// Unified GPU Physics Engine - All computation on GPU
// Combines force-directed, stress majorization, and constraint satisfaction in a single kernel

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

extern "C" {
    
    // Unified vector structure
    struct Vec3 {
        float x, y, z;
    };
    
    // Unified node data with all features
    struct UnifiedNode {
        Vec3 position;
        Vec3 velocity;
        Vec3 target_position;     // For stress majorization
        float mass;
        float importance;
        float semantic_weight;
        float temporal_weight;
        float structural_weight;
        int cluster_id;
        int node_type;            // 0=document, 1=agent, 2=cluster_center
        int flags;
        float stress;             // Current stress value
    };
    
    // Unified edge data
    struct UnifiedEdge {
        int source;
        int target;
        float weight;
        float ideal_distance;     // From stress majorization
        float semantic_sim;
        float temporal_sim;
        float structural_sim;
        int edge_type;           // 0=normal, 1=hierarchical, 2=semantic
    };
    
    // Unified constraint data
    struct UnifiedConstraint {
        int type;                // 0=fixed, 1=separation, 2=alignment, 3=cluster, 4=boundary
        int node_count;
        int nodes[8];            // Up to 8 nodes per constraint
        float params[8];         // Constraint-specific parameters
        float weight;
    };
    
    // Unified physics parameters - single source of truth
    struct UnifiedParams {
        // Force-directed parameters
        float spring_k;
        float repel_k;
        float damping;
        float dt;
        float max_velocity;
        
        // Stress majorization parameters
        float stress_weight;
        float stress_convergence;
        int stress_iterations;
        
        // Semantic parameters
        float semantic_weight;
        float temporal_weight;
        float structural_weight;
        
        // Constraint parameters
        float constraint_weight;
        float separation_radius;
        float cluster_strength;
        
        // Layout parameters
        float viewport_bounds;
        float ideal_edge_length;
        float cooling_rate;
        
        // System parameters
        int iteration;
        int total_nodes;
        int total_edges;
        int total_constraints;
    };
    
    // Device functions for vector operations
    __device__ inline float length3(const Vec3& v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    
    __device__ inline Vec3 normalize3(const Vec3& v) {
        float len = length3(v) + 1e-8f;
        return {v.x / len, v.y / len, v.z / len};
    }
    
    __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    
    __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    
    __device__ inline Vec3 operator*(const Vec3& v, float s) {
        return {v.x * s, v.y * s, v.z * s};
    }
    
    // GPU-based stress calculation for a node
    __device__ float calculate_node_stress(
        int node_idx,
        const UnifiedNode* nodes,
        const UnifiedEdge* edges,
        int num_edges
    ) {
        float stress = 0.0f;
        Vec3 pos = nodes[node_idx].position;
        
        for (int e = 0; e < num_edges; e++) {
            const UnifiedEdge& edge = edges[e];
            int other = -1;
            
            if (edge.source == node_idx) other = edge.target;
            else if (edge.target == node_idx) other = edge.source;
            else continue;
            
            Vec3 other_pos = nodes[other].position;
            float actual_dist = length3(other_pos - pos);
            float ideal_dist = edge.ideal_distance;
            float diff = actual_dist - ideal_dist;
            
            // Weighted stress contribution
            stress += edge.weight * diff * diff;
        }
        
        return stress;
    }
    
    // GPU-based semantic force calculation
    __device__ Vec3 calculate_semantic_force(
        int node_idx,
        const UnifiedNode* nodes,
        const UnifiedEdge* edges,
        int num_edges,
        const UnifiedParams& params
    ) {
        Vec3 force = {0, 0, 0};
        Vec3 pos = nodes[node_idx].position;
        float semantic_w = nodes[node_idx].semantic_weight;
        
        for (int e = 0; e < num_edges; e++) {
            const UnifiedEdge& edge = edges[e];
            int other = -1;
            
            if (edge.source == node_idx) other = edge.target;
            else if (edge.target == node_idx) other = edge.source;
            else continue;
            
            Vec3 other_pos = nodes[other].position;
            Vec3 diff = other_pos - pos;
            float dist = length3(diff) + 0.001f;
            Vec3 dir = normalize3(diff);
            
            // Multi-modal similarity
            float similarity = (edge.semantic_sim * params.semantic_weight +
                              edge.temporal_sim * params.temporal_weight +
                              edge.structural_sim * params.structural_weight) /
                             (params.semantic_weight + params.temporal_weight + params.structural_weight + 0.001f);
            
            // Adaptive ideal distance based on similarity
            float ideal_dist = params.ideal_edge_length * (2.0f - similarity);
            
            // Spring force modulated by similarity
            float spring_force = params.spring_k * (dist - ideal_dist) * similarity;
            force = force + dir * spring_force * semantic_w;
        }
        
        return force;
    }
    
    // GPU-based constraint force calculation
    __device__ Vec3 calculate_constraint_force(
        int node_idx,
        const UnifiedNode* nodes,
        const UnifiedConstraint* constraints,
        int num_constraints,
        const UnifiedParams& params
    ) {
        Vec3 force = {0, 0, 0};
        Vec3 pos = nodes[node_idx].position;
        
        for (int c = 0; c < num_constraints; c++) {
            const UnifiedConstraint& constraint = constraints[c];
            
            // Check if node is affected by this constraint
            bool affected = false;
            int node_pos_in_constraint = -1;
            for (int i = 0; i < constraint.node_count && i < 8; i++) {
                if (constraint.nodes[i] == node_idx) {
                    affected = true;
                    node_pos_in_constraint = i;
                    break;
                }
            }
            
            if (!affected) continue;
            
            Vec3 constraint_force = {0, 0, 0};
            
            switch (constraint.type) {
                case 0: { // Fixed position
                    Vec3 target = {constraint.params[0], constraint.params[1], constraint.params[2]};
                    constraint_force = (target - pos) * 10.0f;
                    break;
                }
                
                case 1: { // Separation
                    if (constraint.node_count >= 2) {
                        int other = (node_pos_in_constraint == 0) ? 
                            constraint.nodes[1] : constraint.nodes[0];
                        if (other >= 0 && other < params.total_nodes) {
                            Vec3 other_pos = nodes[other].position;
                            Vec3 diff = pos - other_pos;
                            float dist = length3(diff);
                            float min_dist = constraint.params[0];
                            
                            if (dist < min_dist && dist > 0.001f) {
                                Vec3 dir = normalize3(diff);
                                float correction = (min_dist - dist) * params.separation_radius;
                                constraint_force = dir * correction;
                            }
                        }
                    }
                    break;
                }
                
                case 2: { // Alignment
                    float target_val = constraint.params[0];
                    int axis = (int)constraint.params[1]; // 0=x, 1=y, 2=z
                    
                    if (axis == 0) constraint_force.x = (target_val - pos.x) * 5.0f;
                    else if (axis == 1) constraint_force.y = (target_val - pos.y) * 5.0f;
                    else if (axis == 2) constraint_force.z = (target_val - pos.z) * 5.0f;
                    break;
                }
                
                case 3: { // Clustering
                    Vec3 center = {0, 0, 0};
                    int cluster_size = 0;
                    
                    // Calculate cluster center
                    for (int i = 0; i < constraint.node_count && i < 8; i++) {
                        int idx = constraint.nodes[i];
                        if (idx >= 0 && idx < params.total_nodes) {
                            center = center + nodes[idx].position;
                            cluster_size++;
                        }
                    }
                    
                    if (cluster_size > 0) {
                        center = center * (1.0f / cluster_size);
                        Vec3 to_center = center - pos;
                        constraint_force = to_center * params.cluster_strength;
                    }
                    break;
                }
                
                case 4: { // Boundary
                    float bounds = params.viewport_bounds;
                    if (fabsf(pos.x) > bounds * 0.9f) 
                        constraint_force.x = -pos.x * 2.0f;
                    if (fabsf(pos.y) > bounds * 0.9f) 
                        constraint_force.y = -pos.y * 2.0f;
                    if (fabsf(pos.z) > bounds * 0.5f) 
                        constraint_force.z = -pos.z * 2.0f;
                    break;
                }
            }
            
            force = force + constraint_force * constraint.weight;
        }
        
        return force * params.constraint_weight;
    }
    
    // GPU-based stress majorization step (iterative refinement)
    __device__ Vec3 calculate_stress_gradient(
        int node_idx,
        const UnifiedNode* nodes,
        const UnifiedEdge* edges,
        int num_edges,
        const UnifiedParams& params
    ) {
        Vec3 gradient = {0, 0, 0};
        Vec3 pos = nodes[node_idx].position;
        float epsilon = 0.001f;
        
        for (int e = 0; e < num_edges; e++) {
            const UnifiedEdge& edge = edges[e];
            int other = -1;
            
            if (edge.source == node_idx) other = edge.target;
            else if (edge.target == node_idx) other = edge.source;
            else continue;
            
            Vec3 other_pos = nodes[other].position;
            Vec3 diff = pos - other_pos;
            float actual_dist = length3(diff) + epsilon;
            float ideal_dist = edge.ideal_distance;
            
            if (actual_dist > epsilon) {
                // Gradient of stress function
                float factor = edge.weight * (1.0f - ideal_dist / actual_dist);
                gradient = gradient + diff * factor;
            }
        }
        
        return gradient * params.stress_weight;
    }
    
    // Main unified physics kernel - combines all forces
    __global__ void unified_physics_kernel(
        UnifiedNode* nodes,
        const UnifiedEdge* edges,
        const UnifiedConstraint* constraints,
        const UnifiedParams params
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.total_nodes) return;
        
        UnifiedNode& node = nodes[idx];
        Vec3 pos = node.position;
        Vec3 vel = node.velocity;
        
        // 1. Calculate all forces
        Vec3 total_force = {0, 0, 0};
        
        // Repulsion between all nodes (Barnes-Hut approximation could be added)
        for (int j = 0; j < params.total_nodes; j++) {
            if (j == idx) continue;
            
            Vec3 other_pos = nodes[j].position;
            Vec3 diff = pos - other_pos;
            float dist = length3(diff) + 0.001f;
            
            if (dist < params.ideal_edge_length * 3.0f) {
                Vec3 dir = normalize3(diff);
                float repel = params.repel_k / (dist * dist);
                
                // Modulate by importance
                repel *= (node.importance + nodes[j].importance) * 0.5f;
                
                total_force = total_force + dir * repel;
            }
        }
        
        // 2. Semantic forces from edges
        Vec3 semantic_force = calculate_semantic_force(idx, nodes, edges, 
                                                       params.total_edges, params);
        total_force = total_force + semantic_force;
        
        // 3. Constraint forces
        Vec3 constraint_force = calculate_constraint_force(idx, nodes, constraints,
                                                          params.total_constraints, params);
        total_force = total_force + constraint_force;
        
        // 4. Stress gradient (for smoother layouts)
        Vec3 stress_grad = calculate_stress_gradient(idx, nodes, edges, 
                                                     params.total_edges, params);
        total_force = total_force - stress_grad; // Negative gradient for minimization
        
        // 5. Global centering force (weak)
        Vec3 center_force = pos * (-0.01f * (1.0f - node.importance));
        total_force = total_force + center_force;
        
        // 6. Apply cooling (simulated annealing)
        float cooling = 1.0f / (1.0f + params.iteration * params.cooling_rate);
        total_force = total_force * cooling;
        
        // 7. Update velocity with damping
        vel = vel * (1.0f - params.damping) + total_force * (params.dt / node.mass);
        
        // 8. Clamp velocity
        float vel_mag = length3(vel);
        if (vel_mag > params.max_velocity) {
            vel = normalize3(vel) * params.max_velocity;
        }
        
        // 9. Update position
        pos = pos + vel * params.dt;
        
        // 10. Hard boundary clamping
        float bounds = params.viewport_bounds;
        pos.x = fmaxf(-bounds, fminf(bounds, pos.x));
        pos.y = fmaxf(-bounds, fminf(bounds, pos.y));
        pos.z = fmaxf(-bounds * 0.5f, fminf(bounds * 0.5f, pos.z));
        
        // 11. Calculate and store node stress for monitoring
        node.stress = calculate_node_stress(idx, nodes, edges, params.total_edges);
        
        // 12. Write back results
        node.position = pos;
        node.velocity = vel;
    }
    
    // Kernel for computing ideal distances (stress majorization preprocessing)
    __global__ void compute_ideal_distances_kernel(
        UnifiedEdge* edges,
        const UnifiedNode* nodes,
        const UnifiedParams params
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.total_edges) return;
        
        UnifiedEdge& edge = edges[idx];
        
        // Base ideal distance from edge weight and similarity
        float similarity = (edge.semantic_sim + edge.temporal_sim + edge.structural_sim) / 3.0f;
        float base_dist = params.ideal_edge_length * (2.0f - similarity);
        
        // Adjust for node types
        const UnifiedNode& source = nodes[edge.source];
        const UnifiedNode& target = nodes[edge.target];
        
        if (source.node_type == 2 || target.node_type == 2) {
            // Cluster centers should be further apart
            base_dist *= 1.5f;
        }
        
        if (source.cluster_id == target.cluster_id && source.cluster_id >= 0) {
            // Same cluster nodes should be closer
            base_dist *= 0.7f;
        }
        
        edge.ideal_distance = base_dist;
    }
    
    // Kernel for updating cluster centers
    __global__ void update_cluster_centers_kernel(
        UnifiedNode* nodes,
        const UnifiedParams params
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.total_nodes) return;
        
        UnifiedNode& node = nodes[idx];
        if (node.node_type != 2) return; // Not a cluster center
        
        // Find all nodes in this cluster
        Vec3 center = {0, 0, 0};
        int count = 0;
        
        for (int i = 0; i < params.total_nodes; i++) {
            if (nodes[i].cluster_id == node.cluster_id && i != idx) {
                center = center + nodes[i].position;
                count++;
            }
        }
        
        if (count > 0) {
            center = center * (1.0f / count);
            // Smoothly move cluster center toward calculated center
            node.target_position = center;
            Vec3 diff = center - node.position;
            node.velocity = node.velocity * 0.5f + diff * 0.1f;
        }
    }
    
    // Entry point for launching unified physics
    void launch_unified_physics(
        void* nodes,
        void* edges,
        void* constraints,
        void* params_ptr,
        int num_nodes,
        int num_edges,
        int num_constraints
    ) {
        const int block_size = 256;
        const int grid_size = (num_nodes + block_size - 1) / block_size;
        
        UnifiedParams* params = (UnifiedParams*)params_ptr;
        params->total_nodes = num_nodes;
        params->total_edges = num_edges;
        params->total_constraints = num_constraints;
        
        // Precompute ideal distances
        if (num_edges > 0) {
            const int edge_grid = (num_edges + block_size - 1) / block_size;
            compute_ideal_distances_kernel<<<edge_grid, block_size>>>(
                (UnifiedEdge*)edges,
                (const UnifiedNode*)nodes,
                *params
            );
        }
        
        // Update cluster centers if needed
        update_cluster_centers_kernel<<<grid_size, block_size>>>(
            (UnifiedNode*)nodes,
            *params
        );
        
        // Main physics step
        unified_physics_kernel<<<grid_size, block_size>>>(
            (UnifiedNode*)nodes,
            (const UnifiedEdge*)edges,
            (const UnifiedConstraint*)constraints,
            *params
        );
        
        // Synchronize
        cudaDeviceSynchronize();
    }
}