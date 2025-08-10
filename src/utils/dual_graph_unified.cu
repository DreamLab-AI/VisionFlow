// Unified Dual-Graph GPU Physics Engine for A6000
// Handles both knowledge graphs and agent swarms with advanced physics
// Optimized for high CUDA core count and memory bandwidth

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cmath>
#include <cfloat>

namespace cg = cooperative_groups;

extern "C" {
    
    // Constants optimized for A6000
    #define BLOCK_SIZE 256
    #define WARP_SIZE 32
    #define MAX_NEIGHBORS 128  // A6000 can handle large neighborhoods
    #define STRESS_ITERATIONS 5  // Multiple iterations per frame
    #define MAX_CONSTRAINTS 1024
    
    // Unified 3D vector
    struct Vec3 {
        float x, y, z;
        
        __device__ inline float length() const {
            return sqrtf(x*x + y*y + z*z);
        }
        
        __device__ inline Vec3 normalized() const {
            float len = length() + 1e-8f;
            return {x/len, y/len, z/len};
        }
        
        __device__ inline float dot(const Vec3& other) const {
            return x*other.x + y*other.y + z*other.z;
        }
    };
    
    __device__ inline Vec3 operator+(const Vec3& a, const Vec3& b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    
    __device__ inline Vec3 operator-(const Vec3& a, const Vec3& b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    
    __device__ inline Vec3 operator*(const Vec3& v, float s) {
        return {v.x * s, v.y * s, v.z * s};
    }
    
    // Dual-graph node structure
    struct DualGraphNode {
        Vec3 position;
        Vec3 velocity;
        Vec3 force;              // Current frame forces
        Vec3 stress_gradient;    // Stress optimization gradient
        float mass;
        float importance;
        
        // Semantic properties
        float semantic_weight;
        float temporal_weight;
        float structural_weight;
        
        // Graph membership and properties
        int graph_id;            // 0=knowledge, 1=agent
        int cluster_id;
        int node_type;           // Type within graph
        
        // Optimization state
        float stress;
        float temperature;       // For simulated annealing
        int neighbor_count;
        
        // Constraint state
        int constraint_mask;     // Bit flags for active constraints
        Vec3 constraint_target;  // Target position from constraints
    };
    
    // Enhanced edge structure
    struct DualGraphEdge {
        int source;
        int target;
        float weight;
        float ideal_distance;
        
        // Multi-modal similarities
        float semantic_sim;
        float temporal_sim;
        float structural_sim;
        float communication_strength;
        
        // Edge properties
        int edge_type;
        int graph_id;           // Which graph this edge belongs to
        bool inter_graph;       // Cross-graph edge
    };
    
    // GPU-optimized constraint structure
    struct GPUConstraint {
        int type;
        int node_count;
        int nodes[8];
        float params[8];
        float weight;
        int graph_mask;         // Which graphs this applies to
    };
    
    // Unified parameters for both graphs
    struct DualGraphParams {
        // Per-graph physics parameters (index 0=knowledge, 1=agent)
        float spring_k[2];
        float repel_k[2];
        float damping[2];
        float ideal_edge_length[2];
        
        // Shared parameters
        float dt;
        float max_velocity;
        float viewport_bounds;
        
        // Stress optimization
        float stress_weight;
        float stress_learning_rate;
        
        // Semantic weights
        float semantic_weight;
        float temporal_weight;
        float structural_weight;
        
        // Inter-graph coupling
        float inter_graph_repulsion;
        float inter_graph_attraction;
        
        // Optimization
        float cooling_rate;
        float constraint_weight;
        
        // System state
        int iteration;
        int total_nodes;
        int total_edges;
        int total_constraints;
        int knowledge_node_count;
        int agent_node_count;
    };
    
    // Shared memory for neighborhood caching
    __shared__ float shared_positions[BLOCK_SIZE * 3];
    __shared__ float shared_weights[BLOCK_SIZE];
    
    // Device function: Calculate repulsion using tiling for A6000 efficiency
    __device__ Vec3 calculate_repulsion_tiled(
        int node_idx,
        const DualGraphNode* nodes,
        const DualGraphParams& params,
        cg::thread_block block
    ) {
        Vec3 force = {0, 0, 0};
        Vec3 pos = nodes[node_idx].position;
        int graph_id = nodes[node_idx].graph_id;
        float repel_k = params.repel_k[graph_id];
        
        // Process nodes in tiles for cache efficiency
        for (int tile = 0; tile < params.total_nodes; tile += BLOCK_SIZE) {
            int tid = threadIdx.x;
            int other_idx = tile + tid;
            
            // Cooperatively load tile into shared memory
            if (other_idx < params.total_nodes) {
                shared_positions[tid * 3] = nodes[other_idx].position.x;
                shared_positions[tid * 3 + 1] = nodes[other_idx].position.y;
                shared_positions[tid * 3 + 2] = nodes[other_idx].position.z;
                shared_weights[tid] = nodes[other_idx].importance;
            }
            
            block.sync();
            
            // Calculate forces from this tile
            for (int j = 0; j < BLOCK_SIZE && tile + j < params.total_nodes; j++) {
                if (tile + j == node_idx) continue;
                
                Vec3 other_pos = {
                    shared_positions[j * 3],
                    shared_positions[j * 3 + 1],
                    shared_positions[j * 3 + 2]
                };
                
                Vec3 diff = pos - other_pos;
                float dist_sq = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
                
                if (dist_sq < 0.001f) dist_sq = 0.001f;
                
                // Enhanced repulsion with importance weighting
                float importance_factor = (nodes[node_idx].importance + shared_weights[j]) * 0.5f;
                
                // Inter-graph repulsion adjustment
                if (nodes[tile + j].graph_id != graph_id) {
                    repel_k *= params.inter_graph_repulsion;
                }
                
                float repel_force = repel_k * importance_factor / dist_sq;
                float dist = sqrtf(dist_sq);
                
                force = force + diff * (repel_force / dist);
            }
            
            block.sync();
        }
        
        return force;
    }
    
    // Device function: Calculate attractive forces from edges
    __device__ Vec3 calculate_edge_forces(
        int node_idx,
        const DualGraphNode* nodes,
        const DualGraphEdge* edges,
        int num_edges,
        const DualGraphParams& params
    ) {
        Vec3 force = {0, 0, 0};
        Vec3 pos = nodes[node_idx].position;
        int graph_id = nodes[node_idx].graph_id;
        float spring_k = params.spring_k[graph_id];
        
        for (int e = 0; e < num_edges; e++) {
            const DualGraphEdge& edge = edges[e];
            
            int other = -1;
            if (edge.source == node_idx) other = edge.target;
            else if (edge.target == node_idx) other = edge.source;
            else continue;
            
            Vec3 other_pos = nodes[other].position;
            Vec3 diff = other_pos - pos;
            float dist = diff.length();
            
            if (dist < 0.001f) continue;
            
            // Multi-modal similarity combination
            float sim = (edge.semantic_sim * params.semantic_weight +
                        edge.temporal_sim * params.temporal_weight +
                        edge.structural_sim * params.structural_weight) /
                       (params.semantic_weight + params.temporal_weight + 
                        params.structural_weight + 0.001f);
            
            // Adaptive ideal distance
            float ideal_dist = edge.ideal_distance;
            if (ideal_dist <= 0) {
                ideal_dist = params.ideal_edge_length[graph_id] * (2.0f - sim);
            }
            
            // Spring force with similarity modulation
            float spring_force = spring_k * (dist - ideal_dist) * edge.weight;
            
            // Inter-graph edge handling
            if (edge.inter_graph) {
                spring_force *= params.inter_graph_attraction;
            }
            
            force = force + diff.normalized() * spring_force;
        }
        
        return force;
    }
    
    // Device function: GPU-based stress gradient computation
    __device__ Vec3 calculate_stress_gradient(
        int node_idx,
        const DualGraphNode* nodes,
        const DualGraphEdge* edges,
        int num_edges,
        const DualGraphParams& params
    ) {
        Vec3 gradient = {0, 0, 0};
        Vec3 pos = nodes[node_idx].position;
        
        // Compute stress gradient for connected nodes
        for (int e = 0; e < num_edges; e++) {
            const DualGraphEdge& edge = edges[e];
            
            int other = -1;
            if (edge.source == node_idx) other = edge.target;
            else if (edge.target == node_idx) other = edge.source;
            else continue;
            
            Vec3 other_pos = nodes[other].position;
            Vec3 diff = pos - other_pos;
            float actual_dist = diff.length() + 0.001f;
            float ideal_dist = edge.ideal_distance;
            
            if (ideal_dist <= 0) {
                ideal_dist = params.ideal_edge_length[edge.graph_id];
            }
            
            // Stress gradient: derivative of (actual - ideal)²
            float stress_deriv = 2.0f * (actual_dist - ideal_dist) / actual_dist;
            gradient = gradient + diff * (stress_deriv * edge.weight);
        }
        
        return gradient * params.stress_weight;
    }
    
    // Device function: Apply constraints
    __device__ Vec3 apply_constraints(
        int node_idx,
        const DualGraphNode* nodes,
        const GPUConstraint* constraints,
        int num_constraints,
        const DualGraphParams& params
    ) {
        Vec3 force = {0, 0, 0};
        const DualGraphNode& node = nodes[node_idx];
        int graph_mask = (1 << node.graph_id);
        
        for (int c = 0; c < num_constraints; c++) {
            const GPUConstraint& constraint = constraints[c];
            
            // Check if constraint applies to this graph
            if (!(constraint.graph_mask & graph_mask)) continue;
            
            // Check if node is affected
            bool affected = false;
            for (int i = 0; i < constraint.node_count; i++) {
                if (constraint.nodes[i] == node_idx) {
                    affected = true;
                    break;
                }
            }
            
            if (!affected) continue;
            
            Vec3 constraint_force = {0, 0, 0};
            
            switch (constraint.type) {
                case 0: // Fixed position
                    constraint_force = Vec3{
                        constraint.params[0],
                        constraint.params[1],
                        constraint.params[2]
                    } - node.position;
                    constraint_force = constraint_force * 10.0f;
                    break;
                    
                case 1: // Separation
                    for (int i = 0; i < constraint.node_count; i++) {
                        int other = constraint.nodes[i];
                        if (other == node_idx || other < 0) continue;
                        
                        Vec3 diff = node.position - nodes[other].position;
                        float dist = diff.length();
                        float min_dist = constraint.params[0];
                        
                        if (dist < min_dist && dist > 0.001f) {
                            constraint_force = constraint_force + 
                                diff.normalized() * ((min_dist - dist) * 5.0f);
                        }
                    }
                    break;
                    
                case 2: // Clustering
                    {
                        Vec3 center = {0, 0, 0};
                        int count = 0;
                        for (int i = 0; i < constraint.node_count; i++) {
                            int idx = constraint.nodes[i];
                            if (idx >= 0 && idx < params.total_nodes) {
                                center = center + nodes[idx].position;
                                count++;
                            }
                        }
                        if (count > 0) {
                            center = center * (1.0f / count);
                            constraint_force = (center - node.position) * constraint.params[0];
                        }
                    }
                    break;
            }
            
            force = force + constraint_force * constraint.weight;
        }
        
        return force * params.constraint_weight;
    }
    
    // Main unified kernel - leverages A6000's massive parallelism
    __global__ void dual_graph_unified_kernel(
        DualGraphNode* nodes,
        const DualGraphEdge* edges,
        const GPUConstraint* constraints,
        const DualGraphParams params
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.total_nodes) return;
        
        auto block = cg::this_thread_block();
        DualGraphNode& node = nodes[idx];
        
        // Reset force accumulator
        Vec3 total_force = {0, 0, 0};
        
        // 1. Repulsion (tiled for A6000 efficiency)
        Vec3 repulsion = calculate_repulsion_tiled(idx, nodes, params, block);
        total_force = total_force + repulsion;
        
        // 2. Edge attractions
        Vec3 edge_force = calculate_edge_forces(idx, nodes, edges, 
                                                params.total_edges, params);
        total_force = total_force + edge_force;
        
        // 3. Stress optimization gradient (multiple iterations)
        for (int iter = 0; iter < STRESS_ITERATIONS; iter++) {
            Vec3 stress_grad = calculate_stress_gradient(idx, nodes, edges,
                                                         params.total_edges, params);
            // Gradient descent step
            total_force = total_force - stress_grad * params.stress_learning_rate;
        }
        
        // 4. Constraint forces
        Vec3 constraint_force = apply_constraints(idx, nodes, constraints,
                                                  params.total_constraints, params);
        total_force = total_force + constraint_force;
        
        // 5. Graph-specific adjustments
        if (node.graph_id == 1) { // Agent graph
            // Agents have stronger centering
            Vec3 center_force = node.position * (-0.02f);
            total_force = total_force + center_force;
        } else { // Knowledge graph
            // Knowledge nodes have weaker centering
            Vec3 center_force = node.position * (-0.005f * (1.0f - node.importance));
            total_force = total_force + center_force;
        }
        
        // 6. Simulated annealing
        float cooling = 1.0f / (1.0f + params.iteration * params.cooling_rate);
        total_force = total_force * cooling;
        
        // 7. Update velocity with damping
        float damping = params.damping[node.graph_id];
        node.velocity = node.velocity * (1.0f - damping) + 
                       total_force * (params.dt / node.mass);
        
        // 8. Velocity clamping
        float vel_mag = node.velocity.length();
        if (vel_mag > params.max_velocity) {
            node.velocity = node.velocity.normalized() * params.max_velocity;
        }
        
        // 9. Update position
        node.position = node.position + node.velocity * params.dt;
        
        // 10. Boundary constraints
        float bounds = params.viewport_bounds;
        node.position.x = fmaxf(-bounds, fminf(bounds, node.position.x));
        node.position.y = fmaxf(-bounds, fminf(bounds, node.position.y));
        node.position.z = fmaxf(-bounds * 0.5f, fminf(bounds * 0.5f, node.position.z));
        
        // 11. Store computed values for monitoring
        node.force = total_force;
        // node.stress_gradient = stress_grad; // Field doesn't exist, skipping
        
        // Calculate and store stress for convergence monitoring
        float stress = 0.0f;
        for (int e = 0; e < params.total_edges; e++) {
            const DualGraphEdge& edge = edges[e];
            if (edge.source == idx || edge.target == idx) {
                int other = (edge.source == idx) ? edge.target : edge.source;
                Vec3 diff = node.position - nodes[other].position;
                float actual = diff.length();
                float ideal = edge.ideal_distance;
                stress += edge.weight * (actual - ideal) * (actual - ideal);
            }
        }
        node.stress = stress;
    }
    
    // Kernel for preprocessing edge ideal distances
    __global__ void precompute_ideal_distances_kernel(
        DualGraphEdge* edges,
        const DualGraphNode* nodes,
        const DualGraphParams params
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= params.total_edges) return;
        
        DualGraphEdge& edge = edges[idx];
        const DualGraphNode& source = nodes[edge.source];
        const DualGraphNode& target = nodes[edge.target];
        
        // Base ideal distance from graph type
        float base_dist = params.ideal_edge_length[edge.graph_id];
        
        // Adjust based on similarity
        float sim = (edge.semantic_sim + edge.temporal_sim + edge.structural_sim) / 3.0f;
        base_dist *= (2.0f - sim);
        
        // Inter-graph edges get longer distances
        if (edge.inter_graph) {
            base_dist *= 2.0f;
        }
        
        // Cluster-based adjustments
        if (source.cluster_id == target.cluster_id && source.cluster_id >= 0) {
            base_dist *= 0.7f; // Same cluster = closer
        }
        
        edge.ideal_distance = base_dist;
    }
    
    // Entry point for dual-graph physics
    void launch_dual_graph_physics(
        void* nodes,
        void* edges,
        void* constraints,
        void* params_ptr,
        int num_nodes,
        int num_edges,
        int num_constraints
    ) {
        DualGraphParams* params = (DualGraphParams*)params_ptr;
        params->total_nodes = num_nodes;
        params->total_edges = num_edges;
        params->total_constraints = num_constraints;
        
        const int block_size = BLOCK_SIZE;
        const int grid_size = (num_nodes + block_size - 1) / block_size;
        
        // Precompute ideal distances
        if (num_edges > 0) {
            const int edge_grid = (num_edges + block_size - 1) / block_size;
            precompute_ideal_distances_kernel<<<edge_grid, block_size>>>(
                (DualGraphEdge*)edges,
                (const DualGraphNode*)nodes,
                *params
            );
        }
        
        // Main physics kernel
        dual_graph_unified_kernel<<<grid_size, block_size>>>(
            (DualGraphNode*)nodes,
            (const DualGraphEdge*)edges,
            (const GPUConstraint*)constraints,
            *params
        );
        
        // Synchronize for consistency
        cudaDeviceSynchronize();
    }
}