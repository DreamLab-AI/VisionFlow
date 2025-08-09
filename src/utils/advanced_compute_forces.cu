// Advanced constraint-aware force computation kernel for knowledge graph layout
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

extern "C" {
    // Enhanced 3D vector structure
    struct Vec3Data { 
        float x; 
        float y; 
        float z; 
    };

    // Enhanced node data with semantic properties
    struct EnhancedBinaryNodeData {
        Vec3Data position;
        Vec3Data velocity;
        unsigned char mass;
        unsigned char flags;
        unsigned char node_type;
        unsigned char cluster_id;
        float semantic_weight;
        float temporal_weight;
        float structural_weight;
        float importance_score;
    };

    // Enhanced edge data with multi-modal similarities
    struct EnhancedEdgeData {
        int source_idx;
        int target_idx;
        float weight;
        float semantic_similarity;
        float structural_similarity;
        float temporal_similarity;
        float communication_strength;
        unsigned char edge_type;
        unsigned char bidirectional;
        unsigned char pad[2];
    };

    // GPU-compatible constraint representation
    struct ConstraintData {
        int kind; // matches ConstraintKind discriminant
        int count; // number of node indices used
        int node_idx[4]; // indices of affected nodes
        float params[8]; // constraint-specific parameters
        float weight; // constraint strength
        float pad[3]; // padding for alignment
    };

    // Advanced simulation parameters
    struct AdvancedSimulationParams {
        // Basic physics
        float spring_k;
        float damping;
        float repel_k;
        float dt;
        float max_repulsion_dist;
        float viewport_bounds;
        
        // Advanced force weights
        float semantic_force_weight;
        float temporal_force_weight;
        float structural_force_weight;
        float constraint_force_weight;
        float boundary_force_weight;
        float separation_factor;
        float knowledge_force_weight;
        float agent_communication_weight;
        
        // Layout optimization
        float target_edge_length;
        float max_velocity;
        float collision_threshold;
        float adaptive_scale;
        
        // Hierarchical layout
        int hierarchical_mode;
        float layer_separation;
        
        // System parameters
        int iteration;
        int total_nodes;
    };

    // Device helper functions
    __device__ inline float3 make_f3(const Vec3Data &v) { 
        return make_float3(v.x, v.y, v.z);
    }
    
    __device__ inline Vec3Data make_v3(const float3 &v) { 
        return Vec3Data{v.x, v.y, v.z}; 
    }
    
    __device__ inline float length3(const float3 &v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    
    __device__ inline float3 normalize3(const float3 &v) {
        float len = length3(v) + 1e-8f;
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    
    __device__ inline float3 clamp3(const float3 &v, float max_val) {
        float len = length3(v);
        if (len > max_val && len > 0.0f) {
            float scale = max_val / len;
            return make_float3(v.x * scale, v.y * scale, v.z * scale);
        }
        return v;
    }

    // Apply soft boundary forces to keep nodes within viewport
    __device__ float3 apply_boundary_force(const float3 &pos, float bounds, float weight) {
        if (bounds <= 0.0f || weight <= 0.0f) return make_float3(0, 0, 0);
        
        float3 force = make_float3(0, 0, 0);
        const float soft_zone = 0.8f * bounds;
        const float hard_zone = 0.95f * bounds;
        
        // Progressive boundary force
        auto boundary_component = [](float p, float soft, float hard) {
            float abs_p = fabsf(p);
            if (abs_p > hard) {
                return -p * 10.0f; // Strong pushback
            } else if (abs_p > soft) {
                float t = (abs_p - soft) / (hard - soft);
                return -p * (1.0f + 9.0f * t * t); // Quadratic ramp
            }
            return 0.0f;
        };
        
        force.x = boundary_component(pos.x, soft_zone, hard_zone) * weight;
        force.y = boundary_component(pos.y, soft_zone, hard_zone) * weight;
        force.z = boundary_component(pos.z, soft_zone * 0.5f, hard_zone * 0.5f) * weight * 0.5f; // Less Z constraint
        
        return force;
    }

    // Calculate adaptive force scaling based on local density
    __device__ float calculate_adaptive_scale(
        int node_idx,
        const EnhancedBinaryNodeData* nodes,
        int num_nodes,
        float radius
    ) {
        int nearby_count = 0;
        float3 pos = make_f3(nodes[node_idx].position);
        
        for (int j = 0; j < num_nodes && nearby_count < 20; j++) {
            if (j == node_idx) continue;
            float3 other_pos = make_f3(nodes[j].position);
            float3 diff = make_float3(
                other_pos.x - pos.x,
                other_pos.y - pos.y,
                other_pos.z - pos.z
            );
            float dist = length3(diff);
            if (dist < radius) nearby_count++;
        }
        
        // Scale forces based on local density (more dense = weaker forces)
        return 1.0f / (1.0f + 0.1f * nearby_count);
    }

    // Apply constraint forces
    __device__ float3 apply_constraint_forces(
        int node_idx,
        const float3 &pos,
        const EnhancedBinaryNodeData* nodes,
        const ConstraintData* constraints,
        int num_constraints,
        const AdvancedSimulationParams &params
    ) {
        float3 total_force = make_float3(0, 0, 0);
        
        for (int c = 0; c < num_constraints; c++) {
            const ConstraintData &constraint = constraints[c];
            
            // Check if this node is affected by the constraint
            bool affected = false;
            int node_pos_in_constraint = -1;
            for (int i = 0; i < constraint.count && i < 4; i++) {
                if (constraint.node_idx[i] == node_idx) {
                    affected = true;
                    node_pos_in_constraint = i;
                    break;
                }
            }
            
            if (!affected) continue;
            
            float3 force = make_float3(0, 0, 0);
            
            switch (constraint.kind) {
                case 0: { // FixedPosition
                    float3 target = make_float3(
                        constraint.params[0],
                        constraint.params[1],
                        constraint.params[2]
                    );
                    float3 diff = make_float3(
                        target.x - pos.x,
                        target.y - pos.y,
                        target.z - pos.z
                    );
                    force = diff;
                    force.x *= 10.0f;
                    force.y *= 10.0f;
                    force.z *= 10.0f;
                    break;
                }
                
                case 1: { // Separation
                    if (constraint.count >= 2) {
                        int other_idx = (node_pos_in_constraint == 0) ? 
                            constraint.node_idx[1] : constraint.node_idx[0];
                        if (other_idx >= 0 && other_idx < params.total_nodes) {
                            float3 other_pos = make_f3(nodes[other_idx].position);
                            float3 diff = make_float3(
                                pos.x - other_pos.x,
                                pos.y - other_pos.y,
                                pos.z - other_pos.z
                            );
                            float dist = length3(diff);
                            float min_dist = constraint.params[0];
                            
                            if (dist < min_dist && dist > 0.001f) {
                                float3 dir = normalize3(diff);
                                float correction = (min_dist - dist) * params.separation_factor;
                                force.x = dir.x * correction;
                                force.y = dir.y * correction;
                                force.z = dir.z * correction;
                            }
                        }
                    }
                    break;
                }
                
                case 2: { // AlignmentHorizontal
                    float target_y = constraint.params[0];
                    force.y = (target_y - pos.y) * 2.0f;
                    break;
                }
                
                case 3: { // AlignmentVertical
                    float target_x = constraint.params[0];
                    force.x = (target_x - pos.x) * 2.0f;
                    break;
                }
                
                case 4: { // AlignmentDepth
                    float target_z = constraint.params[0];
                    force.z = (target_z - pos.z) * 2.0f;
                    break;
                }
                
                case 5: { // Clustering
                    float cluster_id = constraint.params[0];
                    float strength = constraint.params[1];
                    
                    // Calculate cluster center
                    float3 center = make_float3(0, 0, 0);
                    int cluster_count = 0;
                    for (int i = 0; i < constraint.count && i < 4; i++) {
                        int idx = constraint.node_idx[i];
                        if (idx >= 0 && idx < params.total_nodes) {
                            float3 node_pos = make_f3(nodes[idx].position);
                            center.x += node_pos.x;
                            center.y += node_pos.y;
                            center.z += node_pos.z;
                            cluster_count++;
                        }
                    }
                    
                    if (cluster_count > 0) {
                        center.x /= cluster_count;
                        center.y /= cluster_count;
                        center.z /= cluster_count;
                        
                        float3 to_center = make_float3(
                            center.x - pos.x,
                            center.y - pos.y,
                            center.z - pos.z
                        );
                        
                        force.x = to_center.x * strength;
                        force.y = to_center.y * strength;
                        force.z = to_center.z * strength;
                    }
                    break;
                }
                
                case 6: { // Boundary
                    float min_x = constraint.params[0];
                    float max_x = constraint.params[1];
                    float min_y = constraint.params[2];
                    float max_y = constraint.params[3];
                    float min_z = constraint.params[4];
                    float max_z = constraint.params[5];
                    
                    if (pos.x < min_x) force.x = (min_x - pos.x) * 5.0f;
                    if (pos.x > max_x) force.x = (max_x - pos.x) * 5.0f;
                    if (pos.y < min_y) force.y = (min_y - pos.y) * 5.0f;
                    if (pos.y > max_y) force.y = (max_y - pos.y) * 5.0f;
                    if (pos.z < min_z) force.z = (min_z - pos.z) * 5.0f;
                    if (pos.z > max_z) force.z = (max_z - pos.z) * 5.0f;
                    break;
                }
                
                case 7: { // DirectionalFlow
                    float angle = constraint.params[0];
                    float strength = constraint.params[1];
                    force.x = cosf(angle) * strength;
                    force.y = sinf(angle) * strength;
                    break;
                }
                
                case 8: { // RadialDistance
                    float center_x = constraint.params[0];
                    float center_y = constraint.params[1];
                    float center_z = constraint.params[2];
                    float target_radius = constraint.params[3];
                    
                    float3 center = make_float3(center_x, center_y, center_z);
                    float3 to_node = make_float3(
                        pos.x - center.x,
                        pos.y - center.y,
                        pos.z - center.z
                    );
                    float current_radius = length3(to_node);
                    
                    if (current_radius > 0.001f) {
                        float3 dir = normalize3(to_node);
                        float diff = target_radius - current_radius;
                        force.x = dir.x * diff * 2.0f;
                        force.y = dir.y * diff * 2.0f;
                        force.z = dir.z * diff * 2.0f;
                    }
                    break;
                }
                
                case 9: { // LayerDepth
                    float layer_index = constraint.params[0];
                    float z_position = constraint.params[1];
                    force.z = (z_position - pos.z) * 5.0f;
                    break;
                }
            }
            
            // Apply constraint weight
            force.x *= constraint.weight;
            force.y *= constraint.weight;
            force.z *= constraint.weight;
            
            total_force.x += force.x;
            total_force.y += force.y;
            total_force.z += force.z;
        }
        
        return total_force;
    }

    // Main advanced forces kernel
    __global__ void advanced_forces_kernel(
        EnhancedBinaryNodeData* nodes,
        const EnhancedEdgeData* edges,
        int num_nodes,
        int num_edges,
        const ConstraintData* constraints,
        int num_constraints,
        const AdvancedSimulationParams params
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_nodes) return;
        
        float3 pos = make_f3(nodes[i].position);
        float3 vel = make_f3(nodes[i].velocity);
        float mass = (nodes[i].mass == 0) ? 0.5f : ((nodes[i].mass + 1.0f) / 256.0f);
        
        // Node properties for force modulation
        float semantic_w = nodes[i].semantic_weight;
        float temporal_w = nodes[i].temporal_weight;
        float structural_w = nodes[i].structural_weight;
        float importance = nodes[i].importance_score;
        
        float3 total_force = make_float3(0, 0, 0);
        
        // Calculate adaptive scaling if enabled
        float adaptive_scale = 1.0f;
        if (params.adaptive_scale > 0.0f) {
            adaptive_scale = calculate_adaptive_scale(i, nodes, num_nodes, params.max_repulsion_dist * 2.0f);
        }
        
        // Process edges for spring and semantic forces
        for (int e = 0; e < num_edges; e++) {
            const EnhancedEdgeData &edge = edges[e];
            
            int other_idx = -1;
            float direction_factor = 1.0f;
            
            if (edge.source_idx == i) {
                other_idx = edge.target_idx;
            } else if (edge.target_idx == i) {
                other_idx = edge.source_idx;
                direction_factor = edge.bidirectional ? 1.0f : 0.5f;
            }
            
            if (other_idx < 0 || other_idx >= num_nodes) continue;
            
            float3 other_pos = make_f3(nodes[other_idx].position);
            float3 diff = make_float3(
                other_pos.x - pos.x,
                other_pos.y - pos.y,
                other_pos.z - pos.z
            );
            float dist = length3(diff) + 0.001f;
            float3 dir = normalize3(diff);
            
            // Multi-modal similarity combination
            float semantic_sim = edge.semantic_similarity * params.semantic_force_weight;
            float temporal_sim = edge.temporal_similarity * params.temporal_force_weight;
            float structural_sim = edge.structural_similarity * params.structural_force_weight;
            float comm_strength = edge.communication_strength * params.agent_communication_weight;
            
            float combined_similarity = (semantic_sim + temporal_sim + structural_sim + comm_strength) / 
                (params.semantic_force_weight + params.temporal_force_weight + 
                 params.structural_force_weight + params.agent_communication_weight + 0.001f);
            
            // Adaptive ideal distance based on similarity
            float ideal_dist = params.target_edge_length * (2.0f - combined_similarity);
            
            // Spring force with similarity modulation
            float spring_force = -params.spring_k * (dist - ideal_dist) * 
                                (0.5f + 0.5f * combined_similarity) * direction_factor;
            
            // Apply spring force
            total_force.x += dir.x * spring_force * adaptive_scale;
            total_force.y += dir.y * spring_force * adaptive_scale;
            total_force.z += dir.z * spring_force * adaptive_scale;
        }
        
        // Global repulsion between all nodes
        for (int j = 0; j < num_nodes; j++) {
            if (j == i) continue;
            
            float3 other_pos = make_f3(nodes[j].position);
            float3 diff = make_float3(
                pos.x - other_pos.x,
                pos.y - other_pos.y,
                pos.z - other_pos.z
            );
            float dist = length3(diff) + 0.001f;
            
            if (dist < params.max_repulsion_dist) {
                float3 dir = normalize3(diff);
                
                // Importance-weighted repulsion
                float other_importance = nodes[j].importance_score;
                float importance_factor = (importance + other_importance) * 0.5f;
                
                // Collision detection with stronger force
                float repel_force = 0.0f;
                if (dist < params.collision_threshold) {
                    repel_force = params.repel_k * 10.0f / (dist * dist);
                } else {
                    repel_force = params.repel_k * importance_factor / (dist * dist);
                }
                
                total_force.x += dir.x * repel_force * adaptive_scale;
                total_force.y += dir.y * repel_force * adaptive_scale;
                total_force.z += dir.z * repel_force * adaptive_scale;
            }
        }
        
        // Apply boundary forces
        float3 boundary_force = apply_boundary_force(pos, params.viewport_bounds, params.boundary_force_weight);
        total_force.x += boundary_force.x;
        total_force.y += boundary_force.y;
        total_force.z += boundary_force.z;
        
        // Apply constraint forces
        if (num_constraints > 0 && params.constraint_force_weight > 0.0f) {
            float3 constraint_force = apply_constraint_forces(
                i, pos, nodes, constraints, num_constraints, params
            );
            total_force.x += constraint_force.x * params.constraint_force_weight;
            total_force.y += constraint_force.y * params.constraint_force_weight;
            total_force.z += constraint_force.z * params.constraint_force_weight;
        }
        
        // Hierarchical mode adjustments
        if (params.hierarchical_mode) {
            float layer = floorf(nodes[i].cluster_id);
            float target_z = layer * params.layer_separation;
            float z_force = (target_z - pos.z) * 3.0f;
            total_force.z += z_force;
        }
        
        // Center of mass attraction (weak)
        float center_force_strength = 0.01f * (1.0f - importance);
        total_force.x -= pos.x * center_force_strength;
        total_force.y -= pos.y * center_force_strength;
        total_force.z -= pos.z * center_force_strength * 0.5f;
        
        // Clamp total force to prevent instability
        total_force = clamp3(total_force, 1000.0f);
        
        // Update velocity with damping
        vel.x = vel.x * (1.0f - params.damping) + (total_force.x / mass) * params.dt;
        vel.y = vel.y * (1.0f - params.damping) + (total_force.y / mass) * params.dt;
        vel.z = vel.z * (1.0f - params.damping) + (total_force.z / mass) * params.dt;
        
        // Clamp velocity
        vel = clamp3(vel, params.max_velocity);
        
        // Update position
        pos.x += vel.x * params.dt;
        pos.y += vel.y * params.dt;
        pos.z += vel.z * params.dt;
        
        // Hard boundary clamping as final safety
        float hard_bound = params.viewport_bounds * 1.2f;
        pos.x = fmaxf(-hard_bound, fminf(hard_bound, pos.x));
        pos.y = fmaxf(-hard_bound, fminf(hard_bound, pos.y));
        pos.z = fmaxf(-hard_bound * 0.5f, fminf(hard_bound * 0.5f, pos.z));
        
        // Write back results
        nodes[i].position = make_v3(pos);
        nodes[i].velocity = make_v3(vel);
    }

    // Entry point for testing compilation
    void launch_advanced_forces(
        void* nodes,
        void* edges,
        int num_nodes,
        int num_edges,
        void* constraints,
        int num_constraints,
        void* params
    ) {
        dim3 block_size(256);
        dim3 grid_size((num_nodes + block_size.x - 1) / block_size.x);
        
        advanced_forces_kernel<<<grid_size, block_size>>>(
            (EnhancedBinaryNodeData*)nodes,
            (EnhancedEdgeData*)edges,
            num_nodes,
            num_edges,
            (ConstraintData*)constraints,
            num_constraints,
            *(AdvancedSimulationParams*)params
        );
    }
}