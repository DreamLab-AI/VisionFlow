// VisionFlow Unified GPU Kernel - Single kernel for all physics computations
// No external dependencies, pure CUDA C for maximum compatibility
// Supports basic forces, dual graphs, constraints, and visual analytics

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// =============================================================================
// Core Data Structures (Structure of Arrays for coalescing)
// =============================================================================

struct SimParams {
    // Force parameters
    float spring_k;
    float repel_k;
    float damping;
    float dt;
    float max_velocity;
    float max_force;
    
    // Stress majorization
    float stress_weight;
    float stress_alpha;
    
    // Constraints
    float separation_radius;
    float boundary_limit;
    float alignment_strength;
    float cluster_strength;
    
    // System
    float viewport_bounds;
    float temperature;
    int iteration;
    int compute_mode;  // 0=basic, 1=dual, 2=constraints, 3=analytics
};

struct ConstraintData {
    int type;         // 0=none, 1=separation, 2=boundary, 3=alignment, 4=cluster
    float strength;
    float param1;
    float param2;
    int node_mask;    // Bit mask for which nodes are affected
};

// =============================================================================
// Device Helper Functions
// =============================================================================

__device__ inline float3 make_vec3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ inline float vec3_length(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float3 vec3_normalize(float3 v) {
    float len = vec3_length(v) + 1e-8f;
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ inline float3 vec3_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 vec3_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 vec3_scale(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float vec3_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 vec3_clamp(float3 v, float limit) {
    float len = vec3_length(v);
    if (len > limit) {
        return vec3_scale(vec3_normalize(v), limit);
    }
    return v;
}

// =============================================================================
// Basic Force Computation
// =============================================================================

__device__ float3 compute_basic_forces(
    int idx,
    float* pos_x, float* pos_y, float* pos_z,
    int* edge_src, int* edge_dst, float* edge_weight,
    int num_nodes, int num_edges,
    SimParams params
) {
    const float MIN_DISTANCE = 0.15f;  // Minimum separation between nodes
    const float MAX_REPULSION_DIST = 50.0f;  // Cutoff for repulsion
    
    float3 my_pos = make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    float3 total_force = make_vec3(0.0f, 0.0f, 0.0f);
    
    // Repulsive forces (all nodes)
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        
        float3 other_pos = make_vec3(pos_x[j], pos_y[j], pos_z[j]);
        float3 diff = vec3_sub(my_pos, other_pos);
        float dist = vec3_length(diff);
        
        // Enforce minimum distance to prevent collapse
        if (dist < MIN_DISTANCE) {
            // Generate deterministic push direction for coincident nodes
            float3 push_dir;
            if (vec3_dot(diff, diff) > 0.0001f) {
                push_dir = vec3_normalize(diff);
            } else {
                // Use node indices to create unique separation direction
                float angle = (float)(idx - j) * 0.618034f; // Golden ratio
                push_dir = make_vec3(cosf(angle), sinf(angle), 0.1f * (idx - j));
                push_dir = vec3_normalize(push_dir);
            }
            // Strong but controlled repulsion when too close
            float push_force = params.repel_k * (MIN_DISTANCE - dist + 1.0f) / (MIN_DISTANCE * MIN_DISTANCE);
            total_force = vec3_add(total_force, vec3_scale(push_dir, push_force));
        } else if (dist < MAX_REPULSION_DIST) {
            // Normal repulsion with safer distance calculation
            float dist_sq = fmaxf(dist * dist, MIN_DISTANCE * MIN_DISTANCE);
            float repulsion = params.repel_k / dist_sq;
            total_force = vec3_add(total_force, vec3_scale(vec3_normalize(diff), repulsion));
        }
    }
    
    // Attractive forces (connected nodes via edges)
    for (int e = 0; e < num_edges; e++) {
        int src = edge_src[e];
        int dst = edge_dst[e];
        
        if (src == idx || dst == idx) {
            int other = (src == idx) ? dst : src;
            float3 other_pos = make_vec3(pos_x[other], pos_y[other], pos_z[other]);
            float3 diff = vec3_sub(other_pos, my_pos);
            float dist = vec3_length(diff);
            
            if (dist > MIN_DISTANCE) {
                // Spring force with natural length
                float natural_length = 10.0f;  // Increased ideal edge length
                float displacement = dist - natural_length;
                // Limit spring force to prevent instability
                float spring_force = params.spring_k * displacement * edge_weight[e];
                spring_force = fmaxf(-params.max_force * 0.5f, fminf(params.max_force * 0.5f, spring_force));
                total_force = vec3_add(total_force, vec3_scale(vec3_normalize(diff), spring_force));
            }
        }
    }
    
    return total_force;
}

// =============================================================================
// Dual Graph Forces (Knowledge + Agent graphs)
// =============================================================================

__device__ float3 compute_dual_graph_forces(
    int idx,
    float* pos_x, float* pos_y, float* pos_z,
    int* node_graph_id,  // 0=knowledge, 1=agent
    int* edge_src, int* edge_dst, float* edge_weight, int* edge_graph_id,
    int num_nodes, int num_edges,
    SimParams params
) {
    float3 my_pos = make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    float3 total_force = make_vec3(0.0f, 0.0f, 0.0f);
    int my_graph = node_graph_id[idx];
    
    // Different repulsion based on graph membership
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        
        float3 other_pos = make_vec3(pos_x[j], pos_y[j], pos_z[j]);
        float3 diff = vec3_sub(my_pos, other_pos);
        float dist_sq = vec3_dot(diff, diff) + 0.01f;
        
        // Stronger repulsion within same graph, weaker across graphs
        float repel_scale = (node_graph_id[j] == my_graph) ? 1.0f : 0.3f;
        
        if (dist_sq < 100.0f) {
            float repulsion = params.repel_k * repel_scale / dist_sq;
            total_force = vec3_add(total_force, vec3_scale(vec3_normalize(diff), repulsion));
        }
    }
    
    // Graph-aware attraction
    for (int e = 0; e < num_edges; e++) {
        if (edge_src[e] == idx || edge_dst[e] == idx) {
            int other = (edge_src[e] == idx) ? edge_dst[e] : edge_src[e];
            float3 other_pos = make_vec3(pos_x[other], pos_y[other], pos_z[other]);
            float3 diff = vec3_sub(other_pos, my_pos);
            float dist = vec3_length(diff);
            
            // Different spring constants for intra vs inter-graph edges
            float spring_scale = (edge_graph_id[e] == my_graph) ? 1.0f : 0.5f;
            
            if (dist > 0.01f) {
                float attraction = params.spring_k * spring_scale * dist * edge_weight[e];
                total_force = vec3_add(total_force, vec3_scale(vec3_normalize(diff), attraction));
            }
        }
    }
    
    return total_force;
}

// =============================================================================
// Constraint Application
// =============================================================================

__device__ float3 apply_constraints(
    int idx,
    float3 position,
    float3 force,
    ConstraintData* constraints,
    int num_constraints,
    float* pos_x, float* pos_y, float* pos_z,
    int num_nodes,
    SimParams params
) {
    float3 constraint_force = make_vec3(0.0f, 0.0f, 0.0f);
    
    for (int c = 0; c < num_constraints; c++) {
        ConstraintData constraint = constraints[c];
        
        // Check if this node is affected by constraint
        if (!(constraint.node_mask & (1 << (idx % 32)))) continue;
        
        switch (constraint.type) {
            case 1: // Separation constraint
                for (int j = 0; j < num_nodes; j++) {
                    if (j == idx) continue;
                    float3 other_pos = make_vec3(pos_x[j], pos_y[j], pos_z[j]);
                    float3 diff = vec3_sub(position, other_pos);
                    float dist = vec3_length(diff);
                    
                    if (dist < constraint.param1) {  // param1 = min separation
                        float push = (constraint.param1 - dist) * constraint.strength;
                        constraint_force = vec3_add(constraint_force, 
                                                   vec3_scale(vec3_normalize(diff), push));
                    }
                }
                break;
                
            case 2: // Boundary constraint
                if (fabsf(position.x) > constraint.param1) {
                    constraint_force.x -= (position.x - copysignf(constraint.param1, position.x)) * constraint.strength;
                }
                if (fabsf(position.y) > constraint.param1) {
                    constraint_force.y -= (position.y - copysignf(constraint.param1, position.y)) * constraint.strength;
                }
                if (fabsf(position.z) > constraint.param2) {  // param2 = z boundary
                    constraint_force.z -= (position.z - copysignf(constraint.param2, position.z)) * constraint.strength;
                }
                break;
                
            case 3: // Alignment constraint (horizontal)
                constraint_force.y -= position.y * constraint.strength;
                break;
                
            case 4: // Cluster constraint
                // Pull towards cluster center (param1, param2 encode center xy)
                float3 center = make_vec3(constraint.param1, constraint.param2, 0.0f);
                float3 to_center = vec3_sub(center, position);
                constraint_force = vec3_add(constraint_force, 
                                           vec3_scale(to_center, constraint.strength));
                break;
        }
    }
    
    return vec3_add(force, constraint_force);
}

// =============================================================================
// Visual Analytics Mode (simplified)
// =============================================================================

__device__ float3 compute_visual_analytics(
    int idx,
    float* pos_x, float* pos_y, float* pos_z,
    float* node_importance,
    float* node_temporal,
    int* node_cluster,
    int* edge_src, int* edge_dst, float* edge_weight,
    int num_nodes, int num_edges,
    SimParams params
) {
    float3 my_pos = make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    float3 total_force = make_vec3(0.0f, 0.0f, 0.0f);
    float my_importance = node_importance[idx];
    int my_cluster = node_cluster[idx];
    
    // Importance-weighted repulsion
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        
        float3 other_pos = make_vec3(pos_x[j], pos_y[j], pos_z[j]);
        float3 diff = vec3_sub(my_pos, other_pos);
        float dist_sq = vec3_dot(diff, diff) + 0.01f;
        
        // Scale repulsion by importance difference
        float importance_scale = 1.0f + fabsf(my_importance - node_importance[j]);
        
        // Reduce repulsion within same cluster
        float cluster_scale = (node_cluster[j] == my_cluster) ? 0.5f : 1.0f;
        
        if (dist_sq < 100.0f) {
            float repulsion = params.repel_k * importance_scale * cluster_scale / dist_sq;
            total_force = vec3_add(total_force, vec3_scale(vec3_normalize(diff), repulsion));
        }
    }
    
    // Temporal coherence force (attract to previous positions based on temporal weight)
    float temporal_strength = node_temporal[idx] * params.stress_weight;
    float3 temporal_force = vec3_scale(my_pos, -temporal_strength);
    total_force = vec3_add(total_force, temporal_force);
    
    // Standard edge attractions
    for (int e = 0; e < num_edges; e++) {
        if (edge_src[e] == idx || edge_dst[e] == idx) {
            int other = (edge_src[e] == idx) ? edge_dst[e] : edge_src[e];
            float3 other_pos = make_vec3(pos_x[other], pos_y[other], pos_z[other]);
            float3 diff = vec3_sub(other_pos, my_pos);
            float dist = vec3_length(diff);
            
            if (dist > 0.01f) {
                // Scale by importance sum
                float importance_sum = my_importance + node_importance[other];
                float attraction = params.spring_k * dist * edge_weight[e] * importance_sum;
                total_force = vec3_add(total_force, vec3_scale(vec3_normalize(diff), attraction));
            }
        }
    }
    
    return total_force;
}

// =============================================================================
// Main Unified Kernel
// =============================================================================

struct GpuNodeData {
    float* pos_x; float* pos_y; float* pos_z;
    float* vel_x; float* vel_y; float* vel_z;
    float* mass;
    float* importance;
    float* temporal;
    int* graph_id;
    int* cluster;
};

struct GpuEdgeData {
    int* src;
    int* dst;
    float* weight;
    int* graph_id;
};

struct GpuKernelParams {
    GpuNodeData nodes;
    GpuEdgeData edges;
    ConstraintData* constraints;
    SimParams params;
    int num_nodes;
    int num_edges;
    int num_constraints;
};

__global__ void visionflow_compute_kernel(GpuKernelParams p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.num_nodes) return;
    
    // Load current position and velocity
    float3 position = make_vec3(p.nodes.pos_x[idx], p.nodes.pos_y[idx], p.nodes.pos_z[idx]);
    float3 velocity = make_vec3(p.nodes.vel_x[idx], p.nodes.vel_y[idx], p.nodes.vel_z[idx]);
    
    // Compute forces based on mode
    float3 force = make_vec3(0.0f, 0.0f, 0.0f);
    
    switch (p.params.compute_mode) {
        case 0: // Basic force-directed
            force = compute_basic_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            break;
            
        case 1: // Dual graph mode
            force = compute_dual_graph_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z, p.nodes.graph_id,
                p.edges.src, p.edges.dst, p.edges.weight, p.edges.graph_id,
                p.num_nodes, p.num_edges, p.params
            );
            break;
            
        case 2: // With constraints
            force = compute_basic_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            force = apply_constraints(
                idx, position, force, p.constraints, p.num_constraints,
                p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z, p.num_nodes, p.params
            );
            break;
            
        case 3: // Visual analytics
            force = compute_visual_analytics(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.nodes.importance, p.nodes.temporal, p.nodes.cluster,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            break;
            
        default:
            // Fallback to basic
            force = compute_basic_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            break;
    }
    
    // Clamp force magnitude BEFORE any scaling
    force = vec3_clamp(force, p.params.max_force);
    
    // Progressive warmup for stability
    if (p.params.iteration < 200) {
        float warmup = p.params.iteration / 200.0f;
        force = vec3_scale(force, warmup * warmup); // Quadratic warmup
        
        // Extra damping in early iterations
        float extra_damping = 0.98f - 0.13f * warmup; // From 0.98 to 0.85
        p.params.damping = fmaxf(p.params.damping, extra_damping);
    }
    
    // Temperature-based scaling (simulated annealing) - gentler cooling
    float temp_scale = p.params.temperature / (1.0f + p.params.iteration * 0.0001f);
    force = vec3_scale(force, temp_scale);
    
    // Update velocity with damping
    float mass = (p.nodes.mass != nullptr && p.nodes.mass[idx] > 0.0f) ? p.nodes.mass[idx] : 1.0f;
    velocity = vec3_add(velocity, vec3_scale(force, p.params.dt / mass));
    velocity = vec3_scale(velocity, p.params.damping);
    velocity = vec3_clamp(velocity, p.params.max_velocity);
    
    // Zero velocity in very first iterations to prevent explosion
    if (p.params.iteration < 5) {
        velocity = make_vec3(0.0f, 0.0f, 0.0f);
    }
    
    // Update position
    position = vec3_add(position, vec3_scale(velocity, p.params.dt));
    
    // Apply viewport bounds
    position.x = fmaxf(-p.params.viewport_bounds, fminf(p.params.viewport_bounds, position.x));
    position.y = fmaxf(-p.params.viewport_bounds, fminf(p.params.viewport_bounds, position.y));
    position.z = fmaxf(-p.params.viewport_bounds, fminf(p.params.viewport_bounds, position.z));
    
    // Write back results
    p.nodes.pos_x[idx] = position.x;
    p.nodes.pos_y[idx] = position.y;
    p.nodes.pos_z[idx] = position.z;
    p.nodes.vel_x[idx] = velocity.x;
    p.nodes.vel_y[idx] = velocity.y;
    p.nodes.vel_z[idx] = velocity.z;
}

// =============================================================================
// Stress Majorization Kernel (separate for iterative optimization)
// =============================================================================

__global__ void stress_majorization_kernel(
    float* pos_x, float* pos_y, float* pos_z,
    float* ideal_distances,  // N x N matrix
    float* weight_matrix,     // N x N matrix  
    SimParams params,
    int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    float3 position = make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    float3 new_position = make_vec3(0.0f, 0.0f, 0.0f);
    float total_weight = 0.0f;
    
    // Stress majorization step
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        
        int matrix_idx = idx * num_nodes + j;
        float ideal_dist = ideal_distances[matrix_idx];
        float weight = weight_matrix[matrix_idx];
        
        if (weight > 0.0f) {
            float3 other_pos = make_vec3(pos_x[j], pos_y[j], pos_z[j]);
            float3 diff = vec3_sub(other_pos, position);
            float current_dist = vec3_length(diff) + 0.001f;
            
            float ratio = ideal_dist / current_dist;
            new_position = vec3_add(new_position, vec3_scale(other_pos, weight * ratio));
            total_weight += weight;
        }
    }
    
    if (total_weight > 0.0f) {
        new_position = vec3_scale(new_position, 1.0f / total_weight);
        
        // Blend with current position
        position = vec3_add(
            vec3_scale(position, 1.0f - params.stress_alpha),
            vec3_scale(new_position, params.stress_alpha)
        );
        
        // Write back
        pos_x[idx] = position.x;
        pos_y[idx] = position.y;
        pos_z[idx] = position.z;
    }
}

} // extern "C"