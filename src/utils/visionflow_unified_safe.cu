// VisionFlow Unified GPU Kernel - Safe Version with Comprehensive Bounds Checking
// Enhanced with extensive safety measures, overflow protection, and robust error handling
// No external dependencies, pure CUDA C for maximum compatibility

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

extern "C" {

// =============================================================================
// Safety Constants and Macros
// =============================================================================

#define SAFE_MAX_NODES 10000000       // 10M nodes maximum
#define SAFE_MAX_EDGES 50000000       // 50M edges maximum
#define SAFE_MAX_FORCE 1000.0f        // Maximum force magnitude
#define SAFE_MAX_VELOCITY 100.0f      // Maximum velocity magnitude
#define SAFE_MAX_POSITION 1000000.0f  // Maximum position coordinate
#define SAFE_MIN_DISTANCE 0.01f       // Minimum distance to prevent division by zero
#define SAFE_MAX_ITERATIONS 100000    // Maximum iterations

// Safe bounds checking macro
#define SAFE_BOUNDS_CHECK(idx, max_size) \
    if ((idx) < 0 || (idx) >= (max_size)) return make_float3(0.0f, 0.0f, 0.0f)

// Safe value checking macro
#define SAFE_VALUE_CHECK(val) \
    (isfinite(val) && fabsf(val) < SAFE_MAX_POSITION)

// Safe pointer checking macro
#define SAFE_PTR_CHECK(ptr) \
    if ((ptr) == nullptr) return make_float3(0.0f, 0.0f, 0.0f)

// =============================================================================
// Enhanced Data Structures with Safety
// =============================================================================

struct SafeSimParams {
    // Force parameters with bounds
    float spring_k;           // [0.0001, 10.0]
    float repel_k;            // [0.01, 1000.0]
    float damping;            // [0.1, 0.99]
    float dt;                 // [0.001, 0.1]
    float max_velocity;       // [0.1, SAFE_MAX_VELOCITY]
    float max_force;          // [0.1, SAFE_MAX_FORCE]
    
    // Stress majorization
    float stress_weight;      // [0.0, 10.0]
    float stress_alpha;       // [0.0, 1.0]
    
    // Constraints
    float separation_radius;  // [0.01, 100.0]
    float boundary_limit;     // [1.0, SAFE_MAX_POSITION]
    float alignment_strength; // [0.0, 10.0]
    float cluster_strength;   // [0.0, 10.0]
    
    // Boundary control
    float boundary_damping;   // [0.1, 0.99]
    
    // System
    float viewport_bounds;    // [0.0, SAFE_MAX_POSITION]
    float temperature;        // [0.0, 10.0]
    float max_repulsion_dist; // [1.0, 10000.0]
    int iteration;            // [0, SAFE_MAX_ITERATIONS]
    int compute_mode;         // [0, 3]
};

struct SafeConstraintData {
    int type;                 // [0, 4]
    float strength;           // [0.0, 100.0]
    float param1;             // [-SAFE_MAX_POSITION, SAFE_MAX_POSITION]
    float param2;             // [-SAFE_MAX_POSITION, SAFE_MAX_POSITION]
    int node_mask;            // Bit mask for affected nodes
};

// =============================================================================
// Enhanced Helper Functions with Safety Checks
// =============================================================================

__device__ inline float safe_clamp(float value, float min_val, float max_val) {
    if (!isfinite(value)) return 0.0f;
    return fmaxf(min_val, fminf(max_val, value));
}

__device__ inline float3 safe_make_vec3(float x, float y, float z) {
    return make_float3(
        safe_clamp(x, -SAFE_MAX_POSITION, SAFE_MAX_POSITION),
        safe_clamp(y, -SAFE_MAX_POSITION, SAFE_MAX_POSITION),
        safe_clamp(z, -SAFE_MAX_POSITION, SAFE_MAX_POSITION)
    );
}

__device__ inline float safe_vec3_length(float3 v) {
    if (!SAFE_VALUE_CHECK(v.x) || !SAFE_VALUE_CHECK(v.y) || !SAFE_VALUE_CHECK(v.z)) {
        return 0.0f;
    }
    float len_sq = v.x * v.x + v.y * v.y + v.z * v.z;
    return sqrtf(fmaxf(len_sq, SAFE_MIN_DISTANCE * SAFE_MIN_DISTANCE));
}

__device__ inline float3 safe_vec3_normalize(float3 v) {
    float len = safe_vec3_length(v);
    if (len < SAFE_MIN_DISTANCE) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ inline float3 safe_vec3_add(float3 a, float3 b) {
    return safe_make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 safe_vec3_sub(float3 a, float3 b) {
    return safe_make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 safe_vec3_scale(float3 v, float s) {
    if (!isfinite(s) || fabsf(s) > 1000.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return safe_make_vec3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float safe_vec3_dot(float3 a, float3 b) {
    if (!SAFE_VALUE_CHECK(a.x) || !SAFE_VALUE_CHECK(a.y) || !SAFE_VALUE_CHECK(a.z) ||
        !SAFE_VALUE_CHECK(b.x) || !SAFE_VALUE_CHECK(b.y) || !SAFE_VALUE_CHECK(b.z)) {
        return 0.0f;
    }
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 safe_vec3_clamp(float3 v, float limit) {
    if (!isfinite(limit) || limit <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float len = safe_vec3_length(v);
    if (len > limit) {
        return safe_vec3_scale(safe_vec3_normalize(v), limit);
    }
    return v;
}

// Validate simulation parameters
__device__ inline bool validate_sim_params(SafeSimParams params) {
    return (params.spring_k >= 0.0001f && params.spring_k <= 10.0f) &&
           (params.repel_k >= 0.01f && params.repel_k <= 1000.0f) &&
           (params.damping >= 0.1f && params.damping <= 0.99f) &&
           (params.dt >= 0.001f && params.dt <= 0.1f) &&
           (params.max_velocity >= 0.1f && params.max_velocity <= SAFE_MAX_VELOCITY) &&
           (params.max_force >= 0.1f && params.max_force <= SAFE_MAX_FORCE) &&
           (params.iteration >= 0 && params.iteration <= SAFE_MAX_ITERATIONS) &&
           (params.compute_mode >= 0 && params.compute_mode <= 3) &&
           (params.viewport_bounds >= 0.0f && params.viewport_bounds <= SAFE_MAX_POSITION);
}

// =============================================================================
// Safe Basic Force Computation
// =============================================================================

__device__ float3 safe_compute_basic_forces(
    int idx,
    float* pos_x, float* pos_y, float* pos_z,
    int* edge_src, int* edge_dst, float* edge_weight,
    int num_nodes, int num_edges,
    SafeSimParams params
) {
    // Validate input parameters
    if (!validate_sim_params(params)) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    SAFE_BOUNDS_CHECK(idx, num_nodes);
    SAFE_PTR_CHECK(pos_x);
    SAFE_PTR_CHECK(pos_y);
    SAFE_PTR_CHECK(pos_z);
    
    // Additional bounds checks
    if (num_nodes <= 0 || num_nodes > SAFE_MAX_NODES || 
        num_edges < 0 || num_edges > SAFE_MAX_EDGES) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Load and validate position
    if (!SAFE_VALUE_CHECK(pos_x[idx]) || !SAFE_VALUE_CHECK(pos_y[idx]) || !SAFE_VALUE_CHECK(pos_z[idx])) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 my_pos = safe_make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    
    const float MIN_DISTANCE = fmaxf(SAFE_MIN_DISTANCE, 0.15f);
    const float MAX_REPULSION_DIST = safe_clamp(params.max_repulsion_dist, 1.0f, 10000.0f);
    
    // Repulsive forces (all nodes) with safety bounds
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        
        // Validate other node position
        if (!SAFE_VALUE_CHECK(pos_x[j]) || !SAFE_VALUE_CHECK(pos_y[j]) || !SAFE_VALUE_CHECK(pos_z[j])) {
            continue;
        }
        
        float3 other_pos = safe_make_vec3(pos_x[j], pos_y[j], pos_z[j]);
        float3 diff = safe_vec3_sub(my_pos, other_pos);
        float dist = safe_vec3_length(diff);
        
        // Enforce minimum distance to prevent collapse
        if (dist < MIN_DISTANCE) {
            // Generate deterministic push direction for coincident nodes
            float3 push_dir;
            if (safe_vec3_dot(diff, diff) > 0.0001f) {
                push_dir = safe_vec3_normalize(diff);
            } else {
                // Use node indices to create unique separation direction
                float angle = (float)(idx - j) * 0.618034f; // Golden ratio
                push_dir = safe_make_vec3(cosf(angle), sinf(angle), 0.1f * (idx - j));
                push_dir = safe_vec3_normalize(push_dir);
            }
            
            // Strong but controlled repulsion when too close
            float push_force = safe_clamp(
                params.repel_k * (MIN_DISTANCE - dist + 1.0f) / (MIN_DISTANCE * MIN_DISTANCE),
                0.0f, params.max_force * 0.5f
            );
            total_force = safe_vec3_add(total_force, safe_vec3_scale(push_dir, push_force));
        } else if (dist < MAX_REPULSION_DIST) {
            // Smooth distance-based decay for repulsion
            float dist_sq = fmaxf(dist * dist, MIN_DISTANCE * MIN_DISTANCE);
            
            // Add smooth decay: force decreases as we approach MAX_REPULSION_DIST
            float decay_factor = 1.0f - (dist / MAX_REPULSION_DIST);
            decay_factor = safe_clamp(decay_factor * decay_factor, 0.0f, 1.0f); // Quadratic decay
            
            float repulsion = safe_clamp(
                params.repel_k * decay_factor / dist_sq,
                0.0f, params.max_force
            );
            total_force = safe_vec3_add(total_force, safe_vec3_scale(safe_vec3_normalize(diff), repulsion));
        }
        
        // Safety check: prevent force accumulation overflow
        if (safe_vec3_length(total_force) > params.max_force) {
            total_force = safe_vec3_clamp(total_force, params.max_force);
            break; // Stop adding more forces
        }
    }
    
    // Attractive forces (connected nodes via edges) with bounds checking
    if (edge_src != nullptr && edge_dst != nullptr && edge_weight != nullptr) {
        for (int e = 0; e < num_edges; e++) {
            // Validate edge indices
            if (edge_src[e] < 0 || edge_src[e] >= num_nodes || 
                edge_dst[e] < 0 || edge_dst[e] >= num_nodes) {
                continue; // Skip invalid edges
            }
            
            int src = edge_src[e];
            int dst = edge_dst[e];
            
            if (src == idx || dst == idx) {
                int other = (src == idx) ? dst : src;
                
                // Validate other node position
                if (!SAFE_VALUE_CHECK(pos_x[other]) || !SAFE_VALUE_CHECK(pos_y[other]) || !SAFE_VALUE_CHECK(pos_z[other])) {
                    continue;
                }
                
                // Validate edge weight
                if (!isfinite(edge_weight[e]) || edge_weight[e] < 0.0f || edge_weight[e] > 1000.0f) {
                    continue;
                }
                
                float3 other_pos = safe_make_vec3(pos_x[other], pos_y[other], pos_z[other]);
                float3 diff = safe_vec3_sub(other_pos, my_pos);
                float dist = safe_vec3_length(diff);
                
                if (dist > MIN_DISTANCE) {
                    // Spring force with natural length
                    float natural_length = safe_clamp(params.separation_radius * 5.0f, 1.0f, 100.0f);
                    float displacement = dist - natural_length;
                    
                    // Limit spring force to prevent instability
                    float spring_force = safe_clamp(
                        params.spring_k * displacement * safe_clamp(edge_weight[e], 0.0f, 10.0f),
                        -params.max_force * 0.5f, params.max_force * 0.5f
                    );
                    
                    total_force = safe_vec3_add(total_force, safe_vec3_scale(safe_vec3_normalize(diff), spring_force));
                }
            }
            
            // Safety check: prevent force accumulation overflow
            if (safe_vec3_length(total_force) > params.max_force) {
                total_force = safe_vec3_clamp(total_force, params.max_force);
                break;
            }
        }
    }
    
    // Final safety clamp
    return safe_vec3_clamp(total_force, params.max_force);
}

// =============================================================================
// Safe Dual Graph Forces
// =============================================================================

__device__ float3 safe_compute_dual_graph_forces(
    int idx,
    float* pos_x, float* pos_y, float* pos_z,
    int* node_graph_id,
    int* edge_src, int* edge_dst, float* edge_weight, int* edge_graph_id,
    int num_nodes, int num_edges,
    SafeSimParams params
) {
    // Validate input parameters
    if (!validate_sim_params(params)) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    SAFE_BOUNDS_CHECK(idx, num_nodes);
    SAFE_PTR_CHECK(pos_x);
    SAFE_PTR_CHECK(pos_y);
    SAFE_PTR_CHECK(pos_z);
    SAFE_PTR_CHECK(node_graph_id);
    
    if (num_nodes <= 0 || num_nodes > SAFE_MAX_NODES || 
        num_edges < 0 || num_edges > SAFE_MAX_EDGES) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Validate position and graph ID
    if (!SAFE_VALUE_CHECK(pos_x[idx]) || !SAFE_VALUE_CHECK(pos_y[idx]) || !SAFE_VALUE_CHECK(pos_z[idx])) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    int my_graph = node_graph_id[idx];
    if (my_graph < 0 || my_graph > 10) { // Reasonable graph ID limit
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 my_pos = safe_make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    
    const float MIN_DISTANCE = fmaxf(SAFE_MIN_DISTANCE, 0.15f);
    
    // Different repulsion based on graph membership with safety
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        
        // Validate other node
        if (!SAFE_VALUE_CHECK(pos_x[j]) || !SAFE_VALUE_CHECK(pos_y[j]) || !SAFE_VALUE_CHECK(pos_z[j])) {
            continue;
        }
        
        int other_graph = node_graph_id[j];
        if (other_graph < 0 || other_graph > 10) {
            continue;
        }
        
        float3 other_pos = safe_make_vec3(pos_x[j], pos_y[j], pos_z[j]);
        float3 diff = safe_vec3_sub(my_pos, other_pos);
        float dist_sq = safe_vec3_dot(diff, diff) + 0.01f;
        
        // Stronger repulsion within same graph, weaker across graphs
        float repel_scale = (other_graph == my_graph) ? 1.0f : 0.3f;
        
        if (dist_sq < 100.0f) {
            float repulsion = safe_clamp(
                params.repel_k * repel_scale / dist_sq,
                0.0f, params.max_force
            );
            total_force = safe_vec3_add(total_force, safe_vec3_scale(safe_vec3_normalize(diff), repulsion));
        }
        
        // Safety check
        if (safe_vec3_length(total_force) > params.max_force) {
            total_force = safe_vec3_clamp(total_force, params.max_force);
            break;
        }
    }
    
    // Graph-aware attraction with safety
    if (edge_src != nullptr && edge_dst != nullptr && edge_weight != nullptr && edge_graph_id != nullptr) {
        for (int e = 0; e < num_edges; e++) {
            // Validate edge
            if (edge_src[e] < 0 || edge_src[e] >= num_nodes || 
                edge_dst[e] < 0 || edge_dst[e] >= num_nodes) {
                continue;
            }
            
            if (edge_src[e] == idx || edge_dst[e] == idx) {
                int other = (edge_src[e] == idx) ? edge_dst[e] : edge_src[e];
                
                // Validate other node
                if (!SAFE_VALUE_CHECK(pos_x[other]) || !SAFE_VALUE_CHECK(pos_y[other]) || !SAFE_VALUE_CHECK(pos_z[other])) {
                    continue;
                }
                
                // Validate edge data
                if (!isfinite(edge_weight[e]) || edge_weight[e] < 0.0f || edge_weight[e] > 1000.0f) {
                    continue;
                }
                
                int edge_graph = edge_graph_id[e];
                if (edge_graph < 0 || edge_graph > 10) {
                    continue;
                }
                
                float3 other_pos = safe_make_vec3(pos_x[other], pos_y[other], pos_z[other]);
                float3 diff = safe_vec3_sub(other_pos, my_pos);
                float dist = safe_vec3_length(diff);
                
                // Different spring constants for intra vs inter-graph edges
                float spring_scale = (edge_graph == my_graph) ? 1.0f : 0.5f;
                
                if (dist > MIN_DISTANCE) {
                    float natural_length = safe_clamp(params.separation_radius * 5.0f, 1.0f, 100.0f);
                    float displacement = dist - natural_length;
                    
                    float spring_force = safe_clamp(
                        params.spring_k * spring_scale * displacement * safe_clamp(edge_weight[e], 0.0f, 10.0f),
                        -params.max_force * 0.5f, params.max_force * 0.5f
                    );
                    
                    total_force = safe_vec3_add(total_force, safe_vec3_scale(safe_vec3_normalize(diff), spring_force));
                }
            }
            
            // Safety check
            if (safe_vec3_length(total_force) > params.max_force) {
                total_force = safe_vec3_clamp(total_force, params.max_force);
                break;
            }
        }
    }
    
    return safe_vec3_clamp(total_force, params.max_force);
}

// =============================================================================
// Safe Constraint Application
// =============================================================================

__device__ float3 safe_apply_constraints(
    int idx,
    float3 position,
    float3 force,
    SafeConstraintData* constraints,
    int num_constraints,
    float* pos_x, float* pos_y, float* pos_z,
    int num_nodes,
    SafeSimParams params
) {
    SAFE_BOUNDS_CHECK(idx, num_nodes);
    
    if (constraints == nullptr || num_constraints <= 0 || num_constraints > 1000) {
        return force; // No constraints or invalid constraint count
    }
    
    if (num_nodes <= 0 || num_nodes > SAFE_MAX_NODES) {
        return force;
    }
    
    // Validate position
    if (!SAFE_VALUE_CHECK(position.x) || !SAFE_VALUE_CHECK(position.y) || !SAFE_VALUE_CHECK(position.z)) {
        return force;
    }
    
    float3 constraint_force = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int c = 0; c < num_constraints; c++) {
        SafeConstraintData constraint = constraints[c];
        
        // Validate constraint
        if (constraint.type < 0 || constraint.type > 4) continue;
        if (!isfinite(constraint.strength) || constraint.strength < 0.0f || constraint.strength > 100.0f) continue;
        if (!SAFE_VALUE_CHECK(constraint.param1) || !SAFE_VALUE_CHECK(constraint.param2)) continue;
        
        // Check if this node is affected by constraint
        if (!(constraint.node_mask & (1 << (idx % 32)))) continue;
        
        switch (constraint.type) {
            case 1: // Separation constraint
                for (int j = 0; j < num_nodes; j++) {
                    if (j == idx) continue;
                    
                    // Validate other node
                    if (!SAFE_VALUE_CHECK(pos_x[j]) || !SAFE_VALUE_CHECK(pos_y[j]) || !SAFE_VALUE_CHECK(pos_z[j])) {
                        continue;
                    }
                    
                    float3 other_pos = safe_make_vec3(pos_x[j], pos_y[j], pos_z[j]);
                    float3 diff = safe_vec3_sub(position, other_pos);
                    float dist = safe_vec3_length(diff);
                    
                    float min_separation = safe_clamp(constraint.param1, 0.1f, 1000.0f);
                    if (dist < min_separation) {
                        float push = safe_clamp(
                            (min_separation - dist) * constraint.strength,
                            0.0f, params.max_force * 0.1f
                        );
                        constraint_force = safe_vec3_add(constraint_force, 
                                                       safe_vec3_scale(safe_vec3_normalize(diff), push));
                    }
                }
                break;
                
            case 2: // Boundary constraint
                {
                    float boundary_x = safe_clamp(constraint.param1, 1.0f, SAFE_MAX_POSITION);
                    float boundary_y = safe_clamp(constraint.param1, 1.0f, SAFE_MAX_POSITION);
                    float boundary_z = safe_clamp(constraint.param2, 1.0f, SAFE_MAX_POSITION);
                    
                    if (fabsf(position.x) > boundary_x) {
                        float force_x = safe_clamp(
                            -(position.x - copysignf(boundary_x, position.x)) * constraint.strength,
                            -params.max_force * 0.1f, params.max_force * 0.1f
                        );
                        constraint_force.x += force_x;
                    }
                    if (fabsf(position.y) > boundary_y) {
                        float force_y = safe_clamp(
                            -(position.y - copysignf(boundary_y, position.y)) * constraint.strength,
                            -params.max_force * 0.1f, params.max_force * 0.1f
                        );
                        constraint_force.y += force_y;
                    }
                    if (fabsf(position.z) > boundary_z) {
                        float force_z = safe_clamp(
                            -(position.z - copysignf(boundary_z, position.z)) * constraint.strength,
                            -params.max_force * 0.1f, params.max_force * 0.1f
                        );
                        constraint_force.z += force_z;
                    }
                }
                break;
                
            case 3: // Alignment constraint (horizontal)
                {
                    float align_force = safe_clamp(
                        -position.y * constraint.strength,
                        -params.max_force * 0.1f, params.max_force * 0.1f
                    );
                    constraint_force.y += align_force;
                }
                break;
                
            case 4: // Cluster constraint
                {
                    float3 center = safe_make_vec3(constraint.param1, constraint.param2, 0.0f);
                    float3 to_center = safe_vec3_sub(center, position);
                    float cluster_force = safe_clamp(constraint.strength, 0.0f, params.max_force * 0.1f);
                    constraint_force = safe_vec3_add(constraint_force, 
                                                   safe_vec3_scale(safe_vec3_normalize(to_center), cluster_force));
                }
                break;
        }
        
        // Safety check: prevent constraint force overflow
        if (safe_vec3_length(constraint_force) > params.max_force * 0.5f) {
            constraint_force = safe_vec3_clamp(constraint_force, params.max_force * 0.5f);
            break;
        }
    }
    
    // Safely combine forces
    float3 total_force = safe_vec3_add(force, constraint_force);
    return safe_vec3_clamp(total_force, params.max_force);
}

// =============================================================================
// Safe Visual Analytics Mode
// =============================================================================

__device__ float3 safe_compute_visual_analytics(
    int idx,
    float* pos_x, float* pos_y, float* pos_z,
    float* node_importance,
    float* node_temporal,
    int* node_cluster,
    int* edge_src, int* edge_dst, float* edge_weight,
    int num_nodes, int num_edges,
    SafeSimParams params
) {
    // Validate input parameters
    if (!validate_sim_params(params)) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    SAFE_BOUNDS_CHECK(idx, num_nodes);
    SAFE_PTR_CHECK(pos_x);
    SAFE_PTR_CHECK(pos_y);
    SAFE_PTR_CHECK(pos_z);
    
    if (num_nodes <= 0 || num_nodes > SAFE_MAX_NODES || 
        num_edges < 0 || num_edges > SAFE_MAX_EDGES) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Validate position
    if (!SAFE_VALUE_CHECK(pos_x[idx]) || !SAFE_VALUE_CHECK(pos_y[idx]) || !SAFE_VALUE_CHECK(pos_z[idx])) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    
    float3 my_pos = safe_make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    float3 total_force = make_float3(0.0f, 0.0f, 0.0f);
    
    // Safely get node attributes
    float my_importance = 1.0f;
    int my_cluster = 0;
    
    if (node_importance != nullptr && isfinite(node_importance[idx]) && node_importance[idx] >= 0.0f) {
        my_importance = safe_clamp(node_importance[idx], 0.0f, 100.0f);
    }
    
    if (node_cluster != nullptr && node_cluster[idx] >= 0 && node_cluster[idx] < 1000) {
        my_cluster = node_cluster[idx];
    }
    
    const float MIN_DISTANCE = fmaxf(SAFE_MIN_DISTANCE, 0.15f);
    
    // Importance-weighted repulsion with safety
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        
        // Validate other node
        if (!SAFE_VALUE_CHECK(pos_x[j]) || !SAFE_VALUE_CHECK(pos_y[j]) || !SAFE_VALUE_CHECK(pos_z[j])) {
            continue;
        }
        
        float other_importance = 1.0f;
        int other_cluster = 0;
        
        if (node_importance != nullptr && isfinite(node_importance[j]) && node_importance[j] >= 0.0f) {
            other_importance = safe_clamp(node_importance[j], 0.0f, 100.0f);
        }
        
        if (node_cluster != nullptr && node_cluster[j] >= 0 && node_cluster[j] < 1000) {
            other_cluster = node_cluster[j];
        }
        
        float3 other_pos = safe_make_vec3(pos_x[j], pos_y[j], pos_z[j]);
        float3 diff = safe_vec3_sub(my_pos, other_pos);
        float dist_sq = safe_vec3_dot(diff, diff) + 0.01f;
        
        // Scale repulsion by importance difference
        float importance_scale = 1.0f + safe_clamp(fabsf(my_importance - other_importance), 0.0f, 10.0f);
        
        // Reduce repulsion within same cluster
        float cluster_scale = (other_cluster == my_cluster) ? 0.5f : 1.0f;
        
        if (dist_sq < 100.0f) {
            float repulsion = safe_clamp(
                params.repel_k * importance_scale * cluster_scale / dist_sq,
                0.0f, params.max_force
            );
            total_force = safe_vec3_add(total_force, safe_vec3_scale(safe_vec3_normalize(diff), repulsion));
        }
        
        // Safety check
        if (safe_vec3_length(total_force) > params.max_force) {
            total_force = safe_vec3_clamp(total_force, params.max_force);
            break;
        }
    }
    
    // Temporal coherence force with safety
    if (node_temporal != nullptr && isfinite(node_temporal[idx])) {
        float temporal_strength = safe_clamp(
            node_temporal[idx] * params.stress_weight,
            0.0f, params.max_force * 0.1f
        );
        float3 temporal_force = safe_vec3_scale(my_pos, -temporal_strength);
        total_force = safe_vec3_add(total_force, temporal_force);
    }
    
    // Standard edge attractions with importance scaling
    if (edge_src != nullptr && edge_dst != nullptr && edge_weight != nullptr) {
        for (int e = 0; e < num_edges; e++) {
            // Validate edge
            if (edge_src[e] < 0 || edge_src[e] >= num_nodes || 
                edge_dst[e] < 0 || edge_dst[e] >= num_nodes) {
                continue;
            }
            
            if (edge_src[e] == idx || edge_dst[e] == idx) {
                int other = (edge_src[e] == idx) ? edge_dst[e] : edge_src[e];
                
                // Validate other node and edge weight
                if (!SAFE_VALUE_CHECK(pos_x[other]) || !SAFE_VALUE_CHECK(pos_y[other]) || !SAFE_VALUE_CHECK(pos_z[other])) {
                    continue;
                }
                
                if (!isfinite(edge_weight[e]) || edge_weight[e] < 0.0f || edge_weight[e] > 1000.0f) {
                    continue;
                }
                
                float other_importance = 1.0f;
                if (node_importance != nullptr && isfinite(node_importance[other]) && node_importance[other] >= 0.0f) {
                    other_importance = safe_clamp(node_importance[other], 0.0f, 100.0f);
                }
                
                float3 other_pos = safe_make_vec3(pos_x[other], pos_y[other], pos_z[other]);
                float3 diff = safe_vec3_sub(other_pos, my_pos);
                float dist = safe_vec3_length(diff);
                
                if (dist > MIN_DISTANCE) {
                    float natural_length = safe_clamp(params.separation_radius * 5.0f, 1.0f, 100.0f);
                    float displacement = dist - natural_length;
                    
                    // Scale by importance sum
                    float importance_sum = safe_clamp(my_importance + other_importance, 0.1f, 200.0f);
                    
                    float spring_force = safe_clamp(
                        params.spring_k * displacement * safe_clamp(edge_weight[e], 0.0f, 10.0f) * importance_sum,
                        -params.max_force * 0.5f, params.max_force * 0.5f
                    );
                    
                    total_force = safe_vec3_add(total_force, safe_vec3_scale(safe_vec3_normalize(diff), spring_force));
                }
            }
            
            // Safety check
            if (safe_vec3_length(total_force) > params.max_force) {
                total_force = safe_vec3_clamp(total_force, params.max_force);
                break;
            }
        }
    }
    
    return safe_vec3_clamp(total_force, params.max_force);
}

// =============================================================================
// Enhanced Main Kernel with Comprehensive Safety
// =============================================================================

struct SafeGpuNodeData {
    float* pos_x; float* pos_y; float* pos_z;
    float* vel_x; float* vel_y; float* vel_z;
    float* mass;
    float* importance;
    float* temporal;
    int* graph_id;
    int* cluster;
};

struct SafeGpuEdgeData {
    int* src;
    int* dst;
    float* weight;
    int* graph_id;
};

struct SafeGpuKernelParams {
    SafeGpuNodeData nodes;
    SafeGpuEdgeData edges;
    SafeConstraintData* constraints;
    SafeSimParams params;
    int num_nodes;
    int num_edges;
    int num_constraints;
};

__global__ void safe_visionflow_compute_kernel(SafeGpuKernelParams p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Comprehensive bounds checking
    if (idx < 0 || idx >= p.num_nodes) return;
    if (p.num_nodes <= 0 || p.num_nodes > SAFE_MAX_NODES) return;
    if (p.num_edges < 0 || p.num_edges > SAFE_MAX_EDGES) return;
    if (p.num_constraints < 0 || p.num_constraints > 1000) return;
    
    // Validate simulation parameters
    if (!validate_sim_params(p.params)) return;
    
    // Validate node data pointers
    if (p.nodes.pos_x == nullptr || p.nodes.pos_y == nullptr || p.nodes.pos_z == nullptr ||
        p.nodes.vel_x == nullptr || p.nodes.vel_y == nullptr || p.nodes.vel_z == nullptr) {
        return;
    }
    
    // Load and validate current position and velocity
    if (!SAFE_VALUE_CHECK(p.nodes.pos_x[idx]) || !SAFE_VALUE_CHECK(p.nodes.pos_y[idx]) || !SAFE_VALUE_CHECK(p.nodes.pos_z[idx]) ||
        !SAFE_VALUE_CHECK(p.nodes.vel_x[idx]) || !SAFE_VALUE_CHECK(p.nodes.vel_y[idx]) || !SAFE_VALUE_CHECK(p.nodes.vel_z[idx])) {
        // Reset invalid values
        p.nodes.pos_x[idx] = 0.0f;
        p.nodes.pos_y[idx] = 0.0f;
        p.nodes.pos_z[idx] = 0.0f;
        p.nodes.vel_x[idx] = 0.0f;
        p.nodes.vel_y[idx] = 0.0f;
        p.nodes.vel_z[idx] = 0.0f;
        return;
    }
    
    float3 position = safe_make_vec3(p.nodes.pos_x[idx], p.nodes.pos_y[idx], p.nodes.pos_z[idx]);
    float3 velocity = safe_make_vec3(p.nodes.vel_x[idx], p.nodes.vel_y[idx], p.nodes.vel_z[idx]);
    
    // Compute forces based on mode with safety
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    
    switch (p.params.compute_mode) {
        case 0: // Basic force-directed
            force = safe_compute_basic_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            break;
            
        case 1: // Dual graph mode
            force = safe_compute_dual_graph_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z, p.nodes.graph_id,
                p.edges.src, p.edges.dst, p.edges.weight, p.edges.graph_id,
                p.num_nodes, p.num_edges, p.params
            );
            break;
            
        case 2: // With constraints
            force = safe_compute_basic_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            force = safe_apply_constraints(
                idx, position, force, p.constraints, p.num_constraints,
                p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z, p.num_nodes, p.params
            );
            break;
            
        case 3: // Visual analytics
            force = safe_compute_visual_analytics(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.nodes.importance, p.nodes.temporal, p.nodes.cluster,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            break;
            
        default:
            // Fallback to basic with safety
            force = safe_compute_basic_forces(
                idx, p.nodes.pos_x, p.nodes.pos_y, p.nodes.pos_z,
                p.edges.src, p.edges.dst, p.edges.weight,
                p.num_nodes, p.num_edges, p.params
            );
            break;
    }
    
    // Clamp force magnitude BEFORE any scaling
    force = safe_vec3_clamp(force, p.params.max_force);
    
    // Progressive warmup for stability with safety
    if (p.params.iteration < 200) {
        float warmup = safe_clamp((float)p.params.iteration / 200.0f, 0.0f, 1.0f);
        force = safe_vec3_scale(force, warmup * warmup); // Quadratic warmup
        
        // Extra damping in early iterations
        float extra_damping = safe_clamp(0.98f - 0.13f * warmup, 0.5f, 0.99f);
        p.params.damping = fmaxf(p.params.damping, extra_damping);
    }
    
    // Temperature-based scaling (simulated annealing) with safety
    float temp_scale = safe_clamp(
        p.params.temperature / (1.0f + (float)p.params.iteration * 0.0001f),
        0.001f, 10.0f
    );
    force = safe_vec3_scale(force, temp_scale);
    
    // Update velocity with damping and safety
    float mass = 1.0f;
    if (p.nodes.mass != nullptr && isfinite(p.nodes.mass[idx]) && p.nodes.mass[idx] > 0.0f) {
        mass = safe_clamp(p.nodes.mass[idx], 0.1f, 1000.0f);
    }
    
    velocity = safe_vec3_add(velocity, safe_vec3_scale(force, p.params.dt / mass));
    velocity = safe_vec3_scale(velocity, p.params.damping);
    velocity = safe_vec3_clamp(velocity, p.params.max_velocity);
    
    // Zero velocity in very first iterations to prevent explosion
    if (p.params.iteration < 5) {
        velocity = make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Update position with safety
    position = safe_vec3_add(position, safe_vec3_scale(velocity, p.params.dt));
    
    // Enhanced soft viewport bounds with safety
    if (p.params.viewport_bounds > 0.0f) {
        float boundary_margin = p.params.viewport_bounds * 0.85f;
        float boundary_force_strength = 2.0f;
        
        // Check for extreme positions and apply recovery
        float max_distance = fmaxf(fmaxf(fabsf(position.x), fabsf(position.y)), fabsf(position.z));
        if (max_distance > (p.params.viewport_bounds * 10.0f)) {
            // Extreme position recovery
            position = safe_vec3_scale(safe_vec3_normalize(position), p.params.viewport_bounds * 0.5f);
            velocity = make_float3(0.0f, 0.0f, 0.0f);
        } else {
            // Normal boundary processing for each axis
            for (int axis = 0; axis < 3; axis++) {
                float* pos_comp = (axis == 0) ? &position.x : (axis == 1) ? &position.y : &position.z;
                float* vel_comp = (axis == 0) ? &velocity.x : (axis == 1) ? &velocity.y : &velocity.z;
                
                if (fabsf(*pos_comp) > boundary_margin) {
                    float distance_ratio = safe_clamp(
                        (fabsf(*pos_comp) - boundary_margin) / (p.params.viewport_bounds - boundary_margin),
                        0.0f, 1.0f
                    );
                    
                    // Quadratic force increase near boundary for smooth deceleration
                    float dead_zone = (p.params.viewport_bounds > 1500.0f) ? 0.1f : 0.0f;
                    if (distance_ratio > dead_zone) {
                        float boundary_force = safe_clamp(
                            -(distance_ratio - dead_zone) * (distance_ratio - dead_zone) * 
                            boundary_force_strength * copysignf(1.0f, *pos_comp),
                            -SAFE_MAX_FORCE * 0.1f, SAFE_MAX_FORCE * 0.1f
                        );
                        *vel_comp += boundary_force * p.params.dt;
                    }
                    
                    // Progressive damping
                    float progressive_damping = safe_clamp(
                        p.params.boundary_damping * (1.0f - 0.5f * distance_ratio),
                        0.1f, 0.99f
                    );
                    *vel_comp *= progressive_damping;
                    
                    // Soft clamp with margin
                    if (fabsf(*pos_comp) > p.params.viewport_bounds * 0.98f) {
                        *pos_comp = copysignf(p.params.viewport_bounds * 0.98f, *pos_comp);
                        float velocity_damping = (p.params.viewport_bounds > 1500.0f) ? 0.9f : 0.5f;
                        *vel_comp *= velocity_damping;
                    }
                }
            }
        }
    }
    
    // Final safety validation before writing back
    if (!SAFE_VALUE_CHECK(position.x) || !SAFE_VALUE_CHECK(position.y) || !SAFE_VALUE_CHECK(position.z)) {
        position = make_float3(0.0f, 0.0f, 0.0f);
    }
    
    if (!SAFE_VALUE_CHECK(velocity.x) || !SAFE_VALUE_CHECK(velocity.y) || !SAFE_VALUE_CHECK(velocity.z)) {
        velocity = make_float3(0.0f, 0.0f, 0.0f);
    }
    
    // Write back results with final safety checks
    p.nodes.pos_x[idx] = safe_clamp(position.x, -SAFE_MAX_POSITION, SAFE_MAX_POSITION);
    p.nodes.pos_y[idx] = safe_clamp(position.y, -SAFE_MAX_POSITION, SAFE_MAX_POSITION);
    p.nodes.pos_z[idx] = safe_clamp(position.z, -SAFE_MAX_POSITION, SAFE_MAX_POSITION);
    p.nodes.vel_x[idx] = safe_clamp(velocity.x, -SAFE_MAX_VELOCITY, SAFE_MAX_VELOCITY);
    p.nodes.vel_y[idx] = safe_clamp(velocity.y, -SAFE_MAX_VELOCITY, SAFE_MAX_VELOCITY);
    p.nodes.vel_z[idx] = safe_clamp(velocity.z, -SAFE_MAX_VELOCITY, SAFE_MAX_VELOCITY);
}

// =============================================================================
// Safe Kernel Launch Wrapper
// =============================================================================

extern "C" {
    void launch_safe_visionflow_kernel(
        SafeGpuKernelParams* params,
        int grid_size,
        int block_size
    ) {
        // Validate launch parameters
        if (params == nullptr) return;
        if (grid_size <= 0 || grid_size > 65535) return;
        if (block_size <= 0 || block_size > 1024) return;
        if (params->num_nodes <= 0 || params->num_nodes > SAFE_MAX_NODES) return;
        if (params->num_edges < 0 || params->num_edges > SAFE_MAX_EDGES) return;
        
        // Launch kernel with validated parameters
        safe_visionflow_compute_kernel<<<grid_size, block_size>>>(*params);
        
        // Synchronize and check for errors
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            // Log error but don't crash
            printf("CUDA kernel execution failed: %s\n", cudaGetErrorString(error));
        }
    }
}

} // extern "C"