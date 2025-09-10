// VisionFlow Unified GPU Kernel - Rewritten for correctness, performance, and clarity.
// Implements a two-pass (force/integrate) simulation with double-buffering,
// uniform grid spatial hashing for repulsion, and CSR for spring forces.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cfloat>

extern "C" {

// =============================================================================
// Core Data Structures & Constants
// =============================================================================

// Matches the Rust SimParams struct for FFI.
struct SimParams {
    float dt;
    float damping;
    unsigned int warmup_iterations;
    float cooling_rate;
    float spring_k;
    float rest_length;
    float repel_k;
    float repulsion_cutoff;
    float repulsion_softening_epsilon;
    float center_gravity_k;
    float max_force;
    float max_velocity;
    float grid_cell_size;
    unsigned int feature_flags;
    unsigned int seed;
    int iteration;
    // Additional fields for compatibility
    float separation_radius;
    float cluster_strength;
    float alignment_strength;
    float temperature;
    float viewport_bounds;
    // SSSP parameters
    float sssp_alpha;  // Strength of SSSP influence on spring forces
    float boundary_damping;  // Damping applied at boundaries
    // Constraint progressive activation parameters
    unsigned int constraint_ramp_frames;  // Number of frames to fully activate constraints
    float constraint_max_force_per_node;  // Maximum force per node from all constraints
};

// Global constant memory for simulation parameters
__constant__ SimParams c_params;


struct FeatureFlags {
    static const unsigned int ENABLE_REPULSION = 1 << 0;
    static const unsigned int ENABLE_SPRINGS = 1 << 1;
    static const unsigned int ENABLE_CENTERING = 1 << 2;
    static const unsigned int ENABLE_CONSTRAINTS = 1 << 4;  // Enable semantic constraints
    static const unsigned int ENABLE_SSSP_SPRING_ADJUST = 1 << 6;  // Enable SSSP-based spring adjustment
};

struct AABB {
    float3 min;
    float3 max;
};

// GPU-compatible constraint data for CUDA kernel
struct ConstraintData {
    int kind;                    // Discriminant matching ConstraintKind
    int count;                   // Number of node indices used
    int node_idx[4];            // Node indices (max 4 for GPU efficiency)
    float params[8];            // Parameters (max 8 for various constraint types)
    float weight;               // Weight of this constraint
    int activation_frame;       // Frame when this constraint was activated (for progressive activation)
};

// Constraint kinds enum to match Rust
enum ConstraintKind {
    DISTANCE = 0,
    POSITION = 1,
    ANGLE = 2,
    SEMANTIC = 3,
    TEMPORAL = 4,
    GROUP = 5
};

// =============================================================================
// Device Helper Functions
// =============================================================================

__device__ inline float3 make_vec3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ inline float3 vec3_add(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 vec3_sub(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 vec3_scale(float3 v, float s) { return make_float3(v.x * s, v.y * s, v.z * s); }
__device__ inline float vec3_dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float vec3_length_sq(float3 v) { return vec3_dot(v, v); }
__device__ inline float vec3_length(float3 v) { return sqrtf(vec3_length_sq(v)); }

__device__ inline int clamp_int(int x, int min, int max) {
    return (x < min) ? min : (x > max) ? max : x;
}

__device__ inline float clamp_float(float x, float min, float max) {
    return fminf(fmaxf(x, min), max);
}

__device__ inline float3 vec3_normalize(float3 v) {
    float len = vec3_length(v);
    return (len > 1e-6f) ? vec3_scale(v, 1.0f / len) : make_float3(0.0f, 0.0f, 0.0f);
}

__device__ inline float3 vec3_clamp(float3 v, float limit) {
    float len_sq = vec3_length_sq(v);
    if (len_sq > limit * limit) {
        float len = sqrtf(len_sq);
        return vec3_scale(v, limit / len);
    }
    return v;
}

// CAS-based atomic min for float (maximum portability)
__device__ inline float atomicMinFloat(float* addr, float value) {
    float old = __int_as_float(atomicAdd((int*)addr, 0)); // initial read
    while (value < old) {
        int old_i = __float_as_int(old);
        int assumed = atomicCAS((int*)addr, old_i, __float_as_int(value));
        if (assumed == old_i) break;
        old = __int_as_float(assumed);
    }
    return old;
}

// =============================================================================
// Spatial Grid Kernels
// =============================================================================

__global__ void build_grid_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    int* __restrict__ cell_keys,
    const AABB aabb,
    const int3 grid_dims,
    const float cell_size,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float3 pos = make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    
    int grid_x = static_cast<int>((pos.x - aabb.min.x) / cell_size);
    int grid_y = static_cast<int>((pos.y - aabb.min.y) / cell_size);
    int grid_z = static_cast<int>((pos.z - aabb.min.z) / cell_size);

    grid_x = clamp_int(grid_x, 0, grid_dims.x - 1);
    grid_y = clamp_int(grid_y, 0, grid_dims.y - 1);
    grid_z = clamp_int(grid_z, 0, grid_dims.z - 1);

    cell_keys[idx] = grid_z * grid_dims.y * grid_dims.x + grid_y * grid_dims.x + grid_x;
}

__global__ void compute_cell_bounds_kernel(
    const int* __restrict__ sorted_cell_keys,
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    const int num_nodes,
    const int num_grid_cells)
{
    // Each thread checks if the cell key for its corresponding node
    // is different from the previous one, indicating a boundary.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    // The key for the current sorted node
    int current_key = sorted_cell_keys[idx];

    // The key for the previous sorted node (handle boundary case at index 0)
    int prev_key = (idx == 0) ? -1 : sorted_cell_keys[idx - 1];

    // If the key has changed, we've found the end of the previous cell
    // and the start of the current cell.
    if (current_key != prev_key) {
        // Mark the start of the current cell.
        if (current_key >= 0 && current_key < num_grid_cells) {
            cell_start[current_key] = idx;
        }
        // Mark the end of the previous cell.
        if (prev_key >= 0 && prev_key < num_grid_cells) {
            cell_end[prev_key] = idx;
        }
    }

    // The very last node marks the end of its cell.
    if (idx == num_nodes - 1) {
        if (current_key >= 0 && current_key < num_grid_cells) {
            cell_end[current_key] = num_nodes;
        }
    }
}


// =============================================================================
// Force Pass Kernel
// =============================================================================

__global__ void force_pass_kernel(
    const float* __restrict__ pos_in_x,
    const float* __restrict__ pos_in_y,
    const float* __restrict__ pos_in_z,
    float* __restrict__ force_out_x,
    float* __restrict__ force_out_y,
    float* __restrict__ force_out_z,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ sorted_node_indices,
    const int* __restrict__ cell_keys,
    const int3 grid_dims,
    const int* __restrict__ edge_row_offsets,
    const int* __restrict__ edge_col_indices,
    const float* __restrict__ edge_weights,
    const int num_nodes,
    const float* __restrict__ d_sssp_dist,
    const ConstraintData* __restrict__ constraints,
    const int num_constraints,
    // Constraint telemetry buffers (optional, can be nullptr)
    float* __restrict__ constraint_violations,   // [num_constraints] violation magnitudes
    float* __restrict__ constraint_energy,       // [num_constraints] energy values
    float* __restrict__ node_constraint_force)   // [num_nodes] total constraint force per node
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float3 my_pos = make_vec3(pos_in_x[idx], pos_in_y[idx], pos_in_z[idx]);
    float3 total_force = make_vec3(0.0f, 0.0f, 0.0f);

    if (c_params.feature_flags & FeatureFlags::ENABLE_REPULSION) {
        int my_cell_key = cell_keys[idx];
        int grid_x = my_cell_key % grid_dims.x;
        int grid_y = (my_cell_key / grid_dims.x) % grid_dims.y;
        int grid_z = my_cell_key / (grid_dims.x * grid_dims.y);

        for (int z = -1; z <= 1; ++z) {
            for (int y = -1; y <= 1; ++y) {
                for (int x = -1; x <= 1; ++x) {
                    int neighbor_grid_x = grid_x + x;
                    int neighbor_grid_y = grid_y + y;
                    int neighbor_grid_z = grid_z + z;

                    if (neighbor_grid_x >= 0 && neighbor_grid_x < grid_dims.x &&
                        neighbor_grid_y >= 0 && neighbor_grid_y < grid_dims.y &&
                        neighbor_grid_z >= 0 && neighbor_grid_z < grid_dims.z) {
                        
                        int neighbor_cell_key = neighbor_grid_z * grid_dims.y * grid_dims.x + neighbor_grid_y * grid_dims.x + neighbor_grid_x;
                        int start = cell_start[neighbor_cell_key];
                        int end = cell_end[neighbor_cell_key];

                        for (int j = start; j < end; ++j) {
                            int neighbor_idx = sorted_node_indices[j];
                            if (idx == neighbor_idx) continue;

                            float3 neighbor_pos = make_vec3(pos_in_x[neighbor_idx], pos_in_y[neighbor_idx], pos_in_z[neighbor_idx]);
                            float3 diff = vec3_sub(my_pos, neighbor_pos);
                            float dist_sq = vec3_length_sq(diff);

                            if (dist_sq < c_params.repulsion_cutoff * c_params.repulsion_cutoff && dist_sq > 1e-6f) {
                                float dist = sqrtf(dist_sq);
                                float repulsion = c_params.repel_k / (dist_sq + c_params.repulsion_softening_epsilon);
                                
                                // Prevent repulsion force overflow when nodes are too close
                                // Use full max_force instead of arbitrary 0.5 multiplier
                                float max_repulsion = c_params.max_force;
                                repulsion = fminf(repulsion, max_repulsion);
                                
                                // Safety check for NaN/Inf
                                if (isfinite(repulsion) && isfinite(dist) && dist > 0.0f) {
                                    total_force = vec3_add(total_force, vec3_scale(diff, repulsion / dist));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (c_params.feature_flags & FeatureFlags::ENABLE_SPRINGS) {
        int start_edge = edge_row_offsets[idx];
        int end_edge = edge_row_offsets[idx + 1];
        
        float du = 0.0f;
        bool use_sssp = (d_sssp_dist != nullptr) &&
                       (c_params.feature_flags & FeatureFlags::ENABLE_SSSP_SPRING_ADJUST);
        if (use_sssp) {
            du = d_sssp_dist[idx];
        }
        
        for (int i = start_edge; i < end_edge; ++i) {
            int neighbor_idx = edge_col_indices[i];
            float3 neighbor_pos = make_vec3(pos_in_x[neighbor_idx], pos_in_y[neighbor_idx], pos_in_z[neighbor_idx]);
            
            float3 diff = vec3_sub(neighbor_pos, my_pos);
            float dist = vec3_length(diff);
            
            if (dist > 1e-6f) {
                float ideal = c_params.rest_length;
                if (use_sssp) {
                    float dv = d_sssp_dist[neighbor_idx];
                    // Handle disconnected components gracefully
                    if (isfinite(du) && isfinite(dv)) {
                        float delta = fabsf(du - dv);
                        float norm_delta = fminf(delta, 1000.0f); // Cap for stability
                        ideal = c_params.rest_length + c_params.sssp_alpha * norm_delta;
                    }
                }
                float displacement = dist - ideal;
                float spring_force_mag = c_params.spring_k * displacement * edge_weights[i];
                total_force = vec3_add(total_force, vec3_scale(diff, spring_force_mag / dist));
            }
        }
    }
    
    if (c_params.feature_flags & FeatureFlags::ENABLE_CENTERING) {
        total_force = vec3_sub(total_force, vec3_scale(my_pos, c_params.center_gravity_k));
    }

    // Constraint force accumulation
    float total_constraint_force_magnitude = 0.0f;
    if (c_params.feature_flags & FeatureFlags::ENABLE_CONSTRAINTS) {
        for (int c = 0; c < num_constraints; c++) {
            const ConstraintData& constraint = constraints[c];
            
            // Check if this node is involved in this constraint
            bool is_involved = false;
            int node_role = -1; // Which position in the constraint this node occupies
            for (int n = 0; n < constraint.count && n < 4; n++) {
                if (constraint.node_idx[n] == idx) {
                    is_involved = true;
                    node_role = n;
                    break;
                }
            }
            
            if (!is_involved) continue;
            
            float3 constraint_force = make_vec3(0.0f, 0.0f, 0.0f);
            
            // Calculate progressive activation multiplier
            float progressive_multiplier = 1.0f;
            if (c_params.constraint_ramp_frames > 0) {
                int frames_since_activation = c_params.iteration - constraint.activation_frame;
                if (frames_since_activation >= 0 && frames_since_activation < c_params.constraint_ramp_frames) {
                    // Linear ramp from 0 to 1 over constraint_ramp_frames
                    progressive_multiplier = (float)frames_since_activation / (float)c_params.constraint_ramp_frames;
                    progressive_multiplier = fminf(progressive_multiplier, 1.0f);
                }
            }
            
            // Process constraint based on type
            if (constraint.kind == ConstraintKind::DISTANCE && constraint.count >= 2) {
                // Distance constraint: maintain distance between two nodes
                int other_idx = (node_role == 0) ? constraint.node_idx[1] : constraint.node_idx[0];
                if (other_idx >= 0 && other_idx < num_nodes) {
                    float3 other_pos = make_vec3(pos_in_x[other_idx], pos_in_y[other_idx], pos_in_z[other_idx]);
                    float3 diff = vec3_sub(my_pos, other_pos);
                    float current_dist = vec3_length(diff);
                    float target_dist = constraint.params[0];
                    
                    if (current_dist > 1e-6f && isfinite(current_dist) && target_dist > 0.0f) {
                        float error = current_dist - target_dist;
                        // Apply progressive activation multiplier to constraint weight
                        float effective_weight = constraint.weight * progressive_multiplier;
                        float force_magnitude = -effective_weight * error;
                        
                        // Cap constraint forces to prevent instability
                        float max_constraint_force = c_params.constraint_max_force_per_node;
                        force_magnitude = fmaxf(-max_constraint_force, fminf(max_constraint_force, force_magnitude));
                        
                        constraint_force = vec3_scale(diff, force_magnitude / current_dist);
                    }
                }
            }
            else if (constraint.kind == ConstraintKind::POSITION && constraint.count >= 1) {
                // Position constraint: attract node to target position
                float3 target_pos = make_vec3(constraint.params[0], constraint.params[1], constraint.params[2]);
                float3 diff = vec3_sub(target_pos, my_pos);
                float distance = vec3_length(diff);
                
                if (distance > 1e-6f && isfinite(distance)) {
                    // Apply progressive activation multiplier to constraint weight
                    float effective_weight = constraint.weight * progressive_multiplier;
                    float force_magnitude = effective_weight * distance * 0.1f; // Gentle attraction
                    
                    // Cap constraint forces using per-node force limit
                    float max_constraint_force = c_params.constraint_max_force_per_node;
                    force_magnitude = fminf(force_magnitude, max_constraint_force);
                    
                    constraint_force = vec3_scale(diff, force_magnitude / distance);
                }
            }
            
            // Apply constraint force with safety checks and collect telemetry
            if (isfinite(constraint_force.x) && isfinite(constraint_force.y) && isfinite(constraint_force.z)) {
                total_force = vec3_add(total_force, constraint_force);
                
                // Accumulate constraint force magnitude for this node
                float constraint_force_mag = vec3_length(constraint_force);
                total_constraint_force_magnitude += constraint_force_mag;
                
                // Record constraint-specific telemetry (if buffers provided)
                if (constraint_violations != nullptr && constraint_energy != nullptr) {
                    float violation = 0.0f;
                    float energy = 0.0f;
                    
                    // Calculate violation and energy based on constraint type
                    if (constraint.kind == ConstraintKind::DISTANCE && constraint.count >= 2) {
                        int other_idx = (node_role == 0) ? constraint.node_idx[1] : constraint.node_idx[0];
                        if (other_idx >= 0 && other_idx < num_nodes) {
                            float3 other_pos = make_vec3(pos_in_x[other_idx], pos_in_y[other_idx], pos_in_z[other_idx]);
                            float3 diff = vec3_sub(my_pos, other_pos);
                            float current_dist = vec3_length(diff);
                            float target_dist = constraint.params[0];
                            
                            violation = fabsf(current_dist - target_dist);
                            energy = 0.5f * constraint.weight * violation * violation; // Quadratic energy
                        }
                    } else if (constraint.kind == ConstraintKind::POSITION && constraint.count >= 1) {
                        float3 target_pos = make_vec3(constraint.params[0], constraint.params[1], constraint.params[2]);
                        float3 diff = vec3_sub(target_pos, my_pos);
                        violation = vec3_length(diff);
                        energy = 0.5f * constraint.weight * violation * violation;
                    }
                    
                    // Atomically add to constraint telemetry (multiple threads might contribute to same constraint)
                    atomicAdd(&constraint_violations[c], violation);
                    atomicAdd(&constraint_energy[c], energy);
                }
            }
        }
    }

    force_out_x[idx] = total_force.x;
    force_out_y[idx] = total_force.y;
    force_out_z[idx] = total_force.z;
    
    // Record per-node constraint force telemetry
    if (node_constraint_force != nullptr) {
        node_constraint_force[idx] = total_constraint_force_magnitude;
    }
}

// =============================================================================
// SSSP Relaxation Kernel
// =============================================================================

extern "C" __global__ void relaxation_step_kernel(
    float* __restrict__ d_dist,                // [n] distance array
    const int* __restrict__ d_current_frontier,// [frontier_size] active vertices
    int frontier_size,
    const int* __restrict__ d_row_offsets,     // [n+1] CSR row offsets
    const int* __restrict__ d_col_indices,     // [m] CSR column indices  
    const float* __restrict__ d_weights,       // [m] edge weights
    int* __restrict__ d_next_frontier_flags,   // [n] output flags (0/1)
    float B,                                   // distance boundary
    int n                                      // total vertices
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= frontier_size) return;
    
    int u = d_current_frontier[t];
    float du = d_dist[u];
    if (!isfinite(du)) return; // Skip unreachable vertices
    
    int start = d_row_offsets[u];
    int end = d_row_offsets[u + 1];
    
    for (int e = start; e < end; ++e) {
        int v = d_col_indices[e];
        float w = d_weights[e];
        float nd = du + w;
        
        if (nd < B) {
            float old = atomicMinFloat(&d_dist[v], nd);
            if (nd < old) {
                d_next_frontier_flags[v] = 1; // Mark for next frontier
            }
        }
    }
}

// =============================================================================
// Integration Pass Kernel
// =============================================================================

__global__ void integrate_pass_kernel(
    const float* __restrict__ pos_in_x,
    const float* __restrict__ pos_in_y,
    const float* __restrict__ pos_in_z,
    const float* __restrict__ vel_in_x,
    const float* __restrict__ vel_in_y,
    const float* __restrict__ vel_in_z,
    const float* __restrict__ force_x,
    const float* __restrict__ force_y,
    const float* __restrict__ force_z,
    const float* __restrict__ mass,
    float* __restrict__ pos_out_x,
    float* __restrict__ pos_out_y,
    float* __restrict__ pos_out_z,
    float* __restrict__ vel_out_x,
    float* __restrict__ vel_out_y,
    float* __restrict__ vel_out_z,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float3 pos = make_vec3(pos_in_x[idx], pos_in_y[idx], pos_in_z[idx]);
    float3 vel = make_vec3(vel_in_x[idx], vel_in_y[idx], vel_in_z[idx]);
    float3 force = make_vec3(force_x[idx], force_y[idx], force_z[idx]);
    float node_mass = (mass != nullptr && mass[idx] > 0.0f) ? mass[idx] : 1.0f;

    // Force capping using settings values only
    float force_mag = vec3_length(force);
    if (force_mag > c_params.max_force) {
        force = vec3_scale(force, c_params.max_force / force_mag);
    }

    // Use damping exactly as specified in settings
    float effective_damping = c_params.damping;

    // Apply warmup if configured in settings
    if (c_params.iteration < c_params.warmup_iterations) {
        float warmup_factor = (float)c_params.iteration / (float)c_params.warmup_iterations;
        force = vec3_scale(force, warmup_factor);
        // Use cooling_rate from settings for warmup damping adjustment
        effective_damping = c_params.damping + (c_params.cooling_rate - c_params.damping) * (1.0f - warmup_factor);
    }

    // Apply integration with settings-based damping
    vel = vec3_add(vel, vec3_scale(force, c_params.dt / node_mass));
    vel = vec3_scale(vel, effective_damping);
    vel = vec3_clamp(vel, c_params.max_velocity);
    pos = vec3_add(pos, vec3_scale(vel, c_params.dt));

    // Apply enhanced boundary constraints with progressive repulsion
    float boundary_limit = c_params.viewport_bounds;
    if (boundary_limit > 0.0f) {
        // Use boundary damping from settings for margin and strength
        float boundary_margin = boundary_limit * c_params.boundary_damping;
        float boundary_repulsion_strength = c_params.max_force * c_params.boundary_damping;
        
        // Check X boundary
        if (fabsf(pos.x) > boundary_margin) {
            float boundary_dist = fabsf(pos.x) - boundary_margin;
            float boundary_force = boundary_repulsion_strength * (boundary_dist / (boundary_limit - boundary_margin));
            boundary_force = fminf(boundary_force, c_params.max_force); // Cap using max_force setting
            pos.x = pos.x > 0 ? fminf(pos.x, boundary_limit) : fmaxf(pos.x, -boundary_limit);
            vel.x *= c_params.boundary_damping; // Apply boundary damping from settings
            // Add reflection for strong collisions
            if (fabsf(pos.x) >= boundary_limit) {
                vel.x = -vel.x * c_params.boundary_damping; // Reflect with boundary damping
            }
        }
        
        // Check Y boundary
        if (fabsf(pos.y) > boundary_margin) {
            float boundary_dist = fabsf(pos.y) - boundary_margin;
            float boundary_force = boundary_repulsion_strength * (boundary_dist / (boundary_limit - boundary_margin));
            // Use max_force instead of hardcoded 15.0f
            boundary_force = fminf(boundary_force, c_params.max_force);
            pos.y = pos.y > 0 ? fminf(pos.y, boundary_limit) : fmaxf(pos.y, -boundary_limit);
            vel.y *= c_params.boundary_damping;
            if (fabsf(pos.y) >= boundary_limit) {
                vel.y = -vel.y * c_params.boundary_damping;
            }
        }
        
        // Check Z boundary
        if (fabsf(pos.z) > boundary_margin) {
            float boundary_dist = fabsf(pos.z) - boundary_margin;
            float boundary_force = boundary_repulsion_strength * (boundary_dist / (boundary_limit - boundary_margin));
            // Use max_force instead of hardcoded 15.0f
            boundary_force = fminf(boundary_force, c_params.max_force);
            pos.z = pos.z > 0 ? fminf(pos.z, boundary_limit) : fmaxf(pos.z, -boundary_limit);
            vel.z *= c_params.boundary_damping;
            if (fabsf(pos.z) >= boundary_limit) {
                vel.z = -vel.z * c_params.boundary_damping;
            }
        }
    }

    pos_out_x[idx] = pos.x;
    pos_out_y[idx] = pos.y;
    pos_out_z[idx] = pos.z;
    vel_out_x[idx] = vel.x;
    vel_out_y[idx] = vel.y;
    vel_out_z[idx] = vel.z;
}

// =============================================================================
// Device-side Frontier Compaction for SSSP
// =============================================================================

__global__ void compact_frontier_kernel(
    const int* __restrict__ flags,          // Input: per-node flags (1 if in frontier)
    int* __restrict__ compacted_frontier,   // Output: compacted frontier
    int* __restrict__ frontier_counter,     // Output: frontier size (atomic counter)
    const int num_nodes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_nodes && flags[idx] != 0) {
        // Atomically get position in compacted array
        int pos = atomicAdd(frontier_counter, 1);
        compacted_frontier[pos] = idx;
    }
}

// =============================================================================
// Thrust Wrapper Functions for Sorting and Scanning
// =============================================================================

// Wrapper for thrust sort_by_key operation
void thrust_sort_key_value(
    void* d_keys_in,
    void* d_keys_out,
    void* d_values_in, 
    void* d_values_out,
    int num_items,
    cudaStream_t stream
) {
    // Copy input to output first
    cudaMemcpyAsync(d_keys_out, d_keys_in, 
                    num_items * sizeof(int), 
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_values_out, d_values_in,
                    num_items * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Sort in-place on output buffers
    thrust::device_ptr<int> keys(static_cast<int*>(d_keys_out));
    thrust::device_ptr<int> vals(static_cast<int*>(d_values_out));
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                       keys, keys + num_items, vals);
}

// Wrapper for thrust exclusive_scan operation  
void thrust_exclusive_scan(
    void* d_in,
    void* d_out,
    int num_items,
    cudaStream_t stream
) {
    thrust::device_ptr<int> in_ptr(static_cast<int*>(d_in));
    thrust::device_ptr<int> out_ptr(static_cast<int*>(d_out));
    thrust::exclusive_scan(thrust::cuda::par.on(stream),
                          in_ptr, in_ptr + num_items, 
                          out_ptr, 0); // 0 = initial value
}

// =============================================================================
// K-means Clustering Kernels
// =============================================================================

/**
 * Initialize K-means centroids using K-means++ algorithm
 * Grid: (k, 1, 1), Block: (256, 1, 1) where k = num_clusters
 * Each block initializes one centroid
 */
__global__ void init_centroids_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ centroids_x,
    float* __restrict__ centroids_y,
    float* __restrict__ centroids_z,
    float* __restrict__ min_distances,
    int* __restrict__ selected_nodes,
    const int num_nodes,
    const int num_clusters,
    const int current_centroid,
    const unsigned int seed)
{
    const int k = blockIdx.x; // Current centroid index
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Shared memory for reduction operations
    extern __shared__ float shared_data[];
    float* shared_distances = shared_data;
    
    if (current_centroid == 0 && k == 0) {
        // First centroid: select random node
        if (tid == 0) {
            curandState state;
            curand_init(seed, 0, 0, &state);
            int selected = curand(&state) % num_nodes;
            selected_nodes[0] = selected;
            centroids_x[0] = pos_x[selected];
            centroids_y[0] = pos_y[selected];
            centroids_z[0] = pos_z[selected];
        }
        return;
    }
    
    if (k != current_centroid) return; // Only one block processes current centroid
    
    // Calculate distances to nearest existing centroid for all nodes
    for (int node = tid; node < num_nodes; node += block_size) {
        float min_dist = FLT_MAX;
        
        // Find distance to nearest existing centroid
        for (int c = 0; c < current_centroid; c++) {
            float dx = pos_x[node] - centroids_x[c];
            float dy = pos_y[node] - centroids_y[c];
            float dz = pos_z[node] - centroids_z[c];
            float dist = dx * dx + dy * dy + dz * dz;
            min_dist = fminf(min_dist, dist);
        }
        
        min_distances[node] = min_dist;
    }
    
    __syncthreads();
    
    // Sum all squared distances for probability normalization
    float total_dist = 0.0f;
    for (int node = tid; node < num_nodes; node += block_size) {
        total_dist += min_distances[node];
    }
    
    // Block-level reduction to sum distances
    shared_distances[tid] = total_dist;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_distances[tid] += shared_distances[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float total_sum = shared_distances[0];
        
        // Generate random threshold for selection
        curandState state;
        curand_init(seed + current_centroid, 0, 0, &state);
        float threshold = curand_uniform(&state) * total_sum;
        
        // Select node based on probability proportional to squared distance
        float cumulative = 0.0f;
        int selected = 0;
        for (int node = 0; node < num_nodes; node++) {
            cumulative += min_distances[node];
            if (cumulative >= threshold) {
                selected = node;
                break;
            }
        }
        
        selected_nodes[current_centroid] = selected;
        centroids_x[current_centroid] = pos_x[selected];
        centroids_y[current_centroid] = pos_y[selected];
        centroids_z[current_centroid] = pos_z[selected];
    }
}

/**
 * Assign nodes to nearest centroid cluster
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes one node
 */
__global__ void assign_clusters_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ centroids_x,
    const float* __restrict__ centroids_y,
    const float* __restrict__ centroids_z,
    int* __restrict__ cluster_assignments,
    float* __restrict__ distances_to_centroid,
    const int num_nodes,
    const int num_clusters)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;
    
    float node_x = pos_x[node_idx];
    float node_y = pos_y[node_idx];
    float node_z = pos_z[node_idx];
    
    float min_dist = FLT_MAX;
    int best_cluster = 0;
    
    // Find nearest centroid
    for (int k = 0; k < num_clusters; k++) {
        float dx = node_x - centroids_x[k];
        float dy = node_y - centroids_y[k];
        float dz = node_z - centroids_z[k];
        float dist = dx * dx + dy * dy + dz * dz;
        
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = k;
        }
    }
    
    cluster_assignments[node_idx] = best_cluster;
    distances_to_centroid[node_idx] = sqrtf(min_dist);
}

/**
 * Update centroids based on cluster assignments
 * Grid: (num_clusters, 1, 1), Block: (256, 1, 1)
 * Each block processes one cluster centroid
 */
__global__ void update_centroids_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ cluster_assignments,
    float* __restrict__ centroids_x,
    float* __restrict__ centroids_y,
    float* __restrict__ centroids_z,
    int* __restrict__ cluster_sizes,
    const int num_nodes,
    const int num_clusters)
{
    const int cluster_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    if (cluster_id >= num_clusters) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_x = shared_mem;
    float* shared_y = shared_mem + block_size;
    float* shared_z = shared_mem + 2 * block_size;
    int* shared_count = (int*)(shared_mem + 3 * block_size);
    
    // Initialize shared memory
    shared_x[tid] = 0.0f;
    shared_y[tid] = 0.0f;
    shared_z[tid] = 0.0f;
    shared_count[tid] = 0;
    
    // Accumulate positions for nodes assigned to this cluster
    for (int node = tid; node < num_nodes; node += block_size) {
        if (cluster_assignments[node] == cluster_id) {
            shared_x[tid] += pos_x[node];
            shared_y[tid] += pos_y[node];
            shared_z[tid] += pos_z[node];
            shared_count[tid]++;
        }
    }
    
    __syncthreads();
    
    // Block-level reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_x[tid] += shared_x[tid + s];
            shared_y[tid] += shared_y[tid + s];
            shared_z[tid] += shared_z[tid + s];
            shared_count[tid] += shared_count[tid + s];
        }
        __syncthreads();
    }
    
    // Update centroid
    if (tid == 0) {
        int count = shared_count[0];
        if (count > 0) {
            centroids_x[cluster_id] = shared_x[0] / count;
            centroids_y[cluster_id] = shared_y[0] / count;
            centroids_z[cluster_id] = shared_z[0] / count;
            cluster_sizes[cluster_id] = count;
        } else {
            // Keep previous centroid if no nodes assigned
            cluster_sizes[cluster_id] = 0;
        }
    }
}

/**
 * Compute inertia (sum of squared distances to centroids)
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each block computes partial inertia, needs reduction afterward
 */
__global__ void compute_inertia_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ centroids_x,
    const float* __restrict__ centroids_y,
    const float* __restrict__ centroids_z,
    const int* __restrict__ cluster_assignments,
    float* __restrict__ partial_inertia,
    const int num_nodes)
{
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;
    const int start = block_id * block_size;
    const int end = min(start + block_size, num_nodes);
    
    extern __shared__ float shared_inertia[];
    shared_inertia[tid] = 0.0f;
    
    // Compute squared distances for nodes in this block
    for (int node = start + tid; node < end; node += block_size) {
        if (node < num_nodes) {
            int cluster = cluster_assignments[node];
            float dx = pos_x[node] - centroids_x[cluster];
            float dy = pos_y[node] - centroids_y[cluster];
            float dz = pos_z[node] - centroids_z[cluster];
            shared_inertia[tid] += dx * dx + dy * dy + dz * dz;
        }
    }
    
    __syncthreads();
    
    // Block-level reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < block_size) {
            shared_inertia[tid] += shared_inertia[tid + s];
        }
        __syncthreads();
    }
    
    // Store partial result
    if (tid == 0) {
        partial_inertia[block_id] = shared_inertia[0];
    }
}

// =============================================================================
// Anomaly Detection Kernels
// =============================================================================

/**
 * Compute Local Outlier Factor (LOF) for anomaly detection
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes one node
 */
__global__ void compute_lof_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const int* __restrict__ sorted_node_indices,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ cell_keys,
    const int3 grid_dims,
    float* __restrict__ lof_scores,
    float* __restrict__ local_densities,
    const int num_nodes,
    const int k_neighbors,
    const float radius)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    float3 my_pos = make_vec3(pos_x[idx], pos_y[idx], pos_z[idx]);
    
    // Arrays for k-nearest neighbors (using fixed-size for GPU efficiency)
    const int MAX_K = 32; // Compile-time constant
    float neighbor_dists[MAX_K];
    int neighbor_indices[MAX_K];
    int actual_k = min(k_neighbors, MAX_K);
    
    // Initialize neighbor arrays
    for (int i = 0; i < actual_k; i++) {
        neighbor_dists[i] = FLT_MAX;
        neighbor_indices[i] = -1;
    }
    
    // Get my grid cell
    int my_cell_key = cell_keys[idx];
    int grid_x = my_cell_key % grid_dims.x;
    int grid_y = (my_cell_key / grid_dims.x) % grid_dims.y;
    int grid_z = my_cell_key / (grid_dims.x * grid_dims.y);
    
    // Search neighboring cells for k-nearest neighbors
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                int neighbor_grid_x = grid_x + x;
                int neighbor_grid_y = grid_y + y;
                int neighbor_grid_z = grid_z + z;
                
                if (neighbor_grid_x >= 0 && neighbor_grid_x < grid_dims.x &&
                    neighbor_grid_y >= 0 && neighbor_grid_y < grid_dims.y &&
                    neighbor_grid_z >= 0 && neighbor_grid_z < grid_dims.z) {
                    
                    int neighbor_cell_key = neighbor_grid_z * grid_dims.y * grid_dims.x + 
                                          neighbor_grid_y * grid_dims.x + neighbor_grid_x;
                    int start = cell_start[neighbor_cell_key];
                    int end = cell_end[neighbor_cell_key];
                    
                    for (int j = start; j < end; ++j) {
                        int neighbor_idx = sorted_node_indices[j];
                        if (idx == neighbor_idx) continue;
                        
                        float3 neighbor_pos = make_vec3(pos_x[neighbor_idx], pos_y[neighbor_idx], pos_z[neighbor_idx]);
                        float3 diff = vec3_sub(my_pos, neighbor_pos);
                        float dist = vec3_length(diff);
                        
                        if (dist <= radius) {
                            // Insert into k-nearest neighbors if closer than furthest current neighbor
                            if (dist < neighbor_dists[actual_k - 1]) {
                                neighbor_dists[actual_k - 1] = dist;
                                neighbor_indices[actual_k - 1] = neighbor_idx;
                                
                                // Bubble sort to maintain order (small k makes this efficient)
                                for (int i = actual_k - 1; i > 0 && neighbor_dists[i] < neighbor_dists[i-1]; i--) {
                                    float temp_dist = neighbor_dists[i];
                                    int temp_idx = neighbor_indices[i];
                                    neighbor_dists[i] = neighbor_dists[i-1];
                                    neighbor_indices[i] = neighbor_indices[i-1];
                                    neighbor_dists[i-1] = temp_dist;
                                    neighbor_indices[i-1] = temp_idx;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Compute k-distance (distance to k-th nearest neighbor)
    float k_dist = 0.0f;
    int valid_neighbors = 0;
    for (int i = 0; i < actual_k && neighbor_indices[i] != -1; i++) {
        k_dist = neighbor_dists[i]; // Last valid distance is k-distance
        valid_neighbors++;
    }
    
    if (valid_neighbors == 0) {
        lof_scores[idx] = 1.0f; // Normal score for isolated nodes
        local_densities[idx] = 0.0f;
        return;
    }
    
    // Compute local reachability density (LRD)
    float sum_reach_dist = 0.0f;
    for (int i = 0; i < valid_neighbors; i++) {
        // Reachability distance = max(k-distance(neighbor), actual_distance)
        // For simplicity, we approximate neighbor k-distances with current k_dist
        float reach_dist = fmaxf(k_dist, neighbor_dists[i]);
        sum_reach_dist += reach_dist;
    }
    
    float lrd = valid_neighbors / (sum_reach_dist + 1e-6f); // Add epsilon for stability
    local_densities[idx] = lrd;
    
    // Compute LOF by comparing with neighbors' LRDs
    // For GPU efficiency, we approximate neighbors' LRDs
    float lof = 1.0f; // Default normal score
    if (lrd > 1e-6f) {
        float neighbor_lrd_sum = 0.0f;
        
        // Estimate neighbors' LRDs (simplified for GPU performance)
        for (int i = 0; i < valid_neighbors; i++) {
            // Approximate neighbor LRD based on local density estimation
            float approx_neighbor_lrd = valid_neighbors / (neighbor_dists[i] * actual_k + 1e-6f);
            neighbor_lrd_sum += approx_neighbor_lrd;
        }
        
        float avg_neighbor_lrd = neighbor_lrd_sum / valid_neighbors;
        lof = avg_neighbor_lrd / lrd;
    }
    
    // Clamp LOF score for numerical stability
    lof_scores[idx] = fminf(fmaxf(lof, 0.1f), 10.0f);
}

/**
 * Compute Z-score based anomaly detection
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Requires pre-computed mean and standard deviation
 */
__global__ void compute_zscore_kernel(
    const float* __restrict__ feature_values,
    float* __restrict__ zscore_values,
    const float mean_value,
    const float std_value,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    float feature = feature_values[idx];
    
    // Compute Z-score with numerical stability
    if (std_value > 1e-6f) {
        float zscore = (feature - mean_value) / std_value;
        // Clamp extreme values for stability
        zscore_values[idx] = fminf(fmaxf(zscore, -10.0f), 10.0f);
    } else {
        // If no variance, all values are normal
        zscore_values[idx] = 0.0f;
    }
}

/**
 * Compute feature statistics (mean, variance) for Z-score calculation
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each block computes partial sums, needs reduction afterward
 */
__global__ void compute_feature_stats_kernel(
    const float* __restrict__ feature_values,
    float* __restrict__ partial_sums,
    float* __restrict__ partial_sq_sums,
    const int num_nodes)
{
    const int tid = threadIdx.x;
    const int block_id = blockIdx.x;
    const int block_size = blockDim.x;
    const int start = block_id * block_size;
    
    extern __shared__ float shared_stats[];
    float* shared_sum = shared_stats;
    float* shared_sq_sum = shared_stats + block_size;
    
    // Initialize shared memory
    shared_sum[tid] = 0.0f;
    shared_sq_sum[tid] = 0.0f;
    
    // Accumulate values for this block
    for (int i = start + tid; i < num_nodes; i += blockDim.x * gridDim.x) {
        if (i < num_nodes) {
            float val = feature_values[i];
            shared_sum[tid] += val;
            shared_sq_sum[tid] += val * val;
        }
    }
    
    __syncthreads();
    
    // Block-level reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
            shared_sq_sum[tid] += shared_sq_sum[tid + s];
        }
        __syncthreads();
    }
    
    // Store partial results
    if (tid == 0) {
        partial_sums[block_id] = shared_sum[0];
        partial_sq_sums[block_id] = shared_sq_sum[0];
    }
}

// =============================================================================
// Community Detection Kernels (Label Propagation Algorithm)
// =============================================================================

/**
 * Initialize node labels with unique values
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread initializes one node's label
 */
__global__ void init_labels_kernel(
    int* __restrict__ labels,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    // Initialize each node with its own unique label (index)
    labels[idx] = idx;
}

/**
 * Synchronous label propagation kernel - all updates happen simultaneously
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes one node
 */
__global__ void propagate_labels_sync_kernel(
    const int* __restrict__ labels_in,
    int* __restrict__ labels_out,
    const int* __restrict__ edge_row_offsets,
    const int* __restrict__ edge_col_indices,
    const float* __restrict__ edge_weights,
    int* __restrict__ label_counts,
    const int num_nodes,
    const int max_label,
    curandState* __restrict__ rand_states)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int start_edge = edge_row_offsets[idx];
    int end_edge = edge_row_offsets[idx + 1];
    
    if (start_edge == end_edge) {
        // Isolated node keeps its current label
        labels_out[idx] = labels_in[idx];
        return;
    }
    
    // Use shared memory for label frequency counting
    extern __shared__ int shared_memory[];
    int* local_label_counts = shared_memory + threadIdx.x * (max_label + 1);
    
    // Initialize local label counts
    for (int i = 0; i <= max_label; i++) {
        local_label_counts[i] = 0;
    }
    
    // Count weighted neighbor labels
    float total_weight = 0.0f;
    for (int i = start_edge; i < end_edge; ++i) {
        int neighbor_idx = edge_col_indices[i];
        int neighbor_label = labels_in[neighbor_idx];
        float weight = edge_weights[i];
        
        if (neighbor_label >= 0 && neighbor_label <= max_label) {
            // Use weighted voting (multiply by 1000 for integer precision)
            local_label_counts[neighbor_label] += (int)(weight * 1000.0f);
            total_weight += weight;
        }
    }
    
    // Find label with maximum weighted count
    int best_label = labels_in[idx];
    int max_count = 0;
    int ties = 0;
    
    for (int label = 0; label <= max_label; label++) {
        if (local_label_counts[label] > max_count) {
            max_count = local_label_counts[label];
            best_label = label;
            ties = 1;
        } else if (local_label_counts[label] == max_count && max_count > 0) {
            ties++;
        }
    }
    
    // Break ties randomly if multiple labels have same count
    if (ties > 1 && max_count > 0) {
        curandState local_state = rand_states[idx];
        int tie_breaker = curand(&local_state) % ties;
        rand_states[idx] = local_state;
        
        int current_tie = 0;
        for (int label = 0; label <= max_label; label++) {
            if (local_label_counts[label] == max_count) {
                if (current_tie == tie_breaker) {
                    best_label = label;
                    break;
                }
                current_tie++;
            }
        }
    }
    
    labels_out[idx] = best_label;
}

/**
 * Asynchronous label propagation kernel - updates happen in-place
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes one node with immediate updates
 */
__global__ void propagate_labels_async_kernel(
    int* __restrict__ labels,
    const int* __restrict__ edge_row_offsets,
    const int* __restrict__ edge_col_indices,
    const float* __restrict__ edge_weights,
    const int num_nodes,
    const int max_label,
    curandState* __restrict__ rand_states)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int start_edge = edge_row_offsets[idx];
    int end_edge = edge_row_offsets[idx + 1];
    
    if (start_edge == end_edge) {
        return; // Isolated node keeps current label
    }
    
    // Use shared memory for label frequency counting
    extern __shared__ int shared_memory[];
    int* local_label_counts = shared_memory + threadIdx.x * (max_label + 1);
    
    // Initialize local label counts
    for (int i = 0; i <= max_label; i++) {
        local_label_counts[i] = 0;
    }
    
    // Count weighted neighbor labels (reading potentially updated values)
    for (int i = start_edge; i < end_edge; ++i) {
        int neighbor_idx = edge_col_indices[i];
        int neighbor_label = labels[neighbor_idx];  // May be updated by other threads
        float weight = edge_weights[i];
        
        if (neighbor_label >= 0 && neighbor_label <= max_label) {
            local_label_counts[neighbor_label] += (int)(weight * 1000.0f);
        }
    }
    
    // Find label with maximum weighted count
    int best_label = labels[idx];
    int max_count = 0;
    int ties = 0;
    
    for (int label = 0; label <= max_label; label++) {
        if (local_label_counts[label] > max_count) {
            max_count = local_label_counts[label];
            best_label = label;
            ties = 1;
        } else if (local_label_counts[label] == max_count && max_count > 0) {
            ties++;
        }
    }
    
    // Break ties randomly
    if (ties > 1 && max_count > 0) {
        curandState local_state = rand_states[idx];
        int tie_breaker = curand(&local_state) % ties;
        rand_states[idx] = local_state;
        
        int current_tie = 0;
        for (int label = 0; label <= max_label; label++) {
            if (local_label_counts[label] == max_count) {
                if (current_tie == tie_breaker) {
                    best_label = label;
                    break;
                }
                current_tie++;
            }
        }
    }
    
    // Update label in-place (asynchronous)
    labels[idx] = best_label;
}

/**
 * Check convergence by comparing old and new labels
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread checks one node, uses atomics for global convergence flag
 */
__global__ void check_convergence_kernel(
    const int* __restrict__ labels_old,
    const int* __restrict__ labels_new,
    int* __restrict__ convergence_flag,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    // If any label changed, mark as not converged
    if (labels_old[idx] != labels_new[idx]) {
        atomicExch(convergence_flag, 0);
    }
}

/**
 * Compute modularity score for community quality assessment
 * Grid: (ceil(num_edges/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes one edge contribution to modularity
 */
__global__ void compute_modularity_kernel(
    const int* __restrict__ labels,
    const int* __restrict__ edge_row_offsets,
    const int* __restrict__ edge_col_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ node_degrees,
    float* __restrict__ modularity_contributions,
    const int num_nodes,
    const float total_weight)
{
    const int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx >= num_nodes) return;
    
    float contribution = 0.0f;
    int start_edge = edge_row_offsets[node_idx];
    int end_edge = edge_row_offsets[node_idx + 1];
    
    int node_label = labels[node_idx];
    float node_degree = node_degrees[node_idx];
    
    // Process all edges from this node
    for (int i = start_edge; i < end_edge; ++i) {
        int neighbor_idx = edge_col_indices[i];
        int neighbor_label = labels[neighbor_idx];
        float edge_weight = edge_weights[i];
        float neighbor_degree = node_degrees[neighbor_idx];
        
        // Modularity contribution: A_ij - (k_i * k_j)/(2m)
        float expected_weight = (node_degree * neighbor_degree) / (2.0f * total_weight);
        
        if (node_label == neighbor_label) {
            contribution += edge_weight - expected_weight;
        } else {
            contribution -= expected_weight;
        }
    }
    
    modularity_contributions[node_idx] = contribution;
}

/**
 * Initialize random states for tie-breaking in label propagation
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread initializes one random state
 */
__global__ void init_random_states_kernel(
    curandState* __restrict__ rand_states,
    const int num_nodes,
    const unsigned int seed)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    // Initialize random state for this thread
    curand_init(seed + idx, idx, 0, &rand_states[idx]);
}

/**
 * Compute node degrees for modularity calculation
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread computes degree for one node
 */
__global__ void compute_node_degrees_kernel(
    const int* __restrict__ edge_row_offsets,
    const float* __restrict__ edge_weights,
    float* __restrict__ node_degrees,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int start_edge = edge_row_offsets[idx];
    int end_edge = edge_row_offsets[idx + 1];
    
    float degree = 0.0f;
    for (int i = start_edge; i < end_edge; ++i) {
        degree += edge_weights[i];
    }
    
    node_degrees[idx] = degree;
}

/**
 * Count community sizes after label propagation
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes one node and atomically updates community counts
 */
__global__ void count_community_sizes_kernel(
    const int* __restrict__ labels,
    int* __restrict__ community_sizes,
    const int num_nodes,
    const int max_communities)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int label = labels[idx];
    if (label >= 0 && label < max_communities) {
        atomicAdd(&community_sizes[label], 1);
    }
}

/**
 * Relabel communities to remove gaps (compact labeling)
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each thread processes one node
 */
__global__ void relabel_communities_kernel(
    int* __restrict__ labels,
    const int* __restrict__ label_mapping,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    int old_label = labels[idx];
    if (old_label >= 0) {
        labels[idx] = label_mapping[old_label];
    }
}

} // extern "C"