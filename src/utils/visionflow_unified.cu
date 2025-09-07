// VisionFlow Unified GPU Kernel - Rewritten for correctness, performance, and clarity.
// Implements a two-pass (force/integrate) simulation with double-buffering,
// uniform grid spatial hashing for repulsion, and CSR for spring forces.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>

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
    float _padding;             // Padding for alignment
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
    const int num_constraints)
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
                                float max_repulsion = c_params.max_force * 0.5f;
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
                        float force_magnitude = -constraint.weight * error * 0.5f; // Spring-like force
                        
                        // Cap constraint forces to prevent instability
                        float max_constraint_force = c_params.max_force * 0.3f;
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
                    float force_magnitude = constraint.weight * distance * 0.1f; // Gentle attraction
                    
                    // Cap constraint forces
                    float max_constraint_force = c_params.max_force * 0.2f;
                    force_magnitude = fminf(force_magnitude, max_constraint_force);
                    
                    constraint_force = vec3_scale(diff, force_magnitude / distance);
                }
            }
            
            // Apply constraint force with safety checks
            if (isfinite(constraint_force.x) && isfinite(constraint_force.y) && isfinite(constraint_force.z)) {
                total_force = vec3_add(total_force, constraint_force);
            }
        }
    }

    force_out_x[idx] = total_force.x;
    force_out_y[idx] = total_force.y;
    force_out_z[idx] = total_force.z;
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
            boundary_force = fminf(boundary_force, 15.0f);
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
            boundary_force = fminf(boundary_force, 15.0f);
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

} // extern "C"