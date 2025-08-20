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
};

struct FeatureFlags {
    static const unsigned int ENABLE_REPULSION = 1 << 0;
    static const unsigned int ENABLE_SPRINGS = 1 << 1;
    static const unsigned int ENABLE_CENTERING = 1 << 2;
};

struct AABB {
    float3 min;
    float3 max;
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
    const SimParams params,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float3 my_pos = make_vec3(pos_in_x[idx], pos_in_y[idx], pos_in_z[idx]);
    float3 total_force = make_vec3(0.0f, 0.0f, 0.0f);

    if (params.feature_flags & FeatureFlags::ENABLE_REPULSION) {
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

                            if (dist_sq < params.repulsion_cutoff * params.repulsion_cutoff && dist_sq > 1e-6f) {
                                float dist = sqrtf(dist_sq);
                                float repulsion = params.repel_k / (dist_sq + params.repulsion_softening_epsilon);
                                total_force = vec3_add(total_force, vec3_scale(diff, repulsion / dist));
                            }
                        }
                    }
                }
            }
        }
    }

    if (params.feature_flags & FeatureFlags::ENABLE_SPRINGS) {
        int start_edge = edge_row_offsets[idx];
        int end_edge = edge_row_offsets[idx + 1];
        for (int i = start_edge; i < end_edge; ++i) {
            int neighbor_idx = edge_col_indices[i];
            float3 neighbor_pos = make_vec3(pos_in_x[neighbor_idx], pos_in_y[neighbor_idx], pos_in_z[neighbor_idx]);
            
            float3 diff = vec3_sub(neighbor_pos, my_pos);
            float dist = vec3_length(diff);
            
            if (dist > 1e-6f) {
                float displacement = dist - params.rest_length;
                float spring_force_mag = params.spring_k * displacement * edge_weights[i];
                total_force = vec3_add(total_force, vec3_scale(diff, spring_force_mag / dist));
            }
        }
    }
    
    if (params.feature_flags & FeatureFlags::ENABLE_CENTERING) {
        total_force = vec3_sub(total_force, vec3_scale(my_pos, params.center_gravity_k));
    }

    force_out_x[idx] = total_force.x;
    force_out_y[idx] = total_force.y;
    force_out_z[idx] = total_force.z;
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
    const SimParams params,
    const int num_nodes)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float3 pos = make_vec3(pos_in_x[idx], pos_in_y[idx], pos_in_z[idx]);
    float3 vel = make_vec3(vel_in_x[idx], vel_in_y[idx], vel_in_z[idx]);
    float3 force = make_vec3(force_x[idx], force_y[idx], force_z[idx]);
    float node_mass = (mass != nullptr && mass[idx] > 0.0f) ? mass[idx] : 1.0f;

    force = vec3_clamp(force, params.max_force);

    float effective_damping = params.damping;
    if (params.iteration < params.warmup_iterations) {
        float warmup_factor = (float)params.iteration / (float)params.warmup_iterations;
        force = vec3_scale(force, warmup_factor * warmup_factor);
        effective_damping = 1.0f - (1.0f - params.damping) * warmup_factor;
    }

    vel = vec3_add(vel, vec3_scale(force, params.dt / node_mass));
    vel = vec3_scale(vel, effective_damping);
    vel = vec3_clamp(vel, params.max_velocity);
    pos = vec3_add(pos, vec3_scale(vel, params.dt));

    pos_out_x[idx] = pos.x;
    pos_out_y[idx] = pos.y;
    pos_out_z[idx] = pos.z;
    vel_out_x[idx] = vel.x;
    vel_out_y[idx] = vel.y;
    vel_out_z[idx] = vel.z;
}

} // extern "C"