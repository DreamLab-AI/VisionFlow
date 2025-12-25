// Unified Stress Majorization GPU Kernel - AUTHORITATIVE IMPLEMENTATION
// Consolidates duplicate implementations from:
// - src/utils/stress_majorization.cu
// - src/utils/gpu_clustering_kernels.cu
// - src/utils/gpu_landmark_apsp.cu
//
// This kernel provides global layout optimization to complement local force-directed
// layout. It minimizes layout stress by iteratively adjusting node positions to better
// match ideal graph-theoretic distances.
//
// Performance target: Process 100k nodes in <100ms per optimization cycle

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cmath>

extern "C" {

// =============================================================================
// Helper Functions
// =============================================================================

__device__ inline float clamp_float(float x, float min_val, float max_val) {
    return fminf(fmaxf(x, min_val), max_val);
}

__device__ inline float safe_sqrt(float x) {
    return sqrtf(fmaxf(x, 1e-10f));
}

__device__ inline float compute_distance_3d(
    float x1, float y1, float z1,
    float x2, float y2, float z2
) {
    const float dx = x1 - x2;
    const float dy = y1 - y2;
    const float dz = z1 - z2;
    // Use FMA for better performance and accuracy
    return safe_sqrt(fmaf(dx, dx, fmaf(dy, dy, dz * dz)));
}

// =============================================================================
// Stress Majorization Kernels - UNIFIED IMPLEMENTATION
// =============================================================================

/**
 * Compute stress function value
 * Stress = sum(w_ij * (d_ij - ||p_i - p_j||)^2)
 * where d_ij is ideal distance, p_i is current position, w_ij is weight
 */
__global__ void compute_stress_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ ideal_distances,
    const float* __restrict__ weights,
    float* __restrict__ stress_values,
    const int num_nodes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float local_stress = 0.0f;

    // Loop unrolling hint for better performance
    #pragma unroll 8
    for (int j = 0; j < num_nodes; j++) {
        if (i == j) continue;

        // Current distance
        const float current_dist = compute_distance_3d(
            pos_x[i], pos_y[i], pos_z[i],
            pos_x[j], pos_y[j], pos_z[j]
        );

        // Ideal distance from matrix
        const int idx = i * num_nodes + j;
        const float ideal_dist = ideal_distances[idx];
        const float weight = weights[idx];

        // Stress contribution: w_ij * (d_ij - ||p_i - p_j||)^2
        // Use FMA for accumulation
        const float diff = ideal_dist - current_dist;
        local_stress = fmaf(weight, diff * diff, local_stress);
    }

    stress_values[i] = local_stress;
}

/**
 * Compute gradient for stress majorization
 * Gradient_i = sum(w_ij * (1 - d_ij / ||p_i - p_j||) * (p_i - p_j))
 */
__global__ void compute_stress_gradient_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    const float* __restrict__ ideal_distances,
    const float* __restrict__ weights,
    float* __restrict__ grad_x,
    float* __restrict__ grad_y,
    float* __restrict__ grad_z,
    const int num_nodes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float gx = 0.0f;
    float gy = 0.0f;
    float gz = 0.0f;

    // Loop unrolling hint for better performance
    #pragma unroll 8
    for (int j = 0; j < num_nodes; j++) {
        if (i == j) continue;

        // Current position difference
        const float dx = pos_x[i] - pos_x[j];
        const float dy = pos_y[i] - pos_y[j];
        const float dz = pos_z[i] - pos_z[j];

        // Use FMA for distance calculation
        const float current_dist = safe_sqrt(fmaf(dx, dx, fmaf(dy, dy, dz * dz)));

        // Avoid division by zero
        if (current_dist < 1e-6f) continue;

        const int idx = i * num_nodes + j;
        const float ideal_dist = ideal_distances[idx];
        const float weight = weights[idx];

        // Gradient factor: w_ij * (1 - d_ij / ||p_i - p_j||)
        const float factor = weight * (1.0f - ideal_dist / current_dist);

        // Use FMA for gradient accumulation
        gx = fmaf(factor, dx, gx);
        gy = fmaf(factor, dy, gy);
        gz = fmaf(factor, dz, gz);
    }

    grad_x[i] = gx;
    grad_y[i] = gy;
    grad_z[i] = gz;
}

/**
 * Update positions using gradient descent with momentum
 * p_new = p_old - learning_rate * gradient + momentum * velocity
 */
__global__ void update_positions_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ grad_x,
    const float* __restrict__ grad_y,
    const float* __restrict__ grad_z,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ vel_z,
    const float learning_rate,
    const float momentum,
    const float max_displacement,
    const int num_nodes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    // Update velocity with momentum using FMA
    float vx = fmaf(momentum, vel_x[i], -learning_rate * grad_x[i]);
    float vy = fmaf(momentum, vel_y[i], -learning_rate * grad_y[i]);
    float vz = fmaf(momentum, vel_z[i], -learning_rate * grad_z[i]);

    // Clamp displacement magnitude using FMA
    const float displacement_sq = fmaf(vx, vx, fmaf(vy, vy, vz * vz));
    if (displacement_sq > max_displacement * max_displacement) {
        float scale = max_displacement / safe_sqrt(displacement_sq);
        vx *= scale;
        vy *= scale;
        vz *= scale;
    }

    // Update velocity
    vel_x[i] = vx;
    vel_y[i] = vy;
    vel_z[i] = vz;

    // Update position
    pos_x[i] += vx;
    pos_y[i] += vy;
    pos_z[i] += vz;
}

/**
 * UNIFIED sparse stress majorization step using CSR edge list (O(m) instead of O(n²))
 * Combines best implementations from gpu_clustering_kernels.cu and gpu_landmark_apsp.cu
 */
__global__ void stress_majorization_step_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ new_pos_x,
    float* __restrict__ new_pos_y,
    float* __restrict__ new_pos_z,
    const float* __restrict__ target_distances,
    const float* __restrict__ weights,
    const int* __restrict__ edge_row_offsets,
    const int* __restrict__ edge_col_indices,
    const float learning_rate,
    const int num_nodes,
    const float force_epsilon
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float3 pos_i = make_float3(pos_x[i], pos_y[i], pos_z[i]);
    float3 weighted_sum = make_float3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    // Only iterate over edges (CSR sparse format)
    int row_start = edge_row_offsets[i];
    int row_end = edge_row_offsets[i + 1];

    // Unroll for better performance
    #pragma unroll 8
    for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
        const int j = edge_col_indices[edge_idx];

        const float3 pos_j = make_float3(pos_x[j], pos_y[j], pos_z[j]);
        const float weight = weights[i * num_nodes + j];
        const float target_dist = target_distances[i * num_nodes + j];

        if (weight > 0.0f && target_dist > 0.0f) {
            const float dx = pos_i.x - pos_j.x;
            const float dy = pos_i.y - pos_j.y;
            const float dz = pos_i.z - pos_j.z;

            // Use FMA for distance calculation
            const float actual_dist = sqrtf(fmaf(dx, dx, fmaf(dy, dy, dz * dz)));

            if (actual_dist > force_epsilon) {
                const float scale = target_dist / actual_dist;
                const float one_minus_scale = 1.0f - scale;
                const float3 target_pos = make_float3(
                    fmaf(-dx, one_minus_scale, pos_i.x),
                    fmaf(-dy, one_minus_scale, pos_i.y),
                    fmaf(-dz, one_minus_scale, pos_i.z)
                );

                // Use FMA for weighted sum accumulation
                weighted_sum.x = fmaf(weight, target_pos.x, weighted_sum.x);
                weighted_sum.y = fmaf(weight, target_pos.y, weighted_sum.y);
                weighted_sum.z = fmaf(weight, target_pos.z, weighted_sum.z);
                weight_sum += weight;
            }
        }
    }

    // Apply update with learning rate
    if (weight_sum > 0.0f) {
        float3 new_pos = make_float3(
            weighted_sum.x / weight_sum,
            weighted_sum.y / weight_sum,
            weighted_sum.z / weight_sum
        );

        new_pos_x[i] = pos_i.x + learning_rate * (new_pos.x - pos_i.x);
        new_pos_y[i] = pos_i.y + learning_rate * (new_pos.y - pos_i.y);
        new_pos_z[i] = pos_i.z + learning_rate * (new_pos.z - pos_i.z);
    } else {
        // No valid neighbors, keep current position
        new_pos_x[i] = pos_i.x;
        new_pos_y[i] = pos_i.y;
        new_pos_z[i] = pos_i.z;
    }
}

/**
 * Apply stress majorization step using Laplacian system
 * Solves: L * p_new = b, where b is computed from current positions
 */
__global__ void majorization_step_kernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ pos_z,
    const float* __restrict__ ideal_distances,
    const float* __restrict__ weights,
    float* __restrict__ temp_x,
    float* __restrict__ temp_y,
    float* __restrict__ temp_z,
    const int num_nodes,
    const float blend_factor
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    float sum_weights = 0.0f;

    for (int j = 0; j < num_nodes; j++) {
        if (i == j) continue;

        float current_dist = compute_distance_3d(
            pos_x[i], pos_y[i], pos_z[i],
            pos_x[j], pos_y[j], pos_z[j]
        );

        if (current_dist < 1e-6f) continue;

        int idx = i * num_nodes + j;
        float ideal_dist = ideal_distances[idx];
        float weight = weights[idx];

        // Weighted average position adjusted by ideal distance
        float factor = weight * ideal_dist / current_dist;

        sum_x += factor * pos_x[j];
        sum_y += factor * pos_y[j];
        sum_z += factor * pos_z[j];
        sum_weights += weight;
    }

    // New position from majorization
    if (sum_weights > 1e-6f) {
        float new_x = sum_x / sum_weights;
        float new_y = sum_y / sum_weights;
        float new_z = sum_z / sum_weights;

        // Blend with current position (for stability)
        temp_x[i] = blend_factor * new_x + (1.0f - blend_factor) * pos_x[i];
        temp_y[i] = blend_factor * new_y + (1.0f - blend_factor) * pos_y[i];
        temp_z[i] = blend_factor * new_z + (1.0f - blend_factor) * pos_z[i];
    } else {
        // No update if no weights
        temp_x[i] = pos_x[i];
        temp_y[i] = pos_y[i];
        temp_z[i] = pos_z[i];
    }
}

/**
 * Copy temporary positions back to main position buffers
 */
__global__ void copy_positions_kernel(
    float* __restrict__ dest_x,
    float* __restrict__ dest_y,
    float* __restrict__ dest_z,
    const float* __restrict__ src_x,
    const float* __restrict__ src_y,
    const float* __restrict__ src_z,
    const int num_nodes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    dest_x[i] = src_x[i];
    dest_y[i] = src_y[i];
    dest_z[i] = src_z[i];
}

/**
 * Compute convergence metric (maximum displacement)
 */
__global__ void compute_max_displacement_kernel(
    const float* __restrict__ old_x,
    const float* __restrict__ old_y,
    const float* __restrict__ old_z,
    const float* __restrict__ new_x,
    const float* __restrict__ new_y,
    const float* __restrict__ new_z,
    float* __restrict__ displacements,
    const int num_nodes
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float dx = new_x[i] - old_x[i];
    float dy = new_y[i] - old_y[i];
    float dz = new_z[i] - old_z[i];

    displacements[i] = safe_sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Parallel reduction to find maximum value
 */
__global__ void reduce_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : -FLT_MAX;
    __syncthreads();

    // Reduction in shared memory with unrolling
    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Final warp reduction without synchronization
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] = fmaxf(smem[tid], smem[tid + 32]);
        if (blockDim.x >= 32) smem[tid] = fmaxf(smem[tid], smem[tid + 16]);
        if (blockDim.x >= 16) smem[tid] = fmaxf(smem[tid], smem[tid + 8]);
        if (blockDim.x >= 8)  smem[tid] = fmaxf(smem[tid], smem[tid + 4]);
        if (blockDim.x >= 4)  smem[tid] = fmaxf(smem[tid], smem[tid + 2]);
        if (blockDim.x >= 2)  smem[tid] = fmaxf(smem[tid], smem[tid + 1]);
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * Reduce sum for computing total stress
 */
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int n
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Reduction in shared memory with unrolling
    #pragma unroll
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp reduction without synchronization
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64) smem[tid] += smem[tid + 32];
        if (blockDim.x >= 32) smem[tid] += smem[tid + 16];
        if (blockDim.x >= 16) smem[tid] += smem[tid + 8];
        if (blockDim.x >= 8)  smem[tid] += smem[tid + 4];
        if (blockDim.x >= 4)  smem[tid] += smem[tid + 2];
        if (blockDim.x >= 2)  smem[tid] += smem[tid + 1];
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

} // extern "C"
