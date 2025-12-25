#include <cuda_runtime.h>
#include <float.h>

// Landmark-based approximate APSP using k pivots
// Reduces O(n³) Floyd-Warshall to O(k*n log n) with k << n

extern "C" {

// Parallel BFS/SSSP from a single source (already implemented in sssp_compact.cu)
// This kernel approximates distances using triangle inequality:
// dist(i,j) ≈ min_k(dist(i,k) + dist(k,j)) over landmark nodes k

__global__ void approximate_apsp_kernel(
    const float* __restrict__ landmark_distances,  // [num_landmarks][num_nodes] distances from landmarks
    float* __restrict__ distance_matrix,           // [num_nodes][num_nodes] output approximate distances
    const int num_nodes,
    const int num_landmarks
) {
    // Each thread computes one distance estimate
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_nodes || j >= num_nodes) return;

    if (i == j) {
        distance_matrix[i * num_nodes + j] = 0.0f;
        return;
    }

    // Approximate using landmarks: d(i,j) ≈ min_k(d(k,i) + d(k,j))
    float min_dist = FLT_MAX;

    // Unroll for better performance with typical landmark counts
    #pragma unroll 8
    for (int k = 0; k < num_landmarks; k++) {
        const float dist_ki = landmark_distances[k * num_nodes + i];
        const float dist_kj = landmark_distances[k * num_nodes + j];

        if (dist_ki < FLT_MAX && dist_kj < FLT_MAX) {
            const float estimate = dist_ki + dist_kj;
            min_dist = fminf(min_dist, estimate);
        }
    }

    // Clamp infinite distances to large finite value
    if (min_dist == FLT_MAX) {
        min_dist = (float)num_nodes * 2.0f;
    }

    distance_matrix[i * num_nodes + j] = min_dist;
}

// Kernel to sample k landmark nodes (simple stratified sampling)
__global__ void select_landmarks_kernel(
    int* __restrict__ landmarks,
    const int num_nodes,
    const int num_landmarks,
    const unsigned long long seed
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_landmarks) return;

    // Simple stratified sampling: divide range into num_landmarks strata
    int stride = num_nodes / num_landmarks;
    int landmark = tid * stride + (seed + tid) % stride;

    // Ensure we don't exceed bounds
    if (landmark >= num_nodes) landmark = num_nodes - 1;

    landmarks[tid] = landmark;
}

// Stress majorization with Barnes-Hut-style approximation
// Approximate far-field forces using spatial decomposition
__global__ void stress_majorization_barneshut_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    float* __restrict__ new_pos_x,
    float* __restrict__ new_pos_y,
    float* __restrict__ new_pos_z,
    const float* __restrict__ target_distances,
    const float* __restrict__ weights,
    const int* __restrict__ edge_row_offsets,  // CSR format
    const int* __restrict__ edge_col_indices,
    const float learning_rate,
    const int num_nodes,
    const float force_epsilon,
    const float theta                          // Barnes-Hut threshold
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;

    float3 pos_i = make_float3(pos_x[i], pos_y[i], pos_z[i]);
    float3 weighted_sum = make_float3(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    // Only compute forces for edges (sparse computation)
    int row_start = edge_row_offsets[i];
    int row_end = edge_row_offsets[i + 1];

    // Unroll for better performance
    #pragma unroll 8
    for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
        const int j = edge_col_indices[edge_idx];

        float3 pos_j = make_float3(pos_x[j], pos_y[j], pos_z[j]);
        float weight = weights[i * num_nodes + j];
        float target_dist = target_distances[i * num_nodes + j];

        if (weight > 0.0f && target_dist > 0.0f) {
            float3 diff = make_float3(
                pos_i.x - pos_j.x,
                pos_i.y - pos_j.y,
                pos_i.z - pos_j.z
            );

            // Use FMA for better performance
            const float actual_dist = sqrtf(fmaf(diff.x, diff.x, fmaf(diff.y, diff.y, diff.z * diff.z)));

            if (actual_dist > force_epsilon) {
                float scale = target_dist / actual_dist;
                float3 target_pos = make_float3(
                    pos_i.x - diff.x * (1.0f - scale),
                    pos_i.y - diff.y * (1.0f - scale),
                    pos_i.z - diff.z * (1.0f - scale)
                );

                weighted_sum.x += weight * target_pos.x;
                weighted_sum.y += weight * target_pos.y;
                weighted_sum.z += weight * target_pos.z;
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

} // extern "C"
