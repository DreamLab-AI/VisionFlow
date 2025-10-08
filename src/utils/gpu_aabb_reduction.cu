#include <cuda_runtime.h>
#include <float.h>

struct AABB {
    float3 min;
    float3 max;
};

// Warp-level primitives for min/max reduction
__device__ __forceinline__ float warpReduceMin(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// GPU AABB reduction kernel using parallel min/max
// Each block reduces its portion, final reduction happens on CPU
__global__ void compute_aabb_reduction_kernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ pos_z,
    AABB* block_results,
    int num_nodes
) {
    extern __shared__ float sdata[];
    float* s_min_x = sdata;
    float* s_min_y = sdata + blockDim.x;
    float* s_min_z = sdata + 2 * blockDim.x;
    float* s_max_x = sdata + 3 * blockDim.x;
    float* s_max_y = sdata + 4 * blockDim.x;
    float* s_max_z = sdata + 5 * blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize thread-local min/max
    float min_x = FLT_MAX, min_y = FLT_MAX, min_z = FLT_MAX;
    float max_x = -FLT_MAX, max_y = -FLT_MAX, max_z = -FLT_MAX;

    // Grid-stride loop for coalesced memory access
    for (int i = idx; i < num_nodes; i += blockDim.x * gridDim.x) {
        float x = pos_x[i];
        float y = pos_y[i];
        float z = pos_z[i];

        min_x = fminf(min_x, x);
        min_y = fminf(min_y, y);
        min_z = fminf(min_z, z);

        max_x = fmaxf(max_x, x);
        max_y = fmaxf(max_y, y);
        max_z = fmaxf(max_z, z);
    }

    // Warp-level reduction
    min_x = warpReduceMin(min_x);
    min_y = warpReduceMin(min_y);
    min_z = warpReduceMin(min_z);
    max_x = warpReduceMax(max_x);
    max_y = warpReduceMax(max_y);
    max_z = warpReduceMax(max_z);

    // Write warp results to shared memory
    if (tid % 32 == 0) {
        int warp_id = tid / 32;
        s_min_x[warp_id] = min_x;
        s_min_y[warp_id] = min_y;
        s_min_z[warp_id] = min_z;
        s_max_x[warp_id] = max_x;
        s_max_y[warp_id] = max_y;
        s_max_z[warp_id] = max_z;
    }

    __syncthreads();

    // Final reduction in first warp
    if (tid < 32) {
        int num_warps = (blockDim.x + 31) / 32;
        min_x = (tid < num_warps) ? s_min_x[tid] : FLT_MAX;
        min_y = (tid < num_warps) ? s_min_y[tid] : FLT_MAX;
        min_z = (tid < num_warps) ? s_min_z[tid] : FLT_MAX;
        max_x = (tid < num_warps) ? s_max_x[tid] : -FLT_MAX;
        max_y = (tid < num_warps) ? s_max_y[tid] : -FLT_MAX;
        max_z = (tid < num_warps) ? s_max_z[tid] : -FLT_MAX;

        min_x = warpReduceMin(min_x);
        min_y = warpReduceMin(min_y);
        min_z = warpReduceMin(min_z);
        max_x = warpReduceMax(max_x);
        max_y = warpReduceMax(max_y);
        max_z = warpReduceMax(max_z);

        // Thread 0 writes block result
        if (tid == 0) {
            AABB result;
            result.min = make_float3(min_x, min_y, min_z);
            result.max = make_float3(max_x, max_y, max_z);
            block_results[blockIdx.x] = result;
        }
    }
}
