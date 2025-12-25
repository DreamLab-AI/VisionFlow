// GPU PageRank Centrality Implementation
// Implements the power iteration algorithm for PageRank computation
//
// PageRank Formula: PR(v) = (1-d)/N + d * Σ(PR(u)/deg(u))
// where d = damping factor (typically 0.85), N = number of nodes
//
// This kernel uses sparse CSR (Compressed Sparse Row) graph format for efficiency

#include <cuda_runtime.h>
#include <cmath>

// Device constants for PageRank computation
__constant__ float DAMPING_FACTOR = 0.85f;
__constant__ float EPSILON = 1e-6f;

/**
 * Kernel 1: Initialize PageRank values uniformly
 * Each node starts with PR = 1/N
 */
__global__ void pagerank_init_kernel(
    float* __restrict__ pagerank,       // Output: initial PageRank values
    const int num_nodes)                // Number of nodes in graph
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        pagerank[tid] = 1.0f / (float)num_nodes;
    }
}

/**
 * Kernel 2: Compute PageRank iteration using power method
 *
 * CSR Format:
 * - row_offsets[i] = start index in col_indices for node i's edges
 * - col_indices[j] = destination node for edge j
 * - edge_count[i] = number of outgoing edges from node i
 */
__global__ void pagerank_iteration_kernel(
    const float* __restrict__ pagerank_old,     // Previous iteration values
    float* __restrict__ pagerank_new,           // New iteration values
    const int* __restrict__ row_offsets,        // CSR row pointers
    const int* __restrict__ col_indices,        // CSR column indices
    const int* __restrict__ out_degree,         // Outgoing edge count per node
    const int num_nodes,                        // Number of nodes
    const float damping,                        // Damping factor (0.85)
    const float teleport)                       // Teleport probability (1-d)/N
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_nodes) {
        float rank_sum = 0.0f;

        // Sum contributions from all incoming edges
        // For each node, check all other nodes to see if they link to this one
        // Note: This is the "reverse" approach - we iterate over potential sources
        #pragma unroll 8
        for (int src = 0; src < num_nodes; src++) {
            const int edge_start = row_offsets[src];
            const int edge_end = row_offsets[src + 1];
            const int degree = out_degree[src];

            // Skip nodes with no outgoing edges (dangling nodes)
            if (degree == 0) continue;

            // Precompute contribution factor
            const float contribution = pagerank_old[src] / (float)degree;

            // Check if src has an edge to tid
            for (int e = edge_start; e < edge_end; e++) {
                if (col_indices[e] == tid) {
                    // Add contribution using FMA
                    rank_sum = fmaf(damping, contribution, rank_sum);
                    break;
                }
            }
        }

        // Apply PageRank formula: (1-d)/N + d * sum (already applied damping in loop)
        pagerank_new[tid] = teleport + rank_sum;
    }
}

/**
 * Kernel 3: Optimized PageRank iteration using shared memory
 * Uses shared memory to cache frequently accessed data
 */
__global__ void pagerank_iteration_optimized_kernel(
    const float* __restrict__ pagerank_old,
    float* __restrict__ pagerank_new,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const int* __restrict__ out_degree,
    const int num_nodes,
    const float damping,
    const float teleport)
{
    extern __shared__ float shared_pagerank[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Load pagerank_old into shared memory in chunks
    if (tid < num_nodes) {
        shared_pagerank[local_tid] = pagerank_old[tid];
    }
    __syncthreads();

    if (tid < num_nodes) {
        float rank_sum = 0.0f;

        // Iterate through incoming edges
        for (int src = 0; src < num_nodes; src++) {
            int edge_start = row_offsets[src];
            int edge_end = row_offsets[src + 1];
            int degree = out_degree[src];

            if (degree == 0) continue;

            // Check if src links to tid
            for (int e = edge_start; e < edge_end; e++) {
                if (col_indices[e] == tid) {
                    // Use shared memory when possible
                    float src_pr = (src / blockDim.x == blockIdx.x)
                        ? shared_pagerank[src % blockDim.x]
                        : pagerank_old[src];
                    rank_sum += src_pr / (float)degree;
                    break;
                }
            }
        }

        pagerank_new[tid] = teleport + damping * rank_sum;
    }
}

/**
 * Kernel 4: Compute convergence metric (L1 norm of difference)
 * Reduction kernel to check if PageRank has converged
 */
__global__ void pagerank_convergence_kernel(
    const float* __restrict__ pagerank_old,
    const float* __restrict__ pagerank_new,
    float* __restrict__ diff_buffer,        // Output: per-block differences
    const int num_nodes)
{
    extern __shared__ float shared_diff[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Compute local difference
    float local_diff = 0.0f;
    if (tid < num_nodes) {
        local_diff = fabsf(pagerank_new[tid] - pagerank_old[tid]);
    }
    shared_diff[local_tid] = local_diff;
    __syncthreads();

    // Parallel reduction in shared memory with unrolling
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (local_tid < stride) {
            shared_diff[local_tid] += shared_diff[local_tid + stride];
        }
        __syncthreads();
    }

    // Final warp reduction without synchronization
    if (local_tid < 32) {
        volatile float* smem = shared_diff;
        if (blockDim.x >= 64) smem[local_tid] += smem[local_tid + 32];
        if (blockDim.x >= 32) smem[local_tid] += smem[local_tid + 16];
        if (blockDim.x >= 16) smem[local_tid] += smem[local_tid + 8];
        if (blockDim.x >= 8)  smem[local_tid] += smem[local_tid + 4];
        if (blockDim.x >= 4)  smem[local_tid] += smem[local_tid + 2];
        if (blockDim.x >= 2)  smem[local_tid] += smem[local_tid + 1];
    }

    // First thread writes block result
    if (local_tid == 0) {
        diff_buffer[blockIdx.x] = shared_diff[0];
    }
}

/**
 * Kernel 5: Handle dangling nodes (nodes with no outgoing edges)
 * Redistributes their PageRank uniformly across all nodes
 */
__global__ void pagerank_dangling_kernel(
    float* __restrict__ pagerank_new,
    const float* __restrict__ pagerank_old,
    const int* __restrict__ out_degree,
    const int num_nodes,
    const float damping)
{
    __shared__ float dangling_sum;

    // First thread computes total dangling mass
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dangling_sum = 0.0f;
        for (int i = 0; i < num_nodes; i++) {
            if (out_degree[i] == 0) {
                dangling_sum += pagerank_old[i];
            }
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Distribute dangling mass uniformly
    if (tid < num_nodes) {
        float contribution = damping * dangling_sum / (float)num_nodes;
        pagerank_new[tid] += contribution;
    }
}

/**
 * Kernel 6: Normalize PageRank values to sum to 1.0
 * Ensures numerical stability
 */
__global__ void pagerank_normalize_kernel(
    float* __restrict__ pagerank,
    float* __restrict__ sum_buffer,         // Workspace for reduction
    const int num_nodes)
{
    extern __shared__ float shared_sum[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;

    // Load and sum in shared memory
    float local_val = (tid < num_nodes) ? pagerank[tid] : 0.0f;
    shared_sum[local_tid] = local_val;
    __syncthreads();

    // Reduction with unrolling
    #pragma unroll
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (local_tid < stride) {
            shared_sum[local_tid] += shared_sum[local_tid + stride];
        }
        __syncthreads();
    }

    // Final warp reduction without synchronization
    if (local_tid < 32) {
        volatile float* smem = shared_sum;
        if (blockDim.x >= 64) smem[local_tid] += smem[local_tid + 32];
        if (blockDim.x >= 32) smem[local_tid] += smem[local_tid + 16];
        if (blockDim.x >= 16) smem[local_tid] += smem[local_tid + 8];
        if (blockDim.x >= 8)  smem[local_tid] += smem[local_tid + 4];
        if (blockDim.x >= 4)  smem[local_tid] += smem[local_tid + 2];
        if (blockDim.x >= 2)  smem[local_tid] += smem[local_tid + 1];
    }

    // First thread of first block normalizes
    if (threadIdx.x == 0) {
        sum_buffer[blockIdx.x] = shared_sum[0];
    }
    __syncthreads();

    // After all blocks done, normalize (simplified - assumes single block)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < gridDim.x; i++) {
            total_sum += sum_buffer[i];
        }

        if (total_sum > 0.0f) {
            for (int i = 0; i < num_nodes; i++) {
                pagerank[i] /= total_sum;
            }
        }
    }
}

// C-style wrappers for Rust FFI
extern "C" {
    /**
     * Initialize PageRank values
     */
    void pagerank_init(
        float* pagerank,
        int num_nodes,
        void* stream)
    {
        int block_size = 256;
        int grid_size = (num_nodes + block_size - 1) / block_size;

        pagerank_init_kernel<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
            pagerank,
            num_nodes
        );
    }

    /**
     * Execute one PageRank iteration
     */
    void pagerank_iterate(
        const float* pagerank_old,
        float* pagerank_new,
        const int* row_offsets,
        const int* col_indices,
        const int* out_degree,
        int num_nodes,
        float damping,
        void* stream)
    {
        int block_size = 256;
        int grid_size = (num_nodes + block_size - 1) / block_size;

        float teleport = (1.0f - damping) / (float)num_nodes;

        pagerank_iteration_kernel<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
            pagerank_old,
            pagerank_new,
            row_offsets,
            col_indices,
            out_degree,
            num_nodes,
            damping,
            teleport
        );
    }

    /**
     * Execute optimized PageRank iteration with shared memory
     */
    void pagerank_iterate_optimized(
        const float* pagerank_old,
        float* pagerank_new,
        const int* row_offsets,
        const int* col_indices,
        const int* out_degree,
        int num_nodes,
        float damping,
        void* stream)
    {
        int block_size = 256;
        int grid_size = (num_nodes + block_size - 1) / block_size;
        size_t shared_mem_size = block_size * sizeof(float);

        float teleport = (1.0f - damping) / (float)num_nodes;

        pagerank_iteration_optimized_kernel<<<grid_size, block_size, shared_mem_size, (cudaStream_t)stream>>>(
            pagerank_old,
            pagerank_new,
            row_offsets,
            col_indices,
            out_degree,
            num_nodes,
            damping,
            teleport
        );
    }

    /**
     * Check convergence
     */
    float pagerank_check_convergence(
        const float* pagerank_old,
        const float* pagerank_new,
        float* diff_buffer,
        int num_nodes,
        void* stream)
    {
        int block_size = 256;
        int grid_size = (num_nodes + block_size - 1) / block_size;
        size_t shared_mem_size = block_size * sizeof(float);

        pagerank_convergence_kernel<<<grid_size, block_size, shared_mem_size, (cudaStream_t)stream>>>(
            pagerank_old,
            pagerank_new,
            diff_buffer,
            num_nodes
        );

        // Sum up block results on CPU (simplified)
        float total_diff = 0.0f;
        float* host_buffer = new float[grid_size];
        cudaMemcpyAsync(host_buffer, diff_buffer, grid_size * sizeof(float),
                       cudaMemcpyDeviceToHost, (cudaStream_t)stream);
        cudaStreamSynchronize((cudaStream_t)stream);

        for (int i = 0; i < grid_size; i++) {
            total_diff += host_buffer[i];
        }
        delete[] host_buffer;

        return total_diff;
    }

    /**
     * Handle dangling nodes
     */
    void pagerank_handle_dangling(
        float* pagerank_new,
        const float* pagerank_old,
        const int* out_degree,
        int num_nodes,
        float damping,
        void* stream)
    {
        int block_size = 256;
        int grid_size = (num_nodes + block_size - 1) / block_size;

        pagerank_dangling_kernel<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
            pagerank_new,
            pagerank_old,
            out_degree,
            num_nodes,
            damping
        );
    }
}
