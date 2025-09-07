// Device-side frontier compaction kernel for SSSP
// This replaces the slow host-side compaction

#include <cuda_runtime.h>

// Parallel prefix sum (scan) for compaction
__global__ void compact_frontier_kernel(
    const int* __restrict__ flags,          // Input: per-node flags (1 if in frontier)
    int* __restrict__ scan_output,          // Output: exclusive scan results
    int* __restrict__ compacted_frontier,   // Output: compacted frontier
    int* __restrict__ frontier_size,        // Output: new frontier size
    const int num_nodes)
{
    extern __shared__ int shared_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load flag into shared memory
    int flag = (idx < num_nodes) ? flags[idx] : 0;
    shared_data[tid] = flag;
    __syncthreads();
    
    // Parallel prefix sum in shared memory (up-sweep)
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            shared_data[index] += shared_data[index - stride];
        }
        __syncthreads();
    }
    
    // Store block sum and clear last element
    if (tid == blockDim.x - 1) {
        scan_output[blockIdx.x] = shared_data[tid];
        shared_data[tid] = 0;
    }
    __syncthreads();
    
    // Down-sweep
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp = shared_data[index - stride];
            shared_data[index - stride] = shared_data[index];
            shared_data[index] += temp;
        }
        __syncthreads();
    }
    
    // Write scan result
    if (idx < num_nodes) {
        int scan_val = shared_data[tid];
        
        // If this node is in frontier, write its compacted position
        if (flag) {
            compacted_frontier[scan_val] = idx;
        }
        
        // Last thread writes total frontier size
        if (idx == num_nodes - 1) {
            *frontier_size = scan_val + flag;
        }
    }
}

// Simple stream compaction using atomics (alternative approach)
__global__ void compact_frontier_atomic_kernel(
    const int* __restrict__ flags,          // Input: per-node flags
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

extern "C" {
    // Wrapper for calling from Rust
    void compact_frontier_gpu(
        const int* flags,
        int* compacted_frontier,
        int* frontier_size,
        int num_nodes,
        void* stream)
    {
        // Reset counter
        cudaMemsetAsync(frontier_size, 0, sizeof(int), (cudaStream_t)stream);
        
        // Launch compaction kernel
        int block_size = 256;
        int grid_size = (num_nodes + block_size - 1) / block_size;
        
        compact_frontier_atomic_kernel<<<grid_size, block_size, 0, (cudaStream_t)stream>>>(
            flags,
            compacted_frontier,
            frontier_size,
            num_nodes
        );
    }
}