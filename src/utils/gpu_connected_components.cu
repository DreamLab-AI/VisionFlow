// WARNING: This file is not compiled by build.rs. Kernels are accessed via
// kernel_bridge.rs FFI which expects a pre-linked library. See build.rs for
// the list of compiled .cu files.
//
// GPU kernel for connected components using label propagation
//
// This kernel implements parallel label propagation to find connected components.
// Each node starts with its own label, then iteratively adopts the minimum label
// of its neighbors until convergence.

#include <cuda_runtime.h>

extern "C" {

/// Label propagation kernel for connected components
/// Each thread processes one node and updates its label to the minimum of its neighbors
__global__ void label_propagation_kernel(
    const int* __restrict__ edge_row_offsets,   // CSR row offsets [num_nodes + 1]
    const int* __restrict__ edge_col_indices,   // CSR column indices [num_edges]
    int* __restrict__ labels,                   // Node labels [num_nodes]
    int* __restrict__ changed,                  // Flag: did any label change?
    const int num_nodes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node >= num_nodes) return;

    int current_label = labels[node];
    int min_label = current_label;

    // Find minimum label among neighbors
    int row_start = edge_row_offsets[node];
    int row_end = edge_row_offsets[node + 1];

    // Unroll neighbor iteration for better performance
    #pragma unroll 8
    for (int edge_idx = row_start; edge_idx < row_end; edge_idx++) {
        const int neighbor = edge_col_indices[edge_idx];
        const int neighbor_label = labels[neighbor];

        // Use min() intrinsic for branchless comparison
        min_label = min(min_label, neighbor_label);
    }

    // Update label if a smaller one was found
    if (min_label < current_label) {
        labels[node] = min_label;
        atomicAdd(changed, 1);  // Signal that a change occurred
    }
}

/// Initialize labels kernel - each node gets its own ID as initial label
__global__ void initialize_labels_kernel(
    int* __restrict__ labels,
    const int num_nodes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < num_nodes) {
        labels[node] = node;
    }
}

/// Count components kernel - compact unique labels
__global__ void count_components_kernel(
    const int* __restrict__ labels,
    int* __restrict__ component_map,    // Maps old label -> component ID
    int* __restrict__ component_count,  // Output: total components
    const int num_nodes
) {
    // This is a simplified version - in production, use parallel reduction
    // or thrust::unique for better performance

    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < num_nodes) {
        int label = labels[node];

        // Mark this label as seen using atomic operation
        int old_val = atomicCAS(&component_map[label], -1, 0);

        if (old_val == -1) {
            // First time seeing this label - it's a new component
            int comp_id = atomicAdd(component_count, 1);
            component_map[label] = comp_id;
        }
    }
}

/// Host-callable wrapper for connected components computation
void compute_connected_components_gpu(
    const int* edge_row_offsets,
    const int* edge_col_indices,
    int* labels,
    int* num_components,
    const int num_nodes,
    const int max_iterations,
    void* stream
) {
    cudaStream_t cuda_stream = (cudaStream_t)stream;

    int block_size = 256;
    int grid_size = (num_nodes + block_size - 1) / block_size;

    // Allocate changed flag
    int* d_changed;
    cudaError_t err = cudaMalloc(&d_changed, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc d_changed failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // Initialize labels
    initialize_labels_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        labels,
        num_nodes
    );

    // Iteratively propagate labels until convergence
    int iteration = 0;
    int h_changed = 1;

    while (h_changed > 0 && iteration < max_iterations) {
        // Reset changed flag
        cudaMemsetAsync(d_changed, 0, sizeof(int), cuda_stream);

        // Propagate labels
        label_propagation_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
            edge_row_offsets,
            edge_col_indices,
            labels,
            d_changed,
            num_nodes
        );

        // Check if any label changed
        cudaMemcpyAsync(&h_changed, d_changed, sizeof(int),
                       cudaMemcpyDeviceToHost, cuda_stream);
        cudaStreamSynchronize(cuda_stream);

        iteration++;
    }

    // Count unique components
    int* d_component_map;
    int* d_component_count;

    err = cudaMalloc(&d_component_map, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc d_component_map failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_changed);
        return;
    }
    err = cudaMalloc(&d_component_count, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc d_component_count failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_changed);
        cudaFree(d_component_map);
        return;
    }

    // Initialize component map to -1
    cudaMemsetAsync(d_component_map, -1, num_nodes * sizeof(int), cuda_stream);
    cudaMemsetAsync(d_component_count, 0, sizeof(int), cuda_stream);

    // Count components
    count_components_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
        labels,
        d_component_map,
        d_component_count,
        num_nodes
    );

    // Copy result back
    cudaMemcpyAsync(num_components, d_component_count, sizeof(int),
                   cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);

    // Cleanup
    cudaFree(d_changed);
    cudaFree(d_component_map);
    cudaFree(d_component_count);
}

} // extern "C"
