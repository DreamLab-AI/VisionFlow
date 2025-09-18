// VisionFlow GPU Stability Gate Implementation
// Optimized kernel for calculating kinetic energy and determining physics stability
// Prevents 100% GPU usage when graph is stable by early-exiting physics computation

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cfloat>

extern "C" {

// =============================================================================
// Kinetic Energy Calculation Kernel with Reduction
// =============================================================================

/**
 * Calculate per-node kinetic energy and perform block-level reduction
 * Grid: (ceil(num_nodes/256), 1, 1), Block: (256, 1, 1)
 * Each block computes partial kinetic energy sum
 */
__global__ void calculate_kinetic_energy_kernel(
    const float* __restrict__ vel_x,
    const float* __restrict__ vel_y,
    const float* __restrict__ vel_z,
    const float* __restrict__ mass,
    float* __restrict__ partial_kinetic_energy,
    int* __restrict__ active_node_count,
    const int num_nodes,
    const float min_velocity_threshold_sq)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    const int idx = bid * block_size + tid;
    
    // Shared memory for block-level reduction
    extern __shared__ float shared_data[];
    float* shared_ke = shared_data;
    int* shared_active = (int*)&shared_data[block_size];
    
    // Initialize shared memory
    shared_ke[tid] = 0.0f;
    shared_active[tid] = 0;
    
    // Calculate kinetic energy for this thread's node
    if (idx < num_nodes) {
        float vx = vel_x[idx];
        float vy = vel_y[idx];
        float vz = vel_z[idx];
        float vel_sq = vx * vx + vy * vy + vz * vz;
        
        // Check if node is actively moving
        if (vel_sq > min_velocity_threshold_sq) {
            float node_mass = (mass != nullptr && mass[idx] > 0.0f) ? mass[idx] : 1.0f;
            shared_ke[tid] = 0.5f * node_mass * vel_sq;
            shared_active[tid] = 1;
        }
    }
    
    __syncthreads();
    
    // Block-level reduction for both kinetic energy and active count
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_ke[tid] += shared_ke[tid + s];
            shared_active[tid] += shared_active[tid + s];
        }
        __syncthreads();
    }
    
    // Store block results
    if (tid == 0) {
        partial_kinetic_energy[bid] = shared_ke[0];
        atomicAdd(active_node_count, shared_active[0]);
    }
}

/**
 * Final reduction kernel to sum partial kinetic energies
 * Grid: (1, 1, 1), Block: (min(num_blocks, 256), 1, 1)
 * Single block performs final reduction
 */
__global__ void reduce_kinetic_energy_kernel(
    const float* __restrict__ partial_kinetic_energy,
    float* __restrict__ total_kinetic_energy,
    float* __restrict__ avg_kinetic_energy,
    const int* __restrict__ active_node_count,
    const int num_blocks,
    const int num_nodes)
{
    extern __shared__ float shared_ke[];
    
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    // Load partial sums
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += block_size) {
        sum += partial_kinetic_energy[i];
    }
    shared_ke[tid] = sum;
    
    __syncthreads();
    
    // Final reduction
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_ke[tid] += shared_ke[tid + s];
        }
        __syncthreads();
    }
    
    // Store final results
    if (tid == 0) {
        float total_ke = shared_ke[0];
        *total_kinetic_energy = total_ke;
        
        // Calculate average based on active nodes to avoid division by zero
        int active_nodes = *active_node_count;
        if (active_nodes > 0) {
            *avg_kinetic_energy = total_ke / active_nodes;
        } else {
            *avg_kinetic_energy = 0.0f;
        }
    }
}

/**
 * Combined stability check kernel - checks both global and per-node stability
 * Returns early exit flag if system is stable
 * Grid: (1, 1, 1), Block: (1, 1, 1)
 */
__global__ void check_stability_kernel(
    const float* __restrict__ avg_kinetic_energy,
    const int* __restrict__ active_node_count,
    int* __restrict__ should_skip_physics,
    const float stability_threshold,
    const int num_nodes,
    const int iteration)
{
    float avg_ke = *avg_kinetic_energy;
    int active_nodes = *active_node_count;
    
    // System is considered stable if:
    // 1. Average kinetic energy is below threshold
    // 2. Very few nodes are actively moving (< 1% of total)
    bool energy_stable = avg_ke < stability_threshold;
    bool motion_stable = active_nodes < max(1, num_nodes / 100);
    
    if (energy_stable || motion_stable) {
        *should_skip_physics = 1;
        
        // Debug output every 10 seconds at 60 FPS
        if (iteration % 600 == 0) {
            printf("GPU STABILITY GATE: System stable - avg_KE=%.8f, active_nodes=%d/%d\n", 
                   avg_ke, active_nodes, num_nodes);
        }
    } else {
        *should_skip_physics = 0;
    }
}

/**
 * Optimized force pass kernel with early exit capability
 * Includes per-block stability checking for additional optimization
 */
__global__ void force_pass_with_stability_kernel(
    const float* __restrict__ pos_in_x,
    const float* __restrict__ pos_in_y,
    const float* __restrict__ pos_in_z,
    const float* __restrict__ vel_in_x,
    const float* __restrict__ vel_in_y,
    const float* __restrict__ vel_in_z,
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
    const float min_velocity_threshold_sq,
    const int* __restrict__ should_skip_physics)
{
    // Early exit if physics should be skipped
    if (*should_skip_physics) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_nodes) {
            // Zero out forces for stable system
            force_out_x[idx] = 0.0f;
            force_out_y[idx] = 0.0f;
            force_out_z[idx] = 0.0f;
        }
        return;
    }
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    // Check if this specific node is moving
    float vx = vel_in_x[idx];
    float vy = vel_in_y[idx];
    float vz = vel_in_z[idx];
    float vel_sq = vx * vx + vy * vy + vz * vz;
    
    // Skip force calculation for stationary nodes
    if (vel_sq < min_velocity_threshold_sq) {
        force_out_x[idx] = 0.0f;
        force_out_y[idx] = 0.0f;
        force_out_z[idx] = 0.0f;
        return;
    }
    
    // Continue with normal force calculation...
    // (Rest of force calculation code remains the same as original)
}

/**
 * Host-callable function to check system stability
 * Returns true if physics computation can be skipped
 */
__host__ bool check_system_stability(
    const float* d_vel_x,
    const float* d_vel_y,
    const float* d_vel_z,
    const float* d_mass,
    float stability_threshold,
    float min_velocity_threshold,
    int num_nodes,
    int iteration,
    cudaStream_t stream)
{
    const int block_size = 256;
    const int num_blocks = (num_nodes + block_size - 1) / block_size;
    const size_t shared_mem_size = block_size * (sizeof(float) + sizeof(int));
    
    // Allocate temporary buffers
    float* d_partial_ke;
    float* d_total_ke;
    float* d_avg_ke;
    int* d_active_count;
    int* d_should_skip;
    
    cudaMalloc(&d_partial_ke, num_blocks * sizeof(float));
    cudaMalloc(&d_total_ke, sizeof(float));
    cudaMalloc(&d_avg_ke, sizeof(float));
    cudaMalloc(&d_active_count, sizeof(int));
    cudaMalloc(&d_should_skip, sizeof(int));
    
    // Initialize counters
    cudaMemsetAsync(d_active_count, 0, sizeof(int), stream);
    cudaMemsetAsync(d_should_skip, 0, sizeof(int), stream);
    
    float min_vel_threshold_sq = min_velocity_threshold * min_velocity_threshold;
    
    // Step 1: Calculate per-node kinetic energy with block reduction
    calculate_kinetic_energy_kernel<<<num_blocks, block_size, shared_mem_size, stream>>>(
        d_vel_x, d_vel_y, d_vel_z, d_mass,
        d_partial_ke, d_active_count,
        num_nodes, min_vel_threshold_sq
    );
    
    // Step 2: Final reduction
    int reduction_blocks = min(num_blocks, 256);
    reduce_kinetic_energy_kernel<<<1, reduction_blocks, reduction_blocks * sizeof(float), stream>>>(
        d_partial_ke, d_total_ke, d_avg_ke, d_active_count,
        num_blocks, num_nodes
    );
    
    // Step 3: Check stability
    check_stability_kernel<<<1, 1, 0, stream>>>(
        d_avg_ke, d_active_count, d_should_skip,
        stability_threshold, num_nodes, iteration
    );
    
    // Copy result back to host
    int should_skip_host = 0;
    cudaMemcpyAsync(&should_skip_host, d_should_skip, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Cleanup
    cudaFree(d_partial_ke);
    cudaFree(d_total_ke);
    cudaFree(d_avg_ke);
    cudaFree(d_active_count);
    cudaFree(d_should_skip);
    
    return should_skip_host != 0;
}

} // extern "C"