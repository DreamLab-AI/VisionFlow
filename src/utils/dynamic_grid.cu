// Dynamic Grid Sizing for CUDA Kernels
// Automatically adjusts grid dimensions based on workload and GPU characteristics

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <stdio.h>
#include <math.h>

extern "C" {

// Structure to hold grid configuration
struct DynamicGridConfig {
    int block_size;
    int grid_size;
    int shared_memory_size;
    float occupancy_ratio;
    int max_blocks_per_sm;
};

// GPU device properties cache
struct GPUDeviceInfo {
    int max_threads_per_block;
    int max_blocks_per_multiprocessor;
    int multiprocessor_count;
    int shared_memory_per_block;
    int warp_size;
    int max_threads_per_multiprocessor;
    bool initialized;
};

static GPUDeviceInfo g_device_info = {0};

// Initialize GPU device information
__host__ int initialize_device_info() {
    if (g_device_info.initialized) {
        return 0; // Already initialized
    }

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    g_device_info.max_threads_per_block = prop.maxThreadsPerBlock;
    g_device_info.max_blocks_per_multiprocessor = prop.maxBlocksPerMultiProcessor;
    g_device_info.multiprocessor_count = prop.multiProcessorCount;
    g_device_info.shared_memory_per_block = prop.sharedMemPerBlock;
    g_device_info.warp_size = prop.warpSize;
    g_device_info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    g_device_info.initialized = true;

    printf("Initialized GPU device info: %s\n", prop.name);
    printf("  Max threads per block: %d\n", g_device_info.max_threads_per_block);
    printf("  Max blocks per SM: %d\n", g_device_info.max_blocks_per_multiprocessor);
    printf("  Multiprocessor count: %d\n", g_device_info.multiprocessor_count);
    printf("  Shared memory per block: %d bytes\n", g_device_info.shared_memory_per_block);

    return 0;
}

// Calculate optimal block size based on kernel characteristics
__host__ int calculate_optimal_block_size(
    const void* kernel_func,
    int shared_memory_per_block,
    int min_blocks_per_sm
) {
    if (!g_device_info.initialized) {
        if (initialize_device_info() != 0) {
            return 256; // Fallback block size
        }
    }

    int min_grid_size, block_size;
    cudaError_t err = cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        kernel_func,
        shared_memory_per_block,
        0 // No block size limit
    );

    if (err != cudaSuccess) {
        printf("cudaOccupancyMaxPotentialBlockSize failed: %s\n", cudaGetErrorString(err));
        return 256; // Fallback
    }

    // Ensure block size is multiple of warp size
    block_size = (block_size / g_device_info.warp_size) * g_device_info.warp_size;

    // Clamp to reasonable bounds
    block_size = max(block_size, g_device_info.warp_size);
    block_size = min(block_size, g_device_info.max_threads_per_block);

    // Adjust based on minimum blocks per SM requirement
    if (min_blocks_per_sm > 0) {
        int max_threads_for_min_blocks = g_device_info.max_threads_per_multiprocessor / min_blocks_per_sm;
        block_size = min(block_size, max_threads_for_min_blocks);
    }

    return block_size;
}

// Calculate grid size based on workload and block size
__host__ DynamicGridConfig calculate_grid_config(
    int num_elements,
    const void* kernel_func,
    int shared_memory_per_thread,
    int min_blocks_per_sm
) {
    DynamicGridConfig config = {0};

    if (!g_device_info.initialized) {
        if (initialize_device_info() != 0) {
            // Fallback configuration
            config.block_size = 256;
            config.grid_size = (num_elements + 255) / 256;
            config.shared_memory_size = 0;
            config.occupancy_ratio = 0.0f;
            config.max_blocks_per_sm = 1;
            return config;
        }
    }

    // Calculate shared memory requirements
    int shared_memory_per_block = 0;
    if (shared_memory_per_thread > 0) {
        // We'll determine this after block size is calculated
        shared_memory_per_block = shared_memory_per_thread * 256; // Initial estimate
    }

    // Calculate optimal block size
    config.block_size = calculate_optimal_block_size(
        kernel_func,
        shared_memory_per_block,
        min_blocks_per_sm
    );

    // Recalculate shared memory with actual block size
    if (shared_memory_per_thread > 0) {
        config.shared_memory_size = shared_memory_per_thread * config.block_size;

        // Ensure we don't exceed shared memory limits
        if (config.shared_memory_size > g_device_info.shared_memory_per_block) {
            // Reduce block size to fit in shared memory
            int max_threads_for_shared_mem = g_device_info.shared_memory_per_block / shared_memory_per_thread;
            config.block_size = min(config.block_size, max_threads_for_shared_mem);
            config.block_size = (config.block_size / g_device_info.warp_size) * g_device_info.warp_size;
            config.shared_memory_size = shared_memory_per_thread * config.block_size;
        }
    }

    // Calculate grid size
    config.grid_size = (num_elements + config.block_size - 1) / config.block_size;

    // Calculate theoretical occupancy
    int blocks_per_sm = g_device_info.max_threads_per_multiprocessor / config.block_size;
    blocks_per_sm = min(blocks_per_sm, g_device_info.max_blocks_per_multiprocessor);

    if (config.shared_memory_size > 0) {
        int blocks_limited_by_shared_mem = g_device_info.shared_memory_per_block / config.shared_memory_size;
        blocks_per_sm = min(blocks_per_sm, blocks_limited_by_shared_mem);
    }

    config.max_blocks_per_sm = blocks_per_sm;
    int active_threads_per_sm = blocks_per_sm * config.block_size;
    config.occupancy_ratio = (float)active_threads_per_sm / (float)g_device_info.max_threads_per_multiprocessor;

    // Limit grid size to avoid excessive blocks for small workloads
    int max_useful_blocks = g_device_info.multiprocessor_count * blocks_per_sm * 2; // 2x for wave scheduling
    config.grid_size = min(config.grid_size, max_useful_blocks);

    return config;
}

// Adaptive grid configuration based on performance feedback
struct PerformanceHistory {
    float execution_times[16]; // Circular buffer of recent execution times
    DynamicGridConfig configs[16]; // Corresponding configurations
    int current_index;
    int sample_count;
    float best_time;
    DynamicGridConfig best_config;
    bool initialized;
};

static PerformanceHistory g_perf_history = {0};

// Update performance history with new timing data
__host__ void update_performance_history(DynamicGridConfig config, float execution_time_ms) {
    if (!g_perf_history.initialized) {
        g_perf_history.best_time = execution_time_ms;
        g_perf_history.best_config = config;
        g_perf_history.initialized = true;
    }

    // Store in circular buffer
    int idx = g_perf_history.current_index;
    g_perf_history.execution_times[idx] = execution_time_ms;
    g_perf_history.configs[idx] = config;

    g_perf_history.current_index = (idx + 1) % 16;
    g_perf_history.sample_count = min(g_perf_history.sample_count + 1, 16);

    // Update best configuration if this one is better
    if (execution_time_ms < g_perf_history.best_time) {
        g_perf_history.best_time = execution_time_ms;
        g_perf_history.best_config = config;
    }
}

// Get adaptive configuration based on performance history
__host__ DynamicGridConfig get_adaptive_grid_config(
    int num_elements,
    const void* kernel_func,
    int shared_memory_per_thread,
    int min_blocks_per_sm
) {
    // Start with calculated optimal configuration
    DynamicGridConfig base_config = calculate_grid_config(
        num_elements, kernel_func, shared_memory_per_thread, min_blocks_per_sm
    );

    // If we have performance history, consider using the best known configuration
    if (g_perf_history.initialized && g_perf_history.sample_count >= 3) {
        // Use best known configuration if it's significantly better
        return g_perf_history.best_config;
    }

    return base_config;
}

// Specialized configurations for different kernel types
__host__ DynamicGridConfig get_force_kernel_config(int num_nodes) {
    // Force kernels are memory-bound and benefit from higher occupancy
    return calculate_grid_config(
        num_nodes,
        nullptr, // No specific kernel function analysis
        64,      // Moderate shared memory usage for neighbor lists
        2        // Prefer at least 2 blocks per SM for latency hiding
    );
}

__host__ DynamicGridConfig get_reduction_kernel_config(int num_elements) {
    // Reduction kernels benefit from power-of-2 block sizes and higher shared memory
    DynamicGridConfig config = calculate_grid_config(
        num_elements,
        nullptr,
        sizeof(float) * 2, // Shared memory for reduction tree
        4  // Higher parallelism for reduction
    );

    // Ensure power-of-2 block size for efficient reduction
    int power_of_2 = 1;
    while (power_of_2 < config.block_size && power_of_2 < 512) {
        power_of_2 *= 2;
    }
    if (power_of_2 <= 512) {
        config.block_size = power_of_2;
        config.grid_size = (num_elements + config.block_size - 1) / config.block_size;
        config.shared_memory_size = sizeof(float) * config.block_size;
    }

    return config;
}

__host__ DynamicGridConfig get_sorting_kernel_config(int num_elements) {
    // Sorting kernels need balanced compute and memory access
    return calculate_grid_config(
        num_elements,
        nullptr,
        sizeof(int) * 2, // Keys and values
        3  // Moderate parallelism
    );
}

// Print configuration for debugging
__host__ void print_grid_config(const char* kernel_name, DynamicGridConfig config) {
    printf("Grid config for %s:\n", kernel_name);
    printf("  Block size: %d\n", config.block_size);
    printf("  Grid size: %d\n", config.grid_size);
    printf("  Shared memory: %d bytes\n", config.shared_memory_size);
    printf("  Theoretical occupancy: %.2f%%\n", config.occupancy_ratio * 100.0f);
    printf("  Max blocks per SM: %d\n", config.max_blocks_per_sm);
}

// Benchmark a kernel configuration
__host__ float benchmark_kernel_config(
    DynamicGridConfig config,
    void (*kernel_launcher)(DynamicGridConfig, cudaStream_t),
    cudaStream_t stream,
    int num_iterations
) {
    // Warm up
    for (int i = 0; i < 3; i++) {
        kernel_launcher(config, stream);
    }
    cudaStreamSynchronize(stream);

    // Time the kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < num_iterations; i++) {
        kernel_launcher(config, stream);
    }
    cudaEventRecord(stop, stream);
    cudaStreamSynchronize(stream);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds / num_iterations;
}

} // extern "C"