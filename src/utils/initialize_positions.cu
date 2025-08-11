// Position initialization kernel for preventing collapse at origin
#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {

__global__ void initialize_positions_kernel(
    float* pos_x,
    float* pos_y, 
    float* pos_z,
    int num_nodes,
    float spread_radius,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    // Initialize random number generator
    curandState state;
    curand_init(seed + idx * 1337, 0, 0, &state);
    
    // Use golden angle spiral for better initial distribution
    float golden_angle = 3.14159265f * (3.0f - sqrtf(5.0f));
    float theta = idx * golden_angle;
    
    // Vertical position based on node index
    float y = 1.0f - (idx / (float)num_nodes) * 2.0f;
    float radius = sqrtf(1.0f - y * y);
    
    // Add some randomness to prevent perfect patterns
    float rand_scale = 0.8f + curand_uniform(&state) * 0.4f;
    
    // Scale and position
    pos_x[idx] = cosf(theta) * radius * spread_radius * rand_scale;
    pos_y[idx] = y * spread_radius * rand_scale;
    pos_z[idx] = sinf(theta) * radius * spread_radius * rand_scale;
    
    // Add small random perturbation to avoid exact overlaps
    pos_x[idx] += (curand_uniform(&state) - 0.5f) * 0.1f;
    pos_y[idx] += (curand_uniform(&state) - 0.5f) * 0.1f;
    pos_z[idx] += (curand_uniform(&state) - 0.5f) * 0.1f;
}

}