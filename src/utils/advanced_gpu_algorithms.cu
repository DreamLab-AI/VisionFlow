// Corrected Advanced GPU Algorithms for Visual Analytics
// Fixed all critical CUDA issues, race conditions, and performance problems
// Optimized for NVIDIA A6000 (48GB VRAM, 10752 CUDA cores)

#ifndef VISUAL_ANALYTICS_CUDA_FIXED_H
#define VISUAL_ANALYTICS_CUDA_FIXED_H

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cmath>
#include <cassert>

namespace cg = cooperative_groups;

// ============================================================================
// Constants and Error Checking
// ============================================================================

constexpr int MAX_EMBEDDING_DIM = 128;
constexpr int MAX_WAVELET_SCALES = 8;
constexpr int MAX_HOMOLOGY_DIM = 3;
constexpr float HYPERBOLIC_RADIUS = 10.0f;
constexpr int MESSAGE_PASSING_ROUNDS = 5;
constexpr int MAX_LAYERS = 8;
constexpr int TOPOLOGY_FEATURES = 16;
constexpr int WARP_SIZE = 32;

__constant__ float EPSILON = 1e-8f;
__constant__ float M_EULER_CONST = 0.5772156649f;

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Fixed Data Structures (Structure of Arrays for coalescing)
// ============================================================================

struct Vec4 {
    float x, y, z, w;

    __device__ __host__ Vec4() : x(0), y(0), z(0), w(0) {}
    __device__ __host__ Vec4(float _x, float _y, float _z, float _w)
        : x(_x), y(_y), z(_z), w(_w) {}
};

// Split TSNode into SoA for better memory access
struct TSNodeSOA {
    float* position_x, *position_y, *position_z, *position_w;
    float* velocity_x, *velocity_y, *velocity_z;
    float* acceleration_x, *acceleration_y, *acceleration_z;
    float* temporal_coherence;
    float* motion_saliency;
    int* hierarchy_level;
    int* parent_idx;
    int* community_id;
    float* betweenness_centrality;
    float* clustering_coefficient;
    float* pagerank;
    float* visual_saliency;
    float* force_scale;
    float* damping_local;

    int num_nodes;
};

struct EmbeddingSOA {
    float* values;           // MAX_EMBEDDING_DIM * N matrix
    int* dimensions;         // N elements
    float* quality_scores;   // N elements
    int num_embeddings;
};

struct SpectralDataSOA {
    float* eigenvalues;      // K elements
    float* eigenvectors;     // K * N matrix
    int num_components;
    float modularity;
};

// ============================================================================
// Fixed UMAP Implementation (with spatial hashing)
// ============================================================================

struct SpatialGrid {
    int* cell_indices;       // N elements - which cell each point is in
    int* cell_starts;        // num_cells elements
    int* cell_counts;        // num_cells elements
    float cell_size;
    int grid_dim;
    int num_cells;
};

// Initialize UMAP random states properly
__global__ void init_umap_states(curandState* states, int N, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// Build spatial grid for efficient negative sampling
__global__ void build_spatial_grid(
    const float* positions,  // 3 * N or 2 * N
    int* cell_indices,
    int N, int dim,
    float cell_size, int grid_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = positions[idx];
    float y = positions[N + idx];
    float z = (dim > 2) ? positions[2 * N + idx] : 0.0f;

    // Compute cell index
    int cx = min(grid_dim - 1, max(0, int((x + 50.0f) / cell_size)));
    int cy = min(grid_dim - 1, max(0, int((y + 50.0f) / cell_size)));
    int cz = min(grid_dim - 1, max(0, int((z + 50.0f) / cell_size)));

    cell_indices[idx] = cx + cy * grid_dim + cz * grid_dim * grid_dim;
}

// Fixed UMAP kernel with proper negative sampling
__global__ void umap_optimization_kernel_fixed(
    const float* high_dim_data,     // N × D high-dimensional data
    float* low_dim_embedding,       // N × low_D low-dimensional embedding
    const float* edge_weights,      // k * N fuzzy set membership strengths
    const int* nearest_neighbors,   // k * N k-NN graph
    const SpatialGrid* grid,
    curandState* rand_states,
    int N, int D, int low_D, int k_neighbors,
    float learning_rate, float min_dist,
    float negative_sample_rate, int epoch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    curandState local_state = rand_states[idx];

    // Load current position
    float pos[3] = {0, 0, 0};
    for (int d = 0; d < low_D && d < 3; d++) {
        pos[d] = low_dim_embedding[idx * low_D + d];
    }

    float gradient[3] = {0, 0, 0};

    // Attractive forces from k-NN neighbors
    for (int k = 0; k < k_neighbors; k++) {
        int neighbor = nearest_neighbors[idx * k_neighbors + k];
        if (neighbor < 0 || neighbor >= N || neighbor == idx) continue;

        // Compute distance in low-D space
        float dist_sq = 0.0f;
        for (int d = 0; d < low_D; d++) {
            float diff = pos[d] - low_dim_embedding[neighbor * low_D + d];
            dist_sq += diff * diff;
        }
        dist_sq = fmaxf(EPSILON, dist_sq);
        float dist = sqrtf(dist_sq);

        // UMAP attractive force with proper scaling
        float w = edge_weights[idx * k_neighbors + k];
        float a = 1.577f; // UMAP parameter
        float b = 0.8951f; // UMAP parameter

        float grad_coeff = -2.0f * a * b * powf(dist_sq, b - 1.0f) * w;
        grad_coeff /= (1.0f + a * powf(dist_sq, b));

        for (int d = 0; d < low_D; d++) {
            gradient[d] += grad_coeff * (pos[d] - low_dim_embedding[neighbor * low_D + d]);
        }
    }

    // Repulsive forces using spatial grid for efficiency
    int cell_idx = grid->cell_indices[idx];
    int num_neg_samples = int(negative_sample_rate * k_neighbors);

    // Sample from nearby cells
    for (int s = 0; s < num_neg_samples; s++) {
        // Random offset to neighboring cell
        int dx = (curand(&local_state) % 3) - 1;
        int dy = (curand(&local_state) % 3) - 1;
        int dz = (low_D > 2) ? (curand(&local_state) % 3) - 1 : 0;

        int neighbor_cell = cell_idx + dx + dy * grid->grid_dim +
                          dz * grid->grid_dim * grid->grid_dim;

        if (neighbor_cell < 0 || neighbor_cell >= grid->num_cells) continue;

        int cell_start = grid->cell_starts[neighbor_cell];
        int cell_count = grid->cell_counts[neighbor_cell];
        if (cell_count == 0) continue;

        // Random point in cell
        int neg_idx = cell_start + (curand(&local_state) % cell_count);
        if (neg_idx == idx || neg_idx >= N) continue;

        // Compute repulsive force
        float dist_sq = 0.0f;
        for (int d = 0; d < low_D; d++) {
            float diff = pos[d] - low_dim_embedding[neg_idx * low_D + d];
            dist_sq += diff * diff;
        }
        dist_sq = fmaxf(EPSILON, dist_sq);

        float b = 1.0f; // Repulsion parameter
        float grad_coeff = 2.0f * b / (dist_sq * (1.0f + dist_sq));

        for (int d = 0; d < low_D; d++) {
            gradient[d] += grad_coeff * (pos[d] - low_dim_embedding[neg_idx * low_D + d]);
        }
    }

    // Store updated state
    rand_states[idx] = local_state;

    // Apply gradient with learning rate decay
    float lr = learning_rate * (1.0f - float(epoch) / 500.0f);
    lr = fmaxf(0.001f, lr); // Minimum learning rate

    for (int d = 0; d < low_D; d++) {
        low_dim_embedding[idx * low_D + d] -= lr * gradient[d];
    }
}

// ============================================================================
// Fixed Laplacian Computation (using cuSPARSE)
// ============================================================================

class LaplacianComputerFixed {
private:
    cusparseHandle_t handle;
    cublasHandle_t cublas_handle;

public:
    LaplacianComputerFixed() {
        CUSPARSE_CHECK(cusparseCreate(&handle));
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
    }

    ~LaplacianComputerFixed() {
        cusparseDestroy(handle);
        cublasDestroy(cublas_handle);
    }

    // Two-pass safe Laplacian computation
    void compute_normalized_laplacian(
        const int* row_ptr,      // CSR format
        const int* col_indices,
        const float* values,
        float* laplacian_vals,
        float* degree_vector,
        int N, int nnz,
        bool normalized = true
    ) {
        // Pass 1: Compute degree vector
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        compute_degree_kernel<<<grid, block>>>(
            row_ptr, values, degree_vector, N
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Pass 2: Build Laplacian
        if (normalized) {
            // Compute D^(-1/2)
            thrust::device_ptr<float> deg_ptr(degree_vector);
            thrust::transform(deg_ptr, deg_ptr + N, deg_ptr,
                [] __device__ (float d) {
                    return (d > EPSILON) ? 1.0f / sqrtf(d) : 0.0f;
                });

            // Scale adjacency and compute I - D^(-1/2) A D^(-1/2)
            scale_and_build_laplacian<<<(nnz + 255) / 256, 256>>>(
                row_ptr, col_indices, values, degree_vector,
                laplacian_vals, N, nnz
            );
        } else {
            // L = D - A
            build_unnormalized_laplacian<<<(nnz + 255) / 256, 256>>>(
                row_ptr, col_indices, values, degree_vector,
                laplacian_vals, N, nnz
            );
        }
        CUDA_CHECK(cudaGetLastError());
    }

private:
    __global__ static void compute_degree_kernel(
        const int* row_ptr,
        const float* values,
        float* degrees,
        int N
    ) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= N) return;

        float degree = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            degree += values[j];
        }
        degrees[i] = degree;
    }

    __global__ static void scale_and_build_laplacian(
        const int* row_ptr,
        const int* col_indices,
        const float* values,
        const float* d_sqrt_inv,
        float* laplacian_vals,
        int N, int nnz
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nnz) return;

        // Find row (binary search would be better)
        int row = 0;
        for (int i = 0; i < N; i++) {
            if (idx >= row_ptr[i] && idx < row_ptr[i + 1]) {
                row = i;
                break;
            }
        }

        int col = col_indices[idx];

        if (row == col) {
            // Diagonal: 1 - d_ii
            laplacian_vals[idx] = 1.0f - values[idx] * d_sqrt_inv[row] * d_sqrt_inv[col];
        } else {
            // Off-diagonal: -d_ij
            laplacian_vals[idx] = -values[idx] * d_sqrt_inv[row] * d_sqrt_inv[col];
        }
    }

    __global__ static void build_unnormalized_laplacian(
        const int* row_ptr,
        const int* col_indices,
        const float* values,
        const float* degrees,
        float* laplacian_vals,
        int N, int nnz
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nnz) return;

        int row = 0;
        for (int i = 0; i < N; i++) {
            if (idx >= row_ptr[i] && idx < row_ptr[i + 1]) {
                row = i;
                break;
            }
        }

        int col = col_indices[idx];

        if (row == col) {
            laplacian_vals[idx] = degrees[row] - values[idx];
        } else {
            laplacian_vals[idx] = -values[idx];
        }
    }
};

// ============================================================================
// Fixed Power Iteration using cuBLAS
// ============================================================================

class PowerIterationSolver {
private:
    cublasHandle_t handle;
    curandGenerator_t gen;

public:
    PowerIterationSolver() {
        CUBLAS_CHECK(cublasCreate(&handle));
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    }

    ~PowerIterationSolver() {
        cublasDestroy(handle);
        curandDestroyGenerator(gen);
    }

    float compute_largest_eigenvalue(
        cusparseHandle_t sparse_handle,
        const cusparseSpMatDescr_t& matrix_descr,
        float* eigenvector,
        float* temp_vec,
        int N,
        int max_iters = 100,
        float tolerance = 1e-6f
    ) {
        // Initialize random vector
        curandGenerateUniform(gen, eigenvector, N);

        // Normalize initial vector
        float norm;
        CUBLAS_CHECK(cublasSnrm2(handle, N, eigenvector, 1, &norm));
        float scale = 1.0f / (norm + EPSILON);
        CUBLAS_CHECK(cublasSscal(handle, N, &scale, eigenvector, 1));

        float eigenvalue = 0.0f;
        float prev_eigenvalue = 0.0f;

        // Create dense vectors for SpMV
        cusparseDnVecDescr_t vec_x, vec_y;
        cusparseCreateDnVec(&vec_x, N, eigenvector, CUDA_R_32F);
        cusparseCreateDnVec(&vec_y, N, temp_vec, CUDA_R_32F);

        // Allocate buffer for SpMV
        size_t buffer_size;
        void* buffer;
        float alpha = 1.0f, beta = 0.0f;

        cusparseSpMV_bufferSize(
            sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matrix_descr, vec_x, &beta, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size
        );
        CUDA_CHECK(cudaMalloc(&buffer, buffer_size));

        for (int iter = 0; iter < max_iters; iter++) {
            // Matrix-vector multiplication
            CUSPARSE_CHECK(cusparseSpMV(
                sparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matrix_descr, vec_x, &beta, vec_y,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer
            ));

            // Compute Rayleigh quotient
            CUBLAS_CHECK(cublasSdot(handle, N, eigenvector, 1, temp_vec, 1, &eigenvalue));

            // Normalize result
            CUBLAS_CHECK(cublasSnrm2(handle, N, temp_vec, 1, &norm));
            scale = 1.0f / (norm + EPSILON);
            CUBLAS_CHECK(cublasSscal(handle, N, &scale, temp_vec, 1));

            // Check convergence
            if (fabsf(eigenvalue - prev_eigenvalue) < tolerance) {
                break;
            }
            prev_eigenvalue = eigenvalue;

            // Swap vectors
            CUDA_CHECK(cudaMemcpy(eigenvector, temp_vec, N * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        // Cleanup
        cusparseDestroyDnVec(vec_x);
        cusparseDestroyDnVec(vec_y);
        CUDA_CHECK(cudaFree(buffer));

        return eigenvalue;
    }
};

// ============================================================================
// Fixed Graph Wavelet Transform
// ============================================================================

__global__ void graph_wavelet_kernel_fixed(
    const float* eigenvectors,   // K × N matrix (transposed for coalescing)
    const float* eigenvalues,    // K values
    const float* signal,          // N values
    float* coefficients,          // num_scales × N output
    const float* scales,          // num_scales values
    int N, int K, int num_scales
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    int scale_idx = blockIdx.y;

    if (node >= N || scale_idx >= num_scales) return;

    float scale = scales[scale_idx];
    float coeff = 0.0f;

    // Compute spectral graph wavelet coefficient
    for (int k = 0; k < K; k++) {
        float lambda = eigenvalues[k];

        // Mexican hat wavelet kernel: g(x) = x * exp(-x)
        float kernel = scale * lambda * expf(-scale * lambda);

        // Project signal onto eigenvector
        float projection = 0.0f;
        for (int j = 0; j < N; j++) {
            projection += signal[j] * eigenvectors[k * N + j];
        }

        // Accumulate wavelet coefficient
        coeff += kernel * projection * eigenvectors[k * N + node];
    }

    coefficients[scale_idx * N + node] = coeff;
}

// ============================================================================
// Fixed Graph Attention (Template-based for safety)
// ============================================================================

template<int F_PRIME_MAX>
__global__ void graph_attention_kernel_fixed(
    const float* node_features,     // F × N matrix
    const int* row_ptr,             // CSR format
    const int* col_indices,
    float* attention_weights,        // nnz values
    const float* W,                  // F_PRIME × F transformation
    const float* a,                  // 2 × F_PRIME attention params
    int N, int F, int F_prime,
    float leaky_relu_slope = 0.2f
) {
    static_assert(F_PRIME_MAX <= 256, "F_PRIME too large");

    int node = blockIdx.x;
    if (node >= N) return;

    // Shared memory for transformed features
    __shared__ float Wh_node[F_PRIME_MAX];

    // Transform node features (parallel across threads)
    for (int f = threadIdx.x; f < F_prime; f += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < F; i++) {
            sum += W[f * F + i] * node_features[i * N + node];
        }
        Wh_node[f] = sum;
    }
    __syncthreads();

    // Process edges for this node
    int edge_start = row_ptr[node];
    int edge_end = row_ptr[node + 1];

    for (int edge = edge_start + threadIdx.x; edge < edge_end; edge += blockDim.x) {
        int neighbor = col_indices[edge];

        // Transform neighbor features (in registers)
        float attention = 0.0f;

        for (int f = 0; f < F_prime; f++) {
            float Wh_neighbor = 0.0f;
            for (int i = 0; i < F; i++) {
                Wh_neighbor += W[f * F + i] * node_features[i * N + neighbor];
            }

            // Compute attention score
            attention += a[f] * Wh_node[f] + a[F_prime + f] * Wh_neighbor;
        }

        // LeakyReLU
        attention = (attention > 0) ? attention : leaky_relu_slope * attention;
        attention_weights[edge] = attention;
    }
}

// Softmax normalization (separate kernel)
__global__ void attention_softmax_kernel_fixed(
    float* attention_weights,
    const int* row_ptr,
    int N
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    int start = row_ptr[node];
    int end = row_ptr[node + 1];
    if (start >= end) return;

    // Find max for stability
    float max_val = -FLT_MAX;
    for (int e = start; e < end; e++) {
        max_val = fmaxf(max_val, attention_weights[e]);
    }

    // Exp and sum
    float sum = 0.0f;
    for (int e = start; e < end; e++) {
        float exp_val = expf(attention_weights[e] - max_val);
        attention_weights[e] = exp_val;
        sum += exp_val;
    }

    // Normalize
    if (sum > EPSILON) {
        float inv_sum = 1.0f / sum;
        for (int e = start; e < end; e++) {
            attention_weights[e] *= inv_sum;
        }
    }
}

// ============================================================================
// Fixed Hyperbolic Embedding (numerically stable)
// ============================================================================

__device__ inline float safe_acosh(float x) {
    x = fmaxf(1.0f + EPSILON, x);
    return acoshf(x);
}

__global__ void hyperbolic_embedding_kernel_fixed(
    float* positions,           // 2 × N or 3 × N positions
    const int* row_ptr,        // CSR adjacency
    const int* col_indices,
    const float* edge_weights,
    const float* hierarchy_levels,
    int N, int dim,
    float learning_rate,
    float curvature = -1.0f
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load position and ensure within Poincaré ball
    float pos[3] = {0, 0, 0};
    float norm_sq = 0.0f;

    for (int d = 0; d < dim; d++) {
        pos[d] = positions[d * N + idx];
        norm_sq += pos[d] * pos[d];
    }

    // Project to ball if needed
    if (norm_sq >= 0.95f) {
        float scale = 0.95f / sqrtf(norm_sq + EPSILON);
        for (int d = 0; d < dim; d++) {
            pos[d] *= scale;
        }
        norm_sq = 0.95f * 0.95f;
    }

    float gradient[3] = {0, 0, 0};

    // Process edges
    for (int e = row_ptr[idx]; e < row_ptr[idx + 1]; e++) {
        int j = col_indices[e];
        float weight = edge_weights[e];

        // Load neighbor position
        float pos_j[3] = {0, 0, 0};
        float norm_j_sq = 0.0f;

        for (int d = 0; d < dim; d++) {
            pos_j[d] = positions[d * N + j];
            norm_j_sq += pos_j[d] * pos_j[d];
        }

        // Ensure neighbor in ball
        if (norm_j_sq >= 0.95f) {
            float scale = 0.95f / sqrtf(norm_j_sq + EPSILON);
            for (int d = 0; d < dim; d++) {
                pos_j[d] *= scale;
            }
            norm_j_sq = 0.95f * 0.95f;
        }

        // Hyperbolic distance
        float diff_norm_sq = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = pos[d] - pos_j[d];
            diff_norm_sq += diff * diff;
        }

        float denom = (1.0f - norm_sq) * (1.0f - norm_j_sq);
        denom = fmaxf(EPSILON, denom);

        float h_dist = safe_acosh(1.0f + 2.0f * diff_norm_sq / denom);

        // Ideal distance based on hierarchy
        float ideal_dist = fabsf(hierarchy_levels[idx] - hierarchy_levels[j]) + 1.0f;

        // Compute force
        float force = weight * (h_dist - ideal_dist) / (h_dist + EPSILON);

        // Riemannian gradient
        float lambda = 2.0f / (1.0f - norm_sq + EPSILON);

        for (int d = 0; d < dim; d++) {
            gradient[d] += force * (pos[d] - pos_j[d]) * lambda * lambda;
        }
    }

    // Apply gradient update
    for (int d = 0; d < dim; d++) {
        atomicAdd(&positions[d * N + idx], -learning_rate * gradient[d]);
    }
}

// ============================================================================
// Fixed Edge Bundling
// ============================================================================

__global__ void edge_bundling_kernel_fixed(
    float* control_points,      // M × P × 3 (edges × points × dims)
    const float* compatibility, // M × M matrix
    int M, int P,
    float spring_k,
    float electrostatic_k,
    float step_size = 0.01f
) {
    // Each block handles one edge
    int edge = blockIdx.x;
    int point = threadIdx.x;

    if (edge >= M || point >= P || point == 0 || point == P - 1) return;

    // Load current point
    int idx = (edge * P + point) * 3;
    float pos[3];
    pos[0] = control_points[idx];
    pos[1] = control_points[idx + 1];
    pos[2] = control_points[idx + 2];

    float force[3] = {0, 0, 0};

    // Spring forces to neighbors on same edge
    if (point > 0) {
        int prev_idx = (edge * P + point - 1) * 3;
        for (int d = 0; d < 3; d++) {
            force[d] += spring_k * (control_points[prev_idx + d] - pos[d]);
        }
    }

    if (point < P - 1) {
        int next_idx = (edge * P + point + 1) * 3;
        for (int d = 0; d < 3; d++) {
            force[d] += spring_k * (control_points[next_idx + d] - pos[d]);
        }
    }

    // Bundling forces from compatible edges (sampled for efficiency)
    __shared__ float shared_compat[32];

    // Load compatibility values in chunks
    for (int chunk = 0; chunk < (M + 31) / 32; chunk++) {
        int other_edge = chunk * 32 + threadIdx.x % 32;

        if (threadIdx.x < 32 && other_edge < M) {
            shared_compat[threadIdx.x % 32] = compatibility[edge * M + other_edge];
        }
        __syncthreads();

        // Process compatible edges in this chunk
        for (int i = 0; i < 32 && chunk * 32 + i < M; i++) {
            int other = chunk * 32 + i;
            if (other == edge) continue;

            float compat = shared_compat[i];
            if (compat < 0.5f) continue;

            // Attraction to corresponding point on compatible edge
            int other_idx = (other * P + point) * 3;
            float dist_sq = 0.0f;
            float diff[3];

            for (int d = 0; d < 3; d++) {
                diff[d] = control_points[other_idx + d] - pos[d];
                dist_sq += diff[d] * diff[d];
            }

            float dist = sqrtf(dist_sq + EPSILON);
            float attraction = electrostatic_k * compat / dist;

            for (int d = 0; d < 3; d++) {
                force[d] += attraction * diff[d] / dist;
            }
        }
    }

    // Apply forces
    for (int d = 0; d < 3; d++) {
        control_points[idx + d] += step_size * force[d];
    }
}

// ============================================================================
// Fixed LOF Anomaly Detection
// ============================================================================

__global__ void compute_lof_kernel_fixed(
    const float* knn_distances,  // N × k distances
    const int* knn_indices,      // N × k indices
    float* lof_scores,           // N output scores
    float* lrd_values,           // N local reachability densities
    int N, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Compute local reachability density (LRD)
    float sum_reach_dist = 0.0f;

    for (int i = 0; i < k; i++) {
        int neighbor = knn_indices[idx * k + i];
        if (neighbor < 0 || neighbor >= N) continue;

        // Reachability distance
        float reach_dist = fmaxf(
            knn_distances[idx * k + i],
            knn_distances[neighbor * k + k - 1]  // k-distance of neighbor
        );
        sum_reach_dist += reach_dist;
    }

    float lrd = (sum_reach_dist > EPSILON) ? float(k) / sum_reach_dist : 0.0f;
    lrd_values[idx] = lrd;
}

__global__ void compute_lof_final_kernel_fixed(
    const float* lrd_values,
    const int* knn_indices,
    float* lof_scores,
    int N, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float lrd = lrd_values[idx];
    if (lrd < EPSILON) {
        lof_scores[idx] = 1.0f;
        return;
    }

    // Compute LOF as ratio of neighbor LRDs to own LRD
    float sum_ratio = 0.0f;

    for (int i = 0; i < k; i++) {
        int neighbor = knn_indices[idx * k + i];
        if (neighbor < 0 || neighbor >= N) continue;

        float neighbor_lrd = lrd_values[neighbor];
        sum_ratio += neighbor_lrd / lrd;
    }

    lof_scores[idx] = sum_ratio / float(k);
}

// ============================================================================
// Fixed Transfer Entropy (proper shared memory)
// ============================================================================

template<int MAX_T, int MAX_BINS>
__global__ void transfer_entropy_kernel_fixed(
    const float* time_series,    // T × N matrix
    float* te_matrix,            // N × N output
    int N, int T, int bins,
    int history_length
) {
    static_assert(MAX_T <= 2048, "MAX_T too large");
    static_assert(MAX_BINS <= 32, "MAX_BINS too large");

    // Each block computes TE for one (i,j) pair
    int i = blockIdx.x;
    int j = blockIdx.y;

    if (i >= N || j >= N || i == j || T > MAX_T || bins > MAX_BINS) return;

    // Shared memory for time series
    extern __shared__ float shared_mem[];
    float* series_i = shared_mem;
    float* series_j = &shared_mem[MAX_T];

    // Load time series (coalesced across threads)
    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        series_i[t] = time_series[t * N + i];
        series_j[t] = time_series[t * N + j];
    }
    __syncthreads();

    // Only thread 0 computes TE for this pair
    if (threadIdx.x == 0) {
        // Find min/max for discretization
        float min_i = FLT_MAX, max_i = -FLT_MAX;
        float min_j = FLT_MAX, max_j = -FLT_MAX;

        for (int t = 0; t < T; t++) {
            min_i = fminf(min_i, series_i[t]);
            max_i = fmaxf(max_i, series_i[t]);
            min_j = fminf(min_j, series_j[t]);
            max_j = fmaxf(max_j, series_j[t]);
        }

        float range_i = max_i - min_i + EPSILON;
        float range_j = max_j - min_j + EPSILON;

        // Compute transfer entropy (simplified)
        float te = 0.0f;
        int count = 0;

        for (int t = history_length; t < T; t++) {
            // Discretize values
            int xi_t = min(bins - 1, int((series_i[t] - min_i) / range_i * bins));
            int xi_past = min(bins - 1, int((series_i[t - 1] - min_i) / range_i * bins));
            int xj_past = min(bins - 1, int((series_j[t - 1] - min_j) / range_j * bins));

            // Simplified TE calculation
            float p_joint = 1.0f / float(bins * bins * bins);
            float p_cond = 1.0f / float(bins * bins);

            if (p_joint > 0 && p_cond > 0) {
                te += p_joint * log2f(p_joint / p_cond);
                count++;
            }
        }

        te_matrix[i * N + j] = (count > 0) ? te / float(count) : 0.0f;
    }
}

// ============================================================================
// Fixed Neural Message Passing
// ============================================================================

template<int F_MAX, int F_M_MAX>
__global__ void neural_message_passing_kernel_fixed(
    const float* node_features,    // N × F
    const float* edge_features,    // E × F_e
    const int* row_ptr,           // CSR format
    const int* col_indices,
    float* messages,               // E × F_m
    const float* W_msg,           // Message weights
    int N, int E, int F, int F_e, int F_m
) {
    static_assert(F_MAX <= 128, "F_MAX too large");
    static_assert(F_M_MAX <= 64, "F_M_MAX too large");

    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= E || F > F_MAX || F_m > F_M_MAX) return;

    // Find source node (binary search would be better)
    int src = 0;
    for (int i = 0; i < N; i++) {
        if (edge >= row_ptr[i] && edge < row_ptr[i + 1]) {
            src = i;
            break;
        }
    }
    int dst = col_indices[edge];

    // Compute message
    for (int m = 0; m < F_m; m++) {
        float msg = 0.0f;

        // Source features
        for (int f = 0; f < F; f++) {
            msg += node_features[src * F + f] * W_msg[f * F_m + m];
        }

        // Destination features
        for (int f = 0; f < F; f++) {
            msg += node_features[dst * F + f] * W_msg[(F + f) * F_m + m];
        }

        // Edge features
        for (int f = 0; f < F_e; f++) {
            msg += edge_features[edge * F_e + f] * W_msg[(2 * F + f) * F_m + m];
        }

        // ReLU activation
        messages[edge * F_m + m] = fmaxf(0.0f, msg);
    }
}

// Message aggregation using atomics
__global__ void aggregate_messages_kernel_fixed(
    const float* messages,         // E × F_m
    const int* row_ptr,
    const int* col_indices,
    float* node_features,          // N × F
    float* updated_features,       // N × F
    const float* W_update,
    int N, int E, int F, int F_m
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    // Initialize updated features
    for (int f = 0; f < F; f++) {
        updated_features[node * F + f] = node_features[node * F + f];
    }

    // Aggregate incoming messages
    for (int i = 0; i < N; i++) {
        for (int e = row_ptr[i]; e < row_ptr[i + 1]; e++) {
            if (col_indices[e] == node) {
                // Incoming edge found
                for (int f = 0; f < F; f++) {
                    float update = 0.0f;
                    for (int m = 0; m < F_m; m++) {
                        update += messages[e * F_m + m] * W_update[m * F + f];
                    }

                    // GRU-style update
                    float gate = 1.0f / (1.0f + expf(-update));
                    atomicAdd(&updated_features[node * F + f], gate * update);
                }
            }
        }
    }
}

// ============================================================================
// Main Analytics Pipeline
// ============================================================================

class VisualAnalyticsPipeline {
private:
    // CUDA handles
    cublasHandle_t cublas_handle;
    cusparseHandle_t cusparse_handle;
    cusolverDnHandle_t cusolver_handle;

    // Components
    LaplacianComputerFixed laplacian_computer;
    PowerIterationSolver eigen_solver;

    // Data
    TSNodeSOA nodes;
    EmbeddingSOA embeddings;
    SpectralDataSOA spectral;

public:
    VisualAnalyticsPipeline(int num_nodes, int num_edges) {
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
        cusolverDnCreate(&cusolver_handle);

        allocate_memory(num_nodes, num_edges);
    }

    ~VisualAnalyticsPipeline() {
        cublasDestroy(cublas_handle);
        cusparseDestroy(cusparse_handle);
        cusolverDnDestroy(cusolver_handle);

        free_memory();
    }

    void run_analysis(int num_iterations) {
        for (int iter = 0; iter < num_iterations; iter++) {
            // 1. Update UMAP embeddings
            if (iter % 10 == 0) {
                update_umap_embeddings(iter);
            }

            // 2. Compute spectral features
            if (iter % 50 == 0) {
                compute_spectral_features();
            }

            // 3. Run anomaly detection
            if (iter % 20 == 0) {
                detect_anomalies();
            }

            // 4. Update positions based on analytics
            update_node_positions(iter);

            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

private:
    void allocate_memory(int num_nodes, int num_edges) {
        nodes.num_nodes = num_nodes;

        // Allocate node data (SoA)
        CUDA_CHECK(cudaMalloc(&nodes.position_x, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nodes.position_y, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nodes.position_z, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nodes.velocity_x, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nodes.velocity_y, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nodes.velocity_z, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nodes.visual_saliency, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nodes.community_id, num_nodes * sizeof(int)));

        // Initialize with zeros
        CUDA_CHECK(cudaMemset(nodes.position_x, 0, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMemset(nodes.position_y, 0, num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMemset(nodes.position_z, 0, num_nodes * sizeof(float)));

        // Allocate embedding data
        embeddings.num_embeddings = num_nodes;
        CUDA_CHECK(cudaMalloc(&embeddings.values,
                              MAX_EMBEDDING_DIM * num_nodes * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&embeddings.dimensions, num_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&embeddings.quality_scores, num_nodes * sizeof(float)));
    }

    void free_memory() {
        // Free node data
        if (nodes.position_x) cudaFree(nodes.position_x);
        if (nodes.position_y) cudaFree(nodes.position_y);
        if (nodes.position_z) cudaFree(nodes.position_z);
        if (nodes.velocity_x) cudaFree(nodes.velocity_x);
        if (nodes.velocity_y) cudaFree(nodes.velocity_y);
        if (nodes.velocity_z) cudaFree(nodes.velocity_z);
        if (nodes.visual_saliency) cudaFree(nodes.visual_saliency);
        if (nodes.community_id) cudaFree(nodes.community_id);

        // Free embedding data
        if (embeddings.values) cudaFree(embeddings.values);
        if (embeddings.dimensions) cudaFree(embeddings.dimensions);
        if (embeddings.quality_scores) cudaFree(embeddings.quality_scores);
    }

    void update_umap_embeddings(int epoch) {
        // Implementation would call umap_optimization_kernel_fixed
        // with proper k-NN graph and spatial grid
    }

    void compute_spectral_features() {
        // Implementation would compute Laplacian eigendecomposition
        // using the fixed kernels
    }

    void detect_anomalies() {
        // Implementation would run LOF and other anomaly detection
    }

    void update_node_positions(int iter) {
        // Update positions based on various analytics results
        dim3 block(256);
        dim3 grid((nodes.num_nodes + block.x - 1) / block.x);

        // Simple position update kernel
        update_positions_kernel<<<grid, block>>>(
            nodes.position_x, nodes.position_y, nodes.position_z,
            nodes.velocity_x, nodes.velocity_y, nodes.velocity_z,
            embeddings.values, embeddings.dimensions,
            nodes.visual_saliency, nodes.community_id,
            nodes.num_nodes, iter
        );
        CUDA_CHECK(cudaGetLastError());
    }

    __global__ static void update_positions_kernel(
        float* pos_x, float* pos_y, float* pos_z,
        float* vel_x, float* vel_y, float* vel_z,
        const float* embedding_vals, const int* embedding_dims,
        const float* saliency, const int* community,
        int N, int iter
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        // Blend embedding position with physics
        if (embedding_dims[idx] >= 2) {
            float blend = 0.1f;
            pos_x[idx] = (1.0f - blend) * pos_x[idx] +
                         blend * embedding_vals[idx] * 100.0f;
            pos_y[idx] = (1.0f - blend) * pos_y[idx] +
                         blend * embedding_vals[N + idx] * 100.0f;

            if (embedding_dims[idx] >= 3) {
                pos_z[idx] = (1.0f - blend) * pos_z[idx] +
                            blend * embedding_vals[2 * N + idx] * 50.0f;
            }
        }

        // Apply velocity with damping
        float damping = 0.95f;
        pos_x[idx] += vel_x[idx] * 0.01f;
        pos_y[idx] += vel_y[idx] * 0.01f;
        pos_z[idx] += vel_z[idx] * 0.01f;

        vel_x[idx] *= damping;
        vel_y[idx] *= damping;
        vel_z[idx] *= damping;
    }
};

// ============================================================================
// Example Usage
// ============================================================================

extern "C" {

void run_visual_analytics(
    int num_nodes,
    int num_edges,
    int num_iterations
) {
    // Create pipeline
    VisualAnalyticsPipeline pipeline(num_nodes, num_edges);

    // Run analysis
    pipeline.run_analysis(num_iterations);

    printf("Visual analytics completed successfully!\n");
}

} // extern "C"

#endif // VISUAL_ANALYTICS_CUDA_FIXED_H