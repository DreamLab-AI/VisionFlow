# GPU K-means Clustering and Anomaly Detection Implementation

## Overview

This document describes the implementation of GPU-accelerated K-means clustering and anomaly detection kernels in the VisionFlow CUDA codebase.

## Features Implemented

### K-means Clustering

1. **init_centroids_kernel**: Implements K-means++ initialization algorithm
   - Grid: `(k, 1, 1)`, Block: `(256, 1, 1)` where k = num_clusters
   - Uses shared memory for reduction operations
   - Implements probabilistic centroid selection

2. **assign_clusters_kernel**: Assigns nodes to nearest centroid
   - Grid: `(ceil(num_nodes/256), 1, 1)`, Block: `(256, 1, 1)`
   - Each thread processes one node
   - Computes squared Euclidean distances

3. **update_centroids_kernel**: Updates centroids based on assignments
   - Grid: `(num_clusters, 1, 1)`, Block: `(256, 1, 1)`
   - Each block processes one cluster centroid
   - Uses shared memory for position accumulation

4. **compute_inertia_kernel**: Computes sum of squared distances to centroids
   - Grid: `(ceil(num_nodes/256), 1, 1)`, Block: `(256, 1, 1)`
   - Uses shared memory for partial reduction
   - Requires host-side reduction for final inertia

### Anomaly Detection

1. **compute_lof_kernel**: Local Outlier Factor (LOF) anomaly detection
   - Grid: `(ceil(num_nodes/256), 1, 1)`, Block: `(256, 1, 1)`
   - Each thread processes one node
   - Uses spatial grid for efficient neighbor search
   - Computes k-nearest neighbors and local density

2. **compute_zscore_kernel**: Z-score based anomaly detection
   - Grid: `(ceil(num_nodes/256), 1, 1)`, Block: `(256, 1, 1)`
   - Requires pre-computed mean and standard deviation
   - Clamps extreme values for numerical stability

3. **compute_feature_stats_kernel**: Computes feature statistics
   - Grid: `(ceil(num_nodes/256), 1, 1)`, Block: `(256, 1, 1)`
   - Uses shared memory for partial sum reduction
   - Computes mean and variance for Z-score calculation

## Device Buffers Added to UnifiedGPUCompute

### K-means Clustering Buffers
- `centroids_x`, `centroids_y`, `centroids_z`: Centroid positions
- `cluster_assignments`: Node-to-cluster assignments
- `distances_to_centroid`: Distance from each node to its centroid
- `cluster_sizes`: Number of nodes in each cluster
- `partial_inertia`: Partial inertia sums for reduction
- `min_distances`: Minimum distances for K-means++ initialization
- `selected_nodes`: Selected centroid nodes during initialization

### Anomaly Detection Buffers
- `lof_scores`: Local Outlier Factor scores
- `local_densities`: Local density values
- `zscore_values`: Z-score values for each node
- `feature_values`: Input feature values for Z-score computation
- `partial_sums`, `partial_sq_sums`: Partial statistics for reduction

## API Methods

### K-means Clustering
```rust
pub fn run_kmeans(&mut self, num_clusters: usize, max_iterations: u32, tolerance: f32, seed: u32) 
    -> Result<(Vec<i32>, Vec<(f32, f32, f32)>, f32)>
```

Returns cluster assignments, centroids, and final inertia.

### Anomaly Detection
```rust
pub fn run_lof_anomaly_detection(&mut self, k_neighbors: i32, radius: f32) 
    -> Result<(Vec<f32>, Vec<f32>)>

pub fn run_zscore_anomaly_detection(&mut self, feature_data: &[f32]) 
    -> Result<Vec<f32>>
```

## Actor System Integration

### Message Types
- `RunKMeans`: Executes K-means clustering with parameters
- `RunAnomalyDetection`: Executes anomaly detection with method selection

### Parameters
- `KMeansParams`: num_clusters, max_iterations, tolerance, seed
- `AnomalyParams`: method, k_neighbors, radius, feature_data, threshold

### Results  
- `KMeansResult`: cluster_assignments, centroids, inertia, iterations
- `AnomalyResult`: lof_scores, local_densities, zscore_values, num_anomalies

## Performance Optimizations

1. **Shared Memory Usage**:
   - K-means centroid updates use shared memory for reduction
   - LOF computation uses fixed-size arrays for k-nearest neighbors
   - Statistics computation uses shared memory for partial sums

2. **Numerical Stability**:
   - LOF scores clamped to [0.1, 10.0] range
   - Z-scores clamped to [-10.0, 10.0] range
   - Epsilon values added to prevent division by zero

3. **Memory Coalescing**:
   - SoA (Structure of Arrays) layout for position data
   - Aligned memory accesses for optimal bandwidth

## Error Handling

- Validates input parameters (cluster count, feature data size)
- Handles empty nodes gracefully
- Returns meaningful error messages for debugging
- Implements convergence checking for K-means

## Grid/Block Configuration Guidelines

- **Block Size**: 256 threads (optimal for most modern GPUs)
- **Grid Size**: `ceil(num_nodes/block_size)` for node-parallel kernels
- **Shared Memory**: Allocated dynamically based on block size
- **Occupancy**: Designed for high occupancy on modern GPU architectures

## Integration with Spatial Grid

The LOF anomaly detection leverages the existing spatial grid infrastructure:
- Reuses `cell_keys`, `sorted_node_indices`, `cell_start`, `cell_end` buffers
- Benefits from spatial locality for neighbor searches
- Maintains consistency with physics simulation grid

## Future Enhancements

1. **Multi-GPU Support**: Scale across multiple GPUs for larger datasets
2. **Hierarchical Clustering**: Add support for hierarchical clustering algorithms
3. **Online Learning**: Support streaming/online anomaly detection
4. **Custom Kernels**: Support user-defined distance metrics and features
5. **Memory Optimization**: Dynamic buffer sizing based on actual usage