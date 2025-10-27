# GPU Algorithms API Reference

## Overview

VisionFlow implements real GPU-accelerated algorithms using CUDA kernels for clustering, anomaly detection, and graph analysis. All algorithms perform actual computations with no mock data or placeholders.

**Base URL**: `http://localhost:3030/api`

## Supported GPU Algorithms

### Clustering Algorithms

#### K-means Clustering
Real CUDA-accelerated K-means implementation with parallel centroid updates.

```http
POST /api/clustering/start
Content-Type: application/json

{
  "algorithm": "kmeans",
  "clusterCount": 5,
  "maxIterations": 100,
  "convergenceThreshold": 0.001,
  "randomState": 42
}
```

**GPU Implementation Details:**
- Parallel distance calculations across CUDA cores
- Shared memory optimisation for centroid updates
- Real-time convergence monitoring
- Automatic scaling for large datasets

**Response:**
```json
{
  "status": "completed",
  "algorithm": "kmeans",
  "clustersFound": 5,
  "convergenceAchieved": true,
  "iterations": 23,
  "computationTimeMs": 145,
  "gpuAccelerated": true,
  "kernelExecutions": 147,
  "gpuMemoryUsed": "1.2 GB",
  "qualityMetrics": {
    "inertia": 1524.67,
    "silhouetteScore": 0.72,
    "calinskiHarabaszIndex": 298.4
  },
  "centroids": [
    [125.5, 78.2, 45.1],
    [200.1, 150.3, 89.7],
    [75.8, 220.1, 130.4]
  ]
}
```

#### Louvain Community Detection
GPU-accelerated community detection with modularity optimisation.

```http
POST /api/clustering/start
Content-Type: application/json

{
  "algorithm": "louvain",
  "resolution": 1.0,
  "maxIterations": 100,
  "randomState": 42
}
```

**GPU Implementation Details:**
- Parallel modularity calculations
- GPU-accelerated neighbour aggregation
- Real community merge operations
- Dynamic memory allocation for variable cluster sizes

**Response:**
```json
{
  "status": "completed",
  "algorithm": "louvain",
  "clustersFound": 8,
  "modularity": 0.847,
  "resolution": 1.0,
  "iterations": 15,
  "computationTimeMs": 234,
  "gpuAccelerated": true,
  "communityHierarchy": {
    "levels": 3,
    "modularityGain": 0.623
  },
  "clusters": [
    {
      "id": 0,
      "size": 45,
      "density": 0.73,
      "internalEdges": 127,
      "externalEdges": 23
    }
  ]
}
```

#### DBSCAN Clustering
Density-based clustering with GPU-accelerated neighbour searches.

```http
POST /api/clustering/start
Content-Type: application/json

{
  "algorithm": "dbscan",
  "eps": 0.5,
  "minSamples": 5,
  "metric": "euclidean"
}
```

**GPU Implementation Details:**
- GPU k-d tree construction for neighbour queries
- Parallel core point identification
- CUDA-accelerated cluster expansion
- Real noise point detection

**Response:**
```json
{
  "status": "completed",
  "algorithm": "dbscan",
  "clustersFound": 12,
  "noisePoints": 8,
  "corePoints": 142,
  "borderPoints": 67,
  "eps": 0.5,
  "minSamples": 5,
  "computationTimeMs": 189,
  "gpuAccelerated": true,
  "densityMetrics": {
    "averageDensity": 0.68,
    "maxClusterDensity": 0.94,
    "minClusterDensity": 0.42
  }
}
```

### Anomaly Detection Algorithms

#### Local Outlier Factor (LOF)
GPU-accelerated outlier detection with real k-nearest neighbour calculations.

```http
POST /api/analytics/anomaly-detection
Content-Type: application/json

{
  "method": "lof",
  "k": 20,
  "contamination": 0.1,
  "metric": "euclidean"
}
```

**GPU Implementation Details:**
- Parallel k-NN distance calculations
- GPU-accelerated local density computation
- Real outlier score calculations
- Memory-efficient batch processing

**Response:**
```json
{
  "method": "lof",
  "anomaliesDetected": 15,
  "totalDataPoints": 1500,
  "contaminationRate": 0.10,
  "computationTimeMs": 167,
  "gpuAccelerated": true,
  "outlierScores": {
    "min": 0.82,
    "max": 3.47,
    "mean": 1.12,
    "threshold": 1.85
  },
  "anomalies": [
    {
      "nodeId": "node_1247",
      "lofScore": 3.47,
      "kDistance": 0.89,
      "localDensity": 0.23
    }
  ],
  "performance": {
    "knnComputeTime": 89,
    "lofComputeTime": 78,
    "gpuUtilization": 92
  }
}
```

#### Isolation Forest
GPU-accelerated ensemble anomaly detection with parallel tree construction.

```http
POST /api/analytics/anomaly-detection
Content-Type: application/json

{
  "method": "isolation_forest",
  "n_estimators": 100,
  "contamination": 0.1,
  "max_samples": "auto",
  "randomState": 42
}
```

**GPU Implementation Details:**
- Parallel isolation tree construction
- GPU-accelerated path length calculations
- Real ensemble scoring
- Memory-optimised tree storage

**Response:**
```json
{
  "method": "isolation_forest",
  "anomaliesDetected": 12,
  "totalDataPoints": 1500,
  "contaminationRate": 0.08,
  "nEstimators": 100,
  "computationTimeMs": 203,
  "gpuAccelerated": true,
  "isolationScores": {
    "min": 0.34,
    "max": 0.78,
    "mean": 0.52,
    "threshold": 0.65
  },
  "forestMetrics": {
    "averageTreeDepth": 8.7,
    "maxTreeDepth": 12,
    "treesBuilt": 100,
    "pathLengthVariance": 2.3
  }
}
```

#### Z-Score Anomaly Detection
Statistical outlier detection with GPU-accelerated statistical computations.

```http
POST /api/analytics/anomaly-detection
Content-Type: application/json

{
  "method": "zscore",
  "threshold": 2.5,
  "features": ["x", "y", "connectivity"]
}
```

**GPU Implementation Details:**
- Parallel mean and standard deviation calculations
- GPU-accelerated z-score computations
- Real multivariate outlier detection
- Efficient statistical aggregations

**Response:**
```json
{
  "method": "zscore",
  "anomaliesDetected": 8,
  "totalDataPoints": 1500,
  "threshold": 2.5,
  "computationTimeMs": 78,
  "gpuAccelerated": true,
  "statistics": {
    "features": {
      "x": {"mean": 125.4, "std": 45.2, "outliers": 3},
      "y": {"mean": 89.7, "std": 32.1, "outliers": 2},
      "connectivity": {"mean": 4.2, "std": 1.8, "outliers": 3}
    }
  },
  "anomalies": [
    {
      "nodeId": "node_856",
      "zScores": {"x": 3.2, "y": 1.8, "connectivity": 2.9},
      "maxZScore": 3.2,
      "feature": "x"
    }
  ]
}
```

### Graph Analysis Algorithms

#### Stress Majorization
GPU-accelerated force-directed layout with real stress optimisation.

```http
POST /api/graph/layout/stress-majorization
Content-Type: application/json

{
  "iterations": 100,
  "tolerance": 0.001,
  "stepSize": 0.1,
  "adaptiveStepSize": true
}
```

**GPU Implementation Details:**
- Parallel stress gradient calculations
- GPU-accelerated matrix operations
- Real iterative position updates
- Dynamic step size adjustment

**Response:**
```json
{
  "algorithm": "stress_majorization",
  "iterations": 67,
  "finalStress": 0.0008,
  "convergenceAchieved": true,
  "computationTimeMs": 312,
  "gpuAccelerated": true,
  "stressReduction": 0.95,
  "positionUpdates": 1847,
  "performance": {
    "avgIterationTime": 4.7,
    "gradientComputeTime": 156,
    "matrixOperationTime": 89,
    "gpuUtilization": 88
  },
  "finalPositions": [
    {"nodeId": "node_1", "x": 145.2, "y": 78.9, "z": 23.1},
    {"nodeId": "node_2", "x": 89.7, "y": 156.3, "z": 45.8}
  ]
}
```

## Real-Time GPU Monitoring

### GPU Status Endpoint

```http
GET /api/gpu/status
```

**Response:**
```json
{
  "gpuAvailable": true,
  "devices": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "computeCapability": "8.9",
      "totalMemory": "24 GB",
      "freeMemory": "18.5 GB",
      "utilization": 67,
      "temperature": 72,
      "powerUsage": 285
    }
  ],
  "cudaVersion": "12.0",
  "driverVersion": "525.85",
  "activeKernels": 3,
  "memoryPools": {
    "allocated": "5.5 GB",
    "cached": "2.1 GB",
    "reserved": "7.6 GB"
  }
}
```

### Algorithm Performance Metrics

```http
GET /api/gpu/performance/{algorithm}
```

**Response:**
```json
{
  "algorithm": "louvain",
  "recentExecutions": 15,
  "averageExecutionTime": 234,
  "memoryUsagePattern": {
    "peak": "3.2 GB",
    "average": "2.1 GB",
    "allocations": 47
  },
  "kernelPerformance": {
    "modularityKernel": {
      "avgExecutionTime": 12.3,
      "occupancy": 0.85,
      "threadUtilization": 0.92
    },
    "communityMergeKernel": {
      "avgExecutionTime": 8.7,
      "occupancy": 0.78,
      "threadUtilization": 0.88
    }
  },
  "optimisations": {
    "sharedMemoryUsage": 89,
    "coalescedAccess": 0.94,
    "warpEfficiency": 0.87
  }
}
```

## GPU Configuration

### CUDA Parameters

```http
POST /api/gpu/configure
Content-Type: application/json

{
  "device": 0,
  "memoryPoolSize": "8 GB",
  "kernelConfig": {
    "threadsPerBlock": 256,
    "blocksPerGrid": 128,
    "sharedMemorySize": 48
  },
  "optimisations": {
    "coalescedAccess": true,
    "tensorCores": true,
    "memoryPrefetch": true
  }
}
```

### Algorithm-Specific Tuning

```http
POST /api/gpu/algorithms/{algorithm}/tune
Content-Type: application/json

{
  "algorithm": "kmeans",
  "parameters": {
    "batchSize": 1024,
    "convergenceEpsilon": 0.0001,
    "maxThreadsPerSM": 2048
  },
  "autoTune": true
}
```

## Error Handling

### GPU Memory Errors

```json
{
  "error": {
    "code": "GPU_OUT_OF_MEMORY",
    "message": "Insufficient GPU memory for operation",
    "details": {
      "requested": "8.5 GB",
      "available": "6.2 GB",
      "algorithm": "louvain",
      "dataPoints": 50000
    }
  },
  "fallback": {
    "suggestion": "Reduce batch size or use CPU fallback",
    "cpuFallbackAvailable": true,
    "estimatedCpuTime": 15000
  }
}
```

### CUDA Kernel Errors

```json
{
  "error": {
    "code": "KERNEL_LAUNCH_FAILED",
    "message": "CUDA kernel execution failed",
    "details": {
      "kernel": "kmeans_centroids_update",
      "errorCode": "CUDA_ERROR_INVALID_VALUE",
      "threadConfig": {
        "blocks": 128,
        "threads": 256
      }
    }
  },
  "diagnostic": {
    "memoryStatus": "healthy",
    "deviceStatus": "operational",
    "suggestedFix": "Reduce thread count or check input parameters"
  }
}
```

## Performance Optimization Guidelines

### Memory Management
- Use memory pools for frequent allocations
- Implement memory prefetching for large datasets
- Monitor memory fragmentation
- Use pinned memory for host-device transfers

### Kernel Optimization
- Maximize occupancy through proper thread/block sizing
- Utilize shared memory for frequently accessed data
- Ensure coalesced memory access patterns
- Implement warp-level primitives where applicable

### Algorithm Scaling
- Implement adaptive batch sizing
- Use multi-GPU support for large datasets
- Optimize for specific GPU architectures
- Monitor thermal throttling and adjust workloads

## Integration with Agent System

### GPU-Accelerated Agent Tasks

```json
{
  "taskId": "gpu_task_1757967065850",
  "agentId": "agent_1757967065850_dv2zg7",
  "gpuOperation": {
    "algorithm": "louvain",
    "dataSize": "2.5M nodes",
    "estimatedGpuTime": 450,
    "gpuResourcesAllocated": true
  },
  "mcpIntegration": {
    "swarmCoordination": true,
    "resultDistribution": "broadcast",
    "followUpTasks": ["visualisation_update", "cluster_analysis"]
  }
}
```

## Related Documentation

- [REST API](../reference/api/rest-api.md) - HTTP endpoints for GPU operations
- [CUDA Parameters](../cuda-parameters.md) - Detailed CUDA configuration
- [Performance Benchmarks](../../guides/performance.md) - Algorithm performance analysis

---

**[← MCP Protocol](../reference/api/mcp-protocol.md)** | **[Back to API Index →](README.md)**