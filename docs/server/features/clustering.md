# Graph Clustering

*[Server](../index.md) > [Features](../server/features/index.md)*

## Introduction

Graph clustering is a critical analytical feature used to automatically identify communities, functional modules, and semantic groups within the knowledge graph. By partitioning the graph into clusters of densely connected nodes, we can uncover hidden structures and relationships that are not immediately obvious.

This is particularly useful for understanding the high-level structure of a knowledge base, identifying related concepts, and improving the quality of the visualisation.

## Available Algorithms

The system supports several industry-standard clustering algorithms, each with its own strengths and ideal use cases.

-   **Spectral Clustering**:
    *   **Description**: A technique that uses the eigenvalues (the "spectrum") of the graph's Laplacian matrix to perform dimensionality reduction before clustering in a lower-dimensional space.
    *   **Use Case**: Excellent for identifying clean, well-separated clusters of arbitrary shapes. It is often more effective than other methods when clusters are not globular.

-   **K-means Clustering**:
    *   **Description**: An iterative algorithm that partitions the graph into a pre-defined number (`k`) of clusters. It works by assigning each node to the nearest cluster centroid (mean) and then recalculating the centroids.
    *   **Use Case**: Best suited for when the number of clusters is known beforehand and the clusters are roughly spherical and of similar size. It is computationally efficient for large graphs.

-   **Louvain Method**:
    *   **Description**: A fast, greedy optimisation method for community detection in large networks. It works by optimising a "modularity" score, which measures the density of links inside communities compared to links between communities.
    *   **Use Case**: Highly effective and scalable for community detection in very large graphs. It does not require the number of clusters to be specified in advance.

## API Endpoints

The clustering functionality is exposed through a set of REST API endpoints, documented in `src/handlers/api_handler/analytics/mod.rs`.

-   `POST /api/analytics/clustering/run`
    *   **Description**: Initiates a new clustering task. The request body must contain the desired algorithm and any necessary parameters.
    *   **Returns**: A task ID that can be used to check the status of the clustering job.

-   `GET /api/analytics/clustering/status`
    *   **Description**: Retrieves the status and results of a previously submitted clustering task.
    *   **Parameters**: Requires the task ID returned by the `/run` endpoint.
    *   **Returns**: The current status (e.g., `pending`, `running`, `completed`, `failed`) and, if completed, the clustering results.

## GPU Acceleration

To handle the computationally intensive nature of clustering algorithms on large graphs, the system leverages the `GPUComputeActor`. When a clustering task is initiated, the `GraphServiceActor` sends a `PerformGPUClustering` message to the `GPUComputeActor`.

This offloads the heavy matrix operations and iterative calculations to the unified GPU kernel, which can process them in parallel across thousands of threads. This results in a significant performance improvement, allowing for near-real-time clustering on graphs that would be prohibitively slow to process on the CPU.

## Configuration

The clustering process can be configured by sending a `ClusteringParams` struct in the body of the `POST /api/analytics/clustering/run` request.

**Example `ClusteringParams`:**
```json
{
  "algorithm": "louvain",
  "parameters": {
    "resolution": 1.0,
    "tolerance": 0.0001
  },
  "output_property": "clusterId"
}
```

-   `algorithm`: The name of the clustering algorithm to use (`spectral`, `kmeans`, or `louvain`).
-   `parameters`: A nested object containing algorithm-specific parameters (e.g., `k` for K-means, `resolution` for Louvain).
-   `output_property`: The name of the node property where the resulting cluster ID should be stored.

## Related Topics

- [AI Services Documentation](../../server/ai-services.md)
- [Actor System](../../server/actors.md)
- [Adaptive Balancing](../../features/adaptive-balancing.md)
- [Agent Orchestration Architecture](../../features/agent-orchestration.md)
- [Claude Flow MCP Integration](../../server/features/claude-flow-mcp-integration.md)
- [Configuration Architecture](../../server/config.md)
- [Feature Access Control](../../server/feature-access.md)
- [Features Documentation](../../features/index.md)
- [GPU Compute Architecture](../../server/gpu-compute.md)
- [GPU-Accelerated Analytics](../../client/features/gpu-analytics.md)
- [MCP Integration](../../server/mcp-integration.md)
- [Multi Agent Orchestration](../../server/agent-swarm.md)
- [Neural Auto-Balance Feature](../../features/AUTO_BALANCE.md)
- [Ontology Validation](../../server/features/ontology.md)
- [Physics Engine](../../server/physics-engine.md)
- [Request Handlers Architecture](../../server/handlers.md)
- [Semantic Analysis Pipeline](../../server/features/semantic-analysis.md)
- [Server Architecture](../../server/architecture.md)
- [Server Documentation](../../server/index.md)
- [Server-Side Data Models](../../server/models.md)
- [Services Architecture](../../server/services.md)
- [Types Architecture](../../server/types.md)
- [Utilities Architecture](../../server/utils.md)
