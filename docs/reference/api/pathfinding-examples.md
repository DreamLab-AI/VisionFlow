---
title: Pathfinding API Examples
description: The Pathfinding API provides GPU-accelerated graph analytics for shortest paths and connected components analysis. This document provides practical examples for each endpoint.
category: reference
tags:
  - api
  - api
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# Pathfinding API Examples

## Overview

The Pathfinding API provides GPU-accelerated graph analytics for shortest paths and connected components analysis. This document provides practical examples for each endpoint.

## Endpoints

### 1. Single-Source Shortest Path (SSSP)

Computes shortest paths from a single source node to all other reachable nodes.

**Endpoint:** `POST /api/analytics/pathfinding/sssp`

**Use Cases:**
- Path highlighting in graph visualization
- Reachability analysis from a specific node
- Distance-based filtering (nodes within N hops)
- Proximity queries

**Request Example:**
```bash
curl -X POST http://localhost:8080/api/analytics/pathfinding/sssp \
  -H "Content-Type: application/json" \
  -d '{
    "sourceIdx": 0,
    "maxDistance": 5.0
  }'
```

**Request Parameters:**
- `sourceIdx` (required): Index of the source node
- `maxDistance` (optional): Maximum distance cutoff (filters results)

**Response Example:**
```json
{
  "success": true,
  "result": {
    "distances": [0.0, 1.5, 2.3, 3.1, ...],
    "sourceIdx": 0,
    "nodesReached": 1234,
    "maxDistance": 4.8,
    "computationTimeMs": 15
  },
  "error": null
}
```

**Response Fields:**
- `distances`: Array of distances from source (indexed by node index, f32::MAX = unreachable)
- `sourceIdx`: The source node index used
- `nodesReached`: Number of nodes within maxDistance
- `maxDistance`: Maximum distance found in graph
- `computationTimeMs`: GPU computation time

**Advanced Example (No distance limit):**
```bash
curl -X POST http://localhost:8080/api/analytics/pathfinding/sssp \
  -H "Content-Type: application/json" \
  -d '{
    "sourceIdx": 42
  }'
```

---

### 2. All-Pairs Shortest Path (APSP)

Computes approximate shortest paths between all node pairs using landmark-based method.

**Endpoint:** `POST /api/analytics/pathfinding/apsp`

**Use Cases:**
- Distance matrix computation for visualization
- Graph layout with distance preservation
- Centrality analysis (betweenness, closeness)
- Similarity-based clustering

**Request Example:**
```bash
curl -X POST http://localhost:8080/api/analytics/pathfinding/apsp \
  -H "Content-Type: application/json" \
  -d '{
    "numLandmarks": 10,
    "seed": 42
  }'
```

**Request Parameters:**
- `numLandmarks` (optional): Number of landmark nodes for approximation (default: sqrt(n))
- `seed` (optional): Random seed for landmark selection (default: 42)

**Response Example:**
```json
{
  "success": true,
  "result": {
    "distances": [0.0, 1.5, 2.3, ..., 4.1],
    "numNodes": 1000,
    "numLandmarks": 10,
    "landmarks": [5, 123, 456, 789, ...],
    "avgErrorEstimate": 0.15,
    "computationTimeMs": 245
  },
  "error": null
}
```

**Response Fields:**
- `distances`: Flattened distance matrix [numNodes x numNodes] in row-major order
  - Access: `distance[i][j] = distances[i * numNodes + j]`
- `numNodes`: Total number of nodes
- `numLandmarks`: Number of landmarks used
- `landmarks`: Indices of landmark nodes selected
- `avgErrorEstimate`: Average approximation error (typically ~15%)
- `computationTimeMs`: Total GPU computation time

**Accessing Distance Matrix:**
```javascript
// JavaScript example
const getDistance = (i, j, distances, numNodes) => {
  return distances[i * numNodes + j];
};

// Python example
import numpy as np
distances_matrix = np.array(distances).reshape(numNodes, numNodes)
distance_i_j = distances_matrix[i, j]
```

**Default Landmarks Example:**
```bash
# Uses sqrt(numNodes) landmarks automatically
curl -X POST http://localhost:8080/api/analytics/pathfinding/apsp \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

### 3. Connected Components

Detects disconnected regions in the graph using GPU label propagation.

**Endpoint:** `POST /api/analytics/pathfinding/connected-components`

**Use Cases:**
- Identifying graph clusters/islands
- Network fragmentation detection
- Component-based visualization
- Graph partitioning analysis

**Request Example:**
```bash
curl -X POST http://localhost:8080/api/analytics/pathfinding/connected-components \
  -H "Content-Type: application/json" \
  -d '{
    "maxIterations": 100
  }'
```

**Request Parameters:**
- `maxIterations` (optional): Maximum label propagation iterations (default: 100)
- `convergenceThreshold` (optional): Convergence threshold (default: 0.001)

**Response Example:**
```json
{
  "success": true,
  "result": {
    "labels": [0, 0, 0, 1, 1, 2, 2, 2, ...],
    "numComponents": 3,
    "componentSizes": [1024, 512, 256],
    "largestComponentSize": 1024,
    "isConnected": false,
    "iterations": 8,
    "computationTimeMs": 42
  },
  "error": null
}
```

**Response Fields:**
- `labels`: Component label for each node (indexed by node index)
- `numComponents`: Total number of connected components
- `componentSizes`: Size of each component
- `largestComponentSize`: Size of the largest component
- `isConnected`: True if graph is fully connected (1 component)
- `iterations`: Number of iterations until convergence
- `computationTimeMs`: GPU computation time

**Component Analysis Example:**
```javascript
// Group nodes by component
const groupByComponent = (labels) => {
  const components = {};
  labels.forEach((label, nodeIdx) => {
    if (!components[label]) components[label] = [];
    components[label].push(nodeIdx);
  });
  return components;
};
```

---

### 4. SSSP Statistics

Get performance statistics for shortest path computations.

**Endpoint:** `GET /api/analytics/pathfinding/stats/sssp`

**Request Example:**
```bash
curl http://localhost:8080/api/analytics/pathfinding/stats/sssp
```

**Response Example:**
```json
{
  "totalSsspComputations": 142,
  "totalApspComputations": 8,
  "avgSsspTimeMs": 12.3,
  "avgApspTimeMs": 234.5,
  "lastComputationTimeMs": 15
}
```

---

### 5. Connected Components Statistics

Get performance statistics for connected components analysis.

**Endpoint:** `GET /api/analytics/pathfinding/stats/components`

**Request Example:**
```bash
curl http://localhost:8080/api/analytics/pathfinding/stats/components
```

**Response Example:**
```json
{
  "totalComputations": 25,
  "avgComputationTimeMs": 38.2,
  "avgNumComponents": 3.4,
  "lastNumComponents": 4
}
```

---

## Workflow Examples

### Workflow 1: Path Highlighting in UI

```bash
# Step 1: Compute SSSP from selected node
curl -X POST http://localhost:8080/api/analytics/pathfinding/sssp \
  -H "Content-Type: application/json" \
  -d '{
    "sourceIdx": 0,
    "maxDistance": 3.0
  }' | jq '.result.distances' > distances.json

# Step 2: Use distances for visualization
# - Nodes with distance < f32::MAX are reachable
# - Color/highlight nodes based on distance value
```

### Workflow 2: Graph Connectivity Analysis

```bash
# Step 1: Detect components
COMPONENTS=$(curl -X POST http://localhost:8080/api/analytics/pathfinding/connected-components \
  -H "Content-Type: application/json" \
  -d '{}')

# Step 2: Check if graph is connected
IS_CONNECTED=$(echo $COMPONENTS | jq '.result.isConnected')

# Step 3: Analyze fragmentation
NUM_COMPONENTS=$(echo $COMPONENTS | jq '.result.numComponents')
echo "Graph has $NUM_COMPONENTS disconnected regions"
```

### Workflow 3: Distance Matrix for Layout

```bash
# Step 1: Compute APSP with optimal landmarks
curl -X POST http://localhost:8080/api/analytics/pathfinding/apsp \
  -H "Content-Type: application/json" \
  -d '{
    "numLandmarks": 20,
    "seed": 12345
  }' > apsp_result.json

# Step 2: Extract distance matrix
# Use distances array to position nodes in 2D/3D space
# Apply multidimensional scaling (MDS) or force-directed layout
```

---

## Error Handling

### Common Errors

**GPU Not Available:**
```json
{
  "success": false,
  "result": null,
  "error": "GPU features not enabled"
}
```

**Actor Not Initialized:**
```json
{
  "success": false,
  "result": null,
  "error": "Shortest path actor not available"
}
```

**Invalid Parameters:**
```json
{
  "success": false,
  "result": null,
  "error": "Number of landmarks (2000) must be less than number of nodes (1000)"
}
```

**Actor Communication Error:**
```json
{
  "success": false,
  "result": null,
  "error": "Actor communication error: mailbox closed"
}
```

---

## Performance Notes

### SSSP Performance
- **Typical time:** 10-50ms for graphs with 1,000-10,000 nodes
- **Algorithm:** Bellman-Ford-based frontier compaction
- **GPU acceleration:** ~100x faster than CPU for large graphs

### APSP Performance
- **Typical time:** 100-500ms for graphs with 1,000 nodes and 10-20 landmarks
- **Algorithm:** Landmark-based approximation with triangle inequality
- **Trade-off:** More landmarks = better accuracy but slower computation
- **Approximation error:** ~15% on average

### Connected Components Performance
- **Typical time:** 20-100ms for graphs with 1,000-10,000 nodes
- **Algorithm:** GPU label propagation
- **Convergence:** Usually 5-15 iterations for typical graphs
- **GPU acceleration:** ~50x faster than CPU

---

## API Integration Tips

1. **Batch Operations:** For multiple SSSP queries, consider using APSP instead
2. **Caching:** Cache APSP results for repeated distance queries
3. **Landmarks:** Use ~sqrt(n) landmarks for balanced accuracy/performance
4. **Error Handling:** Always check `success` field before accessing `result`
5. **Feature Detection:** Use `/api/analytics/feature-flags` to check GPU availability

---

---

---

## Related Documentation

- [Authentication (DEPRECATED - JWT NOT USED)](01-authentication.md)
- [Semantic Features API Reference](semantic-features-api.md)
- [WebSocket Binary Protocol Reference](../websocket-protocol.md)
- [Database Schema Reference](../DATABASE_SCHEMA_REFERENCE.md)
- [VisionFlow Binary WebSocket Protocol](../protocols/binary-websocket.md)

## Feature Requirements

All pathfinding endpoints require:
- **Feature flag:** `gpu` enabled at compile time
- **Runtime:** GPU compute actor initialized
- **Hardware:** NVIDIA GPU with CUDA support

Check feature availability:
```bash
curl http://localhost:8080/api/analytics/feature-flags
```
