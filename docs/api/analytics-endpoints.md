# Analytics API Endpoints

This document describes the analytics endpoints that bridge the client-server gap for semantic clustering, anomaly detection, and AI insights functionality.

## Overview

The analytics API provides GPU-accelerated graph analysis capabilities including:

- **Semantic Clustering**: Multiple clustering algorithms (spectral, hierarchical, DBSCAN, etc.)
- **Anomaly Detection**: Real-time outlier detection with various methods
- **AI Insights**: Intelligent analysis and recommendations based on graph patterns

## Endpoints

### Clustering

#### POST `/api/analytics/clustering/run`

Starts a clustering analysis task.

**Request Body:**
```json
{
  "method": "spectral",
  "params": {
    "numClusters": 8,
    "minClusterSize": 5,
    "similarity": "cosine",
    "convergenceThreshold": 0.001,
    "maxIterations": 100
  }
}
```

**Response:**
```json
{
  "success": true,
  "clusters": null,
  "method": "spectral",
  "executionTimeMs": null,
  "taskId": "task-uuid-123",
  "error": null
}
```

**Supported Methods:**
- `spectral` - Spectral clustering using eigendecomposition
- `hierarchical` - Tree-based hierarchical decomposition  
- `dbscan` - Density-based spatial clustering
- `kmeans` - K-means++ centroid-based partitioning
- `louvain` - Community detection via modularity optimisation
- `affinity` - Affinity propagation message passing

#### GET `/api/analytics/clustering/status?task_id=<id>`

Get the status of a clustering task.

**Response:**
```json
{
  "success": true,
  "taskId": "task-uuid-123",
  "status": "running",
  "progress": 0.65,
  "method": "spectral", 
  "startedAt": "1640995200",
  "estimatedCompletion": "1640995260",
  "error": null
}
```

**Status Values:**
- `pending` - Task queued but not started
- `running` - Task currently executing
- `completed` - Task finished successfully  
- `failed` - Task encountered an error

#### POST `/api/analytics/clustering/focus`

Focus the visualisation on a specific cluster.

**Request Body:**
```json
{
  "clusterId": "cluster-uuid-123",
  "zoomLevel": 5.0,
  "highlight": true
}
```

### Anomaly Detection

#### POST `/api/analytics/anomaly/toggle`

Enable or disable anomaly detection.

**Request Body:**
```json
{
  "enabled": true,
  "method": "isolation_forest",
  "sensitivity": 0.5,
  "windowSize": 100,
  "updateInterval": 5000
}
```

**Supported Methods:**
- `isolation_forest` - Tree-based anomaly isolation
- `lof` - Local Outlier Factor density-based detection
- `autoencoder` - Neural reconstruction error analysis
- `statistical` - Z-score and IQR based detection
- `temporal` - Time-series anomaly detection

#### GET `/api/analytics/anomaly/current`

Get current anomalies and statistics.

**Response:**
```json
{
  "success": true,
  "anomalies": [
    {
      "id": "anomaly-uuid-123",
      "nodeId": "node_456", 
      "type": "structural_outlier",
      "severity": "high",
      "score": 0.89,
      "description": "Node exhibits unusual structural properties",
      "timestamp": 1640995200,
      "metadata": {
        "detectionMethod": "isolation_forest",
        "confidence": 0.89
      }
    }
  ],
  "stats": {
    "total": 15,
    "critical": 2,
    "high": 5,
    "medium": 6, 
    "low": 2,
    "lastUpdated": 1640995200
  },
  "enabled": true,
  "method": "isolation_forest"
}
```

**Severity Levels:**
- `critical` - Requires immediate attention
- `high` - Significant anomaly
- `medium` - Notable deviation  
- `low` - Minor outlier

### AI Insights

#### GET `/api/analytics/insights`

Get AI-generated insights about the graph structure and patterns.

**Response:**
```json
{
  "success": true,
  "insights": [
    "Graph structure analysis shows balanced connectivity patterns",
    "Identified 8 distinct semantic clusters",
    "Detected 15 anomalies across the graph"
  ],
  "patterns": [
    {
      "id": "pattern-uuid-123",
      "type": "dominant_cluster",
      "description": "Large semantic cluster with 50+ nodes",
      "confidence": 0.85,
      "nodes": [1, 2, 3, 4, 5],
      "significance": "high"
    }
  ],
  "recommendations": [
    "Consider increasing clustering threshold to reduce cluster count",
    "Investigate critical anomalies that may indicate data quality issues"
  ],
  "analysisTimestamp": 1640995200
}
```

**Pattern Types:**
- `dominant_cluster` - Unusually large cluster
- `anomaly_pattern` - Distribution of anomalies
- `connectivity_hub` - Highly connected nodes
- `isolated_component` - Disconnected graph regions

**Significance Levels:**
- `high` - Important pattern requiring attention
- `medium` - Noteworthy pattern
- `low` - Minor observation

## Implementation Details

### GPU Acceleration

All clustering algorithms leverage GPU compute capabilities when available:
- CUDA kernels for parallel processing
- WASM SIMD fallback for CPU-only systems
- Automatic selection based on hardware capabilities

### Real-time Updates

Anomaly detection runs continuously when enabled:
- Background processing with configurable intervals
- WebSocket notifications for real-time updates
- Automatic cleanup of old anomalies

### Performance Considerations

- Clustering tasks run asynchronously to avoid blocking
- Results are cached and reused when appropriate
- Memory-efficient algorithms for large graphs
- Progressive loading for UI responsiveness

### Error Handling

All endpoints return consistent error structures:
```json
{
  "success": false,
  "error": "Descriptive error message",
  "code": "ERROR_CODE"
}
```

### Integration

These endpoints integrate with existing analytics infrastructure:
- GPU compute actor for algorithm execution
- Graph service actor for data access
- Settings service for configuration persistence
- WebSocket system for real-time updates

## Client Usage

The client-side `SemanticClusteringControls.tsx` component uses these endpoints:

```typescript
// Run clustering
const response = await fetch('/api/analytics/clustering/run', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ method: 'spectral', params: {...} })
});

// Check status
const status = await fetch(`/api/analytics/clustering/status?task_id=${taskId}`);

// Toggle anomaly detection  
await fetch('/api/analytics/anomaly/toggle', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ enabled: true, method: 'isolation_forest', ... })
});

// Get current anomalies
const anomalies = await fetch('/api/analytics/anomaly/current');

// Get AI insights
const insights = await fetch('/api/analytics/insights');
```

This ensures perfect compatibility between client expectations and server implementation.