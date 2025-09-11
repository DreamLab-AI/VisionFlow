# GPU Analytics API Documentation

This document describes the GPU-enabled analytics API endpoints for the knowledge graph visualization system. All endpoints support JSON request/response format and include comprehensive error handling.

## Base URL
```
/api/analytics
```

## Core Analytics Endpoints

### GET /params
Get current visual analytics parameters.

**Response:**
```json
{
  "success": true,
  "params": {
    "focusNode": -1,
    "focusRadius": 2.2,
    "temporalDecay": 0.1,
    "nodeCount": 1000,
    "edgeCount": 2000
  }
}
```

### POST /params
Update visual analytics parameters.

**Request:**
```json
{
  "focusNode": 42,
  "focusRadius": 3.0,
  "temporalDecay": 0.15
}
```

## GPU Control and Monitoring

### GET /gpu-metrics
Get comprehensive GPU performance metrics with client integration.

**Response:**
```json
{
  "success": true,
  "gpu_available": true,
  "utilization": 75.0,
  "memory_used_mb": 2048.0,
  "memory_total_mb": 8192.0,
  "memory_percent": 25.0,
  "temperature": 68.0,
  "power_watts": 120.0,
  "compute_nodes": 50000,
  "compute_edges": 125000,
  "kernel_mode": "advanced",
  "clustering_enabled": true,
  "anomaly_detection_enabled": true,
  "last_updated": 1703721600000
}
```

### GET /gpu-status
Get comprehensive GPU status for control center integration.

**Response:**
```json
{
  "success": true,
  "gpu_available": true,
  "status": "active",
  "compute": {
    "kernel_mode": "advanced",
    "nodes_processed": 50000,
    "edges_processed": 125000,
    "fps": 60.0,
    "frame_time_ms": 16.67
  },
  "analytics": {
    "clustering_active": true,
    "active_clustering_tasks": 2,
    "anomaly_detection_enabled": true,
    "anomalies_detected": 5,
    "critical_anomalies": 1
  },
  "performance": {
    "gpu_utilization": 75.0,
    "memory_usage_percent": 45.0,
    "temperature": 68.0,
    "power_draw": 120.0
  },
  "features": {
    "stress_majorization": true,
    "semantic_constraints": true,
    "sssp_integration": true,
    "spatial_hashing": true,
    "real_time_clustering": true,
    "anomaly_detection": true
  },
  "last_updated": 1703721600000
}
```

### GET /gpu-features
Get available GPU features and capabilities.

**Response:**
```json
{
  "success": true,
  "gpu_acceleration": true,
  "features": {
    "clustering": {
      "available": true,
      "methods": ["kmeans", "spectral", "dbscan", "louvain", "hierarchical", "affinity"],
      "gpu_accelerated": true,
      "max_clusters": 50,
      "max_nodes": 100000
    },
    "anomaly_detection": {
      "available": true,
      "methods": ["isolation_forest", "lof", "autoencoder", "statistical", "temporal"],
      "real_time": true,
      "gpu_accelerated": true
    },
    "graph_algorithms": {
      "sssp": true,
      "stress_majorization": true,
      "spatial_hashing": true,
      "constraint_solving": true
    },
    "visualization": {
      "real_time_updates": true,
      "dynamic_layout": true,
      "focus_regions": true,
      "multi_graph_support": true
    }
  },
  "performance": {
    "expected_speedup": "10-50x",
    "memory_efficiency": "High",
    "concurrent_tasks": true,
    "batch_processing": true
  }
}
```

## Clustering Endpoints

### POST /clustering/run
Start GPU-accelerated clustering analysis.

**Request:**
```json
{
  "method": "kmeans",
  "params": {
    "num_clusters": 8,
    "max_iterations": 100,
    "convergence_threshold": 0.001,
    "random_state": 42
  }
}
```

**Response:**
```json
{
  "success": true,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "kmeans",
  "clusters": null,
  "execution_time_ms": null
}
```

### GET /clustering/status?task_id={id}
Get clustering task progress and status.

**Response:**
```json
{
  "success": true,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": 0.65,
  "method": "kmeans",
  "started_at": "1703721600",
  "estimated_completion": "1703721630"
}
```

### POST /clustering/cancel?task_id={id}
Cancel a running clustering task.

**Response:**
```json
{
  "success": true,
  "message": "Task cancelled successfully",
  "task_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### POST /clustering/focus
Focus visualization on a specific cluster.

**Request:**
```json
{
  "cluster_id": "cluster_0",
  "zoom_level": 5.0,
  "highlight": true
}
```

## Anomaly Detection

### POST /anomaly/toggle
Enable/disable GPU-accelerated anomaly detection.

**Request:**
```json
{
  "enabled": true,
  "method": "isolation_forest",
  "sensitivity": 0.7,
  "window_size": 100,
  "update_interval": 5
}
```

**Response:**
```json
{
  "success": true,
  "enabled": true,
  "method": "isolation_forest",
  "stats": {
    "total": 0,
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  }
}
```

### GET /anomaly/current
Get current anomalies detected by GPU algorithms.

**Response:**
```json
{
  "success": true,
  "anomalies": [
    {
      "id": "anom_001",
      "node_id": "42",
      "type": "outlier",
      "severity": "high",
      "score": 0.92,
      "description": "Node significantly deviates from expected pattern",
      "timestamp": 1703721600,
      "metadata": {
        "detection_method": "isolation_forest",
        "confidence": 0.92,
        "isolation_depth": 5
      }
    }
  ],
  "stats": {
    "total": 5,
    "critical": 1,
    "high": 2,
    "medium": 1,
    "low": 1,
    "last_updated": 1703721600
  },
  "enabled": true,
  "method": "isolation_forest"
}
```

### GET /anomaly/config
Get current anomaly detection configuration.

**Response:**
```json
{
  "success": true,
  "config": {
    "enabled": true,
    "method": "isolation_forest",
    "sensitivity": 0.7,
    "window_size": 100,
    "update_interval": 5
  },
  "stats": {
    "total": 5,
    "critical": 1,
    "high": 2,
    "medium": 1,
    "low": 1
  },
  "supported_methods": [
    "isolation_forest",
    "lof",
    "autoencoder", 
    "statistical",
    "temporal"
  ]
}
```

## AI Insights

### GET /insights
Get comprehensive AI-generated insights about the graph.

**Response:**
```json
{
  "success": true,
  "insights": [
    "Graph structure analysis shows balanced connectivity patterns",
    "Identified 8 distinct semantic clusters",
    "Detected 2 high-severity anomalies requiring attention"
  ],
  "patterns": [
    {
      "id": "pattern_001",
      "type": "dominant_cluster",
      "description": "Large semantic cluster 'Core Concepts' with 1247 nodes",
      "confidence": 0.87,
      "nodes": [1, 2, 3, 4, 5],
      "significance": "high"
    }
  ],
  "recommendations": [
    "Consider increasing clustering threshold to reduce cluster count",
    "Investigate critical anomalies that may indicate data quality issues"
  ],
  "analysis_timestamp": 1703721600
}
```

### GET /insights/realtime
Get real-time AI insights with urgency assessment.

**Response:**
```json
{
  "success": true,
  "insights": [
    "Graph density: 0.234 - moderately connected",
    "Clustering in progress: kmeans method at 65.0% completion",
    "CRITICAL: 1 critical anomalies detected!"
  ],
  "urgency_level": "critical",
  "timestamp": 1703721600,
  "requires_action": true,
  "next_update_ms": 5000
}
```

## Graph Algorithms

### POST /shortest-path
Compute GPU-accelerated single-source shortest paths.

**Request:**
```json
{
  "source_node_id": 42
}
```

**Response:**
```json
{
  "success": true,
  "distances": {
    "1": 2.5,
    "2": 1.8,
    "3": null,
    "4": 3.2
  },
  "unreachable_count": 1
}
```

## Control Center Integration

### GET /dashboard-status
Get comprehensive dashboard status for control center.

**Response:**
```json
{
  "success": true,
  "system": {
    "status": "healthy",
    "gpu_available": true,
    "uptime_ms": 3600000,
    "issues": []
  },
  "analytics": {
    "clustering": {
      "active_tasks": 1,
      "completed_tasks": 5,
      "total_tasks": 6
    },
    "anomaly_detection": {
      "enabled": true,
      "total_anomalies": 3,
      "critical": 0,
      "high": 1,
      "medium": 1,
      "low": 1
    }
  },
  "last_updated": 1703721600
}
```

### GET /health-check
Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "timestamp": 1703721600,
  "service": "analytics"
}
```

## Feature Management

### GET /feature-flags
Get current feature flag configuration.

**Response:**
```json
{
  "success": true,
  "flags": {
    "gpu_clustering": true,
    "gpu_anomaly_detection": true,
    "real_time_insights": true,
    "advanced_visualizations": true,
    "performance_monitoring": true,
    "stress_majorization": false,
    "semantic_constraints": false,
    "sssp_integration": true
  },
  "description": {
    "gpu_clustering": "Enable GPU-accelerated clustering algorithms",
    "gpu_anomaly_detection": "Enable GPU-accelerated anomaly detection",
    "real_time_insights": "Enable real-time AI insights generation",
    "advanced_visualizations": "Enable advanced visualization features",
    "performance_monitoring": "Enable detailed performance monitoring",
    "stress_majorization": "Enable stress majorization layout algorithm",
    "semantic_constraints": "Enable semantic constraint processing",
    "sssp_integration": "Enable single-source shortest path integration"
  }
}
```

### POST /feature-flags
Update feature flag configuration.

**Request:**
```json
{
  "gpu_clustering": true,
  "gpu_anomaly_detection": true,
  "real_time_insights": true,
  "advanced_visualizations": true,
  "performance_monitoring": true,
  "stress_majorization": true,
  "semantic_constraints": true,
  "sssp_integration": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Feature flags updated successfully",
  "flags": {
    "gpu_clustering": true,
    "gpu_anomaly_detection": true,
    "real_time_insights": true,
    "advanced_visualizations": true,
    "performance_monitoring": true,
    "stress_majorization": true,
    "semantic_constraints": true,
    "sssp_integration": true
  }
}
```

## WebSocket Real-Time Updates

### WebSocket /ws
Connect to real-time GPU analytics updates.

**Connection URL:**
```
ws://localhost:8080/api/analytics/ws
```

**Message Types:**

#### Connected
```json
{
  "messageType": "connected",
  "data": {
    "clientId": "client-123",
    "capabilities": {
      "gpuMetrics": true,
      "clusteringProgress": true,
      "anomalyAlerts": true,
      "insightsUpdates": true,
      "realTimeUpdates": true
    },
    "defaultUpdateInterval": 5000
  },
  "timestamp": 1703721600000
}
```

#### GPU Metrics Update
```json
{
  "messageType": "gpuMetricsUpdate", 
  "data": {
    "gpu_utilization": 75.0,
    "memory_usage_percent": 45.0,
    "temperature": 68.0,
    "power_draw": 120.0,
    "active_kernels": 3,
    "compute_nodes": 50000,
    "compute_edges": 125000,
    "fps": 60.0,
    "frame_time_ms": 16.67
  },
  "timestamp": 1703721600000
}
```

#### Clustering Progress
```json
{
  "messageType": "clusteringProgress",
  "data": {
    "task_id": "task-123",
    "method": "kmeans",
    "progress": 0.65,
    "status": "running",
    "clusters_found": null,
    "estimated_completion": 1703721630000
  },
  "timestamp": 1703721600000
}
```

#### Anomaly Alert
```json
{
  "messageType": "anomalyAlert",
  "data": {
    "anomaly_id": "anom-456",
    "node_id": "42",
    "severity": "critical",
    "score": 0.95,
    "detection_method": "isolation_forest",
    "description": "Node shows critical deviation from expected behavior",
    "requires_action": true
  },
  "timestamp": 1703721600000
}
```

#### Insights Update
```json
{
  "messageType": "insightsUpdate",
  "data": {
    "insights": [
      "Performance warning: Frame rate below 30 FPS",
      "High alert: 3 high-severity anomalies"
    ],
    "urgency_level": "high",
    "requires_action": true,
    "performance_warnings": ["Frame rate below 30 FPS"],
    "recommendations": ["Consider reducing graph complexity"]
  },
  "timestamp": 1703721600000
}
```

## Error Responses

All endpoints return consistent error formats:

```json
{
  "success": false,
  "error": "Error description",
  "code": "ERROR_CODE",
  "timestamp": 1703721600000
}
```

**Common Error Codes:**
- `GPU_NOT_AVAILABLE` - GPU acceleration not available
- `TASK_NOT_FOUND` - Clustering task not found
- `INVALID_PARAMETERS` - Invalid request parameters
- `SERVICE_UNAVAILABLE` - Backend service unavailable
- `COMPUTATION_FAILED` - GPU computation failed

## Performance Expectations

- **GPU Clustering**: 10-50x speedup over CPU
- **Anomaly Detection**: Real-time processing for up to 100K nodes
- **WebSocket Updates**: 5-second default intervals, configurable
- **API Response Times**: < 100ms for status endpoints, variable for compute operations
- **Memory Usage**: Linear scaling with node/edge count

## Client Integration Examples

### JavaScript/TypeScript
```javascript
// REST API usage
const response = await fetch('/api/analytics/gpu-metrics');
const metrics = await response.json();
console.log('GPU utilization:', metrics.utilization);

// WebSocket connection
const ws = new WebSocket('ws://localhost:8080/api/analytics/ws');
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  switch(message.messageType) {
    case 'gpuMetricsUpdate':
      updateDashboard(message.data);
      break;
    case 'anomalyAlert':
      showAlert(message.data);
      break;
  }
};
```

### Python
```python
import requests
import websocket
import json

# REST API usage
response = requests.get('http://localhost:8080/api/analytics/gpu-status')
status = response.json()
print(f"GPU Status: {status['status']}")

# WebSocket connection
def on_message(ws, message):
    data = json.loads(message)
    if data['messageType'] == 'anomalyAlert':
        print(f"Anomaly detected: {data['data']['description']}")

ws = websocket.WebSocketApp("ws://localhost:8080/api/analytics/ws",
                           on_message=on_message)
ws.run_forever()
```

This comprehensive API provides full client integration capabilities for GPU-accelerated analytics with real-time updates, progress tracking, and detailed monitoring suitable for both dashboard applications and programmatic access.