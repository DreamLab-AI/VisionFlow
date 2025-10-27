# REST API Reference

## Overview

The VisionFlow REST API provides HTTP endpoints for managing agents, graphs, settings, and system operations. All endpoints return JSON responses and follow RESTful conventions.

**Base URL**: `http://localhost:3001/api`

## Authentication

Most endpoints require authentication via JWT tokens:

```http
Authorization: Bearer <jwt_token>
X-Nostr-Pubkey: <public_key>
```

## Core Endpoints

### Agent Management

#### Get Agent Data
Retrieves the complete list of agents with their metadata and current state from the real agent swarm via MCP protocol.

```http
GET /api/bots/data
```

**Response:**
```json
{
  "nodes": [
    {
      "id": "agent_1757967065850_dv2zg7",
      "name": "Coder Agent",
      "type": "agent",
      "position": { "x": 10.5, "y": 20.3, "z": 0 },
      "metadata": {
        "swarmId": "swarm_1757880683494_yl81sece5",
        "capabilities": ["code", "review", "rust", "python"],
        "status": "active",
        "model": "claude-3-opus",
        "temperature": 0.7,
        "lastActive": "2025-01-22T10:00:00Z",
        "mcpConnected": true
      }
    }
  ],
  "edges": [
    {
      "source": "agent_1757967065850_dv2zg7",
      "target": "agent_1757967065851_abc123",
      "weight": 0.8,
      "type": "coordination"
    }
  ],
  "mcpStatus": {
    "connected": true,
    "host": "multi-agent-container",
    "port": 9500,
    "activeConnections": 3
  }
}
```

#### Get Agent Status
Returns detailed telemetry and health information for all agents.

```http
GET /api/bots/status
```

**Response:**
```json
{
  "agents": [
    {
      "id": "agent_123",
      "status": "active",
      "health": {
        "cpu": 25.5,
        "memory": 512,
        "uptime": 3600
      },
      "workload": {
        "tasksCompleted": 15,
        "tasksInProgress": 2,
        "taskQueue": 5
      }
    }
  ],
  "timestamp": "2025-01-22T10:00:00Z"
}
```

#### Submit Task
Submits a new task to the agent swarm for processing via MCP task orchestration protocol.

```http
POST /api/bots/submit-task
Content-Type: application/json

{
  "task": "Analyze the authentication module for security vulnerabilities",
  "priority": "high",
  "strategy": "adaptive",
  "swarmId": "swarm_1757880683494_yl81sece5",
  "timeout": 300,
  "requiredCapabilities": ["code", "security", "review"]
}
```

**Response:**
```json
{
  "taskId": "task_1757967065850_abc123",
  "status": "queued",
  "estimatedDuration": 300,
  "assignedAgents": [
    "agent_1757967065850_dv2zg7",
    "agent_1757967065851_def456"
  ],
  "mcpTaskId": "mcp_task_1757967065850_xyz789",
  "swarmId": "swarm_1757880683494_yl81sece5",
  "queuePosition": 2,
  "orchestrationMethod": "consensus",
  "agentRequirements": {
    "minAgents": 2,
    "maxAgents": 5,
    "consensusThreshold": 0.7
  }
}
```

#### Get Task Status
Retrieves the current status and progress of a submitted task.

```http
GET /api/bots/task-status/{taskId}
```

**Response:**
```json
{
  "taskId": "task_789",
  "status": "in_progress",
  "progress": 65,
  "startTime": "2025-01-22T10:00:00Z",
  "updates": [
    {
      "timestamp": "2025-01-22T10:05:00Z",
      "agent": "agent_123",
      "message": "Completed initial code analysis"
    }
  ],
  "results": null
}
```

#### Initialize Swarm
Creates and initializes a new agent swarm with specified configuration.

```http
POST /api/bots/initialize-swarm
Content-Type: application/json

{
  "topology": "mesh",
  "maxAgents": 5,
  "strategy": "balanced",
  "capabilities": ["code", "test", "review"]
}
```

**Response:**
```json
{
  "swarmId": "swarm_001",
  "status": "initializing",
  "agents": [
    { "id": "agent_001", "type": "coordinator" },
    { "id": "agent_002", "type": "worker" }
  ]
}
```

#### Spawn Agent
Creates a new agent via MCP TCP protocol with real agent execution.

```http
POST /api/bots/spawn
Content-Type: application/json

{
  "type": "coder",
  "instructions": "Implement REST API endpoints with error handling",
  "capabilities": ["python", "rust", "javascript"],
  "config": {
    "model": "claude-3-opus",
    "temperature": 0.7,
    "maxTokens": 4096
  },
  "swarmId": "swarm_1757880683494_yl81sece5"
}
```

**Response:**
```json
{
  "agentId": "agent_1757967065850_xyz789",
  "status": "active",
  "mcpTaskId": "mcp_task_1757967065850_abc123",
  "swarmId": "swarm_1757880683494_yl81sece5",
  "capabilities": ["python", "rust", "javascript"],
  "tcpConnection": {
    "host": "multi-agent-container",
    "port": 9500,
    "connected": true
  },
  "createdAt": "2025-01-22T10:00:00Z"
}
```

#### List Agents
Retrieves all currently running agents from MCP.

```http
GET /api/bots/list
```

**Response:**
```json
{
  "agents": [
    {
      "id": "agent_1757967065850_abc123",
      "type": "coder",
      "status": "active",
      "uptime": 3600,
      "tasksCompleted": 15,
      "currentTask": "Implementing authentication middleware",
      "mcpConnected": true
    },
    {
      "id": "agent_1757967065851_def456",
      "type": "reviewer",
      "status": "idle",
      "uptime": 2400,
      "tasksCompleted": 8,
      "currentTask": null,
      "mcpConnected": true
    }
  ],
  "totalAgents": 2,
  "activeAgents": 1,
  "mcpStatus": {
    "connected": true,
    "host": "multi-agent-container",
    "port": 9500,
    "activeConnections": 2
  }
}
```

#### Orchestrate Tasks
Distributes tasks across multiple agents using real MCP orchestration.

```http
POST /api/bots/orchestrate
Content-Type: application/json

{
  "tasks": [
    {
      "description": "Implement user authentication",
      "priority": "high",
      "requiredCapabilities": ["security", "backend"]
    },
    {
      "description": "Write comprehensive tests",
      "priority": "medium",
      "requiredCapabilities": ["testing", "python"]
    }
  ],
  "strategy": "consensus",
  "swarmId": "swarm_1757880683494_yl81sece5"
}
```

**Response:**
```json
{
  "orchestrationId": "orch_1757967065850_xyz789",
  "tasksDistributed": 2,
  "assignments": [
    {
      "taskId": "task_1757967065850_abc123",
      "agentId": "agent_1757967065850_def456",
      "estimatedDuration": 1800
    },
    {
      "taskId": "task_1757967065851_ghi789",
      "agentId": "agent_1757967065851_jkl012",
      "estimatedDuration": 900
    }
  ],
  "mcpOrchestrationActive": true,
  "consensusRequired": true
}
```

### Graph Management

#### Update Graph Data
Updates node positions and graph structure.

```http
POST /api/graph/update
Content-Type: application/json

{
  "nodes": [
    {
      "id": "node_1",
      "position": { "x": 100, "y": 200, "z": 0 },
      "metadata": { "label": "Updated Node" }
    }
  ],
  "edges": [
    {
      "source": "node_1",
      "target": "node_2",
      "weight": 0.5
    }
  ]
}
```

### GPU-Accelerated Analytics

#### GPU Clustering Analysis
Executes real GPU-accelerated clustering using K-means, Louvain, and DBSCAN algorithms with CUDA kernels.

```http
GET /api/analytics/clustering
```

**Response:**
```json
{
  "kmeans": {
    "clusters": [
      {
        "id": 0,
        "centroid": [125.4, 89.2, 156.7],
        "nodeCount": 23,
        "inertia": 0.156,
        "nodes": ["node_1", "node_5", "node_12"]
      },
      {
        "id": 1,
        "centroid": [220.1, 145.8, 78.3],
        "nodeCount": 18,
        "inertia": 0.142,
        "nodes": ["node_2", "node_7", "node_9"]
      }
    ],
    "totalInertia": 0.298,
    "iterations": 12,
    "converged": true,
    "computationTimeMs": 89,
    "gpuAccelerated": true
  },
  "louvain": {
    "communities": [
      {
        "id": 0,
        "nodes": ["node_1", "node_3", "node_8", "node_15"],
        "modularity": 0.234,
        "size": 4
      },
      {
        "id": 1,
        "nodes": ["node_2", "node_6", "node_11"],
        "modularity": 0.189,
        "size": 3
      }
    ],
    "totalModularity": 0.847,
    "iterations": 23,
    "computationTimeMs": 156,
    "gpuAccelerated": true
  },
  "dbscan": {
    "clusters": [
      {
        "id": 0,
        "corePoints": ["node_1", "node_4", "node_7"],
        "borderPoints": ["node_2", "node_5"],
        "totalPoints": 5,
        "density": 0.83
      }
    ],
    "outliers": ["node_20", "node_35"],
    "epsilon": 0.5,
    "minSamples": 3,
    "computationTimeMs": 67,
    "gpuAccelerated": true
  },
  "gpuStatus": {
    "device": "NVIDIA GeForce RTX 4090",
    "memoryUsed": "2.8 GB",
    "utilization": 92,
    "kernelExecutions": 234
  },
  "lastUpdated": "2025-01-22T10:15:30Z"
}
```

#### GPU Status
Returns real-time GPU utilization and performance metrics.

```http
GET /api/gpu/status
```

**Response:**
```json
{
  "gpus": [
    {
      "id": 0,
      "name": "NVIDIA GeForce RTX 4090",
      "utilization": 87,
      "memory": {
        "total": "24 GB",
        "used": "18.2 GB",
        "free": "5.8 GB",
        "utilization": 76
      },
      "temperature": 72,
      "powerDraw": 385,
      "fanSpeed": 68,
      "clockSpeeds": {
        "core": 2520,
        "memory": 10500
      }
    }
  ],
  "clustering": {
    "active": true,
    "algorithm": "louvain",
    "progress": 0.89,
    "eta": 45
  },
  "kernelStats": {
    "totalExecutions": 1547,
    "avgExecutionTime": 12.4,
    "failedExecutions": 0
  },
  "lastUpdated": "2025-01-22T10:15:30Z"
}
```

#### GPU Anomaly Detection
Executes real anomaly detection using LOF, Z-Score, and Isolation Forest algorithms.

```http
GET /api/gpu/anomalies
```

**Response:**
```json
{
  "lof": {
    "anomalies": [
      {
        "nodeId": "node_47",
        "score": 2.34,
        "position": [245.7, 189.3, 67.1],
        "neighbours": 8,
        "distance": 23.7
      },
      {
        "nodeId": "node_82",
        "score": 1.89,
        "position": [78.2, 345.9, 123.4],
        "neighbours": 12,
        "distance": 19.2
      }
    ],
    "threshold": 1.5,
    "totalAnomalies": 2,
    "computationTimeMs": 87
  },
  "zscore": {
    "anomalies": [
      {
        "nodeId": "node_23",
        "score": 3.45,
        "feature": "degree_centrality",
        "value": 0.89,
        "mean": 0.23,
        "stddev": 0.19
      }
    ],
    "threshold": 3.0,
    "totalAnomalies": 1,
    "computationTimeMs": 34
  },
  "isolationForest": {
    "anomalies": [
      {
        "nodeId": "node_156",
        "score": -0.67,
        "pathLength": 4.2,
        "features": {
          "betweenness": 0.78,
          "closeness": 0.45,
          "eigenvector": 0.23
        }
      }
    ],
    "threshold": -0.5,
    "treeCount": 100,
    "totalAnomalies": 1,
    "computationTimeMs": 156
  },
  "gpuAccelerated": true,
  "totalComputationTimeMs": 277,
  "lastUpdated": "2025-01-22T10:15:30Z"
}
```

### Voice Processing

#### Speech-to-Text Transcription
Transcribes audio using real Whisper models with GPU acceleration.

```http
POST /api/voice/transcribe
Content-Type: multipart/form-data

Form data:
- audio: [audio file - WAV, MP3, or OGG]
- model: "whisper-large-v3" (optional)
- language: "en" (optional)
```

**Response:**
```json
{
  "text": "Create a new REST API endpoint for user authentication with JWT tokens",
  "confidence": 0.94,
  "duration": 4.7,
  "language": "en",
  "model": "whisper-large-v3",
  "processingTimeMs": 890,
  "gpuAccelerated": true,
  "segments": [
    {
      "start": 0.0,
      "end": 2.3,
      "text": "Create a new REST API endpoint",
      "confidence": 0.96
    },
    {
      "start": 2.3,
      "end": 4.7,
      "text": "for user authentication with JWT tokens",
      "confidence": 0.92
    }
  ]
}
```

#### Text-to-Speech Synthesis
Synthesizes speech using real Kokoro TTS models.

```http
POST /api/voice/synthesize
Content-Type: application/json

{
  "text": "The clustering analysis has completed successfully with 7 communities detected.",
  "voice": "nova",
  "speed": 1.0,
  "format": "wav"
}
```

**Response:**
```json
{
  "audioUrl": "/api/voice/audio/synthesis_1757967065850_abc123.wav",
  "duration": 3.8,
  "sampleRate": 22050,
  "format": "wav",
  "voice": "nova",
  "processingTimeMs": 1240,
  "fileSize": 167834,
  "checksum": "sha256:a7b9c3d4e5f6789..."
}
```

#### Voice Command Execution
Processes voice commands and executes them via agent orchestration.

```http
POST /api/voice/execute
Content-Type: multipart/form-data

Form data:
- audio: [audio file]
- executeImmediately: true (optional)
```

**Response:**
```json
{
  "transcription": {
    "text": "Run clustering analysis on the current graph data",
    "confidence": 0.91
  },
  "interpretation": {
    "action": "clustering",
    "parameters": {
      "algorithm": "louvain",
      "resolution": 1.0
    },
    "confidence": 0.87
  },
  "execution": {
    "taskId": "voice_task_1757967065850_xyz789",
    "agentId": "agent_1757967065850_abc123",
    "status": "executing",
    "estimatedDuration": 180
  },
  "response": {
    "text": "Starting Louvain clustering analysis on the current graph data. This should take approximately 3 minutes.",
    "audioUrl": "/api/voice/audio/response_1757967065850_def456.wav"
  }
}
```

#### Get Graph State
Retrieves the current graph state including physics simulation data.

```http
GET /api/graph/state
```

**Response:**
```json
{
  "nodes": [...],
  "edges": [...],
  "physics": {
    "enabled": true,
    "temperature": 0.01,
    "iterations": 50
  },
  "bounds": {
    "min": { "x": -500, "y": -500, "z": -500 },
    "max": { "x": 500, "y": 500, "z": 500 }
  }
}
```

### Settings Management

#### Get Settings
Retrieves current application settings.

```http
GET /api/settings
```

#### Update Settings
Updates application settings (requires authentication).

```http
POST /api/settings
Content-Type: application/json

{
  "visualisation": {
    "rendering": {
      "enable_shadows": true,
      "shadow_map_size": 2048
    }
  }
}
```

#### Batch Update Settings
Efficiently updates multiple settings at once.

```http
POST /api/settings/batch
Content-Type: application/json

{
  "updates": [
    {
      "path": "visualisation.rendering.enable_shadows",
      "value": true
    },
    {
      "path": "system.network.rate_limit_requests",
      "value": 10000
    }
  ]
}
```

### System Operations

#### Health Check
Basic health check endpoint.

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime": 3600,
  "services": {
    "database": "connected",
    "redis": "connected",
    "mcp": "connected"
  }
}
```

#### MCP Connection Status
Checks the Model Context Protocol connection status.

```http
GET /api/bots/mcp-status
```

**Response:**
```json
{
  "connected": true,
  "host": "multi-agent-container",
  "port": 9500,
  "latency": 5,
  "lastPing": "2025-01-22T10:00:00Z"
}
```

## Timeouts and Error Handling

To ensure system stability and prevent indefinite hangs, the VisionFlow API implements a two-layer timeout strategy.

### 1. HTTP-Level Timeout (Global)

-   **Duration:** 30 seconds
-   **Scope:** Applies to all incoming HTTP requests.
-   **Behavior:** If a request is not fully processed within 30 seconds, the connection is terminated, and the client will receive a `504 Gateway Timeout` response. This is a global safeguard against long-running or stuck processes.

### 2. Actor-Level Timeout (Per-Operation)

-   **Default Duration:** 5 seconds
-   **Scope:** Applies to internal actor communications for specific operations.
-   **Behavior:** When an API handler sends a message to an actor (e.g., `GraphServiceActor`), it waits a maximum of 5 seconds for a response. If the actor fails to respond within this window, the handler will return a `504 Gateway Timeout` with a specific error message related to the operation. This prevents long delays in one part of the system from cascading and affecting the entire application.
-   **Extended Durations:** Certain long-running operations may use extended timeouts (e.g., 10 seconds) where appropriate.
## Error Responses

All error responses follow a consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "priority",
      "reason": "Priority must be one of: low, medium, high, critical"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `RATE_LIMITED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |

## Rate Limiting

Default rate limits per client:
- 1000 requests per 15 minutes
- Burst allowance: 50 requests
- Rate limit headers included in responses:
  - `X-RateLimit-Limit`: Maximum requests
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Pagination

List endpoints support pagination:

```http
GET /api/bots/data?page=2&limit=50
```

**Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 100, max: 1000)

**Response includes:**
```json
{
  "data": [...],
  "pagination": {
    "page": 2,
    "limit": 50,
    "total": 245,
    "totalPages": 5
  }
}
```

## Filtering and Sorting

Many endpoints support filtering and sorting:

```http
GET /api/bots/data?status=active&sort=name:asc
```

**Common parameters:**
- `status`: Filter by status (active, idle, error)
- `type`: Filter by agent type
- `sort`: Sort field and direction (field:asc or field:desc)

## Webhooks

Configure webhooks for real-time notifications:

```http
POST /api/webhooks
Content-Type: application/json

{
  "url": "https://your-server.com/webhook",
  "events": ["task.completed", "agent.error"],
  "secret": "your_webhook_secret"
}
```

## API Versioning

Current version: `v1`

Version is included in the URL path. Future versions:
- `v1` (current): `/api/...`
- `v2` (future): `/api/v2/...`

## Testing

### Example cURL Commands

```bash
# Get agent data
curl -X GET http://localhost:3001/api/bots/data \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Submit task
curl -X POST http://localhost:3001/api/bots/submit-task \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"task": "Test task", "priority": "medium"}'

# Update settings
curl -X POST http://localhost:3001/api/settings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"visualisation": {"rendering": {"enable_shadows": true}}}'
```

## Related Documentation

- [WebSocket API](websocket-api.md)
- [Binary Protocol](binary-protocol.md)
- [MCP Protocol](mcp-protocol.md)
- [Authentication Guide](../../guides/authentication.md)

---

**[← API Overview](index.md)** | **[WebSocket API →](websocket-api.md)**