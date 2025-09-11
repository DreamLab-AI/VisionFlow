# VisionFlow API Reference

Comprehensive API documentation for VisionFlow REST and WebSocket endpoints.

## Base URL

```
http://localhost:8080/api
ws://localhost:8080/wss
```

## Authentication

Currently, the API uses optional Nostr authentication for user-specific settings.

```http
Authorization: Bearer <nostr-token>
```

## REST API Endpoints

### Graph Management

#### Get Graph Data
```http
GET /api/graph
```

Returns the current graph structure including nodes and edges.

**Response:**
```json
{
  "nodes": [
    {
      "id": 1,
      "label": "Node 1",
      "x": 100.0,
      "y": 200.0,
      "z": 0.0,
      "properties": {}
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "weight": 1.0
    }
  ]
}
```

#### Update Graph
```http
POST /api/graph
Content-Type: application/json
```

Updates the graph structure.

**Request Body:**
```json
{
  "nodes": [...],
  "edges": [...]
}
```

### AI Agent Management

#### Spawn Agent Swarm
```http
POST /api/agents/spawn
Content-Type: application/json
```

Creates a new AI agent swarm with specified configuration.

**Request Body:**
```json
{
  "topology": "mesh",
  "maxAgents": 8,
  "strategy": "balanced",
  "agentTypes": ["coordinator", "researcher", "coder"]
}
```

**Response:**
```json
{
  "swarmId": "swarm-123",
  "status": "initialised",
  "agents": [...]
}
```

#### Get Swarm Status
```http
GET /api/agents/swarm/{swarmId}/status
```

Returns current status of an agent swarm.

**Response:**
```json
{
  "swarmId": "swarm-123",
  "topology": "mesh",
  "activeAgents": 8,
  "tasks": {
    "pending": 3,
    "inProgress": 2,
    "completed": 15
  },
  "performance": {
    "avgResponseTime": 125,
    "successRate": 0.95
  }
}
```

#### Orchestrate Task
```http
POST /api/agents/task
Content-Type: application/json
```

Assigns a task to the agent swarm.

**Request Body:**
```json
{
  "swarmId": "swarm-123",
  "task": "Analyse codebase and identify performance bottlenecks",
  "priority": "high",
  "strategy": "parallel"
}
```

### Analytics

#### Run Clustering
```http
POST /api/analytics/clustering
Content-Type: application/json
```

Executes clustering analysis on the graph.

**Request Body:**
```json
{
  "algorithm": "kmeans",
  "parameters": {
    "k": 5,
    "maxIterations": 100
  }
}
```

**Response:**
```json
{
  "clusters": [
    {
      "id": 0,
      "nodes": [1, 2, 3],
      "centroid": [100.0, 200.0, 0.0]
    }
  ],
  "metrics": {
    "silhouetteScore": 0.75,
    "inertia": 1234.5
  }
}
```

#### Detect Anomalies
```http
POST /api/analytics/anomaly
Content-Type: application/json
```

Runs anomaly detection on graph data.

**Request Body:**
```json
{
  "algorithm": "lof",
  "sensitivity": 0.8
}
```

**Response:**
```json
{
  "anomalies": [
    {
      "nodeId": 42,
      "score": 0.95,
      "type": "structural",
      "explanation": "Node has unusual connectivity pattern"
    }
  ]
}
```

#### Run Community Detection
```http
POST /api/analytics/community
Content-Type: application/json
```

Identifies communities within the graph.

**Request Body:**
```json
{
  "algorithm": "louvain",
  "resolution": 1.0
}
```

### GPU Status

#### Get GPU Status
```http
GET /api/analytics/gpu-status
```

Returns current GPU compute status and metrics.

**Response:**
```json
{
  "isInitialised": true,
  "deviceName": "NVIDIA RTX 4090",
  "memoryUsed": 2048,
  "memoryTotal": 24576,
  "utilisation": 45,
  "temperature": 65,
  "activeActors": {
    "forceCompute": true,
    "clustering": true,
    "anomalyDetection": false,
    "stressMajorisation": false
  }
}
```

### Physics Simulation

#### Update Simulation Parameters
```http
POST /api/physics/params
Content-Type: application/json
```

Updates physics simulation parameters.

**Request Body:**
```json
{
  "attractionK": 0.1,
  "repelK": 100000,
  "centerGravityK": 0.05,
  "damping": 0.95,
  "dt": 0.016,
  "maxVelocity": 1000
}
```

#### Toggle Physics
```http
POST /api/physics/toggle
```

Enables or disables physics simulation.

### Settings

#### Get User Settings
```http
GET /api/settings
```

Returns current user settings.

**Response:**
```json
{
  "physics": {
    "enabled": true,
    "mode": "advanced"
  },
  "visualisation": {
    "nodeSize": 5,
    "edgeOpacity": 0.5,
    "showLabels": true
  },
  "performance": {
    "targetFPS": 60,
    "maxNodes": 1000
  }
}
```

#### Update Settings
```http
PUT /api/settings
Content-Type: application/json
```

Updates user settings.

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/wss');
```

### Binary Protocol Format

VisionFlow uses a custom binary protocol for position updates:

```
Header (4 bytes):
  - Message Type (1 byte)
  - Flags (1 byte)
  - Node Count (2 bytes, little-endian)

Per Node (28 bytes):
  - Node ID (4 bytes, little-endian)
  - Position X (4 bytes, float32)
  - Position Y (4 bytes, float32)
  - Position Z (4 bytes, float32)
  - Velocity X (4 bytes, float32)
  - Velocity Y (4 bytes, float32)
  - Velocity Z (4 bytes, float32)
```

### Message Types

#### Position Update (0x01)
Sent every frame with node positions and velocities.

#### Graph Update (0x02)
Sent when graph structure changes.

#### Agent Update (0x03)
Sent when agent status changes.

#### Analytics Result (0x04)
Sent when analytics computation completes.

### Client Messages

#### Subscribe to Updates
```json
{
  "type": "subscribe",
  "channels": ["positions", "agents", "analytics"]
}
```

#### Request Computation
```json
{
  "type": "compute",
  "operation": "clustering",
  "parameters": {...}
}
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `201 Created` - Resource created
- `400 Bad Request` - Invalid request
- `401 Unauthorised` - Authentication required
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

Error responses include details:

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Parameter 'k' must be positive integer",
    "details": {
      "parameter": "k",
      "value": -1,
      "expected": "positive integer"
    }
  }
}
```

## Rate Limiting

API requests are rate-limited to:
- 100 requests per minute for unauthenticated users
- 1000 requests per minute for authenticated users
- WebSocket messages: 60 per second

## Performance Considerations

- Binary protocol reduces bandwidth by 85% vs JSON
- Position updates batched at 60 FPS
- GPU computations queued and processed asynchronously
- Large graphs (>1000 nodes) automatically paginated

## SDK Examples

### JavaScript/TypeScript

```typescript
import { VisionFlowClient } from '@visionflow/client';

const client = new VisionFlowClient({
  baseUrl: 'http://localhost:8080',
  wsUrl: 'ws://localhost:8080/wss'
});

// Spawn agent swarm
const swarm = await client.agents.spawn({
  topology: 'mesh',
  maxAgents: 8
});

// Subscribe to position updates
client.on('positions', (positions) => {
  positions.forEach(node => {
    console.log(`Node ${node.id}: ${node.x}, ${node.y}, ${node.z}`);
  });
});

// Run clustering
const result = await client.analytics.cluster({
  algorithm: 'kmeans',
  k: 5
});
```

### Rust

```rust
use visionflow_client::Client;

#[tokio::main]
async fn main() {
    let client = Client::new("http://localhost:8080");
    
    // Get graph data
    let graph = client.get_graph().await?;
    
    // Run anomaly detection
    let anomalies = client.detect_anomalies(
        Algorithm::LOF,
        Sensitivity::High
    ).await?;
}
```

## Versioning

The API follows semantic versioning. Current version: `v1.0.0`

Version information available at:
```http
GET /api/version
```

## Support

For API support and questions:
- GitHub Issues: [Report issues](https://github.com/visionflow/issues)
- Email: api@visionflow.ai

---

*Last updated: January 2025*