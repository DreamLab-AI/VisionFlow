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
Spawns a new agent within an existing swarm.

```http
POST /api/bots/spawn-agent
Content-Type: application/json

{
  "agentType": "coder",
  "swarmId": "swarm_456",
  "capabilities": ["python", "rust"],
  "config": {
    "model": "gpt-4",
    "temperature": 0.7
  }
}
```

**Response:**
```json
{
  "agentId": "agent_999",
  "swarmId": "swarm_456",
  "status": "spawning",
  "estimatedReadyTime": "2025-01-22T10:00:30Z"
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