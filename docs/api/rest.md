# REST API Reference

## Overview

The VisionFlow REST API provides comprehensive endpoints for graph management, AI agent orchestration, system configuration, and service integration. All endpoints follow RESTful conventions and return JSON responses.

## Base URL

```
http://localhost:3001/api
```

## Authentication

### Nostr Authentication
Primary authentication using Nostr protocol (NIP-07).

```http
POST /api/nostr/auth
```

**Request:**
```json
{
  "id": "event_id_hex",
  "pubkey": "user_pubkey_hex",
  "created_at": 1678886400,
  "kind": 22242,
  "tags": [
    ["relay", "wss://relay.nostr.com"],
    ["challenge", "random_challenge"]
  ],
  "content": "VisionFlow Authentication",
  "sig": "signature_hex"
}
```

**Response:**
```json
{
  "user": {
    "pubkey": "user_pubkey",
    "npub": "npub_encoded",
    "isPowerUser": true
  },
  "token": "session_token",
  "expiresAt": 1234567890,
  "features": ["graph", "agents", "xr"]
}
```

## Graph Endpoints

### Get Graph Data
```http
GET /api/graph/data
```

**Query Parameters:**
- `includeMetadata` (boolean) - Include node/edge metadata
- `layout` (string) - Layout algorithm: `force`, `hierarchical`, `circular`

**Response:**
```json
{
  "nodes": [
    {
      "id": 1,
      "label": "Knowledge Node",
      "x": 100.0,
      "y": 200.0,
      "z": 50.0,
      "type": "concept",
      "metadata": {
        "fileSize": "1024",
        "lastModified": "2024-01-01T00:00:00Z"
      }
    }
  ],
  "edges": [
    {
      "id": 1,
      "source": 1,
      "target": 2,
      "weight": 1.0,
      "type": "semantic"
    }
  ],
  "metadata": {
    "nodeCount": 100,
    "edgeCount": 150,
    "lastUpdated": "2024-01-01T00:00:00Z"
  }
}
```

### Get Paginated Graph Data
```http
GET /api/graph/data/paginated
```

**Query Parameters:**
- `page` (number) - Page number (1-based)
- `page_size` (number) - Items per page (default: 100)
- `query` (string) - Search query
- `sort` (string) - Sort field
- `filter` (string) - Filter criteria

**Response:**
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {...},
  "totalPages": 10,
  "currentPage": 1,
  "totalItems": 1000,
  "pageSize": 100
}
```

### Update Graph
```http
POST /api/graph/update
```

**Response:**
```json
{
  "success": true,
  "message": "Graph updated with 5 new files",
  "processedFiles": ["file1.md", "file2.md"]
}
```

### Refresh Graph
```http
POST /api/graph/refresh
```

**Response:**
```json
{
  "success": true,
  "message": "Graph refreshed successfully"
}
```

## Agent/Bots Endpoints

### List Agents
```http
GET /api/bots/data
```

**Response:**
```json
{
  "agents": [
    {
      "id": "agent-001",
      "name": "Research Agent Alpha",
      "type": "researcher",
      "status": "active",
      "health": 0.95,
      "workload": 0.7,
      "capabilities": ["web_search", "document_analysis"],
      "currentTask": "Analyzing research papers",
      "metrics": {
        "tasksCompleted": 42,
        "successRate": 0.95,
        "averageResponseTime": 250,
        "tokensUsed": 15000
      }
    }
  ],
  "multi-agentTopology": "hierarchical",
  "totalAgents": 15,
  "connections": [
    {
      "from": "agent-001",
      "to": "agent-002",
      "strength": 0.8,
      "messageRate": 12.5
    }
  ]
}
```

### Update Agent Data
```http
POST /api/bots/data
```

**Request:**
```json
{
  "agents": [
    {
      "id": "agent-001",
      "status": "idle",
      "metrics": {
        "tokensUsed": 15100
      }
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Agent data updated successfully",
  "updatedAgents": 1
}
```

### Initialize multi-agent
```http
POST /api/bots/initialize-multi-agent
```

**Request:**
```json
{
  "topology": "hierarchical",
  "maxAgents": 10,
  "agentTypes": ["coordinator", "researcher", "coder"],
  "enableNeural": true,
  "customPrompt": "Build a REST API"
}
```

**Response:**
```json
{
  "success": true,
  "multi-agentId": "multi-agent-001",
  "message": "multi-agent initialized successfully",
  "agents": [
    {
      "id": "agent-001",
      "type": "coordinator",
      "status": "initializing"
    }
  ]
}
```

## Settings Endpoints

### Get Settings
```http
GET /api/settings
```

**Response:**
```json
{
  "theme": "dark",
  "physics": {
    "enabled": true,
    "gravity": -9.8,
    "damping": 0.95,
    "iterations": 5,
    "constraints": {
      "separation": 100.0,
      "cohesion": 0.1
    }
  },
  "visualization": {
    "nodeSize": 10,
    "edgeThickness": 2,
    "showLabels": true,
    "hologramEffect": true,
    "renderDistance": 1000
  },
  "xr": {
    "enabled": false,
    "handTracking": true,
    "passthrough": true,
    "spatialAnchors": false
  },
  "websocket": {
    "minUpdateRate": 5,
    "maxUpdateRate": 60,
    "motionThreshold": 0.01,
    "heartbeatInterval": 30000
  },
  "gpu": {
    "enabled": true,
    "memoryLimit": 1024,
    "computeShaders": true
  }
}
```

### Update Settings
```http
POST /api/settings
```

**Request:**
```json
{
  "path": "physics.gravity",
  "value": -12.0
}
```

**Response:**
```json
{
  "success": true,
  "message": "Setting updated successfully",
  "updatedPath": "physics.gravity",
  "newValue": -12.0
}
```

### Reset Settings
```http
POST /api/settings/reset
```

**Response:**
```json
{
  "success": true,
  "message": "Settings reset to default values"
}
```

## Files Endpoints

### Process Files
```http
POST /api/files/process
```

**Response:**
```json
{
  "status": "success",
  "processedFiles": [
    {
      "fileName": "knowledge.md",
      "size": 1024,
      "processed": true
    }
  ],
  "totalSize": 10240,
  "count": 10
}
```

### Get File Content
```http
GET /api/files/get_content/{filename}
```

**Response:** Raw file content (text/markdown)

### Refresh Graph from Files
```http
POST /api/files/refresh_graph
```

**Response:**
```json
{
  "status": "success",
  "message": "Graph refreshed successfully"
}
```

### Update Graph from Files
```http
POST /api/files/update_graph
```

**Response:**
```json
{
  "status": "success",
  "message": "Graph updated successfully"
}
```

## Quest 3 / XR Endpoints

### Quest 3 Status
```http
GET /api/quest3/status
```

**Response:**
```json
{
  "connected": true,
  "device": {
    "type": "Meta Quest 3",
    "browser": "Wolvic",
    "capabilities": ["handTracking", "passthrough", "spatialAnchors"],
    "webxrSupported": true
  },
  "session": {
    "active": true,
    "mode": "immersive-ar",
    "referenceSpace": "local-floor",
    "features": ["hand-tracking", "layers"]
  }
}
```

### Initialize XR Session
```http
POST /api/quest3/init
```

**Request:**
```json
{
  "mode": "immersive-ar",
  "requiredFeatures": ["hand-tracking"],
  "optionalFeatures": ["layers", "anchors"]
}
```

**Response:**
```json
{
  "success": true,
  "sessionId": "xr-session-001",
  "supportedFeatures": ["hand-tracking", "layers"]
}
```

## Visualization Endpoints

### Get Visualization Config
```http
GET /api/visualisation/config
```

**Response:**
```json
{
  "renderer": {
    "antialias": true,
    "pixelRatio": 2,
    "shadowMap": true,
    "toneMapping": "ACESFilmic"
  },
  "camera": {
    "fov": 75,
    "near": 0.1,
    "far": 10000,
    "position": [0, 100, 500]
  },
  "effects": {
    "bloom": true,
    "outline": true,
    "hologram": true,
    "particles": false,
    "postProcessing": true
  }
}
```

### Update Camera Position
```http
POST /api/visualisation/camera
```

**Request:**
```json
{
  "position": [100, 200, 300],
  "target": [0, 0, 0],
  "fov": 60
}
```

**Response:**
```json
{
  "success": true,
  "message": "Camera position updated"
}
```

## Analytics Endpoints

### System Analytics
```http
GET /api/analytics/system
```

**Response:**
```json
{
  "performance": {
    "fps": 60,
    "frameTime": 16.67,
    "drawCalls": 150,
    "memoryUsage": 512
  },
  "gpu": {
    "utilization": 75,
    "memory": 2048,
    "temperature": 65,
    "vendor": "NVIDIA"
  },
  "network": {
    "websocketClients": 5,
    "bandwidth": 1024,
    "latency": 15,
    "messagesPerSecond": 120
  }
}
```

### Graph Analytics
```http
GET /api/analytics/graph
```

**Response:**
```json
{
  "topology": {
    "density": 0.05,
    "clustering": 0.7,
    "diameter": 6,
    "avgPathLength": 3.2
  },
  "centrality": {
    "mostCentral": ["node_1", "node_2"],
    "bridges": ["edge_5", "edge_7"],
    "communities": 5
  },
  "growth": {
    "nodesAdded": 50,
    "edgesAdded": 125,
    "timeframe": "24h"
  }
}
```

## Health & Status

### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "timestamp": "2024-01-01T00:00:00Z",
  "services": {
    "graph": "operational",
    "agents": "operational",
    "gpu": "operational",
    "mcp": "connected",
    "websockets": "operational"
  },
  "metrics": {
    "memoryUsage": "512MB",
    "cpuUsage": "45%",
    "activeConnections": 5
  }
}
```

### MCP Health
```http
GET /api/mcp/health
```

**Response:**
```json
{
  "connected": true,
  "claudeFlow": {
    "url": "ws://localhost:3002",
    "status": "connected",
    "latency": 5,
    "lastHeartbeat": "2024-01-01T00:00:00Z"
  },
  "tools": {
    "available": 50,
    "active": 3,
    "categories": ["multi-agent", "neural", "memory"]
  },
  "sessions": {
    "active": 2,
    "total": 15
  }
}
```

## External Service Integrations

### RAGFlow Query
```http
POST /api/ragflow/query
```

**Request:**
```json
{
  "query": "What is the architecture?",
  "context": "technical",
  "maxResults": 5,
  "sessionId": "session-001"
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "content": "VisionFlow uses a modular architecture...",
      "score": 0.95,
      "source": "architecture.md",
      "metadata": {
        "pageNumber": 1,
        "section": "Overview"
      }
    }
  ],
  "totalResults": 12,
  "processingTime": 250
}
```

### Perplexity Search
```http
POST /api/perplexity/search
```

**Request:**
```json
{
  "query": "Latest AI developments",
  "sources": ["academic", "news"],
  "limit": 10
}
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "title": "Recent AI Breakthrough",
      "content": "Summary of recent developments...",
      "url": "https://example.com/article",
      "source": "academic",
      "date": "2024-01-01",
      "relevance": 0.95
    }
  ],
  "totalResults": 45,
  "query": "Latest AI developments"
}
```

### GitHub Integration
```http
GET /api/github/repos/{owner}/{repo}
```

**Response:**
```json
{
  "repository": {
    "name": "visionflow",
    "fullName": "owner/visionflow",
    "stars": 100,
    "forks": 20,
    "issues": 5,
    "language": "Rust",
    "size": 10240
  },
  "pullRequests": [
    {
      "number": 1,
      "title": "Add new feature",
      "state": "open",
      "author": "developer",
      "createdAt": "2024-01-01T00:00:00Z"
    }
  ],
  "commits": [
    {
      "sha": "abc123",
      "message": "Fix bug in binary protocol",
      "author": "developer",
      "date": "2024-01-01T00:00:00Z"
    }
  ]
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was invalid",
    "details": {
      "field": "agentId",
      "reason": "Required field missing",
      "validValues": ["agent-001", "agent-002"]
    },
    "requestId": "req-123"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `UNAUTHORIZED` | Authentication required | 401 |
| `FORBIDDEN` | Insufficient permissions | 403 |
| `NOT_FOUND` | Resource not found | 404 |
| `INVALID_REQUEST` | Invalid request parameters | 400 |
| `RATE_LIMITED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Server error | 500 |
| `SERVICE_UNAVAILABLE` | Service temporarily unavailable | 503 |
| `AGENT_NOT_FOUND` | Specified agent does not exist | 404 |
| `multi-agent_ERROR` | multi-agent operation failed | 500 |
| `GRAPH_ERROR` | Graph operation failed | 500 |
| `GPU_ERROR` | GPU computation error | 500 |

## Rate Limiting

| Endpoint Category | Rate Limit | Window | Burst |
|------------------|------------|--------|-------|
| Graph Operations | 100/min | 1 minute | 20 requests |
| Agent Operations | 50/min | 1 minute | 10 requests |
| File Operations | 10/min | 1 minute | 5 requests |
| Analytics | 200/min | 1 minute | 50 requests |
| Settings | 100/min | 1 minute | 20 requests |
| Health Checks | 600/min | 1 minute | 100 requests |

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1609459200
X-RateLimit-Window: 60
```

## Response Formats

### Success Response Template
```json
{
  "success": true,
  "data": { /* endpoint-specific data */ },
  "metadata": {
    "timestamp": "2024-01-01T00:00:00Z",
    "requestId": "req-123",
    "processingTime": 250
  }
}
```

### Pagination Response Template
```json
{
  "data": [ /* array of items */ ],
  "pagination": {
    "currentPage": 1,
    "totalPages": 10,
    "totalItems": 1000,
    "pageSize": 100,
    "hasNext": true,
    "hasPrevious": false
  }
}
```

## Request/Response Headers

### Common Request Headers
```http
Content-Type: application/json
Accept: application/json
User-Agent: VisionFlow-Client/1.0.0
X-Request-ID: req-123
X-Client-Version: 1.0.0
```

### Common Response Headers
```http
Content-Type: application/json; charset=utf-8
X-Response-Time: 25ms
X-Request-ID: req-123
Cache-Control: no-cache, no-store, must-revalidate
```

## CORS Configuration

```javascript
// Allowed origins
Access-Control-Allow-Origin: http://localhost:3001
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization, X-Request-ID
Access-Control-Allow-Credentials: true
Access-Control-Max-Age: 86400
```

## API Versioning

The API uses URL-based versioning:
- Current version: `v1`
- Base URL: `http://localhost:3001/api`
- Future versions will use: `http://localhost:3001/api/v2`

### Version Compatibility
- **Backward compatibility** maintained within major versions
- **Deprecation notices** provided 6 months before removal
- **Migration guides** provided for major version changes

## Performance Considerations

### Caching Strategy
- ETags for resource versioning
- Cache-Control headers for static resources
- In-memory caching for frequently accessed data
- Redis caching for session data

### Optimization Tips
1. Use pagination for large datasets
2. Implement client-side caching where appropriate
3. Use compression for large payloads
4. Batch multiple operations when possible
5. Use WebSocket connections for real-time updates

## Security Considerations

### Input Validation
- All inputs validated against schemas
- SQL injection prevention
- XSS prevention measures
- File upload restrictions

### Authentication & Authorization
- Session-based authentication via Nostr
- Feature-based access control
- Rate limiting per user/IP
- Audit logging for sensitive operations

## Testing & Development

### Testing Endpoints
```bash
# Health check
curl http://localhost:3001/api/health

# Get graph data
curl http://localhost:3001/api/graph/data

# Get agents
curl http://localhost:3001/api/bots/data
```

### Development Tools
- Postman collection available at `/docs/postman/`
- OpenAPI specification at `/docs/openapi.json`
- Interactive API documentation at `/docs/swagger`

## Migration from Legacy Endpoints

### Deprecated Endpoints (Remove by v2.0)
| Legacy | New | Status |
|--------|-----|--------|
| `/ws` | `/wss` | Deprecated |
| `/api/v0/graph` | `/api/graph/data` | Deprecated |
| `/api/bots` | `/api/bots/data` | Deprecated |

### Breaking Changes in v1.0
1. Node IDs changed from u16 to u32
2. WebSocket binary protocol updated
3. Authentication moved to Nostr-based system
4. Response format standardized across endpoints