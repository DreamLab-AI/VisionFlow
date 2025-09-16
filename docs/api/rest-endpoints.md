# REST API Endpoints

*[API Documentation](README.md) > REST Endpoints*

## Overview

VisionFlow's REST API provides comprehensive endpoints for graph management, AI agent orchestration, system configuration, and analytics. All endpoints follow RESTful conventions with automatic camelCase ↔ snake_case conversion and comprehensive validation.

## Base Configuration

### URLs and Headers
```
Development: http://localhost:3001/api
Production:  https://api.visionflow.dev/api

Headers:
Content-Type: application/json
Accept: application/json
Authorization: Bearer <token>
X-Request-ID: <uuid>
```

### Standard Response Format
```json
{
  "success": true,
  "data": {
    // Endpoint-specific data
  },
  "metadata": {
    "timestamp": "2025-09-16T00:00:00Z",
    "requestId": "req-123",
    "processingTime": 250
  }
}
```

## Authentication

### Nostr Authentication
**Endpoint:** `POST /api/auth/nostr`

Authenticate using Nostr protocol (NIP-07):

```json
// Request
{
  "pubkey": "user_public_key_hex",
  "signature": "signature_hex",
  "challenge": "server_challenge",
  "relay": "wss://relay.nostr.com"
}

// Response
{
  "success": true,
  "user": {
    "pubkey": "user_public_key",
    "npub": "npub_encoded_key",
    "isPowerUser": true
  },
  "token": "session_token",
  "expiresAt": 1234567890,
  "features": ["graph", "agents", "xr", "gpu"]
}
```

## Graph Management

### Get Graph Data
**Endpoint:** `GET /api/graph/data`

Retrieve complete graph structure with nodes and edges:

```json
// Response
{
  "success": true,
  "data": {
    "nodes": [
      {
        "id": 1,
        "label": "Node Label",
        "x": 10.5,
        "y": 20.3,
        "z": 0.0,
        "color": "#ff6b6b",
        "size": 8.0,
        "metadata": {
          "type": "knowledge",
          "category": "concept"
        }
      }
    ],
    "edges": [
      {
        "source": 1,
        "target": 2,
        "weight": 1.0,
        "color": "#4ecdc4",
        "metadata": {
          "relationship": "relates_to"
        }
      }
    ],
    "metadata": {
      "nodeCount": 1500,
      "edgeCount": 3200,
      "avgDegree": 4.27,
      "components": 3
    }
  }
}
```

### Get Paginated Graph Data
**Endpoint:** `GET /api/graph/data/paginated`

**Query Parameters:**
- `page` (number): Page number (default: 1)
- `pageSize` (number): Items per page (default: 100, max: 1000)
- `nodeType` (string): Filter by node type ("agent", "knowledge")
- `sortBy` (string): Sort field ("id", "label", "degree")

```json
// Response
{
  "data": {
    "nodes": [...],
    "edges": [...]
  },
  "pagination": {
    "currentPage": 1,
    "totalPages": 15,
    "totalItems": 1500,
    "pageSize": 100,
    "hasNext": true,
    "hasPrevious": false
  }
}
```

### Update Graph
**Endpoint:** `POST /api/graph/update`

Update graph from file sources or direct data:

```json
// Request
{
  "source": "file", // or "data"
  "filePath": "/path/to/graph.json", // if source is "file"
  "data": {
    "nodes": [...],
    "edges": [...]
  }, // if source is "data"
  "mergeStrategy": "replace", // "replace", "merge", "append"
  "validateConsistency": true
}

// Response
{
  "success": true,
  "data": {
    "nodesAdded": 150,
    "nodesUpdated": 25,
    "edgesAdded": 300,
    "edgesUpdated": 50,
    "processingTime": 1250
  }
}
```

### Refresh Graph
**Endpoint:** `POST /api/graph/refresh`

Refresh graph topology and recompute derived properties:

```json
// Request
{
  "recomputeLayout": true,
  "validateIntegrity": true,
  "clearCache": false
}

// Response
{
  "success": true,
  "data": {
    "nodesProcessed": 1500,
    "edgesProcessed": 3200,
    "layoutRecomputed": true,
    "cacheCleared": false
  }
}
```

## Agent Orchestration

### List Agents
**Endpoint:** `GET /api/bots/data`

Get current state of all active AI agents:

```json
// Response
{
  "success": true,
  "data": {
    "agents": [
      {
        "id": "agent-001",
        "name": "Research Agent Alpha",
        "type": "researcher",
        "status": "active", // "idle", "busy", "error", "offline"
        "health": 0.95,
        "workload": 0.7,
        "capabilities": ["web_search", "document_analysis"],
        "currentTask": "Analyzing research papers on graph theory",
        "position": {
          "x": 150.5,
          "y": 200.3,
          "z": 50.0
        },
        "metrics": {
          "tasksCompleted": 42,
          "successRate": 0.95,
          "avgResponseTime": 250,
          "tokenRate": 1523.4
        }
      }
    ],
    "swarms": [
      {
        "id": "swarm_1757880683494_yl81sece5",
        "topology": "hierarchical",
        "agentCount": 5,
        "status": "active",
        "performance": {
          "throughput": 15.3,
          "efficiency": 0.87
        }
      }
    ]
  }
}
```

### Update Agents
**Endpoint:** `POST /api/bots/data`

Update agent configurations and states:

```json
// Request
{
  "agents": [
    {
      "id": "agent-001",
      "status": "busy",
      "currentTask": "New task assignment",
      "position": {
        "x": 160.0,
        "y": 210.0,
        "z": 55.0
      }
    }
  ]
}

// Response
{
  "success": true,
  "data": {
    "agentsUpdated": 1,
    "invalidAgents": [],
    "warnings": []
  }
}
```

### Initialize Swarm
**Endpoint:** `POST /api/bots/initialize-swarm`

Create a new multi-agent swarm with specified topology:

```json
// Request
{
  "topology": "hierarchical", // "mesh", "ring", "star"
  "maxAgents": 10,
  "agentTypes": [
    {
      "type": "researcher",
      "count": 3,
      "capabilities": ["web_search", "document_analysis"]
    },
    {
      "type": "coder",
      "count": 2,
      "capabilities": ["code_generation", "debugging"]
    }
  ],
  "strategy": "balanced" // "specialized", "adaptive"
}

// Response
{
  "success": true,
  "data": {
    "swarmId": "swarm_1757880683494_yl81sece5",
    "topology": "hierarchical",
    "agentsCreated": 5,
    "coordinatorId": "agent_1757967065850_coord",
    "estimatedStartupTime": 15000
  }
}
```

## System Configuration

### Get Settings
**Endpoint:** `GET /api/settings`

Retrieve current system configuration:

```json
// Response
{
  "success": true,
  "data": {
    "physics": {
      "simulation": {
        "repulsionStrength": 100.0,
        "attractionStrength": 0.1,
        "centeringStrength": 0.05,
        "damping": 0.9,
        "timeStep": 0.016,
        "useGPU": true
      }
    },
    "visualization": {
      "nodeSize": 8.0,
      "edgeWidth": 2.0,
      "showLabels": true,
      "backgroundColor": "#1a1a1a",
      "enableBloom": true,
      "bloomIntensity": 0.8
    },
    "features": {
      "enabledFeatures": ["gpu", "agents", "xr"],
      "powerUserKeys": ["ctrl+shift+d", "ctrl+alt+g"]
    },
    "performance": {
      "maxNodes": 100000,
      "maxEdges": 500000,
      "targetFPS": 60,
      "enableLOD": true
    }
  }
}
```

### Update Settings
**Endpoint:** `POST /api/settings`

Update system configuration with validation:

```json
// Request
{
  "physics": {
    "simulation": {
      "repulsionStrength": 150.0,
      "damping": 0.85
    }
  },
  "visualization": {
    "nodeSize": 10.0,
    "enableBloom": false
  }
}

// Response
{
  "success": true,
  "data": {
    "settingsUpdated": ["physics.simulation.repulsionStrength", "physics.simulation.damping", "visualization.nodeSize", "visualization.enableBloom"],
    "validationWarnings": [],
    "restartRequired": false
  }
}
```

### Reset Settings
**Endpoint:** `POST /api/settings/reset`

Reset configuration to default values:

```json
// Request
{
  "sections": ["physics", "visualization"], // or omit for full reset
  "preserveUserPreferences": true
}

// Response
{
  "success": true,
  "data": {
    "sectionsReset": ["physics", "visualization"],
    "settingsPreserved": ["features.powerUserKeys"]
  }
}
```

## Analytics & Monitoring

### System Metrics
**Endpoint:** `GET /api/analytics/system`

Get comprehensive system performance metrics:

```json
// Response
{
  "success": true,
  "data": {
    "performance": {
      "fps": 59.2,
      "frameTime": 16.8,
      "gpuUtilization": 0.75,
      "memoryUsage": {
        "used": 2.1,
        "total": 8.0,
        "unit": "GB"
      }
    },
    "network": {
      "wsConnections": 3,
      "messageRate": 150.5,
      "bandwidth": {
        "inbound": 2.3,
        "outbound": 5.7,
        "unit": "MB/s"
      }
    },
    "agents": {
      "totalAgents": 12,
      "activeAgents": 8,
      "averageWorkload": 0.65,
      "throughput": 45.2
    }
  }
}
```

### Graph Analytics
**Endpoint:** `GET /api/analytics/graph`

Analyze graph topology and structure:

```json
// Response
{
  "success": true,
  "data": {
    "topology": {
      "nodes": 1500,
      "edges": 3200,
      "density": 0.0028,
      "avgDegree": 4.27,
      "maxDegree": 47,
      "components": 3,
      "diameter": 12,
      "clustering": 0.35
    },
    "centrality": {
      "betweenness": [
        {"nodeId": 42, "score": 0.15},
        {"nodeId": 15, "score": 0.12}
      ],
      "eigenvector": [
        {"nodeId": 23, "score": 0.31},
        {"nodeId": 67, "score": 0.28}
      ]
    },
    "communities": [
      {
        "id": 1,
        "nodes": [1, 5, 12, 23, 34],
        "size": 5,
        "modularity": 0.42
      }
    ]
  }
}
```

### Shortest Path Computation
**Endpoint:** `POST /api/analytics/shortest-path`

Compute GPU-accelerated shortest paths from source node:

```json
// Request
{
  "sourceNodeId": 42,
  "algorithm": "dijkstra", // "bellman-ford", "gpu-optimized"
  "includePathReconstruction": false
}

// Response
{
  "success": true,
  "data": {
    "sourceNodeId": 42,
    "distances": {
      "42": 0.0,
      "43": 1.5,
      "44": 2.3,
      "45": null, // unreachable
      "46": 3.7
    },
    "unreachableCount": 1,
    "computeTime": 8.5,
    "algorithm": "gpu-optimized"
  }
}
```

### Clustering Analysis
**Endpoint:** `GET /api/analytics/clustering`

Run semantic clustering on graph nodes:

```json
// Query Parameters:
// ?algorithm=louvain&resolution=1.0&includeMetrics=true

// Response
{
  "success": true,
  "data": {
    "algorithm": "louvain",
    "clusterCount": 8,
    "modularity": 0.67,
    "clusters": [
      {
        "id": 1,
        "nodes": [1, 5, 12, 23],
        "centroid": {"x": 150.5, "y": 200.3},
        "coherence": 0.85,
        "keywords": ["machine learning", "neural networks"]
      }
    ],
    "metrics": {
      "silhouetteScore": 0.73,
      "intraClusterDistance": 45.2,
      "interClusterDistance": 152.8
    }
  }
}
```

## Health & Status

### Service Health
**Endpoint:** `GET /api/health`

Get comprehensive service health status:

```json
// Response
{
  "status": "healthy",
  "timestamp": "2025-09-16T12:00:00Z",
  "version": "1.0.0",
  "components": {
    "database": {
      "status": "healthy",
      "responseTime": 2.5,
      "connections": 8
    },
    "gpu": {
      "status": "healthy",
      "utilization": 0.45,
      "memory": 0.62,
      "temperature": 65
    },
    "websocket": {
      "status": "healthy",
      "connections": 12,
      "messageRate": 150.5
    },
    "agents": {
      "status": "healthy",
      "activeCount": 8,
      "averageHealth": 0.92
    }
  },
  "uptime": 86400,
  "requestCount": 15420,
  "errorRate": 0.002
}
```

### MCP Health
**Endpoint:** `GET /api/mcp/health`

Check MCP server connection status:

```json
// Response
{
  "status": "connected",
  "mcpServer": {
    "host": "multi-agent-container",
    "port": 9500,
    "protocol": "JSON-RPC 2.0",
    "version": "2.0.0-alpha.101",
    "uptime": 3600
  },
  "swarms": [
    {
      "id": "swarm_1757880683494_yl81sece5",
      "status": "active",
      "agentCount": 5,
      "lastActivity": "2025-09-16T11:58:30Z"
    }
  ],
  "performance": {
    "requestCount": 847,
    "avgResponseTime": 15.3,
    "errorRate": 0.001
  }
}
```

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_FAILED",
    "message": "Request validation failed",
    "requestId": "req-123",
    "timestamp": "2025-09-16T00:00:00Z",
    "details": {
      "field": "physics.repulsionStrength",
      "provided": -100.0,
      "expected": "number between 1 and 1000",
      "errorCode": "VALUE_OUT_OF_RANGE"
    },
    "suggestions": [
      "Repulsion strength must be a positive number",
      "Typical values range from 10 to 500"
    ]
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status | Recovery Action |
|------|-------------|-------------|-----------------|
| `UNAUTHORIZED` | Authentication required | 401 | Provide valid auth token |
| `FORBIDDEN` | Insufficient permissions | 403 | Check user permissions |
| `NOT_FOUND` | Resource not found | 404 | Verify resource ID |
| `VALIDATION_ERROR` | Input validation failed | 422 | Fix request parameters |
| `RATE_LIMITED` | Too many requests | 429 | Implement backoff |
| `RESOURCE_LIMIT_EXCEEDED` | Request too large | 413 | Reduce request size |
| `INTERNAL_ERROR` | Server error | 500 | Retry with backoff |
| `SERVICE_UNAVAILABLE` | Service down | 503 | Check service status |

## Rate Limiting

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1694851200
X-RateLimit-Retry-After: 60
```

### Limits by Category

| Category | Requests/Min | Burst | Window | Notes |
|----------|--------------|-------|--------|-------|
| Graph Operations | 100 | 20 | 60s | CRUD operations |
| Agent Operations | 50 | 10 | 60s | Swarm management |
| Settings Updates | 100 | 20 | 60s | Configuration changes |
| Analytics Queries | 200 | 50 | 60s | Computations and analysis |
| Health Checks | 600 | 100 | 60s | Status monitoring |
| Authentication | 10 | 5 | 300s | Login attempts |

## Case Conversion

The API automatically handles case conversion between client and server:

### Client → Server Flow
1. Client sends `camelCase` JSON
2. Server converts to `snake_case` for processing
3. Storage uses `snake_case` format

### Server → Client Flow
1. Server processes `snake_case` internally
2. Server converts to `camelCase` for response
3. Client receives `camelCase` JSON

### Example
```javascript
// Client sends
{
  "repulsionStrength": 150.0,
  "showLabels": true
}

// Server processes
{
  "repulsion_strength": 150.0,
  "show_labels": true
}

// Client receives
{
  "repulsionStrength": 150.0,
  "showLabels": true
}
```

## Security Features

### Input Validation
- **Syntax Validation**: JSON schema compliance
- **Semantic Validation**: Business rule enforcement
- **Security Validation**: Malicious content detection
- **Resource Validation**: Memory and size limits

### Security Headers
```http
Strict-Transport-Security: max-age=31536000
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
```

### CORS Configuration
```http
Access-Control-Allow-Origin: http://localhost:3001
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Allow-Credentials: true
```

## Performance Optimization

### Best Practices
1. **Use pagination** for large datasets (`/api/graph/data/paginated`)
2. **Enable compression** via `Accept-Encoding: gzip`
3. **Batch operations** when updating multiple entities
4. **Cache responses** using ETags and conditional requests
5. **Use WebSockets** for real-time updates instead of polling

### Caching Strategy
- **ETags**: Resource versioning for conditional requests
- **Cache-Control**: Client-side caching directives
- **Compression**: GZIP compression for payloads >1KB
- **CDN**: Static resource distribution

## SDK Integration

### curl Examples
```bash
# Get graph data
curl -H "Authorization: Bearer TOKEN" \
     http://localhost:3001/api/graph/data

# Update settings
curl -X POST \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer TOKEN" \
     -d '{"physics":{"simulation":{"repulsionStrength":150}}}' \
     http://localhost:3001/api/settings

# Initialize swarm
curl -X POST \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer TOKEN" \
     -d '{"topology":"hierarchical","maxAgents":10}' \
     http://localhost:3001/api/bots/initialize-swarm
```

### Integration Testing
```bash
# Run API integration tests
npm run test:api

# Test with specific endpoint
npm run test:api -- --grep "settings"

# Performance benchmarks
npm run benchmark:api
```

---

*For WebSocket protocols and binary formats, see [websocket-streams.md](websocket-streams.md) and [binary-protocol.md](binary-protocol.md).*