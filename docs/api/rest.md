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
POST /api/auth/nostr
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
GET /api/graph
```

**Query Parameters:**
- `includeMetadata` (boolean) - Include node/edge metadata
- `layout` (string) - Layout algorithm: `force`, `hierarchical`, `circular`

**Response:**
```json
{
  "nodes": [
    {
      "id": "node_1",
      "label": "Knowledge Node",
      "x": 100.0,
      "y": 200.0,
      "z": 50.0,
      "type": "concept",
      "metadata": {}
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "node_1",
      "target": "node_2",
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

### Update Graph
```http
POST /api/graph
```

**Request:**
```json
{
  "nodes": [...],
  "edges": [...],
  "operation": "merge" // or "replace"
}
```

### Get Graph Metadata
```http
GET /api/graph/metadata
```

**Response:**
```json
{
  "statistics": {
    "totalNodes": 1000,
    "totalEdges": 5000,
    "connectedComponents": 3,
    "averageDegree": 5.0
  },
  "lastPhysicsUpdate": "2024-01-01T00:00:00Z",
  "gpuEnabled": true
}
```

## Agent/Bots Endpoints

### List Agents
```http
GET /api/bots
```

**Response:**
```json
{
  "agents": [
    {
      "id": "agent_1",
      "name": "Coordinator Alpha",
      "type": "coordinator",
      "status": "active",
      "health": {
        "cpu": 45.2,
        "memory": 512,
        "uptime": 3600
      },
      "capabilities": ["orchestration", "planning"],
      "currentTask": "task_123"
    }
  ],
  "swarmTopology": "hierarchical",
  "totalAgents": 15
}
```

### Spawn Agent
```http
POST /api/bots/spawn
```

**Request:**
```json
{
  "type": "researcher",
  "name": "Research Bot 1",
  "capabilities": ["search", "analysis"],
  "config": {
    "maxTasks": 5,
    "priority": "high"
  }
}
```

### Agent Metrics
```http
GET /api/bots/{agentId}/metrics
```

**Response:**
```json
{
  "agentId": "agent_1",
  "performance": {
    "tasksCompleted": 42,
    "successRate": 0.95,
    "averageResponseTime": 250,
    "tokensUsed": 15000
  },
  "resources": {
    "cpuUsage": 35.5,
    "memoryMB": 256,
    "networkKbps": 100
  }
}
```

### Orchestrate Task
```http
POST /api/bots/orchestrate
```

**Request:**
```json
{
  "task": "Analyze the codebase and generate documentation",
  "strategy": "parallel",
  "maxAgents": 5,
  "priority": "high",
  "timeout": 300
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
    "iterations": 5
  },
  "visualization": {
    "nodeSize": 10,
    "edgeThickness": 2,
    "showLabels": true,
    "hologramEffect": true
  },
  "xr": {
    "enabled": false,
    "handTracking": true,
    "passthrough": true
  }
}
```

### Update Settings
```http
PUT /api/settings
```

**Request:**
```json
{
  "path": "physics.gravity",
  "value": -12.0
}
```

### Get Protected Settings
```http
GET /api/settings/protected
Authorization: Bearer {token}
```

## Files Endpoints

### List Files
```http
GET /api/files
```

**Query Parameters:**
- `path` (string) - Directory path
- `type` (string) - File type filter: `markdown`, `json`, `all`

**Response:**
```json
{
  "files": [
    {
      "name": "knowledge.md",
      "path": "/data/markdown/knowledge.md",
      "size": 1024,
      "modified": "2024-01-01T00:00:00Z",
      "type": "markdown"
    }
  ],
  "totalSize": 10240,
  "count": 10
}
```

### Upload File
```http
POST /api/files/upload
Content-Type: multipart/form-data
```

**Form Data:**
- `file` - File to upload
- `path` - Target directory
- `overwrite` - Boolean to overwrite existing

### Download File
```http
GET /api/files/download?path=/data/file.md
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
    "capabilities": ["handTracking", "passthrough", "spatialAnchors"]
  },
  "session": {
    "active": true,
    "mode": "immersive-ar",
    "referenceSpace": "local-floor"
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
    "shadowMap": true
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
    "particles": false
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
    "drawCalls": 150
  },
  "gpu": {
    "utilization": 75,
    "memory": 2048,
    "temperature": 65
  },
  "network": {
    "websocketClients": 5,
    "bandwidth": 1024,
    "latency": 15
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
  "services": {
    "graph": "operational",
    "agents": "operational",
    "gpu": "operational",
    "mcp": "connected"
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
    "latency": 5
  },
  "tools": {
    "available": 50,
    "active": 3
  }
}
```

## External Service Integrations

### GitHub Integration
```http
GET /api/github/repos/{owner}/{repo}
```

**Response:**
```json
{
  "repository": {
    "name": "visionflow",
    "stars": 100,
    "forks": 20,
    "issues": 5
  },
  "pullRequests": [...],
  "commits": [...]
}
```

### RAGFlow Query
```http
POST /api/ragflow/query
```

**Request:**
```json
{
  "query": "What is the architecture?",
  "context": "technical",
  "maxResults": 5
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

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request was invalid",
    "details": {
      "field": "name",
      "reason": "Required field missing"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes
- `UNAUTHORIZED` - Authentication required
- `FORBIDDEN` - Insufficient permissions
- `NOT_FOUND` - Resource not found
- `INVALID_REQUEST` - Invalid request parameters
- `RATE_LIMITED` - Too many requests
- `INTERNAL_ERROR` - Server error

## Rate Limiting

| Endpoint Category | Rate Limit | Window |
|------------------|------------|--------|
| Graph Operations | 100/min | 1 minute |
| Agent Operations | 50/min | 1 minute |
| File Operations | 10/min | 1 minute |
| Analytics | 200/min | 1 minute |
| Settings | 100/min | 1 minute |