---
title: REST API Reference
description: Complete REST API endpoint reference for VisionFlow
category: reference
difficulty-level: intermediate
tags:
  - api
  - rest
  - backend
updated-date: 2025-01-29
---

# REST API Reference

**Base URL**: `http://localhost:9090/api`

Complete reference for VisionFlow REST API endpoints.

---

## HTTP Methods

| Method | Idempotent | Safe | Use Case |
|--------|------------|------|----------|
| **GET** | Yes | Yes | Retrieve resources |
| **POST** | No | No | Create resources, actions |
| **PUT** | Yes | No | Update/replace resources |
| **PATCH** | No | No | Partial updates |
| **DELETE** | Yes | No | Remove resources |

---

## Core Endpoints

### Health Check

```http
GET /api/health
```

**Response**:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2025-12-18T12:00:00Z",
  "components": {
    "database": "healthy",
    "gpu": "healthy",
    "websocket": "healthy"
  }
}
```

### Application Configuration

```http
GET /api/config
```

**Response**:
```json
{
  "version": "0.1.0",
  "features": {
    "ragflow": true,
    "perplexity": false,
    "openai": true,
    "gpu": true,
    "xr": true
  },
  "websocket": {
    "minUpdateRate": 16,
    "maxUpdateRate": 120,
    "protocol": "binary-v2"
  }
}
```

---

## Graph Endpoints

### Get Complete Graph

```http
GET /api/graph/data
Authorization: Bearer {token}
```

**Response**:
```json
{
  "nodes": [
    {
      "id": 1,
      "metadataId": "node-uuid",
      "label": "Example Node",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
      "metadata": {
        "owl-class-iri": "http://example.org/Class",
        "file-source": "example.md"
      },
      "type": "concept",
      "size": 1.0,
      "color": "#3498db"
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "relationshipType": "related-to",
      "weight": 1.0
    }
  ],
  "metadata": {},
  "settlementState": {
    "isSettled": false,
    "kineticEnergy": 12.5
  }
}
```

### Paginated Graph Data

```http
GET /api/graph/data/paginated?page=1&page-size=100
```

**Query Parameters**:
- `page` (integer): Page number (1-indexed)
- `page-size` (integer): Items per page (default: 100, max: 1000)
- `filter` (string, optional): Node type filter
- `sort` (string, optional): Sort field

### Update Graph

```http
POST /api/graph/update
Authorization: Bearer {token}
```

Triggers graph update from GitHub repository.

---

## Ontology Endpoints

### Get Class Hierarchy

```http
GET /api/ontology/hierarchy?ontology-id=default&max-depth=5
```

**Query Parameters**:
- `ontology-id` (string, optional): Ontology identifier
- `max-depth` (integer, optional): Maximum traversal depth

### Load Ontology

```http
POST /api/ontology/load
Content-Type: application/json
Authorization: Bearer {token}

{
  "source": "https://example.org/ontology.owl",
  "format": "rdf/xml",
  "validateImmediately": true
}
```

**Supported Formats**:
- `rdf/xml`
- `turtle`
- `n-triples`
- `json-ld`

### Validate Ontology

```http
POST /api/ontology/validate
Content-Type: application/json

{
  "ontologyId": "ontology-123",
  "mode": "full",
  "priority": 5,
  "enableWebsocketUpdates": true
}
```

**Validation Modes**:
- `quick`: Fast structural checks only
- `full`: Complete validation with reasoning
- `incremental`: Only validate changes

---

## Physics Endpoints

### Start/Stop Simulation

```http
POST /api/physics/start
POST /api/physics/stop
```

### Get Simulation Status

```http
GET /api/physics/status
```

**Response**:
```json
{
  "isRunning": true,
  "isSettled": false,
  "stableFrameCount": 45,
  "kineticEnergy": 12.5,
  "fps": 60,
  "nodeCount": 150,
  "gpu": {
    "enabled": true,
    "utilization": 45,
    "memory": "400MB / 16GB"
  }
}
```

### Update Physics Parameters

```http
POST /api/physics/parameters
Content-Type: application/json

{
  "gravity": 0.1,
  "charge": -30.0,
  "linkStrength": 0.5,
  "friction": 0.9,
  "theta": 0.8,
  "enableGPU": true
}
```

---

## Analytics Endpoints

### Detect Communities

```http
POST /api/semantic/communities
Content-Type: application/json

{
  "algorithm": "louvain",
  "resolution": 1.0,
  "minSize": 5
}
```

**Supported Algorithms**:
- `louvain`: Louvain method (fast, hierarchical)
- `label-propagation`: Label propagation (very fast)
- `modularity`: Modularity optimization

### Compute Centrality

```http
POST /api/semantic/centrality
Content-Type: application/json

{
  "metric": "betweenness",
  "normalized": true,
  "nodeIds": [1, 2, 3]
}
```

**Supported Metrics**:
- `degree`: Node degree centrality
- `betweenness`: Betweenness centrality
- `closeness`: Closeness centrality
- `eigenvector`: Eigenvector centrality
- `pagerank`: PageRank algorithm

---

## Rate Limiting

### Default Limits

| Endpoint Type | Limit | Window | Per |
|---------------|-------|--------|-----|
| Authentication | 10 requests | 1 minute | IP |
| REST API | 100 requests | 1 minute | IP |
| REST API (Authenticated) | 1000 requests | 1 hour | User |
| WebSocket Messages | 120 messages | 1 second | Connection |
| File Upload | 10 requests | 1 hour | User |

### Rate Limit Headers

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1702915200
```

---

## Content Negotiation

### Accept Header

```http
GET /api/graph/data HTTP/1.1
Accept: application/json                    # JSON response (default)
Accept: application/ld+json                 # JSON-LD (RDF)
Accept: text/turtle                         # Turtle (RDF)
Accept: application/rdf+xml                 # RDF/XML
Accept: text/csv                            # CSV export
```

### Compression

```http
GET /api/graph/data HTTP/1.1
Accept-Encoding: gzip, deflate, br          # Brotli preferred
```

---

## Related Documentation

- [Complete REST API Reference](./rest-api-complete.md) - Full 110+ endpoints
- [WebSocket API](./websocket-api.md)
- [Authentication](./authentication.md)
- [Error Codes](../error-codes.md)
