---
title: API Complete Reference
description: **Version**: 1.0.0 **Base URL**: `http://localhost:9090/api` **WebSocket**: `ws://localhost:9090/ws`
category: reference
tags:
  - api
  - rest
  - websocket
updated-date: 2025-12-18
difficulty-level: intermediate
---


# API Complete Reference

**Version**: 1.0.0
**Base URL**: `http://localhost:9090/api`
**WebSocket**: `ws://localhost:9090/ws`
**Last Updated**: 2025-11-04
**Status**: Production Reference

---

## Table of Contents

1. 
2. [REST API Reference](#rest-api-reference)
3. [Request/Response Formats](#requestresponse-formats)
4. [Error Responses](#error-responses)
5. [Rate Limiting](#rate-limiting)
6. [Pagination](#pagination)
7. 
8. [Bulk Operations](#bulk-operations)
9. [Webhooks](#webhooks)
10. [API Versioning](#api-versioning)
11. [Examples](#examples)

---

## Authentication & Authorization

### Authentication Methods

VisionFlow supports three authentication methods:

1. **JWT Tokens** (Session-based)
2. **API Keys** (Programmatic access)
3. **OAuth 2.0** (Third-party integrations)

### JWT Authentication

**Login Endpoint**: `POST /api/auth/login`

**Request**:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expiresIn": 86400,
    "user": {
      "id": "uuid-123",
      "email": "user@example.com",
      "role": "user",
      "permissions": ["read:graph", "write:graph"]
    }
  }
}
```

**Token Usage**:
```bash
curl -H "Authorization: Bearer YOUR-JWT-TOKEN" \
  http://localhost:9090/api/graph/data
```

**Token Expiration**:
- Access tokens: 24 hours
- Refresh tokens: 30 days
- Sliding expiration on activity

### Token Refresh

**Endpoint**: `POST /api/auth/refresh`

**Request**:
```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expiresIn": 86400
  }
}
```

### API Keys

**Generate API Key**: `POST /api/auth/api-keys`

**Request**:
```json
{
  "name": "Production API Key",
  "scopes": ["read:graph", "write:graph", "read:ontology"],
  "expiresIn": 31536000
}
```

**Response** (201 Created):
```json
{
  "success": true,
  "data": {
    "id": "key-uuid-123",
    "key": "vf_live_a1b2c3d4e5f6g7h8i9j0",
    "name": "Production API Key",
    "scopes": ["read:graph", "write:graph", "read:ontology"],
    "createdAt": "2025-11-04T12:00:00Z",
    "expiresAt": "2026-11-04T12:00:00Z"
  }
}
```

**API Key Usage**:
```bash
curl -H "X-API-Key: vf_live_a1b2c3d4e5f6g7h8i9j0" \
  http://localhost:9090/api/graph/data
```

**API Key Management**:
- List keys: `GET /api/auth/api-keys`
- Revoke key: `DELETE /api/auth/api-keys/:id`
- Rotate key: `POST /api/auth/api-keys/:id/rotate`

### OAuth 2.0

**Supported Providers**:
- GitHub
- Google
- Microsoft

**Authorization Flow**:

1. **Initiate OAuth**: `GET /api/auth/oauth/:provider`
   ```bash
   GET /api/auth/oauth/github?redirect_uri=https://myapp.com/callback
   ```

2. **Callback**: `GET /api/auth/oauth/:provider/callback`
   - Receives authorization code
   - Exchanges for access token
   - Returns JWT

3. **Use JWT**: Include in Authorization header

### Permission Scopes

| Scope | Description |
|-------|-------------|
| `read:graph` | Read graph data and metadata |
| `write:graph` | Create/update/delete nodes and edges |
| `read:ontology` | Read ontology classes and properties |
| `write:ontology` | Load and validate ontologies |
| `admin:physics` | Control physics simulation |
| `admin:system` | System administration |
| `read:analytics` | Access analytics endpoints |
| `write:workspace` | Manage workspaces |

---

## REST API Reference

### Health & Configuration

#### Health Check

**GET** `/api/health`

Returns system health status.

**Response** (200 OK):
```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2025-11-04T12:00:00Z",
  "services": {
    "database": "healthy",
    "physics": "running",
    "gpu": "available"
  }
}
```

#### Application Configuration

**GET** `/api/config`

Returns complete application configuration.

**Response** (200 OK):
```json
{
  "version": "0.1.0",
  "features": {
    "ragflow": true,
    "perplexity": false,
    "openai": true,
    "whisper": true
  },
  "websocket": {
    "minUpdateRate": 16,
    "maxUpdateRate": 60,
    "motionThreshold": 0.001,
    "motionDamping": 0.95
  },
  "rendering": {
    "ambientLightIntensity": 0.5,
    "enableAmbientOcclusion": true,
    "backgroundColor": "#1a1a1a"
  },
  "xr": {
    "enabled": true,
    "roomScale": 5.0,
    "spaceType": "unbounded"
  }
}
```

### Graph Endpoints

#### Get Complete Graph Data

**GET** `/api/graph/data`

Returns complete graph with nodes, edges, physics positions, and metadata.

**Query Parameters**:
- `format` (optional): Response format (`json` | `graphml` | `cytoscape`)
- `include-metadata` (optional): Include full metadata (default: `true`)
- `include-positions` (optional): Include physics positions (default: `true`)

**Response** (200 OK):
```json
{
  "nodes": [
    {
      "id": 1,
      "metadataId": "node-uuid-123",
      "label": "Example Node",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
      "metadata": {
        "owl-class-iri": "http://example.org/Class",
        "file-source": "example.md"
      },
      "type": "concept",
      "size": 1.0,
      "color": "#3498db",
      "weight": 1.0,
      "group": "category1"
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "relationshipType": "related-to",
      "weight": 1.0,
      "metadata": {}
    }
  ],
  "metadata": {
    "node-uuid-123": {
      "id": "node-uuid-123",
      "fileName": "example.md",
      "title": "Example",
      "sha1": "abc123",
      "properties": {}
    }
  },
  "settlementState": {
    "isSettled": false,
    "stableFrameCount": 45,
    "kineticEnergy": 12.5
  },
  "statistics": {
    "nodeCount": 150,
    "edgeCount": 320,
    "density": 0.028
  }
}
```

**cURL Example**:
```bash
curl -H "Authorization: Bearer YOUR-JWT" \
  "http://localhost:9090/api/graph/data?format=json&include-metadata=true"
```

#### Get Paginated Graph Data

**GET** `/api/graph/data/paginated`

Returns paginated graph data for large graphs.

**Query Parameters**:
- `page` (required): Page number (1-indexed)
- `page-size` (optional): Items per page (default: 100, max: 1000)
- `sort-by` (optional): Sort field (`id` | `label` | `weight`)
- `sort-order` (optional): Sort direction (`asc` | `desc`)

**Response** (200 OK):
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata": {...},
  "pagination": {
    "currentPage": 1,
    "totalPages": 10,
    "pageSize": 100,
    "totalItems": 1000,
    "hasNext": true,
    "hasPrevious": false
  }
}
```

**cURL Example**:
```bash
curl "http://localhost:9090/api/graph/data/paginated?page=1&page-size=50&sort-by=label"
```

#### Update Graph

**POST** `/api/graph/update`

Triggers graph update by fetching new files from GitHub.

**Request Body** (optional):
```json
{
  "forceFull": false,
  "filterPaths": ["docs/", "concepts/"],
  "maxFiles": 100
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Graph updated with 5 new files",
  "data": {
    "filesProcessed": 5,
    "nodesCreated": 12,
    "edgesCreated": 8,
    "duration": "2.5s"
  }
}
```

#### Refresh Graph

**POST** `/api/graph/refresh`

Returns current graph state without modification.

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Graph data retrieved successfully",
  "data": {
    "nodes": [...],
    "edges": [...],
    "metadata": {...}
  }
}
```

### Ontology Endpoints

#### Get Class Hierarchy

**GET** `/api/ontology/hierarchy`

Returns complete class hierarchy.

**Query Parameters**:
- `ontology-id` (optional): Specific ontology (default: "default")
- `max-depth` (optional): Maximum traversal depth
- `include-annotations` (optional): Include class annotations

**Response** (200 OK):
```json
{
  "rootClasses": [
    "http://example.org/Person"
  ],
  "hierarchy": {
    "http://example.org/Person": {
      "iri": "http://example.org/Person",
      "label": "Person",
      "parentIri": null,
      "childrenIris": [
        "http://example.org/Student"
      ],
      "nodeCount": 5,
      "depth": 0,
      "annotations": {
        "rdfs:comment": "A human being",
        "dc:creator": "System"
      }
    }
  }
}
```

#### Load Ontology

**POST** `/api/ontology/load`

Loads ontology from file, URL, or inline content.

**Request Body**:
```json
{
  "source": "https://example.org/ontology.owl",
  "format": "rdf/xml",
  "validateImmediately": true,
  "options": {
    "enableReasoning": true,
    "enableInference": true,
    "maxInferenceDepth": 5
  }
}
```

**Supported Formats**:
- `rdf/xml` - RDF/XML format
- `turtle` - Turtle format
- `json-ld` - JSON-LD format
- `owl/xml` - OWL/XML format
- `functional` - OWL Functional Syntax

**Response** (201 Created):
```json
{
  "ontologyId": "ont-uuid-123",
  "loadedAt": "2025-11-04T12:00:00Z",
  "axiomCount": 150,
  "classCount": 45,
  "propertyCount": 30,
  "loadingTimeMs": 450,
  "validationJobId": "job-uuid-456"
}
```

#### Validate Ontology

**POST** `/api/ontology/validate`

Triggers ontology validation.

**Request Body**:
```json
{
  "ontologyId": "ont-123",
  "mode": "full",
  "priority": 5,
  "enableWebsocketUpdates": true,
  "clientId": "client-abc",
  "options": {
    "validateCardinality": true,
    "validateDomainsRanges": true,
    "validateDisjointClasses": true,
    "reasoningTimeout": 60
  }
}
```

**Validation Modes**:
- `quick`: Structural checks only (5-10 seconds)
- `full`: Complete validation with reasoning (30-60 seconds)
- `incremental`: Only validates changes (5-15 seconds)

**Response** (202 Accepted):
```json
{
  "jobId": "job-uuid-789",
  "status": "queued",
  "estimatedCompletion": "2025-11-04T12:01:00Z",
  "queuePosition": 1,
  "websocketUrl": "/api/ontology/ws?client-id=client-abc"
}
```

#### Get Validation Report

**GET** `/api/ontology/reports/:id`

Retrieves validation report.

**Path Parameters**:
- `id`: Report ID

**Response** (200 OK):
```json
{
  "id": "report-123",
  "ontologyId": "ont-123",
  "timestamp": "2025-11-04T12:00:30Z",
  "status": "completed",
  "duration": "15.5s",
  "violations": [
    {
      "type": "cardinality",
      "severity": "error",
      "message": "Person class violates cardinality constraint",
      "location": "http://example.org/Person",
      "details": {
        "property": "hasAge",
        "expected": "exactly 1",
        "found": "0"
      }
    }
  ],
  "inferredTriples": [
    {
      "subject": "http://example.org/john",
      "predicate": "rdf:type",
      "object": "http://example.org/Agent"
    }
  ],
  "statistics": {
    "totalClasses": 50,
    "totalProperties": 30,
    "totalIndividuals": 100,
    "totalViolations": 2,
    "errorCount": 1,
    "warningCount": 1
  }
}
```

### Physics Endpoints

#### Start Simulation

**POST** `/api/physics/start`

Starts physics simulation.

**Request Body** (optional):
```json
{
  "parameters": {
    "gravity": 0.1,
    "charge": -30.0,
    "linkStrength": 0.5,
    "friction": 0.9
  }
}
```

**Response** (200 OK):
```json
{
  "status": "running",
  "timestamp": "2025-11-04T12:00:00Z",
  "parameters": {
    "gravity": 0.1,
    "charge": -30.0
  }
}
```

#### Get Simulation Status

**GET** `/api/physics/status`

Returns simulation status.

**Response** (200 OK):
```json
{
  "isRunning": true,
  "isSettled": false,
  "stableFrameCount": 45,
  "kineticEnergy": 12.5,
  "fps": 60,
  "nodeCount": 150,
  "parameters": {
    "gravity": 0.1,
    "charge": -30.0,
    "linkStrength": 0.5
  }
}
```

---

## Request/Response Formats

### Standard Request Format

All POST/PUT requests should use JSON:

**Headers**:
```
Content-Type: application/json
Authorization: Bearer YOUR-JWT-TOKEN
Accept: application/json
```

**Body**:
```json
{
  "field1": "value1",
  "field2": "value2"
}
```

### Standard Response Format

All responses follow this structure:

**Success Response**:
```json
{
  "success": true,
  "data": {
    // Response data
  },
  "meta": {
    "timestamp": "2025-11-04T12:00:00Z",
    "requestId": "req-uuid-123"
  }
}
```

**Error Response**:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "details": [
      {
        "field": "email",
        "message": "Email is required"
      }
    ]
  },
  "meta": {
    "timestamp": "2025-11-04T12:00:00Z",
    "requestId": "req-uuid-123"
  }
}
```

### JSON Schema Validation

All endpoints validate requests against JSON schemas.

**Example Schema** (Node Creation):
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "label": {
      "type": "string",
      "minLength": 1,
      "maxLength": 255
    },
    "type": {
      "type": "string",
      "enum": ["concept", "entity", "relationship"]
    },
    "position": {
      "type": "object",
      "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"}
      },
      "required": ["x", "y", "z"]
    }
  },
  "required": ["label", "type"]
}
```

---

## Error Responses

### Error Code Structure

Format: `[SYSTEM]-[SEVERITY]-[NUMBER]`

**Systems**:
- `AP`: API/Application Layer
- `DB`: Database Layer
- `GR`: Graph/Ontology Reasoning
- `GP`: GPU/Physics Computing
- `WS`: WebSocket/Network
- `AU`: Authentication/Authorization
- `ST`: Storage/File Management

**Severity Levels**:
- `E`: Error (recoverable)
- `F`: Fatal (requires restart)
- `W`: Warning (degraded performance)
- `I`: Info (informational)

### Common Error Codes

| Code | HTTP Status | Message | Resolution |
|------|-------------|---------|-----------|
| `AP-E-001` | 400 | Invalid Request Format | Verify JSON syntax |
| `AP-E-002` | 400 | Missing Required Field | Add missing field |
| `AP-E-101` | 401 | Missing Auth Token | Provide JWT token |
| `AP-E-102` | 401 | Invalid Token | Refresh token |
| `AP-E-104` | 403 | Insufficient Permissions | Request elevated permissions |
| `AP-E-201` | 404 | Resource Not Found | Verify resource ID |
| `AP-E-305` | 429 | Rate Limit Exceeded | Slow down requests |
| `DB-E-001` | 503 | Connection Failed | Check database connectivity |
| `GR-E-102` | 400 | Inconsistent Ontology | Fix logical contradictions |
| `GP-E-001` | 503 | No GPU Found | Use CPU fallback |

### Error Response Examples

**Validation Error** (400):
```json
{
  "success": false,
  "error": {
    "code": "AP-E-002",
    "message": "Missing required field 'label'",
    "details": [
      {
        "field": "label",
        "constraint": "required",
        "message": "Label field is required"
      }
    ]
  },
  "meta": {
    "timestamp": "2025-11-04T12:00:00Z",
    "requestId": "req-uuid-123"
  }
}
```

**Authentication Error** (401):
```json
{
  "success": false,
  "error": {
    "code": "AP-E-101",
    "message": "Authorization header missing",
    "details": {
      "expectedHeader": "Authorization: Bearer <token>"
    }
  },
  "meta": {
    "timestamp": "2025-11-04T12:00:00Z"
  }
}
```

**Rate Limit Error** (429):
```json
{
  "success": false,
  "error": {
    "code": "AP-E-305",
    "message": "Rate limit exceeded, retry after 60s",
    "details": {
      "limit": 100,
      "used": 100,
      "reset": "2025-11-04T12:01:00Z"
    }
  },
  "meta": {
    "timestamp": "2025-11-04T12:00:00Z"
  }
}
```

---

## Rate Limiting

### Default Limits

| User Type | Requests/Minute | Requests/Hour | Requests/Day |
|-----------|----------------|---------------|--------------|
| Anonymous | 20 | 100 | 500 |
| Authenticated | 100 | 1000 | 10000 |
| Premium | 500 | 5000 | 50000 |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Rate Limit Headers

All responses include rate limit headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1730736000
X-RateLimit-Policy: per-user
```

### Retry Strategy

When rate limited, use exponential backoff:

**Algorithm**:
```python
def retry_with_backoff(request_func, max_retries=5):
    for attempt in range(max_retries):
        response = request_func()
        if response.status == 429:
            reset_time = response.headers['X-RateLimit-Reset']
            wait_seconds = min(2 ** attempt, reset_time - time.now())
            time.sleep(wait_seconds)
        else:
            return response
    raise RateLimitException()
```

---

## Pagination

### Cursor-Based Pagination

For large datasets (>1000 items):

**Request**:
```
GET /api/graph/nodes?cursor=eyJpZCI6MTIzfQ&limit=50
```

**Response**:
```json
{
  "data": [...],
  "pagination": {
    "cursor": "eyJpZCI6MTczfQ",
    "hasNext": true,
    "limit": 50
  }
}
```

### Offset-Based Pagination

For smaller datasets:

**Request**:
```
GET /api/graph/nodes?page=2&page-size=25
```

**Response**:
```json
{
  "data": [...],
  "pagination": {
    "currentPage": 2,
    "totalPages": 10,
    "pageSize": 25,
    "totalItems": 250,
    "hasNext": true,
    "hasPrevious": true
  }
}
```

### Sorting

**Query Parameters**:
- `sort-by`: Field to sort by
- `sort-order`: `asc` or `desc`

**Example**:
```
GET /api/graph/nodes?sort-by=label&sort-order=asc
```

---

## Filtering & Search

### Query Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `eq` | Equals | `?type[eq]=concept` |
| `ne` | Not equals | `?type[ne]=relationship` |
| `gt` | Greater than | `?weight[gt]=0.5` |
| `gte` | Greater or equal | `?weight[gte]=0.5` |
| `lt` | Less than | `?weight[lt]=2.0` |
| `lte` | Less or equal | `?weight[lte]=2.0` |
| `in` | In array | `?type[in]=concept,entity` |
| `nin` | Not in array | `?type[nin]=relationship` |
| `contains` | String contains | `?label[contains]=test` |
| `startsWith` | String starts with | `?label[startsWith]=node` |

### Complex Filters

**Example**: Find all concept nodes with weight > 0.5:
```
GET /api/graph/nodes?type[eq]=concept&weight[gt]=0.5
```

### Full-Text Search

**Endpoint**: `GET /api/graph/search`

**Query Parameters**:
- `q`: Search query
- `fields`: Fields to search (comma-separated)
- `fuzzy`: Enable fuzzy matching

**Example**:
```bash
curl "http://localhost:9090/api/graph/search?q=quantum&fields=label,description&fuzzy=true"
```

**Response**:
```json
{
  "results": [
    {
      "id": 123,
      "label": "Quantum Computing",
      "score": 0.95,
      "highlights": {
        "label": "<em>Quantum</em> Computing"
      }
    }
  ],
  "total": 5,
  "took": "12ms"
}
```

---

## Bulk Operations

### Batch Create Nodes

**POST** `/api/graph/nodes/batch`

**Request**:
```json
{
  "nodes": [
    {
      "label": "Node 1",
      "type": "concept",
      "position": {"x": 0, "y": 0, "z": 0}
    },
    {
      "label": "Node 2",
      "type": "entity",
      "position": {"x": 1, "y": 0, "z": 0}
    }
  ]
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "created": 2,
    "failed": 0,
    "results": [
      {"id": 1, "status": "created"},
      {"id": 2, "status": "created"}
    ]
  }
}
```

### Batch Update

**PUT** `/api/graph/nodes/batch`

**Request**:
```json
{
  "updates": [
    {"id": 1, "label": "Updated Node 1"},
    {"id": 2, "label": "Updated Node 2"}
  ]
}
```

### Batch Delete

**DELETE** `/api/graph/nodes/batch`

**Request**:
```json
{
  "ids": [1, 2, 3, 4, 5]
}
```

---

## Webhooks

### Register Webhook

**POST** `/api/webhooks`

**Request**:
```json
{
  "url": "https://myapp.com/webhook",
  "events": ["node.created", "node.updated", "graph.updated"],
  "secret": "your-webhook-secret",
  "active": true
}
```

**Response**:
```json
{
  "id": "webhook-uuid-123",
  "url": "https://myapp.com/webhook",
  "events": ["node.created", "node.updated"],
  "createdAt": "2025-11-04T12:00:00Z"
}
```

### Webhook Events

| Event | Description | Payload |
|-------|-------------|---------|
| `node.created` | Node created | `{node: {...}}` |
| `node.updated` | Node updated | `{node: {...}, changes: {...}}` |
| `node.deleted` | Node deleted | `{nodeId: 123}` |
| `edge.created` | Edge created | `{edge: {...}}` |
| `edge.deleted` | Edge deleted | `{edgeId: 456}` |
| `graph.updated` | Graph updated | `{nodesChanged: 10, edgesChanged: 5}` |
| `validation.completed` | Validation done | `{reportId: "report-123"}` |

### Webhook Payload

**Headers**:
```
X-Webhook-Signature: sha256=...
X-Webhook-Event: node.created
X-Webhook-Delivery: delivery-uuid-123
Content-Type: application/json
```

**Body**:
```json
{
  "event": "node.created",
  "timestamp": "2025-11-04T12:00:00Z",
  "data": {
    "node": {
      "id": 123,
      "label": "New Node"
    }
  }
}
```

### Signature Verification

**Algorithm** (HMAC-SHA256):
```python
import hmac
import hashlib

def verify_signature(payload, signature, secret):
    expected = hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

### Webhook Retries

- Retry on failure: 5 attempts
- Backoff: exponential (1s, 2s, 4s, 8s, 16s)
- Timeout: 30 seconds per attempt

---

## API Versioning

### Version Header

**Recommended**: Use `Accept` header:
```
Accept: application/vnd.visionflow.v1+json
```

### URL Versioning

**Alternative**: Include version in URL:
```
GET /api/v1/graph/data
```

### Version Compatibility

| Version | Status | Deprecation | End of Life |
|---------|--------|-------------|-------------|
| v1.0 | Current | - | - |
| v0.9 | Deprecated | 2025-12-01 | 2026-06-01 |

### Migration Guides

**Migrating from v0.9 to v1.0**:

1. Update authentication to JWT (API keys still supported)
2. Replace `/graph/get` with `/graph/data`
3. Update WebSocket binary protocol from 32 to 36 bytes
4. Use new error code format (`AP-E-001` vs `ERR-001`)

---

## Examples

### cURL Examples

**Get Graph Data**:
```bash
curl -X GET \
  -H "Authorization: Bearer YOUR-JWT" \
  -H "Accept: application/json" \
  http://localhost:9090/api/graph/data
```

**Create Node**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR-JWT" \
  -H "Content-Type: application/json" \
  -d '{"label":"New Node","type":"concept","position":{"x":0,"y":0,"z":0}}' \
  http://localhost:9090/api/graph/nodes
```

**Search Nodes**:
```bash
curl -X GET \
  -H "Authorization: Bearer YOUR-JWT" \
  "http://localhost:9090/api/graph/search?q=quantum&fuzzy=true"
```

### JavaScript Examples

**Fetch Graph Data**:
```javascript
async function fetchGraphData() {
  const response = await fetch('http://localhost:9090/api/graph/data', {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Accept': 'application/json'
    }
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error.message);
  }

  return response.json();
}
```

**Create Node with Error Handling**:
```javascript
async function createNode(label, type, position) {
  try {
    const response = await fetch('http://localhost:9090/api/graph/nodes', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ label, type, position })
    });

    const data = await response.json();

    if (!data.success) {
      console.error('Error:', data.error.code, data.error.message);
      return null;
    }

    return data.data;
  } catch (error) {
    console.error('Network error:', error);
    return null;
  }
}
```

### Python Examples

**Get Graph Data**:
```python
import requests

def get_graph_data(token):
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json'
    }

    response = requests.get(
        'http://localhost:9090/api/graph/data',
        headers=headers
    )

    response.raise_for_status()
    return response.json()
```

**Create Node**:
```python
def create_node(token, label, node_type, position):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    data = {
        'label': label,
        'type': node_type,
        'position': position
    }

    response = requests.post(
        'http://localhost:9090/api/graph/nodes',
        headers=headers,
        json=data
    )

    result = response.json()

    if not result['success']:
        raise Exception(f"Error: {result['error']['message']}")

    return result['data']
```

**Batch Operations**:
```python
def batch_create_nodes(token, nodes):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    response = requests.post(
        'http://localhost:9090/api/graph/nodes/batch',
        headers=headers,
        json={'nodes': nodes}
    )

    return response.json()

# Usage
nodes = [
    {'label': 'Node 1', 'type': 'concept', 'position': {'x': 0, 'y': 0, 'z': 0}},
    {'label': 'Node 2', 'type': 'entity', 'position': {'x': 1, 'y': 0, 'z': 0}}
]

result = batch_create_nodes(token, nodes)
print(f"Created {result['data']['created']} nodes")
```

---

## Related Documentation

- [Error Codes Reference](error-codes.md) - Complete error code catalog
- [WebSocket Protocol](websocket-protocol.md) - Binary WebSocket specification
-  - All configuration options
- [Architecture Overview](../../explanations/architecture/system-overview.md) - System design
- [Getting Started](../../tutorials/01-installation.md) - Installation guide

---

**Last Updated**: 2025-11-04
**Version**: 1.0.0
**Status**: Production Reference
**Maintainer**: VisionFlow API Team
