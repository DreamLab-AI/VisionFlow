# VisionFlow REST API Complete Reference

**Version**: 1.0
**Base URL**: `http://localhost:9090/api`
**Last Updated**: November 3, 2025

---

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [Graph Endpoints](#graph-endpoints)
4. [Ontology Endpoints](#ontology-endpoints)
5. [Physics Endpoints](#physics-endpoints)
6. [File Management](#file-management)
7. [Bots & Swarm Endpoints](#bots--swarm-endpoints)
8. [Analytics Endpoints](#analytics-endpoints)
9. [Workspace Management](#workspace-management)
10. [Advanced Features](#advanced-features)
11. [WebSocket Protocol](#websocket-protocol)
12. [Error Handling](#error-handling)

---

## Authentication

VisionFlow uses JWT (JSON Web Tokens) for authentication.

### Login

**Endpoint**: `POST /api/auth/login`

**Request**:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user": {
      "id": "uuid",
      "email": "user@example.com",
      "role": "user"
    }
  }
}
```

### Using Tokens

Include JWT in Authorization header:

```bash
curl -H "Authorization: Bearer YOUR-JWT-TOKEN" \
  http://localhost:9090/api/graph/data
```

### API Keys

Generate API keys for programmatic access:

```bash
curl -X POST http://localhost:9090/api/auth/api-keys \
  -H "Authorization: Bearer YOUR-JWT-TOKEN"
```

Use API keys:
```bash
curl -H "X-API-Key: YOUR-API-KEY" \
  http://localhost:9090/api/graph/data
```

### Token Expiration

- JWT tokens expire in 24 hours
- Refresh tokens expire in 30 days
- Use `/auth/refresh` to get new tokens

---

## Core Endpoints

### Health Check

**Endpoint**: `GET /api/health`

Returns system health status including version and timestamp.

**Response**:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "timestamp": "2025-11-03T12:00:00Z"
}
```

**cURL Example**:
```bash
curl http://localhost:9090/api/health
```

### Application Configuration

**Endpoint**: `GET /api/config`

Returns complete application configuration including features, WebSocket settings, rendering options, and XR configuration.

**Response**:
```json
{
  "version": "0.1.0",
  "features": {
    "ragflow": true,
    "perplexity": false,
    "openai": true,
    "kokoro": false,
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

**cURL Example**:
```bash
curl http://localhost:9090/api/config
```

---

## Graph Endpoints

### Get Complete Graph Data

**Endpoint**: `GET /api/graph/data`

Returns complete graph data with nodes, edges, physics positions, and metadata.

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
      "color": "#3498db",
      "weight": 1.0,
      "group": "category1"
    }
  ],
  "edges": [
    {
      "source": 1,
      "target": 2,
      "relationshipType": "related-to"
    }
  ],
  "metadata": {
    "node-uuid": {
      "id": "node-uuid",
      "fileName": "example.md",
      "title": "Example",
      "sha1": "abc123"
    }
  },
  "settlementState": {
    "isSettled": false,
    "stableFrameCount": 45,
    "kineticEnergy": 12.5
  }
}
```

**cURL Example**:
```bash
curl http://localhost:9090/api/graph/data
```

**TypeScript Example**:
```typescript
interface GraphResponse {
  nodes: NodeWithPosition[];
  edges: Edge[];
  metadata: Record<string, Metadata>;
  settlementState: SettlementState;
}

async function fetchGraph(): Promise<GraphResponse> {
  const response = await fetch('/api/graph/data');
  return response.json();
}
```

### Get Paginated Graph Data

**Endpoint**: `GET /api/graph/data/paginated`

Returns paginated graph data for large graphs.

**Query Parameters**:
- `page` (number): Page number (1-indexed)
- `page-size` (number): Items per page (default: 100)

**Response**:
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

**cURL Example**:
```bash
curl "http://localhost:9090/api/graph/data/paginated?page=1&page-size=50"
```

### Update Graph

**Endpoint**: `POST /api/graph/update`

Triggers graph update by fetching and processing new files from GitHub.

**Response**:
```json
{
  "success": true,
  "message": "Graph updated with 5 new files"
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:9090/api/graph/update
```

### Refresh Graph

**Endpoint**: `POST /api/graph/refresh`

Returns current graph state without modification.

**Response**:
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

### Auto-Balance Notifications

**Endpoint**: `GET /api/graph/auto-balance-notifications`

Returns physics auto-balance events and notifications.

**Query Parameters**:
- `since` (number): Unix timestamp to filter notifications

**Response**:
```json
{
  "success": true,
  "notifications": [
    {
      "timestamp": 1699012345,
      "type": "auto-balance-triggered",
      "message": "Physics simulation auto-balanced",
      "details": {
        "previousEnergy": 156.7,
        "newEnergy": 45.2
      }
    }
  ]
}
```

---

## Ontology Endpoints

### Get Class Hierarchy

**Endpoint**: `GET /api/ontology/hierarchy`

Returns complete class hierarchy with parent-child relationships, depth information, and descendant counts.

**Query Parameters**:
- `ontology-id` (string, optional): Specific ontology identifier (default: "default")
- `max-depth` (number, optional): Maximum depth to traverse

**Response**:
```json
{
  "rootClasses": [
    "http://example.org/Person",
    "http://example.org/Organization"
  ],
  "hierarchy": {
    "http://example.org/Person": {
      "iri": "http://example.org/Person",
      "label": "Person",
      "parentIri": null,
      "childrenIris": [
        "http://example.org/Student",
        "http://example.org/Teacher"
      ],
      "nodeCount": 5,
      "depth": 0
    },
    "http://example.org/Student": {
      "iri": "http://example.org/Student",
      "label": "Student",
      "parentIri": "http://example.org/Person",
      "childrenIris": [
        "http://example.org/GraduateStudent"
      ],
      "nodeCount": 2,
      "depth": 1
    }
  }
}
```

**cURL Example**:
```bash
curl "http://localhost:9090/api/ontology/hierarchy?ontology-id=default&max-depth=5"
```

**TypeScript Example**:
```typescript
interface ClassHierarchy {
  rootClasses: string[];
  hierarchy: Record<string, ClassNode>;
}

interface ClassNode {
  iri: string;
  label: string;
  parentIri?: string;
  childrenIris: string[];
  nodeCount: number;
  depth: number;
}

async function fetchHierarchy(ontologyId?: string): Promise<ClassHierarchy> {
  const params = new URLSearchParams();
  if (ontologyId) params.set('ontology-id', ontologyId);

  const response = await fetch(`/api/ontology/hierarchy?${params}`);
  return response.json();
}
```

### Load Ontology Axioms

**Endpoint**: `POST /api/ontology/load` or `POST /api/ontology/load-axioms`

Loads ontology axioms from a file path, URL, or inline content.

**Request**:
```json
{
  "source": "https://example.org/ontology.owl",
  "format": "rdf/xml",
  "validateImmediately": true
}
```

**Response**:
```json
{
  "ontologyId": "ontology-uuid-123",
  "loadedAt": "2025-11-03T12:00:00Z",
  "axiomCount": 150,
  "loadingTimeMs": 450,
  "validationJobId": "job-uuid-456"
}
```

**cURL Example**:
```bash
curl -X POST http://localhost:9090/api/ontology/load \
  -H "Content-Type: application/json" \
  -d '{"source": "https://example.org/ontology.owl", "format": "rdf/xml"}'
```

### Validate Ontology

**Endpoint**: `POST /api/ontology/validate`

Triggers ontology validation with specified mode.

**Request**:
```json
{
  "ontologyId": "ontology-123",
  "mode": "full",
  "priority": 5,
  "enableWebsocketUpdates": true,
  "clientId": "client-abc"
}
```

**Validation Modes**:
- `quick`: Fast validation (structural checks only)
- `full`: Complete validation (includes reasoning)
- `incremental`: Only validates changes

**Response**:
```json
{
  "jobId": "job-uuid-789",
  "status": "queued",
  "estimatedCompletion": "2025-11-03T12:00:30Z",
  "queuePosition": 1,
  "websocketUrl": "/api/ontology/ws?client-id=client-abc"
}
```

### Get Validation Report

**Endpoint**: `GET /api/ontology/report` or `GET /api/ontology/reports/{id}`

Retrieves validation report by ID.

**Response**:
```json
{
  "id": "report-123",
  "ontologyId": "ontology-123",
  "timestamp": "2025-11-03T12:00:00Z",
  "violations": [
    {
      "type": "cardinality",
      "severity": "error",
      "message": "Person class violates cardinality constraint",
      "location": "http://example.org/Person"
    }
  ],
  "inferredTriples": [...],
  "statistics": {
    "totalClasses": 50,
    "totalProperties": 30,
    "totalViolations": 2
  }
}
```

### Apply Inferences

**Endpoint**: `POST /api/ontology/apply`

Applies ontology inference rules to RDF triples.

**Request**:
```json
{
  "rdfTriples": [
    {
      "subject": "http://example.org/john",
      "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
      "object": "http://example.org/Person",
      "isLiteral": false
    }
  ],
  "maxDepth": 3,
  "updateGraph": true
}
```

**Response**:
```json
{
  "inputCount": 1,
  "inferredTriples": [
    {
      "subject": "http://example.org/john",
      "predicate": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
      "object": "http://example.org/Agent",
      "isLiteral": false
    }
  ],
  "processingTimeMs": 45,
  "graphUpdated": true
}
```

### List OWL Classes

**Endpoint**: `GET /api/owl/classes`

Lists all OWL classes in the ontology.

**Response**:
```json
{
  "classes": [
    {
      "iri": "http://example.org/Person",
      "label": "Person",
      "parentClasses": [],
      "equivalentClasses": [],
      "disjointWith": ["http://example.org/Organization"]
    }
  ],
  "count": 50
}
```

### Get OWL Class Details

**Endpoint**: `GET /api/owl/classes/{iri}`

Retrieves detailed information about a specific OWL class.

### Update Ontology Mapping

**Endpoint**: `POST /api/ontology/mapping`

Updates validation configuration and mapping rules.

**Request**:
```json
{
  "config": {
    "enableReasoning": true,
    "reasoningTimeoutSeconds": 60,
    "enableInference": true,
    "maxInferenceDepth": 5,
    "enableCaching": true,
    "cacheTtlSeconds": 3600,
    "validateCardinality": true,
    "validateDomainsRanges": true,
    "validateDisjointClasses": true
  },
  "applyToAll": false
}
```

### Ontology Health Check

**Endpoint**: `GET /api/ontology/health`

Returns ontology system health metrics.

**Response**:
```json
{
  "status": "healthy",
  "health": {
    "loadedOntologies": 5,
    "cachedReports": 10,
    "validationQueueSize": 2,
    "lastValidation": "2025-11-03T11:55:00Z",
    "cacheHitRate": 0.85,
    "avgValidationTimeMs": 1500.0,
    "activeJobs": 1,
    "memoryUsageMb": 256.0
  },
  "ontologyValidationEnabled": true,
  "timestamp": "2025-11-03T12:00:00Z"
}
```

### Clear Ontology Caches

**Endpoint**: `DELETE /api/ontology/cache`

Clears all ontology-related caches.

### Ontology WebSocket

**Endpoint**: `GET /api/ontology/ws`

WebSocket endpoint for real-time ontology validation updates.

**Query Parameters**:
- `client-id` (string, optional): Client identifier

---

## Physics Endpoints

### Start Simulation

**Endpoint**: `POST /api/physics/start`

Starts the physics simulation.

**Response**:
```json
{
  "status": "running",
  "timestamp": "2025-11-03T12:00:00Z"
}
```

### Stop Simulation

**Endpoint**: `POST /api/physics/stop`

Stops the physics simulation.

### Get Simulation Status

**Endpoint**: `GET /api/physics/status`

Returns current physics simulation status.

**Response**:
```json
{
  "isRunning": true,
  "isSettled": false,
  "stableFrameCount": 45,
  "kineticEnergy": 12.5,
  "fps": 60,
  "nodeCount": 150
}
```

### Optimize Layout

**Endpoint**: `POST /api/physics/optimize`

Triggers physics-based layout optimization.

### Apply Forces

**Endpoint**: `POST /api/physics/forces/apply`

Applies custom forces to nodes.

**Request**:
```json
{
  "forces": [
    {
      "nodeId": 1,
      "force": {"x": 10.0, "y": 0.0, "z": 0.0}
    }
  ]
}
```

### Pin/Unpin Nodes

**Endpoints**:
- `POST /api/physics/nodes/pin`
- `POST /api/physics/nodes/unpin`

Pins or unpins nodes in fixed positions.

**Request**:
```json
{
  "nodeIds": [1, 2, 3],
  "position": {"x": 0.0, "y": 0.0, "z": 0.0}
}
```

### Update Physics Parameters

**Endpoint**: `POST /api/physics/parameters`

Updates physics simulation parameters.

**Request**:
```json
{
  "gravity": 0.1,
  "charge": -30.0,
  "linkStrength": 0.5,
  "friction": 0.9,
  "theta": 0.8
}
```

### Reset Simulation

**Endpoint**: `POST /api/physics/reset`

Resets physics simulation to initial state.

---

## File Management

### Process Files

**Endpoint**: `POST /api/files/process`

Fetches and processes markdown files from GitHub repository.

**Response**:
```json
{
  "status": "success",
  "processedFiles": [
    "concepts/example.md",
    "guides/tutorial.md"
  ]
}
```

### Get File Content

**Endpoint**: `GET /api/files/get-content/{filename}`

Retrieves content of a specific markdown file.

**Response**: Raw file content (text/markdown)

### Refresh Graph from Files

**Endpoint**: `POST /api/files/refresh-graph`

Refreshes graph data from current file state.

### Update Graph from Files

**Endpoint**: `POST /api/files/update-graph`

Updates graph structure from processed files.

---

## Bots & Swarm Endpoints

### Get Bots Data

**Endpoint**: `GET /api/bots/data`

Retrieves current bots/swarm configuration.

### Update Bots Graph

**Endpoint**: `POST /api/bots/update`

Updates bots graph structure.

### Initialize Swarm

**Endpoint**: `POST /api/bots/initialize-swarm`

Initializes hive mind swarm coordination.

**Request**:
```json
{
  "topology": "hierarchical",
  "maxAgents": 10,
  "strategy": "adaptive"
}
```

### Get Connection Status

**Endpoint**: `GET /api/bots/status`

Returns bots/swarm connection status.

### List Agents

**Endpoint**: `GET /api/bots/agents`

Lists all active agents in the swarm.

### Spawn Hybrid Agent

**Endpoint**: `POST /api/bots/spawn-agent-hybrid`

Spawns a new hybrid agent with specified capabilities.

**Request**:
```json
{
  "type": "researcher",
  "capabilities": ["analysis", "synthesis"],
  "priority": "high"
}
```

### Remove Task

**Endpoint**: `DELETE /api/bots/remove-task/{id}`

Removes a specific task from the swarm queue.

---

## Analytics Endpoints

### Detect Communities

**Endpoint**: `POST /api/semantic/communities`

Detects community structures in the graph.

**Request**:
```json
{
  "algorithm": "louvain",
  "resolution": 1.0
}
```

**Response**:
```json
{
  "communities": [
    {
      "id": "community-1",
      "nodeIds": [1, 2, 3, 4],
      "density": 0.75
    }
  ],
  "modularity": 0.82
}
```

### Compute Centrality

**Endpoint**: `POST /api/semantic/centrality`

Computes centrality metrics for nodes.

**Request**:
```json
{
  "metric": "betweenness",
  "normalized": true
}
```

**Centrality Metrics**:
- `degree`: Degree centrality
- `betweenness`: Betweenness centrality
- `closeness`: Closeness centrality
- `eigenvector`: Eigenvector centrality
- `pagerank`: PageRank

### Shortest Path

**Endpoint**: `POST /api/semantic/shortest-path`

Computes shortest path between nodes.

**Request**:
```json
{
  "sourceId": 1,
  "targetId": 10,
  "weighted": true
}
```

### Graph Statistics

**Endpoint**: `GET /api/semantic/statistics`

Returns comprehensive graph statistics.

**Response**:
```json
{
  "nodeCount": 150,
  "edgeCount": 320,
  "density": 0.028,
  "averageDegree": 4.27,
  "diameter": 8,
  "averagePathLength": 3.2,
  "clusteringCoefficient": 0.45
}
```

### Clustering Operations

Configure clustering:
- `POST /api/clustering/configure`

Start clustering:
- `POST /api/clustering/start`

Get status:
- `GET /api/clustering/status`

Get results:
- `GET /api/clustering/results`

Export assignments:
- `POST /api/clustering/export`

---

## Workspace Management

### List Workspaces

**Endpoint**: `GET /api/workspace/list`

Lists all workspaces for the current user.

**Response**:
```json
{
  "workspaces": [
    {
      "id": "workspace-123",
      "name": "My Research",
      "description": "Research project workspace",
      "isFavorite": true,
      "isArchived": false,
      "createdAt": "2025-10-01T10:00:00Z",
      "updatedAt": "2025-11-03T12:00:00Z"
    }
  ]
}
```

### Create Workspace

**Endpoint**: `POST /api/workspace/create`

Creates a new workspace.

**Request**:
```json
{
  "name": "New Workspace",
  "description": "Workspace description",
  "settings": {}
}
```

### Get Workspace

**Endpoint**: `GET /api/workspace/{id}`

Retrieves workspace details by ID.

### Update Workspace

**Endpoint**: `PUT /api/workspace/{id}`

Updates workspace information.

### Delete Workspace

**Endpoint**: `DELETE /api/workspace/{id}`

Deletes a workspace.

### Toggle Favorite

**Endpoint**: `POST /api/workspace/{id}/favorite`

Toggles workspace favorite status.

### Archive Workspace

**Endpoint**: `POST /api/workspace/{id}/archive`

Archives or unarchives a workspace.

### Workspace Count

**Endpoint**: `GET /api/workspace/count`

Returns total workspace count.

---

## Advanced Features

### Graph Export

**Export Graph**:
- `POST /api/graph-export/export`

**Share Graph**:
- `POST /api/graph-export/share`

**Get Shared Graph**:
- `GET /api/graph-export/shared/{id}`

**Delete Shared Graph**:
- `DELETE /api/graph-export/shared/{id}`

**Publish Graph**:
- `POST /api/graph-export/publish`

**Export Statistics**:
- `GET /api/graph-export/stats`

### Graph State Management

**Get State**:
- `GET /api/graph-state/state`

**Get Statistics**:
- `GET /api/graph-state/statistics`

**Node Operations**:
- `POST /api/graph-state/nodes` - Add node
- `GET /api/graph-state/nodes/{id}` - Get node
- `PUT /api/graph-state/nodes/{id}` - Update node
- `DELETE /api/graph-state/nodes/{id}` - Remove node

**Edge Operations**:
- `POST /api/graph-state/edges` - Add edge
- `PUT /api/graph-state/edges/{id}` - Update edge

**Batch Operations**:
- `POST /api/graph-state/positions/batch` - Batch update positions

### RAGFlow Integration

**Create Session**:
- `POST /api/ragflow/session`

**Send Message**:
- `POST /api/ragflow/message`

**Chat**:
- `POST /api/ragflow/chat`

**Session History**:
- `GET /api/ragflow/history/{session-id}`

### Constraints Management

**Define Constraints**:
- `POST /api/constraints/define`

**Apply Constraints**:
- `POST /api/constraints/apply`

**Remove Constraints**:
- `POST /api/constraints/remove`

**List Constraints**:
- `GET /api/constraints/list`

**Validate Definition**:
- `POST /api/constraints/validate`

### Validation Testing

**Validate Payload**:
- `POST /api/validation/test/{type}`

**Validation Statistics**:
- `GET /api/validation/stats`

### Multi-MCP WebSocket

**WebSocket Connection**:
- `GET /api/multi-mcp/ws`

**MCP Server Status**:
- `GET /api/multi-mcp/status`

**Refresh Discovery**:
- `POST /api/multi-mcp/refresh`

### Admin Operations

**Trigger GitHub Sync**:
- `POST /api/admin/sync`

Triggers GitHub repository synchronization to import ontology files.

**Environment Variables**:
- `FORCE-FULL-SYNC=1` - Bypass SHA1 filtering, process all files

**Response**:
```json
{
  "status": "success",
  "filesProcessed": 50,
  "nodesCreated": 45,
  "edgesCreated": 12
}
```

### Health Monitoring

**Unified Health Check**:
- `GET /api/health`

**Physics Health**:
- `GET /api/health/physics`

**MCP Relay**:
- `POST /api/health/mcp-relay/start`
- `GET /api/health/mcp-relay/logs`

---

## WebSocket Protocol

VisionFlow uses a **36-byte binary WebSocket protocol** for real-time graph updates.

### Connection

```typescript
const ws = new WebSocket('ws://localhost:9090/ws?token=YOUR-JWT-TOKEN');
ws.binaryType = 'arraybuffer';
```

### Binary Message Format

Each node update is exactly 36 bytes:

```
Byte Layout (Little-Endian):
┌──────────┬───────────────────────────────────────────┐
│ Offset   │ Field                                     │
├──────────┼───────────────────────────────────────────┤
│ [0-3]    │ Node ID (u32)                             │
│ [4-7]    │ X position (f32)                          │
│ [8-11]   │ Y position (f32)                          │
│ [12-15]  │ Z position (f32)                          │
│ [16-19]  │ VX velocity (f32)                         │
│ [20-23]  │ VY velocity (f32)                         │
│ [24-27]  │ VZ velocity (f32)                         │
│ [28-31]  │ Mass (f32)                                │
│ [32-35]  │ Charge (f32)                              │
└──────────┴───────────────────────────────────────────┘
```

### Parsing Example

```typescript
class BinaryProtocolParser {
  private view: DataView;

  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer);
  }

  parseNodeUpdates(): NodeUpdate[] {
    const nodeCount = this.view.byteLength / 36;
    const updates: NodeUpdate[] = [];

    for (let i = 0; i < nodeCount; i++) {
      const offset = i * 36;

      updates.push({
        id: this.view.getUint32(offset + 0, true),
        position: [
          this.view.getFloat32(offset + 4, true),
          this.view.getFloat32(offset + 8, true),
          this.view.getFloat32(offset + 12, true),
        ],
        velocity: [
          this.view.getFloat32(offset + 16, true),
          this.view.getFloat32(offset + 20, true),
          this.view.getFloat32(offset + 24, true),
        ],
        mass: this.view.getFloat32(offset + 28, true),
        charge: this.view.getFloat32(offset + 32, true),
      });
    }

    return updates;
  }
}
```

### Performance

| Metric | Binary V2 | JSON V1 | Improvement |
|--------|-----------|---------|-------------|
| **Message Size** | 3.6 MB | 18 MB | 80% smaller |
| **Parse Time** | 0.8 ms | 12 ms | 15x faster |
| **Network Latency** | <10 ms | 45 ms | 4.5x faster |
| **CPU Usage** | 5% | 28% | 5.6x lower |

**See**: [Complete WebSocket Documentation](./03-websocket.md)

---

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION-ERROR",
    "message": "Invalid input",
    "details": [
      {
        "field": "name",
        "message": "Name is required"
      }
    ]
  }
}
```

### Ontology-Specific Errors

```json
{
  "error": "Ontology validation feature is disabled",
  "code": "FEATURE-DISABLED",
  "details": {
    "message": "Enable the ontology-validation feature flag to use this endpoint"
  },
  "timestamp": "2025-11-03T12:00:00Z",
  "traceId": "uuid-here"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200 OK` | Success |
| `201 Created` | Resource created |
| `202 Accepted` | Request accepted (async processing) |
| `301 Moved Permanently` | Endpoint relocated |
| `400 Bad Request` | Invalid input |
| `401 Unauthorized` | Missing/invalid authentication |
| `403 Forbidden` | Insufficient permissions |
| `404 Not Found` | Resource not found |
| `429 Too Many Requests` | Rate limit exceeded |
| `500 Internal Server Error` | Server error |
| `503 Service Unavailable` | Feature disabled or service unavailable |

---

## Database Architecture

The API uses a **unified database architecture** with `unified.db` containing all domain tables:

- `nodes` - Knowledge graph nodes
- `edges` - Relationships between nodes
- `owl-classes` - OWL ontology classes
- `owl-properties` - OWL ontology properties
- `github-sync-state` - Synchronization tracking
- `workspaces` - User workspaces
- `validation-reports` - Ontology validation results

---

## Rate Limiting

**Default Limits**:
- 100 requests per minute per IP
- 1000 requests per hour per user

**Rate Limit Headers**:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1699012800
```

---

## Pagination

Paginated endpoints follow this structure:

**Request**:
```
GET /api/resource?page=1&page-size=50
```

**Response**:
```json
{
  "data": [...],
  "pagination": {
    "total": 1000,
    "limit": 50,
    "offset": 0,
    "totalPages": 20,
    "currentPage": 1
  }
}
```

---

## Changelog

### v0.1.0 (2025-11-03)
- Initial unified API documentation
- Binary WebSocket protocol (80% bandwidth reduction)
- Complete ontology validation system
- Physics simulation endpoints
- Workspace management
- Graph export and sharing
- RAGFlow integration
- Multi-MCP WebSocket support

---

## Support & Resources

- **WebSocket Protocol**: [03-websocket.md](./03-websocket.md)
- **Architecture Overview**: [../../concepts/architecture/00-ARCHITECTURE-overview.md](../../concepts/architecture/00-ARCHITECTURE-overview.md)
- **GitHub Repository**: Contact administrator for access
- **Issue Tracker**: Internal JIRA

---

**Last Updated**: November 3, 2025
**Maintainer**: VisionFlow API Team
**Version**: 1.0.0
