---
title: VisionFlow REST API Complete Reference
description: Complete REST API documentation for VisionFlow including authentication, graph, ontology, physics, file management, analytics, workspace, and Solid integration endpoints.
category: reference
tags:
  - api
  - rest
  - backend
  - visionflow
updated-date: 2025-01-29
difficulty-level: intermediate
---

# VisionFlow REST API Complete Reference

**Version**: 1.0
**Base URL**: `http://localhost:9090/api`
**Last Updated**: January 29, 2025

---

## Table of Contents

1. [Authentication](#authentication)
2. [Core Endpoints](#core-endpoints)
3. [Graph Endpoints](#graph-endpoints)
4. [Ontology Endpoints](#ontology-endpoints)
5. [Physics Endpoints](#physics-endpoints)
6. [File Management](#file-management)
7. [Bots and Swarm Endpoints](#bots--swarm-endpoints)
8. [Analytics Endpoints](#analytics-endpoints)
9. [Workspace Management](#workspace-management)
10. [Advanced Features](#advanced-features)
11. [Solid Integration](#solid-integration-endpoints)
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

### Update Graph

**Endpoint**: `POST /api/graph/update`

Triggers graph update by fetching and processing new files from GitHub.

### Refresh Graph

**Endpoint**: `POST /api/graph/refresh`

Returns current graph state without modification.

### Auto-Balance Notifications

**Endpoint**: `GET /api/graph/auto-balance-notifications`

Returns physics auto-balance events and notifications.

**Query Parameters**:
- `since` (number): Unix timestamp to filter notifications

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
    }
  }
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

### Validate Ontology

**Endpoint**: `POST /api/ontology/validate`

Triggers ontology validation with specified mode.

**Validation Modes**:
- `quick`: Fast validation (structural checks only)
- `full`: Complete validation (includes reasoning)
- `incremental`: Only validates changes

### Get Validation Report

**Endpoint**: `GET /api/ontology/report` or `GET /api/ontology/reports/{id}`

Retrieves validation report by ID.

### Apply Inferences

**Endpoint**: `POST /api/ontology/apply`

Applies ontology inference rules to RDF triples.

### List OWL Classes

**Endpoint**: `GET /api/owl/classes`

Lists all OWL classes in the ontology.

### Get OWL Class Details

**Endpoint**: `GET /api/owl/classes/{iri}`

Retrieves detailed information about a specific OWL class.

### Update Ontology Mapping

**Endpoint**: `POST /api/ontology/mapping`

Updates validation configuration and mapping rules.

### Ontology Health Check

**Endpoint**: `GET /api/ontology/health`

Returns ontology system health metrics.

### Clear Ontology Caches

**Endpoint**: `DELETE /api/ontology/cache`

Clears all ontology-related caches.

### Ontology WebSocket

**Endpoint**: `GET /api/ontology/ws`

WebSocket endpoint for real-time ontology validation updates.

---

## Physics Endpoints

### Start Simulation

**Endpoint**: `POST /api/physics/start`

Starts the physics simulation.

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

### Get File Content

**Endpoint**: `GET /api/files/get-content/{filename}`

Retrieves content of a specific markdown file.

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

### Compute Centrality

**Endpoint**: `POST /api/semantic/centrality`

Computes centrality metrics for nodes.

**Centrality Metrics**:
- `degree`: Degree centrality
- `betweenness`: Betweenness centrality
- `closeness`: Closeness centrality
- `eigenvector`: Eigenvector centrality
- `pagerank`: PageRank

### Shortest Path

**Endpoint**: `POST /api/semantic/shortest-path`

Computes shortest path between nodes.

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

- `POST /api/clustering/configure` - Configure clustering
- `POST /api/clustering/start` - Start clustering
- `GET /api/clustering/status` - Get status
- `GET /api/clustering/results` - Get results
- `POST /api/clustering/export` - Export assignments

---

## Workspace Management

### List Workspaces

**Endpoint**: `GET /api/workspace/list`

Lists all workspaces for the current user.

### Create Workspace

**Endpoint**: `POST /api/workspace/create`

Creates a new workspace.

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

- `POST /api/graph-export/export` - Export Graph
- `POST /api/graph-export/share` - Share Graph
- `GET /api/graph-export/shared/{id}` - Get Shared Graph
- `DELETE /api/graph-export/shared/{id}` - Delete Shared Graph
- `POST /api/graph-export/publish` - Publish Graph
- `GET /api/graph-export/stats` - Export Statistics

### Graph State Management

- `GET /api/graph-state/state` - Get State
- `GET /api/graph-state/statistics` - Get Statistics
- `POST /api/graph-state/nodes` - Add node
- `GET /api/graph-state/nodes/{id}` - Get node
- `PUT /api/graph-state/nodes/{id}` - Update node
- `DELETE /api/graph-state/nodes/{id}` - Remove node
- `POST /api/graph-state/edges` - Add edge
- `PUT /api/graph-state/edges/{id}` - Update edge
- `POST /api/graph-state/positions/batch` - Batch update positions

### RAGFlow Integration

- `POST /api/ragflow/session` - Create Session
- `POST /api/ragflow/message` - Send Message
- `POST /api/ragflow/chat` - Chat
- `GET /api/ragflow/history/{session-id}` - Session History

### Constraints Management

- `POST /api/constraints/define` - Define Constraints
- `POST /api/constraints/apply` - Apply Constraints
- `POST /api/constraints/remove` - Remove Constraints
- `GET /api/constraints/list` - List Constraints
- `POST /api/constraints/validate` - Validate Definition

### Multi-MCP WebSocket

- `GET /api/multi-mcp/ws` - WebSocket Connection
- `GET /api/multi-mcp/status` - MCP Server Status
- `POST /api/multi-mcp/refresh` - Refresh Discovery

### Admin Operations

**Trigger GitHub Sync**: `POST /api/admin/sync`

Triggers GitHub repository synchronization to import ontology files.

**Environment Variables**:
- `FORCE-FULL-SYNC=1` - Bypass SHA1 filtering, process all files

### Health Monitoring

- `GET /api/health` - Unified Health Check
- `GET /api/health/physics` - Physics Health
- `POST /api/health/mcp-relay/start` - Start MCP Relay
- `GET /api/health/mcp-relay/logs` - MCP Relay Logs

---

## Solid Integration Endpoints

VisionFlow integrates with Solid pods via the JSON Solid Server (JSS) sidecar for decentralized data storage and Linked Data Platform (LDP) compliance.

### GET /solid/pods

List available Solid pods for the authenticated user.

**Response** (200 OK):
```json
{
  "pods": [
    {
      "id": "pod-user123",
      "webId": "https://visionflow.example.com/pods/user123/profile/card#me",
      "storage": "https://visionflow.example.com/pods/user123/",
      "createdAt": "2025-11-03T10:00:00Z",
      "quota": {
        "used": 52428800,
        "total": 1073741824
      }
    }
  ]
}
```

### GET /solid/pods/{podId}/graph

Retrieve graph data from a Solid pod in RDF format.

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `format` | string | No | `turtle` | Output format: `turtle`, `jsonld`, `ntriples` |
| `container` | string | No | `/` | Container path within pod |

### POST /solid/pods/{podId}/graph

Create a new resource in the pod's graph container.

### PUT /solid/pods/{podId}/graph/{resource}

Replace an existing resource.

### PATCH /solid/pods/{podId}/graph/{resource}

Partially update a resource using SPARQL Update or N3 Patch.

### DELETE /solid/pods/{podId}/graph/{resource}

Delete a resource from the pod.

### GET /solid/ws

WebSocket endpoint for real-time Solid notifications.

### POST /solid/sync

Trigger synchronization between Neo4j graph and Solid pod.

**Sync Directions**:
| Direction | Description |
|-----------|-------------|
| `neo4j-to-solid` | Export graph data to Solid pod |
| `solid-to-neo4j` | Import pod data into graph |
| `bidirectional` | Two-way sync with conflict resolution |

### GET /solid/sync/{jobId}

Check synchronization job status.

### Authentication for Solid Endpoints

Solid endpoints support two authentication methods:

**1. JWT Bearer Token** (VisionFlow session):
```http
Authorization: Bearer <visionflow_jwt>
```

**2. NIP-98 Nostr Authentication**:
```http
Authorization: Nostr <base64_encoded_event>
```

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

## Related Documentation

- [WebSocket Protocol Reference](../protocols/websocket-binary-v2.md)
- [WebSocket Endpoints](./websocket-endpoints.md)
- [Database Schema Reference](../database/schema-catalog.md)
- [Protocol Reference](../protocols/README.md)

---

**Last Updated**: January 29, 2025
**Maintainer**: VisionFlow API Team
**Version**: 1.0.0
