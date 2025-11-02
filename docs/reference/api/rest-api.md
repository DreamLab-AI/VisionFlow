# REST API Reference

**Version:** 3.1.0
**Last Updated:** 2025-10-25
**Base URL:** `http://localhost:3030` (development)

---

## Authentication

VisionFlow implements **three-tier authentication**:

### Public Access (No Auth Required)
- Health check endpoints
- Documentation endpoints
- Public graph views (read-only)

### User Authentication (JWT Token)
- Standard graph operations (read/write)
- User settings management
- Physics simulation control

**Header Format:**
```
Authorization: Bearer <jwt_token>
```

### Developer Authentication (API Key)
- System configuration
- Developer settings
- Administrative operations

**Header Format:**
```
X-API-Key: <api_key>
```

---

## Settings API

### Get All Settings
**Endpoint:** `GET /api/settings`
**Authentication:** User
**CQRS Handler:** `GetAllSettingsQuery`

### Update All Settings
**Endpoint:** `POST /api/settings`
**Authentication:** User
**CQRS Handler:** `SaveAllSettingsDirective`

### Get Single Setting
**Endpoint:** `GET /api/settings/path/{path}`
**Authentication:** User
**CQRS Handler:** `GetSettingQuery`

### Get Physics Settings
**Endpoint:** `GET /api/settings/physics/{graph_name}`
**Authentication:** User
**Parameters:** `graph_name`: `logseq` | `visionflow` | `ontology` | `default`

### Update Physics Settings
**Endpoint:** `PUT /api/settings/physics/{graph_name}`
**Authentication:** User
**CQRS Handler:** `UpdatePhysicsSettingsDirective`

---

## Knowledge Graph API

### Get Full Graph
**Endpoint:** `GET /api/graph`
**Authentication:** User
**CQRS Handler:** `GetGraphQuery`

### Get Complete Graph Data
**Endpoint:** `GET /api/graph/data`
**Authentication:** User
**Description:** Returns complete graph data with nodes, edges, and metadata

**Response Structure:**
```json
{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "position": {"x": 0.0, "y": 0.0, "z": 0.0},
      "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
      "metadata": {
        "owl_class_iri": "string",
        "file_source": "string"
      },
      "visual": {
        "color": "#hexcolor",
        "size": 1.0
      }
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id",
      "relationship_type": "string"
    }
  ],
  "metadata": {
    "node_count": 50,
    "edge_count": 12,
    "last_updated": "2025-11-02T13:00:00Z"
  }
}
```

**Example:** Typical response: 50 nodes, 12 edges (~17KB JSON)

### Add Node
**Endpoint:** `POST /api/graph/node`
**Authentication:** User
**CQRS Handler:** `AddNodeDirective`

### Update Node
**Endpoint:** `PUT /api/graph/node/{id}`
**Authentication:** User
**CQRS Handler:** `UpdateNodeDirective`

### Delete Node
**Endpoint:** `DELETE /api/graph/node/{id}`
**Authentication:** User
**CQRS Handler:** `RemoveNodeDirective`

### Add Edge
**Endpoint:** `POST /api/graph/edge`
**Authentication:** User
**CQRS Handler:** `AddEdgeDirective`

### Delete Edge
**Endpoint:** `DELETE /api/graph/edge/{id}`
**Authentication:** User
**CQRS Handler:** `RemoveEdgeDirective`

### Query Nodes
**Endpoint:** `POST /api/graph/query`
**Authentication:** User
**CQRS Handler:** `QueryNodesQuery`

### Get Graph Statistics
**Endpoint:** `GET /api/graph/statistics`
**Authentication:** User
**CQRS Handler:** `GetGraphStatisticsQuery`

---

## Admin API

### Trigger GitHub Synchronization
**Endpoint:** `POST /api/admin/sync`
**Authentication:** Developer
**Description:** Triggers GitHub repository synchronization to import ontology files

**Environment Variables:**
- `FORCE_FULL_SYNC=1` - Bypass SHA1 filtering, process all files

**Response:**
```json
{
  "status": "success",
  "files_processed": 50,
  "nodes_created": 45,
  "edges_created": 12
}
```

---

## Ontology API

### Get Ontology Graph
**Endpoint:** `GET /api/ontology/graph`
**Authentication:** User
**CQRS Handler:** `GetOntologyGraphQuery`

### Add OWL Class
**Endpoint:** `POST /api/ontology/class`
**Authentication:** Developer
**CQRS Handler:** `AddOwlClassDirective`

### Get OWL Class
**Endpoint:** `GET /api/ontology/class/{iri}`
**Authentication:** User
**CQRS Handler:** `GetOwlClassQuery`

### Add OWL Property
**Endpoint:** `POST /api/ontology/property`
**Authentication:** Developer
**CQRS Handler:** `AddOwlPropertyDirective`

### Run Inference
**Endpoint:** `POST /api/ontology/infer`
**Authentication:** Developer
**CQRS Handler:** `RunInferenceDirective`

---

## Physics API

### Get Simulation State
**Endpoint:** `GET /api/physics/state`
**Authentication:** User
**CQRS Handler:** `GetSimulationStateQuery`

### Update Simulation Parameters
**Endpoint:** `PUT /api/physics/params`
**Authentication:** User
**CQRS Handler:** `UpdateSimulationParamsDirective`

### Apply Constraints
**Endpoint:** `POST /api/physics/constraints`
**Authentication:** User
**CQRS Handler:** `ApplyConstraintsDirective`

### Reset Simulation
**Endpoint:** `POST /api/physics/reset`
**Authentication:** User
**CQRS Handler:** `ResetSimulationDirective`

---

## Error Handling

### Standard Error Response

All API errors follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "additional context"
    }
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Request validation failed |
| `NOT_FOUND` | 404 | Resource does not exist |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `CONFLICT` | 409 | Resource already exists |
| `INTERNAL_ERROR` | 500 | Server-side error |
| `DATABASE_ERROR` | 500 | Database operation failed |
| `GPU_ERROR` | 503 | GPU computation failed |

---

## Rate Limiting

### Rate Limit Headers

All responses include rate limit information:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1729591200
```

### Rate Limits by Tier

| Tier | Requests/Minute | Burst |
|------|-----------------|-------|
| Public | 60 | 10 |
| User | 1000 | 100 |
| Developer | 5000 | 500 |

---

## Health & Monitoring

### Unified Health Check
**Endpoint:** `GET /api/health`
**Authentication:** Public

### Physics Simulation Health
**Endpoint:** `GET /api/health/physics`
**Authentication:** Public

---

## Database Architecture

VisionFlow uses a **unified database architecture** with `unified.db`:

### Database Schema
- **nodes** - Knowledge graph nodes with positions and metadata
- **edges** - Relationships between nodes
- **owl_classes** - OWL ontology classes
- **owl_properties** - OWL ontology properties
- **github_sync_state** - GitHub synchronization tracking (SHA1 hashes)

All domain data is consolidated in a single database for consistency and performance.

---

**Document Maintained By:** VisionFlow API Team
**Last Review:** 2025-11-02
**Next Review:** 2025-12-02
