# VisionFlow API Reference

**Version:** 3.0.0
**Last Updated:** 2025-10-22
**Base URL:** `http://localhost:8080` (development)

---

## Table of Contents

1. [Authentication](#authentication)
2. [Settings API](#settings-api)
3. [Knowledge Graph API](#knowledge-graph-api)
4. [Ontology API](#ontology-api)
5. [Physics API](#physics-api)
6. [WebSocket Protocol](#websocket-protocol)
7. [Binary Protocol Specification](#binary-protocol-specification)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)

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

**Response:**
```json
{
  "application": {
    "theme": "dark",
    "language": "en"
  },
  "visualisation": {
    "graphs": {
      "logseq": { "nodes": {...}, "edges": {...} },
      "visionflow": { "nodes": {...}, "edges": {...} }
    }
  },
  "developer": {
    "debug_mode": false,
    "log_level": "info"
  }
}
```

### Update All Settings
**Endpoint:** `POST /api/settings`
**Authentication:** User
**CQRS Handler:** `SaveAllSettingsDirective`

**Request Body:**
```json
{
  "application": {...},
  "visualisation": {...},
  "developer": {...}
}
```

### Get Single Setting
**Endpoint:** `GET /api/settings/path/{path}`
**Authentication:** User
**CQRS Handler:** `GetSettingQuery`

**Example:** `GET /api/settings/path/application.theme`

**Response:**
```json
{
  "value": "dark",
  "type": "string"
}
```

### Update Single Setting
**Endpoint:** `PUT /api/settings/path/{path}`
**Authentication:** User
**CQRS Handler:** `UpdateSettingDirective`

**Request Body:**
```json
{
  "value": "dark"
}
```

### Get Physics Settings
**Endpoint:** `GET /api/settings/physics/{graph_name}`
**Authentication:** User
**CQRS Handler:** `GetPhysicsSettingsQuery`

**Parameters:**
- `graph_name`: `logseq` | `visionflow` | `ontology` | `default`

**Response:**
```json
{
  "time_step": 0.016,
  "damping": 0.85,
  "repulsion_strength": 500.0,
  "attraction_strength": 0.01,
  "max_velocity": 100.0,
  "convergence_threshold": 0.001
}
```

### Update Physics Settings
**Endpoint:** `PUT /api/settings/physics/{graph_name}`
**Authentication:** User
**CQRS Handler:** `UpdatePhysicsSettingsDirective`

**Request Body:**
```json
{
  "time_step": 0.016,
  "damping": 0.85,
  "repulsion_strength": 500.0,
  "attraction_strength": 0.01
}
```

**⚠️ Important:** Physics settings are **per-graph** to prevent conflation between logseq and visionflow graphs.

### Reset Settings to Defaults
**Endpoint:** `POST /api/settings/reset`
**Authentication:** Developer
**CQRS Handler:** `ResetSettingsDirective`

### Clear Settings Cache
**Endpoint:** `POST /api/settings/cache/clear`
**Authentication:** Developer

### Settings Health Check
**Endpoint:** `GET /api/settings/health`
**Authentication:** Public

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "cache_size": 42,
  "last_update": "2025-10-22T10:30:00Z"
}
```

---

## Knowledge Graph API

### Get Full Graph
**Endpoint:** `GET /api/graph`
**Authentication:** User
**CQRS Handler:** `GetGraphQuery`

**Response:**
```json
{
  "nodes": [
    {
      "id": 1,
      "label": "Node Label",
      "type": "concept",
      "position": { "x": 10.5, "y": 20.3, "z": 5.1 },
      "metadata_id": "file_abc123",
      "properties": {}
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": 1,
      "target": 2,
      "type": "links_to",
      "weight": 1.0
    }
  ],
  "statistics": {
    "node_count": 1523,
    "edge_count": 4821,
    "last_updated": "2025-10-22T10:30:00Z"
  }
}
```

### Add Node
**Endpoint:** `POST /api/graph/node`
**Authentication:** User
**CQRS Handler:** `AddNodeDirective`

**Request Body:**
```json
{
  "label": "New Node",
  "type": "concept",
  "metadata_id": "file_xyz789",
  "position": { "x": 0, "y": 0, "z": 0 },
  "properties": {}
}
```

**Response:**
```json
{
  "node_id": 1524
}
```

### Update Node
**Endpoint:** `PUT /api/graph/node/{id}`
**Authentication:** User
**CQRS Handler:** `UpdateNodeDirective`

**Request Body:**
```json
{
  "label": "Updated Label",
  "position": { "x": 15.5, "y": 25.3, "z": 10.1 }
}
```

### Delete Node
**Endpoint:** `DELETE /api/graph/node/{id}`
**Authentication:** User
**CQRS Handler:** `RemoveNodeDirective`

### Add Edge
**Endpoint:** `POST /api/graph/edge`
**Authentication:** User
**CQRS Handler:** `AddEdgeDirective`

**Request Body:**
```json
{
  "source": 1,
  "target": 2,
  "type": "references",
  "weight": 1.0
}
```

**Response:**
```json
{
  "edge_id": "edge_4822"
}
```

### Delete Edge
**Endpoint:** `DELETE /api/graph/edge/{id}`
**Authentication:** User
**CQRS Handler:** `RemoveEdgeDirective`

### Query Nodes
**Endpoint:** `POST /api/graph/query`
**Authentication:** User
**CQRS Handler:** `QueryNodesQuery`

**Request Body:**
```json
{
  "filter": {
    "type": "concept",
    "metadata_id": "file_abc*"
  },
  "limit": 100,
  "offset": 0
}
```

### Get Graph Statistics
**Endpoint:** `GET /api/graph/statistics`
**Authentication:** User
**CQRS Handler:** `GetGraphStatisticsQuery`

**Response:**
```json
{
  "node_count": 1523,
  "edge_count": 4821,
  "average_degree": 6.3,
  "connected_components": 3,
  "last_updated": "2025-10-22T10:30:00Z"
}
```

---

## Ontology API

### Get Ontology Graph
**Endpoint:** `GET /api/ontology/graph`
**Authentication:** User
**CQRS Handler:** `GetOntologyGraphQuery`

**Response:**
```json
{
  "nodes": [...],
  "edges": [...],
  "classes": 45,
  "properties": 28,
  "axioms": 102
}
```

### Add OWL Class
**Endpoint:** `POST /api/ontology/class`
**Authentication:** Developer
**CQRS Handler:** `AddOwlClassDirective`

**Request Body:**
```json
{
  "iri": "http://example.org/ontology#Person",
  "label": "Person",
  "description": "A human being",
  "parent_classes": ["http://example.org/ontology#Agent"],
  "properties": {}
}
```

### Get OWL Class
**Endpoint:** `GET /api/ontology/class/{iri}`
**Authentication:** User
**CQRS Handler:** `GetOwlClassQuery`

**Parameters:**
- `iri`: URL-encoded IRI (e.g., `http%3A%2F%2Fexample.org%2Fontology%23Person`)

### List All OWL Classes
**Endpoint:** `GET /api/ontology/classes`
**Authentication:** User
**CQRS Handler:** `ListOwlClassesQuery`

### Add OWL Property
**Endpoint:** `POST /api/ontology/property`
**Authentication:** Developer
**CQRS Handler:** `AddOwlPropertyDirective`

**Request Body:**
```json
{
  "iri": "http://example.org/ontology#hasAge",
  "label": "has age",
  "property_type": "DataProperty",
  "domain": ["http://example.org/ontology#Person"],
  "range": ["http://www.w3.org/2001/XMLSchema#integer"]
}
```

### Run Inference
**Endpoint:** `POST /api/ontology/infer`
**Authentication:** Developer
**CQRS Handler:** `RunInferenceDirective`

**Request Body:**
```json
{
  "reasoner": "whelk",
  "options": {
    "timeout_seconds": 60,
    "max_inferences": 10000
  }
}
```

**Response:**
```json
{
  "inferred_axioms": 23,
  "inference_time_ms": 1523,
  "reasoner_version": "whelk-rs-0.4.2"
}
```

### Get Inference Results
**Endpoint:** `GET /api/ontology/inference/results`
**Authentication:** User
**CQRS Handler:** `GetInferenceResultsQuery`

**Response:**
```json
{
  "timestamp": "2025-10-22T10:35:00Z",
  "inferred_axioms": [
    {
      "axiom_type": "SubClassOf",
      "subject": "http://example.org/ontology#Student",
      "object": "http://example.org/ontology#Person"
    }
  ],
  "inference_time_ms": 1523
}
```

### Validate Ontology
**Endpoint:** `GET /api/ontology/validate`
**Authentication:** User
**CQRS Handler:** `ValidateOntologyQuery`

**Response:**
```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [
    "Class 'Example' has no instances"
  ],
  "timestamp": "2025-10-22T10:30:00Z"
}
```

---

## Physics API

### Get Simulation State
**Endpoint:** `GET /api/physics/state`
**Authentication:** User
**CQRS Handler:** `GetSimulationStateQuery`

**Response:**
```json
{
  "running": true,
  "iteration": 1523,
  "fps": 59.8,
  "kinetic_energy": 123.45,
  "potential_energy": 678.90,
  "convergence_delta": 0.0015
}
```

### Update Simulation Parameters
**Endpoint:** `PUT /api/physics/params`
**Authentication:** User
**CQRS Handler:** `UpdateSimulationParamsDirective`

**Request Body:**
```json
{
  "time_step": 0.016,
  "damping": 0.85,
  "max_iterations": 10000
}
```

### Apply Constraints
**Endpoint:** `POST /api/physics/constraints`
**Authentication:** User
**CQRS Handler:** `ApplyConstraintsDirective`

**Request Body:**
```json
{
  "constraints": [
    {
      "type": "pin",
      "node_id": 1,
      "position": { "x": 0, "y": 0, "z": 0 }
    },
    {
      "type": "distance",
      "node_id_a": 2,
      "node_id_b": 3,
      "distance": 50.0
    }
  ]
}
```

### Reset Simulation
**Endpoint:** `POST /api/physics/reset`
**Authentication:** User
**CQRS Handler:** `ResetSimulationDirective`

### Get Physics Statistics
**Endpoint:** `GET /api/physics/statistics`
**Authentication:** User
**CQRS Handler:** `GetPhysicsStatisticsQuery`

**Response:**
```json
{
  "total_steps": 15234,
  "average_step_time_ms": 2.3,
  "current_fps": 59.8,
  "gpu_memory_used_mb": 256.5,
  "gpu_utilization_percent": 87.3
}
```

---

## WebSocket Protocol

### Connection

**Endpoints:**
- Knowledge Graph: `ws://localhost:8080/api/graph/stream`
- Ontology Graph: `ws://localhost:8080/api/ontology/graph/stream`
- Agent Visualization: `ws://localhost:8080/api/agents/stream`

**Authentication:**
```javascript
const ws = new WebSocket('ws://localhost:8080/api/graph/stream');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: '<jwt_token>'
  }));
};
```

### Message Types

#### Server → Client Messages

**Node Update (Binary Protocol V2):**
```
36-byte binary message (see Binary Protocol Specification)
```

**Edge Update:**
```json
{
  "type": "edge_update",
  "edge": {
    "id": "edge_1",
    "source": 1,
    "target": 2,
    "weight": 1.0
  }
}
```

**Simulation State:**
```json
{
  "type": "simulation_state",
  "running": true,
  "fps": 59.8,
  "iteration": 1523
}
```

**Error:**
```json
{
  "type": "error",
  "code": "INVALID_NODE",
  "message": "Node ID does not exist"
}
```

#### Client → Server Messages

**Request Full Sync:**
```json
{
  "type": "request_sync"
}
```

**Update Node Position (Manual Drag):**
```json
{
  "type": "update_position",
  "node_id": 1,
  "position": { "x": 15.5, "y": 25.3, "z": 10.1 }
}
```

**Subscribe to Graph Mode:**
```json
{
  "type": "subscribe",
  "graph_mode": "knowledge" // or "ontology"
}
```

### Adaptive Broadcasting

**Active State (Physics Running):**
- 60 FPS (16.6ms interval)
- Full node position updates

**Settled State (Physics Converged):**
- 5 Hz (200ms interval)
- Delta updates only (changed nodes)

**On-Demand:**
- Client can request full graph sync at any time

---

## Binary Protocol Specification

### Protocol Version 2.0

VisionFlow uses a highly optimized **36-byte binary protocol** for real-time node updates:

#### Message Structure

```
Offset | Size | Type  | Field         | Description
-------|------|-------|---------------|----------------------------------
0      | 1    | u8    | msg_type      | 0x01 = NodeUpdate, 0x02 = EdgeUpdate
1      | 4    | u32   | node_id       | Node identifier (4.3 billion max)
5      | 4    | f32   | position_x    | X coordinate
9      | 4    | f32   | position_y    | Y coordinate
13     | 4    | f32   | position_z    | Z coordinate
17     | 4    | f32   | velocity_x    | X velocity
21     | 4    | f32   | velocity_y    | Y velocity
25     | 4    | f32   | velocity_z    | Z velocity
29     | 4    | u32   | color_rgba    | Packed RGBA (8 bits per channel)
33     | 3    | u8[3] | flags         | State flags
```

#### Node ID Format (u32)

**Bits 31-30:** Graph type flags
- `00` (0): Knowledge graph node (local markdown)
- `01` (1): Ontology graph node (GitHub markdown)
- `10` (2): Agent visualization node
- `11` (3): Reserved

**Bits 29-0:** Actual node ID (1.07 billion max per graph type)

**Example:**
```rust
// Knowledge graph node ID 12345
let node_id: u32 = 0b00_000000000000000000000000011000000111001; // 12345

// Ontology graph node ID 12345
let node_id: u32 = 0b01_000000000000000000000000011000000111001; // 1,073,754,489

// Extract graph type
let graph_type = (node_id >> 30) & 0b11;

// Extract actual ID
let actual_id = node_id & 0x3FFFFFFF;
```

#### Color Format (u32 RGBA)

```
Bits 31-24: Red channel (0-255)
Bits 23-16: Green channel (0-255)
Bits 15-8:  Blue channel (0-255)
Bits 7-0:   Alpha channel (0-255)
```

**Example:**
```rust
// Pack RGBA
let color: u32 = (red << 24) | (green << 16) | (blue << 8) | alpha;

// Unpack RGBA
let red   = (color >> 24) & 0xFF;
let green = (color >> 16) & 0xFF;
let blue  = (color >> 8) & 0xFF;
let alpha = color & 0xFF;
```

#### Flags (3 bytes)

```
Byte 33 (flags[0]):
  Bit 0: is_pinned (manually positioned)
  Bit 1: is_selected (user selected)
  Bit 2: is_highlighted (search result)
  Bit 3: is_visible (culled if false)
  Bits 4-7: Reserved

Byte 34 (flags[1]):
  Bits 0-7: Node type (0-255)

Byte 35 (flags[2]):
  Bits 0-7: Reserved
```

### Protocol Benefits

- **~80% bandwidth reduction** vs JSON (36 bytes vs ~200 bytes per node)
- **60 FPS sustained** at 100k nodes (3.6 MB/s vs 20 MB/s JSON)
- **<10ms latency** for physics updates
- **Prevents graph conflation** (knowledge vs ontology via node ID bits)
- **Scalable** to millions of nodes

### Client-Side Parsing (TypeScript)

```typescript
function parseBinaryNodeUpdate(buffer: ArrayBuffer): NodeUpdate {
  const view = new DataView(buffer);
  const msgType = view.getUint8(0);

  if (msgType !== 0x01) {
    throw new Error('Not a node update message');
  }

  const nodeId = view.getUint32(1, true); // little-endian
  const graphType = (nodeId >> 30) & 0b11;
  const actualId = nodeId & 0x3FFFFFFF;

  return {
    graphType: ['knowledge', 'ontology', 'agent', 'reserved'][graphType],
    nodeId: actualId,
    position: {
      x: view.getFloat32(5, true),
      y: view.getFloat32(9, true),
      z: view.getFloat32(13, true),
    },
    velocity: {
      x: view.getFloat32(17, true),
      y: view.getFloat32(21, true),
      z: view.getFloat32(25, true),
    },
    color: {
      r: (view.getUint32(29, true) >> 24) & 0xFF,
      g: (view.getUint32(29, true) >> 16) & 0xFF,
      b: (view.getUint32(29, true) >> 8) & 0xFF,
      a: view.getUint32(29, true) & 0xFF,
    },
    isPinned: (view.getUint8(33) & 0b1) !== 0,
    isSelected: (view.getUint8(33) & 0b10) !== 0,
    isHighlighted: (view.getUint8(33) & 0b100) !== 0,
    isVisible: (view.getUint8(33) & 0b1000) !== 0,
    nodeType: view.getUint8(34),
  };
}
```

---

## Error Handling

### Error Response Format

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
| `INFERENCE_TIMEOUT` | 504 | Ontology inference timed out |

### Example Error Response

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Validation failed for field 'graph_name'",
    "details": {
      "field": "graph_name",
      "value": "invalid_graph",
      "allowed_values": ["logseq", "visionflow", "ontology", "default"]
    }
  }
}
```

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

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please retry after 60 seconds.",
    "details": {
      "retry_after": 60
    }
  }
}
```

---

## Additional Resources

- [VisionFlow Architecture](/docs/ARCHITECTURE.md)
- [Developer Guide](/docs/DEVELOPER_GUIDE.md)
- [Database Documentation](/docs/DATABASE.md)
- [Client Integration Guide](/docs/CLIENT_INTEGRATION.md)

---

**Document Maintained By:** VisionFlow API Team
**Last Review:** 2025-10-22
**Next Review:** 2025-11-22

