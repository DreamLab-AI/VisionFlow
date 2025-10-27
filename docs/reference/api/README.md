# API Reference

Welcome to the VisionFlow API documentation. Choose the protocol that best fits your needs.

---

## Quick Start

### I want to...

**Access graph data**
→ [REST API - Knowledge Graph](./rest-api.md#knowledge-graph-api)

**Get real-time position updates**
→ [WebSocket API](./websocket-api.md)

**Understand the binary wire format**
→ [Binary Protocol Specification](./binary-protocol.md)

**Update application settings**
→ [REST API - Settings](./rest-api.md#settings-api)

**Work with semantic data**
→ [REST API - Ontology](./rest-api.md#ontology-api)

**Control physics simulation**
→ [REST API - Physics](./rest-api.md#physics-api)

---

## API Protocols

### REST API (HTTP)

**Best for:** Queries, one-time operations, administrative tasks

- **Synchronous** - Get immediate responses
- **Stateless** - No connection overhead
- **JSON format** - Easy to parse
- **CRUD operations** - Standard HTTP methods

**Common Uses:**
- Fetch graph data
- Add/update nodes and edges
- Manage settings
- Run ontology inference

**[Learn More →](./rest-api.md)**

### WebSocket API (Binary & JSON)

**Best for:** Real-time updates, streaming data, interactive applications

- **Persistent connection** - Low latency
- **Binary protocol** - 82% bandwidth savings
- **Full-duplex** - Bidirectional communication
- **Event streaming** - Push updates to client

**Common Uses:**
- Real-time node position updates
- Physics simulation state
- Interactive visualization
- Live graph changes

**[Learn More →](./websocket-api.md)**

---

## Authentication

VisionFlow implements **three-tier authentication**:

### Public Access (No Auth)
- Health checks
- Documentation
- Public graph views (read-only)

**Header:** None required

### User Authentication (JWT)
- Standard operations (read/write)
- User settings
- Physics control

**Header:** `Authorization: Bearer <jwt_token>`

### Developer Authentication (API Key)
- System configuration
- Administrative operations

**Header:** `X-API-Key: <api_key>`

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Base URL | `http://localhost:3030` |
| REST Format | JSON over HTTP |
| WebSocket Format | Binary (V2) + JSON |
| Authentication | JWT + API Keys |
| Rate Limiting | Yes (per tier) |

---

## Rate Limiting

All APIs have rate limits based on authentication tier:

| Tier | Requests/Min | Burst |
|------|--------------|-------|
| Public | 60 | 10 |
| User | 1000 | 100 |
| Developer | 5000 | 500 |

**Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1729591200
```

---

## API Overview

### Core Domains

#### Settings API
- Get/update application configuration
- Manage user preferences
- Control physics parameters per graph

**[Details →](./rest-api.md#settings-api)**

#### Knowledge Graph API
- Add/remove nodes
- Create/delete edges
- Query graph with filters
- Get statistics

**[Details →](./rest-api.md#knowledge-graph-api)**

#### Ontology API
- Define OWL classes and properties
- Run semantic reasoning
- Validate ontologies
- Access inference results

**[Details →](./rest-api.md#ontology-api)**

#### Physics API
- Get simulation state
- Update parameters
- Apply constraints (pin nodes, distance limits)
- Monitor GPU utilization

**[Details →](./rest-api.md#physics-api)**

#### Health & Monitoring
- Unified health check
- Physics simulation health
- MCP relay logs

**[Details →](./rest-api.md#health--monitoring)**

---

## Binary Protocol (V2)

For maximum efficiency, WebSocket uses a 36-byte binary protocol:

```
[msg_type: u8][node_id: u32][pos_x: f32]...[color_rgba: u32][flags: u8][type: u8][reserved: u8]
36 bytes total
```

**Benefits:**
- 82% bandwidth reduction vs JSON
- 60 FPS at 100k nodes
- <10ms latency

**[Detailed Specification →](./binary-protocol.md)**

---

## Performance Characteristics

### REST API

- **Typical Latency:** <50ms
- **Throughput:** 10-100 req/sec per client
- **Payload Size:** 200 bytes - 1 MB (JSON)

### WebSocket API

- **Latency:** <10ms
- **Throughput:** 60 FPS (16.6ms per frame)
- **Payload Size:** 36 bytes (binary), 200 bytes (JSON)
- **Bandwidth:** 3.6 MB/s @ 100k nodes, 60 FPS

---

## Error Handling

### Standard Error Response

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {
      "field": "additional context"
    }
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_INPUT` | 400 | Validation failed |
| `NOT_FOUND` | 404 | Resource not found |
| `UNAUTHORIZED` | 401 | Auth required |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `CONFLICT` | 409 | Resource exists |
| `INTERNAL_ERROR` | 500 | Server error |
| `DATABASE_ERROR` | 500 | Database failure |
| `GPU_ERROR` | 503 | GPU computation failed |

---

## Code Examples

### JavaScript/TypeScript

#### REST: Get Full Graph

```javascript
fetch('http://localhost:3030/api/graph', {
  headers: {
    'Authorization': 'Bearer <token>'
  }
})
.then(r => r.json())
.then(graph => {
  console.log(`Nodes: ${graph.nodes.length}`);
  console.log(`Edges: ${graph.edges.length}`);
});
```

#### WebSocket: Real-Time Updates

```javascript
const ws = new WebSocket('ws://localhost:3030/api/graph/stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: '<jwt_token>'
  }));
};

ws.onmessage = (event) => {
  // Handle binary node updates (36 bytes)
  if (event.data instanceof ArrayBuffer) {
    const update = parseBinaryNodeUpdate(event.data);
    updateNodePosition(update.nodeId, update.position);
  }
};
```

### Python

#### REST: Add Node

```python
import requests

response = requests.post(
    'http://localhost:3030/api/graph/node',
    headers={
        'Authorization': 'Bearer <token>'
    },
    json={
        'label': 'New Concept',
        'type': 'concept',
        'position': {'x': 0, 'y': 0, 'z': 0}
    }
)

node_id = response.json()['node_id']
print(f'Created node: {node_id}')
```

#### REST: Query Nodes

```python
response = requests.post(
    'http://localhost:3030/api/graph/query',
    headers={
        'Authorization': 'Bearer <token>'
    },
    json={
        'filter': {'type': 'concept'},
        'limit': 100
    }
)

nodes = response.json()
for node in nodes:
    print(f"{node['id']}: {node['label']}")
```

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| [REST API](./rest-api.md) | HTTP endpoints for all operations |
| [WebSocket API](./websocket-api.md) | Real-time streaming protocol |
| [Binary Protocol](./binary-protocol.md) | V2 wire format specification |

---

## Support Resources

- **[Architecture Guide](../architecture/)** - Understand how APIs work
- **[Developer Guides](../../guides/developer/)** - Implementation tutorials
- **[Concepts](../../concepts/)** - Domain knowledge

---

## API Status

| Component | Status | Version |
|-----------|--------|---------|
| REST API | ✅ Stable | 3.1.0 |
| WebSocket API | ✅ Stable | 3.1.0 |
| Binary Protocol | ✅ Stable | 2.0 |
| Authentication | ✅ Stable | 1.0 |

---

**Last Updated:** 2025-10-25
**Base URL:** `http://localhost:3030`
**Maintained By:** VisionFlow API Team
**Next Review:** 2025-11-25
