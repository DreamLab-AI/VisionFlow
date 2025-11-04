# REST API Complete Reference

**VisionFlow Ontology and Graph API Documentation**

---

## Base URL

```
http://localhost:8080/api
```

## Authentication

Currently no authentication required (development mode).

**Production**: Will use Bearer token authentication.

---

## Ontology Endpoints

### GET /ontology/hierarchy

Retrieve complete ontology class hierarchy with parent-child relationships.

**Request**:
```http
GET /api/ontology/hierarchy?ontology_id=default&max_depth=10
```

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ontology_id` | string | No | "default" | Ontology identifier |
| `max_depth` | integer | No | unlimited | Maximum hierarchy depth to return |

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

**TypeScript Interface**:
```typescript
interface ClassHierarchy {
  rootClasses: string[];
  hierarchy: { [iri: string]: ClassNode };
}

interface ClassNode {
  iri: string;
  label: string;
  parentIri: string | null;
  childrenIris: string[];
  nodeCount: number;      // Descendant count
  depth: number;          // Hierarchy level
}
```

**Error Responses**:
- `500 Internal Server Error`: Failed to build hierarchy
- `503 Service Unavailable`: Feature disabled

**Example Usage** (JavaScript):
```javascript
const response = await fetch('/api/ontology/hierarchy?ontology_id=default');
const data = await response.json();

console.log('Root classes:', data.rootClasses);
for (const [iri, node] of Object.entries(data.hierarchy)) {
  console.log(`${node.label} (depth: ${node.depth}, children: ${node.childrenIris.length})`);
}
```

**Example Usage** (Python):
```python
import requests

response = requests.get('http://localhost:8080/api/ontology/hierarchy')
data = response.json()

for class_iri, node in data['hierarchy'].items():
    print(f"{node['label']} - Depth: {node['depth']}")
```

**Implementation**: See [ontology_handler.rs:936-1090](../../src/handlers/api_handler/ontology/mod.rs)

---

### POST /ontology/reasoning/infer

Trigger OWL reasoning and return inferred axioms.

**Request**:
```http
POST /api/ontology/reasoning/infer
Content-Type: application/json

{
  "ontology_id": "default"
}
```

**Request Body**:
```typescript
interface ReasoningRequest {
  ontology_id: string;
}
```

**Response** (200 OK):
```json
{
  "inferred_axioms": [
    {
      "axiomType": "SubClassOf",
      "subjectIri": "http://example.org/GraduateStudent",
      "objectIri": "http://example.org/Person",
      "confidence": 0.95,
      "reasoningMethod": "whelk-el++"
    }
  ],
  "cache_hit": false,
  "reasoning_time_ms": 245
}
```

**TypeScript Interface**:
```typescript
interface InferredAxiom {
  axiomType: string;          // "SubClassOf", "DisjointWith", etc.
  subjectIri: string;         // Subject class IRI
  objectIri: string;          // Object class IRI
  confidence: number;         // 0.0-1.0
  reasoningMethod: string;    // "whelk-el++"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid ontology_id
- `500 Internal Server Error`: Reasoning failed
- `503 Service Unavailable`: Reasoning feature disabled

---

### GET /ontology/disjoint-classes

Get all disjoint class pairs from ontology.

**Request**:
```http
GET /api/ontology/disjoint-classes?ontology_id=default
```

**Response** (200 OK):
```json
{
  "disjoint_pairs": [
    {
      "classA": "http://example.org/Animal",
      "classB": "http://example.org/Plant"
    },
    {
      "classA": "http://example.org/Animal",
      "classB": "http://example.org/Mineral"
    }
  ]
}
```

**TypeScript Interface**:
```typescript
interface DisjointClassPair {
  classA: string;
  classB: string;
}
```

---

## Graph Endpoints

### GET /graph/nodes

Retrieve graph nodes with optional filtering.

**Request**:
```http
GET /api/graph/nodes?limit=1000&offset=0&class_iri=http://example.org/Person
```

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 1000 | Maximum nodes to return |
| `offset` | integer | No | 0 | Pagination offset |
| `class_iri` | string | No | - | Filter by class IRI |

**Response** (200 OK):
```json
{
  "nodes": [
    {
      "id": "node-123",
      "label": "John Doe",
      "metadata": {
        "classIri": "http://example.org/Person",
        "properties": {
          "age": 30,
          "email": "john@example.com"
        }
      }
    }
  ],
  "total_count": 1523,
  "has_more": true
}
```

**TypeScript Interface**:
```typescript
interface GraphNode {
  id: string;
  label: string;
  metadata?: {
    classIri?: string;
    properties?: { [key: string]: any };
  };
}
```

---

### GET /graph/edges

Retrieve graph edges with optional filtering.

**Request**:
```http
GET /api/graph/edges?source_id=node-123&relationship=knows
```

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_id` | string | No | - | Filter by source node |
| `target_id` | string | No | - | Filter by target node |
| `relationship` | string | No | - | Filter by relationship type |
| `limit` | integer | No | 1000 | Maximum edges to return |

**Response** (200 OK):
```json
{
  "edges": [
    {
      "id": "edge-456",
      "source": "node-123",
      "target": "node-789",
      "relationship": "knows",
      "metadata": {
        "since": "2020-01-15"
      }
    }
  ],
  "total_count": 342
}
```

**TypeScript Interface**:
```typescript
interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relationship: string;
  metadata?: { [key: string]: any };
}
```

---

## Physics Constraints Endpoints

### POST /constraints/generate

Generate physics constraints from ontology axioms.

**Request**:
```http
POST /api/constraints/generate
Content-Type: application/json

{
  "ontology_id": "default",
  "constraint_types": ["Separation", "HierarchicalAttraction"],
  "config": {
    "disjoint_repel_multiplier": 2.0,
    "subclass_spring_multiplier": 0.5
  }
}
```

**Request Body**:
```typescript
interface ConstraintGenerationRequest {
  ontology_id: string;
  constraint_types?: string[];  // Optional filter
  config?: SemanticPhysicsConfig;
}

interface SemanticPhysicsConfig {
  disjoint_repel_multiplier?: number;
  subclass_spring_multiplier?: number;
  equivalent_colocation_dist?: number;
  partof_containment_radius?: number;
}
```

**Response** (200 OK):
```json
{
  "constraints": [
    {
      "constraintType": "Separation",
      "nodeA": "http://example.org/Animal",
      "nodeB": "http://example.org/Plant",
      "minDistance": 70.0,
      "strength": 0.8,
      "priority": 5
    },
    {
      "constraintType": "HierarchicalAttraction",
      "child": "http://example.org/Student",
      "parent": "http://example.org/Person",
      "idealDistance": 20.0,
      "strength": 0.3,
      "priority": 5
    }
  ],
  "total_count": 245,
  "generation_time_ms": 123
}
```

**TypeScript Interface**:
```typescript
interface SemanticConstraint {
  constraintType: string;
  nodeA?: string;
  nodeB?: string;
  child?: string;
  parent?: string;
  minDistance?: number;
  idealDistance?: number;
  strength: number;
  priority: number;
}
```

---

## WebSocket Endpoints

### WS /graph/updates

Real-time graph updates via WebSocket (binary protocol).

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8080/api/graph/updates');
```

**Binary Message Format**:

**Client → Server (Subscribe)**:
```
MessageType: 0x01 (Subscribe)
Payload: JSON { "node_ids": ["node-123", "node-456"] }
```

**Server → Client (Update)**:
```
MessageType: 0x02 (NodeUpdate)
Payload:
  - node_id: String (length-prefixed)
  - position_x: f32
  - position_y: f32
  - position_z: f32
```

**Client → Server (Unsubscribe)**:
```
MessageType: 0x03 (Unsubscribe)
```

**Example** (TypeScript):
```typescript
const ws = new WebSocket('ws://localhost:8080/api/graph/updates');

ws.onopen = () => {
  // Subscribe to updates
  const subscribe = new Uint8Array([
    0x01,  // MessageType: Subscribe
    ...encodeJSON({ node_ids: ['node-123'] })
  ]);
  ws.send(subscribe);
};

ws.onmessage = (event) => {
  const data = new Uint8Array(event.data);
  const messageType = data[0];

  if (messageType === 0x02) {  // NodeUpdate
    const { nodeId, position } = decodeNodeUpdate(data.slice(1));
    updateNodePosition(nodeId, position);
  }
};
```

**See**: [Binary Protocol Documentation](./websocket-binary-protocol.md)

---

## Error Responses

All endpoints return consistent error format:

```json
{
  "error": "Failed to retrieve hierarchy",
  "code": "INTERNAL_ERROR",
  "details": {
    "ontology_id": "default",
    "cause": "Database connection failed"
  },
  "timestamp": "2025-11-03T12:34:56.789Z",
  "trace_id": "abc123def456"
}
```

**Error Codes**:
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request or invalid parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Feature disabled or service down |
| `TIMEOUT` | 504 | Request timeout |

---

## Rate Limiting

**Current**: No rate limiting (development)

**Production**:
- 100 requests/minute per IP
- 1000 requests/hour per API key
- WebSocket: 1 connection per client

---

## CORS Configuration

**Development**:
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Content-Type, Authorization
```

**Production**: Restricted to specific origins

---

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:

```
GET /api/documentation
```

Swagger UI available at:

```
http://localhost:8080/swagger-ui
```

---

## SDK Examples

### JavaScript/TypeScript

```typescript
import { VisionFlowClient } from '@visionflow/client';

const client = new VisionFlowClient({
  baseURL: 'http://localhost:8080/api'
});

// Get hierarchy
const hierarchy = await client.ontology.getHierarchy('default');

// Trigger reasoning
const inferred = await client.ontology.infer('default');

// Get graph nodes
const nodes = await client.graph.getNodes({
  limit: 100,
  classIri: 'http://example.org/Person'
});
```

### Python

```python
from visionflow import VisionFlowClient

client = VisionFlowClient(base_url='http://localhost:8080/api')

# Get hierarchy
hierarchy = client.ontology.get_hierarchy('default')

# Trigger reasoning
inferred = client.ontology.infer('default')

# Get graph nodes
nodes = client.graph.get_nodes(
    limit=100,
    class_iri='http://example.org/Person'
)
```

### Rust

```rust
use visionflow_client::VisionFlowClient;

let client = VisionFlowClient::new("http://localhost:8080/api");

// Get hierarchy
let hierarchy = client.ontology().get_hierarchy("default").await?;

// Trigger reasoning
let inferred = client.ontology().infer("default").await?;

// Get graph nodes
let nodes = client.graph().get_nodes()
    .limit(100)
    .class_iri("http://example.org/Person")
    .execute()
    .await?;
```

---

## Performance Considerations

### Caching

- Hierarchy endpoint: 1-hour cache with ontology hash validation
- Reasoning results: Persistent cache with Blake3 hashing
- Graph queries: No caching (real-time data)

### Pagination

Large result sets automatically paginated:
- Default page size: 1000 items
- Maximum page size: 10000 items
- Use `offset` and `limit` parameters

### Response Times

| Endpoint | Typical | Maximum |
|----------|---------|---------|
| GET /hierarchy | <50ms | 200ms |
| POST /reasoning/infer | 100-500ms | 5s |
| GET /graph/nodes | <100ms | 500ms |
| GET /graph/edges | <100ms | 500ms |

---

## Changelog

### v1.0.0 (2025-11-03)
- Initial API release
- Ontology hierarchy endpoint
- Reasoning integration
- Graph query endpoints
- Physics constraint generation

---

## Related Documentation

- [Ontology Reasoning Pipeline](../../concepts/architecture/ontology-reasoning-pipeline.md)
- [Semantic Physics System](../../concepts/architecture/semantic-physics-system.md)
- [WebSocket Binary Protocol](./websocket-binary-protocol.md)
- [User Integration Guide](../guides/developer-integration-guide.md)

---

**Last Updated**: 2025-11-03
**API Version**: 1.0.0
**Status**: Production Ready
