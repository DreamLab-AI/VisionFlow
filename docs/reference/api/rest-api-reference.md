---
layout: default
title: "REST API Complete Reference"
parent: API
grand_parent: Reference
nav_order: 99
---

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
GET /api/ontology/hierarchy?ontology-id=default&max-depth=10
```

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `ontology-id` | string | No | "default" | Ontology identifier |
| `max-depth` | integer | No | unlimited | Maximum hierarchy depth to return |

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
const response = await fetch('/api/ontology/hierarchy?ontology-id=default');
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

for class-iri, node in data['hierarchy'].items():
    print(f"{node['label']} - Depth: {node['depth']}")
```

**Implementation**: See 

---

### POST /ontology/reasoning/infer

Trigger OWL reasoning and return inferred axioms.

**Request**:
```http
POST /api/ontology/reasoning/infer
Content-Type: application/json

{
  "ontology-id": "default"
}
```

**Request Body**:
```typescript
interface ReasoningRequest {
  ontology-id: string;
}
```

**Response** (200 OK):
```json
{
  "inferred-axioms": [
    {
      "axiomType": "SubClassOf",
      "subjectIri": "http://example.org/GraduateStudent",
      "objectIri": "http://example.org/Person",
      "confidence": 0.95,
      "reasoningMethod": "whelk-el++"
    }
  ],
  "cache-hit": false,
  "reasoning-time-ms": 245
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
- `400 Bad Request`: Invalid ontology-id
- `500 Internal Server Error`: Reasoning failed
- `503 Service Unavailable`: Reasoning feature disabled

---

### GET /ontology/disjoint-classes

Get all disjoint class pairs from ontology.

**Request**:
```http
GET /api/ontology/disjoint-classes?ontology-id=default
```

**Response** (200 OK):
```json
{
  "disjoint-pairs": [
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
GET /api/graph/nodes?limit=1000&offset=0&class-iri=http://example.org/Person
```

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | 1000 | Maximum nodes to return |
| `offset` | integer | No | 0 | Pagination offset |
| `class-iri` | string | No | - | Filter by class IRI |

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
  "total-count": 1523,
  "has-more": true
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
GET /api/graph/edges?source-id=node-123&relationship=knows
```

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source-id` | string | No | - | Filter by source node |
| `target-id` | string | No | - | Filter by target node |
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
  "total-count": 342
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
  "ontology-id": "default",
  "constraint-types": ["Separation", "HierarchicalAttraction"],
  "config": {
    "disjoint-repel-multiplier": 2.0,
    "subclass-spring-multiplier": 0.5
  }
}
```

**Request Body**:
```typescript
interface ConstraintGenerationRequest {
  ontology-id: string;
  constraint-types?: string[];  // Optional filter
  config?: SemanticPhysicsConfig;
}

interface SemanticPhysicsConfig {
  disjoint-repel-multiplier?: number;
  subclass-spring-multiplier?: number;
  equivalent-colocation-dist?: number;
  partof-containment-radius?: number;
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
  "total-count": 245,
  "generation-time-ms": 123
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
Payload: JSON { "node-ids": ["node-123", "node-456"] }
```

**Server → Client (Update)**:
```
MessageType: 0x02 (NodeUpdate)
Payload:
  - node-id: String (length-prefixed)
  - position-x: f32
  - position-y: f32
  - position-z: f32
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
    ...encodeJSON({ node-ids: ['node-123'] })
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

**See**: 

---

## Error Responses

All endpoints return consistent error format:

```json
{
  "error": "Failed to retrieve hierarchy",
  "code": "INTERNAL-ERROR",
  "details": {
    "ontology-id": "default",
    "cause": "Database connection failed"
  },
  "timestamp": "2025-11-03T12:34:56.789Z",
  "trace-id": "abc123def456"
}
```

**Error Codes**:
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID-REQUEST` | 400 | Malformed request or invalid parameters |
| `NOT-FOUND` | 404 | Resource not found |
| `INTERNAL-ERROR` | 500 | Internal server error |
| `SERVICE-UNAVAILABLE` | 503 | Feature disabled or service down |
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

client = VisionFlowClient(base-url='http://localhost:8080/api')

# Get hierarchy
hierarchy = client.ontology.get-hierarchy('default')

# Trigger reasoning
inferred = client.ontology.infer('default')

# Get graph nodes
nodes = client.graph.get-nodes(
    limit=100,
    class-iri='http://example.org/Person'
)
```

### Rust

```rust
use visionflow-client::VisionFlowClient;

let client = VisionFlowClient::new("http://localhost:8080/api");

// Get hierarchy
let hierarchy = client.ontology().get-hierarchy("default").await?;

// Trigger reasoning
let inferred = client.ontology().infer("default").await?;

// Get graph nodes
let nodes = client.graph().get-nodes()
    .limit(100)
    .class-iri("http://example.org/Person")
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

- [Ontology Reasoning Pipeline](../../../explanations/architecture/ontology-reasoning-pipeline.md)
- [Semantic Physics System](../../../explanations/architecture/semantic-physics-system.md)
- 
- 

---

**Last Updated**: 2025-11-03
**API Version**: 1.0.0
**Status**: Production Ready
