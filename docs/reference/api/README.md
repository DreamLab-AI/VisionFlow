---
title: VisionFlow API Documentation
description: Complete REST API Reference for VisionFlow Knowledge Graph Platform
category: reference
diataxis: reference
tags:
  - api
  - rest
  - websocket
  - frontend
updated-date: 2025-01-29
difficulty-level: intermediate
---


# VisionFlow API Documentation

**Complete REST API Reference for VisionFlow Knowledge Graph Platform**

---

## Documentation Index

### Primary Documentation

- **[rest-api-complete.md](./rest-api-complete.md)** - **MASTER REFERENCE**
  - Complete API documentation with all 100+ endpoints
  - Authentication, request/response examples
  - cURL and TypeScript examples for every endpoint
  - WebSocket binary protocol integration
  - Error handling and status codes

### Specialized Documentation

- **[01-authentication.md](./01-authentication.md)** - Authentication & Security
  - JWT token management
  - API key generation
  - Security best practices

- **[03-websocket.md](./03-websocket.md)** - WebSocket Binary Protocol
  - 36-byte binary message format
  - 80% bandwidth reduction vs JSON
  - Real-time physics updates
  - Client implementation examples

- **[solid-api.md](./solid-api.md)** - Solid Pod Integration
  - Pod creation and management
  - LDP resource operations (CRUD)
  - Agent memory persistence
  - WebSocket notifications for Pods
  - Nostr NIP-98 authentication

### Quick References

- **** - Fast lookup guide
- **** - Implementation notes

---

## Quick Start

### 1. Health Check

```bash
curl http://localhost:9090/api/health
```

### 2. Get Application Config

```bash
curl http://localhost:9090/api/config
```

### 3. Fetch Graph Data

```bash
curl http://localhost:9090/api/graph/data
```

### 4. Connect to WebSocket (Binary Protocol)

```typescript
const ws = new WebSocket('ws://localhost:9090/ws?token=YOUR-JWT');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
  const parser = new BinaryProtocolParser(event.data);
  const updates = parser.parseNodeUpdates();
  // Process 100k+ node updates at 60 FPS
};
```

---

## API Categories

### Core Endpoints
- Health & Configuration
- Authentication & Authorization
- WebSocket Connections

### Graph Management
- Get/Update Graph Data
- Paginated Graph Queries
- Auto-Balance Notifications
- Graph Export & Sharing

### Ontology System
- Class Hierarchy
- Validation & Inference
- OWL Classes & Properties
- Axiom Management

### Physics Simulation
- Start/Stop Simulation
- Apply Forces
- Pin/Unpin Nodes
- Layout Optimization

### File Management
- Process Markdown Files
- GitHub Synchronization
- File Content Retrieval

### Advanced Features
- Bots & Swarm Coordination
- Analytics & Clustering
- Workspace Management
- Constraints System

---

## Endpoint Count by Category

| Category | Endpoints | Documentation |
|----------|-----------|---------------|
| **Graph** | 15 | Graph data, pagination, updates |
| **Ontology** | 25 | Classes, validation, inference |
| **Physics** | 10 | Simulation control, forces |
| **Files** | 4 | Processing, content retrieval |
| **Bots/Swarm** | 7 | Agent coordination |
| **Analytics** | 12 | Communities, centrality, clustering |
| **Workspace** | 8 | CRUD operations |
| **Auth** | 6 | Login, tokens, API keys |
| **Solid** | 10 | Pod management, LDP operations |
| **Advanced** | 20+ | Export, constraints, RAGFlow |

**Total**: 110+ documented endpoints

---

## Key Features

### Binary WebSocket Protocol
- **80% bandwidth reduction** (3.6 MB vs 18 MB per frame at 100k nodes)
- **15x faster parsing** (0.8ms vs 12ms)
- **Sub-10ms latency** for real-time updates
- **60 FPS** streaming capability

### Comprehensive Ontology System
- Full OWL 2 support
- Real-time validation
- Inference engine
- Class hierarchy navigation
- WebSocket progress updates

### Advanced Physics
- GPU-accelerated simulation
- Auto-balance detection
- Custom force application
- Node pinning
- Layout optimization

---

## Usage Examples

### TypeScript Client

```typescript
// Fetch complete graph with physics data
interface GraphResponse {
  nodes: NodeWithPosition[];
  edges: Edge[];
  metadata: Record<string, Metadata>;
  settlementState: SettlementState;
}

async function fetchGraph(): Promise<GraphResponse> {
  const response = await fetch('/api/graph/data');
  if (!response.ok) throw new Error('Failed to fetch graph');
  return response.json();
}

// Use with React
function useGraphData() {
  const [graph, setGraph] = useState<GraphResponse | null>(null);

  useEffect(() => {
    fetchGraph().then(setGraph);
  }, []);

  return graph;
}
```

### Python Client

```python
import requests

# Get ontology hierarchy
def get-hierarchy(ontology-id="default", max-depth=None):
    url = "http://localhost:9090/api/ontology/hierarchy"
    params = {"ontology-id": ontology-id}
    if max-depth:
        params["max-depth"] = max-depth

    response = requests.get(url, params=params)
    response.raise-for-status()
    return response.json()

# Navigate class hierarchy
hierarchy = get-hierarchy()
root-classes = hierarchy["rootClasses"]

for class-iri in root-classes:
    node = hierarchy["hierarchy"][class-iri]
    print(f"{node['label']} ({node['nodeCount']} descendants)")
```

---

## Authentication

All endpoints except `/health` and `/config` require authentication:

```bash
# Login to get JWT token
curl -X POST http://localhost:9090/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token in requests
curl -H "Authorization: Bearer YOUR-JWT-TOKEN" \
  http://localhost:9090/api/graph/data

# Or use API key
curl -H "X-API-Key: YOUR-API-KEY" \
  http://localhost:9090/api/graph/data
```

---

## ️ Database

Unified SQLite database (`unified.db`) with tables:
- `nodes` - Knowledge graph nodes
- `edges` - Node relationships
- `owl-classes` - OWL ontology classes
- `owl-properties` - OWL properties
- `github-sync-state` - Sync tracking
- `workspaces` - User workspaces
- `validation-reports` - Ontology validation

---

## ️ Error Handling

Standard error format:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION-ERROR",
    "message": "Invalid input",
    "details": [...]
  }
}
```

HTTP status codes:
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `500` - Internal Server Error
- `503` - Service Unavailable

---

## Performance Benchmarks

### WebSocket Binary Protocol (100k nodes @ 60 FPS)

| Metric | Binary | JSON | Improvement |
|--------|--------|------|-------------|
| Message Size | 3.6 MB | 18 MB | 80% smaller |
| Parse Time | 0.8 ms | 12 ms | 15x faster |
| Latency | <10 ms | 45 ms | 4.5x faster |
| CPU Usage | 5% | 28% | 5.6x lower |

---

## Migration Notes

### Deprecated Endpoints

The following endpoints have been consolidated:
- ~~`/api/endpoints/`~~ → Use `/api/graph/`, `/api/ontology/`, etc.
- ~~`/api/ontology/hierarchy-endpoint/`~~ → Merged into `rest-api-complete.md`
- ~~`/visualisation/`~~ → Redirects to `/api/settings`

---

## Support

- **Full Documentation**: [rest-api-complete.md](./rest-api-complete.md)
- **WebSocket Protocol**: [03-websocket.md](./03-websocket.md)
- **Solid API**: [solid-api.md](./solid-api.md)
- **Integration Guide**: [../../guides/solid-integration.md](../../guides/solid-integration.md)

---

**Last Updated**: December 29, 2025
**Version**: 1.1.0
**Maintainer**: VisionFlow API Documentation Team
