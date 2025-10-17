# Ontology Data API - Quick Start Guide

## Quick Setup

### 1. Enable Ontology Feature

```bash
# Build with ontology support
cargo build --features ontology

# Run server
cargo run --features ontology
```

### 2. Test Basic Endpoints

```bash
# Health check
curl http://localhost:8080/api/health

# List domains
curl http://localhost:8080/api/ontology/domains

# List classes
curl http://localhost:8080/api/ontology/classes?limit=5

# Get entity
curl http://localhost:8080/api/ontology/entities/vnf-123
```

### 3. Connect WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/api/ontology/stream');
ws.onmessage = (e) => console.log(JSON.parse(e.data));
```

## Common Use Cases

### Use Case 1: Browse ETSI Domains

```bash
# Get all domains with statistics
curl "http://localhost:8080/api/ontology/domains?include_stats=true" | jq
```

**Response**:
```json
{
  "domains": [
    {
      "id": "etsi-nfv",
      "name": "ETSI NFV",
      "classCount": 125,
      "propertyCount": 250
    }
  ]
}
```

### Use Case 2: Explore VNF Classes

```bash
# Get all VNF-related classes with properties
curl "http://localhost:8080/api/ontology/classes?filter=vnf&include_properties=true" | jq
```

### Use Case 3: Find All Object Properties

```bash
# List object properties with constraints
curl "http://localhost:8080/api/ontology/properties?property_type=object_property&include_constraints=true" | jq
```

### Use Case 4: Get Entity Relationships

```bash
# Get entity with full relationship graph
curl "http://localhost:8080/api/ontology/entities/vnf-123?include_inferred=true&max_depth=2" | jq
```

### Use Case 5: Query VNF Instances

```bash
# Query all VNF instances
curl -X POST http://localhost:8080/api/ontology/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT ?entity WHERE { ?entity a vnf }",
    "limit": 10
  }' | jq
```

### Use Case 6: Visualize Ontology Graph

```bash
# Get graph data for visualization
curl "http://localhost:8080/api/ontology/graph?domain=etsi-nfv&max_nodes=100" | jq
```

## Integration Examples

### React/TypeScript

```typescript
// Fetch domains
async function fetchDomains(): Promise<DomainInfo[]> {
  const response = await fetch('/api/ontology/domains?include_stats=true');
  const data = await response.json();
  return data.domains;
}

// Query ontology
async function queryOntology(query: string): Promise<QueryResult> {
  const response = await fetch('/api/ontology/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit: 100 })
  });
  return response.json();
}

// WebSocket updates
const ws = new WebSocket('ws://localhost:8080/api/ontology/stream');
ws.onmessage = (event) => {
  const update: OntologyUpdate = JSON.parse(event.data);

  switch (update.type) {
    case 'EntityAdded':
      console.log('New entity:', update.entityId);
      break;
    case 'ValidationCompleted':
      console.log('Validation done:', update.reportId);
      break;
  }
};
```

### Python

```python
import requests
import json

# Fetch classes
def fetch_classes(domain='etsi-nfv', limit=50):
    response = requests.get(
        f'http://localhost:8080/api/ontology/classes',
        params={'domain': domain, 'limit': limit}
    )
    return response.json()['classes']

# Execute query
def query_ontology(query_str):
    response = requests.post(
        'http://localhost:8080/api/ontology/query',
        json={'query': query_str, 'limit': 100}
    )
    return response.json()['results']

# Example: Find all deployed VNFs
results = query_ontology('SELECT ?entity WHERE { ?entity a vnf }')
for result in results:
    print(result['entity'])
```

### Go

```go
package main

import (
    "encoding/json"
    "fmt"
    "net/http"
)

type DomainInfo struct {
    ID           string `json:"id"`
    Name         string `json:"name"`
    ClassCount   int    `json:"classCount"`
    PropertyCount int   `json:"propertyCount"`
}

func fetchDomains() ([]DomainInfo, error) {
    resp, err := http.Get("http://localhost:8080/api/ontology/domains")
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result struct {
        Domains []DomainInfo `json:"domains"`
    }

    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return result.Domains, nil
}
```

## Advanced Queries

### Query 1: Find All Relationships of Type

```json
{
  "query": "SELECT ?source ?target WHERE { ?source hasVNFC ?target }",
  "limit": 100
}
```

### Query 2: Count Entities by Type

```json
{
  "query": "COUNT WHERE { ?entity a ?type }",
  "parameters": {
    "domain": "etsi-nfv"
  }
}
```

### Query 3: Describe Entity

```json
{
  "query": "DESCRIBE vnf-123",
  "explain": true
}
```

## Performance Tips

### 1. Use Pagination
```bash
# Fetch in chunks
curl "http://localhost:8080/api/ontology/classes?offset=0&limit=50"
curl "http://localhost:8080/api/ontology/classes?offset=50&limit=50"
```

### 2. Filter at Query Time
```bash
# Filter by domain to reduce result set
curl "http://localhost:8080/api/ontology/classes?domain=etsi-nfv"
```

### 3. Use Cache-Friendly Queries
```bash
# Repeated queries benefit from cache
for i in {1..5}; do
  curl "http://localhost:8080/api/ontology/domains" -w "\nTime: %{time_total}s\n"
done
```

### 4. Limit Graph Visualization
```bash
# Cap nodes for performance
curl "http://localhost:8080/api/ontology/graph?max_nodes=200"
```

## Troubleshooting

### Issue: Empty Results

**Problem**: Endpoints return empty arrays

**Solution**: Check if ontology data is loaded
```bash
curl http://localhost:8080/api/ontology/health
```

### Issue: WebSocket Disconnects

**Problem**: WebSocket connection drops

**Solution**: Implement reconnect logic
```javascript
function connectWebSocket() {
  const ws = new WebSocket('ws://localhost:8080/api/ontology/stream');

  ws.onclose = () => {
    console.log('Disconnected, reconnecting...');
    setTimeout(connectWebSocket, 5000);
  };

  return ws;
}
```

### Issue: Slow Queries

**Problem**: Query takes too long

**Solution**: Add timeout and limit results
```json
{
  "query": "SELECT ?entity WHERE { ?entity a vnf }",
  "limit": 50,
  "timeoutSeconds": 10
}
```

### Issue: Cache Miss Rate

**Problem**: Low cache hit rate

**Solution**: Check cache statistics
```bash
# TODO: Add cache stats endpoint
curl http://localhost:8080/api/ontology/cache/stats
```

## Development Workflow

### 1. Add New Domain

```rust
// In db.rs, add to list_domains()
DomainInfo {
    id: "my-custom-domain".to_string(),
    name: "My Custom Domain".to_string(),
    description: "Custom domain description".to_string(),
    class_count: 0,
    property_count: 0,
    namespace: "http://example.org/custom#".to_string(),
    updated_at: Utc::now(),
}
```

### 2. Add New Class

```rust
// In db.rs, add to list_classes()
ClassInfo {
    id: "my-class".to_string(),
    name: "MyClass".to_string(),
    description: Some("Custom class".to_string()),
    parent_classes: vec!["base-class".to_string()],
    child_classes: vec![],
    domain: "my-custom-domain".to_string(),
    properties: vec![],
    instance_count: 0,
    namespace: "http://example.org/custom#".to_string(),
}
```

### 3. Add New Query Type

```rust
// In query.rs, add to execute_query()
match parsed.query_type.as_str() {
    "SELECT" => self.execute_select(&parsed, parameters, limit, offset),
    "COUNT" => self.execute_count(&parsed, parameters),
    "DESCRIBE" => self.execute_describe(&parsed, parameters),
    "MY_QUERY" => self.execute_my_query(&parsed, parameters), // NEW
    _ => Err(format!("Unsupported query type: {}", parsed.query_type)),
}
```

### 4. Add WebSocket Update Type

```rust
// In mod.rs, add to OntologyUpdate enum
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum OntologyUpdate {
    // ... existing variants
    MyCustomUpdate {
        data: String,
        timestamp: DateTime<Utc>,
    },
}
```

## Testing Commands

```bash
# Run all tests
cargo test --features ontology ontology_data

# Run specific test
cargo test --features ontology cache_basic_operations

# Run with output
cargo test --features ontology -- --nocapture

# Run benchmarks (if implemented)
cargo bench --features ontology
```

## Monitoring

### Health Check
```bash
curl http://localhost:8080/api/ontology/health | jq
```

### Cache Statistics
```bash
# TODO: Implement cache stats endpoint
curl http://localhost:8080/api/ontology/cache/stats | jq
```

### Query Performance
```bash
# Measure query time
time curl -X POST http://localhost:8080/api/ontology/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT ?entity WHERE { ?entity a vnf }"}'
```

## Resources

- **Full API Documentation**: `/docs/ontology-data-api.md`
- **Implementation Summary**: `/ONTOLOGY_DATA_API_SUMMARY.md`
- **Source Code**: `/src/handlers/api_handler/ontology_data/`
- **Database Schema**: See documentation for complete schema

## Support

For issues or questions:
1. Check the full documentation
2. Review error messages and trace IDs
3. Check server logs
4. File an issue with reproduction steps

---

**Quick Reference Card**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/ontology/domains` | GET | List domains |
| `/api/ontology/classes` | GET | List classes |
| `/api/ontology/properties` | GET | List properties |
| `/api/ontology/entities/:id` | GET | Get entity |
| `/api/ontology/query` | POST | Execute query |
| `/api/ontology/graph` | GET | Graph viz data |
| `/api/ontology/stream` | WebSocket | Real-time updates |

**Cache Configuration**
- Default capacity: 1000 entries
- Default TTL: 3600 seconds (1 hour)
- Eviction: LRU policy
- Background cleanup: Every 5 minutes

**Query Limits**
- Default limit: 50 items
- Maximum limit: 500 items
- Maximum graph nodes: 2000 nodes
- Default timeout: 30 seconds
