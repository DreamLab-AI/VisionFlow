# Ontology Data API Documentation

## Overview

The Ontology Data API provides comprehensive REST endpoints and WebSocket handlers for exposing ontology data to clients. It includes domain listings, class hierarchies, property schemas, entity relationships, advanced query interfaces, and real-time updates.

**Location**: `/home/devuser/workspace/project/src/handlers/api_handler/ontology_data/mod.rs`

## Architecture

```
ontology_data/
├── mod.rs           # Main API endpoints and WebSocket handlers
├── db.rs            # SQLite database layer
├── cache.rs         # LRU cache with TTL
└── query.rs         # SPARQL-like query engine
```

### Components

#### 1. Database Layer (`db.rs`)
- **Technology**: SQLite with embedded storage
- **Features**:
  - Domain, class, and property metadata storage
  - Entity and relationship persistence
  - Full-text search capabilities
  - Efficient indexing

**Database Schema** (implemented):
```sql
-- Domains table
domains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    namespace TEXT NOT NULL,
    class_count INTEGER DEFAULT 0,
    property_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP
)

-- Classes table
classes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    domain_id TEXT REFERENCES domains(id),
    namespace TEXT,
    instance_count INTEGER DEFAULT 0,
    created_at TIMESTAMP
)

-- Class hierarchy
class_hierarchy (
    parent_id TEXT REFERENCES classes(id),
    child_id TEXT REFERENCES classes(id),
    PRIMARY KEY (parent_id, child_id)
)

-- Properties table
properties (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    property_type TEXT CHECK(property_type IN ('object_property', 'data_property', 'annotation_property')),
    domain_id TEXT REFERENCES domains(id),
    is_functional BOOLEAN DEFAULT 0,
    is_inverse_functional BOOLEAN DEFAULT 0,
    is_transitive BOOLEAN DEFAULT 0,
    is_symmetric BOOLEAN DEFAULT 0
)

-- Property constraints
property_domain_constraints (
    property_id TEXT REFERENCES properties(id),
    class_id TEXT REFERENCES classes(id),
    PRIMARY KEY (property_id, class_id)
)

property_range_constraints (
    property_id TEXT REFERENCES properties(id),
    class_id TEXT,
    PRIMARY KEY (property_id, class_id)
)

property_cardinality (
    property_id TEXT PRIMARY KEY REFERENCES properties(id),
    min_cardinality INTEGER,
    max_cardinality INTEGER,
    exact_cardinality INTEGER
)

-- Entities table
entities (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    domain_id TEXT REFERENCES domains(id),
    properties_json TEXT, -- JSON blob for flexible properties
    created_at TIMESTAMP,
    updated_at TIMESTAMP
)

-- Relationships table
relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES entities(id),
    target_id TEXT REFERENCES entities(id),
    relationship_type TEXT NOT NULL,
    properties_json TEXT,
    is_inferred BOOLEAN DEFAULT 0,
    confidence REAL,
    created_at TIMESTAMP
)

-- Full-text search index
entity_fts (
    entity_id TEXT,
    label TEXT,
    properties_text TEXT
) -- Virtual FTS5 table
```

#### 2. Cache Layer (`cache.rs`)
- **Technology**: LRU cache with TTL
- **Capacity**: Configurable (default: 1000 entries)
- **TTL**: Configurable (default: 1 hour)
- **Features**:
  - Query result caching
  - Entity data caching
  - Pattern-based invalidation
  - Cache statistics (hit rate, evictions)
  - Background eviction task

**Cache Statistics**:
```rust
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size: usize,
    pub capacity: usize,
    pub hit_rate: f32,
}
```

#### 3. Query Engine (`query.rs`)
- **Query Language**: SPARQL-like syntax
- **Supported Operations**:
  - SELECT: Entity filtering and selection
  - COUNT: Aggregations
  - DESCRIBE: Entity descriptions
- **Features**:
  - Parameter binding
  - Timeout control
  - Pagination
  - Execution plan generation

## REST API Endpoints

### 1. List ETSI Domains

**Endpoint**: `GET /api/ontology/domains`

**Query Parameters**:
- `filter` (optional): Filter by domain name pattern
- `include_stats` (optional): Include domain statistics (default: false)

**Response**:
```json
{
  "domains": [
    {
      "id": "etsi-nfv",
      "name": "ETSI NFV",
      "description": "Network Functions Virtualization domain",
      "classCount": 125,
      "propertyCount": 250,
      "namespace": "http://etsi.org/nfv#",
      "updatedAt": "2025-10-17T12:00:00Z"
    }
  ],
  "totalCount": 3,
  "timestamp": "2025-10-17T12:00:00Z"
}
```

**Example**:
```bash
curl "http://localhost:8080/api/ontology/domains?include_stats=true"
```

### 2. List Ontology Classes

**Endpoint**: `GET /api/ontology/classes`

**Query Parameters**:
- `domain` (optional): Filter by domain
- `filter` (optional): Filter by class name pattern
- `include_subclasses` (optional): Include subclass IDs (default: false)
- `include_properties` (optional): Include class properties (default: false)
- `offset` (optional): Pagination offset (default: 0)
- `limit` (optional): Pagination limit (default: 50, max: 500)

**Response**:
```json
{
  "classes": [
    {
      "id": "vnf",
      "name": "VirtualNetworkFunction",
      "description": "A virtualized network function",
      "parentClasses": ["network-function"],
      "childClasses": ["vnfc", "vnf-instance"],
      "domain": "etsi-nfv",
      "properties": [...],
      "instanceCount": 42,
      "namespace": "http://etsi.org/nfv#"
    }
  ],
  "totalCount": 125,
  "offset": 0,
  "limit": 50,
  "timestamp": "2025-10-17T12:00:00Z"
}
```

**Example**:
```bash
curl "http://localhost:8080/api/ontology/classes?domain=etsi-nfv&include_properties=true&limit=10"
```

### 3. List Ontology Properties

**Endpoint**: `GET /api/ontology/properties`

**Query Parameters**:
- `domain` (optional): Filter by domain
- `filter` (optional): Filter by property name pattern
- `property_type` (optional): Filter by type (object_property, data_property, annotation_property)
- `include_constraints` (optional): Include cardinality constraints (default: false)
- `offset` (optional): Pagination offset
- `limit` (optional): Pagination limit (default: 50, max: 500)

**Response**:
```json
{
  "properties": [
    {
      "id": "has-vnfc",
      "name": "hasVNFC",
      "description": "VNF has VNFC components",
      "propertyType": "object_property",
      "domainClasses": ["vnf"],
      "rangeClasses": ["vnfc"],
      "isFunctional": false,
      "isInverseFunctional": false,
      "isTransitive": false,
      "isSymmetric": false,
      "cardinality": {
        "min": 1,
        "max": null,
        "exact": null
      },
      "domain": "etsi-nfv"
    }
  ],
  "totalCount": 250,
  "offset": 0,
  "limit": 50,
  "timestamp": "2025-10-17T12:00:00Z"
}
```

**Example**:
```bash
curl "http://localhost:8080/api/ontology/properties?property_type=object_property&include_constraints=true"
```

### 4. Get Entity Details

**Endpoint**: `GET /api/ontology/entities/:id`

**Path Parameters**:
- `id`: Entity identifier

**Query Parameters**:
- `include_incoming` (optional): Include incoming relationships (default: true)
- `include_outgoing` (optional): Include outgoing relationships (default: true)
- `include_inferred` (optional): Include inferred relationships (default: false)
- `max_depth` (optional): Maximum relationship traversal depth (default: 1)

**Response**:
```json
{
  "id": "vnf-123",
  "label": "Example VNF Instance",
  "entityType": "vnf",
  "properties": {
    "deploymentStatus": "deployed",
    "version": "1.2.3"
  },
  "incomingRelationships": [
    {
      "id": "rel-1",
      "sourceId": "vnf-manager-1",
      "targetId": "vnf-123",
      "relationshipType": "manages",
      "properties": {},
      "isInferred": false,
      "confidence": null
    }
  ],
  "outgoingRelationships": [...],
  "inferredRelationships": [...],
  "relatedEntities": ["vnf-manager-1", "vnfc-456"],
  "domain": "etsi-nfv",
  "createdAt": "2025-09-17T12:00:00Z",
  "updatedAt": "2025-10-17T10:00:00Z"
}
```

**Example**:
```bash
curl "http://localhost:8080/api/ontology/entities/vnf-123?include_inferred=true&max_depth=2"
```

### 5. Advanced Query Interface

**Endpoint**: `POST /api/ontology/query`

**Request Body**:
```json
{
  "query": "SELECT ?entity ?type WHERE { ?entity a ?type }",
  "parameters": {
    "domain": "etsi-nfv"
  },
  "limit": 100,
  "offset": 0,
  "explain": true,
  "timeoutSeconds": 30
}
```

**Response**:
```json
{
  "queryId": "550e8400-e29b-41d4-a716-446655440000",
  "results": [
    {
      "entity": "vnf-123",
      "type": "vnf"
    }
  ],
  "totalCount": 42,
  "executionTimeMs": 150,
  "executionPlan": "Query execution plan...",
  "fromCache": false,
  "timestamp": "2025-10-17T12:00:00Z"
}
```

**Query Examples**:

```sparql
-- Select all VNF instances
SELECT ?vnf WHERE { ?vnf a vnf }

-- Count entities by type
COUNT WHERE { ?entity a ?type }

-- Describe specific entity
DESCRIBE vnf-123

-- Filter by domain
SELECT ?entity ?status WHERE {
  ?entity a vnf .
  ?entity deploymentStatus ?status
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/api/ontology/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT ?entity WHERE { ?entity a vnf }",
    "limit": 50
  }'
```

### 6. Graph Visualization Data

**Endpoint**: `GET /api/ontology/graph`

**Query Parameters**:
- `domain` (optional): Filter by domain
- `max_nodes` (optional): Maximum nodes to return (default: 500, max: 2000)

**Response**:
```json
{
  "nodes": [
    {
      "id": "vnf-123",
      "label": "Example VNF",
      "nodeType": "vnf",
      "domain": "etsi-nfv",
      "size": 1.5,
      "color": "#4a90e2",
      "metadata": {
        "status": "deployed"
      }
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "source": "vnf-123",
      "target": "vnfc-456",
      "edgeType": "hasVNFC",
      "label": "has component",
      "isInferred": false,
      "weight": 1.0,
      "metadata": {}
    }
  ],
  "metadata": {
    "generatedAt": "2025-10-17T12:00:00Z",
    "nodeCount": 150,
    "edgeCount": 300
  }
}
```

**Example**:
```bash
curl "http://localhost:8080/api/ontology/graph?domain=etsi-nfv&max_nodes=200"
```

## WebSocket API

### Real-time Ontology Updates

**Endpoint**: `GET /api/ontology/stream` (WebSocket upgrade)

**Query Parameters**:
- `client_id` (optional): Client identifier for tracking

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:8080/api/ontology/stream?client_id=my-client-123');

ws.onopen = () => {
  console.log('Connected to ontology stream');
};

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  console.log('Received update:', update);
};
```

**Update Message Types**:

1. **Connection Established**
```json
{
  "type": "connected",
  "clientId": "my-client-123",
  "timestamp": "2025-10-17T12:00:00Z"
}
```

2. **Ontology Loaded**
```json
{
  "type": "OntologyLoaded",
  "ontologyId": "etsi-nfv-v2.1",
  "timestamp": "2025-10-17T12:00:00Z"
}
```

3. **Validation Started**
```json
{
  "type": "ValidationStarted",
  "jobId": "job-123",
  "timestamp": "2025-10-17T12:00:00Z"
}
```

4. **Validation Progress**
```json
{
  "type": "ValidationProgress",
  "jobId": "job-123",
  "progress": 0.65,
  "currentStep": "Checking cardinality constraints",
  "timestamp": "2025-10-17T12:00:30Z"
}
```

5. **Validation Completed**
```json
{
  "type": "ValidationCompleted",
  "jobId": "job-123",
  "reportId": "report-456",
  "violationsCount": 5,
  "timestamp": "2025-10-17T12:01:00Z"
}
```

6. **Entity Added**
```json
{
  "type": "EntityAdded",
  "entityId": "vnf-789",
  "entityType": "vnf",
  "timestamp": "2025-10-17T12:00:00Z"
}
```

7. **Entity Updated**
```json
{
  "type": "EntityUpdated",
  "entityId": "vnf-123",
  "changes": {
    "deploymentStatus": "terminated"
  },
  "timestamp": "2025-10-17T12:00:00Z"
}
```

8. **Relationship Added**
```json
{
  "type": "RelationshipAdded",
  "relationshipId": "rel-789",
  "sourceId": "vnf-123",
  "targetId": "vnfc-456",
  "relationshipType": "hasVNFC",
  "timestamp": "2025-10-17T12:00:00Z"
}
```

**Client Commands**:

Subscribe to specific updates:
```json
{
  "type": "subscribe",
  "filter": {
    "domain": "etsi-nfv",
    "entityTypes": ["vnf", "vnfc"]
  }
}
```

Unsubscribe:
```json
{
  "type": "unsubscribe"
}
```

## Integration with Existing Systems

### Actor System Integration

The ontology data API integrates with the existing actor system:

```rust
// Access ontology actor from AppState
if let Some(ontology_actor) = state.ontology_actor_addr.as_ref() {
    // Query ontology health
    let health = ontology_actor.send(GetOntologyHealth).await?;

    // Get cached ontologies
    let ontologies = ontology_actor.send(GetCachedOntologies).await?;
}
```

### Graph Visualization Integration

The `/api/ontology/graph` endpoint provides nodes and edges compatible with the existing graph visualization system:

```javascript
// Fetch ontology graph data
const response = await fetch('/api/ontology/graph?domain=etsi-nfv');
const graphData = await response.json();

// Integrate with existing graph
graphData.nodes.forEach(node => {
  // Add to graph renderer
  addOntologyNode(node);
});

graphData.edges.forEach(edge => {
  // Add relationships
  addOntologyEdge(edge);
});
```

## Performance Optimization

### Caching Strategy

1. **Query Results**: Cached for 1 hour with LRU eviction
2. **Entity Data**: Cached for 1 hour per entity
3. **Domain/Class Listings**: Cached until ontology updates
4. **Pattern Invalidation**: Invalidate related caches on updates

### Database Indexing

```sql
CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_domain ON entities(domain_id);
CREATE INDEX idx_relationships_source ON relationships(source_id);
CREATE INDEX idx_relationships_target ON relationships(target_id);
CREATE INDEX idx_classes_domain ON classes(domain_id);
CREATE INDEX idx_properties_domain ON properties(domain_id);
```

### Pagination Limits

- Default limit: 50 items
- Maximum limit: 500 items (domains/classes/properties)
- Maximum graph nodes: 2000 nodes

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Entity not found",
  "code": "ENTITY_NOT_FOUND",
  "details": {
    "entityId": "vnf-999"
  },
  "timestamp": "2025-10-17T12:00:00Z",
  "traceId": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Codes**:
- `DB_ERROR`: Database initialization or query failed
- `DOMAIN_LIST_FAILED`: Failed to list domains
- `CLASS_LIST_FAILED`: Failed to list classes
- `PROPERTY_LIST_FAILED`: Failed to list properties
- `ENTITY_NOT_FOUND`: Entity does not exist
- `ENTITY_RETRIEVAL_FAILED`: Failed to retrieve entity
- `QUERY_FAILED`: Query execution failed
- `GRAPH_VIZ_FAILED`: Graph visualization generation failed

## Testing

Run tests:
```bash
cargo test --features ontology ontology_data
```

Example test:
```bash
# Test domain listing
curl http://localhost:8080/api/ontology/domains

# Test class listing with filters
curl "http://localhost:8080/api/ontology/classes?domain=etsi-nfv&limit=5"

# Test entity retrieval
curl http://localhost:8080/api/ontology/entities/vnf-123

# Test query interface
curl -X POST http://localhost:8080/api/ontology/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT ?entity WHERE { ?entity a vnf }", "limit": 10}'

# Test WebSocket
wscat -c "ws://localhost:8080/api/ontology/stream?client_id=test-123"
```

## Future Enhancements

1. **Full SPARQL Support**: Implement complete SPARQL 1.1 query language
2. **GraphQL Interface**: Add GraphQL endpoint for flexible queries
3. **Batch Operations**: Support batch entity/relationship creation
4. **Export Formats**: Support RDF/XML, Turtle, JSON-LD export
5. **Access Control**: Add fine-grained permissions for domains/classes
6. **Versioning**: Track ontology versions and changes over time
7. **Audit Log**: Record all entity and relationship modifications
8. **Analytics**: Provide usage analytics and query performance metrics

## Dependencies

```toml
[dependencies]
rusqlite = { version = "0.32", features = ["bundled"] }
lru = "0.12"
actix-web = "4.11"
actix-web-actors = "4.3"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1.18", features = ["v4", "serde"] }
log = "0.4"
```

## See Also

- [Ontology Validation API](./ontology-physics-integration.md)
- [Actor System Documentation](./architecture/ARCHITECTURE_INDEX.md)
- [Graph Visualization Guide](./specialized/graph-visualization.md)
