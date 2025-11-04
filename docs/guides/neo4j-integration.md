# Neo4j Integration Guide

**Status**: ⚙️ Code exists, wiring needed
**Last Updated**: November 3, 2025

---

## Overview

VisionFlow supports optional Neo4j graph database integration for advanced graph analytics alongside the primary SQLite unified.db storage. This enables Cypher queries, multi-hop path analysis, and OWL semantic relationship exploration.

---

## Quick Start

### 1. Environment Configuration

```bash
# Required
export NEO4J-URI="bolt://localhost:7687"
export NEO4J-USER="neo4j"
export NEO4J-PASSWORD="your-password"

# Optional
export NEO4J-DATABASE="neo4j"  # Default database
export NEO4J-STRICT-MODE="false"  # Log errors but don't fail
```

### 2. Start Neo4j

```bash
# Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J-AUTH=neo4j/your-password \
  neo4j:latest
```

### 3. Verify Connection

```bash
# Health check
curl http://localhost:4000/api/neo4j/health

# Get query examples
curl http://localhost:4000/api/query/cypher/examples
```

---

## Architecture

### Dual-Write Strategy

```
Write Request
    ↓
DualGraphRepository
    ├─→ SQLite (Primary) - MUST succeed
    └─→ Neo4j (Secondary) - MAY fail (if strict-mode=false)
```

### Read Strategy

All reads come from **SQLite (unified.db)** for consistency.

---

## Integration Checklist

### Step 1: Add Module Export

**File**: `src/handlers/mod.rs`

Add after line 10:
```rust
pub mod cypher-query-handler;
```

### Step 2: Add AppState Fields

**File**: `src/app-state.rs`

Add after existing neo4j-adapter field (if not present):
```rust
/// Neo4j adapter for graph analytics (optional)
pub neo4j-adapter: Option<Arc<Neo4jAdapter>>,
```

### Step 3: Initialize Neo4j in AppState::new()

**File**: `src/app-state.rs`

Add after unified graph repository creation:
```rust
// ============================================================================
// Neo4j Integration (Optional Dual-Write Repository)
// ============================================================================
info!("[AppState::new] Checking for Neo4j configuration...");

let neo4j-adapter: Option<Arc<Neo4jAdapter>> = if std::env::var("NEO4J-URI").is-ok() {
    info!("[AppState::new] NEO4J-URI detected, attempting to connect...");

    match Neo4jAdapter::new(Neo4jConfig::default()).await {
        Ok(adapter) => {
            info!("✅ Neo4j adapter connected successfully");
            Some(Arc::new(adapter))
        }
        Err(e) => {
            warn!("⚠️  Failed to initialize Neo4j adapter: {}", e);
            warn!("⚠️  Continuing with SQLite-only mode");
            None
        }
    }
} else {
    info!("ℹ️  NEO4J-URI not set - running SQLite-only mode");
    None
};

// Wrap in DualGraphRepository if Neo4j is available
let knowledge-graph-repository: Arc<dyn KnowledgeGraphRepository> = if neo4j-adapter.is-some() {
    let strict-mode = std::env::var("NEO4J-STRICT-MODE")
        .ok()
        .and-then(|v| v.parse::<bool>().ok())
        .unwrap-or(false);

    info!("🔗 Creating DualGraphRepository (SQLite + Neo4j)");
    info!("   Strict mode: {}", strict-mode);

    Arc::new(DualGraphRepository::new(
        knowledge-graph-repository,
        neo4j-adapter.clone(),
        strict-mode,
    ))
} else {
    knowledge-graph-repository
};
```

### Step 4: Register Routes in main.rs

**File**: `src/main.rs`

Add to imports:
```rust
cypher-query-handler,
```

Add to route configuration:
```rust
.configure(cypher-query-handler::configure-routes)
```

---

## REST API Endpoints

### Health Check
```bash
GET /api/neo4j/health
```

**Response**:
```json
{
  "status": "connected",
  "uri": "bolt://localhost:7687",
  "database": "neo4j"
}
```

### Execute Cypher Query
```bash
POST /api/query/cypher
Content-Type: application/json

{
  "query": "MATCH (n:GraphNode) RETURN n.id, n.label LIMIT 10",
  "parameters": {},
  "limit": 10,
  "timeout": 30
}
```

**Response**:
```json
{
  "results": [
    {"n.id": 1, "n.label": "Person"},
    {"n.id": 2, "n.label": "Organization"}
  ],
  "count": 2,
  "truncated": false,
  "execution-time-ms": 42
}
```

### Get Query Examples
```bash
GET /api/query/cypher/examples
```

---

## Example Queries

### Find Neighbors (1-3 hops)
```cypher
MATCH (n:GraphNode {id: $node-id})-[:EDGE*1..3]-(m:GraphNode)
RETURN DISTINCT m.id, m.label
```

### Find Semantic Paths (OWL)
```cypher
MATCH (n:GraphNode {id: $start-id})-[r:EDGE*1..3]->(m:GraphNode)
WHERE ALL(rel IN r WHERE rel.owl-property-iri = 'http://www.w3.org/2000/01/rdf-schema#subClassOf')
RETURN m.id, m.label, m.owl-class-iri, length(r) AS depth
ORDER BY depth
LIMIT 50
```

### Find Disjoint Classes
```cypher
MATCH (c1:GraphNode)-[:EDGE {owl-property-iri: 'http://www.w3.org/2002/07/owl#disjointWith'}]->(c2:GraphNode)
RETURN c1.label, c2.label
```

---

## Safety Features

### Query Restrictions
The Cypher handler **blocks** destructive operations:
- `DELETE`
- `SET`
- `CREATE`
- `MERGE`

### Timeout Protection
- Default timeout: 30 seconds
- Maximum timeout: 300 seconds

### Result Limits
- Default limit: 100 results
- Maximum limit: 10,000 results

---

## Troubleshooting

### Issue: "Neo4j Not Configured"
**Solution**: Set `NEO4J-URI` environment variable and restart.

### Issue: Connection Refused
**Solution**: Verify Neo4j is running on port 7687.

### Issue: Authentication Failed
**Solution**: Check `NEO4J-USER` and `NEO4J-PASSWORD` match Neo4j configuration.

### Issue: Dual-Write Failures (strict-mode=true)
**Solution**: Set `NEO4J-STRICT-MODE=false` to continue on Neo4j errors.

---

## Migration from Legacy

If upgrading from the old three-database architecture:

1. **Export from SQLite**: Existing unified.db data is preserved
2. **Import to Neo4j**: Use dual-write mode to populate Neo4j
3. **Verify**: Query both databases to ensure consistency

---

## Schema

### Node Labels
- `GraphNode`: All nodes from unified.db

### Node Properties
- `id`: Integer node ID
- `label`: Human-readable label
- `metadata-id`: JSON metadata reference
- `owl-class-iri`: OWL class IRI (if applicable)
- `owl-property-iri`: OWL property IRI (if applicable)

### Relationship Type
- `EDGE`: All edges from unified.db

### Indexes
- Uniqueness constraint on `GraphNode.id`
- Index on `metadata-id`
- Index on `owl-class-iri`

---

## Performance Considerations

### When to Use Neo4j
- Multi-hop path queries (3+ hops)
- Graph analytics (centrality, communities)
- OWL semantic reasoning queries

### When to Use SQLite
- Simple CRUD operations
- Single-node lookups
- Bulk imports/exports

### Dual-Write Overhead
Minimal (~5-10ms per write) in non-strict mode.

---

## References

- [Neo4j Adapter Code](../../src/adapters/neo4j-adapter.rs)
- [Dual Repository Code](../../src/adapters/dual-graph-repository.rs)
- [Cypher Handler Code](../../src/handlers/cypher-query-handler.rs)
- [Integration Checklist (Historical)](../NEO4j-integration-checklist.md)
