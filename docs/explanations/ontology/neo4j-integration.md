---
title: Neo4j Integration Documentation
description: The Neo4j integration provides dual persistence to both SQLite (`unified.db`) and Neo4j graph database, enabling:
type: explanation
status: stable
---

# Neo4j Integration Documentation

## Overview

The Neo4j integration provides dual persistence to both SQLite (`unified.db`) and Neo4j graph database, enabling:

- **SQLite (unified.db)**: Fast local queries, physics state persistence, primary source of truth
- **Neo4j**: Complex graph traversals, multi-hop reasoning, semantic analysis via Cypher

## Architecture

### Components

1. **Neo4jAdapter** (`src/adapters/neo4j-adapter.rs`)
   - Implements `KnowledgeGraphRepository` port
   - Handles Neo4j connection and schema management
   - Provides Cypher query execution

2. **DualGraphRepository** (`src/adapters/dual-graph-repository.rs`)
   - Wraps both `UnifiedGraphRepository` (SQLite) and `Neo4jAdapter`
   - Implements dual-write pattern
   - Configurable strict/non-strict mode for error handling

3. **CypherQueryHandler** (`src/handlers/cypher-query-handler.rs`)
   - REST API endpoints for Cypher queries
   - Safety features: timeouts, result limits, read-only enforcement
   - Example queries and documentation

4. **Sync Script** (`scripts/sync-neo4j.rs`)
   - Migrates data from unified.db to Neo4j
   - Supports full and incremental sync
   - Dry-run mode for testing

## Database Schema

### Neo4j Schema

**Nodes** (`GraphNode`):
```cypher
(:GraphNode {
  id: Integer,              // Unique numeric ID
  metadata-id: String,      // File path or identifier
  label: String,            // Display label
  x, y, z: Float,          // Position
  vx, vy, vz: Float,       // Velocity
  mass: Float,             // Physics mass
  owl-class-iri: String,   // OWL ontology class (optional)
  color: String,           // Visualization color (optional)
  size: Float,             // Node size (optional)
  node-type: String,       // Type classification (optional)
  weight: Float,           // Graph weight (optional)
  group-name: String,      // Group/cluster (optional)
  metadata: String         // JSON metadata (optional)
})
```

**Relationships** (`EDGE`):
```cypher
[:EDGE {
  weight: Float,           // Edge weight
  relation-type: String,   // Edge type (optional)
  owl-property-iri: String, // OWL property IRI (optional)
  metadata: String         // JSON metadata (optional)
}]
```

**Indexes and Constraints**:
```cypher
CREATE CONSTRAINT graph-node-id IF NOT EXISTS
FOR (n:GraphNode) REQUIRE n.id IS UNIQUE

CREATE INDEX graph-node-metadata-id IF NOT EXISTS
FOR (n:GraphNode) ON (n.metadata-id)

CREATE INDEX graph-node-owl-class IF NOT EXISTS
FOR (n:GraphNode) ON (n.owl-class-iri)
```

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Neo4j Configuration
NEO4J-URI=bolt://localhost:7687
NEO4J-USER=neo4j
NEO4J-PASSWORD=your-secure-password
NEO4J-DATABASE=neo4j
NEO4J-ENABLED=true
```

### Docker Compose

Add Neo4j service to `docker-compose.yml`:

```yaml
services:
  neo4j:
    image: neo4j:5.15-community
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J-AUTH: neo4j/your-secure-password
      NEO4J-PLUGINS: '["apoc"]'
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
    networks:
      - webxr-network

volumes:
  neo4j-data:
  neo4j-logs:
```

### Application Integration

**Option 1: Dual-Write Mode** (Recommended for new deployments)

```rust
use webxr::adapters::neo4j-adapter::{Neo4jAdapter, Neo4jConfig};
use webxr::adapters::dual-graph-repository::DualGraphRepository;
use webxr::repositories::unified-graph-repository::UnifiedGraphRepository;

// Initialize repositories
let sqlite-repo = Arc::new(UnifiedGraphRepository::new("/app/data/unified.db")?);

let neo4j-config = Neo4jConfig::default();
let neo4j = Arc::new(Neo4jAdapter::new(neo4j-config).await?);

// Create dual repository
let dual-repo = Arc::new(DualGraphRepository::new(
    sqlite-repo,
    Some(neo4j),
    false, // Non-strict mode: log Neo4j errors but don't fail
));

// Use dual-repo as KnowledgeGraphRepository
```

**Option 2: SQLite-Only Mode** (Existing deployments)

```rust
// Continue using UnifiedGraphRepository directly
let repo = Arc::new(UnifiedGraphRepository::new("/app/data/unified.db")?);
```

## Migration Guide

### Initial Sync: unified.db → Neo4j

1. **Start Neo4j**:
   ```bash
   docker-compose up -d neo4j
   ```

2. **Verify Neo4j is accessible**:
   ```bash
   curl http://localhost:7474
   # Or open in browser
   ```

3. **Run full sync**:
   ```bash
   cargo run --bin sync-neo4j -- --full
   ```

4. **Verify sync**:
   ```bash
   # Check in Neo4j Browser (http://localhost:7474)
   MATCH (n:GraphNode) RETURN count(n) AS nodes
   MATCH ()-[r:EDGE]->() RETURN count(r) AS edges
   ```

### Incremental Sync

For existing deployments, sync only new/modified data:

```bash
# Dry run first
cargo run --bin sync-neo4j -- --dry-run

# Actual sync
cargo run --bin sync-neo4j
```

### Sync Options

```bash
# Full sync (clears Neo4j first)
cargo run --bin sync-neo4j -- --full

# Dry run (preview without changes)
cargo run --bin sync-neo4j -- --dry-run

# Custom database path
cargo run --bin sync-neo4j -- --db=/custom/path/unified.db
```

## API Endpoints

### Execute Cypher Query

**POST** `/api/query/cypher`

Execute a Cypher query with safety limits.

**Request**:
```json
{
  "query": "MATCH (n:GraphNode {id: $node-id})-[:EDGE*1..3]-(m:GraphNode) RETURN m.label, m.owl-class-iri LIMIT 10",
  "parameters": {
    "node-id": 42
  },
  "limit": 100,
  "timeout": 30
}
```

**Response**:
```json
{
  "results": [
    {
      "m.label": "Example Node",
      "m.owl-class-iri": "http://example.org/ontology#Class"
    }
  ],
  "count": 1,
  "truncated": false,
  "execution-time-ms": 45
}
```

**Safety Features**:
- Query timeout (max 300 seconds)
- Result limit (max 10,000 results)
- Write operation blocking (DELETE, CREATE, SET, MERGE)
- Parameterized queries prevent injection

### Get Example Queries

**GET** `/api/query/cypher/examples`

Returns common Cypher query patterns.

## Cypher Query Examples

### 1. Find Neighbors

```cypher
MATCH (n:GraphNode {id: $node-id})-[:EDGE]-(m:GraphNode)
RETURN m
```

### 2. Multi-Hop Path Analysis

```cypher
MATCH (n:GraphNode {id: $node-id})-[:EDGE*1..3]-(m:GraphNode)
RETURN DISTINCT m.id, m.label
```

### 3. Shortest Path

```cypher
MATCH p=shortestPath((n:GraphNode {id: $start-id})-[:EDGE*]-(m:GraphNode {id: $end-id}))
RETURN p, length(p) AS hops
```

### 4. Nodes by OWL Class

```cypher
MATCH (n:GraphNode {owl-class-iri: $iri})
RETURN n.id, n.label, n.metadata
```

### 5. High-Degree Nodes (Hubs)

```cypher
MATCH (n:GraphNode)-[r:EDGE]-()
WITH n, count(r) AS degree
ORDER BY degree DESC
LIMIT $limit
RETURN n.id, n.label, degree
```

### 6. Semantic Path by OWL Properties

```cypher
MATCH (n:GraphNode {id: $start-id})-[r:EDGE*1..5]->(m:GraphNode)
WHERE ALL(rel IN r WHERE rel.owl-property-iri = $property-iri)
RETURN m.id, m.label
```

### 7. Cluster Detection

```cypher
MATCH (n:GraphNode)-[:EDGE]-(m:GraphNode)
WHERE n.group-name = $group
RETURN n, m
```

## Performance Considerations

### Dual-Write Strategy

- **Primary (SQLite)**: All operations execute here first
- **Secondary (Neo4j)**: Operations execute asynchronously
- **Failure Handling**:
  - **Strict mode** (`strict-mode: true`): Fail entire operation if Neo4j fails
  - **Non-strict mode** (`strict-mode: false`): Log Neo4j errors, continue with SQLite

### Query Performance

- **Read queries**: Always from SQLite (faster for simple queries)
- **Complex graph queries**: Use Cypher endpoint for Neo4j
- **Indexes**: Automatically created on `id`, `metadata-id`, `owl-class-iri`

### Scaling

- **SQLite**: Single-node, file-based
- **Neo4j**: Can scale to millions of nodes/relationships
- **Recommendation**: Use SQLite for ≤100k nodes, Neo4j for larger graphs

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to Neo4j

```
Failed to connect to Neo4j: Connection refused
```

**Solutions**:
1. Verify Neo4j is running: `docker-compose ps`
2. Check Neo4j logs: `docker-compose logs neo4j`
3. Verify URI in `.env`: `NEO4J-URI=bolt://localhost:7687`
4. Test connection: `curl http://localhost:7474`

### Sync Failures

**Problem**: Nodes/edges fail to sync

```
⚠️  Failed to sync node 42: Constraint violation
```

**Solutions**:
1. Run full sync to clear conflicts: `--full`
2. Check for duplicate IDs in unified.db
3. Verify Neo4j constraints: `SHOW CONSTRAINTS`

### Query Timeout

**Problem**: Cypher query times out

```
Query exceeded timeout of 30 seconds
```

**Solutions**:
1. Increase timeout in request: `"timeout": 300`
2. Optimize query with indexes
3. Add `LIMIT` clauses to reduce result size
4. Use pagination for large result sets

## Security Best Practices

1. **Authentication**:
   - Change default Neo4j password
   - Use strong passwords (16+ characters)
   - Rotate credentials regularly

2. **Network Security**:
   - Restrict Neo4j ports to internal network
   - Use TLS for production (`bolt+s://`)
   - Firewall external access

3. **Query Safety**:
   - Write operations blocked by default in API
   - Use parameterized queries
   - Enforce timeouts and result limits
   - Validate user input

4. **Data Privacy**:
   - Encrypt Neo4j data at rest
   - Use access control lists (ACL)
   - Audit query logs

## Development vs Production

### Development

```bash
# Use Docker Compose
docker-compose up neo4j

# Connect to localhost
NEO4J-URI=bolt://localhost:7687
```

### Production

```bash
# Use managed Neo4j (AuraDB, EC2, etc.)
NEO4J-URI=neo4j+s://xxxxx.databases.neo4j.io

# Enable TLS
# Use strong passwords
# Set up monitoring
# Configure backups
```

## Monitoring

### Health Checks

```rust
// Check both databases
let sqlite-ok = dual-repo.health-check().await?;

// Check Neo4j specifically
let neo4j-ok = neo4j.health-check().await?;
```

### Statistics

```cypher
// Node/edge counts
MATCH (n:GraphNode) RETURN count(n) AS nodes
MATCH ()-[r:EDGE]->() RETURN count(r) AS edges

// Average degree
MATCH (n:GraphNode)-[r:EDGE]-()
WITH n, count(r) AS degree
RETURN avg(degree) AS avg-degree

// Storage size
CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes')
YIELD attributes
RETURN attributes
```

## Future Enhancements

1. **Real-time Sync**: Change Data Capture (CDC) from SQLite
2. **Conflict Resolution**: Automatic merging of divergent states
3. **Partitioning**: Distribute graph across multiple Neo4j instances
4. **Graph Algorithms**: PageRank, community detection, centrality
5. **Visualization**: Neo4j Bloom integration

## Support

- **Neo4j Documentation**: https://neo4j.com/docs/
- **Cypher Reference**: https://neo4j.com/docs/cypher-manual/
- **APOC Library**: https://neo4j.com/labs/apoc/

## License

Same license as the main project.
