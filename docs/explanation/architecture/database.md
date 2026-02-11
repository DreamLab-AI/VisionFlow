---
title: Database Architecture
description: VisionFlow uses **Neo4j** as the **single source of truth** for all graph data, user settings, ontology information, and application state.
category: explanation
tags:
  - architecture
  - design
  - patterns
  - structure
  - api
related-docs:
  - concepts/architecture/core/server.md
  - concepts/hexagonal-architecture.md
  - guides/architecture/actor-system.md
  - guides/graphserviceactor-migration.md
  - README.md
updated-date: 2026-02-11
difficulty-level: advanced
dependencies:
  - Neo4j database
---

# Database Architecture

**Database**: Neo4j 5.x
**Status**: Production
**Migration**: Completed November 2025 (from SQLite)
**Last Updated**: February 11, 2026

---

## Executive Summary

VisionFlow uses **Neo4j** as the **single source of truth** for all graph data, user settings, ontology information, and application state. The migration from SQLite to Neo4j was completed in November 2025, providing:

- **Graph-native storage** - Natural fit for node/edge data structures
- **Cypher queries** - Expressive query language for complex graph patterns
- **ACID transactions** - Full consistency guarantees
- **User authentication** - Nostr-based identity with per-user settings
- **Scalability** - Neo4j clustering support for future horizontal scaling

### What Works Well ✅

- **Performance**: Simple queries <2ms, complex traversals <50ms
- **Data model**: Natural graph structure matches domain perfectly
- **User isolation**: Clean separation of user data via Nostr public keys
- **Ontology storage**: OWL classes/properties map cleanly to Neo4j
- **Migrations**: Bolt protocol makes schema updates straightforward

### Current Limitations ⚠️

- **Cost**: Neo4j Enterprise required for clustering ($$$)
- **Learning curve**: Cypher syntax different from SQL
- **Connection pool**: Fixed size (10 connections), no dynamic scaling
- **Backup strategy**: Manual backups only, no automated point-in-time recovery
- **Deployment complexity**: Requires separate Neo4j server (can't embed)

---

## Neo4j Schema

### Node Labels

#### Graph Visualization Data

**Node** - Graph visualization nodes
```cypher
CREATE (n:Node {
  id: 1,                           // Internal VisionFlow ID (u32)
  label: "Example Node",           // Display label
  x: 1.5, y: 2.3, z: 0.8,         // 3D position
  metadata_id: "node-abc-123",    // Optional metadata reference
  quality_score: 0.85,            // Quality score (0.0-1.0)
  created_at: datetime(),
  updated_at: datetime()
})
```

**Edge** - Graph connections
```cypher
CREATE (n1:Node {id: 1})-[:EDGE {
  relationship: "depends_on",      // Relationship type
  weight: 1.0,                     // Edge weight
  created_at: datetime()
}]->(n2:Node {id: 2})
```

#### User Data (Nostr Authentication)

**User** - Authenticated users
```cypher
CREATE (u:User {
  npub: "npub1abc...",            // Nostr public key (identity)
  created_at: datetime(),
  last_seen: datetime()
})
```

**UserSettings** - Per-user configuration
```cypher
CREATE (s:UserSettings {
  user_id: "npub1abc...",         // Links to User
  key: "theme",                    // Setting key
  value: "dark",                   // Setting value (string)
  category: "appearance",          // Category grouping
  updated_at: datetime()
})

// Relationship to user
MATCH (u:User {npub: "npub1abc..."})
CREATE (u)-[:HAS_SETTINGS]->(s)
```

**FilterState** - Per-client filter state
```cypher
CREATE (f:FilterState {
  client_id: "ws-conn-12345",     // WebSocket connection ID
  user_id: "npub1abc...",         // Links to User
  filter_json: '{"threshold": 0.5}', // JSON filter config
  created_at: datetime(),
  expires_at: datetime()
})
```

#### Ontology Data

**OntologyClass** - OWL classes
```cypher
CREATE (c:OntologyClass {
  iri: "http://example.org/Class1",
  label: "Class 1",
  description: "Example class",
  is_inferred: false              // False = explicit, True = reasoner-inferred
})
```

**OntologyProperty** - OWL properties
```cypher
CREATE (p:OntologyProperty {
  iri: "http://example.org/property1",
  label: "property 1",
  domain: "http://example.org/Class1",
  range: "xsd:string",
  is_functional: true
})
```

**Axiom** - OWL axioms
```cypher
CREATE (a:Axiom {
  subject: "http://example.org/Class1",
  predicate: "rdfs:subClassOf",
  object: "http://example.org/Class2",
  is_inferred: false
})
```

**Constraint** - Semantic constraints (for physics)
```cypher
CREATE (c:Constraint {
  source_node_id: 1,
  target_node_id: 2,
  constraint_type: "min_distance",
  value: 5.0,                     // Minimum distance in 3D space
  created_at: datetime()
})
```

### Indexes

Performance-critical indexes:

```cypher
// Node lookup by ID (primary key)
CREATE INDEX node_id_index FOR (n:Node) ON (n.id);

// User lookup by Nostr public key
CREATE INDEX user_npub_index FOR (u:User) ON (u.npub);

// Settings lookup by user and key
CREATE INDEX settings_user_key_index FOR (s:UserSettings) ON (s.user_id, s.key);

// Filter state lookup by client
CREATE INDEX filter_client_index FOR (f:FilterState) ON (f.client_id);

// Ontology class lookup by IRI
CREATE INDEX ontology_class_iri_index FOR (c:OntologyClass) ON (c.iri);
```

### Relationships

```cypher
// Graph edges
(Node)-[:EDGE {relationship, weight}]->(Node)

// User settings
(User)-[:HAS_SETTINGS]->(UserSettings)

// User filter states
(User)-[:HAS_FILTER_STATE]->(FilterState)

// Ontology class hierarchy
(OntologyClass)-[:SUBCLASS_OF]->(OntologyClass)

// Property domain/range
(OntologyProperty)-[:HAS_DOMAIN]->(OntologyClass)
(OntologyProperty)-[:HAS_RANGE]->(OntologyClass)
```

---

## Common Query Patterns

### Graph Data Queries

#### Get All Nodes and Edges

```cypher
// Get nodes
MATCH (n:Node)
RETURN n.id AS id, n.label AS label, n.x AS x, n.y AS y, n.z AS z, n.metadata_id AS metadata_id, n.quality_score AS quality_score
ORDER BY n.id;

// Get edges
MATCH (n1:Node)-[e:EDGE]->(n2:Node)
RETURN n1.id AS source, n2.id AS target, e.relationship AS relationship, e.weight AS weight;
```

**Performance**: ~12ms for 1,000 nodes + 5,000 edges

#### Add Node with Transaction

```cypher
// Atomic node creation
BEGIN
  CREATE (n:Node {
    id: $id,
    label: $label,
    x: $x, y: $y, z: $z,
    metadata_id: $metadata_id,
    quality_score: $quality_score,
    created_at: datetime(),
    updated_at: datetime()
  })
  RETURN n;
COMMIT
```

#### Batch Update Node Positions (Physics)

```cypher
// Efficient batch update (called every physics step)
UNWIND $positions AS pos
MATCH (n:Node {id: pos.id})
SET n.x = pos.x, n.y = pos.y, n.z = pos.z, n.updated_at = datetime()
RETURN count(n) AS updated_count;
```

**Performance**: ~4ms for 100 nodes, ~16ms for 1,000 nodes

### User Settings Queries

#### Get All Settings for User

```cypher
MATCH (u:User {npub: $npub})-[:HAS_SETTINGS]->(s:UserSettings)
RETURN s.key AS key, s.value AS value, s.category AS category
ORDER BY s.category, s.key;
```

#### Upsert User Setting (Atomic)

```cypher
MERGE (u:User {npub: $npub})
ON CREATE SET u.created_at = datetime()
SET u.last_seen = datetime()

MERGE (u)-[:HAS_SETTINGS]->(s:UserSettings {user_id: $npub, key: $key})
ON CREATE SET s.created_at = datetime()
SET s.value = $value, s.category = $category, s.updated_at = datetime()

RETURN s;
```

**Performance**: ~3ms per setting

### Filter State Queries

#### Get Filter State for Client

```cypher
MATCH (f:FilterState {client_id: $client_id})
WHERE f.expires_at > datetime()
RETURN f.filter_json AS filter_json;
```

#### Clean Up Expired Filter States (Maintenance)

```cypher
MATCH (f:FilterState)
WHERE f.expires_at < datetime()
DELETE f;
```

### Ontology Queries

#### Get Ontology Class Hierarchy

```cypher
// Get class with all superclasses
MATCH path = (c:OntologyClass {iri: $iri})-[:SUBCLASS_OF*]->(parent:OntologyClass)
RETURN c, collect(parent) AS ancestors;
```

#### Find Inferred Axioms

```cypher
// Get all axioms inferred by reasoner
MATCH (a:Axiom {is_inferred: true})
RETURN a.subject AS subject, a.predicate AS predicate, a.object AS object
ORDER BY a.subject;
```

---

## Performance Benchmarks

Measured on development hardware (Neo4j 5.13, 16 GB RAM):

| Query Type | Dataset Size | Latency (p50) | Latency (p95) | Notes |
|------------|--------------|---------------|---------------|-------|
| Get all nodes | 1,000 nodes | 8ms | 15ms | Simple MATCH |
| Get all edges | 5,000 edges | 12ms | 28ms | Includes relationships |
| Get nodes + edges | 1K nodes, 5K edges | 18ms | 35ms | Combined query |
| Add single node | 1 node | 3ms | 8ms | Single CREATE |
| Batch update positions | 100 nodes | 4ms | 10ms | UNWIND + SET |
| Batch update positions | 1,000 nodes | 16ms | 38ms | UNWIND + SET |
| Get user settings | 20 settings | 2ms | 5ms | MATCH + relationship |
| Upsert setting | 1 setting | 3ms | 9ms | MERGE + SET |
| Ontology traversal | 10 levels deep | 25ms | 60ms | Path query |

### Optimization Notes

**What's fast**:
- Single-node lookups by ID (indexed): <2ms
- Direct relationship traversals: <5ms
- Batch operations with UNWIND: 4-20ms

**What's slow**:
- Deep graph traversals (>5 hops): 50-200ms
- Full table scans without indexes: 100ms+
- Complex aggregations: Variable

**Future optimizations**:
- Add more indexes for common query patterns
- Use query result caching (Redis)
- Partition large graphs by domain

---

## Migration from SQLite

### Migration Timeline

- **November 2, 2025**: Unified database architecture designed
- **November 3, 2025**: Neo4j adapter implemented
- **November 4, 2025**: User settings migration completed
- **November 5, 2025**: GraphServiceActor removed, Neo4j as source of truth

### Why Neo4j?

**Advantages over SQLite**:
1. **Graph-native**: Natural fit for node/edge data structures
2. **Cypher queries**: More expressive than SQL for graph patterns
3. **Relationships**: First-class citizens, not foreign keys
4. **Performance**: Optimized for graph traversals
5. **Scalability**: Clustering support for horizontal scaling
6. **ACID**: Full transactional guarantees

**What we gave up**:
1. **Simplicity**: SQLite = single file, Neo4j = separate server
2. **Cost**: Neo4j Enterprise required for production clustering
3. **Portability**: SQLite is portable, Neo4j requires deployment
4. **Familiarity**: SQL is more widely known than Cypher

### Migration Lessons Learned

**What went well**:
- Bolt protocol makes migrations straightforward
- Cypher is intuitive for graph operations
- Transaction support prevented data corruption
- Neo4j performance exceeded expectations

**What was challenging**:
- Learning Cypher syntax (different from SQL)
- Setting up Neo4j server (vs embedded SQLite)
- Connection pool tuning (trial and error)
- Query optimization (different patterns than SQL)

**Technical debt**:
- ActorGraphRepository bridge pattern (low priority to remove)
- Some queries not fully optimized (need profiling)
- Manual backup strategy (need automation)
- No distributed deployment yet (planned for Q2 2026)

---

## Connection Management

### Connection Pool Configuration

```rust
// src/adapters/neo4j_adapter.rs
pub struct Neo4jConfig {
    pub uri: String,                  // "bolt://localhost:7687"
    pub username: String,             // "neo4j"
    pub password: String,
    pub max_connections: usize,       // Default: 10
    pub connection_timeout: Duration, // Default: 30s
    pub max_query_time: Duration,     // Default: 60s
}
```

**Current settings**:
- Max connections: **10** (fixed)
- Connection timeout: **30 seconds**
- Max query time: **60 seconds**
- Pool exhaustion strategy: **Wait** (blocks until connection available)

**Limitations**:
- No dynamic pool sizing (fixed at 10)
- No connection eviction (connections live forever)
- No connection health checks
- No retry logic (caller must retry)

**Future improvements**:
- Dynamic pool sizing (min 5, max 50)
- Connection health checks (ping every 5s)
- Automatic retry with exponential backoff
- Circuit breaker for connection failures

---

## Backup and Recovery

### Current Strategy: Manual Backups

**Daily backup script**:
```bash
#!/bin/bash
# scripts/backup-neo4j.sh

NEO4J_HOME="/var/lib/neo4j"
BACKUP_DIR="/backups/neo4j"
DATE=$(date +%Y-%m-%d-%H%M%S)

# Stop Neo4j
systemctl stop neo4j

# Copy data directory
cp -r $NEO4J_HOME/data $BACKUP_DIR/neo4j-backup-$DATE

# Start Neo4j
systemctl start neo4j

# Keep last 7 days of backups
find $BACKUP_DIR -name "neo4j-backup-*" -mtime +7 -delete
```

**Limitations**:
- Requires Neo4j downtime (minutes)
- No point-in-time recovery
- No incremental backups
- Manual restore process

### Future Strategy: Automated Backups

**Planned for Q1 2026**:
1. **Online backups** - Neo4j Enterprise feature (no downtime)
2. **Incremental backups** - Daily incrementals, weekly fulls
3. **Point-in-time recovery** - Transaction log archiving
4. **Automated restore testing** - Monthly restore drills
5. **Off-site replication** - AWS S3 backup storage

---

## Query Optimization Techniques

### 1. Use Indexes

```cypher
// ❌ BAD: Full scan
MATCH (n:Node)
WHERE n.label = "Example"
RETURN n;
// Scans all nodes

// ✅ GOOD: Index lookup
CREATE INDEX node_label_index FOR (n:Node) ON (n.label);
MATCH (n:Node {label: "Example"})
RETURN n;
// Uses index
```

### 2. Limit Results

```cypher
// ❌ BAD: Returns all nodes
MATCH (n:Node)
RETURN n;
// Returns 10,000+ nodes

// ✅ GOOD: Limit results
MATCH (n:Node)
RETURN n
LIMIT 100;
// Returns 100 nodes
```

### 3. Use Parameters

```cypher
// ❌ BAD: Query plan not cached
MATCH (n:Node {id: 123})
RETURN n;

// ✅ GOOD: Query plan cached
MATCH (n:Node {id: $id})
RETURN n;
// Use with parameter: {id: 123}
```

### 4. Profile Queries

```cypher
// Profile query performance
PROFILE
MATCH (n:Node)-[:EDGE]->(m:Node)
WHERE n.quality_score > 0.8
RETURN n, m;

// Shows query plan with:
// - DB hits
// - Rows processed
// - Time taken
```

---

## Security Considerations

### Authentication

**Nostr-based identity**:
- Users authenticated via Nostr public key (npub)
- No passwords stored in database
- Nostr signature verification in application layer
- Session tokens managed in application (not database)

### Authorization

**User data isolation**:
```cypher
// Each user can only access their own settings
MATCH (u:User {npub: $authenticated_npub})-[:HAS_SETTINGS]->(s:UserSettings)
WHERE s.key = $requested_key
RETURN s.value;
// Cannot access other users' settings
```

### SQL Injection Prevention

**Cypher injection protection**:
- Always use parameterized queries
- Never concatenate user input into Cypher strings
- Validate input types before query execution

```rust
// ❌ BAD: Injection risk
let query = format!("MATCH (n:Node {{label: '{}'}}) RETURN n", user_input);

// ✅ GOOD: Parameterized query
let query = "MATCH (n:Node {label: $label}) RETURN n";
let params = vec![("label", user_input)];
```

### Connection Security

**Bolt protocol encryption**:
- TLS enabled for production (bolt+s://)
- Certificate verification enforced
- Credentials never logged
- Connection pool encrypted

---

## Monitoring and Observability

### Database Metrics (Planned)

**Key metrics to track**:
- Query latency (p50, p95, p99)
- Connection pool utilization
- Cache hit rate
- Transaction throughput
- Error rate by query type

**Tools**:
- Neo4j built-in metrics (query.log)
- Prometheus exporter (planned)
- Grafana dashboards (planned)

### Query Logging

```rust
// Enable query logging in development
debug!("Executing Cypher query: {}", query);
debug!("Parameters: {:?}", params);

let start = Instant::now();
let result = graph.execute(query(query).params(params)).await?;
let duration = start.elapsed();

debug!("Query completed in {:?}", duration);
if duration > Duration::from_millis(100) {
    warn!("Slow query detected: {:?}", duration);
}
```

---

### Architecture Docs
- [Server Architecture](../../concepts/architecture/core/server.md) - Overall system design
- [Hexagonal Architecture](hexagonal-cqrs.md) - Ports and adapters pattern
- [Actor System Guide](../../guides/architecture/actor-system.md) - Actor patterns and Neo4j interaction

### Implementation References
- [Settings System](../../guides/user-settings.md) - User settings with Nostr auth
- [User Settings Implementation Summary](../../docs/user-settings-implementation-summary.md) - Settings migration details
- [Neo4j Settings Schema](../../docs/neo4j-user-settings-schema.md) - Schema documentation

### Historical References
- [SQLite to Neo4j Migration](../../guides/sqlite-to-neo4j-migration.md) - Migration history (if exists)
- [GraphServiceActor Migration](../../guides/graphserviceactor-migration.md) - Related actor migration

---

---

## Related Documentation

- [Blender MCP Unified System Architecture](../../architecture/blender-mcp-unified-architecture.md)
- [Hexagonal Architecture Migration Status Report](../../concepts/hexagonal-architecture.md)
- [Server Architecture](../../concepts/architecture/core/server.md)
- [VisionFlow Documentation Modernization - Final Report](../../DOCUMENTATION_MODERNIZATION_COMPLETE.md)
- [VisionFlow GPU CUDA Architecture - Complete Technical Documentation](../../diagrams/infrastructure/gpu/cuda-architecture-complete.md)

## Changelog

**December 2, 2025**
- Initial database architecture documentation
- Added Neo4j schema documentation
- Included performance benchmarks
- Documented backup strategy
- Added query optimization techniques
- Included candid assessments of limitations

**November 5, 2025**
- Neo4j migration complete
- SQLite fully replaced
- GraphServiceActor removed
