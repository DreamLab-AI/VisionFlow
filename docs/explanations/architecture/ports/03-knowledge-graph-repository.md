---
title: KnowledgeGraphRepository Port
description: The **KnowledgeGraphRepository** port manages the main knowledge graph structure parsed from local markdown files (Logseq, Obsidian). It provides comprehensive graph data access, manipulation, and ...
category: explanation
tags:
  - architecture
  - database
  - backend
updated-date: 2025-12-18
difficulty-level: advanced
---


# KnowledgeGraphRepository Port

## Purpose

The **KnowledgeGraphRepository** port manages the main knowledge graph structure parsed from local markdown files (Logseq, Obsidian). It provides comprehensive graph data access, manipulation, and query capabilities.

## Location

- **Trait Definition**: `src/ports/knowledge-graph-repository.rs`
- **Adapter Implementation**: `src/adapters/sqlite-knowledge-graph-repository.rs`

## Interface

```rust
#[async-trait]
pub trait KnowledgeGraphRepository: Send + Sync {
    // Graph-level operations
    async fn load-graph(&self) -> Result<Arc<GraphData>>;
    async fn save-graph(&self, graph: &GraphData) -> Result<()>;
    async fn clear-graph(&self) -> Result<()>;
    async fn get-statistics(&self) -> Result<GraphStatistics>;

    // Node operations
    async fn add-node(&self, node: &Node) -> Result<u32>;
    async fn batch-add-nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;
    async fn update-node(&self, node: &Node) -> Result<()>;
    async fn batch-update-nodes(&self, nodes: Vec<Node>) -> Result<()>;
    async fn remove-node(&self, node-id: u32) -> Result<()>;
    async fn batch-remove-nodes(&self, node-ids: Vec<u32>) -> Result<()>;

    // Node queries
    async fn get-node(&self, node-id: u32) -> Result<Option<Node>>;
    async fn get-nodes(&self, node-ids: Vec<u32>) -> Result<Vec<Node>>;
    async fn get-nodes-by-metadata-id(&self, metadata-id: &str) -> Result<Vec<Node>>;
    async fn search-nodes-by-label(&self, label: &str) -> Result<Vec<Node>>;
    async fn query-nodes(&self, query: &str) -> Result<Vec<Node>>;

    // Edge operations
    async fn add-edge(&self, edge: &Edge) -> Result<String>;
    async fn batch-add-edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;
    async fn update-edge(&self, edge: &Edge) -> Result<()>;
    async fn remove-edge(&self, edge-id: &str) -> Result<()>;
    async fn batch-remove-edges(&self, edge-ids: Vec<String>) -> Result<()>;

    // Edge queries
    async fn get-node-edges(&self, node-id: u32) -> Result<Vec<Edge>>;
    async fn get-edges-between(&self, source-id: u32, target-id: u32) -> Result<Vec<Edge>>;

    // Graph algorithms
    async fn get-neighbors(&self, node-id: u32) -> Result<Vec<Node>>;
    async fn batch-update-positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> Result<()>;

    // Transaction support
    async fn begin-transaction(&self) -> Result<()>;
    async fn commit-transaction(&self) -> Result<()>;
    async fn rollback-transaction(&self) -> Result<()>;

    // Health check
    async fn health-check(&self) -> Result<bool>;
}
```

## Types

### GraphData

Core graph structure:

```rust
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: GraphMetadata,
}

pub struct GraphMetadata {
    pub name: String,
    pub source: GraphSource,
    pub created-at: DateTime<Utc>,
    pub updated-at: DateTime<Utc>,
}

pub enum GraphSource {
    Logseq { directory: PathBuf },
    Obsidian { vault: PathBuf },
    GitHub { repo: String, branch: String },
}
```

### Node

Graph node representation:

```rust
pub struct Node {
    pub id: u32,
    pub label: String,
    pub position: Vec3,       // (x, y, z)
    pub velocity: Vec3,       // Physics simulation
    pub color: String,        // Hex color
    pub size: f32,
    pub metadata-id: String,  // Markdown file ID
    pub properties: HashMap<String, String>,
}
```

### Edge

Graph edge representation:

```rust
pub struct Edge {
    pub id: String,
    pub source: u32,
    pub target: u32,
    pub edge-type: EdgeType,
    pub weight: f32,
    pub properties: HashMap<String, String>,
}

pub enum EdgeType {
    Link,            // Standard link
    Backlink,        // Reverse link
    Hierarchical,    // Parent-child
    Semantic,        // Inferred relationship
    Custom(String),
}
```

### GraphStatistics

Graph analytics:

```rust
pub struct GraphStatistics {
    pub node-count: usize,
    pub edge-count: usize,
    pub average-degree: f32,
    pub connected-components: usize,
    pub last-updated: DateTime<Utc>,
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum KnowledgeGraphRepositoryError {
    #[error("Graph not found")]
    NotFound,

    #[error("Node not found: {0}")]
    NodeNotFound(u32),

    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Concurrent modification detected")]
    ConcurrentModification,
}
```

## Usage Examples

### Loading and Saving Graphs

```rust
let repo: Arc<dyn KnowledgeGraphRepository> = Arc::new(SqliteGraphRepository::new(pool));

// Load complete graph
let graph = repo.load-graph().await?;
println!("Loaded {} nodes and {} edges", graph.nodes.len(), graph.edges.len());

// Save modified graph
repo.save-graph(&modified-graph).await?;

// Get graph statistics
let stats = repo.get-statistics().await?;
println!("Average degree: {}", stats.average-degree);
println!("Components: {}", stats.connected-components);
```

### Node Operations

```rust
// Add a single node
let node = Node {
    id: 0, // Will be assigned by repository
    label: "My Note".to-string(),
    position: Vec3::new(0.0, 0.0, 0.0),
    velocity: Vec3::ZERO,
    color: "#3498db".to-string(),
    size: 1.0,
    metadata-id: "note-123".to-string(),
    properties: HashMap::new(),
};

let node-id = repo.add-node(&node).await?;
println!("Created node with ID: {}", node-id);

// Batch add nodes (more efficient)
let nodes = vec![node1, node2, node3];
let node-ids = repo.batch-add-nodes(nodes).await?;
println!("Created {} nodes", node-ids.len());

// Update a node
let mut node = repo.get-node(node-id).await?.unwrap();
node.label = "Updated Label".to-string();
repo.update-node(&node).await?;

// Remove nodes
repo.remove-node(node-id).await?;
repo.batch-remove-nodes(vec![1, 2, 3]).await?;
```

### Node Queries

```rust
// Get a single node
if let Some(node) = repo.get-node(42).await? {
    println!("Node: {}", node.label);
}

// Get multiple nodes by IDs
let nodes = repo.get-nodes(vec![1, 2, 3, 4]).await?;

// Search by metadata ID (markdown file)
let nodes = repo.get-nodes-by-metadata-id("daily/2025-10-27").await?;

// Search by label (partial matching)
let nodes = repo.search-nodes-by-label("rust").await?;

// Advanced query (SQL-like syntax)
let nodes = repo.query-nodes("color = '#3498db' AND size > 1.0").await?;

// Get neighbors
let neighbors = repo.get-neighbors(node-id).await?;
println!("Node has {} neighbors", neighbors.len());
```

### Edge Operations

```rust
// Add an edge
let edge = Edge {
    id: String::new(), // Will be assigned
    source: 1,
    target: 2,
    edge-type: EdgeType::Link,
    weight: 1.0,
    properties: HashMap::new(),
};

let edge-id = repo.add-edge(&edge).await?;

// Batch add edges
let edges = vec![edge1, edge2, edge3];
let edge-ids = repo.batch-add-edges(edges).await?;

// Get all edges for a node
let edges = repo.get-node-edges(node-id).await?;

// Get edges between two specific nodes
let edges = repo.get-edges-between(source-id, target-id).await?;

// Remove edges
repo.remove-edge(&edge-id).await?;
repo.batch-remove-edges(vec!["edge1".to-string(), "edge2".to-string()]).await?;
```

### Physics Simulation Updates

```rust
// Update node positions after physics step
let positions = vec![
    (1, 10.5, 20.3, 0.0),
    (2, 15.2, 18.7, 0.0),
    (3, 12.1, 22.4, 0.0),
];

repo.batch-update-positions(positions).await?;
```

### Transaction Support

```rust
// Begin transaction
repo.begin-transaction().await?;

// Perform multiple operations
let node-ids = repo.batch-add-nodes(nodes).await?;
let edge-ids = repo.batch-add-edges(edges).await?;

// Commit or rollback
if all-ok {
    repo.commit-transaction().await?;
} else {
    repo.rollback-transaction().await?;
}
```

## Implementation Notes

### Batch Operations Optimization

Always prefer batch operations for bulk updates:

```rust
// ❌ Slow: Individual operations
for node in nodes {
    repo.add-node(&node).await?;
}

// ✅ Fast: Batch operation
repo.batch-add-nodes(nodes).await?;
```

**Performance Gains**:
- Batch add: 10-50x faster than individual adds
- Single transaction vs multiple transactions
- Reduced lock contention

### Indexing Strategy

Database indexes for optimal query performance:

```sql
CREATE INDEX idx-nodes-metadata-id ON nodes(metadata-id);
CREATE INDEX idx-nodes-label ON nodes(label);
CREATE INDEX idx-edges-source ON edges(source);
CREATE INDEX idx-edges-target ON edges(target);
CREATE INDEX idx-edges-type ON edges(edge-type);
```

### Caching Strategy

Implement multi-level caching:

```rust
pub struct CachedGraphRepository {
    repo: Arc<dyn KnowledgeGraphRepository>,
    node-cache: Arc<RwLock<LruCache<u32, Node>>>,
    edge-cache: Arc<RwLock<LruCache<String, Edge>>>,
    graph-cache: Arc<RwLock<Option<(Arc<GraphData>, Instant)>>>,
}
```

**Cache Invalidation**:
- Invalidate on write operations
- TTL-based expiration
- LRU eviction policy

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    z REAL NOT NULL,
    vx REAL DEFAULT 0.0,
    vy REAL DEFAULT 0.0,
    vz REAL DEFAULT 0.0,
    color TEXT NOT NULL,
    size REAL NOT NULL,
    metadata-id TEXT NOT NULL,
    properties TEXT, -- JSON
    created-at DATETIME DEFAULT CURRENT-TIMESTAMP,
    updated-at DATETIME DEFAULT CURRENT-TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source INTEGER NOT NULL,
    target INTEGER NOT NULL,
    edge-type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    properties TEXT, -- JSON
    created-at DATETIME DEFAULT CURRENT-TIMESTAMP,
    FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
);
```

## Testing

### Mock Implementation

```rust
pub struct MockKnowledgeGraphRepository {
    graph: Arc<RwLock<GraphData>>,
    next-node-id: Arc<AtomicU32>,
}

#[async-trait]
impl KnowledgeGraphRepository for MockKnowledgeGraphRepository {
    async fn load-graph(&self) -> Result<Arc<GraphData>> {
        Ok(Arc::new(self.graph.read().await.clone()))
    }

    async fn add-node(&self, node: &Node) -> Result<u32> {
        let id = self.next-node-id.fetch-add(1, Ordering::SeqCst);
        let mut node = node.clone();
        node.id = id;
        self.graph.write().await.nodes.push(node);
        Ok(id)
    }

    // ... implement remaining methods
}
```

### Contract Tests

```rust
#[tokio::test]
async fn test-knowledge-graph-repository-contract() {
    let repo = MockKnowledgeGraphRepository::new();

    // Test node operations
    let node = create-test-node();
    let node-id = repo.add-node(&node).await.unwrap();
    assert!(node-id > 0);

    let loaded-node = repo.get-node(node-id).await.unwrap().unwrap();
    assert-eq!(loaded-node.label, node.label);

    // Test batch operations
    let nodes = vec![create-test-node(), create-test-node()];
    let ids = repo.batch-add-nodes(nodes).await.unwrap();
    assert-eq!(ids.len(), 2);

    // Test queries
    let results = repo.search-nodes-by-label("test").await.unwrap();
    assert!(results.len() > 0);
}
```

## Performance Considerations

### Benchmarks

Target performance (SQLite adapter):
- Load graph (1000 nodes): < 100ms
- Add single node: < 5ms
- Batch add (100 nodes): < 50ms
- Node query by ID: < 1ms
- Search by label: < 20ms
- Batch position update (1000 nodes): < 100ms

### Optimization Tips

1. **Use batch operations** for bulk inserts/updates
2. **Implement connection pooling** with r2d2
3. **Cache frequently accessed data**
4. **Use prepared statements**
5. **Optimize indexes** based on query patterns

---

---

## Related Documentation

- [Hexagonal Architecture Ports - Overview](01-overview.md)
- [Ontology Storage Architecture](../ontology-storage-architecture.md)
- [Architecture Overview (OBSOLETE - WRONG STACK)](../../../archive/deprecated-patterns/03-architecture-WRONG-STACK.md)
- [OntologyRepository Port](04-ontology-repository.md)
- [InferenceEngine Port](05-inference-engine.md)

## References

- **Graph Databases**: https://neo4j.com/developer/graph-database/
- **Repository Pattern**: https://martinfowler.com/eaaCatalog/repository.html
- **SQLite Optimization**: https://www.sqlite.org/optoverview.html

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
