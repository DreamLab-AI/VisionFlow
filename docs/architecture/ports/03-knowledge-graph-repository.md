# KnowledgeGraphRepository Port

## Purpose

The **KnowledgeGraphRepository** port manages the main knowledge graph structure parsed from local markdown files (Logseq, Obsidian). It provides comprehensive graph data access, manipulation, and query capabilities.

## Location

- **Trait Definition**: `src/ports/knowledge_graph_repository.rs`
- **Adapter Implementation**: `src/adapters/sqlite_knowledge_graph_repository.rs`

## Interface

```rust
#[async_trait]
pub trait KnowledgeGraphRepository: Send + Sync {
    // Graph-level operations
    async fn load_graph(&self) -> Result<Arc<GraphData>>;
    async fn save_graph(&self, graph: &GraphData) -> Result<()>;
    async fn clear_graph(&self) -> Result<()>;
    async fn get_statistics(&self) -> Result<GraphStatistics>;

    // Node operations
    async fn add_node(&self, node: &Node) -> Result<u32>;
    async fn batch_add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;
    async fn update_node(&self, node: &Node) -> Result<()>;
    async fn batch_update_nodes(&self, nodes: Vec<Node>) -> Result<()>;
    async fn remove_node(&self, node_id: u32) -> Result<()>;
    async fn batch_remove_nodes(&self, node_ids: Vec<u32>) -> Result<()>;

    // Node queries
    async fn get_node(&self, node_id: u32) -> Result<Option<Node>>;
    async fn get_nodes(&self, node_ids: Vec<u32>) -> Result<Vec<Node>>;
    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> Result<Vec<Node>>;
    async fn search_nodes_by_label(&self, label: &str) -> Result<Vec<Node>>;
    async fn query_nodes(&self, query: &str) -> Result<Vec<Node>>;

    // Edge operations
    async fn add_edge(&self, edge: &Edge) -> Result<String>;
    async fn batch_add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;
    async fn update_edge(&self, edge: &Edge) -> Result<()>;
    async fn remove_edge(&self, edge_id: &str) -> Result<()>;
    async fn batch_remove_edges(&self, edge_ids: Vec<String>) -> Result<()>;

    // Edge queries
    async fn get_node_edges(&self, node_id: u32) -> Result<Vec<Edge>>;
    async fn get_edges_between(&self, source_id: u32, target_id: u32) -> Result<Vec<Edge>>;

    // Graph algorithms
    async fn get_neighbors(&self, node_id: u32) -> Result<Vec<Node>>;
    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> Result<()>;

    // Transaction support
    async fn begin_transaction(&self) -> Result<()>;
    async fn commit_transaction(&self) -> Result<()>;
    async fn rollback_transaction(&self) -> Result<()>;

    // Health check
    async fn health_check(&self) -> Result<bool>;
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
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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
    pub metadata_id: String,  // Markdown file ID
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
    pub edge_type: EdgeType,
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
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub connected_components: usize,
    pub last_updated: DateTime<Utc>,
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
let graph = repo.load_graph().await?;
println!("Loaded {} nodes and {} edges", graph.nodes.len(), graph.edges.len());

// Save modified graph
repo.save_graph(&modified_graph).await?;

// Get graph statistics
let stats = repo.get_statistics().await?;
println!("Average degree: {}", stats.average_degree);
println!("Components: {}", stats.connected_components);
```

### Node Operations

```rust
// Add a single node
let node = Node {
    id: 0, // Will be assigned by repository
    label: "My Note".to_string(),
    position: Vec3::new(0.0, 0.0, 0.0),
    velocity: Vec3::ZERO,
    color: "#3498db".to_string(),
    size: 1.0,
    metadata_id: "note-123".to_string(),
    properties: HashMap::new(),
};

let node_id = repo.add_node(&node).await?;
println!("Created node with ID: {}", node_id);

// Batch add nodes (more efficient)
let nodes = vec![node1, node2, node3];
let node_ids = repo.batch_add_nodes(nodes).await?;
println!("Created {} nodes", node_ids.len());

// Update a node
let mut node = repo.get_node(node_id).await?.unwrap();
node.label = "Updated Label".to_string();
repo.update_node(&node).await?;

// Remove nodes
repo.remove_node(node_id).await?;
repo.batch_remove_nodes(vec![1, 2, 3]).await?;
```

### Node Queries

```rust
// Get a single node
if let Some(node) = repo.get_node(42).await? {
    println!("Node: {}", node.label);
}

// Get multiple nodes by IDs
let nodes = repo.get_nodes(vec![1, 2, 3, 4]).await?;

// Search by metadata ID (markdown file)
let nodes = repo.get_nodes_by_metadata_id("daily/2025-10-27").await?;

// Search by label (partial matching)
let nodes = repo.search_nodes_by_label("rust").await?;

// Advanced query (SQL-like syntax)
let nodes = repo.query_nodes("color = '#3498db' AND size > 1.0").await?;

// Get neighbors
let neighbors = repo.get_neighbors(node_id).await?;
println!("Node has {} neighbors", neighbors.len());
```

### Edge Operations

```rust
// Add an edge
let edge = Edge {
    id: String::new(), // Will be assigned
    source: 1,
    target: 2,
    edge_type: EdgeType::Link,
    weight: 1.0,
    properties: HashMap::new(),
};

let edge_id = repo.add_edge(&edge).await?;

// Batch add edges
let edges = vec![edge1, edge2, edge3];
let edge_ids = repo.batch_add_edges(edges).await?;

// Get all edges for a node
let edges = repo.get_node_edges(node_id).await?;

// Get edges between two specific nodes
let edges = repo.get_edges_between(source_id, target_id).await?;

// Remove edges
repo.remove_edge(&edge_id).await?;
repo.batch_remove_edges(vec!["edge1".to_string(), "edge2".to_string()]).await?;
```

### Physics Simulation Updates

```rust
// Update node positions after physics step
let positions = vec![
    (1, 10.5, 20.3, 0.0),
    (2, 15.2, 18.7, 0.0),
    (3, 12.1, 22.4, 0.0),
];

repo.batch_update_positions(positions).await?;
```

### Transaction Support

```rust
// Begin transaction
repo.begin_transaction().await?;

// Perform multiple operations
let node_ids = repo.batch_add_nodes(nodes).await?;
let edge_ids = repo.batch_add_edges(edges).await?;

// Commit or rollback
if all_ok {
    repo.commit_transaction().await?;
} else {
    repo.rollback_transaction().await?;
}
```

## Implementation Notes

### Batch Operations Optimization

Always prefer batch operations for bulk updates:

```rust
// ❌ Slow: Individual operations
for node in nodes {
    repo.add_node(&node).await?;
}

// ✅ Fast: Batch operation
repo.batch_add_nodes(nodes).await?;
```

**Performance Gains**:
- Batch add: 10-50x faster than individual adds
- Single transaction vs multiple transactions
- Reduced lock contention

### Indexing Strategy

Database indexes for optimal query performance:

```sql
CREATE INDEX idx_nodes_metadata_id ON nodes(metadata_id);
CREATE INDEX idx_nodes_label ON nodes(label);
CREATE INDEX idx_edges_source ON edges(source);
CREATE INDEX idx_edges_target ON edges(target);
CREATE INDEX idx_edges_type ON edges(edge_type);
```

### Caching Strategy

Implement multi-level caching:

```rust
pub struct CachedGraphRepository {
    repo: Arc<dyn KnowledgeGraphRepository>,
    node_cache: Arc<RwLock<LruCache<u32, Node>>>,
    edge_cache: Arc<RwLock<LruCache<String, Edge>>>,
    graph_cache: Arc<RwLock<Option<(Arc<GraphData>, Instant)>>>,
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
    metadata_id TEXT NOT NULL,
    properties TEXT, -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    source INTEGER NOT NULL,
    target INTEGER NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    properties TEXT, -- JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
);
```

## Testing

### Mock Implementation

```rust
pub struct MockKnowledgeGraphRepository {
    graph: Arc<RwLock<GraphData>>,
    next_node_id: Arc<AtomicU32>,
}

#[async_trait]
impl KnowledgeGraphRepository for MockKnowledgeGraphRepository {
    async fn load_graph(&self) -> Result<Arc<GraphData>> {
        Ok(Arc::new(self.graph.read().await.clone()))
    }

    async fn add_node(&self, node: &Node) -> Result<u32> {
        let id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
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
async fn test_knowledge_graph_repository_contract() {
    let repo = MockKnowledgeGraphRepository::new();

    // Test node operations
    let node = create_test_node();
    let node_id = repo.add_node(&node).await.unwrap();
    assert!(node_id > 0);

    let loaded_node = repo.get_node(node_id).await.unwrap().unwrap();
    assert_eq!(loaded_node.label, node.label);

    // Test batch operations
    let nodes = vec![create_test_node(), create_test_node()];
    let ids = repo.batch_add_nodes(nodes).await.unwrap();
    assert_eq!(ids.len(), 2);

    // Test queries
    let results = repo.search_nodes_by_label("test").await.unwrap();
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

## References

- **Graph Databases**: https://neo4j.com/developer/graph-database/
- **Repository Pattern**: https://martinfowler.com/eaaCatalog/repository.html
- **SQLite Optimization**: https://www.sqlite.org/optoverview.html

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
