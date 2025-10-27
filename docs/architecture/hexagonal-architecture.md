# Hexagonal Architecture Guide

## 📐 Overview

VisionFlow v1.0.0 implements **Hexagonal Architecture** (also known as Ports and Adapters) to achieve clean separation of concerns, testability, and maintainability at enterprise scale.

### What is Hexagonal Architecture?

Hexagonal Architecture is a software design pattern that:
- **Isolates core business logic** from external concerns (databases, UI, APIs)
- **Defines clear boundaries** through ports (interfaces) and adapters (implementations)
- **Enables easy testing** by swapping real adapters with mocks
- **Supports evolution** by decoupling components

### Key Principles

1. **Domain-Centric Design**: Business logic is king, infrastructure serves it
2. **Dependency Inversion**: Dependencies point inward toward the domain
3. **Interface Segregation**: Small, focused ports for specific purposes
4. **Testability**: Mock external systems easily for unit tests

---

## 🏗️ Architecture Layers

VisionFlow's hexagonal architecture consists of four concentric layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    External Systems                          │
│  (Clients, Databases, GPU, Actors, File System, Network)     │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ implements
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Adapters Layer                          │
│  (Concrete Implementations of Ports)                         │
│                                                               │
│  ├─ SqliteKnowledgeGraphRepository                          │
│  ├─ ActorGraphRepository                                    │
│  ├─ CudaPhysicsSimulator                                    │
│  ├─ WhelkOntologyValidator                                  │
│  └─ ... more adapters ...                                   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ uses
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Application Layer                          │
│  (CQRS Commands, Queries, Event Handlers)                   │
│                                                               │
│  ├─ Command Handlers (Directives)                           │
│  │   ├─ SaveGraphCommand                                    │
│  │   ├─ CreateNodeCommand                                   │
│  │   └─ LoadOntologyCommand                                 │
│  │                                                           │
│  ├─ Query Handlers (Queries)                                │
│  │   ├─ GetGraphQuery                                       │
│  │   ├─ SearchNodesQuery                                    │
│  │   └─ GetOntologyQuery                                    │
│  │                                                           │
│  └─ Event Handlers                                          │
│      ├─ GraphEventHandler                                   │
│      ├─ OntologyEventHandler                                │
│      └─ AuditEventHandler                                   │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ uses
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Ports Layer                             │
│  (Abstract Interfaces - Traits)                              │
│                                                               │
│  ├─ Repository Ports (Data Access)                          │
│  │   ├─ KnowledgeGraphRepository                            │
│  │   ├─ OntologyRepository                                  │
│  │   └─ SettingsRepository                                  │
│  │                                                           │
│  └─ Service Ports (Business Operations)                     │
│      ├─ PhysicsSimulator                                    │
│      ├─ SemanticAnalyzer                                    │
│      ├─ OntologyValidator                                   │
│      ├─ NotificationService                                 │
│      └─ AuditLogger                                         │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ uses
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Domain Layer                            │
│  (Pure Business Logic - No Dependencies)                     │
│                                                               │
│  ├─ Domain Models                                           │
│  │   ├─ Node, Edge, Graph                                   │
│  │   ├─ Ontology, Term, Axiom                               │
│  │   └─ Settings, PhysicsConfig                             │
│  │                                                           │
│  ├─ Domain Events                                           │
│  │   ├─ NodeCreated, NodeUpdated                            │
│  │   ├─ EdgeCreated, EdgeUpdated                            │
│  │   └─ OntologyLoaded, ValidationCompleted                 │
│  │                                                           │
│  └─ Domain Services                                         │
│      └─ (Pure business logic operations)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📦 Layer Details

### 1. Domain Layer (Core)

**Location**: `src/models/`, `src/events/domain_events.rs`

**Characteristics**:
- **Zero external dependencies** (only std library)
- **Pure business logic** with no infrastructure concerns
- **Domain-driven design** (DDD) entities and value objects
- **Framework-agnostic** (can be moved to any framework)

**Example - Domain Model**:
```rust
// src/models/node.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub label: String,
    pub position: Vector3,
    pub metadata: NodeMetadata,
}

impl Node {
    /// Pure domain logic - no database, no I/O
    pub fn calculate_distance(&self, other: &Node) -> f32 {
        self.position.distance(other.position)
    }

    pub fn is_connected_to(&self, edges: &[Edge]) -> bool {
        edges.iter().any(|e| e.source == self.id || e.target == self.id)
    }
}
```

**Example - Domain Event**:
```rust
// src/events/domain_events.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainEvent {
    NodeCreated {
        node_id: NodeId,
        graph_id: GraphId,
        timestamp: DateTime<Utc>,
    },
    EdgeCreated {
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
        timestamp: DateTime<Utc>,
    },
}
```

### 2. Ports Layer (Interfaces)

**Location**: `src/ports/`

**Characteristics**:
- **Abstract trait definitions** (no implementation)
- **Dependency contracts** between layers
- **Technology-agnostic** interfaces
- **Testable** through mocking

**Repository Ports**:
```rust
// src/ports/knowledge_graph_repository.rs
#[async_trait]
pub trait KnowledgeGraphRepository: Send + Sync {
    /// Save entire graph to storage
    async fn save_graph(&self, graph: &Graph) -> Result<(), RepositoryError>;

    /// Retrieve graph by ID
    async fn get_graph(&self, graph_id: &GraphId) -> Result<Graph, RepositoryError>;

    /// Get node by ID
    async fn get_node(&self, node_id: &NodeId) -> Result<Node, RepositoryError>;

    /// Search nodes by label
    async fn search_nodes(&self, query: &str) -> Result<Vec<Node>, RepositoryError>;

    /// Create new node
    async fn create_node(&self, node: &Node) -> Result<NodeId, RepositoryError>;

    /// Update existing node
    async fn update_node(&self, node: &Node) -> Result<(), RepositoryError>;

    /// Delete node
    async fn delete_node(&self, node_id: &NodeId) -> Result<(), RepositoryError>;

    // Edge operations...
    async fn create_edge(&self, edge: &Edge) -> Result<EdgeId, RepositoryError>;
    async fn get_edges_for_node(&self, node_id: &NodeId) -> Result<Vec<Edge>, RepositoryError>;
}
```

**Service Ports**:
```rust
// src/ports/physics_simulator.rs
#[async_trait]
pub trait PhysicsSimulator: Send + Sync {
    /// Run physics simulation step
    async fn simulate_step(
        &self,
        nodes: &mut [Node],
        edges: &[Edge],
        delta_time: f32,
    ) -> Result<(), PhysicsError>;

    /// Calculate forces for a single node
    fn calculate_forces(&self, node: &Node, neighbors: &[Node]) -> Vector3;

    /// Update node positions based on forces
    fn apply_forces(&self, nodes: &mut [Node], delta_time: f32);
}
```

### 3. Application Layer (Use Cases)

**Location**: `src/application/`

**Characteristics**:
- **CQRS pattern** (Commands for writes, Queries for reads)
- **Use case orchestration** (coordinating multiple ports)
- **Event publishing** (domain events)
- **Transaction management** (when needed)

**Command Handler (Write)**:
```rust
// src/application/commands/save_graph_command.rs
pub struct SaveGraphCommand {
    pub graph_id: GraphId,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

pub struct SaveGraphCommandHandler {
    graph_repo: Arc<dyn KnowledgeGraphRepository>,
    event_bus: Arc<EventBus>,
}

impl SaveGraphCommandHandler {
    pub async fn handle(&self, cmd: SaveGraphCommand) -> Result<(), ApplicationError> {
        // 1. Validate command
        self.validate(&cmd)?;

        // 2. Create domain model
        let graph = Graph {
            id: cmd.graph_id,
            nodes: cmd.nodes,
            edges: cmd.edges,
        };

        // 3. Save via repository port
        self.graph_repo.save_graph(&graph).await?;

        // 4. Publish domain event
        self.event_bus.publish(DomainEvent::GraphSaved {
            graph_id: graph.id,
            timestamp: Utc::now(),
        }).await?;

        Ok(())
    }
}
```

**Query Handler (Read)**:
```rust
// src/application/queries/get_graph_query.rs
pub struct GetGraphQuery {
    pub graph_id: GraphId,
}

pub struct GetGraphQueryHandler {
    graph_repo: Arc<dyn KnowledgeGraphRepository>,
}

impl GetGraphQueryHandler {
    pub async fn handle(&self, query: GetGraphQuery) -> Result<GraphDTO, ApplicationError> {
        // Query doesn't modify state - pure read
        let graph = self.graph_repo.get_graph(&query.graph_id).await?;

        // Transform to DTO (Data Transfer Object)
        Ok(GraphDTO::from(graph))
    }
}
```

### 4. Adapters Layer (Implementations)

**Location**: `src/adapters/`

**Characteristics**:
- **Concrete implementations** of port traits
- **Technology-specific code** (SQLite, CUDA, Actix, etc.)
- **Interchangeable** (swap SQLite for PostgreSQL)
- **Testable** (integration tests per adapter)

**SQLite Repository Adapter**:
```rust
// src/adapters/sqlite_knowledge_graph_repository.rs
pub struct SqliteKnowledgeGraphRepository {
    pool: Pool<SqliteConnectionManager>,
}

#[async_trait]
impl KnowledgeGraphRepository for SqliteKnowledgeGraphRepository {
    async fn save_graph(&self, graph: &Graph) -> Result<(), RepositoryError> {
        let conn = self.pool.get()
            .map_err(|e| RepositoryError::ConnectionError(e.to_string()))?;

        // Start transaction
        conn.execute("BEGIN TRANSACTION", [])?;

        // Insert nodes
        for node in &graph.nodes {
            conn.execute(
                "INSERT OR REPLACE INTO kg_nodes (id, label, x, y, z, metadata_json)
                 VALUES (?, ?, ?, ?, ?, ?)",
                params![
                    node.id.to_string(),
                    node.label,
                    node.position.x,
                    node.position.y,
                    node.position.z,
                    serde_json::to_string(&node.metadata)?,
                ],
            )?;
        }

        // Insert edges
        for edge in &graph.edges {
            conn.execute(
                "INSERT OR REPLACE INTO kg_edges (id, source, target, weight)
                 VALUES (?, ?, ?, ?)",
                params![
                    edge.id.to_string(),
                    edge.source.to_string(),
                    edge.target.to_string(),
                    edge.weight,
                ],
            )?;
        }

        // Commit transaction
        conn.execute("COMMIT", [])?;

        Ok(())
    }

    async fn get_graph(&self, graph_id: &GraphId) -> Result<Graph, RepositoryError> {
        let conn = self.pool.get()?;

        // Query nodes
        let mut stmt = conn.prepare("SELECT id, label, x, y, z, metadata_json FROM kg_nodes WHERE graph_id = ?")?;
        let nodes = stmt.query_map([graph_id.to_string()], |row| {
            Ok(Node {
                id: NodeId::from_str(row.get(0)?)?,
                label: row.get(1)?,
                position: Vector3::new(row.get(2)?, row.get(3)?, row.get(4)?),
                metadata: serde_json::from_str(row.get(5)?)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        // Query edges
        let mut stmt = conn.prepare("SELECT id, source, target, weight FROM kg_edges WHERE graph_id = ?")?;
        let edges = stmt.query_map([graph_id.to_string()], |row| {
            Ok(Edge {
                id: EdgeId::from_str(row.get(0)?)?,
                source: NodeId::from_str(row.get(1)?)?,
                target: NodeId::from_str(row.get(2)?)?,
                weight: row.get(3)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        Ok(Graph {
            id: *graph_id,
            nodes,
            edges,
        })
    }

    // ... other methods ...
}
```

**Actor Adapter**:
```rust
// src/adapters/actor_graph_repository.rs
pub struct ActorGraphRepository {
    graph_actor: Addr<GraphServiceActor>,
}

#[async_trait]
impl KnowledgeGraphRepository for ActorGraphRepository {
    async fn save_graph(&self, graph: &Graph) -> Result<(), RepositoryError> {
        // Wrap repository call in actor message
        let msg = SaveGraphMessage {
            graph: graph.clone(),
        };

        self.graph_actor
            .send(msg)
            .await
            .map_err(|e| RepositoryError::ActorError(e.to_string()))?
            .map_err(|e| RepositoryError::SaveError(e.to_string()))
    }

    // ... other methods use actor messages ...
}
```

---

## 🔄 Dependency Flow

### The Dependency Rule

**All dependencies point INWARD** toward the domain:

```
External Systems  →  Adapters  →  Application  →  Ports  →  Domain
   (SQLite)          (Impl)      (Use Cases)    (Traits)  (Logic)
```

### Why This Matters

1. **Domain is Stable**: Core business logic never changes due to database choice
2. **Easy Testing**: Swap real adapters with mocks
3. **Technology Independence**: Change SQLite to PostgreSQL without touching domain
4. **Clear Responsibilities**: Each layer has a single, well-defined purpose

---

## 🧪 Testing Strategy

### Unit Tests (Domain Layer)

**Test pure business logic in isolation:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_calculates_distance_correctly() {
        let node1 = Node {
            id: NodeId::new(),
            position: Vector3::new(0.0, 0.0, 0.0),
            ..Default::default()
        };

        let node2 = Node {
            id: NodeId::new(),
            position: Vector3::new(3.0, 4.0, 0.0),
            ..Default::default()
        };

        assert_eq!(node1.calculate_distance(&node2), 5.0);
    }
}
```

### Integration Tests (Adapters)

**Test adapters against real systems:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn sqlite_repository_saves_and_retrieves_graph() {
        // Arrange
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let repo = SqliteKnowledgeGraphRepository::new(&db_path).unwrap();

        let graph = Graph {
            id: GraphId::new(),
            nodes: vec![/* test nodes */],
            edges: vec![/* test edges */],
        };

        // Act
        repo.save_graph(&graph).await.unwrap();
        let retrieved = repo.get_graph(&graph.id).await.unwrap();

        // Assert
        assert_eq!(retrieved.nodes.len(), graph.nodes.len());
        assert_eq!(retrieved.edges.len(), graph.edges.len());
    }
}
```

### Unit Tests with Mocks (Application Layer)

**Test use cases with mock ports:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mockall::*;

    // Create mock repository
    mock! {
        GraphRepository {}

        #[async_trait]
        impl KnowledgeGraphRepository for GraphRepository {
            async fn save_graph(&self, graph: &Graph) -> Result<(), RepositoryError>;
            async fn get_graph(&self, id: &GraphId) -> Result<Graph, RepositoryError>;
        }
    }

    #[tokio::test]
    async fn command_handler_saves_graph_via_repository() {
        // Arrange
        let mut mock_repo = MockGraphRepository::new();
        mock_repo
            .expect_save_graph()
            .times(1)
            .returning(|_| Ok(()));

        let handler = SaveGraphCommandHandler {
            graph_repo: Arc::new(mock_repo),
            event_bus: Arc::new(EventBus::new()),
        };

        let cmd = SaveGraphCommand {
            graph_id: GraphId::new(),
            nodes: vec![],
            edges: vec![],
        };

        // Act
        let result = handler.handle(cmd).await;

        // Assert
        assert!(result.is_ok());
    }
}
```

---

## 🔌 Benefits of Hexagonal Architecture

### 1. Testability
- **Unit test** domain logic without databases
- **Mock** external dependencies easily
- **Fast tests** (no I/O overhead)

### 2. Flexibility
- **Swap** SQLite for PostgreSQL (change adapter)
- **Add** Redis caching (new adapter)
- **Migrate** to different framework (ports stay same)

### 3. Maintainability
- **Clear boundaries** between layers
- **Single responsibility** per component
- **Easy to locate** code (by layer)

### 4. Evolvability
- **Add features** without modifying core
- **Refactor** adapters independently
- **Upgrade** dependencies safely

### 5. Team Collaboration
- **Frontend/Backend** teams work independently
- **Domain experts** work on pure logic
- **Infrastructure team** works on adapters

---

## 📚 Further Reading

### Books
- **"Clean Architecture"** by Robert C. Martin
- **"Implementing Domain-Driven Design"** by Vaughn Vernon
- **"Patterns of Enterprise Application Architecture"** by Martin Fowler

### Articles
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [CQRS Pattern](https://martinfowler.com/bliki/CQRS.html)
- [Dependency Inversion Principle](https://en.wikipedia.org/wiki/Dependency_inversion_principle)

### VisionFlow Documentation
- [Ports Design](./ports-and-adapters.md)
- [CQRS Implementation](./cqrs-pattern.md)
- [Event-Driven Architecture](./event-driven.md)
- [Testing Strategy](./testing-strategy.md)

---

**VisionFlow Hexagonal Architecture Guide**
Version 1.0.0 | Last Updated: 2025-10-27
