---
title: Hexagonal Architecture
description: Understanding VisionFlow's ports-and-adapters architecture with CQRS for clean separation of concerns
category: explanation
tags:
  - architecture
  - hexagonal
  - cqrs
  - design-patterns
related-docs:
  - concepts/actor-model.md
  - architecture/patterns/hexagonal-cqrs.md
  - architecture/ports/01-overview.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# Hexagonal Architecture

VisionFlow employs hexagonal (ports-and-adapters) architecture with CQRS to achieve clean separation between domain logic, infrastructure, and presentation.

---

## Core Concept

Hexagonal architecture inverts dependencies: the domain core knows nothing about databases, APIs, or UI. Instead:

- **Ports** define abstract interfaces the domain needs
- **Adapters** implement those interfaces for specific technologies
- **Application layer** orchestrates domain operations via CQRS handlers

This enables:
- Swapping databases without changing domain logic
- Testing domain in isolation
- Multiple presentation layers (REST, GraphQL, WebSocket)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                 Hexagonal Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Infrastructure (Adapters)                 │  │
│  │                                                        │  │
│  │  Neo4j Adapters    Actix Adapters    GPU Adapters     │  │
│  │  (Database)        (Actors)          (CUDA)           │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓ implements                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Domain (Ports)                       │  │
│  │                                                        │  │
│  │  GraphRepository   PhysicsSimulator   InferenceEngine │  │
│  │  OntologyRepository SettingsRepository SemanticAnalyzer│  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↑ uses                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Application Layer (CQRS)                  │  │
│  │                                                        │  │
│  │  CommandBus    QueryBus    114 Handlers               │  │
│  │  (Directives)  (Queries)   (Domain Operations)        │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↑ invokes                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Presentation (Actors)                     │  │
│  │                                                        │  │
│  │  GraphStateActor   PhysicsOrchestrator   ClientCoord  │  │
│  │  (State Machine)   (GPU Coordination)    (WebSocket)  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Ports (Domain Interfaces)

Ports define **what** the domain needs, not how it's implemented.

### Core Ports

| Port | Purpose | Methods |
|------|---------|---------|
| `GraphRepository` | Graph data access | `get_node`, `add_node`, `query_edges` |
| `OntologyRepository` | OWL storage | `load_ontology`, `save_axiom`, `query_classes` |
| `SettingsRepository` | Configuration | `get_setting`, `update_setting` |
| `PhysicsSimulator` | Physics abstraction | `compute_forces`, `integrate` |
| `InferenceEngine` | OWL reasoning | `infer_axioms`, `get_hierarchy` |
| `SemanticAnalyzer` | AI features | `extract_embeddings`, `compute_similarity` |
| `GPUPhysicsAdapter` | GPU integration | `upload_graph`, `execute_kernel` |
| `GPUSemanticAnalyzer` | GPU ML | `batch_embed`, `cluster` |
| `KnowledgeGraphRepository` | KG operations | `create_entity`, `link_entities` |

### Port Definition Example

```rust
// src/ports/graph_repository.rs
#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_node(&self, id: u32) -> Result<Node, RepositoryError>;
    async fn add_node(&self, node: Node) -> Result<u32, RepositoryError>;
    async fn remove_node(&self, id: u32) -> Result<(), RepositoryError>;
    async fn get_edges(&self, node_id: u32) -> Result<Vec<Edge>, RepositoryError>;
    async fn add_edge(&self, edge: Edge) -> Result<(), RepositoryError>;
}
```

---

## Adapters (Infrastructure Implementations)

Adapters implement ports for specific technologies.

### Current Adapters

| Adapter | Port | Technology |
|---------|------|------------|
| `Neo4jGraphRepository` | GraphRepository | Neo4j database |
| `Neo4jOntologyRepository` | OntologyRepository | Neo4j database |
| `Neo4jSettingsRepository` | SettingsRepository | Neo4j database |
| `ActixPhysicsAdapter` | PhysicsSimulator | Actix actor system |
| `ActixSemanticAdapter` | SemanticAnalyzer | Actix actor system |
| `WhelkInferenceEngine` | InferenceEngine | whelk-rs EL++ reasoner |
| `PhysicsOrchestratorAdapter` | GPUPhysicsAdapter | CUDA kernels |
| `GPUSemanticAnalyzerAdapter` | GPUSemanticAnalyzer | CUDA ML |
| `ActorGraphRepository` | GraphRepository | Bridge to Actix actors |

### Adapter Implementation Example

```rust
// src/adapters/neo4j_graph_repository.rs
pub struct Neo4jGraphRepository {
    graph: Arc<Graph>,
    config: Neo4jConfig,
}

#[async_trait]
impl GraphRepository for Neo4jGraphRepository {
    async fn get_node(&self, id: u32) -> Result<Node, RepositoryError> {
        let query = "MATCH (n) WHERE n.id = $id RETURN n";
        let result = self.graph.execute(query!(id = id)).await?;
        // ... parse result into Node
    }

    async fn add_node(&self, node: Node) -> Result<u32, RepositoryError> {
        let query = "CREATE (n:Node $props) RETURN id(n)";
        // ... execute and return ID
    }
}
```

### Bridge Adapter

The `ActorGraphRepository` bridges hexagonal architecture to the Actix actor system:

```rust
// src/adapters/actor_graph_repository.rs
pub struct ActorGraphRepository {
    graph_actor: Addr<GraphStateActor>,
}

#[async_trait]
impl GraphRepository for ActorGraphRepository {
    async fn get_node(&self, id: u32) -> Result<Node, RepositoryError> {
        self.graph_actor
            .send(GetNode { id })
            .await
            .map_err(|e| RepositoryError::Actor(e.to_string()))?
    }
}
```

---

## CQRS Pattern

Command Query Responsibility Segregation separates reads from writes.

### Commands (Directives)

Commands mutate state and return confirmation:

```rust
// src/application/ontology/directives.rs
pub struct CreateClassDirective {
    pub ontology_id: String,
    pub class_iri: String,
    pub label: String,
}

pub struct CreateClassHandler {
    ontology_repo: Arc<dyn OntologyRepository>,
}

#[async_trait]
impl DirectiveHandler<CreateClassDirective> for CreateClassHandler {
    type Result = Result<ClassId, DomainError>;

    async fn handle(&self, cmd: CreateClassDirective) -> Self::Result {
        let class = OntologyClass::new(cmd.class_iri, cmd.label);
        self.ontology_repo.save_class(class).await
    }
}
```

### Queries

Queries read state without mutation:

```rust
// src/application/ontology/queries.rs
pub struct GetClassHierarchyQuery {
    pub ontology_id: String,
    pub root_class: Option<String>,
}

pub struct GetClassHierarchyHandler {
    ontology_repo: Arc<dyn OntologyRepository>,
    inference_engine: Arc<dyn InferenceEngine>,
}

#[async_trait]
impl QueryHandler<GetClassHierarchyQuery> for GetClassHierarchyHandler {
    type Result = Result<ClassHierarchy, DomainError>;

    async fn handle(&self, query: GetClassHierarchyQuery) -> Self::Result {
        self.inference_engine
            .get_class_hierarchy(&query.ontology_id)
            .await
    }
}
```

### Bus Infrastructure

Type-safe routing via `TypeId`:

```rust
// src/cqrs/bus.rs
pub struct CommandBus {
    handlers: HashMap<TypeId, Box<dyn AnyHandler>>,
}

impl CommandBus {
    pub fn register<C, H>(&mut self, handler: H)
    where
        C: 'static,
        H: DirectiveHandler<C> + 'static,
    {
        self.handlers.insert(TypeId::of::<C>(), Box::new(handler));
    }

    pub async fn dispatch<C: 'static>(&self, command: C) -> H::Result {
        let handler = self.handlers.get(&TypeId::of::<C>())
            .ok_or(BusError::NoHandler)?;
        handler.handle(command).await
    }
}
```

---

## Handler Distribution

114 CQRS handlers across 4 domains:

| Domain | Queries | Directives | Total |
|--------|---------|------------|-------|
| Ontology | 20 | 18 | 38 |
| Knowledge Graph | 10 | 16 | 26 |
| Settings | 10 | 12 | 22 |
| Physics | 4 | 8 | 12 |
| Graph | 16 | 0 | 16 |
| **Total** | **60** | **54** | **114** |

---

## Migration from Monolith

VisionFlow migrated from a monolithic `GraphServiceActor` (2000+ lines) to hexagonal architecture:

### Before (Monolith)

```
GraphServiceActor
├── Graph state management
├── Physics simulation
├── Semantic processing
├── Client coordination
├── Database access
├── Ontology reasoning
└── Settings management
```

**Problems**: God object, tight coupling, hard to test, difficult to change.

### After (Hexagonal)

```
9 Port traits
12 Adapters
114 CQRS handlers
4 Specialised actors (3,371 lines total)
```

**Benefits**: Single responsibility, testable, swappable, maintainable.

---

## Testing Strategy

### Port Testing

Test domain logic with mock adapters:

```rust
#[tokio::test]
async fn test_create_class() {
    let mock_repo = MockOntologyRepository::new();
    let handler = CreateClassHandler::new(Arc::new(mock_repo));

    let result = handler.handle(CreateClassDirective {
        ontology_id: "test".into(),
        class_iri: "http://example.org/Class1".into(),
        label: "Class 1".into(),
    }).await;

    assert!(result.is_ok());
}
```

### Adapter Testing

Test adapters against real infrastructure:

```rust
#[tokio::test]
async fn test_neo4j_repository() {
    let repo = Neo4jGraphRepository::new(test_config()).await;

    let id = repo.add_node(test_node()).await.unwrap();
    let node = repo.get_node(id).await.unwrap();

    assert_eq!(node.label, "Test Node");
}
```

### Integration Testing

Test full flow through actors and adapters:

```rust
#[actix_rt::test]
async fn test_end_to_end_graph_creation() {
    let app = create_test_app().await;

    let response = app.post("/api/nodes")
        .json(&json!({ "label": "Test" }))
        .send()
        .await;

    assert_eq!(response.status(), 201);
}
```

---

## Architecture Health

### Metrics (as of migration completion)

| Metric | Value |
|--------|-------|
| Port traits | 9 |
| Adapters | 12 |
| CQRS handlers | 114 |
| Specialised actors | 4 |
| Total actor lines | 3,371 |
| Technical debt markers | 2 |
| God objects | 0 |

### Verification Checklist

- [x] No direct database access from actors
- [x] All domain operations via CQRS handlers
- [x] Ports have multiple implementations
- [x] Infrastructure dependencies injectable
- [x] Domain testable in isolation

---

## Related Concepts

- **[Actor Model](actor-model.md)**: How actors use CQRS handlers
- **[GPU Acceleration](gpu-acceleration.md)**: GPU adapter implementation
- **[Ontology Reasoning](ontology-reasoning.md)**: InferenceEngine port and whelk adapter
