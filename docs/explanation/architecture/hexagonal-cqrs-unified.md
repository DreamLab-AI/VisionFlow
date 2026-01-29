---
title: Hexagonal Architecture with CQRS
description: Complete reference for VisionFlow's hexagonal (ports-and-adapters) architecture with CQRS patterns, 21 actors, 9 ports, 12 adapters, and Neo4j persistence.
category: explanation
tags:
  - architecture
  - hexagonal
  - cqrs
  - design-patterns
  - ports-adapters
updated-date: 2025-01-29
difficulty-level: advanced
---

# Hexagonal Architecture with CQRS

VisionFlow employs hexagonal (ports-and-adapters) architecture with CQRS to achieve clean separation between domain logic, infrastructure, and presentation.

---

## Executive Summary

VisionFlow has successfully implemented a **production-grade hexagonal architecture** with complete separation of concerns, fault-tolerant actor system, and unified Neo4j persistence.

### Architecture at a Glance

- **21 Specialised Actors** - Supervised hierarchy with fault tolerance
- **9 Port Interfaces** - Technology-agnostic domain boundaries
- **12 Adapter Implementations** - Neo4j, GPU, WebSocket, HTTP
- **~114 CQRS Handlers** - Separate read/write operations
- **Neo4j Database** - Single source of truth
- **Real-time WebSocket** - Per-client filtering and updates

### Migration Success

| Before (Oct 2025) | After (Dec 2025) | Status |
|-------------------|------------------|--------|
| Monolithic GraphServiceActor (48K tokens) | 21 modular actors | Complete |
| Stale in-memory cache | Neo4j source of truth | Complete |
| Tight coupling | Hexagonal ports/adapters | Complete |
| No CQRS | Command/Query separation | Complete |
| SQLite fragmentation | Unified Neo4j database | Complete |

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
+-------------------------------------------------------------+
|                 Hexagonal Architecture                       |
+-------------------------------------------------------------+
|                                                              |
|  +-------------------------------------------------------+  |
|  |              Infrastructure (Adapters)                 |  |
|  |                                                        |  |
|  |  Neo4j Adapters    Actix Adapters    GPU Adapters     |  |
|  |  (Database)        (Actors)          (CUDA)           |  |
|  +-------------------------------------------------------+  |
|                            | implements                      |
|  +-------------------------------------------------------+  |
|  |                   Domain (Ports)                       |  |
|  |                                                        |  |
|  |  GraphRepository   PhysicsSimulator   InferenceEngine |  |
|  |  OntologyRepository SettingsRepository SemanticAnalyzer|  |
|  +-------------------------------------------------------+  |
|                            ^ uses                           |
|  +-------------------------------------------------------+  |
|  |              Application Layer (CQRS)                  |  |
|  |                                                        |  |
|  |  CommandBus    QueryBus    114 Handlers               |  |
|  |  (Directives)  (Queries)   (Domain Operations)        |  |
|  +-------------------------------------------------------+  |
|                            ^ invokes                        |
|  +-------------------------------------------------------+  |
|  |              Presentation (Actors)                     |  |
|  |                                                        |  |
|  |  GraphStateActor   PhysicsOrchestrator   ClientCoord  |  |
|  |  (State Machine)   (GPU Coordination)    (WebSocket)  |  |
|  +-------------------------------------------------------+  |
|                                                              |
+-------------------------------------------------------------+
```

---

## Port Interfaces (9 Total)

Located in `src/ports/`, these define technology-agnostic boundaries:

| Port | Purpose | Methods | Used By |
|------|---------|---------|---------|
| GraphRepository | Graph CRUD | get_graph, save_graph, add_node, update_positions | GraphStateActor, CQRS handlers |
| KnowledgeGraphRepository | KG operations | save_nodes, get_metadata, sync_from_github | GitHub sync, OntologyActor |
| OntologyRepository | Ontology storage | save_axioms, get_classes, reason | OntologyActor, SemanticProcessor |
| SettingsRepository | User settings | get_setting, set_setting, get_all | OptimizedSettingsActor, API handlers |
| PhysicsSimulator | Physics compute | simulate_step, calculate_forces | PhysicsOrchestratorActor |
| SemanticAnalyzer | Semantic analysis | analyze_communities, detect_patterns | SemanticProcessorActor |
| GpuPhysicsAdapter | GPU physics | batch_force_compute, optimize_layout | ForceComputeActor, GPU actors |
| GpuSemanticAnalyzer | GPU semantic | pagerank, clustering, pathfinding | GPU semantic actors |
| InferenceEngine | OWL reasoning | infer_axioms, classify_hierarchy | OntologyActor |

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

## Adapter Implementations (12 Total)

Located in `src/adapters/`, these implement ports with concrete technologies.

### Neo4j Adapters (5)

| Adapter | Implements Port | Technology | Performance |
|---------|----------------|------------|-------------|
| Neo4jAdapter | KnowledgeGraphRepository | Bolt protocol | ~2ms per query |
| Neo4jGraphRepository | GraphRepository | Cypher queries | ~12ms for full graph |
| Neo4jSettingsRepository | SettingsRepository | Cypher + auth | ~3ms per setting |
| Neo4jOntologyRepository | OntologyRepository | Graph storage | ~25ms for traversal |
| ActorGraphRepository | GraphRepository | Actor bridge | ~15ms (adds overhead) |

### GPU Adapters (2)

| Adapter | Implements Port | Technology | Performance |
|---------|----------------|------------|-------------|
| GpuSemanticAnalyzerAdapter | GpuSemanticAnalyzer | CUDA kernels | ~4ms per step |
| ActixPhysicsAdapter | GpuPhysicsAdapter | Actor wrapper | ~16ms per step |

### Other Adapters (5)

| Adapter | Implements Port | Technology | Performance |
|---------|----------------|------------|-------------|
| ActixSemanticAdapter | SemanticAnalyzer | Actor wrapper | ~20ms per analysis |
| PhysicsOrchestratorAdapter | PhysicsSimulator | Actor coordination | ~16ms per step |
| WhelkInferenceEngine | InferenceEngine | Rust OWL reasoner | ~100ms per reasoning |
| ActixWebSocketAdapter | (implicit) | WebSocket protocol | ~3ms per broadcast |
| ActixHttpAdapter | (implicit) | HTTP handlers | <1ms routing |

### Bridge Adapter Example

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

## Event-Driven Architecture

### Event Bus

```rust
#[async_trait]
pub trait EventBus: Send + Sync {
    async fn publish(&self, event: GraphEvent) -> Result<(), String>;
    async fn subscribe(&self, event_type: &str, handler: Arc<dyn EventHandler>) -> Result<(), String>;
}
```

### Domain Events

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEvent {
    NodeCreated {
        node_id: u32,
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },
    NodePositionChanged {
        node_id: u32,
        old_position: (f32, f32, f32),
        new_position: (f32, f32, f32),
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },
    PhysicsStepCompleted {
        iteration: usize,
        nodes_updated: usize,
        timestamp: DateTime<Utc>,
    },
    GitHubSyncCompleted {
        total_nodes: usize,
        total_edges: usize,
        timestamp: DateTime<Utc>,
    },
}
```

### Cache Invalidation Strategy

**Current approach**: No caching (Neo4j is source of truth)

**Future approach** (planned):
1. Redis cache for frequently accessed queries
2. Event-driven invalidation on writes
3. TTL-based expiration for safety

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

## Directory Structure

```
src/
+-- application/              # Application layer (CQRS)
|   +-- graph/
|   |   +-- commands.rs       # Write operations
|   |   +-- command_handlers.rs
|   |   +-- queries.rs        # Read operations
|   |   +-- query_handlers.rs
|   +-- physics/
|   +-- ontology/
|
+-- domain/                   # Domain layer (business logic)
|   +-- events.rs             # Domain events
|   +-- services/
|       +-- physics_service.rs
|       +-- semantic_service.rs
|
+-- ports/                    # Port interfaces (traits)
|   +-- graph_repository.rs
|   +-- event_store.rs
|   +-- websocket_gateway.rs
|   +-- physics_simulator.rs
|
+-- adapters/                 # Adapter implementations
|   +-- neo4j_graph_repository.rs
|   +-- actix_websocket_adapter.rs
|   +-- gpu_physics_adapter.rs
|
+-- infrastructure/           # Infrastructure concerns
    +-- event_bus.rs
    +-- cache_service.rs
```

---

## Performance Characteristics

| Query Type | Typical Time |
|------------|--------------|
| Neo4j query (simple) | 2-3 ms |
| Neo4j full graph | 12 ms |
| CQRS handler dispatch | <1 ms |
| Event publication | <10 ms |
| WebSocket broadcast | 3 ms |

---

## Related Documentation

- [Actor System](../concepts/actor-model.md)
- [Database Schema Catalog](../../reference/database/schema-catalog.md)
- [REST API Reference](../../reference/api/rest-api.md)

---

**Last Updated**: January 29, 2025
**Maintainer**: VisionFlow Architecture Team
