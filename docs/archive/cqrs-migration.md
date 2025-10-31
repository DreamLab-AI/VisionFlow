# CQRS Migration and Monolith Deprecation Strategy

**Version:** 1.0
**Date:** 2025-10-27
**Status:** In Progress

---

## 1. Overview

This document outlines the strategy for migrating the legacy `GraphServiceActor` monolith to a modern CQRS (Command Query Responsibility Segregation) and Hexagonal Architecture. This migration is a critical step in reducing technical debt, improving maintainability, and enabling future development.

The core of this effort involves:
-   **Separating Reads and Writes:** Implementing distinct paths for queries (reading data) and commands (mutating data).
-   **Adopting a Ports and Adapters Pattern:** Decoupling the application's core logic from external concerns like databases, APIs, and the actor model.
-   **Phased Deprecation:** Gradually and safely removing the 4,566-line `GraphServiceActor` monolith.

## 2. CQRS Implementation Guide

### 2.1. File Structure

The new CQRS structure for a given domain (e.g., `graph`) should follow this pattern:

```
src/
├── application/
│   └── graph/
│       ├── mod.rs              # Module exports
│       ├── queries.rs          # Read operations (Phase 1)
│       └── directives.rs       # Write operations (Phase 2)
├── ports/
│   └── graph_repository.rs     # Repository interface
├── adapters/
│   └── actor_graph_repository.rs  # Actor-based adapter for transition
```

### 2.2. Query Handler Template

Queries are used for all read operations. They should not have any side effects.

```rust
// src/application/graph/queries.rs

use hexser::{HexResult, Hexserror, QueryHandler};
use std::sync::Arc;
use crate::models::graph::GraphData;
use crate::ports::graph_repository::GraphRepository;

/// Query to retrieve the current graph structure
#[derive(Debug, Clone)]
pub struct GetGraphData;

/// Handler for GetGraphData query
pub struct GetGraphDataHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetGraphDataHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetGraphData, Arc<GraphData>> for GetGraphDataHandler {
    fn handle(&self, _query: GetGraphData) -> HexResult<Arc<GraphData>> {
        log::debug!("Executing GetGraphData query");

        let repository = self.repository.clone();

        tokio::runtime::Handle::current()
            .block_on(async move {
                repository.get_graph()
                    .await
                    .map_err(|e| Hexserror::port("E_GRAPH_001", &format!("Failed to get graph: {}", e)))
            })
    }
}
```

### 2.3. Repository Port

The repository port defines the interface for data access, abstracting the underlying implementation.

```rust
// src/ports/graph_repository.rs

use async_trait::async_trait;
use std::sync::Arc;
use crate::models::graph::GraphData;

#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_graph(&self) -> Result<Arc<GraphData>>;
    // Additional read/write methods will be defined here.
}
```

### 2.4. Transitional Actor-Based Adapter

During the migration, an adapter can bridge the new CQRS handlers to the existing actor system, allowing for an incremental transition.

```rust
// src/adapters/actor_graph_repository.rs

use async_trait::async_trait;
use std::sync::Arc;
use actix::Addr;
use crate::actors::graph_actor::GraphServiceActor;
use crate::actors::messages as actor_msgs;
use crate::models::graph::GraphData;
use crate::ports::graph_repository::{GraphRepository, GraphRepositoryError, Result};

pub struct ActorGraphRepository {
    actor_addr: Addr<GraphServiceActor>,
}

#[async_trait]
impl GraphRepository for ActorGraphRepository {
    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        self.actor_addr
            .send(actor_msgs::GetGraphData)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }
}
```

## 3. Monolith Deprecation Strategy

The `GraphServiceActor` will be deprecated in three phases over 6-9 months.

### Phase 1: Safe Deprecation of Read Operations (Months 1-2)

-   **Objective:** Mark low-risk, stateless read methods as `#[deprecated]`.
-   **Strategy:** Create CQRS query handler equivalents for all read operations. The API layer will be updated to use these new handlers, which will initially delegate to the actor via the `ActorGraphRepository` adapter.

### Phase 2: Create CQRS Equivalents for Write Operations (Months 3-5)

-   **Objective:** Build CQRS command handlers for all stateful operations.
-   **Strategy:** Implement an event bus to synchronize state between the new CQRS write path and the legacy actor's in-memory state. This allows both systems to run in parallel. The `GraphServiceActor` will become an event subscriber to keep its state consistent with changes made via CQRS commands.

### Phase 3: Traffic Routing & Legacy Removal (Months 6-9)

-   **Objective:** Route all API traffic to the new CQRS handlers and remove the legacy actor code.
-   **Strategy:** Use feature flags to gradually shift traffic from the legacy actor path to the new CQRS path. Once 100% of traffic is on the CQRS path and the system is stable, the `GraphServiceActor` will be refactored into a thin real-time gateway responsible only for WebSocket communication and physics orchestration. Deprecated methods and handlers will be deleted.

## 4. Legacy Code Removal Timeline

The full migration is estimated to take 7-8 weeks of engineering time (304 hours).

-   **Phase 0: Preparation (Week 1):** Add `#[deprecated]` attributes, establish baselines, and create integration tests.
-   **Phase 1: GraphServiceActor Migration (Weeks 2-4):** Build the hexagonal foundation (domain, repository, CQRS handlers) for graph operations and migrate API handlers.
-   **Phase 2: PhysicsOrchestratorActor Migration (Week 5):** Consolidate physics logic into a new `PhysicsService`.
-   **Phase 3: GPU Actors Migration (Weeks 6-7):** Migrate GPU worker actors to services, removing the actor-based abstraction.
-   **Phase 4: Final Cleanup (Week 8):** Remove all remaining legacy actor infrastructure, clean up `AppState`, and verify system stability.

By following this phased approach, we can safely and incrementally migrate our monolithic actor system to a modern, maintainable, and scalable architecture.