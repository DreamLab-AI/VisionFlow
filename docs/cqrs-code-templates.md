# CQRS Code Templates - Quick Reference

## Overview

This document provides copy-paste ready templates for implementing CQRS query handlers in the Graph domain, based on successful patterns from Settings and Ontology domains.

## File Structure

```
src/
├── application/
│   └── graph/
│       ├── mod.rs              # Module exports
│       ├── queries.rs          # Read operations (Phase 1)
│       └── directives.rs       # Write operations (Phase 2)
├── ports/
│   └── graph_repository.rs     # Repository interface (extend existing)
├── adapters/
│   └── actor_graph_repository.rs  # Actor-based adapter
└── handlers/
    └── api_handler/
        └── graph/
            └── mod.rs          # Update API routes
```

## 1. Query Handler Template

### Basic Query (No Parameters)

```rust
// src/application/graph/queries.rs

use hexser::{HexResult, Hexserror, QueryHandler};
use std::sync::Arc;

use crate::models::graph::GraphData;
use crate::ports::graph_repository::GraphRepository;

// ============================================================================
// GET GRAPH DATA
// ============================================================================

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

### Query with Parameters

```rust
// ============================================================================
// GET NODE BY ID
// ============================================================================

/// Query to retrieve a specific node by ID
#[derive(Debug, Clone)]
pub struct GetNodeById {
    pub node_id: u32,
}

/// Handler for GetNodeById query
pub struct GetNodeByIdHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetNodeByIdHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetNodeById, Option<Node>> for GetNodeByIdHandler {
    fn handle(&self, query: GetNodeById) -> HexResult<Option<Node>> {
        log::debug!("Executing GetNodeById query: node_id={}", query.node_id);

        let repository = self.repository.clone();
        let node_id = query.node_id;

        tokio::runtime::Handle::current()
            .block_on(async move {
                repository.get_node_by_id(node_id)
                    .await
                    .map_err(|e| Hexserror::port(
                        "E_GRAPH_002",
                        &format!("Failed to get node {}: {}", node_id, e)
                    ))
            })
    }
}
```

### Query with Computed Result

```rust
// ============================================================================
// GET PHYSICS STATE
// ============================================================================

use crate::actors::graph_actor::PhysicsState;

/// Query to get current physics simulation state
#[derive(Debug, Clone)]
pub struct GetPhysicsState;

/// Handler for GetPhysicsState query
pub struct GetPhysicsStateHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetPhysicsStateHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetPhysicsState, PhysicsState> for GetPhysicsStateHandler {
    fn handle(&self, _query: GetPhysicsState) -> HexResult<PhysicsState> {
        log::debug!("Executing GetPhysicsState query");

        let repository = self.repository.clone();

        tokio::runtime::Handle::current()
            .block_on(async move {
                repository.get_physics_state()
                    .await
                    .map_err(|e| Hexserror::port(
                        "E_GRAPH_003",
                        &format!("Failed to get physics state: {}", e)
                    ))
            })
    }
}
```

## 2. Module Export Template

```rust
// src/application/graph/mod.rs

//! Graph Domain - CQRS Application Layer
//!
//! This module implements the CQRS pattern for graph operations:
//! - Queries (queries.rs): Read operations that return data
//! - Directives (directives.rs): Write operations that modify state (Phase 2)

pub mod queries;
// pub mod directives; // Phase 2

// Re-export all queries
pub use queries::{
    GetGraphData, GetGraphDataHandler,
    GetNodeMap, GetNodeMapHandler,
    GetPhysicsState, GetPhysicsStateHandler,
    GetNodePositions, GetNodePositionsHandler,
    GetBotsGraphData, GetBotsGraphDataHandler,
    GetConstraints, GetConstraintsHandler,
    GetAutoBalanceNotifications, GetAutoBalanceNotificationsHandler,
};
```

## 3. Repository Port Extension

```rust
// src/ports/graph_repository.rs

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::constraints::ConstraintSet;
use crate::actors::graph_actor::{PhysicsState, AutoBalanceNotification};

pub type Result<T> = std::result::Result<T, GraphRepositoryError>;

#[derive(Debug, thiserror::Error)]
pub enum GraphRepositoryError {
    #[error("Graph not found")]
    NotFound,

    #[error("Graph access error: {0}")]
    AccessError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Actor communication error: {0}")]
    ActorError(String),
}

/// Port for graph data repository operations
#[async_trait]
pub trait GraphRepository: Send + Sync {
    // ========== READ OPERATIONS (Phase 1) ==========

    /// Get the current graph structure (knowledge graph)
    async fn get_graph(&self) -> Result<Arc<GraphData>>;

    /// Get all nodes with current physics positions
    async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>>;

    /// Get current physics simulation state
    async fn get_physics_state(&self) -> Result<PhysicsState>;

    /// Get node positions as vector of (id, position) tuples
    async fn get_node_positions(&self) -> Result<Vec<(u32, glam::Vec3)>>;

    /// Get the bots/agents graph structure
    async fn get_bots_graph(&self) -> Result<Arc<GraphData>>;

    /// Get current constraint set
    async fn get_constraints(&self) -> Result<ConstraintSet>;

    /// Get auto-balance notification history
    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>>;

    // ========== WRITE OPERATIONS (Phase 2) ==========
    // async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;
    // async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;
    // async fn update_positions(&self, updates: Vec<(u32, BinaryNodeData)>) -> Result<()>;
}
```

## 4. Actor-Based Repository Adapter

```rust
// src/adapters/actor_graph_repository.rs

//! Actor-based Graph Repository Adapter
//!
//! Implements GraphRepository port using the existing GraphServiceActor.
//! This allows CQRS queries to work with the current actor system during
//! the migration period.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use actix::Addr;

use crate::actors::graph_actor::{GraphServiceActor, PhysicsState, AutoBalanceNotification};
use crate::actors::messages as actor_msgs;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::constraints::ConstraintSet;
use crate::ports::graph_repository::{GraphRepository, GraphRepositoryError, Result};

/// Actor-based implementation of GraphRepository
pub struct ActorGraphRepository {
    actor_addr: Addr<GraphServiceActor>,
}

impl ActorGraphRepository {
    /// Create new actor-based repository
    pub fn new(actor_addr: Addr<GraphServiceActor>) -> Self {
        log::info!("Initializing ActorGraphRepository adapter");
        Self { actor_addr }
    }
}

#[async_trait]
impl GraphRepository for ActorGraphRepository {
    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        log::debug!("ActorGraphRepository::get_graph - sending message to actor");

        self.actor_addr
            .send(actor_msgs::GetGraphData)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }

    async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>> {
        log::debug!("ActorGraphRepository::get_node_map - sending message to actor");

        self.actor_addr
            .send(actor_msgs::GetNodeMap)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }

    async fn get_physics_state(&self) -> Result<PhysicsState> {
        log::debug!("ActorGraphRepository::get_physics_state - sending message to actor");

        self.actor_addr
            .send(actor_msgs::GetPhysicsState)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }

    async fn get_node_positions(&self) -> Result<Vec<(u32, glam::Vec3)>> {
        log::debug!("ActorGraphRepository::get_node_positions - sending message to actor");

        self.actor_addr
            .send(actor_msgs::GetNodePositions)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }

    async fn get_bots_graph(&self) -> Result<Arc<GraphData>> {
        log::debug!("ActorGraphRepository::get_bots_graph - sending message to actor");

        self.actor_addr
            .send(actor_msgs::GetBotsGraphData)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }

    async fn get_constraints(&self) -> Result<ConstraintSet> {
        log::debug!("ActorGraphRepository::get_constraints - sending message to actor");

        self.actor_addr
            .send(actor_msgs::GetConstraints)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }

    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>> {
        log::debug!("ActorGraphRepository::get_auto_balance_notifications - sending message to actor");

        self.actor_addr
            .send(actor_msgs::GetAutoBalanceNotifications)
            .await
            .map_err(|e| GraphRepositoryError::ActorError(format!("Mailbox error: {}", e)))?
            .map_err(|e| GraphRepositoryError::AccessError(e))
    }
}
```

## 5. AppState Update

```rust
// src/app_state.rs (additions)

use std::sync::Arc;
use crate::application::graph::{
    GetGraphDataHandler,
    GetNodeMapHandler,
    GetPhysicsStateHandler,
    GetNodePositionsHandler,
    GetBotsGraphDataHandler,
    GetConstraintsHandler,
    GetAutoBalanceNotificationsHandler,
};
use crate::adapters::actor_graph_repository::ActorGraphRepository;
use crate::ports::graph_repository::GraphRepository;

/// Query handlers for graph read operations
pub struct GraphQueryHandlers {
    pub get_graph_data: Arc<GetGraphDataHandler>,
    pub get_node_map: Arc<GetNodeMapHandler>,
    pub get_physics_state: Arc<GetPhysicsStateHandler>,
    pub get_node_positions: Arc<GetNodePositionsHandler>,
    pub get_bots_graph: Arc<GetBotsGraphDataHandler>,
    pub get_constraints: Arc<GetConstraintsHandler>,
    pub get_auto_balance_notifications: Arc<GetAutoBalanceNotificationsHandler>,
}

impl GraphQueryHandlers {
    /// Initialize all query handlers with the given repository
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        log::info!("Initializing GraphQueryHandlers");
        Self {
            get_graph_data: Arc::new(GetGraphDataHandler::new(repository.clone())),
            get_node_map: Arc::new(GetNodeMapHandler::new(repository.clone())),
            get_physics_state: Arc::new(GetPhysicsStateHandler::new(repository.clone())),
            get_node_positions: Arc::new(GetNodePositionsHandler::new(repository.clone())),
            get_bots_graph: Arc::new(GetBotsGraphDataHandler::new(repository.clone())),
            get_constraints: Arc::new(GetConstraintsHandler::new(repository.clone())),
            get_auto_balance_notifications: Arc::new(GetAutoBalanceNotificationsHandler::new(repository)),
        }
    }
}

pub struct AppState {
    // Existing fields...
    pub graph_service_addr: Addr<GraphServiceActor>,

    // NEW: CQRS query handlers
    pub graph_query_handlers: GraphQueryHandlers,
}
```

## 6. Main.rs Initialization

```rust
// src/main.rs (in app initialization)

use crate::adapters::actor_graph_repository::ActorGraphRepository;
use crate::app_state::GraphQueryHandlers;

// ... in main() or app setup ...

// Create actor-based repository adapter
let graph_repository = Arc::new(ActorGraphRepository::new(graph_service_addr.clone()));

// Initialize query handlers
let graph_query_handlers = GraphQueryHandlers::new(graph_repository);

// Build AppState
let app_state = web::Data::new(AppState {
    graph_service_addr,
    graph_query_handlers,
    // ... other fields
});
```

## 7. API Route Update

```rust
// src/handlers/api_handler/graph/mod.rs

use actix_web::{web, HttpResponse, Responder};
use crate::AppState;
use crate::application::graph::{GetGraphData, GetNodeMap, GetPhysicsState};

/// Get graph data with physics positions (CQRS version)
pub async fn get_graph_data(state: web::Data<AppState>) -> impl Responder {
    log::info!("API: get_graph_data using CQRS query handlers");

    // Clone handlers for parallel execution
    let graph_handler = state.graph_query_handlers.get_graph_data.clone();
    let node_map_handler = state.graph_query_handlers.get_node_map.clone();
    let physics_handler = state.graph_query_handlers.get_physics_state.clone();

    // Execute queries in parallel using spawn_blocking
    let (graph_result, node_map_result, physics_result) = tokio::join!(
        tokio::task::spawn_blocking(move || graph_handler.handle(GetGraphData)),
        tokio::task::spawn_blocking(move || node_map_handler.handle(GetNodeMap)),
        tokio::task::spawn_blocking(move || physics_handler.handle(GetPhysicsState))
    );

    // Handle results
    match (graph_result, node_map_result, physics_result) {
        (Ok(Ok(graph_data)), Ok(Ok(node_map)), Ok(Ok(physics_state))) => {
            log::debug!(
                "Successfully fetched graph data: {} nodes, {} edges, settled: {}",
                graph_data.nodes.len(),
                graph_data.edges.len(),
                physics_state.is_settled
            );

            // Build response with positions
            let nodes_with_positions: Vec<NodeWithPosition> = graph_data
                .nodes
                .iter()
                .map(|node| {
                    let (position, velocity) = if let Some(physics_node) = node_map.get(&node.id) {
                        (physics_node.data.position(), physics_node.data.velocity())
                    } else {
                        (node.data.position(), node.data.velocity())
                    };

                    NodeWithPosition {
                        id: node.id,
                        metadata_id: node.metadata_id.clone(),
                        label: node.label.clone(),
                        position,
                        velocity,
                        metadata: node.metadata.clone(),
                        node_type: node.node_type.clone(),
                        size: node.size,
                        color: node.color.clone(),
                        weight: node.weight,
                        group: node.group.clone(),
                    }
                })
                .collect();

            let response = GraphResponseWithPositions {
                nodes: nodes_with_positions,
                edges: graph_data.edges.clone(),
                metadata: graph_data.metadata.clone(),
                settlement_state: SettlementState {
                    is_settled: physics_state.is_settled,
                    stable_frame_count: physics_state.stable_frame_count,
                    kinetic_energy: physics_state.kinetic_energy,
                },
            };

            HttpResponse::Ok().json(response)
        }
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
            log::error!("Task join error: {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "Query execution failed"}))
        }
        (Ok(Err(e)), _, _) | (_, Ok(Err(e)), _) | (_, _, Ok(Err(e))) => {
            log::error!("Query handler error: {}", e);
            HttpResponse::InternalServerError()
                .json(serde_json::json!({"error": "Failed to retrieve graph data"}))
        }
    }
}
```

## 8. Feature Flag Pattern

```rust
// src/handlers/api_handler/graph/mod.rs

use std::env;

pub async fn get_graph_data(state: web::Data<AppState>) -> impl Responder {
    // Check feature flag
    if env::var("USE_CQRS_QUERIES").is_ok() {
        log::info!("Using CQRS query handlers (feature flag enabled)");
        get_graph_data_cqrs(state).await
    } else {
        log::info!("Using legacy actor approach (feature flag disabled)");
        get_graph_data_legacy(state).await
    }
}

/// CQRS implementation
async fn get_graph_data_cqrs(state: web::Data<AppState>) -> impl Responder {
    // Use query handlers...
}

/// Legacy implementation (fallback)
async fn get_graph_data_legacy(state: web::Data<AppState>) -> impl Responder {
    // Direct actor calls...
}
```

## 9. Unit Test Template

```rust
// src/application/graph/queries.rs (test module)

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Mock repository for testing
    struct MockGraphRepository {
        graph: Arc<GraphData>,
        node_map: Arc<HashMap<u32, Node>>,
    }

    #[async_trait]
    impl GraphRepository for MockGraphRepository {
        async fn get_graph(&self) -> Result<Arc<GraphData>> {
            Ok(Arc::clone(&self.graph))
        }

        async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>> {
            Ok(Arc::clone(&self.node_map))
        }

        // Implement other methods...
    }

    #[test]
    fn test_get_graph_data_handler() {
        // Arrange
        let mock_graph = Arc::new(GraphData {
            nodes: vec![],
            edges: vec![],
            metadata: HashMap::new(),
        });

        let mock_repo = Arc::new(MockGraphRepository {
            graph: mock_graph.clone(),
            node_map: Arc::new(HashMap::new()),
        });

        let handler = GetGraphDataHandler::new(mock_repo);

        // Act
        let result = handler.handle(GetGraphData);

        // Assert
        assert!(result.is_ok());
        let graph = result.unwrap();
        assert_eq!(graph.nodes.len(), 0);
    }

    #[test]
    fn test_get_graph_data_handler_error() {
        // Test error handling...
    }
}
```

## 10. Integration Test Template

```rust
// tests/integration/graph_queries_test.rs

use actix_web::{test, web, App};
use visionflow::handlers::api_handler::graph;
use visionflow::app_state::AppState;

#[actix_web::test]
async fn test_get_graph_data_endpoint() {
    // Initialize test app state
    let app_state = web::Data::new(AppState {
        // ... initialize with test data
    });

    // Create test app
    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .route("/api/graph/data", web::get().to(graph::get_graph_data))
    ).await;

    // Make request
    let req = test::TestRequest::get()
        .uri("/api/graph/data")
        .to_request();

    let resp = test::call_service(&app, req).await;

    // Assert
    assert!(resp.status().is_success());
}
```

## Complete Query Handlers File

Here's a complete `queries.rs` with all 7 handlers:

```rust
// src/application/graph/queries.rs

//! Graph Domain - Read Operations (Queries)
//!
//! All queries for reading graph state following CQRS patterns.
//! This module provides read-only access to graph data, physics state,
//! and constraints without side effects.

use hexser::{HexResult, Hexserror, QueryHandler};
use std::collections::HashMap;
use std::sync::Arc;

use crate::actors::graph_actor::{AutoBalanceNotification, PhysicsState};
use crate::models::constraints::ConstraintSet;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::graph_repository::GraphRepository;

// ============================================================================
// GET GRAPH DATA
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetGraphData;

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
        tokio::runtime::Handle::current().block_on(async move {
            repository.get_graph().await.map_err(|e| {
                Hexserror::port("E_GRAPH_001", &format!("Failed to get graph: {}", e))
            })
        })
    }
}

// ============================================================================
// GET NODE MAP
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetNodeMap;

pub struct GetNodeMapHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetNodeMapHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetNodeMap, Arc<HashMap<u32, Node>>> for GetNodeMapHandler {
    fn handle(&self, _query: GetNodeMap) -> HexResult<Arc<HashMap<u32, Node>>> {
        log::debug!("Executing GetNodeMap query");
        let repository = self.repository.clone();
        tokio::runtime::Handle::current().block_on(async move {
            repository.get_node_map().await.map_err(|e| {
                Hexserror::port("E_GRAPH_002", &format!("Failed to get node map: {}", e))
            })
        })
    }
}

// ============================================================================
// GET PHYSICS STATE
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetPhysicsState;

pub struct GetPhysicsStateHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetPhysicsStateHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetPhysicsState, PhysicsState> for GetPhysicsStateHandler {
    fn handle(&self, _query: GetPhysicsState) -> HexResult<PhysicsState> {
        log::debug!("Executing GetPhysicsState query");
        let repository = self.repository.clone();
        tokio::runtime::Handle::current().block_on(async move {
            repository.get_physics_state().await.map_err(|e| {
                Hexserror::port("E_GRAPH_003", &format!("Failed to get physics state: {}", e))
            })
        })
    }
}

// ============================================================================
// GET NODE POSITIONS
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetNodePositions;

pub struct GetNodePositionsHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetNodePositionsHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetNodePositions, Vec<(u32, glam::Vec3)>> for GetNodePositionsHandler {
    fn handle(&self, _query: GetNodePositions) -> HexResult<Vec<(u32, glam::Vec3)>> {
        log::debug!("Executing GetNodePositions query");
        let repository = self.repository.clone();
        tokio::runtime::Handle::current().block_on(async move {
            repository.get_node_positions().await.map_err(|e| {
                Hexserror::port("E_GRAPH_004", &format!("Failed to get node positions: {}", e))
            })
        })
    }
}

// ============================================================================
// GET BOTS GRAPH DATA
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetBotsGraphData;

pub struct GetBotsGraphDataHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetBotsGraphDataHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetBotsGraphData, Arc<GraphData>> for GetBotsGraphDataHandler {
    fn handle(&self, _query: GetBotsGraphData) -> HexResult<Arc<GraphData>> {
        log::debug!("Executing GetBotsGraphData query");
        let repository = self.repository.clone();
        tokio::runtime::Handle::current().block_on(async move {
            repository.get_bots_graph().await.map_err(|e| {
                Hexserror::port("E_GRAPH_005", &format!("Failed to get bots graph: {}", e))
            })
        })
    }
}

// ============================================================================
// GET CONSTRAINTS
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetConstraints;

pub struct GetConstraintsHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetConstraintsHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetConstraints, ConstraintSet> for GetConstraintsHandler {
    fn handle(&self, _query: GetConstraints) -> HexResult<ConstraintSet> {
        log::debug!("Executing GetConstraints query");
        let repository = self.repository.clone();
        tokio::runtime::Handle::current().block_on(async move {
            repository.get_constraints().await.map_err(|e| {
                Hexserror::port("E_GRAPH_006", &format!("Failed to get constraints: {}", e))
            })
        })
    }
}

// ============================================================================
// GET AUTO BALANCE NOTIFICATIONS
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetAutoBalanceNotifications;

pub struct GetAutoBalanceNotificationsHandler {
    repository: Arc<dyn GraphRepository>,
}

impl GetAutoBalanceNotificationsHandler {
    pub fn new(repository: Arc<dyn GraphRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetAutoBalanceNotifications, Vec<AutoBalanceNotification>>
    for GetAutoBalanceNotificationsHandler
{
    fn handle(
        &self,
        _query: GetAutoBalanceNotifications,
    ) -> HexResult<Vec<AutoBalanceNotification>> {
        log::debug!("Executing GetAutoBalanceNotifications query");
        let repository = self.repository.clone();
        tokio::runtime::Handle::current().block_on(async move {
            repository
                .get_auto_balance_notifications()
                .await
                .map_err(|e| {
                    Hexserror::port(
                        "E_GRAPH_007",
                        &format!("Failed to get auto balance notifications: {}", e),
                    )
                })
        })
    }
}
```

## Summary

These templates provide everything needed to implement CQRS Phase 1 for graph read operations:

1. ✅ Query handler pattern with error handling
2. ✅ Repository port interface extensions
3. ✅ Actor-based adapter for gradual migration
4. ✅ AppState updates for dependency injection
5. ✅ API route modifications
6. ✅ Feature flag support for safe rollout
7. ✅ Unit and integration test templates

**Next Step**: Copy these templates and implement Phase 1!
