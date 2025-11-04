# CQRS Directive Handler Template
**Copy-Paste Template for Creating Graph Directive Handlers**

---

## Template Structure

### File: `src/application/graph/directives.rs`

```rust
//! Graph Domain - Write Operations (Directives)
//!
//! All directives for modifying graph state following CQRS patterns.
//! This file replaces actor message handlers for write operations.

use hexser::{Directive, DirectiveHandler, HexResult, Hexserror};
use std::sync::Arc;

use crate::application::events::{DomainEventPublisher, GraphEvent};
use crate::models::edge::Edge;
use crate::models::node::Node;
use crate::ports::graph_repository::GraphRepository;

// ============================================================================
// CREATE NODE
// ============================================================================

/// Directive to create a new node in the graph.
#[derive(Debug, Clone)]
pub struct CreateNode {
    pub node: Node,
}

impl Directive for CreateNode {
    fn validate(&self) -> HexResult<()> {
        if self.node.metadata_id.is_empty() {
            return Err(Hexserror::validation("Node metadata_id cannot be empty"));
        }

        if self.node.label.is_empty() {
            return Err(Hexserror::validation("Node label cannot be empty"));
        }

        Ok(())
    }
}

/// Handler for CreateNode directive.
pub struct CreateNodeHandler {
    repository: Arc<dyn GraphRepository>,
    event_publisher: Arc<dyn DomainEventPublisher>,
}

impl CreateNodeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event_publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event_publisher,
        }
    }
}

impl DirectiveHandler<CreateNode> for CreateNodeHandler {
    fn handle(&self, directive: CreateNode) -> HexResult<()> {
        log::info!(
            "Executing CreateNode directive: metadata_id={}, label={}",
            directive.node.metadata_id,
            directive.node.label
        );

        // 1. Validate directive
        directive.validate()?;

        // 2. Execute domain logic
        let node = directive.node.clone();

        // 3. Persist via repository
        let repository = self.repository.clone();
        let event_publisher = self.event_publisher.clone();

        // Use tokio runtime for async repository call
        tokio::runtime::Handle::current().block_on(async move {
            repository
                .add_nodes(vec![node.clone()])
                .await
                .map_err(|e| {
                    Hexserror::adapter(
                        "E_GRAPH_CREATE_001",
                        &format!("Failed to create node: {}", e),
                    )
                })?;

            // 4. Emit domain event
            let event = GraphEvent::NodeCreated {
                node_id: node.id,
                metadata_id: node.metadata_id.clone(),
                label: node.label.clone(),
                timestamp: chrono::Utc::now(),
            };

            event_publisher.publish(event).map_err(|e| {
                Hexserror::adapter(
                    "E_GRAPH_CREATE_002",
                    &format!("Failed to publish NodeCreated event: {}", e),
                )
            })?;

            log::info!("✅ Node created successfully: id={}", node.id);

            Ok(())
        })
    }
}

// ============================================================================
// CREATE EDGE
// ============================================================================

/// Directive to create a new edge in the graph.
#[derive(Debug, Clone)]
pub struct CreateEdge {
    pub edge: Edge,
}

impl Directive for CreateEdge {
    fn validate(&self) -> HexResult<()> {
        if self.edge.id.is_empty() {
            return Err(Hexserror::validation("Edge id cannot be empty"));
        }

        if self.edge.source == 0 {
            return Err(Hexserror::validation("Edge source node_id cannot be 0"));
        }

        if self.edge.target == 0 {
            return Err(Hexserror::validation("Edge target node_id cannot be 0"));
        }

        if self.edge.source == self.edge.target {
            return Err(Hexserror::validation("Edge cannot connect node to itself"));
        }

        Ok(())
    }
}

/// Handler for CreateEdge directive.
pub struct CreateEdgeHandler {
    repository: Arc<dyn GraphRepository>,
    event_publisher: Arc<dyn DomainEventPublisher>,
}

impl CreateEdgeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event_publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event_publisher,
        }
    }
}

impl DirectiveHandler<CreateEdge> for CreateEdgeHandler {
    fn handle(&self, directive: CreateEdge) -> HexResult<()> {
        log::info!(
            "Executing CreateEdge directive: id={}, source={}, target={}",
            directive.edge.id,
            directive.edge.source,
            directive.edge.target
        );

        // 1. Validate directive
        directive.validate()?;

        // 2. Execute domain logic
        let edge = directive.edge.clone();

        // 3. Persist via repository
        let repository = self.repository.clone();
        let event_publisher = self.event_publisher.clone();

        tokio::runtime::Handle::current().block_on(async move {
            repository
                .add_edges(vec![edge.clone()])
                .await
                .map_err(|e| {
                    Hexserror::adapter(
                        "E_GRAPH_CREATE_003",
                        &format!("Failed to create edge: {}", e),
                    )
                })?;

            // 4. Emit domain event
            let event = GraphEvent::EdgeCreated {
                edge_id: edge.id.clone(),
                source: edge.source,
                target: edge.target,
                edge_type: edge.edge_type.clone(),
                timestamp: chrono::Utc::now(),
            };

            event_publisher.publish(event).map_err(|e| {
                Hexserror::adapter(
                    "E_GRAPH_CREATE_004",
                    &format!("Failed to publish EdgeCreated event: {}", e),
                )
            })?;

            log::info!("✅ Edge created successfully: id={}", edge.id);

            Ok(())
        })
    }
}

// ============================================================================
// UPDATE NODE POSITION
// ============================================================================

/// Directive to update a single node's position.
#[derive(Debug, Clone)]
pub struct UpdateNodePosition {
    pub node_id: u32,
    pub position: (f32, f32, f32),
    pub source: UpdateSource,
}

/// Source of position update (for event context).
#[derive(Debug, Clone)]
pub enum UpdateSource {
    UserInteraction,
    PhysicsSimulation,
    GitHubSync,
    SemanticAnalysis,
}

impl Directive for UpdateNodePosition {
    fn validate(&self) -> HexResult<()> {
        if self.node_id == 0 {
            return Err(Hexserror::validation("Node ID cannot be 0"));
        }

        // Validate position values are not NaN or infinite
        if !self.position.0.is_finite()
            || !self.position.1.is_finite()
            || !self.position.2.is_finite()
        {
            return Err(Hexserror::validation("Position values must be finite"));
        }

        Ok(())
    }
}

/// Handler for UpdateNodePosition directive.
pub struct UpdateNodePositionHandler {
    repository: Arc<dyn GraphRepository>,
    event_publisher: Arc<dyn DomainEventPublisher>,
}

impl UpdateNodePositionHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event_publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event_publisher,
        }
    }
}

impl DirectiveHandler<UpdateNodePosition> for UpdateNodePositionHandler {
    fn handle(&self, directive: UpdateNodePosition) -> HexResult<()> {
        log::debug!(
            "Executing UpdateNodePosition directive: node_id={}, position={:?}",
            directive.node_id,
            directive.position
        );

        // 1. Validate directive
        directive.validate()?;

        // 2. Persist via repository
        let repository = self.repository.clone();
        let event_publisher = self.event_publisher.clone();
        let node_id = directive.node_id;
        let position = directive.position;
        let source = directive.source.clone();

        tokio::runtime::Handle::current().block_on(async move {
            repository
                .update_positions(vec![(node_id, position)])
                .await
                .map_err(|e| {
                    Hexserror::adapter(
                        "E_GRAPH_UPDATE_001",
                        &format!("Failed to update node position: {}", e),
                    )
                })?;

            // 3. Emit domain event
            let event = GraphEvent::NodePositionChanged {
                node_id,
                new_position: position,
                source,
                timestamp: chrono::Utc::now(),
            };

            event_publisher.publish(event).map_err(|e| {
                Hexserror::adapter(
                    "E_GRAPH_UPDATE_002",
                    &format!("Failed to publish NodePositionChanged event: {}", e),
                )
            })?;

            Ok(())
        })
    }
}

// ============================================================================
// BATCH UPDATE POSITIONS
// ============================================================================

/// Directive to update multiple node positions at once (physics simulation).
#[derive(Debug, Clone)]
pub struct BatchUpdatePositions {
    pub updates: Vec<(u32, (f32, f32, f32))>,
    pub source: UpdateSource,
}

impl Directive for BatchUpdatePositions {
    fn validate(&self) -> HexResult<()> {
        if self.updates.is_empty() {
            return Err(Hexserror::validation("Cannot update empty position list"));
        }

        // Validate all positions
        for (node_id, position) in &self.updates {
            if *node_id == 0 {
                return Err(Hexserror::validation("Node ID cannot be 0"));
            }

            if !position.0.is_finite() || !position.1.is_finite() || !position.2.is_finite() {
                return Err(Hexserror::validation(&format!(
                    "Position values must be finite for node {}",
                    node_id
                )));
            }
        }

        Ok(())
    }
}

/// Handler for BatchUpdatePositions directive.
pub struct BatchUpdatePositionsHandler {
    repository: Arc<dyn GraphRepository>,
    event_publisher: Arc<dyn DomainEventPublisher>,
}

impl BatchUpdatePositionsHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event_publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event_publisher,
        }
    }
}

impl DirectiveHandler<BatchUpdatePositions> for BatchUpdatePositionsHandler {
    fn handle(&self, directive: BatchUpdatePositions) -> HexResult<()> {
        log::debug!(
            "Executing BatchUpdatePositions directive: {} nodes",
            directive.updates.len()
        );

        // 1. Validate directive
        directive.validate()?;

        // 2. Persist via repository
        let repository = self.repository.clone();
        let event_publisher = self.event_publisher.clone();
        let updates = directive.updates.clone();
        let source = directive.source.clone();

        tokio::runtime::Handle::current().block_on(async move {
            repository.update_positions(updates.clone()).await.map_err(|e| {
                Hexserror::adapter(
                    "E_GRAPH_BATCH_001",
                    &format!("Failed to batch update positions: {}", e),
                )
            })?;

            // 3. Emit domain event
            let node_ids: Vec<u32> = updates.iter().map(|(id, _)| *id).collect();

            let event = GraphEvent::PositionsUpdated {
                node_ids,
                count: updates.len(),
                source,
                timestamp: chrono::Utc::now(),
            };

            event_publisher.publish(event).map_err(|e| {
                Hexserror::adapter(
                    "E_GRAPH_BATCH_002",
                    &format!("Failed to publish PositionsUpdated event: {}", e),
                )
            })?;

            log::debug!("✅ Batch positions updated: {} nodes", updates.len());

            Ok(())
        })
    }
}

// ============================================================================
// DELETE NODE
// ============================================================================

/// Directive to delete a node from the graph.
#[derive(Debug, Clone)]
pub struct DeleteNode {
    pub node_id: u32,
}

impl Directive for DeleteNode {
    fn validate(&self) -> HexResult<()> {
        if self.node_id == 0 {
            return Err(Hexserror::validation("Node ID cannot be 0"));
        }

        Ok(())
    }
}

/// Handler for DeleteNode directive.
pub struct DeleteNodeHandler {
    repository: Arc<dyn GraphRepository>,
    event_publisher: Arc<dyn DomainEventPublisher>,
}

impl DeleteNodeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event_publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event_publisher,
        }
    }
}

impl DirectiveHandler<DeleteNode> for DeleteNodeHandler {
    fn handle(&self, directive: DeleteNode) -> HexResult<()> {
        log::info!("Executing DeleteNode directive: node_id={}", directive.node_id);

        // 1. Validate directive
        directive.validate()?;

        // 2. Delete via repository
        let repository = self.repository.clone();
        let event_publisher = self.event_publisher.clone();
        let node_id = directive.node_id;

        tokio::runtime::Handle::current().block_on(async move {
            // Note: Repository needs delete_node method
            // This is a placeholder - actual implementation depends on repository trait
            // repository.delete_node(node_id).await.map_err(|e| {
            //     Hexserror::adapter(
            //         "E_GRAPH_DELETE_001",
            //         &format!("Failed to delete node: {}", e),
            //     )
            // })?;

            // 3. Emit domain event
            let event = GraphEvent::NodeDeleted {
                node_id,
                timestamp: chrono::Utc::now(),
            };

            event_publisher.publish(event).map_err(|e| {
                Hexserror::adapter(
                    "E_GRAPH_DELETE_002",
                    &format!("Failed to publish NodeDeleted event: {}", e),
                )
            })?;

            log::info!("✅ Node deleted successfully: id={}", node_id);

            Ok(())
        })
    }
}

// ============================================================================
// DELETE EDGE
// ============================================================================

/// Directive to delete an edge from the graph.
#[derive(Debug, Clone)]
pub struct DeleteEdge {
    pub edge_id: String,
}

impl Directive for DeleteEdge {
    fn validate(&self) -> HexResult<()> {
        if self.edge_id.is_empty() {
            return Err(Hexserror::validation("Edge ID cannot be empty"));
        }

        Ok(())
    }
}

/// Handler for DeleteEdge directive.
pub struct DeleteEdgeHandler {
    repository: Arc<dyn GraphRepository>,
    event_publisher: Arc<dyn DomainEventPublisher>,
}

impl DeleteEdgeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event_publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event_publisher,
        }
    }
}

impl DirectiveHandler<DeleteEdge> for DeleteEdgeHandler {
    fn handle(&self, directive: DeleteEdge) -> HexResult<()> {
        log::info!("Executing DeleteEdge directive: edge_id={}", directive.edge_id);

        // 1. Validate directive
        directive.validate()?;

        // 2. Delete via repository
        let repository = self.repository.clone();
        let event_publisher = self.event_publisher.clone();
        let edge_id = directive.edge_id.clone();

        tokio::runtime::Handle::current().block_on(async move {
            // Note: Repository needs delete_edge method
            // repository.delete_edge(&edge_id).await.map_err(|e| {
            //     Hexserror::adapter(
            //         "E_GRAPH_DELETE_003",
            //         &format!("Failed to delete edge: {}", e),
            //     )
            // })?;

            // 3. Emit domain event
            let event = GraphEvent::EdgeDeleted {
                edge_id: edge_id.clone(),
                timestamp: chrono::Utc::now(),
            };

            event_publisher.publish(event).map_err(|e| {
                Hexserror::adapter(
                    "E_GRAPH_DELETE_004",
                    &format!("Failed to publish EdgeDeleted event: {}", e),
                )
            })?;

            log::info!("✅ Edge deleted successfully: id={}", edge_id);

            Ok(())
        })
    }
}

// ============================================================================
// EXPORTS
// ============================================================================

// Re-export for convenience
pub use self::{
    BatchUpdatePositions, BatchUpdatePositionsHandler, CreateEdge, CreateEdgeHandler, CreateNode,
    CreateNodeHandler, DeleteEdge, DeleteEdgeHandler, DeleteNode, DeleteNodeHandler,
    UpdateNodePosition, UpdateNodePositionHandler, UpdateSource,
};
```

---

## Domain Events

### File: `src/application/events.rs` (Enhance Existing)

```rust
//! Domain events for graph operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Update source for tracking event origin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateSource {
    UserInteraction,
    PhysicsSimulation,
    GitHubSync,
    SemanticAnalysis,
}

/// Graph domain events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEvent {
    /// Node was created
    NodeCreated {
        node_id: u32,
        metadata_id: String,
        label: String,
        timestamp: DateTime<Utc>,
    },

    /// Node was deleted
    NodeDeleted {
        node_id: u32,
        timestamp: DateTime<Utc>,
    },

    /// Edge was created
    EdgeCreated {
        edge_id: String,
        source: u32,
        target: u32,
        edge_type: String,
        timestamp: DateTime<Utc>,
    },

    /// Edge was deleted
    EdgeDeleted {
        edge_id: String,
        timestamp: DateTime<Utc>,
    },

    /// Single node position changed
    NodePositionChanged {
        node_id: u32,
        new_position: (f32, f32, f32),
        source: UpdateSource,
        timestamp: DateTime<Utc>,
    },

    /// Multiple node positions updated (batch)
    PositionsUpdated {
        node_ids: Vec<u32>,
        count: usize,
        source: UpdateSource,
        timestamp: DateTime<Utc>,
    },

    /// ⭐ CRITICAL FOR CACHE BUG FIX: GitHub sync completed
    GraphSyncCompleted {
        total_nodes: usize,
        total_edges: usize,
        timestamp: DateTime<Utc>,
    },
}

/// Domain event publisher trait
pub trait DomainEventPublisher: Send + Sync {
    fn publish(&self, event: GraphEvent) -> Result<(), String>;
}

/// Domain event subscriber trait
pub trait DomainEventSubscriber: Send + Sync {
    fn on_event(&self, event: &GraphEvent) -> Result<(), String>;
}
```

---

## HTTP Handler Integration

### File: `src/handlers/api_handler/graph/mod.rs` (Update)

```rust
// BEFORE (uses actor messages)
pub async fn upload_and_process(
    state: web::Data<AppState>,
    // ...
) -> impl Responder {
    // ❌ Old way: Send actor message
    let result = state
        .graph_service_actor
        .send(AddNodesFromMetadata { metadata })
        .await??;

    HttpResponse::Ok().json(result)
}

// AFTER (uses CQRS directive handlers)
pub async fn upload_and_process(
    state: web::Data<AppState>,
    // ...
) -> impl Responder {
    // ✅ New way: Use directive handler
    let handler = state.graph_directive_handlers.create_node.clone();

    // Build nodes from metadata
    let nodes = build_nodes_from_metadata(&metadata)?;

    // Execute directive for each node
    for node in nodes {
        let directive = CreateNode { node };

        execute_in_thread(move || handler.handle(directive))
            .await
            .map_err(|e| {
                error!("Failed to create node: {}", e);
                actix_web::error::ErrorInternalServerError(e)
            })??;
    }

    HttpResponse::Ok().json(json!({"success": true}))
}
```

---

## AppState Wiring

### File: `src/app_state.rs` (Add Directive Handlers)

```rust
use crate::application::graph::directives::{
    CreateNodeHandler, CreateEdgeHandler, UpdateNodePositionHandler,
    BatchUpdatePositionsHandler, DeleteNodeHandler, DeleteEdgeHandler,
};

pub struct GraphDirectiveHandlers {
    pub create_node: Arc<CreateNodeHandler>,
    pub create_edge: Arc<CreateEdgeHandler>,
    pub update_position: Arc<UpdateNodePositionHandler>,
    pub batch_update_positions: Arc<BatchUpdatePositionsHandler>,
    pub delete_node: Arc<DeleteNodeHandler>,
    pub delete_edge: Arc<DeleteEdgeHandler>,
}

pub struct AppState {
    pub graph_query_handlers: GraphQueryHandlers,  // ✅ Already exists
    pub graph_directive_handlers: GraphDirectiveHandlers,  // 🔧 Add this
    // ...
}

// In initialization
let unified_graph_repo = Arc::new(UnifiedGraphRepository::new(&db_path));
let event_publisher = Arc::new(InMemoryEventBus::new());  // Simple event bus

let graph_directive_handlers = GraphDirectiveHandlers {
    create_node: Arc::new(CreateNodeHandler::new(
        unified_graph_repo.clone(),
        event_publisher.clone(),
    )),
    create_edge: Arc::new(CreateEdgeHandler::new(
        unified_graph_repo.clone(),
        event_publisher.clone(),
    )),
    update_position: Arc::new(UpdateNodePositionHandler::new(
        unified_graph_repo.clone(),
        event_publisher.clone(),
    )),
    batch_update_positions: Arc::new(BatchUpdatePositionsHandler::new(
        unified_graph_repo.clone(),
        event_publisher.clone(),
    )),
    delete_node: Arc::new(DeleteNodeHandler::new(
        unified_graph_repo.clone(),
        event_publisher.clone(),
    )),
    delete_edge: Arc::new(DeleteEdgeHandler::new(
        unified_graph_repo.clone(),
        event_publisher.clone(),
    )),
};
```

---

## Testing Template

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::repositories::unified_graph_repository::UnifiedGraphRepository;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_create_node_directive() {
        // Arrange
        let db_path = ":memory:";  // In-memory SQLite for testing
        let repo = Arc::new(UnifiedGraphRepository::new(db_path));
        let event_bus = Arc::new(MockEventPublisher::new());

        let handler = CreateNodeHandler::new(repo.clone(), event_bus.clone());

        let directive = CreateNode {
            node: Node {
                id: 1,
                metadata_id: "test-node".to_string(),
                label: "Test Node".to_string(),
                data: Default::default(),
                ..Default::default()
            },
        };

        // Act
        let result = handler.handle(directive);

        // Assert
        assert!(result.is_ok());

        // Verify node was persisted
        let graph = repo.get_graph().await.unwrap();
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].id, 1);

        // Verify event was published
        let events = event_bus.get_published_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            GraphEvent::NodeCreated { node_id: 1, .. }
        ));
    }
}
```

---

**This template provides**:
- ✅ Complete directive handler implementations
- ✅ Domain event definitions
- ✅ HTTP handler integration examples
- ✅ AppState wiring
- ✅ Testing patterns

**Copy this template to create**: `src/application/graph/directives.rs`
