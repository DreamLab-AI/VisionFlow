---
title: CQRS Directive Handler Template
description: **Copy-Paste Template for Creating Graph Directive Handlers**
category: explanation
tags:
  - architecture
  - backend
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: advanced
---


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
use crate::ports::graph-repository::GraphRepository;

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
        if self.node.metadata-id.is-empty() {
            return Err(Hexserror::validation("Node metadata-id cannot be empty"));
        }

        if self.node.label.is-empty() {
            return Err(Hexserror::validation("Node label cannot be empty"));
        }

        Ok(())
    }
}

/// Handler for CreateNode directive.
pub struct CreateNodeHandler {
    repository: Arc<dyn GraphRepository>,
    event-publisher: Arc<dyn DomainEventPublisher>,
}

impl CreateNodeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event-publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event-publisher,
        }
    }
}

impl DirectiveHandler<CreateNode> for CreateNodeHandler {
    fn handle(&self, directive: CreateNode) -> HexResult<()> {
        log::info!(
            "Executing CreateNode directive: metadata-id={}, label={}",
            directive.node.metadata-id,
            directive.node.label
        );

        // 1. Validate directive
        directive.validate()?;

        // 2. Execute domain logic
        let node = directive.node.clone();

        // 3. Persist via repository
        let repository = self.repository.clone();
        let event-publisher = self.event-publisher.clone();

        // Use tokio runtime for async repository call
        tokio::runtime::Handle::current().block-on(async move {
            repository
                .add-nodes(vec![node.clone()])
                .await
                .map-err(|e| {
                    Hexserror::adapter(
                        "E-GRAPH-CREATE-001",
                        &format!("Failed to create node: {}", e),
                    )
                })?;

            // 4. Emit domain event
            let event = GraphEvent::NodeCreated {
                node-id: node.id,
                metadata-id: node.metadata-id.clone(),
                label: node.label.clone(),
                timestamp: chrono::Utc::now(),
            };

            event-publisher.publish(event).map-err(|e| {
                Hexserror::adapter(
                    "E-GRAPH-CREATE-002",
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
        if self.edge.id.is-empty() {
            return Err(Hexserror::validation("Edge id cannot be empty"));
        }

        if self.edge.source == 0 {
            return Err(Hexserror::validation("Edge source node-id cannot be 0"));
        }

        if self.edge.target == 0 {
            return Err(Hexserror::validation("Edge target node-id cannot be 0"));
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
    event-publisher: Arc<dyn DomainEventPublisher>,
}

impl CreateEdgeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event-publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event-publisher,
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
        let event-publisher = self.event-publisher.clone();

        tokio::runtime::Handle::current().block-on(async move {
            repository
                .add-edges(vec![edge.clone()])
                .await
                .map-err(|e| {
                    Hexserror::adapter(
                        "E-GRAPH-CREATE-003",
                        &format!("Failed to create edge: {}", e),
                    )
                })?;

            // 4. Emit domain event
            let event = GraphEvent::EdgeCreated {
                edge-id: edge.id.clone(),
                source: edge.source,
                target: edge.target,
                edge-type: edge.edge-type.clone(),
                timestamp: chrono::Utc::now(),
            };

            event-publisher.publish(event).map-err(|e| {
                Hexserror::adapter(
                    "E-GRAPH-CREATE-004",
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
    pub node-id: u32,
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
        if self.node-id == 0 {
            return Err(Hexserror::validation("Node ID cannot be 0"));
        }

        // Validate position values are not NaN or infinite
        if !self.position.0.is-finite()
            || !self.position.1.is-finite()
            || !self.position.2.is-finite()
        {
            return Err(Hexserror::validation("Position values must be finite"));
        }

        Ok(())
    }
}

/// Handler for UpdateNodePosition directive.
pub struct UpdateNodePositionHandler {
    repository: Arc<dyn GraphRepository>,
    event-publisher: Arc<dyn DomainEventPublisher>,
}

impl UpdateNodePositionHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event-publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event-publisher,
        }
    }
}

impl DirectiveHandler<UpdateNodePosition> for UpdateNodePositionHandler {
    fn handle(&self, directive: UpdateNodePosition) -> HexResult<()> {
        log::debug!(
            "Executing UpdateNodePosition directive: node-id={}, position={:?}",
            directive.node-id,
            directive.position
        );

        // 1. Validate directive
        directive.validate()?;

        // 2. Persist via repository
        let repository = self.repository.clone();
        let event-publisher = self.event-publisher.clone();
        let node-id = directive.node-id;
        let position = directive.position;
        let source = directive.source.clone();

        tokio::runtime::Handle::current().block-on(async move {
            repository
                .update-positions(vec![(node-id, position)])
                .await
                .map-err(|e| {
                    Hexserror::adapter(
                        "E-GRAPH-UPDATE-001",
                        &format!("Failed to update node position: {}", e),
                    )
                })?;

            // 3. Emit domain event
            let event = GraphEvent::NodePositionChanged {
                node-id,
                new-position: position,
                source,
                timestamp: chrono::Utc::now(),
            };

            event-publisher.publish(event).map-err(|e| {
                Hexserror::adapter(
                    "E-GRAPH-UPDATE-002",
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
        if self.updates.is-empty() {
            return Err(Hexserror::validation("Cannot update empty position list"));
        }

        // Validate all positions
        for (node-id, position) in &self.updates {
            if *node-id == 0 {
                return Err(Hexserror::validation("Node ID cannot be 0"));
            }

            if !position.0.is-finite() || !position.1.is-finite() || !position.2.is-finite() {
                return Err(Hexserror::validation(&format!(
                    "Position values must be finite for node {}",
                    node-id
                )));
            }
        }

        Ok(())
    }
}

/// Handler for BatchUpdatePositions directive.
pub struct BatchUpdatePositionsHandler {
    repository: Arc<dyn GraphRepository>,
    event-publisher: Arc<dyn DomainEventPublisher>,
}

impl BatchUpdatePositionsHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event-publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event-publisher,
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
        let event-publisher = self.event-publisher.clone();
        let updates = directive.updates.clone();
        let source = directive.source.clone();

        tokio::runtime::Handle::current().block-on(async move {
            repository.update-positions(updates.clone()).await.map-err(|e| {
                Hexserror::adapter(
                    "E-GRAPH-BATCH-001",
                    &format!("Failed to batch update positions: {}", e),
                )
            })?;

            // 3. Emit domain event
            let node-ids: Vec<u32> = updates.iter().map(|(id, -)| *id).collect();

            let event = GraphEvent::PositionsUpdated {
                node-ids,
                count: updates.len(),
                source,
                timestamp: chrono::Utc::now(),
            };

            event-publisher.publish(event).map-err(|e| {
                Hexserror::adapter(
                    "E-GRAPH-BATCH-002",
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
    pub node-id: u32,
}

impl Directive for DeleteNode {
    fn validate(&self) -> HexResult<()> {
        if self.node-id == 0 {
            return Err(Hexserror::validation("Node ID cannot be 0"));
        }

        Ok(())
    }
}

/// Handler for DeleteNode directive.
pub struct DeleteNodeHandler {
    repository: Arc<dyn GraphRepository>,
    event-publisher: Arc<dyn DomainEventPublisher>,
}

impl DeleteNodeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event-publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event-publisher,
        }
    }
}

impl DirectiveHandler<DeleteNode> for DeleteNodeHandler {
    fn handle(&self, directive: DeleteNode) -> HexResult<()> {
        log::info!("Executing DeleteNode directive: node-id={}", directive.node-id);

        // 1. Validate directive
        directive.validate()?;

        // 2. Delete via repository
        let repository = self.repository.clone();
        let event-publisher = self.event-publisher.clone();
        let node-id = directive.node-id;

        tokio::runtime::Handle::current().block-on(async move {
            // Note: Repository needs delete-node method
            // This is a placeholder - actual implementation depends on repository trait
            // repository.delete-node(node-id).await.map-err(|e| {
            //     Hexserror::adapter(
            //         "E-GRAPH-DELETE-001",
            //         &format!("Failed to delete node: {}", e),
            //     )
            // })?;

            // 3. Emit domain event
            let event = GraphEvent::NodeDeleted {
                node-id,
                timestamp: chrono::Utc::now(),
            };

            event-publisher.publish(event).map-err(|e| {
                Hexserror::adapter(
                    "E-GRAPH-DELETE-002",
                    &format!("Failed to publish NodeDeleted event: {}", e),
                )
            })?;

            log::info!("✅ Node deleted successfully: id={}", node-id);

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
    pub edge-id: String,
}

impl Directive for DeleteEdge {
    fn validate(&self) -> HexResult<()> {
        if self.edge-id.is-empty() {
            return Err(Hexserror::validation("Edge ID cannot be empty"));
        }

        Ok(())
    }
}

/// Handler for DeleteEdge directive.
pub struct DeleteEdgeHandler {
    repository: Arc<dyn GraphRepository>,
    event-publisher: Arc<dyn DomainEventPublisher>,
}

impl DeleteEdgeHandler {
    pub fn new(
        repository: Arc<dyn GraphRepository>,
        event-publisher: Arc<dyn DomainEventPublisher>,
    ) -> Self {
        Self {
            repository,
            event-publisher,
        }
    }
}

impl DirectiveHandler<DeleteEdge> for DeleteEdgeHandler {
    fn handle(&self, directive: DeleteEdge) -> HexResult<()> {
        log::info!("Executing DeleteEdge directive: edge-id={}", directive.edge-id);

        // 1. Validate directive
        directive.validate()?;

        // 2. Delete via repository
        let repository = self.repository.clone();
        let event-publisher = self.event-publisher.clone();
        let edge-id = directive.edge-id.clone();

        tokio::runtime::Handle::current().block-on(async move {
            // Note: Repository needs delete-edge method
            // repository.delete-edge(&edge-id).await.map-err(|e| {
            //     Hexserror::adapter(
            //         "E-GRAPH-DELETE-003",
            //         &format!("Failed to delete edge: {}", e),
            //     )
            // })?;

            // 3. Emit domain event
            let event = GraphEvent::EdgeDeleted {
                edge-id: edge-id.clone(),
                timestamp: chrono::Utc::now(),
            };

            event-publisher.publish(event).map-err(|e| {
                Hexserror::adapter(
                    "E-GRAPH-DELETE-004",
                    &format!("Failed to publish EdgeDeleted event: {}", e),
                )
            })?;

            log::info!("✅ Edge deleted successfully: id={}", edge-id);

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
        node-id: u32,
        metadata-id: String,
        label: String,
        timestamp: DateTime<Utc>,
    },

    /// Node was deleted
    NodeDeleted {
        node-id: u32,
        timestamp: DateTime<Utc>,
    },

    /// Edge was created
    EdgeCreated {
        edge-id: String,
        source: u32,
        target: u32,
        edge-type: String,
        timestamp: DateTime<Utc>,
    },

    /// Edge was deleted
    EdgeDeleted {
        edge-id: String,
        timestamp: DateTime<Utc>,
    },

    /// Single node position changed
    NodePositionChanged {
        node-id: u32,
        new-position: (f32, f32, f32),
        source: UpdateSource,
        timestamp: DateTime<Utc>,
    },

    /// Multiple node positions updated (batch)
    PositionsUpdated {
        node-ids: Vec<u32>,
        count: usize,
        source: UpdateSource,
        timestamp: DateTime<Utc>,
    },

    /// ⭐ CRITICAL FOR CACHE BUG FIX: GitHub sync completed
    GraphSyncCompleted {
        total-nodes: usize,
        total-edges: usize,
        timestamp: DateTime<Utc>,
    },
}

/// Domain event publisher trait
pub trait DomainEventPublisher: Send + Sync {
    fn publish(&self, event: GraphEvent) -> Result<(), String>;
}

/// Domain event subscriber trait
pub trait DomainEventSubscriber: Send + Sync {
    fn on-event(&self, event: &GraphEvent) -> Result<(), String>;
}
```

---

## HTTP Handler Integration

### File: `src/handlers/api-handler/graph/mod.rs` (Update)

```rust
// BEFORE (uses actor messages)
pub async fn upload-and-process(
    state: web::Data<AppState>,
    // ...
) -> impl Responder {
    // ❌ Old way: Send actor message
    let result = state
        .graph-service-actor
        .send(AddNodesFromMetadata { metadata })
        .await??;

    HttpResponse::Ok().json(result)
}

// AFTER (uses CQRS directive handlers)
pub async fn upload-and-process(
    state: web::Data<AppState>,
    // ...
) -> impl Responder {
    // ✅ New way: Use directive handler
    let handler = state.graph-directive-handlers.create-node.clone();

    // Build nodes from metadata
    let nodes = build-nodes-from-metadata(&metadata)?;

    // Execute directive for each node
    for node in nodes {
        let directive = CreateNode { node };

        execute-in-thread(move || handler.handle(directive))
            .await
            .map-err(|e| {
                error!("Failed to create node: {}", e);
                actix-web::error::ErrorInternalServerError(e)
            })??;
    }

    HttpResponse::Ok().json(json!({"success": true}))
}
```

---

## AppState Wiring

### File: `src/app-state.rs` (Add Directive Handlers)

```rust
use crate::application::graph::directives::{
    CreateNodeHandler, CreateEdgeHandler, UpdateNodePositionHandler,
    BatchUpdatePositionsHandler, DeleteNodeHandler, DeleteEdgeHandler,
};

pub struct GraphDirectiveHandlers {
    pub create-node: Arc<CreateNodeHandler>,
    pub create-edge: Arc<CreateEdgeHandler>,
    pub update-position: Arc<UpdateNodePositionHandler>,
    pub batch-update-positions: Arc<BatchUpdatePositionsHandler>,
    pub delete-node: Arc<DeleteNodeHandler>,
    pub delete-edge: Arc<DeleteEdgeHandler>,
}

pub struct AppState {
    pub graph-query-handlers: GraphQueryHandlers,  // ✅ Already exists
    pub graph-directive-handlers: GraphDirectiveHandlers,  // 🔧 Add this
    // ...
}

// In initialization
let unified-graph-repo = Arc::new(UnifiedGraphRepository::new(&db-path));
let event-publisher = Arc::new(InMemoryEventBus::new());  // Simple event bus

let graph-directive-handlers = GraphDirectiveHandlers {
    create-node: Arc::new(CreateNodeHandler::new(
        unified-graph-repo.clone(),
        event-publisher.clone(),
    )),
    create-edge: Arc::new(CreateEdgeHandler::new(
        unified-graph-repo.clone(),
        event-publisher.clone(),
    )),
    update-position: Arc::new(UpdateNodePositionHandler::new(
        unified-graph-repo.clone(),
        event-publisher.clone(),
    )),
    batch-update-positions: Arc::new(BatchUpdatePositionsHandler::new(
        unified-graph-repo.clone(),
        event-publisher.clone(),
    )),
    delete-node: Arc::new(DeleteNodeHandler::new(
        unified-graph-repo.clone(),
        event-publisher.clone(),
    )),
    delete-edge: Arc::new(DeleteEdgeHandler::new(
        unified-graph-repo.clone(),
        event-publisher.clone(),
    )),
};
```

---

## Testing Template

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::repositories::unified-graph-repository::UnifiedGraphRepository;
    use std::sync::Arc;

    #[tokio::test]
    async fn test-create-node-directive() {
        // Arrange
        let db-path = ":memory:";  // In-memory SQLite for testing
        let repo = Arc::new(UnifiedGraphRepository::new(db-path));
        let event-bus = Arc::new(MockEventPublisher::new());

        let handler = CreateNodeHandler::new(repo.clone(), event-bus.clone());

        let directive = CreateNode {
            node: Node {
                id: 1,
                metadata-id: "test-node".to-string(),
                label: "Test Node".to-string(),
                data: Default::default(),
                ..Default::default()
            },
        };

        // Act
        let result = handler.handle(directive);

        // Assert
        assert!(result.is-ok());

        // Verify node was persisted
        let graph = repo.get-graph().await.unwrap();
        assert-eq!(graph.nodes.len(), 1);
        assert-eq!(graph.nodes[0].id, 1);

        // Verify event was published
        let events = event-bus.get-published-events();
        assert-eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            GraphEvent::NodeCreated { node-id: 1, .. }
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
