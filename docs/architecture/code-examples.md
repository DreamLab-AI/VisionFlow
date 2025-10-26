# Code Examples for Hexagonal/CQRS Architecture
**Production-Ready Code Samples**

---

## Table of Contents
1. [Query Handler Examples](#query-handler-examples)
2. [Command Handler Examples](#command-handler-examples)
3. [Event Definitions](#event-definitions)
4. [Event Bus Implementation](#event-bus-implementation)
5. [Repository Implementations](#repository-implementations)
6. [WebSocket Integration](#websocket-integration)
7. [API Handler Examples](#api-handler-examples)
8. [Testing Examples](#testing-examples)

---

## Query Handler Examples

### GetGraphDataQueryHandler
```rust
// src/application/graph/query_handlers.rs

use crate::ports::graph_repository::GraphRepository;
use crate::application::graph::queries::{GetGraphDataQuery, GraphFilter};
use crate::models::graph::GraphData;
use std::sync::Arc;
use async_trait::async_trait;

pub struct GetGraphDataQueryHandler {
    graph_repo: Arc<dyn GraphRepository>,
}

impl GetGraphDataQueryHandler {
    pub fn new(graph_repo: Arc<dyn GraphRepository>) -> Self {
        Self { graph_repo }
    }

    /// Handle GetGraphDataQuery
    ///
    /// This always reads fresh data from the database (no stale cache!)
    pub async fn handle(&self, query: GetGraphDataQuery) -> Result<GraphData, String> {
        log::debug!("Handling GetGraphDataQuery: include_edges={}", query.include_edges);

        // 1. Read from repository (ALWAYS fresh from SQLite!)
        let mut graph_data = self.graph_repo.get_graph().await
            .map_err(|e| format!("Failed to load graph: {}", e))?;

        // 2. Apply optional filters
        if let Some(filter) = query.filter {
            graph_data = self.apply_filter(graph_data, filter)?;
        }

        // 3. Optionally exclude edges for performance
        if !query.include_edges {
            log::debug!("Excluding edges from response");
            graph_data.edges.clear();
        }

        log::info!("Returning graph data: {} nodes, {} edges",
            graph_data.nodes.len(),
            graph_data.edges.len()
        );

        // 4. Return DTO
        Ok(graph_data)
    }

    fn apply_filter(&self, mut graph: GraphData, filter: GraphFilter) -> Result<GraphData, String> {
        match filter {
            GraphFilter::NodeLabelContains(substring) => {
                graph.nodes.retain(|_, node| node.label.contains(&substring));
                Ok(graph)
            },
            GraphFilter::NodeIdsIn(ids) => {
                graph.nodes.retain(|id, _| ids.contains(id));
                Ok(graph)
            },
            GraphFilter::MetadataIdEquals(metadata_id) => {
                graph.nodes.retain(|_, node| {
                    node.metadata_id.as_ref().map(|m| m == &metadata_id).unwrap_or(false)
                });
                Ok(graph)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::mocks::MockGraphRepository;

    #[tokio::test]
    async fn test_get_graph_data_no_filter() {
        let mock_repo = Arc::new(MockGraphRepository::with_nodes(10));
        let handler = GetGraphDataQueryHandler::new(mock_repo);

        let query = GetGraphDataQuery {
            include_edges: true,
            filter: None,
        };

        let result = handler.handle(query).await.unwrap();
        assert_eq!(result.nodes.len(), 10);
    }

    #[tokio::test]
    async fn test_get_graph_data_with_filter() {
        let mock_repo = Arc::new(MockGraphRepository::with_nodes(10));
        let handler = GetGraphDataQueryHandler::new(mock_repo);

        let query = GetGraphDataQuery {
            include_edges: true,
            filter: Some(GraphFilter::NodeLabelContains("test".to_string())),
        };

        let result = handler.handle(query).await.unwrap();
        assert!(result.nodes.len() <= 10);
    }
}
```

### GetNodeByIdQueryHandler
```rust
// src/application/graph/query_handlers.rs

use crate::ports::graph_repository::GraphRepository;
use crate::application::graph::queries::GetNodeByIdQuery;
use crate::models::node::Node;
use std::sync::Arc;

pub struct GetNodeByIdQueryHandler {
    graph_repo: Arc<dyn GraphRepository>,
}

impl GetNodeByIdQueryHandler {
    pub fn new(graph_repo: Arc<dyn GraphRepository>) -> Self {
        Self { graph_repo }
    }

    pub async fn handle(&self, query: GetNodeByIdQuery) -> Result<Option<Node>, String> {
        log::debug!("Fetching node by ID: {}", query.node_id);

        let node = self.graph_repo.get_node(query.node_id).await
            .map_err(|e| format!("Failed to get node: {}", e))?;

        if let Some(ref n) = node {
            log::debug!("Found node: {} (label: {})", n.id, n.label);
        } else {
            log::debug!("Node {} not found", query.node_id);
        }

        Ok(node)
    }
}
```

---

## Command Handler Examples

### CreateNodeCommandHandler
```rust
// src/application/graph/command_handlers.rs

use crate::ports::graph_repository::GraphRepository;
use crate::infrastructure::event_bus::EventBus;
use crate::domain::events::{GraphEvent, UpdateSource};
use crate::application::graph::commands::CreateNodeCommand;
use crate::models::node::Node;
use std::sync::Arc;
use chrono::Utc;

pub struct CreateNodeCommandHandler {
    graph_repo: Arc<dyn GraphRepository>,
    event_bus: Arc<dyn EventBus>,
}

impl CreateNodeCommandHandler {
    pub fn new(
        graph_repo: Arc<dyn GraphRepository>,
        event_bus: Arc<dyn EventBus>,
    ) -> Self {
        Self { graph_repo, event_bus }
    }

    pub async fn handle(&self, cmd: CreateNodeCommand) -> Result<u32, String> {
        log::info!("Creating node: {} (label: {})", cmd.node_id, cmd.label);

        // 1. Validate command
        self.validate(&cmd)?;

        // 2. Create domain entity
        let node = Node {
            id: cmd.node_id,
            label: cmd.label.clone(),
            position: cmd.position,
            metadata_id: cmd.metadata_id.clone(),
            velocity: (0.0, 0.0, 0.0),
            properties: std::collections::HashMap::new(),
        };

        // 3. Persist via repository (write to SQLite)
        self.graph_repo.add_node(node.clone()).await
            .map_err(|e| format!("Failed to persist node: {}", e))?;

        log::debug!("Node {} persisted to database", node.id);

        // 4. Emit domain event (event sourcing!)
        let event = GraphEvent::NodeCreated {
            node_id: node.id,
            label: node.label.clone(),
            timestamp: Utc::now(),
            source: UpdateSource::UserInteraction,
        };

        self.event_bus.publish(event).await
            .map_err(|e| format!("Failed to publish event: {}", e))?;

        log::info!("✅ Node {} created successfully", node.id);

        Ok(node.id)
    }

    fn validate(&self, cmd: &CreateNodeCommand) -> Result<(), String> {
        if cmd.label.is_empty() {
            return Err("Node label cannot be empty".to_string());
        }

        if cmd.label.len() > 255 {
            return Err("Node label too long (max 255 characters)".to_string());
        }

        // Validate position (no NaN or infinity)
        if cmd.position.0.is_nan() || cmd.position.0.is_infinite() {
            return Err("Invalid position: x is NaN or infinite".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::mocks::{MockGraphRepository, MockEventBus};

    #[tokio::test]
    async fn test_create_node_success() {
        let mock_repo = Arc::new(MockGraphRepository::new());
        let mock_bus = Arc::new(MockEventBus::new());
        let handler = CreateNodeCommandHandler::new(mock_repo.clone(), mock_bus.clone());

        let cmd = CreateNodeCommand {
            node_id: 1,
            label: "Test Node".to_string(),
            position: (0.0, 0.0, 0.0),
            metadata_id: None,
        };

        let result = handler.handle(cmd).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        assert_eq!(mock_repo.add_node_calls(), 1);
        assert_eq!(mock_bus.published_events().len(), 1);
    }

    #[tokio::test]
    async fn test_create_node_empty_label() {
        let mock_repo = Arc::new(MockGraphRepository::new());
        let mock_bus = Arc::new(MockEventBus::new());
        let handler = CreateNodeCommandHandler::new(mock_repo, mock_bus);

        let cmd = CreateNodeCommand {
            node_id: 1,
            label: "".to_string(),
            position: (0.0, 0.0, 0.0),
            metadata_id: None,
        };

        let result = handler.handle(cmd).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Node label cannot be empty");
    }
}
```

### UpdateNodePositionCommandHandler
```rust
// src/application/graph/command_handlers.rs

use crate::ports::graph_repository::GraphRepository;
use crate::infrastructure::event_bus::EventBus;
use crate::domain::events::{GraphEvent, UpdateSource};
use crate::application::graph::commands::UpdateNodePositionCommand;
use std::sync::Arc;
use chrono::Utc;

pub struct UpdateNodePositionCommandHandler {
    graph_repo: Arc<dyn GraphRepository>,
    event_bus: Arc<dyn EventBus>,
}

impl UpdateNodePositionCommandHandler {
    pub fn new(
        graph_repo: Arc<dyn GraphRepository>,
        event_bus: Arc<dyn EventBus>,
    ) -> Self {
        Self { graph_repo, event_bus }
    }

    pub async fn handle(&self, cmd: UpdateNodePositionCommand) -> Result<(), String> {
        log::debug!("Updating position for node {}: {:?}", cmd.node_id, cmd.new_position);

        // 1. Get current node (for old position)
        let old_node = self.graph_repo.get_node(cmd.node_id).await?
            .ok_or_else(|| format!("Node {} not found", cmd.node_id))?;

        // 2. Update position in repository
        self.graph_repo.update_node_position(cmd.node_id, cmd.new_position).await?;

        // 3. Emit event
        let event = GraphEvent::NodePositionChanged {
            node_id: cmd.node_id,
            old_position: old_node.position,
            new_position: cmd.new_position,
            timestamp: Utc::now(),
            source: cmd.source,
        };

        self.event_bus.publish(event).await?;

        Ok(())
    }
}
```

---

## Event Definitions

### Complete Event System
```rust
// src/domain/events.rs

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Update source for tracking event origin
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UpdateSource {
    UserInteraction,
    PhysicsSimulation,
    GitHubSync,
    SemanticAnalysis,
    SystemInternal,
}

/// Domain events for graph operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphEvent {
    /// Node was created
    NodeCreated {
        node_id: u32,
        label: String,
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },

    /// Node was updated
    NodeUpdated {
        node_id: u32,
        old_label: String,
        new_label: String,
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },

    /// Node position changed
    NodePositionChanged {
        node_id: u32,
        old_position: (f32, f32, f32),
        new_position: (f32, f32, f32),
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },

    /// Node was deleted
    NodeDeleted {
        node_id: u32,
        timestamp: DateTime<Utc>,
        source: UpdateSource,
    },

    /// Edge was created
    EdgeCreated {
        edge_id: String,
        source_id: u32,
        target_id: u32,
        timestamp: DateTime<Utc>,
    },

    /// Edge was deleted
    EdgeDeleted {
        edge_id: String,
        timestamp: DateTime<Utc>,
    },

    /// Physics simulation step completed
    PhysicsStepCompleted {
        iteration: usize,
        nodes_updated: usize,
        timestamp: DateTime<Utc>,
    },

    /// ⭐ CRITICAL: GitHub sync completed (fixes cache bug!)
    GitHubSyncCompleted {
        total_nodes: usize,
        total_edges: usize,
        kg_files: usize,
        ontology_files: usize,
        timestamp: DateTime<Utc>,
    },

    /// WebSocket client connected
    WebSocketClientConnected {
        client_id: String,
        timestamp: DateTime<Utc>,
    },

    /// WebSocket client disconnected
    WebSocketClientDisconnected {
        client_id: String,
        timestamp: DateTime<Utc>,
    },

    /// Semantic analysis completed
    SemanticAnalysisCompleted {
        constraints_generated: usize,
        communities_detected: usize,
        timestamp: DateTime<Utc>,
    },
}

impl GraphEvent {
    /// Get event type as string
    pub fn event_type(&self) -> &str {
        match self {
            GraphEvent::NodeCreated { .. } => "NodeCreated",
            GraphEvent::NodeUpdated { .. } => "NodeUpdated",
            GraphEvent::NodePositionChanged { .. } => "NodePositionChanged",
            GraphEvent::NodeDeleted { .. } => "NodeDeleted",
            GraphEvent::EdgeCreated { .. } => "EdgeCreated",
            GraphEvent::EdgeDeleted { .. } => "EdgeDeleted",
            GraphEvent::PhysicsStepCompleted { .. } => "PhysicsStepCompleted",
            GraphEvent::GitHubSyncCompleted { .. } => "GitHubSyncCompleted",
            GraphEvent::WebSocketClientConnected { .. } => "WebSocketClientConnected",
            GraphEvent::WebSocketClientDisconnected { .. } => "WebSocketClientDisconnected",
            GraphEvent::SemanticAnalysisCompleted { .. } => "SemanticAnalysisCompleted",
        }
    }

    /// Get event timestamp
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            GraphEvent::NodeCreated { timestamp, .. } => *timestamp,
            GraphEvent::NodeUpdated { timestamp, .. } => *timestamp,
            GraphEvent::NodePositionChanged { timestamp, .. } => *timestamp,
            GraphEvent::NodeDeleted { timestamp, .. } => *timestamp,
            GraphEvent::EdgeCreated { timestamp, .. } => *timestamp,
            GraphEvent::EdgeDeleted { timestamp, .. } => *timestamp,
            GraphEvent::PhysicsStepCompleted { timestamp, .. } => *timestamp,
            GraphEvent::GitHubSyncCompleted { timestamp, .. } => *timestamp,
            GraphEvent::WebSocketClientConnected { timestamp, .. } => *timestamp,
            GraphEvent::WebSocketClientDisconnected { timestamp, .. } => *timestamp,
            GraphEvent::SemanticAnalysisCompleted { timestamp, .. } => *timestamp,
        }
    }

    /// Generate unique event ID
    pub fn event_id(&self) -> String {
        format!("{}-{}", self.event_type(), uuid::Uuid::new_v4())
    }
}
```

---

## Event Bus Implementation

### In-Memory Event Bus
```rust
// src/infrastructure/event_bus.rs

use crate::domain::events::GraphEvent;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;

#[async_trait]
pub trait EventBus: Send + Sync {
    /// Publish event to all subscribers
    async fn publish(&self, event: GraphEvent) -> Result<(), String>;

    /// Subscribe to specific event types
    async fn subscribe(&self, event_type: &str, handler: Arc<dyn EventHandler>) -> Result<(), String>;

    /// Unsubscribe handler
    async fn unsubscribe(&self, event_type: &str, handler_id: &str) -> Result<(), String>;
}

#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Handle event
    async fn handle(&self, event: &GraphEvent) -> Result<(), String>;

    /// Get handler ID
    fn id(&self) -> String;
}

/// In-memory event bus implementation
pub struct InMemoryEventBus {
    subscribers: Arc<RwLock<HashMap<String, Vec<Arc<dyn EventHandler>>>>>,
}

impl InMemoryEventBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get subscriber count for event type
    pub fn subscriber_count(&self, event_type: &str) -> usize {
        let subscribers = self.subscribers.read().unwrap();
        subscribers.get(event_type).map(|v| v.len()).unwrap_or(0)
    }
}

#[async_trait]
impl EventBus for InMemoryEventBus {
    async fn publish(&self, event: GraphEvent) -> Result<(), String> {
        let event_type = event.event_type().to_string();
        log::debug!("Publishing event: {}", event_type);

        let subscribers = self.subscribers.read().unwrap();

        if let Some(handlers) = subscribers.get(&event_type) {
            log::debug!("Found {} subscribers for {}", handlers.len(), event_type);

            // Spawn handlers in parallel
            let mut join_handles = Vec::new();

            for handler in handlers {
                let handler_clone = handler.clone();
                let event_clone = event.clone();

                let handle = tokio::spawn(async move {
                    if let Err(e) = handler_clone.handle(&event_clone).await {
                        log::error!("Event handler {} failed: {}", handler_clone.id(), e);
                    }
                });

                join_handles.push(handle);
            }

            // Wait for all handlers to complete
            for handle in join_handles {
                let _ = handle.await;
            }
        } else {
            log::debug!("No subscribers for {}", event_type);
        }

        Ok(())
    }

    async fn subscribe(&self, event_type: &str, handler: Arc<dyn EventHandler>) -> Result<(), String> {
        let mut subscribers = self.subscribers.write().unwrap();
        subscribers.entry(event_type.to_string())
            .or_insert_with(Vec::new)
            .push(handler);

        log::info!("Subscriber added for event type: {}", event_type);
        Ok(())
    }

    async fn unsubscribe(&self, event_type: &str, handler_id: &str) -> Result<(), String> {
        let mut subscribers = self.subscribers.write().unwrap();

        if let Some(handlers) = subscribers.get_mut(event_type) {
            handlers.retain(|h| h.id() != handler_id);
            log::info!("Subscriber {} removed from {}", handler_id, event_type);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::events::{GraphEvent, UpdateSource};
    use chrono::Utc;

    struct TestEventHandler {
        id: String,
        handled: Arc<RwLock<Vec<String>>>,
    }

    #[async_trait]
    impl EventHandler for TestEventHandler {
        async fn handle(&self, event: &GraphEvent) -> Result<(), String> {
            self.handled.write().unwrap().push(event.event_type().to_string());
            Ok(())
        }

        fn id(&self) -> String {
            self.id.clone()
        }
    }

    #[tokio::test]
    async fn test_event_bus_publish_subscribe() {
        let event_bus = InMemoryEventBus::new();
        let handled = Arc::new(RwLock::new(Vec::new()));

        let handler = Arc::new(TestEventHandler {
            id: "test-handler".to_string(),
            handled: handled.clone(),
        });

        event_bus.subscribe("NodeCreated", handler).await.unwrap();

        let event = GraphEvent::NodeCreated {
            node_id: 1,
            label: "Test".to_string(),
            timestamp: Utc::now(),
            source: UpdateSource::UserInteraction,
        };

        event_bus.publish(event).await.unwrap();

        // Wait for async handler
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        let handled_events = handled.read().unwrap();
        assert_eq!(handled_events.len(), 1);
        assert_eq!(handled_events[0], "NodeCreated");
    }
}
```

---

## Repository Implementations

### SQLite Graph Repository
```rust
// src/adapters/sqlite_graph_repository.rs

use crate::ports::graph_repository::GraphRepository;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use rusqlite::{Connection, params};
use std::sync::Arc;
use async_trait::async_trait;

pub struct SqliteGraphRepository {
    db_path: String,
}

impl SqliteGraphRepository {
    pub fn new(db_path: &str) -> Self {
        Self {
            db_path: db_path.to_string(),
        }
    }

    fn open_connection(&self) -> Result<Connection, String> {
        Connection::open(&self.db_path)
            .map_err(|e| format!("Failed to open database: {}", e))
    }
}

#[async_trait]
impl GraphRepository for SqliteGraphRepository {
    async fn get_graph(&self) -> Result<GraphData, String> {
        let conn = self.open_connection()?;

        // Load nodes
        let mut stmt = conn.prepare("SELECT id, label, x, y, z, metadata_id FROM nodes")
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let nodes = stmt.query_map([], |row| {
            Ok(Node {
                id: row.get(0)?,
                label: row.get(1)?,
                position: (row.get(2)?, row.get(3)?, row.get(4)?),
                metadata_id: row.get(5)?,
                velocity: (0.0, 0.0, 0.0),
                properties: std::collections::HashMap::new(),
            })
        })
        .map_err(|e| format!("Failed to query nodes: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect nodes: {}", e))?;

        // Load edges
        let mut stmt = conn.prepare("SELECT id, source, target, strength FROM edges")
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let edges = stmt.query_map([], |row| {
            Ok(Edge {
                id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                strength: row.get(3)?,
            })
        })
        .map_err(|e| format!("Failed to query edges: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect edges: {}", e))?;

        let nodes_map = nodes.into_iter().map(|n| (n.id, n)).collect();

        Ok(GraphData {
            nodes: nodes_map,
            edges,
        })
    }

    async fn add_node(&self, node: Node) -> Result<(), String> {
        let conn = self.open_connection()?;

        conn.execute(
            "INSERT INTO nodes (id, label, x, y, z, metadata_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                node.id,
                node.label,
                node.position.0,
                node.position.1,
                node.position.2,
                node.metadata_id,
            ],
        )
        .map_err(|e| format!("Failed to insert node: {}", e))?;

        Ok(())
    }

    async fn get_node(&self, node_id: u32) -> Result<Option<Node>, String> {
        let conn = self.open_connection()?;

        let mut stmt = conn.prepare("SELECT id, label, x, y, z, metadata_id FROM nodes WHERE id = ?1")
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let node = stmt.query_row(params![node_id], |row| {
            Ok(Node {
                id: row.get(0)?,
                label: row.get(1)?,
                position: (row.get(2)?, row.get(3)?, row.get(4)?),
                metadata_id: row.get(5)?,
                velocity: (0.0, 0.0, 0.0),
                properties: std::collections::HashMap::new(),
            })
        })
        .optional()
        .map_err(|e| format!("Failed to query node: {}", e))?;

        Ok(node)
    }

    async fn update_node_position(&self, node_id: u32, position: (f32, f32, f32)) -> Result<(), String> {
        let conn = self.open_connection()?;

        conn.execute(
            "UPDATE nodes SET x = ?1, y = ?2, z = ?3 WHERE id = ?4",
            params![position.0, position.1, position.2, node_id],
        )
        .map_err(|e| format!("Failed to update position: {}", e))?;

        Ok(())
    }

    async fn batch_update_positions(&self, updates: Vec<(u32, (f32, f32, f32))>) -> Result<(), String> {
        let conn = self.open_connection()?;

        let tx = conn.transaction()
            .map_err(|e| format!("Failed to start transaction: {}", e))?;

        for (node_id, position) in updates {
            tx.execute(
                "UPDATE nodes SET x = ?1, y = ?2, z = ?3 WHERE id = ?4",
                params![position.0, position.1, position.2, node_id],
            )
            .map_err(|e| format!("Failed to update position: {}", e))?;
        }

        tx.commit()
            .map_err(|e| format!("Failed to commit transaction: {}", e))?;

        Ok(())
    }

    async fn add_edge(&self, edge: Edge) -> Result<(), String> {
        let conn = self.open_connection()?;

        conn.execute(
            "INSERT INTO edges (id, source, target, strength) VALUES (?1, ?2, ?3, ?4)",
            params![edge.id, edge.source, edge.target, edge.strength],
        )
        .map_err(|e| format!("Failed to insert edge: {}", e))?;

        Ok(())
    }

    async fn get_node_edges(&self, node_id: u32) -> Result<Vec<Edge>, String> {
        let conn = self.open_connection()?;

        let mut stmt = conn.prepare("SELECT id, source, target, strength FROM edges WHERE source = ?1 OR target = ?1")
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let edges = stmt.query_map(params![node_id], |row| {
            Ok(Edge {
                id: row.get(0)?,
                source: row.get(1)?,
                target: row.get(2)?,
                strength: row.get(3)?,
            })
        })
        .map_err(|e| format!("Failed to query edges: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect edges: {}", e))?;

        Ok(edges)
    }
}
```

---

## WebSocket Integration

### WebSocket Event Subscriber
```rust
// src/infrastructure/websocket_event_subscriber.rs

use crate::domain::events::GraphEvent;
use crate::infrastructure::event_bus::EventHandler;
use crate::ports::websocket_gateway::WebSocketGateway;
use std::sync::Arc;
use async_trait::async_trait;
use serde_json::json;

pub struct WebSocketEventSubscriber {
    ws_gateway: Arc<dyn WebSocketGateway>,
}

impl WebSocketEventSubscriber {
    pub fn new(ws_gateway: Arc<dyn WebSocketGateway>) -> Self {
        Self { ws_gateway }
    }
}

#[async_trait]
impl EventHandler for WebSocketEventSubscriber {
    async fn handle(&self, event: &GraphEvent) -> Result<(), String> {
        match event {
            GraphEvent::NodeCreated { node_id, label, .. } => {
                self.ws_gateway.broadcast(json!({
                    "type": "nodeCreated",
                    "nodeId": node_id,
                    "label": label,
                })).await?;
            },

            GraphEvent::NodePositionChanged { node_id, new_position, source, .. } => {
                self.ws_gateway.broadcast(json!({
                    "type": "nodePositionUpdate",
                    "nodeId": node_id,
                    "position": new_position,
                    "source": format!("{:?}", source),
                })).await?;
            },

            GraphEvent::PhysicsStepCompleted { iteration, nodes_updated, .. } => {
                self.ws_gateway.broadcast(json!({
                    "type": "physicsUpdate",
                    "iteration": iteration,
                    "nodesUpdated": nodes_updated,
                })).await?;
            },

            GraphEvent::GitHubSyncCompleted { total_nodes, total_edges, .. } => {
                // ⭐ THIS NOTIFIES CLIENTS AFTER GITHUB SYNC!
                self.ws_gateway.broadcast(json!({
                    "type": "graphReloaded",
                    "totalNodes": total_nodes,
                    "totalEdges": total_edges,
                    "message": "Graph data synchronized from GitHub",
                })).await?;
            },

            _ => {}
        }
        Ok(())
    }

    fn id(&self) -> String {
        "websocket-event-subscriber".to_string()
    }
}
```

---

## Testing Examples

### Unit Test Example
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::mocks::{MockGraphRepository, MockEventBus};
    use crate::application::graph::commands::CreateNodeCommand;

    #[tokio::test]
    async fn test_create_node_command_handler() {
        // Arrange
        let mock_repo = Arc::new(MockGraphRepository::new());
        let mock_bus = Arc::new(MockEventBus::new());
        let handler = CreateNodeCommandHandler::new(mock_repo.clone(), mock_bus.clone());

        let cmd = CreateNodeCommand {
            node_id: 1,
            label: "Test Node".to_string(),
            position: (10.0, 20.0, 0.0),
            metadata_id: Some("test-metadata".to_string()),
        };

        // Act
        let result = handler.handle(cmd).await;

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1);
        assert_eq!(mock_repo.add_node_calls(), 1);
        assert_eq!(mock_bus.published_events().len(), 1);

        let event = &mock_bus.published_events()[0];
        assert!(matches!(event, GraphEvent::NodeCreated { node_id: 1, .. }));
    }
}
```

---

**Code examples by**: Hive Mind Architecture Planner
**Date**: 2025-10-26
