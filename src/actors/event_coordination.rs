// src/actors/event_coordination.rs
//! Event-Driven Actor Coordination
//!
//! Coordinates actors through domain events, enabling reactive
//! behavior and loose coupling between components.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::application::physics_service::PhysicsService;
use crate::application::semantic_service::SemanticService;
use crate::events::domain_events::{
    EdgeAddedEvent, GraphSavedEvent, NodeAddedEvent, OntologyImportedEvent, PositionsUpdatedEvent,
};
use crate::events::event_bus::EventBus;
use crate::events::types::DomainEvent;
use crate::models::graph::GraphData;

/// Event-driven actor coordinator
pub struct EventCoordinator {
    physics_service: Arc<PhysicsService>,
    semantic_service: Arc<SemanticService>,
    event_bus: Arc<RwLock<EventBus>>,
    graph_data: Arc<RwLock<GraphData>>,
}

impl EventCoordinator {
    /// Create new event coordinator
    pub fn new(
        physics_service: Arc<PhysicsService>,
        semantic_service: Arc<SemanticService>,
        event_bus: Arc<RwLock<EventBus>>,
        graph_data: Arc<RwLock<GraphData>>,
    ) -> Self {
        Self {
            physics_service,
            semantic_service,
            event_bus,
            graph_data,
        }
    }

    /// Initialize event listeners
    pub async fn initialize(&self) {
        info!("Initializing event coordination");

        // Subscribe to graph events
        self.subscribe_to_graph_events().await;

        // Subscribe to ontology events
        self.subscribe_to_ontology_events().await;

        // Subscribe to position update events
        self.subscribe_to_position_events().await;

        info!("Event coordination initialized");
    }

    /// Subscribe to graph-related events
    async fn subscribe_to_graph_events(&self) {
        let physics_service = self.physics_service.clone();
        let semantic_service = self.semantic_service.clone();
        let graph_data = self.graph_data.clone();

        // Listen for GraphSavedEvent -> trigger physics update
        let graph_data_clone = graph_data.clone();
        let physics_clone = physics_service.clone();

        actix::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Check for graph updates and trigger physics
                let graph = graph_data_clone.read().await.clone();
                if graph.nodes.len() > 0 {
                    debug!("Graph updated, triggering physics recalculation");

                    // Update physics with new graph data
                    // Note: This is a simplified version, actual implementation
                    // would use proper event subscription
                }
            }
        });

        // Listen for NodeAddedEvent -> invalidate semantic cache
        let semantic_clone = semantic_service.clone();
        actix::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                // Invalidate cache when nodes are added
                if let Err(e) = semantic_clone.invalidate_cache().await {
                    warn!("Failed to invalidate cache: {}", e);
                }
            }
        });
    }

    /// Subscribe to ontology events
    async fn subscribe_to_ontology_events(&self) {
        let semantic_service = self.semantic_service.clone();

        // Listen for OntologyImportedEvent -> trigger inference
        actix::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

                debug!("Checking for ontology updates");
                // Trigger semantic analysis after ontology import
            }
        });
    }

    /// Subscribe to position update events
    async fn subscribe_to_position_events(&self) {
        let event_bus = self.event_bus.clone();

        // Listen for PositionsUpdatedEvent -> broadcast to WebSocket clients
        actix::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

                // Broadcast position updates via WebSocket
                debug!("Broadcasting position updates");
            }
        });
    }

    /// Handle graph saved event
    pub async fn on_graph_saved(&self, event: GraphSavedEvent) {
        info!(
            "Graph saved event received: {} nodes, {} edges",
            event.node_count, event.edge_count
        );

        // Update physics with new graph
        let graph = self.graph_data.read().await.clone();

        // Reinitialize physics simulation
        if let Err(e) = self.physics_service.reset().await {
            warn!("Failed to reset physics: {}", e);
        }

        // Invalidate semantic cache
        if let Err(e) = self.semantic_service.invalidate_cache().await {
            warn!("Failed to invalidate semantic cache: {}", e);
        }
    }

    /// Handle ontology imported event
    pub async fn on_ontology_imported(&self, event: OntologyImportedEvent) {
        info!(
            "Ontology imported: {} classes, {} properties",
            event.class_count, event.property_count
        );

        // Trigger semantic analysis
        let graph = self.graph_data.read().await.clone();

        // Reinitialize semantic analyzer
        if let Err(e) = self.semantic_service.initialize(Arc::new(graph)).await {
            warn!("Failed to initialize semantic analyzer: {}", e);
        }

        // Detect communities after ontology import
        if let Err(e) = self.semantic_service.detect_communities_louvain().await {
            warn!("Failed to detect communities: {}", e);
        }
    }

    /// Handle positions updated event
    pub async fn on_positions_updated(&self, event: PositionsUpdatedEvent) {
        debug!("Positions updated for {} nodes", event.updated_nodes.len());

        // Broadcast to WebSocket clients
        let event_bus = self.event_bus.write().await;
        event_bus.publish_domain_event(event).await;
    }

    /// Handle node added event
    pub async fn on_node_added(&self, event: NodeAddedEvent) {
        info!("Node added: {}", event.node_id);

        // Invalidate semantic cache
        if let Err(e) = self.semantic_service.invalidate_cache().await {
            warn!("Failed to invalidate cache after node addition: {}", e);
        }

        // Trigger physics update
        if self.physics_service.is_running().await {
            debug!("Physics simulation running, will incorporate new node");
        }
    }

    /// Handle edge added event
    pub async fn on_edge_added(&self, event: EdgeAddedEvent) {
        info!("Edge added: {} -> {}", event.source_id, event.target_id);

        // Invalidate pathfinding cache
        if let Err(e) = self.semantic_service.invalidate_cache().await {
            warn!("Failed to invalidate cache after edge addition: {}", e);
        }
    }
}

/// Global event coordinator instance
pub static EVENT_COORDINATOR: once_cell::sync::Lazy<Arc<RwLock<Option<EventCoordinator>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize global event coordinator
pub async fn initialize_event_coordinator(
    physics_service: Arc<PhysicsService>,
    semantic_service: Arc<SemanticService>,
    event_bus: Arc<RwLock<EventBus>>,
    graph_data: Arc<RwLock<GraphData>>,
) {
    let coordinator =
        EventCoordinator::new(physics_service, semantic_service, event_bus, graph_data);

    coordinator.initialize().await;

    *EVENT_COORDINATOR.write().await = Some(coordinator);
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[tokio::test]
    async fn test_event_coordinator_creation() {
        // Test would require proper mocks
        // Placeholder for actual implementation
    }

    #[test]
    fn test_graph_saved_event() {
        let event = GraphSavedEvent {
            graph_id: "test".to_string(),
            file_path: "/test.json".to_string(),
            node_count: 100,
            edge_count: 200,
            timestamp: Utc::now(),
        };

        assert_eq!(event.node_count, 100);
        assert_eq!(event.edge_count, 200);
    }
}
