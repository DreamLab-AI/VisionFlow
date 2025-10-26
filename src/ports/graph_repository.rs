// src/ports/graph_repository.rs
//! Graph Repository Port
//!
//! Defines the interface for graph data access and manipulation.
//! This port abstracts away the concrete implementation (actor-based, direct access, etc.)

use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::actors::graph_actor::{AutoBalanceNotification, PhysicsState};
use crate::models::constraints::ConstraintSet;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use glam::Vec3;

// Placeholder for BinaryNodeData - will use actual type from GPU module
pub type BinaryNodeData = (f32, f32, f32);

pub type Result<T> = std::result::Result<T, GraphRepositoryError>;

#[derive(Debug, thiserror::Error)]
pub enum GraphRepositoryError {
    #[error("Graph not found")]
    NotFound,

    #[error("Graph access error: {0}")]
    AccessError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Parameters for pathfinding operations
#[derive(Debug, Clone)]
pub struct PathfindingParams {
    pub start_node: u32,
    pub end_node: u32,
    pub max_depth: Option<usize>,
}

/// Result of pathfinding computation
#[derive(Debug, Clone)]
pub struct PathfindingResult {
    pub path: Vec<u32>,
    pub total_distance: f32,
}

/// Port for graph data repository operations
#[async_trait]
pub trait GraphRepository: Send + Sync {
    // === Write Operations (Commands) ===

    /// Add nodes to the graph
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;

    /// Add edges to the graph
    async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;

    /// Update node positions (from physics simulation)
    async fn update_positions(&self, updates: Vec<(u32, BinaryNodeData)>) -> Result<()>;

    /// Clear dirty node tracking
    async fn clear_dirty_nodes(&self) -> Result<()>;

    // === Read Operations (Queries) - CQRS Pattern ===

    /// Get the current graph state
    async fn get_graph(&self) -> Result<Arc<GraphData>>;

    /// Get the node map (id -> Node)
    async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>>;

    /// Get current physics state
    async fn get_physics_state(&self) -> Result<PhysicsState>;

    /// Get node positions as a vector of (id, position) tuples
    async fn get_node_positions(&self) -> Result<Vec<(u32, Vec3)>>;

    /// Get the graph data for bots/pathfinding
    async fn get_bots_graph(&self) -> Result<Arc<GraphData>>;

    /// Get current constraint set
    async fn get_constraints(&self) -> Result<ConstraintSet>;

    /// Get auto-balance notifications
    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>>;

    /// Get equilibrium status
    async fn get_equilibrium_status(&self) -> Result<bool>;

    /// Compute shortest paths between nodes
    async fn compute_shortest_paths(&self, params: PathfindingParams) -> Result<PathfindingResult>;

    /// Get nodes that have changed since last sync
    async fn get_dirty_nodes(&self) -> Result<HashSet<u32>>;
}
