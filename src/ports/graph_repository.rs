// src/ports/graph_repository.rs
//! Graph Repository Port
//!
//! Defines the interface for graph data access and manipulation.
//! This port abstracts away the concrete implementation (actor-based, direct access, etc.)

use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;

use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;

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

/// Port for graph data repository operations
#[async_trait]
pub trait GraphRepository: Send + Sync {
    /// Get the current graph state
    async fn get_graph(&self) -> Result<Arc<GraphData>>;

    /// Add nodes to the graph
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;

    /// Add edges to the graph
    async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;

    /// Update node positions (from physics simulation)
    async fn update_positions(&self, updates: Vec<(u32, BinaryNodeData)>) -> Result<()>;

    /// Get nodes that have changed since last sync
    async fn get_dirty_nodes(&self) -> Result<HashSet<u32>>;

    /// Clear dirty node tracking
    async fn clear_dirty_nodes(&self) -> Result<()>;
}
