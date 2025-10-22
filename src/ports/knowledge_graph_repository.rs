// src/ports/knowledge_graph_repository.rs
//! Knowledge Graph Repository Port
//!
//! Manages the main knowledge graph structure parsed from local markdown files.
//! This port provides comprehensive graph data access and manipulation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;

pub type Result<T> = std::result::Result<T, KnowledgeGraphRepositoryError>;

#[derive(Debug, thiserror::Error)]
pub enum KnowledgeGraphRepositoryError {
    #[error("Graph not found")]
    NotFound,

    #[error("Node not found: {0}")]
    NodeNotFound(u32),

    #[error("Edge not found: {0}")]
    EdgeNotFound(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),

    #[error("Concurrent modification detected")]
    ConcurrentModification,
}

/// Graph statistics for monitoring and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub connected_components: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Port for knowledge graph repository operations
#[async_trait]
pub trait KnowledgeGraphRepository: Send + Sync {
    /// Load complete graph structure from database
    async fn load_graph(&self) -> Result<Arc<GraphData>>;

    /// Save complete graph structure to database
    async fn save_graph(&self, graph: &GraphData) -> Result<()>;

    /// Add a single node to the graph
    /// Returns the assigned node ID
    async fn add_node(&self, node: &Node) -> Result<u32>;

    /// Update an existing node
    async fn update_node(&self, node: &Node) -> Result<()>;

    /// Remove a node by ID
    async fn remove_node(&self, node_id: u32) -> Result<()>;

    /// Get a node by ID
    async fn get_node(&self, node_id: u32) -> Result<Option<Node>>;

    /// Get nodes by metadata ID
    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> Result<Vec<Node>>;

    /// Add an edge between two nodes
    /// Returns the assigned edge ID
    async fn add_edge(&self, edge: &Edge) -> Result<String>;

    /// Update an existing edge
    async fn update_edge(&self, edge: &Edge) -> Result<()>;

    /// Remove an edge by ID
    async fn remove_edge(&self, edge_id: &str) -> Result<()>;

    /// Get all edges connected to a node
    async fn get_node_edges(&self, node_id: u32) -> Result<Vec<Edge>>;

    /// Batch update node positions (for physics simulation)
    /// Format: Vec<(node_id, x, y, z)>
    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> Result<()>;

    /// Query nodes by properties (e.g., "color = red", "size > 10")
    async fn query_nodes(&self, query: &str) -> Result<Vec<Node>>;

    /// Get graph statistics
    async fn get_statistics(&self) -> Result<GraphStatistics>;
}
