// Port: GraphRepository
// Defines the interface for graph state management
// Future: Add #[derive(HexPort)] when Hexser is available

use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashSet;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::utils::socket_flow_messages::BinaryNodeData;

pub type Result<T> = std::result::Result<T, String>;

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

    /// Clear the dirty nodes list
    async fn clear_dirty_nodes(&self) -> Result<()>;

    /// Get current graph version for optimistic locking
    async fn get_version(&self) -> Result<u64>;

    /// Get specific nodes by ID
    async fn get_nodes(&self, node_ids: Vec<u32>) -> Result<Vec<Node>>;

    /// Remove nodes from the graph
    async fn remove_nodes(&self, node_ids: Vec<u32>) -> Result<()>;

    /// Clear the entire graph
    async fn clear(&self) -> Result<()>;
}
