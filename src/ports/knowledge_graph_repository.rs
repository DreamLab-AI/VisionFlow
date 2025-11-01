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

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub connected_components: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/
#[async_trait]
pub trait KnowledgeGraphRepository: Send + Sync {
    
    async fn load_graph(&self) -> Result<Arc<GraphData>>;

    
    async fn save_graph(&self, graph: &GraphData) -> Result<()>;

    
    
    async fn add_node(&self, node: &Node) -> Result<u32>;

    
    
    async fn batch_add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;

    
    async fn update_node(&self, node: &Node) -> Result<()>;

    
    async fn batch_update_nodes(&self, nodes: Vec<Node>) -> Result<()>;

    
    async fn remove_node(&self, node_id: u32) -> Result<()>;

    
    async fn batch_remove_nodes(&self, node_ids: Vec<u32>) -> Result<()>;

    
    async fn get_node(&self, node_id: u32) -> Result<Option<Node>>;

    
    async fn get_nodes(&self, node_ids: Vec<u32>) -> Result<Vec<Node>>;

    
    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> Result<Vec<Node>>;

    
    async fn search_nodes_by_label(&self, label: &str) -> Result<Vec<Node>>;

    
    
    async fn add_edge(&self, edge: &Edge) -> Result<String>;

    
    
    async fn batch_add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;

    
    async fn update_edge(&self, edge: &Edge) -> Result<()>;

    
    async fn remove_edge(&self, edge_id: &str) -> Result<()>;

    
    async fn batch_remove_edges(&self, edge_ids: Vec<String>) -> Result<()>;

    
    async fn get_node_edges(&self, node_id: u32) -> Result<Vec<Edge>>;

    
    async fn get_edges_between(&self, source_id: u32, target_id: u32) -> Result<Vec<Edge>>;

    
    
    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> Result<()>;

    
    async fn query_nodes(&self, query: &str) -> Result<Vec<Node>>;

    
    async fn get_neighbors(&self, node_id: u32) -> Result<Vec<Node>>;

    
    async fn get_statistics(&self) -> Result<GraphStatistics>;

    
    async fn clear_graph(&self) -> Result<()>;

    
    async fn begin_transaction(&self) -> Result<()>;

    
    async fn commit_transaction(&self) -> Result<()>;

    
    async fn rollback_transaction(&self) -> Result<()>;

    
    async fn health_check(&self) -> Result<bool>;
}
