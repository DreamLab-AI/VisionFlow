// Port: SemanticAnalyzer
// Defines the interface for graph algorithms and semantic analysis
// Future: Add #[derive(HexPort)] when Hexser is available

use async_trait::async_trait;
use std::collections::HashMap;
use crate::models::graph::GraphData;

pub type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
pub struct SSSPResult {
    pub distances: HashMap<u32, f32>,
    pub parents: HashMap<u32, i32>,
    pub source: u32,
}

#[derive(Debug, Clone)]
pub struct ClusteringResult {
    pub clusters: HashMap<u32, u32>,  // node_id -> cluster_id
    pub cluster_count: u32,
    pub algorithm: ClusterAlgorithm,
}

#[derive(Debug, Clone, Copy)]
pub enum ClusterAlgorithm {
    KMeans { k: u32 },
    DBSCAN { eps: f32, min_samples: u32 },
    Hierarchical { num_clusters: u32 },
}

#[derive(Debug, Clone)]
pub struct CommunityResult {
    pub communities: HashMap<u32, u32>,  // node_id -> community_id
    pub modularity: f32,
}

#[async_trait]
pub trait SemanticAnalyzer: Send + Sync {
    /// Run Single-Source Shortest Path from a source node
    async fn run_sssp(&self, graph: &GraphData, source: u32) -> Result<SSSPResult>;

    /// Run clustering algorithm on the graph
    async fn run_clustering(&self, graph: &GraphData, algorithm: ClusterAlgorithm) -> Result<ClusteringResult>;

    /// Detect communities using Louvain modularity
    async fn detect_communities(&self, graph: &GraphData) -> Result<CommunityResult>;

    /// Get shortest path between two nodes
    async fn get_shortest_path(&self, graph: &GraphData, source: u32, target: u32) -> Result<Vec<u32>>;

    /// Invalidate algorithm caches
    async fn invalidate_cache(&self) -> Result<()>;
}
