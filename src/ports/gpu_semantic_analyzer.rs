// src/ports/gpu_semantic_analyzer.rs
//! GPU Semantic Analyzer Port
//!
//! Provides GPU-accelerated semantic analysis, clustering, and pathfinding.
//! This port abstracts CUDA/OpenCL implementations for graph algorithms.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::constraints::ConstraintSet;
use crate::models::graph::GraphData;

pub type Result<T> = std::result::Result<T, GpuSemanticAnalyzerError>;

#[derive(Debug, thiserror::Error)]
pub enum GpuSemanticAnalyzerError {
    #[error("GPU not available")]
    GpuNotAvailable,

    #[error("Analysis error: {0}")]
    AnalysisError(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("Algorithm not supported: {0}")]
    UnsupportedAlgorithm(String),

    #[error("CUDA error: {0}")]
    CudaError(String),
}

/// Clustering algorithm options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClusteringAlgorithm {
    Louvain,
    LabelPropagation,
    ConnectedComponents,
    HierarchicalClustering { min_cluster_size: usize },
}

/// Community detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionResult {
    pub clusters: HashMap<u32, usize>,        // node_id -> cluster_id
    pub cluster_sizes: HashMap<usize, usize>, // cluster_id -> size
    pub modularity: f32,
    pub computation_time_ms: f32,
}

/// Pathfinding result from SSSP
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathfindingResult {
    pub source_node: u32,
    pub distances: HashMap<u32, f32>,  // node_id -> distance
    pub paths: HashMap<u32, Vec<u32>>, // node_id -> path (sequence of nodes)
    pub computation_time_ms: f32,
}

/// Semantic constraint configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConstraintConfig {
    pub similarity_threshold: f32,
    pub enable_clustering_constraints: bool,
    pub enable_importance_constraints: bool,
    pub enable_topic_constraints: bool,
    pub max_constraints: usize,
}

/// Layout optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub converged: bool,
    pub iterations: u32,
    pub final_stress: f32,
    pub convergence_delta: f32,
    pub computation_time_ms: f32,
}

/// Node importance algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportanceAlgorithm {
    PageRank { damping: f32, max_iterations: usize },
    Betweenness,
    Closeness,
    Eigenvector,
    Degree,
}

/// Semantic analysis statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticStatistics {
    pub total_analyses: u64,
    pub average_clustering_time_ms: f32,
    pub average_pathfinding_time_ms: f32,
    pub cache_hit_rate: f32,
    pub gpu_memory_used_mb: f32,
}

/// Port for GPU semantic analysis operations
#[async_trait]
pub trait GpuSemanticAnalyzer: Send + Sync {
    /// Initialize semantic analyzer with graph data
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<()>;

    /// Perform GPU-accelerated community detection
    async fn detect_communities(
        &mut self,
        algorithm: ClusteringAlgorithm,
    ) -> Result<CommunityDetectionResult>;

    /// Compute shortest paths from a source node (GPU-accelerated SSSP)
    /// Uses CUDA kernel from sssp_compact.cu
    async fn compute_shortest_paths(&mut self, source_node_id: u32) -> Result<PathfindingResult>;

    /// Compute single-source shortest paths and return only distances
    /// Optimized for cases where path reconstruction is not needed
    async fn compute_sssp_distances(&mut self, source_node_id: u32) -> Result<Vec<f32>>;

    /// Compute all-pairs shortest paths using landmark-based approximation
    /// Uses CUDA kernel from gpu_landmark_apsp.cu
    /// Returns: HashMap mapping (source, target) -> path (Vec of node IDs)
    async fn compute_all_pairs_shortest_paths(&mut self) -> Result<HashMap<(u32, u32), Vec<u32>>>;

    /// Compute approximate APSP using k landmarks
    /// More efficient than exact APSP for large graphs
    /// num_landmarks: typically sqrt(num_nodes) for good approximation
    async fn compute_landmark_apsp(&mut self, num_landmarks: usize) -> Result<Vec<Vec<f32>>>;

    /// Generate semantic constraints based on graph analysis
    async fn generate_semantic_constraints(
        &mut self,
        config: SemanticConstraintConfig,
    ) -> Result<ConstraintSet>;

    /// Perform stress majorization layout optimization
    async fn optimize_layout(
        &mut self,
        constraints: &ConstraintSet,
        max_iterations: usize,
    ) -> Result<OptimizationResult>;

    /// Analyze node importance (PageRank, centrality, etc.)
    async fn analyze_node_importance(
        &mut self,
        algorithm: ImportanceAlgorithm,
    ) -> Result<HashMap<u32, f32>>;

    /// Update graph data for analysis
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<()>;

    /// Get semantic analysis statistics
    async fn get_statistics(&self) -> Result<SemanticStatistics>;

    /// Invalidate pathfinding cache (call after graph structure changes)
    async fn invalidate_pathfinding_cache(&mut self) -> Result<()>;
}
