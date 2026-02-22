//! Analytics-domain messages: clustering, anomaly detection, community detection,
//! SSSP, PageRank, and related parameter/result types.

use actix::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// K-means clustering
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansResult {
    pub cluster_assignments: Vec<i32>,
    pub centroids: Vec<(f32, f32, f32)>,
    pub inertia: f32,
    pub iterations: u32,
    pub clusters: Vec<crate::handlers::api_handler::analytics::Cluster>,
    pub stats: crate::actors::gpu::clustering_actor::ClusteringStats,
    pub converged: bool,
    pub final_iteration: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansParams {
    pub num_clusters: usize,
    pub max_iterations: Option<u32>,
    pub tolerance: Option<f32>,
    pub seed: Option<u32>,
}

#[derive(Message)]
#[rtype(result = "Result<KMeansResult, String>")]
pub struct RunKMeans {
    pub params: KMeansParams,
}

// ---------------------------------------------------------------------------
// Anomaly detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyResult {
    pub lof_scores: Option<Vec<f32>>,
    pub local_densities: Option<Vec<f32>>,
    pub zscore_values: Option<Vec<f32>>,
    pub anomaly_threshold: f32,
    pub num_anomalies: usize,
    pub anomalies: Vec<crate::actors::gpu::anomaly_detection_actor::AnomalyNode>,
    pub stats: AnomalyDetectionStats,
    pub method: AnomalyDetectionMethod,
    pub threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionStats {
    pub total_nodes_analyzed: u32,
    pub anomalies_found: usize,
    pub detection_threshold: f32,
    pub computation_time_ms: u64,
    pub method: AnomalyDetectionMethod,
    pub average_anomaly_score: f32,
    pub max_anomaly_score: f32,
    pub min_anomaly_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyMethod {
    LocalOutlierFactor,
    ZScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    LOF,
    ZScore,
    IsolationForest,
    DBSCAN,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyParams {
    pub method: AnomalyMethod,
    pub k_neighbors: i32,
    pub radius: f32,
    pub feature_data: Option<Vec<f32>>,
    pub threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionParams {
    pub method: AnomalyDetectionMethod,
    pub threshold: Option<f32>,
    pub k_neighbors: Option<i32>,
    pub window_size: Option<usize>,
    pub feature_data: Option<Vec<f32>>,
}

#[derive(Message)]
#[rtype(result = "Result<AnomalyResult, String>")]
pub struct RunAnomalyDetection {
    pub params: AnomalyParams,
}

// ---------------------------------------------------------------------------
// Community detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionResult {
    pub node_labels: Vec<i32>,
    pub num_communities: usize,
    pub modularity: f32,
    pub iterations: u32,
    pub community_sizes: Vec<i32>,
    pub converged: bool,
    pub communities: Vec<crate::actors::gpu::clustering_actor::Community>,
    pub stats: crate::actors::gpu::clustering_actor::CommunityDetectionStats,
    pub algorithm: CommunityDetectionAlgorithm,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunityDetectionAlgorithm {
    LabelPropagation,
    Louvain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionParams {
    pub algorithm: CommunityDetectionAlgorithm,
    pub max_iterations: Option<u32>,
    pub convergence_tolerance: Option<f32>,
    pub synchronous: Option<bool>,
    pub seed: Option<u32>,
}

#[derive(Message)]
#[rtype(result = "Result<CommunityDetectionResult, String>")]
pub struct RunCommunityDetection {
    pub params: CommunityDetectionParams,
}

// ---------------------------------------------------------------------------
// GPU Clustering (higher-level orchestration messages)
// ---------------------------------------------------------------------------

#[derive(Message, Clone)]
#[rtype(result = "Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>")]
pub struct PerformGPUClustering {
    pub method: String,
    pub params: crate::handlers::api_handler::analytics::ClusteringParams,
    pub task_id: String,
}

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct StartGPUClustering {
    pub algorithm: String,
    pub cluster_count: u32,
    pub task_id: String,
}

#[derive(Message)]
#[rtype(result = "Result<serde_json::Value, String>")]
pub struct GetClusteringStatus;

#[derive(Message)]
#[rtype(result = "Result<serde_json::Value, String>")]
pub struct GetClusteringResults;

#[derive(Message)]
#[rtype(result = "Result<String, String>")]
pub struct ExportClusterAssignments {
    pub format: String,
}

// ---------------------------------------------------------------------------
// SSSP (Single-Source Shortest Path)
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ComputeSSSP {
    pub source_node: u32,
}

#[derive(Message)]
#[rtype(result = "Result<super::PathfindingResult, String>")]
pub struct ComputeShortestPaths {
    pub source_node_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<HashMap<(u32, u32), Vec<u32>>, String>")]
pub struct ComputeAllPairsShortestPaths {
    pub num_landmarks: Option<usize>,
}

// ---------------------------------------------------------------------------
// PageRank Centrality (P1-2)
// ---------------------------------------------------------------------------

#[derive(Message)]
#[rtype(result = "Result<crate::actors::gpu::pagerank_actor::PageRankResult, String>")]
pub struct ComputePageRank {
    pub params: Option<crate::actors::gpu::pagerank_actor::PageRankParams>,
}

#[derive(Message)]
#[rtype(result = "Option<crate::actors::gpu::pagerank_actor::PageRankResult>")]
pub struct GetPageRankResult;

#[derive(Message)]
#[rtype(result = "()")]
pub struct ClearPageRankCache;

// ---------------------------------------------------------------------------
// Connected Components
// ---------------------------------------------------------------------------

/// Update the cached edge list used by ConnectedComponentsActor for label propagation.
/// Send this whenever graph edges change so connected-component queries use real data.
#[derive(Message, Debug, Clone)]
#[rtype(result = "()")]
pub struct UpdateComponentEdges {
    /// Edge list as (source_node_id, target_node_id) pairs
    pub edges: Vec<(u32, u32)>,
}
