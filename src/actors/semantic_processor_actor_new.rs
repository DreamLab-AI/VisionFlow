//! Semantic Processor Actor
//!
//! Orchestrates graph algorithms including SSSP, clustering, and community detection.
//! Integrates with GPU compute actors for performance-critical operations.

use actix::prelude::*;
use log::{debug, error, info, warn};
use std::collections::HashMap;

use crate::actors::messages::*;
use crate::actors::graph_state_actor::GraphStateActor;
use crate::actors::gpu::GPUManagerActor;
use crate::ports::semantic_analyzer::{SSSPResult, ClusteringResult, CommunityResult, ClusterAlgorithm};

/// Message: Run SSSP from source node
#[derive(Message)]
#[rtype(result = "Result<SSSPResult, String>")]
pub struct RunSSSP {
    pub source: u32,
}

/// Message: Run clustering with specified algorithm
#[derive(Message)]
#[rtype(result = "Result<ClusteringResult, String>")]
pub struct RunClustering {
    pub algorithm: ClusterAlgorithm,
}

/// Message: Detect communities in graph
#[derive(Message)]
#[rtype(result = "Result<CommunityResult, String>")]
pub struct DetectCommunities;

/// Message: Get shortest path between two nodes
#[derive(Message)]
#[rtype(result = "Result<Vec<u32>, String>")]
pub struct GetShortestPath {
    pub source: u32,
    pub target: u32,
}

/// Message: Invalidate all caches
#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct InvalidateCache;

/// Semantic Processor Actor - Orchestrates graph algorithms
pub struct SemanticProcessorActor {
    graph_state: Addr<GraphStateActor>,
    gpu_manager: Option<Addr<GPUManagerActor>>,
    sssp_cache: HashMap<u32, SSSPResult>,
    clustering_cache: HashMap<String, ClusteringResult>,
    community_cache: Option<CommunityResult>,
}
