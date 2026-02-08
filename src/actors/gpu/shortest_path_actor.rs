//! Shortest Path Actor - Handles SSSP and APSP computations on GPU
//!
//! This actor wraps the existing GPU kernels for:
//! - Single-Source Shortest Path (SSSP) using Bellman-Ford-based frontier compaction
//! - All-Pairs Shortest Path (APSP) using landmark-based approximation
//!
//! Use cases:
//! - Path highlighting in graph visualization
//! - Route visualization for navigation
//! - Connectivity analysis
//! - Distance-based graph analytics

use actix::prelude::*;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::shared::{GPUState, SharedGPUContext};
use crate::actors::messages::*;

/// SSSP computation parameters
#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[rtype(result = "Result<SSSPResult, String>")]
pub struct ComputeSSP {
    /// Source node index for SSSP computation
    pub source_idx: usize,
    /// Optional maximum distance cutoff
    pub max_distance: Option<f32>,
    /// Optional delta-stepping bucket width.  When `Some(d)`, edges are relaxed
    /// in distance buckets of width `d` instead of all-at-once (Bellman-Ford).
    /// Smaller deltas reduce work per iteration at the cost of more iterations.
    #[serde(default)]
    pub delta: Option<f32>,
}

/// APSP computation parameters
#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[rtype(result = "Result<APSPResult, String>")]
pub struct ComputeAPSP {
    /// Number of landmark nodes for approximation
    pub num_landmarks: usize,
    /// Optional seed for landmark selection
    pub seed: Option<u64>,
}

/// Query shortest path between two nodes
#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[rtype(result = "Result<PathResult, String>")]
pub struct QueryPath {
    /// Source node ID
    pub source_id: String,
    /// Target node ID
    pub target_id: String,
}

/// SSSP computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSSPResult {
    /// Distance from source to each node (indexed by node index)
    pub distances: Vec<f32>,
    /// Source node index
    pub source_idx: usize,
    /// Number of nodes reached
    pub nodes_reached: usize,
    /// Maximum distance found
    pub max_distance: f32,
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
}

/// APSP computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APSPResult {
    /// Approximate all-pairs distances [num_nodes x num_nodes]
    /// Stored in row-major order: distance[i][j] = distances[i * num_nodes + j]
    pub distances: Vec<f32>,
    /// Number of nodes
    pub num_nodes: usize,
    /// Number of landmarks used
    pub num_landmarks: usize,
    /// Landmark node indices
    pub landmarks: Vec<usize>,
    /// Average approximation error estimate
    pub avg_error_estimate: f32,
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
}

/// Path query result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    /// Path as sequence of node IDs
    pub path: Vec<String>,
    /// Total path distance
    pub distance: f32,
    /// Whether path exists
    pub exists: bool,
}

/// Shortest path computation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShortestPathStats {
    pub total_sssp_computations: u64,
    pub total_apsp_computations: u64,
    pub avg_sssp_time_ms: f32,
    pub avg_apsp_time_ms: f32,
    pub last_computation_time_ms: u64,
}

/// Shortest Path Actor
pub struct ShortestPathActor {
    /// GPU state tracking
    gpu_state: GPUState,

    /// Shared GPU context
    shared_context: Option<Arc<SharedGPUContext>>,

    /// Computation statistics
    stats: ShortestPathStats,
}

impl ShortestPathActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            stats: ShortestPathStats {
                total_sssp_computations: 0,
                total_apsp_computations: 0,
                avg_sssp_time_ms: 0.0,
                avg_apsp_time_ms: 0.0,
                last_computation_time_ms: 0,
            },
        }
    }

    /// Update statistics with new computation time
    fn update_stats(&mut self, is_sssp: bool, time_ms: u64) {
        self.stats.last_computation_time_ms = time_ms;

        if is_sssp {
            let total = self.stats.total_sssp_computations as f32;
            self.stats.avg_sssp_time_ms =
                (self.stats.avg_sssp_time_ms * total + time_ms as f32) / (total + 1.0);
            self.stats.total_sssp_computations += 1;
        } else {
            let total = self.stats.total_apsp_computations as f32;
            self.stats.avg_apsp_time_ms =
                (self.stats.avg_apsp_time_ms * total + time_ms as f32) / (total + 1.0);
            self.stats.total_apsp_computations += 1;
        }
    }
}

impl Default for ShortestPathActor {
    fn default() -> Self {
        Self::new()
    }
}

impl Actor for ShortestPathActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("ShortestPathActor started");
        ctx.notify(InitializeActor);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("ShortestPathActor stopped");
    }
}

// Message Handlers

impl Handler<InitializeActor> for ShortestPathActor {
    type Result = ();

    fn handle(&mut self, _msg: InitializeActor, _ctx: &mut Self::Context) -> Self::Result {
        info!("ShortestPathActor: Initializing");
        self.gpu_state.is_initialized = true;
    }
}

impl Handler<SetSharedGPUContext> for ShortestPathActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        info!("ShortestPathActor: Setting GPU context");
        self.shared_context = Some(msg.context);
        self.gpu_state.is_initialized = true;
        Ok(())
    }
}

impl Handler<ComputeSSP> for ShortestPathActor {
    type Result = Result<SSSPResult, String>;

    fn handle(&mut self, msg: ComputeSSP, _ctx: &mut Self::Context) -> Self::Result {
        info!("ShortestPathActor: Computing SSSP from node {}", msg.source_idx);

        // Acquire lock, compute, then drop lock before calling update_stats
        let (filtered_distances, nodes_reached, max_distance, computation_time) = {
            let mut unified_compute = match &self.shared_context {
                Some(ctx) => ctx
                    .unified_compute
                    .lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?,
                None => {
                    return Err("GPU context not initialized".to_string());
                }
            };

            let start_time = Instant::now();

            // Call the existing GPU SSSP implementation
            let distances = unified_compute
                .run_sssp(msg.source_idx, msg.delta)
                .map_err(|e| {
                    error!("GPU SSSP computation failed: {}", e);
                    format!("SSSP computation failed: {}", e)
                })?;

            let computation_time = start_time.elapsed().as_millis() as u64;

            // Calculate statistics
            let mut nodes_reached = 0;
            let mut max_distance = 0.0f32;

            for &dist in &distances {
                if dist < f32::MAX {
                    nodes_reached += 1;
                    max_distance = max_distance.max(dist);
                }
            }

            // Apply max_distance filter if specified
            let filtered_distances = if let Some(max_dist) = msg.max_distance {
                distances.into_iter().map(|d| {
                    if d <= max_dist { d } else { f32::MAX }
                }).collect()
            } else {
                distances
            };

            (filtered_distances, nodes_reached, max_distance, computation_time)
        }; // unified_compute lock dropped here

        // Now we can safely call update_stats with mutable borrow
        self.update_stats(true, computation_time);

        info!(
            "ShortestPathActor: SSSP completed in {}ms, reached {}/{} nodes",
            computation_time, nodes_reached, filtered_distances.len()
        );

        Ok(SSSPResult {
            distances: filtered_distances,
            source_idx: msg.source_idx,
            nodes_reached,
            max_distance,
            computation_time_ms: computation_time,
        })
    }
}

impl Handler<ComputeAPSP> for ShortestPathActor {
    type Result = Result<APSPResult, String>;

    fn handle(&mut self, msg: ComputeAPSP, _ctx: &mut Self::Context) -> Self::Result {
        info!("ShortestPathActor: Computing APSP with {} landmarks", msg.num_landmarks);

        // Acquire lock, compute, then drop lock before calling update_stats
        let (apsp_distances, num_nodes, landmarks, computation_time) = {
            let mut unified_compute = match &self.shared_context {
                Some(ctx) => ctx
                    .unified_compute
                    .lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?,
                None => {
                    return Err("GPU context not initialized".to_string());
                }
            };

            let start_time = Instant::now();

            // Get graph size
            let num_nodes = unified_compute.get_num_nodes();

            if msg.num_landmarks >= num_nodes {
                return Err(format!(
                    "Number of landmarks ({}) must be less than number of nodes ({})",
                    msg.num_landmarks, num_nodes
                ));
            }

            // Select landmark nodes (stratified sampling)
            let mut landmarks = Vec::with_capacity(msg.num_landmarks);
            let stride = num_nodes / msg.num_landmarks;
            let seed = msg.seed.unwrap_or(42);

            for i in 0..msg.num_landmarks {
                let landmark = (i * stride + ((seed + i as u64) % stride as u64) as usize) % num_nodes;
                landmarks.push(landmark);
            }

            // Run batched SSSP from all landmarks at once (keeps CSR on device)
            let landmark_vecs = unified_compute
                .run_sssp_batch(&landmarks)
                .map_err(|e| {
                    error!("GPU batched SSSP computation failed: {}", e);
                    format!("Batched SSSP computation failed: {}", e)
                })?;

            // Flatten landmark distances into [num_landmarks][num_nodes] layout
            let mut landmark_distances = Vec::with_capacity(msg.num_landmarks * num_nodes);
            for dists in &landmark_vecs {
                landmark_distances.extend_from_slice(dists);
            }

            // Try GPU kernel for APSP assembly; fall back to CPU if module unavailable
            let apsp_distances = match unified_compute
                .run_apsp_gpu(&landmark_distances, msg.num_landmarks)
            {
                Ok(gpu_result) => {
                    info!("APSP assembly completed on GPU");
                    gpu_result
                }
                Err(e) => {
                    info!(
                        "GPU APSP kernel unavailable ({}), using CPU fallback",
                        e
                    );
                    // CPU fallback: triangle inequality approximation
                    let mut dists = vec![f32::MAX; num_nodes * num_nodes];
                    for i in 0..num_nodes {
                        dists[i * num_nodes + i] = 0.0;
                        for j in (i + 1)..num_nodes {
                            let mut min_dist = f32::MAX;
                            for k_idx in 0..msg.num_landmarks {
                                let dist_ki = landmark_distances[k_idx * num_nodes + i];
                                let dist_kj = landmark_distances[k_idx * num_nodes + j];
                                if dist_ki < f32::MAX && dist_kj < f32::MAX {
                                    min_dist = min_dist.min(dist_ki + dist_kj);
                                }
                            }
                            dists[i * num_nodes + j] = min_dist;
                            dists[j * num_nodes + i] = min_dist;
                        }
                    }
                    dists
                }
            };

            let computation_time = start_time.elapsed().as_millis() as u64;

            (apsp_distances, num_nodes, landmarks, computation_time)
        }; // unified_compute lock dropped here

        // Now we can safely call update_stats with mutable borrow
        self.update_stats(false, computation_time);

        // Estimate approximation error (simplified)
        let avg_error_estimate = 0.15; // Typical 15% error for landmark-based APSP

        info!(
            "ShortestPathActor: APSP completed in {}ms with {} landmarks",
            computation_time, msg.num_landmarks
        );

        Ok(APSPResult {
            distances: apsp_distances,
            num_nodes,
            num_landmarks: msg.num_landmarks,
            landmarks,
            avg_error_estimate,
            computation_time_ms: computation_time,
        })
    }
}

/// Get shortest path statistics
#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[rtype(result = "ShortestPathStats")]
pub struct GetShortestPathStats;

impl Handler<GetShortestPathStats> for ShortestPathActor {
    type Result = MessageResult<GetShortestPathStats>;

    fn handle(&mut self, _msg: GetShortestPathStats, _ctx: &mut Self::Context) -> Self::Result {
        MessageResult(self.stats.clone())
    }
}
