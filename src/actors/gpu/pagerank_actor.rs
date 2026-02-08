//! PageRank Actor - Handles PageRank centrality computation on GPU
//!
//! This actor implements the PageRank algorithm using GPU acceleration for efficient
//! centrality computation on large graphs. PageRank is a measure of node importance
//! based on the structure of incoming links.
//!
//! ## Algorithm
//!
//! PageRank uses the power iteration method:
//! ```
//! PR(v) = (1-d)/N + d * Σ(PR(u)/out_degree(u))
//! ```
//! where:
//! - d = damping factor (typically 0.85)
//! - N = number of nodes
//! - The sum is over all nodes u that link to v
//!
//! ## Features
//!
//! - GPU-accelerated computation using CUDA
//! - Convergence detection with configurable epsilon
//! - Dangling node handling (nodes with no outgoing edges)
//! - Normalization to ensure sum of PageRank values = 1.0
//! - Performance metrics tracking
//!
//! ## Visual Analytics Integration
//!
//! PageRank values can be used to:
//! - Size nodes proportionally to their importance
//! - Color nodes using a gradient (low → high centrality)
//! - Filter/highlight influential nodes
//! - Drive layout forces (important nodes at center)

use actix::prelude::*;
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::shared::{GPUState, SharedGPUContext};
use crate::actors::messages::*;

/// PageRank computation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankParams {
    /// Damping factor (probability of following a link vs random jump)
    /// Typical value: 0.85
    pub damping_factor: Option<f32>,

    /// Maximum number of iterations
    pub max_iterations: Option<u32>,

    /// Convergence threshold (L1 norm of difference between iterations)
    pub epsilon: Option<f32>,

    /// Whether to normalize results (ensure sum = 1.0)
    pub normalize: Option<bool>,

    /// Use optimized kernel with shared memory
    pub use_optimized: Option<bool>,
}

impl Default for PageRankParams {
    fn default() -> Self {
        Self {
            damping_factor: Some(0.85),
            max_iterations: Some(100),
            epsilon: Some(1e-6),
            normalize: Some(true),
            use_optimized: Some(true),
        }
    }
}

/// PageRank computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankResult {
    /// PageRank value for each node (index = node_id)
    pub pagerank_values: Vec<f32>,

    /// Number of iterations performed
    pub iterations: u32,

    /// Whether the algorithm converged
    pub converged: bool,

    /// Final convergence metric (L1 norm of difference)
    pub convergence_value: f32,

    /// Top K most important nodes (sorted by PageRank)
    pub top_nodes: Vec<PageRankNode>,

    /// Statistical summary
    pub stats: PageRankStats,
}

/// Node with its PageRank value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankNode {
    pub node_id: u32,
    pub pagerank: f32,
    pub rank: usize, // 1-based rank (1 = highest)
}

/// Statistical summary of PageRank results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageRankStats {
    pub total_nodes: u32,
    pub max_pagerank: f32,
    pub min_pagerank: f32,
    pub mean_pagerank: f32,
    pub median_pagerank: f32,
    pub std_deviation: f32,
    pub computation_time_ms: u64,
    pub converged: bool,
    pub iterations: u32,
}

/// PageRank Actor for GPU-accelerated centrality computation
pub struct PageRankActor {
    /// GPU state and resource management
    gpu_state: GPUState,

    /// Shared GPU context with compute engine
    shared_context: Option<Arc<SharedGPUContext>>,

    /// Last computed PageRank results (cached)
    last_result: Option<PageRankResult>,
}

impl PageRankActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            last_result: None,
        }
    }

    /// Perform PageRank computation on GPU
    #[allow(dead_code)]
    async fn compute_pagerank(
        &mut self,
        params: PageRankParams,
    ) -> Result<PageRankResult, String> {
        info!("PageRankActor: Starting PageRank computation");

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

        // Extract parameters with defaults
        let damping = params.damping_factor.unwrap_or(0.85);
        let max_iter = params.max_iterations.unwrap_or(100) as usize;
        let epsilon = params.epsilon.unwrap_or(1e-6);
        let normalize = params.normalize.unwrap_or(true);
        let use_optimized = params.use_optimized.unwrap_or(true);

        // Call GPU PageRank computation
        let gpu_result = unified_compute
            .run_pagerank_centrality(damping, max_iter, epsilon, normalize, use_optimized)
            .map_err(|e| {
                error!("GPU PageRank computation failed: {}", e);
                format!("PageRank computation failed: {}", e)
            })?;

        let computation_time = start_time.elapsed();
        info!(
            "PageRankActor: PageRank computation completed in {:?}",
            computation_time
        );

        // Unpack GPU result and convert iterations to u32 for PageRankResult
        let (pagerank_values, iterations, converged, convergence_value) = gpu_result;
        let iterations = iterations as u32;

        // Compute statistics
        let stats = self.calculate_statistics(
            &pagerank_values,
            iterations,
            converged,
            computation_time.as_millis() as u64,
        );

        // Extract top K nodes (top 10 by default)
        let top_nodes = self.extract_top_nodes(&pagerank_values, 10);

        let result = PageRankResult {
            pagerank_values,
            iterations,
            converged,
            convergence_value,
            top_nodes,
            stats,
        };

        // Cache result
        self.last_result = Some(result.clone());

        Ok(result)
    }

    /// Calculate statistics from PageRank values
    fn calculate_statistics(
        &self,
        values: &[f32],
        iterations: u32,
        converged: bool,
        computation_time_ms: u64,
    ) -> PageRankStats {
        if values.is_empty() {
            return PageRankStats {
                total_nodes: 0,
                max_pagerank: 0.0,
                min_pagerank: 0.0,
                mean_pagerank: 0.0,
                median_pagerank: 0.0,
                std_deviation: 0.0,
                computation_time_ms,
                converged,
                iterations,
            };
        }

        let total_nodes = values.len() as u32;
        let max_pagerank = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_pagerank = values.iter().cloned().fold(f32::INFINITY, f32::min);
        let sum: f32 = values.iter().sum();
        let mean_pagerank = sum / total_nodes as f32;

        // Calculate median
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_pagerank = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        // Calculate standard deviation
        let variance: f32 = values
            .iter()
            .map(|&v| {
                let diff = v - mean_pagerank;
                diff * diff
            })
            .sum::<f32>()
            / total_nodes as f32;
        let std_deviation = variance.sqrt();

        PageRankStats {
            total_nodes,
            max_pagerank,
            min_pagerank,
            mean_pagerank,
            median_pagerank,
            std_deviation,
            computation_time_ms,
            converged,
            iterations,
        }
    }

    /// Extract top K nodes sorted by PageRank
    fn extract_top_nodes(&self, values: &[f32], k: usize) -> Vec<PageRankNode> {
        let mut nodes_with_values: Vec<(u32, f32)> = values
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx as u32, val))
            .collect();

        // Sort by PageRank descending
        nodes_with_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top K
        nodes_with_values
            .iter()
            .take(k.min(values.len()))
            .enumerate()
            .map(|(rank, &(node_id, pagerank))| PageRankNode {
                node_id,
                pagerank,
                rank: rank + 1,
            })
            .collect()
    }

    /// Get cached PageRank results
    fn get_cached_result(&self) -> Option<PageRankResult> {
        self.last_result.clone()
    }

    /// Clear cached results
    fn clear_cache(&mut self) {
        self.last_result = None;
    }
}

impl Default for PageRankActor {
    fn default() -> Self {
        Self::new()
    }
}

impl Actor for PageRankActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("PageRankActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("PageRankActor stopped");
    }
}

// Message handler for computing PageRank
impl Handler<ComputePageRank> for PageRankActor {
    type Result = ResponseActFuture<Self, Result<PageRankResult, String>>;

    fn handle(&mut self, msg: ComputePageRank, _ctx: &mut Context<Self>) -> Self::Result {
        info!("PageRankActor: Received ComputePageRank message");

        let params = msg.params.unwrap_or_default();

        // Get shared context before async boundary
        let shared_ctx = match &self.shared_context {
            Some(ctx) => Arc::clone(ctx),
            None => {
                return Box::pin(
                    async { Err("GPU context not initialized".to_string()) }
                        .into_actor(self)
                );
            }
        };

        // Create the async computation future
        let future = async move {
            let mut unified_compute = match shared_ctx
                .unified_compute
                .lock()
            {
                Ok(guard) => guard,
                Err(e) => return Err(format!("Failed to acquire GPU compute lock: {}", e)),
            };

            let start_time = Instant::now();

            // Extract parameters with defaults
            let damping = params.damping_factor.unwrap_or(0.85);
            let max_iter = params.max_iterations.unwrap_or(100) as usize;
            let epsilon = params.epsilon.unwrap_or(1e-6);
            let normalize = params.normalize.unwrap_or(true);
            let use_optimized = params.use_optimized.unwrap_or(true);

            // Call GPU PageRank computation
            let gpu_result = unified_compute
                .run_pagerank_centrality(damping, max_iter, epsilon, normalize, use_optimized)
                .map_err(|e| {
                    error!("GPU PageRank computation failed: {}", e);
                    format!("PageRank computation failed: {}", e)
                })?;

            let computation_time = start_time.elapsed();
            info!(
                "PageRankActor: PageRank computation completed in {:?}",
                computation_time
            );

            // Unpack GPU result and convert iterations to u32 for PageRankResult
            let (pagerank_values, iterations, converged, convergence_value) = gpu_result;
            let iterations = iterations as u32;

            Ok((pagerank_values, iterations, converged, convergence_value, computation_time))
        };

        // Use into_actor to re-enter actor context and finish processing
        Box::pin(
            future
                .into_actor(self)
                .map(|result, actor, _ctx| {
                    match result {
                        Ok((pagerank_values, iterations, converged, convergence_value, computation_time)) => {
                            // Compute statistics in actor context
                            let stats = actor.calculate_statistics(
                                &pagerank_values,
                                iterations,
                                converged,
                                computation_time.as_millis() as u64,
                            );

                            // Extract top K nodes (top 10 by default)
                            let top_nodes = actor.extract_top_nodes(&pagerank_values, 10);

                            let result = PageRankResult {
                                pagerank_values,
                                iterations,
                                converged,
                                convergence_value,
                                top_nodes,
                                stats,
                            };

                            // Cache the result
                            actor.last_result = Some(result.clone());
                            actor.gpu_state.record_utilization(0.8);

                            Ok(result)
                        }
                        Err(e) => Err(e),
                    }
                })
        )
    }
}

// Message handler for getting cached PageRank results
impl Handler<GetPageRankResult> for PageRankActor {
    type Result = MessageResult<GetPageRankResult>;

    fn handle(&mut self, _msg: GetPageRankResult, _ctx: &mut Context<Self>) -> Self::Result {
        MessageResult(self.get_cached_result())
    }
}

// Message handler for clearing PageRank cache
impl Handler<ClearPageRankCache> for PageRankActor {
    type Result = MessageResult<ClearPageRankCache>;

    fn handle(&mut self, _msg: ClearPageRankCache, _ctx: &mut Context<Self>) -> Self::Result {
        self.clear_cache();
        MessageResult(())
    }
}

// Message handler for updating GPU context
impl Handler<SetSharedGPUContext> for PageRankActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Context<Self>) -> Self::Result {
        info!("PageRankActor: Updating GPU context");
        self.shared_context = Some(msg.context);
        self.gpu_state.is_initialized = true;
        Ok(())
    }
}

// Message handler for initializing actor
impl Handler<InitializeActor> for PageRankActor {
    type Result = ();

    fn handle(&mut self, _msg: InitializeActor, _ctx: &mut Context<Self>) {
        info!("PageRankActor: Actor initialized");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_params_default() {
        let params = PageRankParams::default();
        assert_eq!(params.damping_factor, Some(0.85));
        assert_eq!(params.max_iterations, Some(100));
        assert!(params.epsilon.unwrap() < 1e-5);
        assert_eq!(params.normalize, Some(true));
    }

    #[test]
    fn test_extract_top_nodes() {
        let actor = PageRankActor::new();
        let values = vec![0.1, 0.5, 0.2, 0.8, 0.3];
        let top = actor.extract_top_nodes(&values, 3);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0].node_id, 3); // Index 3 has value 0.8
        assert_eq!(top[0].rank, 1);
        assert_eq!(top[1].node_id, 1); // Index 1 has value 0.5
        assert_eq!(top[1].rank, 2);
    }

    #[test]
    fn test_calculate_statistics() {
        let actor = PageRankActor::new();
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let stats = actor.calculate_statistics(&values, 10, true, 100);

        assert_eq!(stats.total_nodes, 5);
        assert_eq!(stats.max_pagerank, 0.5);
        assert_eq!(stats.min_pagerank, 0.1);
        assert!((stats.mean_pagerank - 0.3).abs() < 0.001);
        assert_eq!(stats.median_pagerank, 0.3);
        assert!(stats.converged);
        assert_eq!(stats.iterations, 10);
    }
}
