// Hybrid CPU-WASM/GPU SSSP Implementation
// Implements the "Breaking the Sorting Barrier" O(m log^(2/3) n) algorithm
// using CPU-WASM for recursive control and GPU for parallel relaxation

pub mod wasm_controller;
pub mod gpu_kernels;
pub mod communication_bridge;
pub mod adaptive_heap;


/// Hybrid SSSP configuration
#[derive(Debug, Clone)]
pub struct HybridSSPConfig {
    /// Use hybrid CPU-WASM/GPU implementation
    pub enable_hybrid: bool,

    /// Maximum recursion depth (log n / t)
    pub max_recursion_depth: u32,

    /// Pivot detection parameter k = log^(1/3)(n)
    pub pivot_k: u32,

    /// Recursion branching factor t = log^(2/3)(n)
    pub branching_t: u32,

    /// Use pinned memory for zero-copy transfers
    pub use_pinned_memory: bool,

    /// Enable performance profiling
    pub enable_profiling: bool,
}

impl Default for HybridSSPConfig {
    fn default() -> Self {
        Self {
            enable_hybrid: false,  // Default to traditional GPU-only
            max_recursion_depth: 10,
            pivot_k: 10,
            branching_t: 100,
            use_pinned_memory: true,
            enable_profiling: false,
        }
    }
}

/// Result of hybrid SSSP computation
#[derive(Debug)]
pub struct HybridSSPResult {
    /// Distance array for all vertices
    pub distances: Vec<f32>,

    /// Parent array for path reconstruction
    pub parents: Vec<i32>,

    /// Performance metrics
    pub metrics: SSPMetrics,
}

/// Performance metrics for analysis
#[derive(Debug, Default, Clone)]
pub struct SSPMetrics {
    /// Total execution time (ms)
    pub total_time_ms: f32,

    /// Time spent in CPU orchestration (ms)
    pub cpu_time_ms: f32,

    /// Time spent in GPU computation (ms)
    pub gpu_time_ms: f32,

    /// Time spent in CPU-GPU communication (ms)
    pub transfer_time_ms: f32,

    /// Number of recursion levels executed
    pub recursion_levels: u32,

    /// Total number of edge relaxations
    pub total_relaxations: u64,

    /// Number of pivots selected
    pub pivots_selected: u32,

    /// Achieved complexity (edges × log factor)
    pub complexity_factor: f32,
}

/// Main hybrid SSSP executor
pub struct HybridSSPExecutor {
    config: HybridSSPConfig,
    wasm_controller: Option<wasm_controller::WASMController>,
    gpu_bridge: communication_bridge::GPUBridge,
    metrics: SSPMetrics,
}

impl HybridSSPExecutor {
    /// Create new hybrid executor
    pub fn new(config: HybridSSPConfig) -> Self {
        Self {
            config: config.clone(),
            wasm_controller: None,
            gpu_bridge: communication_bridge::GPUBridge::new(config.use_pinned_memory),
            metrics: SSPMetrics::default(),
        }
    }

    /// Initialize WASM controller
    pub async fn initialize(&mut self) -> Result<(), String> {
        if self.config.enable_hybrid {
            self.wasm_controller = Some(
                wasm_controller::WASMController::new(&self.config).await?
            );
        }
        Ok(())
    }

    /// Execute hybrid SSSP algorithm
    pub async fn execute(
        &mut self,
        num_nodes: usize,
        num_edges: usize,
        sources: &[u32],
        csr_row_offsets: &[u32],
        csr_col_indices: &[u32],
        csr_weights: &[f32],
    ) -> Result<HybridSSPResult, String> {
        let start_time = std::time::Instant::now();

        // Calculate algorithm parameters
        let n = num_nodes as f32;
        let k = n.log2().cbrt().floor() as u32;  // log^(1/3)(n)
        let t = n.log2().powf(2.0/3.0).floor() as u32;  // log^(2/3)(n)
        let max_depth = ((n.log2() / t as f32).ceil() as u32).max(1);

        self.config.pivot_k = k;
        self.config.branching_t = t;
        self.config.max_recursion_depth = max_depth;

        log::info!(
            "Hybrid SSSP: n={}, m={}, k={}, t={}, max_depth={}, theoretical complexity=O(m·log^(2/3) n)=O({})",
            num_nodes, num_edges, k, t, max_depth,
            (num_edges as f32 * n.log2().powf(2.0/3.0)) as u64
        );

        let result = if self.config.enable_hybrid && self.wasm_controller.is_some() {
            // Execute hybrid CPU-WASM/GPU algorithm
            self.execute_hybrid(
                num_nodes,
                num_edges,
                sources,
                csr_row_offsets,
                csr_col_indices,
                csr_weights,
            ).await?
        } else {
            // Fallback to traditional GPU-only implementation
            self.execute_gpu_only(
                num_nodes,
                sources,
                csr_row_offsets,
                csr_col_indices,
                csr_weights,
            ).await?
        };

        self.metrics.total_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.metrics.complexity_factor =
            self.metrics.total_relaxations as f32 / (num_edges as f32 * n.log2().powf(2.0/3.0));

        if self.config.enable_profiling {
            self.log_performance_metrics();
        }

        Ok(HybridSSPResult {
            distances: result.0,
            parents: result.1,
            metrics: self.metrics.clone(),
        })
    }

    /// Execute hybrid CPU-WASM/GPU algorithm
    async fn execute_hybrid(
        &mut self,
        num_nodes: usize,
        num_edges: usize,
        sources: &[u32],
        csr_row_offsets: &[u32],
        csr_col_indices: &[u32],
        csr_weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        let controller = self.wasm_controller.as_mut()
            .ok_or("WASM controller not initialized")?;

        // Transfer graph to GPU (stays resident during execution)
        self.gpu_bridge.upload_graph(
            num_nodes,
            csr_row_offsets,
            csr_col_indices,
            csr_weights,
        ).await?;

        // Execute recursive BMSSP via WASM controller
        let (distances, parents) = controller.execute_bmssp(
            sources,
            num_nodes,
            &mut self.gpu_bridge,
            &mut self.metrics,
        ).await?;

        Ok((distances, parents))
    }

    /// Fallback to traditional GPU-only implementation
    async fn execute_gpu_only(
        &mut self,
        num_nodes: usize,
        sources: &[u32],
        csr_row_offsets: &[u32],
        csr_col_indices: &[u32],
        csr_weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        // This would call the existing GPU implementation
        // For now, returning placeholder
        let distances = vec![f32::INFINITY; num_nodes];
        let parents = vec![-1i32; num_nodes];

        // Set source distances to 0
        for &source in sources {
            if (source as usize) < num_nodes {
                // distances[source as usize] = 0.0;
            }
        }

        log::warn!("GPU-only SSSP not yet connected - using placeholder");
        Ok((distances, parents))
    }

    /// Log performance metrics for analysis
    fn log_performance_metrics(&self) {
        log::info!("=== Hybrid SSSP Performance Metrics ===");
        log::info!("Total time: {:.2} ms", self.metrics.total_time_ms);
        log::info!("  CPU orchestration: {:.2} ms ({:.1}%)",
            self.metrics.cpu_time_ms,
            100.0 * self.metrics.cpu_time_ms / self.metrics.total_time_ms
        );
        log::info!("  GPU computation: {:.2} ms ({:.1}%)",
            self.metrics.gpu_time_ms,
            100.0 * self.metrics.gpu_time_ms / self.metrics.total_time_ms
        );
        log::info!("  CPU-GPU transfer: {:.2} ms ({:.1}%)",
            self.metrics.transfer_time_ms,
            100.0 * self.metrics.transfer_time_ms / self.metrics.total_time_ms
        );
        log::info!("Recursion levels: {}", self.metrics.recursion_levels);
        log::info!("Pivots selected: {}", self.metrics.pivots_selected);
        log::info!("Total relaxations: {}", self.metrics.total_relaxations);
        log::info!("Complexity factor: {:.2}x theoretical", self.metrics.complexity_factor);
        log::info!("=====================================");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_calculation() {
        let n = 100000.0;
        let k = n.log2().cbrt().floor() as u32;
        let t = n.log2().powf(2.0/3.0).floor() as u32;
        let max_depth = ((n.log2() / t as f32).ceil() as u32).max(1);

        // For n=100,000: log2(n) ≈ 16.6
        // k = log^(1/3)(n) ≈ 2.5 → 2
        // t = log^(2/3)(n) ≈ 6.4 → 6
        // max_depth = log(n)/t ≈ 16.6/6 ≈ 2.8 → 3

        assert!(k >= 2 && k <= 3);
        assert!(t >= 6 && t <= 7);
        assert!(max_depth >= 2 && max_depth <= 3);
    }
}