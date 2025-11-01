// Hybrid CPU-WASM/GPU SSSP Implementation
// Implements the "Breaking the Sorting Barrier" O(m log^(2/3) n) algorithm
// using CPU-WASM for recursive control and GPU for parallel relaxation

pub mod adaptive_heap;
pub mod communication_bridge;
pub mod gpu_kernels;
pub mod wasm_controller;

/
#[derive(Debug, Clone)]
pub struct HybridSSPConfig {
    
    pub enable_hybrid: bool,

    
    pub max_recursion_depth: u32,

    
    pub pivot_k: u32,

    
    pub branching_t: u32,

    
    pub use_pinned_memory: bool,

    
    pub enable_profiling: bool,
}

impl Default for HybridSSPConfig {
    fn default() -> Self {
        Self {
            enable_hybrid: false, 
            max_recursion_depth: 10,
            pivot_k: 10,
            branching_t: 100,
            use_pinned_memory: true,
            enable_profiling: false,
        }
    }
}

/
#[derive(Debug)]
pub struct HybridSSPResult {
    
    pub distances: Vec<f32>,

    
    pub parents: Vec<i32>,

    
    pub metrics: SSPMetrics,
}

/
#[derive(Debug, Default, Clone)]
pub struct SSPMetrics {
    
    pub total_time_ms: f32,

    
    pub cpu_time_ms: f32,

    
    pub gpu_time_ms: f32,

    
    pub transfer_time_ms: f32,

    
    pub recursion_levels: u32,

    
    pub total_relaxations: u64,

    
    pub pivots_selected: u32,

    
    pub complexity_factor: f32,
}

/
pub struct HybridSSPExecutor {
    config: HybridSSPConfig,
    wasm_controller: Option<wasm_controller::WASMController>,
    gpu_bridge: communication_bridge::GPUBridge,
    metrics: SSPMetrics,
}

impl HybridSSPExecutor {
    
    pub fn new(config: HybridSSPConfig) -> Self {
        Self {
            config: config.clone(),
            wasm_controller: None,
            gpu_bridge: communication_bridge::GPUBridge::new(config.use_pinned_memory),
            metrics: SSPMetrics::default(),
        }
    }

    
    pub async fn initialize(&mut self) -> Result<(), String> {
        if self.config.enable_hybrid {
            self.wasm_controller = Some(wasm_controller::WASMController::new(&self.config).await?);
        }
        Ok(())
    }

    
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

        
        let n = num_nodes as f32;
        let k = n.log2().cbrt().floor() as u32; 
        let t = n.log2().powf(2.0 / 3.0).floor() as u32; 
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
            
            self.execute_hybrid(
                num_nodes,
                num_edges,
                sources,
                csr_row_offsets,
                csr_col_indices,
                csr_weights,
            )
            .await?
        } else {
            
            self.execute_gpu_only(
                num_nodes,
                sources,
                csr_row_offsets,
                csr_col_indices,
                csr_weights,
            )
            .await?
        };

        self.metrics.total_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.metrics.complexity_factor =
            self.metrics.total_relaxations as f32 / (num_edges as f32 * n.log2().powf(2.0 / 3.0));

        if self.config.enable_profiling {
            self.log_performance_metrics();
        }

        Ok(HybridSSPResult {
            distances: result.0,
            parents: result.1,
            metrics: self.metrics.clone(),
        })
    }

    
    async fn execute_hybrid(
        &mut self,
        num_nodes: usize,
        _num_edges: usize,
        sources: &[u32],
        csr_row_offsets: &[u32],
        csr_col_indices: &[u32],
        csr_weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        let controller = self
            .wasm_controller
            .as_mut()
            .ok_or("WASM controller not initialized")?;

        
        self.gpu_bridge
            .upload_graph(num_nodes, csr_row_offsets, csr_col_indices, csr_weights)
            .await?;

        
        let (distances, parents) = controller
            .execute_bmssp(sources, num_nodes, &mut self.gpu_bridge, &mut self.metrics)
            .await?;

        Ok((distances, parents))
    }

    
    async fn execute_gpu_only(
        &mut self,
        num_nodes: usize,
        sources: &[u32],
        _csr_row_offsets: &[u32],
        _csr_col_indices: &[u32],
        _csr_weights: &[f32],
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        
        
        let distances = vec![f32::INFINITY; num_nodes];
        let parents = vec![-1i32; num_nodes];

        
        for &source in sources {
            if (source as usize) < num_nodes {
                
            }
        }

        log::warn!("GPU-only SSSP not yet connected - using placeholder");
        Ok((distances, parents))
    }

    
    fn log_performance_metrics(&self) {
        log::info!("=== Hybrid SSSP Performance Metrics ===");
        log::info!("Total time: {:.2} ms", self.metrics.total_time_ms);
        log::info!(
            "  CPU orchestration: {:.2} ms ({:.1}%)",
            self.metrics.cpu_time_ms,
            100.0 * self.metrics.cpu_time_ms / self.metrics.total_time_ms
        );
        log::info!(
            "  GPU computation: {:.2} ms ({:.1}%)",
            self.metrics.gpu_time_ms,
            100.0 * self.metrics.gpu_time_ms / self.metrics.total_time_ms
        );
        log::info!(
            "  CPU-GPU transfer: {:.2} ms ({:.1}%)",
            self.metrics.transfer_time_ms,
            100.0 * self.metrics.transfer_time_ms / self.metrics.total_time_ms
        );
        log::info!("Recursion levels: {}", self.metrics.recursion_levels);
        log::info!("Pivots selected: {}", self.metrics.pivots_selected);
        log::info!("Total relaxations: {}", self.metrics.total_relaxations);
        log::info!(
            "Complexity factor: {:.2}x theoretical",
            self.metrics.complexity_factor
        );
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
        let t = n.log2().powf(2.0 / 3.0).floor() as u32;
        let max_depth = ((n.log2() / t as f32).ceil() as u32).max(1);

        
        
        
        

        assert!(k >= 2 && k <= 3);
        assert!(t >= 6 && t <= 7);
        assert!(max_depth >= 2 && max_depth <= 3);
    }
}
