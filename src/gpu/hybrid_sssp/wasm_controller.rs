// WASM Controller for Recursive BMSSP Orchestration
// Handles the complex recursive structure that doesn't map well to GPU

use super::adaptive_heap::AdaptiveHeap;
use super::communication_bridge::GPUBridge;
use super::{HybridSSPConfig, SSPMetrics};
use std::collections::VecDeque;

///
pub struct WASMController {
    config: HybridSSPConfig,
    adaptive_heap: AdaptiveHeap,
    recursion_stack: Vec<RecursionFrame>,
}

///
struct RecursionFrame {
    level: u32,
    bound: f32,
    frontier: Vec<u32>,
    pivots: Vec<u32>,
    subproblem_id: u32,
}

impl WASMController {
    
    pub async fn new(config: &HybridSSPConfig) -> Result<Self, String> {
        Ok(Self {
            config: config.clone(),
            adaptive_heap: AdaptiveHeap::new(1000000), 
            recursion_stack: Vec::with_capacity(config.max_recursion_depth as usize),
        })
    }

    
    pub async fn execute_bmssp(
        &mut self,
        sources: &[u32],
        num_nodes: usize,
        gpu_bridge: &mut GPUBridge,
        metrics: &mut SSPMetrics,
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        let start_time = std::time::Instant::now();

        
        let mut distances = vec![f32::INFINITY; num_nodes];
        let mut parents = vec![-1i32; num_nodes];

        
        for &source in sources {
            distances[source as usize] = 0.0;
        }

        
        let initial_frontier: Vec<u32> = sources.to_vec();

        
        
        self.bmssp_iterative(
            f32::INFINITY, 
            initial_frontier,
            &mut distances,
            &mut parents,
            gpu_bridge,
            metrics,
        )
        .await?;

        metrics.cpu_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        Ok((distances, parents))
    }

    
    async fn bmssp_iterative(
        &mut self,
        initial_bound: f32,
        initial_frontier: Vec<u32>,
        distances: &mut [f32],
        parents: &mut [i32],
        gpu_bridge: &mut GPUBridge,
        metrics: &mut SSPMetrics,
    ) -> Result<(), String> {
        
        let mut work_queue = VecDeque::new();

        
        work_queue.push_back((
            self.config.max_recursion_depth, 
            initial_bound,
            initial_frontier,
        ));

        while let Some((level, bound, frontier)) = work_queue.pop_front() {
            metrics.recursion_levels = metrics
                .recursion_levels
                .max(self.config.max_recursion_depth - level + 1);

            
            if level == 0 || frontier.is_empty() {
                self.base_case_gpu_dijkstra(
                    bound, frontier, distances, parents, gpu_bridge, metrics,
                )
                .await?;
                continue;
            }

            log::debug!(
                "BMSSP Level {}: frontier_size={}, bound={}",
                level,
                frontier.len(),
                bound
            );

            
            let pivots = self
                .find_pivots(&frontier, distances, gpu_bridge, metrics)
                .await?;

            metrics.pivots_selected += pivots.len() as u32;

            
            let partitions = self.partition_frontier(&frontier, &pivots, distances);

            
            let t = self.config.branching_t as usize;
            let num_partitions = partitions.len().min(t);

            
            for (i, partition) in partitions
                .into_iter()
                .take(num_partitions)
                .enumerate()
                .rev()
            {
                if !partition.is_empty() {
                    
                    let sub_bound = bound / (2_f32.powi(i as i32));

                    
                    work_queue.push_front((level - 1, sub_bound, partition));
                }
            }
        }

        Ok(())
    }

    
    async fn find_pivots(
        &mut self,
        frontier: &[u32],
        distances: &[f32],
        gpu_bridge: &mut GPUBridge,
        metrics: &mut SSPMetrics,
    ) -> Result<Vec<u32>, String> {
        let k = self.config.pivot_k;

        
        let (temp_distances, spt_sizes) = gpu_bridge
            .k_step_relaxation(frontier, distances, k, metrics)
            .await?;

        
        let mut pivots = Vec::new();
        for (vertex, &spt_size) in spt_sizes.iter().enumerate() {
            if spt_size >= k && temp_distances[vertex] < f32::INFINITY {
                pivots.push(vertex as u32);
            }
        }

        
        let max_pivots = ((frontier.len() as f32) / (k as f32)).ceil() as usize;
        if pivots.len() > max_pivots {
            
            pivots.sort_by_key(|&v| std::cmp::Reverse(spt_sizes[v as usize]));
            pivots.truncate(max_pivots);
        }

        log::debug!(
            "FindPivots: frontier_size={}, k={}, pivots_found={}",
            frontier.len(),
            k,
            pivots.len()
        );

        Ok(pivots)
    }

    
    fn partition_frontier(
        &self,
        frontier: &[u32],
        pivots: &[u32],
        distances: &[f32],
    ) -> Vec<Vec<u32>> {
        let t = self.config.branching_t as usize;
        let mut partitions = vec![Vec::new(); t];

        if pivots.is_empty() {
            
            partitions[0] = frontier.to_vec();
            return partitions;
        }

        
        for &vertex in frontier {
            let mut min_dist = f32::INFINITY;
            let mut best_partition = 0;

            for (i, &pivot) in pivots.iter().enumerate() {
                
                let dist = if distances[vertex as usize] < f32::INFINITY
                    && distances[pivot as usize] < f32::INFINITY
                {
                    (distances[vertex as usize] - distances[pivot as usize]).abs()
                } else {
                    (vertex as f32 - pivot as f32).abs()
                };

                if dist < min_dist {
                    min_dist = dist;
                    best_partition = i % t;
                }
            }

            partitions[best_partition].push(vertex);
        }

        
        partitions.retain(|p| !p.is_empty());
        partitions
    }

    
    async fn base_case_gpu_dijkstra(
        &mut self,
        bound: f32,
        frontier: Vec<u32>,
        distances: &mut [f32],
        parents: &mut [i32],
        gpu_bridge: &mut GPUBridge,
        metrics: &mut SSPMetrics,
    ) -> Result<(), String> {
        if frontier.is_empty() {
            return Ok(());
        }

        log::debug!(
            "Base case GPU Dijkstra: frontier_size={}, bound={}",
            frontier.len(),
            bound
        );

        
        let (new_distances, new_parents, relaxations) = gpu_bridge
            .bounded_dijkstra(&frontier, distances, bound, metrics)
            .await?;

        
        for i in 0..distances.len() {
            if new_distances[i] < distances[i] {
                distances[i] = new_distances[i];
                parents[i] = new_parents[i];
            }
        }

        metrics.total_relaxations += relaxations;
        Ok(())
    }
}

///
#[cfg(target_arch = "wasm32")]
mod wasm_helpers {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WASMSSPSolver {
        controller: super::WASMController,
    }

    #[wasm_bindgen]
    impl WASMSSPSolver {
        #[wasm_bindgen(constructor)]
        pub async fn new() -> Result<WASMSSPSolver, JsValue> {
            
            let config = super::HybridSSPConfig::default();
            Ok(WASMSSPSolver {
                controller: super::WASMController::new(&config)
                    .await
                    .map_err(|e| JsValue::from_str(&e))?,
            })
        }

        #[wasm_bindgen]
        pub async fn solve_sssp(
            &mut self,
            sources: Vec<u32>,
            num_nodes: usize,
            _row_offsets: Vec<u32>,
            _col_indices: Vec<u32>,
            _weights: Vec<f32>,
        ) -> Result<JsValue, JsValue> {
            
            
            Ok(JsValue::from_str(&format!(
                "WASM SSSP solver ready for {} nodes with {} sources",
                num_nodes,
                sources.len()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_controller_creation() {
        let config = HybridSSPConfig::default();
        let controller = WASMController::new(&config).await;
        assert!(controller.is_ok());
    }

    #[test]
    fn test_frontier_partitioning() {
        let config = HybridSSPConfig::default();
        let controller = WASMController {
            config,
            adaptive_heap: AdaptiveHeap::new(100),
            recursion_stack: Vec::new(),
        };

        let frontier = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let pivots = vec![2, 5, 8];
        let distances = vec![0.0; 10];

        let partitions = controller.partition_frontier(&frontier, &pivots, &distances);

        
        assert!(!partitions.is_empty());

        
        let total: usize = partitions.iter().map(|p| p.len()).sum();
        assert_eq!(total, frontier.len());
    }
}
