// WASM Controller for Recursive BMSSP Orchestration
// Handles the complex recursive structure that doesn't map well to GPU

use super::{HybridSSPConfig, SSPMetrics};
use super::communication_bridge::GPUBridge;
use super::adaptive_heap::AdaptiveHeap;
use std::collections::VecDeque;

/// WASM Controller for recursive BMSSP algorithm
pub struct WASMController {
    config: HybridSSPConfig,
    adaptive_heap: AdaptiveHeap,
    recursion_stack: Vec<RecursionFrame>,
}

/// Frame for managing recursion state
struct RecursionFrame {
    level: u32,
    bound: f32,
    frontier: Vec<u32>,
    pivots: Vec<u32>,
    subproblem_id: u32,
}

impl WASMController {
    /// Create new WASM controller
    pub async fn new(config: &HybridSSPConfig) -> Result<Self, String> {
        Ok(Self {
            config: config.clone(),
            adaptive_heap: AdaptiveHeap::new(1000000), // 1M elements capacity
            recursion_stack: Vec::with_capacity(config.max_recursion_depth as usize),
        })
    }

    /// Execute recursive BMSSP algorithm
    pub async fn execute_bmssp(
        &mut self,
        sources: &[u32],
        num_nodes: usize,
        gpu_bridge: &mut GPUBridge,
        metrics: &mut SSPMetrics,
    ) -> Result<(Vec<f32>, Vec<i32>), String> {
        let start_time = std::time::Instant::now();

        // Initialize distances and parents
        let mut distances = vec![f32::INFINITY; num_nodes];
        let mut parents = vec![-1i32; num_nodes];

        // Set source distances
        for &source in sources {
            distances[source as usize] = 0.0;
        }

        // Initial frontier is the source vertices
        let initial_frontier: Vec<u32> = sources.to_vec();

        // Execute recursive BMSSP with initial bound B = ∞
        // Use iterative approach to avoid recursive async issues
        self.bmssp_iterative(
            f32::INFINITY,  // Initial bound
            initial_frontier,
            &mut distances,
            &mut parents,
            gpu_bridge,
            metrics,
        ).await?;

        metrics.cpu_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        Ok((distances, parents))
    }

    /// Iterative BMSSP implementation to avoid recursive async issues
    async fn bmssp_iterative(
        &mut self,
        initial_bound: f32,
        initial_frontier: Vec<u32>,
        distances: &mut [f32],
        parents: &mut [i32],
        gpu_bridge: &mut GPUBridge,
        metrics: &mut SSPMetrics,
    ) -> Result<(), String> {
        // Use a work queue to simulate recursion iteratively
        let mut work_queue = VecDeque::new();

        // Start with the initial problem
        work_queue.push_back((
            self.config.max_recursion_depth,  // Start at max level
            initial_bound,
            initial_frontier,
        ));

        while let Some((level, bound, frontier)) = work_queue.pop_front() {
            metrics.recursion_levels = metrics.recursion_levels.max(
                self.config.max_recursion_depth - level + 1
            );

            // Base case: level 0 or empty frontier - use GPU Dijkstra
            if level == 0 || frontier.is_empty() {
                self.base_case_gpu_dijkstra(
                    bound,
                    frontier,
                    distances,
                    parents,
                    gpu_bridge,
                    metrics,
                ).await?;
                continue;
            }

            log::debug!("BMSSP Level {}: frontier_size={}, bound={}",
                level, frontier.len(), bound);

            // Step 1: FindPivots - identify pivot vertices
            let pivots = self.find_pivots(
                &frontier,
                distances,
                gpu_bridge,
                metrics,
            ).await?;

            metrics.pivots_selected += pivots.len() as u32;

            // Step 2: Partition frontier based on pivots
            let partitions = self.partition_frontier(
                &frontier,
                &pivots,
                distances,
            );

            // Step 3: Add subproblems to work queue (simulating recursive calls)
            let t = self.config.branching_t as usize;
            let num_partitions = partitions.len().min(t);

            // Add partitions in reverse order so they're processed in correct order
            for (i, partition) in partitions.into_iter()
                .take(num_partitions)
                .enumerate()
                .rev()
            {
                if !partition.is_empty() {
                    // Calculate sub-bound for this partition
                    let sub_bound = bound / (2_f32.powi(i as i32));

                    // Add to work queue (simulating recursive call)
                    work_queue.push_front((
                        level - 1,
                        sub_bound,
                        partition,
                    ));
                }
            }
        }

        Ok(())
    }

    /// FindPivots algorithm (Algorithm 1 from paper)
    async fn find_pivots(
        &mut self,
        frontier: &[u32],
        distances: &[f32],
        gpu_bridge: &mut GPUBridge,
        metrics: &mut SSPMetrics,
    ) -> Result<Vec<u32>, String> {
        let k = self.config.pivot_k;

        // Step 1: Perform k-step relaxation from frontier on GPU
        let (temp_distances, spt_sizes) = gpu_bridge.k_step_relaxation(
            frontier,
            distances,
            k,
            metrics,
        ).await?;

        // Step 2: Identify vertices with SPT size >= k
        let mut pivots = Vec::new();
        for (vertex, &spt_size) in spt_sizes.iter().enumerate() {
            if spt_size >= k && temp_distances[vertex] < f32::INFINITY {
                pivots.push(vertex as u32);
            }
        }

        // Step 3: Ensure at most |frontier|/k pivots
        let max_pivots = ((frontier.len() as f32) / (k as f32)).ceil() as usize;
        if pivots.len() > max_pivots {
            // Sort by SPT size and take top pivots
            pivots.sort_by_key(|&v| std::cmp::Reverse(spt_sizes[v as usize]));
            pivots.truncate(max_pivots);
        }

        log::debug!("FindPivots: frontier_size={}, k={}, pivots_found={}",
            frontier.len(), k, pivots.len());

        Ok(pivots)
    }

    /// Partition frontier based on pivots
    fn partition_frontier(
        &self,
        frontier: &[u32],
        pivots: &[u32],
        distances: &[f32],
    ) -> Vec<Vec<u32>> {
        let t = self.config.branching_t as usize;
        let mut partitions = vec![Vec::new(); t];

        if pivots.is_empty() {
            // No pivots: put all in first partition
            partitions[0] = frontier.to_vec();
            return partitions;
        }

        // Assign each frontier vertex to nearest pivot's partition
        for &vertex in frontier {
            let mut min_dist = f32::INFINITY;
            let mut best_partition = 0;

            for (i, &pivot) in pivots.iter().enumerate() {
                // Use graph distance if available, otherwise use vertex ID difference
                let dist = if distances[vertex as usize] < f32::INFINITY &&
                              distances[pivot as usize] < f32::INFINITY {
                    (distances[vertex as usize] - distances[pivot as usize]).abs()
                } else {
                    (vertex as f32 - pivot as f32).abs()
                };

                if dist < min_dist {
                    min_dist = dist;
                    best_partition = (i % t);
                }
            }

            partitions[best_partition].push(vertex);
        }

        // Remove empty partitions
        partitions.retain(|p| !p.is_empty());
        partitions
    }

    /// Base case: Use GPU Dijkstra for small subproblems
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

        log::debug!("Base case GPU Dijkstra: frontier_size={}, bound={}",
            frontier.len(), bound);

        // Execute bounded Dijkstra on GPU
        let (new_distances, new_parents, relaxations) = gpu_bridge.bounded_dijkstra(
            &frontier,
            distances,
            bound,
            metrics,
        ).await?;

        // Update distances and parents for affected vertices
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

/// Helper functions for WASM compilation
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
            // Initialize WASM controller
            let config = super::HybridSSPConfig::default();
            Ok(WASMSSPSolver {
                controller: super::WASMController::new(&config).await
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
            // This would connect to the GPU bridge via JavaScript
            // For now, returning placeholder
            Ok(JsValue::from_str(&format!(
                "WASM SSSP solver ready for {} nodes with {} sources",
                num_nodes, sources.len()
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

        // Should create t partitions
        assert!(!partitions.is_empty());

        // All frontier vertices should be assigned
        let total: usize = partitions.iter().map(|p| p.len()).sum();
        assert_eq!(total, frontier.len());
    }
}