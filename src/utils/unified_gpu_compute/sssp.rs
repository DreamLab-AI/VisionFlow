//! Single-Source Shortest Path (SSSP) and All-Pairs Shortest Path (APSP) algorithms.

use super::construction::UnifiedGPUCompute;
use anyhow::{anyhow, Result};
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer};

impl UnifiedGPUCompute {
    /// Run single-source shortest path from `source_idx`.
    ///
    /// * `delta` - When `Some(d)`, use delta-stepping with bucket width `d`.
    ///   The kernel processes edges whose tentative distance falls within the
    ///   current bucket boundary `B`, then advances `B` by `delta` until the
    ///   frontier is exhausted.  When `None`, `B = INFINITY` (classic
    ///   Bellman-Ford frontier behaviour).
    ///
    /// On success the computed distances are also retained on the GPU in
    /// `sssp_device_distances` so the force kernel can read them without an
    /// extra host-device round-trip.
    pub fn run_sssp(&mut self, source_idx: usize, delta: Option<f32>) -> Result<Vec<f32>> {
        self.sssp_available = false;

        let result = (|| -> Result<Vec<f32>> {
            // Initialise distances: source = 0, everything else = INF
            let mut host_dist = vec![f32::INFINITY; self.num_nodes];
            host_dist[source_idx] = 0.0;
            self.dist.copy_from(&host_dist)?;

            // Seed frontier with just the source node
            let mut frontier_host = vec![-1i32; self.num_nodes];
            frontier_host[0] = source_idx as i32;
            self.current_frontier.copy_from(&frontier_host)?;
            let mut frontier_len = 1usize;

            let s = self.sssp_stream.as_ref().unwrap_or(&self.stream);
            let max_iters = 10 * self.num_nodes.max(1);

            match delta {
                // ---- Delta-stepping: iterate over buckets [0,d), [d,2d), ... ----
                Some(d) if d > 0.0 && d < f32::INFINITY => {
                    let mut bucket_boundary = d;
                    let mut total_iters = 0usize;

                    // Outer loop: advance bucket boundary until no more work
                    while frontier_len > 0 {
                        // Inner loop: drain the current bucket
                        loop {
                            total_iters += 1;
                            if total_iters > max_iters {
                                log::warn!(
                                    "SSSP delta-stepping safety cap reached ({} iters, B={:.2})",
                                    total_iters,
                                    bucket_boundary,
                                );
                                frontier_len = 0;
                                break;
                            }

                            let zeros = vec![0i32; self.num_nodes];
                            self.next_frontier_flags.copy_from(&zeros)?;

                            let block = 256u32;
                            let grid = (frontier_len as u32 + block - 1) / block;

                            let func = self._module.get_function("relaxation_step_kernel")?;
                            // SAFETY: Same invariants as the original Bellman-Ford loop.
                            // B = bucket_boundary restricts relaxation to the current bucket.
                            unsafe {
                                launch!(func<<<grid, block, 0, s>>>(
                                    self.dist.as_device_ptr(),
                                    self.current_frontier.as_device_ptr(),
                                    frontier_len as i32,
                                    self.edge_row_offsets.as_device_ptr(),
                                    self.edge_col_indices.as_device_ptr(),
                                    self.edge_weights.as_device_ptr(),
                                    self.next_frontier_flags.as_device_ptr(),
                                    bucket_boundary,
                                    self.num_nodes as i32
                                ))?;
                            }

                            // Compact the frontier
                            let d_frontier_counter = DeviceBuffer::from_slice(&[0i32])?;
                            let compact_func = self._module.get_function("compact_frontier_kernel")?;
                            let compact_grid = ((self.num_nodes as u32 + 255) / 256, 1, 1);
                            let compact_block = (256, 1, 1);

                            // SAFETY: Same invariants as the original compact kernel launch.
                            unsafe {
                                launch!(compact_func<<<compact_grid, compact_block, 0, s>>>(
                                    self.next_frontier_flags.as_device_ptr(),
                                    self.current_frontier.as_device_ptr(),
                                    d_frontier_counter.as_device_ptr(),
                                    self.num_nodes as i32
                                ))?;
                            }

                            let mut new_frontier_size = [0i32; 1];
                            d_frontier_counter.copy_to(&mut new_frontier_size)?;
                            frontier_len = new_frontier_size[0] as usize;

                            if frontier_len == 0 {
                                // Current bucket drained -- advance to next bucket.
                                break;
                            }
                        }

                        if frontier_len == 0 {
                            // Advance bucket and re-seed frontier with nodes whose
                            // distance falls in [old_B, new_B).
                            bucket_boundary += d;

                            // Copy distances to host to build the next frontier.
                            // This is O(n) per bucket but delta-stepping has few buckets.
                            let mut tmp_dist = vec![0.0f32; self.num_nodes];
                            self.dist.copy_to(&mut tmp_dist)?;

                            let mut new_frontier = vec![-1i32; self.num_nodes];
                            let mut count = 0usize;
                            for (i, &dval) in tmp_dist.iter().enumerate() {
                                if dval >= (bucket_boundary - d)
                                    && dval < bucket_boundary
                                    && dval.is_finite()
                                {
                                    new_frontier[count] = i as i32;
                                    count += 1;
                                }
                            }

                            if count == 0 {
                                break; // No more reachable nodes in any future bucket
                            }

                            self.current_frontier.copy_from(&new_frontier)?;
                            frontier_len = count;
                        }
                    }
                }

                // ---- Classic Bellman-Ford: B = INFINITY ----
                _ => {
                    let mut iter_count = 0usize;
                    while frontier_len > 0 {
                        iter_count += 1;
                        if iter_count > max_iters {
                            log::warn!(
                                "SSSP safety cap reached ({} iters) with frontier_len={}",
                                iter_count,
                                frontier_len,
                            );
                            break;
                        }

                        let zeros = vec![0i32; self.num_nodes];
                        self.next_frontier_flags.copy_from(&zeros)?;

                        let block = 256u32;
                        let grid = (frontier_len as u32 + block - 1) / block;

                        let func = self._module.get_function("relaxation_step_kernel")?;
                        // SAFETY: Same invariants as documented in the original implementation.
                        unsafe {
                            launch!(func<<<grid, block, 0, s>>>(
                                self.dist.as_device_ptr(),
                                self.current_frontier.as_device_ptr(),
                                frontier_len as i32,
                                self.edge_row_offsets.as_device_ptr(),
                                self.edge_col_indices.as_device_ptr(),
                                self.edge_weights.as_device_ptr(),
                                self.next_frontier_flags.as_device_ptr(),
                                f32::INFINITY,
                                self.num_nodes as i32
                            ))?;
                        }

                        let d_frontier_counter = DeviceBuffer::from_slice(&[0i32])?;
                        let compact_func = self._module.get_function("compact_frontier_kernel")?;
                        let compact_grid = ((self.num_nodes as u32 + 255) / 256, 1, 1);
                        let compact_block = (256, 1, 1);

                        // SAFETY: Same invariants as documented in the original implementation.
                        unsafe {
                            launch!(compact_func<<<compact_grid, compact_block, 0, s>>>(
                                self.next_frontier_flags.as_device_ptr(),
                                self.current_frontier.as_device_ptr(),
                                d_frontier_counter.as_device_ptr(),
                                self.num_nodes as i32
                            ))?;
                        }

                        let mut new_frontier_size = [0i32; 1];
                        d_frontier_counter.copy_to(&mut new_frontier_size)?;
                        frontier_len = new_frontier_size[0] as usize;
                    }
                }
            }

            // Copy final distances to host
            self.dist.copy_to(&mut host_dist)?;

            // Persist a device-side copy for the force kernel to read via d_sssp_dist.
            // We clone into a separate buffer so that self.dist can be reused for the
            // next SSSP run without corrupting the force kernel's input.
            let mut sssp_buf = DeviceBuffer::zeroed(self.num_nodes)?;
            sssp_buf.copy_from(&host_dist)?;
            self.sssp_device_distances = Some(sssp_buf);

            Ok(host_dist)
        })();

        match result {
            Ok(distances) => {
                self.sssp_available = true;
                log::info!("SSSP computation successful from source {}", source_idx);
                Ok(distances)
            }
            Err(e) => {
                self.sssp_available = false;
                self.sssp_device_distances = None;
                log::error!("SSSP computation failed: {}. State invalidated.", e);
                Err(e)
            }
        }
    }

    /// Batched SSSP: runs SSSP from multiple sources while keeping the graph CSR
    /// on device. Only copies distance results back at the end, avoiding redundant
    /// host-device transfers of the graph structure between calls.
    pub fn run_sssp_batch(&mut self, sources: &[usize]) -> Result<Vec<Vec<f32>>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.num_nodes;
        let mut all_distances: Vec<Vec<f32>> = Vec::with_capacity(sources.len());

        // The CSR (edge_row_offsets, edge_col_indices, edge_weights) stays on device
        // across all iterations. We only reset dist/frontier per source and copy
        // the distance result back after each BFS completes.
        let s = self.sssp_stream.as_ref().unwrap_or(&self.stream);

        for &source_idx in sources {
            if source_idx >= n {
                return Err(anyhow!(
                    "Source index {} out of range (num_nodes = {})",
                    source_idx,
                    n
                ));
            }

            // Initialize distance buffer: infinity everywhere, 0 at source
            let mut host_dist = vec![f32::INFINITY; n];
            host_dist[source_idx] = 0.0;
            self.dist.copy_from(&host_dist)?;

            // Initialize frontier with just the source node
            let mut frontier_host = vec![-1i32; n];
            frontier_host[0] = source_idx as i32;
            self.current_frontier.copy_from(&frontier_host)?;
            let mut frontier_len = 1usize;

            let mut iter_count = 0usize;
            let max_iters = 10 * n.max(1);

            while frontier_len > 0 {
                iter_count += 1;
                if iter_count > max_iters {
                    log::warn!(
                        "SSSP batch safety cap reached ({} iters) for source {}",
                        iter_count,
                        source_idx
                    );
                    break;
                }

                // Clear next frontier flags
                let zeros = vec![0i32; n];
                self.next_frontier_flags.copy_from(&zeros)?;

                // Launch relaxation kernel
                let block = 256;
                let grid = ((frontier_len as u32 + block - 1) / block) as u32;

                let func = self._module.get_function("relaxation_step_kernel")?;
                // SAFETY: Same invariants as run_sssp - all buffers valid, bounds checked
                unsafe {
                    launch!(func<<<grid, block, 0, s>>>(
                        self.dist.as_device_ptr(),
                        self.current_frontier.as_device_ptr(),
                        frontier_len as i32,
                        self.edge_row_offsets.as_device_ptr(),
                        self.edge_col_indices.as_device_ptr(),
                        self.edge_weights.as_device_ptr(),
                        self.next_frontier_flags.as_device_ptr(),
                        f32::INFINITY,
                        self.num_nodes as i32
                    ))?;
                }

                // Compact frontier
                let d_frontier_counter = DeviceBuffer::from_slice(&[0i32])?;
                let compact_func = self._module.get_function("compact_frontier_kernel")?;
                let compact_grid = ((self.num_nodes as u32 + 255) / 256, 1, 1);
                let compact_block = (256, 1, 1);

                // SAFETY: Same invariants as run_sssp compact step
                unsafe {
                    launch!(compact_func<<<compact_grid, compact_block, 0, s>>>(
                        self.next_frontier_flags.as_device_ptr(),
                        self.current_frontier.as_device_ptr(),
                        d_frontier_counter.as_device_ptr(),
                        self.num_nodes as i32
                    ))?;
                }

                let mut new_frontier_size = vec![0i32; 1];
                d_frontier_counter.copy_to(&mut new_frontier_size)?;
                frontier_len = new_frontier_size[0] as usize;
            }

            // Copy distances back for this source only (graph CSR stays on device)
            self.dist.copy_to(&mut host_dist)?;
            all_distances.push(host_dist);
        }

        self.sssp_available = true;
        log::info!(
            "Batched SSSP completed for {} sources ({} nodes each)",
            sources.len(),
            n
        );
        Ok(all_distances)
    }

    /// Launch the GPU approximate_apsp_kernel from gpu_landmark_apsp.cu.
    /// Takes flattened landmark distances [num_landmarks][num_nodes] and produces
    /// the full [num_nodes][num_nodes] approximate distance matrix on GPU.
    pub fn run_apsp_gpu(
        &self,
        landmark_distances: &[f32],
        num_landmarks: usize,
    ) -> Result<Vec<f32>> {
        let apsp_mod = self
            .apsp_module
            .as_ref()
            .ok_or_else(|| anyhow!("APSP module not loaded - GPU APSP unavailable"))?;

        let n = self.num_nodes;
        if landmark_distances.len() != num_landmarks * n {
            return Err(anyhow!(
                "landmark_distances length ({}) != num_landmarks ({}) * num_nodes ({})",
                landmark_distances.len(),
                num_landmarks,
                n
            ));
        }

        // Pre-check: ensure APSP matrix won't exceed GPU memory limits.
        // An n*n f32 matrix can grow quadratically; cap at 4 GB to prevent OOM.
        let required_bytes = (n as u64) * (n as u64) * (std::mem::size_of::<f32>() as u64);
        const MAX_APSP_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GB limit
        if required_bytes > MAX_APSP_BYTES {
            return Err(anyhow!(
                "APSP matrix too large: {} nodes requires {} GB, max is {} GB",
                n,
                required_bytes / (1024 * 1024 * 1024),
                MAX_APSP_BYTES / (1024 * 1024 * 1024)
            ));
        }

        // Upload landmark distances to device
        let d_landmark = DeviceBuffer::from_slice(landmark_distances)?;

        // Allocate output distance matrix on device
        let d_output: DeviceBuffer<f32> = DeviceBuffer::zeroed(n * n)?;

        // Launch 2D grid: each thread computes one (i,j) pair
        let block_dim = 16u32;
        let grid_x = ((n as u32) + block_dim - 1) / block_dim;
        let grid_y = ((n as u32) + block_dim - 1) / block_dim;

        let func = apsp_mod.get_function("approximate_apsp_kernel")?;
        let s = self.sssp_stream.as_ref().unwrap_or(&self.stream);

        // SAFETY: approximate_apsp_kernel launch is safe because:
        // 1. d_landmark is a valid device buffer of size [num_landmarks * num_nodes]
        // 2. d_output is a valid zeroed device buffer of size [num_nodes * num_nodes]
        // 3. Grid/block dimensions produce threads covering all (i,j) pairs
        // 4. The kernel reads landmark_distances and writes distance_matrix with bounds checks
        unsafe {
            launch!(func<<<(grid_x, grid_y, 1), (block_dim, block_dim, 1), 0, s>>>(
                d_landmark.as_device_ptr(),
                d_output.as_device_ptr(),
                n as i32,
                num_landmarks as i32
            ))?;
        }

        // Copy result back to host
        let mut host_output = vec![0.0f32; n * n];
        d_output.copy_to(&mut host_output)?;

        log::info!(
            "GPU APSP kernel completed: {} nodes, {} landmarks",
            n,
            num_landmarks
        );
        Ok(host_output)
    }
}
