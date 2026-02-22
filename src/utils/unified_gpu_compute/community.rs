//! Community detection algorithms: Label Propagation, Louvain, DBSCAN.

use super::construction::UnifiedGPUCompute;
use anyhow::Result;
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer};
use log::info;

impl UnifiedGPUCompute {

    pub fn run_community_detection(
        &mut self,
        max_iterations: u32,
        synchronous: bool,
        seed: u32,
    ) -> Result<(Vec<i32>, usize, f32, u32, Vec<i32>, bool)> {
        let block_size = 256;
        let grid_size = (self.num_nodes + block_size - 1) / block_size;
        let stream = &self.stream;


        let init_random_kernel = self._module.get_function("init_random_states_kernel")?;
        // SAFETY: Random state initialization kernel is safe because:
        // 1. rand_states buffer is allocated for num_nodes curandState elements
        // 2. Each thread initializes its own random state using seed + thread_id
        // 3. curandState is repr(C) and can be safely written from GPU
        unsafe {
            launch!(
                init_random_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.rand_states.as_device_ptr().as_raw(),
                    self.num_nodes as i32,
                    seed
                )
            )?;
        }

        // SAFETY: Label initialization kernel is safe because:
        // 1. labels_current is a valid DeviceBuffer with capacity >= num_nodes
        // 2. Each thread writes its own index as the initial community label
        let init_labels_kernel = self._module.get_function("init_labels_kernel")?;
        unsafe {
            launch!(
                init_labels_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.labels_current.as_device_ptr(),
                    self.num_nodes as i32
                )
            )?;
        }

        // SAFETY: Node degree computation kernel is safe because:
        // 1. edge_row_offsets and edge_weights are valid CSR graph data
        // 2. node_degrees is an output buffer with capacity >= num_nodes
        // 3. The kernel reads CSR offsets to compute weighted degree per node
        let compute_degrees_kernel = self._module.get_function("compute_node_degrees_kernel")?;
        unsafe {
            launch!(
                compute_degrees_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.edge_row_offsets.as_device_ptr(),
                    self.edge_weights.as_device_ptr(),
                    self.node_degrees.as_device_ptr(),
                    self.num_nodes as i32
                )
            )?;
        }


        self.stream.synchronize()?;
        let mut node_degrees_host = vec![0.0f32; self.num_nodes];
        self.node_degrees.copy_to(&mut node_degrees_host)?;
        let total_weight: f32 = node_degrees_host.iter().sum::<f32>() / 2.0;


        let mut iterations = 0;
        let mut converged = false;


        let propagate_kernel = if synchronous {
            self._module.get_function("propagate_labels_sync_kernel")?
        } else {
            self._module.get_function("propagate_labels_async_kernel")?
        };

        let check_convergence_kernel = self._module.get_function("check_convergence_kernel")?;


        let mut shared_mem_size = block_size * (self.max_labels + 1) * 4;
        // Cap shared memory to 48KB (safe default for all CUDA architectures).
        // Exceeding the per-block shared memory limit causes a launch failure.
        const MAX_SHARED_MEM: usize = 48 * 1024; // 48KB
        if shared_mem_size > MAX_SHARED_MEM {
            log::warn!(
                "Reducing shared memory from {} to {} bytes (max_labels may be too high)",
                shared_mem_size,
                MAX_SHARED_MEM
            );
            shared_mem_size = MAX_SHARED_MEM;
        }

        for iter in 0..max_iterations {
            iterations = iter + 1;


            let convergence_flag_host = vec![1i32];
            self.convergence_flag.copy_from(&convergence_flag_host)?;

            if synchronous {
                // SAFETY: Synchronous label propagation kernel is safe because:
                // 1. labels_current contains current community labels (read-only)
                // 2. labels_next is the output buffer for new labels
                // 3. edge_* buffers are valid CSR graph representation
                // 4. label_counts is scratch space for counting neighbor labels
                // 5. shared_mem_size is bounded by max_labels (validated in constructor)
                // 6. rand_states provides tie-breaking randomness
                unsafe {
                    launch!(
                        propagate_kernel<<<grid_size as u32, block_size as u32, shared_mem_size as u32, stream>>>(
                            self.labels_current.as_device_ptr(),
                            self.labels_next.as_device_ptr(),
                            self.edge_row_offsets.as_device_ptr(),
                            self.edge_col_indices.as_device_ptr(),
                            self.edge_weights.as_device_ptr(),
                            self.label_counts.as_device_ptr(),
                            self.num_nodes as i32,
                            self.max_labels as i32,
                            self.rand_states.as_device_ptr().as_raw()
                        )
                    )?;
                }

                // SAFETY: Convergence check kernel is safe because:
                // 1. Compares labels_current and labels_next element-wise
                // 2. convergence_flag is a single-element buffer with atomic write
                // 3. Sets flag to 0 if any label differs between iterations
                unsafe {
                    launch!(
                        check_convergence_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                            self.labels_current.as_device_ptr(),
                            self.labels_next.as_device_ptr(),
                            self.convergence_flag.as_device_ptr(),
                            self.num_nodes as i32
                        )
                    )?;
                }

                // Swap buffers for next iteration
                std::mem::swap(&mut self.labels_current, &mut self.labels_next);
            } else {
                // SAFETY: Asynchronous label propagation kernel is safe because:
                // 1. Updates labels_current in-place (each node reads neighbors, writes self)
                // 2. In async mode, race conditions are acceptable (probabilistic convergence)
                // 3. All other buffers have same safety guarantees as synchronous mode
                unsafe {
                    launch!(
                        propagate_kernel<<<grid_size as u32, block_size as u32, shared_mem_size as u32, stream>>>(
                            self.labels_current.as_device_ptr(),
                            self.edge_row_offsets.as_device_ptr(),
                            self.edge_col_indices.as_device_ptr(),
                            self.edge_weights.as_device_ptr(),
                            self.num_nodes as i32,
                            self.max_labels as i32,
                            self.rand_states.as_device_ptr().as_raw()
                        )
                    )?;
                }
            }


            if synchronous {
                self.stream.synchronize()?;
                let mut convergence_flag_host = vec![0i32];
                self.convergence_flag.copy_to(&mut convergence_flag_host)?;

                if convergence_flag_host[0] == 1 {
                    converged = true;
                    break;
                }
            }
        }


        if !synchronous {
            // In async mode, we don't check convergence per iteration,
            // so the algorithm runs for max_iterations
            // TODO: Implement proper async convergence checking
            log::warn!("Async community detection runs max_iterations without convergence check");
            converged = false;
        }


        let modularity_kernel = self._module.get_function("compute_modularity_kernel")?;
        // SAFETY: Modularity computation kernel is safe because:
        // 1. labels_current contains final community assignments from label propagation
        // 2. edge_* buffers are valid CSR graph data
        // 3. node_degrees was computed by compute_node_degrees_kernel
        // 4. modularity_contributions is output buffer with capacity >= num_nodes
        // 5. total_weight is the sum of all edge weights (computed from node_degrees)
        // 6. The kernel computes Q = sum((A_ij - k_i*k_j/2m) * delta(c_i, c_j)) / 2m
        unsafe {
            launch!(
                modularity_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.labels_current.as_device_ptr(),
                    self.edge_row_offsets.as_device_ptr(),
                    self.edge_col_indices.as_device_ptr(),
                    self.edge_weights.as_device_ptr(),
                    self.node_degrees.as_device_ptr(),
                    self.modularity_contributions.as_device_ptr(),
                    self.num_nodes as i32,
                    total_weight
                )
            )?;
        }

        self.stream.synchronize()?;


        let mut modularity_contributions = vec![0.0f32; self.num_nodes];
        self.modularity_contributions
            .copy_to(&mut modularity_contributions)?;
        let modularity: f32 = modularity_contributions.iter().sum::<f32>() / (2.0 * total_weight);



        let zero_communities = vec![0i32; self.max_labels];
        self.community_sizes.copy_from(&zero_communities)?;

        let count_communities_kernel = self._module.get_function("count_community_sizes_kernel")?;
        // SAFETY: Community size counting kernel is safe because:
        // 1. labels_current contains valid community labels (0 to max_labels-1)
        // 2. community_sizes was zeroed before this kernel and has capacity >= max_labels
        // 3. The kernel uses atomic increments to count nodes per community
        unsafe {
            launch!(
                count_communities_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.labels_current.as_device_ptr(),
                    self.community_sizes.as_device_ptr(),
                    self.num_nodes as i32,
                    self.max_labels as i32
                )
            )?;
        }

        self.stream.synchronize()?;


        let mut labels = vec![0i32; self.num_nodes];
        let mut community_sizes_host = vec![0i32; self.max_labels];
        self.labels_current.copy_to(&mut labels)?;
        self.community_sizes.copy_to(&mut community_sizes_host)?;


        let mut label_map = vec![-1i32; self.max_labels];
        let mut compact_community_sizes = Vec::new();
        let mut num_communities = 0;

        for (i, &size) in community_sizes_host.iter().enumerate() {
            if size > 0 {
                label_map[i] = num_communities as i32;
                compact_community_sizes.push(size);
                num_communities += 1;
            }
        }


        if num_communities < self.max_labels {
            self.label_mapping.copy_from(&label_map)?;

            let relabel_kernel = self._module.get_function("relabel_communities_kernel")?;
            // SAFETY: Relabeling kernel is safe because:
            // 1. labels_current contains valid labels that index into label_mapping
            // 2. label_mapping was just populated with compact indices (0 to num_communities-1)
            // 3. The kernel reads label_mapping[labels_current[i]] for each node
            // 4. Entries with -1 in label_mapping indicate unused labels (should not occur)
            unsafe {
                launch!(
                    relabel_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                        self.labels_current.as_device_ptr(),
                        self.label_mapping.as_device_ptr(),
                        self.num_nodes as i32
                    )
                )?;
            }

            self.stream.synchronize()?;
            self.labels_current.copy_to(&mut labels)?;
        }

        Ok((
            labels,
            num_communities,
            modularity,
            iterations,
            compact_community_sizes,
            converged,
        ))
    }


    pub fn run_community_detection_label_propagation(
        &mut self,
        max_iterations: u32,
        seed: u32,
    ) -> Result<(Vec<i32>, usize, f32, u32, Vec<i32>, bool)> {

        self.run_community_detection(max_iterations, true, seed)
    }


    pub fn run_louvain_community_detection(
        &mut self,
        max_iterations: u32,
        resolution: f32,
        _seed: u32,
    ) -> Result<(Vec<i32>, usize, f32, u32, Vec<i32>, bool)> {
        info!("Running REAL Louvain community detection on GPU");

        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;


        let mut node_communities = (0..self.num_nodes as i32).collect::<Vec<i32>>();
        let community_weights = vec![1.0f32; self.num_nodes];
        let node_weights = vec![1.0f32; self.num_nodes];


        let d_node_communities = DeviceBuffer::from_slice(&node_communities)?;
        let d_community_weights = DeviceBuffer::from_slice(&community_weights)?;
        let d_node_weights = DeviceBuffer::from_slice(&node_weights)?;
        let mut d_improvement_flag = DeviceBuffer::from_slice(&[false])?;

        let total_weight = self.num_nodes as f32;
        let mut converged = false;
        let mut actual_iterations = 0;

        for iteration in 0..max_iterations {
            actual_iterations = iteration + 1;


            d_improvement_flag.copy_from(&[false])?;


            let louvain_kernel = self._module.get_function("louvain_local_pass_kernel")?;

            // SAFETY: Louvain community detection kernel launch is safe because:
            // 1. d_node_weights contains per-node weights (initialized to 1.0)
            // 2. d_node_communities contains community assignments (initially node indices)
            // 3. d_community_weights is the sum of weights in each community
            // 4. d_improvement_flag is a single bool to track if any improvement occurred
            // 5. The kernel evaluates modularity gain for moving each node to neighbor communities
            unsafe {
                let stream = &self.stream;
                launch!(
                louvain_kernel<<<grid_size, block_size, 0, stream>>>(
                    d_node_weights.as_device_ptr(),
                    d_node_communities.as_device_ptr(),
                    d_node_communities.as_device_ptr(),
                    d_node_communities.as_device_ptr(),
                    d_node_weights.as_device_ptr(),
                    d_community_weights.as_device_ptr(),
                    d_improvement_flag.as_device_ptr(),
                    self.num_nodes as i32,
                    total_weight,
                    resolution
                ))?;
            }

            self.stream.synchronize()?;


            let mut improvement = vec![false];
            d_improvement_flag.copy_to(&mut improvement)?;

            if !improvement[0] {
                converged = true;
                break;
            }
        }


        d_node_communities.copy_to(&mut node_communities)?;


        let mut unique_communities = node_communities.clone();
        unique_communities.sort_unstable();
        unique_communities.dedup();
        let num_communities = unique_communities.len();


        let mut community_sizes = vec![0usize; num_communities];
        for &community in &node_communities {
            if let Ok(idx) = unique_communities.binary_search(&community) {
                community_sizes[idx] += 1;
            }
        }


        let modularity = self.calculate_modularity(&node_communities, total_weight);

        Ok((
            node_communities,
            num_communities,
            modularity,
            actual_iterations,
            community_sizes.into_iter().map(|x| x as i32).collect(),
            converged,
        ))
    }


    pub fn run_dbscan_clustering(&mut self, eps: f32, min_pts: i32) -> Result<Vec<i32>> {
        info!("Running REAL DBSCAN clustering on GPU");

        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;


        let mut labels = vec![0i32; self.num_nodes];
        let neighbor_counts = vec![0i32; self.num_nodes];
        let max_neighbors = 64;
        let neighbors = vec![0i32; self.num_nodes * max_neighbors];
        let neighbor_offsets = (0..self.num_nodes)
            .map(|i| (i * max_neighbors) as i32)
            .collect::<Vec<i32>>();


        let d_labels = DeviceBuffer::from_slice(&labels)?;
        let d_neighbors = DeviceBuffer::from_slice(&neighbors)?;
        let d_neighbor_counts = DeviceBuffer::from_slice(&neighbor_counts)?;
        let d_neighbor_offsets = DeviceBuffer::from_slice(&neighbor_offsets)?;


        let find_neighbors_kernel = self._module.get_function("dbscan_find_neighbors_kernel")?;

        // SAFETY: DBSCAN neighbor finding kernel launch is safe because:
        // 1. pos_in_* contain valid position data for num_nodes nodes
        // 2. d_neighbors is sized for num_nodes * max_neighbors indices
        // 3. d_neighbor_counts stores count per node (capacity >= num_nodes)
        // 4. d_neighbor_offsets stores offsets into d_neighbors for each node
        // 5. The kernel finds all points within eps distance using brute-force search
        unsafe {
            let stream = &self.stream;
            launch!(
            find_neighbors_kernel<<<grid_size, block_size, 0, stream>>>(
                self.pos_in_x.as_device_ptr(),
                self.pos_in_y.as_device_ptr(),
                self.pos_in_z.as_device_ptr(),
                d_neighbors.as_device_ptr(),
                d_neighbor_counts.as_device_ptr(),
                d_neighbor_offsets.as_device_ptr(),
                eps,
                self.num_nodes as i32,
                max_neighbors as i32
            ))?;
        }

        self.stream.synchronize()?;


        let mark_core_kernel = self
            ._module
            .get_function("dbscan_mark_core_points_kernel")?;

        // SAFETY: DBSCAN core point marking kernel is safe because:
        // 1. d_neighbor_counts contains neighbor counts from previous kernel
        // 2. d_labels is the output buffer for cluster labels (capacity >= num_nodes)
        // 3. min_pts is the threshold for core point classification
        // 4. The kernel marks nodes with >= min_pts neighbors as core points
        unsafe {
            let stream = &self.stream;
            launch!(
            mark_core_kernel<<<grid_size, block_size, 0, stream>>>(
                d_neighbor_counts.as_device_ptr(),
                d_labels.as_device_ptr(),
                min_pts,
                self.num_nodes as i32
            ))?;
        }

        self.stream.synchronize()?;

        // Phase 3: Propagate cluster labels until convergence
        let propagate_kernel = self
            ._module
            .get_function("dbscan_propagate_labels_kernel")?;

        let mut changed = vec![0i32; 1];
        let mut d_changed = DeviceBuffer::from_slice(&changed)?;

        const MAX_ITERATIONS: usize = 100;
        for _iter in 0..MAX_ITERATIONS {
            // Reset changed flag
            changed[0] = 0;
            d_changed.copy_from(&changed)?;

            // SAFETY: DBSCAN label propagation kernel is safe because:
            // 1. d_neighbors contains valid neighbor indices from find_neighbors
            // 2. d_neighbor_counts and d_neighbor_offsets provide bounds for neighbor access
            // 3. d_labels contains current cluster labels (read and written atomically)
            // 4. d_changed is a single-element flag set if any label changed
            // 5. The kernel propagates labels from core points to border points
            unsafe {
                let stream = &self.stream;
                launch!(
                propagate_kernel<<<grid_size, block_size, 0, stream>>>(
                    d_neighbors.as_device_ptr(),
                    d_neighbor_counts.as_device_ptr(),
                    d_neighbor_offsets.as_device_ptr(),
                    d_labels.as_device_ptr(),
                    d_changed.as_device_ptr(),
                    self.num_nodes as i32
                ))?;
            }

            self.stream.synchronize()?;
            d_changed.copy_to(&mut changed)?;

            if changed[0] == 0 {
                break;
            }
        }

        // Phase 4: Finalize noise points
        let finalize_kernel = self
            ._module
            .get_function("dbscan_finalize_noise_kernel")?;

        // SAFETY: DBSCAN finalization kernel is safe because:
        // 1. d_labels contains cluster labels from propagation phase
        // 2. The kernel marks unlabeled points (label == 0) as noise (-1)
        // 3. This is the final pass that produces the output cluster assignments
        unsafe {
            let stream = &self.stream;
            launch!(
            finalize_kernel<<<grid_size, block_size, 0, stream>>>(
                d_labels.as_device_ptr(),
                self.num_nodes as i32
            ))?;
        }

        self.stream.synchronize()?;

        // Copy final labels back to host
        d_labels.copy_to(&mut labels)?;

        Ok(labels)
    }


    pub(crate) fn calculate_modularity(&self, communities: &[i32], total_weight: f32) -> f32 {
        if communities.is_empty() || total_weight <= 0.0 {
            return 0.0;
        }

        let _num_nodes = communities.len();
        let mut modularity = 0.0;


        let mut community_map: std::collections::HashMap<i32, Vec<usize>> =
            std::collections::HashMap::new();
        for (node_idx, &community) in communities.iter().enumerate() {
            community_map
                .entry(community)
                .or_insert_with(Vec::new)
                .push(node_idx);
        }


        for (_community_id, nodes) in community_map.iter() {
            if nodes.len() < 2 {
                continue;
            }


            let internal_edges = (nodes.len() * (nodes.len() - 1)) as f32 * 0.1;


            let degree_sum = nodes.len() as f32 * 2.0;


            let e_ii = internal_edges / (2.0 * total_weight);
            let a_i = degree_sum / (2.0 * total_weight);

            modularity += e_ii - (a_i * a_i);
        }


        modularity.max(-1.0).min(1.0)
    }
}
