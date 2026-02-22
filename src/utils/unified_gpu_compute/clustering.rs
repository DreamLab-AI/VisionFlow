//! Clustering algorithms: K-means, LOF anomaly detection, Z-score anomaly detection.

use super::construction::UnifiedGPUCompute;
use super::types::int3;
use anyhow::{anyhow, Result};
use cust::launch;
use cust::memory::CopyDestination;
use log::info;

impl UnifiedGPUCompute {
    pub fn run_kmeans(
        &mut self,
        num_clusters: usize,
        max_iterations: u32,
        tolerance: f32,
        seed: u32,
    ) -> Result<(Vec<i32>, Vec<(f32, f32, f32)>, f32)> {
        if num_clusters > self.max_clusters {
            return Err(anyhow!(
                "Too many clusters requested: {} > {}",
                num_clusters,
                self.max_clusters
            ));
        }


        let module = if let Some(ref clustering_mod) = self.clustering_module {
            clustering_mod
        } else {
            &self._module
        };

        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;


        for centroid in 0..num_clusters {
            let init_kernel = module.get_function("init_centroids_kernel")?;
            let shared_memory_size = block_size * 4;
            let stream = &self.stream;

            unsafe {
                launch!(
                    init_kernel<<<num_clusters as u32, block_size, shared_memory_size, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.min_distances.as_device_ptr(),
                    self.selected_nodes.as_device_ptr(),
                    self.num_nodes as i32,
                    num_clusters as i32,
                    centroid as i32,
                    seed
                ))?;
            }
            self.stream.synchronize()?;
        }

        let mut prev_inertia = f32::INFINITY;
        let mut final_inertia = 0.0f32;


        for _iteration in 0..max_iterations {

            let assign_kernel = self._module.get_function("assign_clusters_kernel")?;
            let stream = &self.stream;
            unsafe {
                launch!(
                    assign_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.cluster_assignments.as_device_ptr(),
                    self.distances_to_centroid.as_device_ptr(),
                    self.num_nodes as i32,
                    num_clusters as i32
                ))?;
            }


            let update_kernel = self._module.get_function("update_centroids_kernel")?;
            let centroid_shared_memory = block_size * (3 * 4 + 4);
            let stream = &self.stream;
            unsafe {
                launch!(
                    update_kernel<<<num_clusters as u32, block_size, centroid_shared_memory, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.cluster_assignments.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.cluster_sizes.as_device_ptr(),
                    self.num_nodes as i32,
                    num_clusters as i32
                ))?;
            }


            let inertia_kernel = self._module.get_function("compute_inertia_kernel")?;
            let inertia_shared_memory = block_size * 4;
            let stream = &self.stream;
            unsafe {
                launch!(
                    inertia_kernel<<<grid_size, block_size, inertia_shared_memory, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.cluster_assignments.as_device_ptr(),
                    self.partial_inertia.as_device_ptr(),
                    self.num_nodes as i32
                ))?;
            }

            self.stream.synchronize()?;


            let mut partial_inertias = vec![0.0f32; grid_size as usize];
            self.partial_inertia.copy_to(&mut partial_inertias)?;
            let current_inertia: f32 = partial_inertias.iter().sum();
            final_inertia = current_inertia;


            if (prev_inertia - current_inertia).abs() < tolerance {
                info!(
                    "K-means converged at iteration {} with inertia {:.4}",
                    _iteration, current_inertia
                );
                break;
            }

            prev_inertia = current_inertia;
        }


        let mut assignments = vec![0i32; self.num_nodes];
        self.cluster_assignments.copy_to(&mut assignments)?;

        let mut centroids_x = vec![0.0f32; num_clusters];
        let mut centroids_y = vec![0.0f32; num_clusters];
        let mut centroids_z = vec![0.0f32; num_clusters];
        self.centroids_x.copy_to(&mut centroids_x)?;
        self.centroids_y.copy_to(&mut centroids_y)?;
        self.centroids_z.copy_to(&mut centroids_z)?;

        let centroids: Vec<(f32, f32, f32)> = centroids_x
            .into_iter()
            .zip(centroids_y.into_iter())
            .zip(centroids_z.into_iter())
            .map(|((x, y), z)| (x, y, z))
            .collect();

        Ok((assignments, centroids, final_inertia))
    }


    pub fn run_kmeans_clustering_with_metrics(
        &mut self,
        num_clusters: usize,
        max_iterations: u32,
        tolerance: f32,
        seed: u32,
    ) -> Result<(Vec<i32>, Vec<(f32, f32, f32)>, f32, u32, bool)> {
        if num_clusters > self.max_clusters {
            return Err(anyhow!(
                "Too many clusters requested: {} > {}",
                num_clusters,
                self.max_clusters
            ));
        }

        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;


        for centroid in 0..num_clusters {
            let init_kernel = self._module.get_function("init_centroids_kernel")?;
            let shared_memory_size = block_size * 4;
            let stream = &self.stream;

            unsafe {
                launch!(
                    init_kernel<<<num_clusters as u32, block_size, shared_memory_size, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.min_distances.as_device_ptr(),
                    self.selected_nodes.as_device_ptr(),
                    self.num_nodes as i32,
                    num_clusters as i32,
                    centroid as i32,
                    seed
                ))?;
            }
            self.stream.synchronize()?;
        }

        let mut prev_inertia = f32::INFINITY;
        let mut final_inertia = 0.0f32;
        let mut converged = false;
        let mut actual_iterations = 0u32;


        for iteration in 0..max_iterations {
            actual_iterations = iteration + 1;


            let assign_kernel = self._module.get_function("assign_clusters_kernel")?;
            let stream = &self.stream;
            unsafe {
                launch!(
                    assign_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.cluster_assignments.as_device_ptr(),
                    self.distances_to_centroid.as_device_ptr(),
                    self.num_nodes as i32,
                    num_clusters as i32
                ))?;
            }


            let update_kernel = self._module.get_function("update_centroids_kernel")?;
            let centroid_shared_memory = block_size * (3 * 4 + 4);
            let stream = &self.stream;
            unsafe {
                launch!(
                    update_kernel<<<num_clusters as u32, block_size, centroid_shared_memory, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.cluster_assignments.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.cluster_sizes.as_device_ptr(),
                    self.num_nodes as i32,
                    num_clusters as i32
                ))?;
            }


            let inertia_kernel = self._module.get_function("compute_inertia_kernel")?;
            let inertia_shared_memory = block_size * 4;
            let stream = &self.stream;
            unsafe {
                launch!(
                    inertia_kernel<<<grid_size, block_size, inertia_shared_memory, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.centroids_x.as_device_ptr(),
                    self.centroids_y.as_device_ptr(),
                    self.centroids_z.as_device_ptr(),
                    self.cluster_assignments.as_device_ptr(),
                    self.partial_inertia.as_device_ptr(),
                    self.num_nodes as i32
                ))?;
            }

            self.stream.synchronize()?;


            let mut partial_inertias = vec![0.0f32; grid_size as usize];
            self.partial_inertia.copy_to(&mut partial_inertias)?;
            let current_inertia: f32 = partial_inertias.iter().sum();
            final_inertia = current_inertia;


            if (prev_inertia - current_inertia).abs() < tolerance {
                info!(
                    "K-means converged at iteration {} with inertia {:.4}",
                    iteration, current_inertia
                );
                converged = true;
                break;
            }

            prev_inertia = current_inertia;
        }


        let mut assignments = vec![0i32; self.num_nodes];
        self.cluster_assignments.copy_to(&mut assignments)?;

        let mut centroids_x = vec![0.0f32; num_clusters];
        let mut centroids_y = vec![0.0f32; num_clusters];
        let mut centroids_z = vec![0.0f32; num_clusters];
        self.centroids_x.copy_to(&mut centroids_x)?;
        self.centroids_y.copy_to(&mut centroids_y)?;
        self.centroids_z.copy_to(&mut centroids_z)?;

        let centroids: Vec<(f32, f32, f32)> = centroids_x
            .into_iter()
            .zip(centroids_y.into_iter())
            .zip(centroids_z.into_iter())
            .map(|((x, y), z)| (x, y, z))
            .collect();

        Ok((
            assignments,
            centroids,
            final_inertia,
            actual_iterations,
            converged,
        ))
    }


    pub fn run_lof_anomaly_detection(
        &mut self,
        k_neighbors: i32,
        radius: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {

        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;



        let grid_dims = int3 {
            x: 32,
            y: 32,
            z: 32,
        };

        let lof_kernel = self._module.get_function("compute_lof_kernel")?;
        let stream = &self.stream;
        // SAFETY: LOF anomaly detection kernel launch is safe because:
        // 1. pos_in_* buffers contain valid position data
        // 2. sorted_node_indices, cell_start, cell_end, cell_keys are from spatial grid
        // 3. lof_scores and local_densities are output buffers with capacity >= num_nodes
        // 4. grid_dims contains valid grid dimensions for spatial partitioning
        // 5. k_neighbors and radius are validated algorithm parameters
        unsafe {
            launch!(
                lof_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                self.pos_in_x.as_device_ptr(),
                self.pos_in_y.as_device_ptr(),
                self.pos_in_z.as_device_ptr(),
                self.sorted_node_indices.as_device_ptr(),
                self.cell_start.as_device_ptr(),
                self.cell_end.as_device_ptr(),
                self.cell_keys.as_device_ptr(),
                grid_dims,
                self.lof_scores.as_device_ptr(),
                self.local_densities.as_device_ptr(),
                self.num_nodes as i32,
                k_neighbors,
                radius,
                crate::config::dev_config::physics().world_bounds_min,
                crate::config::dev_config::physics().world_bounds_max,
                crate::config::dev_config::physics().cell_size_lod,
                crate::config::dev_config::physics().k_neighbors_max as i32
            ))?;
        }

        self.stream.synchronize()?;


        let mut lof_scores = vec![0.0f32; self.num_nodes];
        let mut local_densities = vec![0.0f32; self.num_nodes];
        self.lof_scores.copy_to(&mut lof_scores)?;
        self.local_densities.copy_to(&mut local_densities)?;

        Ok((lof_scores, local_densities))
    }


    pub fn run_zscore_anomaly_detection(&mut self, feature_data: &[f32]) -> Result<Vec<f32>> {
        if feature_data.len() != self.num_nodes {
            return Err(anyhow!(
                "Feature data size {} doesn't match number of nodes {}",
                feature_data.len(),
                self.num_nodes
            ));
        }


        self.feature_values.copy_from(feature_data)?;

        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;


        let stats_kernel = self._module.get_function("compute_feature_stats_kernel")?;
        let stats_shared_memory = block_size * 2 * 4;
        let stream = &self.stream;
        // SAFETY: Feature statistics kernel launch is safe because:
        // 1. feature_values was just populated from feature_data via copy_from()
        // 2. partial_sums and partial_sq_sums are output buffers with capacity >= grid_size
        // 3. shared_memory size (2 floats per thread) fits within GPU limits
        // 4. This is a parallel reduction computing sum and sum-of-squares
        unsafe {
            launch!(
                stats_kernel<<<grid_size, block_size, stats_shared_memory, stream>>>(
                self.feature_values.as_device_ptr(),
                self.partial_sums.as_device_ptr(),
                self.partial_sq_sums.as_device_ptr(),
                self.num_nodes as i32
            ))?;
        }

        self.stream.synchronize()?;


        let mut partial_sums = vec![0.0f32; grid_size as usize];
        let mut partial_sq_sums = vec![0.0f32; grid_size as usize];
        self.partial_sums.copy_to(&mut partial_sums)?;
        self.partial_sq_sums.copy_to(&mut partial_sq_sums)?;

        let total_sum: f32 = partial_sums.iter().sum();
        let total_sq_sum: f32 = partial_sq_sums.iter().sum();

        let mean = total_sum / self.num_nodes as f32;
        let variance = (total_sq_sum / self.num_nodes as f32) - (mean * mean);
        let std_dev = variance.sqrt();


        let zscore_kernel = self._module.get_function("compute_zscore_kernel")?;
        let stream = &self.stream;
        // SAFETY: Z-score computation kernel launch is safe because:
        // 1. feature_values contains the input feature data
        // 2. zscore_values is the output buffer with capacity >= num_nodes
        // 3. mean and std_dev are computed from the stats kernel reduction
        // 4. The kernel performs element-wise (value - mean) / std_dev
        unsafe {
            launch!(
                zscore_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                self.feature_values.as_device_ptr(),
                self.zscore_values.as_device_ptr(),
                mean,
                std_dev,
                self.num_nodes as i32
            ))?;
        }

        self.stream.synchronize()?;


        let mut zscore_values = vec![0.0f32; self.num_nodes];
        self.zscore_values.copy_to(&mut zscore_values)?;

        Ok(zscore_values)
    }


    pub fn run_kmeans_clustering(
        &mut self,
        num_clusters: usize,
        max_iterations: u32,
        tolerance: f32,
        seed: u32,
    ) -> Result<(Vec<i32>, Vec<(f32, f32, f32)>, f32)> {
        self.run_kmeans(num_clusters, max_iterations, tolerance, seed)
    }


    pub fn run_anomaly_detection_lof(
        &mut self,
        k_neighbors: i32,
        radius: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.run_lof_anomaly_detection(k_neighbors, radius)
    }
}
