//! Memory management, buffer resizing, and data upload/download operations.

use super::construction::UnifiedGPUCompute;
use super::types::ComputeMode;
use crate::models::constraints::ConstraintData;
use crate::models::simulation_params::SimParams;
use anyhow::{anyhow, Result};
use cust::memory::{CopyDestination, DeviceBuffer};
use log::{debug, info, warn};

impl UnifiedGPUCompute {
    pub fn upload_positions(&mut self, x: &[f32], y: &[f32], z: &[f32]) -> Result<()> {

        if x.len() != self.num_nodes || y.len() != self.num_nodes || z.len() != self.num_nodes {
            return Err(anyhow!(
                "Position array size mismatch: expected {} nodes, got x:{}, y:{}, z:{}",
                self.num_nodes,
                x.len(),
                y.len(),
                z.len()
            ));
        }


        if x.len() < self.allocated_nodes {
            let mut padded_x = x.to_vec();
            let mut padded_y = y.to_vec();
            let mut padded_z = z.to_vec();
            padded_x.resize(self.allocated_nodes, 0.0);
            padded_y.resize(self.allocated_nodes, 0.0);
            padded_z.resize(self.allocated_nodes, 0.0);
            self.pos_in_x.copy_from(&padded_x)?;
            self.pos_in_y.copy_from(&padded_y)?;
            self.pos_in_z.copy_from(&padded_z)?;
        } else {
            self.pos_in_x.copy_from(x)?;
            self.pos_in_y.copy_from(y)?;
            self.pos_in_z.copy_from(z)?;
        }
        Ok(())
    }

    /// Upload ontology class metadata for class-based physics
    /// Maps owl_class_iri to integer class IDs and sets class-specific force parameters
    pub fn upload_class_metadata(
        &mut self,
        class_ids: &[i32],
        class_charges: &[f32],
        class_masses: &[f32],
    ) -> Result<()> {
        if class_ids.len() != self.num_nodes {
            return Err(anyhow!(
                "Class ID array size mismatch: expected {} nodes, got {}",
                self.num_nodes,
                class_ids.len()
            ));
        }
        if class_charges.len() != self.num_nodes {
            return Err(anyhow!(
                "Class charge array size mismatch: expected {} nodes, got {}",
                self.num_nodes,
                class_charges.len()
            ));
        }
        if class_masses.len() != self.num_nodes {
            return Err(anyhow!(
                "Class mass array size mismatch: expected {} nodes, got {}",
                self.num_nodes,
                class_masses.len()
            ));
        }

        // Upload to GPU buffers
        self.class_id.copy_from(class_ids)?;
        self.class_charge.copy_from(class_charges)?;
        self.class_mass.copy_from(class_masses)?;

        Ok(())
    }

    pub fn upload_edges_csr(
        &mut self,
        row_offsets: &[i32],
        col_indices: &[i32],
        weights: &[f32],
    ) -> Result<()> {

        if row_offsets.len() != self.num_nodes + 1 {
            return Err(anyhow!(
                "Row offsets size mismatch: expected {} (num_nodes + 1), got {}",
                self.num_nodes + 1,
                row_offsets.len()
            ));
        }


        if col_indices.len() != weights.len() {
            return Err(anyhow!(
                "Edge arrays size mismatch: col_indices has {}, weights has {}",
                col_indices.len(),
                weights.len()
            ));
        }


        if col_indices.len() > self.allocated_edges {
            return Err(anyhow!(
                "Too many edges: trying to upload {}, but only {} allocated",
                col_indices.len(),
                self.allocated_edges
            ));
        }



        if row_offsets.len() <= self.allocated_nodes + 1 {

            let mut padded_row_offsets = row_offsets.to_vec();
            let last_val = *padded_row_offsets.last().unwrap_or(&0);
            padded_row_offsets.resize(self.allocated_nodes + 1, last_val);
            self.edge_row_offsets.copy_from(&padded_row_offsets)?;
        } else {
            self.edge_row_offsets.copy_from(row_offsets)?;
        }


        if col_indices.len() < self.allocated_edges {
            let mut padded_col_indices = col_indices.to_vec();
            let mut padded_weights = weights.to_vec();
            padded_col_indices.resize(self.allocated_edges, 0);
            padded_weights.resize(self.allocated_edges, 0.0);
            self.edge_col_indices.copy_from(&padded_col_indices)?;
            self.edge_weights.copy_from(&padded_weights)?;
        } else {
            self.edge_col_indices.copy_from(col_indices)?;
            self.edge_weights.copy_from(weights)?;
        }

        self.num_edges = col_indices.len();
        Ok(())
    }

    /// Download the CSR graph structure from GPU device memory.
    /// Returns (row_offsets, col_indices) where row_offsets has length num_nodes+1
    /// and col_indices has length num_edges.
    pub fn download_csr(&self) -> Result<(Vec<i32>, Vec<i32>)> {
        let mut row_offsets = vec![0i32; self.num_nodes + 1];
        let mut col_indices = vec![0i32; self.num_edges];
        self.edge_row_offsets.copy_to(&mut row_offsets)?;
        if self.num_edges > 0 {
            self.edge_col_indices.copy_to(&mut col_indices)?;
        }
        Ok((row_offsets, col_indices))
    }

    pub fn download_positions(&self, x: &mut [f32], y: &mut [f32], z: &mut [f32]) -> Result<()> {
        self.pos_in_x.copy_to(x)?;
        self.pos_in_y.copy_to(y)?;
        self.pos_in_z.copy_to(z)?;
        Ok(())
    }

    pub fn download_velocities(&self, x: &mut [f32], y: &mut [f32], z: &mut [f32]) -> Result<()> {
        self.vel_in_x.copy_to(x)?;
        self.vel_in_y.copy_to(y)?;
        self.vel_in_z.copy_to(z)?;
        Ok(())
    }

    pub fn swap_buffers(&mut self) {
        std::mem::swap(&mut self.pos_in_x, &mut self.pos_out_x);
        std::mem::swap(&mut self.pos_in_y, &mut self.pos_out_y);
        std::mem::swap(&mut self.pos_in_z, &mut self.pos_out_z);
        std::mem::swap(&mut self.vel_in_x, &mut self.vel_out_x);
        std::mem::swap(&mut self.vel_in_y, &mut self.vel_out_y);
        std::mem::swap(&mut self.vel_in_z, &mut self.vel_out_z);
    }

    pub fn get_memory_metrics(&self) -> (usize, f32, usize) {
        let current_usage =
            Self::calculate_memory_usage(self.num_nodes, self.num_edges, self.max_grid_cells);
        let allocated_usage = Self::calculate_memory_usage(
            self.allocated_nodes,
            self.allocated_edges,
            self.max_grid_cells,
        );
        let utilization = current_usage as f32 / allocated_usage as f32;
        (current_usage, utilization, self.resize_count)
    }


    pub fn get_grid_occupancy(&self, num_grid_cells: usize) -> f32 {
        if num_grid_cells == 0 {
            return 0.0;
        }
        let avg_nodes_per_cell = self.num_nodes as f32 / num_grid_cells as f32;

        let optimal_occupancy = 8.0;
        (avg_nodes_per_cell / optimal_occupancy).min(1.0)
    }


    pub fn resize_cell_buffers(&mut self, required_cells: usize) -> Result<()> {
        if required_cells <= self.max_grid_cells {
            return Ok(());
        }


        if required_cells > self.max_allowed_grid_cells {
            warn!(
                "Grid size {} exceeds maximum allowed {}, capping to maximum",
                required_cells, self.max_allowed_grid_cells
            );
            let capped_size = self.max_allowed_grid_cells;
            return self.resize_cell_buffers_internal(capped_size);
        }


        let new_size = ((required_cells as f32 * self.cell_buffer_growth_factor) as usize)
            .min(self.max_allowed_grid_cells);

        self.resize_cell_buffers_internal(new_size)
    }


    fn resize_cell_buffers_internal(&mut self, new_size: usize) -> Result<()> {
        info!(
            "Resizing cell buffers from {} to {} cells ({}x growth)",
            self.max_grid_cells, new_size, self.cell_buffer_growth_factor
        );


        let preserve_data = self.max_grid_cells > 0 && self.iteration > 0;

        let old_cell_start_data = if preserve_data {
            let mut data = vec![0i32; self.max_grid_cells];
            self.cell_start.copy_to(&mut data).unwrap_or_else(|e| {
                warn!("Failed to preserve cell_start data: {}", e);
            });
            Some(data)
        } else {
            None
        };

        let old_cell_end_data = if preserve_data {
            let mut data = vec![0i32; self.max_grid_cells];
            self.cell_end.copy_to(&mut data).unwrap_or_else(|e| {
                warn!("Failed to preserve cell_end data: {}", e);
            });
            Some(data)
        } else {
            None
        };


        self.cell_start = DeviceBuffer::zeroed(new_size).map_err(|e| {
            anyhow!(
                "Failed to allocate cell_start buffer of size {}: {}",
                new_size,
                e
            )
        })?;
        self.cell_end = DeviceBuffer::zeroed(new_size).map_err(|e| {
            anyhow!(
                "Failed to allocate cell_end buffer of size {}: {}",
                new_size,
                e
            )
        })?;


        if let (Some(start_data), Some(end_data)) = (old_cell_start_data, old_cell_end_data) {
            let copy_size = start_data.len().min(new_size);
            if copy_size > 0 {
                self.cell_start.copy_from(&start_data[..copy_size])?;
                self.cell_end.copy_from(&end_data[..copy_size])?;
                debug!("Preserved {} cells of data during resize", copy_size);
            }
        }


        let old_memory = self.total_memory_allocated;
        self.max_grid_cells = new_size;
        self.zero_buffer = vec![0i32; new_size];
        self.resize_count += 1;
        self.total_memory_allocated = Self::calculate_memory_usage(
            self.allocated_nodes,
            self.allocated_edges,
            self.max_grid_cells,
        );

        let memory_delta = self.total_memory_allocated as i64 - old_memory as i64;
        info!(
            "Cell buffer resize complete. Memory change: {:+} bytes, Total: {} MB",
            memory_delta,
            self.total_memory_allocated / 1024 / 1024
        );


        if self.resize_count > 10 {
            warn!("High resize frequency detected ({} resizes). Consider increasing initial buffer size.",
                  self.resize_count);
        }

        Ok(())
    }


    pub fn resize_buffers(&mut self, new_num_nodes: usize, new_num_edges: usize) -> Result<()> {

        if new_num_nodes <= self.num_nodes && new_num_edges <= self.num_edges {
            self.num_nodes = new_num_nodes;
            self.num_edges = new_num_edges;
            return Ok(());
        }

        info!(
            "Resizing GPU buffers from {}/{} to {}/{} nodes/edges",
            self.num_nodes, self.num_edges, new_num_nodes, new_num_edges
        );


        let actual_new_nodes = ((new_num_nodes as f32 * 1.5) as usize).max(self.num_nodes);
        let actual_new_edges = ((new_num_edges as f32 * 1.5) as usize).max(self.num_edges);


        let mut pos_x_data = vec![0.0f32; self.num_nodes];
        let mut pos_y_data = vec![0.0f32; self.num_nodes];
        let mut pos_z_data = vec![0.0f32; self.num_nodes];
        let mut vel_x_data = vec![0.0f32; self.num_nodes];
        let mut vel_y_data = vec![0.0f32; self.num_nodes];
        let mut vel_z_data = vec![0.0f32; self.num_nodes];


        self.pos_in_x.copy_to(&mut pos_x_data)?;
        self.pos_in_y.copy_to(&mut pos_y_data)?;
        self.pos_in_z.copy_to(&mut pos_z_data)?;
        self.vel_in_x.copy_to(&mut vel_x_data)?;
        self.vel_in_y.copy_to(&mut vel_y_data)?;
        self.vel_in_z.copy_to(&mut vel_z_data)?;


        pos_x_data.resize(actual_new_nodes, 0.0);
        pos_y_data.resize(actual_new_nodes, 0.0);
        pos_z_data.resize(actual_new_nodes, 0.0);
        vel_x_data.resize(actual_new_nodes, 0.0);
        vel_y_data.resize(actual_new_nodes, 0.0);
        vel_z_data.resize(actual_new_nodes, 0.0);


        self.pos_in_x = DeviceBuffer::from_slice(&pos_x_data)?;
        self.pos_in_y = DeviceBuffer::from_slice(&pos_y_data)?;
        self.pos_in_z = DeviceBuffer::from_slice(&pos_z_data)?;
        self.vel_in_x = DeviceBuffer::from_slice(&vel_x_data)?;
        self.vel_in_y = DeviceBuffer::from_slice(&vel_y_data)?;
        self.vel_in_z = DeviceBuffer::from_slice(&vel_z_data)?;

        self.pos_out_x = DeviceBuffer::from_slice(&pos_x_data)?;
        self.pos_out_y = DeviceBuffer::from_slice(&pos_y_data)?;
        self.pos_out_z = DeviceBuffer::from_slice(&pos_z_data)?;
        self.vel_out_x = DeviceBuffer::from_slice(&vel_x_data)?;
        self.vel_out_y = DeviceBuffer::from_slice(&vel_y_data)?;
        self.vel_out_z = DeviceBuffer::from_slice(&vel_z_data)?;


        self.mass = DeviceBuffer::from_slice(&vec![1.0f32; actual_new_nodes])?;
        self.node_graph_id = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.edge_row_offsets = DeviceBuffer::zeroed(actual_new_nodes + 1)?;
        self.edge_col_indices = DeviceBuffer::zeroed(actual_new_edges)?;
        self.edge_weights = DeviceBuffer::zeroed(actual_new_edges)?;
        self.force_x = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.force_y = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.force_z = DeviceBuffer::zeroed(actual_new_nodes)?;


        self.cell_keys = DeviceBuffer::zeroed(actual_new_nodes)?;
        let sorted_indices: Vec<i32> = (0..actual_new_nodes as i32).collect();
        self.sorted_node_indices = DeviceBuffer::from_slice(&sorted_indices)?;


        self.total_memory_allocated = Self::calculate_memory_usage(
            self.allocated_nodes,
            self.allocated_edges,
            self.max_grid_cells,
        );


        self.cluster_assignments = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.distances_to_centroid = DeviceBuffer::zeroed(actual_new_nodes)?;
        let new_num_blocks = (actual_new_nodes + 255) / 256;
        self.partial_inertia = DeviceBuffer::zeroed(new_num_blocks)?;
        self.min_distances = DeviceBuffer::zeroed(actual_new_nodes)?;


        self.lof_scores = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.local_densities = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.zscore_values = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.feature_values = DeviceBuffer::zeroed(actual_new_nodes)?;
        self.partial_sums = DeviceBuffer::zeroed(new_num_blocks)?;
        self.partial_sq_sums = DeviceBuffer::zeroed(new_num_blocks)?;


        self.num_nodes = new_num_nodes;
        self.num_edges = new_num_edges;
        self.allocated_nodes = actual_new_nodes;
        self.allocated_edges = actual_new_edges;

        info!(
            "Successfully resized GPU buffers to {}/{} allocated nodes/edges",
            actual_new_nodes, actual_new_edges
        );
        Ok(())
    }

    pub fn set_params(&mut self, params: SimParams) -> Result<()> {

        info!(
            "Setting SimParams - spring_k: {:.4}, repel_k: {:.2}, damping: {:.3}, dt: {:.3}",
            params.spring_k, params.repel_k, params.damping, params.dt
        );

        self.params = params;

        info!("SimParams successfully updated");
        Ok(())
    }

    pub fn set_mode(&mut self, _mode: ComputeMode) {

    }

    pub fn set_constraints(&mut self, mut constraints: Vec<ConstraintData>) -> Result<()> {

        let current_iteration = self.iteration;
        for constraint in &mut constraints {
            if constraint.activation_frame == 0 {
                constraint.activation_frame = current_iteration as i32;
                debug!(
                    "Setting activation frame {} for constraint type {}",
                    current_iteration, constraint.kind
                );
            }
        }


        if constraints.len() > self.constraint_data.len() {
            info!(
                "Resizing constraint buffer from {} to {} with progressive activation",
                self.constraint_data.len(),
                constraints.len()
            );

            let new_constraint_buffer = DeviceBuffer::from_slice(&constraints)?;
            self.constraint_data = new_constraint_buffer;
        } else if !constraints.is_empty() {

            let constraint_len = self.constraint_data.len();
            let copy_len = constraints.len().min(constraint_len);
            self.constraint_data.copy_from(&constraints[..copy_len])?;
        }

        self.num_constraints = constraints.len();
        debug!(
            "Updated GPU constraints: {} active constraints with progressive activation support",
            self.num_constraints
        );
        Ok(())
    }

    pub fn clear_constraints(&mut self) -> Result<()> {
        self.num_constraints = 0;


        let empty_constraints = vec![ConstraintData::default(); self.constraint_data.len()];
        self.constraint_data.copy_from(&empty_constraints)?;

        Ok(())
    }

    pub fn upload_constraints(
        &mut self,
        constraints: &[crate::models::constraints::ConstraintData],
    ) -> Result<()> {
        self.num_constraints = constraints.len();

        if constraints.is_empty() {
            return self.clear_constraints();
        }


        let mut constraint_data = Vec::new();
        for constraint in constraints {

            constraint_data.extend_from_slice(&[
                constraint.kind as f32,
                constraint.node_idx[0] as f32,
                constraint.params[0],
                constraint.params[1],
                constraint.params[2],
                constraint.weight,
                constraint.params[3],
            ]);
        }


        if !constraint_data.is_empty() {

            let mut gpu_constraints = Vec::new();
            for chunk in constraint_data.chunks(7) {

                if chunk.len() == 7 {
                    let mut constraint = ConstraintData::default();
                    constraint.kind = chunk[0] as i32;
                    constraint.node_idx[0] = chunk[1] as i32;
                    constraint.params[0] = chunk[2];
                    constraint.params[1] = chunk[3];
                    constraint.params[2] = chunk[4];
                    constraint.weight = chunk[5];
                    constraint.params[3] = chunk[6];
                    gpu_constraints.push(constraint);
                }
            }

            if gpu_constraints.len() > self.constraint_data.len() {

                self.constraint_data = DeviceBuffer::from_slice(&gpu_constraints)?;
            } else {

                self.constraint_data.copy_from(&gpu_constraints)?;
            }
        }

        info!(
            "Uploaded {} constraints to GPU ({} floats)",
            constraints.len(),
            constraint_data.len()
        );
        Ok(())
    }


    pub fn initialize_graph(
        &mut self,
        row_offsets: Vec<i32>,
        col_indices: Vec<i32>,
        edge_weights: Vec<f32>,
        positions_x: Vec<f32>,
        positions_y: Vec<f32>,
        positions_z: Vec<f32>,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<()> {

        if num_nodes != self.num_nodes || num_edges != self.num_edges {
            self.resize_buffers(num_nodes, num_edges)?;
        }


        self.upload_edges_csr(&row_offsets, &col_indices, &edge_weights)?;


        self.upload_positions(&positions_x, &positions_y, &positions_z)?;

        info!(
            "Graph initialized with {} nodes and {} edges",
            num_nodes, num_edges
        );
        Ok(())
    }


    pub fn update_positions_only(
        &mut self,
        positions_x: &[f32],
        positions_y: &[f32],
        positions_z: &[f32],
    ) -> Result<()> {
        self.upload_positions(positions_x, positions_y, positions_z)?;
        Ok(())
    }

    /// Get the number of nodes in the GPU compute context
    /// Returns the actual node count from the position buffer size
    pub fn get_num_nodes(&self) -> usize {
        self.pos_in_x.len()
    }

    /// Returns the raw device pointer to the persisted SSSP distance buffer,
    /// or a null pointer if no SSSP has been computed yet.
    ///
    /// This pointer is suitable for passing as `d_sssp_dist` to CUDA kernels.
    pub fn get_sssp_device_ptr(&self) -> cust::memory::DevicePointer<f32> {
        match &self.sssp_device_distances {
            Some(buf) => buf.as_device_ptr(),
            None => cust::memory::DevicePointer::null(),
        }
    }

    /// Toggle the SSSP spring-adjust feature at runtime.
    ///
    /// When enabled and SSSP distances are available, the force kernel will use
    /// graph-theoretic distances to modulate spring rest lengths.
    pub fn enable_sssp_spring_adjust(&mut self, enabled: bool) {
        self.sssp_spring_adjust_enabled = enabled;
    }

    /// Returns whether the SSSP spring-adjust feature is currently enabled.
    pub fn is_sssp_spring_adjust_enabled(&self) -> bool {
        self.sssp_spring_adjust_enabled
    }
}
