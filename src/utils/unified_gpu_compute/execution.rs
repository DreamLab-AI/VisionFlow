//! Physics simulation execution pipeline (force computation, integration, stability).

use super::construction::UnifiedGPUCompute;
use super::types::{int3, thrust_sort_key_value, AABB};
use crate::models::simulation_params::SimParams;
use anyhow::{anyhow, Result};
use cust::context::Context;
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer, DevicePointer};
use log::{debug, info, warn};
use std::ffi::CStr;

impl UnifiedGPUCompute {
    pub fn execute(&mut self, mut params: SimParams) -> Result<()> {
        // Make CUDA context current for this thread (required when called from spawn_blocking threads)
        // Context::new() on the same device retains the primary context and makes it current
        let _thread_context = Context::new(self.device.clone())
            .map_err(|e| anyhow!("Failed to set CUDA context: {}", e))?;

        params.iteration = self.iteration;
        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;


        if self.num_nodes > self.allocated_nodes {
            return Err(anyhow!("CRITICAL: num_nodes ({}) exceeds allocated_nodes ({}). This would cause buffer overflow!", self.num_nodes, self.allocated_nodes));
        }


        self.params = params;


        let mut c_params_global = self
            ._module
            .get_global(CStr::from_bytes_with_nul(b"c_params\0").unwrap())?;
        c_params_global.copy_from(&[params])?;



        if self.num_nodes > 0 && params.stability_threshold > 0.0 {
            let num_blocks = (self.num_nodes + block_size as usize - 1) / block_size as usize;
            let shared_mem_size =
                block_size * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()) as u32;


            self.active_node_count.copy_from(&[0i32])?;
            self.should_skip_physics.copy_from(&[0i32])?;


            let ke_kernel = self
                ._module
                .get_function("calculate_kinetic_energy_kernel")?;
            // SAFETY: Kernel launch is safe because:
            // 1. All DeviceBuffer pointers (vel_in_*, mass, partial_kinetic_energy, active_node_count)
            //    are valid allocations created during UnifiedGPUCompute::new()
            // 2. num_nodes <= allocated_nodes was verified at function entry
            // 3. shared_mem_size is computed based on block_size and type sizes
            // 4. self.stream is a valid CUDA stream created in UnifiedGPUCompute::new()
            // 5. The kernel function was loaded from a valid PTX module
            unsafe {
                let stream = &self.stream;
                launch!(
                    ke_kernel<<<num_blocks as u32, block_size, shared_mem_size, stream>>>(
                        self.vel_in_x.as_device_ptr(),
                        self.vel_in_y.as_device_ptr(),
                        self.vel_in_z.as_device_ptr(),
                        self.mass.as_device_ptr(),
                        self.partial_kinetic_energy.as_device_ptr(),
                        self.active_node_count.as_device_ptr(),
                        self.num_nodes as i32,
                        params.min_velocity_threshold
                    )
                )?;
            }


            let stability_kernel = self._module.get_function("check_system_stability_kernel")?;
            let reduction_blocks = (num_blocks as u32).min(256);
            // SAFETY: Kernel launch is safe because:
            // 1. All DeviceBuffer arguments are valid allocations from UnifiedGPUCompute::new()
            // 2. reduction_blocks is bounded to max 256 (valid CUDA block size)
            // 3. Shared memory (reduction_blocks * 4) fits within GPU limits
            // 4. This reduction kernel reads from partial_kinetic_energy computed by prior kernel
            unsafe {
                let stream = &self.stream;
                launch!(
                    stability_kernel<<<1, reduction_blocks, reduction_blocks * 4, stream>>>(
                        self.partial_kinetic_energy.as_device_ptr(),
                        self.active_node_count.as_device_ptr(),
                        self.should_skip_physics.as_device_ptr(),
                        self.system_kinetic_energy.as_device_ptr(),
                        num_blocks as i32,
                        self.num_nodes as i32,
                        params.stability_threshold,
                        self.iteration
                    )
                )?;
            }


            let mut skip_physics = vec![0i32; 1];
            self.should_skip_physics.copy_to(&mut skip_physics)?;

            if skip_physics[0] != 0 {

                self.iteration += 1;
                return Ok(());
            }
        }


        crate::utils::gpu_diagnostics::validate_kernel_launch(
            "unified_gpu_execute",
            grid_size,
            block_size,
            self.num_nodes,
        )
        .map_err(|e| anyhow::anyhow!(e))?;


        let aabb_kernel = self._module.get_function("compute_aabb_reduction_kernel")?;
        let aabb_block_size = 256u32;
        let aabb_grid_size = self.aabb_num_blocks as u32;
        let shared_mem = 6 * aabb_block_size * std::mem::size_of::<f32>() as u32;

        // SAFETY: AABB reduction kernel launch is safe because:
        // 1. pos_in_* buffers contain valid position data from prior physics step
        // 2. aabb_block_results is sized for aabb_num_blocks * sizeof(AABB)
        // 3. shared_mem is computed as 6 floats per thread (min/max x,y,z)
        // 4. aabb_grid_size and aabb_block_size are validated during construction
        unsafe {
            let s = &self.stream;
            launch!(
                aabb_kernel<<<aabb_grid_size, aabb_block_size, shared_mem, s>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.aabb_block_results.as_device_ptr(),
                    self.num_nodes as i32
                )
            )?;
        }


        let mut block_results = vec![AABB::default(); self.aabb_num_blocks];
        self.aabb_block_results.copy_to(&mut block_results)?;

        let mut aabb = AABB {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        };
        for block_aabb in block_results.iter().take(self.aabb_num_blocks) {
            aabb.min[0] = aabb.min[0].min(block_aabb.min[0]);
            aabb.min[1] = aabb.min[1].min(block_aabb.min[1]);
            aabb.min[2] = aabb.min[2].min(block_aabb.min[2]);
            aabb.max[0] = aabb.max[0].max(block_aabb.max[0]);
            aabb.max[1] = aabb.max[1].max(block_aabb.max[1]);
            aabb.max[2] = aabb.max[2].max(block_aabb.max[2]);
        }

        let scene_volume =
            (aabb.max[0] - aabb.min[0]) * (aabb.max[1] - aabb.min[1]) * (aabb.max[2] - aabb.min[2]);
        let target_neighbors_per_cell = 8.0;
        let optimal_cells = self.num_nodes as f32 / target_neighbors_per_cell;
        let optimal_cell_size = (scene_volume / optimal_cells).powf(1.0 / 3.0);


        let auto_tuned_cell_size = if optimal_cell_size > 10.0 && optimal_cell_size < 1000.0 {
            optimal_cell_size
        } else {
            params.grid_cell_size
        };

        debug!(
            "Spatial hashing: scene_volume={:.2}, optimal_cell_size={:.2}, using_size={:.2}",
            scene_volume, optimal_cell_size, auto_tuned_cell_size
        );


        aabb.min[0] -= auto_tuned_cell_size;
        aabb.max[0] += auto_tuned_cell_size;
        aabb.min[1] -= auto_tuned_cell_size;
        aabb.max[1] += auto_tuned_cell_size;
        aabb.min[2] -= auto_tuned_cell_size;
        aabb.max[2] += auto_tuned_cell_size;


        let grid_dims = int3 {
            x: ((aabb.max[0] - aabb.min[0]) / auto_tuned_cell_size).ceil() as i32,
            y: ((aabb.max[1] - aabb.min[1]) / auto_tuned_cell_size).ceil() as i32,
            z: ((aabb.max[2] - aabb.min[2]) / auto_tuned_cell_size).ceil() as i32,
        };
        let num_grid_cells = (grid_dims.x * grid_dims.y * grid_dims.z) as usize;


        let occupancy = self.get_grid_occupancy(num_grid_cells);
        if occupancy < 0.1 {
            warn!("Low grid occupancy detected: {:.1}% (avg {:.1} nodes/cell). Consider larger cell size.",
                  occupancy * 100.0, self.num_nodes as f32 / num_grid_cells as f32);
        } else if occupancy > 2.0 {
            warn!("High grid occupancy detected: {:.1}% (avg {:.1} nodes/cell). Consider smaller cell size.",
                  occupancy * 100.0, self.num_nodes as f32 / num_grid_cells as f32);
        }


        if num_grid_cells > self.max_grid_cells {
            self.resize_cell_buffers(num_grid_cells)?;
            debug!(
                "Grid buffer resize completed. Current grid: {}x{}x{} = {} cells",
                grid_dims.x, grid_dims.y, grid_dims.z, num_grid_cells
            );
        }


        crate::utils::gpu_diagnostics::validate_kernel_launch(
            self.build_grid_kernel_name,
            grid_size,
            block_size,
            self.num_nodes,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
        let build_grid_kernel = self
            ._module
            .get_function(self.build_grid_kernel_name)
            .map_err(|e| {
                let diagnosis = crate::utils::gpu_diagnostics::diagnose_ptx_error(&format!(
                    "Kernel '{}' not found: {}",
                    self.build_grid_kernel_name, e
                ));
                anyhow!(
                    "Failed to get kernel function '{}':\n{}",
                    self.build_grid_kernel_name,
                    diagnosis
                )
            })?;
        // SAFETY: Grid building kernel launch is safe because:
        // 1. pos_in_* buffers are valid DeviceBuffers with capacity >= num_nodes
        // 2. cell_keys buffer is sized for allocated_nodes elements
        // 3. aabb and grid_dims are computed from valid position data
        // 4. auto_tuned_cell_size is a positive float computed from AABB dimensions
        // 5. validate_kernel_launch() was called above to verify launch parameters
        unsafe {
            let stream = &self.stream;
            launch!(
                build_grid_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                self.pos_in_x.as_device_ptr(),
                self.pos_in_y.as_device_ptr(),
                self.pos_in_z.as_device_ptr(),
                self.cell_keys.as_device_ptr(),
                aabb,
                grid_dims,
                auto_tuned_cell_size,
                self.num_nodes as i32
            ))?;
        }


        let d_keys_in = self.cell_keys.as_slice();
        let d_values_in = self.sorted_node_indices.as_slice();

        let d_keys_out = DeviceBuffer::<i32>::zeroed(self.allocated_nodes)?;
        let mut d_values_out = DeviceBuffer::<i32>::zeroed(self.allocated_nodes)?;

        // SAFETY: Thrust sort FFI call is safe because:
        // 1. d_keys_in (cell_keys) is a valid DeviceBuffer allocated for allocated_nodes elements
        // 2. d_keys_out is a freshly allocated DeviceBuffer::zeroed(allocated_nodes)
        // 3. d_values_in (sorted_node_indices) is a valid DeviceBuffer for allocated_nodes elements
        // 4. d_values_out is a freshly allocated DeviceBuffer::zeroed(allocated_nodes)
        // 5. num_items is bounded by min(num_nodes, allocated_nodes) preventing out-of-bounds
        // 6. stream_ptr is obtained from a valid cust::Stream via as_inner()
        // 7. Thrust internally synchronizes on the provided stream before returning
        unsafe {
            let stream_ptr = self.stream.as_inner() as *mut ::std::os::raw::c_void;
            thrust_sort_key_value(
                d_keys_in.as_device_ptr().as_raw() as *const ::std::os::raw::c_void,
                d_keys_out.as_device_ptr().as_raw() as *mut ::std::os::raw::c_void,
                d_values_in.as_device_ptr().as_raw() as *const ::std::os::raw::c_void,
                d_values_out.as_device_ptr().as_raw() as *mut ::std::os::raw::c_void,
                self.num_nodes.min(self.allocated_nodes) as ::std::os::raw::c_int,
                stream_ptr,
            );
        }

        let sorted_keys = d_keys_out;

        std::mem::swap(&mut self.sorted_node_indices, &mut d_values_out);




        self.cell_start.copy_from(&self.zero_buffer)?;
        self.cell_end.copy_from(&self.zero_buffer)?;

        let grid_cells_blocks = (num_grid_cells as u32 + 255) / 256;
        let compute_cell_bounds_kernel = self
            ._module
            .get_function(self.compute_cell_bounds_kernel_name)?;
        // SAFETY: Cell bounds kernel launch is safe because:
        // 1. sorted_keys is the output from thrust_sort_key_value (valid device memory)
        // 2. cell_start and cell_end were zeroed and have capacity >= num_grid_cells
        // 3. num_grid_cells was computed from validated grid dimensions
        // 4. The kernel reads sorted_keys and writes cell boundaries atomically
        unsafe {
            let stream = &self.stream;
            launch!(
                compute_cell_bounds_kernel<<<grid_cells_blocks, 256, 0, stream>>>(
                sorted_keys.as_device_ptr(),
                self.cell_start.as_device_ptr(),
                self.cell_end.as_device_ptr(),
                self.num_nodes as i32,
                num_grid_cells as i32
            ))?;
        }



        let force_kernel_name = if params.stability_threshold > 0.0 {
            "force_pass_with_stability_kernel"
        } else {
            self.force_pass_kernel_name
        };
        let force_pass_kernel = self._module.get_function(force_kernel_name)?;
        let stream = &self.stream;


        let d_sssp = if (self.sssp_available || self.sssp_device_distances.is_some())
            && (params.feature_flags
                & crate::models::simulation_params::FeatureFlags::ENABLE_SSSP_SPRING_ADJUST
                != 0)
        {
            // Prefer the persistent sssp_device_distances buffer (stable across run_sssp calls)
            // over self.dist which is the working buffer that gets overwritten each SSSP run.
            match &self.sssp_device_distances {
                Some(buf) => buf.as_device_ptr(),
                None => self.dist.as_device_ptr(),
            }
        } else {
            DevicePointer::null()
        };

        // SAFETY: Force computation kernel launch is safe because:
        // 1. All position, velocity, and force buffers are valid DeviceBuffers with capacity >= num_nodes
        // 2. cell_start, cell_end, sorted_node_indices, cell_keys are from the spatial grid build phase
        // 3. edge_row_offsets, edge_col_indices, edge_weights are CSR graph data loaded at construction
        // 4. d_sssp is either a valid DevicePointer to dist buffer or DevicePointer::null()
        // 5. constraint_data has capacity for num_constraints ConstraintData elements
        // 6. should_skip_physics is a valid single-element DeviceBuffer for stability gating
        // 7. grid_size and block_size are validated via validate_kernel_launch()
        unsafe {
            if params.stability_threshold > 0.0 {
                // Force pass with stability checking variant
                launch!(
                    force_pass_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.vel_in_x.as_device_ptr(),
                    self.vel_in_y.as_device_ptr(),
                    self.vel_in_z.as_device_ptr(),
                    self.force_x.as_device_ptr(),
                    self.force_y.as_device_ptr(),
                    self.force_z.as_device_ptr(),
                    self.cell_start.as_device_ptr(),
                    self.cell_end.as_device_ptr(),
                    self.sorted_node_indices.as_device_ptr(),
                    self.cell_keys.as_device_ptr(),
                    grid_dims,
                    self.edge_row_offsets.as_device_ptr(),
                    self.edge_col_indices.as_device_ptr(),
                    self.edge_weights.as_device_ptr(),
                    self.num_nodes as i32,
                    d_sssp,
                    self.constraint_data.as_device_ptr(),
                    self.num_constraints as i32,
                    self.should_skip_physics.as_device_ptr()
                ))?;
            } else {

                launch!(
                    force_pass_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                    self.pos_in_x.as_device_ptr(),
                    self.pos_in_y.as_device_ptr(),
                    self.pos_in_z.as_device_ptr(),
                    self.force_x.as_device_ptr(),
                    self.force_y.as_device_ptr(),
                    self.force_z.as_device_ptr(),
                    self.cell_start.as_device_ptr(),
                    self.cell_end.as_device_ptr(),
                    self.sorted_node_indices.as_device_ptr(),
                    self.cell_keys.as_device_ptr(),
                    grid_dims,
                    self.edge_row_offsets.as_device_ptr(),
                    self.edge_col_indices.as_device_ptr(),
                    self.edge_weights.as_device_ptr(),
                    self.num_nodes as i32,
                    d_sssp,
                    self.constraint_data.as_device_ptr(),
                    self.num_constraints as i32,
                    DevicePointer::<f32>::null(),
                    DevicePointer::<f32>::null(),
                    DevicePointer::<f32>::null(),
                    // Ontology class metadata
                    self.class_id.as_device_ptr(),
                    self.class_charge.as_device_ptr(),
                    self.class_mass.as_device_ptr()
                ))?;
            }
        }


        let integrate_pass_kernel = self._module.get_function(self.integrate_pass_kernel_name)?;
        let stream = &self.stream;
        // SAFETY: Integration kernel launch is safe because:
        // 1. All input buffers (pos_in_*, vel_in_*, force_*, mass) contain data from force pass
        // 2. All output buffers (pos_out_*, vel_out_*) are valid DeviceBuffers with capacity >= num_nodes
        // 3. class_id, class_charge, class_mass are ontology metadata buffers loaded at construction
        // 4. The kernel performs Verlet integration using c_params constants from device memory
        // 5. After this kernel, swap_buffers() exchanges input/output for next iteration
        unsafe {
            launch!(
                integrate_pass_kernel<<<grid_size as u32, block_size as u32, 0, stream>>>(
                self.pos_in_x.as_device_ptr(),
                self.pos_in_y.as_device_ptr(),
                self.pos_in_z.as_device_ptr(),
                self.vel_in_x.as_device_ptr(),
                self.vel_in_y.as_device_ptr(),
                self.vel_in_z.as_device_ptr(),
                self.force_x.as_device_ptr(),
                self.force_y.as_device_ptr(),
                self.force_z.as_device_ptr(),
                self.mass.as_device_ptr(),
                self.pos_out_x.as_device_ptr(),
                self.pos_out_y.as_device_ptr(),
                self.pos_out_z.as_device_ptr(),
                self.vel_out_x.as_device_ptr(),
                self.vel_out_y.as_device_ptr(),
                self.vel_out_z.as_device_ptr(),
                self.num_nodes as i32,
                // Ontology class metadata
                self.class_id.as_device_ptr(),
                self.class_charge.as_device_ptr(),
                self.class_mass.as_device_ptr()
            ))?;
        }



        let completion_event = cust::event::Event::new(cust::event::EventFlags::DEFAULT)?;
        completion_event.record(&self.stream)?;


        while completion_event
            .query()
            .unwrap_or(cust::event::EventStatus::Ready)
            != cust::event::EventStatus::Ready
        {

            std::thread::yield_now();
        }

        self.swap_buffers();
        self.iteration += 1;


        if self.iteration % 100 == 0 {
            let (memory_used, utilization, resize_count) = self.get_memory_metrics();
            let grid_occupancy = self.get_grid_occupancy(num_grid_cells);
            info!("Performance metrics [iter {}]: Memory: {:.1}MB ({:.1}% utilized), Grid occupancy: {:.1}%, Resizes: {}",
                  self.iteration, memory_used as f32 / 1024.0 / 1024.0,
                  utilization * 100.0, grid_occupancy * 100.0, resize_count);
        }

        Ok(())
    }

    pub fn execute_physics_step(
        &mut self,
        params: &crate::models::simulation_params::SimulationParams,
    ) -> Result<()> {
        self.execute_physics_step_with_bypass(params, false)
    }

    pub fn execute_physics_step_with_bypass(
        &mut self,
        params: &crate::models::simulation_params::SimulationParams,
        stability_bypass: bool,
    ) -> Result<()> {
        // Build feature_flags from the SimulationParams and runtime toggles,
        // mirroring the logic in SimulationParams::to_sim_params().
        let mut feature_flags: u32 = 0;
        if params.repel_k > 0.0 {
            feature_flags |= crate::models::simulation_params::FeatureFlags::ENABLE_REPULSION;
        }
        if params.spring_k > 0.0 {
            feature_flags |= crate::models::simulation_params::FeatureFlags::ENABLE_SPRINGS;
        }
        if params.center_gravity_k > 0.0 {
            feature_flags |= crate::models::simulation_params::FeatureFlags::ENABLE_CENTERING;
        }
        // Honour both the SimulationParams flag and the runtime toggle
        if params.use_sssp_distances || self.sssp_spring_adjust_enabled {
            feature_flags |= crate::models::simulation_params::FeatureFlags::ENABLE_SSSP_SPRING_ADJUST;
        }

        // Use SimulationParams::to_sim_params() which correctly maps ALL user-facing
        // settings to the GPU-compatible SimParams struct. Previous implementation
        // hardcoded many values (temperature, separation_radius, repulsion_cutoff, etc.)
        // which caused "nothing moves when I change settings" because those settings
        // never reached the GPU kernel.
        let mut sim_params = params.to_sim_params();
        sim_params.feature_flags = feature_flags;

        // When stability_bypass is true, disable the GPU stability check so physics
        // runs unconditionally. This prevents the check_system_stability_kernel from
        // skipping physics when the system was at equilibrium before a parameter change.
        if stability_bypass {
            sim_params.stability_threshold = 0.0;
        }

        // Log GPU params on first iteration to confirm forces are enabled
        if self.iteration == 0 {
            info!(
                "GPU execute_physics_step: FIRST iter — feature_flags=0b{:b} (repel={}, spring={}, center={}), dt={}, repel_k={}, spring_k={}, damping={}, stability_bypass={}",
                feature_flags,
                feature_flags & 1 != 0,
                feature_flags & 2 != 0,
                feature_flags & 4 != 0,
                sim_params.dt, sim_params.repel_k, sim_params.spring_k,
                sim_params.damping, stability_bypass
            );
        }

        self.execute(sim_params)
    }

    pub fn get_node_positions(&mut self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {


        let mut pos_x = vec![0.0f32; self.allocated_nodes];
        let mut pos_y = vec![0.0f32; self.allocated_nodes];
        let mut pos_z = vec![0.0f32; self.allocated_nodes];


        self.pos_in_x.copy_to(&mut pos_x)?;
        self.pos_in_y.copy_to(&mut pos_y)?;
        self.pos_in_z.copy_to(&mut pos_z)?;


        pos_x.truncate(self.num_nodes);
        pos_y.truncate(self.num_nodes);
        pos_z.truncate(self.num_nodes);

        Ok((pos_x, pos_y, pos_z))
    }

    pub fn get_node_velocities(&mut self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {


        let mut vel_x = vec![0.0f32; self.allocated_nodes];
        let mut vel_y = vec![0.0f32; self.allocated_nodes];
        let mut vel_z = vec![0.0f32; self.allocated_nodes];


        self.vel_in_x.copy_to(&mut vel_x)?;
        self.vel_in_y.copy_to(&mut vel_y)?;
        self.vel_in_z.copy_to(&mut vel_z)?;


        vel_x.truncate(self.num_nodes);
        vel_y.truncate(self.num_nodes);
        vel_z.truncate(self.num_nodes);

        Ok((vel_x, vel_y, vel_z))
    }
}
