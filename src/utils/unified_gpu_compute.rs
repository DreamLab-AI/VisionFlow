pub use crate::models::simulation_params::SimParams;
use anyhow::{anyhow, Result};
use cust::context::Context;
use cust::device::Device;
use cust::launch;
use cust::memory::{DeviceBuffer, CopyDestination, DevicePointer};
use cust_core::DeviceCopy;
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};

// External CUDA/Thrust function for sorting
// This is provided by the compiled CUDA object file
unsafe extern "C" {
    fn thrust_sort_key_value(
        d_keys_in: *const ::std::os::raw::c_void,
        d_keys_out: *mut ::std::os::raw::c_void,
        d_values_in: *const ::std::os::raw::c_void,
        d_values_out: *mut ::std::os::raw::c_void,
        num_items: ::std::os::raw::c_int,
        stream: *mut ::std::os::raw::c_void,
    );
}

// Define AABB and int3 structs to match CUDA
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, DeviceCopy)]
struct AABB {
    min: [f32; 3],
    max: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, DeviceCopy)]
struct int3 {
    x: i32,
    y: i32,
    z: i32,
}

pub struct UnifiedGPUCompute {
    // Context and modules
    _context: Context,
    _module: Module,
    stream: Stream,

    // Kernel names for lookup
    build_grid_kernel_name: &'static str,
    compute_cell_bounds_kernel_name: &'static str,
    force_pass_kernel_name: &'static str,
    integrate_pass_kernel_name: &'static str,

    // Node data (double buffered)
    pub pos_in_x: DeviceBuffer<f32>,
    pub pos_in_y: DeviceBuffer<f32>,
    pub pos_in_z: DeviceBuffer<f32>,
    pub vel_in_x: DeviceBuffer<f32>,
    pub vel_in_y: DeviceBuffer<f32>,
    pub vel_in_z: DeviceBuffer<f32>,

    pub pos_out_x: DeviceBuffer<f32>,
    pub pos_out_y: DeviceBuffer<f32>,
    pub pos_out_z: DeviceBuffer<f32>,
    pub vel_out_x: DeviceBuffer<f32>,
    pub vel_out_y: DeviceBuffer<f32>,
    pub vel_out_z: DeviceBuffer<f32>,

    // Other node data
    pub mass: DeviceBuffer<f32>,
    pub node_graph_id: DeviceBuffer<i32>,

    // Edge data (CSR format)
    pub edge_row_offsets: DeviceBuffer<i32>,
    pub edge_col_indices: DeviceBuffer<i32>,
    pub edge_weights: DeviceBuffer<f32>,

    // Force buffer
    force_x: DeviceBuffer<f32>,
    force_y: DeviceBuffer<f32>,
    force_z: DeviceBuffer<f32>,

    // Spatial grid data
    cell_keys: DeviceBuffer<i32>,
    sorted_node_indices: DeviceBuffer<i32>,
    cell_start: DeviceBuffer<i32>,
    cell_end: DeviceBuffer<i32>,
    
    // Temporary storage for CUB
    cub_temp_storage: DeviceBuffer<u8>,

    // State
    num_nodes: usize,
    num_edges: usize,
    allocated_nodes: usize,  // Track allocated buffer size
    allocated_edges: usize,  // Track allocated buffer size
    max_grid_cells: usize,   // Track allocated grid cell buffer size
    iteration: i32,
    
    // Reusable host buffer for zeroing grid cells
    zero_buffer: Vec<i32>,
    
    // SSSP state
    pub dist: DeviceBuffer<f32>,                // [n] distances
    pub current_frontier: DeviceBuffer<i32>,    // Dynamic frontier
    pub next_frontier_flags: DeviceBuffer<i32>, // [n] flags
    pub parents: Option<DeviceBuffer<i32>>,     // Optional for paths
    
    // Dedicated SSSP stream for overlap
    sssp_stream: Option<Stream>,
    
    // State validity flag
    pub sssp_available: bool,
}


impl UnifiedGPUCompute {
    pub fn new(num_nodes: usize, num_edges: usize, ptx_content: &str) -> Result<Self> {
        let device = Device::get_device(0)?;
        let _context = Context::new(device)?;
        let module = Module::from_ptx(ptx_content, &[])?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        // Allocate double buffers for position and velocity
        let pos_in_x = DeviceBuffer::zeroed(num_nodes)?;
        let pos_in_y = DeviceBuffer::zeroed(num_nodes)?;
        let pos_in_z = DeviceBuffer::zeroed(num_nodes)?;
        let vel_in_x = DeviceBuffer::zeroed(num_nodes)?;
        let vel_in_y = DeviceBuffer::zeroed(num_nodes)?;
        let vel_in_z = DeviceBuffer::zeroed(num_nodes)?;

        let pos_out_x = DeviceBuffer::zeroed(num_nodes)?;
        let pos_out_y = DeviceBuffer::zeroed(num_nodes)?;
        let pos_out_z = DeviceBuffer::zeroed(num_nodes)?;
        let vel_out_x = DeviceBuffer::zeroed(num_nodes)?;
        let vel_out_y = DeviceBuffer::zeroed(num_nodes)?;
        let vel_out_z = DeviceBuffer::zeroed(num_nodes)?;

        // Allocate other buffers
        let mass = DeviceBuffer::from_slice(&vec![1.0f32; num_nodes])?;
        let node_graph_id = DeviceBuffer::zeroed(num_nodes)?;
        let edge_row_offsets = DeviceBuffer::zeroed(num_nodes + 1)?;
        let edge_col_indices = DeviceBuffer::zeroed(num_edges)?;
        let edge_weights = DeviceBuffer::zeroed(num_edges)?;
        let force_x = DeviceBuffer::zeroed(num_nodes)?;
        let force_y = DeviceBuffer::zeroed(num_nodes)?;
        let force_z = DeviceBuffer::zeroed(num_nodes)?;

        // Allocate spatial grid buffers
        let cell_keys = DeviceBuffer::zeroed(num_nodes)?;
        let mut sorted_node_indices = DeviceBuffer::zeroed(num_nodes)?;
        // Initialize sorted_node_indices with 0, 1, 2, ...
        let initial_indices: Vec<i32> = (0..num_nodes as i32).collect();
        sorted_node_indices.copy_from(&initial_indices)?;

        // Grid dimensions will be calculated on the fly, but we need a buffer for cell starts/ends.
        // Allocate for a reasonably large grid, e.g., 128^3. This can be resized if needed.
        let max_grid_cells = 128 * 128 * 128;
        let cell_start = DeviceBuffer::zeroed(max_grid_cells)?;
        let cell_end = DeviceBuffer::zeroed(max_grid_cells)?;

        // Allocate temporary storage for CUB operations (sorting, prefix sum)
        let cub_temp_storage = Self::calculate_cub_temp_storage(
            num_nodes,
            max_grid_cells,
        )?;

        // SSSP buffers
        let dist = DeviceBuffer::from_slice(&vec![f32::INFINITY; num_nodes])?;
        let current_frontier = DeviceBuffer::zeroed(num_nodes)?;
        let next_frontier_flags = DeviceBuffer::zeroed(num_nodes)?;
        let sssp_stream = Some(Stream::new(StreamFlags::NON_BLOCKING, None)?);
        
        // Store the module for kernel lookup
        let kernel_module = module;

        Ok(Self {
            _context,
            _module: kernel_module,
            stream,
            build_grid_kernel_name: "build_grid_kernel",
            compute_cell_bounds_kernel_name: "compute_cell_bounds_kernel",
            force_pass_kernel_name: "force_pass_kernel",
            integrate_pass_kernel_name: "integrate_pass_kernel",
            pos_in_x,
            pos_in_y,
            pos_in_z,
            vel_in_x,
            vel_in_y,
            vel_in_z,
            pos_out_x,
            pos_out_y,
            pos_out_z,
            vel_out_x,
            vel_out_y,
            vel_out_z,
            mass,
            node_graph_id,
            edge_row_offsets,
            edge_col_indices,
            edge_weights,
            force_x,
            force_y,
            force_z,
            cell_keys,
            sorted_node_indices,
            cell_start,
            cell_end,
            cub_temp_storage,
            num_nodes,
            num_edges,
            allocated_nodes: num_nodes,
            allocated_edges: num_edges,
            max_grid_cells,
            iteration: 0,
            zero_buffer: vec![0i32; max_grid_cells], // Pre-allocate for reuse
            // SSSP fields
            dist,
            current_frontier,
            next_frontier_flags,
            parents: None,  // Optional, not initialized by default
            sssp_stream,
            sssp_available: false,
        })
    }

    fn calculate_cub_temp_storage(num_nodes: usize, num_cells: usize) -> Result<DeviceBuffer<u8>> {
        let mut sort_bytes = 0;
        let mut scan_bytes = 0;
        let mut error;

        // Get storage size for sorting
        let d_keys_temp = DeviceBuffer::<i32>::zeroed(0)?;
        let d_keys_null = d_keys_temp.as_slice();
        let d_values_temp = DeviceBuffer::<i32>::zeroed(0)?;
        let d_values_null = d_values_temp.as_slice();
        // Thrust handles temp storage internally
        sort_bytes = 0; // Not needed with Thrust
        error = 0; // Success
        if error != 0 {
            return Err(anyhow!("CUB sort storage calculation failed with code {}", error));
        }

        // Get storage size for prefix sum (scan)
        let d_scan_temp = DeviceBuffer::<i32>::zeroed(0)?;
        let d_scan_null = d_scan_temp.as_slice();
        // Thrust handles temp storage internally
        scan_bytes = 0; // Not needed with Thrust
        error = 0; // Success
        if error != 0 {
            return Err(anyhow!("CUB scan storage calculation failed with code {}", error));
        }

        let total_bytes = sort_bytes.max(scan_bytes);
        DeviceBuffer::zeroed(total_bytes)
            .map_err(|e| anyhow!("Failed to allocate CUB temp storage: {}", e))
    }

    pub fn upload_positions(&mut self, x: &[f32], y: &[f32], z: &[f32]) -> Result<()> {
        // Check sizes match
        if x.len() != self.num_nodes || y.len() != self.num_nodes || z.len() != self.num_nodes {
            return Err(anyhow!(
                "Position array size mismatch: expected {} nodes, got x:{}, y:{}, z:{}",
                self.num_nodes, x.len(), y.len(), z.len()
            ));
        }
        
        self.pos_in_x.copy_from(x)?;
        self.pos_in_y.copy_from(y)?;
        self.pos_in_z.copy_from(z)?;
        Ok(())
    }

    pub fn upload_edges_csr(&mut self, row_offsets: &[i32], col_indices: &[i32], weights: &[f32]) -> Result<()> {
        // Check row_offsets size
        if row_offsets.len() != self.num_nodes + 1 {
            return Err(anyhow!(
                "Row offsets size mismatch: expected {} (num_nodes + 1), got {}",
                self.num_nodes + 1, row_offsets.len()
            ));
        }
        
        // Check that edge data arrays have same length
        if col_indices.len() != weights.len() {
            return Err(anyhow!(
                "Edge arrays size mismatch: col_indices has {}, weights has {}",
                col_indices.len(), weights.len()
            ));
        }
        
        // Check that we don't exceed allocated edge buffer size
        if col_indices.len() > self.allocated_edges {
            return Err(anyhow!(
                "Too many edges: trying to upload {}, but only {} allocated",
                col_indices.len(), self.allocated_edges
            ));
        }
        
        self.edge_row_offsets.copy_from(row_offsets)?;
        self.edge_col_indices.copy_from(col_indices)?;
        self.edge_weights.copy_from(weights)?;
        self.num_edges = col_indices.len();
        Ok(())
    }

    pub fn download_positions(&self, x: &mut [f32], y: &mut [f32], z: &mut [f32]) -> Result<()> {
        self.pos_in_x.copy_to(x)?;
        self.pos_in_y.copy_to(y)?;
        self.pos_in_z.copy_to(z)?;
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

    // NOTE: resize_buffers is intentionally not implemented as it would require
    // reallocating all DeviceBuffers which is complex and error-prone.
    // Instead, create a new UnifiedGPUCompute instance when size requirements change.
    // pub fn resize_buffers(&mut self, num_nodes: usize, num_edges: usize) -> Result<()> {
    //     unimplemented!("Use a new UnifiedGPUCompute instance for different sizes")
    // }

    pub fn set_params(&mut self, _params: SimParams) {
        // This is a placeholder. A real implementation would likely copy the params
        // to a constant memory buffer on the GPU.
    }

    pub fn set_mode(&mut self, _mode: ComputeMode) {
        // Placeholder for setting compute mode
    }

    pub fn set_constraints(&mut self, _constraints: Vec<ConstraintData>) -> Result<()> {
        // Placeholder for setting constraints
        Ok(())
    }

    pub fn execute(&mut self, mut params: SimParams) -> Result<()> {
        params.iteration = self.iteration;
        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;

        // 1. Calculate AABB (on CPU for now, can be moved to GPU later)
        // Use allocated_nodes for buffer sizes to ensure they match GPU buffers
        let mut host_pos_x = vec![0.0; self.allocated_nodes];
        self.pos_in_x.copy_to(&mut host_pos_x)?;
        let mut host_pos_y = vec![0.0; self.allocated_nodes];
        self.pos_in_y.copy_to(&mut host_pos_y)?;
        let mut host_pos_z = vec![0.0; self.allocated_nodes];
        self.pos_in_z.copy_to(&mut host_pos_z)?;

        let mut aabb = AABB {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        };
        // Only iterate over actual nodes, not allocated buffer size
        for i in 0..self.num_nodes {
            aabb.min[0] = aabb.min[0].min(host_pos_x[i]);
            aabb.min[1] = aabb.min[1].min(host_pos_y[i]);
            aabb.min[2] = aabb.min[2].min(host_pos_z[i]);
            aabb.max[0] = aabb.max[0].max(host_pos_x[i]);
            aabb.max[1] = aabb.max[1].max(host_pos_y[i]);
            aabb.max[2] = aabb.max[2].max(host_pos_z[i]);
        }
        // Add padding to AABB
        aabb.min[0] -= params.grid_cell_size; aabb.max[0] += params.grid_cell_size;
        aabb.min[1] -= params.grid_cell_size; aabb.max[1] += params.grid_cell_size;
        aabb.min[2] -= params.grid_cell_size; aabb.max[2] += params.grid_cell_size;

        // 2. Define grid dimensions
        let grid_dims = int3 {
            x: ((aabb.max[0] - aabb.min[0]) / params.grid_cell_size).ceil() as i32,
            y: ((aabb.max[1] - aabb.min[1]) / params.grid_cell_size).ceil() as i32,
            z: ((aabb.max[2] - aabb.min[2]) / params.grid_cell_size).ceil() as i32,
        };
        let num_grid_cells = (grid_dims.x * grid_dims.y * grid_dims.z) as usize;

        if num_grid_cells > self.max_grid_cells {
            return Err(anyhow!("Grid size {} exceeds allocated buffer {}. Re-allocate with larger max_grid_cells.", 
                       num_grid_cells, self.max_grid_cells));
        }

        // 3. Build Grid: Assign cell keys to each node
        let build_grid_kernel = self._module.get_function(self.build_grid_kernel_name)?;
        unsafe {
            let stream = &self.stream;
            launch!(
                build_grid_kernel<<<grid_size, block_size, 0, stream>>>(
                self.pos_in_x.as_device_ptr(),
                self.pos_in_y.as_device_ptr(),
                self.pos_in_z.as_device_ptr(),
                self.cell_keys.as_device_ptr(),
                aabb,
                grid_dims,
                params.grid_cell_size,
                self.num_nodes as i32
            ))?;
        }

        // 4. Sort nodes by cell key
        let d_keys_in = self.cell_keys.as_slice();
        let d_values_in = self.sorted_node_indices.as_slice();
        // Use allocated_nodes to match buffer sizes
        let d_keys_out = DeviceBuffer::<i32>::zeroed(self.allocated_nodes)?;
        let mut d_values_out = DeviceBuffer::<i32>::zeroed(self.allocated_nodes)?;
        
        unsafe {
            // Get the raw CUDA stream handle to ensure Thrust uses the same stream as kernels
            let stream_ptr = self.stream.as_inner() as *mut ::std::os::raw::c_void;
            thrust_sort_key_value(
                d_keys_in.as_device_ptr().as_raw() as *const ::std::os::raw::c_void,
                d_keys_out.as_device_ptr().as_raw() as *mut ::std::os::raw::c_void,
                d_values_in.as_device_ptr().as_raw() as *const ::std::os::raw::c_void,
                d_values_out.as_device_ptr().as_raw() as *mut ::std::os::raw::c_void,
                self.num_nodes as ::std::os::raw::c_int,
                stream_ptr, // Use our custom stream, not default
            );
        }
        // The sorted keys are in d_keys_out, sorted values (node indices) in d_values_out
        let sorted_keys = d_keys_out;
        // We need the sorted node indices for the force kernel, so we swap it into our struct
        std::mem::swap(&mut self.sorted_node_indices, &mut d_values_out);

        // 5. Find cell start/end indices using our new kernel
        // Zero out the full allocated buffers to ensure all cells are initialized
        // Use pre-allocated zero buffer to avoid allocation every frame
        if num_grid_cells <= self.max_grid_cells {
            self.cell_start.copy_from(&self.zero_buffer)?;
            self.cell_end.copy_from(&self.zero_buffer)?;
        } else {
            return Err(anyhow!("Grid cells {} exceeds max {}", num_grid_cells, self.max_grid_cells));
        }

        let grid_cells_blocks = (num_grid_cells as u32 + 255) / 256;
        let compute_cell_bounds_kernel = self._module.get_function(self.compute_cell_bounds_kernel_name)?;
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

        // 6. Force Pass Kernel
        let force_pass_kernel = self._module.get_function(self.force_pass_kernel_name)?;
        let stream = &self.stream;
        
        // Get SSSP distances pointer if available
        let d_sssp = if self.sssp_available && 
                      (params.feature_flags & crate::models::simulation_params::FeatureFlags::ENABLE_SSSP_SPRING_ADJUST != 0) {
            self.dist.as_device_ptr()
        } else {
            DevicePointer::null()
        };
        
        unsafe {
            launch!(
                force_pass_kernel<<<grid_size, block_size, 0, stream>>>(
                self.pos_in_x.as_device_ptr(),
                self.pos_in_y.as_device_ptr(),
                self.pos_in_z.as_device_ptr(),
                self.force_x.as_device_ptr(),
                self.force_y.as_device_ptr(),
                self.force_z.as_device_ptr(),
                self.cell_start.as_device_ptr(),
                self.cell_end.as_device_ptr(),
                self.sorted_node_indices.as_device_ptr(),
                self.cell_keys.as_device_ptr(), // Unsorted keys, but matches original node indices
                grid_dims,
                self.edge_row_offsets.as_device_ptr(),
                self.edge_col_indices.as_device_ptr(),
                self.edge_weights.as_device_ptr(),
                params,
                self.num_nodes as i32,
                d_sssp
            ))?;
        }

        // 7. Integration Pass Kernel
        let integrate_pass_kernel = self._module.get_function(self.integrate_pass_kernel_name)?;
        let stream = &self.stream;
        unsafe {
            launch!(
                integrate_pass_kernel<<<grid_size, block_size, 0, stream>>>(
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
                params,
                self.num_nodes as i32
            ))?;
        }

        self.stream.synchronize()?;
        self.swap_buffers();
        self.iteration += 1;

        Ok(())
    }
    
    pub fn run_sssp(&mut self, source_idx: usize) -> Result<Vec<f32>> {
        // Invalidate previous results immediately
        self.sssp_available = false;
        
        // Wrap main logic for clean error handling
        let result = (|| -> Result<Vec<f32>> {
            // Initialize distances
            let mut host_dist = vec![f32::INFINITY; self.num_nodes];
            host_dist[source_idx] = 0.0;
            self.dist.copy_from(&host_dist)?;
            
            // Initialize frontier
            let mut host_frontier = vec![source_idx as i32];
            self.current_frontier = DeviceBuffer::from_slice(&host_frontier)?;
            
            // Compute k parameter
            let k = ((self.num_nodes as f32).log2().cbrt().ceil() as u32).max(3);
            let s = self.sssp_stream.as_ref().unwrap_or(&self.stream);
            
            // Main iteration loop
            for iteration in 0..k {
                // Clear next frontier flags
                let zeros = vec![0i32; self.num_nodes];
                self.next_frontier_flags.copy_from(&zeros)?;
                
                // Check for convergence
                let frontier_len = host_frontier.len();
                if frontier_len == 0 {
                    log::debug!("SSSP converged at iteration {}", iteration);
                    break;
                }
                
                // Launch relaxation kernel
                let block = 256;
                let grid = ((frontier_len as u32 + block - 1) / block) as u32;
                
                let func = self._module.get_function("relaxation_step_kernel")?;
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
                
                // Host-side frontier compaction (v1)
                let mut flags = vec![0i32; self.num_nodes];
                self.next_frontier_flags.copy_to(&mut flags)?;
                
                host_frontier.clear();
                host_frontier.reserve(frontier_len * 2);
                for (i, &flag) in flags.iter().enumerate() {
                    if flag != 0 {
                        host_frontier.push(i as i32);
                    }
                }
                
                if !host_frontier.is_empty() {
                    self.current_frontier = DeviceBuffer::from_slice(&host_frontier)?;
                }
            }
            
            // Copy results back
            self.dist.copy_to(&mut host_dist)?;
            Ok(host_dist)
        })();
        
        // Handle result and update state
        match result {
            Ok(distances) => {
                self.sssp_available = true;
                log::info!("SSSP computation successful from source {}", source_idx);
                Ok(distances)
            }
            Err(e) => {
                self.sssp_available = false;
                log::error!("SSSP computation failed: {}. State invalidated.", e);
                Err(e)
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ComputeMode {
    Basic,
    Constraints,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ConstraintData {
    pub constraint_type: i32,
    pub strength: f32,
    pub param1: f32,
    pub param2: f32,
    pub node_mask: i32,
}

// Additional Thrust wrapper function for scanning
unsafe extern "C" {
    fn thrust_exclusive_scan(
        d_in: *const ::std::os::raw::c_void,
        d_out: *mut ::std::os::raw::c_void,
        num_items: ::std::os::raw::c_int,
        stream: *mut ::std::os::raw::c_void,
    );
}