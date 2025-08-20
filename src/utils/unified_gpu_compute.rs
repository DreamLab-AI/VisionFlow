use crate::models::simulation_params::{FeatureFlags, SimParams};
use anyhow::{anyhow, Result};
use cust::context::{Context, CurrentContext};
use cust::device::Device;
use cust::function::Function;
use cust::launch;
use cust::memory::{CopyDestination, DeviceBuffer, DeviceSlice};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use std::ffi::CString;
use std::sync::Arc;

// Define AABB and int3 structs to match CUDA
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct AABB {
    min: [f32; 3],
    max: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct int3 {
    x: i32,
    y: i32,
    z: i32,
}

pub struct UnifiedGPUCompute {
    // Context and modules
    _context: Context,
    module: Module,
    stream: Stream,

    // Kernels
    build_grid_kernel: Function<'static>,
    compute_cell_bounds_kernel: Function<'static>,
    force_pass_kernel: Function<'static>,
    integrate_pass_kernel: Function<'static>,

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
    iteration: i32,
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

        // Load kernels from the module
        let build_grid_kernel = module.get_function("build_grid_kernel")?.into_static();
        let compute_cell_bounds_kernel = module.get_function("compute_cell_bounds_kernel")?.into_static();
        let force_pass_kernel = module.get_function("force_pass_kernel")?.into_static();
        let integrate_pass_kernel = module.get_function("integrate_pass_kernel")?.into_static();

        Ok(Self {
            _context,
            module,
            stream,
            build_grid_kernel,
            compute_cell_bounds_kernel,
            force_pass_kernel,
            integrate_pass_kernel,
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
            iteration: 0,
        })
    }

    fn calculate_cub_temp_storage(num_nodes: usize, num_cells: usize) -> Result<DeviceBuffer<u8>> {
        let mut sort_bytes = 0;
        let mut scan_bytes = 0;
        let mut error = 0;

        // Get storage size for sorting
        let d_keys_null: DeviceSlice<i32> = DeviceBuffer::zeroed(0)?.as_slice();
        let d_values_null: DeviceSlice<i32> = DeviceBuffer::zeroed(0)?.as_slice();
        unsafe {
            error = cub_sys::CUB_SORT_PAIRS(
                std::ptr::null_mut(),
                &mut sort_bytes,
                d_keys_null.as_device_ptr(),
                d_keys_null.as_device_ptr(), // out
                d_values_null.as_device_ptr(),
                d_values_null.as_device_ptr(), // out
                num_nodes as i32,
                0,
                32,
                0 as cust_core::CUstream,
            );
        }
        if error != 0 {
            return Err(anyhow!("CUB sort storage calculation failed with code {}", error));
        }

        // Get storage size for prefix sum (scan)
        let d_scan_null: DeviceSlice<i32> = DeviceBuffer::zeroed(0)?.as_slice();
        unsafe {
            error = cub_sys::CUB_EXCLUSIVE_SCAN(
                std::ptr::null_mut(),
                &mut scan_bytes,
                d_scan_null.as_device_ptr(),
                d_scan_null.as_device_ptr(), // out
                cub_sys::cubSumOp,
                0,
                num_cells as i32,
                0 as cust_core::CUstream,
            );
        }
        if error != 0 {
            return Err(anyhow!("CUB scan storage calculation failed with code {}", error));
        }

        let total_bytes = sort_bytes.max(scan_bytes);
        DeviceBuffer::zeroed(total_bytes)
            .map_err(|e| anyhow!("Failed to allocate CUB temp storage: {}", e))
    }

    pub fn upload_positions(&mut self, x: &[f32], y: &[f32], z: &[f32]) -> Result<()> {
        self.pos_in_x.copy_from(x)?;
        self.pos_in_y.copy_from(y)?;
        self.pos_in_z.copy_from(z)?;
        Ok(())
    }

    pub fn upload_edges_csr(&mut self, row_offsets: &[i32], col_indices: &[i32], weights: &[f32]) -> Result<()> {
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

    pub fn execute(&mut self, mut params: SimParams) -> Result<()> {
        params.iteration = self.iteration;
        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;

        // 1. Calculate AABB (on CPU for now, can be moved to GPU later)
        let mut host_pos_x = vec![0.0; self.num_nodes];
        self.pos_in_x.copy_to(&mut host_pos_x)?;
        let mut host_pos_y = vec![0.0; self.num_nodes];
        self.pos_in_y.copy_to(&mut host_pos_y)?;
        let mut host_pos_z = vec![0.0; self.num_nodes];
        self.pos_in_z.copy_to(&mut host_pos_z)?;

        let mut aabb = AABB {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        };
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

        if num_grid_cells > self.cell_start.len() {
            return Err(anyhow!("Grid size exceeds allocated buffer. Re-allocate with larger max_grid_cells."));
        }

        // 3. Build Grid: Assign cell keys to each node
        unsafe {
            launch!(
                self.build_grid_kernel<
                self.pos_in_x.as_device_ptr(),
                self.pos_in_y.as_device_ptr(),
                self.pos_in_z.as_device_ptr(),
                self.cell_keys.as_device_ptr(),
                aabb,
                grid_dims,
                params.grid_cell_size,
                self.num_nodes as i32
            >>>(grid_size, block_size, 0, &self.stream))?;
        }

        // 4. Sort nodes by cell key
        let mut d_keys_in = self.cell_keys.as_slice();
        let mut d_values_in = self.sorted_node_indices.as_slice();
        let mut d_keys_out = DeviceBuffer::zeroed(self.num_nodes)?;
        let mut d_values_out = DeviceBuffer::zeroed(self.num_nodes)?;
        
        unsafe {
            let mut temp_storage_bytes = self.cub_temp_storage.len();
            cub_sys::CUB_SORT_PAIRS(
                self.cub_temp_storage.as_device_ptr() as *mut _,
                &mut temp_storage_bytes,
                d_keys_in.as_device_ptr(),
                d_keys_out.as_device_ptr(),
                d_values_in.as_device_ptr(),
                d_values_out.as_device_ptr(),
                self.num_nodes as i32,
                0,
                32,
                self.stream.as_inner(),
            );
        }
        // The sorted keys are in d_keys_out, sorted values (node indices) in d_values_out
        let sorted_keys = d_keys_out;
        // We need the sorted node indices for the force kernel, so we swap it into our struct
        std::mem::swap(&mut self.sorted_node_indices, &mut d_values_out);

        // 5. Find cell start/end indices using our new kernel
        // First, we need to zero out the cell_start and cell_end buffers as the kernel
        // only writes the boundaries. A `memset` would be more efficient.
        self.cell_start.copy_from(&vec![0i32; num_grid_cells])?;
        self.cell_end.copy_from(&vec![0i32; num_grid_cells])?;

        unsafe {
            launch!(
                self.compute_cell_bounds_kernel<
                sorted_keys.as_device_ptr(),
                self.cell_start.as_device_ptr(),
                self.cell_end.as_device_ptr(),
                self.num_nodes as i32,
                num_grid_cells as i32
            >>>(&self.stream))?;
        }

        // 6. Force Pass Kernel
        unsafe {
            launch!(
                self.force_pass_kernel<
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
                self.num_nodes as i32
            >>>(grid_size, block_size, 0, &self.stream))?;
        }

        // 7. Integration Pass Kernel
        unsafe {
            launch!(
                self.integrate_pass_kernel<
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
            >>>(grid_size, block_size, 0, &self.stream))?;
        }

        self.stream.synchronize()?;
        self.swap_buffers();
        self.iteration += 1;

        Ok(())
    }
}

// CUB FFI bindings
mod cub_sys {
    use cust_core::CUdeviceptr;
    use cust_core::CUstream;

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct cubSumOp {
        _unused: [u8; 0],
    }

    extern "C" {
        pub fn CUB_SORT_PAIRS(
            d_temp_storage: CUdeviceptr,
            temp_storage_bytes: *mut usize,
            d_keys_in: CUdeviceptr,
            d_keys_out: CUdeviceptr,
            d_values_in: CUdeviceptr,
            d_values_out: CUdeviceptr,
            num_items: ::std::os::raw::c_int,
            begin_bit: ::std::os::raw::c_int,
            end_bit: ::std::os::raw::c_int,
            stream: CUstream,
        ) -> ::std::os::raw::c_int;

        pub fn CUB_EXCLUSIVE_SCAN(
            d_temp_storage: CUdeviceptr,
            temp_storage_bytes: *mut usize,
            d_in: CUdeviceptr,
            d_out: CUdeviceptr,
            op: cubSumOp,
            initial_value: ::std::os::raw::c_int,
            num_items: ::std::os::raw::c_int,
            stream: CUstream,
        ) -> ::std::os::raw::c_int;
    }
}