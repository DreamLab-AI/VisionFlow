// Unified GPU Compute Module - Rewritten for correctness, performance, and clarity.
// Implements a two-pass (force/integrate) simulation with double-buffering.

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, DeviceRepr, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::io::{Error, ErrorKind};
use log::{info, error};
use std::path::Path;
use crate::models::simulation_params::SimParams as InternalSimParams;
use crate::utils::memory_bounds::{MemoryBounds, MemoryBoundsRegistry};

// Re-export SimParams for external modules
pub use crate::models::simulation_params::SimParams as SimParams;

// ComputeMode enum to match usage in other modules
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeMode {
    Basic = 0,
    DualGraph = 1,
    Constraints = 2,
    VisualAnalytics = 3,
}

// Constraint data matching CUDA ConstraintData
// For manual implementation - DeviceRepr and ValidAsZeroBits are traits, not derive macros
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConstraintData {
    pub constraint_type: i32,
    pub strength: f32,
    pub param1: f32,
    pub param2: f32,
    pub node_mask: i32,
}

// Manual trait implementations for CUDA compatibility
unsafe impl DeviceRepr for ConstraintData {}
unsafe impl ValidAsZeroBits for ConstraintData {}

// Structs for grouping kernel parameters, matching the new CUDA design.
// Manually implement required traits - DeviceRepr is a trait, not a derive macro
#[repr(C)]
struct GpuNodeData<'a> {
    // Input buffers (read-only for the kernel)
    pos_in_x: &'a CudaSlice<f32>,
    pos_in_y: &'a CudaSlice<f32>,
    pos_in_z: &'a CudaSlice<f32>,
    vel_in_x: &'a CudaSlice<f32>,
    vel_in_y: &'a CudaSlice<f32>,
    vel_in_z: &'a CudaSlice<f32>,

    // Output buffers (write-only for the kernel)
    pos_out_x: &'a mut CudaSlice<f32>,
    pos_out_y: &'a mut CudaSlice<f32>,
    pos_out_z: &'a mut CudaSlice<f32>,
    vel_out_x: &'a mut CudaSlice<f32>,
    vel_out_y: &'a mut CudaSlice<f32>,
    vel_out_z: &'a mut CudaSlice<f32>,

    // Optional attributes
    mass: &'a CudaSlice<f32>,
    importance: &'a CudaSlice<f32>,
    temporal: &'a CudaSlice<f32>,
    graph_id: &'a CudaSlice<i32>,
    cluster: &'a CudaSlice<i32>,
}

#[repr(C)]
struct GpuEdgeData<'a> {
    // Using CSR (Compressed Sparse Row) format
    row_offsets: &'a CudaSlice<i32>,
    col_indices: &'a CudaSlice<i32>,
    weights: &'a CudaSlice<f32>,
    graph_ids: &'a CudaSlice<i32>,
}

#[repr(C)]
struct GpuKernelParams<'a> {
    nodes: GpuNodeData<'a>,
    edges: GpuEdgeData<'a>,
    constraints: &'a CudaSlice<ConstraintData>,
    params: InternalSimParams,
    num_nodes: i32,
    num_edges: i32,
    num_constraints: i32,
}

#[derive(Debug)]
pub struct UnifiedGPUCompute {
    device: Arc<CudaDevice>,
    force_kernel: CudaFunction,
    integrate_kernel: CudaFunction,
    stress_kernel: Option<CudaFunction>,

    // Double-buffered node data (Structure of Arrays)
    pos_in_x: CudaSlice<f32>,
    pos_in_y: CudaSlice<f32>,
    pos_in_z: CudaSlice<f32>,
    vel_in_x: CudaSlice<f32>,
    vel_in_y: CudaSlice<f32>,
    vel_in_z: CudaSlice<f32>,

    pos_out_x: CudaSlice<f32>,
    pos_out_y: CudaSlice<f32>,
    pos_out_z: CudaSlice<f32>,
    vel_out_x: CudaSlice<f32>,
    vel_out_y: CudaSlice<f32>,
    vel_out_z: CudaSlice<f32>,
    
    // Force buffer
    force_x: CudaSlice<f32>,
    force_y: CudaSlice<f32>,
    force_z: CudaSlice<f32>,

    // Optional node attributes
    node_mass: Option<CudaSlice<f32>>,
    node_importance: Option<CudaSlice<f32>>,
    node_temporal: Option<CudaSlice<f32>>,
    node_graph_id: Option<CudaSlice<i32>>,
    node_cluster: Option<CudaSlice<i32>>,

    // Edge buffers (CSR format)
    edge_row_offsets: CudaSlice<i32>,
    edge_col_indices: CudaSlice<i32>,
    edge_weights: CudaSlice<f32>,
    edge_graph_ids: Option<CudaSlice<i32>>,

    // Constraints
    constraints: Option<CudaSlice<ConstraintData>>,

    // Parameters
    params: InternalSimParams,
    compute_mode: ComputeMode,
    num_nodes: usize,
    num_edges: usize,
    num_constraints: usize,

    // Memory bounds checking
    bounds_registry: MemoryBoundsRegistry,
}

impl UnifiedGPUCompute {
    pub fn new(
        device: Arc<CudaDevice>,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<Self, Error> {
        info!("Initializing Unified GPU Compute with {} nodes, {} edges", num_nodes, num_edges);

        let ptx_path = find_ptx_path()?;
        info!("Loading PTX from: {}", ptx_path);
        let ptx = Ptx::from_file(ptx_path);
        
        Self::create_with_ptx(device, ptx, num_nodes, num_edges)
    }

    fn create_with_ptx(
        device: Arc<CudaDevice>,
        ptx: Ptx,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<Self, Error> {
        let module_name = "visionflow_unified_rewrite";
        let kernel_names = &["force_pass_kernel", "integrate_pass_kernel", "stress_majorization_kernel"];
        device.load_ptx(ptx, module_name, kernel_names)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to load PTX module: {:?}", e)))?;

        let force_kernel = device.get_func(module_name, "force_pass_kernel")
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "force_pass_kernel not found"))?;
        let integrate_kernel = device.get_func(module_name, "integrate_pass_kernel")
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "integrate_pass_kernel not found"))?;
        let stress_kernel = device.get_func(module_name, "stress_majorization_kernel");

        // Initialize memory bounds registry (1GB limit)
        let mut bounds_registry = MemoryBoundsRegistry::new(1024 * 1024 * 1024);
        
        // Register all GPU buffer allocations
        for buffer_name in &[
            "pos_in_x", "pos_in_y", "pos_in_z", "vel_in_x", "vel_in_y", "vel_in_z",
            "pos_out_x", "pos_out_y", "pos_out_z", "vel_out_x", "vel_out_y", "vel_out_z",
            "force_x", "force_y", "force_z"
        ] {
            let bounds = MemoryBounds::new(
                buffer_name.to_string(), 
                num_nodes * std::mem::size_of::<f32>(), 
                std::mem::size_of::<f32>(), 
                std::mem::align_of::<f32>()
            );
            bounds_registry.register_allocation(bounds)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to register {} bounds: {:?}", buffer_name, e)))?;
        }
        
        // Register edge buffers
        let edge_bounds_row = MemoryBounds::new(
            "edge_row_offsets".to_string(),
            (num_nodes + 1) * std::mem::size_of::<i32>(),
            std::mem::size_of::<i32>(),
            std::mem::align_of::<i32>()
        );
        bounds_registry.register_allocation(edge_bounds_row)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to register edge_row_offsets bounds: {:?}", e)))?;
            
        for buffer_name in &["edge_col_indices", "edge_weights"] {
            let element_size = if buffer_name.contains("indices") { std::mem::size_of::<i32>() } else { std::mem::size_of::<f32>() };
            let bounds = MemoryBounds::new(
                buffer_name.to_string(),
                num_edges * 2 * element_size,
                element_size,
                element_size  // Alignment same as element size for primitives
            );
            bounds_registry.register_allocation(bounds)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to register {} bounds: {:?}", buffer_name, e)))?;
        }

        Ok(Self {
            device: device.clone(),
            force_kernel,
            integrate_kernel,
            stress_kernel,
            pos_in_x: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            pos_in_y: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            pos_in_z: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            vel_in_x: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            vel_in_y: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            vel_in_z: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            pos_out_x: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            pos_out_y: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            pos_out_z: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            vel_out_x: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            vel_out_y: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            vel_out_z: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            force_x: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            force_y: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            force_z: device.alloc_zeros(num_nodes).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            node_mass: None,
            node_importance: None,
            node_temporal: None,
            node_graph_id: None,
            node_cluster: None,
            edge_row_offsets: device.alloc_zeros(num_nodes + 1).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            edge_col_indices: device.alloc_zeros(num_edges * 2).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            edge_weights: device.alloc_zeros(num_edges * 2).map_err(|e| Error::new(ErrorKind::Other, format!("GPU allocation failed: {:?}", e)))?,
            edge_graph_ids: None,
            constraints: None,
            params: InternalSimParams::new(),
            compute_mode: ComputeMode::Basic,
            num_nodes,
            num_edges,
            num_constraints: 0,
            bounds_registry,
        })
    }

    pub fn set_params(&mut self, params: InternalSimParams) {
        log::info!("UnifiedGPUCompute::set_params called with: {:?}", params);
        self.params = params;
    }

    pub fn set_mode(&mut self, mode: ComputeMode) {
        log::info!("UnifiedGPUCompute::set_mode called with: {:?}", mode);
        self.compute_mode = mode;
    }

    pub fn get_mode(&self) -> ComputeMode {
        self.compute_mode
    }

    pub fn upload_positions(&mut self, positions: &[(f32, f32, f32)]) -> Result<(), Error> {
        // Memory bounds validation using registry
        self.bounds_registry.check_range_access(
            "pos_in_x", 0, positions.len().min(self.num_nodes), true
        ).map_err(|e| Error::new(ErrorKind::InvalidInput, format!("Position upload bounds check failed: {:?}", e)))?;
        
        // Additional bounds checking
        if positions.len() > self.num_nodes {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Position count {} exceeds allocated buffer size {}", positions.len(), self.num_nodes)
            ));
        }

        // If no positions provided, nothing to upload
        if positions.is_empty() {
            return Ok(());
        }

        // Separate positions into SoA (Structure of Arrays) format
        let mut pos_x: Vec<f32> = Vec::with_capacity(self.num_nodes);
        let mut pos_y: Vec<f32> = Vec::with_capacity(self.num_nodes);
        let mut pos_z: Vec<f32> = Vec::with_capacity(self.num_nodes);

        for (i, &(x, y, z)) in positions.iter().enumerate() {
            if i >= self.num_nodes {
                break; // Additional safety check
            }
            pos_x.push(x);
            pos_y.push(y);
            pos_z.push(z);
        }

        // Fill remaining with zeros if needed
        while pos_x.len() < self.num_nodes {
            pos_x.push(0.0);
            pos_y.push(0.0);
            pos_z.push(0.0);
        }

        // Upload to GPU with error handling
        self.device.htod_sync_copy_into(&pos_x, &mut self.pos_in_x)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload X positions to GPU: {:?}", e)))?;
        
        self.device.htod_sync_copy_into(&pos_y, &mut self.pos_in_y)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload Y positions to GPU: {:?}", e)))?;
        
        self.device.htod_sync_copy_into(&pos_z, &mut self.pos_in_z)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload Z positions to GPU: {:?}", e)))?;

        info!("Successfully uploaded {} positions to GPU", positions.len());
        Ok(())
    }

    pub fn upload_edges_csr(&mut self, offsets: &[i32], indices: &[i32], weights: &[f32]) -> Result<(), Error> {
        // Bounds checking for CSR format
        if offsets.len() != self.num_nodes + 1 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("CSR offsets length {} != num_nodes + 1 ({})", offsets.len(), self.num_nodes + 1)
            ));
        }

        if indices.len() != weights.len() {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("CSR indices length {} != weights length {}", indices.len(), weights.len())
            ));
        }

        let max_edges = self.num_edges * 2; // Buffer size calculation
        if indices.len() > max_edges {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Edge count {} exceeds allocated buffer size {}", indices.len(), max_edges)
            ));
        }

        // Validate indices are within bounds
        for &idx in indices.iter() {
            if idx < 0 || idx as usize >= self.num_nodes {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Edge index {} is out of bounds [0, {})", idx, self.num_nodes)
                ));
            }
        }

        // Validate offsets are monotonically increasing
        for i in 1..offsets.len() {
            if offsets[i] < offsets[i - 1] {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("CSR offsets not monotonically increasing at index {}: {} < {}", i, offsets[i], offsets[i - 1])
                ));
            }
        }

        // Upload CSR data to GPU
        self.device.htod_sync_copy_into(offsets, &mut self.edge_row_offsets)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload CSR offsets to GPU: {:?}", e)))?;

        // Create padded vectors if needed
        let mut padded_indices = indices.to_vec();
        let mut padded_weights = weights.to_vec();
        
        while padded_indices.len() < max_edges {
            padded_indices.push(0);
            padded_weights.push(0.0);
        }

        self.device.htod_sync_copy_into(&padded_indices, &mut self.edge_col_indices)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload CSR indices to GPU: {:?}", e)))?;

        self.device.htod_sync_copy_into(&padded_weights, &mut self.edge_weights)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload CSR weights to GPU: {:?}", e)))?;

        info!("Successfully uploaded CSR edge data: {} nodes, {} edges", self.num_nodes, indices.len());
        Ok(())
    }
    
    pub fn upload_edges(&mut self, edges: &[(i32, i32, f32)]) -> Result<(), Error> {
        // Convert edge list to CSR format
        if edges.is_empty() {
            // Initialize empty CSR structure
            let offsets = vec![0; self.num_nodes + 1];
            let indices = vec![];
            let weights = vec![];
            return self.upload_edges_csr(&offsets, &indices, &weights);
        }

        // Build CSR representation
        let mut adjacency_lists: Vec<Vec<(i32, f32)>> = vec![Vec::new(); self.num_nodes];
        
        for &(src, dst, weight) in edges.iter() {
            // Bounds checking for edge vertices
            if src < 0 || src as usize >= self.num_nodes {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Source vertex {} is out of bounds [0, {})", src, self.num_nodes)
                ));
            }
            if dst < 0 || dst as usize >= self.num_nodes {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Destination vertex {} is out of bounds [0, {})", dst, self.num_nodes)
                ));
            }

            adjacency_lists[src as usize].push((dst, weight));
            // For undirected graphs, add reverse edge
            if src != dst {
                adjacency_lists[dst as usize].push((src, weight));
            }
        }

        // Convert to CSR format
        let mut offsets = Vec::with_capacity(self.num_nodes + 1);
        let mut indices = Vec::new();
        let mut weights = Vec::new();

        let mut offset = 0;
        offsets.push(offset);

        for adj_list in adjacency_lists {
            for (neighbor, weight) in adj_list {
                indices.push(neighbor);
                weights.push(weight);
                offset += 1;
            }
            offsets.push(offset);
        }

        self.upload_edges_csr(&offsets, &indices, &weights)
    }
    
    pub fn set_constraints(&mut self, constraints: Vec<ConstraintData>) -> Result<(), Error> {
        if constraints.is_empty() {
            self.constraints = None;
            self.num_constraints = 0;
            return Ok(());
        }

        // Validate constraint data
        for (i, constraint) in constraints.iter().enumerate() {
            if constraint.strength < 0.0 || constraint.strength > 1.0 {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Constraint {} has invalid strength {}: must be in [0, 1]", i, constraint.strength)
                ));
            }
        }

        // Allocate GPU buffer for constraints
        let constraint_buffer = self.device.htod_copy(constraints.clone())
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate GPU memory for constraints: {:?}", e)))?;

        self.constraints = Some(constraint_buffer);
        self.num_constraints = constraints.len();

        info!("Successfully uploaded {} constraints to GPU", constraints.len());
        Ok(())
    }
    
    pub fn resize_buffers(&mut self, new_node_count: usize, new_edge_count: usize) -> Result<(), Error> {
        // Validate new sizes
        if new_node_count == 0 {
            return Err(Error::new(ErrorKind::InvalidInput, "Node count cannot be zero"));
        }
        if new_node_count > 1_000_000 {
            return Err(Error::new(ErrorKind::InvalidInput, "Node count exceeds maximum limit (1M)"));
        }
        if new_edge_count > 10_000_000 {
            return Err(Error::new(ErrorKind::InvalidInput, "Edge count exceeds maximum limit (10M)"));
        }

        info!("Resizing GPU buffers from {}x{} to {}x{} nodes/edges", 
              self.num_nodes, self.num_edges, new_node_count, new_edge_count);

        // Only resize if the new size is different
        if new_node_count != self.num_nodes {
            // Reallocate node buffers
            self.pos_in_x = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize pos_in_x buffer: {:?}", e)))?;
            self.pos_in_y = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize pos_in_y buffer: {:?}", e)))?;
            self.pos_in_z = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize pos_in_z buffer: {:?}", e)))?;
            self.vel_in_x = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize vel_in_x buffer: {:?}", e)))?;
            self.vel_in_y = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize vel_in_y buffer: {:?}", e)))?;
            self.vel_in_z = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize vel_in_z buffer: {:?}", e)))?;
            self.pos_out_x = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize pos_out_x buffer: {:?}", e)))?;
            self.pos_out_y = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize pos_out_y buffer: {:?}", e)))?;
            self.pos_out_z = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize pos_out_z buffer: {:?}", e)))?;
            self.vel_out_x = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize vel_out_x buffer: {:?}", e)))?;
            self.vel_out_y = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize vel_out_y buffer: {:?}", e)))?;
            self.vel_out_z = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize vel_out_z buffer: {:?}", e)))?;
            self.force_x = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize force_x buffer: {:?}", e)))?;
            self.force_y = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize force_y buffer: {:?}", e)))?;
            self.force_z = self.device.alloc_zeros(new_node_count)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize force_z buffer: {:?}", e)))?;

            // Resize CSR row offsets (num_nodes + 1)
            self.edge_row_offsets = self.device.alloc_zeros(new_node_count + 1)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize edge_row_offsets buffer: {:?}", e)))?;
        }

        if new_edge_count != self.num_edges {
            // Reallocate edge buffers (multiply by 2 for undirected graphs)
            let edge_buffer_size = new_edge_count * 2;
            self.edge_col_indices = self.device.alloc_zeros(edge_buffer_size)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize edge_col_indices buffer: {:?}", e)))?;
            self.edge_weights = self.device.alloc_zeros(edge_buffer_size)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize edge_weights buffer: {:?}", e)))?;
        }

        // Update size tracking
        self.num_nodes = new_node_count;
        self.num_edges = new_edge_count;

        info!("Successfully resized GPU buffers to {} nodes, {} edges", new_node_count, new_edge_count);
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

    pub fn execute(&mut self) -> Result<Vec<(f32, f32, f32)>, Error> {
        self.params.iteration += 1;

        let block_size = 256;
        let grid_size = (self.num_nodes as u32 + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // TODO: Build GpuKernelParams here

        // Launch force kernel
        // unsafe { self.force_kernel.clone().launch(config, (kernel_params,))? };

        // Launch integrate kernel
        // unsafe { self.integrate_kernel.clone().launch(config, (kernel_params,))? };
        
        self.device.synchronize().map_err(|e| Error::new(ErrorKind::Other, format!("GPU synchronization failed: {:?}", e)))?;
        
        self.swap_buffers();

        // Return updated positions
        self.get_positions()
    }

    pub fn get_positions(&self) -> Result<Vec<(f32, f32, f32)>, Error> {
        if self.num_nodes == 0 {
            return Ok(Vec::new());
        }

        // Allocate host buffers with bounds checking
        let mut pos_x = vec![0.0f32; self.num_nodes];
        let mut pos_y = vec![0.0f32; self.num_nodes];
        let mut pos_z = vec![0.0f32; self.num_nodes];

        // Validate buffer sizes match before copying
        if pos_x.len() != self.num_nodes {
            return Err(Error::new(ErrorKind::Other, 
                format!("Host buffer size mismatch: {} != {}", pos_x.len(), self.num_nodes)));
        }

        // Copy from GPU with comprehensive error handling
        self.device.dtoh_sync_copy_into(&self.pos_in_x, &mut pos_x)
            .map_err(|e| Error::new(ErrorKind::Other, 
                format!("Failed to copy X positions from GPU (buffer size: {}): {:?}", self.num_nodes, e)))?;
        
        self.device.dtoh_sync_copy_into(&self.pos_in_y, &mut pos_y)
            .map_err(|e| Error::new(ErrorKind::Other, 
                format!("Failed to copy Y positions from GPU (buffer size: {}): {:?}", self.num_nodes, e)))?;
        
        self.device.dtoh_sync_copy_into(&self.pos_in_z, &mut pos_z)
            .map_err(|e| Error::new(ErrorKind::Other, 
                format!("Failed to copy Z positions from GPU (buffer size: {}): {:?}", self.num_nodes, e)))?;

        // Validate data integrity
        for i in 0..self.num_nodes {
            if !pos_x[i].is_finite() || !pos_y[i].is_finite() || !pos_z[i].is_finite() {
                return Err(Error::new(ErrorKind::InvalidData, 
                    format!("Invalid position data at index {}: ({}, {}, {})", i, pos_x[i], pos_y[i], pos_z[i])));
            }
        }

        let positions: Vec<(f32, f32, f32)> = (0..self.num_nodes)
            .map(|i| (pos_x[i], pos_y[i], pos_z[i]))
            .collect();

        Ok(positions)
    }

    pub fn test_gpu() -> Result<(), Error> {
        info!("Testing GPU functionality...");
        
        // Test CUDA device creation with detailed error reporting
        let device = match CudaDevice::new(0) {
            Ok(dev) => {
                info!("CUDA device created successfully");
                dev
            },
            Err(e) => {
                error!("Failed to create CUDA device: {:?}", e);
                return Err(Error::new(ErrorKind::Other, 
                    format!("CUDA device initialization failed. This may indicate:\n\
                    1. No CUDA-capable GPU found\n\
                    2. CUDA driver not installed or outdated\n\
                    3. GPU memory exhausted\n\
                    Original error: {:?}", e)));
            }
        };
        
        // Test memory allocation with progressive size checks
        let test_sizes = [100, 1_000, 10_000, 100_000];
        let mut max_allocation = 0;
        
        for &size in &test_sizes {
            match device.alloc_zeros::<f32>(size) {
                Ok(buffer) => {
                    info!("Successfully allocated {} floats ({} bytes)", size, size * 4);
                    max_allocation = size;
                    drop(buffer); // Free immediately
                },
                Err(e) => {
                    if max_allocation == 0 {
                        error!("Failed to allocate even {} floats: {:?}", size, e);
                        return Err(Error::new(ErrorKind::Other, 
                            format!("Critical GPU memory allocation failure. Cannot allocate {} bytes.\n\
                            This may indicate GPU memory exhaustion or driver issues.\n\
                            Original error: {:?}", size * 4, e)));
                    } else {
                        info!("Memory allocation limit reached at {} elements", max_allocation);
                        break;
                    }
                }
            }
        }
        
        // Test GPU synchronization
        device.synchronize().map_err(|e| {
            error!("GPU synchronization failed: {:?}", e);
            Error::new(ErrorKind::Other, 
                format!("GPU synchronization test failed. This may indicate:\n\
                1. GPU hang or timeout\n\
                2. Driver instability\n\
                3. Hardware issues\n\
                Original error: {:?}", e))
        })?;
        
        info!("GPU test completed successfully - max allocation: {} elements", max_allocation);
        Ok(())
    }
    
    /// Get detailed GPU memory information
    pub fn get_gpu_memory_info(&self) -> Result<GPUMemoryInfo, Error> {
        let usage_report = self.bounds_registry.get_usage_report();
        
        Ok(GPUMemoryInfo {
            total_allocated: usage_report.total_allocated,
            allocation_count: usage_report.allocation_count,
            largest_allocation: usage_report.largest_allocation,
            usage_percentage: usage_report.usage_percentage(),
            buffer_breakdown: usage_report.buffer_types,
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
            num_constraints: self.num_constraints,
        })
    }
}

/// GPU Memory Information
#[derive(Debug, Clone)]
pub struct GPUMemoryInfo {
    pub total_allocated: usize,
    pub allocation_count: usize,
    pub largest_allocation: usize,
    pub usage_percentage: f64,
    pub buffer_breakdown: std::collections::HashMap<String, usize>,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub num_constraints: usize,
}

fn find_ptx_path() -> Result<&'static str, Error> {
    let ptx_paths = [
        "src/utils/ptx/visionflow_unified_rewrite.ptx",
        "./src/utils/ptx/visionflow_unified_rewrite.ptx",
        "/app/src/utils/ptx/visionflow_unified_rewrite.ptx",
        "src/utils/ptx/visionflow_unified.ptx", // Fallback
        "./src/utils/ptx/visionflow_unified.ptx", // Fallback
        "/app/src/utils/ptx/visionflow_unified.ptx", // Fallback
    ];

    for path in &ptx_paths {
        if Path::new(path).exists() {
            info!("Found PTX file at: {}", path);
            return Ok(path);
        }
    }
    
    error!("PTX file not found in any of the expected locations");
    Err(Error::new(
        ErrorKind::NotFound,
        format!("Unified PTX not found. Tried paths: {:?}\n\
                Please ensure the CUDA kernel has been compiled correctly.", ptx_paths)
    ))
}

// From impl removed - use .map_err() at call sites instead