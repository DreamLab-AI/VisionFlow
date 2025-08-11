// Unified GPU Compute Module - Clean, single-kernel implementation
// Replaces the complex multi-kernel system with fallbacks

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync, DeviceRepr, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::io::{Error, ErrorKind};
use log::{info, warn};
use std::path::Path;

// Compute modes matching the CUDA kernel
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeMode {
    Basic = 0,         // Basic force-directed layout
    DualGraph = 1,     // Dual graph (knowledge + agent)
    Constraints = 2,   // With constraint satisfaction
    VisualAnalytics = 3, // Advanced visual analytics
}

// Simulation parameters matching CUDA SimParams
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SimParams {
    // Force parameters
    pub spring_k: f32,
    pub repel_k: f32,
    pub damping: f32,
    pub dt: f32,
    pub max_velocity: f32,
    pub max_force: f32,
    
    // Stress majorization
    pub stress_weight: f32,
    pub stress_alpha: f32,
    
    // Constraints
    pub separation_radius: f32,
    pub boundary_limit: f32,
    pub alignment_strength: f32,
    pub cluster_strength: f32,
    
    // System
    pub viewport_bounds: f32,
    pub temperature: f32,
    pub iteration: i32,
    pub compute_mode: i32,
}

// Implement DeviceRepr traits for GPU transfer
unsafe impl DeviceRepr for SimParams {}
unsafe impl ValidAsZeroBits for SimParams {}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            spring_k: 0.005,       // Very gentle springs
            repel_k: 50.0,         // Moderate repulsion
            damping: 0.9,          // Higher damping for stability
            dt: 0.01,              // Smaller timestep for stability
            max_velocity: 1.0,     // Lower max velocity
            max_force: 2.0,        // Much lower max force
            stress_weight: 0.5,
            stress_alpha: 0.1,
            separation_radius: 2.0, // Reasonable separation
            boundary_limit: 100.0,
            alignment_strength: 0.1,
            cluster_strength: 0.2,
            viewport_bounds: 200.0, // Smaller viewport
            temperature: 0.5,      // Lower initial temperature
            iteration: 0,
            compute_mode: 0,
        }
    }
}

// Constraint data matching CUDA ConstraintData
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConstraintData {
    pub constraint_type: i32,
    pub strength: f32,
    pub param1: f32,
    pub param2: f32,
    pub node_mask: i32,
}

// Implement DeviceRepr traits for GPU transfer
unsafe impl DeviceRepr for ConstraintData {}
unsafe impl ValidAsZeroBits for ConstraintData {}

pub struct UnifiedGPUCompute {
    device: Arc<CudaDevice>,
    compute_kernel: CudaFunction,
    stress_kernel: Option<CudaFunction>,
    
    // Node buffers (Structure of Arrays)
    pos_x: CudaSlice<f32>,
    pos_y: CudaSlice<f32>,
    pos_z: CudaSlice<f32>,
    vel_x: CudaSlice<f32>,
    vel_y: CudaSlice<f32>,
    vel_z: CudaSlice<f32>,
    
    // Optional node attributes
    node_mass: Option<CudaSlice<f32>>,
    node_importance: Option<CudaSlice<f32>>,
    node_temporal: Option<CudaSlice<f32>>,
    node_graph_id: Option<CudaSlice<i32>>,
    node_cluster: Option<CudaSlice<i32>>,
    
    // Edge buffers (CSR format)
    edge_src: CudaSlice<i32>,
    edge_dst: CudaSlice<i32>,
    edge_weight: CudaSlice<f32>,
    edge_graph_id: Option<CudaSlice<i32>>,
    
    // Constraints
    constraints: Option<CudaSlice<ConstraintData>>,
    
    // Parameters
    params: SimParams,
    num_nodes: usize,
    num_edges: usize,
    num_constraints: usize,
    compute_mode: ComputeMode,
}

impl UnifiedGPUCompute {
    pub fn new(
        device: Arc<CudaDevice>,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<Self, Error> {
        info!("Initializing Unified GPU Compute with {} nodes, {} edges", num_nodes, num_edges);
        
        // Load the unified PTX
        let ptx_path = "/app/src/utils/ptx/visionflow_unified.ptx";
        if !Path::new(ptx_path).exists() {
            // Try local path for development
            let local_path = "src/utils/ptx/visionflow_unified.ptx";
            if !Path::new(local_path).exists() {
                return Err(Error::new(
                    ErrorKind::NotFound,
                    format!("Unified PTX not found at {} or {}", ptx_path, local_path)
                ));
            }
            info!("Using local PTX path: {}", local_path);
            let ptx = Ptx::from_file(local_path);
            Self::create_with_ptx(device, ptx, num_nodes, num_edges)
        } else {
            info!("Loading PTX from: {}", ptx_path);
            let ptx = Ptx::from_file(ptx_path);
            Self::create_with_ptx(device, ptx, num_nodes, num_edges)
        }
    }
    
    fn create_with_ptx(
        device: Arc<CudaDevice>,
        ptx: Ptx,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<Self, Error> {
        // Load compute kernel
        let compute_kernel = device
            .load_ptx(ptx.clone(), "visionflow_compute", &["visionflow_compute_kernel"])
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to load compute kernel: {}", e)))?;
        
        // Try to load stress kernel (optional)
        let stress_kernel = device
            .load_ptx(ptx, "stress_majorization", &["stress_majorization_kernel"])
            .ok();
        
        if stress_kernel.is_some() {
            info!("Stress majorization kernel loaded successfully");
        } else {
            warn!("Stress majorization kernel not available");
        }
        
        // Allocate GPU buffers
        let pos_x = device.alloc_zeros(num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let pos_y = device.alloc_zeros(num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let pos_z = device.alloc_zeros(num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let vel_x = device.alloc_zeros(num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let vel_y = device.alloc_zeros(num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let vel_z = device.alloc_zeros(num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        let edge_src = device.alloc_zeros(num_edges)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let edge_dst = device.alloc_zeros(num_edges)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let edge_weight = device.alloc_zeros(num_edges)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        Ok(Self {
            device,
            compute_kernel,
            stress_kernel,
            pos_x,
            pos_y,
            pos_z,
            vel_x,
            vel_y,
            vel_z,
            node_mass: None,
            node_importance: None,
            node_temporal: None,
            node_graph_id: None,
            node_cluster: None,
            edge_src,
            edge_dst,
            edge_weight,
            edge_graph_id: None,
            constraints: None,
            params: SimParams::default(),
            num_nodes,
            num_edges,
            num_constraints: 0,
            compute_mode: ComputeMode::Basic,
        })
    }
    
    pub fn set_mode(&mut self, mode: ComputeMode) {
        self.compute_mode = mode;
        self.params.compute_mode = mode as i32;
        info!("Set compute mode to: {:?}", mode);
    }
    
    pub fn set_params(&mut self, params: SimParams) {
        self.params = params;
        self.params.compute_mode = self.compute_mode as i32;
    }
    
    pub fn upload_positions(&mut self, positions: &[(f32, f32, f32)]) -> Result<(), Error> {
        if positions.len() != self.num_nodes {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Position count mismatch: {} vs {}", positions.len(), self.num_nodes)
            ));
        }
        
        let mut pos_x = vec![0.0f32; self.num_nodes];
        let mut pos_y = vec![0.0f32; self.num_nodes];
        let mut pos_z = vec![0.0f32; self.num_nodes];
        
        // Check if positions are all zero (uninitialized)
        let needs_init = positions.iter().all(|&(x, y, z)| x == 0.0 && y == 0.0 && z == 0.0);
        
        if needs_init {
            info!("Initializing random positions to prevent collapse");
            // Generate initial positions using golden angle spiral
            let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
            let spread_radius = 30.0; // Increased initial spread
            
            for i in 0..self.num_nodes {
                let theta = i as f32 * golden_angle;
                let y = 1.0 - (i as f32 / self.num_nodes as f32) * 2.0;
                let radius = (1.0 - y * y).sqrt();
                
                pos_x[i] = theta.cos() * radius * spread_radius;
                pos_y[i] = y * spread_radius;
                pos_z[i] = theta.sin() * radius * spread_radius;
                
                // Add small random perturbation
                use rand::Rng;
                let mut rng = rand::thread_rng();
                pos_x[i] += rng.gen_range(-0.5..0.5);
                pos_y[i] += rng.gen_range(-0.5..0.5);
                pos_z[i] += rng.gen_range(-0.5..0.5);
            }
        } else {
            for (i, &(x, y, z)) in positions.iter().enumerate() {
                pos_x[i] = x;
                pos_y[i] = y;
                pos_z[i] = z;
            }
        }
        
        self.device.htod_sync_copy_into(&pos_x, &mut self.pos_x)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        self.device.htod_sync_copy_into(&pos_y, &mut self.pos_y)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        self.device.htod_sync_copy_into(&pos_z, &mut self.pos_z)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        Ok(())
    }
    
    pub fn upload_edges(&mut self, edges: &[(i32, i32, f32)]) -> Result<(), Error> {
        if edges.len() != self.num_edges {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!("Edge count mismatch: {} vs {}", edges.len(), self.num_edges)
            ));
        }
        
        let mut edge_src = vec![0i32; self.num_edges];
        let mut edge_dst = vec![0i32; self.num_edges];
        let mut edge_weight = vec![0.0f32; self.num_edges];
        
        for (i, &(src, dst, weight)) in edges.iter().enumerate() {
            edge_src[i] = src;
            edge_dst[i] = dst;
            edge_weight[i] = weight;
        }
        
        self.device.htod_sync_copy_into(&edge_src, &mut self.edge_src)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        self.device.htod_sync_copy_into(&edge_dst, &mut self.edge_dst)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        self.device.htod_sync_copy_into(&edge_weight, &mut self.edge_weight)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        Ok(())
    }
    
    pub fn set_constraints(&mut self, constraints: Vec<ConstraintData>) -> Result<(), Error> {
        self.num_constraints = constraints.len();
        
        if self.num_constraints > 0 {
            let gpu_constraints = self.device.alloc_zeros(self.num_constraints)
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            
            self.device.htod_sync_copy_into(&constraints, &gpu_constraints)
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            
            self.constraints = Some(gpu_constraints);
        } else {
            self.constraints = None;
        }
        
        info!("Set {} constraints", self.num_constraints);
        Ok(())
    }
    
    pub fn enable_dual_graph(&mut self, node_graph_ids: Vec<i32>) -> Result<(), Error> {
        if node_graph_ids.len() != self.num_nodes {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "Graph ID count mismatch"
            ));
        }
        
        let gpu_graph_ids = self.device.alloc_zeros(self.num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        self.device.htod_sync_copy_into(&node_graph_ids, &gpu_graph_ids)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        self.node_graph_id = Some(gpu_graph_ids);
        self.set_mode(ComputeMode::DualGraph);
        
        Ok(())
    }
    
    pub fn execute(&mut self) -> Result<Vec<(f32, f32, f32)>, Error> {
        // Update iteration counter
        self.params.iteration += 1;
        
        // Calculate launch configuration
        let block_size = 256;
        let grid_size = (self.num_nodes + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Launch kernel with appropriate parameters based on mode
        unsafe {
            self.compute_kernel.launch(
                config,
                (
                    &self.pos_x, &self.pos_y, &self.pos_z,
                    &self.vel_x, &self.vel_y, &self.vel_z,
                    self.node_mass.as_ref().unwrap_or(&self.pos_x), // Dummy if not used
                    self.node_importance.as_ref().unwrap_or(&self.pos_x),
                    self.node_temporal.as_ref().unwrap_or(&self.pos_x),
                    self.node_graph_id.as_ref().unwrap_or(&self.edge_src), // Dummy
                    self.node_cluster.as_ref().unwrap_or(&self.edge_src),
                    &self.edge_src,
                    &self.edge_dst,
                    &self.edge_weight,
                    self.edge_graph_id.as_ref().unwrap_or(&self.edge_src),
                    self.constraints.as_ref().unwrap_or(&self.edge_src), // Dummy
                    self.params,
                    self.num_nodes as i32,
                    self.num_edges as i32,
                    self.num_constraints as i32,
                ),
            ).map_err(|e| Error::new(ErrorKind::Other, format!("Kernel launch failed: {}", e)))?;
        }
        
        // Synchronize
        self.device.synchronize()
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        // Download positions
        let mut pos_x = vec![0.0f32; self.num_nodes];
        let mut pos_y = vec![0.0f32; self.num_nodes];
        let mut pos_z = vec![0.0f32; self.num_nodes];
        
        self.device.dtoh_sync_copy_into(&self.pos_x, &mut pos_x)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        self.device.dtoh_sync_copy_into(&self.pos_y, &mut pos_y)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        self.device.dtoh_sync_copy_into(&self.pos_z, &mut pos_z)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        // Combine into tuples
        let positions: Vec<(f32, f32, f32)> = (0..self.num_nodes)
            .map(|i| (pos_x[i], pos_y[i], pos_z[i]))
            .collect();
        
        Ok(positions)
    }
    
    pub fn execute_stress_majorization(&mut self, ideal_distances: &[f32], weight_matrix: &[f32]) -> Result<(), Error> {
        if let Some(ref stress_kernel) = self.stress_kernel {
            let n = self.num_nodes;
            if ideal_distances.len() != n * n || weight_matrix.len() != n * n {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "Distance/weight matrix size mismatch"
                ));
            }
            
            // Upload matrices
            let mut gpu_distances = self.device.alloc_zeros::<f32>(ideal_distances.len())
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            self.device.htod_sync_copy_into(ideal_distances, &mut gpu_distances)
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                
            let mut gpu_weights = self.device.alloc_zeros::<f32>(weight_matrix.len())
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            self.device.htod_sync_copy_into(weight_matrix, &mut gpu_weights)
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            
            // Launch stress kernel
            let block_size = 256;
            let grid_size = (n + block_size - 1) / block_size;
            
            let config = LaunchConfig {
                grid_dim: (grid_size as u32, 1, 1),
                block_dim: (block_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            
            unsafe {
                stress_kernel.launch(
                    config,
                    (
                        &self.pos_x, &self.pos_y, &self.pos_z,
                        &gpu_distances,
                        &gpu_weights,
                        self.params,
                        n as i32,
                    ),
                ).map_err(|e| Error::new(ErrorKind::Other, format!("Stress kernel failed: {}", e)))?;
            }
            
            self.device.synchronize()
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            
            Ok(())
        } else {
            warn!("Stress majorization kernel not available");
            Ok(())
        }
    }
}