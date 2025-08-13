// Unified GPU Compute Module - Clean, single-kernel implementation
// Replaces the complex multi-kernel system with fallbacks

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync, DeviceRepr, ValidAsZeroBits, DevicePtrMut};
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

// Conversion from SimulationParams to SimParams
impl From<&crate::models::simulation_params::SimulationParams> for SimParams {
    fn from(params: &crate::models::simulation_params::SimulationParams) -> Self {
        Self {
            spring_k: params.spring_strength,
            repel_k: params.repulsion,
            damping: params.damping,
            dt: params.time_step,
            max_velocity: params.max_velocity,
            max_force: params.repulsion * 0.2, // Calculated as 20% of repulsion
            stress_weight: 0.5,  // Default stress values
            stress_alpha: 0.1,
            separation_radius: params.collision_radius,
            boundary_limit: params.viewport_bounds,
            alignment_strength: params.attraction_strength,
            cluster_strength: 0.2,  // Default cluster strength
            viewport_bounds: params.viewport_bounds,
            temperature: params.temperature,
            iteration: 0,
            compute_mode: 0,  // Will be set based on ComputeMode
        }
    }
}

impl Default for SimParams {
    fn default() -> Self {
        // Default values should match settings.yaml physics settings
        // These are fallback values only - actual values come from settings.yaml
        Self {
            spring_k: 0.005,       // From settings.yaml spring_strength
            repel_k: 50.0,         // From settings.yaml repulsion_strength
            damping: 0.9,          // From settings.yaml damping
            dt: 0.01,              // From settings.yaml time_step
            max_velocity: 1.0,     // From settings.yaml max_velocity
            max_force: 10.0,       // Calculated based on forces
            stress_weight: 0.5,
            stress_alpha: 0.1,
            separation_radius: 0.15, // From settings.yaml collision_radius
            boundary_limit: 200.0,   // From settings.yaml bounds_size
            alignment_strength: 0.001, // From settings.yaml attraction_strength
            cluster_strength: 0.2,
            viewport_bounds: 200.0,  // From settings.yaml bounds_size
            temperature: 0.5,        // From settings.yaml temperature
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

// New structs to group kernel parameters
#[repr(C)]
struct GpuNodeData {
    pos_x: *mut f32, pos_y: *mut f32, pos_z: *mut f32,
    vel_x: *mut f32, vel_y: *mut f32, vel_z: *mut f32,
    mass: *mut f32,
    importance: *mut f32,
    temporal: *mut f32,
    graph_id: *mut i32,
    cluster: *mut i32,
}
unsafe impl DeviceRepr for GpuNodeData {}

#[repr(C)]
struct GpuEdgeData {
    src: *mut i32,
    dst: *mut i32,
    weight: *mut f32,
    graph_id: *mut i32,
}
unsafe impl DeviceRepr for GpuEdgeData {}

#[repr(C)]
struct GpuKernelParams {
    nodes: GpuNodeData,
    edges: GpuEdgeData,
    constraints: *mut ConstraintData,
    params: SimParams,
    num_nodes: i32,
    num_edges: i32,
    num_constraints: i32,
}
unsafe impl DeviceRepr for GpuKernelParams {}

use std::fmt;

impl fmt::Debug for UnifiedGPUCompute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnifiedGPUCompute")
            .field("num_nodes", &self.num_nodes)
            .field("num_edges", &self.num_edges)
            .field("num_constraints", &self.num_constraints)
            .field("compute_mode", &self.compute_mode)
            .field("params", &self.params)
            .finish_non_exhaustive()
    }
}

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

    // Dummy buffers for unused optional inputs
    dummy_f32: CudaSlice<f32>,
    dummy_i32: CudaSlice<i32>,
    dummy_constraints: CudaSlice<ConstraintData>,

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

        // Load the unified PTX - try multiple paths
        let ptx_paths = [
            "/src/utils/ptx/visionflow_unified.ptx",  // Workspace path
            "/app/src/utils/ptx/visionflow_unified.ptx",            // Container path
            "src/utils/ptx/visionflow_unified.ptx",                 // Relative path
            "./src/utils/ptx/visionflow_unified.ptx",               // Relative with ./
        ];

        let mut ptx_path_found = None;
        for path in &ptx_paths {
            if Path::new(path).exists() {
                ptx_path_found = Some(*path);
                break;
            }
        }

        match ptx_path_found {
            Some(path) => {
                info!("Loading PTX from: {}", path);
                let ptx = Ptx::from_file(path);
                Self::create_with_ptx(device, ptx, num_nodes, num_edges)
            }
            None => {
                Err(Error::new(
                    ErrorKind::NotFound,
                    format!("Unified PTX not found. Tried paths: {:?}", ptx_paths)
                ))
            }
        }
    }

    fn create_with_ptx(
        device: Arc<CudaDevice>,
        ptx: Ptx,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<Self, Error> {
        // Load kernels from PTX module, specifying the functions to load
        device.load_ptx(ptx, "visionflow_unified", &["visionflow_compute_kernel", "stress_majorization_kernel"])
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to load PTX module: {}", e)))?;

        let compute_kernel = device.get_func("visionflow_unified", "visionflow_compute_kernel")
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "visionflow_compute_kernel not found"))?;

        let stress_kernel = device.get_func("visionflow_unified", "stress_majorization_kernel");

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

        // Allocate dummy buffers
        let dummy_f32 = device.alloc_zeros(0)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let dummy_i32 = device.alloc_zeros(0)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let dummy_constraints = device.alloc_zeros(0)
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
            dummy_f32,
            dummy_i32,
            dummy_constraints,
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
            let mut gpu_constraints = self.device.alloc_zeros(self.num_constraints)
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;

            self.device.htod_sync_copy_into(&constraints, &mut gpu_constraints)
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

        let mut gpu_graph_ids = self.device.alloc_zeros(self.num_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;

        self.device.htod_sync_copy_into(&node_graph_ids, &mut gpu_graph_ids)
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

        // Prepare kernel parameters
        let kernel_params = GpuKernelParams {
            nodes: GpuNodeData {
                pos_x: *self.pos_x.device_ptr_mut() as *mut f32,
                pos_y: *self.pos_y.device_ptr_mut() as *mut f32,
                pos_z: *self.pos_z.device_ptr_mut() as *mut f32,
                vel_x: *self.vel_x.device_ptr_mut() as *mut f32,
                vel_y: *self.vel_y.device_ptr_mut() as *mut f32,
                vel_z: *self.vel_z.device_ptr_mut() as *mut f32,
                mass: *self.node_mass.as_mut().map_or(self.dummy_f32.device_ptr_mut(), |s| s.device_ptr_mut()) as *mut f32,
                importance: *self.node_importance.as_mut().map_or(self.dummy_f32.device_ptr_mut(), |s| s.device_ptr_mut()) as *mut f32,
                temporal: *self.node_temporal.as_mut().map_or(self.dummy_f32.device_ptr_mut(), |s| s.device_ptr_mut()) as *mut f32,
                graph_id: *self.node_graph_id.as_mut().map_or(self.dummy_i32.device_ptr_mut(), |s| s.device_ptr_mut()) as *mut i32,
                cluster: *self.node_cluster.as_mut().map_or(self.dummy_i32.device_ptr_mut(), |s| s.device_ptr_mut()) as *mut i32,
            },
            edges: GpuEdgeData {
                src: *self.edge_src.device_ptr_mut() as *mut i32,
                dst: *self.edge_dst.device_ptr_mut() as *mut i32,
                weight: *self.edge_weight.device_ptr_mut() as *mut f32,
                graph_id: *self.edge_graph_id.as_mut().map_or(self.dummy_i32.device_ptr_mut(), |s| s.device_ptr_mut()) as *mut i32,
            },
            constraints: *self.constraints.as_mut().map_or(self.dummy_constraints.device_ptr_mut(), |s| s.device_ptr_mut()) as *mut ConstraintData,
            params: self.params,
            num_nodes: self.num_nodes as i32,
            num_edges: self.num_edges as i32,
            num_constraints: self.num_constraints as i32,
        };

        // Launch kernel
        unsafe {
            self.compute_kernel.clone().launch(config, (kernel_params,))
                .map_err(|e| Error::new(ErrorKind::Other, format!("Kernel launch failed: {}", e)))?;
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
        if let Some(stress_kernel) = &self.stress_kernel {
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
                stress_kernel.clone().launch(
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