use actix::prelude::*;
use log::{error, warn, info, trace};
use std::io::{Error, ErrorKind};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
use cudarc::nvrtc::Ptx;
use cudarc::driver::sys::CUdevice_attribute_enum;

use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::{Constraint, AdvancedParams};
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::utils::edge_data::EdgeData;
use crate::utils::advanced_gpu_compute::{AdvancedGPUContext, EnhancedBinaryNodeData, EnhancedEdgeData};
use crate::gpu::visual_analytics::{VisualAnalyticsGPU, VisualAnalyticsParams, TSNode, TSEdge, IsolationLayer, Vec4};
use crate::types::vec3::Vec3Data;
use crate::actors::messages::*;
use std::path::Path;
use std::env;
use std::sync::Arc;
use actix::fut::{ActorFutureExt}; // For .map() on ActorFuture
use serde::{Serialize, Deserialize};
// use futures_util::future::FutureExt as _; // For .into_actor() - note the `as _` to avoid name collision if FutureExt is also in scope from elsewhere

// Constants for GPU computation
const BLOCK_SIZE: u32 = 256;
const MAX_NODES: u32 = 1_000_000;
const NODE_SIZE: u32 = std::mem::size_of::<BinaryNodeData>() as u32;
const SHARED_MEM_SIZE: u32 = BLOCK_SIZE * NODE_SIZE;
const DEBUG_THROTTLE: u32 = 60;

// Constants for retry mechanism
// const MAX_GPU_INIT_RETRIES: u32 = 3; // Unused
// const RETRY_DELAY_MS: u64 = 500; // Unused

// Constants for error watchdog (Insight 1.9)
const MAX_GPU_FAILURES: u32 = 5;
const FAILURE_RESET_INTERVAL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphType {
    Knowledge,  // Logseq knowledge graph
    Agent,      // AI agent swarm
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComputeMode {
    Legacy,     // Single-graph mode with compute_forces.ptx
    DualGraph,  // Dual-graph mode with compute_dual_graphs.ptx  
    Advanced,   // Advanced mode with constraints and advanced_compute_forces.ptx
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelMode {
    Legacy,          // Standard legacy kernel for simple graphs (<1000 nodes)
    Advanced,        // Advanced kernel for medium complexity (1000-10000 nodes)
    VisualAnalytics, // Visual analytics kernel for complex analysis (>10000 nodes or isolation layers)
}

#[derive(Debug)]
pub struct GPUComputeActor {
    device: Option<Arc<CudaDevice>>,
    
    // Five kernels for different compute modes
    force_kernel: Option<CudaFunction>,        // Legacy single-graph
    dual_graph_kernel: Option<CudaFunction>,   // Dual-graph physics
    advanced_kernel: Option<CudaFunction>,     // Advanced with constraints
    visual_analytics_kernel: Option<CudaFunction>, // Visual analytics core kernel
    advanced_gpu_kernel: Option<CudaFunction>, // Advanced GPU algorithms kernel
    
    // Separate data for each graph type
    knowledge_node_data: Option<CudaSlice<BinaryNodeData>>,
    agent_node_data: Option<CudaSlice<BinaryNodeData>>,
    
    // Combined node data for legacy single-graph mode
    node_data: Option<CudaSlice<BinaryNodeData>>,
    
    // CRITICAL FIX: Add edge data for spring force calculations
    edge_data: Option<CudaSlice<EdgeData>>,
    num_edges: u32,
    
    num_knowledge_nodes: u32,
    num_agent_nodes: u32,
    num_nodes: u32,  // Total for legacy compatibility
    
    // Track which graph each node belongs to
    graph_type_map: HashMap<u32, GraphType>,
    knowledge_node_indices: HashMap<u32, usize>,
    agent_node_indices: HashMap<u32, usize>,
    node_indices: HashMap<u32, usize>,  // Legacy combined map
    
    // Separate physics parameters for each graph type
    knowledge_sim_params: SimulationParams,
    agent_sim_params: SimulationParams,
    simulation_params: SimulationParams,  // Legacy combined params
    
    // Advanced physics support
    advanced_params: AdvancedParams,
    constraints: Vec<Constraint>,
    advanced_gpu_context: Option<AdvancedGPUContext>,
    
    // Visual analytics support
    visual_analytics_gpu: Option<VisualAnalyticsGPU>,
    visual_analytics_params: Option<VisualAnalyticsParams>,
    isolation_layers: Vec<IsolationLayer>,
    
    iteration_count: u32,
    gpu_failure_count: u32,
    last_failure_reset: Instant,
    cpu_fallback_active: bool,
    
    // Current compute mode and kernel mode
    compute_mode: ComputeMode,
    kernel_mode: KernelMode,
    
    // Stress majorization settings
    stress_majorization_interval: u32,
    last_stress_majorization: u32,
}

// Struct to hold the results of GPU initialization
struct GpuInitializationResult {
    device: Arc<CudaDevice>,
    force_kernel: CudaFunction,
    dual_graph_kernel: Option<CudaFunction>,
    advanced_kernel: Option<CudaFunction>,
    visual_analytics_kernel: Option<CudaFunction>,
    advanced_gpu_kernel: Option<CudaFunction>,
    node_data: CudaSlice<BinaryNodeData>,
    edge_data: Option<CudaSlice<EdgeData>>,  // CRITICAL FIX: Include edge data
    num_nodes: u32,
    num_edges: u32,  // CRITICAL FIX: Include edge count
    node_indices: HashMap<u32, usize>,
}

impl GPUComputeActor {
    pub fn new() -> Self {
        // Create default physics params for knowledge graph (gentler forces)
        let mut knowledge_params = SimulationParams::default();
        knowledge_params.spring_strength = 0.15;  // Gentler springs
        knowledge_params.damping = 0.85;          // More damping for stability
        knowledge_params.repulsion = 50.0;        // Less repulsion
        knowledge_params.time_step = 0.016;       // Standard 60 FPS timestep
        
        // Agent graph uses default params (more dynamic)
        let agent_params = SimulationParams::default();
        
        Self {
            device: None,
            force_kernel: None,
            dual_graph_kernel: None,
            advanced_kernel: None,
            visual_analytics_kernel: None,
            advanced_gpu_kernel: None,
            
            knowledge_node_data: None,
            agent_node_data: None,
            node_data: None,
            edge_data: None,  // CRITICAL FIX: Initialize edge data
            
            num_knowledge_nodes: 0,
            num_agent_nodes: 0,
            num_nodes: 0,
            num_edges: 0,  // CRITICAL FIX: Initialize edge count
            
            graph_type_map: HashMap::new(),
            knowledge_node_indices: HashMap::new(),
            agent_node_indices: HashMap::new(),
            node_indices: HashMap::new(),
            
            knowledge_sim_params: knowledge_params,
            agent_sim_params: agent_params,
            simulation_params: SimulationParams::default(),
            
            // Initialize advanced physics support
            advanced_params: AdvancedParams::default(),
            constraints: Vec::new(),
            advanced_gpu_context: None,
            
            // Initialize visual analytics support
            visual_analytics_gpu: None,
            visual_analytics_params: None,
            isolation_layers: Vec::new(),
            
            iteration_count: 0,
            gpu_failure_count: 0,
            last_failure_reset: Instant::now(),
            cpu_fallback_active: false,
            
            // Start in legacy mode for backward compatibility
            compute_mode: ComputeMode::Legacy,
            kernel_mode: KernelMode::Legacy,
            
            // Stress majorization every 300 iterations (5 seconds at 60fps)
            stress_majorization_interval: 300,
            last_stress_majorization: 0,
        }
    }

    // --- Static GPU Initialization Logic ---

    async fn static_test_gpu_capabilities() -> Result<(), Error> {
        info!("(Static) Testing CUDA capabilities");
        match CudaDevice::count() {
            Ok(count) => {
                info!("Found {} CUDA device(s)", count);
                if count == 0 {
                    Err(Error::new(ErrorKind::NotFound, "No CUDA devices found. Ensure NVIDIA drivers are installed and working."))
                } else {
                    Ok(())
                }
            }
            Err(e) => Err(Error::new(ErrorKind::Other, format!("Failed to get CUDA device count: {}. Check NVIDIA drivers.", e))),
        }
    }

    async fn static_create_cuda_device() -> Result<Arc<CudaDevice>, Error> {
        trace!("(Static) Starting CUDA device initialization sequence");
        if let Ok(uuid) = env::var("NVIDIA_GPU_UUID") {
            info!("(Static) Using GPU UUID {} via environment variables", uuid);
        }
        info!("(Static) Creating CUDA device with index 0");
        match CudaDevice::new(0) {
            Ok(device) => {
                let max_threads = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let compute_mode = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as _).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                let multiprocessor_count = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as _).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                
                info!("(Static) GPU Device detected:");
                info!("  Max threads per MP: {}", max_threads);
                info!("  Multiprocessor count: {}", multiprocessor_count);
                info!("  Compute mode: {}", compute_mode);

                if max_threads < 256 {
                    Err(Error::new(ErrorKind::Other, format!("GPU capability too low: {} threads per multiprocessor; minimum required is 256", max_threads)))
                } else {
                    Ok(device.into())
                }
            }
            Err(e) => Err(Error::new(ErrorKind::Other, format!("Failed to create CUDA device: {}", e))),
        }
    }

    async fn static_load_compute_kernels(
        device: Arc<CudaDevice>, 
        num_nodes: u32,
        graph_nodes: &[crate::models::node::Node], // Pass slice of nodes
    ) -> Result<(CudaFunction, Option<CudaFunction>, Option<CudaFunction>, Option<CudaFunction>, Option<CudaFunction>, CudaSlice<BinaryNodeData>, HashMap<u32, usize>), Error> {
        // Load legacy compute_forces kernel (required)
        let legacy_ptx_path = "/app/src/utils/compute_forces.ptx";
        if !Path::new(legacy_ptx_path).exists() {
            return Err(Error::new(ErrorKind::NotFound, format!("Legacy PTX file not found at {}", legacy_ptx_path)));
        }
        
        let ptx = Ptx::from_file(legacy_ptx_path);
        info!("(Static) Successfully loaded legacy PTX file");
        
        device.load_ptx(ptx, "compute_forces_kernel", &["compute_forces_kernel"]).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        let force_kernel = device.get_func("compute_forces_kernel", "compute_forces_kernel").ok_or_else(|| Error::new(ErrorKind::Other, "Function compute_forces_kernel not found"))?;
        
        // Load dual graph kernel (optional)
        let dual_graph_kernel = {
            let dual_ptx_path = "/app/src/utils/compute_dual_graphs.ptx";
            if Path::new(dual_ptx_path).exists() {
                info!("(Static) Loading dual graph PTX file");
                let ptx = Ptx::from_file(dual_ptx_path);
                device.load_ptx(ptx, "compute_dual_graphs_kernel", &["compute_dual_graphs_kernel"]).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                Some(device.get_func("compute_dual_graphs_kernel", "compute_dual_graphs_kernel").ok_or_else(|| Error::new(ErrorKind::Other, "Function compute_dual_graphs_kernel not found"))?)
            } else {
                warn!("(Static) Dual graph PTX not found, dual graph mode will not be available");
                None
            }
        };
        
        // Load advanced kernel (optional)
        let advanced_kernel = {
            let advanced_ptx_path = "/app/src/utils/advanced_compute_forces.ptx";
            if Path::new(advanced_ptx_path).exists() {
                info!("(Static) Loading advanced physics PTX file");
                let ptx = Ptx::from_file(advanced_ptx_path);
                device.load_ptx(ptx, "advanced_forces_kernel", &["advanced_forces_kernel"]).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
                Some(device.get_func("advanced_forces_kernel", "advanced_forces_kernel").ok_or_else(|| Error::new(ErrorKind::Other, "Function advanced_forces_kernel not found"))?)
            } else {
                warn!("(Static) Advanced physics PTX not found, advanced mode will not be available");
                None
            }
        };
        
        // Load visual analytics kernel (optional)
        let visual_analytics_kernel = {
            // Try multiple possible paths for the visual analytics PTX
            let va_ptx_paths = [
                "/app/src/utils/visual_analytics_core_manual.ptx",
                "/workspace/ext/src/utils/visual_analytics_core_manual.ptx",
                "/app/src/utils/visual_analytics_core.ptx",
                "/workspace/ext/src/utils/visual_analytics_core.ptx"
            ];
            
            let mut kernel = None;
            for va_ptx_path in &va_ptx_paths {
                if Path::new(va_ptx_path).exists() {
                    info!("(Static) Loading visual analytics core PTX file from {}", va_ptx_path);
                    match Ptx::from_file(va_ptx_path) {
                        Ok(ptx) => {
                            match device.load_ptx(ptx, "visual_analytics_kernel", &["visual_analytics_kernel"]) {
                                Ok(_) => {
                                    kernel = device.get_func("visual_analytics_kernel", "visual_analytics_kernel");
                                    break;
                                },
                                Err(e) => warn!("Failed to load PTX from {}: {}", va_ptx_path, e),
                            }
                        },
                        Err(e) => warn!("Failed to read PTX from {}: {}", va_ptx_path, e),
                    }
                }
            }
            
            if kernel.is_none() {
                warn!("(Static) Visual analytics PTX not found at any location, visual analytics mode will not be available");
            }
            
            kernel
        };
        
        // Load advanced GPU algorithms kernel (optional)
        let advanced_gpu_kernel = {
            // Try multiple possible paths for the advanced GPU algorithms PTX
            let aga_ptx_paths = [
                "/app/src/utils/advanced_gpu_algorithms.ptx",
                "/workspace/ext/src/utils/advanced_gpu_algorithms.ptx"
            ];
            
            let mut kernel = None;
            for aga_ptx_path in &aga_ptx_paths {
                if Path::new(aga_ptx_path).exists() {
                    info!("(Static) Loading advanced GPU algorithms PTX file from {}", aga_ptx_path);
                    match Ptx::from_file(aga_ptx_path) {
                        Ok(ptx) => {
                            match device.load_ptx(ptx, "advanced_gpu_algorithms_kernel", &["advanced_gpu_algorithms_kernel"]) {
                                Ok(_) => {
                                    kernel = device.get_func("advanced_gpu_algorithms_kernel", "advanced_gpu_algorithms_kernel");
                                    break;
                                },
                                Err(e) => warn!("Failed to load PTX from {}: {}", aga_ptx_path, e),
                            }
                        },
                        Err(e) => warn!("Failed to read PTX from {}: {}", aga_ptx_path, e),
                    }
                }
            }
            
            if kernel.is_none() {
                warn!("(Static) Advanced GPU algorithms PTX not found at any location, will use fallback methods");
            }
            
            kernel
        };
        
        info!("(Static) Allocating device memory for {} nodes", num_nodes);
        let mut node_data_gpu = device.alloc_zeros::<BinaryNodeData>(num_nodes as usize).map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
        
        let mut node_indices = HashMap::new();
        let mut node_data_host = Vec::with_capacity(graph_nodes.len());

        for (idx, node) in graph_nodes.iter().enumerate() {
            node_indices.insert(node.id, idx);
            node_data_host.push(BinaryNodeData {
                position: node.data.position.clone(),
                velocity: node.data.velocity.clone(),
                mass: node.data.mass,
                flags: node.data.flags,
                padding: node.data.padding,
            });
        }
        
        device.htod_sync_copy_into(&node_data_host, &mut node_data_gpu).map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy node data to GPU: {}", e)))?;
        
        Ok((force_kernel, dual_graph_kernel, advanced_kernel, visual_analytics_kernel, advanced_gpu_kernel, node_data_gpu, node_indices))
    }

    async fn perform_gpu_initialization(graph: GraphData) -> Result<GpuInitializationResult, Error> {
        let num_nodes = graph.nodes.len() as u32;
        info!("(Static Logic) Initializing GPU for {} nodes", num_nodes);

        if num_nodes > MAX_NODES {
            return Err(Error::new(ErrorKind::Other, format!("Node count {} exceeds limit {}", num_nodes, MAX_NODES)));
        }

        Self::static_test_gpu_capabilities().await?;
        info!("(Static Logic) GPU capabilities check passed");

        let device = Self::static_create_cuda_device().await?;
        info!("(Static Logic) CUDA device created successfully");
        
        // Pass graph.nodes which is Vec<Node>
        let (force_kernel, dual_graph_kernel, advanced_kernel, visual_analytics_kernel, advanced_gpu_kernel, node_data, node_indices) = Self::static_load_compute_kernels(device.clone(), num_nodes, &graph.nodes).await?;
        info!("(Static Logic) Compute kernels loaded and data copied");
        
        // CRITICAL FIX: Process and upload edge data during initialization
        let num_edges = graph.edges.len() as u32;
        let edge_data = if !graph.edges.is_empty() {
            info!("(Static Logic) Processing {} edges for spring forces", num_edges);
            let mut host_edge_data = Vec::with_capacity(graph.edges.len());
            
            for edge in &graph.edges {
                // Find indices for source and target nodes
                let source_idx = node_indices.get(&edge.source)
                    .copied()
                    .unwrap_or(0) as i32;
                let target_idx = node_indices.get(&edge.target)
                    .copied()
                    .unwrap_or(0) as i32;
                
                host_edge_data.push(EdgeData {
                    source_idx,
                    target_idx,
                    weight: edge.weight,
                });
            }
            
            let mut edge_slice = device.alloc_zeros::<EdgeData>(graph.edges.len())
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate edge buffer: {}", e)))?;
            
            device.htod_sync_copy_into(&host_edge_data, &mut edge_slice)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy edge data to GPU: {}", e)))?;
            
            info!("(Static Logic) Successfully uploaded {} edges to GPU", num_edges);
            Some(edge_slice)
        } else {
            info!("(Static Logic) No edges in graph - spring forces will not be applied");
            None
        };
        
        Ok(GpuInitializationResult {
            device, // No Some() needed, it's Arc<CudaDevice>
            force_kernel, // No Some()
            dual_graph_kernel,
            advanced_kernel,
            visual_analytics_kernel,
            advanced_gpu_kernel,
            node_data,    // No Some()
            edge_data,    // CRITICAL FIX: Include edge data
            num_nodes,
            num_edges,    // CRITICAL FIX: Include edge count
            node_indices,
        })
    }

    // --- Instance Methods ---

    fn update_graph_data_internal(&mut self, graph: &GraphData) -> Result<(), Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let node_data_slice = self.node_data.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;
 
        info!("Updating graph data for {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        
        self.node_indices.clear();
        for (idx, node) in graph.nodes.iter().enumerate() {
            self.node_indices.insert(node.id, idx);
        }

        if graph.nodes.len() as u32 != self.num_nodes {
            info!("Reallocating GPU buffer for {} nodes", graph.nodes.len());
            *node_data_slice = device.alloc_zeros::<BinaryNodeData>(graph.nodes.len())
                .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
            self.num_nodes = graph.nodes.len() as u32;
            self.iteration_count = 0; // Reset iteration count on realloc
        }

        let mut host_node_data = Vec::with_capacity(graph.nodes.len());
        for node in &graph.nodes {
            host_node_data.push(BinaryNodeData {
                position: node.data.position.clone(),
                velocity: node.data.velocity.clone(),
                mass: node.data.mass,
                flags: node.data.flags,
                padding: node.data.padding,
            });
        }

        device.htod_sync_copy_into(&host_node_data, node_data_slice)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy node data to GPU: {}", e)))?;
        
        // CRITICAL FIX: Process and upload edge data for spring forces
        if graph.edges.len() as u32 != self.num_edges || self.edge_data.is_none() {
            info!("Reallocating GPU buffer for {} edges", graph.edges.len());
            if graph.edges.is_empty() {
                self.edge_data = None;
                self.num_edges = 0;
            } else {
                let edge_slice = device.alloc_zeros::<EdgeData>(graph.edges.len())
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate edge buffer: {}", e)))?;
                self.edge_data = Some(edge_slice);
                self.num_edges = graph.edges.len() as u32;
            }
        }
        
        // Convert edges to EdgeData format with node indices
        if !graph.edges.is_empty() {
            let mut host_edge_data = Vec::with_capacity(graph.edges.len());
            for edge in &graph.edges {
                // Find indices for source and target nodes
                let source_idx = self.node_indices.get(&edge.source)
                    .copied()
                    .unwrap_or(0) as i32;
                let target_idx = self.node_indices.get(&edge.target)
                    .copied()
                    .unwrap_or(0) as i32;
                
                host_edge_data.push(EdgeData {
                    source_idx,
                    target_idx,
                    weight: edge.weight,
                });
            }
            
            if let Some(edge_slice) = self.edge_data.as_mut() {
                device.htod_sync_copy_into(&host_edge_data, edge_slice)
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy edge data to GPU: {}", e)))?;
                info!("Successfully uploaded {} edges to GPU", host_edge_data.len());
            }
        }
        
        Ok(())
    }

    /// Determine the optimal kernel mode based on graph complexity and analysis requirements
    fn determine_kernel_mode(&self, num_nodes: u32, has_isolation_layers: bool, has_complex_analysis: bool) -> KernelMode {
        // Priority 1: Use visual analytics for complex analysis or isolation layers
        if (has_isolation_layers || has_complex_analysis) && self.visual_analytics_kernel.is_some() {
            return KernelMode::VisualAnalytics;
        }
        
        // Priority 2: Use advanced kernel for medium-large graphs
        if num_nodes >= 1000 && num_nodes <= 10000 && self.advanced_gpu_kernel.is_some() {
            return KernelMode::Advanced;
        }
        
        // Priority 3: Use visual analytics for very large graphs
        if num_nodes > 10000 && self.visual_analytics_kernel.is_some() {
            return KernelMode::VisualAnalytics;
        }
        
        // Fallback to legacy for small graphs or when advanced kernels are not available
        KernelMode::Legacy
    }

    /// Update kernel mode based on current graph state
    fn update_kernel_mode(&mut self) {
        let has_isolation_layers = !self.isolation_layers.is_empty();
        let has_complex_analysis = !self.constraints.is_empty() || 
                                 matches!(self.compute_mode, ComputeMode::Advanced);
        
        let new_mode = self.determine_kernel_mode(self.num_nodes, has_isolation_layers, has_complex_analysis);
        
        if new_mode != self.kernel_mode {
            info!("GPU: Switching kernel mode from {:?} to {:?} (nodes: {}, layers: {}, complex: {})", 
                  self.kernel_mode, new_mode, self.num_nodes, has_isolation_layers, has_complex_analysis);
            self.kernel_mode = new_mode;
        }
    }

    fn compute_forces_internal(&mut self) -> Result<(), Error> {
        if self.cpu_fallback_active {
            warn!("GPU compute in CPU fallback mode, skipping GPU kernel");
            return Ok(());
        }

        if self.last_failure_reset.elapsed() > FAILURE_RESET_INTERVAL {
            if self.gpu_failure_count > 0 {
                info!("Resetting GPU failure count after {} seconds", FAILURE_RESET_INTERVAL.as_secs());
                self.gpu_failure_count = 0;
                self.cpu_fallback_active = false; // Attempt to re-enable GPU
            }
            self.last_failure_reset = Instant::now();
        }

        // Update kernel mode based on current state
        self.update_kernel_mode();

        // Check for stress majorization trigger
        if self.iteration_count.saturating_sub(self.last_stress_majorization) >= self.stress_majorization_interval {
            if !self.constraints.is_empty() {
                info!("Triggering periodic stress majorization at iteration {}", self.iteration_count);
                if let Err(e) = self.perform_stress_majorization() {
                    warn!("Stress majorization failed: {}", e);
                }
                self.last_stress_majorization = self.iteration_count;
            }
        }

        // Choose compute method based on kernel mode (with fallback to compute mode)
        match self.kernel_mode {
            KernelMode::Legacy => {
                match self.compute_mode {
                    ComputeMode::Legacy => self.compute_forces_legacy(),
                    ComputeMode::DualGraph => self.compute_forces_dual_graph(),
                    ComputeMode::Advanced => self.compute_forces_advanced(),
                }
            },
            KernelMode::Advanced => self.compute_forces_with_advanced_gpu(),
            KernelMode::VisualAnalytics => self.compute_forces_with_visual_analytics(),
        }
    }

    fn compute_forces_legacy(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let force_kernel = self.force_kernel.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Legacy kernel not initialized"))?;
        let node_data = self.node_data.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;

        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Starting legacy force computation (iteration {})", self.iteration_count);
        }

        let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

        // BREADCRUMB: Log physics params being sent to GPU kernel
        if self.iteration_count % 60 == 0 { // Log every second at 60 FPS
            info!("GPU kernel params - spring: {}, damping: {}, repulsion: {}, timestep: {}, edges: {}",
                  self.simulation_params.spring_strength,
                  self.simulation_params.damping,
                  self.simulation_params.repulsion,
                  self.simulation_params.time_step,
                  self.num_edges);
        }
        
        // CRITICAL FIX: Include edge data in kernel launch
        let launch_result = if let Some(edge_data) = self.edge_data.as_ref() {
            unsafe {
                force_kernel.clone().launch(cfg, (
                    node_data,
                    edge_data,
                    self.num_nodes as i32,
                    self.num_edges as i32,
                    self.simulation_params.spring_strength,
                    self.simulation_params.damping,
                    self.simulation_params.repulsion,
                    self.simulation_params.time_step,
                    self.simulation_params.max_repulsion_distance,
                    if self.simulation_params.enable_bounds {
                        self.simulation_params.viewport_bounds
                    } else {
                        f32::MAX
                    },
                    self.iteration_count as i32,
                ))
            }
        } else {
            // If no edges, create a dummy edge buffer to satisfy kernel signature
            warn!("No edges in graph - spring forces will not be applied!");
            let dummy_edges = device.alloc_zeros::<EdgeData>(1)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate dummy edge buffer: {}", e)))?;
            unsafe {
                force_kernel.clone().launch(cfg, (
                    node_data,
                    &dummy_edges,
                    self.num_nodes as i32,
                    0i32,
                    self.simulation_params.spring_strength,
                    self.simulation_params.damping,
                    self.simulation_params.repulsion,
                    self.simulation_params.time_step,
                    self.simulation_params.max_repulsion_distance,
                    if self.simulation_params.enable_bounds {
                        self.simulation_params.viewport_bounds
                    } else {
                        f32::MAX
                    },
                    self.iteration_count as i32,
                ))
            }
        };

        self.finish_gpu_computation(launch_result)
    }

    fn compute_forces_dual_graph(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let dual_kernel = self.dual_graph_kernel.as_ref().ok_or_else(|| {
            warn!("Dual graph kernel not available, falling back to legacy");
            self.compute_mode = ComputeMode::Legacy;
            Error::new(ErrorKind::Other, "Dual graph kernel not initialized")
        })?;

        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Starting dual graph force computation (iteration {}), knowledge: {}, agent: {}", 
                   self.iteration_count, self.num_knowledge_nodes, self.num_agent_nodes);
        }

        // Check if we have both graphs initialized
        if self.knowledge_node_data.is_none() || self.agent_node_data.is_none() {
            warn!("Dual graph data not properly initialized, falling back to legacy");
            self.compute_mode = ComputeMode::Legacy;
            return self.compute_forces_legacy();
        }

        let knowledge_data = self.knowledge_node_data.as_ref().unwrap();
        let agent_data = self.agent_node_data.as_ref().unwrap();
        let edge_data = self.edge_data.as_ref();

        // Calculate blocks for the maximum number of nodes
        let total_nodes = self.num_knowledge_nodes + self.num_agent_nodes;
        let blocks = ((total_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

        // Create separate edge buffers for knowledge and agent graphs
        // For now, we'll use the same edge buffer but filter by node type
        let launch_result = if let Some(edges) = edge_data {
            unsafe {
                dual_kernel.clone().launch(cfg, (
                    knowledge_data,           // Knowledge graph nodes
                    agent_data,               // Agent graph nodes
                    edges,                    // Knowledge graph edges (same buffer for now)
                    edges,                    // Agent graph edges (same buffer for now)
                    self.num_knowledge_nodes as i32,
                    self.num_agent_nodes as i32,
                    self.num_edges as i32,    // Knowledge edges count
                    0i32,                     // Agent edges count (0 for now)
                    self.knowledge_sim_params.spring_strength,
                    self.knowledge_sim_params.damping,
                    self.knowledge_sim_params.repulsion,
                    self.agent_sim_params.spring_strength,
                    self.agent_sim_params.damping,
                    self.agent_sim_params.repulsion,
                    self.simulation_params.time_step,
                    self.simulation_params.max_repulsion_distance,
                    self.simulation_params.viewport_bounds,
                    self.iteration_count as i32,
                    true,   // process_knowledge
                    true,   // process_agents
                ))
            }
        } else {
            // Create dummy edge buffer if no edges
            let dummy_edges = device.alloc_zeros::<EdgeData>(1)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate dummy edge buffer: {}", e)))?;
            unsafe {
                dual_kernel.clone().launch(cfg, (
                    knowledge_data,
                    agent_data,
                    &dummy_edges,
                    &dummy_edges,
                    self.num_knowledge_nodes as i32,
                    self.num_agent_nodes as i32,
                    0i32,
                    0i32,
                    self.knowledge_sim_params.spring_strength,
                    self.knowledge_sim_params.damping,
                    self.knowledge_sim_params.repulsion,
                    self.agent_sim_params.spring_strength,
                    self.agent_sim_params.damping,
                    self.agent_sim_params.repulsion,
                    self.simulation_params.time_step,
                    self.simulation_params.max_repulsion_distance,
                    self.simulation_params.viewport_bounds,
                    self.iteration_count as i32,
                    true,
                    true,
                ))
            }
        };

        self.finish_gpu_computation(launch_result)
    }

    fn compute_forces_advanced(&mut self) -> Result<(), Error> {
        // Use the advanced GPU context if available
        if let Some(ref mut ctx) = self.advanced_gpu_context {
            if self.iteration_count % DEBUG_THROTTLE == 0 {
                trace!("Starting advanced force computation (iteration {})", self.iteration_count);
            }
            
            match ctx.step_with_constraints(&self.constraints) {
                Ok(_) => {
                    self.iteration_count += 1;
                    Ok(())
                },
                Err(e) => {
                    warn!("Advanced physics failed: {}, falling back to legacy", e);
                    self.compute_mode = ComputeMode::Legacy;
                    self.compute_forces_legacy()
                }
            }
        } else {
            // Fall back to using the advanced kernel directly if context is not available
            let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
            let advanced_kernel = self.advanced_kernel.as_ref().ok_or_else(|| {
                warn!("Advanced kernel not available, falling back to legacy");
                self.compute_mode = ComputeMode::Legacy;
                Error::new(ErrorKind::Other, "Advanced kernel not initialized")
            })?;

            if self.iteration_count % DEBUG_THROTTLE == 0 {
                trace!("Starting advanced force computation with direct kernel (iteration {})", self.iteration_count);
            }

            // This is a simplified direct kernel launch - the full implementation
            // would use the AdvancedGPUContext for proper constraint handling
            warn!("Using simplified advanced kernel without full constraint support");
            self.compute_forces_legacy() // Fallback for now
        }
    }

    fn compute_forces_with_advanced_gpu(&mut self) -> Result<(), Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let advanced_gpu_kernel = self.advanced_gpu_kernel.as_ref().ok_or_else(|| {
            warn!("Advanced GPU algorithms kernel not available, falling back to legacy");
            self.kernel_mode = KernelMode::Legacy;
            Error::new(ErrorKind::Other, "Advanced GPU algorithms kernel not initialized")
        })?;

        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Starting advanced GPU algorithms computation (iteration {})", self.iteration_count);
        }

        let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: SHARED_MEM_SIZE,
        };

        let node_data = self.node_data.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;
        let edge_data = self.edge_data.as_ref();

        let launch_result = if let Some(edge_data) = edge_data {
            unsafe {
                advanced_gpu_kernel.clone().launch(cfg, (
                    node_data,
                    edge_data,
                    self.num_nodes as i32,
                    self.num_edges as i32,
                    self.simulation_params.spring_strength,
                    self.simulation_params.damping,
                    self.simulation_params.repulsion,
                    self.simulation_params.time_step,
                    self.simulation_params.max_repulsion_distance,
                    if self.simulation_params.enable_bounds {
                        self.simulation_params.viewport_bounds
                    } else {
                        f32::MAX
                    },
                    self.iteration_count as i32,
                ))
            }
        } else {
            warn!("No edge data for advanced GPU algorithms, falling back to legacy");
            self.kernel_mode = KernelMode::Legacy;
            return self.compute_forces_legacy();
        };

        self.finish_gpu_computation(launch_result)
    }

    fn compute_forces_with_visual_analytics(&mut self) -> Result<(), Error> {
        // If we have a visual analytics GPU context, use it
        if let Some(ref mut va_gpu) = self.visual_analytics_gpu {
            if self.iteration_count % DEBUG_THROTTLE == 0 {
                trace!("Starting visual analytics GPU computation (iteration {})", self.iteration_count);
            }

            // Convert legacy node data to visual analytics format
            let ts_nodes = self.convert_to_ts_nodes()?;
            let ts_edges = self.convert_to_ts_edges()?;

            // Stream data to visual analytics GPU
            if let Err(e) = va_gpu.stream_nodes(&ts_nodes) {
                warn!("Failed to stream nodes to visual analytics GPU: {}, falling back", e);
                self.kernel_mode = KernelMode::Legacy;
                return self.compute_forces_legacy();
            }

            if let Err(e) = va_gpu.stream_edges(&ts_edges) {
                warn!("Failed to stream edges to visual analytics GPU: {}, falling back", e);
                self.kernel_mode = KernelMode::Legacy;
                return self.compute_forces_legacy();
            }

            if let Err(e) = va_gpu.update_layers(&self.isolation_layers) {
                warn!("Failed to update isolation layers: {}, falling back", e);
                self.kernel_mode = KernelMode::Legacy;
                return self.compute_forces_legacy();
            }

            // Execute visual analytics pipeline
            if let Some(ref params) = self.visual_analytics_params {
                match va_gpu.execute(params, ts_nodes.len(), ts_edges.len(), self.isolation_layers.len()) {
                    Ok(_) => {
                        // Copy results back to legacy format
                        if let Err(e) = self.copy_visual_analytics_results(va_gpu) {
                            warn!("Failed to copy visual analytics results: {}", e);
                        }
                        self.iteration_count += 1;
                        Ok(())
                    },
                    Err(e) => {
                        warn!("Visual analytics execution failed: {}, falling back", e);
                        self.kernel_mode = KernelMode::Legacy;
                        self.compute_forces_legacy()
                    }
                }
            } else {
                warn!("Visual analytics parameters not initialized, falling back");
                self.kernel_mode = KernelMode::Legacy;
                self.compute_forces_legacy()
            }
        } else {
            // Use the visual analytics kernel directly
            let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
            let va_kernel = self.visual_analytics_kernel.as_ref().ok_or_else(|| {
                warn!("Visual analytics kernel not available, falling back to legacy");
                self.kernel_mode = KernelMode::Legacy;
                Error::new(ErrorKind::Other, "Visual analytics kernel not initialized")
            })?;

            if self.iteration_count % DEBUG_THROTTLE == 0 {
                trace!("Starting visual analytics kernel computation (iteration {})", self.iteration_count);
            }

            // For now, fall back to legacy - full visual analytics integration would require
            // converting all data structures and implementing the full pipeline
            warn!("Direct visual analytics kernel execution not yet fully implemented, falling back");
            self.kernel_mode = KernelMode::Legacy;
            self.compute_forces_legacy()
        }
    }

    fn finish_gpu_computation(&mut self, launch_result: Result<(), cudarc::driver::DriverError>) -> Result<(), Error> {
        let device = self.device.as_ref().unwrap();
        
        match launch_result {
            Ok(_) => {
                match device.synchronize() {
                    Ok(_) => {
                        if self.iteration_count % DEBUG_THROTTLE == 0 {
                            trace!("Force computation completed successfully");
                        }
                        self.iteration_count += 1;
                        Ok(())
                    },
                    Err(e) => self.handle_gpu_error(format!("GPU synchronization failed: {}", e)),
                }
            },
            Err(e) => self.handle_gpu_error(format!("Kernel launch failed: {}", e)),
        }
    }

    fn perform_stress_majorization(&mut self) -> Result<(), Error> {
        // This is a placeholder for stress majorization implementation
        // In a full implementation, this would:
        // 1. Copy current node positions from GPU
        // 2. Run stress majorization algorithm on CPU or separate GPU kernel
        // 3. Apply the optimized positions back to GPU
        
        info!("Performing stress majorization with {} constraints", self.constraints.len());
        
        // For now, just log that we would perform stress majorization
        // A real implementation would use the stress_majorization module
        
        Ok(())
    }

    fn handle_gpu_error(&mut self, error_msg: String) -> Result<(), Error> {
        self.gpu_failure_count += 1;
        error!("GPU error (failure {}/{}): {}", self.gpu_failure_count, MAX_GPU_FAILURES, error_msg);

        if self.gpu_failure_count >= MAX_GPU_FAILURES {
            warn!("GPU failure count exceeded limit, activating CPU fallback mode");
            self.cpu_fallback_active = true;
            // Reset failure count to allow retry later, but keep fallback active until reset interval
            // self.gpu_failure_count = 0; // Don't reset immediately, let the interval handle it
            // self.last_failure_reset = Instant::now();
        }
        Err(Error::new(ErrorKind::Other, error_msg))
    }

    /// Convert legacy BinaryNodeData to visual analytics TSNode format
    fn convert_to_ts_nodes(&self) -> Result<Vec<TSNode>, Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let node_data = self.node_data.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;

        let mut gpu_raw_data = vec![BinaryNodeData {
            position: Vec3Data::zero(),
            velocity: Vec3Data::zero(),
            mass: 0,
            flags: 0,
            padding: [0, 0],
        }; self.num_nodes as usize];

        device.dtoh_sync_copy_into(node_data, &mut gpu_raw_data)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy data from GPU: {}", e)))?;

        let mut ts_nodes = Vec::with_capacity(gpu_raw_data.len());
        for (idx, node) in gpu_raw_data.iter().enumerate() {
            let mut ts_node = TSNode::default();
            
            // Convert position and velocity
            ts_node.position.x = node.position.x;
            ts_node.position.y = node.position.y;
            ts_node.position.z = node.position.z;
            ts_node.position.t = self.iteration_count as f32 * self.simulation_params.time_step;
            
            ts_node.velocity.x = node.velocity.x;
            ts_node.velocity.y = node.velocity.y;
            ts_node.velocity.z = node.velocity.z;
            ts_node.velocity.t = 0.0; // Temporal velocity not used yet
            
            // Set basic properties
            ts_node.visual_saliency = if node.mass > 0 { node.mass as f32 / 100.0 } else { 1.0 };
            ts_node.force_scale = 1.0;
            ts_node.damping_local = self.simulation_params.damping;
            
            // Set hierarchy level based on node index (simple heuristic)
            ts_node.hierarchy_level = if idx < 100 { 0 } else if idx < 1000 { 1 } else { 2 };
            
            ts_nodes.push(ts_node);
        }

        Ok(ts_nodes)
    }

    /// Convert legacy EdgeData to visual analytics TSEdge format
    fn convert_to_ts_edges(&self) -> Result<Vec<TSEdge>, Error> {
        if let Some(edge_data) = &self.edge_data {
            let device = self.device.as_ref().unwrap();
            let mut gpu_edge_data = vec![EdgeData {
                source_idx: 0,
                target_idx: 0,
                weight: 0.0,
            }; self.num_edges as usize];

            device.dtoh_sync_copy_into(edge_data, &mut gpu_edge_data)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy edge data from GPU: {}", e)))?;

            let mut ts_edges = Vec::with_capacity(gpu_edge_data.len());
            for edge in gpu_edge_data.iter() {
                let ts_edge = TSEdge {
                    source: edge.source_idx,
                    target: edge.target_idx,
                    structural_weight: edge.weight,
                    semantic_weight: edge.weight * 0.8, // Simple heuristic
                    temporal_weight: 1.0,
                    causal_weight: 1.0,
                    weight_history: [edge.weight; 8],
                    formation_time: 0.0,
                    stability: 0.9,
                    bundling_strength: 0.5,
                    control_points: [crate::gpu::visual_analytics::Vec4::default(); 2],
                    layer_mask: 1, // Default to primary layer
                    information_flow: edge.weight,
                    latency: 1.0,
                };
                ts_edges.push(ts_edge);
            }

            Ok(ts_edges)
        } else {
            Ok(Vec::new())
        }
    }

    /// Copy results from visual analytics GPU back to legacy format
    fn copy_visual_analytics_results(&mut self, va_gpu: &VisualAnalyticsGPU) -> Result<(), Error> {
        let positions = va_gpu.get_positions()
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to get positions from visual analytics: {}", e)))?;

        // Convert positions back to BinaryNodeData format and upload to GPU
        let device = self.device.as_ref().unwrap();
        let node_data = self.node_data.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;

        let mut updated_nodes = vec![BinaryNodeData {
            position: Vec3Data::zero(),
            velocity: Vec3Data::zero(),
            mass: 0,
            flags: 0,
            padding: [0, 0],
        }; self.num_nodes as usize];

        // Copy current data from GPU
        device.dtoh_sync_copy_into(node_data, &mut updated_nodes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy current node data: {}", e)))?;

        // Update positions from visual analytics results
        for (idx, node) in updated_nodes.iter_mut().enumerate() {
            if idx * 4 + 2 < positions.len() {
                node.position.x = positions[idx * 4];
                node.position.y = positions[idx * 4 + 1];
                node.position.z = positions[idx * 4 + 2];
                // positions[idx * 4 + 3] is the time component, not used in legacy format
            }
        }

        // Upload updated data back to GPU
        device.htod_sync_copy_into(&updated_nodes, node_data)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload updated node data: {}", e)))?;

        Ok(())
    }

    fn get_node_data_internal(&self) -> Result<Vec<BinaryNodeData>, Error> {
        let device = self.device.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Device not initialized"))?;
        let node_data = self.node_data.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Node data not initialized"))?;

        let mut gpu_raw_data = vec![BinaryNodeData {
            position: Vec3Data::zero(),
            velocity: Vec3Data::zero(),
            mass: 0,
            flags: 0,
            padding: [0, 0],
        }; self.num_nodes as usize];

        device.dtoh_sync_copy_into(node_data, &mut gpu_raw_data)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy data from GPU: {}", e)))?;

        Ok(gpu_raw_data)
    }
}

impl Actor for GPUComputeActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("GPUComputeActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("GPUComputeActor stopped");
    }
}

impl Handler<InitializeGPU> for GPUComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: InitializeGPU, _ctx: &mut Self::Context) -> Self::Result {
        let graph_data_owned = msg.graph;
        let node_count = graph_data_owned.nodes.len();
        info!("GPU: InitializeGPU received with {} nodes", node_count);
        
        let fut = GPUComputeActor::perform_gpu_initialization(graph_data_owned);
        
        // Use FutureActorExt trait's into_actor method
        let actor_fut = fut.into_actor(self);

        Box::pin(
            actor_fut.map(|result_of_logic, actor, _ctx_map| {
                match result_of_logic {
                    Ok(init_result) => {
                        actor.device = Some(init_result.device);
                        actor.force_kernel = Some(init_result.force_kernel);
                        actor.dual_graph_kernel = init_result.dual_graph_kernel;
                        actor.advanced_kernel = init_result.advanced_kernel;
                        actor.visual_analytics_kernel = init_result.visual_analytics_kernel;
                        actor.advanced_gpu_kernel = init_result.advanced_gpu_kernel;
                        actor.node_data = Some(init_result.node_data);
                        actor.edge_data = init_result.edge_data;  // CRITICAL FIX
                        actor.num_nodes = init_result.num_nodes;
                        actor.num_edges = init_result.num_edges;  // CRITICAL FIX
                        actor.node_indices = init_result.node_indices;
                        
                        // Reset other relevant state
                        actor.iteration_count = 0;
                        actor.gpu_failure_count = 0;
                        actor.last_failure_reset = Instant::now();
                        actor.cpu_fallback_active = false;

                        info!("GPU initialization successful (applied static logic result)");
                        Ok(())
                    }
                    Err(e) => {
                        error!("GPU initialization failed (static logic): {}", e);
                        actor.device = None;
                        actor.force_kernel = None;
                        actor.node_data = None;
                        actor.num_nodes = 0;
                        actor.node_indices.clear();
                        actor.cpu_fallback_active = true; // Fallback on init failure
                        Err(e.to_string())
                    }
                }
            })
        )
    }
}

impl Handler<UpdateGPUGraphData> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGPUGraphData, _ctx: &mut Self::Context) -> Self::Result {
        let node_count = msg.graph.nodes.len();
        info!("GPU: UpdateGPUGraphData received with {} nodes", node_count);
        
        if self.device.is_none() {
            error!("GPU NOT INITIALIZED! Cannot update graph data. Need to call InitializeGPU first!");
            return Err("GPU not initialized - call InitializeGPU first".to_string());
        }
        
        match self.update_graph_data_internal(&msg.graph) {
            Ok(_) => {
                info!("GPU: Graph data updated successfully with {} nodes", node_count);
                Ok(())
            },
            Err(e) => {
                error!("Failed to update graph data: {}", e);
                Err(e.to_string())
            }
        }
    }
}

impl Handler<UpdateSimulationParams> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateSimulationParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Received physics update - damping: {}, spring: {}, repulsion: {}, iterations: {}", 
              msg.params.damping, msg.params.spring_strength, 
              msg.params.repulsion, msg.params.iterations);
        self.simulation_params = msg.params;
        info!("GPU: Physics parameters updated successfully");
        Ok(())
    }
}

impl Handler<UpdatePhysicsParams> for GPUComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdatePhysicsParams, _ctx: &mut Self::Context) -> Self::Result {
        match msg.graph_type {
            GraphType::Knowledge => {
                self.knowledge_sim_params = msg.params.clone();
                info!("Updated knowledge graph physics parameters");
            }
            GraphType::Agent => {
                self.agent_sim_params = msg.params.clone();
                info!("Updated agent graph physics parameters");
            }
        }
        
        // Also update legacy combined params for backwards compatibility
        self.simulation_params = msg.params;
        
        Ok(())
    }
}

impl Handler<ComputeForces> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: ComputeForces, _ctx: &mut Self::Context) -> Self::Result {
        if self.iteration_count % 60 == 0 { // Log every second
            info!("GPU: ComputeForces called (iteration {}), nodes: {}", 
                  self.iteration_count, self.num_nodes);
        }
        
        if self.device.is_none() {
            error!("GPU NOT INITIALIZED! Cannot compute forces!");
            return Err("GPU not initialized".to_string());
        }
        
        if self.num_nodes == 0 {
            warn!("GPU: No nodes to compute forces for!");
            return Ok(());
        }
        
        match self.compute_forces_internal() {
            Ok(_) => {
                // iteration_count is already incremented in compute_forces_internal
                Ok(())
            },
            Err(e) => {
                if self.cpu_fallback_active {
                    warn!("GPU compute failed, CPU fallback active: {}", e);
                    Ok(())
                } else {
                    error!("GPU compute failed: {}", e);
                    Err(e.to_string())
                }
            }
        }
    }
}

impl Handler<GetNodeData> for GPUComputeActor {
    type Result = Result<Vec<BinaryNodeData>, String>;

    fn handle(&mut self, _msg: GetNodeData, _ctx: &mut Self::Context) -> Self::Result {
         if self.device.is_none() {
            warn!("Attempted to get node data, but GPU is not initialized.");
            return Err("GPU not initialized".to_string());
        }
        match self.get_node_data_internal() {
            Ok(data) => {
                trace!("Retrieved {} node data items from GPU", data.len());
                Ok(data)
            },
            Err(e) => {
                error!("Failed to get node data from GPU: {}", e);
                Err(e.to_string())
            }
        }
    }
}

impl Handler<GetGPUStatus> for GPUComputeActor {
    type Result = MessageResult<GetGPUStatus>;

    fn handle(&mut self, _msg: GetGPUStatus, _ctx: &mut Self::Context) -> Self::Result {
        MessageResult(GPUStatus {
            is_initialized: self.device.is_some(),
            cpu_fallback_active: self.cpu_fallback_active,
            failure_count: self.gpu_failure_count,
            iteration_count: self.iteration_count,
            num_nodes: self.num_nodes,
        })
    }
}

// New message handlers for advanced physics

impl Handler<UpdateConstraints> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateConstraints, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Updating constraints with new constraint data");
        
        // Parse constraints from JSON value
        match serde_json::from_value::<Vec<Constraint>>(msg.constraint_data) {
            Ok(constraints) => {
                info!("GPU: Parsed {} constraints", constraints.len());
                self.constraints = constraints;
                
                // Switch to advanced mode if we have constraints and the kernel is available
                if !self.constraints.is_empty() && self.advanced_kernel.is_some() {
                    if matches!(self.compute_mode, ComputeMode::Legacy) {
                        info!("GPU: Switching to advanced compute mode due to constraints");
                        self.compute_mode = ComputeMode::Advanced;
                    }
                } else if self.constraints.is_empty() && matches!(self.compute_mode, ComputeMode::Advanced) {
                    info!("GPU: Switching back to legacy mode - no constraints");
                    self.compute_mode = ComputeMode::Legacy;
                }
                
                Ok(())
            },
            Err(e) => {
                error!("Failed to parse constraints: {}", e);
                Err(format!("Failed to parse constraints: {}", e))
            }
        }
    }
}

impl Handler<UpdateAdvancedParams> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateAdvancedParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Updating advanced physics parameters");
        self.advanced_params = msg.params;
        
        // Update the advanced GPU context if it exists
        if let Some(ref mut ctx) = self.advanced_gpu_context {
            ctx.update_advanced_params(self.advanced_params.clone());
        }
        
        Ok(())
    }
}

impl Handler<TriggerStressMajorization> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: TriggerStressMajorization, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Manually triggering stress majorization");
        
        match self.perform_stress_majorization() {
            Ok(_) => {
                self.last_stress_majorization = self.iteration_count;
                info!("GPU: Stress majorization completed successfully");
                Ok(())
            },
            Err(e) => {
                error!("GPU: Stress majorization failed: {}", e);
                Err(format!("Stress majorization failed: {}", e))
            }
        }
    }
}


impl Handler<SetComputeMode> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetComputeMode, _ctx: &mut Self::Context) -> Self::Result {
        let old_mode = self.compute_mode;
        
        // Validate that the requested mode is supported
        match msg.mode {
            ComputeMode::Legacy => {
                if self.force_kernel.is_some() {
                    self.compute_mode = msg.mode;
                    info!("GPU: Switched from {:?} to {:?} compute mode", old_mode, msg.mode);
                } else {
                    return Err("Legacy kernel not available".to_string());
                }
            },
            ComputeMode::DualGraph => {
                if self.dual_graph_kernel.is_some() {
                    self.compute_mode = msg.mode;
                    info!("GPU: Switched from {:?} to {:?} compute mode", old_mode, msg.mode);
                } else {
                    return Err("Dual graph kernel not available".to_string());
                }
            },
            ComputeMode::Advanced => {
                if self.advanced_kernel.is_some() || self.advanced_gpu_context.is_some() {
                    self.compute_mode = msg.mode;
                    info!("GPU: Switched from {:?} to {:?} compute mode", old_mode, msg.mode);
                } else {
                    return Err("Advanced kernel not available".to_string());
                }
            },
        }
        
        Ok(())
    }
}

impl Handler<GetPhysicsStats> for GPUComputeActor {
    type Result = Result<PhysicsStats, String>;

    fn handle(&mut self, _msg: GetPhysicsStats, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_physics_stats())
    }
}

impl GPUComputeActor {
    /// Initialize the advanced GPU context for constraint-based physics
    pub async fn initialize_advanced_context(&mut self, num_nodes: u32, num_edges: u32) -> Result<(), Error> {
        info!("Initializing advanced GPU context for {} nodes, {} edges", num_nodes, num_edges);
        
        match AdvancedGPUContext::new(
            num_nodes,
            num_edges,
            self.simulation_params.clone(),
            self.advanced_params.clone(),
        ).await {
            Ok(ctx) => {
                self.advanced_gpu_context = Some(ctx);
                info!("Advanced GPU context initialized successfully");
                Ok(())
            },
            Err(e) => {
                warn!("Failed to initialize advanced GPU context: {}, will use direct kernels", e);
                Err(e)
            }
        }
    }

    /// Convert legacy node data to enhanced format for advanced physics
    fn convert_to_enhanced_nodes(&self, nodes: &[BinaryNodeData]) -> Vec<EnhancedBinaryNodeData> {
        nodes.iter().map(|node| EnhancedBinaryNodeData::from(*node)).collect()
    }

    /// Convert enhanced node data back to legacy format
    fn convert_from_enhanced_nodes(&self, nodes: &[EnhancedBinaryNodeData]) -> Vec<BinaryNodeData> {
        nodes.iter().map(|node| BinaryNodeData::from(*node)).collect()
    }

    /// Get current compute mode as string for logging
    pub fn get_compute_mode_string(&self) -> &'static str {
        match self.compute_mode {
            ComputeMode::Legacy => "Legacy",
            ComputeMode::DualGraph => "DualGraph", 
            ComputeMode::Advanced => "Advanced",
        }
    }

    /// Check if advanced features are available
    pub fn has_advanced_features(&self) -> bool {
        self.advanced_kernel.is_some() || self.advanced_gpu_context.is_some()
    }

    /// Check if dual graph features are available
    pub fn has_dual_graph_features(&self) -> bool {
        self.dual_graph_kernel.is_some()
    }


    /// Add or update an isolation layer
    pub fn add_isolation_layer(&mut self, layer: IsolationLayer) {
        let layer_id = layer.layer_id;
        self.isolation_layers.push(layer);
        info!("Added isolation layer {}, total layers: {}", layer_id, self.isolation_layers.len());
    }

    /// Remove an isolation layer by ID
    pub fn remove_isolation_layer(&mut self, layer_id: i32) -> bool {
        let initial_len = self.isolation_layers.len();
        self.isolation_layers.retain(|layer| layer.layer_id != layer_id);
        let removed = self.isolation_layers.len() < initial_len;
        if removed {
            info!("Removed isolation layer {}", layer_id);
        }
        removed
    }

    /// Update visual analytics parameters
    pub fn update_visual_analytics_params(&mut self, params: VisualAnalyticsParams) {
        self.visual_analytics_params = Some(params);
        info!("Updated visual analytics parameters");
    }

    /// Get current kernel mode as string for logging
    pub fn get_kernel_mode_string(&self) -> &'static str {
        match self.kernel_mode {
            KernelMode::Legacy => "Legacy",
            KernelMode::Advanced => "Advanced",
            KernelMode::VisualAnalytics => "VisualAnalytics",
        }
    }

    /// Check if visual analytics features are available
    pub fn has_visual_analytics_features(&self) -> bool {
        self.visual_analytics_kernel.is_some() || self.visual_analytics_gpu.is_some()
    }

    /// Get statistics for monitoring
    pub fn get_physics_stats(&self) -> PhysicsStats {
        PhysicsStats {
            compute_mode: self.get_compute_mode_string().to_string(),
            kernel_mode: self.get_kernel_mode_string().to_string(),
            iteration_count: self.iteration_count,
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
            num_constraints: self.constraints.len() as u32,
            num_isolation_layers: self.isolation_layers.len() as u32,
            stress_majorization_interval: self.stress_majorization_interval,
            last_stress_majorization: self.last_stress_majorization,
            gpu_failure_count: self.gpu_failure_count,
            cpu_fallback_active: self.cpu_fallback_active,
            has_advanced_features: self.has_advanced_features(),
            has_dual_graph_features: self.has_dual_graph_features(),
            has_visual_analytics_features: self.has_visual_analytics_features(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsStats {
    pub compute_mode: String,
    pub kernel_mode: String,
    pub iteration_count: u32,
    pub num_nodes: u32,
    pub num_edges: u32,
    pub num_constraints: u32,
    pub num_isolation_layers: u32,
    pub stress_majorization_interval: u32,
    pub last_stress_majorization: u32,
    pub gpu_failure_count: u32,
    pub cpu_fallback_active: bool,
    pub has_advanced_features: bool,
    pub has_dual_graph_features: bool,
    pub has_visual_analytics_features: bool,
}

// New message handlers for visual analytics support

impl Handler<InitializeVisualAnalytics> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitializeVisualAnalytics, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: InitializeVisualAnalytics received for {} nodes, {} edges", msg.max_nodes, msg.max_edges);
        
        // For now, just mark as initialized without the async complexity
        // The visual analytics GPU is initialized on-demand when needed
        self.visual_analytics_params = Some(VisualAnalyticsParams {
            // GPU optimization
            total_nodes: msg.max_nodes as i32,
            total_edges: msg.max_edges as i32,
            active_layers: 4,
            hierarchy_depth: 3,
            
            // Temporal dynamics
            current_frame: 0,
            time_step: 0.016,
            temporal_decay: 0.95,
            history_weight: 0.8,
            
            // Force parameters (multi-resolution)
            force_scale: [1.0, 0.8, 0.6, 0.4],
            damping: [0.9, 0.85, 0.8, 0.75],
            temperature: [1.0, 0.8, 0.6, 0.4],
            
            // Isolation and focus
            isolation_strength: 0.5,
            focus_gamma: 1.2,
            primary_focus_node: -1,
            context_alpha: 0.7,
            
            // Visual comprehension
            complexity_threshold: 0.8,
            saliency_boost: 1.5,
            information_bandwidth: 100.0,
            
            // Topology analysis
            community_algorithm: 0,
            modularity_resolution: 1.0,
            topology_update_interval: 10,
            
            // Semantic analysis
            semantic_influence: 0.7,
            drift_threshold: 0.5,
            embedding_dims: 128,
            
            // Viewport
            camera_position: Vec4 { x: 0.0, y: 0.0, z: 500.0, t: 0.0 },
            viewport_bounds: Vec4 { x: -1000.0, y: -1000.0, z: 1000.0, t: 1000.0 },
            zoom_level: 1.0,
            time_window: 10.0,
        });
        
        info!("Visual analytics parameters initialized");
        Ok(())
    }
}

impl Handler<UpdateVisualAnalyticsParams> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateVisualAnalyticsParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Updating visual analytics parameters");
        self.update_visual_analytics_params(msg.params);
        Ok(())
    }
}

impl Handler<AddIsolationLayer> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddIsolationLayer, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Adding isolation layer {}", msg.layer.layer_id);
        self.add_isolation_layer(msg.layer);
        Ok(())
    }
}

impl Handler<RemoveIsolationLayer> for GPUComputeActor {
    type Result = Result<bool, String>;

    fn handle(&mut self, msg: RemoveIsolationLayer, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Removing isolation layer {}", msg.layer_id);
        let removed = self.remove_isolation_layer(msg.layer_id);
        Ok(removed)
    }
}

impl Handler<GetKernelMode> for GPUComputeActor {
    type Result = Result<String, String>;

    fn handle(&mut self, _msg: GetKernelMode, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_kernel_mode_string().to_string())
    }
}
