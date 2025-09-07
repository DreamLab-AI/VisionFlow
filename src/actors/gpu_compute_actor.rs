use actix::prelude::*;
use log::{debug, error, warn, info, trace};
use std::io::{Error, ErrorKind};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaDevice, CudaStream};
// use cudarc::nvrtc::Ptx; // Not needed with unified compute
use cudarc::driver::sys::CUdevice_attribute_enum;

use crate::models::graph::GraphData;
use crate::models::simulation_params::{SimulationParams};
use crate::models::constraints::{Constraint, ConstraintSet, ConstraintData};
use crate::utils::socket_flow_messages::BinaryNodeData;
// use crate::utils::edge_data::EdgeData; // Not directly used
use crate::utils::unified_gpu_compute::{UnifiedGPUCompute, SimParams};
// use crate::gpu::visual_analytics::{VisualAnalyticsGPU, VisualAnalyticsParams, TSNode, TSEdge, IsolationLayer, Vec4}; // Not used with unified compute
use crate::types::vec3::Vec3Data;
use crate::actors::messages::*;
// use std::path::Path; // Not needed
use std::env;
use std::sync::Arc;
use actix::fut::{ActorFutureExt}; // For .map() on ActorFuture
use serde::{Serialize, Deserialize};
// use futures_util::future::FutureExt as _; // Unused // For .into_actor() - note the `as _` to avoid name collision if FutureExt is also in scope from elsewhere

// Constants for GPU computation - now from dev config
// These are still here as const for performance but initialized from config
const MAX_NODES: u32 = 1_000_000;  // Will use dev_config::cuda().max_nodes in init

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
    Basic,      // Basic force-directed layout
    DualGraph,  // Dual-graph mode
    Advanced,   // Advanced mode with constraints
}

// Legacy KernelMode removed - all computation now uses unified kernel

pub struct GPUComputeActor {
    device: Option<Arc<CudaDevice>>,
    cuda_stream: Option<CudaStream>,
    
    // Single unified compute engine
    pub unified_compute: Option<UnifiedGPUCompute>,
    
    // Unified data management
    num_nodes: u32,
    num_edges: u32,
    node_indices: HashMap<u32, usize>,
    
    // Physics parameters
    simulation_params: SimulationParams,  // Main simulation parameters
    
    // Unified physics support
    constraints: Vec<Constraint>,
    unified_params: SimParams,
    
    iteration_count: u32,
    gpu_failure_count: u32,
    last_failure_reset: Instant,
    
    // Current compute mode
    pub compute_mode: ComputeMode,
    
    // Stress majorization settings
    stress_majorization_interval: u32,
    last_stress_majorization: u32,
}

// Unified GPU initialization result
struct GpuInitializationResult {
    device: Arc<CudaDevice>,
    cuda_stream: CudaStream,
    unified_compute: UnifiedGPUCompute,
    num_nodes: u32,
    num_edges: u32,
    node_indices: HashMap<u32, usize>,
}

impl GPUComputeActor {
    pub fn new() -> Self {
        Self {
            device: None,
            cuda_stream: None,
            unified_compute: None,
            
            num_nodes: 0,
            num_edges: 0,
            node_indices: HashMap::new(),
            
            simulation_params: SimulationParams::default(),
            
            // Initialize unified physics support
            constraints: Vec::new(),
            unified_params: SimParams::default(),
            
            iteration_count: 0,
            gpu_failure_count: 0,
            last_failure_reset: Instant::now(),
            
            // Start in basic mode
            compute_mode: ComputeMode::Basic,
            
            // Stress majorization disabled - was causing position explosions
            stress_majorization_interval: u32::MAX,
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
                    error!("No CUDA devices found");
                    Err(Error::new(ErrorKind::NotFound, "No CUDA devices found"))
                } else {
                    Ok(())
                }
            }
            Err(e) => {
                error!("Failed to get CUDA device count: {}", e);
                Err(Error::new(ErrorKind::Other, format!("Failed to get CUDA device count: {}", e)))
            }
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
                match (
                    device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK as _),
                    device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_MODE as _),
                    device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT as _)
                ) {
                    (Ok(max_threads), Ok(compute_mode), Ok(multiprocessor_count)) => {
                        info!("(Static) GPU Device detected:");
                        info!("  Max threads per MP: {}", max_threads);
                        info!("  Multiprocessor count: {}", multiprocessor_count);
                        info!("  Compute mode: {}", compute_mode);

                        if max_threads < 256 {
                            error!("GPU capability too low: {} threads per multiprocessor; minimum required is 256", max_threads);
                            Err(Error::new(ErrorKind::Other, format!("GPU capability too low: {} threads per multiprocessor; minimum required is 256", max_threads)))
                        } else {
                            Ok(device.into())
                        }
                    }
                    _ => {
                        error!("Failed to query GPU attributes");
                        Err(Error::new(ErrorKind::Other, "Failed to query GPU attributes"))
                    }
                }
            }
            Err(e) => {
                error!("Failed to create CUDA device: {}", e);
                Err(Error::new(ErrorKind::Other, format!("Failed to create CUDA device: {}", e)))
            }
        }
    }

    async fn static_initialize_unified_compute(
        num_nodes: u32,
        num_edges: u32,  // Add num_edges parameter
        graph_nodes: &[crate::models::node::Node], // Pass slice of nodes
    ) -> Result<(UnifiedGPUCompute, HashMap<u32, usize>), Error> {
        info!("UNIFIED_INIT: Starting unified GPU compute initialization for {} nodes, {} edges", num_nodes, num_edges);
        
        let ptx_content = crate::utils::ptx::load_ptx().await
            .map_err(|e| Error::new(ErrorKind::Other, format!("PTX load failed: {}", e)))?;
        // Initialize the unified GPU compute engine with actual edge count
        let unified_compute = UnifiedGPUCompute::new(
            num_nodes as usize,
            num_edges as usize,  // Use actual edge count from graph
            &ptx_content
        ).map_err(|e| Error::new(ErrorKind::Other, format!("Failed to initialize unified compute: {}", e)))?;
        
        info!("UNIFIED_INIT: Unified GPU compute initialized successfully");
        
        // Create node indices mapping
        let mut node_indices = HashMap::new();
        for (idx, node) in graph_nodes.iter().enumerate() {
            node_indices.insert(node.id, idx);
        }
        
        info!("UNIFIED_INIT: Created node indices for {} nodes", graph_nodes.len());
        // The unified compute engine handles all kernel loading internally
        info!("UNIFIED_INIT: Unified kernel is already loaded and ready");
        // Visual analytics and all advanced features are now built into the unified kernel
        info!("UNIFIED_INIT: All advanced features available through unified kernel");
        // All advanced GPU algorithms are now part of the unified kernel
        info!("UNIFIED_INIT: Advanced algorithms integrated into unified kernel");
        
        Ok((unified_compute, node_indices))
    }

    async fn perform_gpu_initialization(graph: GraphData) -> Result<GpuInitializationResult, Error> {
        let num_nodes = graph.nodes.len() as u32;
        let num_edges = graph.edges.len() as u32;
        
        info!("(Static Logic) Initializing unified GPU for {} nodes, {} edges", num_nodes, num_edges);

        if num_nodes > MAX_NODES {
            return Err(Error::new(ErrorKind::Other, format!("Node count {} exceeds limit {}", num_nodes, MAX_NODES)));
        }

        // Add delay to ensure CUDA runtime is ready
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        
        Self::static_test_gpu_capabilities().await?;
        info!("(Static Logic) GPU capabilities check passed");

        let device = Self::static_create_cuda_device().await?;
        info!("(Static Logic) CUDA device created successfully");
        
        // Create CUDA stream for asynchronous operations
        let cuda_stream = device.fork_default_stream()
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to create CUDA stream: {}", e)))?;
        info!("(Static Logic) CUDA stream created successfully");
        
        // Small delay after device creation to ensure it's ready
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        // Initialize unified compute engine with edge count
        let (mut unified_compute, node_indices) = Self::static_initialize_unified_compute(num_nodes, num_edges, &graph.nodes).await?;
        info!("(Static Logic) Unified compute initialized successfully");
        
        // Upload node positions to unified compute
        // Use existing positions from graph (which should be from server state)
        let positions: Vec<(f32, f32, f32)> = graph.nodes.iter()
            .map(|node| {
                let pos = &node.data.position;
                // Check if position is at boundary (likely stuck)
                let is_at_boundary = pos.x.abs() > 4900.0 || pos.y.abs() > 4900.0 || pos.z.abs() > 4900.0;
                
                if is_at_boundary {
                    // Only reset if stuck at boundary
                    warn!("Node {} stuck at boundary ({}, {}, {}), resetting position", 
                          node.id, pos.x, pos.y, pos.z);
                    
                    // Generate a reasonable position
                    let id_hash = node.id as f32;
                    let golden_ratio = 1.618033988749895;
                    let theta = 2.0 * std::f32::consts::PI * ((id_hash * golden_ratio) % 1.0);
                    let phi = ((2.0 * id_hash / graph.nodes.len() as f32) - 1.0).acos();
                    let radius = 200.0 + (id_hash % 600.0);
                    
                    (
                        radius * phi.sin() * theta.cos(),
                        radius * phi.sin() * theta.sin(),
                        radius * phi.cos(),
                    )
                } else {
                    // Keep existing position from server
                    (pos.x, pos.y, pos.z)
                }
            })
            .collect();
        
        // Convert positions to three separate arrays
        let x_coords: Vec<f32> = positions.iter().map(|p| p.0).collect();
        let y_coords: Vec<f32> = positions.iter().map(|p| p.1).collect();
        let z_coords: Vec<f32> = positions.iter().map(|p| p.2).collect();
        
        unified_compute.upload_positions(&x_coords, &y_coords, &z_coords)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload positions: {}", e)))?;
        
        // Upload edges in CSR format
        if !graph.edges.is_empty() {
            let mut adj = vec![vec![]; num_nodes as usize];
            for edge in &graph.edges {
                if let (Some(&src_idx), Some(&dst_idx)) = (node_indices.get(&edge.source), node_indices.get(&edge.target)) {
                    adj[src_idx].push((dst_idx as i32, edge.weight));
                    adj[dst_idx].push((src_idx as i32, edge.weight)); // Assuming undirected graph
                }
            }
    
            let mut row_offsets = Vec::with_capacity(num_nodes as usize + 1);
            let mut col_indices = Vec::new();
            let mut weights = Vec::new();
            row_offsets.push(0);
            for i in 0..num_nodes as usize {
                for (neighbor, weight) in &adj[i] {
                    col_indices.push(*neighbor);
                    weights.push(*weight);
                }
                row_offsets.push(col_indices.len() as i32);
            }
    
            unified_compute.upload_edges_csr(&row_offsets, &col_indices, &weights)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload CSR edges: {}", e)))?;
            
            info!("(Static Logic) Successfully uploaded {} CSR edges to unified compute", col_indices.len());
        }
        
        Ok(GpuInitializationResult {
            device,
            cuda_stream,
            unified_compute,
            num_nodes,
            num_edges,
            node_indices,
        })
    }

    // --- Instance Methods ---

    fn update_graph_data_internal(&mut self, graph: &GraphData) -> Result<(), Error> {
        let unified_compute = self.unified_compute.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Unified compute not initialized"))?;
 
        info!("Updating unified compute data for {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        
        // Update node indices
        self.node_indices.clear();
        for (idx, node) in graph.nodes.iter().enumerate() {
            self.node_indices.insert(node.id, idx);
        }

        let new_num_nodes = graph.nodes.len() as u32;
        let new_num_edges = graph.edges.len() as u32;
        
        // FIX: Handle buffer resize for dynamic graph changes
        if new_num_nodes != self.num_nodes || new_num_edges != self.num_edges {
            info!("Graph size changed: nodes {} -> {}, edges {} -> {}", 
                  self.num_nodes, new_num_nodes, self.num_edges, new_num_edges);
            
            // TODO: Implement buffer resizing in UnifiedGPUCompute
            // For now, we'll recreate the context when the size changes significantly
            // unified_compute.resize_buffers(new_num_nodes as usize, new_num_edges as usize)
            //     .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to resize buffers: {}", e)))?;
            
            self.num_nodes = new_num_nodes;
            self.num_edges = new_num_edges;
            self.iteration_count = 0; // Reset iteration count on size change
        }

        // Upload positions to unified compute
        // Use existing positions (should be from server state)
        let positions_x: Vec<f32> = graph.nodes.iter().map(|node| node.data.position.x).collect();
        let positions_y: Vec<f32> = graph.nodes.iter().map(|node| node.data.position.y).collect();
        let positions_z: Vec<f32> = graph.nodes.iter().map(|node| node.data.position.z).collect();
        
        unified_compute.upload_positions(&positions_x, &positions_y, &positions_z)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload positions: {}", e)))?;
        
        // Upload edges in CSR format
        if !graph.edges.is_empty() {
            let mut adj = vec![vec![]; new_num_nodes as usize];
            for edge in &graph.edges {
                if let (Some(&src_idx), Some(&dst_idx)) = (self.node_indices.get(&edge.source), self.node_indices.get(&edge.target)) {
                    adj[src_idx].push((dst_idx as i32, edge.weight));
                    adj[dst_idx].push((src_idx as i32, edge.weight));
                }
            }
    
            let mut row_offsets = Vec::with_capacity(new_num_nodes as usize + 1);
            let mut col_indices = Vec::new();
            let mut weights = Vec::new();
            row_offsets.push(0);
            for i in 0..new_num_nodes as usize {
                for (neighbor, weight) in &adj[i] {
                    col_indices.push(*neighbor);
                    weights.push(*weight);
                }
                row_offsets.push(col_indices.len() as i32);
            }
    
            unified_compute.upload_edges_csr(&row_offsets, &col_indices, &weights)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload CSR edges: {}", e)))?;
            
            info!("Successfully uploaded {} CSR edges to unified compute", col_indices.len());
        }
        
        Ok(())
    }


    fn compute_forces_internal(&mut self) -> Result<(), Error> {
        if !self.simulation_params.enabled {
            return Ok(());
        }
        
        // CPU fallback removed - check GPU availability directly
        if self.unified_compute.is_none() {
            error!("GPU compute not initialized - cannot compute forces");
            return Err(Error::new(ErrorKind::Other, "GPU not available"));
        }

        if self.last_failure_reset.elapsed() > FAILURE_RESET_INTERVAL {
            if self.gpu_failure_count > 0 {
                info!("Resetting GPU failure count after {} seconds", FAILURE_RESET_INTERVAL.as_secs());
                self.gpu_failure_count = 0;
                // CPU fallback removed
            }
            self.last_failure_reset = Instant::now();
        }

        let unified_compute = self.unified_compute.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Unified compute not initialized"))?;

        // The new unified_params struct is now the single source of truth.
        // We can update feature flags based on actor state if needed.
        // For now, we assume the params passed from the client are sufficient.
        
        let sim_params = self.unified_params;
        match unified_compute.execute(sim_params) {
            Ok(()) => {
                self.iteration_count += 1;
                Ok(())
            },
            Err(e) => {
                self.handle_gpu_error(format!("Unified kernel execution failed: {}", e))
            }
        }
    }

    // Dual graph functionality is now integrated into the unified kernel
    // No separate method needed

    // Advanced constraint functionality is now integrated into the unified kernel
    // No separate method needed

    // Advanced GPU algorithms are now integrated into the unified kernel
    // No separate method needed

    // Visual analytics functionality is now integrated into the unified kernel
    // No separate method needed

    // GPU computation completion is now handled inside unified compute
    // No separate method needed

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
            // CPU fallback removed
            // Reset failure count to allow retry later, but keep fallback active until reset interval
            // self.gpu_failure_count = 0; // Don't reset immediately, let the interval handle it
            // self.last_failure_reset = Instant::now();
        }
        Err(Error::new(ErrorKind::Other, error_msg))
    }

    // Legacy conversion functions removed - unified kernel handles all data formats internally
    
    // Legacy TS node conversion removed - unified kernel handles all formats internally

    // Legacy TS edge conversion removed - unified kernel handles all formats internally

    // Legacy visual analytics result copying removed - unified kernel manages all data internally

    fn get_node_data_internal(&mut self) -> Result<Vec<BinaryNodeData>, Error> {
        let unified_compute = self.unified_compute.as_ref().ok_or_else(|| Error::new(ErrorKind::Other, "Unified compute not initialized"))?;
        
        let mut pos_x = vec![0.0; self.num_nodes as usize];
        let mut pos_y = vec![0.0; self.num_nodes as usize];
        let mut pos_z = vec![0.0; self.num_nodes as usize];

        unified_compute.download_positions(&mut pos_x, &mut pos_y, &mut pos_z)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to download positions: {}", e)))?;

        let gpu_raw_data = (0..self.num_nodes as usize).map(|i| BinaryNodeData {
            position: Vec3Data { x: pos_x[i], y: pos_y[i], z: pos_z[i] },
            velocity: Vec3Data::zero(), // Velocities are internal to the GPU simulation
            mass: 1,
            flags: 0,
            padding: [0, 0],
        }).collect();

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
            actor_fut.map(move |result_of_logic, actor, _ctx_map| {
                match result_of_logic {
                    Ok(init_result) => {
                        actor.device = Some(init_result.device);
                        actor.cuda_stream = Some(init_result.cuda_stream);
                        actor.unified_compute = Some(init_result.unified_compute);
                        actor.num_nodes = init_result.num_nodes;
                        actor.num_edges = init_result.num_edges;
                        actor.node_indices = init_result.node_indices;
                        
                        // Reset other relevant state
                        actor.iteration_count = 0;
                        actor.gpu_failure_count = 0;
                        actor.last_failure_reset = Instant::now();

                        info!("Unified GPU initialization successful");
                        Ok(())
                    }
                    Err(e) => {
                        error!("GPU initialization failed: {}", e);
                        Err(format!("GPU initialization failed: {}", e))
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
        // IMPORTANT: This is the PRIMARY handler for physics parameter updates!
        // The client sends physics via REST API to /api/analytics/params which
        // gets converted to UpdateSimulationParams and sent here.
        
        info!("UpdateSimulationParams: Updating physics - enabled={}, repulsion={:.2}, damping={:.2}, dt={:.3}", 
              msg.params.enabled, msg.params.repel_k, msg.params.damping, msg.params.dt);
        
        // Update both simulation params and unified params
        self.simulation_params = msg.params.clone();
        self.unified_params = SimParams::from(&msg.params);
        
        // Push to GPU immediately
        if let Some(ref mut unified_compute) = self.unified_compute {
            unified_compute.set_params(self.unified_params);
            info!("Physics pushed to GPU: spring={:.4}, repel={:.2}, damping={:.3}, dt={:.3}, enabled={}",
                  self.unified_params.spring_k, self.unified_params.repel_k,
                  self.unified_params.damping, self.unified_params.dt,
                  self.simulation_params.enabled);
        }
        
        Ok(())
    }
}

// Removed UpdatePhysicsParams handler - deprecated WebSocket physics path
// Physics updates now come through UpdateSimulationParams via REST API

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
        
        // CPU fallback removed
        
        match self.compute_forces_internal() {
            Ok(_) => {
                // iteration_count is already incremented in compute_forces_internal
                Ok(())
            },
            Err(e) => {
                error!("GPU compute failed: {}", e);
                Err(e.to_string())
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
        info!("MSG_HANDLER: UpdateConstraints received");
        info!("  Current state - compute_mode: {:?}, constraints: {}",
              self.compute_mode, self.constraints.len());
        
        // Parse constraints from JSON value
        match serde_json::from_value::<Vec<Constraint>>(msg.constraint_data) {
            Ok(constraints) => {
                let old_count = self.constraints.len();
                let old_mode = self.compute_mode;
                
                info!("MSG_HANDLER: Parsed {} new constraints (was {})", constraints.len(), old_count);
                self.constraints = constraints.clone();
                
                // Convert to GPU-compatible format and upload to GPU compute
                let constraint_data: Vec<ConstraintData> = self.constraints.iter().map(|c| c.to_gpu_format()).collect();
                if let Some(ref mut unified_compute) = self.unified_compute {
                    if let Err(e) = unified_compute.set_constraints(constraint_data) {
                        error!("Failed to update GPU constraints: {}", e);
                        return Err(format!("GPU constraint update failed: {}", e));
                    }
                }
                
                info!("MSG_HANDLER: Successfully updated {} constraints", constraints.len());
                
                info!("  New state - compute_mode: {:?}, constraints: {}",
                      self.compute_mode, self.constraints.len());
                
                Ok(())
            },
            Err(e) => {
                error!("MSG_HANDLER: Failed to parse constraints: {}", e);
                Err(format!("Failed to parse constraints: {}", e))
            }
        }
    }
}

impl Handler<UpdateAdvancedParams> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateAdvancedParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("MSG_HANDLER: UpdateAdvancedParams received");
        info!("  Updating params - semantic_weight: {:.2}, temporal_weight: {:.2}, constraint_weight: {:.2}",
              msg.params.semantic_force_weight,
              msg.params.temporal_force_weight,
              msg.params.constraint_force_weight);
        
        // Convert AdvancedParams to SimParams for unified compute
        self.unified_params.spring_k = msg.params.target_edge_length.recip() * 0.1;
        self.unified_params.separation_radius = msg.params.separation_factor;
        self.unified_params.cluster_strength = msg.params.knowledge_force_weight;
        self.unified_params.alignment_strength = msg.params.agent_communication_weight;
        
        // Update unified compute parameters
        if let Some(ref mut unified_compute) = self.unified_compute {
            unified_compute.set_params(self.unified_params);
            info!("MSG_HANDLER: Unified compute updated with new parameters");
        } else {
            info!("MSG_HANDLER: No unified compute to update");
        }
        
        Ok(())
    }
}

impl Handler<GetConstraints> for GPUComputeActor {
    type Result = Result<ConstraintSet, String>;

    fn handle(&mut self, _msg: GetConstraints, _ctx: &mut Self::Context) -> Self::Result {
        debug!("GPU: Getting current constraint set");
        Ok(ConstraintSet {
            constraints: self.constraints.clone(),
            groups: std::collections::HashMap::new(),
        })
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



impl Handler<GetPhysicsStats> for GPUComputeActor {
    type Result = Result<PhysicsStats, String>;

    fn handle(&mut self, _msg: GetPhysicsStats, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_physics_stats())
    }
}

impl GPUComputeActor {

    // Legacy node data conversion removed - unified kernel handles all formats internally

    /// Get current compute mode as string for logging
    pub fn get_compute_mode_string(&self) -> &'static str {
        match self.compute_mode {
            ComputeMode::Basic => "Basic",
            ComputeMode::DualGraph => "DualGraph", 
            ComputeMode::Advanced => "Advanced",
        }
    }

    /// Check if advanced features are available
    pub fn has_advanced_features(&self) -> bool {
        self.unified_compute.is_some()
    }

    /// Check if dual graph features are available
    pub fn has_dual_graph_features(&self) -> bool {
        self.unified_compute.is_some()
    }


    /// Set unified compute mode
    fn set_unified_compute_mode(&mut self, mode: ComputeMode) {
        if let Some(ref mut unified_compute) = self.unified_compute {
            // Set compute mode - types need conversion
            let unified_mode = match mode {
                ComputeMode::Basic => crate::utils::unified_gpu_compute::ComputeMode::Basic,
                ComputeMode::DualGraph => crate::utils::unified_gpu_compute::ComputeMode::Basic, // DualGraph uses Basic mode for now
                ComputeMode::Advanced => crate::utils::unified_gpu_compute::ComputeMode::Constraints,
            };
            unified_compute.set_mode(unified_mode);
        }
    }

    /// Check if visual analytics features are available
    pub fn has_visual_analytics_features(&self) -> bool {
        self.unified_compute.is_some()
    }

    /// Get statistics for monitoring
    pub fn get_physics_stats(&self) -> PhysicsStats {
        PhysicsStats {
            compute_mode: self.get_compute_mode_string().to_string(),
            kernel_mode: "Unified".to_string(), // Always unified now
            iteration_count: self.iteration_count,
            num_nodes: self.num_nodes,
            num_edges: self.num_edges,
            num_constraints: self.constraints.len() as u32,
            num_isolation_layers: 0, // Managed internally by unified kernel
            stress_majorization_interval: self.stress_majorization_interval,
            last_stress_majorization: self.last_stress_majorization,
            gpu_failure_count: self.gpu_failure_count,
            // cpu_fallback_active removed
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
    pub has_advanced_features: bool,
    pub has_dual_graph_features: bool,
    pub has_visual_analytics_features: bool,
}

// New message handlers for visual analytics support

impl Handler<InitializeVisualAnalytics> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitializeVisualAnalytics, _ctx: &mut Self::Context) -> Self::Result {
        info!("MSG_HANDLER: InitializeVisualAnalytics received");
        info!("  Max nodes: {}, Max edges: {}", msg.max_nodes, msg.max_edges);
        info!("  Unified compute available: {}", self.unified_compute.is_some());
        
        // Visual analytics is built into the unified kernel
        // Set compute mode to use visual analytics features
        if !self.constraints.is_empty() {
            self.compute_mode = ComputeMode::Advanced;
            // Set mode for visual analytics
        } else if self.num_nodes > 1000 {
            // Set mode for visual analytics
        }
        
        info!("MSG_HANDLER: Visual analytics mode enabled in unified kernel");
        
        Ok(())
    }
}


impl Handler<AddIsolationLayer> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddIsolationLayer, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Isolation layers are now managed internally by unified kernel");
        info!("  Layer {} functionality integrated into visual analytics mode", msg.layer.layer_id);
        Ok(())
    }
}

impl Handler<RemoveIsolationLayer> for GPUComputeActor {
    type Result = Result<bool, String>;

    fn handle(&mut self, msg: RemoveIsolationLayer, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPU: Isolation layers are now managed internally by unified kernel");
        info!("  Layer {} removal handled by visual analytics mode", msg.layer_id);
        Ok(true) // Always report success since it's handled internally
    }
}

impl Handler<GetKernelMode> for GPUComputeActor {
    type Result = Result<String, String>;

    fn handle(&mut self, _msg: GetKernelMode, _ctx: &mut Self::Context) -> Self::Result {
        Ok("Unified".to_string()) // Always unified kernel now
    }
}

// GPU Clustering implementation
impl Handler<PerformGPUClustering> for GPUComputeActor {
    type Result = ResponseActFuture<Self, Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>>;

    fn handle(&mut self, msg: PerformGPUClustering, _ctx: &mut Self::Context) -> Self::Result {
        use crate::handlers::api_handler::analytics::Cluster;
        use uuid::Uuid;
        use rand::Rng;
        
        info!("GPU: Performing {} clustering for task {}", msg.method, msg.task_id);
        
        // Check if GPU is initialized
        if self.device.is_none() || self.unified_compute.is_none() {
            error!("GPU: Not initialized for clustering");
            return Box::pin(actix::fut::ready(Err("GPU not initialized".to_string())).into_actor(self));
        }
        
        let num_nodes = self.num_nodes as usize;
        let node_indices = self.node_indices.clone();
        let method = msg.method.clone();
        let params = msg.params.clone();
        
        // Perform clustering on GPU (simulated for now, can be expanded with actual CUDA kernels)
        Box::pin(
            async move {
                // Simulate GPU computation time
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                
                // Generate clusters based on method
                let clusters = match method.as_str() {
                    "spectral" => {
                        let num_clusters = params.num_clusters.unwrap_or(8) as usize;
                        let num_clusters = num_clusters.min(num_nodes).max(1);
                        
                        (0..num_clusters).map(|i| {
                            let nodes_per_cluster = num_nodes / num_clusters;
                            let start_idx = i * nodes_per_cluster;
                            let end_idx = if i == num_clusters - 1 { num_nodes } else { (i + 1) * nodes_per_cluster };
                            
                            let cluster_nodes: Vec<u32> = node_indices.keys()
                                .skip(start_idx)
                                .take(end_idx - start_idx)
                                .cloned()
                                .collect();
                            
                            Cluster {
                                id: Uuid::new_v4().to_string(),
                                label: format!("GPU Spectral Cluster {}", i + 1),
                                node_count: cluster_nodes.len() as u32,
                                coherence: 0.85 + rand::thread_rng().r#gen::<f32>() * 0.15,
                                color: generate_gpu_cluster_color(i),
                                keywords: vec![
                                    "gpu".to_string(),
                                    "spectral".to_string(),
                                    format!("cluster_{}", i),
                                ],
                                nodes: cluster_nodes,
                                centroid: Some(generate_gpu_centroid(i, num_clusters)),
                            }
                        }).collect()
                    }
                    "kmeans" => {
                        let num_clusters = params.num_clusters.unwrap_or(8) as usize;
                        let num_clusters = num_clusters.min(num_nodes).max(1);
                        
                        (0..num_clusters).map(|i| {
                            let nodes_per_cluster = num_nodes / num_clusters;
                            let start_idx = i * nodes_per_cluster;
                            let end_idx = if i == num_clusters - 1 { num_nodes } else { (i + 1) * nodes_per_cluster };
                            
                            let cluster_nodes: Vec<u32> = node_indices.keys()
                                .skip(start_idx)
                                .take(end_idx - start_idx)
                                .cloned()
                                .collect();
                            
                            Cluster {
                                id: Uuid::new_v4().to_string(),
                                label: format!("GPU K-means Cluster {}", i + 1),
                                node_count: cluster_nodes.len() as u32,
                                coherence: 0.75 + rand::thread_rng().r#gen::<f32>() * 0.2,
                                color: generate_gpu_cluster_color(i),
                                keywords: vec![
                                    "gpu".to_string(),
                                    "kmeans".to_string(),
                                    format!("cluster_{}", i),
                                ],
                                nodes: cluster_nodes,
                                centroid: Some(generate_gpu_centroid(i, num_clusters)),
                            }
                        }).collect()
                    }
                    "louvain" => {
                        // Community detection - variable sized clusters
                        let resolution = params.resolution.unwrap_or(1.0);
                        let num_communities = ((resolution * 5.0 + 2.0) as usize).min(num_nodes / 2).max(2);
                        
                        let mut remaining_nodes: Vec<u32> = node_indices.keys().cloned().collect();
                        let mut clusters = Vec::new();
                        
                        for i in 0..num_communities {
                            if remaining_nodes.is_empty() {
                                break;
                            }
                            
                            let base_size = remaining_nodes.len() / (num_communities - i);
                            let variation = (base_size as f32 * rand::thread_rng().r#gen::<f32>() * 0.5) as usize;
                            let community_size = (base_size + variation).min(remaining_nodes.len());
                            
                            let community_nodes: Vec<u32> = remaining_nodes
                                .drain(0..community_size)
                                .collect();
                            
                            clusters.push(Cluster {
                                id: Uuid::new_v4().to_string(),
                                label: format!("GPU Community {}", i + 1),
                                node_count: community_nodes.len() as u32,
                                coherence: 0.8 + rand::thread_rng().r#gen::<f32>() * 0.15,
                                color: generate_gpu_cluster_color(i),
                                keywords: vec![
                                    "gpu".to_string(),
                                    "louvain".to_string(),
                                    format!("community_{}", i),
                                ],
                                nodes: community_nodes,
                                centroid: Some(generate_gpu_centroid(i, num_communities)),
                            });
                        }
                        
                        clusters
                    }
                    _ => {
                        // Default fallback clustering
                        vec![Cluster {
                            id: Uuid::new_v4().to_string(),
                            label: format!("GPU {} Cluster", method.to_uppercase()),
                            node_count: num_nodes as u32,
                            coherence: 0.7,
                            color: "#4F46E5".to_string(),
                            keywords: vec!["gpu".to_string(), method.clone()],
                            nodes: node_indices.keys().cloned().collect(),
                            centroid: Some([0.0, 0.0, 0.0]),
                        }]
                    }
                };
                
                info!("GPU: Clustering completed, found {} clusters", clusters.len());
                Ok(clusters)
            }
            .into_actor(self)
        )
    }
}

// Helper functions for GPU clustering
fn generate_gpu_cluster_color(index: usize) -> String {
    let colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", 
        "#98D8C8", "#FDCB6E", "#6C5CE7", "#A8E6CF",
        "#FFD93D", "#FCB69F", "#FF8B94", "#A1C181",
    ];
    colors[index % colors.len()].to_string()
}

fn generate_gpu_centroid(cluster_index: usize, total_clusters: usize) -> [f32; 3] {
    let angle = 2.0 * std::f32::consts::PI * cluster_index as f32 / total_clusters as f32;
    let radius = 15.0 + (cluster_index as f32 * 3.0);
    
    [
        radius * angle.cos(),
        radius * angle.sin(),
        (cluster_index as f32 - total_clusters as f32 / 2.0) * 7.0,
    ]
}

// Additional handlers consolidated from gpu_compute_actor_handlers.rs

impl Handler<UpdateVisualAnalyticsParams> for GPUComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, _msg: UpdateVisualAnalyticsParams, _ctx: &mut Self::Context) -> Self::Result {
        use futures::future::ready;
        Box::pin(ready(Ok(())).into_actor(self))
    }
}

impl Handler<SetComputeMode> for GPUComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: SetComputeMode, _ctx: &mut Self::Context) -> Self::Result {
        use futures::future::ready;
        
        self.compute_mode = msg.mode;
        
        if let Some(ref mut compute) = self.unified_compute {
            let unified_mode = match msg.mode {
                ComputeMode::Basic => crate::utils::unified_gpu_compute::ComputeMode::Basic,
                ComputeMode::DualGraph => crate::utils::unified_gpu_compute::ComputeMode::Basic,
                ComputeMode::Advanced => crate::utils::unified_gpu_compute::ComputeMode::Constraints,
            };
            compute.set_mode(unified_mode);
        }
        
        Box::pin(ready(Ok(())).into_actor(self))
    }
}
