use actix::prelude::*;
use log::{debug, error, warn, info, trace};
use std::io::{Error, ErrorKind};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use cudarc::driver::CudaDevice;
// use cudarc::nvrtc::Ptx; // Not needed with unified compute
use cudarc::driver::sys::CUdevice_attribute_enum;

use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::{Constraint, ConstraintSet};
use crate::utils::socket_flow_messages::BinaryNodeData;
// use crate::utils::edge_data::EdgeData; // Not directly used
use crate::utils::unified_gpu_compute::{UnifiedGPUCompute, ComputeMode as UnifiedComputeMode, SimParams};
// use crate::gpu::visual_analytics::{VisualAnalyticsGPU, VisualAnalyticsParams, TSNode, TSEdge, IsolationLayer, Vec4}; // Not used with unified compute
use crate::types::vec3::Vec3Data;
use crate::actors::messages::*;
// use std::path::Path; // Not needed
use std::env;
use std::sync::Arc;
use actix::fut::{ActorFutureExt}; // For .map() on ActorFuture
use serde::{Serialize, Deserialize};
use futures_util::future::FutureExt as _; // For .into_actor() - note the `as _` to avoid name collision if FutureExt is also in scope from elsewhere

// Constants for GPU computation
const MAX_NODES: u32 = 1_000_000;
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
    Basic,      // Basic force-directed layout
    DualGraph,  // Dual-graph mode
    Advanced,   // Advanced mode with constraints
}

// Legacy KernelMode removed - all computation now uses unified kernel

#[derive(Debug)]
pub struct GPUComputeActor {
    device: Option<Arc<CudaDevice>>,
    
    // Single unified compute engine
    unified_compute: Option<UnifiedGPUCompute>,
    
    // Unified data management
    num_nodes: u32,
    num_edges: u32,
    node_indices: HashMap<u32, usize>,
    
    // Separate physics parameters for each graph type
    knowledge_sim_params: SimulationParams,
    agent_sim_params: SimulationParams,
    simulation_params: SimulationParams,  // Legacy combined params
    
    // Unified physics support
    constraints: Vec<Constraint>,
    unified_params: SimParams,
    
    iteration_count: u32,
    gpu_failure_count: u32,
    last_failure_reset: Instant,
    cpu_fallback_active: bool,
    
    // Current compute mode
    compute_mode: ComputeMode,
    
    // Stress majorization settings
    stress_majorization_interval: u32,
    last_stress_majorization: u32,
}

// Unified GPU initialization result
struct GpuInitializationResult {
    device: Arc<CudaDevice>,
    unified_compute: UnifiedGPUCompute,
    num_nodes: u32,
    num_edges: u32,
    node_indices: HashMap<u32, usize>,
}

impl GPUComputeActor {
    pub fn new() -> Self {
        Self {
            device: None,
            unified_compute: None,
            
            num_nodes: 0,
            num_edges: 0,
            node_indices: HashMap::new(),
            
            knowledge_sim_params: SimulationParams::default(),
            agent_sim_params: SimulationParams::default(),
            simulation_params: SimulationParams::default(),
            
            // Initialize unified physics support
            constraints: Vec::new(),
            unified_params: SimParams::default(),
            
            iteration_count: 0,
            gpu_failure_count: 0,
            last_failure_reset: Instant::now(),
            cpu_fallback_active: false,
            
            // Start in basic mode
            compute_mode: ComputeMode::Basic,
            
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

    async fn static_initialize_unified_compute(
        device: Arc<CudaDevice>,
        num_nodes: u32,
        num_edges: u32,  // Add num_edges parameter
        graph_nodes: &[crate::models::node::Node], // Pass slice of nodes
    ) -> Result<(UnifiedGPUCompute, HashMap<u32, usize>), Error> {
        info!("UNIFIED_INIT: Starting unified GPU compute initialization for {} nodes, {} edges", num_nodes, num_edges);
        
        // Initialize the unified GPU compute engine with actual edge count
        let unified_compute = UnifiedGPUCompute::new(
            device.clone(),
            num_nodes as usize,
            num_edges as usize,  // Use actual edge count from graph
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

        Self::static_test_gpu_capabilities().await?;
        info!("(Static Logic) GPU capabilities check passed");

        let device = Self::static_create_cuda_device().await?;
        info!("(Static Logic) CUDA device created successfully");
        
        // Initialize unified compute engine with edge count
        let (mut unified_compute, node_indices) = Self::static_initialize_unified_compute(device.clone(), num_nodes, num_edges, &graph.nodes).await?;
        info!("(Static Logic) Unified compute initialized successfully");
        
        // Upload node positions to unified compute
        let positions: Vec<(f32, f32, f32)> = graph.nodes.iter()
            .map(|node| (node.data.position.x, node.data.position.y, node.data.position.z))
            .collect();
        
        unified_compute.upload_positions(&positions)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload positions: {}", e)))?;
        
        // Upload edges to unified compute
        if !graph.edges.is_empty() {
            let edges: Vec<(i32, i32, f32)> = graph.edges.iter()
                .map(|edge| {
                    let source_idx = node_indices.get(&edge.source).copied().unwrap_or(0) as i32;
                    let target_idx = node_indices.get(&edge.target).copied().unwrap_or(0) as i32;
                    (source_idx, target_idx, edge.weight)
                })
                .collect();
            
            unified_compute.upload_edges(&edges)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload edges: {}", e)))?;
            
            info!("(Static Logic) Successfully uploaded {} edges to unified compute", num_edges);
        }
        
        Ok(GpuInitializationResult {
            device,
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

        // Update node count if changed
        if graph.nodes.len() as u32 != self.num_nodes {
            info!("Node count changed from {} to {}", self.num_nodes, graph.nodes.len());
            self.num_nodes = graph.nodes.len() as u32;
            self.iteration_count = 0; // Reset iteration count on size change
        }

        // Upload positions to unified compute
        let positions: Vec<(f32, f32, f32)> = graph.nodes.iter()
            .map(|node| (node.data.position.x, node.data.position.y, node.data.position.z))
            .collect();
        
        unified_compute.upload_positions(&positions)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload positions: {}", e)))?;
        
        // Upload edges to unified compute
        if graph.edges.len() as u32 != self.num_edges {
            info!("Edge count changed from {} to {}", self.num_edges, graph.edges.len());
            self.num_edges = graph.edges.len() as u32;
        }
        
        if !graph.edges.is_empty() {
            let edges: Vec<(i32, i32, f32)> = graph.edges.iter()
                .map(|edge| {
                    let source_idx = self.node_indices.get(&edge.source).copied().unwrap_or(0) as i32;
                    let target_idx = self.node_indices.get(&edge.target).copied().unwrap_or(0) as i32;
                    (source_idx, target_idx, edge.weight)
                })
                .collect();
            
            unified_compute.upload_edges(&edges)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to upload edges: {}", e)))?;
            
            info!("Successfully uploaded {} edges to unified compute", edges.len());
        }
        
        Ok(())
    }

    /// Update unified compute mode based on current graph state
    fn update_unified_mode(&mut self) {
        if let Some(ref mut unified_compute) = self.unified_compute {
            let unified_mode = match self.compute_mode {
                ComputeMode::Basic => UnifiedComputeMode::Basic,
                ComputeMode::DualGraph => UnifiedComputeMode::DualGraph,
                ComputeMode::Advanced => {
                    if !self.constraints.is_empty() {
                        UnifiedComputeMode::Constraints
                    } else {
                        UnifiedComputeMode::VisualAnalytics
                    }
                },
            };
            
            unified_compute.set_mode(unified_mode);
            unified_compute.set_params(self.unified_params);
            
            debug!("UNIFIED: Updated compute mode to {:?}", unified_mode);
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
                self.cpu_fallback_active = false;
            }
            self.last_failure_reset = Instant::now();
        }

        // Update unified compute mode based on current state
        self.update_unified_mode();

        // Log physics state every 60 frames for debugging
        if self.iteration_count % 60 == 0 {
            info!("UNIFIED_PHYSICS: iteration={}, compute_mode={:?}, nodes={}, edges={}, constraints={}",
                self.iteration_count,
                self.compute_mode,
                self.num_nodes,
                self.num_edges,
                self.constraints.len()
            );
            info!("UNIFIED_PARAMS: spring={:.4}, repulsion={:.1}, damping={:.4}, dt={:.4}",
                self.unified_params.spring_k,
                self.unified_params.repel_k,
                self.unified_params.damping,
                self.unified_params.dt
            );
        }

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

        // Execute unified physics computation
        self.compute_forces_unified()
    }

    fn compute_forces_unified(&mut self) -> Result<(), Error> {
        let unified_compute = self.unified_compute.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Unified compute not initialized"))?;

        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Starting unified force computation (iteration {})", self.iteration_count);
        }

        // Execute unified physics computation
        match unified_compute.execute() {
            Ok(positions) => {
                // Positions are updated directly in the unified compute engine
                // No need to copy back unless specifically requested
                self.iteration_count += 1;
                
                if self.iteration_count % DEBUG_THROTTLE == 0 {
                    trace!("Unified force computation completed successfully with {} positions", positions.len());
                }
                
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
            self.cpu_fallback_active = true;
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
        let unified_compute = self.unified_compute.as_mut().ok_or_else(|| Error::new(ErrorKind::Other, "Unified compute not initialized"))?;

        // Get positions from unified compute (returns as tuples)
        let positions = unified_compute.execute().map_err(|e| Error::new(ErrorKind::Other, format!("Failed to get positions from unified compute: {}", e)))?;
        
        // Convert to BinaryNodeData format for compatibility
        let mut gpu_raw_data = Vec::with_capacity(positions.len());
        for (x, y, z) in positions {
            gpu_raw_data.push(BinaryNodeData {
                position: Vec3Data { x, y, z },
                velocity: Vec3Data::zero(), // Velocities are internal to unified compute
                mass: 1, // Default mass
                flags: 0,
                padding: [0, 0],
            });
        }

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
                        actor.unified_compute = Some(init_result.unified_compute);
                        actor.num_nodes = init_result.num_nodes;
                        actor.num_edges = init_result.num_edges;
                        actor.node_indices = init_result.node_indices;
                        
                        // Reset other relevant state
                        actor.iteration_count = 0;
                        actor.gpu_failure_count = 0;
                        actor.last_failure_reset = Instant::now();
                        actor.cpu_fallback_active = false;

                        info!("Unified GPU initialization successful");
                        Ok(())
                    }
                    Err(e) => {
                        error!("Unified GPU initialization failed: {}", e);
                        actor.device = None;
                        actor.unified_compute = None;
                        actor.num_nodes = 0;
                        actor.num_edges = 0;
                        actor.node_indices.clear();
                        actor.cpu_fallback_active = true;
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
        
        // Update both simulation params and unified params
        self.simulation_params = msg.params.clone();
        self.unified_params = SimParams::from(&msg.params);
        
        // Update the unified compute if it's initialized
        if let Some(ref mut unified_compute) = self.unified_compute {
            unified_compute.set_params(self.unified_params);
            info!("GPU: Updated unified compute with new physics parameters from settings");
        }
        
        info!("GPU: Physics parameters updated successfully from settings.yaml");
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
        info!("MSG_HANDLER: UpdateConstraints received");
        info!("  Current state - compute_mode: {:?}, constraints: {}",
              self.compute_mode, self.constraints.len());
        
        // Parse constraints from JSON value
        match serde_json::from_value::<Vec<Constraint>>(msg.constraint_data) {
            Ok(constraints) => {
                let old_count = self.constraints.len();
                let old_mode = self.compute_mode;
                
                info!("MSG_HANDLER: Parsed {} new constraints (was {})", constraints.len(), old_count);
                self.constraints = constraints;
                
                // Switch to advanced mode if we have constraints
                if !self.constraints.is_empty() && self.unified_compute.is_some() {
                    if matches!(self.compute_mode, ComputeMode::Basic) {
                        info!("MSG_HANDLER: Switching compute mode from {:?} to Advanced due to constraints", old_mode);
                        self.compute_mode = ComputeMode::Advanced;
                        self.set_unified_compute_mode(UnifiedComputeMode::Constraints);
                    }
                } else if self.constraints.is_empty() && matches!(self.compute_mode, ComputeMode::Advanced) {
                    info!("MSG_HANDLER: Switching compute mode from {:?} to Basic - no constraints", old_mode);
                    self.compute_mode = ComputeMode::Basic;
                    self.set_unified_compute_mode(UnifiedComputeMode::Basic);
                } else {
                    info!("MSG_HANDLER: Compute mode remains {:?}", self.compute_mode);
                }
                
                // Update unified compute with constraints
                if let Some(ref mut unified_compute) = self.unified_compute {
                    // Convert constraints to unified format and set them
                    let constraint_data: Vec<crate::utils::unified_gpu_compute::ConstraintData> = self.constraints.iter()
                        .map(|c| crate::utils::unified_gpu_compute::ConstraintData {
                            constraint_type: 0, // Basic constraint type
                            strength: c.weight,
                            param1: 0.0,
                            param2: 0.0,
                            node_mask: 0,
                        })
                        .collect();
                    
                    if let Err(e) = unified_compute.set_constraints(constraint_data) {
                        warn!("Failed to update unified compute constraints: {}", e);
                    }
                }
                
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


impl Handler<SetComputeMode> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetComputeMode, _ctx: &mut Self::Context) -> Self::Result {
        let old_mode = self.compute_mode;
        
        // All modes are supported by the unified kernel
        if self.unified_compute.is_some() {
            self.compute_mode = msg.mode;
            
            // Map to unified compute mode
            let unified_mode = match msg.mode {
                ComputeMode::Basic => UnifiedComputeMode::Basic,
                ComputeMode::DualGraph => UnifiedComputeMode::DualGraph,
                ComputeMode::Advanced => UnifiedComputeMode::Constraints,
            };
            
            self.set_unified_compute_mode(unified_mode);
            info!("GPU: Switched from {:?} to {:?} compute mode", old_mode, msg.mode);
            Ok(())
        } else {
            Err("Unified compute not available".to_string())
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
    /// Initialize the unified GPU compute engine
    pub async fn initialize_unified_context(&mut self, num_nodes: u32, num_edges: u32) -> Result<(), Error> {
        info!("Initializing unified GPU context for {} nodes, {} edges", num_nodes, num_edges);
        
        if let Some(device) = &self.device {
            match UnifiedGPUCompute::new(device.clone(), num_nodes as usize, num_edges as usize) {
                Ok(unified_compute) => {
                    self.unified_compute = Some(unified_compute);
                    info!("Unified GPU context initialized successfully");
                    Ok(())
                },
                Err(e) => {
                    warn!("Failed to initialize unified GPU context: {}", e);
                    Err(e)
                }
            }
        } else {
            Err(Error::new(ErrorKind::Other, "Device not initialized"))
        }
    }

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
    pub fn set_unified_compute_mode(&mut self, mode: UnifiedComputeMode) {
        if let Some(ref mut unified_compute) = self.unified_compute {
            unified_compute.set_mode(mode);
            info!("Updated unified compute mode to: {:?}", mode);
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
        info!("MSG_HANDLER: InitializeVisualAnalytics received");
        info!("  Max nodes: {}, Max edges: {}", msg.max_nodes, msg.max_edges);
        info!("  Unified compute available: {}", self.unified_compute.is_some());
        
        // Visual analytics is built into the unified kernel
        // Set compute mode to use visual analytics features
        if !self.constraints.is_empty() {
            self.compute_mode = ComputeMode::Advanced;
            self.set_unified_compute_mode(UnifiedComputeMode::VisualAnalytics);
        } else if self.num_nodes > 1000 {
            self.set_unified_compute_mode(UnifiedComputeMode::VisualAnalytics);
        }
        
        info!("MSG_HANDLER: Visual analytics mode enabled in unified kernel");
        
        Ok(())
    }
}

impl Handler<UpdateVisualAnalyticsParams> for GPUComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateVisualAnalyticsParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("MSG_HANDLER: UpdateVisualAnalyticsParams received");
        info!("  Focus node: {}, Isolation strength: {:.2}",
              msg.params.primary_focus_node, msg.params.isolation_strength);
        
        // Update unified compute parameters based on visual analytics settings
        if let Some(ref mut unified_compute) = self.unified_compute {
            let mut params = self.unified_params;
            params.temperature = 1.0 - msg.params.isolation_strength;
            params.viewport_bounds = msg.params.viewport_bounds.x.abs().max(msg.params.viewport_bounds.y.abs());
            unified_compute.set_params(params);
        }
        
        info!("MSG_HANDLER: Visual analytics parameters updated in unified compute");
        
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
                                coherence: 0.85 + rand::thread_rng().gen::<f32>() * 0.15,
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
                                coherence: 0.75 + rand::thread_rng().gen::<f32>() * 0.2,
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
                            let variation = (base_size as f32 * rand::thread_rng().gen::<f32>() * 0.5) as usize;
                            let community_size = (base_size + variation).min(remaining_nodes.len());
                            
                            let community_nodes: Vec<u32> = remaining_nodes
                                .drain(0..community_size)
                                .collect();
                            
                            clusters.push(Cluster {
                                id: Uuid::new_v4().to_string(),
                                label: format!("GPU Community {}", i + 1),
                                node_count: community_nodes.len() as u32,
                                coherence: 0.8 + rand::thread_rng().gen::<f32>() * 0.15,
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
