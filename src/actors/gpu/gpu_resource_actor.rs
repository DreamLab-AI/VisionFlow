//! GPU Resource Actor - Handles GPU initialization, memory management, and device status

use actix::prelude::*;
use log::{error, info, trace};
use std::sync::Arc;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{Error, ErrorKind};
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaStream};
use cudarc::driver::sys::CUdevice_attribute_enum;

use crate::actors::messages::*;
use crate::models::graph::GraphData;
use crate::utils::unified_gpu_compute::UnifiedGPUCompute;
use crate::utils::socket_flow_messages::BinaryNodeData;
use super::shared::GPUState;

/// Constants for GPU computation
const MAX_NODES: u32 = 1_000_000;
const MAX_GPU_FAILURES: u32 = 5;

/// GPU Resource Actor - manages GPU device initialization and resources
pub struct GPUResourceActor {
    /// GPU device handle
    device: Option<Arc<CudaDevice>>,
    
    /// CUDA stream for operations
    cuda_stream: Option<CudaStream>,
    
    /// Unified GPU compute engine
    unified_compute: Option<UnifiedGPUCompute>,
    
    /// Shared GPU state
    gpu_state: GPUState,
    
    /// Last failure reset time for recovery logic
    last_failure_reset: Instant,
}

impl GPUResourceActor {
    pub fn new() -> Self {
        Self {
            device: None,
            cuda_stream: None,
            unified_compute: None,
            gpu_state: GPUState::default(),
            last_failure_reset: Instant::now(),
        }
    }
    
    /// Initialize GPU device and unified compute engine
    async fn perform_gpu_initialization(&mut self, graph_data: Arc<GraphData>) -> Result<(), String> {
        info!("GPUResourceActor: Starting GPU initialization with {} nodes", graph_data.nodes.len());
        
        // Test GPU capabilities first
        Self::static_test_gpu_capabilities().await
            .map_err(|e| format!("GPU capabilities test failed: {}", e))?;
        
        // Initialize CUDA device
        let device = CudaDevice::new(0).map_err(|e| format!("Failed to create CUDA device: {}", e))?;
        info!("CUDA device initialized successfully");
        
        // Create CUDA stream
        let cuda_stream = device.fork_default_stream().map_err(|e| format!("Failed to create CUDA stream: {}", e))?;
        info!("CUDA stream created successfully");
        
        // Get device capabilities
        let max_threads_per_block = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            .map_err(|e| format!("Failed to get device attributes: {}", e))? as u32;
        
        let compute_capability_major = device.attribute(CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|e| format!("Failed to get compute capability: {}", e))?;
        
        info!("GPU Capabilities - Max threads per block: {}, Compute capability: {}.x", 
              max_threads_per_block, compute_capability_major);
        
        // Initialize unified compute engine
        let mut unified_compute = UnifiedGPUCompute::new(1000, 1000, include_str!("../../utils/visionflow_unified.ptx"))
            .map_err(|e| format!("Failed to create unified compute: {}", e))?;
        
        info!("UnifiedGPUCompute engine initialized successfully");
        
        // Prepare graph data and CSR representation
        let csr_result = self.create_csr_from_graph_data(&graph_data)
            .map_err(|e| format!("Failed to create CSR representation: {}", e))?;
        
        // Initialize unified compute with graph data
        unified_compute.initialize_graph(
            csr_result.row_offsets.iter().map(|&x| x as i32).collect(),
            csr_result.col_indices.iter().map(|&x| x as i32).collect(),
            csr_result.edge_weights,
            csr_result.positions_x,
            csr_result.positions_y,
            csr_result.positions_z,
            csr_result.num_nodes as usize,
            csr_result.num_edges as usize
        ).map_err(|e| format!("Failed to initialize graph in unified compute: {}", e))?;
        
        info!("Graph data uploaded to GPU successfully");
        
        // Store initialized resources
        self.device = Some(device);
        self.cuda_stream = Some(cuda_stream);
        self.unified_compute = Some(unified_compute);
        
        // Update GPU state
        self.gpu_state.num_nodes = csr_result.num_nodes;
        self.gpu_state.num_edges = csr_result.num_edges;
        self.gpu_state.node_indices = csr_result.node_indices;
        self.gpu_state.graph_structure_hash = Self::calculate_graph_structure_hash(&graph_data);
        self.gpu_state.positions_hash = Self::calculate_positions_hash(&graph_data);
        self.gpu_state.csr_structure_uploaded = true;
        
        info!("GPU initialization completed successfully");
        Ok(())
    }
    
    /// Test GPU capabilities
    async fn static_test_gpu_capabilities() -> Result<(), Error> {
        info!("Testing CUDA capabilities");
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
                Err(Error::new(ErrorKind::Other, format!("CUDA error: {}", e)))
            }
        }
    }
    
    /// Create CSR representation from graph data
    fn create_csr_from_graph_data(&self, graph_data: &GraphData) -> Result<CsrResult, String> {
        let num_nodes = graph_data.nodes.len() as u32;
        let num_edges = graph_data.edges.len() as u32;
        
        if num_nodes == 0 {
            return Err("Cannot create CSR from empty graph".to_string());
        }
        
        info!("Creating CSR representation: {} nodes, {} edges", num_nodes, num_edges);
        
        // Build node index mapping
        let mut node_indices = HashMap::new();
        for (i, node) in graph_data.nodes.iter().enumerate() {
            node_indices.insert(node.id, i);
        }
        
        // Initialize CSR structures
        let mut row_offsets = vec![0u32; (num_nodes + 1) as usize];
        let mut col_indices = Vec::new();
        let mut edge_weights = Vec::new();
        
        // Extract positions
        let positions_x: Vec<f32> = graph_data.nodes.iter().map(|n| n.data.position.x).collect();
        let positions_y: Vec<f32> = graph_data.nodes.iter().map(|n| n.data.position.y).collect();
        let positions_z: Vec<f32> = graph_data.nodes.iter().map(|n| n.data.position.z).collect();
        
        // Build adjacency lists for CSR conversion
        let mut adjacency_lists: Vec<Vec<(u32, f32)>> = vec![Vec::new(); num_nodes as usize];
        
        for edge in &graph_data.edges {
            if let (Some(&source_idx), Some(&target_idx)) = 
                (node_indices.get(&edge.source), node_indices.get(&edge.target)) {
                
                let weight = edge.weight;
                adjacency_lists[source_idx].push((target_idx as u32, weight));
                
                // Add reverse edge for undirected graph
                if source_idx != target_idx {
                    adjacency_lists[target_idx].push((source_idx as u32, weight));
                }
            }
        }
        
        // Convert to CSR format
        let mut edge_count = 0;
        for (i, adj_list) in adjacency_lists.iter().enumerate() {
            row_offsets[i] = edge_count;
            
            for &(target, weight) in adj_list {
                col_indices.push(target);
                edge_weights.push(weight);
                edge_count += 1;
            }
        }
        row_offsets[num_nodes as usize] = edge_count;
        
        info!("CSR conversion complete: {} total edges (including reverse edges)", edge_count);
        
        Ok(CsrResult {
            row_offsets,
            col_indices,
            edge_weights,
            positions_x,
            positions_y,
            positions_z,
            num_nodes,
            num_edges: edge_count,
            node_indices,
        })
    }
    
    /// Calculate hash for graph structure (nodes, edges, connectivity)
    fn calculate_graph_structure_hash(graph_data: &GraphData) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        // Hash node count and edge count
        graph_data.nodes.len().hash(&mut hasher);
        graph_data.edges.len().hash(&mut hasher);
        
        // Hash edge connectivity (source/target pairs)
        for edge in &graph_data.edges {
            edge.source.hash(&mut hasher);
            edge.target.hash(&mut hasher);
            // Use to_bits() for consistent float hashing
            edge.weight.to_bits().hash(&mut hasher);
        }
        
        hasher.finish()
    }
    
    /// Calculate hash for node positions only
    fn calculate_positions_hash(graph_data: &GraphData) -> u64 {
        let mut hasher = DefaultHasher::new();
        
        for node in &graph_data.nodes {
            // Use to_bits() for consistent float hashing across runs
            node.data.position.x.to_bits().hash(&mut hasher);
            node.data.position.y.to_bits().hash(&mut hasher);
            node.data.position.z.to_bits().hash(&mut hasher);
        }
        
        hasher.finish()
    }
}

/// CSR creation result
struct CsrResult {
    row_offsets: Vec<u32>,
    col_indices: Vec<u32>,
    edge_weights: Vec<f32>,
    positions_x: Vec<f32>,
    positions_y: Vec<f32>,
    positions_z: Vec<f32>,
    num_nodes: u32,
    num_edges: u32,
    node_indices: HashMap<u32, usize>,
}

impl Actor for GPUResourceActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("GPU Resource Actor started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("GPU Resource Actor stopped");
    }
}

// === Message Handlers ===

impl Handler<InitializeGPU> for GPUResourceActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;
    
    fn handle(&mut self, msg: InitializeGPU, _ctx: &mut Self::Context) -> Self::Result {
        info!("GPUResourceActor: InitializeGPU received with {} nodes", msg.graph.nodes.len());
        
        let graph_data = msg.graph;
        let graph_service_addr = msg.graph_service_addr;
        
        // Perform GPU initialization
        Box::pin(async move { 
            // This will call our async initialization method
            Ok::<(), ()>(())
        }.into_actor(self).map(move |result, actor, _ctx| {
            match result {
                Ok(_) => {
                    // Perform the actual GPU initialization
                    let initialization_result = futures::executor::block_on(
                        actor.perform_gpu_initialization(graph_data)
                    );
                    
                    match initialization_result {
                        Ok(_) => {
                            info!("GPU initialization completed successfully - notifying GraphServiceActor");
                            
                            // Send GPUInitialized message to GraphServiceActor if address provided
                            if let Some(addr) = graph_service_addr {
                                if let Err(e) = addr.try_send(crate::actors::messages::GPUInitialized) {
                                    error!("Failed to send GPUInitialized message: {}", e);
                                    return Err("Failed to notify GraphServiceActor of GPU initialization".to_string());
                                }
                                info!("GPUInitialized message sent successfully");
                            } else {
                                info!("No GraphServiceActor address provided, skipping notification");
                            }
                            Ok(())
                        },
                        Err(e) => {
                            error!("GPU initialization failed: {}", e);
                            Err(e)
                        }
                    }
                },
                Err(_) => Err("Failed to start GPU initialization".to_string())
            }
        }))
    }
}

impl Handler<UpdateGPUGraphData> for GPUResourceActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateGPUGraphData, _ctx: &mut Self::Context) -> Self::Result {
        if self.device.is_none() {
            error!("GPU not initialized! Cannot update graph data");
            return Err("GPU not initialized".to_string());
        }
        
        // Perform optimized graph data update with hash-based change detection
        self.update_graph_data_internal_optimized(&msg.graph)
    }
}

impl Handler<GetNodeData> for GPUResourceActor {
    type Result = Result<Vec<BinaryNodeData>, String>;
    
    fn handle(&mut self, _msg: GetNodeData, _ctx: &mut Self::Context) -> Self::Result {
        if let Some(ref mut unified_compute) = self.unified_compute {
            match unified_compute.get_node_positions() {
                Ok((positions_x, positions_y, positions_z)) => {
                    let mut node_data = Vec::new();
                    
                    for i in 0..positions_x.len().min(positions_y.len()).min(positions_z.len()) {
                        node_data.push(BinaryNodeData {
                            position: crate::types::vec3::Vec3Data {
                                x: positions_x[i],
                                y: positions_y[i],
                                z: positions_z[i],
                            },
                            velocity: crate::types::vec3::Vec3Data { x: 0.0, y: 0.0, z: 0.0 },
                            mass: 1,
                            flags: 0,
                            padding: [0, 0],
                        });
                    }
                    
                    Ok(node_data)
                },
                Err(e) => {
                    error!("Failed to get node positions from GPU: {}", e);
                    Err(format!("Failed to get node positions: {}", e))
                }
            }
        } else {
            Err("GPU not initialized".to_string())
        }
    }
}

impl GPUResourceActor {
    /// Optimized graph data update with hash-based change detection
    fn update_graph_data_internal_optimized(&mut self, graph_data: &Arc<GraphData>) -> Result<(), String> {
        let new_structure_hash = Self::calculate_graph_structure_hash(graph_data);
        let new_positions_hash = Self::calculate_positions_hash(graph_data);
        
        let structure_changed = new_structure_hash != self.gpu_state.graph_structure_hash;
        let positions_changed = new_positions_hash != self.gpu_state.positions_hash;
        
        info!("GPU upload optimization - structure_changed: {}, positions_changed: {}", 
              structure_changed, positions_changed);
        
        // Skip upload if no changes detected
        if !structure_changed && !positions_changed {
            trace!("GPU upload skipped - no changes detected");
            return Ok(());
        }
        
        if structure_changed {
            // Full structure update required
            info!("GPU: Full structure update required");
            
            let csr_result = self.create_csr_from_graph_data(graph_data)?;
            
            let unified_compute = self.unified_compute.as_mut()
                .ok_or_else(|| "Unified compute not initialized".to_string())?;
            
            unified_compute.initialize_graph(
                csr_result.row_offsets.iter().map(|&x| x as i32).collect(),
                csr_result.col_indices.iter().map(|&x| x as i32).collect(),
                csr_result.edge_weights,
                csr_result.positions_x,
                csr_result.positions_y,
                csr_result.positions_z,
                csr_result.num_nodes as usize,
                csr_result.num_edges as usize
            ).map_err(|e| format!("Failed to upload full graph structure: {}", e))?;
            
            // Update state
            self.gpu_state.num_nodes = csr_result.num_nodes;
            self.gpu_state.num_edges = csr_result.num_edges;
            self.gpu_state.node_indices = csr_result.node_indices;
            self.gpu_state.graph_structure_hash = new_structure_hash;
            self.gpu_state.positions_hash = new_positions_hash;
            self.gpu_state.csr_structure_uploaded = true;
            
        } else if positions_changed {
            // Position-only update
            info!("GPU: Position-only update");
            
            let positions_x: Vec<f32> = graph_data.nodes.iter().map(|n| n.data.position.x).collect();
            let positions_y: Vec<f32> = graph_data.nodes.iter().map(|n| n.data.position.y).collect();
            let positions_z: Vec<f32> = graph_data.nodes.iter().map(|n| n.data.position.z).collect();
            
            let unified_compute = self.unified_compute.as_mut()
                .ok_or_else(|| "Unified compute not initialized".to_string())?;
            
            unified_compute.update_positions_only(&positions_x, &positions_y, &positions_z)
                .map_err(|e| format!("Failed to update positions: {}", e))?;
            
            self.gpu_state.positions_hash = new_positions_hash;
        }
        
        Ok(())
    }
}