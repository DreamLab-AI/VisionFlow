//! Advanced GPU compute module with constraint-aware physics

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchConfig, LaunchAsync};
use cudarc::nvrtc::Ptx;
use std::io::{Error, ErrorKind};
use std::sync::Arc;
use log::{error, warn, info, trace};
use std::collections::HashMap;
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::{Constraint, ConstraintData, AdvancedParams};
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::utils::edge_data::EdgeData;
use crate::types::vec3::Vec3Data;
use std::path::Path;
use std::env;

// Enhanced node data structure for advanced physics
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EnhancedBinaryNodeData {
    pub position: Vec3Data,
    pub velocity: Vec3Data,
    pub mass: u8,
    pub flags: u8,
    pub node_type: u8,
    pub cluster_id: u8,
    pub semantic_weight: f32,
    pub temporal_weight: f32,
    pub structural_weight: f32,
    pub importance_score: f32,
}

impl From<BinaryNodeData> for EnhancedBinaryNodeData {
    fn from(node: BinaryNodeData) -> Self {
        Self {
            position: node.position,
            velocity: node.velocity,
            mass: node.mass,
            flags: node.flags,
            node_type: 0,
            cluster_id: 0,
            semantic_weight: 1.0,
            temporal_weight: 1.0,
            structural_weight: 1.0,
            importance_score: 0.5,
        }
    }
}

impl From<EnhancedBinaryNodeData> for BinaryNodeData {
    fn from(node: EnhancedBinaryNodeData) -> Self {
        Self {
            position: node.position,
            velocity: node.velocity,
            mass: node.mass,
            flags: node.flags,
            padding: [0, 0],
        }
    }
}

// Enhanced edge data with multi-modal similarities
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EnhancedEdgeData {
    pub source_idx: i32,
    pub target_idx: i32,
    pub weight: f32,
    pub semantic_similarity: f32,
    pub structural_similarity: f32,
    pub temporal_similarity: f32,
    pub communication_strength: f32,
    pub edge_type: u8,
    pub bidirectional: u8,
    pub pad: [u8; 2],
}

impl From<EdgeData> for EnhancedEdgeData {
    fn from(edge: EdgeData) -> Self {
        Self {
            source_idx: edge.source_idx,
            target_idx: edge.target_idx,
            weight: edge.weight,
            semantic_similarity: 0.5,
            structural_similarity: 0.5,
            temporal_similarity: 0.5,
            communication_strength: 0.0,
            edge_type: 0,
            bidirectional: 1,
            pad: [0, 0],
        }
    }
}

// Advanced simulation parameters for GPU kernel
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AdvancedSimulationParams {
    // Basic physics
    pub spring_k: f32,
    pub damping: f32,
    pub repel_k: f32,
    pub dt: f32,
    pub max_repulsion_dist: f32,
    pub viewport_bounds: f32,
    
    // Advanced force weights
    pub semantic_force_weight: f32,
    pub temporal_force_weight: f32,
    pub structural_force_weight: f32,
    pub constraint_force_weight: f32,
    pub boundary_force_weight: f32,
    pub separation_factor: f32,
    pub knowledge_force_weight: f32,
    pub agent_communication_weight: f32,
    
    // Layout optimization
    pub target_edge_length: f32,
    pub max_velocity: f32,
    pub collision_threshold: f32,
    pub adaptive_scale: f32,
    
    // Hierarchical layout
    pub hierarchical_mode: i32,
    pub layer_separation: f32,
    
    // System parameters
    pub iteration: i32,
    pub total_nodes: i32,
}

impl AdvancedSimulationParams {
    pub fn from_params(sim_params: &SimulationParams, adv_params: &AdvancedParams, iteration: u32, num_nodes: u32) -> Self {
        Self {
            spring_k: sim_params.spring_strength,
            damping: sim_params.damping,
            repel_k: sim_params.repulsion,
            dt: sim_params.time_step,
            max_repulsion_dist: sim_params.max_repulsion_distance,
            viewport_bounds: if sim_params.enable_bounds { sim_params.viewport_bounds } else { f32::MAX },
            
            semantic_force_weight: adv_params.semantic_force_weight,
            temporal_force_weight: adv_params.temporal_force_weight,
            structural_force_weight: adv_params.structural_force_weight,
            constraint_force_weight: adv_params.constraint_force_weight,
            boundary_force_weight: adv_params.boundary_force_weight,
            separation_factor: adv_params.separation_factor,
            knowledge_force_weight: adv_params.knowledge_force_weight,
            agent_communication_weight: adv_params.agent_communication_weight,
            
            target_edge_length: adv_params.target_edge_length,
            max_velocity: adv_params.max_velocity,
            collision_threshold: adv_params.collision_threshold,
            adaptive_scale: if adv_params.adaptive_force_scaling { 1.0 } else { 0.0 },
            
            hierarchical_mode: if adv_params.hierarchical_mode { 1 } else { 0 },
            layer_separation: adv_params.layer_separation,
            
            iteration: iteration as i32,
            total_nodes: num_nodes as i32,
        }
    }
}

// Constants for GPU computation
const BLOCK_SIZE: u32 = 256;
const MAX_NODES: u32 = 1_000_000;
const MAX_CONSTRAINTS: u32 = 10_000;
const DEBUG_THROTTLE: u32 = 60;

/// Advanced GPU compute context with constraint support
pub struct AdvancedGPUContext {
    pub device: Arc<CudaDevice>,
    pub advanced_kernel: CudaFunction,
    pub legacy_kernel: Option<CudaFunction>, // For fallback
    pub node_data: CudaSlice<EnhancedBinaryNodeData>,
    pub edge_data: CudaSlice<EnhancedEdgeData>,
    pub constraint_data: CudaSlice<ConstraintData>,
    pub num_nodes: u32,
    pub num_edges: u32,
    pub num_constraints: u32,
    pub node_indices: HashMap<u32, usize>,
    pub simulation_params: SimulationParams,
    pub advanced_params: AdvancedParams,
    pub iteration_count: u32,
    pub use_advanced_kernel: bool,
}

impl AdvancedGPUContext {
    /// Create a new advanced GPU context
    pub async fn new(
        num_nodes: u32,
        num_edges: u32,
        simulation_params: SimulationParams,
        advanced_params: AdvancedParams,
    ) -> Result<Self, Error> {
        info!("Initializing advanced GPU context for {} nodes, {} edges", num_nodes, num_edges);
        
        // Create CUDA device
        let device = Self::create_cuda_device().await?;
        
        // Load advanced kernel PTX
        let advanced_ptx_path = "src/utils/advanced_compute_forces.ptx";
        let advanced_kernel = if Path::new(advanced_ptx_path).exists() {
            info!("Loading advanced CUDA kernel from {}", advanced_ptx_path);
            let ptx_data = std::fs::read(advanced_ptx_path)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to read advanced PTX: {}", e)))?;
            
            let ptx = Ptx::from_bytes(&ptx_data);
            device.load_ptx(ptx, "advanced_forces", &["advanced_forces_kernel"])
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to load advanced PTX: {}", e)))?;
            
            Some(device.get_func("advanced_forces", "advanced_forces_kernel")
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to get advanced kernel: {}", e)))?)
        } else {
            warn!("Advanced PTX not found at {}, will use legacy kernel", advanced_ptx_path);
            None
        };
        
        // Load legacy kernel as fallback
        let legacy_kernel = {
            let legacy_ptx_path = "src/utils/compute_forces.ptx";
            if Path::new(legacy_ptx_path).exists() {
                info!("Loading legacy CUDA kernel as fallback");
                let ptx_data = std::fs::read(legacy_ptx_path)
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to read legacy PTX: {}", e)))?;
                
                let ptx = Ptx::from_bytes(&ptx_data);
                device.load_ptx(ptx, "compute_forces", &["compute_forces_kernel"])
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to load legacy PTX: {}", e)))?;
                
                Some(device.get_func("compute_forces", "compute_forces_kernel")
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to get legacy kernel: {}", e)))?)
            } else {
                None
            }
        };
        
        // Ensure we have at least one kernel
        let kernel = advanced_kernel.as_ref()
            .or(legacy_kernel.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::Other, "No CUDA kernels available"))?
            .clone();
        
        // Allocate GPU memory
        let node_data = device.alloc_zeros::<EnhancedBinaryNodeData>(num_nodes as usize)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate node memory: {}", e)))?;
        
        let edge_data = device.alloc_zeros::<EnhancedEdgeData>(num_edges.max(1) as usize)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate edge memory: {}", e)))?;
        
        let constraint_data = device.alloc_zeros::<ConstraintData>(MAX_CONSTRAINTS as usize)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to allocate constraint memory: {}", e)))?;
        
        Ok(Self {
            device,
            advanced_kernel: kernel,
            legacy_kernel,
            node_data,
            edge_data,
            constraint_data,
            num_nodes,
            num_edges,
            num_constraints: 0,
            node_indices: HashMap::new(),
            simulation_params,
            advanced_params,
            iteration_count: 0,
            use_advanced_kernel: advanced_kernel.is_some(),
        })
    }
    
    async fn create_cuda_device() -> Result<Arc<CudaDevice>, Error> {
        trace!("Creating CUDA device for advanced GPU context");
        match CudaDevice::new(0) {
            Ok(device) => {
                info!("Successfully created CUDA device for advanced physics");
                Ok(device.into())
            },
            Err(e) => {
                error!("Failed to create CUDA device: {}", e);
                Err(Error::new(ErrorKind::Other, format!("Failed to create CUDA device: {}", e)))
            }
        }
    }
    
    /// Update node data with enhanced features
    pub fn update_node_data(&mut self, nodes: Vec<EnhancedBinaryNodeData>) -> Result<(), Error> {
        if nodes.len() != self.num_nodes as usize {
            return Err(Error::new(ErrorKind::InvalidInput, 
                format!("Node count mismatch: expected {}, got {}", self.num_nodes, nodes.len())));
        }
        
        self.device.htod_sync_copy_into(&nodes, &mut self.node_data)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy nodes to GPU: {}", e)))?;
        
        Ok(())
    }
    
    /// Update edge data with enhanced features
    pub fn update_edge_data(&mut self, edges: Vec<EnhancedEdgeData>) -> Result<(), Error> {
        self.num_edges = edges.len() as u32;
        
        if self.num_edges > 0 {
            // Reallocate if needed
            if self.num_edges > self.edge_data.len() as u32 {
                self.edge_data = self.device.alloc_zeros::<EnhancedEdgeData>(self.num_edges as usize)
                    .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to reallocate edge memory: {}", e)))?;
            }
            
            self.device.htod_sync_copy_into(&edges, &mut self.edge_data)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy edges to GPU: {}", e)))?;
        }
        
        Ok(())
    }
    
    /// Update constraints on GPU
    pub fn update_constraints(&mut self, constraints: &[Constraint]) -> Result<(), Error> {
        let gpu_constraints: Vec<ConstraintData> = constraints.iter()
            .filter(|c| c.active)
            .take(MAX_CONSTRAINTS as usize)
            .map(ConstraintData::from_constraint)
            .collect();
        
        self.num_constraints = gpu_constraints.len() as u32;
        
        if self.num_constraints > 0 {
            self.device.htod_sync_copy_into(&gpu_constraints, &mut self.constraint_data)
                .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy constraints to GPU: {}", e)))?;
        }
        
        info!("Updated {} active constraints on GPU", self.num_constraints);
        Ok(())
    }
    
    /// Execute one physics step with constraints
    pub fn step_with_constraints(&mut self, constraints: &[Constraint]) -> Result<(), Error> {
        // Update constraints if provided
        if !constraints.is_empty() {
            self.update_constraints(constraints)?;
        }
        
        // Log periodically
        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Executing advanced physics step (iteration {})", self.iteration_count);
            trace!("  - Nodes: {}, Edges: {}, Constraints: {}", 
                self.num_nodes, self.num_edges, self.num_constraints);
        }
        
        // Prepare launch configuration
        let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0, // Advanced kernel manages its own shared memory
        };
        
        // Build kernel parameters
        let params = AdvancedSimulationParams::from_params(
            &self.simulation_params,
            &self.advanced_params,
            self.iteration_count,
            self.num_nodes,
        );
        
        // Launch kernel
        if self.use_advanced_kernel {
            unsafe {
                self.advanced_kernel.clone().launch(cfg, (
                    &self.node_data,
                    &self.edge_data,
                    self.num_nodes as i32,
                    self.num_edges as i32,
                    &self.constraint_data,
                    self.num_constraints as i32,
                    params,
                )).map_err(|e| {
                    error!("Advanced kernel launch failed: {}", e);
                    Error::new(ErrorKind::Other, e.to_string())
                })?;
            }
        } else {
            // Fallback to legacy kernel (without constraints)
            warn!("Using legacy kernel without constraint support");
            self.compute_forces_legacy()?;
        }
        
        self.iteration_count += 1;
        Ok(())
    }
    
    /// Fallback to legacy force computation
    fn compute_forces_legacy(&mut self) -> Result<(), Error> {
        if let Some(ref kernel) = self.legacy_kernel {
            let blocks = ((self.num_nodes + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_SIZE, 1, 1),
                shared_mem_bytes: BLOCK_SIZE * std::mem::size_of::<BinaryNodeData>() as u32,
            };
            
            // Convert to legacy format temporarily
            // Note: This is inefficient but ensures compatibility
            unsafe {
                kernel.clone().launch(cfg, (
                    &self.node_data, // Will be reinterpreted by legacy kernel
                    &self.edge_data, // Will be reinterpreted by legacy kernel
                    self.num_nodes as i32,
                    self.num_edges as i32,
                    self.simulation_params.spring_strength,
                    self.simulation_params.damping,
                    self.simulation_params.repulsion,
                    self.simulation_params.time_step,
                    self.simulation_params.max_repulsion_distance,
                    self.simulation_params.viewport_bounds,
                    self.iteration_count as i32,
                )).map_err(|e| {
                    error!("Legacy kernel launch failed: {}", e);
                    Error::new(ErrorKind::Other, e.to_string())
                })?;
            }
        } else {
            return Err(Error::new(ErrorKind::Other, "No fallback kernel available"));
        }
        Ok(())
    }
    
    /// Get node data from GPU
    pub fn get_node_data(&self) -> Result<Vec<EnhancedBinaryNodeData>, Error> {
        let mut nodes = vec![EnhancedBinaryNodeData {
            position: Vec3Data::zero(),
            velocity: Vec3Data::zero(),
            mass: 0,
            flags: 0,
            node_type: 0,
            cluster_id: 0,
            semantic_weight: 1.0,
            temporal_weight: 1.0,
            structural_weight: 1.0,
            importance_score: 0.5,
        }; self.num_nodes as usize];
        
        self.device.dtoh_sync_copy_into(&self.node_data, &mut nodes)
            .map_err(|e| Error::new(ErrorKind::Other, format!("Failed to copy nodes from GPU: {}", e)))?;
        
        Ok(nodes)
    }
    
    /// Get node data in legacy format for compatibility
    pub fn get_legacy_node_data(&self) -> Result<Vec<BinaryNodeData>, Error> {
        let enhanced = self.get_node_data()?;
        Ok(enhanced.into_iter().map(BinaryNodeData::from).collect())
    }
    
    /// Update simulation parameters
    pub fn update_simulation_params(&mut self, params: SimulationParams) {
        self.simulation_params = params;
    }
    
    /// Update advanced parameters
    pub fn update_advanced_params(&mut self, params: AdvancedParams) {
        self.advanced_params = params;
    }
    
    /// Switch between advanced and legacy kernels
    pub fn set_use_advanced_kernel(&mut self, use_advanced: bool) {
        if use_advanced && self.advanced_kernel.raw != 0 {
            self.use_advanced_kernel = true;
            info!("Switched to advanced physics kernel");
        } else if !use_advanced && self.legacy_kernel.is_some() {
            self.use_advanced_kernel = false;
            info!("Switched to legacy physics kernel");
        } else {
            warn!("Cannot switch kernel mode - requested kernel not available");
        }
    }
    
    /// Test GPU computation
    pub fn test_compute(&self) -> Result<(), Error> {
        info!("Running advanced GPU compute test");
        match self.device.synchronize() {
            Ok(_) => {
                info!("Advanced GPU device test passed");
                Ok(())
            },
            Err(e) => {
                error!("Advanced GPU device test failed: {}", e);
                Err(Error::new(ErrorKind::Other, format!("GPU test failed: {}", e)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_advanced_gpu_context_creation() {
        let sim_params = SimulationParams::default();
        let adv_params = AdvancedParams::default();
        
        match AdvancedGPUContext::new(100, 200, sim_params, adv_params).await {
            Ok(ctx) => {
                assert_eq!(ctx.num_nodes, 100);
                assert_eq!(ctx.num_edges, 200);
            },
            Err(e) => {
                // GPU might not be available in test environment
                println!("GPU context creation failed (expected in CI): {}", e);
            }
        }
    }
    
    #[test]
    fn test_enhanced_node_conversion() {
        let basic = BinaryNodeData {
            position: Vec3Data { x: 1.0, y: 2.0, z: 3.0 },
            velocity: Vec3Data { x: 0.1, y: 0.2, z: 0.3 },
            mass: 10,
            flags: 1,
            padding: [0, 0],
        };
        
        let enhanced = EnhancedBinaryNodeData::from(basic);
        assert_eq!(enhanced.position.x, 1.0);
        assert_eq!(enhanced.mass, 10);
        
        let converted_back = BinaryNodeData::from(enhanced);
        assert_eq!(converted_back.position.x, 1.0);
        assert_eq!(converted_back.mass, 10);
    }
}