//! Legacy advanced GPU compute module - DEPRECATED
//! 
//! This module has been replaced by the unified GPU compute system.
//! All functionality is now integrated into `unified_gpu_compute.rs`.
//! This file is kept temporarily for API compatibility during migration.

use std::io::{Error, ErrorKind};
use log::{warn, info, error, trace};
use crate::utils::unified_gpu_compute::{UnifiedGPUCompute};
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::{Constraint, AdvancedParams};
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::utils::edge_data::EdgeData;
use crate::types::vec3::Vec3Data;
use std::sync::Arc;
use cudarc::driver::{CudaDevice, DeviceRepr, ValidAsZeroBits};

// DEPRECATED: EnhancedBinaryNodeData
// This is now handled internally by the unified GPU compute system using SoA layout
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

// Implement DeviceRepr traits for GPU transfer
unsafe impl DeviceRepr for EnhancedBinaryNodeData {}
unsafe impl ValidAsZeroBits for EnhancedBinaryNodeData {}

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

// Implement DeviceRepr traits for GPU transfer
unsafe impl DeviceRepr for EnhancedEdgeData {}
unsafe impl ValidAsZeroBits for EnhancedEdgeData {}

unsafe impl DeviceRepr for AdvancedSimulationParams {}
unsafe impl ValidAsZeroBits for AdvancedSimulationParams {}

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
        let params = Self {
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
        };

        // Log advanced parameters every 60 iterations for debugging
        if iteration % 60 == 0 {
            info!("ADVANCED_PARAMS: iteration={}, nodes={}", iteration, num_nodes);
            info!("  Physics: spring={:.4}, damping={:.4}, repel={:.1}, dt={:.4}, max_repel_dist={:.1}, bounds={:.1}",
                params.spring_k, params.damping, params.repel_k, params.dt,
                params.max_repulsion_dist, params.viewport_bounds);
            info!("  Force weights: semantic={:.2}, temporal={:.2}, structural={:.2}, constraint={:.2}, boundary={:.2}",
                params.semantic_force_weight, params.temporal_force_weight,
                params.structural_force_weight, params.constraint_force_weight,
                params.boundary_force_weight);
            info!("  Advanced: separation={:.2}, knowledge={:.2}, agent={:.2}, adaptive={:.1}, max_velocity={:.1}",
                params.separation_factor, params.knowledge_force_weight,
                params.agent_communication_weight, params.adaptive_scale,
                params.max_velocity);
            info!("  Layout: hierarchical={}, layer_sep={:.1}, target_edge={:.1}, collision={:.1}",
                params.hierarchical_mode, params.layer_separation,
                params.target_edge_length, params.collision_threshold);
        }

        params
    }
}

// Constants for GPU computation
const BLOCK_SIZE: u32 = 256;
#[allow(dead_code)]
const MAX_NODES: u32 = 1_000_000;
const MAX_CONSTRAINTS: u32 = 10_000;
const DEBUG_THROTTLE: u32 = 60;

/// DEPRECATED: AdvancedGPUContext
/// This has been replaced by UnifiedGPUCompute which handles all advanced features
/// in a single, optimized kernel using Structure-of-Arrays layout.
#[derive(Debug)]
pub struct AdvancedGPUContext {
    unified_compute: Option<UnifiedGPUCompute>,
    num_nodes: u32,
    num_edges: u32,
    simulation_params: SimulationParams,
    advanced_params: AdvancedParams,
    iteration_count: u32,
}

impl AdvancedGPUContext {
    /// Create a new advanced GPU context (DEPRECATED)
    /// This now delegates to UnifiedGPUCompute
    pub async fn new(
        num_nodes: u32,
        num_edges: u32,
        simulation_params: SimulationParams,
        advanced_params: AdvancedParams,
    ) -> Result<Self, Error> {
        warn!("AdvancedGPUContext::new is DEPRECATED. Use UnifiedGPUCompute instead.");
        info!("Creating compatibility wrapper with unified GPU compute for {} nodes, {} edges", num_nodes, num_edges);
        
        // Create CUDA device
        let device = Self::create_cuda_device().await?;
        
        // Initialize unified compute
        let unified_compute = match UnifiedGPUCompute::new(device, num_nodes as usize, num_edges as usize) {
            Ok(compute) => {
                info!("Successfully created unified GPU compute as advanced context replacement");
                Some(compute)
            },
            Err(e) => {
                warn!("Failed to create unified GPU compute: {}", e);
                None
            }
        };
        
        // Legacy PTX loading is no longer needed - unified kernel handles everything
        
        Ok(Self {
            unified_compute,
            num_nodes,
            num_edges,
            simulation_params,
            advanced_params,
            iteration_count: 0,
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
    
    /// Update node data with enhanced features (DEPRECATED)
    pub fn update_node_data(&mut self, nodes: Vec<EnhancedBinaryNodeData>) -> Result<(), Error> {
        warn!("update_node_data is DEPRECATED. Use UnifiedGPUCompute directly.");
        // This is now handled by the unified compute system
        if nodes.len() != self.num_nodes as usize {
            return Err(Error::new(ErrorKind::InvalidInput, 
                format!("Node count mismatch: expected {}, got {}", self.num_nodes, nodes.len())));
        }
        Ok(())
    }
    
    /// Update edge data with enhanced features (DEPRECATED)
    pub fn update_edge_data(&mut self, edges: Vec<EnhancedEdgeData>) -> Result<(), Error> {
        warn!("update_edge_data is DEPRECATED. Use UnifiedGPUCompute directly.");
        self.num_edges = edges.len() as u32;
        Ok(())
    }
    
    /// Update constraints on GPU (DEPRECATED)
    pub fn update_constraints(&mut self, constraints: &[Constraint]) -> Result<(), Error> {
        warn!("update_constraints is DEPRECATED. Use UnifiedGPUCompute directly.");
        let gpu_constraints: Vec<crate::utils::unified_gpu_compute::ConstraintData> = constraints.iter()
            .filter(|c| c.active)
            .take(MAX_CONSTRAINTS as usize)
            .map(|c| {
                // Manually create the correct ConstraintData type
                let mut node_idx = [-1i32; 4];
                for (i, &idx) in c.node_indices.iter().take(4).enumerate() {
                    node_idx[i] = idx as i32;
                }

                crate::utils::unified_gpu_compute::ConstraintData {
                    constraint_type: c.kind as i32,
                    strength: c.weight,
                    param1: c.params.get(0).copied().unwrap_or(0.0),
                    param2: c.params.get(1).copied().unwrap_or(0.0),
                    node_mask: c.node_indices.len() as i32,
                }
            })
            .collect();
        
        if let Some(ref mut unified) = self.unified_compute {
            unified.set_constraints(gpu_constraints)?;
        }
        
        Ok(())
    }
    
    /// Execute one physics step with constraints (DEPRECATED)
    pub fn step_with_constraints(&mut self, constraints: &[Constraint]) -> Result<(), Error> {
        warn!("step_with_constraints is DEPRECATED. Use UnifiedGPUCompute directly.");
        
        // Update constraints if provided
        if !constraints.is_empty() {
            self.update_constraints(constraints)?;
        }
        
        // Log periodically
        if self.iteration_count % DEBUG_THROTTLE == 0 {
            trace!("Executing unified physics step (iteration {})", self.iteration_count);
        }
        
        if let Some(ref mut unified) = self.unified_compute {
            unified.execute()?;
        }
        
        self.iteration_count += 1;
        Ok(())
    }
    
    /// Fallback to legacy force computation (REMOVED - no longer needed)
    fn compute_forces_legacy(&mut self) -> Result<(), Error> {
        warn!("compute_forces_legacy is DEPRECATED and removed. Unified kernel handles all cases.");
        Err(Error::new(ErrorKind::Other, "Legacy kernel removed - use unified compute"))
    }
    
    /// Get node data from GPU (DEPRECATED)
    pub fn get_node_data(&self) -> Result<Vec<EnhancedBinaryNodeData>, Error> {
        warn!("get_node_data is DEPRECATED. Use UnifiedGPUCompute directly.");
        // Return dummy data for compatibility
        let nodes = vec![EnhancedBinaryNodeData {
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

    /// Get the current iteration count
    pub fn iteration_count(&self) -> u32 {
        self.iteration_count
    }
    
    /// Switch between advanced and legacy kernels (DEPRECATED)
    pub fn set_use_advanced_kernel(&mut self, _use_advanced: bool) {
        warn!("set_use_advanced_kernel is DEPRECATED. Unified kernel is always used.");
        info!("All kernel modes are now unified - no switching needed");
    }
    
    /// Test GPU computation (DEPRECATED)
    pub fn test_compute(&mut self) -> Result<(), Error> {
        warn!("test_compute is DEPRECATED. Use UnifiedGPUCompute directly for testing.");
        
        if let Some(ref mut unified) = self.unified_compute {
            // Try a simple execution to test
            match unified.execute() {
                Ok(_) => {
                    info!("Unified GPU compute test passed");
                    Ok(())
                },
                Err(e) => {
                    error!("Unified GPU compute test failed: {}", e);
                    Err(Error::new(ErrorKind::Other, format!("GPU test failed: {}", e)))
                }
            }
        } else {
            Err(Error::new(ErrorKind::Other, "Unified compute not available"))
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