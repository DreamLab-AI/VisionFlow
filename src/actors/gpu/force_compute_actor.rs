//! Force Compute Actor - Handles physics force computation and simulation

use actix::prelude::*;
use log::{error, info, trace};
use serde::{Serialize, Deserialize};

use crate::actors::messages::*;
use crate::models::simulation_params::SimulationParams;
use crate::actors::gpu_compute_actor::ComputeMode;
use crate::utils::unified_gpu_compute::SimParams;
use super::shared::{SharedGPUContext, GPUState};

/// Physics statistics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsStats {
    pub iteration_count: u32,
    pub gpu_failure_count: u32,
    pub current_params: SimulationParams,
    pub compute_mode: ComputeMode,
    pub nodes_count: u32,
    pub edges_count: u32,
    
    // Physics state information
    pub average_velocity: f32,
    pub kinetic_energy: f32,
    pub total_forces: f32,
    
    // Performance metrics
    pub last_step_duration_ms: f32,
    pub fps: f32,
    
    // Add missing fields for compatibility
    pub num_edges: u32,
    pub total_force_calculations: u32,
}

/// Force Compute Actor - handles physics force computation and simulation
pub struct ForceComputeActor {
    /// Shared GPU state
    gpu_state: GPUState,
    
    /// Shared GPU context reference
    shared_context: Option<SharedGPUContext>,
    
    /// Physics simulation parameters
    simulation_params: SimulationParams,
    
    /// Unified physics parameters
    unified_params: SimParams,
    
    /// Current compute mode
    compute_mode: ComputeMode,
}

impl ForceComputeActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            simulation_params: SimulationParams::default(),
            unified_params: SimParams::default(),
            compute_mode: ComputeMode::Basic,
        }
    }
    
    /// Perform GPU force computation step
    fn perform_force_computation(&mut self) -> Result<(), String> {
        if self.iteration_count() % 60 == 0 { // Log every second at 60 FPS
            info!("ForceComputeActor: Computing forces (iteration {}), nodes: {}", 
                  self.iteration_count(), self.gpu_state.num_nodes);
        }
        
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?
            },
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };
        
        // Update unified parameters from simulation params
        let mut current_unified_params = self.unified_params.clone();
        self.sync_simulation_to_unified_params(&mut current_unified_params);
        
        // Execute force computation on GPU
        // Convert SimParams to SimulationParams
        let sim_params = &self.simulation_params;
        unified_compute.execute_physics_step(sim_params)
            .map_err(|e| {
                error!("GPU force computation failed: {}", e);
                self.gpu_state.gpu_failure_count += 1;
                format!("Force computation failed: {}", e)
            })?;
        
        // Increment iteration count
        self.gpu_state.iteration_count += 1;
        
        // Log performance metrics occasionally
        if self.iteration_count() % 300 == 0 { // Every 5 seconds
            info!("ForceComputeActor: {} iterations completed, {} GPU failures", 
                  self.iteration_count(), self.gpu_state.gpu_failure_count);
        }
        
        Ok(())
    }
    
    /// Synchronize simulation parameters to unified parameters
    fn sync_simulation_to_unified_params(&self, unified_params: &mut SimParams) {
        // Update unified params from simulation params
        // TODO: Map SimParams fields properly - using spring_k for attraction
        // unified_params.attraction_k = self.simulation_params.attraction_k;
        // TODO: Map SimParams fields properly - using repel_k for repulsion
        // unified_params.repulsion_k = self.simulation_params.repel_k;
        unified_params.damping = self.simulation_params.damping;
        unified_params.dt = self.simulation_params.dt;
        unified_params.max_velocity = self.simulation_params.max_velocity;
        // TODO: Map SimParams fields properly - using center_gravity_k for center strength
        // unified_params.center_strength = self.simulation_params.center_gravity_k;
        
        // Update physics mode based on compute mode
        match self.compute_mode {
            ComputeMode::Basic => {
                // TODO: Map SimParams fields properly
                // unified_params.enable_advanced_forces = false;
                // TODO: Map SimParams fields properly
                // unified_params.semantic_force_weight = 0.0;
                // TODO: Map SimParams fields properly
                // unified_params.temporal_force_weight = 0.0;
                // TODO: Map SimParams fields properly
                // unified_params.constraint_weight = 0.0;
            },
            ComputeMode::Advanced => {
                // TODO: Map SimParams fields properly
                // unified_params.enable_advanced_forces = true;
                // These would come from advanced params if available
                // TODO: Map SimParams fields properly
                // unified_params.semantic_force_weight = 0.3;
                // TODO: Map SimParams fields properly
                // unified_params.temporal_force_weight = 0.2;
                // TODO: Map SimParams fields properly
                // unified_params.constraint_weight = 0.5;
            },
            ComputeMode::DualGraph => {
                // TODO: Implement dual graph mode
                // For now, use Advanced mode behavior
                todo!("DualGraph mode not yet implemented")
            },
        }
        
        trace!("Unified params updated: damping={:.3}", unified_params.damping);
    }
    
    /// Get current iteration count
    fn iteration_count(&self) -> u32 {
        self.gpu_state.iteration_count
    }
    
    /// Update physics simulation parameters
    fn update_simulation_parameters(&mut self, params: SimulationParams) {
        info!("ForceComputeActor: Updating simulation parameters");
        info!("  attraction_k: {:.3} -> {:.3}", self.simulation_params.attraction_k, params.attraction_k);
        info!("  repel_k: {:.3} -> {:.3}", self.simulation_params.repel_k, params.repel_k);
        info!("  damping: {:.3} -> {:.3}", self.simulation_params.damping, params.damping);
        
        self.simulation_params = params;
        
        // Update unified params immediately
        // TODO: Implement sync_simulation_to_unified_params method
        // self.sync_simulation_to_unified_params(&mut self.unified_params);
    }
    
    /// Get current physics statistics
    fn get_physics_stats(&self) -> PhysicsStats {
        PhysicsStats {
            iteration_count: self.gpu_state.iteration_count,
            gpu_failure_count: self.gpu_state.gpu_failure_count,
            current_params: self.simulation_params.clone(),
            compute_mode: self.compute_mode.clone(),
            nodes_count: self.gpu_state.num_nodes,
            edges_count: self.gpu_state.num_edges,
            
            // Physics state information
            average_velocity: 0.0, // TODO: Calculate from GPU if needed
            kinetic_energy: 0.0,   // TODO: Calculate from GPU if needed
            total_forces: 0.0,     // TODO: Calculate from GPU if needed
            
            // Performance metrics
            last_step_duration_ms: 0.0, // TODO: Track timing
            fps: if self.gpu_state.iteration_count > 0 { 60.0 } else { 0.0 }, // Approximate
            
            // Missing fields for compatibility
            num_edges: self.gpu_state.num_edges,
            total_force_calculations: self.gpu_state.iteration_count * self.gpu_state.num_nodes,
        }
    }
}

impl Actor for ForceComputeActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("Force Compute Actor started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Force Compute Actor stopped");
    }
}

// === Message Handlers ===

impl Handler<ComputeForces> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: ComputeForces, _ctx: &mut Self::Context) -> Self::Result {
        self.perform_force_computation()
    }
}

impl Handler<UpdateSimulationParams> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateSimulationParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UpdateSimulationParams received");
        info!("  New params - attraction_k: {:.3}, repulsion_k: {:.3}, damping: {:.3}",
              msg.params.attraction_k, msg.params.repel_k, msg.params.damping);
        
        self.update_simulation_parameters(msg.params);
        Ok(())
    }
}

impl Handler<SetComputeMode> for ForceComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;
    
    fn handle(&mut self, msg: SetComputeMode, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: Setting compute mode to {:?}", msg.mode);
        
        self.compute_mode = msg.mode;
        
        // Update unified params to reflect new compute mode  
        let mut temp_params = self.unified_params.clone();
        self.sync_simulation_to_unified_params(&mut temp_params);
        self.unified_params = temp_params;
        
        use futures::future::ready;
        Box::pin(ready(Ok(())).into_actor(self))
    }
}

impl Handler<GetPhysicsStats> for ForceComputeActor {
    type Result = Result<PhysicsStats, String>;
    
    fn handle(&mut self, _msg: GetPhysicsStats, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_physics_stats())
    }
}

impl Handler<UpdateAdvancedParams> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateAdvancedParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UpdateAdvancedParams received");
        info!("  Advanced params - semantic_weight: {:.2}, temporal_weight: {:.2}, constraint_weight: {:.2}",
              msg.params.semantic_force_weight, msg.params.temporal_force_weight, msg.params.constraint_force_weight);
        
        // Update unified params with advanced physics parameters
        // TODO: Add these fields to SimParams
        // self.unified_params.semantic_force_weight = msg.params.semantic_force_weight;
        // self.unified_params.temporal_force_weight = msg.params.temporal_force_weight;
        // self.unified_params.constraint_weight = msg.params.constraint_force_weight;
        // self.unified_params.enable_advanced_forces = true;
        
        info!("Advanced physics parameters processed (fields need to be added to SimParams)");
        
        // Switch to advanced compute mode if not already
        if matches!(self.compute_mode, ComputeMode::Basic) {
            info!("ForceComputeActor: Switching to Advanced compute mode due to advanced params");
            self.compute_mode = ComputeMode::Advanced;
        }
        
        Ok(())
    }
}

// Position upload support for external updates
impl Handler<UploadPositions> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UploadPositions, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UploadPositions received - {} nodes", msg.positions_x.len());
        
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?
            },
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };
        
        // Upload positions to GPU for physics computation
        unified_compute.update_positions_only(&msg.positions_x, &msg.positions_y, &msg.positions_z)
            .map_err(|e| format!("Failed to upload positions: {}", e))?;
        
        info!("ForceComputeActor: Position upload completed successfully");
        Ok(())
    }
}