//! Force Compute Actor - Handles physics force computation and simulation

use actix::prelude::*;
use log::{error, info, trace};
use serde::{Serialize, Deserialize};
use std::time::Instant;

use crate::actors::messages::*;
use crate::models::simulation_params::SimulationParams;
use crate::utils::unified_gpu_compute::ComputeMode;
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

    /// Last computation step timing
    last_step_start: Option<Instant>,
    last_step_duration_ms: f32,
}

impl ForceComputeActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            simulation_params: SimulationParams::default(),
            unified_params: SimParams::default(),
            compute_mode: ComputeMode::Basic,
            last_step_start: None,
            last_step_duration_ms: 0.0,
        }
    }
    
    /// Perform GPU force computation step
    fn perform_force_computation(&mut self) -> Result<(), String> {
        // Start timing the computation step
        let step_start = Instant::now();

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

        // Record timing for this computation step
        self.last_step_duration_ms = step_start.elapsed().as_millis() as f32;

        // Log performance metrics occasionally
        if self.iteration_count() % 300 == 0 { // Every 5 seconds
            info!("ForceComputeActor: {} iterations completed, {} GPU failures, last step: {:.2}ms",
                  self.iteration_count(), self.gpu_state.gpu_failure_count, self.last_step_duration_ms);
        }

        Ok(())
    }
    
    /// Synchronize simulation parameters to unified parameters
    fn sync_simulation_to_unified_params(&self, unified_params: &mut SimParams) {
        // Update unified params from simulation params
        unified_params.spring_k = self.simulation_params.spring_k;
        unified_params.repel_k = self.simulation_params.repel_k;
        unified_params.damping = self.simulation_params.damping;
        unified_params.dt = self.simulation_params.dt;
        unified_params.max_velocity = self.simulation_params.max_velocity;
        unified_params.center_gravity_k = self.simulation_params.center_gravity_k;
        
        // Update physics mode based on compute mode
        match self.compute_mode {
            ComputeMode::Basic => {
                // Basic mode - all features enabled, no additional processing needed
                // Note: Feature flags are already set in SimParams based on force values
            },
            ComputeMode::Advanced => {
                // Advanced mode - use enhanced physics features
                // These would come from advanced params if available
                unified_params.temperature = self.simulation_params.temperature;
                unified_params.alignment_strength = self.simulation_params.alignment_strength;
                unified_params.cluster_strength = self.simulation_params.cluster_strength;
            },
            ComputeMode::DualGraph => {
                // TODO: Implement dual graph mode
                // For now, use Advanced mode behavior
                unified_params.temperature = self.simulation_params.temperature;
                unified_params.alignment_strength = self.simulation_params.alignment_strength;
                unified_params.cluster_strength = self.simulation_params.cluster_strength;
            },
            ComputeMode::Constraints => {
                // Constraints mode - maximum physics features
                unified_params.temperature = self.simulation_params.temperature;
                unified_params.alignment_strength = self.simulation_params.alignment_strength;
                unified_params.cluster_strength = self.simulation_params.cluster_strength;
                unified_params.constraint_ramp_frames = self.simulation_params.constraint_ramp_frames;
                unified_params.constraint_max_force_per_node = self.simulation_params.constraint_max_force_per_node;
            },
        }
        
        trace!("Unified params updated: spring_k={:.3}, repel_k={:.3}, center_gravity_k={:.3}, damping={:.3}",
               unified_params.spring_k, unified_params.repel_k, unified_params.center_gravity_k, unified_params.damping);
    }
    
    /// Get current iteration count
    fn iteration_count(&self) -> u32 {
        self.gpu_state.iteration_count
    }
    
    /// Update physics simulation parameters
    fn update_simulation_parameters(&mut self, params: SimulationParams) {
        info!("ForceComputeActor: Updating simulation parameters");
        info!("  spring_k: {:.3} -> {:.3}", self.simulation_params.spring_k, params.spring_k);
        info!("  repel_k: {:.3} -> {:.3}", self.simulation_params.repel_k, params.repel_k);
        info!("  damping: {:.3} -> {:.3}", self.simulation_params.damping, params.damping);
        
        self.simulation_params = params;
        
        // Update unified params immediately
        {
            let unified_params = &mut self.unified_params;
            unified_params.spring_k = self.simulation_params.spring_k;
            unified_params.repel_k = self.simulation_params.repel_k;
            unified_params.damping = self.simulation_params.damping;
            unified_params.dt = self.simulation_params.dt;
        }
    }
    
    /// Get current physics statistics
    fn get_physics_stats(&self) -> PhysicsStats {
        // Calculate physics metrics from GPU data if available
        let (average_velocity, kinetic_energy, total_forces) = self.calculate_physics_metrics();

        // Calculate FPS from last step duration
        let fps = if self.last_step_duration_ms > 0.0 {
            1000.0 / self.last_step_duration_ms
        } else {
            0.0
        };

        PhysicsStats {
            iteration_count: self.gpu_state.iteration_count,
            gpu_failure_count: self.gpu_state.gpu_failure_count,
            current_params: self.simulation_params.clone(),
            compute_mode: self.compute_mode.clone(),
            nodes_count: self.gpu_state.num_nodes,
            edges_count: self.gpu_state.num_edges,

            // Physics state information
            average_velocity,
            kinetic_energy,
            total_forces,

            // Performance metrics
            last_step_duration_ms: self.last_step_duration_ms,
            fps,

            // Missing fields for compatibility
            num_edges: self.gpu_state.num_edges,
            total_force_calculations: self.gpu_state.iteration_count * self.gpu_state.num_nodes,
        }
    }

    /// Calculate physics metrics from GPU data
    fn calculate_physics_metrics(&self) -> (f32, f32, f32) {
        // Try to access unified compute to get actual data
        if let Some(ctx) = &self.shared_context {
            if let Ok(unified_compute) = ctx.unified_compute.lock() {
                return self.extract_gpu_metrics(&*unified_compute);
            }
        }

        // Fallback to estimated values based on simulation parameters if GPU data unavailable
        let estimated_velocity = self.simulation_params.max_velocity * 0.3; // Assume 30% of max velocity
        let estimated_kinetic_energy = 0.5 * (self.gpu_state.num_nodes as f32) * estimated_velocity.powi(2);
        let estimated_total_forces = self.simulation_params.spring_k * (self.gpu_state.num_edges as f32) * 0.5;

        (estimated_velocity, estimated_kinetic_energy, estimated_total_forces)
    }

    /// Extract actual physics metrics from GPU data
    fn extract_gpu_metrics(&self, unified_compute: &crate::utils::unified_gpu_compute::UnifiedGPUCompute) -> (f32, f32, f32) {
        let num_nodes = unified_compute.num_nodes;

        // Download velocity data to calculate average velocity and kinetic energy
        let mut vel_x = vec![0.0f32; num_nodes];
        let mut vel_y = vec![0.0f32; num_nodes];
        let mut vel_z = vec![0.0f32; num_nodes];

        // Attempt to download velocities (may fail if GPU is busy)
        if unified_compute.download_velocities(&mut vel_x, &mut vel_y, &mut vel_z).is_ok() {
            // Calculate average velocity magnitude
            let total_velocity: f32 = vel_x.iter().zip(&vel_y).zip(&vel_z)
                .map(|((vx, vy), vz)| (vx*vx + vy*vy + vz*vz).sqrt())
                .sum();
            let average_velocity = if num_nodes > 0 {
                total_velocity / num_nodes as f32
            } else {
                0.0
            };

            // Calculate kinetic energy (assuming unit mass for all nodes)
            let kinetic_energy: f32 = vel_x.iter().zip(&vel_y).zip(&vel_z)
                .map(|((vx, vy), vz)| 0.5 * (vx*vx + vy*vy + vz*vz))
                .sum();

            // Estimate total forces based on velocity changes and damping
            let estimated_total_forces = total_velocity * self.simulation_params.damping * num_nodes as f32;

            (average_velocity, kinetic_energy, estimated_total_forces)
        } else {
            // GPU data unavailable, use fallback estimates
            let estimated_velocity = self.simulation_params.max_velocity * 0.3;
            let estimated_kinetic_energy = 0.5 * (num_nodes as f32) * estimated_velocity.powi(2);
            let estimated_total_forces = self.simulation_params.spring_k * (self.gpu_state.num_edges as f32) * 0.5;

            (estimated_velocity, estimated_kinetic_energy, estimated_total_forces)
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
        info!("  New params - spring_k: {:.3}, repel_k: {:.3}, damping: {:.3}",
              msg.params.spring_k, msg.params.repel_k, msg.params.damping);
        
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

// === Additional Message Handlers for Compatibility ===

impl Handler<InitializeGPU> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: InitializeGPU, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: InitializeGPU received");
        
        // Store graph data dimensions
        self.gpu_state.num_nodes = msg.graph.nodes.len() as u32;
        self.gpu_state.num_edges = msg.graph.edges.len() as u32;
        
        info!("ForceComputeActor: GPU initialized with {} nodes, {} edges", 
              self.gpu_state.num_nodes, self.gpu_state.num_edges);
        
        Ok(())
    }
}

impl Handler<UpdateGPUGraphData> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateGPUGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UpdateGPUGraphData received");
        
        // Update graph dimensions
        self.gpu_state.num_nodes = msg.graph.nodes.len() as u32;
        self.gpu_state.num_edges = msg.graph.edges.len() as u32;
        
        info!("ForceComputeActor: Graph data updated - {} nodes, {} edges", 
              self.gpu_state.num_nodes, self.gpu_state.num_edges);
        
        Ok(())
    }
}

impl Handler<GetNodeData> for ForceComputeActor {
    type Result = Result<Vec<crate::utils::socket_flow_messages::BinaryNodeData>, String>;
    
    fn handle(&mut self, _msg: GetNodeData, _ctx: &mut Self::Context) -> Self::Result {
        // Return empty vector - actual node data should be retrieved from GraphServiceActor
        Ok(Vec::new())
    }
}

impl Handler<GetGPUStatus> for ForceComputeActor {
    type Result = GPUStatus;
    
    fn handle(&mut self, _msg: GetGPUStatus, _ctx: &mut Self::Context) -> Self::Result {
        GPUStatus {
            is_initialized: self.shared_context.is_some(),
            failure_count: self.gpu_state.gpu_failure_count,
            iteration_count: self.gpu_state.iteration_count,
            num_nodes: self.gpu_state.num_nodes,
        }
    }
}

impl Handler<GetGPUMetrics> for ForceComputeActor {
    type Result = Result<serde_json::Value, String>;
    
    fn handle(&mut self, _msg: GetGPUMetrics, _ctx: &mut Self::Context) -> Self::Result {
        use serde_json::json;
        
        Ok(json!({
            "memory_usage_mb": 0.0,
            "gpu_utilization": 0.0,
            "temperature_c": 0.0,
            "power_usage_w": 0.0,
            "compute_units": 0,
            "max_threads": 0,
            "clock_speed_mhz": 0,
        }))
    }
}

impl Handler<RunCommunityDetection> for ForceComputeActor {
    type Result = Result<CommunityDetectionResult, String>;
    
    fn handle(&mut self, _msg: RunCommunityDetection, _ctx: &mut Self::Context) -> Self::Result {
        // Community detection should be handled by ClusteringActor
        Err("Community detection should be handled by ClusteringActor".to_string())
    }
}

impl Handler<UpdateVisualAnalyticsParams> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: UpdateVisualAnalyticsParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UpdateVisualAnalyticsParams received (no-op, handled by other actors)");
        Ok(())
    }
}

impl Handler<GetConstraints> for ForceComputeActor {
    type Result = Result<crate::models::constraints::ConstraintSet, String>;
    
    fn handle(&mut self, _msg: GetConstraints, _ctx: &mut Self::Context) -> Self::Result {
        // Constraints should be handled by ConstraintActor
        Err("Constraints should be handled by ConstraintActor".to_string())
    }
}

impl Handler<UpdateConstraints> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: UpdateConstraints, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UpdateConstraints received (forwarding to ConstraintActor would be done by GPUManagerActor)");
        Ok(())
    }
}

impl Handler<UploadConstraintsToGPU> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: UploadConstraintsToGPU, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UploadConstraintsToGPU received (forwarding to ConstraintActor would be done by GPUManagerActor)");
        Ok(())
    }
}

impl Handler<TriggerStressMajorization> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: TriggerStressMajorization, _ctx: &mut Self::Context) -> Self::Result {
        // Stress majorization should be handled by StressMajorizationActor
        Err("Stress majorization should be handled by StressMajorizationActor".to_string())
    }
}

impl Handler<GetStressMajorizationStats> for ForceComputeActor {
    type Result = Result<crate::actors::gpu::stress_majorization_actor::StressMajorizationStats, String>;
    
    fn handle(&mut self, _msg: GetStressMajorizationStats, _ctx: &mut Self::Context) -> Self::Result {
        // Stress majorization should be handled by StressMajorizationActor
        Err("Stress majorization stats should be retrieved from StressMajorizationActor".to_string())
    }
}

impl Handler<ResetStressMajorizationSafety> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: ResetStressMajorizationSafety, _ctx: &mut Self::Context) -> Self::Result {
        // Stress majorization should be handled by StressMajorizationActor
        Err("Stress majorization safety reset should be handled by StressMajorizationActor".to_string())
    }
}

impl Handler<UpdateStressMajorizationParams> for ForceComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: UpdateStressMajorizationParams, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UpdateStressMajorizationParams received (forwarding to StressMajorizationActor would be done by GPUManagerActor)");
        Ok(())
    }
}