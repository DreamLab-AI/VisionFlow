//! Force Compute Actor - Handles physics force computation and simulation

use actix::prelude::*;
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::shared::{GPUOperation, GPUState, SharedGPUContext};
use crate::actors::messages::*;
use crate::models::simulation_params::SimulationParams;
use crate::telemetry::agent_telemetry::{
    get_telemetry_logger, CorrelationId, LogLevel, TelemetryEvent,
};
use crate::utils::socket_flow_messages::{glam_to_vec3data, BinaryNodeDataClient};
use crate::utils::unified_gpu_compute::ComputeMode;
use crate::utils::unified_gpu_compute::SimParams;
use crate::gpu::broadcast_optimizer::{BroadcastConfig, BroadcastOptimizer};
use glam::Vec3;

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsStats {
    pub iteration_count: u32,
    pub gpu_failure_count: u32,
    pub current_params: SimulationParams,
    pub compute_mode: ComputeMode,
    pub nodes_count: u32,
    pub edges_count: u32,

    
    pub average_velocity: f32,
    pub kinetic_energy: f32,
    pub total_forces: f32,

    
    pub last_step_duration_ms: f32,
    pub fps: f32,

    
    pub num_edges: u32,
    pub total_force_calculations: u32,
}

///
pub struct ForceComputeActor {

    gpu_state: GPUState,


    shared_context: Option<Arc<SharedGPUContext>>,


    simulation_params: SimulationParams,


    unified_params: SimParams,


    compute_mode: ComputeMode,


    last_step_start: Option<Instant>,
    last_step_duration_ms: f32,


    is_computing: bool,


    skipped_frames: u32,



    reheat_factor: f32,


    stability_iterations: u32,


    graph_service_addr: Option<Addr<crate::actors::GraphServiceSupervisor>>,


    ontology_constraint_addr: Option<Addr<super::ontology_constraint_actor::OntologyConstraintActor>>,

    /// Cached constraint buffer from OntologyConstraintActor for GPU upload
    cached_constraint_buffer: Vec<crate::models::constraints::ConstraintData>,

    /// Semantic forces actor for DAG layout, type clustering, and collision
    semantic_forces_addr: Option<Addr<super::semantic_forces_actor::SemanticForcesActor>>,

    /// Broadcast optimizer for delta compression and spatial culling
    broadcast_optimizer: BroadcastOptimizer,
}

impl ForceComputeActor {
    pub fn new() -> Self {
        // Initialize broadcast optimizer with default config
        let broadcast_config = BroadcastConfig {
            target_fps: 25, // 25fps broadcast, 60fps physics
            delta_threshold: 0.01, // 1cm movement threshold
            enable_spatial_culling: false, // Disabled by default, can be enabled via API
            camera_bounds: None,
        };

        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            simulation_params: SimulationParams::default(),
            unified_params: SimParams::default(),
            compute_mode: ComputeMode::Basic,
            last_step_start: None,
            last_step_duration_ms: 0.0,
            is_computing: false,
            skipped_frames: 0,
            reheat_factor: 0.0,
            stability_iterations: 0,
            graph_service_addr: None,
            ontology_constraint_addr: None,
            cached_constraint_buffer: Vec::new(),
            semantic_forces_addr: None,
            broadcast_optimizer: BroadcastOptimizer::new(broadcast_config),
        }
    }

    
    fn perform_force_computation(&mut self) -> Result<(), String> {
        
        if self.gpu_state.is_gpu_overloaded() {
            self.skipped_frames += 1;
            if self.skipped_frames % 60 == 0 {
                info!("ForceComputeActor: Skipped {} frames due to GPU overload (utilization: {:.1}%, concurrent ops: {})",
                      self.skipped_frames, self.gpu_state.get_average_utilization(), self.gpu_state.concurrent_access_count);
            }
            return Ok(()); 
        }

        
        if self.is_computing {
            self.skipped_frames += 1;
            if self.skipped_frames % 60 == 0 {
                info!(
                    "ForceComputeActor: Skipped {} frames due to ongoing GPU computation",
                    self.skipped_frames
                );
            }
            return Ok(()); 
        }

        self.is_computing = true;

        
        self.gpu_state
            .start_operation(GPUOperation::ForceComputation);

        
        let step_start = Instant::now();
        let correlation_id = CorrelationId::new();
        let iteration = self.iteration_count();

        if iteration % 60 == 0 {
            
            info!(
                "ForceComputeActor: Computing forces (iteration {}), nodes: {}",
                iteration, self.gpu_state.num_nodes
            );
        }

        
        if let Some(logger) = get_telemetry_logger() {
            let event = TelemetryEvent::new(
                correlation_id.clone(),
                LogLevel::DEBUG,
                "gpu_compute",
                "force_computation_start",
                &format!(
                    "Starting force computation iteration {} for {} nodes",
                    iteration, self.gpu_state.num_nodes
                ),
                "force_compute_actor",
            )
            .with_metadata("iteration", serde_json::json!(iteration))
            .with_metadata("node_count", serde_json::json!(self.gpu_state.num_nodes))
            .with_metadata("edge_count", serde_json::json!(self.gpu_state.num_edges))
            .with_metadata(
                "compute_mode",
                serde_json::json!(format!("{:?}", self.compute_mode)),
            );

            logger.log_event(event);
        }

        // Apply ontology constraints before acquiring shared context
        // This avoids borrow conflicts since apply_ontology_forces needs mutable access
        {
            if let Err(e) = self.apply_ontology_forces() {
                warn!("ForceComputeActor: Failed to apply ontology forces: {}", e);
            }
        }


        let shared_context = match &self.shared_context {
            Some(ctx) => ctx,
            None => {
                let error_msg = "GPU context not initialized".to_string();


                if let Some(logger) = get_telemetry_logger() {
                    let event = TelemetryEvent::new(
                        correlation_id.clone(),
                        LogLevel::ERROR,
                        "gpu_compute",
                        "context_not_initialized",
                        &error_msg,
                        "force_compute_actor",
                    )
                    .with_metadata("iteration", serde_json::json!(iteration));

                    logger.log_event(event);
                }

                self.is_computing = false;
                self.gpu_state
                    .complete_operation(&GPUOperation::ForceComputation);
                return Err(error_msg);
            }
        };



        let _gpu_guard =
            futures::executor::block_on(shared_context.acquire_gpu_access()).map_err(|e| {
                let error_msg = format!("Failed to acquire GPU lock: {}", e);


                if let Some(logger) = get_telemetry_logger() {
                    let event = TelemetryEvent::new(
                        correlation_id.clone(),
                        LogLevel::ERROR,
                        "gpu_compute",
                        "exclusive_lock_acquisition_failed",
                        &error_msg,
                        "force_compute_actor",
                    )
                    .with_metadata("error_type", serde_json::json!("exclusive_lock_failed"))
                    .with_metadata("iteration", serde_json::json!(iteration));

                    logger.log_event(event);
                }

                self.is_computing = false;
                self.gpu_state
                    .complete_operation(&GPUOperation::ForceComputation);
                error_msg
            })?;

        let mut unified_compute = shared_context.unified_compute.lock().map_err(|e| {
            let error_msg = format!("Failed to acquire GPU compute lock: {}", e);


            if let Some(logger) = get_telemetry_logger() {
                let event = TelemetryEvent::new(
                    correlation_id.clone(),
                    LogLevel::ERROR,
                    "gpu_compute",
                    "lock_acquisition_failed",
                    &error_msg,
                    "force_compute_actor",
                )
                .with_metadata("error_type", serde_json::json!("mutex_lock_failed"))
                .with_metadata("iteration", serde_json::json!(iteration));

                logger.log_event(event);
            }

            self.is_computing = false;
            self.gpu_state
                .complete_operation(&GPUOperation::ForceComputation);
            error_msg
        })?;


        let mut current_unified_params = self.unified_params.clone();
        self.sync_simulation_to_unified_params(&mut current_unified_params);



        let _sim_params_with_reheat = self.simulation_params.clone();
        if self.reheat_factor > 0.0 {
            info!(
                "Reheating physics with factor {:.2} to break equilibrium after parameter change",
                self.reheat_factor
            );

            self.stability_iterations = 0;


        }

        let sim_params = &self.simulation_params;
        let gpu_result = unified_compute.execute_physics_step(sim_params);


        if self.reheat_factor > 0.0 {
            self.reheat_factor = 0.0;
        }


        self.stability_iterations += 1;

        let execution_duration = step_start.elapsed().as_secs_f64() * 1000.0;
        self.last_step_duration_ms = execution_duration as f32;

        match gpu_result {
            Ok(_) => {

                let gpu_utilization = self.calculate_gpu_utilization(execution_duration);
                self.gpu_state.record_utilization(gpu_utilization);


                if let Err(e) = shared_context.update_utilization(gpu_utilization) {
                    log::warn!("Failed to update shared GPU utilization metrics: {}", e);
                }

                
                if let Some(logger) = get_telemetry_logger() {
                    
                    let gpu_memory_mb = (self.gpu_state.num_nodes as f32 * 48.0 +
                                        self.gpu_state.num_edges as f32 * 24.0) / (1024.0 * 1024.0);

                    logger.log_gpu_execution(
                        "force_computation_kernel",
                        self.gpu_state.num_nodes,
                        execution_duration,
                        gpu_memory_mb
                    );

                    
                    if iteration % 300 == 0 { 
                        let event = TelemetryEvent::new(
                            correlation_id,
                            LogLevel::TRACE,
                            "position_tracking",
                            "gpu_position_update",
                            &format!("GPU force computation completed for {} nodes at iteration {} (utilization: {:.1}%)",
                                   self.gpu_state.num_nodes, iteration, gpu_utilization),
                            "force_compute_actor"
                        )
                        .with_metadata("execution_time_ms", serde_json::json!(execution_duration))
                        .with_metadata("nodes_processed", serde_json::json!(self.gpu_state.num_nodes))
                        .with_metadata("compute_mode", serde_json::json!(format!("{:?}", self.compute_mode)))
                        .with_metadata("gpu_utilization_percent", serde_json::json!(gpu_utilization))
                        .with_metadata("concurrent_ops", serde_json::json!(self.gpu_state.concurrent_access_count))
                        .with_metadata("average_utilization", serde_json::json!(self.gpu_state.get_average_utilization()));

                        logger.log_event(event);
                    }
                }


                // Download positions from GPU and apply broadcast optimization
                // The broadcast optimizer handles:
                // - Adaptive broadcast frequency (25fps instead of 60fps)
                // - Delta compression (only send nodes that moved > threshold)
                // - Spatial culling (if enabled)

                let positions_result = unified_compute.get_node_positions();
                let velocities_result = unified_compute.get_node_velocities();

                if let (Ok((pos_x, pos_y, pos_z)), Ok((vel_x, vel_y, vel_z))) =
                    (positions_result, velocities_result) {

                    // Prepare position/velocity data for optimizer
                    let mut positions_velocities = Vec::with_capacity(pos_x.len());
                    let mut node_ids = Vec::with_capacity(pos_x.len());

                    for i in 0..pos_x.len() {
                        let position = Vec3::new(pos_x[i], pos_y[i], pos_z[i]);
                        let velocity = Vec3::new(vel_x[i], vel_y[i], vel_z[i]);
                        positions_velocities.push((position, velocity));
                        node_ids.push(i as u32);
                    }

                    // Apply broadcast optimization
                    let (should_broadcast, filtered_indices) =
                        self.broadcast_optimizer.process_frame(&positions_velocities, &node_ids);

                    if should_broadcast && !filtered_indices.is_empty() {
                        // Build node updates for only the filtered indices
                        let mut node_updates = Vec::with_capacity(filtered_indices.len());
                        for &idx in &filtered_indices {
                            let node_id = node_ids[idx];
                            let (position, velocity) = positions_velocities[idx];

                            node_updates.push((node_id, BinaryNodeDataClient::new(
                                node_id,
                                glam_to_vec3data(position),
                                glam_to_vec3data(velocity),
                            )));
                        }

                        if let Some(ref graph_addr) = self.graph_service_addr {
                            graph_addr.do_send(crate::actors::messages::UpdateNodePositions {
                                positions: node_updates,
                                correlation_id: None,
                            });

                            // Log bandwidth savings every 5 seconds
                            if iteration % 300 == 0 {
                                let stats = self.broadcast_optimizer.get_performance_stats();
                                info!("ForceComputeActor: Broadcast stats - Total nodes: {}, Sent: {}, Reduction: {:.1}%",
                                      pos_x.len(), filtered_indices.len(), stats.average_bandwidth_reduction);
                            }
                        } else if iteration % 60 == 0 {
                            log::warn!("ForceComputeActor: No GraphServiceActor address - positions not being sent to clients!");
                        }
                    } else if should_broadcast && iteration % 300 == 0 {
                        // Log when no updates needed (high compression)
                        debug!("ForceComputeActor: No position updates needed (all nodes stable)");
                    }
                } else {
                    error!("ForceComputeActor: Failed to download positions/velocities from GPU");
                }

                Ok(())
            },
            Err(e) => {
                let error_msg = format!("GPU force computation failed: {}", e);

                
                if let Some(logger) = get_telemetry_logger() {
                    let event = TelemetryEvent::new(
                        correlation_id,
                        LogLevel::ERROR,
                        "gpu_compute",
                        "force_computation_failed",
                        &error_msg,
                        "force_compute_actor"
                    )
                    .with_gpu_info("force_computation_kernel", execution_duration, 0.0)
                    .with_metadata("iteration", serde_json::json!(iteration))
                    .with_metadata("node_count", serde_json::json!(self.gpu_state.num_nodes))
                    .with_metadata("error_message", serde_json::json!(e.to_string()));

                    logger.log_event(event);
                }

                self.is_computing = false; 
                Err(error_msg)
            }
        }
            .map_err(|e| {
                error!("GPU force computation failed: {}", e);
                self.gpu_state.gpu_failure_count += 1;
                self.is_computing = false; 
                format!("Force computation failed: {}", e)
            })?;

        
        self.gpu_state.iteration_count += 1;

        
        self.last_step_duration_ms = step_start.elapsed().as_millis() as f32;

        
        if self.iteration_count() % 300 == 0 {
            
            info!("ForceComputeActor: {} iterations completed, {} GPU failures, {} skipped frames, last step: {:.2}ms",
                  self.iteration_count(), self.gpu_state.gpu_failure_count, self.skipped_frames, self.last_step_duration_ms);
        }

        
        self.is_computing = false;

        Ok(())
    }

    
    fn sync_simulation_to_unified_params(&self, unified_params: &mut SimParams) {
        
        unified_params.spring_k = self.simulation_params.spring_k;
        unified_params.repel_k = self.simulation_params.repel_k;
        unified_params.damping = self.simulation_params.damping;
        unified_params.dt = self.simulation_params.dt;
        unified_params.max_velocity = self.simulation_params.max_velocity;
        unified_params.center_gravity_k = self.simulation_params.center_gravity_k;

        
        match self.compute_mode {
            ComputeMode::Basic => {
                
                
            }
            ComputeMode::Advanced => {
                
                
                unified_params.temperature = self.simulation_params.temperature;
                unified_params.alignment_strength = self.simulation_params.alignment_strength;
                unified_params.cluster_strength = self.simulation_params.cluster_strength;
            }
            ComputeMode::DualGraph => {
                
                
                unified_params.temperature = self.simulation_params.temperature;
                unified_params.alignment_strength = self.simulation_params.alignment_strength;
                unified_params.cluster_strength = self.simulation_params.cluster_strength;
            }
            ComputeMode::Constraints => {
                
                unified_params.temperature = self.simulation_params.temperature;
                unified_params.alignment_strength = self.simulation_params.alignment_strength;
                unified_params.cluster_strength = self.simulation_params.cluster_strength;
                unified_params.constraint_ramp_frames =
                    self.simulation_params.constraint_ramp_frames;
                unified_params.constraint_max_force_per_node =
                    self.simulation_params.constraint_max_force_per_node;
            }
        }

        trace!("Unified params updated: spring_k={:.3}, repel_k={:.3}, center_gravity_k={:.3}, damping={:.3}",
               unified_params.spring_k, unified_params.repel_k, unified_params.center_gravity_k, unified_params.damping);
    }

    
    fn iteration_count(&self) -> u32 {
        self.gpu_state.iteration_count
    }

    
    fn update_simulation_parameters(&mut self, params: SimulationParams) {
        info!("ForceComputeActor: Updating simulation parameters");
        info!(
            "  spring_k: {:.3} -> {:.3}",
            self.simulation_params.spring_k, params.spring_k
        );
        info!(
            "  repel_k: {:.3} -> {:.3}",
            self.simulation_params.repel_k, params.repel_k
        );
        info!(
            "  damping: {:.3} -> {:.3}",
            self.simulation_params.damping, params.damping
        );

        self.simulation_params = params;

        
        {
            let unified_params = &mut self.unified_params;
            unified_params.spring_k = self.simulation_params.spring_k;
            unified_params.repel_k = self.simulation_params.repel_k;
            unified_params.damping = self.simulation_params.damping;
            unified_params.dt = self.simulation_params.dt;
        }
    }

    
    fn get_physics_stats(&self) -> PhysicsStats {
        
        let (average_velocity, kinetic_energy, total_forces) = self.calculate_physics_metrics();

        
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

            
            average_velocity,
            kinetic_energy,
            total_forces,

            
            last_step_duration_ms: self.last_step_duration_ms,
            fps,

            
            num_edges: self.gpu_state.num_edges,
            total_force_calculations: self.gpu_state.iteration_count * self.gpu_state.num_nodes,
        }
    }

    
    fn calculate_physics_metrics(&self) -> (f32, f32, f32) {
        
        if let Some(ctx) = &self.shared_context {
            if let Ok(unified_compute) = ctx.unified_compute.lock() {
                return self.extract_gpu_metrics(&*unified_compute);
            }
        }

        
        let estimated_velocity = self.simulation_params.max_velocity * 0.3; 
        let estimated_kinetic_energy =
            0.5 * (self.gpu_state.num_nodes as f32) * estimated_velocity.powi(2);
        let estimated_total_forces =
            self.simulation_params.spring_k * (self.gpu_state.num_edges as f32) * 0.5;

        (
            estimated_velocity,
            estimated_kinetic_energy,
            estimated_total_forces,
        )
    }

    
    fn extract_gpu_metrics(
        &self,
        unified_compute: &crate::utils::unified_gpu_compute::UnifiedGPUCompute,
    ) -> (f32, f32, f32) {
        let num_nodes = unified_compute.num_nodes;

        
        let mut vel_x = vec![0.0f32; num_nodes];
        let mut vel_y = vec![0.0f32; num_nodes];
        let mut vel_z = vec![0.0f32; num_nodes];

        
        if unified_compute
            .download_velocities(&mut vel_x, &mut vel_y, &mut vel_z)
            .is_ok()
        {
            
            let total_velocity: f32 = vel_x
                .iter()
                .zip(&vel_y)
                .zip(&vel_z)
                .map(|((vx, vy), vz)| (vx * vx + vy * vy + vz * vz).sqrt())
                .sum();
            let average_velocity = if num_nodes > 0 {
                total_velocity / num_nodes as f32
            } else {
                0.0
            };

            
            let kinetic_energy: f32 = vel_x
                .iter()
                .zip(&vel_y)
                .zip(&vel_z)
                .map(|((vx, vy), vz)| 0.5 * (vx * vx + vy * vy + vz * vz))
                .sum();

            
            let estimated_total_forces =
                total_velocity * self.simulation_params.damping * num_nodes as f32;

            (average_velocity, kinetic_energy, estimated_total_forces)
        } else {
            
            let estimated_velocity = self.simulation_params.max_velocity * 0.3;
            let estimated_kinetic_energy = 0.5 * (num_nodes as f32) * estimated_velocity.powi(2);
            let estimated_total_forces =
                self.simulation_params.spring_k * (self.gpu_state.num_edges as f32) * 0.5;

            (
                estimated_velocity,
                estimated_kinetic_energy,
                estimated_total_forces,
            )
        }
    }

    

    fn calculate_gpu_utilization(&self, execution_time_ms: f64) -> f32 {

        const TARGET_FRAME_TIME_MS: f64 = 16.67;


        let utilization_percent = (execution_time_ms / TARGET_FRAME_TIME_MS * 100.0) as f32;


        utilization_percent.min(100.0).max(0.0)
    }

    /// Apply ontology-derived constraint forces to the physics simulation
    ///
    /// This method integrates ontology constraints from the OntologyConstraintActor
    /// into the physics pipeline, enabling semantic relationships to influence node positions.
    ///
    /// # Implementation Notes
    ///
    /// This is the final integration point for P0-2 ontology constraints. It:
    /// 1. Retrieves constraint buffer from OntologyConstraintActor (via shared memory/coordination)
    /// 2. Uploads constraints to GPU via UnifiedGPUCompute::upload_constraints()
    /// 3. Constraints are automatically applied during execute_physics_step()
    ///
    /// The constraint buffer contains ConstraintData structs generated from OWL axioms
    /// by OntologyConstraintTranslator, which are processed by ontology_constraints.cu kernels.
    fn apply_ontology_forces(&mut self) -> Result<(), String> {
        trace!("ForceComputeActor: Applying ontology constraint forces");

        // Check if we have a shared context with access to the GPU compute system
        let shared_context = match &self.shared_context {
            Some(ctx) => ctx,
            None => {
                trace!("ForceComputeActor: No shared context available for ontology forces");
                return Ok(()); // Not an error, just not available yet
            }
        };

        // Use the cached constraint buffer (updated via UpdateOntologyConstraintBuffer message)
        let constraint_buffer = &self.cached_constraint_buffer;

        // Skip if no constraints to apply
        if constraint_buffer.is_empty() {
            trace!("ForceComputeActor: No ontology constraints to apply");
            return Ok(());
        }

        // Access unified compute system and upload constraints to GPU
        let mut unified_compute = shared_context
            .unified_compute
            .lock()
            .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?;

        // Upload constraints to GPU - this is the critical integration point
        // The upload_constraints method:
        // 1. Converts ConstraintData to GPU-compatible format
        // 2. Allocates/updates constraint buffer on GPU
        // 3. Prepares constraints for processing by ontology_constraints.cu kernels
        unified_compute
            .upload_constraints(constraint_buffer)
            .map_err(|e| format!("Failed to upload ontology constraints to GPU: {}", e))?;

        debug!(
            "ForceComputeActor: Uploaded {} ontology constraints to GPU",
            constraint_buffer.len()
        );

        // Constraints are now on GPU and will be automatically applied
        // during the next execute_physics_step() call
        trace!("ForceComputeActor: Ontology constraint upload complete");
        Ok(())
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
        info!(
            "  New params - spring_k: {:.3}, repel_k: {:.3}, damping: {:.3}",
            msg.params.spring_k, msg.params.repel_k, msg.params.damping
        );

        
        self.update_simulation_parameters(msg.params);

        
        
        
        let previous_iteration = self.gpu_state.iteration_count;
        self.gpu_state.iteration_count = 0;

        
        self.stability_iterations = 0;

        
        
        self.reheat_factor = 0.3;

        info!(
            "ForceComputeActor: Reset iteration counter from {} to 0 to restart physics",
            previous_iteration
        );
        info!("ForceComputeActor: Stability gate will allow physics to run for at least 600 iterations");

        Ok(())
    }
}

impl Handler<SetComputeMode> for ForceComputeActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: SetComputeMode, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: Setting compute mode to {:?}", msg.mode);

        self.compute_mode = msg.mode;

        
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

        
        
        if msg.params.semantic_force_weight > 0.0 {
            self.unified_params.temperature *= msg.params.semantic_force_weight;
        }

        
        if msg.params.temporal_force_weight > 0.0 {
            self.unified_params.alignment_strength *= msg.params.temporal_force_weight;
        }

        
        if msg.params.constraint_force_weight > 0.0 {
            self.unified_params.cluster_strength *= msg.params.constraint_force_weight;
        }

        info!("Advanced physics parameters applied to unified compute params");

        
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
        info!(
            "ForceComputeActor: UploadPositions received - {} nodes",
            msg.positions_x.len()
        );

        let mut unified_compute = match &self.shared_context {
            Some(ctx) => ctx
                .unified_compute
                .lock()
                .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?,
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };

        
        unified_compute
            .update_positions_only(&msg.positions_x, &msg.positions_y, &msg.positions_z)
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


        self.gpu_state.num_nodes = msg.graph.nodes.len() as u32;
        self.gpu_state.num_edges = msg.graph.edges.len() as u32;


        if msg.graph_service_addr.is_some() {
            self.graph_service_addr = msg.graph_service_addr;
            info!("ForceComputeActor: GraphServiceActor address stored for position updates");
        }

        info!(
            "ForceComputeActor: GPU initialized with {} nodes, {} edges",
            self.gpu_state.num_nodes, self.gpu_state.num_edges
        );

        // H4: Send acknowledgment
        if let Some(correlation_id) = msg.correlation_id {
            use crate::actors::messaging::MessageAck;
            if let Some(ref orchestrator_addr) = msg.physics_orchestrator_addr {
                orchestrator_addr.do_send(MessageAck::success(correlation_id)
                    .with_metadata("nodes", self.gpu_state.num_nodes.to_string())
                    .with_metadata("edges", self.gpu_state.num_edges.to_string()));
            }
        }

        Ok(())
    }
}

impl Handler<UpdateGPUGraphData> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGPUGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: UpdateGPUGraphData received");


        self.gpu_state.num_nodes = msg.graph.nodes.len() as u32;
        self.gpu_state.num_edges = msg.graph.edges.len() as u32;

        info!(
            "ForceComputeActor: Graph data updated - {} nodes, {} edges",
            self.gpu_state.num_nodes, self.gpu_state.num_edges
        );

        // H4: Send acknowledgment
        if let Some(correlation_id) = msg.correlation_id {
            use crate::actors::messaging::MessageAck;
            // Note: We don't have a direct physics orchestrator reference here,
            // but acknowledgments can still be sent if the reference is added in the future
            // For now, this demonstrates the pattern
            debug!("UpdateGPUGraphData completed with correlation_id: {}", correlation_id);
        }

        Ok(())
    }
}

impl Handler<GetNodeData> for ForceComputeActor {
    type Result = Result<Vec<crate::utils::socket_flow_messages::BinaryNodeData>, String>;

    fn handle(&mut self, _msg: GetNodeData, _ctx: &mut Self::Context) -> Self::Result {
        
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
        
        Err("Community detection should be handled by ClusteringActor".to_string())
    }
}

impl Handler<UpdateVisualAnalyticsParams> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: UpdateVisualAnalyticsParams,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("ForceComputeActor: UpdateVisualAnalyticsParams received (no-op, handled by other actors)");
        Ok(())
    }
}

impl Handler<GetConstraints> for ForceComputeActor {
    type Result = Result<crate::models::constraints::ConstraintSet, String>;

    fn handle(&mut self, _msg: GetConstraints, _ctx: &mut Self::Context) -> Self::Result {
        
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

    fn handle(
        &mut self,
        _msg: TriggerStressMajorization,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        
        Err("Stress majorization should be handled by StressMajorizationActor".to_string())
    }
}

impl Handler<GetStressMajorizationStats> for ForceComputeActor {
    type Result =
        Result<crate::actors::gpu::stress_majorization_actor::StressMajorizationStats, String>;

    fn handle(
        &mut self,
        _msg: GetStressMajorizationStats,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        
        Err(
            "Stress majorization stats should be retrieved from StressMajorizationActor"
                .to_string(),
        )
    }
}

impl Handler<ResetStressMajorizationSafety> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: ResetStressMajorizationSafety,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        
        Err(
            "Stress majorization safety reset should be handled by StressMajorizationActor"
                .to_string(),
        )
    }
}

impl Handler<UpdateStressMajorizationParams> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: UpdateStressMajorizationParams,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("ForceComputeActor: UpdateStressMajorizationParams received (forwarding to StressMajorizationActor would be done by GPUManagerActor)");
        Ok(())
    }
}

impl Handler<PerformGPUClustering> for ForceComputeActor {
    type Result = Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>;

    fn handle(&mut self, _msg: PerformGPUClustering, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: PerformGPUClustering received - forwarding to ClusteringActor would be done by GPUManagerActor");
        
        
        Err("Clustering should be handled by ClusteringActor, not ForceComputeActor".to_string())
    }
}

impl Handler<GetClusteringResults> for ForceComputeActor {
    type Result = Result<serde_json::Value, String>;

    fn handle(&mut self, _msg: GetClusteringResults, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: GetClusteringResults received - forwarding to ClusteringActor would be done by GPUManagerActor");


        Err(
            "Clustering results should be retrieved from ClusteringActor, not ForceComputeActor"
                .to_string(),
        )
    }
}

/// Handler for UpdateOntologyConstraintBuffer
/// Updates the cached constraint buffer when ontology constraints change
impl Handler<crate::actors::messages::UpdateOntologyConstraintBuffer> for ForceComputeActor {
    type Result = ();

    fn handle(&mut self, msg: crate::actors::messages::UpdateOntologyConstraintBuffer, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: Received updated ontology constraint buffer with {} constraints",
              msg.constraint_buffer.len());

        // Update the cached constraint buffer
        self.cached_constraint_buffer = msg.constraint_buffer;

        debug!("ForceComputeActor: Ontology constraint buffer cached, will be uploaded to GPU on next physics step");
    }
}

///
impl Handler<SetSharedGPUContext> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: Received SharedGPUContext from ResourceActor");


        self.shared_context = Some(msg.context);


        if let Some(addr) = msg.graph_service_addr {
            self.graph_service_addr = Some(addr);
            info!("ForceComputeActor: GraphServiceActor address stored - position updates will be sent to clients!");
        } else {
            warn!("ForceComputeActor: No GraphServiceActor address provided - positions won't be sent to clients");
        }


        self.gpu_state.is_initialized = true;

        info!("ForceComputeActor: SharedGPUContext stored successfully - GPU physics enabled!");
        info!(
            "ForceComputeActor: Physics can now run with {} nodes and {} edges",
            self.gpu_state.num_nodes, self.gpu_state.num_edges
        );

        // H4: Send acknowledgment
        if let Some(correlation_id) = msg.correlation_id {
            use crate::actors::messaging::MessageAck;
            debug!("SetSharedGPUContext completed with correlation_id: {}", correlation_id);
            // Note: Future enhancement - send ack to physics orchestrator if reference available
        }

        Ok(())
    }
}

/// Handler for ConfigureStressMajorization message
impl Handler<ConfigureStressMajorization> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ConfigureStressMajorization, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: ConfigureStressMajorization received");

        // Store stress majorization configuration in unified params
        // These parameters affect graph layout optimization
        if let Some(learning_rate) = msg.learning_rate {
            info!("  Setting learning_rate: {:.3}", learning_rate);
            // Apply learning rate to temperature for optimization
            self.unified_params.temperature = learning_rate * 100.0;
        }

        if let Some(momentum) = msg.momentum {
            info!("  Setting momentum: {:.3}", momentum);
            // Momentum affects velocity damping
            self.unified_params.damping = 1.0 - momentum;
        }

        if let Some(max_iterations) = msg.max_iterations {
            info!("  Setting max_iterations: {}", max_iterations);
            // This would be used by stress majorization algorithm
            // For now, we log it as it affects the optimization convergence
        }

        if let Some(auto_run_interval) = msg.auto_run_interval {
            info!("  Setting auto_run_interval: {} frames", auto_run_interval);
            // Auto-run interval affects periodic layout optimization
        }

        info!("ForceComputeActor: Stress majorization configuration applied");
        Ok(())
    }
}

/// Handler for GetStressMajorizationConfig message
impl Handler<GetStressMajorizationConfig> for ForceComputeActor {
    type Result = Result<StressMajorizationConfig, String>;

    fn handle(&mut self, _msg: GetStressMajorizationConfig, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: GetStressMajorizationConfig received");

        // Return current stress majorization configuration based on unified params
        let config = StressMajorizationConfig {
            learning_rate: self.unified_params.temperature / 100.0,
            momentum: 1.0 - self.unified_params.damping,
            max_iterations: 100, // Default value
            auto_run_interval: 60, // Default: every 60 frames
            current_stress: 0.0, // Would be computed from current layout
            converged: self.stability_iterations > 600, // Converged after stability
            iterations_completed: self.gpu_state.iteration_count as usize,
        };

        info!("ForceComputeActor: Returning stress majorization config (learning_rate: {:.3}, momentum: {:.3})",
              config.learning_rate, config.momentum);

        Ok(config)
    }
}

// =============================================================================
// Phase 7: Broadcast Optimization Message Handlers
// =============================================================================

/// Handler for ConfigureBroadcastOptimization
impl Handler<crate::actors::messages::ConfigureBroadcastOptimization> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: crate::actors::messages::ConfigureBroadcastOptimization, _ctx: &mut Self::Context) -> Self::Result {
        info!("ForceComputeActor: ConfigureBroadcastOptimization received");

        // Get current stats before update
        let old_stats = self.broadcast_optimizer.get_performance_stats();

        // Build new config from current + updates
        let mut new_config = BroadcastConfig {
            target_fps: msg.target_fps.unwrap_or(old_stats.target_fps),
            delta_threshold: msg.delta_threshold.unwrap_or(old_stats.delta_threshold),
            enable_spatial_culling: msg.enable_spatial_culling.unwrap_or(false),
            camera_bounds: None, // Updated separately via UpdateCameraFrustum
        };

        // Validate parameters
        if new_config.target_fps == 0 || new_config.target_fps > 60 {
            return Err(format!("Invalid target_fps: {} (must be 1-60)", new_config.target_fps));
        }

        if new_config.delta_threshold < 0.0 {
            return Err(format!("Invalid delta_threshold: {} (must be >= 0.0)", new_config.delta_threshold));
        }

        info!("  Target FPS: {} -> {}", old_stats.target_fps, new_config.target_fps);
        info!("  Delta threshold: {:.4} -> {:.4}", old_stats.delta_threshold, new_config.delta_threshold);
        info!("  Spatial culling: {}", new_config.enable_spatial_culling);

        // Apply new configuration
        self.broadcast_optimizer.update_config(new_config);

        Ok(())
    }
}

/// Handler for UpdateCameraFrustum
impl Handler<crate::actors::messages::UpdateCameraFrustum> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: crate::actors::messages::UpdateCameraFrustum, _ctx: &mut Self::Context) -> Self::Result {
        debug!("ForceComputeActor: UpdateCameraFrustum received - min: {:?}, max: {:?}",
               msg.min, msg.max);

        let min = Vec3::new(msg.min.0, msg.min.1, msg.min.2);
        let max = Vec3::new(msg.max.0, msg.max.1, msg.max.2);
        self.broadcast_optimizer.update_camera_bounds(min, max);
        Ok(())
    }
}

/// Handler for GetBroadcastStats
impl Handler<crate::actors::messages::GetBroadcastStats> for ForceComputeActor {
    type Result = Result<crate::actors::messages::BroadcastPerformanceStats, String>;

    fn handle(&mut self, _msg: crate::actors::messages::GetBroadcastStats, _ctx: &mut Self::Context) -> Self::Result {
        let stats = self.broadcast_optimizer.get_performance_stats();

        // Convert from gpu::broadcast_optimizer::BroadcastPerformanceStats
        // to actors::messages::BroadcastPerformanceStats
        Ok(crate::actors::messages::BroadcastPerformanceStats {
            total_frames_processed: stats.total_frames_processed,
            total_nodes_sent: stats.total_nodes_sent,
            total_nodes_processed: stats.total_nodes_processed,
            average_bandwidth_reduction: stats.average_bandwidth_reduction,
            target_fps: stats.target_fps,
            delta_threshold: stats.delta_threshold,
        })
    }
}
