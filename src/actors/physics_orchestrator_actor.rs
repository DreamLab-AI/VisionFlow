//! Physics Orchestrator Actor - Dedicated physics simulation management
//!
//! This actor coordinates all physics simulation activities in the VisionFlow system,
//! providing focused management of force calculations, position updates, and GPU acceleration.

use actix::prelude::*;
use actix::MessageResult;
use glam::Vec3;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use log::{debug, info, warn};

use crate::errors::VisionFlowError;
use crate::actors::messages::PositionSnapshot;

use crate::actors::gpu::force_compute_actor::ForceComputeActor;
use crate::actors::gpu::force_compute_actor::{
    PhysicsStats
};
use crate::actors::messages::{
    InitializeGPU, UpdateGPUGraphData
};
// GraphStateActor will be implemented separately - using direct graph data access
use crate::models::simulation_params::SimulationParams;
use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::models::constraints::ConstraintSet;
use crate::actors::messages::{
    StartSimulation, StopSimulation, SimulationStep, UpdateNodePosition,
    RequestPositionSnapshot, PhysicsPauseMessage, NodeInteractionMessage,
    ForceResumePhysics, StoreGPUComputeAddress, UpdateSimulationParams,
    UpdateNodePositions, ApplyOntologyConstraints, ConstraintMergeMode,
    SetConstraintGroupActive, GetConstraintStats, ConstraintStats
};

/// Physics orchestration actor responsible for managing all physics simulation
/// This actor coordinates between GPU compute, graph state, and simulation parameters
pub struct PhysicsOrchestratorActor {
    /// Whether physics simulation is currently running
    simulation_running: AtomicBool,

    /// Current physics simulation parameters
    simulation_params: SimulationParams,

    /// Target parameters for smooth transitions
    target_params: SimulationParams,

    /// GPU compute actor address for hardware acceleration
    gpu_compute_addr: Option<Addr<ForceComputeActor>>,

    /// Ontology actor address for constraint generation
    #[cfg(feature = "ontology")]
    ontology_actor_addr: Option<Addr<crate::actors::ontology_actor::OntologyActor>>,

    /// Graph data reference for physics calculations
    graph_data_ref: Option<Arc<GraphData>>,

    /// Whether GPU is initialized and ready
    gpu_initialized: bool,

    /// GPU initialization in progress flag
    gpu_init_in_progress: bool,

    /// Last physics step execution time
    last_step_time: Option<Instant>,

    /// Current physics statistics
    physics_stats: Option<PhysicsStats>,

    /// Parameter interpolation rate for smooth transitions
    param_interpolation_rate: f32,

    /// Auto-balance threshold tracking
    auto_balance_last_check: Option<Instant>,

    /// Force resume after user interaction timer
    force_resume_timer: Option<Instant>,

    /// Last known node count for equilibrium detection
    last_node_count: usize,

    /// Current iteration counter
    current_iteration: u64,

    /// Performance metrics
    performance_metrics: PhysicsPerformanceMetrics,

    /// Ontology-derived constraints
    ontology_constraints: Option<ConstraintSet>,

    /// User-defined constraints (non-ontology)
    user_constraints: Option<ConstraintSet>,
}

/// Physics performance tracking metrics
#[derive(Debug, Default, Clone)]
pub struct PhysicsPerformanceMetrics {
    pub total_steps: u64,
    pub average_step_time_ms: f32,
    pub gpu_utilization: f32,
    pub last_fps: f32,
    pub gpu_memory_usage_mb: f32,
    pub convergence_rate: f32,
}

impl PhysicsOrchestratorActor {
    /// Create a new physics orchestrator
    pub fn new(
        simulation_params: SimulationParams,
        gpu_compute_addr: Option<Addr<ForceComputeActor>>,
        graph_data: Option<Arc<GraphData>>,
    ) -> Self {
        let target_params = simulation_params.clone();

        Self {
            simulation_running: AtomicBool::new(false),
            simulation_params,
            target_params,
            gpu_compute_addr,
            #[cfg(feature = "ontology")]
            ontology_actor_addr: None,
            graph_data_ref: graph_data,
            gpu_initialized: false,
            gpu_init_in_progress: false,
            last_step_time: None,
            physics_stats: None,
            param_interpolation_rate: 0.1, // Smooth parameter transitions
            auto_balance_last_check: None,
            force_resume_timer: None,
            last_node_count: 0,
            current_iteration: 0,
            performance_metrics: PhysicsPerformanceMetrics::default(),
            ontology_constraints: None,
            user_constraints: None,
        }
    }

    /// Set the ontology actor address for constraint generation
    #[cfg(feature = "ontology")]
    pub fn set_ontology_actor(&mut self, addr: Addr<crate::actors::ontology_actor::OntologyActor>) {
        info!("PhysicsOrchestratorActor: Ontology actor address set");
        self.ontology_actor_addr = Some(addr);
    }

    /// Start the physics simulation loop
    fn start_simulation_loop(&self, ctx: &mut Context<Self>) {
        if self.simulation_running.load(Ordering::SeqCst) {
            warn!("Physics simulation already running");
            return;
        }

        self.simulation_running.store(true, Ordering::SeqCst);
        info!("Starting physics simulation loop");

        // Schedule recurring physics steps
        ctx.run_interval(Duration::from_millis(16), |act, ctx| { // ~60 FPS
            if !act.simulation_running.load(Ordering::SeqCst) {
                return; // Stop if simulation disabled
            }

            act.physics_step(ctx);
        });
    }

    /// Stop the physics simulation
    fn stop_simulation(&mut self) {
        self.simulation_running.store(false, Ordering::SeqCst);
        info!("Physics simulation stopped");
    }

    /// Execute a single physics simulation step
    fn physics_step(&mut self, ctx: &mut Context<Self>) {
        let start_time = Instant::now();

        // Check if physics is paused
        if self.simulation_params.is_physics_paused {
            self.handle_physics_paused_state(ctx);
            return;
        }

        // Update target parameter interpolation
        self.interpolate_parameters();

        // Ensure GPU is initialized
        if !self.gpu_initialized && self.gpu_compute_addr.is_some() {
            self.initialize_gpu_if_needed(ctx);
            return;
        }

        // Perform auto-balancing if enabled
        if self.simulation_params.auto_balance {
            self.perform_auto_balance_check();
        }

        // Execute physics computation
        if let Some(gpu_addr) = self.gpu_compute_addr.clone() {
            // GPU-accelerated physics step
            self.execute_gpu_physics_step(&gpu_addr, ctx);
        } else {
            // CPU fallback (if implemented)
            self.execute_cpu_physics_step(ctx);
        }

        // Update performance metrics
        let step_time = start_time.elapsed();
        self.update_performance_metrics(step_time);

        // Check for equilibrium and auto-pause
        self.check_equilibrium_and_auto_pause();

        self.last_step_time = Some(start_time);
    }

    /// Handle physics paused state interactions
    fn handle_physics_paused_state(&mut self, _ctx: &mut Context<Self>) {
        // Check for force resume timer expiration
        if let Some(resume_time) = self.force_resume_timer {
            if resume_time.elapsed() > Duration::from_millis(500) {
                self.resume_physics();
                self.force_resume_timer = None;
            }
        }
    }

    /// Smoothly interpolate simulation parameters towards target
    fn interpolate_parameters(&mut self) {
        let rate = self.param_interpolation_rate;

        // Interpolate core physics parameters
        self.simulation_params.repel_k = self.simulation_params.repel_k * (1.0 - rate)
            + self.target_params.repel_k * rate;
        self.simulation_params.damping = self.simulation_params.damping * (1.0 - rate)
            + self.target_params.damping * rate;
        self.simulation_params.max_velocity = self.simulation_params.max_velocity * (1.0 - rate)
            + self.target_params.max_velocity * rate;
        self.simulation_params.spring_k = self.simulation_params.spring_k * (1.0 - rate)
            + self.target_params.spring_k * rate;
        self.simulation_params.viewport_bounds = self.simulation_params.viewport_bounds * (1.0 - rate)
            + self.target_params.viewport_bounds * rate;

        // Additional parameter interpolations
        self.simulation_params.max_repulsion_dist = self.simulation_params.max_repulsion_dist * (1.0 - rate)
            + self.target_params.max_repulsion_dist * rate;
        self.simulation_params.boundary_force_strength = self.simulation_params.boundary_force_strength * (1.0 - rate)
            + self.target_params.boundary_force_strength * rate;
        self.simulation_params.cooling_rate = self.simulation_params.cooling_rate * (1.0 - rate)
            + self.target_params.cooling_rate * rate;

        // Boolean parameters snap immediately
        if (self.target_params.enable_bounds as i32 - self.simulation_params.enable_bounds as i32).abs() > 0 {
            self.simulation_params.enable_bounds = self.target_params.enable_bounds;
        }
    }

    /// Initialize GPU compute if available and needed
    fn initialize_gpu_if_needed(&mut self, ctx: &mut Context<Self>) {
        if self.gpu_init_in_progress || self.gpu_initialized {
            return;
        }

        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            self.gpu_init_in_progress = true;
            info!("Initializing GPU compute for physics");

            // Initialize GPU with current graph data
            if let Some(ref graph_data) = self.graph_data_ref {
                gpu_addr.do_send(InitializeGPU {
                    graph: Arc::clone(graph_data),
                    graph_service_addr: None, // Will be set later if needed
                    gpu_manager_addr: None,  // Will be filled by GPUManagerActor
                });

                // Update GPU with current graph data
                gpu_addr.do_send(UpdateGPUGraphData {
                    graph: Arc::clone(graph_data)
                });

                self.gpu_initialized = true;
                self.gpu_init_in_progress = false;
            }
        }
    }

    /// Update graph data reference
    fn update_graph_data(&mut self, graph_data: Arc<GraphData>) {
        self.graph_data_ref = Some(graph_data.clone());
        self.last_node_count = graph_data.nodes.len();
    }

    /// Execute GPU-accelerated physics step
    fn execute_gpu_physics_step(&mut self, gpu_addr: &Addr<ForceComputeActor>, _ctx: &mut Context<Self>) {
        if !self.gpu_initialized {
            return;
        }

        // Update iteration counter
        self.current_iteration += 1;
        self.performance_metrics.total_steps = self.current_iteration;

        // For now, just log the physics step - actual GPU step implementation
        // would need to be coordinated with the existing ForceComputeActor
        debug!("Physics step {} executed", self.current_iteration);
    }

    /// Handle physics step completion
    fn handle_physics_step_completion(&mut self) {
        // Update performance metrics
        debug!("Physics step {} completed", self.current_iteration);
    }

    /// Execute CPU fallback physics step
    fn execute_cpu_physics_step(&mut self, _ctx: &mut Context<Self>) {
        // CPU physics implementation would go here
        // For now, this is a placeholder that logs the fallback
        warn!("CPU physics fallback not fully implemented - using GPU compute");
    }

    /// Broadcast position updates to interested actors
    fn broadcast_position_updates(&self, _positions: Vec<(u32, BinaryNodeData)>, _ctx: &mut Context<Self>) {
        // Implementation for broadcasting position updates
        // This could notify UI, WebSocket handlers, etc.
    }

    /// Perform auto-balance parameter checking
    fn perform_auto_balance_check(&mut self) {
        let now = Instant::now();

        // Check if enough time has elapsed since last auto-balance check
        if let Some(last_check) = self.auto_balance_last_check {
            let interval = Duration::from_millis(self.simulation_params.auto_balance_interval_ms as u64);
            if now.duration_since(last_check) < interval {
                return;
            }
        }

        self.auto_balance_last_check = Some(now);

        // Perform neural auto-balancing logic
        self.neural_auto_balance();
    }

    /// Neural auto-balancing algorithm
    fn neural_auto_balance(&mut self) {
        let config = &self.simulation_params.auto_balance_config;

        // Get current physics stats for decision making
        if let Some(ref stats) = self.physics_stats {
            let mut new_target = self.target_params.clone();

            // Analyze kinetic energy and adjust parameters
            if stats.kinetic_energy > 1000.0 { // Placeholder threshold
                // High energy - increase damping, reduce forces
                let damping_factor = 1.0 + config.min_adjustment_factor;
                let force_factor = 1.0 - config.max_adjustment_factor;

                new_target.damping = (self.simulation_params.damping * damping_factor).min(0.99);
                new_target.repel_k = self.simulation_params.repel_k * force_factor;

                info!("Auto-balance: Reducing forces due to high energy");
            } else if stats.kinetic_energy < 10.0 { // Placeholder threshold
                // Low energy - decrease damping, increase forces
                let damping_factor = 1.0 - config.min_adjustment_factor;
                let force_factor = 1.0 + config.max_adjustment_factor;

                new_target.damping = (self.simulation_params.damping * damping_factor).max(0.1);
                new_target.repel_k = self.simulation_params.repel_k * force_factor;

                info!("Auto-balance: Increasing forces due to low energy");
            }

            // Apply clustering-based adjustments
            if stats.kinetic_energy < config.clustering_distance_threshold {
                // Low clustering - increase attraction
                new_target.spring_k = self.simulation_params.spring_k * (1.0 + config.min_adjustment_factor);
            }

            // Update target parameters
            self.target_params = new_target;
        }
    }

    /// Check for equilibrium conditions and auto-pause if configured
    fn check_equilibrium_and_auto_pause(&mut self) {
        let node_count = self.graph_data_ref.as_ref().map(|g| g.nodes.len()).unwrap_or(0);

        if !self.simulation_params.auto_pause_config.enabled || node_count == 0 {
            return;
        }

        let config = &self.simulation_params.auto_pause_config;

        // Check if system is in equilibrium based on kinetic energy
        let is_equilibrium = if let Some(ref stats) = self.physics_stats {
            stats.kinetic_energy < config.equilibrium_energy_threshold
        } else {
            false
        };

        if is_equilibrium {
            self.simulation_params.equilibrium_stability_counter += 1;

            // Check if equilibrium has been stable long enough
            if self.simulation_params.equilibrium_stability_counter >= config.equilibrium_check_frames {
                if !self.simulation_params.is_physics_paused && config.pause_on_equilibrium {
                    info!("Auto-pause: System reached equilibrium, pausing physics");
                    self.simulation_params.is_physics_paused = true;

                    // Broadcast pause event
                    self.broadcast_physics_paused();
                }
            }
        } else {
            // Reset counter if not in equilibrium
            if !self.simulation_params.is_physics_paused {
                self.simulation_params.equilibrium_stability_counter = 0;
            }
        }
    }

    /// Resume physics simulation
    fn resume_physics(&mut self) {
        if self.simulation_params.is_physics_paused {
            self.simulation_params.is_physics_paused = false;
            self.simulation_params.equilibrium_stability_counter = 0;
            info!("Physics simulation resumed");

            // Broadcast resume event
            self.broadcast_physics_resumed();
        }
    }

    /// Broadcast physics paused event
    fn broadcast_physics_paused(&self) {
        // Implementation for notifying other actors about physics pause
        debug!("Broadcasting physics paused event");
    }

    /// Broadcast physics resumed event
    fn broadcast_physics_resumed(&self) {
        // Implementation for notifying other actors about physics resume
        debug!("Broadcasting physics resumed event");
    }

    /// Update performance metrics based on step execution
    fn update_performance_metrics(&mut self, step_time: Duration) {
        let step_time_ms = step_time.as_secs_f32() * 1000.0;

        // Update running average of step time
        if self.performance_metrics.total_steps == 0 {
            self.performance_metrics.average_step_time_ms = step_time_ms;
        } else {
            let alpha = 0.1; // Smoothing factor
            self.performance_metrics.average_step_time_ms =
                (1.0 - alpha) * self.performance_metrics.average_step_time_ms + alpha * step_time_ms;
        }

        // Calculate FPS
        self.performance_metrics.last_fps = if step_time_ms > 0.0 {
            1000.0 / step_time_ms
        } else {
            0.0
        };

        // Update GPU metrics if available
        if let Some(ref stats) = self.physics_stats {
            // GPU metrics would be extracted from available stats fields
            self.performance_metrics.gpu_utilization = 0.0; // Placeholder
            self.performance_metrics.gpu_memory_usage_mb = 0.0; // Placeholder
            self.performance_metrics.convergence_rate = 0.0; // Placeholder
        }
    }

    /// Get current physics status for monitoring
    pub fn get_physics_status(&self) -> PhysicsStatus {
        PhysicsStatus {
            simulation_running: self.simulation_running.load(Ordering::SeqCst),
            is_paused: self.simulation_params.is_physics_paused,
            gpu_enabled: self.gpu_compute_addr.is_some(),
            gpu_initialized: self.gpu_initialized,
            node_count: self.last_node_count,
            performance: self.performance_metrics.clone(),
            current_params: self.simulation_params.clone(),
        }
    }

    /// Apply ontology constraints to the physics system
    fn apply_ontology_constraints_internal(
        &mut self,
        constraint_set: ConstraintSet,
        merge_mode: &ConstraintMergeMode,
    ) -> Result<(), String> {
        match merge_mode {
            ConstraintMergeMode::Replace => {
                // Replace all ontology constraints
                let constraints_len = constraint_set.constraints.len();
                let groups_len = constraint_set.groups.len();
                self.ontology_constraints = Some(constraint_set);
                info!("Replaced ontology constraints: {} constraints in {} groups",
                      constraints_len, groups_len);
            }
            ConstraintMergeMode::Merge => {
                // Merge with existing ontology constraints
                if let Some(ref mut existing) = self.ontology_constraints {
                    let start_count = existing.constraints.len();
                    existing.constraints.extend(constraint_set.constraints);

                    // Merge groups
                    for (group_name, indices) in constraint_set.groups {
                        let offset = start_count;
                        let adjusted_indices: Vec<usize> = indices.iter()
                            .map(|&idx| idx + offset)
                            .collect();

                        existing.groups.entry(group_name)
                            .or_insert_with(Vec::new)
                            .extend(adjusted_indices);
                    }

                    info!("Merged ontology constraints: {} total constraints",
                          existing.constraints.len());
                } else {
                    self.ontology_constraints = Some(constraint_set);
                }
            }
            ConstraintMergeMode::AddIfNoConflict => {
                // Add only non-conflicting constraints
                if let Some(ref mut existing) = self.ontology_constraints {
                    let start_count = existing.constraints.len();
                    let mut added = 0;

                    for constraint in constraint_set.constraints {
                        // Simple conflict check: don't add if same nodes are already constrained
                        let has_conflict = existing.constraints.iter().any(|c| {
                            c.kind == constraint.kind &&
                            c.node_indices == constraint.node_indices
                        });

                        if !has_conflict {
                            existing.constraints.push(constraint);
                            added += 1;
                        }
                    }

                    // Update groups for added constraints
                    for (group_name, indices) in constraint_set.groups {
                        let adjusted_indices: Vec<usize> = indices.iter()
                            .filter_map(|&idx| {
                                if idx < added {
                                    Some(idx + start_count)
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !adjusted_indices.is_empty() {
                            existing.groups.entry(group_name)
                                .or_insert_with(Vec::new)
                                .extend(adjusted_indices);
                        }
                    }

                    info!("Added {} non-conflicting constraints", added);
                } else {
                    self.ontology_constraints = Some(constraint_set);
                }
            }
        }

        // Upload constraints to GPU if initialized
        self.upload_constraints_to_gpu();

        Ok(())
    }

    /// Upload all active constraints to GPU
    fn upload_constraints_to_gpu(&self) {
        if !self.gpu_initialized || self.gpu_compute_addr.is_none() {
            return;
        }

        // Combine ontology and user constraints
        let mut all_constraints = Vec::new();

        if let Some(ref ont_constraints) = self.ontology_constraints {
            all_constraints.extend(ont_constraints.active_constraints());
        }

        if let Some(ref user_constraints) = self.user_constraints {
            all_constraints.extend(user_constraints.active_constraints());
        }

        if all_constraints.is_empty() {
            debug!("No active constraints to upload to GPU");
            return;
        }

        // Convert to GPU format
        let gpu_constraints: Vec<_> = all_constraints.iter()
            .map(|c| c.to_gpu_format())
            .collect();

        info!("Uploading {} active constraints to GPU", gpu_constraints.len());

        // Send to GPU actor
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            use crate::actors::messages::UploadConstraintsToGPU;
            gpu_addr.do_send(UploadConstraintsToGPU {
                constraint_data: gpu_constraints,
            });
        }
    }

    /// Get constraint statistics
    fn get_constraint_statistics(&self) -> ConstraintStats {
        let mut total_constraints = 0;
        let mut active_constraints = 0;
        let mut constraint_groups = HashMap::new();
        let mut ontology_constraints = 0;
        let mut user_constraints = 0;

        // Count ontology constraints
        if let Some(ref ont) = self.ontology_constraints {
            total_constraints += ont.constraints.len();
            ontology_constraints = ont.constraints.len();
            active_constraints += ont.active_constraints().len();

            for (group_name, indices) in &ont.groups {
                constraint_groups.insert(
                    format!("ontology_{}", group_name),
                    indices.len()
                );
            }
        }

        // Count user constraints
        if let Some(ref user) = self.user_constraints {
            total_constraints += user.constraints.len();
            user_constraints = user.constraints.len();
            active_constraints += user.active_constraints().len();

            for (group_name, indices) in &user.groups {
                constraint_groups.insert(
                    format!("user_{}", group_name),
                    indices.len()
                );
            }
        }

        ConstraintStats {
            total_constraints,
            active_constraints,
            constraint_groups,
            ontology_constraints,
            user_constraints,
        }
    }

    /// Set constraint group active/inactive status
    fn set_constraint_group_active(&mut self, group_name: &str, active: bool) -> Result<(), String> {
        let mut found = false;

        // Check in ontology constraints
        if let Some(ref mut ont) = self.ontology_constraints {
            if ont.groups.contains_key(group_name) {
                ont.set_group_active(group_name, active);
                found = true;
            }
        }

        // Check in user constraints
        if let Some(ref mut user) = self.user_constraints {
            if user.groups.contains_key(group_name) {
                user.set_group_active(group_name, active);
                found = true;
            }
        }

        if found {
            info!("Set constraint group '{}' active={}", group_name, active);
            self.upload_constraints_to_gpu();
            Ok(())
        } else {
            Err(format!("Constraint group '{}' not found", group_name))
        }
    }
}

/// Physics orchestrator status information
#[derive(Debug, Clone)]
pub struct PhysicsStatus {
    pub simulation_running: bool,
    pub is_paused: bool,
    pub gpu_enabled: bool,
    pub gpu_initialized: bool,
    pub node_count: usize,
    pub performance: PhysicsPerformanceMetrics,
    pub current_params: SimulationParams,
}

impl Actor for PhysicsOrchestratorActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("Physics Orchestrator Actor started");

        // Initialize GPU if available
        if self.gpu_compute_addr.is_some() {
            self.initialize_gpu_if_needed(ctx);
        }
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Physics Orchestrator Actor stopped");
        self.stop_simulation();
    }
}

// Message Handler Implementations

impl Handler<StartSimulation> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: StartSimulation, ctx: &mut Self::Context) -> Self::Result {
        info!("Starting physics simulation");
        self.start_simulation_loop(ctx);
        Ok(())
    }
}

impl Handler<StopSimulation> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: StopSimulation, _ctx: &mut Self::Context) -> Self::Result {
        info!("Stopping physics simulation");
        self.stop_simulation();
        Ok(())
    }
}

impl Handler<SimulationStep> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: SimulationStep, ctx: &mut Self::Context) -> Self::Result {
        // Execute single physics step on demand
        self.physics_step(ctx);
        Ok(())
    }
}

impl Handler<UpdateNodePositions> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: UpdateNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        // Update GPU with new positions if needed
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            if let Some(ref graph_data) = self.graph_data_ref {
                gpu_addr.do_send(UpdateGPUGraphData {
                    graph: Arc::clone(graph_data)
                });
            }
        }

        Ok(())
    }
}

impl Handler<UpdateNodePosition> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: UpdateNodePosition, _ctx: &mut Self::Context) -> Self::Result {
        // Log single node position update
        debug!("Single node position update received");
        Ok(())
    }
}

impl Handler<RequestPositionSnapshot> for PhysicsOrchestratorActor {
    type Result = Result<PositionSnapshot, String>;

    fn handle(&mut self, msg: RequestPositionSnapshot, _ctx: &mut Self::Context) -> Self::Result {
        use crate::actors::messages::PositionSnapshot;
        use crate::utils::socket_flow_messages::BinaryNodeDataClient;

        // Return current node positions from graph data
        if let Some(ref graph_data) = self.graph_data_ref {
            let knowledge_nodes: Vec<(u32, BinaryNodeData)> = graph_data.nodes.iter()
                .map(|node| (node.id, node.data.clone()))
                .collect();

            let snapshot = PositionSnapshot {
                knowledge_nodes,
                agent_nodes: Vec::new(), // Empty for now
                timestamp: Instant::now(),
            };

            Ok(snapshot)
        } else {
            Err("No graph data available".to_string())
        }
    }
}

impl Handler<PhysicsPauseMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: PhysicsPauseMessage, _ctx: &mut Self::Context) -> Self::Result {
        info!("Physics pause requested: pause={}", msg.pause);

        if msg.pause {
            self.simulation_params.is_physics_paused = true;
        } else {
            self.resume_physics();
        }

        Ok(())
    }
}

impl Handler<NodeInteractionMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: NodeInteractionMessage, _ctx: &mut Self::Context) -> Self::Result {
        info!("Node interaction detected: {:?}", msg.interaction_type);

        // Resume physics on user interaction if configured
        if self.simulation_params.auto_pause_config.resume_on_interaction {
            if self.simulation_params.is_physics_paused {
                self.resume_physics();
            }

            // Set timer for potential re-pause
            self.force_resume_timer = Some(Instant::now());
        }

        Ok(())
    }
}

impl Handler<ForceResumePhysics> for PhysicsOrchestratorActor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, _msg: ForceResumePhysics, _ctx: &mut Self::Context) -> Self::Result {
        info!("Force resume physics requested");

        let was_paused = self.simulation_params.is_physics_paused;
        self.resume_physics();

        Ok(())
    }
}

impl Handler<StoreGPUComputeAddress> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, msg: StoreGPUComputeAddress, ctx: &mut Self::Context) -> Self::Result {
        info!("Storing GPU compute address");
        // Convert from GPUComputeActor to ForceComputeActor address if needed
        // For now, just log - actual implementation would handle address conversion
        debug!("GPU address stored: {:?}", msg.addr.is_some());

        if self.gpu_compute_addr.is_some() {
            self.initialize_gpu_if_needed(ctx);
        }
    }
}

impl Handler<UpdateSimulationParams> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateSimulationParams, ctx: &mut Self::Context) -> Self::Result {
        info!("Updating simulation parameters");

        // Check if auto-balance was just enabled
        let auto_balance_just_enabled = !self.simulation_params.auto_balance && msg.params.auto_balance;

        // Update target parameters for smooth transition
        self.target_params = msg.params.clone();

        // Immediately update certain critical parameters
        self.simulation_params.enabled = msg.params.enabled;
        self.simulation_params.auto_balance = msg.params.auto_balance;
        self.simulation_params.auto_balance_config = msg.params.auto_balance_config.clone();
        self.simulation_params.auto_pause_config = msg.params.auto_pause_config.clone();

        // If auto-balance was just enabled, reset timers
        if auto_balance_just_enabled {
            self.auto_balance_last_check = None;
        }

        // Update GPU parameters if initialized
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            if self.gpu_initialized {
                if let Some(ref graph_data) = self.graph_data_ref {
                    gpu_addr.do_send(UpdateGPUGraphData {
                        graph: Arc::clone(graph_data)
                    });
                }
            }
        }

        info!("Physics parameters updated - repel_k: {}, damping: {}",
              self.target_params.repel_k, self.target_params.damping);

        Ok(())
    }
}

/// Message to get current physics status
#[derive(Message)]
#[rtype(result = "PhysicsStatus")]
pub struct GetPhysicsStatus;

impl Handler<GetPhysicsStatus> for PhysicsOrchestratorActor {
    type Result = MessageResult<GetPhysicsStatus>;

    fn handle(&mut self, _msg: GetPhysicsStatus, _ctx: &mut Self::Context) -> Self::Result {
        MessageResult(self.get_physics_status())
    }
}

/// Message to update physics statistics from GPU
#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdatePhysicsStats {
    pub stats: PhysicsStats,
}

impl Handler<UpdatePhysicsStats> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, msg: UpdatePhysicsStats, _ctx: &mut Self::Context) -> Self::Result {
        self.physics_stats = Some(msg.stats);
    }
}

/// Message to update graph data
#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateGraphData {
    pub graph_data: Arc<GraphData>,
}

impl Handler<UpdateGraphData> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateGraphData, _ctx: &mut Self::Context) -> Self::Result {
        self.update_graph_data(msg.graph_data);
    }
}

/// Message to force parameter interpolation completion
#[derive(Message)]
#[rtype(result = "()")]
pub struct FlushParameterTransitions;

impl Handler<FlushParameterTransitions> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, _msg: FlushParameterTransitions, _ctx: &mut Self::Context) -> Self::Result {
        // Immediately apply target parameters
        self.simulation_params = self.target_params.clone();
        info!("Parameter transitions flushed");
    }
}

/// Message to adjust interpolation rate
#[derive(Message)]
#[rtype(result = "()")]
pub struct SetParameterInterpolationRate {
    pub rate: f32,
}

impl Handler<SetParameterInterpolationRate> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, msg: SetParameterInterpolationRate, _ctx: &mut Self::Context) -> Self::Result {
        self.param_interpolation_rate = msg.rate.clamp(0.01, 1.0);
        info!("Parameter interpolation rate set to: {}", self.param_interpolation_rate);
    }
}

/// Handler for applying ontology-derived constraints
impl Handler<ApplyOntologyConstraints> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ApplyOntologyConstraints, _ctx: &mut Self::Context) -> Self::Result {
        info!("Applying ontology constraints: {} constraints, merge mode: {:?}",
              msg.constraint_set.constraints.len(), msg.merge_mode);

        self.apply_ontology_constraints_internal(msg.constraint_set, &msg.merge_mode)
    }
}

/// Handler for enabling/disabling constraint groups
impl Handler<SetConstraintGroupActive> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetConstraintGroupActive, _ctx: &mut Self::Context) -> Self::Result {
        self.set_constraint_group_active(&msg.group_name, msg.active)
    }
}

/// Handler for getting constraint statistics
impl Handler<GetConstraintStats> for PhysicsOrchestratorActor {
    type Result = Result<ConstraintStats, String>;

    fn handle(&mut self, _msg: GetConstraintStats, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_constraint_statistics())
    }
}

/// Message to set ontology actor address in physics orchestrator
#[cfg(feature = "ontology")]
#[derive(Message)]
#[rtype(result = "()")]
pub struct SetOntologyActor {
    pub addr: Addr<crate::actors::ontology_actor::OntologyActor>,
}

/// Handler for setting ontology actor address
#[cfg(feature = "ontology")]
impl Handler<SetOntologyActor> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, msg: SetOntologyActor, _ctx: &mut Self::Context) -> Self::Result {
        self.set_ontology_actor(msg.addr);
    }
}