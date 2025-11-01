//! Stress Majorization Actor - Handles stress optimization and layout algorithms

use actix::prelude::*;
use log::{error, info, trace, warn};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::shared::{GPUState, SharedGPUContext, StressMajorizationSafety};
use crate::actors::messages::*;

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressMajorizationParams {
    pub max_iterations: u32,
    pub tolerance: f32,
    pub learning_rate: f32,
    pub interval_frames: Option<u32>,
    pub max_displacement_threshold: Option<f32>,
    pub max_position_magnitude: Option<f32>,
    pub convergence_threshold: Option<f32>,
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressMajorizationStats {
    pub stress_value: f32,
    pub iterations_performed: u32,
    pub converged: bool,
    pub computation_time_ms: u64,
}

/
pub struct StressMajorizationActor {
    
    gpu_state: GPUState,

    
    shared_context: Option<Arc<SharedGPUContext>>,

    
    safety: StressMajorizationSafety,

    
    stress_majorization_interval: u32,

    
    last_stress_majorization: u32,
}

impl StressMajorizationActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            safety: StressMajorizationSafety::new(),
            stress_majorization_interval: 600, 
            last_stress_majorization: 0,
        }
    }

    
    fn perform_stress_majorization(&mut self) -> Result<(), String> {
        info!("StressMajorizationActor: Performing stress majorization");

        
        if !self.safety.is_safe_to_run() {
            let reason = if self.safety.is_emergency_stopped {
                format!(
                    "Emergency stopped: {}",
                    self.safety.last_emergency_stop_reason
                )
            } else {
                format!(
                    "Too many consecutive failures: {}",
                    self.safety.consecutive_failures
                )
            };

            warn!(
                "StressMajorizationActor: Skipping stress majorization - {}",
                reason
            );
            return Err(reason);
        }

        let mut unified_compute = match &self.shared_context {
            Some(ctx) => ctx
                .unified_compute
                .lock()
                .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?,
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };

        let start_time = Instant::now();

        
        let result = unified_compute.run_stress_majorization().map_err(|e| {
            error!("GPU stress majorization failed: {}", e);
            self.safety
                .record_failure(format!("GPU execution failed: {}", e));
            format!("Stress majorization failed: {}", e)
        });

        let computation_time = start_time.elapsed();

        match result {
            Ok(stress_info) => {
                
                self.safety
                    .record_success(computation_time.as_millis() as u64);

                
                
                let (positions_x, positions_y, positions_z) = stress_info;
                let stress_value =
                    self.calculate_stress_value(&positions_x, &positions_y, &positions_z)?;
                let max_displacement =
                    self.calculate_max_displacement(&positions_x, &positions_y, &positions_z)?;
                let converged = stress_value < self.safety.convergence_threshold;

                self.safety
                    .record_iteration(stress_value, max_displacement, converged);

                
                self.last_stress_majorization = self.gpu_state.iteration_count;

                info!(
                    "StressMajorizationActor: Completed successfully in {:?}",
                    computation_time
                );
                info!(
                    "  Final stress: {:.2}, Max displacement: {:.2}, Converged: {}",
                    stress_value, max_displacement, converged
                );

                
                self.apply_position_safety_clamping()?;

                Ok(())
            }
            Err(e) => {
                error!("StressMajorizationActor: Failed - {}", e);
                Err(e)
            }
        }
    }

    
    fn apply_position_safety_clamping(&self) -> Result<(), String> {
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => ctx
                .unified_compute
                .lock()
                .map_err(|e| format!("Failed to acquire GPU compute lock for clamping: {}", e))?,
            None => {
                return Err("GPU context not initialized for position clamping".to_string());
            }
        };

        
        let (positions_x, positions_y, positions_z) = unified_compute
            .get_node_positions()
            .map_err(|e| format!("Failed to get positions for clamping: {}", e))?;

        
        let mut clamping_needed = false;
        let mut clamped_x = positions_x.clone();
        let mut clamped_y = positions_y.clone();
        let mut clamped_z = positions_z.clone();

        for i in 0..positions_x.len() {
            let pos = [positions_x[i], positions_y[i], positions_z[i]];

            let clamped_pos = self.safety.clamp_position(&pos);

            if (clamped_pos[0] - pos[0]).abs() > 1e-6
                || (clamped_pos[1] - pos[1]).abs() > 1e-6
                || (clamped_pos[2] - pos[2]).abs() > 1e-6
            {
                clamping_needed = true;
                clamped_x[i] = clamped_pos[0];
                clamped_y[i] = clamped_pos[1];
                clamped_z[i] = clamped_pos[2];
            }
        }

        
        if clamping_needed {
            warn!("StressMajorizationActor: Position clamping applied to prevent numerical instability");
            unified_compute
                .update_positions_only(&clamped_x, &clamped_y, &clamped_z)
                .map_err(|e| format!("Failed to update clamped positions: {}", e))?;
        }

        Ok(())
    }

    
    fn should_run_stress_majorization(&self) -> bool {
        if !self.safety.is_safe_to_run() {
            return false;
        }

        let iterations_since_last = self
            .gpu_state
            .iteration_count
            .saturating_sub(self.last_stress_majorization);
        iterations_since_last >= self.stress_majorization_interval
    }

    
    fn update_stress_majorization_params(&mut self, params: StressMajorizationParams) {
        info!("StressMajorizationActor: Updating stress majorization parameters");

        
        if let Some(interval) = params.interval_frames {
            self.stress_majorization_interval = interval;
            info!("  Updated interval to {} frames", interval);
        }

        
        if let Some(max_displacement) = params.max_displacement_threshold {
            self.safety.max_displacement_threshold = max_displacement;
            info!(
                "  Updated max displacement threshold to {:.2}",
                max_displacement
            );
        }

        if let Some(max_position) = params.max_position_magnitude {
            self.safety.max_position_magnitude = max_position;
            info!("  Updated max position magnitude to {:.2}", max_position);
        }

        if let Some(convergence) = params.convergence_threshold {
            self.safety.convergence_threshold = convergence;
            info!("  Updated convergence threshold to {:.4}", convergence);
        }
    }

    
    fn get_stress_majorization_stats(&self) -> StressMajorizationStats {
        self.safety.get_stats()
    }

    
    fn reset_safety_state(&mut self) {
        self.safety.reset_safety_state();
        info!("StressMajorizationActor: Safety state has been reset");
    }

    
    fn should_disable_stress_majorization(&self) -> bool {
        self.safety.should_disable()
    }

    
    fn calculate_stress_value(
        &self,
        pos_x: &[f32],
        pos_y: &[f32],
        pos_z: &[f32],
    ) -> Result<f32, String> {
        if pos_x.len() != pos_y.len() || pos_y.len() != pos_z.len() {
            return Err("Position arrays have mismatched lengths".to_string());
        }

        let mut total_stress = 0.0f32;
        let n = pos_x.len();

        
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pos_x[i] - pos_x[j];
                let dy = pos_y[i] - pos_y[j];
                let dz = pos_z[i] - pos_z[j];
                let actual_dist = (dx * dx + dy * dy + dz * dz).sqrt();

                
                let target_dist = ((i as f32 - j as f32).abs() + 1.0).ln();
                let weight = 1.0; 

                let diff = actual_dist - target_dist;
                total_stress += weight * diff * diff;
            }
        }

        Ok(total_stress)
    }

    
    fn calculate_max_displacement(
        &self,
        pos_x: &[f32],
        pos_y: &[f32],
        pos_z: &[f32],
    ) -> Result<f32, String> {
        
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => ctx.unified_compute.lock().map_err(|e| {
                format!(
                    "Failed to acquire GPU compute lock for displacement calculation: {}",
                    e
                )
            })?,
            None => {
                return Ok(0.0);
            }
        };

        let (prev_x, prev_y, prev_z) = unified_compute
            .get_node_positions()
            .map_err(|e| format!("Failed to get previous positions: {}", e))?;

        let mut max_displacement = 0.0f32;

        for i in 0..pos_x.len().min(prev_x.len()) {
            let dx = pos_x[i] - prev_x[i];
            let dy = pos_y[i] - prev_y[i];
            let dz = pos_z[i] - prev_z[i];
            let displacement = (dx * dx + dy * dy + dz * dz).sqrt();
            max_displacement = max_displacement.max(displacement);
        }

        Ok(max_displacement)
    }
}

impl Actor for StressMajorizationActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("Stress Majorization Actor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Stress Majorization Actor stopped");
    }
}

// === Message Handlers ===

impl Handler<TriggerStressMajorization> for StressMajorizationActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: TriggerStressMajorization,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("StressMajorizationActor: Manual stress majorization trigger received");

        if self.shared_context.is_none() {
            error!("StressMajorizationActor: GPU not initialized");
            return Err("GPU not initialized".to_string());
        }

        self.perform_stress_majorization()
    }
}

// FIXME: Type conflict - commented for compilation


impl Handler<ResetStressMajorizationSafety> for StressMajorizationActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: ResetStressMajorizationSafety,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        self.reset_safety_state();
        Ok(())
    }
}

impl Handler<UpdateStressMajorizationParams> for StressMajorizationActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        msg: UpdateStressMajorizationParams,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        
        let stress_params = StressMajorizationParams {
            max_iterations: 100, 
            tolerance: 0.001,    
            learning_rate: 0.1,  
            interval_frames: Some(msg.params.stress_step_interval_frames),
            max_displacement_threshold: None, 
            max_position_magnitude: None,     
            convergence_threshold: None,      
        };
        self.update_stress_majorization_params(stress_params);
        Ok(())
    }
}

/
impl Handler<CheckStressMajorization> for StressMajorizationActor {
    type Result = Result<bool, String>;

    fn handle(&mut self, _msg: CheckStressMajorization, _ctx: &mut Self::Context) -> Self::Result {
        if self.should_run_stress_majorization() {
            info!("StressMajorizationActor: Automatic stress majorization triggered");
            match self.perform_stress_majorization() {
                Ok(_) => Ok(true),
                Err(e) => {
                    warn!(
                        "StressMajorizationActor: Automatic stress majorization failed: {}",
                        e
                    );
                    Ok(false) 
                }
            }
        } else {
            trace!("StressMajorizationActor: Stress majorization not needed yet");
            Ok(false)
        }
    }
}

// Custom message for internal stress majorization checks
#[derive(Message)]
#[rtype(result = "Result<bool, String>")]
pub struct CheckStressMajorization;

/
impl Handler<SetSharedGPUContext> for StressMajorizationActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        info!("StressMajorizationActor: Received SharedGPUContext from ResourceActor");
        self.shared_context = Some(msg.context);
        
        info!("StressMajorizationActor: SharedGPUContext stored successfully");
        Ok(())
    }
}
