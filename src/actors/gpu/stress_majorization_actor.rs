//! Stress Majorization Actor - Handles stress optimization and layout algorithms

use actix::prelude::*;
use log::{error, info, trace, warn};
use std::time::Instant;
use serde::{Serialize, Deserialize};

use crate::actors::messages::*;
use super::shared::{SharedGPUContext, GPUState, StressMajorizationSafety};

/// Stress majorization parameters
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

/// Stress majorization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressMajorizationStats {
    pub stress_value: f32,
    pub iterations_performed: u32,
    pub converged: bool,
    pub computation_time_ms: u64,
}

/// Stress Majorization Actor - handles stress optimization and layout algorithms
pub struct StressMajorizationActor {
    /// Shared GPU state
    gpu_state: GPUState,
    
    /// Shared GPU context reference
    shared_context: Option<SharedGPUContext>,
    
    /// Stress majorization safety controls
    safety: StressMajorizationSafety,
    
    /// Stress majorization execution interval (in iterations)
    stress_majorization_interval: u32,
    
    /// Last iteration when stress majorization was performed
    last_stress_majorization: u32,
}

impl StressMajorizationActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
            safety: StressMajorizationSafety::new(),
            stress_majorization_interval: 600, // Default: every 10 seconds at 60 FPS
            last_stress_majorization: 0,
        }
    }
    
    /// Perform stress majorization with safety controls
    fn perform_stress_majorization(&mut self) -> Result<(), String> {
        info!("StressMajorizationActor: Performing stress majorization");
        
        // Safety check - verify it's safe to run
        if !self.safety.is_safe_to_run() {
            let reason = if self.safety.is_emergency_stopped {
                format!("Emergency stopped: {}", self.safety.last_emergency_stop_reason)
            } else {
                format!("Too many consecutive failures: {}", self.safety.consecutive_failures)
            };
            
            warn!("StressMajorizationActor: Skipping stress majorization - {}", reason);
            return Err(reason);
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
        
        let start_time = Instant::now();
        
        // Execute stress majorization on GPU with safety monitoring
        let result = unified_compute.run_stress_majorization()
            .map_err(|e| {
                error!("GPU stress majorization failed: {}", e);
                self.safety.record_failure(format!("GPU execution failed: {}", e));
                format!("Stress majorization failed: {}", e)
            });
        
        let computation_time = start_time.elapsed();
        
        match result {
            Ok(stress_info) => {
                // Record successful execution
                self.safety.record_success(computation_time.as_millis() as u64);
                
                // Record iteration metrics for safety monitoring
                // Extract info from tuple result
                let (positions_x, positions_y, positions_z) = stress_info;
                let stress_value = self.calculate_stress_value(&positions_x, &positions_y, &positions_z)?;
                let max_displacement = self.calculate_max_displacement(&positions_x, &positions_y, &positions_z)?;
                let converged = stress_value < self.safety.convergence_threshold;
                
                self.safety.record_iteration(stress_value, max_displacement, converged);
                
                // Update last execution tracking
                self.last_stress_majorization = self.gpu_state.iteration_count;
                
                info!("StressMajorizationActor: Completed successfully in {:?}", computation_time);
                info!("  Final stress: {:.2}, Max displacement: {:.2}, Converged: {}", 
                      stress_value, max_displacement, converged);
                
                // Apply position clamping for safety
                self.apply_position_safety_clamping()?;
                
                Ok(())
            },
            Err(e) => {
                error!("StressMajorizationActor: Failed - {}", e);
                Err(e)
            }
        }
    }
    
    /// Apply position safety clamping to prevent numerical explosions
    fn apply_position_safety_clamping(&self) -> Result<(), String> {
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock for clamping: {}", e))?
            },
            None => {
                return Err("GPU context not initialized for position clamping".to_string());
            }
        };
        
        // Get current positions from GPU
        let (positions_x, positions_y, positions_z) = unified_compute.get_node_positions()
            .map_err(|e| format!("Failed to get positions for clamping: {}", e))?;
        
        // Check if any positions need clamping
        let mut clamping_needed = false;
        let mut clamped_x = positions_x.clone();
        let mut clamped_y = positions_y.clone();
        let mut clamped_z = positions_z.clone();
        
        for i in 0..positions_x.len() {
            let pos = [positions_x[i], positions_y[i], positions_z[i]];
            
            let clamped_pos = self.safety.clamp_position(&pos);
            
            if (clamped_pos[0] - pos[0]).abs() > 1e-6 ||
               (clamped_pos[1] - pos[1]).abs() > 1e-6 ||
               (clamped_pos[2] - pos[2]).abs() > 1e-6 {
                clamping_needed = true;
                clamped_x[i] = clamped_pos[0];
                clamped_y[i] = clamped_pos[1];
                clamped_z[i] = clamped_pos[2];
            }
        }
        
        // Update positions on GPU if clamping was needed
        if clamping_needed {
            warn!("StressMajorizationActor: Position clamping applied to prevent numerical instability");
            unified_compute.update_positions_only(&clamped_x, &clamped_y, &clamped_z)
                .map_err(|e| format!("Failed to update clamped positions: {}", e))?;
        }
        
        Ok(())
    }
    
    /// Check if stress majorization should be triggered based on interval
    fn should_run_stress_majorization(&self) -> bool {
        if !self.safety.is_safe_to_run() {
            return false;
        }
        
        let iterations_since_last = self.gpu_state.iteration_count.saturating_sub(self.last_stress_majorization);
        iterations_since_last >= self.stress_majorization_interval
    }
    
    /// Update stress majorization parameters
    fn update_stress_majorization_params(&mut self, params: StressMajorizationParams) {
        info!("StressMajorizationActor: Updating stress majorization parameters");
        
        // Update interval
        if let Some(interval) = params.interval_frames {
            self.stress_majorization_interval = interval;
            info!("  Updated interval to {} frames", interval);
        }
        
        // Update safety thresholds
        if let Some(max_displacement) = params.max_displacement_threshold {
            self.safety.max_displacement_threshold = max_displacement;
            info!("  Updated max displacement threshold to {:.2}", max_displacement);
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
    
    /// Get stress majorization statistics
    fn get_stress_majorization_stats(&self) -> StressMajorizationStats {
        self.safety.get_stats()
    }
    
    /// Reset safety state
    fn reset_safety_state(&mut self) {
        self.safety.reset_safety_state();
        info!("StressMajorizationActor: Safety state has been reset");
    }
    
    /// Check if stress majorization should be disabled due to safety
    fn should_disable_stress_majorization(&self) -> bool {
        self.safety.should_disable()
    }

    /// Calculate stress function value from current positions
    fn calculate_stress_value(&self, pos_x: &[f32], pos_y: &[f32], pos_z: &[f32]) -> Result<f32, String> {
        if pos_x.len() != pos_y.len() || pos_y.len() != pos_z.len() {
            return Err("Position arrays have mismatched lengths".to_string());
        }

        let mut total_stress = 0.0f32;
        let n = pos_x.len();

        // Calculate stress as sum of squared differences between actual and target distances
        for i in 0..n {
            for j in (i+1)..n {
                let dx = pos_x[i] - pos_x[j];
                let dy = pos_y[i] - pos_y[j];
                let dz = pos_z[i] - pos_z[j];
                let actual_dist = (dx*dx + dy*dy + dz*dz).sqrt();

                // Use graph-theoretic distance as target (simplified)
                let target_dist = ((i as f32 - j as f32).abs() + 1.0).ln();
                let weight = 1.0; // Uniform weighting for simplicity

                let diff = actual_dist - target_dist;
                total_stress += weight * diff * diff;
            }
        }

        Ok(total_stress)
    }

    /// Calculate maximum displacement from previous positions
    fn calculate_max_displacement(&self, pos_x: &[f32], pos_y: &[f32], pos_z: &[f32]) -> Result<f32, String> {
        // Get previous positions from GPU state (simplified)
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock for displacement calculation: {}", e))?
            },
            None => {
                return Ok(0.0);
            }
        };

        let (prev_x, prev_y, prev_z) = unified_compute.get_node_positions()
            .map_err(|e| format!("Failed to get previous positions: {}", e))?;

        let mut max_displacement = 0.0f32;

        for i in 0..pos_x.len().min(prev_x.len()) {
            let dx = pos_x[i] - prev_x[i];
            let dy = pos_y[i] - prev_y[i];
            let dz = pos_z[i] - prev_z[i];
            let displacement = (dx*dx + dy*dy + dz*dz).sqrt();
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
    
    fn handle(&mut self, _msg: TriggerStressMajorization, _ctx: &mut Self::Context) -> Self::Result {
        info!("StressMajorizationActor: Manual stress majorization trigger received");
        
        if self.shared_context.is_none() {
            error!("StressMajorizationActor: GPU not initialized");
            return Err("GPU not initialized".to_string());
        }
        
        self.perform_stress_majorization()
    }
}

// FIXME: Type conflict - commented for compilation
/*
impl Handler<GetStressMajorizationStats> for StressMajorizationActor {
    type Result = Result<crate::actors::gpu::stress_majorization_actor::StressMajorizationStats, String>;
    
    fn handle(&mut self, _msg: GetStressMajorizationStats, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_stress_majorization_stats())
    }
}
*/

impl Handler<ResetStressMajorizationSafety> for StressMajorizationActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: ResetStressMajorizationSafety, _ctx: &mut Self::Context) -> Self::Result {
        self.reset_safety_state();
        Ok(())
    }
}

impl Handler<UpdateStressMajorizationParams> for StressMajorizationActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateStressMajorizationParams, _ctx: &mut Self::Context) -> Self::Result {
        // Extract relevant fields from AdvancedParams and map to StressMajorizationParams
        let stress_params = StressMajorizationParams {
            max_iterations: 100, // Default value since not available in AdvancedParams
            tolerance: 0.001,    // Default value since not available in AdvancedParams
            learning_rate: 0.1,  // Default value since not available in AdvancedParams
            interval_frames: Some(msg.params.stress_step_interval_frames),
            max_displacement_threshold: None, // Not available in AdvancedParams
            max_position_magnitude: None,     // Not available in AdvancedParams
            convergence_threshold: None,      // Not available in AdvancedParams
        };
        self.update_stress_majorization_params(stress_params);
        Ok(())
    }
}

/// Internal handler for automatic stress majorization checks during physics simulation
impl Handler<CheckStressMajorization> for StressMajorizationActor {
    type Result = Result<bool, String>;
    
    fn handle(&mut self, _msg: CheckStressMajorization, _ctx: &mut Self::Context) -> Self::Result {
        if self.should_run_stress_majorization() {
            info!("StressMajorizationActor: Automatic stress majorization triggered");
            match self.perform_stress_majorization() {
                Ok(_) => Ok(true),
                Err(e) => {
                    warn!("StressMajorizationActor: Automatic stress majorization failed: {}", e);
                    Ok(false) // Don't fail the entire physics step
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