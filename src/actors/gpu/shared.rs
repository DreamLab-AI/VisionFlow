//! Shared data structures and utilities for GPU actors

use std::sync::Arc;
use std::collections::HashMap;
use actix::Addr;
use cudarc::driver::{CudaDevice, CudaStream};
use serde::{Serialize, Deserialize};

use crate::utils::unified_gpu_compute::{UnifiedGPUCompute, SimParams};
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::Constraint;

// Import the child actors for address storage
// use super::{GPUResourceActor, ForceComputeActor, ClusteringActor, 
//            AnomalyDetectionActor, StressMajorizationActor, ConstraintActor};

/// Child actor addresses for the GPU manager

/// Shared GPU context that gets passed between child actors
// Note: CudaStream wrapped in Arc<Mutex> for thread safety
pub struct SharedGPUContext {
    pub device: Arc<CudaDevice>,
    pub stream: Arc<std::sync::Mutex<CudaStream>>,
    pub unified_compute: Arc<std::sync::Mutex<UnifiedGPUCompute>>,
}

/// GPU state shared among child actors
#[derive(Debug, Clone)]
pub struct GPUState {
    pub num_nodes: u32,
    pub num_edges: u32,
    pub node_indices: HashMap<u32, usize>,
    pub simulation_params: SimulationParams,
    pub unified_params: SimParams,
    pub constraints: Vec<Constraint>,
    pub iteration_count: u32,
    pub gpu_failure_count: u32,
    pub is_initialized: bool,

    // GPU Upload Optimization tracking
    pub graph_structure_hash: u64,
    pub positions_hash: u64,
    pub csr_structure_uploaded: bool,
}

impl Default for GPUState {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            node_indices: HashMap::new(),
            simulation_params: SimulationParams::default(),
            unified_params: SimParams::default(),
            constraints: Vec::new(),
            iteration_count: 0,
            gpu_failure_count: 0,
            is_initialized: false,
            graph_structure_hash: 0,
            positions_hash: 0,
            csr_structure_uploaded: false,
        }
    }
}

/// Safety controls for stress majorization - moved from main actor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressMajorizationSafety {
    /// Maximum allowed displacement per iteration
    pub max_displacement_threshold: f32,
    /// Maximum allowed position magnitude
    pub max_position_magnitude: f32,
    /// Number of consecutive failures before disabling
    pub max_consecutive_failures: u32,
    /// Convergence threshold for displacement
    pub convergence_threshold: f32,
    /// Maximum allowed stress value before emergency stop
    pub max_stress_threshold: f32,
    
    // Runtime state
    pub consecutive_failures: u32,
    pub last_stress_values: Vec<f32>,
    pub last_displacement_values: Vec<f32>,
    pub total_runs: u64,
    pub successful_runs: u64,
    pub total_computation_time_ms: u64,
    pub is_emergency_stopped: bool,
    pub last_emergency_stop_reason: String,
}

impl StressMajorizationSafety {
    pub fn new() -> Self {
        Self {
            max_displacement_threshold: 1000.0,
            max_position_magnitude: 5000.0,
            max_consecutive_failures: 3,
            convergence_threshold: 0.01,
            max_stress_threshold: 1e6,
            
            consecutive_failures: 0,
            last_stress_values: Vec::with_capacity(10),
            last_displacement_values: Vec::with_capacity(10),
            total_runs: 0,
            successful_runs: 0,
            total_computation_time_ms: 0,
            is_emergency_stopped: false,
            last_emergency_stop_reason: String::new(),
        }
    }
    
    pub fn is_safe_to_run(&self) -> bool {
        !self.is_emergency_stopped && self.consecutive_failures < self.max_consecutive_failures
    }
    
    pub fn record_failure(&mut self, reason: String) {
        self.consecutive_failures += 1;
        self.total_runs += 1;
        if self.consecutive_failures >= self.max_consecutive_failures {
            self.is_emergency_stopped = true;
            self.last_emergency_stop_reason = reason;
        }
    }
    
    pub fn record_success(&mut self, computation_time_ms: u64) {
        self.consecutive_failures = 0;
        self.total_runs += 1;
        self.successful_runs += 1;
        self.total_computation_time_ms += computation_time_ms;
    }
    
    pub fn record_iteration(&mut self, stress: f32, displacement: f32, converged: bool) {
        self.last_stress_values.push(stress);
        if self.last_stress_values.len() > 10 {
            self.last_stress_values.remove(0);
        }
        
        self.last_displacement_values.push(displacement);
        if self.last_displacement_values.len() > 10 {
            self.last_displacement_values.remove(0);
        }
        
        // Check for emergency conditions
        if stress > self.max_stress_threshold {
            self.record_failure(format!("Stress exceeded threshold: {}", stress));
        }
        
        if displacement > self.max_displacement_threshold {
            self.record_failure(format!("Displacement exceeded threshold: {}", displacement));
        }
    }
    
    pub fn clamp_position(&self, position: &[f32; 3]) -> [f32; 3] {
        let magnitude = (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt();
        if magnitude > self.max_position_magnitude {
            let scale = self.max_position_magnitude / magnitude;
            [
                position[0] * scale,
                position[1] * scale,
                position[2] * scale,
            ]
        } else {
            *position
        }
    }
    
    pub fn get_stats(&self) -> crate::actors::gpu::stress_majorization_actor::StressMajorizationStats {
        crate::actors::gpu::stress_majorization_actor::StressMajorizationStats {
            stress_value: 0.0, // Default value
            iterations_performed: self.total_runs as u32,
            converged: !self.is_emergency_stopped,
            computation_time_ms: if self.successful_runs > 0 {
                self.total_computation_time_ms / self.successful_runs
            } else {
                0
            },
        }
    }
    
    pub fn reset_safety_state(&mut self) {
        self.consecutive_failures = 0;
        self.is_emergency_stopped = false;
        self.last_emergency_stop_reason.clear();
    }
    
    pub fn should_disable(&self) -> bool {
        self.is_emergency_stopped
    }
}

/// Child actor addresses for the manager
#[derive(Clone)]
pub struct ChildActorAddresses {
    pub resource_actor: Addr<super::GPUResourceActor>,
    pub force_compute_actor: Addr<super::ForceComputeActor>,
    pub clustering_actor: Addr<super::ClusteringActor>,
    pub anomaly_detection_actor: Addr<super::AnomalyDetectionActor>,
    pub stress_majorization_actor: Addr<super::StressMajorizationActor>,
    pub constraint_actor: Addr<super::ConstraintActor>,
}