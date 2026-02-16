use crate::config::{AutoBalanceConfig, AutoPauseConfig, PhysicsSettings};
use bytemuck::{Pod, Zeroable};
use cudarc::driver::DeviceRepr;
use cust_core::DeviceCopy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationMode {
    Remote, 
    Local,  
}

impl Default for SimulationMode {
    fn default() -> Self {
        SimulationMode::Remote
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationPhase {
    Initial,  
    Dynamic,  
    Finalize, 
}

impl Default for SimulationPhase {
    fn default() -> Self {
        SimulationPhase::Initial
    }
}

// GPU-compatible simulation parameters, matching the new CUDA kernel design.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimParams {
    
    pub dt: f32,
    pub damping: f32,
    pub warmup_iterations: u32,
    pub cooling_rate: f32,

    
    pub spring_k: f32,
    pub rest_length: f32,

    
    pub repel_k: f32,
    pub repulsion_cutoff: f32,
    pub repulsion_softening_epsilon: f32,

    
    pub center_gravity_k: f32,
    pub max_force: f32,
    pub max_velocity: f32,

    
    pub grid_cell_size: f32,

    
    pub feature_flags: u32,
    pub seed: u32,
    pub iteration: i32,

    
    pub separation_radius: f32,
    pub cluster_strength: f32,
    pub alignment_strength: f32,
    pub temperature: f32,
    pub viewport_bounds: f32,
    pub sssp_alpha: f32,       
    pub boundary_damping: f32, 

    
    pub constraint_ramp_frames: u32, 
    pub constraint_max_force_per_node: f32, 

    
    pub stability_threshold: f32, 
    pub min_velocity_threshold: f32, 

    
    pub world_bounds_min: f32,         
    pub world_bounds_max: f32,         
    pub cell_size_lod: f32,            
    pub k_neighbors_max: u32,          
    pub anomaly_detection_radius: f32, 
    pub learning_rate_default: f32,    

    
    pub norm_delta_cap: f32,
    pub position_constraint_attraction: f32,
    pub lof_score_min: f32,
    pub lof_score_max: f32,
    pub weight_precision_multiplier: f32,
    // NOTE: Stress majorization params removed from GPU SimParams (not used by GPU kernels).
    // Stress optimization is handled by SemanticProcessorActor on CPU.
}

// SAFETY: SimParams is repr(C) with only POD types, safe for GPU transfer
// All fields are primitives (f32, u32, i32) with well-defined memory layout
unsafe impl DeviceRepr for SimParams {}

// SAFETY: SimParams is repr(C) with only POD types, safe for GPU transfer
// All fields are primitives (f32, u32, i32) with well-defined memory layout
unsafe impl DeviceCopy for SimParams {}

pub struct FeatureFlags;
impl FeatureFlags {
    pub const ENABLE_REPULSION: u32 = 1 << 0;
    pub const ENABLE_SPRINGS: u32 = 1 << 1;
    pub const ENABLE_CENTERING: u32 = 1 << 2;
    pub const ENABLE_TEMPORAL_COHERENCE: u32 = 1 << 3;
    pub const ENABLE_CONSTRAINTS: u32 = 1 << 4; 
    pub const ENABLE_STRESS_MAJORIZATION: u32 = 1 << 5;
    pub const ENABLE_SSSP_SPRING_ADJUST: u32 = 1 << 6; 
}

#[derive(Default, Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SimulationParams {
    
    pub enabled: bool, 

    
    pub auto_balance: bool,                     
    pub auto_balance_interval_ms: u32,          
    pub auto_balance_config: AutoBalanceConfig, 

    
    pub auto_pause_config: AutoPauseConfig, 
    pub is_physics_paused: bool,            
    pub equilibrium_stability_counter: u32, 

    
    pub iterations: u32, 
    pub dt: f32,         

    
    pub spring_k: f32, 
    pub repel_k: f32,  

    
    pub mass_scale: f32,       
    pub damping: f32,          
    pub boundary_damping: f32, 

    
    pub viewport_bounds: f32, 
    pub enable_bounds: bool,  

    
    pub max_velocity: f32,      
    pub max_force: f32,         
    pub separation_radius: f32, 
    pub temperature: f32,       
    pub center_gravity_k: f32,  

    
    pub stress_weight: f32,
    pub stress_alpha: f32,
    pub boundary_limit: f32,
    pub alignment_strength: f32,
    pub cluster_strength: f32,
    pub compute_mode: i32,
    pub min_distance: f32,
    pub max_repulsion_dist: f32,
    pub boundary_margin: f32,
    pub boundary_force_strength: f32,
    pub warmup_iterations: u32,
    pub cooling_rate: f32,

    
    pub use_sssp_distances: bool, 
    pub sssp_alpha: Option<f32>,  

    
    pub constraint_ramp_frames: u32, 
    pub constraint_max_force_per_node: f32, 

    
    pub phase: SimulationPhase, 
    pub mode: SimulationMode,   
}

impl SimulationParams {
    pub fn new() -> Self {
        
        let default_physics = PhysicsSettings::default();
        Self::from(&default_physics)
    }

    pub fn with_phase(phase: SimulationPhase) -> Self {
        let mut params = Self::new();
        params.phase = phase;

        
        
        match phase {
            SimulationPhase::Initial => {
                
                params.iterations = params.iterations.max(500);
                params.warmup_iterations = params.warmup_iterations.max(300);
            }
            SimulationPhase::Dynamic => {
                
            }
            SimulationPhase::Finalize => {
                
                params.iterations = params.iterations.max(300);
            }
        }

        params
    }

    
    pub fn to_sim_params(&self) -> SimParams {
        
        let mut feature_flags = 0;
        if self.repel_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_REPULSION;
        }
        if self.spring_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_SPRINGS;
        }
        
        if self.center_gravity_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_CENTERING;
        }
        
        if self.use_sssp_distances {
            feature_flags |= FeatureFlags::ENABLE_SSSP_SPRING_ADJUST;
        }
        

        SimParams {
            dt: self.dt,
            damping: self.damping,
            warmup_iterations: self.warmup_iterations,
            cooling_rate: self.cooling_rate,
            spring_k: self.spring_k,
            rest_length: self.separation_radius * 2.0, 
            repel_k: self.repel_k,
            repulsion_cutoff: self.max_repulsion_dist,
            repulsion_softening_epsilon: 1e-4, 
            center_gravity_k: self.center_gravity_k, 
            max_force: self.max_force,
            max_velocity: self.max_velocity,
            grid_cell_size: self.max_repulsion_dist, 
            feature_flags,
            seed: 1337,
            iteration: 0, 
            separation_radius: self.separation_radius,
            cluster_strength: self.cluster_strength,
            alignment_strength: self.alignment_strength,
            temperature: self.temperature,
            viewport_bounds: self.viewport_bounds,
            sssp_alpha: self.sssp_alpha.unwrap_or(0.0),
            boundary_damping: self.boundary_damping,
            constraint_ramp_frames: self.constraint_ramp_frames,
            constraint_max_force_per_node: self.constraint_max_force_per_node,
            
            stability_threshold: crate::config::dev_config::physics().stability_threshold,
            min_velocity_threshold: crate::config::dev_config::physics().min_velocity_threshold,

            
            world_bounds_min: crate::config::dev_config::physics().world_bounds_min,
            world_bounds_max: crate::config::dev_config::physics().world_bounds_max,
            cell_size_lod: crate::config::dev_config::physics().cell_size_lod,
            k_neighbors_max: crate::config::dev_config::physics().k_neighbors_max,
            anomaly_detection_radius: crate::config::dev_config::physics().anomaly_detection_radius,
            learning_rate_default: crate::config::dev_config::physics().learning_rate_default,

            
            norm_delta_cap: crate::config::dev_config::physics().norm_delta_cap,
            position_constraint_attraction: crate::config::dev_config::physics()
                .position_constraint_attraction,
            lof_score_min: crate::config::dev_config::physics().lof_score_min,
            lof_score_max: crate::config::dev_config::physics().lof_score_max,
            weight_precision_multiplier: crate::config::dev_config::physics()
                .weight_precision_multiplier,
        }
    }
}

// Implementation for SimParams (GPU-aligned struct)
impl Default for SimParams {
    fn default() -> Self {
        Self::new()
    }
}

impl SimParams {
    pub fn new() -> Self {
        
        let params = SimulationParams::new();
        params.to_sim_params()
    }

    
    pub fn set_iteration(&mut self, iteration: i32) {
        self.iteration = iteration;
    }

    
    pub fn to_simulation_params(&self) -> SimulationParams {
        SimulationParams {
            enabled: true,
            auto_balance: false,
            auto_balance_interval_ms: 100,
            auto_balance_config: AutoBalanceConfig::default(),
            auto_pause_config: AutoPauseConfig::default(),
            equilibrium_stability_counter: 0,
            is_physics_paused: false,
            iterations: 100,
            dt: self.dt,
            repel_k: self.repel_k,
            mass_scale: 1.0,
            damping: self.damping,
            boundary_damping: 0.9,
            viewport_bounds: self.viewport_bounds,
            enable_bounds: true,
            max_velocity: self.max_velocity,
            max_force: self.max_force,
            spring_k: 0.0, 
            separation_radius: self.separation_radius,
            center_gravity_k: self.center_gravity_k, 
            temperature: self.temperature,
            stress_weight: 1.0,
            stress_alpha: 0.1,
            boundary_limit: 1000.0,
            alignment_strength: self.alignment_strength,
            cluster_strength: self.cluster_strength,
            compute_mode: 0,
            min_distance: 1.0,
            max_repulsion_dist: self.repulsion_cutoff,
            boundary_margin: 50.0,
            boundary_force_strength: 1.0,
            warmup_iterations: self.warmup_iterations,
            cooling_rate: self.cooling_rate,
            use_sssp_distances: false, 
            sssp_alpha: Some(self.sssp_alpha),
            constraint_ramp_frames: self.constraint_ramp_frames,
            constraint_max_force_per_node: self.constraint_max_force_per_node,
            phase: SimulationPhase::Dynamic,
            mode: SimulationMode::Remote,
        }
    }
}

// Conversion from SimulationParams to SimParams
impl From<&SimulationParams> for SimParams {
    fn from(params: &SimulationParams) -> Self {
        params.to_sim_params()
    }
}

// Conversion from SimParams to SimulationParams
impl From<&SimParams> for SimulationParams {
    fn from(params: &SimParams) -> Self {
        params.to_simulation_params()
    }
}

// Direct conversion from PhysicsSettings to SimParams for the new CUDA kernel
impl From<&PhysicsSettings> for SimParams {
    fn from(physics: &PhysicsSettings) -> Self {
        let mut feature_flags = 0;
        if physics.repel_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_REPULSION;
        }
        if physics.spring_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_SPRINGS;
        }
        if physics.center_gravity_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_CENTERING;
        }

        SimParams {
            dt: physics.dt,
            damping: physics.damping,
            warmup_iterations: physics.warmup_iterations,
            cooling_rate: physics.cooling_rate,
            spring_k: physics.spring_k,
            rest_length: physics.rest_length,
            repel_k: physics.repel_k,
            repulsion_cutoff: physics.max_repulsion_dist,
            repulsion_softening_epsilon: physics.repulsion_softening_epsilon,
            center_gravity_k: physics.center_gravity_k,
            max_force: physics.max_force,
            max_velocity: physics.max_velocity,
            grid_cell_size: physics.grid_cell_size,
            feature_flags,
            seed: 1337,
            iteration: 0,
            separation_radius: physics.separation_radius,
            cluster_strength: physics.cluster_strength,
            alignment_strength: physics.alignment_strength,
            temperature: physics.temperature,
            viewport_bounds: physics.bounds_size,
            sssp_alpha: 0.0,
            boundary_damping: physics.boundary_damping,
            constraint_ramp_frames: physics.constraint_ramp_frames,
            constraint_max_force_per_node: physics.constraint_max_force_per_node,

            stability_threshold: crate::config::dev_config::physics().stability_threshold,
            min_velocity_threshold: crate::config::dev_config::physics().min_velocity_threshold,


            world_bounds_min: crate::config::dev_config::physics().world_bounds_min,
            world_bounds_max: crate::config::dev_config::physics().world_bounds_max,
            cell_size_lod: crate::config::dev_config::physics().cell_size_lod,
            k_neighbors_max: crate::config::dev_config::physics().k_neighbors_max,
            anomaly_detection_radius: crate::config::dev_config::physics().anomaly_detection_radius,
            learning_rate_default: crate::config::dev_config::physics().learning_rate_default,


            norm_delta_cap: crate::config::dev_config::physics().norm_delta_cap,
            position_constraint_attraction: crate::config::dev_config::physics()
                .position_constraint_attraction,
            lof_score_min: crate::config::dev_config::physics().lof_score_min,
            lof_score_max: crate::config::dev_config::physics().lof_score_max,
            weight_precision_multiplier: crate::config::dev_config::physics()
                .weight_precision_multiplier,
        }
    }
}

// Conversion from PhysicsSettings to SimulationParams
impl From<&PhysicsSettings> for SimulationParams {
    fn from(physics: &PhysicsSettings) -> Self {
        Self {
            enabled: physics.enabled,
            auto_balance: physics.auto_balance,
            auto_balance_interval_ms: physics.auto_balance_interval_ms,
            auto_balance_config: physics.auto_balance_config.clone(),
            auto_pause_config: physics.auto_pause.clone(),
            is_physics_paused: false, 
            equilibrium_stability_counter: 0,
            iterations: physics.iterations,
            dt: physics.dt,
            spring_k: physics.spring_k,
            repel_k: physics.repel_k,
            mass_scale: physics.mass_scale,
            damping: physics.damping,
            boundary_damping: physics.boundary_damping,
            viewport_bounds: physics.bounds_size,
            enable_bounds: physics.enable_bounds,
            max_velocity: physics.max_velocity,
            max_force: physics.max_force, 
            separation_radius: physics.separation_radius,
            temperature: physics.temperature,
            center_gravity_k: physics.center_gravity_k,
            
            stress_weight: physics.stress_weight,
            stress_alpha: physics.stress_alpha,
            boundary_limit: physics.boundary_limit,
            alignment_strength: physics.alignment_strength,
            cluster_strength: physics.cluster_strength,
            compute_mode: physics.compute_mode,
            min_distance: physics.min_distance,
            max_repulsion_dist: physics.max_repulsion_dist,
            boundary_margin: physics.boundary_margin,
            boundary_force_strength: physics.boundary_force_strength,
            warmup_iterations: physics.warmup_iterations,
            cooling_rate: physics.cooling_rate,
            use_sssp_distances: false, 
            sssp_alpha: None,          
            constraint_ramp_frames: physics.constraint_ramp_frames,
            constraint_max_force_per_node: physics.constraint_max_force_per_node,
            phase: SimulationPhase::Dynamic,
            mode: SimulationMode::Remote,
        }
    }
}
