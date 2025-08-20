use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use crate::config::{PhysicsSettings, AutoBalanceConfig};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationMode {
    Remote,  // GPU-accelerated remote computation (default)
    Local,   // CPU-based computation (fallback only)
}

impl Default for SimulationMode {
    fn default() -> Self {
        SimulationMode::Remote
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum SimulationPhase {
    Initial,    // Heavy computation for initial layout
    Dynamic,    // Lighter computation for dynamic updates
    Finalize,   // Final positioning and cleanup
}

impl Default for SimulationPhase {
    fn default() -> Self {
        SimulationPhase::Initial
    }
}

// GPU-compatible simulation parameters - exactly matches CUDA kernel SimParams
#[repr(C)]
#[derive(Default, Clone, Copy, Pod, Zeroable, Debug)]
pub struct SimParams {
    // Force parameters
    pub spring_k: f32,
    pub repel_k: f32,
    pub damping: f32,
    pub dt: f32,
    pub max_velocity: f32,
    pub max_force: f32,
    
    // Stress majorization
    pub stress_weight: f32,
    pub stress_alpha: f32,
    
    // Constraints
    pub separation_radius: f32,
    pub boundary_limit: f32,
    pub alignment_strength: f32,
    pub cluster_strength: f32,
    
    // Boundary control
    pub boundary_damping: f32,
    
    // System
    pub viewport_bounds: f32,
    pub temperature: f32,
    pub iteration: i32,
    pub compute_mode: i32,  // 0=basic, 1=dual, 2=constraints, 3=analytics
    
    // Additional GPU fields
    pub min_distance: f32,
    pub max_repulsion_dist: f32,  // Maximum distance for repulsion forces
    pub boundary_margin: f32,
    pub boundary_force_strength: f32,
    pub warmup_iterations: u32,
    pub cooling_rate: f32,
}


#[derive(Default, Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SimulationParams {
    // Master enable flag
    pub enabled: bool,             // FIX: Added to allow disabling physics entirely
    
    // Auto-balance parameters
    pub auto_balance: bool,        // Enable neural auto-balancing
    pub auto_balance_interval_ms: u32, // Interval between auto-balance checks
    pub auto_balance_config: AutoBalanceConfig, // Configuration for auto-balance thresholds
    
    // Core iteration parameters
    pub iterations: u32,           // Range: 1-500, Default: varies by phase
    pub dt: f32,                  // Range: 0.01-1, Default: 0.2 (5fps)
    
    // Force parameters
    pub spring_k: f32,            // Range: 0.1-10, Default: 0.5
    pub repel_k: f32,             // Default: 100
    
    // Mass and damping
    pub mass_scale: f32,          // Default: 1.0, Affects force scaling
    pub damping: f32,             // Range: 0-1, Default: 0.5
    pub boundary_damping: f32,    // Range: 0.5-1, Default: 0.9
    
    // Boundary control
    pub viewport_bounds: f32,     // Range: 100-5000, Default: 1000
    pub enable_bounds: bool,      // Default: true
    
    // Additional physics parameters
    pub max_velocity: f32,        // Maximum velocity for nodes
    pub max_force: f32,           // Maximum force magnitude to prevent instability
    pub attraction_k: f32,        // Attraction between connected nodes
    pub separation_radius: f32,   // Minimum separation between nodes
    pub temperature: f32,          // System temperature for simulated annealing
    
    // GPU-specific parameters
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
    
    // Simulation state
    pub phase: SimulationPhase,   // Current simulation phase
    pub mode: SimulationMode,     // Computation mode
}

impl SimulationParams {
    pub fn new() -> Self {
        // Use default PhysicsSettings as base
        let default_physics = PhysicsSettings::default();
        Self::from(&default_physics)
    }

    pub fn with_phase(phase: SimulationPhase) -> Self {
        let mut params = Self::new();
        params.phase = phase;
        
        // Phase-specific adjustments are minimal - most values come from settings
        // Only adjust critical parameters that need to vary by phase
        match phase {
            SimulationPhase::Initial => {
                // Initial phase may need more iterations for settling
                params.iterations = params.iterations.max(500);
                params.warmup_iterations = params.warmup_iterations.max(300);
            },
            SimulationPhase::Dynamic => {
                // Dynamic phase uses default settings
            },
            SimulationPhase::Finalize => {
                // Finalize phase may need more iterations to stabilize
                params.iterations = params.iterations.max(300);
            },
        }
        
        params
    }

    // Convert to GPU-compatible parameters (new GPU-aligned format)
    pub fn to_sim_params(&self) -> SimParams {
        SimParams {
            spring_k: self.spring_k,
            repel_k: self.repel_k,
            damping: self.damping,
            dt: self.dt,
            max_velocity: self.max_velocity,
            max_force: self.max_force,
            stress_weight: self.stress_weight,
            stress_alpha: self.stress_alpha,
            separation_radius: self.separation_radius,
            boundary_limit: if self.enable_bounds { self.boundary_limit } else { 0.0 },
            alignment_strength: self.alignment_strength,
            cluster_strength: self.cluster_strength,
            boundary_damping: self.boundary_damping,
            viewport_bounds: if self.enable_bounds { self.viewport_bounds } else { 0.0 },
            temperature: self.temperature,
            iteration: 0,  // Will be set by the simulation loop
            compute_mode: self.compute_mode,
            min_distance: self.min_distance,
            max_repulsion_dist: self.max_repulsion_dist,
            boundary_margin: self.boundary_margin,
            boundary_force_strength: self.boundary_force_strength,
            warmup_iterations: self.warmup_iterations,
            cooling_rate: self.cooling_rate,
        }
    }

}

// Implementation for SimParams (GPU-aligned struct)
impl SimParams {
    pub fn new() -> Self {
        // Create from default SimulationParams which uses PhysicsSettings
        let params = SimulationParams::new();
        params.to_sim_params()
    }

    // Convert from SimParams back to SimulationParams
    pub fn to_simulation_params(&self) -> SimulationParams {
        SimulationParams {
            enabled: true,
            auto_balance: false,
            auto_balance_interval_ms: 500,
            auto_balance_config: AutoBalanceConfig::default(),
            iterations: 200,  // Default iteration count
            dt: self.dt,
            spring_k: self.spring_k,
            repel_k: self.repel_k,
            mass_scale: 1.0,  // Default mass scale
            damping: self.damping,
            boundary_damping: self.boundary_damping,
            viewport_bounds: self.viewport_bounds,
            enable_bounds: self.boundary_limit > 0.0,
            max_velocity: self.max_velocity,
            max_force: self.max_force,
            attraction_k: self.spring_k,         // Use spring_k as attraction
            separation_radius: self.separation_radius,
            temperature: self.temperature,
            // GPU parameters
            stress_weight: self.stress_weight,
            stress_alpha: self.stress_alpha,
            boundary_limit: self.boundary_limit,
            alignment_strength: self.alignment_strength,
            cluster_strength: self.cluster_strength,
            compute_mode: self.compute_mode,
            min_distance: self.min_distance,
            max_repulsion_dist: self.max_repulsion_dist,
            boundary_margin: self.boundary_margin,
            boundary_force_strength: self.boundary_force_strength,
            warmup_iterations: self.warmup_iterations,
            cooling_rate: self.cooling_rate,
            phase: SimulationPhase::Dynamic,
            mode: SimulationMode::Remote,
        }
    }

    // Update iteration count (called from simulation loop)
    pub fn set_iteration(&mut self, iteration: i32) {
        self.iteration = iteration;
    }

    // Update compute mode
    pub fn set_compute_mode(&mut self, mode: i32) {
        self.compute_mode = mode;
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

// Conversion from PhysicsSettings to SimParams
impl From<&PhysicsSettings> for SimParams {
    fn from(physics: &PhysicsSettings) -> Self {
        Self {
            spring_k: physics.spring_k,
            repel_k: physics.repel_k,
            damping: physics.damping,
            dt: physics.dt,
            max_velocity: physics.max_velocity,
            max_force: physics.max_force,
            stress_weight: physics.stress_weight,
            stress_alpha: physics.stress_alpha,
            separation_radius: physics.separation_radius,
            boundary_limit: if physics.enable_bounds { physics.boundary_limit } else { 0.0 },
            alignment_strength: physics.alignment_strength,
            cluster_strength: physics.cluster_strength,
            boundary_damping: physics.boundary_damping,
            viewport_bounds: physics.bounds_size,
            temperature: physics.temperature,
            iteration: 0,
            compute_mode: physics.compute_mode,
            min_distance: physics.min_distance,
            max_repulsion_dist: physics.max_repulsion_dist,
            boundary_margin: physics.boundary_margin,
            boundary_force_strength: physics.boundary_force_strength,
            warmup_iterations: physics.warmup_iterations,
            cooling_rate: physics.cooling_rate,
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
            max_force: physics.max_force,  // Use from settings
            attraction_k: physics.attraction_k,
            separation_radius: physics.separation_radius,
            temperature: physics.temperature,
            // GPU parameters from physics settings
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
            phase: SimulationPhase::Dynamic,
            mode: SimulationMode::Remote,
        }
    }
}
