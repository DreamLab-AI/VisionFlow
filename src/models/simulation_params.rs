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
        Self {
            enabled: true,              // Physics enabled by default
            auto_balance: false,
            auto_balance_interval_ms: 500,
            auto_balance_config: AutoBalanceConfig::default(),
            iterations: 200,
            dt: 0.01,                   // Much smaller for stability
            spring_k: 0.005,             // Very gentle springs
            repel_k: 50.0,               // Dramatically reduced from 1000.0
            mass_scale: 1.0,
            damping: 0.9,               // High damping for stability
            boundary_damping: 0.95,     
            viewport_bounds: 200.0,     // Smaller viewport
            enable_bounds: true,
            max_velocity: 1.0,          // From settings.yaml
            max_force: 10.0,            // Independent max force limit
            attraction_k: 0.001,         // From settings.yaml
            separation_radius: 0.15,     // From settings.yaml
            temperature: 0.5,            // From settings.yaml
            // GPU parameters with defaults
            stress_weight: 0.1,
            stress_alpha: 0.1,
            boundary_limit: 196.0,      // 98% of viewport_bounds
            alignment_strength: 0.0,
            cluster_strength: 0.0,
            compute_mode: 0,
            min_distance: 0.15,
            max_repulsion_dist: 50.0,
            boundary_margin: 0.85,
            boundary_force_strength: 2.0,
            warmup_iterations: 200,
            cooling_rate: 0.0001,
            phase: SimulationPhase::Initial,
            mode: SimulationMode::Remote,
        }
    }

    pub fn with_phase(phase: SimulationPhase) -> Self {
        match phase {
            SimulationPhase::Initial => Self {
                enabled: true,              // Physics enabled by default
                auto_balance: false,
                auto_balance_interval_ms: 500,
                auto_balance_config: AutoBalanceConfig::default(),
                iterations: 500,
                dt: 0.01,                  // Small timestep for stability
                spring_k: 0.005,            // Very gentle springs
                repel_k: 50.0,              // Much lower repulsion
                mass_scale: 1.0,           // Standard mass
                damping: 0.95,             // Very high damping
                boundary_damping: 0.95,
                viewport_bounds: 8000.0,   // Much larger bounds
                enable_bounds: true,
                max_velocity: 1.0,
                max_force: 5.0,            // Lower force limit for initial phase
                attraction_k: 0.001,
                separation_radius: 0.15,
                temperature: 0.5,
                // GPU parameters
                stress_weight: 0.1,
                stress_alpha: 0.1,
                boundary_limit: 7840.0,    // 98% of viewport_bounds
                alignment_strength: 0.0,
                cluster_strength: 0.0,
                compute_mode: 0,
                min_distance: 0.15,
                max_repulsion_dist: 50.0,
                boundary_margin: 0.85,
                boundary_force_strength: 2.0,
                warmup_iterations: 200,
                cooling_rate: 0.0001,
                phase,
                mode: SimulationMode::Remote,
            },
            SimulationPhase::Dynamic => Self {
                enabled: true,
                auto_balance: false,
                auto_balance_interval_ms: 500,
                auto_balance_config: AutoBalanceConfig::default(),
                iterations: 100,
                dt: 0.12,                  // Further reduced for optimal stability
                spring_k: 0.01,            // Reduced spring strength
                repel_k: 600.0,            // Further reduced from 800.0
                mass_scale: 1.5,
                damping: 0.9,  // Increased from 0.85
                boundary_damping: 0.95,  // Increased boundary damping
                viewport_bounds: 5000.0,
                enable_bounds: true,
                max_velocity: 2.0,
                max_force: 10.0,           // Standard force limit
                attraction_k: 0.002,
                separation_radius: 0.2,
                temperature: 0.3,
                // GPU parameters
                stress_weight: 0.1,
                stress_alpha: 0.1,
                boundary_limit: 4900.0,    // 98% of viewport_bounds
                alignment_strength: 0.0,
                cluster_strength: 0.0,
                compute_mode: 0,
                min_distance: 0.15,
                max_repulsion_dist: 1500.0,
                boundary_margin: 0.85,
                boundary_force_strength: 2.0,
                warmup_iterations: 200,
                cooling_rate: 0.0001,
                phase,
                mode: SimulationMode::Remote,
            },
            SimulationPhase::Finalize => Self {
                enabled: true,
                auto_balance: false,
                auto_balance_interval_ms: 500,
                auto_balance_config: AutoBalanceConfig::default(),
                iterations: 300,
                dt: 0.15,                  // Reduced from 0.2
                spring_k: 0.005,            // Minimal spring forces
                repel_k: 600.0,             // Reduced from 800.0
                mass_scale: 1.2,           // Moderate mass influence
                damping: 0.95,             // High damping for stability
                boundary_damping: 0.95,
                viewport_bounds: 5000.0,
                enable_bounds: true,
                max_velocity: 1.5,
                max_force: 15.0,           // Higher force limit for final positioning
                attraction_k: 0.001,
                separation_radius: 0.15,
                temperature: 0.2,
                // GPU parameters
                stress_weight: 0.1,
                stress_alpha: 0.1,
                boundary_limit: 4900.0,    // 98% of viewport_bounds
                alignment_strength: 0.0,
                cluster_strength: 0.0,
                compute_mode: 0,
                min_distance: 0.15,
                max_repulsion_dist: 1200.0,
                boundary_margin: 0.85,
                boundary_force_strength: 2.0,
                warmup_iterations: 200,
                cooling_rate: 0.0001,
                phase,
                mode: SimulationMode::Remote,
            },
        }
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
        Self {
            spring_k: 0.005,
            repel_k: 50.0,
            damping: 0.9,
            dt: 0.01,
            max_velocity: 1.0,
            max_force: 10.0,
            stress_weight: 0.1,
            stress_alpha: 0.1,
            separation_radius: 0.15,
            boundary_limit: 196.0,  // 98% of viewport_bounds
            alignment_strength: 0.0,
            cluster_strength: 0.0,
            boundary_damping: 0.95,
            viewport_bounds: 200.0,
            temperature: 0.5,
            iteration: 0,
            compute_mode: 0,
            min_distance: 0.15,
            max_repulsion_dist: 50.0,
            boundary_margin: 0.85,
            boundary_force_strength: 2.0,
            warmup_iterations: 200,
            cooling_rate: 0.0001,
        }
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
            max_force: 10.0,  // Default max force
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
            min_distance: 0.15,  // Default values for fields not in PhysicsSettings
            max_repulsion_dist: physics.max_repulsion_dist,
            boundary_margin: 0.85,
            boundary_force_strength: 2.0,
            warmup_iterations: 200,
            cooling_rate: 0.0001,
        }
    }
}

// Conversion from PhysicsSettings to SimulationParams
impl From<&PhysicsSettings> for SimulationParams {
    fn from(physics: &PhysicsSettings) -> Self {
        Self {
            enabled: physics.enabled,  // Use the enabled flag from settings
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
            max_force: 10.0,  // Default independent max force
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
            min_distance: 0.15,  // Default values for fields not in PhysicsSettings
            max_repulsion_dist: physics.max_repulsion_dist,
            boundary_margin: 0.85,
            boundary_force_strength: 2.0,
            warmup_iterations: 200,
            cooling_rate: 0.0001,
            phase: SimulationPhase::Dynamic,
            mode: SimulationMode::Remote,
        }
    }
}
