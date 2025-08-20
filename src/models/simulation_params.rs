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

// GPU-compatible simulation parameters, matching the new CUDA kernel design.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SimParams {
    // Integration and Damping
    pub dt: f32,
    pub damping: f32,
    pub warmup_iterations: u32,
    pub cooling_rate: f32,

    // Spring Forces
    pub spring_k: f32,
    pub rest_length: f32,

    // Repulsion Forces
    pub repel_k: f32,
    pub repulsion_cutoff: f32,
    pub repulsion_softening_epsilon: f32,

    // Global Forces & Clamping
    pub center_gravity_k: f32,
    pub max_force: f32,
    pub max_velocity: f32,

    // Spatial Grid
    pub grid_cell_size: f32,

    // System State
    pub feature_flags: u32,
    pub seed: u32,
    pub iteration: i32,
    
    // Additional fields for compatibility
    pub separation_radius: f32,
    pub cluster_strength: f32,
    pub alignment_strength: f32,
    pub temperature: f32,
    pub viewport_bounds: f32,
}

/// Bitmask for enabling/disabling features in the CUDA kernel.
pub struct FeatureFlags;
impl FeatureFlags {
    pub const ENABLE_REPULSION: u32 = 1 << 0;
    pub const ENABLE_SPRINGS: u32 = 1 << 1;
    pub const ENABLE_CENTERING: u32 = 1 << 2;
    pub const ENABLE_TEMPORAL_COHERENCE: u32 = 1 << 3;
    pub const ENABLE_CONSTRAINTS: u32 = 1 << 4;
    pub const ENABLE_STRESS_MAJORIZATION: u32 = 1 << 5;
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

    // Convert to GPU-compatible parameters
    pub fn to_sim_params(&self) -> SimParams {
        // This conversion maps the high-level host settings to the low-level GPU struct.
        let mut feature_flags = 0;
        if self.repel_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_REPULSION;
        }
        if self.spring_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_SPRINGS;
        }
        if self.attraction_k > 0.0 {
            feature_flags |= FeatureFlags::ENABLE_CENTERING;
        }
        // Add other feature flags based on settings as needed.

        SimParams {
            dt: self.dt,
            damping: self.damping,
            warmup_iterations: self.warmup_iterations,
            cooling_rate: self.cooling_rate,
            spring_k: self.spring_k,
            rest_length: self.separation_radius * 2.0, // Default derivation - will be overridden by direct conversion
            repel_k: self.repel_k,
            repulsion_cutoff: self.max_repulsion_dist,
            repulsion_softening_epsilon: 1e-4, // Default - will be overridden by direct conversion
            center_gravity_k: self.attraction_k, // Use attraction_k as center gravity
            max_force: self.max_force,
            max_velocity: self.max_velocity,
            grid_cell_size: self.max_repulsion_dist, // Default - will be overridden by direct conversion
            feature_flags,
            seed: 1337,
            iteration: 0, // Set by the simulation loop
            separation_radius: self.separation_radius,
            cluster_strength: self.cluster_strength,
            alignment_strength: self.alignment_strength,
            temperature: self.temperature,
            viewport_bounds: self.viewport_bounds,
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
        // Create from default SimulationParams which uses PhysicsSettings
        let params = SimulationParams::new();
        params.to_sim_params()
    }

    // Update iteration count (called from simulation loop)
    pub fn set_iteration(&mut self, iteration: i32) {
        self.iteration = iteration;
    }

    // Convert back to SimulationParams (for the From implementation)
    pub fn to_simulation_params(&self) -> SimulationParams {
        SimulationParams {
            enabled: true,
            auto_balance: false,
            auto_balance_interval_ms: 100,
            auto_balance_config: AutoBalanceConfig::default(),
            iterations: 100,
            dt: self.dt,
            spring_k: self.spring_k,
            repel_k: self.repel_k,
            mass_scale: 1.0,
            damping: self.damping,
            boundary_damping: 0.9,
            viewport_bounds: self.viewport_bounds,
            enable_bounds: true,
            max_velocity: self.max_velocity,
            max_force: self.max_force,
            attraction_k: self.center_gravity_k,
            separation_radius: self.separation_radius,
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
        if physics.attraction_k > 0.0 || physics.center_gravity_k > 0.0 {
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
            iteration: 0, // Set by the simulation loop
            separation_radius: physics.separation_radius,
            cluster_strength: physics.cluster_strength,
            alignment_strength: physics.alignment_strength,
            temperature: physics.temperature,
            viewport_bounds: physics.bounds_size,
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
