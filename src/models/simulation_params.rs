use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};
use crate::config::PhysicsSettings;

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

// GPU-compatible simulation parameters
#[repr(C)]
#[derive(Default, Clone, Copy, Pod, Zeroable, Debug)]
pub struct GPUSimulationParams {
    pub iterations: u32,
    pub spring_strength: f32,
    pub repulsion: f32,
    pub damping: f32,
    pub max_repulsion_distance: f32,
    pub viewport_bounds: f32,
    pub mass_scale: f32,
    pub boundary_damping: f32,
}

#[derive(Default, Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SimulationParams {
    // Master enable flag
    pub enabled: bool,             // FIX: Added to allow disabling physics entirely
    
    // Core iteration parameters
    pub iterations: u32,           // Range: 1-500, Default: varies by phase
    pub time_step: f32,           // Range: 0.01-1, Default: 0.2 (5fps)
    
    // Force parameters
    pub spring_strength: f32,      // Range: 0.1-10, Default: 0.5
    pub repulsion: f32,           // Default: 100
    pub max_repulsion_distance: f32, // Default: 500
    
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
    pub attraction_strength: f32, // Attraction between connected nodes
    pub collision_radius: f32,    // Minimum separation between nodes
    pub temperature: f32,          // System temperature for simulated annealing
    
    // Simulation state
    pub phase: SimulationPhase,   // Current simulation phase
    pub mode: SimulationMode,     // Computation mode
}

impl SimulationParams {
    pub fn new() -> Self {
        Self {
            enabled: true,              // Physics enabled by default
            iterations: 200,
            time_step: 0.01,            // Much smaller for stability
            spring_strength: 0.005,      // Very gentle springs
            repulsion: 50.0,            // Dramatically reduced from 1000.0
            max_repulsion_distance: 50.0,  // Smaller cutoff radius
            mass_scale: 1.0,
            damping: 0.9,               // High damping for stability
            boundary_damping: 0.95,     
            viewport_bounds: 200.0,     // Smaller viewport
            enable_bounds: true,
            max_velocity: 1.0,          // From settings.yaml
            max_force: 10.0,            // Independent max force limit
            attraction_strength: 0.001,  // From settings.yaml
            collision_radius: 0.15,      // From settings.yaml
            temperature: 0.5,            // From settings.yaml
            phase: SimulationPhase::Initial,
            mode: SimulationMode::Remote,
        }
    }

    pub fn with_phase(phase: SimulationPhase) -> Self {
        match phase {
            SimulationPhase::Initial => Self {
                enabled: true,              // Physics enabled by default
                iterations: 500,
                time_step: 0.01,           // Small timestep for stability
                spring_strength: 0.005,     // Very gentle springs
                repulsion: 50.0,           // Much lower repulsion
                max_repulsion_distance: 50.0, // Limited range
                mass_scale: 1.0,           // Standard mass
                damping: 0.95,             // Very high damping
                boundary_damping: 0.95,
                viewport_bounds: 8000.0,   // Much larger bounds
                enable_bounds: true,
                max_velocity: 1.0,
                max_force: 5.0,            // Lower force limit for initial phase
                attraction_strength: 0.001,
                collision_radius: 0.15,
                temperature: 0.5,
                phase,
                mode: SimulationMode::Remote,
            },
            SimulationPhase::Dynamic => Self {
                enabled: true,
                iterations: 100,
                time_step: 0.12,  // Further reduced for optimal stability
                spring_strength: 0.01,  // Reduced spring strength
                repulsion: 600.0,  // Further reduced from 800.0
                max_repulsion_distance: 1500.0,
                mass_scale: 1.5,
                damping: 0.9,  // Increased from 0.85
                boundary_damping: 0.95,  // Increased boundary damping
                viewport_bounds: 5000.0,
                enable_bounds: true,
                max_velocity: 2.0,
                max_force: 10.0,           // Standard force limit
                attraction_strength: 0.002,
                collision_radius: 0.2,
                temperature: 0.3,
                phase,
                mode: SimulationMode::Remote,
            },
            SimulationPhase::Finalize => Self {
                enabled: true,
                iterations: 300,
                time_step: 0.15,           // Reduced from 0.2
                spring_strength: 0.005,      // Minimal spring forces
                repulsion: 600.0,           // Reduced from 800.0
                max_repulsion_distance: 1200.0, // Maintain spacing
                mass_scale: 1.2,           // Moderate mass influence
                damping: 0.95,             // High damping for stability
                boundary_damping: 0.95,
                viewport_bounds: 5000.0,
                enable_bounds: true,
                max_velocity: 1.5,
                max_force: 15.0,           // Higher force limit for final positioning
                attraction_strength: 0.001,
                collision_radius: 0.15,
                temperature: 0.2,
                phase,
                mode: SimulationMode::Remote,
            },
        }
    }

    // Convert to GPU-compatible parameters
    pub fn to_gpu_params(&self) -> GPUSimulationParams {
        GPUSimulationParams {
            iterations: self.iterations,
            spring_strength: self.spring_strength,
            repulsion: self.repulsion,
            damping: self.damping,
            max_repulsion_distance: self.max_repulsion_distance,
            viewport_bounds: if self.enable_bounds { self.viewport_bounds } else { 0.0 },
            mass_scale: self.mass_scale,
            boundary_damping: self.boundary_damping,
        }
    }
}

// Conversion from PhysicsSettings to SimulationParams
impl From<&PhysicsSettings> for SimulationParams {
    fn from(physics: &PhysicsSettings) -> Self {
        Self {
            enabled: physics.enabled,  // Use the enabled flag from settings
            iterations: physics.iterations,
            time_step: physics.time_step,
            spring_strength: physics.spring_strength,
            repulsion: physics.repulsion_strength,
            max_repulsion_distance: physics.repulsion_distance,
            mass_scale: physics.mass_scale,
            damping: physics.damping,
            boundary_damping: physics.boundary_damping,
            viewport_bounds: physics.bounds_size,
            enable_bounds: physics.enable_bounds,
            max_velocity: physics.max_velocity,
            max_force: 10.0,  // Default independent max force
            attraction_strength: physics.attraction_strength,
            collision_radius: physics.collision_radius,
            temperature: physics.temperature,
            phase: SimulationPhase::Dynamic,
            mode: SimulationMode::Remote,
        }
    }
}
