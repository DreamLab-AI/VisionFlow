// src/ports/physics_simulator.rs
//! Physics Simulator Port
//!
//! Defines the interface for physics simulation operations.
//! Abstracts GPU compute, CPU fallback, or any other physics engine.

use async_trait::async_trait;

use crate::models::graph::GraphData;

// Placeholder for BinaryNodeData - will use actual type from GPU module
pub type BinaryNodeData = (f32, f32, f32);
use crate::config::PhysicsSettings;

pub type Result<T> = std::result::Result<T, PhysicsSimulatorError>;

#[derive(Debug, thiserror::Error)]
pub enum PhysicsSimulatorError {
    #[error("Simulation error: {0}")]
    SimulationError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("GPU error: {0}")]
    GpuError(String),
}

#[derive(Debug, Clone)]
pub struct SimulationParams {
    pub settings: PhysicsSettings,
    pub graph_name: String,
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub node_id: u32,
    pub constraint_type: ConstraintType,
    pub target_position: Option<(f32, f32, f32)>,
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub enum ConstraintType {
    Fixed,
    Spring,
    Boundary,
}

/// Port for physics simulation operations
#[async_trait]
pub trait PhysicsSimulator: Send + Sync {
    /// Run a single simulation step and return position updates
    async fn run_simulation_step(&self, graph: &GraphData) -> Result<Vec<(u32, BinaryNodeData)>>;

    /// Update simulation parameters
    async fn update_params(&self, params: SimulationParams) -> Result<()>;

    /// Apply constraints from ontology or user-defined
    async fn apply_constraints(&self, constraints: Vec<Constraint>) -> Result<()>;

    /// Start continuous simulation
    async fn start_simulation(&self) -> Result<()>;

    /// Stop continuous simulation
    async fn stop_simulation(&self) -> Result<()>;

    /// Check if simulation is running
    async fn is_running(&self) -> Result<bool>;
}
