// Port: PhysicsSimulator
// Defines the interface for physics simulation
// Future: Add #[derive(HexPort)] when Hexser is available

use async_trait::async_trait;
use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::Constraint;
use crate::utils::socket_flow_messages::BinaryNodeData;

pub type Result<T> = std::result::Result<T, String>;

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

    /// Set SSSP source node for pathfinding visualization
    async fn set_sssp_source(&self, source: Option<u32>) -> Result<()>;
}
