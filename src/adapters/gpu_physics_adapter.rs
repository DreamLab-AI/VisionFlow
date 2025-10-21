// src/adapters/gpu_physics_adapter.rs
//! GPU Physics Simulator Adapter
//!
//! Implements PhysicsSimulator port using GPU compute actor

use async_trait::async_trait;

use crate::ports::physics_simulator::{PhysicsSimulator, Result, PhysicsSimulatorError, SimulationParams, Constraint};
use crate::ports::physics_simulator::BinaryNodeData;
use crate::models::graph::GraphData;

/// Adapter that implements PhysicsSimulator using GPU compute
pub struct GpuPhysicsAdapter {
    // Will be populated with actual GPU actor address later
}

impl GpuPhysicsAdapter {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl PhysicsSimulator for GpuPhysicsAdapter {
    async fn run_simulation_step(&self, _graph: &GraphData) -> Result<Vec<(u32, BinaryNodeData)>> {
        // Placeholder - will call ForceComputeActor
        Err(PhysicsSimulatorError::SimulationError("Not yet implemented".to_string()))
    }

    async fn update_params(&self, _params: SimulationParams) -> Result<()> {
        // Placeholder - will call ForceComputeActor
        Err(PhysicsSimulatorError::SimulationError("Not yet implemented".to_string()))
    }

    async fn apply_constraints(&self, _constraints: Vec<Constraint>) -> Result<()> {
        // Placeholder - will call ForceComputeActor
        Err(PhysicsSimulatorError::SimulationError("Not yet implemented".to_string()))
    }

    async fn start_simulation(&self) -> Result<()> {
        // Placeholder - will call ForceComputeActor
        Err(PhysicsSimulatorError::SimulationError("Not yet implemented".to_string()))
    }

    async fn stop_simulation(&self) -> Result<()> {
        // Placeholder - will call ForceComputeActor
        Err(PhysicsSimulatorError::SimulationError("Not yet implemented".to_string()))
    }

    async fn is_running(&self) -> Result<bool> {
        // Placeholder - will call ForceComputeActor
        Ok(false)
    }
}
