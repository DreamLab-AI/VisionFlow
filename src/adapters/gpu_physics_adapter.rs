// GpuPhysicsAdapter - Adapter wrapping PhysicsOrchestratorActor and GPUManagerActor
// Implements PhysicsSimulator port for hexagonal architecture
// Delegates to existing physics actors for GPU-accelerated simulation

use async_trait::async_trait;
use actix::Addr;

use crate::ports::physics_simulator::{PhysicsSimulator, Result};
use crate::actors::physics_orchestrator_actor::PhysicsOrchestratorActor;
use crate::actors::gpu::gpu_manager_actor::GPUManagerActor;
use crate::actors::messages::{
    StartSimulation, StopSimulation, UpdateSimulationParams,
    ApplyOntologyConstraints, ConstraintMergeMode,
    RequestPositionSnapshot,
};
use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::{Constraint, ConstraintSet};
use crate::utils::socket_flow_messages::BinaryNodeData;

/// Adapter that wraps PhysicsOrchestratorActor and GPUManagerActor
/// to implement PhysicsSimulator trait
pub struct GpuPhysicsAdapter {
    physics_orchestrator: Addr<PhysicsOrchestratorActor>,
    gpu_manager: Option<Addr<GPUManagerActor>>,
    sssp_source: Arc<std::sync::RwLock<Option<u32>>>,
}

impl GpuPhysicsAdapter {
    /// Create new adapter with physics orchestrator
    pub fn new(
        physics_orchestrator: Addr<PhysicsOrchestratorActor>,
        gpu_manager: Option<Addr<GPUManagerActor>>,
    ) -> Self {
        Self {
            physics_orchestrator,
            gpu_manager,
            sssp_source: Arc::new(std::sync::RwLock::new(None)),
        }
    }
}

use std::sync::Arc;

#[async_trait]
impl PhysicsSimulator for GpuPhysicsAdapter {
    async fn run_simulation_step(&self, _graph: &GraphData) -> Result<Vec<(u32, BinaryNodeData)>> {
        // Get position snapshot from physics orchestrator
        let snapshot = self.physics_orchestrator
            .send(RequestPositionSnapshot {
                include_knowledge_graph: true,
                include_agent_graph: false,
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))??;

        Ok(snapshot.knowledge_nodes)
    }

    async fn update_params(&self, params: SimulationParams) -> Result<()> {
        self.physics_orchestrator
            .send(UpdateSimulationParams { params })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }

    async fn apply_constraints(&self, constraints: Vec<Constraint>) -> Result<()> {
        // Build constraint set
        let mut constraint_set = ConstraintSet::default();
        for constraint in constraints {
            constraint_set.constraints.push(constraint);
        }

        // Apply as ontology constraints with replace mode
        self.physics_orchestrator
            .send(ApplyOntologyConstraints {
                constraint_set,
                merge_mode: ConstraintMergeMode::Replace,
                graph_id: 0,
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }

    async fn start_simulation(&self) -> Result<()> {
        self.physics_orchestrator
            .send(StartSimulation)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }

    async fn stop_simulation(&self) -> Result<()> {
        self.physics_orchestrator
            .send(StopSimulation)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }

    async fn is_running(&self) -> Result<bool> {
        // Query physics status through GetPhysicsStatus message
        use crate::actors::physics_orchestrator_actor::GetPhysicsStatus;

        let status = self.physics_orchestrator
            .send(GetPhysicsStatus)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?;

        Ok(status.simulation_running)
    }

    async fn set_sssp_source(&self, source: Option<u32>) -> Result<()> {
        // Store SSSP source for visualization
        self.sssp_source
            .write()
            .map(|mut s| *s = source)
            .map_err(|e| format!("Lock poisoned: {}", e))?;

        // If GPU manager available, could trigger SSSP computation
        // Note: GPUManagerActor doesn't currently implement ComputeShortestPaths
        // SSSP computation should be delegated to SemanticProcessorActor instead
        if let Some(_gpu_manager) = &self.gpu_manager {
            if let Some(_source_node) = source {
                // TODO: Implement SSSP visualization trigger via SemanticProcessorActor
                log::debug!("SSSP source set to {:?}, but computation not triggered (not implemented in GPUManagerActor)", source);
            }
        }

        Ok(())
    }
}
