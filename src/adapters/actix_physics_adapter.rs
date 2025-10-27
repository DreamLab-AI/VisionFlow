// src/adapters/actix_physics_adapter.rs
//! Actix Physics Adapter
//!
//! Implements the GpuPhysicsAdapter port by wrapping the PhysicsOrchestratorActor.
//! This adapter bridges the hexagonal architecture port interface with the Actix actor system.

use actix::prelude::*;
use async_trait::async_trait;
use log::{debug, info, warn};
use std::sync::Arc;
use std::time::Duration;

use crate::actors::physics_orchestrator_actor::PhysicsOrchestratorActor;
use crate::adapters::messages::*;
use crate::models::graph::GraphData;
use crate::ports::gpu_physics_adapter::{
    GpuDeviceInfo, GpuPhysicsAdapter, NodeForce, PhysicsParameters, PhysicsStatistics,
    PhysicsStepResult, Result as PortResult,
};

/// Default timeout for actor message responses (30 seconds)
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Actix-based implementation of GpuPhysicsAdapter
///
/// This adapter wraps a PhysicsOrchestratorActor and translates port method calls
/// into actor messages. It handles:
/// - Actor lifecycle management (start/stop)
/// - Message passing with timeouts
/// - Error translation between actor and port layers
/// - Type conversion between port and actor domains
pub struct ActixPhysicsAdapter {
    /// Address of the physics orchestrator actor
    actor_addr: Option<Addr<PhysicsOrchestratorActor>>,

    /// Message timeout duration
    timeout: Duration,

    /// Whether the adapter has been initialized
    initialized: bool,

    /// Current physics parameters for reference
    current_params: Option<PhysicsParameters>,
}

impl ActixPhysicsAdapter {
    /// Create a new ActixPhysicsAdapter
    ///
    /// The adapter starts without an actor address - initialize() must be called
    /// to create and start the actor.
    pub fn new() -> Self {
        info!("Creating ActixPhysicsAdapter");
        Self {
            actor_addr: None,
            timeout: DEFAULT_TIMEOUT,
            initialized: false,
            current_params: None,
        }
    }

    /// Create adapter with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        info!(
            "Creating ActixPhysicsAdapter with custom timeout: {:?}",
            timeout
        );
        Self {
            actor_addr: None,
            timeout,
            initialized: false,
            current_params: None,
        }
    }

    /// Create adapter with existing actor address (for testing)
    pub fn from_actor(actor_addr: Addr<PhysicsOrchestratorActor>) -> Self {
        info!("Creating ActixPhysicsAdapter from existing actor");
        Self {
            actor_addr: Some(actor_addr),
            timeout: DEFAULT_TIMEOUT,
            initialized: true,
            current_params: None,
        }
    }

    /// Get reference to actor address
    pub fn actor_addr(&self) -> Option<&Addr<PhysicsOrchestratorActor>> {
        self.actor_addr.as_ref()
    }

    /// Set custom timeout for messages
    pub fn set_timeout(&mut self, timeout: Duration) {
        self.timeout = timeout;
    }

    /// Helper to send message with timeout and error conversion
    async fn send_message<M>(&self, msg: M) -> PortResult<M::Result>
    where
        M: Message + Send + 'static,
        M::Result: Send,
        PhysicsOrchestratorActor: Handler<M>,
    {
        let addr = self.actor_addr.as_ref().ok_or_else(|| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::GraphNotLoaded
        })?;

        tokio::time::timeout(self.timeout, addr.send(msg))
            .await
            .map_err(|_| {
                warn!("Actor message timeout");
                crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(
                    "Actor communication timeout".to_string(),
                )
            })?
            .map_err(|e| {
                warn!("Actor mailbox error: {}", e);
                crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(
                    format!("Actor communication error: {}", e),
                )
            })
    }
}

impl Default for ActixPhysicsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GpuPhysicsAdapter for ActixPhysicsAdapter {
    /// Initialize physics simulation with graph data and parameters
    ///
    /// This creates and starts the PhysicsOrchestratorActor if not already running,
    /// then sends initialization message with graph and parameters.
    async fn initialize(
        &mut self,
        graph: Arc<GraphData>,
        params: PhysicsParameters,
    ) -> PortResult<()> {
        info!(
            "Initializing ActixPhysicsAdapter with {} nodes",
            graph.nodes.len()
        );

        // Create actor if not exists
        if self.actor_addr.is_none() {
            // Note: In production, would pass GPU compute address and graph data to actor
            // For now, create basic actor
            let simulation_params = crate::models::simulation_params::SimulationParams::default();

            #[cfg(feature = "gpu")]
            let actor = PhysicsOrchestratorActor::new(
                simulation_params,
                None, // GPU compute address would be set here
                Some(graph.clone()),
            );

            #[cfg(not(feature = "gpu"))]
            let actor = PhysicsOrchestratorActor::new(simulation_params, Some(graph.clone()));

            let addr = actor.start();
            self.actor_addr = Some(addr);
        }

        // Send initialization message
        let msg = InitializePhysicsMessage::new(graph, params.clone());
        self.send_message(msg).await?;

        self.initialized = true;
        self.current_params = Some(params);

        Ok(())
    }

    /// Compute forces for all nodes
    async fn compute_forces(&mut self) -> PortResult<Vec<NodeForce>> {
        debug!("Computing forces via actor");

        let addr = self.actor_addr.as_ref().ok_or_else(|| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::GraphNotLoaded
        })?;

        let result = tokio::time::timeout(self.timeout, addr.send(ComputeForcesMessage))
            .await
            .map_err(|_| {
                warn!("Actor message timeout");
                crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(
                    "Actor communication timeout".to_string(),
                )
            })?
            .map_err(|e| {
                warn!("Actor mailbox error: {}", e);
                crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(
                    format!("Actor communication error: {}", e),
                )
            })?;

        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Update node positions based on computed forces
    async fn update_positions(
        &mut self,
        forces: &[NodeForce],
    ) -> PortResult<Vec<(u32, f32, f32, f32)>> {
        debug!("Updating positions for {} nodes via actor", forces.len());
        let msg = UpdatePositionsMessage::new(forces.to_vec());
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Perform complete physics simulation step
    async fn step(&mut self) -> PortResult<PhysicsStepResult> {
        debug!("Executing physics step via actor");
        let msg = PhysicsStepMessage;
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Run simulation until convergence or max iterations
    async fn simulate_until_convergence(&mut self) -> PortResult<PhysicsStepResult> {
        info!("Running simulation until convergence via actor");
        let msg = SimulateUntilConvergenceMessage;
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Apply custom external forces to specific nodes
    async fn apply_external_forces(&mut self, forces: Vec<(u32, f32, f32, f32)>) -> PortResult<()> {
        debug!(
            "Applying external forces to {} nodes via actor",
            forces.len()
        );
        let msg = ApplyExternalForcesMessage::new(forces);
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Pin nodes at specific positions
    async fn pin_nodes(&mut self, nodes: Vec<(u32, f32, f32, f32)>) -> PortResult<()> {
        debug!("Pinning {} nodes via actor", nodes.len());
        let msg = PinNodesMessage::new(nodes);
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Unpin nodes to allow free movement
    async fn unpin_nodes(&mut self, node_ids: Vec<u32>) -> PortResult<()> {
        debug!("Unpinning {} nodes via actor", node_ids.len());
        let msg = UnpinNodesMessage::new(node_ids);
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Update physics parameters without reinitializing
    async fn update_parameters(&mut self, params: PhysicsParameters) -> PortResult<()> {
        info!("Updating physics parameters via actor");
        let msg = UpdatePhysicsParametersMessage::new(params.clone());
        self.send_message(msg).await?;

        self.current_params = Some(params);
        Ok(())
    }

    /// Update graph data (e.g., after nodes/edges added/removed)
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> PortResult<()> {
        info!(
            "Updating graph data with {} nodes via actor",
            graph.nodes.len()
        );
        let msg = UpdatePhysicsGraphDataMessage::new(graph);
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Get GPU device information
    async fn get_gpu_status(&self) -> PortResult<GpuDeviceInfo> {
        debug!("Getting GPU status via actor");
        let msg = GetGpuStatusMessage;
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Get physics simulation statistics
    async fn get_statistics(&self) -> PortResult<PhysicsStatistics> {
        debug!("Getting physics statistics via actor");
        let msg = GetPhysicsStatisticsMessage;
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Reset simulation state (clear velocities, forces)
    async fn reset(&mut self) -> PortResult<()> {
        info!("Resetting physics simulation via actor");
        let msg = ResetPhysicsMessage;
        let result = self.send_message(msg).await?;
        result.map_err(|e| {
            crate::ports::gpu_physics_adapter::GpuPhysicsAdapterError::ComputationError(e)
        })
    }

    /// Free GPU resources and cleanup
    async fn cleanup(&mut self) -> PortResult<()> {
        info!("Cleaning up physics adapter");

        if let Some(addr) = self.actor_addr.take() {
            let msg = CleanupPhysicsMessage;

            // Send cleanup message with timeout
            if let Err(e) = addr.send(msg).timeout(self.timeout).await {
                warn!("Cleanup message failed: {}", e);
            }

            // Stop actor (send stop message)
            // Note: Actor will stop when all addresses are dropped
        }

        self.initialized = false;
        self.current_params = None;

        Ok(())
    }
}

// ============================================================================
// Message Handlers for PhysicsOrchestratorActor
// ============================================================================

// These handlers translate between the adapter messages and the actor's internal methods

impl Handler<InitializePhysicsMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitializePhysicsMessage, _ctx: &mut Self::Context) -> Self::Result {
        info!("PhysicsOrchestratorActor: Handling initialization");

        // Update graph data in actor
        use crate::actors::physics_orchestrator_actor::UpdateGraphData;
        self.handle(
            UpdateGraphData {
                graph_data: msg.graph,
            },
            _ctx,
        );

        // Update parameters
        use crate::actors::messages::UpdateSimulationParams;
        let simulation_params = crate::models::simulation_params::SimulationParams {
            repel_k: msg.params.repulsion_strength,
            spring_k: msg.params.spring_constant,
            damping: msg.params.damping,
            max_velocity: msg.params.max_velocity,
            ..Default::default()
        };

        self.handle(
            UpdateSimulationParams {
                params: simulation_params,
            },
            _ctx,
        )?;

        Ok(())
    }
}

impl Handler<ComputeForcesMessage> for PhysicsOrchestratorActor {
    type Result = Result<Vec<NodeForce>, String>;

    fn handle(&mut self, _msg: ComputeForcesMessage, _ctx: &mut Self::Context) -> Self::Result {
        debug!("PhysicsOrchestratorActor: Computing forces");

        // Placeholder: actual implementation would compute forces from GPU
        // For now, return empty forces
        Ok(Vec::new())
    }
}

impl Handler<UpdatePositionsMessage> for PhysicsOrchestratorActor {
    type Result = Result<Vec<(u32, f32, f32, f32)>, String>;

    fn handle(&mut self, msg: UpdatePositionsMessage, _ctx: &mut Self::Context) -> Self::Result {
        debug!(
            "PhysicsOrchestratorActor: Updating positions for {} forces",
            msg.forces.len()
        );

        // Placeholder: actual implementation would update positions in GPU
        // For now, return empty positions
        Ok(Vec::new())
    }
}

impl Handler<PhysicsStepMessage> for PhysicsOrchestratorActor {
    type Result = Result<PhysicsStepResult, String>;

    fn handle(&mut self, _msg: PhysicsStepMessage, ctx: &mut Self::Context) -> Self::Result {
        debug!("PhysicsOrchestratorActor: Executing physics step");

        // Use existing SimulationStep handler
        use crate::actors::messages::SimulationStep;
        self.handle(SimulationStep, ctx)?;

        // Return placeholder result
        Ok(PhysicsStepResult {
            nodes_updated: 0,
            total_energy: 0.0,
            max_displacement: 0.0,
            converged: false,
            computation_time_ms: 0.0,
        })
    }
}

impl Handler<SimulateUntilConvergenceMessage> for PhysicsOrchestratorActor {
    type Result = Result<PhysicsStepResult, String>;

    fn handle(
        &mut self,
        _msg: SimulateUntilConvergenceMessage,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("PhysicsOrchestratorActor: Simulating until convergence");

        // Placeholder: actual implementation would run loop until convergence
        Ok(PhysicsStepResult {
            nodes_updated: 0,
            total_energy: 0.0,
            max_displacement: 0.0,
            converged: true,
            computation_time_ms: 0.0,
        })
    }
}

impl Handler<ApplyExternalForcesMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        msg: ApplyExternalForcesMessage,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        debug!(
            "PhysicsOrchestratorActor: Applying {} external forces",
            msg.forces.len()
        );
        Ok(())
    }
}

impl Handler<PinNodesMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: PinNodesMessage, _ctx: &mut Self::Context) -> Self::Result {
        debug!(
            "PhysicsOrchestratorActor: Pinning {} nodes",
            msg.nodes.len()
        );
        Ok(())
    }
}

impl Handler<UnpinNodesMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UnpinNodesMessage, _ctx: &mut Self::Context) -> Self::Result {
        debug!(
            "PhysicsOrchestratorActor: Unpinning {} nodes",
            msg.node_ids.len()
        );
        Ok(())
    }
}

impl Handler<UpdatePhysicsParametersMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        msg: UpdatePhysicsParametersMessage,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("PhysicsOrchestratorActor: Updating physics parameters");

        use crate::actors::messages::UpdateSimulationParams;
        let simulation_params = crate::models::simulation_params::SimulationParams {
            repel_k: msg.params.repulsion_strength,
            spring_k: msg.params.spring_constant,
            damping: msg.params.damping,
            max_velocity: msg.params.max_velocity,
            ..Default::default()
        };

        self.handle(
            UpdateSimulationParams {
                params: simulation_params,
            },
            ctx,
        )
    }
}

impl Handler<UpdatePhysicsGraphDataMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        msg: UpdatePhysicsGraphDataMessage,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("PhysicsOrchestratorActor: Updating graph data");

        use crate::actors::physics_orchestrator_actor::UpdateGraphData;
        self.handle(
            UpdateGraphData {
                graph_data: msg.graph,
            },
            ctx,
        );
        Ok(())
    }
}

impl Handler<GetGpuStatusMessage> for PhysicsOrchestratorActor {
    type Result = Result<GpuDeviceInfo, String>;

    fn handle(&mut self, _msg: GetGpuStatusMessage, _ctx: &mut Self::Context) -> Self::Result {
        debug!("PhysicsOrchestratorActor: Getting GPU status");

        // Placeholder: actual implementation would query GPU
        Ok(GpuDeviceInfo {
            device_id: 0,
            device_name: "Simulated GPU".to_string(),
            compute_capability: (7, 5),
            total_memory_mb: 8192,
            free_memory_mb: 4096,
            multiprocessor_count: 40,
            warp_size: 32,
            max_threads_per_block: 1024,
        })
    }
}

impl Handler<GetPhysicsStatisticsMessage> for PhysicsOrchestratorActor {
    type Result = Result<PhysicsStatistics, String>;

    fn handle(
        &mut self,
        _msg: GetPhysicsStatisticsMessage,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        debug!("PhysicsOrchestratorActor: Getting physics statistics");

        // Placeholder: actual implementation would collect real stats
        Ok(PhysicsStatistics {
            total_steps: 0,
            average_step_time_ms: 0.0,
            average_energy: 0.0,
            gpu_memory_used_mb: 0.0,
            cache_hit_rate: 0.0,
            last_convergence_iterations: 0,
        })
    }
}

impl Handler<ResetPhysicsMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: ResetPhysicsMessage, _ctx: &mut Self::Context) -> Self::Result {
        info!("PhysicsOrchestratorActor: Resetting simulation");
        Ok(())
    }
}

impl Handler<CleanupPhysicsMessage> for PhysicsOrchestratorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: CleanupPhysicsMessage, _ctx: &mut Self::Context) -> Self::Result {
        info!("PhysicsOrchestratorActor: Cleaning up resources");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::node::Node;
    use crate::utils::socket_flow_messages::BinaryNodeData;

    #[actix_rt::test]
    async fn test_adapter_creation() {
        let adapter = ActixPhysicsAdapter::new();
        assert!(!adapter.initialized);
        assert!(adapter.actor_addr.is_none());
    }

    #[actix_rt::test]
    async fn test_adapter_with_timeout() {
        let timeout = Duration::from_secs(60);
        let adapter = ActixPhysicsAdapter::with_timeout(timeout);
        assert_eq!(adapter.timeout, timeout);
    }

    #[actix_rt::test]
    async fn test_adapter_initialize() {
        let mut adapter = ActixPhysicsAdapter::new();

        let nodes = vec![Node {
            id: 1,
            data: BinaryNodeData::default(),
        }];
        let graph = Arc::new(GraphData {
            nodes,
            edges: Vec::new(),
        });

        let params = PhysicsParameters::default();

        let result = adapter.initialize(graph, params).await;
        assert!(result.is_ok());
        assert!(adapter.initialized);
    }
}
