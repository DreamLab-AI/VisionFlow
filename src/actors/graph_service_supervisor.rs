//! Graph Service Supervisor - Lightweight supervisor for managing graph service actors
//!
//! This module implements a supervisor pattern that:
//! - Spawns and manages 4 child actors (GraphState, Physics, Semantic, Client)
//! - Routes messages to appropriate actors based on message type
//! - Handles actor restarts on failure with configurable policies
//! - Coordinates inter-actor communication and state synchronization
//! - Provides health monitoring and performance metrics
//!
//! ## Architecture
//!
//! ```
//! GraphServiceSupervisor
//! ├── GraphStateActor          (State management & persistence)
//! ├── PhysicsOrchestratorActor (Physics simulation & GPU compute)
//! ├── SemanticProcessorActor   (Semantic analysis & AI features)
//! └── ClientCoordinatorActor   (WebSocket & client management)
//! ```
//!
//! ## Supervision Strategies
//!
//! - **OneForOne**: Restart only the failed actor
//! - **OneForAll**: Restart all actors when one fails
//! - **RestForOne**: Restart failed actor and all actors started after it
//! - **Escalate**: Escalate failure to parent supervisor
//!
//! ## Message Routing
//!
//! Messages are routed based on their type:
//! - Graph operations → GraphStateActor
//! - Physics/GPU operations → PhysicsOrchestratorActor
//! - Semantic analysis → SemanticProcessorActor
//! - Client management → ClientCoordinatorActor

use actix::dev::{MessageResponse, OneshotSender};
use actix::prelude::*;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::actors::{
    ClientCoordinatorActor, GraphServiceActor, PhysicsOrchestratorActor, SemanticProcessorActor,
};
// Removed unused import - we don't use graph_messages types for handlers
use crate::actors::messages as msgs;
// Removed graph_messages::GetGraphData import - not used
use crate::errors::{ActorError, VisionFlowError};

/// Graph service supervision strategy for handling actor failures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphSupervisionStrategy {
    /// Restart only the failed actor
    OneForOne,
    /// Restart all actors when one fails
    OneForAll,
    /// Restart failed actor and all actors started after it
    RestForOne,
    /// Escalate failure to parent supervisor
    Escalate,
}

/// Actor health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActorHealth {
    Healthy,
    Degraded,
    Failed,
    Restarting,
    Unknown,
}

/// Actor restart policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartPolicy {
    pub max_restarts: u32,
    pub within_time_period: Duration,
    pub backoff_strategy: BackoffStrategy,
    pub escalation_threshold: u32,
}

/// Backoff strategy for actor restarts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed(Duration),
    Linear(Duration),
    Exponential { initial: Duration, max: Duration },
}

/// Actor metadata for supervision
#[derive(Debug)]
pub struct ActorInfo {
    pub name: String,
    pub actor_type: ActorType,
    pub health: ActorHealth,
    pub last_heartbeat: Option<Instant>,
    pub restart_count: u32,
    pub last_restart: Option<Instant>,
    pub message_buffer: Vec<SupervisedMessage>,
    pub stats: ActorStats,
}

/// Types of supervised actors
#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum ActorType {
    GraphState,
    PhysicsOrchestrator,
    SemanticProcessor,
    ClientCoordinator,
}

/// Actor performance statistics
#[derive(Debug, Clone)]
pub struct ActorStats {
    pub messages_processed: u64,
    pub messages_failed: u64,
    pub average_response_time: Duration,
    pub last_activity: Option<Instant>,
    pub uptime: Duration,
    pub memory_usage: Option<u64>,
}

/// Simple operation result message for supervised actors
#[derive(Message, Debug, Clone)]
#[rtype(result = "()")]
pub struct OperationResult {
    pub success: bool,
    pub error: Option<String>,
}

impl From<Result<(), VisionFlowError>> for OperationResult {
    fn from(result: Result<(), VisionFlowError>) -> Self {
        match result {
            Ok(()) => OperationResult {
                success: true,
                error: None,
            },
            Err(e) => OperationResult {
                success: false,
                error: Some(e.to_string()),
            },
        }
    }
}

/// Buffered message during actor restart
pub struct SupervisedMessage {
    pub message: Box<dyn Message<Result = ()> + Send>,
    pub sender: Option<Recipient<OperationResult>>,
    pub timestamp: Instant,
    pub retry_count: u32,
}

impl std::fmt::Debug for SupervisedMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupervisedMessage")
            .field("timestamp", &self.timestamp)
            .field("retry_count", &self.retry_count)
            .finish()
    }
}

/// Main supervisor actor managing all graph service actors
pub struct GraphServiceSupervisor {
    // Child actor addresses
    graph_state: Option<Addr<GraphServiceActor>>,
    physics: Option<Addr<PhysicsOrchestratorActor>>,
    semantic: Option<Addr<SemanticProcessorActor>>,
    client: Option<Addr<ClientCoordinatorActor>>,

    // Supervision configuration
    strategy: GraphSupervisionStrategy,
    restart_policy: RestartPolicy,

    // Actor management
    actor_info: HashMap<ActorType, ActorInfo>,

    // Health monitoring
    health_check_interval: Duration,
    last_health_check: Instant,

    // Message routing and buffering
    message_buffer_size: usize,
    total_messages_routed: u64,

    // Performance metrics
    supervision_stats: SupervisionStats,
}

/// Supervisor performance statistics
#[derive(Debug, Clone)]
pub struct SupervisionStats {
    pub actors_supervised: u32,
    pub total_restarts: u32,
    pub messages_routed: u64,
    pub messages_buffered: u64,
    pub average_routing_time: Duration,
    pub last_failure: Option<Instant>,
    pub uptime: Duration,
    pub health_checks_performed: u64,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 5,
            within_time_period: Duration::from_secs(300), // 5 minutes
            backoff_strategy: BackoffStrategy::Exponential {
                initial: Duration::from_secs(1),
                max: Duration::from_secs(60),
            },
            escalation_threshold: 3,
        }
    }
}

impl Default for ActorStats {
    fn default() -> Self {
        Self {
            messages_processed: 0,
            messages_failed: 0,
            average_response_time: Duration::from_millis(0),
            last_activity: None,
            uptime: Duration::from_secs(0),
            memory_usage: None,
        }
    }
}

impl GraphServiceSupervisor {
    /// Create new supervisor with default configuration
    pub fn new() -> Self {
        Self {
            graph_state: None,
            physics: None,
            semantic: None,
            client: None,
            strategy: GraphSupervisionStrategy::OneForOne,
            restart_policy: RestartPolicy::default(),
            actor_info: HashMap::new(),
            health_check_interval: Duration::from_secs(30),
            last_health_check: Instant::now(),
            message_buffer_size: 1000,
            total_messages_routed: 0,
            supervision_stats: SupervisionStats::default(),
        }
    }

    /// Create supervisor with custom configuration
    pub fn with_config(
        strategy: GraphSupervisionStrategy,
        restart_policy: RestartPolicy,
        health_check_interval: Duration,
    ) -> Self {
        let mut supervisor = Self::new();
        supervisor.strategy = strategy;
        supervisor.restart_policy = restart_policy;
        supervisor.health_check_interval = health_check_interval;
        supervisor
    }

    /// Create supervisor with dependencies for GraphServiceActor compatibility
    /// This is a transitional method that creates a GraphServiceActor as the managed child
    /// This allows for gradual migration to the full supervisor architecture
    pub fn with_dependencies(
        client_manager_addr: Option<Addr<crate::actors::ClientCoordinatorActor>>,
        gpu_manager_addr: Option<Addr<crate::actors::GPUManagerActor>>,
    ) -> TransitionalGraphSupervisor {
        info!("Creating TransitionalGraphSupervisor with GraphServiceActor as managed child");

        // Create the transitional supervisor that wraps GraphServiceActor
        TransitionalGraphSupervisor::new(client_manager_addr, gpu_manager_addr)
    }

    /// Initialize all child actors
    fn initialize_actors(&mut self, ctx: &mut Context<Self>) {
        info!("Initializing supervised actors");

        // Initialize actor info structures
        self.actor_info.insert(
            ActorType::GraphState,
            ActorInfo {
                name: "GraphState".to_string(),
                actor_type: ActorType::GraphState,
                health: ActorHealth::Unknown,
                last_heartbeat: None,
                restart_count: 0,
                last_restart: None,
                message_buffer: Vec::new(),
                stats: ActorStats::default(),
            },
        );

        self.actor_info.insert(
            ActorType::PhysicsOrchestrator,
            ActorInfo {
                name: "PhysicsOrchestrator".to_string(),
                actor_type: ActorType::PhysicsOrchestrator,
                health: ActorHealth::Unknown,
                last_heartbeat: None,
                restart_count: 0,
                last_restart: None,
                message_buffer: Vec::new(),
                stats: ActorStats::default(),
            },
        );

        self.actor_info.insert(
            ActorType::SemanticProcessor,
            ActorInfo {
                name: "SemanticProcessor".to_string(),
                actor_type: ActorType::SemanticProcessor,
                health: ActorHealth::Unknown,
                last_heartbeat: None,
                restart_count: 0,
                last_restart: None,
                message_buffer: Vec::new(),
                stats: ActorStats::default(),
            },
        );

        self.actor_info.insert(
            ActorType::ClientCoordinator,
            ActorInfo {
                name: "ClientCoordinator".to_string(),
                actor_type: ActorType::ClientCoordinator,
                health: ActorHealth::Unknown,
                last_heartbeat: None,
                restart_count: 0,
                last_restart: None,
                message_buffer: Vec::new(),
                stats: ActorStats::default(),
            },
        );

        // Start actors in dependency order
        // ClientCoordinator must start first as GraphState depends on it
        self.start_actor(ActorType::ClientCoordinator, ctx);
        self.start_actor(ActorType::PhysicsOrchestrator, ctx);
        self.start_actor(ActorType::SemanticProcessor, ctx);
        self.start_actor(ActorType::GraphState, ctx); // GraphState last - depends on ClientCoordinator

        // Schedule health checks
        ctx.run_interval(self.health_check_interval, |act, ctx| {
            act.perform_health_check(ctx);
        });

        self.supervision_stats.actors_supervised = 4;
        info!("All supervised actors initialized successfully");
    }

    /// Start a specific actor
    fn start_actor(&mut self, actor_type: ActorType, _ctx: &mut Context<Self>) {
        info!("Starting actor: {:?}", actor_type);

        match actor_type {
            ActorType::GraphState => {
                // Temporarily use GraphServiceActor as the graph state manager
                // This will be replaced with a dedicated GraphStateActor during gradual refactoring
                info!("Starting GraphServiceActor as temporary GraphState manager");

                // GraphServiceActor needs client_manager and optionally gpu_compute addresses
                // For now we'll create it without these dependencies and add them later
                // The supervisor will coordinate message routing
                let client_manager = self.client.as_ref().map(|addr| addr.clone());
                if let Some(client_addr) = client_manager {
                    let actor = GraphServiceActor::new(
                        client_addr,
                        None, // GPU compute will be linked later
                        None, // Settings actor will be linked later
                    )
                    .start();
                    self.graph_state = Some(actor);
                    info!("GraphServiceActor started successfully as GraphState manager");
                } else {
                    warn!("Cannot start GraphServiceActor without ClientCoordinator - will retry after client actor starts");
                }
            }
            ActorType::PhysicsOrchestrator => {
                use crate::models::simulation_params::SimulationParams;
                let params = SimulationParams::default();
                let actor = PhysicsOrchestratorActor::new(params, None, None).start();
                self.physics = Some(actor);
            }
            ActorType::SemanticProcessor => {
                let config = Some(
                    crate::actors::semantic_processor_actor::SemanticProcessorConfig::default(),
                );
                let actor = SemanticProcessorActor::new(config).start();
                self.semantic = Some(actor);
            }
            ActorType::ClientCoordinator => {
                let actor = ClientCoordinatorActor::new().start();
                self.client = Some(actor);
            }
        }

        // Update actor info
        if let Some(info) = self.actor_info.get_mut(&actor_type) {
            info.health = ActorHealth::Healthy;
            info.last_heartbeat = Some(Instant::now());
            info.stats.uptime = Duration::from_secs(0);
        }
    }

    /// Restart a failed actor
    fn restart_actor(&mut self, actor_type: ActorType, ctx: &mut Context<Self>) {
        warn!("Restarting failed actor: {:?}", actor_type);

        // Update actor info
        if let Some(info) = self.actor_info.get_mut(&actor_type) {
            info.health = ActorHealth::Restarting;
            info.restart_count += 1;
            info.last_restart = Some(Instant::now());

            // Check restart limits
            if info.restart_count > self.restart_policy.max_restarts {
                error!(
                    "Actor {:?} exceeded maximum restarts ({}), escalating",
                    actor_type, self.restart_policy.max_restarts
                );
                self.escalate_failure(actor_type, ctx);
                return;
            }
        }

        // Apply backoff strategy
        let backoff_duration = self.calculate_backoff(&actor_type);
        let actor_type_clone = actor_type.clone();
        let actor_type_clone2 = actor_type.clone();

        ctx.run_later(backoff_duration, move |act, ctx| {
            act.start_actor(actor_type_clone, ctx);
            act.replay_buffered_messages(actor_type_clone2);
        });

        self.supervision_stats.total_restarts += 1;
    }

    /// Calculate backoff duration for restart
    fn calculate_backoff(&self, actor_type: &ActorType) -> Duration {
        if let Some(info) = self.actor_info.get(actor_type) {
            match &self.restart_policy.backoff_strategy {
                BackoffStrategy::Fixed(duration) => *duration,
                BackoffStrategy::Linear(duration) => *duration * info.restart_count,
                BackoffStrategy::Exponential { initial, max } => {
                    let exponential = *initial * 2_u32.pow(info.restart_count.min(10));
                    exponential.min(*max)
                }
            }
        } else {
            Duration::from_secs(1)
        }
    }

    /// Escalate failure to parent or shutdown
    fn escalate_failure(&mut self, actor_type: ActorType, ctx: &mut Context<Self>) {
        error!("Escalating failure for actor: {:?}", actor_type);

        match self.strategy {
            GraphSupervisionStrategy::OneForAll => {
                warn!("Restarting all actors due to escalation");
                self.restart_all_actors(ctx);
            }
            GraphSupervisionStrategy::Escalate => {
                error!("Escalating to parent supervisor");
                // TODO: Send escalation message to parent
                ctx.stop();
            }
            _ => {
                error!("Actor {:?} failed beyond recovery limits", actor_type);
                if let Some(info) = self.actor_info.get_mut(&actor_type) {
                    info.health = ActorHealth::Failed;
                }
            }
        }
    }

    /// Restart all supervised actors
    fn restart_all_actors(&mut self, ctx: &mut Context<Self>) {
        info!("Restarting all supervised actors");

        // Clear current actors
        self.graph_state = None;
        self.physics = None;
        self.semantic = None;
        self.client = None;

        // Restart all
        self.start_actor(ActorType::GraphState, ctx);
        self.start_actor(ActorType::PhysicsOrchestrator, ctx);
        self.start_actor(ActorType::SemanticProcessor, ctx);
        self.start_actor(ActorType::ClientCoordinator, ctx);
    }

    /// Buffer message during actor restart
    fn buffer_message(&mut self, actor_type: ActorType, message: SupervisedMessage) {
        if let Some(info) = self.actor_info.get_mut(&actor_type) {
            if info.message_buffer.len() < self.message_buffer_size {
                info.message_buffer.push(message);
                self.supervision_stats.messages_buffered += 1;
            } else {
                warn!(
                    "Message buffer full for actor {:?}, dropping message",
                    actor_type
                );
            }
        }
    }

    /// Replay buffered messages after actor restart
    fn replay_buffered_messages(&mut self, actor_type: ActorType) {
        if let Some(info) = self.actor_info.get_mut(&actor_type) {
            let messages = std::mem::take(&mut info.message_buffer);
            info!(
                "Replaying {} buffered messages for actor {:?}",
                messages.len(),
                actor_type
            );

            // TODO: Replay messages to restarted actor
            // This would require message serialization/deserialization
        }
    }

    /// Perform health check on all actors
    fn perform_health_check(&mut self, _ctx: &mut Context<Self>) {
        debug!("Performing health check on supervised actors");

        let now = Instant::now();
        self.last_health_check = now;
        self.supervision_stats.health_checks_performed += 1;

        for (actor_type, info) in &mut self.actor_info {
            // Check heartbeat timeout
            if let Some(last_heartbeat) = info.last_heartbeat {
                if now.duration_since(last_heartbeat) > Duration::from_secs(60) {
                    warn!("Actor {:?} heartbeat timeout", actor_type);
                    info.health = ActorHealth::Degraded;
                }
            }

            // Update uptime
            if let Some(last_restart) = info.last_restart {
                info.stats.uptime = now.duration_since(last_restart);
            }
        }
    }

    /// Route message to appropriate actor
    fn route_message(
        &mut self,
        message: SupervisorMessage,
        _ctx: &mut Context<Self>,
    ) -> Result<(), VisionFlowError> {
        let start_time = Instant::now();

        let result = match message {
            SupervisorMessage::GraphOperation(_msg) => {
                if let Some(ref _addr) = self.graph_state {
                    // Forward message to graph state actor
                    // For now this is a placeholder - full implementation would deserialize and forward
                    debug!("Forwarding graph operation to GraphState actor");
                    Ok(())
                } else {
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable(
                        "GraphState".to_string(),
                    )))
                }
            }
            SupervisorMessage::PhysicsOperation(_msg) => {
                if let Some(ref _addr) = self.physics {
                    debug!("Forwarding physics operation to Physics actor");
                    Ok(())
                } else {
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable(
                        "Physics".to_string(),
                    )))
                }
            }
            SupervisorMessage::SemanticOperation(_msg) => {
                if let Some(ref _addr) = self.semantic {
                    debug!("Forwarding semantic operation to Semantic actor");
                    Ok(())
                } else {
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable(
                        "Semantic".to_string(),
                    )))
                }
            }
            SupervisorMessage::ClientOperation(_msg) => {
                if let Some(ref _addr) = self.client {
                    debug!("Forwarding client operation to Client actor");
                    Ok(())
                } else {
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable(
                        "Client".to_string(),
                    )))
                }
            }
        };

        // Update routing statistics
        let routing_time = start_time.elapsed();
        self.total_messages_routed += 1;
        self.supervision_stats.messages_routed += 1;

        // Update average routing time (simple moving average)
        let current_avg = self.supervision_stats.average_routing_time;
        let new_avg = (current_avg + routing_time) / 2;
        self.supervision_stats.average_routing_time = new_avg;

        result
    }

    /// Get supervisor status and statistics
    pub fn get_status(&self) -> SupervisorStatus {
        SupervisorStatus {
            strategy: self.strategy.clone(),
            actor_health: self
                .actor_info
                .iter()
                .map(|(actor_type, info)| (actor_type.clone(), info.health.clone()))
                .collect(),
            supervision_stats: self.supervision_stats.clone(),
            last_health_check: self.last_health_check,
            total_messages_routed: self.total_messages_routed,
        }
    }
}

impl Default for SupervisionStats {
    fn default() -> Self {
        Self {
            actors_supervised: 0,
            total_restarts: 0,
            messages_routed: 0,
            messages_buffered: 0,
            average_routing_time: Duration::from_millis(0),
            last_failure: None,
            uptime: Duration::from_secs(0),
            health_checks_performed: 0,
        }
    }
}

impl Actor for GraphServiceSupervisor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("GraphServiceSupervisor started");
        self.initialize_actors(ctx);
        self.supervision_stats.uptime = Duration::from_secs(0);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("GraphServiceSupervisor stopped");
    }
}

// Message definitions for supervisor communication

/// Main supervisor message enum for routing
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub enum SupervisorMessage {
    GraphOperation(Box<dyn Message<Result = Result<(), VisionFlowError>> + Send>),
    PhysicsOperation(Box<dyn Message<Result = Result<(), VisionFlowError>> + Send>),
    SemanticOperation(Box<dyn Message<Result = Result<(), VisionFlowError>> + Send>),
    ClientOperation(Box<dyn Message<Result = Result<(), VisionFlowError>> + Send>),
}

/// Actor heartbeat message
#[derive(Message)]
#[rtype(result = "()")]
pub struct ActorHeartbeat {
    pub actor_type: ActorType,
    pub timestamp: Instant,
    pub health: ActorHealth,
    pub stats: Option<ActorStats>,
}

/// Request supervisor status
#[derive(Message)]
#[rtype(result = "SupervisorStatus")]
pub struct GetSupervisorStatus;

/// Supervisor status response
#[derive(Debug, Clone)]
pub struct SupervisorStatus {
    pub strategy: GraphSupervisionStrategy,
    pub actor_health: HashMap<ActorType, ActorHealth>,
    pub supervision_stats: SupervisionStats,
    pub last_health_check: Instant,
    pub total_messages_routed: u64,
}

impl<A, M> MessageResponse<A, M> for SupervisorStatus
where
    A: Actor,
    M: Message<Result = SupervisorStatus>,
{
    fn handle(self, _ctx: &mut A::Context, tx: Option<OneshotSender<M::Result>>) {
        if let Some(tx) = tx {
            let _ = tx.send(self);
        }
    }
}

/// Request to restart specific actor
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct RestartActor {
    pub actor_type: ActorType,
}

/// Request to restart all actors
#[derive(Message)]
#[rtype(result = "Result<(), VisionFlowError>")]
pub struct RestartAllActors;

// Message handlers

impl Handler<SupervisorMessage> for GraphServiceSupervisor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: SupervisorMessage, ctx: &mut Self::Context) -> Self::Result {
        self.route_message(msg, ctx)
    }
}

impl Handler<ActorHeartbeat> for GraphServiceSupervisor {
    type Result = ();

    fn handle(&mut self, msg: ActorHeartbeat, _ctx: &mut Self::Context) -> Self::Result {
        if let Some(info) = self.actor_info.get_mut(&msg.actor_type) {
            info.last_heartbeat = Some(msg.timestamp);
            info.health = msg.health;

            if let Some(stats) = msg.stats {
                info.stats = stats;
            }
        }
    }
}

impl Handler<GetSupervisorStatus> for GraphServiceSupervisor {
    type Result = SupervisorStatus;

    fn handle(&mut self, _msg: GetSupervisorStatus, _ctx: &mut Self::Context) -> Self::Result {
        self.get_status()
    }
}

impl Handler<RestartActor> for GraphServiceSupervisor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: RestartActor, ctx: &mut Self::Context) -> Self::Result {
        self.restart_actor(msg.actor_type, ctx);
        Ok(())
    }
}

impl Handler<RestartAllActors> for GraphServiceSupervisor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, _msg: RestartAllActors, ctx: &mut Self::Context) -> Self::Result {
        self.restart_all_actors(ctx);
        Ok(())
    }
}

// ============================================================================
// KEY MESSAGE HANDLERS - Bridge to existing GraphServiceActor functionality
// ============================================================================

/// For now, forward key graph messages directly to maintain compatibility
/// In a full refactor, these would be decomposed and routed to specialized actors

// Removed GetGraphData handler from graph_messages - GraphServiceActor doesn't implement it

// BuildGraphFromMetadata handler is already implemented below for TransitionalGraphSupervisor (line 1078)

impl Handler<msgs::UpdateGraphData> for GraphServiceSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, _msg: msgs::UpdateGraphData, _ctx: &mut Self::Context) -> Self::Result {
        warn!("UpdateGraphData: Supervisor not fully implemented");
        let result = Err("Supervisor not yet fully implemented".to_string());
        Box::pin(actix::fut::ready(result))
    }
}

impl Handler<msgs::AddNodesFromMetadata> for GraphServiceSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(
        &mut self,
        _msg: msgs::AddNodesFromMetadata,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        warn!("AddNodesFromMetadata: Supervisor not fully implemented");
        let result = Err("Supervisor not yet fully implemented".to_string());
        Box::pin(actix::fut::ready(result))
    }
}

// Removed UpdateNodePosition handler from graph_messages - GraphServiceActor doesn't implement it

// Additional commonly used messages
impl Handler<msgs::StartSimulation> for GraphServiceSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, _msg: msgs::StartSimulation, _ctx: &mut Self::Context) -> Self::Result {
        warn!("StartSimulation: Supervisor not fully implemented");
        let result = Err("Supervisor not yet fully implemented".to_string());
        Box::pin(actix::fut::ready(result))
    }
}

impl Handler<msgs::SimulationStep> for GraphServiceSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, _msg: msgs::SimulationStep, _ctx: &mut Self::Context) -> Self::Result {
        warn!("SimulationStep: Supervisor not fully implemented");
        let result = Err("Supervisor not yet fully implemented".to_string());
        Box::pin(actix::fut::ready(result))
    }
}

impl Handler<msgs::GetBotsGraphData> for GraphServiceSupervisor {
    type Result =
        ResponseActFuture<Self, Result<std::sync::Arc<crate::models::graph::GraphData>, String>>;

    fn handle(&mut self, _msg: msgs::GetBotsGraphData, _ctx: &mut Self::Context) -> Self::Result {
        warn!("GetBotsGraphData: Supervisor not fully implemented");
        let result = Err("Supervisor not fully implemented".to_string()); // Return error for now since we need Arc<GraphData>
        Box::pin(actix::fut::ready(result))
    }
}

impl Handler<msgs::UpdateSimulationParams> for GraphServiceSupervisor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: msgs::UpdateSimulationParams,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        warn!("UpdateSimulationParams: Supervisor not fully implemented");
        // This is a fire-and-forget message, so we just log and return
        Ok(())
    }
}

impl Handler<msgs::InitializeGPUConnection> for GraphServiceSupervisor {
    type Result = ();

    fn handle(
        &mut self,
        _msg: msgs::InitializeGPUConnection,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        warn!("InitializeGPUConnection: Supervisor not fully implemented");
        // This is a fire-and-forget message, so we just log and return
    }
}

// ============================================================================
// TRANSITIONAL SUPERVISOR - Wraps GraphServiceActor for gradual migration
// ============================================================================

/// Transitional supervisor that wraps GraphServiceActor
/// This allows for gradual migration from the monolithic actor to the full supervisor pattern
/// while maintaining compatibility with existing code
pub struct TransitionalGraphSupervisor {
    /// The wrapped GraphServiceActor
    graph_service_actor: Option<Addr<GraphServiceActor>>,
    /// Client manager dependency
    client_manager_addr: Option<Addr<crate::actors::ClientCoordinatorActor>>,
    /// GPU manager dependency
    gpu_manager_addr: Option<Addr<crate::actors::GPUManagerActor>>,
    /// Supervisor statistics
    start_time: Instant,
    messages_forwarded: u64,
}

impl TransitionalGraphSupervisor {
    pub fn new(
        client_manager_addr: Option<Addr<crate::actors::ClientCoordinatorActor>>,
        gpu_manager_addr: Option<Addr<crate::actors::GPUManagerActor>>,
    ) -> Self {
        Self {
            graph_service_actor: None,
            client_manager_addr,
            gpu_manager_addr,
            start_time: Instant::now(),
            messages_forwarded: 0,
        }
    }

    /// Get or create the wrapped GraphServiceActor
    fn get_or_create_actor(
        &mut self,
        _ctx: &mut Context<Self>,
    ) -> Option<&Addr<GraphServiceActor>> {
        if self.graph_service_actor.is_none() {
            // Create the GraphServiceActor with the provided dependencies
            if let Some(ref client_manager) = self.client_manager_addr {
                info!("TransitionalGraphSupervisor: Creating managed GraphServiceActor");
                let actor = GraphServiceActor::new(
                    client_manager.clone(),
                    None, // GPU compute actor will be linked later
                    None, // Settings actor will be linked later
                )
                .start();
                self.graph_service_actor = Some(actor);
            } else {
                warn!("TransitionalGraphSupervisor: Cannot create GraphServiceActor without client manager");
                return None;
            }
        }
        self.graph_service_actor.as_ref()
    }
}

/// Handler for GetGraphServiceActor - returns the internal GraphServiceActor address
impl Handler<msgs::GetGraphServiceActor> for TransitionalGraphSupervisor {
    type Result = Option<Addr<GraphServiceActor>>;

    fn handle(
        &mut self,
        _msg: msgs::GetGraphServiceActor,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        self.get_or_create_actor(ctx).cloned()
    }
}

impl Actor for TransitionalGraphSupervisor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("TransitionalGraphSupervisor started - managing GraphServiceActor lifecycle");

        // Create the wrapped actor immediately
        self.get_or_create_actor(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        let uptime = self.start_time.elapsed();
        info!(
            "TransitionalGraphSupervisor stopped - uptime: {:?}, messages forwarded: {}",
            uptime, self.messages_forwarded
        );
    }
}

// Forward all GraphServiceActor messages to the wrapped actor
// This maintains full compatibility while adding supervision

// Removed GetGraphData handler from graph_messages - GraphServiceActor doesn't implement it

// Handler for messages::GetGraphData (different from graph_messages::GetGraphData)
impl Handler<msgs::GetGraphData> for TransitionalGraphSupervisor {
    type Result =
        ResponseActFuture<Self, Result<std::sync::Arc<crate::models::graph::GraphData>, String>>;

    fn handle(&mut self, msg: msgs::GetGraphData, ctx: &mut Self::Context) -> Self::Result {
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

// Handler for GetNodeMap - NEW for position-aware initialization
impl Handler<msgs::GetNodeMap> for TransitionalGraphSupervisor {
    type Result = ResponseActFuture<
        Self,
        Result<std::sync::Arc<std::collections::HashMap<u32, crate::models::node::Node>>, String>,
    >;

    fn handle(&mut self, msg: msgs::GetNodeMap, ctx: &mut Self::Context) -> Self::Result {
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

// Handler for GetPhysicsState - NEW for settlement state reporting
impl Handler<msgs::GetPhysicsState> for TransitionalGraphSupervisor {
    type Result = ResponseActFuture<Self, Result<crate::actors::graph_actor::PhysicsState, String>>;

    fn handle(&mut self, msg: msgs::GetPhysicsState, ctx: &mut Self::Context) -> Self::Result {
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

// Handler for BuildGraphFromMetadata from messages module
impl Handler<msgs::BuildGraphFromMetadata> for TransitionalGraphSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(
        &mut self,
        msg: msgs::BuildGraphFromMetadata,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("[TransitionalGraphSupervisor] BuildGraphFromMetadata handler invoked with {} entries", msg.metadata.len());
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            info!("[TransitionalGraphSupervisor] Forwarding BuildGraphFromMetadata to GraphServiceActor");
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    info!("[TransitionalGraphSupervisor] Sending BuildGraphFromMetadata to actor...");
                    match addr.send(msg).await {
                        Ok(result) => {
                            info!("[TransitionalGraphSupervisor] BuildGraphFromMetadata response received: {:?}", result);
                            result
                        }
                        Err(e) => {
                            error!("[TransitionalGraphSupervisor] BuildGraphFromMetadata actor communication error: {}", e);
                            Err(format!("Actor communication error: {}", e))
                        }
                    }
                }
                .into_actor(self),
            )
        } else {
            error!("[TransitionalGraphSupervisor] No GraphServiceActor available to handle BuildGraphFromMetadata");
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

impl Handler<msgs::UpdateGraphData> for TransitionalGraphSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: msgs::UpdateGraphData, ctx: &mut Self::Context) -> Self::Result {
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

impl Handler<msgs::AddNodesFromMetadata> for TransitionalGraphSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: msgs::AddNodesFromMetadata, ctx: &mut Self::Context) -> Self::Result {
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

// Removed UpdateNodePosition handler from graph_messages - GraphServiceActor doesn't implement it

impl Handler<msgs::StartSimulation> for TransitionalGraphSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: msgs::StartSimulation, ctx: &mut Self::Context) -> Self::Result {
        if let Some(actor) = self.get_or_create_actor(ctx) {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

impl Handler<msgs::SimulationStep> for TransitionalGraphSupervisor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: msgs::SimulationStep, ctx: &mut Self::Context) -> Self::Result {
        if let Some(actor) = self.get_or_create_actor(ctx) {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

impl Handler<msgs::GetBotsGraphData> for TransitionalGraphSupervisor {
    type Result =
        ResponseActFuture<Self, Result<std::sync::Arc<crate::models::graph::GraphData>, String>>;

    fn handle(&mut self, msg: msgs::GetBotsGraphData, ctx: &mut Self::Context) -> Self::Result {
        if let Some(actor) = self.get_or_create_actor(ctx) {
            let addr = actor.clone();
            self.messages_forwarded += 1;
            Box::pin(
                async move {
                    match addr.send(msg).await {
                        Ok(result) => result,
                        Err(e) => Err(format!("Actor communication error: {}", e)),
                    }
                }
                .into_actor(self),
            )
        } else {
            Box::pin(actix::fut::ready(Err(
                "Failed to create GraphServiceActor".to_string()
            )))
        }
    }
}

impl Handler<msgs::UpdateSimulationParams> for TransitionalGraphSupervisor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        msg: msgs::UpdateSimulationParams,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        if let Some(actor) = self.get_or_create_actor(ctx) {
            actor.do_send(msg);
            self.messages_forwarded += 1;
            Ok(())
        } else {
            Err("Failed to create GraphServiceActor".to_string())
        }
    }
}

impl Handler<msgs::InitializeGPUConnection> for TransitionalGraphSupervisor {
    type Result = ();

    fn handle(
        &mut self,
        msg: msgs::InitializeGPUConnection,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            actor.do_send(msg);
            self.messages_forwarded += 1;
        }
    }
}

impl Handler<msgs::UpdateBotsGraph> for TransitionalGraphSupervisor {
    type Result = ();

    fn handle(&mut self, msg: msgs::UpdateBotsGraph, ctx: &mut Self::Context) -> Self::Result {
        let actor_result = self.get_or_create_actor(ctx);
        if let Some(actor) = actor_result {
            actor.do_send(msg);
            self.messages_forwarded += 1;
        }
    }
}

// Add handlers for other messages that might be sent
// These provide basic forwarding functionality

macro_rules! forward_message {
    ($msg_type:ty, $result_type:ty) => {
        impl Handler<$msg_type> for TransitionalGraphSupervisor {
            type Result = ResponseActFuture<Self, $result_type>;

            fn handle(&mut self, msg: $msg_type, ctx: &mut Self::Context) -> Self::Result {
                let actor_result = self.get_or_create_actor(ctx);
                if let Some(actor) = actor_result {
                    let addr = actor.clone();
                    self.messages_forwarded += 1;
                    Box::pin(
                        async move {
                            match addr.send(msg).await {
                                Ok(result) => result,
                                Err(e) => Err(format!("Actor communication error: {}", e)),
                            }
                        }
                        .into_actor(self),
                    )
                } else {
                    Box::pin(actix::fut::ready(Err(
                        "Failed to create GraphServiceActor".to_string()
                    )))
                }
            }
        }
    };
}

// Forward common messages that the GraphServiceActor handles
// Note: Some types may be private to graph_actor.rs, so we use String for now
forward_message!(msgs::ComputeShortestPaths, Result<msgs::PathfindingResult, String>);
forward_message!(msgs::RequestPositionSnapshot, Result<crate::actors::messages::PositionSnapshot, String>);
forward_message!(
    msgs::GetAutoBalanceNotifications,
    Result<Vec<crate::actors::graph_actor::AutoBalanceNotification>, String>
);
forward_message!(msgs::InitialClientSync, Result<(), String>);
forward_message!(msgs::UpdateNodePosition, Result<(), String>);

#[cfg(test)]
mod tests {
    use super::*;
    use actix::System;

    #[actix_rt::test]
    async fn test_supervisor_initialization() {
        let system = System::new();

        system.block_on(async {
            let supervisor = GraphServiceSupervisor::new();
            assert_eq!(supervisor.strategy, GraphSupervisionStrategy::OneForOne);
            assert_eq!(supervisor.actor_info.len(), 0);
        });
    }

    #[actix_rt::test]
    async fn test_restart_policy_default() {
        let policy = RestartPolicy::default();
        assert_eq!(policy.max_restarts, 5);
        assert_eq!(policy.within_time_period, Duration::from_secs(300));
    }

    #[actix_rt::test]
    async fn test_backoff_calculation() {
        let supervisor = GraphServiceSupervisor::new();

        // Test with no actor info
        let backoff = supervisor.calculate_backoff(&ActorType::GraphState);
        assert_eq!(backoff, Duration::from_secs(1));
    }
}
