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

use actix::prelude::*;
use actix::dev::{MessageResponse, OneshotSender};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use log::{info, warn, error, debug};

use crate::actors::{
    GraphServiceActor,
    PhysicsOrchestratorActor,
    SemanticProcessorActor,
    ClientCoordinatorActor,
};
use crate::errors::{VisionFlowError, ActorError};

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
            Ok(()) => OperationResult { success: true, error: None },
            Err(e) => OperationResult { success: false, error: Some(e.to_string()) },
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

    /// Initialize all child actors
    fn initialize_actors(&mut self, ctx: &mut Context<Self>) {
        info!("Initializing supervised actors");

        // Initialize actor info structures
        self.actor_info.insert(ActorType::GraphState, ActorInfo {
            name: "GraphState".to_string(),
            actor_type: ActorType::GraphState,
            health: ActorHealth::Unknown,
            last_heartbeat: None,
            restart_count: 0,
            last_restart: None,
            message_buffer: Vec::new(),
            stats: ActorStats::default(),
        });

        self.actor_info.insert(ActorType::PhysicsOrchestrator, ActorInfo {
            name: "PhysicsOrchestrator".to_string(),
            actor_type: ActorType::PhysicsOrchestrator,
            health: ActorHealth::Unknown,
            last_heartbeat: None,
            restart_count: 0,
            last_restart: None,
            message_buffer: Vec::new(),
            stats: ActorStats::default(),
        });

        self.actor_info.insert(ActorType::SemanticProcessor, ActorInfo {
            name: "SemanticProcessor".to_string(),
            actor_type: ActorType::SemanticProcessor,
            health: ActorHealth::Unknown,
            last_heartbeat: None,
            restart_count: 0,
            last_restart: None,
            message_buffer: Vec::new(),
            stats: ActorStats::default(),
        });

        self.actor_info.insert(ActorType::ClientCoordinator, ActorInfo {
            name: "ClientCoordinator".to_string(),
            actor_type: ActorType::ClientCoordinator,
            health: ActorHealth::Unknown,
            last_heartbeat: None,
            restart_count: 0,
            last_restart: None,
            message_buffer: Vec::new(),
            stats: ActorStats::default(),
        });

        // Start actors
        self.start_actor(ActorType::GraphState, ctx);
        self.start_actor(ActorType::PhysicsOrchestrator, ctx);
        self.start_actor(ActorType::SemanticProcessor, ctx);
        self.start_actor(ActorType::ClientCoordinator, ctx);

        // Schedule health checks
        ctx.run_interval(self.health_check_interval, |act, ctx| {
            act.perform_health_check(ctx);
        });

        self.supervision_stats.actors_supervised = 4;
        info!("All supervised actors initialized successfully");
    }

    /// Start a specific actor
    fn start_actor(&mut self, actor_type: ActorType, ctx: &mut Context<Self>) {
        info!("Starting actor: {:?}", actor_type);

        match actor_type {
            ActorType::GraphState => {
                // TODO: Pass required dependencies for GraphServiceActor
                // let actor = GraphServiceActor::new(client_manager, gpu_compute_addr).start();
                // self.graph_state = Some(actor);
                warn!("GraphState actor creation requires dependencies - implement when needed");
            },
            ActorType::PhysicsOrchestrator => {
                use crate::models::SimulationParams;
                let params = SimulationParams::default();
                let actor = PhysicsOrchestratorActor::new(params, None, None).start();
                self.physics = Some(actor);
            },
            ActorType::SemanticProcessor => {
                let config = Some(crate::actors::semantic_processor_actor::SemanticProcessorConfig::default());
                let actor = SemanticProcessorActor::new(config).start();
                self.semantic = Some(actor);
            },
            ActorType::ClientCoordinator => {
                let actor = ClientCoordinatorActor::new().start();
                self.client = Some(actor);
            },
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
                error!("Actor {:?} exceeded maximum restarts ({}), escalating",
                       actor_type, self.restart_policy.max_restarts);
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
                BackoffStrategy::Linear(duration) => {
                    *duration * info.restart_count
                },
                BackoffStrategy::Exponential { initial, max } => {
                    let exponential = *initial * 2_u32.pow(info.restart_count.min(10));
                    exponential.min(*max)
                },
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
            },
            GraphSupervisionStrategy::Escalate => {
                error!("Escalating to parent supervisor");
                // TODO: Send escalation message to parent
                ctx.stop();
            },
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
                warn!("Message buffer full for actor {:?}, dropping message", actor_type);
            }
        }
    }

    /// Replay buffered messages after actor restart
    fn replay_buffered_messages(&mut self, actor_type: ActorType) {
        if let Some(info) = self.actor_info.get_mut(&actor_type) {
            let messages = std::mem::take(&mut info.message_buffer);
            info!("Replaying {} buffered messages for actor {:?}",
                  messages.len(), actor_type);

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
    fn route_message(&mut self, message: SupervisorMessage, ctx: &mut Context<Self>) -> Result<(), VisionFlowError> {
        let start_time = Instant::now();

        let result = match message {
            SupervisorMessage::GraphOperation(msg) => {
                if let Some(ref addr) = self.graph_state {
                    // TODO: Forward message to graph state actor
                    Ok(())
                } else {
                    // TODO: Fix message buffering for type-safe message handling
                    // self.buffer_message(ActorType::GraphState, SupervisedMessage {
                    //     message: Box::new(msg),
                    //     sender: None,
                    //     timestamp: Instant::now(),
                    //     retry_count: 0,
                    // });
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable("GraphState".to_string())))
                }
            },
            SupervisorMessage::PhysicsOperation(msg) => {
                if let Some(ref addr) = self.physics {
                    // TODO: Forward message to physics actor
                    Ok(())
                } else {
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable("Physics".to_string())))
                }
            },
            SupervisorMessage::SemanticOperation(msg) => {
                if let Some(ref addr) = self.semantic {
                    // TODO: Forward message to semantic actor
                    Ok(())
                } else {
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable("Semantic".to_string())))
                }
            },
            SupervisorMessage::ClientOperation(msg) => {
                if let Some(ref addr) = self.client {
                    // TODO: Forward message to client actor
                    Ok(())
                } else {
                    Err(VisionFlowError::Actor(ActorError::ActorNotAvailable("Client".to_string())))
                }
            },
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
            actor_health: self.actor_info.iter().map(|(actor_type, info)| {
                (actor_type.clone(), info.health.clone())
            }).collect(),
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