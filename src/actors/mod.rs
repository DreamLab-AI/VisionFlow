//! Actor system modules for replacing Arc<RwLock<T>> patterns with Actix actors

pub mod agent_monitor_actor;
pub mod client_coordinator_actor;
pub mod gpu; 
pub mod graph_actor;
pub mod metadata_actor;
pub mod optimized_settings_actor;
pub mod physics_orchestrator_actor;
pub mod protected_settings_actor;
pub mod supervisor;
pub mod voice_commands;
// pub mod supervisor_voice; 
pub mod graph_messages;
pub mod graph_service_supervisor;
pub mod messages;
pub mod multi_mcp_visualization_actor;
pub mod ontology_actor;
pub mod semantic_processor_actor;
pub mod task_orchestrator_actor;
pub mod workspace_actor;

pub use agent_monitor_actor::AgentMonitorActor;
pub use client_coordinator_actor::{
    ClientCoordinatorActor, ClientCoordinatorStats, ClientManager, ClientState,
};
pub use gpu::GPUManagerActor; 
pub use graph_actor::GraphServiceActor;
pub use graph_service_supervisor::{
    ActorHealth, ActorHeartbeat, ActorType, BackoffStrategy, GetSupervisorStatus,
    GraphServiceSupervisor, GraphSupervisionStrategy, RestartActor, RestartAllActors,
    RestartPolicy, SupervisorMessage, SupervisorStatus,
};
pub use messages::*;
pub use metadata_actor::MetadataActor;
pub use multi_mcp_visualization_actor::MultiMcpVisualizationActor;
pub use ontology_actor::{
    ActorStatistics as OntologyActorStatistics, JobPriority, JobStatus, OntologyActor,
    OntologyActorConfig, ValidationJob,
};
pub use optimized_settings_actor::OptimizedSettingsActor;
pub use physics_orchestrator_actor::PhysicsOrchestratorActor;
pub use protected_settings_actor::ProtectedSettingsActor;
pub use semantic_processor_actor::{
    AISemanticFeatures, SemanticProcessorActor, SemanticProcessorConfig, SemanticStats,
};
pub use supervisor::{
    SupervisedActorInfo, SupervisedActorTrait, SupervisionStrategy, SupervisorActor,
};
pub use task_orchestrator_actor::{
    CreateTask, GetSystemStatus, GetTaskStatus, ListActiveTasks, StopTask, SystemStatusInfo,
    TaskOrchestratorActor, TaskState,
};
pub use voice_commands::{SwarmIntent, SwarmVoiceResponse, VoiceCommand, VoicePreamble};
pub use workspace_actor::WorkspaceActor;

// Phase 5: Actor lifecycle management and coordination
pub mod backward_compat;
pub mod event_coordination;
pub mod lifecycle;

pub use backward_compat::{LegacyActorCompat, MigrationHelper};
pub use event_coordination::{initialize_event_coordinator, EventCoordinator};
pub use lifecycle::{
    initialize_actor_system, shutdown_actor_system, ActorLifecycleManager,
    SupervisionStrategy as Phase5SupervisionStrategy,
};
