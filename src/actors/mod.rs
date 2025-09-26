//! Actor system modules for replacing Arc<RwLock<T>> patterns with Actix actors

pub mod graph_actor;
pub mod physics_orchestrator_actor;
pub mod settings_actor;
pub mod metadata_actor;
pub mod client_manager_actor;
pub mod client_coordinator_actor;
pub mod gpu; // Modular GPU actors system
pub mod protected_settings_actor;
pub mod claude_flow_actor;
pub mod tcp_connection_actor;
pub mod jsonrpc_client;
pub mod supervisor;
pub mod voice_commands;
// pub mod supervisor_voice; // Removed - duplicate handlers in supervisor.rs
pub mod multi_mcp_visualization_actor;
pub mod workspace_actor;
pub mod semantic_processor_actor;
pub mod graph_service_supervisor;
pub mod messages;
pub mod graph_messages;

pub use graph_actor::GraphServiceActor;
pub use physics_orchestrator_actor::PhysicsOrchestratorActor;
pub use settings_actor::SettingsActor;
pub use metadata_actor::MetadataActor;
pub use client_manager_actor::ClientManagerActor;
pub use client_coordinator_actor::{ClientCoordinatorActor, ClientCoordinatorStats, ClientManager, ClientState};
pub use gpu::GPUManagerActor; // Modular GPU manager system
pub use protected_settings_actor::ProtectedSettingsActor;
pub use claude_flow_actor::ClaudeFlowActorTcp as ClaudeFlowActor;
pub use tcp_connection_actor::TcpConnectionActor;
pub use jsonrpc_client::JsonRpcClient;
pub use supervisor::{SupervisorActor, SupervisionStrategy, SupervisedActorTrait, SupervisedActorInfo};
pub use voice_commands::{VoiceCommand, SwarmVoiceResponse, SwarmIntent, VoicePreamble};
pub use multi_mcp_visualization_actor::MultiMcpVisualizationActor;
pub use workspace_actor::WorkspaceActor;
pub use semantic_processor_actor::{SemanticProcessorActor, SemanticProcessorConfig, SemanticStats, AISemanticFeatures};
pub use graph_service_supervisor::{
    GraphServiceSupervisor, GraphSupervisionStrategy, RestartPolicy, BackoffStrategy,
    ActorHealth, ActorType, SupervisorStatus, SupervisorMessage, ActorHeartbeat,
    GetSupervisorStatus, RestartActor, RestartAllActors
};
pub use messages::*;