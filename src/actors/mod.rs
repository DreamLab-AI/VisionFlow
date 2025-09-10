//! Actor system modules for replacing Arc<RwLock<T>> patterns with Actix actors

pub mod graph_actor;
pub mod settings_actor;
pub mod metadata_actor;
pub mod client_manager_actor;
pub mod gpu_compute_actor;
pub mod gpu; // New modular GPU actors
// gpu_compute_actor_handlers consolidated into gpu_compute_actor.rs
pub mod protected_settings_actor;
pub mod claude_flow_actor_tcp_refactored;
pub mod tcp_connection_actor;
pub mod jsonrpc_client;
pub mod supervisor;
pub mod voice_commands;
pub mod supervisor_voice;
// pub mod multi_mcp_visualization_actor; // Removed - file not implemented
pub mod messages;

pub use graph_actor::GraphServiceActor;
pub use settings_actor::SettingsActor;
pub use metadata_actor::MetadataActor;
pub use client_manager_actor::ClientManagerActor;
pub use gpu_compute_actor::GPUComputeActor;
pub use gpu::GPUManagerActor; // New modular GPU manager
pub use protected_settings_actor::ProtectedSettingsActor;
// Export the refactored TCP actor as the only ClaudeFlowActor
pub use claude_flow_actor_tcp_refactored::ClaudeFlowActorTcp as ClaudeFlowActor;
pub use tcp_connection_actor::TcpConnectionActor;
pub use jsonrpc_client::JsonRpcClient;
pub use supervisor::{SupervisorActor, SupervisionStrategy, SupervisedActorTrait, SupervisedActorInfo};
pub use voice_commands::{VoiceCommand, SwarmVoiceResponse, SwarmIntent, VoicePreamble};
// pub use multi_mcp_visualization_actor::MultiMcpVisualizationActor; // Removed - not implemented
pub use messages::*;