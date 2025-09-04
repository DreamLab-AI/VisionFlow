//! Actor system modules for replacing Arc<RwLock<T>> patterns with Actix actors

pub mod graph_actor;
pub mod settings_actor;
pub mod metadata_actor;
pub mod client_manager_actor;
pub mod gpu_compute_actor;
// gpu_compute_actor_handlers consolidated into gpu_compute_actor.rs
pub mod protected_settings_actor;
pub mod claude_flow_actor_tcp;
pub mod supervisor;
// pub mod multi_mcp_visualization_actor; // Removed - file not implemented
pub mod messages;

pub use graph_actor::GraphServiceActor;
pub use settings_actor::SettingsActor;
pub use metadata_actor::MetadataActor;
pub use client_manager_actor::ClientManagerActor;
pub use gpu_compute_actor::GPUComputeActor;
pub use protected_settings_actor::ProtectedSettingsActor;
// Export the TCP actor as the ONLY ClaudeFlowActor
pub use claude_flow_actor_tcp::ClaudeFlowActorTcp as ClaudeFlowActor;
pub use supervisor::{SupervisorActor, SupervisionStrategy, SupervisedActorTrait, SupervisedActorInfo};
// pub use multi_mcp_visualization_actor::MultiMcpVisualizationActor; // Removed - not implemented
pub use messages::*;