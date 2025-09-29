pub mod actors;
pub mod app_state;
pub mod config;
pub mod errors;
pub mod gpu;
pub mod handlers;
pub mod models;
pub mod physics;
pub mod services;
pub mod telemetry;
pub mod types;
pub mod utils;

// Neural integration modules
pub mod neural_swarm_controller;
pub mod neural_actor_system;
pub mod neural_gpu_service;
pub mod neural_websocket_handler;
pub mod neural_docker_orchestrator;
pub mod neural_consensus;
pub mod neural_memory;

// #[cfg(test)]
// pub mod test_settings_fix;

pub use app_state::AppState;
pub use actors::{GraphServiceActor, OptimizedSettingsActor, MetadataActor, ClientCoordinatorActor};
pub use models::metadata::MetadataStore;
pub use models::protected_settings::ProtectedSettings;
pub use models::simulation_params::SimulationParams;
// pub use models::ui_settings::UISettings; // Removed - consolidated into AppFullSettings"
pub use models::user_settings::UserSettings;

// Neural system exports
pub use neural_swarm_controller::{NeuralSwarmController, SwarmTopology, NeuralSwarmAgent, AgentRole, SwarmStatus};
pub use neural_actor_system::{NeuralActorSystem, CognitivePattern, NeuralActor, TaskResult};
pub use neural_gpu_service::{NeuralGpuService, NeuralNetworkConfig, NeuralTask, NeuralResult};
pub use neural_websocket_handler::{NeuralWebSocketSession, CognitiveProfile, neural_websocket_handler};
pub use neural_docker_orchestrator::{NeuralDockerOrchestrator, NeuralContainer, NeuralCluster};
pub use neural_consensus::{NeuralConsensus, ConsensusProposal, ConsensusResult, ConsensusVote};
pub use neural_memory::{NeuralMemory, MemoryType, ExperienceData, MemoryQuery, MemoryResult};
