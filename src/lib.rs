pub mod actors;
pub mod app_state;
pub mod config;
pub mod errors;
// REMOVED: pub mod gpu; - legacy modules superseded by unified_gpu_compute
pub mod handlers;
pub mod models;
pub mod physics;
pub mod services;
pub mod types;
pub mod utils;

#[cfg(test)]
pub mod tests;


pub use app_state::AppState;
pub use actors::{GraphServiceActor, SettingsActor, MetadataActor, ClientManagerActor};
pub use models::metadata::MetadataStore;
pub use models::protected_settings::ProtectedSettings;
pub use models::simulation_params::SimulationParams;
pub use models::user_settings::UserSettings;
