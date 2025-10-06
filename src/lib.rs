pub mod actors;
pub mod app_state;
pub mod client;
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

// #[cfg(test)]
// pub mod test_settings_fix;

pub use app_state::AppState;
pub use actors::{GraphServiceActor, OptimizedSettingsActor, MetadataActor, ClientCoordinatorActor};
pub use models::metadata::MetadataStore;
pub use models::protected_settings::ProtectedSettings;
pub use models::simulation_params::SimulationParams;
// pub use models::ui_settings::UISettings; // Removed - consolidated into AppFullSettings"
pub use models::user_settings::UserSettings;
