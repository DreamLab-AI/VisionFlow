pub mod actors;
pub mod adapters;
pub mod app_state;
pub mod application; 
pub mod client;
pub mod config;
pub mod cqrs; 
pub mod errors;
pub mod events; 
pub mod gpu;
pub mod handlers;
pub mod inference; 
pub mod middleware;
pub mod migrations; 
pub mod models;
pub mod ontology;
pub mod reasoning;
pub mod physics;
pub mod ports;
pub mod repositories; 
pub mod services;
pub mod settings; 
pub mod telemetry;
pub mod types;
pub mod utils;

// #[cfg(test)]
// pub mod test_settings_fix;

pub use actors::{
    ClientCoordinatorActor, GraphServiceActor, MetadataActor, OptimizedSettingsActor,
};
pub use app_state::AppState;
pub use models::metadata::MetadataStore;
pub use models::protected_settings::ProtectedSettings;
pub use models::simulation_params::SimulationParams;
// pub use models::ui_settings::UISettings; 
pub use models::user_settings::UserSettings;
