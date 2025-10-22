pub mod analytics;
pub mod bots;
pub mod files;
pub mod graph;
#[cfg(feature = "ontology")]
pub mod ontology;
pub mod quest3;
pub mod sessions;
pub mod visualisation;

// Re-export specific types and functions
// Re-export specific types and functions
pub use files::{fetch_and_process_files, get_file_content};

pub use graph::{get_graph_data, get_paginated_graph_data, refresh_graph, update_graph};

pub use visualisation::get_visualisation_settings;

use actix_web::web;

// Configure all API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("") // Removed redundant /api prefix
            .configure(files::config)
            .configure(graph::config)
            .configure(crate::handlers::graph_state_handler::config) // CQRS-refactored graph state
            .configure(crate::handlers::ontology_handler::config) // CQRS-based ontology handler
            .configure(visualisation::config)
            .configure(bots::config)
            .configure(sessions::config)
            .configure(analytics::config)
            .configure(quest3::config)
            .configure(crate::handlers::nostr_handler::config)
            .configure(crate::handlers::settings_handler::config) // CQRS-refactored settings
            .configure(crate::handlers::settings_paths::configure_settings_paths)
            .configure(crate::handlers::ragflow_handler::config)
            .configure(crate::handlers::clustering_handler::config)
            .configure(crate::handlers::constraints_handler::config),
    );

    // Legacy ontology module (will be deprecated in favor of ontology_handler)
    #[cfg(feature = "ontology")]
    cfg.service(web::scope("").configure(ontology::config));
}
