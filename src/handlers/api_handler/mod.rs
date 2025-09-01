pub mod files;
pub mod graph;
// REMOVED: pub mod visualisation; - legacy module moved to settings
pub mod bots;
pub mod analytics;
pub mod quest3;

// Re-export specific types and functions
// Re-export specific types and functions
pub use files::{
    fetch_and_process_files,
    get_file_content,
};

pub use graph::{
    get_graph_data,
    get_paginated_graph_data,
    refresh_graph,
    update_graph,
};

// REMOVED: pub use visualisation::get_visualisation_settings; - legacy endpoint

use actix_web::web;

// Configure all API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("") // Removed redundant /api prefix
            .configure(files::config)
            .configure(graph::config)
            // REMOVED: .configure(visualisation::config) - legacy module
            .configure(bots::config)
            .configure(analytics::config)
            .configure(quest3::config)
            .configure(crate::handlers::nostr_handler::config)
            .configure(crate::handlers::settings_handler::configure_routes)
            .configure(crate::handlers::ragflow_handler::config)
            .configure(crate::handlers::clustering_handler::config)
            .configure(crate::handlers::constraints_handler::config)
    );
}
