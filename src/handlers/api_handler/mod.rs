pub mod analytics;
pub mod bots;
pub mod files;
pub mod graph;
#[cfg(feature = "ontology")]
pub mod ontology;
pub mod quest3;
// pub mod sessions; // REMOVED: Deprecated sessions API (returns 410 Gone)
pub mod visualisation;

// Re-export specific types and functions
// Re-export specific types and functions
pub use files::{fetch_and_process_files, get_file_content};

pub use graph::{get_graph_data, get_paginated_graph_data, refresh_graph, update_graph};

pub use visualisation::get_visualisation_settings;

use crate::handlers::utils::execute_in_thread;
use actix_web::{web, HttpResponse, Responder};
use log::{error, info};
use serde_json::json;

/// GET /api/health - Simple health check endpoint for UI
async fn health_check() -> impl Responder {
    info!("Health check requested");
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}

/// GET /api/config - Return app configuration for UI (CQRS-based)
async fn get_app_config(state: web::Data<crate::AppState>) -> impl Responder {
    info!("App config requested via CQRS");

    // Use CQRS to load settings from database
    use crate::application::settings::{LoadAllSettings, LoadAllSettingsHandler};
    use hexser::QueryHandler;

    let handler = LoadAllSettingsHandler::new(state.settings_repository.clone());

    // Execute query in a separate OS thread to escape Tokio runtime
    let result = execute_in_thread(move || handler.handle(LoadAllSettings)).await;

    match result {
        Ok(Ok(Some(settings))) => HttpResponse::Ok().json(json!({
            "version": env!("CARGO_PKG_VERSION"),
            "features": {
                "ragflow": settings.ragflow.is_some(),
                "perplexity": settings.perplexity.is_some(),
                "openai": settings.openai.is_some(),
                "kokoro": settings.kokoro.is_some(),
                "whisper": settings.whisper.is_some(),
            },
            "websocket": {
                "minUpdateRate": settings.system.websocket.min_update_rate,
                "maxUpdateRate": settings.system.websocket.max_update_rate,
                "motionThreshold": settings.system.websocket.motion_threshold,
                "motionDamping": settings.system.websocket.motion_damping,
            },
            "rendering": {
                "ambientLightIntensity": settings.visualisation.rendering.ambient_light_intensity,
                "enableAmbientOcclusion": settings.visualisation.rendering.enable_ambient_occlusion,
                "backgroundColor": settings.visualisation.rendering.background_color,
            },
            "xr": {
                "enabled": settings.xr.enabled.unwrap_or(false),
                "roomScale": settings.xr.room_scale,
                "spaceType": settings.xr.space_type,
            }
        })),
        Ok(Ok(None)) => {
            log::warn!("No settings found, using defaults");
            use crate::config::AppFullSettings;
            let settings = AppFullSettings::default();
            HttpResponse::Ok().json(json!({
                "version": env!("CARGO_PKG_VERSION"),
                "features": {
                    "ragflow": settings.ragflow.is_some(),
                    "perplexity": settings.perplexity.is_some(),
                    "openai": settings.openai.is_some(),
                    "kokoro": settings.kokoro.is_some(),
                    "whisper": settings.whisper.is_some(),
                },
                "websocket": {
                    "minUpdateRate": settings.system.websocket.min_update_rate,
                    "maxUpdateRate": settings.system.websocket.max_update_rate,
                    "motionThreshold": settings.system.websocket.motion_threshold,
                    "motionDamping": settings.system.websocket.motion_damping,
                },
                "rendering": {
                    "ambientLightIntensity": settings.visualisation.rendering.ambient_light_intensity,
                    "enableAmbientOcclusion": settings.visualisation.rendering.enable_ambient_occlusion,
                    "backgroundColor": settings.visualisation.rendering.background_color,
                },
                "xr": {
                    "enabled": settings.xr.enabled.unwrap_or(false),
                    "roomScale": settings.xr.room_scale,
                    "spaceType": settings.xr.space_type,
                }
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to load settings via CQRS: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Failed to retrieve configuration"
            }))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            }))
        }
    }
}

// Configure all API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("") // Removed redundant /api prefix
            // Core API endpoints
            .route("/health", web::get().to(health_check))
            .route("/config", web::get().to(get_app_config))
            .route(
                "/settings-test",
                web::get().to(|| async { HttpResponse::Ok().json(json!({"test": "works"})) }),
            )
            // Existing module routes
            .configure(files::config)
            .configure(graph::config)
            .configure(crate::handlers::graph_state_handler::config) // CQRS-refactored graph state
            .configure(crate::handlers::ontology_handler::config) // CQRS-based ontology handler
            .configure(visualisation::config)
            .configure(bots::config)
            // .configure(sessions::config)  // REMOVED: Deprecated sessions API (returns 410 Gone)
            .configure(analytics::config)
            .configure(quest3::config)
            .configure(crate::handlers::nostr_handler::config)
            .configure(crate::handlers::settings_handler::config)
            // DISABLED: Causes route conflicts/timeouts with settings_handler
            // .configure(crate::handlers::settings_paths::configure_settings_paths)
            .configure(crate::handlers::ragflow_handler::config)
            .configure(crate::handlers::clustering_handler::config)
            .configure(crate::handlers::constraints_handler::config),
    );

    // Legacy ontology module (will be deprecated in favor of ontology_handler)
    #[cfg(feature = "ontology")]
    cfg.service(web::scope("").configure(ontology::config));
}
