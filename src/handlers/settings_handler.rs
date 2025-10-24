// NEW: CQRS-Based Settings Handler
// Direct connection: UI → Handler → CQRS Application Layer → Database
//
// This replaces direct service calls with CQRS directives/queries
// All settings operations go through application layer handlers

use crate::app_state::AppState;
use crate::config::AppFullSettings;
use actix_web::{web, Error, HttpResponse};
use log::{error, info, warn};
use crate::handlers::utils::execute_in_thread;
use serde_json::{json, Value};

// Import CQRS handlers
use crate::application::settings::{
    ClearSettingsCache,
    ClearSettingsCacheHandler,
    GetPhysicsSettings,
    GetPhysicsSettingsHandler,
    GetSetting,
    GetSettingHandler,
    GetSettingsBatch,
    GetSettingsBatchHandler,
    // Queries
    LoadAllSettings,
    LoadAllSettingsHandler,
    // Directives
    SaveAllSettings,
    SaveAllSettingsHandler,
    UpdatePhysicsSettings,
    UpdatePhysicsSettingsHandler,
    UpdateSetting,
    UpdateSettingHandler,
};
use crate::ports::settings_repository::SettingValue;
use hexser::{DirectiveHandler, QueryHandler};

/// Configure settings routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/settings")
            .route("", web::get().to(get_settings))
            .route("", web::post().to(update_settings))
            .route("/health", web::get().to(settings_health))
            .route("/reset", web::post().to(reset_settings))
            .route("/export", web::get().to(export_settings))
            .route("/import", web::post().to(import_settings))
            .route("/cache/clear", web::post().to(clear_cache))
            .route("/path/{path:.*}", web::get().to(get_setting_by_path))
            .route("/path/{path:.*}", web::put().to(update_setting_by_path))
            .route("/batch", web::post().to(get_settings_batch))
            .route("/physics/{graph_name}", web::get().to(get_physics_settings))
            .route(
                "/physics/{graph_name}",
                web::put().to(update_physics_settings),
            ),
    );
}

/// Get all settings from database using CQRS query
pub async fn get_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] GET /settings - Loading via CQRS");

    // Create query handler using repository from AppState
    let handler = LoadAllSettingsHandler::new(state.settings_repository.clone());

    // Execute query
    let result = execute_in_thread(move || handler.handle(LoadAllSettings)).await;


    match result {
        Ok(Ok(Some(settings))) => {
            info!("[Settings Handler] Settings loaded successfully via CQRS");
            Ok(HttpResponse::Ok().json(settings))
        }
        Ok(Ok(None)) => {
            warn!("[Settings Handler] No settings found, using defaults");
            let default_settings = AppFullSettings::default();

            // Save defaults using directive
            let save_handler = SaveAllSettingsHandler::new(state.settings_repository.clone());
            let settings_clone = default_settings.clone();
            let result = execute_in_thread(move || save_handler.handle(SaveAllSettings {
                settings: settings_clone,
            })).await;


            match result {
                Ok(Ok(())) => { /* Success */ }
                Ok(Err(e)) => error!("[Settings Handler] Failed to save defaults: {}", e),
                Err(e) => error!("[Settings Handler] Thread error saving defaults: {}", e),
            }

            Ok(HttpResponse::Ok().json(default_settings))
        }
        Ok(Err(e)) => {
            error!("[Settings Handler] CQRS query error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to load settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Update settings in database using CQRS directive
pub async fn update_settings(
    state: web::Data<AppState>,
    payload: web::Json<AppFullSettings>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings - Updating via CQRS");

    let settings = payload.into_inner();

    // CRITICAL: Validate that graph settings are separate (logseq vs visionflow)
    if settings.visualisation.graphs.logseq.nodes == settings.visualisation.graphs.visionflow.nodes
    {
        warn!("[Settings Handler] WARNING: Logseq and Visionflow graphs have identical node settings!");
    }

    // Create directive handler
    let handler = SaveAllSettingsHandler::new(state.settings_repository.clone());

    // Execute directive
    let result = execute_in_thread(move || handler.handle(SaveAllSettings { settings })).await;


    match result {
        Ok(Ok(())) => {
            info!("[Settings Handler] Settings saved successfully via CQRS");
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Settings saved to database"
            })))
        }
        Ok(Err(e)) => {
            error!("[Settings Handler] CQRS directive error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to save settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Get a single setting by path using CQRS query
pub async fn get_setting_by_path(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let setting_path = path.into_inner();
    info!(
        "[Settings Handler] GET /settings/path/{} - CQRS query",
        setting_path
    );

    // Create query handler
    let handler = GetSettingHandler::new(state.settings_repository.clone());
    let path_clone = setting_path.clone();

    // Execute query
    let result = execute_in_thread(move || handler.handle(GetSetting {
        key: path_clone,
    })).await;


    match result {
        Ok(Ok(Some(value))) => {
            info!(
                "[Settings Handler] Setting '{}' found via CQRS",
                setting_path
            );
            // Convert SettingValue to JSON
            let json_value = setting_value_to_json(&value);
            Ok(HttpResponse::Ok().json(json!({
                "path": setting_path,
                "value": json_value
            })))
        }
        Ok(Ok(None)) => {
            info!("[Settings Handler] Setting '{}' not found", setting_path);
            Ok(HttpResponse::NotFound().json(json!({
                "error": "not_found",
                "path": setting_path
            })))
        }
        Ok(Err(e)) => {
            error!(
                "[Settings Handler] CQRS query error for '{}': {}",
                setting_path, e
            );
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to get setting: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Update a single setting by path using CQRS directive
pub async fn update_setting_by_path(
    state: web::Data<AppState>,
    path: web::Path<String>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let setting_path = path.into_inner();
    let value = payload.into_inner();

    info!(
        "[Settings Handler] PUT /settings/path/{} - CQRS directive",
        setting_path
    );

    // Convert JSON value to SettingValue
    let setting_value = json_to_setting_value(&value);

    // Create directive handler
    let handler = UpdateSettingHandler::new(state.settings_repository.clone());
    let path_clone = setting_path.clone();

    // Execute directive
    let result = execute_in_thread(move || handler.handle(UpdateSetting {
        key: path_clone,
        value: setting_value,
        description: None,
    })).await;


    match result {
        Ok(Ok(())) => {
            info!(
                "[Settings Handler] Setting '{}' updated via CQRS",
                setting_path
            );
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "path": setting_path,
                "value": value
            })))
        }
        Ok(Err(e)) => {
            error!(
                "[Settings Handler] CQRS directive error for '{}': {}",
                setting_path, e
            );
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to update setting: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Get batch of settings by paths using CQRS query
pub async fn get_settings_batch(
    state: web::Data<AppState>,
    payload: web::Json<Vec<String>>,
) -> Result<HttpResponse, Error> {
    let paths = payload.into_inner();
    info!(
        "[Settings Handler] POST /settings/batch - Getting {} settings via CQRS",
        paths.len()
    );

    // Create query handler
    let handler = GetSettingsBatchHandler::new(state.settings_repository.clone());
    let paths_len = paths.len();

    // Execute query
    let result = execute_in_thread(move || handler.handle(GetSettingsBatch {
        keys: paths.clone(),
    })).await;


    match result {
        Ok(Ok(results)) => {
            info!(
                "[Settings Handler] Batch get successful: {}/{} settings found",
                results.len(),
                paths_len
            );
            // Convert SettingValue map to JSON map
            let json_results: std::collections::HashMap<String, Value> = results
                .into_iter()
                .map(|(k, v)| (k, setting_value_to_json(&v)))
                .collect();
            Ok(HttpResponse::Ok().json(json_results))
        }
        Ok(Err(e)) => {
            error!("[Settings Handler] CQRS batch query error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to get settings batch: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Reset settings to defaults using CQRS directive
pub async fn reset_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings/reset - CQRS directive");

    let default_settings = AppFullSettings::default();

    // Create directive handler
    let handler = SaveAllSettingsHandler::new(state.settings_repository.clone());

    // Execute directive in a separate OS thread to escape Tokio runtime
    let result = execute_in_thread(move || handler.handle(SaveAllSettings {
        settings: default_settings,
    })).await;

    match result {
        Ok(Ok(())) => {
            info!("[Settings Handler] Settings reset via CQRS");
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Settings reset to defaults"
            })))
        }
        Ok(Err(e)) => {
            error!("[Settings Handler] CQRS reset directive error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to reset settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Health check for settings service
pub async fn settings_health(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    // Try to load settings using CQRS query
    let handler = LoadAllSettingsHandler::new(state.settings_repository.clone());

    let result = execute_in_thread(move || handler.handle(LoadAllSettings)).await;

    let healthy = match result {
        Ok(Ok(_)) => true,
        _ => false,
    };

    // Get cache stats
    let cache_stats = state.settings_service.get_cache_stats().await;

    if healthy {
        Ok(HttpResponse::Ok().json(json!({
            "status": "healthy",
            "database": "connected",
            "cache_entries": cache_stats.entries,
            "cache_age_seconds": cache_stats.last_updated
        })))
    } else {
        Ok(HttpResponse::ServiceUnavailable().json(json!({
            "status": "unhealthy",
            "database": "error"
        })))
    }
}

/// Get physics settings for a specific graph using CQRS query
pub async fn get_physics_settings(
    state: web::Data<AppState>,
    graph_name: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let graph = graph_name.into_inner();
    info!(
        "[Settings Handler] GET /settings/physics/{} - CQRS query",
        graph
    );

    // CRITICAL: Ensure graph separation (logseq vs visionflow)
    if graph != "logseq" && graph != "visionflow" && graph != "default" {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "invalid_graph",
            "message": format!("Invalid graph name: {}. Must be 'logseq', 'visionflow', or 'default'", graph)
        })));
    }

    // Create query handler
    let handler = GetPhysicsSettingsHandler::new(state.settings_repository.clone());
    let graph_clone = graph.clone();

    // Execute query
    let result = execute_in_thread(move || handler.handle(GetPhysicsSettings {
        profile_name: graph_clone,
    })).await;


    match result {
        Ok(Ok(settings)) => {
            info!(
                "[Settings Handler] Physics settings for '{}' loaded via CQRS",
                graph
            );
            Ok(HttpResponse::Ok().json(settings))
        }
        Ok(Err(e)) => {
            error!(
                "[Settings Handler] CQRS physics query error for '{}': {}",
                graph, e
            );
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to get physics settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Update physics settings for a specific graph using CQRS directive
pub async fn update_physics_settings(
    state: web::Data<AppState>,
    graph_name: web::Path<String>,
    payload: web::Json<crate::config::PhysicsSettings>,
) -> Result<HttpResponse, Error> {
    let graph = graph_name.into_inner();
    let physics_settings = payload.into_inner();

    info!(
        "[Settings Handler] PUT /settings/physics/{} - CQRS directive",
        graph
    );

    // CRITICAL: Validate graph name to prevent conflation
    if graph != "logseq" && graph != "visionflow" && graph != "default" {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "invalid_graph",
            "message": format!("Invalid graph name: {}. Must be 'logseq', 'visionflow', or 'default'", graph)
        })));
    }

    // Create directive handler
    let handler = UpdatePhysicsSettingsHandler::new(state.settings_repository.clone());
    let graph_clone = graph.clone();

    // Execute directive
    let result = execute_in_thread(move || handler.handle(UpdatePhysicsSettings {
        profile_name: graph_clone,
        settings: physics_settings,
    })).await;


    match result {
        Ok(Ok(())) => {
            info!(
                "[Settings Handler] Physics settings for '{}' saved via CQRS",
                graph
            );
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "graph": graph,
                "message": "Physics settings saved"
            })))
        }
        Ok(Err(e)) => {
            error!(
                "[Settings Handler] CQRS physics directive error for '{}': {}",
                graph, e
            );
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to save physics settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Export settings as JSON using CQRS query
pub async fn export_settings(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] GET /settings/export - CQRS query");

    // Create query handler
    let handler = LoadAllSettingsHandler::new(state.settings_repository.clone());

    // Execute query
    let result = execute_in_thread(move || handler.handle(LoadAllSettings)).await;


    match result {
        Ok(Ok(Some(settings))) => match serde_json::to_string_pretty(&settings) {
            Ok(json_string) => Ok(HttpResponse::Ok()
                .content_type("application/json")
                .insert_header((
                    "Content-Disposition",
                    "attachment; filename=\"settings.json\"",
                ))
                .body(json_string)),
            Err(e) => {
                error!("[Settings Handler] Failed to serialize settings: {}", e);
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "serialization_error"
                })))
            }
        },
        Ok(Ok(None)) => Ok(HttpResponse::NotFound().json(json!({
            "error": "no_settings"
        }))),
        Ok(Err(e)) => {
            error!("[Settings Handler] CQRS export query error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to export settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Import settings from JSON using CQRS directive
pub async fn import_settings(
    state: web::Data<AppState>,
    payload: web::Json<AppFullSettings>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings/import - CQRS directive");

    let settings = payload.into_inner();

    // Create directive handler
    let handler = SaveAllSettingsHandler::new(state.settings_repository.clone());

    // Execute directive
    let result = execute_in_thread(move || handler.handle(SaveAllSettings { settings })).await;


    match result {
        Ok(Ok(())) => {
            info!("[Settings Handler] Settings imported successfully via CQRS");
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Settings imported"
            })))
        }
        Ok(Err(e)) => {
            error!("[Settings Handler] CQRS import directive error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to import settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

/// Clear settings cache using CQRS directive
pub async fn clear_cache(state: web::Data<AppState>) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings/cache/clear - CQRS directive");

    // Create directive handler
    let handler = ClearSettingsCacheHandler::new(state.settings_repository.clone());

    // Execute directive
    let result = execute_in_thread(move || handler.handle(ClearSettingsCache)).await;


    match result {
        Ok(Ok(())) => Ok(HttpResponse::Ok().json(json!({
            "success": true,
            "message": "Cache cleared"
        }))),
        Ok(Err(e)) => {
            error!("[Settings Handler] CQRS cache clear error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "cache_error",
                "message": format!("Failed to clear cache: {}", e)
            })))
        }
        Err(e) => {
            error!("Thread execution error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error"
            })))
        }
    }
}

// Helper functions for SettingValue conversion

fn setting_value_to_json(value: &SettingValue) -> Value {
    match value {
        SettingValue::String(s) => json!(s),
        SettingValue::Integer(i) => json!(i),
        SettingValue::Float(f) => json!(f),
        SettingValue::Boolean(b) => json!(b),
        SettingValue::Json(j) => j.clone(),
    }
}

fn json_to_setting_value(value: &Value) -> SettingValue {
    match value {
        Value::String(s) => SettingValue::String(s.clone()),
        Value::Number(n) if n.is_f64() => SettingValue::Float(n.as_f64().unwrap()),
        Value::Number(n) if n.is_i64() => SettingValue::Integer(n.as_i64().unwrap()),
        Value::Bool(b) => SettingValue::Boolean(*b),
        _ => SettingValue::Json(value.clone()),
    }
}
