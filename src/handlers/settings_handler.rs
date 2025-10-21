// NEW: Database-Backed Settings Handler
// Direct connection: UI → Handler → SettingsService → Database
//
// This replaces the old actor-based approach with direct database access
// All settings operations go through SettingsService which handles caching and persistence

use actix_web::{web, Error, HttpResponse, HttpRequest};
use crate::app_state::AppState;
use crate::config::AppFullSettings;
use serde_json::{json, Value};
use log::{info, warn, error};

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
            .route("/physics/{graph_name}", web::put().to(update_physics_settings))
    );
}

/// Get all settings from database
pub async fn get_settings(
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] GET /settings - Loading from database");

    match state.settings_service.load_all_settings() {
        Ok(Some(settings)) => {
            info!("[Settings Handler] Settings loaded successfully from database");
            Ok(HttpResponse::Ok().json(settings))
        }
        Ok(None) => {
            warn!("[Settings Handler] No settings found in database, using defaults");
            let default_settings = AppFullSettings::default();

            // Save defaults to database for next time
            if let Err(e) = state.settings_service.save_all_settings(&default_settings) {
                error!("[Settings Handler] Failed to save default settings: {}", e);
            }

            Ok(HttpResponse::Ok().json(default_settings))
        }
        Err(e) => {
            error!("[Settings Handler] Database error loading settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to load settings: {}", e)
            })))
        }
    }
}

/// Update settings in database
pub async fn update_settings(
    state: web::Data<AppState>,
    payload: web::Json<AppFullSettings>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings - Updating database");

    let settings = payload.into_inner();

    // CRITICAL: Validate that graph settings are separate (logseq vs visionflow)
    // This prevents the conflation bug mentioned by user
    if settings.visualisation.graphs.logseq.nodes == settings.visualisation.graphs.visionflow.nodes {
        warn!("[Settings Handler] WARNING: Logseq and Visionflow graphs have identical node settings - possible conflation!");
    }

    match state.settings_service.save_all_settings(&settings) {
        Ok(()) => {
            info!("[Settings Handler] Settings saved successfully to database");
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Settings saved to database"
            })))
        }
        Err(e) => {
            error!("[Settings Handler] Database error saving settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to save settings: {}", e)
            })))
        }
    }
}

/// Get a single setting by path (supports camelCase and snake_case)
pub async fn get_setting_by_path(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let setting_path = path.into_inner();
    info!("[Settings Handler] GET /settings/path/{} - Direct database lookup", setting_path);

    match state.settings_service.get_setting(&setting_path).await {
        Ok(Some(value)) => {
            info!("[Settings Handler] Setting '{}' found in database", setting_path);
            // Convert SettingValue to JSON
            let json_value = match value {
                crate::services::database_service::SettingValue::String(s) => json!(s),
                crate::services::database_service::SettingValue::Integer(i) => json!(i),
                crate::services::database_service::SettingValue::Float(f) => json!(f),
                crate::services::database_service::SettingValue::Boolean(b) => json!(b),
                crate::services::database_service::SettingValue::Json(j) => j,
            };
            Ok(HttpResponse::Ok().json(json!({
                "path": setting_path,
                "value": json_value
            })))
        }
        Ok(None) => {
            info!("[Settings Handler] Setting '{}' not found", setting_path);
            Ok(HttpResponse::NotFound().json(json!({
                "error": "not_found",
                "path": setting_path
            })))
        }
        Err(e) => {
            error!("[Settings Handler] Database error getting setting '{}': {}", setting_path, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to get setting: {}", e)
            })))
        }
    }
}

/// Update a single setting by path
pub async fn update_setting_by_path(
    state: web::Data<AppState>,
    path: web::Path<String>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let setting_path = path.into_inner();
    let value = payload.into_inner();

    info!("[Settings Handler] PUT /settings/path/{} - Direct database update", setting_path);

    // Convert JSON value to SettingValue
    use crate::services::database_service::SettingValue;
    let setting_value = match &value {
        Value::String(s) => SettingValue::String(s.clone()),
        Value::Number(n) if n.is_f64() => SettingValue::Float(n.as_f64().unwrap()),
        Value::Number(n) if n.is_i64() => SettingValue::Integer(n.as_i64().unwrap()),
        Value::Bool(b) => SettingValue::Boolean(*b),
        _ => SettingValue::Json(value.clone()),
    };

    match state.settings_service.set_setting(&setting_path, setting_value).await {
        Ok(()) => {
            info!("[Settings Handler] Setting '{}' updated in database", setting_path);
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "path": setting_path,
                "value": value
            })))
        }
        Err(e) => {
            error!("[Settings Handler] Database error updating setting '{}': {}", setting_path, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to update setting: {}", e)
            })))
        }
    }
}

/// Get batch of settings by paths
pub async fn get_settings_batch(
    state: web::Data<AppState>,
    payload: web::Json<Vec<String>>,
) -> Result<HttpResponse, Error> {
    let paths = payload.into_inner();
    info!("[Settings Handler] POST /settings/batch - Getting {} settings from database", paths.len());

    match state.settings_service.get_settings_batch(&paths).await {
        Ok(results) => {
            info!("[Settings Handler] Batch get successful: {}/{} settings found", results.len(), paths.len());
            // Convert SettingValue map to JSON map
            let json_results: std::collections::HashMap<String, Value> = results.into_iter().map(|(k, v)| {
                let json_value = match v {
                    crate::services::database_service::SettingValue::String(s) => json!(s),
                    crate::services::database_service::SettingValue::Integer(i) => json!(i),
                    crate::services::database_service::SettingValue::Float(f) => json!(f),
                    crate::services::database_service::SettingValue::Boolean(b) => json!(b),
                    crate::services::database_service::SettingValue::Json(j) => j,
                };
                (k, json_value)
            }).collect();
            Ok(HttpResponse::Ok().json(json_results))
        }
        Err(e) => {
            error!("[Settings Handler] Database error in batch get: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to get settings batch: {}", e)
            })))
        }
    }
}

/// Reset settings to defaults
pub async fn reset_settings(
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings/reset - Resetting to defaults in database");

    let default_settings = AppFullSettings::default();

    match state.settings_service.save_all_settings(&default_settings) {
        Ok(()) => {
            info!("[Settings Handler] Settings reset to defaults in database");
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Settings reset to defaults"
            })))
        }
        Err(e) => {
            error!("[Settings Handler] Database error resetting settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to reset settings: {}", e)
            })))
        }
    }
}

/// Health check for settings service
pub async fn settings_health(
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    // Try to load settings from database
    let healthy = state.settings_service.load_all_settings().is_ok();

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

/// Get physics settings for a specific graph
pub async fn get_physics_settings(
    state: web::Data<AppState>,
    graph_name: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let graph = graph_name.into_inner();
    info!("[Settings Handler] GET /settings/physics/{} - Loading from database", graph);

    // CRITICAL: Ensure graph separation (logseq vs visionflow)
    if graph != "logseq" && graph != "visionflow" && graph != "default" {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "invalid_graph",
            "message": format!("Invalid graph name: {}. Must be 'logseq', 'visionflow', or 'default'", graph)
        })));
    }

    match state.settings_service.get_physics_settings(&graph) {
        Ok(settings) => {
            info!("[Settings Handler] Physics settings for '{}' loaded from database", graph);
            Ok(HttpResponse::Ok().json(settings))
        }
        Err(e) => {
            error!("[Settings Handler] Database error getting physics for '{}': {}", graph, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to get physics settings: {}", e)
            })))
        }
    }
}

/// Update physics settings for a specific graph
pub async fn update_physics_settings(
    state: web::Data<AppState>,
    graph_name: web::Path<String>,
    payload: web::Json<crate::config::PhysicsSettings>,
) -> Result<HttpResponse, Error> {
    let graph = graph_name.into_inner();
    let physics_settings = payload.into_inner();

    info!("[Settings Handler] PUT /settings/physics/{} - Updating database", graph);

    // CRITICAL: Validate graph name to prevent conflation
    if graph != "logseq" && graph != "visionflow" && graph != "default" {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "invalid_graph",
            "message": format!("Invalid graph name: {}. Must be 'logseq', 'visionflow', or 'default'", graph)
        })));
    }

    match state.settings_service.save_physics_settings(&graph, &physics_settings) {
        Ok(()) => {
            info!("[Settings Handler] Physics settings for '{}' saved to database", graph);
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "graph": graph,
                "message": "Physics settings saved"
            })))
        }
        Err(e) => {
            error!("[Settings Handler] Database error saving physics for '{}': {}", graph, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to save physics settings: {}", e)
            })))
        }
    }
}

/// Export settings as JSON
pub async fn export_settings(
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] GET /settings/export - Exporting from database");

    match state.settings_service.load_all_settings() {
        Ok(Some(settings)) => {
            match serde_json::to_string_pretty(&settings) {
                Ok(json_string) => {
                    Ok(HttpResponse::Ok()
                        .content_type("application/json")
                        .insert_header(("Content-Disposition", "attachment; filename=\"settings.json\""))
                        .body(json_string))
                }
                Err(e) => {
                    error!("[Settings Handler] Failed to serialize settings: {}", e);
                    Ok(HttpResponse::InternalServerError().json(json!({
                        "error": "serialization_error"
                    })))
                }
            }
        }
        Ok(None) => {
            Ok(HttpResponse::NotFound().json(json!({
                "error": "no_settings"
            })))
        }
        Err(e) => {
            error!("[Settings Handler] Database error exporting settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to export settings: {}", e)
            })))
        }
    }
}

/// Import settings from JSON
pub async fn import_settings(
    state: web::Data<AppState>,
    payload: web::Json<AppFullSettings>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings/import - Importing to database");

    let settings = payload.into_inner();

    match state.settings_service.save_all_settings(&settings) {
        Ok(()) => {
            info!("[Settings Handler] Settings imported successfully to database");
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "message": "Settings imported"
            })))
        }
        Err(e) => {
            error!("[Settings Handler] Database error importing settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "database_error",
                "message": format!("Failed to import settings: {}", e)
            })))
        }
    }
}

/// Clear settings cache
pub async fn clear_cache(
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("[Settings Handler] POST /settings/cache/clear - Clearing cache");

    state.settings_service.clear_cache().await;

    Ok(HttpResponse::Ok().json(json!({
        "success": true,
        "message": "Cache cleared"
    })))
}
