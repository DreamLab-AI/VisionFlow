// Path-based Settings API Handler
// Provides efficient field-level access to settings without full serialization overhead

use crate::actors::messages::{GetSettings, UpdateSettings};
use crate::app_state::AppState;
use crate::config::path_access::JsonPathAccessible;
use crate::config::AppFullSettings;
use crate::utils::validation::rate_limit::extract_client_id;
use actix_web::{web, HttpRequest, HttpResponse, Result as ActixResult};
use log::{debug, error, info, warn};
use serde_json::{json, Value};

/// Get a specific settings value by path
///
/// GET /api/settings/path?path=visualisation.physics.damping
/// Returns: { "value": 0.95, "path": "visualisation.physics.damping" }
pub async fn get_settings_by_path(
    req: HttpRequest,
    query: web::Query<PathQuery>,
    state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let _client_id = extract_client_id(&req);

    // Rate limiting - would need to be implemented if rate_limiter exists in AppState
    // Currently commented out as rate_limiter field doesn't exist in current AppState
    /*
    if let Some(rate_limiter) = &state.rate_limiter {
        if let Err(retry_after) = rate_limiter.check_rate_limit(
            &client_id,
            "settings_get_path",
            &EndpointRateLimits::default()
        ) {
            return Ok(HttpResponse::TooManyRequests()
                .insert_header(("Retry-After", retry_after.to_string()))
                .json(json!({
                    "error": "Rate limit exceeded",
                    "retryAfter": retry_after
                })));
        }
    }
    */

    let path = &query.path;
    debug!("Getting settings value by path: {}", path);

    // Validate path format
    if path.is_empty() {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "Path cannot be empty",
            "path": path
        })));
    }

    // Get current settings from actor
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            // Use JsonPathAccessible trait to get the value
            match settings.get_json_by_path(path) {
                Ok(value) => {
                    info!("Successfully retrieved settings value at path: {}", path);
                    Ok(HttpResponse::Ok().json(json!({
                        "value": value,
                        "path": path,
                        "success": true
                    })))
                }
                Err(err) => {
                    warn!("Failed to get settings value at path '{}': {}", path, err);
                    Ok(HttpResponse::NotFound().json(json!({
                        "error": format!("Path not found: {}", err),
                        "path": path,
                        "success": false
                    })))
                }
            }
        }
        Ok(Err(err)) => {
            error!("Settings actor returned error: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get settings",
                "details": err,
                "success": false
            })))
        }
        Err(err) => {
            error!("Failed to communicate with settings actor: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error",
                "success": false
            })))
        }
    }
}

/// Update a specific settings value by path
///
/// PUT /api/settings/path
/// Body: { "path": "visualisation.physics.damping", "value": 0.98 }
/// Returns: { "success": true, "path": "visualisation.physics.damping", "previousValue": 0.95 }
pub async fn update_settings_by_path(
    req: HttpRequest,
    body: web::Json<PathUpdateRequest>,
    state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let _client_id = extract_client_id(&req);

    // Rate limiting - commented out (rate_limiter field doesn't exist in AppState)
    /*
    if let Some(rate_limiter) = &state.rate_limiter {
        if let Err(retry_after) = rate_limiter.check_rate_limit(
            &client_id,
            "settings_update_path",
            &EndpointRateLimits::default()
        ) {
            return Ok(HttpResponse::TooManyRequests()
                .insert_header(("Retry-After", retry_after.to_string()))
                .json(json!({
                    "error": "Rate limit exceeded",
                    "retryAfter": retry_after
                })));
        }
    }
    */

    let path = &body.path;
    let new_value = &body.value;

    debug!(
        "Updating settings value by path: {} = {:?}",
        path, new_value
    );

    // Validate path format
    if path.is_empty() {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "Path cannot be empty",
            "path": path,
            "success": false
        })));
    }

    // Get current settings from actor
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut settings)) => {
            // Get previous value for response
            let previous_value = settings.get_json_by_path(path).ok();

            // Attempt to set the new value using JsonPathAccessible
            match settings.set_json_by_path(path, new_value.clone()) {
                Ok(()) => {
                    // Validate the updated settings
                    if let Err(validation_errors) = settings.validate_config_camel_case() {
                        let error_details =
                            AppFullSettings::get_validation_errors_camel_case(&validation_errors);
                        warn!(
                            "Settings validation failed for path '{}': {:?}",
                            path, error_details
                        );

                        return Ok(HttpResponse::BadRequest().json(json!({
                            "error": "Validation failed",
                            "path": path,
                            "validationErrors": error_details,
                            "success": false
                        })));
                    }

                    // Send updated settings back to actor
                    match state.settings_addr.send(UpdateSettings { settings }).await {
                        Ok(Ok(())) => {
                            info!(
                                "Successfully updated settings at path: {} = {:?}",
                                path, new_value
                            );

                            // Notify WebSocket clients of the change - commented out (websocket_connections field doesn't exist)
                            /*
                            let change_notification = json!({
                                "type": "settingsChanged",
                                "path": path,
                                "value": new_value,
                                "previousValue": previous_value,
                                "timestamp": chrono::Utc::now().timestamp_millis()
                            });

                            if let Err(err) = state.websocket_connections.do_send(
                                crate::actors::messages::BroadcastMessage::new(change_notification.clone())
                            ) {
                                warn!("Failed to notify WebSocket clients: {}", err);
                            }
                            */

                            Ok(HttpResponse::Ok().json(json!({
                                "success": true,
                                "path": path,
                                "value": new_value,
                                "previousValue": previous_value,
                                "message": "Settings updated successfully"
                            })))
                        }
                        Ok(Err(err)) => {
                            error!("Failed to save updated settings: {}", err);
                            Ok(HttpResponse::InternalServerError().json(json!({
                                "error": "Failed to save settings",
                                "details": err,
                                "success": false
                            })))
                        }
                        Err(err) => {
                            error!("Failed to communicate with settings actor: {}", err);
                            Ok(HttpResponse::InternalServerError().json(json!({
                                "error": "Internal server error",
                                "success": false
                            })))
                        }
                    }
                }
                Err(err) => {
                    warn!("Failed to set settings value at path '{}': {}", path, err);
                    Ok(HttpResponse::BadRequest().json(json!({
                        "error": format!("Failed to update path: {}", err),
                        "path": path,
                        "success": false
                    })))
                }
            }
        }
        Ok(Err(err)) => {
            error!("Settings actor returned error: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings",
                "details": err,
                "success": false
            })))
        }
        Err(err) => {
            error!("Failed to communicate with settings actor: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error",
                "success": false
            })))
        }
    }
}

/// Batch read multiple settings values by path
///
/// POST /api/settings/batch
/// Body: { "paths": ["visualisation.physics.damping", "visualisation.physics.gravity"] }
/// Returns: { "values": [{"path": "...", "value": ...}, ...] }
pub async fn batch_read_settings_by_path(
    req: HttpRequest,
    body: web::Json<BatchPathReadRequest>,
    state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let _client_id = extract_client_id(&req);

    debug!("Batch reading {} settings paths", body.paths.len());

    if body.paths.is_empty() {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "No paths provided",
            "success": false
        })));
    }

    // Limit batch size to prevent abuse
    if body.paths.len() > 50 {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "Batch size exceeds maximum of 50 paths",
            "success": false
        })));
    }

    // Get current settings from actor
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            // Use a Map for direct key-value results (client expectation)
            let mut results = serde_json::Map::new();

            // Read all requested paths
            for path in &body.paths {
                if let Ok(value) = settings.get_json_by_path(path) {
                    results.insert(path.clone(), value);
                } else {
                    // Insert null for paths that don't exist
                    results.insert(path.clone(), serde_json::Value::Null);
                }
            }

            // Return the map directly as the JSON body (simple key-value format)
            Ok(HttpResponse::Ok().json(results))
        }
        Ok(Err(err)) => {
            error!("Settings actor returned error: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings",
                "details": err,
                "success": false
            })))
        }
        Err(err) => {
            error!("Failed to communicate with settings actor: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error",
                "success": false
            })))
        }
    }
}

/// Batch update multiple settings values by path
///
/// PUT /api/settings/batch
/// Body: {
///   "updates": [
///     { "path": "visualisation.physics.damping", "value": 0.98 },
///     { "path": "visualisation.physics.gravity", "value": 0.001 }
///   ]
/// }
pub async fn batch_update_settings_by_path(
    req: HttpRequest,
    body: web::Json<BatchPathUpdateRequest>,
    state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let _client_id = extract_client_id(&req);

    // Rate limiting - commented out (rate_limiter field doesn't exist in AppState)
    /*
    if let Some(rate_limiter) = &state.rate_limiter {
        if let Err(retry_after) = rate_limiter.check_rate_limit(
            &client_id,
            "settings_batch_update",
            &EndpointRateLimits::default()
        ) {
            return Ok(HttpResponse::TooManyRequests()
                .insert_header(("Retry-After", retry_after.to_string()))
                .json(json!({
                    "error": "Rate limit exceeded",
                    "retryAfter": retry_after
                })));
        }
    }
    */

    debug!("Batch updating {} settings paths", body.updates.len());

    if body.updates.is_empty() {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "No updates provided",
            "success": false
        })));
    }

    // Limit batch size to prevent abuse
    if body.updates.len() > 50 {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "Batch size exceeds maximum of 50 updates",
            "success": false
        })));
    }

    // Get current settings from actor
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut settings)) => {
            let mut results = Vec::new();
            let mut has_errors = false;

            // Apply all updates
            for update in &body.updates {
                let previous_value = settings.get_json_by_path(&update.path).ok();

                match settings.set_json_by_path(&update.path, update.value.clone()) {
                    Ok(()) => {
                        results.push(json!({
                            "path": update.path,
                            "success": true,
                            "value": update.value,
                            "previousValue": previous_value
                        }));
                    }
                    Err(err) => {
                        has_errors = true;
                        results.push(json!({
                            "path": update.path,
                            "success": false,
                            "error": err
                        }));
                    }
                }
            }

            if has_errors {
                return Ok(HttpResponse::BadRequest().json(json!({
                    "success": false,
                    "message": "Some updates failed",
                    "results": results
                })));
            }

            // Validate the updated settings
            if let Err(validation_errors) = settings.validate_config_camel_case() {
                let error_details =
                    AppFullSettings::get_validation_errors_camel_case(&validation_errors);
                warn!("Batch settings validation failed: {:?}", error_details);

                return Ok(HttpResponse::BadRequest().json(json!({
                    "error": "Validation failed after batch update",
                    "validationErrors": error_details,
                    "success": false
                })));
            }

            // Send updated settings back to actor
            match state.settings_addr.send(UpdateSettings { settings }).await {
                Ok(Ok(())) => {
                    info!(
                        "Successfully applied {} batch settings updates",
                        body.updates.len()
                    );

                    // Notify WebSocket clients of the changes - commented out (websocket_connections field doesn't exist)
                    /*
                    let change_notification = json!({
                        "type": "settingsBatchChanged",
                        "updates": results,
                        "timestamp": chrono::Utc::now().timestamp_millis()
                    });

                    if let Err(err) = state.websocket_connections.do_send(
                        crate::actors::messages::BroadcastMessage::new(change_notification.clone())
                    ) {
                        warn!("Failed to notify WebSocket clients: {}", err);
                    }
                    */

                    Ok(HttpResponse::Ok().json(json!({
                        "success": true,
                        "message": format!("Successfully updated {} settings", body.updates.len()),
                        "results": results
                    })))
                }
                Ok(Err(err)) => {
                    error!("Failed to save batch updated settings: {}", err);
                    Ok(HttpResponse::InternalServerError().json(json!({
                        "error": "Failed to save settings",
                        "details": err,
                        "success": false
                    })))
                }
                Err(err) => {
                    error!("Failed to communicate with settings actor: {}", err);
                    Ok(HttpResponse::InternalServerError().json(json!({
                        "error": "Internal server error",
                        "success": false
                    })))
                }
            }
        }
        Ok(Err(err)) => {
            error!("Settings actor returned error: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings",
                "details": err,
                "success": false
            })))
        }
        Err(err) => {
            error!("Failed to communicate with settings actor: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error",
                "success": false
            })))
        }
    }
}

/// Get the schema/structure of settings for a given path
///
/// GET /api/settings/schema?path=visualisation.physics
/// Returns the structure and validation rules for the specified path
pub async fn get_settings_schema(
    req: HttpRequest,
    query: web::Query<PathQuery>,
    state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let _client_id = extract_client_id(&req);

    // Rate limiting - commented out (rate_limiter field doesn't exist in AppState)
    /*
    if let Some(rate_limiter) = &state.rate_limiter {
        if let Err(retry_after) = rate_limiter.check_rate_limit(
            &client_id,
            "settings_schema",
            &EndpointRateLimits::default()
        ) {
            return Ok(HttpResponse::TooManyRequests()
                .insert_header(("Retry-After", retry_after.to_string()))
                .json(json!({
                    "error": "Rate limit exceeded",
                    "retryAfter": retry_after
                })));
        }
    }
    */

    let path = &query.path;
    debug!("Getting settings schema for path: {}", path);

    // Get current settings to extract schema
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => match settings.get_json_by_path(path) {
            Ok(value) => {
                let schema = generate_value_schema(&value, path);
                Ok(HttpResponse::Ok().json(json!({
                    "path": path,
                    "schema": schema,
                    "success": true
                })))
            }
            Err(err) => Ok(HttpResponse::NotFound().json(json!({
                "error": format!("Path not found: {}", err),
                "path": path,
                "success": false
            }))),
        },
        Ok(Err(err)) => {
            error!("Settings actor returned error: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get settings",
                "details": err,
                "success": false
            })))
        }
        Err(err) => {
            error!("Failed to communicate with settings actor: {}", err);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Internal server error",
                "success": false
            })))
        }
    }
}

// Request/Response DTOs

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathQuery {
    pub path: String,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PathUpdateRequest {
    pub path: String,
    pub value: Value,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchPathUpdateRequest {
    pub updates: Vec<PathUpdateRequest>,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchPathReadRequest {
    pub paths: Vec<String>,
}

// Helper functions

/// Generate a basic schema description for a JSON value
fn generate_value_schema(value: &Value, path: &str) -> Value {
    match value {
        Value::Bool(_) => json!({
            "type": "boolean",
            "path": path
        }),
        Value::Number(n) => {
            if n.is_f64() {
                json!({
                    "type": "number",
                    "format": "float",
                    "path": path
                })
            } else {
                json!({
                    "type": "integer",
                    "path": path
                })
            }
        }
        Value::String(_) => json!({
            "type": "string",
            "path": path
        }),
        Value::Array(arr) => {
            let item_type = if let Some(first) = arr.first() {
                generate_value_schema(first, &format!("{}[0]", path))
            } else {
                json!({"type": "unknown"})
            };
            json!({
                "type": "array",
                "items": item_type,
                "path": path
            })
        }
        Value::Object(obj) => {
            let mut properties = serde_json::Map::new();
            for (key, val) in obj {
                properties.insert(
                    key.clone(),
                    generate_value_schema(val, &format!("{}.{}", path, key)),
                );
            }
            json!({
                "type": "object",
                "properties": properties,
                "path": path
            })
        }
        Value::Null => json!({
            "type": "null",
            "path": path
        }),
    }
}

// Configuration for path-based settings routes
pub fn configure_settings_paths(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/settings")
            .route("/path", web::get().to(get_settings_by_path))
            .route("/path", web::put().to(update_settings_by_path))
            .route("/batch", web::post().to(batch_read_settings_by_path))
            .route("/batch", web::put().to(batch_update_settings_by_path))
            .route("/schema", web::get().to(get_settings_schema)),
    );
}
