// Refactored Settings Handler - Granular API with direct AppFullSettings serialization
use actix_web::{web, HttpResponse, HttpRequest, Result};
use crate::app_state::AppState;
use crate::actors::messages::{GetSettingsByPaths, SetSettingsByPaths};
// Remove unused rate limiting for now
use log::{info, error, debug};
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use std::collections::HashMap;

/// Request payload for setting multiple values at once
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsSetRequest {
    pub updates: Vec<SettingsUpdate>,
}

/// Individual setting update
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsUpdate {
    pub path: String,
    pub value: Value,
}

/// Response for getting settings
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsGetResponse {
    pub settings: HashMap<String, Value>,
}

/// Success response for setting operations
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsSetResponse {
    pub success: bool,
    pub updated_paths: Vec<String>,
    pub errors: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_errors: Option<HashMap<String, String>>,
}

/// GET /api/settings/get?paths=path1,path2,path3
/// Retrieves specific settings by dot-notation paths
pub async fn get_settings(
    req: HttpRequest,
    query: web::Query<HashMap<String, String>>,
    data: web::Data<AppState>,
) -> Result<HttpResponse> {
    // Rate limiting removed for now - can be added back later if needed

    // Extract paths from query parameter
    let empty_string = String::new();
    let paths_str = query.get("paths").unwrap_or(&empty_string);
    let requested_paths: Vec<&str> = if paths_str.is_empty() {
        vec![] // Return all settings if no paths specified
    } else {
        paths_str.split(',').collect()
    };

    // Always use granular operations for better performance
    let paths = if requested_paths.is_empty() {
        // If no specific paths requested, return error - client should specify what they need
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "No paths specified. Please specify which settings you need using ?paths=path1,path2",
            "code": "MISSING_PATHS"
        })));
    } else {
        requested_paths.into_iter().map(|s| s.to_string()).collect()
    };
    
    // Use granular GetSettingsByPaths - MUCH MORE EFFICIENT!
    match data.settings_addr.send(GetSettingsByPaths { paths }).await {
        Ok(Ok(response_settings)) => {
            info!("Successfully retrieved {} setting paths using granular operation", response_settings.len());
            Ok(HttpResponse::Ok().json(SettingsGetResponse {
                settings: response_settings,
            }))
        }
        Ok(Err(e)) => {
            error!("Settings actor returned error: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to retrieve settings",
                "code": "ACTOR_ERROR",
                "details": e
            })))
        }
        Err(e) => {
            error!("Failed to communicate with settings actor: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Settings service unavailable",
                "code": "SERVICE_UNAVAILABLE"
            })))
        }
    }
}

/// POST /api/settings/set
/// Sets multiple settings using dot-notation paths
pub async fn set_settings(
    req: HttpRequest,
    request: web::Json<SettingsSetRequest>,
    data: web::Data<AppState>,
) -> Result<HttpResponse> {
    // Rate limiting removed for now - can be added back later if needed

    if request.updates.is_empty() {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "No updates provided",
            "code": "EMPTY_UPDATE"
        })));
    }

    debug!("Processing {} setting updates using granular operations", request.updates.len());

    // Prepare updates for granular operation - MUCH MORE EFFICIENT!
    let updates: Vec<(String, Value)> = request.updates.iter()
        .map(|u| (u.path.clone(), u.value.clone()))
        .collect();
    
    // Validation is now handled by the SettingsActor using the validator crate on structs
    // This ensures single source of truth for validation

    // Use granular SetSettingsByPaths - NO MORE FULL OBJECT OPERATIONS!
    match data.settings_addr.send(SetSettingsByPaths { updates }).await {
        Ok(Ok(())) => {
            let updated_paths: Vec<String> = request.updates.iter()
                .map(|u| u.path.clone())
                .collect();
            info!("Successfully updated {} settings using granular operation", updated_paths.len());
            Ok(HttpResponse::Ok().json(SettingsSetResponse {
                success: true,
                updated_paths,
                errors: vec![], // Granular operation handles errors internally
                validation_errors: None,
            }))
        }
        Ok(Err(e)) => {
            error!("Settings actor failed to update: {}", e);
            
            // Check if this is a validation error and extract details
            if e.starts_with("Validation failed:") {
                // Parse validation errors from the error message
                let validation_part = e.strip_prefix("Validation failed: ").unwrap_or(&e);
                let validation_errors: std::collections::HashMap<String, String> = validation_part
                    .split("; ")
                    .filter_map(|error_str| {
                        let parts: Vec<&str> = error_str.splitn(2, ": ").collect();
                        if parts.len() == 2 {
                            Some((parts[0].to_string(), parts[1].to_string()))
                        } else {
                            None
                        }
                    })
                    .collect();
                
                Ok(HttpResponse::BadRequest().json(SettingsSetResponse {
                    success: false,
                    updated_paths: vec![],
                    errors: vec!["Settings validation failed".to_string()],
                    validation_errors: Some(validation_errors),
                }))
            } else {
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "Failed to persist settings",
                    "code": "PERSISTENCE_ERROR",
                    "details": e
                })))
            }
        }
        Err(e) => {
            error!("Failed to communicate with settings actor: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Settings service unavailable",
                "code": "SERVICE_UNAVAILABLE"
            })))
        }
    }
}

// Path operations are now handled by the SettingsActor for better performance

// Validation is now centralized in the SettingsActor using the validator crate
// The server validates the entire AppFullSettings struct after updates
// This eliminates duplication and ensures consistent validation

// Path operations are now handled by the SettingsActor with proper validation

/// Configure routes for the settings handler
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/settings")
            .route("/get", web::get().to(get_settings))
            .route("/set", web::post().to(set_settings))
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_settings_request_structures() {
        // Test SettingsUpdate deserialization
        let update_json = json!({
            "path": "visualisation.physics.damping",
            "value": 0.95
        });
        
        let update: SettingsUpdate = serde_json::from_value(update_json).unwrap();
        assert_eq!(update.path, "visualisation.physics.damping");
        assert_eq!(update.value, json!(0.95));
        
        // Test SettingsSetRequest deserialization
        let request_json = json!({
            "updates": [
                {
                    "path": "visualisation.physics.damping",
                    "value": 0.95
                },
                {
                    "path": "system.debug.enabled",
                    "value": true
                }
            ]
        });
        
        let request: SettingsSetRequest = serde_json::from_value(request_json).unwrap();
        assert_eq!(request.updates.len(), 2);
        assert_eq!(request.updates[0].path, "visualisation.physics.damping");
        assert_eq!(request.updates[1].path, "system.debug.enabled");
    }
}