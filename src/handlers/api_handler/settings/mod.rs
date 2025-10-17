// REST API for Settings Management
// Provides complete CRUD operations for settings with validation and permission checks

use actix_web::{web, HttpRequest, HttpResponse, Error as ActixError};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use std::sync::Arc;
use log::{info, error, debug, warn};

use crate::app_state::AppState;
use crate::services::settings_service::SettingsService;
use crate::services::database_service::SettingValue;
use crate::config::PhysicsSettings;

/// Response DTO for settings list
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsListResponse {
    pub settings: Vec<SettingItem>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingItem {
    pub key: String,
    pub value: JsonValue,
    pub value_type: String,
}

/// Request DTO for setting update
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateSettingRequest {
    pub value: JsonValue,
}

/// Request DTO for validation only
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidateSettingRequest {
    pub key: String,
    pub value: JsonValue,
}

/// Response DTO for validation
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ValidationResponse {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

/// Extract user ID from request (from Nostr pubkey or auth token)
fn extract_user_id(req: &HttpRequest) -> Option<String> {
    req.headers()
        .get("x-nostr-pubkey")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
}

/// Check if user has power user permissions
fn check_power_user(app_state: &AppState, user_id: Option<&str>) -> bool {
    match user_id {
        Some(uid) => app_state.is_power_user(uid),
        None => false,
    }
}

/// GET /api/settings - List all settings with permission filtering
pub async fn list_settings(
    req: HttpRequest,
    app_state: web::Data<AppState>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let user_id = extract_user_id(&req);
    let is_power_user = check_power_user(&app_state, user_id.as_deref());

    debug!("Listing settings for user: {:?} (power: {})", user_id, is_power_user);

    match settings_service.list_all_settings().await {
        Ok(settings) => {
            let items: Vec<SettingItem> = settings
                .into_iter()
                .map(|(key, value)| {
                    let (json_value, value_type) = convert_setting_value_to_json(&value);
                    SettingItem {
                        key,
                        value: json_value,
                        value_type,
                    }
                })
                .collect();

            let total = items.len();
            Ok(HttpResponse::Ok().json(SettingsListResponse {
                settings: items,
                total,
            }))
        }
        Err(e) => {
            error!("Failed to list settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to list settings",
                "details": e
            })))
        }
    }
}

/// GET /api/settings/{key} - Get specific setting (supports dots in key)
pub async fn get_setting(
    path: web::Path<String>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let key = path.into_inner();

    debug!("Getting setting: {}", key);

    match settings_service.get_setting(&key).await {
        Ok(Some(value)) => {
            let (json_value, value_type) = convert_setting_value_to_json(&value);
            Ok(HttpResponse::Ok().json(json!({
                "key": key,
                "value": json_value,
                "valueType": value_type
            })))
        }
        Ok(None) => Ok(HttpResponse::NotFound().json(json!({
            "error": "Setting not found",
            "key": key
        }))),
        Err(e) => {
            error!("Failed to get setting {}: {}", key, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get setting",
                "details": e
            })))
        }
    }
}

/// PUT /api/settings/{key} - Update setting (requires power user)
pub async fn update_setting(
    req: HttpRequest,
    path: web::Path<String>,
    body: web::Json<UpdateSettingRequest>,
    app_state: web::Data<AppState>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let key = path.into_inner();
    let user_id = extract_user_id(&req);

    // Check power user permission
    if !check_power_user(&app_state, user_id.as_deref()) {
        warn!("Unauthorized settings update attempt by user: {:?}", user_id);
        return Ok(HttpResponse::Forbidden().json(json!({
            "error": "Power user permission required"
        })));
    }

    debug!("Updating setting: {} by user: {:?}", key, user_id);

    // Convert JSON value to SettingValue
    let setting_value = match json_value_to_setting_value(&body.value) {
        Ok(v) => v,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().json(json!({
                "error": "Invalid value type",
                "details": e
            })));
        }
    };

    match settings_service
        .set_setting(&key, setting_value, user_id.as_deref())
        .await
    {
        Ok(()) => {
            info!("Setting {} updated by user {:?}", key, user_id);
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "key": key,
                "message": "Setting updated successfully"
            })))
        }
        Err(e) => {
            error!("Failed to update setting {}: {}", key, e);
            Ok(HttpResponse::BadRequest().json(json!({
                "error": "Failed to update setting",
                "details": e
            })))
        }
    }
}

/// GET /api/settings/tree/{prefix} - Get hierarchical tree
pub async fn get_settings_tree(
    path: web::Path<String>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let prefix = path.into_inner();

    debug!("Getting settings tree for prefix: {}", prefix);

    match settings_service.get_settings_tree(&prefix).await {
        Ok(tree) => Ok(HttpResponse::Ok().json(tree)),
        Err(e) => {
            error!("Failed to get settings tree for {}: {}", prefix, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get settings tree",
                "details": e
            })))
        }
    }
}

/// GET /api/settings/physics/{profile} - Get physics profile
pub async fn get_physics_profile(
    path: web::Path<String>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let profile_name = path.into_inner();

    debug!("Getting physics profile: {}", profile_name);

    match settings_service.get_physics_profile(&profile_name).await {
        Ok(profile) => Ok(HttpResponse::Ok().json(profile)),
        Err(e) => {
            error!("Failed to get physics profile {}: {}", profile_name, e);
            Ok(HttpResponse::NotFound().json(json!({
                "error": "Physics profile not found",
                "profile": profile_name,
                "details": e
            })))
        }
    }
}

/// PUT /api/settings/physics/{profile} - Update physics profile (requires power user)
pub async fn update_physics_profile(
    req: HttpRequest,
    path: web::Path<String>,
    body: web::Json<PhysicsSettings>,
    app_state: web::Data<AppState>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let profile_name = path.into_inner();
    let user_id = extract_user_id(&req);

    // Check power user permission
    if !check_power_user(&app_state, user_id.as_deref()) {
        warn!(
            "Unauthorized physics profile update attempt by user: {:?}",
            user_id
        );
        return Ok(HttpResponse::Forbidden().json(json!({
            "error": "Power user permission required"
        })));
    }

    debug!(
        "Updating physics profile: {} by user: {:?}",
        profile_name, user_id
    );

    match settings_service
        .update_physics_profile(&profile_name, body.into_inner(), user_id.as_deref())
        .await
    {
        Ok(()) => {
            info!(
                "Physics profile {} updated by user {:?}",
                profile_name, user_id
            );
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "profile": profile_name,
                "message": "Physics profile updated successfully"
            })))
        }
        Err(e) => {
            error!("Failed to update physics profile {}: {}", profile_name, e);
            Ok(HttpResponse::BadRequest().json(json!({
                "error": "Failed to update physics profile",
                "details": e
            })))
        }
    }
}

/// POST /api/settings/validate - Validate settings without saving
pub async fn validate_setting(
    body: web::Json<ValidateSettingRequest>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    debug!("Validating setting: {}", body.key);

    // Convert JSON value to SettingValue
    let setting_value = match json_value_to_setting_value(&body.value) {
        Ok(v) => v,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().json(json!({
                "error": "Invalid value type",
                "details": e
            })));
        }
    };

    // Perform validation
    let validator = &settings_service.validator;
    match validator.validate_setting(&body.key, &setting_value) {
        Ok(result) => Ok(HttpResponse::Ok().json(ValidationResponse {
            is_valid: result.is_valid,
            errors: result.errors,
            warnings: result.warnings,
        })),
        Err(e) => {
            error!("Validation error for {}: {}", body.key, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Validation failed",
                "details": e
            })))
        }
    }
}

/// GET /api/settings/search?q=pattern - Search settings
pub async fn search_settings(
    query: web::Query<SearchQuery>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let pattern = &query.q;

    debug!("Searching settings with pattern: {}", pattern);

    match settings_service.search_settings(pattern).await {
        Ok(results) => {
            let items: Vec<SettingItem> = results
                .into_iter()
                .map(|(key, value)| {
                    let (json_value, value_type) = convert_setting_value_to_json(&value);
                    SettingItem {
                        key,
                        value: json_value,
                        value_type,
                    }
                })
                .collect();

            Ok(HttpResponse::Ok().json(json!({
                "results": items,
                "count": items.len()
            })))
        }
        Err(e) => {
            error!("Failed to search settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to search settings",
                "details": e
            })))
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    pub q: String,
}

/// DELETE /api/settings/{key} - Reset to default (requires power user)
pub async fn reset_setting(
    req: HttpRequest,
    path: web::Path<String>,
    app_state: web::Data<AppState>,
    settings_service: web::Data<Arc<SettingsService>>,
) -> Result<HttpResponse, ActixError> {
    let key = path.into_inner();
    let user_id = extract_user_id(&req);

    // Check power user permission
    if !check_power_user(&app_state, user_id.as_deref()) {
        warn!("Unauthorized reset attempt by user: {:?}", user_id);
        return Ok(HttpResponse::Forbidden().json(json!({
            "error": "Power user permission required"
        })));
    }

    debug!("Resetting setting: {} by user: {:?}", key, user_id);

    match settings_service
        .reset_to_default(&key, user_id.as_deref())
        .await
    {
        Ok(()) => {
            info!("Setting {} reset to default by user {:?}", key, user_id);
            Ok(HttpResponse::Ok().json(json!({
                "success": true,
                "key": key,
                "message": "Setting reset to default"
            })))
        }
        Err(e) => {
            error!("Failed to reset setting {}: {}", key, e);
            Ok(HttpResponse::BadRequest().json(json!({
                "error": "Failed to reset setting",
                "details": e
            })))
        }
    }
}

/// Helper: Convert SettingValue to JSON
fn convert_setting_value_to_json(value: &SettingValue) -> (JsonValue, String) {
    match value {
        SettingValue::String(s) => (json!(s), "string".to_string()),
        SettingValue::Integer(i) => (json!(i), "integer".to_string()),
        SettingValue::Float(f) => (json!(f), "float".to_string()),
        SettingValue::Boolean(b) => (json!(b), "boolean".to_string()),
        SettingValue::Json(j) => (j.clone(), "json".to_string()),
    }
}

/// Helper: Convert JSON value to SettingValue
fn json_value_to_setting_value(value: &JsonValue) -> Result<SettingValue, String> {
    match value {
        JsonValue::String(s) => Ok(SettingValue::String(s.clone())),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(SettingValue::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(SettingValue::Float(f))
            } else {
                Err("Invalid number type".to_string())
            }
        }
        JsonValue::Bool(b) => Ok(SettingValue::Boolean(*b)),
        JsonValue::Object(_) | JsonValue::Array(_) => Ok(SettingValue::Json(value.clone())),
        JsonValue::Null => Err("Cannot store null values".to_string()),
    }
}

/// Configure settings API routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/settings")
            .route("", web::get().to(list_settings))
            .route("/search", web::get().to(search_settings))
            .route("/validate", web::post().to(validate_setting))
            .route("/tree/{prefix:.*}", web::get().to(get_settings_tree))
            .route("/physics/{profile}", web::get().to(get_physics_profile))
            .route("/physics/{profile}", web::put().to(update_physics_profile))
            .route("/{key:.*}", web::get().to(get_setting))
            .route("/{key:.*}", web::put().to(update_setting))
            .route("/{key:.*}", web::delete().to(reset_setting)),
    );
}
