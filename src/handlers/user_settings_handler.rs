use actix_web::{web, Error, HttpResponse, HttpRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use log::{error, info};

use crate::services::user_service::{UserService, UserServiceError, SettingValue, UserSetting};
use crate::middleware::permissions::extract_auth_context;

#[derive(Debug, Serialize)]
struct UserSettingsResponse {
    settings: Vec<UserSettingDTO>,
}

#[derive(Debug, Serialize)]
struct UserSettingDTO {
    key: String,
    value: JsonValue,
    created_at: i64,
    updated_at: i64,
}

#[derive(Debug, Deserialize)]
struct SetUserSettingRequest {
    key: String,
    value: JsonValue,
}

#[derive(Debug, Serialize)]
struct MessageResponse {
    message: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

fn setting_value_to_json(value: &SettingValue) -> JsonValue {
    match value {
        SettingValue::String(s) => JsonValue::String(s.clone()),
        SettingValue::Integer(i) => JsonValue::Number(serde_json::Number::from(*i)),
        SettingValue::Float(f) => {
            serde_json::Number::from_f64(*f)
                .map(JsonValue::Number)
                .unwrap_or(JsonValue::Null)
        }
        SettingValue::Boolean(b) => JsonValue::Bool(*b),
        SettingValue::Json(j) => j.clone(),
    }
}

fn json_to_setting_value(value: JsonValue) -> SettingValue {
    match value {
        JsonValue::String(s) => SettingValue::String(s),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                SettingValue::Integer(i)
            } else if let Some(f) = n.as_f64() {
                SettingValue::Float(f)
            } else {
                SettingValue::String(n.to_string())
            }
        }
        JsonValue::Bool(b) => SettingValue::Boolean(b),
        _ => SettingValue::Json(value),
    }
}

pub async fn get_user_settings(
    req: HttpRequest,
    user_service: web::Data<UserService>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    match user_service.get_user_settings(auth_context.user_id).await {
        Ok(settings) => {
            let dto: Vec<UserSettingDTO> = settings
                .into_iter()
                .map(|s| UserSettingDTO {
                    key: s.key,
                    value: setting_value_to_json(&s.value),
                    created_at: s.created_at,
                    updated_at: s.updated_at,
                })
                .collect();

            info!(
                "Retrieved {} settings for user_id={}",
                dto.len(),
                auth_context.user_id
            );
            Ok(HttpResponse::Ok().json(UserSettingsResponse { settings: dto }))
        }
        Err(e) => {
            error!("Failed to get user settings: {:?}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to fetch user settings".to_string(),
            }))
        }
    }
}

pub async fn set_user_setting(
    req: HttpRequest,
    user_service: web::Data<UserService>,
    payload: web::Json<SetUserSettingRequest>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    if !auth_context.is_power_user {
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error: "Power user access required to modify settings".to_string(),
        }));
    }

    let setting_value = json_to_setting_value(payload.value.clone());

    match user_service
        .set_user_setting(auth_context.user_id, &payload.key, setting_value)
        .await
    {
        Ok(()) => {
            info!(
                "User {} set setting {}",
                auth_context.nostr_pubkey, payload.key
            );
            Ok(HttpResponse::Ok().json(MessageResponse {
                message: format!("Setting {} updated successfully", payload.key),
            }))
        }
        Err(e) => {
            error!("Failed to set user setting: {:?}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to update setting".to_string(),
            }))
        }
    }
}

pub async fn delete_user_setting(
    req: HttpRequest,
    user_service: web::Data<UserService>,
    path: web::Path<String>,
) -> Result<HttpResponse, Error> {
    let auth_context = extract_auth_context(&req)
        .ok_or_else(|| actix_web::error::ErrorUnauthorized("Authentication required"))?;

    if !auth_context.is_power_user {
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error: "Power user access required to delete settings".to_string(),
        }));
    }

    let key = path.into_inner();

    match user_service
        .delete_user_setting(auth_context.user_id, &key)
        .await
    {
        Ok(()) => {
            info!(
                "User {} deleted setting {}",
                auth_context.nostr_pubkey, key
            );
            Ok(HttpResponse::Ok().json(MessageResponse {
                message: format!("Setting {} deleted successfully", key),
            }))
        }
        Err(UserServiceError::UserNotFound) => Ok(HttpResponse::NotFound().json(ErrorResponse {
            error: "Setting not found".to_string(),
        })),
        Err(e) => {
            error!("Failed to delete user setting: {:?}", e);
            Ok(HttpResponse::InternalServerError().json(ErrorResponse {
                error: "Failed to delete setting".to_string(),
            }))
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/user-settings")
            .route("", web::get().to(get_user_settings))
            .route("", web::post().to(set_user_setting))
            .route("/{key}", web::delete().to(delete_user_setting)),
    );
}
