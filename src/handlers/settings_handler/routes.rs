// Route configuration and read-only route handler functions

use crate::actors::messages::GetSettings;
use crate::app_state::AppState;
use crate::config::path_access::JsonPathAccessible;
use actix_web::{web, Error, HttpRequest, HttpResponse};
use log::{error, warn};
use serde_json::{json, Value};
use std::borrow::Cow;
use crate::{ok_json, error_json, not_found, service_unavailable};

use super::enhanced::EnhancedSettingsHandler;
use super::types::SettingsResponseDTO;
use super::write_handlers;

pub fn config(cfg: &mut web::ServiceConfig) {
    let handler = web::Data::new(EnhancedSettingsHandler::new());

    cfg.app_data(handler.clone())
        .service(
            web::scope("/settings")
                .route("/path", web::get().to(get_setting_by_path))
                .route("/path", web::put().to(update_setting_by_path))
                .route("/schema", web::get().to(get_settings_schema))
                .route("/current", web::get().to(get_current_settings))
                .route("", web::get().to(get_settings))
                .route("", web::post().to(write_handlers::update_settings))
                .route("/reset", web::post().to(write_handlers::reset_settings))
                .route("/save", web::post().to(write_handlers::save_settings))
                .route(
                    "/validation/stats",
                    web::get().to(
                        |req, handler: web::Data<EnhancedSettingsHandler>| async move {
                            handler.get_validation_stats(req).await
                        },
                    ),
                ),
        )
        .service(
            web::scope("/api/physics").route("/compute-mode", web::post().to(super::physics::update_compute_mode)),
        )
        .service(
            web::scope("/api/clustering")
                .route("/algorithm", web::post().to(super::physics::update_clustering_algorithm)),
        )
        .service(
            web::scope("/api/constraints").route("/update", web::post().to(super::physics::update_constraints)),
        )
        .service(
            web::scope("/api/analytics").route("/clusters", web::get().to(super::physics::get_cluster_analytics)),
        )
        .service(
            web::scope("/api/stress")
                .route("/optimization", web::post().to(super::physics::update_stress_optimization)),
        );
}

async fn get_setting_by_path(
    req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let path = req
        .query_string()
        .split('&')
        .find(|param| param.starts_with("path="))
        .and_then(|p| p.strip_prefix("path="))
        .map(|p| {
            urlencoding::decode(p)
                .unwrap_or(Cow::Borrowed(p))
                .to_string()
        })
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing 'path' query parameter"))?;

    let app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return error_json!("Failed to retrieve settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    match app_settings.get_json_by_path(&path) {
        Ok(value_json) => ok_json!(json!({
            "success": true,
            "path": path,
            "value": value_json
        })),
        Err(e) => {
            warn!("Path not found '{}': {}", path, e);
            not_found!("Path not found", e)
        }
    }
}

async fn update_setting_by_path(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    use crate::actors::messages::UpdateSettings;
    use log::info;
    use crate::bad_request;
    use super::physics::propagate_physics_to_gpu;

    let update = payload.into_inner();
    let path = update
        .get("path")
        .and_then(|p| p.as_str())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing 'path' in request body"))?
        .to_string();
    let value = update
        .get("value")
        .ok_or_else(|| actix_web::error::ErrorBadRequest("Missing 'value' in request body"))?
        .clone();

    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return error_json!("Failed to retrieve settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    let previous_value = app_settings.get_json_by_path(&path).ok();

    match app_settings.set_json_by_path(&path, value.clone()) {
        Ok(()) => {
            match state
                .settings_addr
                .send(UpdateSettings {
                    settings: app_settings.clone(),
                })
                .await
            {
                Ok(Ok(())) => {
                    info!("Updated setting at path: {}", path);

                    if path.contains(".physics.")
                        || path.contains(".graphs.logseq.")
                        || path.contains(".graphs.visionflow.")
                    {
                        info!("Physics setting changed, propagating to GPU actors");

                        let graph_name = if path.contains(".graphs.logseq.") {
                            "logseq"
                        } else if path.contains(".graphs.visionflow.") {
                            "visionflow"
                        } else {
                            "logseq"
                        };

                        propagate_physics_to_gpu(&state, &app_settings, graph_name).await;
                    }

                    ok_json!(json!({
                        "success": true,
                        "path": path,
                        "value": update.get("value").cloned().unwrap_or(Value::Null),
                        "previousValue": previous_value
                    }))
                }
                Ok(Err(e)) => {
                    error!("Failed to save settings: {}", e);
                    Ok(HttpResponse::InternalServerError().json(json!({
                        "error": format!("Failed to save settings: {}", e),
                        "path": path
                    })))
                }
                Err(e) => {
                    error!("Settings actor error: {}", e);
                    service_unavailable!("Settings service unavailable")
                }
            }
        }
        Err(e) => {
            warn!("Failed to update path '{}': {}", path, e);
            bad_request!("Invalid path or value", e)
        }
    }
}

async fn get_settings_schema(
    req: HttpRequest,
    _state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let path = req
        .query_string()
        .split('&')
        .find(|param| param.starts_with("path="))
        .and_then(|p| p.strip_prefix("path="))
        .map(|p| {
            urlencoding::decode(p)
                .unwrap_or(Cow::Borrowed(p))
                .to_string()
        })
        .unwrap_or_default();

    let schema = json!({
        "type": "object",
        "properties": {
            "damping": { "type": "number", "description": "Physics damping factor (0.0-1.0)" },
            "gravity": { "type": "number", "description": "Physics gravity strength" },
        },
        "path": path
    });

    ok_json!(json!({
        "success": true,
        "path": path,
        "schema": schema
    }))
}

async fn get_settings(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return error_json!("Failed to retrieve settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    let response_dto: SettingsResponseDTO = (&app_settings).into();

    ok_json!(response_dto)
}

async fn get_current_settings(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return error_json!("Failed to retrieve settings");
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return service_unavailable!("Settings service unavailable");
        }
    };

    let response_dto: SettingsResponseDTO = (&app_settings).into();

    ok_json!(json!({
        "settings": response_dto,
        "version": app_settings.version,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }))
}
