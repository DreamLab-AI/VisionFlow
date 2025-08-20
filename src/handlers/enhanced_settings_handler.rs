// Enhanced Settings Handler with Comprehensive Input Validation
use actix_web::{web, Error, HttpResponse, HttpRequest};
use crate::app_state::AppState;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, UpdateSimulationParams};
use crate::handlers::validation_handler::ValidationService;
use crate::utils::validation::rate_limit::{RateLimiter, RateLimitConfig, EndpointRateLimits, extract_client_id};
use crate::utils::validation::sanitization::Sanitizer;
use crate::utils::validation::errors::{DetailedValidationError, ValidationErrorCollection};
use crate::utils::validation::{ValidationContext, MAX_REQUEST_SIZE};
use log::{info, warn, error, debug};
use serde_json::{json, Value};
use std::sync::Arc;

/// Enhanced settings handler with comprehensive validation
pub struct EnhancedSettingsHandler {
    validation_service: ValidationService,
    rate_limiter: Arc<RateLimiter>,
}

impl EnhancedSettingsHandler {
    pub fn new() -> Self {
        let config = EndpointRateLimits::settings_update();
        let rate_limiter = Arc::new(RateLimiter::new(config));

        Self {
            validation_service: ValidationService::new(),
            rate_limiter,
        }
    }

    /// Enhanced settings update with full validation
    pub async fn update_settings_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
        payload: web::Json<Value>,
    ) -> Result<HttpResponse, Error> {
        let client_id = extract_client_id(&req);
        
        // Rate limiting check
        if !self.rate_limiter.is_allowed(&client_id) {
            warn!("Rate limit exceeded for settings update from client: {}", client_id);
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many settings update requests. Please wait before retrying.",
                "client_id": client_id,
                "retry_after": self.rate_limiter.reset_time(&client_id).as_secs()
            })));
        }

        // Request size check
        let payload_size = serde_json::to_vec(&*payload).unwrap_or_default().len();
        if payload_size > MAX_REQUEST_SIZE {
            error!("Settings update payload too large: {} bytes", payload_size);
            return Ok(HttpResponse::PayloadTooLarge().json(json!({
                "error": "payload_too_large",
                "message": format!("Payload size {} bytes exceeds limit of {} bytes", payload_size, MAX_REQUEST_SIZE),
                "max_size": MAX_REQUEST_SIZE
            })));
        }

        info!("Processing enhanced settings update from client: {} (size: {} bytes)", client_id, payload_size);

        // Comprehensive validation
        let validated_payload = match self.validation_service.validate_settings_update(&payload) {
            Ok(sanitized) => sanitized,
            Err(validation_error) => {
                warn!("Settings validation failed for client {}: {}", client_id, validation_error);
                return Ok(validation_error.to_http_response());
            }
        };

        debug!("Settings validation passed for client: {}", client_id);

        // Get current settings
        let mut app_settings = match state.settings_addr.send(GetSettings).await {
            Ok(Ok(settings)) => settings,
            Ok(Err(e)) => {
                error!("Failed to get current settings: {}", e);
                return Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "settings_retrieval_failed",
                    "message": "Failed to retrieve current settings"
                })));
            }
            Err(e) => {
                error!("Settings actor error: {}", e);
                return Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "settings_service_unavailable",
                    "message": "Settings service is temporarily unavailable"
                })));
            }
        };

        // Apply validated changes
        if let Err(e) = app_settings.merge_update(validated_payload.clone()) {
            error!("Failed to merge validated settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "settings_merge_failed",
                "message": format!("Failed to apply settings changes: {}", e)
            })));
        }

        // Save updated settings
        match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
            Ok(Ok(())) => {
                info!("Settings updated successfully for client: {}", client_id);
                
                // Propagate physics updates if needed
                self.propagate_physics_updates(&state, &app_settings, &validated_payload).await;
                
                // Return updated settings
                let response_settings = app_settings.to_camel_case_json()
                    .map_err(|e| {
                        error!("Failed to serialize updated settings: {}", e);
                        actix_web::error::ErrorInternalServerError("Serialization error")
                    })?;

                Ok(HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "Settings updated successfully",
                    "settings": response_settings,
                    "client_id": client_id,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })))
            }
            Ok(Err(e)) => {
                error!("Failed to save settings: {}", e);
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "settings_save_failed",
                    "message": format!("Failed to save settings: {}", e)
                })))
            }
            Err(e) => {
                error!("Settings actor error during save: {}", e);
                Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "settings_service_unavailable",
                    "message": "Settings service unavailable during save operation"
                })))
            }
        }
    }

    /// Enhanced get settings with validation metadata
    pub async fn get_settings_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
    ) -> Result<HttpResponse, Error> {
        let client_id = extract_client_id(&req);

        // Rate limiting (more permissive for GET requests)
        let get_rate_limiter = Arc::new(RateLimiter::new(RateLimitConfig {
            requests_per_minute: 120,
            burst_size: 20,
            ..Default::default()
        }));

        if !get_rate_limiter.is_allowed(&client_id) {
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many get settings requests"
            })));
        }

        debug!("Processing get settings request from client: {}", client_id);

        let app_settings = match state.settings_addr.send(GetSettings).await {
            Ok(Ok(settings)) => settings,
            Ok(Err(e)) => {
                error!("Failed to get settings: {}", e);
                return Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "settings_retrieval_failed",
                    "message": "Failed to retrieve settings"
                })));
            }
            Err(e) => {
                error!("Settings actor error: {}", e);
                return Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "settings_service_unavailable", 
                    "message": "Settings service unavailable"
                })));
            }
        };

        let settings_json = app_settings.to_camel_case_json()
            .map_err(|e| {
                error!("Failed to serialize settings: {}", e);
                actix_web::error::ErrorInternalServerError("Serialization error")
            })?;

        Ok(HttpResponse::Ok().json(json!({
            "status": "success",
            "settings": settings_json,
            "validation_info": {
                "input_sanitization": "enabled",
                "rate_limiting": "active",
                "schema_validation": "enforced"
            },
            "client_id": client_id,
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }

    /// Reset settings with validation
    pub async fn reset_settings_enhanced(
        &self,
        req: HttpRequest,
        state: web::Data<AppState>,
    ) -> Result<HttpResponse, Error> {
        let client_id = extract_client_id(&req);

        // Stricter rate limiting for reset operations
        let reset_rate_limiter = Arc::new(RateLimiter::new(RateLimitConfig {
            requests_per_minute: 10,
            burst_size: 2,
            ..Default::default()
        }));

        if !reset_rate_limiter.is_allowed(&client_id) {
            warn!("Rate limit exceeded for settings reset from client: {}", client_id);
            return Ok(HttpResponse::TooManyRequests().json(json!({
                "error": "rate_limit_exceeded",
                "message": "Too many reset requests. This is a destructive operation with strict limits."
            })));
        }

        info!("Processing settings reset request from client: {}", client_id);

        // Load default settings
        let default_settings = match AppFullSettings::new() {
            Ok(settings) => settings,
            Err(e) => {
                error!("Failed to load default settings: {}", e);
                return Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "default_settings_failed",
                    "message": "Failed to load default settings"
                })));
            }
        };

        // Save as current settings
        match state.settings_addr.send(UpdateSettings { settings: default_settings.clone() }).await {
            Ok(Ok(())) => {
                info!("Settings reset to defaults for client: {}", client_id);
                
                let response_settings = default_settings.to_camel_case_json()
                    .map_err(|e| {
                        error!("Failed to serialize reset settings: {}", e);
                        actix_web::error::ErrorInternalServerError("Serialization error")
                    })?;

                Ok(HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "Settings reset to defaults successfully",
                    "settings": response_settings,
                    "client_id": client_id,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })))
            }
            Ok(Err(e)) => {
                error!("Failed to reset settings: {}", e);
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "settings_reset_failed",
                    "message": format!("Failed to reset settings: {}", e)
                })))
            }
            Err(e) => {
                error!("Settings actor error during reset: {}", e);
                Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "settings_service_unavailable",
                    "message": "Settings service unavailable during reset"
                })))
            }
        }
    }

    /// Get validation statistics for settings
    pub async fn get_validation_stats(
        &self,
        req: HttpRequest,
    ) -> Result<HttpResponse, Error> {
        let client_id = extract_client_id(&req);
        debug!("Validation stats request from client: {}", client_id);

        let stats = self.rate_limiter.get_stats();

        Ok(HttpResponse::Ok().json(json!({
            "validation_service": "active",
            "rate_limiting": {
                "total_clients": stats.total_clients,
                "banned_clients": stats.banned_clients,
                "active_clients": stats.active_clients,
                "config": stats.config
            },
            "security_features": [
                "comprehensive_input_validation",
                "xss_prevention", 
                "sql_injection_prevention",
                "path_traversal_prevention",
                "malicious_content_detection",
                "rate_limiting",
                "request_size_validation"
            ],
            "endpoints_protected": [
                "/settings",
                "/settings/reset", 
                "/physics/update",
                "/physics/compute-mode",
                "/clustering/algorithm",
                "/constraints/update",
                "/stress/optimization"
            ],
            "timestamp": chrono::Utc::now().to_rfc3339()
        })))
    }

    /// Propagate physics updates to GPU actors
    async fn propagate_physics_updates(
        &self,
        state: &web::Data<AppState>,
        settings: &AppFullSettings,
        update: &Value,
    ) {
        // Check if physics was updated
        let has_physics_update = update.get("visualisation")
            .and_then(|v| v.get("graphs"))
            .map(|g| {
                g.as_object()
                    .map(|obj| obj.values().any(|graph| graph.get("physics").is_some()))
                    .unwrap_or(false)
            })
            .unwrap_or(false);

        if has_physics_update {
            info!("Propagating physics updates to GPU actors");
            
            // Propagate to both graph types
            for graph_name in &["logseq", "visionflow"] {
                let physics = settings.get_physics(graph_name);
                let sim_params = crate::models::simulation_params::SimulationParams::from(physics);
                
                if let Some(gpu_addr) = &state.gpu_compute_addr {
                    if let Err(e) = gpu_addr.send(UpdateSimulationParams { params: sim_params }).await {
                        error!("Failed to update GPU simulation params for {}: {}", graph_name, e);
                    } else {
                        debug!("GPU simulation params updated for {}", graph_name);
                    }
                }
            }
        }
    }
}

impl Default for EnhancedSettingsHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for enhanced settings routes
pub fn config_enhanced(cfg: &mut web::ServiceConfig) {
    let handler = web::Data::new(EnhancedSettingsHandler::new());
    
    cfg.app_data(handler.clone())
        .service(
            web::scope("/settings/v2")
                .route("", web::get().to(|req, state, handler: web::Data<EnhancedSettingsHandler>| {
                    handler.get_settings_enhanced(req, state)
                }))
                .route("", web::post().to(|req, state, payload, handler: web::Data<EnhancedSettingsHandler>| {
                    handler.update_settings_enhanced(req, state, payload)
                }))
                .route("/reset", web::post().to(|req, state, handler: web::Data<EnhancedSettingsHandler>| {
                    handler.reset_settings_enhanced(req, state)
                }))
                .route("/validation/stats", web::get().to(|req, handler: web::Data<EnhancedSettingsHandler>| {
                    handler.get_validation_stats(req)
                }))
        );
}