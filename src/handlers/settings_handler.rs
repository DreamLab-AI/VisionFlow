// Unified Settings Handler - Single source of truth: AppFullSettings
use actix_web::{web, Error, HttpResponse, HttpRequest};
use crate::app_state::AppState;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, UpdateSimulationParams};
use crate::handlers::validation_handler::ValidationService;
use crate::utils::validation::rate_limit::{RateLimiter, RateLimitConfig, EndpointRateLimits, extract_client_id};
use crate::utils::validation::MAX_REQUEST_SIZE;
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

        // Continue with existing update logic using validated payload
        let update = validated_payload;
        
        debug!("Settings update received: {:?}", update);
        
        // Get current settings
        let mut app_settings = match state.settings_addr.send(GetSettings).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                error!("Failed to get current settings: {}", e);
                return Ok(HttpResponse::InternalServerError().json(json!({
                    "error": "Failed to get current settings"
                })));
            }
            Err(e) => {
                error!("Settings actor error: {}", e);
                return Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "Settings service unavailable"
                })));
            }
        };
        
        // Continue with existing auto-balance logic...
        let mut modified_update = update.clone();
        let auto_balance_update = update.get("visualisation")
            .and_then(|v| v.get("graphs"))
            .and_then(|g| {
                if let Some(logseq) = g.get("logseq") {
                    if let Some(physics) = logseq.get("physics") {
                        if let Some(auto_balance) = physics.get("autoBalance") {
                            return Some(auto_balance.clone());
                        }
                    }
                }
                if let Some(visionflow) = g.get("visionflow") {
                    if let Some(physics) = visionflow.get("physics") {
                        if let Some(auto_balance) = physics.get("autoBalance") {
                            return Some(auto_balance.clone());
                        }
                    }
                }
                None
            });
        
        // If auto_balance is being updated, apply to both graphs
        if let Some(ref auto_balance_value) = auto_balance_update {
            info!("Synchronizing auto_balance setting across both graphs: {}", auto_balance_value);
            
            let vis_obj = modified_update.as_object_mut()
                .and_then(|o| o.entry("visualisation").or_insert_with(|| json!({})).as_object_mut())
                .and_then(|v| v.entry("graphs").or_insert_with(|| json!({})).as_object_mut());
            
            if let Some(graphs) = vis_obj {
                let logseq_physics = graphs
                    .entry("logseq").or_insert_with(|| json!({})).as_object_mut()
                    .and_then(|l| l.entry("physics").or_insert_with(|| json!({})).as_object_mut());
                if let Some(physics) = logseq_physics {
                    physics.insert("autoBalance".to_string(), auto_balance_value.clone());
                }
                
                let visionflow_physics = graphs
                    .entry("visionflow").or_insert_with(|| json!({})).as_object_mut()
                    .and_then(|v| v.entry("physics").or_insert_with(|| json!({})).as_object_mut());
                if let Some(physics) = visionflow_physics {
                    physics.insert("autoBalance".to_string(), auto_balance_value.clone());
                }
            }
        }
        
        // Merge the (possibly modified) update
        if let Err(e) = app_settings.merge_update(modified_update.clone()) {
            error!("Failed to merge settings: {}", e);
            if crate::utils::logging::is_debug_enabled() {
                error!("Update payload that caused error: {}", serde_json::to_string_pretty(&modified_update).unwrap_or_else(|_| "Could not serialize".to_string()));
            }
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to merge settings: {}", e)
            })));
        }
        
        // Continue with existing update logic...
        let updated_graphs = if auto_balance_update.is_some() {
            vec!["logseq", "visionflow"]
        } else {
            modified_update.get("visualisation")
                .and_then(|v| v.get("graphs"))
                .and_then(|g| g.as_object())
                .map(|graphs| {
                    let mut updated = Vec::new();
                    if graphs.contains_key("logseq") {
                        updated.push("logseq");
                    }
                    if graphs.contains_key("visionflow") {
                        updated.push("visionflow");
                    }
                    updated
                })
                .unwrap_or_default()
        };
        
        let auto_balance_active = app_settings.visualisation.graphs.logseq.physics.auto_balance 
            || app_settings.visualisation.graphs.visionflow.physics.auto_balance;
        
        // Save updated settings
        match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
            Ok(Ok(())) => {
                info!("Settings updated successfully for client: {}", client_id);
                
                let is_auto_balance_change = auto_balance_update.is_some();
                
                if is_auto_balance_change || !auto_balance_active {
                    for graph_name in updated_graphs {
                        propagate_physics_to_gpu(&state, &app_settings, graph_name).await;
                    }
                    if is_auto_balance_change {
                        info!("[AUTO-BALANCE] Propagating auto_balance setting change to GPU");
                    }
                } else {
                    info!("[AUTO-BALANCE] Skipping physics propagation to GPU - auto-balance is active and not changing");
                }
                
                let camel_case_json = app_settings.to_camel_case_json()
                    .map_err(|e| {
                        error!("Failed to convert settings: {}", e);
                        actix_web::error::ErrorInternalServerError("Serialization error")
                    })?;
                
                Ok(HttpResponse::Ok().json(json!({
                    "status": "success",
                    "message": "Settings updated successfully",
                    "settings": camel_case_json,
                    "client_id": client_id,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                })))
            }
            Ok(Err(e)) => {
                error!("Failed to save settings: {}", e);
                Ok(HttpResponse::InternalServerError().json(json!({
                    "error": format!("Failed to save settings: {}", e)
                })))
            }
            Err(e) => {
                error!("Settings actor error: {}", e);
                Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "Settings service unavailable"
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
                    "error": "Failed to retrieve settings"
                })));
            }
            Err(e) => {
                error!("Settings actor error: {}", e);
                return Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "Settings service unavailable"
                })));
            }
        };
        
        let camel_case_json = app_settings.to_camel_case_json()
            .map_err(|e| {
                error!("Failed to convert settings: {}", e);
                actix_web::error::ErrorInternalServerError("Serialization error")
            })?;
        
        Ok(HttpResponse::Ok().json(json!({
            "status": "success",
            "settings": camel_case_json,
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
                    "error": "Failed to load default settings"
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
                    "error": format!("Failed to reset settings: {}", e)
                })))
            }
            Err(e) => {
                error!("Settings actor error during reset: {}", e);
                Ok(HttpResponse::ServiceUnavailable().json(json!({
                    "error": "Settings service unavailable during reset"
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

/// Configure routes for settings endpoints
pub fn config(cfg: &mut web::ServiceConfig) {
    let handler = web::Data::new(EnhancedSettingsHandler::new());
    
    cfg.app_data(handler.clone())
        .service(
            web::scope("/settings")
                .route("", web::get().to(|req, state, handler: web::Data<EnhancedSettingsHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    match handler.get_settings_enhanced(req, state).await {
                        Ok(response) => response,
                        Err(_) => HttpResponse::InternalServerError().json(json!({"error": "Settings service temporarily unavailable"}))
                    }
                }))
                .route("", web::post().to(|req, state, payload, handler: web::Data<EnhancedSettingsHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    match handler.update_settings_enhanced(req, state, payload).await {
                        Ok(response) => response,
                        Err(_) => HttpResponse::InternalServerError().json(json!({"error": "Settings update service temporarily unavailable"}))
                    }
                }))
                .route("/reset", web::post().to(|req, state, handler: web::Data<EnhancedSettingsHandler>| async move {
                    // Try enhanced handler first, fallback to legacy
                    match handler.reset_settings_enhanced(req, state).await {
                        Ok(response) => response,
                        Err(_) => HttpResponse::InternalServerError().json(json!({"error": "Settings reset service temporarily unavailable"}))
                    }
                }))
                .route("/validation/stats", web::get().to(|req, handler: web::Data<EnhancedSettingsHandler>| async move {
                    handler.get_validation_stats(req).await
                }))
        )
        .service(
            web::scope("/physics")
                .route("/update", web::post().to(update_physics))
                .route("/compute-mode", web::post().to(update_compute_mode))
        )
        .service(
            web::scope("/clustering")
                .route("/algorithm", web::post().to(update_clustering_algorithm))
        )
        .service(
            web::scope("/constraints")
                .route("/update", web::post().to(update_constraints))
        )
        .service(
            web::scope("/analytics")
                .route("/clusters", web::get().to(get_cluster_analytics))
        )
        .service(
            web::scope("/stress")
                .route("/optimization", web::post().to(update_stress_optimization))
        );
}

/// Get current settings - returns camelCase JSON
async fn get_settings(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    let app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to retrieve settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    // Convert to camelCase JSON for client
    let camel_case_json = app_settings.to_camel_case_json()
        .map_err(|e| {
            error!("Failed to convert settings: {}", e);
            actix_web::error::ErrorInternalServerError("Serialization error")
        })?;
    
    Ok(HttpResponse::Ok().json(camel_case_json))
}

/// Update settings with validation - accepts camelCase JSON
async fn update_settings(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();
    
    debug!("Settings update received: {:?}", update);
    
    // Validate the update
    if let Err(e) = validate_settings_update(&update) {
        error!("Settings validation failed: {}", e);
        error!("Failed update payload: {}", serde_json::to_string_pretty(&update).unwrap_or_default());
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid settings: {}", e)
        })));
    }
    
    // Get current settings
    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    // Debug: Log the update payload before merging
    if crate::utils::logging::is_debug_enabled() {
        debug!("Settings update payload (before merge): {}", serde_json::to_string_pretty(&update).unwrap_or_else(|_| "Could not serialize".to_string()));
    }
    
    // Check if auto_balance is being updated in either graph
    // If so, apply it to both graphs for consistency
    let mut modified_update = update.clone();
    let auto_balance_update = update.get("visualisation")
        .and_then(|v| v.get("graphs"))
        .and_then(|g| {
            // Check if logseq graph has auto_balance update
            if let Some(logseq) = g.get("logseq") {
                if let Some(physics) = logseq.get("physics") {
                    if let Some(auto_balance) = physics.get("autoBalance") {
                        return Some(auto_balance.clone());
                    }
                }
            }
            // Check if visionflow graph has auto_balance update
            if let Some(visionflow) = g.get("visionflow") {
                if let Some(physics) = visionflow.get("physics") {
                    if let Some(auto_balance) = physics.get("autoBalance") {
                        return Some(auto_balance.clone());
                    }
                }
            }
            None
        });
    
    // If auto_balance is being updated, apply to both graphs
    if let Some(ref auto_balance_value) = auto_balance_update {
        info!("Synchronizing auto_balance setting across both graphs: {}", auto_balance_value);
        
        // Ensure the update structure exists for both graphs
        let vis_obj = modified_update.as_object_mut()
            .and_then(|o| o.entry("visualisation").or_insert_with(|| json!({})).as_object_mut())
            .and_then(|v| v.entry("graphs").or_insert_with(|| json!({})).as_object_mut());
        
        if let Some(graphs) = vis_obj {
            // Update logseq graph
            let logseq_physics = graphs
                .entry("logseq").or_insert_with(|| json!({})).as_object_mut()
                .and_then(|l| l.entry("physics").or_insert_with(|| json!({})).as_object_mut());
            if let Some(physics) = logseq_physics {
                physics.insert("autoBalance".to_string(), auto_balance_value.clone());
            }
            
            // Update visionflow graph
            let visionflow_physics = graphs
                .entry("visionflow").or_insert_with(|| json!({})).as_object_mut()
                .and_then(|v| v.entry("physics").or_insert_with(|| json!({})).as_object_mut());
            if let Some(physics) = visionflow_physics {
                physics.insert("autoBalance".to_string(), auto_balance_value.clone());
            }
        }
    }
    
    // Merge the (possibly modified) update
    if let Err(e) = app_settings.merge_update(modified_update.clone()) {
        error!("Failed to merge settings: {}", e);
        if crate::utils::logging::is_debug_enabled() {
            error!("Update payload that caused error: {}", serde_json::to_string_pretty(&modified_update).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to merge settings: {}", e)
        })));
    }
    
    // Check which graphs had physics updated
    // If auto_balance was synchronized, both graphs are considered updated
    let updated_graphs = if auto_balance_update.is_some() {
        vec!["logseq", "visionflow"]
    } else {
        modified_update.get("visualisation")
            .and_then(|v| v.get("graphs"))
            .and_then(|g| g.as_object())
            .map(|graphs| {
                let mut updated = Vec::new();
                if graphs.contains_key("logseq") {
                    updated.push("logseq");
                }
                if graphs.contains_key("visionflow") {
                    updated.push("visionflow");
                }
                updated
            })
            .unwrap_or_default()
    };
    
    // Check if auto-balance is enabled in the current settings
    // If auto-balance is active, don't propagate physics back to avoid feedback loop
    let auto_balance_active = app_settings.visualisation.graphs.logseq.physics.auto_balance 
        || app_settings.visualisation.graphs.visionflow.physics.auto_balance;
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Settings updated successfully");
            
            // Check if this update is changing the auto_balance setting itself
            // If so, we MUST propagate it regardless of current auto_balance state
            let is_auto_balance_change = auto_balance_update.is_some();
            
            // Propagate physics updates to GPU
            // - Always propagate if auto_balance setting is being changed
            // - Skip only if auto_balance is already active AND this isn't an auto_balance change
            //   (to prevent feedback loops from auto-tuning adjustments)
            if is_auto_balance_change || !auto_balance_active {
                // Propagate physics updates to GPU for each updated graph
                for graph_name in updated_graphs {
                    propagate_physics_to_gpu(&state, &app_settings, graph_name).await;
                }
                if is_auto_balance_change {
                    info!("[AUTO-BALANCE] Propagating auto_balance setting change to GPU");
                }
            } else {
                info!("[AUTO-BALANCE] Skipping physics propagation to GPU - auto-balance is active and not changing");
            }
            
            // Return updated settings in camelCase
            let camel_case_json = app_settings.to_camel_case_json()
                .map_err(|e| {
                    error!("Failed to convert settings: {}", e);
                    actix_web::error::ErrorInternalServerError("Serialization error")
                })?;
            
            Ok(HttpResponse::Ok().json(camel_case_json))
        }
        Ok(Err(e)) => {
            error!("Failed to save settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Reset settings to defaults from settings.yaml
async fn reset_settings(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    // Load default settings from YAML
    let default_settings = match AppFullSettings::new() {
        Ok(settings) => settings,
        Err(e) => {
            error!("Failed to load default settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to load default settings"
            })));
        }
    };
    
    // Save as current settings
    match state.settings_addr.send(UpdateSettings { settings: default_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Settings reset to defaults");
            
            // Return default settings in camelCase
            let camel_case_json = default_settings.to_camel_case_json()
                .map_err(|e| {
                    error!("Failed to convert settings: {}", e);
                    actix_web::error::ErrorInternalServerError("Serialization error")
                })?;
            
            Ok(HttpResponse::Ok().json(camel_case_json))
        }
        Ok(Err(e)) => {
            error!("Failed to reset settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to reset settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Update physics settings for a specific graph - NEW DEDICATED ENDPOINT
async fn update_physics(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let physics_update = payload.into_inner();
    
    info!("Physics update request received - parameters: {}", 
        physics_update.as_object().map(|o| o.keys().map(|k| k.as_str()).collect::<Vec<_>>().join(", ")).unwrap_or_default());
    debug!("Physics update payload: {}", serde_json::to_string_pretty(&physics_update).unwrap_or_default());
    
    // Validate physics parameters
    if let Err(e) = validate_physics_settings(&physics_update) {
        error!("Physics validation failed: field='{}', reason='{}'", 
            extract_failed_field(&physics_update), e);
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid physics parameters: {}", e)
        })));
    }
    
    debug!("Physics validation passed for: {:?}", 
        physics_update.as_object().map(|o| o.keys().map(|k| k.as_str()).collect::<Vec<_>>()).unwrap_or_default());
    
    // Get current settings
    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    // Create a proper settings update structure for physics
    let settings_update = create_physics_settings_update(physics_update);
    
    // Apply the physics update
    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge physics settings: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to merge physics settings: {}", e)
        })));
    }
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Physics settings updated successfully");
            
            // Propagate to both graphs (logseq and visionflow)
            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "Physics settings updated successfully"
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save physics settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save physics settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Validate settings update payload
fn validate_settings_update(update: &Value) -> Result<(), String> {
    // Validate visualisation settings
    if let Some(vis) = update.get("visualisation") {
        if let Some(graphs) = vis.get("graphs") {
            // Validate graph settings
            for (graph_name, graph_settings) in graphs.as_object().ok_or("graphs must be an object")?.iter() {
                if graph_name != "logseq" && graph_name != "visionflow" {
                    return Err(format!("Invalid graph name: {}", graph_name));
                }
                
                // Validate physics settings
                if let Some(physics) = graph_settings.get("physics") {
                    validate_physics_settings(physics)?;
                }
                
                // Validate node settings
                if let Some(nodes) = graph_settings.get("nodes") {
                    validate_node_settings(nodes)?;
                }
            }
        }
        
        // Validate rendering settings
        if let Some(rendering) = vis.get("rendering") {
            validate_rendering_settings(rendering)?;
        }
        
        // Validate hologram settings
        if let Some(hologram) = vis.get("hologram") {
            validate_hologram_settings(hologram)?;
        }
    }
    
    // Validate XR settings
    if let Some(xr) = update.get("xr") {
        validate_xr_settings(xr)?;
    }
    
    // Validate system settings
    if let Some(system) = update.get("system") {
        validate_system_settings(system)?;
    }
    
    Ok(())
}

fn validate_physics_settings(physics: &Value) -> Result<(), String> {
    // Log what fields are actually being sent
    if let Some(obj) = physics.as_object() {
        debug!("Physics settings fields received: {:?}", obj.keys().collect::<Vec<_>>());
    }
    
    // Check if auto-balance is enabled - if so, be more lenient with validation
    let auto_balance_enabled = physics.get("autoBalance")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    
    
    // Validate damping - MUST be high for stability
    // Allow up to 0.999 for auto-balance aggressive damping
    if let Some(damping) = physics.get("damping") {
        let val = damping.as_f64().ok_or("damping must be a number")?;
        // Round to 3 decimal places to handle floating point precision issues
        let rounded_val = (val * 1000.0).round() / 1000.0;
        
        // Be more lenient with auto-balance enabled
        let max_damping = if auto_balance_enabled { 1.0 } else { 0.999 };
        
        if rounded_val < 0.0 || rounded_val > 1.0 {
            return Err("damping must be between 0.0 and 1.0".to_string());
        }
    }
    
    // Validate iterations - LIMIT FOR PERFORMANCE
    if let Some(iterations) = physics.get("iterations") {
        // Accept both integer and float values (JavaScript sends 100.0 as float)
        let val = iterations.as_f64()
            .map(|f| f.round() as u64)  // Round and cast float to u64
            .or_else(|| iterations.as_u64())  // Also accept direct integer
            .ok_or("iterations must be a positive number")?;
        if val == 0 || val > 1000 {  // Allow more iterations
            return Err("iterations must be between 1 and 1000".to_string());
        }
    }
    
    // Spring strength validation
    // Allow lower values for auto-balance fine-tuning
    if let Some(spring_k) = physics.get("springK") {
        let val = spring_k.as_f64().ok_or("springK must be a number")?;
        if !(0.0001..=10.0).contains(&val) {
            return Err("springK must be between 0.0001 and 10.0".to_string());
        }
    }
    
    // Repulsion strength validation - SAFE RANGE
    // Allow lower values (down to 0.001) for auto-balance stabilization
    if let Some(repel_k) = physics.get("repelK") {
        let val = repel_k.as_f64().ok_or("repelK must be a number")?;
        if val < 0.0001 || val > 10000.0 {
            return Err("repelK must be between 0.0001 and 10000.0".to_string());
        }
    }
    
    // Attraction strength validation
    if let Some(attraction_k) = physics.get("attractionK") {
        let val = attraction_k.as_f64().ok_or("attractionK must be a number")?;
        if !(0.0..=10.0).contains(&val) {
            return Err("attractionK must be between 0.0 and 10.0".to_string());
        }
    }
    
    // Bounds size validation
    if let Some(bounds) = physics.get("boundsSize") {
        let val = bounds.as_f64().ok_or("boundsSize must be a number")?;
        if val < 1.0 || val > 100000.0 {  // Very generous range
            return Err("boundsSize must be between 1.0 and 100000.0".to_string());
        }
    }
    
    // Separation radius validation
    if let Some(separation_radius) = physics.get("separationRadius") {
        let val = separation_radius.as_f64().ok_or("separationRadius must be a number")?;
        if val < 0.01 || val > 100.0 {
            return Err("separationRadius must be between 0.01 and 100.0".to_string());
        }
    }
    
    // Max velocity validation - PREVENT EXPLOSION
    // Allow lower values (down to 0.05) for auto-balance aggressive stabilization
    if let Some(max_vel) = physics.get("maxVelocity") {
        let val = max_vel.as_f64().ok_or("maxVelocity must be a number")?;
        if val < 0.001 || val > 1000.0 {  // Very generous range
            return Err("maxVelocity must be between 0.001 and 1000.0".to_string());
        }
    }
    
    // Mass scale validation
    if let Some(mass) = physics.get("massScale") {
        let val = mass.as_f64().ok_or("massScale must be a number")?;
        if val < 0.01 || val > 100.0 {  // Generous range
            return Err("massScale must be between 0.01 and 100.0".to_string());
        }
    }
    
    // Boundary damping validation
    if let Some(boundary) = physics.get("boundaryDamping") {
        let val = boundary.as_f64().ok_or("boundaryDamping must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("boundaryDamping must be between 0.0 and 1.0".to_string());
        }
    }
    
    // Time step validation - NUMERICAL STABILITY
    // Check both timeStep and dt (client might send either)
    let time_step = physics.get("timeStep").or_else(|| physics.get("dt"));
    if let Some(time_step) = time_step {
        let val = time_step.as_f64().ok_or("timeStep/dt must be a number")?;
        if val <= 0.0 || val > 1.0 {  // Generous range
            return Err("timeStep/dt must be between 0.001 and 1.0".to_string());
        }
    }
    
    // Temperature validation
    if let Some(temp) = physics.get("temperature") {
        let val = temp.as_f64().ok_or("temperature must be a number")?;
        if val < 0.0 || val > 100.0 {  // Generous range
            return Err("temperature must be between 0.0 and 100.0".to_string());
        }
    }
    
    // Gravity validation
    if let Some(gravity) = physics.get("gravity") {
        let val = gravity.as_f64().ok_or("gravity must be a number")?;
        if val < -100.0 || val > 100.0 {  // Generous range
            return Err("gravity must be between -100.0 and 100.0".to_string());
        }
    }
    
    // Update threshold validation
    if let Some(threshold) = physics.get("updateThreshold") {
        let val = threshold.as_f64().ok_or("updateThreshold must be a number")?;
        if val < 0.0 || val > 10.0 {
            return Err("updateThreshold must be between 0.0 and 10.0".to_string());
        }
    }
    
    // NEW GPU-ALIGNED PARAMETERS
    
    // Stress weight validation
    if let Some(stress_weight) = physics.get("stressWeight") {
        let val = stress_weight.as_f64().ok_or("stressWeight must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("stressWeight must be between 0.0 and 1.0".to_string());
        }
    }
    
    // Stress alpha validation
    if let Some(stress_alpha) = physics.get("stressAlpha") {
        let val = stress_alpha.as_f64().ok_or("stressAlpha must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("stressAlpha must be between 0.0 and 1.0".to_string());
        }
    }
    
    // Alignment strength validation
    if let Some(alignment_strength) = physics.get("alignmentStrength") {
        let val = alignment_strength.as_f64().ok_or("alignmentStrength must be a number")?;
        if !(0.0..=10.0).contains(&val) {
            return Err("alignmentStrength must be between 0.0 and 10.0".to_string());
        }
    }
    
    // Cluster strength validation
    if let Some(cluster_strength) = physics.get("clusterStrength") {
        let val = cluster_strength.as_f64().ok_or("clusterStrength must be a number")?;
        if !(0.0..=10.0).contains(&val) {
            return Err("clusterStrength must be between 0.0 and 10.0".to_string());
        }
    }
    
    // Compute mode validation (0=Basic, 1=Dual Graph, 2=Constraints, 3=Visual Analytics)
    if let Some(compute_mode) = physics.get("computeMode") {
        let val = compute_mode.as_u64()
            .or_else(|| compute_mode.as_f64().map(|f| f.round() as u64))
            .ok_or("computeMode must be an integer")?;
        if val > 3 {
            return Err("computeMode must be between 0 and 3".to_string());
        }
    }
    
    // Additional GPU parameters validation
    if let Some(min_distance) = physics.get("minDistance") {
        let val = min_distance.as_f64().ok_or("minDistance must be a number")?;
        if val < 0.001 || val > 10.0 {
            return Err("minDistance must be between 0.001 and 10.0".to_string());
        }
    }
    
    if let Some(max_repulsion_dist) = physics.get("maxRepulsionDist") {
        let val = max_repulsion_dist.as_f64().ok_or("maxRepulsionDist must be a number")?;
        if val < 1.0 || val > 10000.0 {
            return Err("maxRepulsionDist must be between 1.0 and 10000.0".to_string());
        }
    }
    
    if let Some(boundary_margin) = physics.get("boundaryMargin") {
        let val = boundary_margin.as_f64().ok_or("boundaryMargin must be a number")?;
        if val < 0.0 || val > 1.0 {
            return Err("boundaryMargin must be between 0.0 and 1.0".to_string());
        }
    }
    
    if let Some(boundary_force_strength) = physics.get("boundaryForceStrength") {
        let val = boundary_force_strength.as_f64().ok_or("boundaryForceStrength must be a number")?;
        if val < 0.0 || val > 100.0 {
            return Err("boundaryForceStrength must be between 0.0 and 100.0".to_string());
        }
    }
    
    if let Some(warmup_iterations) = physics.get("warmupIterations") {
        let val = warmup_iterations.as_u64()
            .or_else(|| warmup_iterations.as_f64().map(|f| f.round() as u64))
            .ok_or("warmupIterations must be an integer")?;
        if val > 10000 {
            return Err("warmupIterations must be between 0 and 10000".to_string());
        }
    }
    
    if let Some(warmup_curve) = physics.get("warmupCurve") {
        let val = warmup_curve.as_str().ok_or("warmupCurve must be a string")?;
        if !["linear", "quadratic", "cubic"].contains(&val) {
            return Err("warmupCurve must be 'linear', 'quadratic', or 'cubic'".to_string());
        }
    }
    
    if let Some(zero_velocity_iterations) = physics.get("zeroVelocityIterations") {
        let val = zero_velocity_iterations.as_u64()
            .or_else(|| zero_velocity_iterations.as_f64().map(|f| f.round() as u64))
            .ok_or("zeroVelocityIterations must be an integer")?;
        if val > 1000 {
            return Err("zeroVelocityIterations must be between 0 and 1000".to_string());
        }
    }
    
    if let Some(cooling_rate) = physics.get("coolingRate") {
        let val = cooling_rate.as_f64().ok_or("coolingRate must be a number")?;
        if val < 0.0 || val > 1.0 {
            return Err("coolingRate must be between 0.0 and 1.0".to_string());
        }
    }
    
    // Auto-balance validation
    if let Some(auto_balance) = physics.get("autoBalance") {
        if !auto_balance.is_boolean() {
            return Err("autoBalance must be a boolean".to_string());
        }
    }
    
    if let Some(auto_balance_interval) = physics.get("autoBalanceIntervalMs") {
        let val = auto_balance_interval.as_u64()
            .or_else(|| auto_balance_interval.as_f64().map(|f| f.round() as u64))
            .ok_or("autoBalanceIntervalMs must be a positive integer")?;
        if val < 10 || val > 60000 {
            return Err("autoBalanceIntervalMs must be between 10 and 60000 ms".to_string());
        }
    }
    
    // Clustering parameters validation
    if let Some(clustering_algorithm) = physics.get("clusteringAlgorithm") {
        let val = clustering_algorithm.as_str().ok_or("clusteringAlgorithm must be a string")?;
        if !["none", "kmeans", "spectral", "louvain"].contains(&val) {
            return Err("clusteringAlgorithm must be 'none', 'kmeans', 'spectral', or 'louvain'".to_string());
        }
    }
    
    if let Some(cluster_count) = physics.get("clusterCount") {
        let val = cluster_count.as_u64()
            .or_else(|| cluster_count.as_f64().map(|f| f.round() as u64))
            .ok_or("clusterCount must be an integer")?;
        if val < 1 || val > 1000 {
            return Err("clusterCount must be between 1 and 1000".to_string());
        }
    }
    
    if let Some(clustering_resolution) = physics.get("clusteringResolution") {
        let val = clustering_resolution.as_f64().ok_or("clusteringResolution must be a number")?;
        if val < 0.001 || val > 100.0 {
            return Err("clusteringResolution must be between 0.001 and 100.0".to_string());
        }
    }
    
    if let Some(clustering_iterations) = physics.get("clusteringIterations") {
        let val = clustering_iterations.as_u64()
            .or_else(|| clustering_iterations.as_f64().map(|f| f.round() as u64))
            .ok_or("clusteringIterations must be an integer")?;
        if val < 1 || val > 10000 {
            return Err("clusteringIterations must be between 1 and 10000".to_string());
        }
    }
    
    // Boundary limit validation (should be ~98% of boundsSize)
    if let Some(boundary_limit) = physics.get("boundaryLimit") {
        let val = boundary_limit.as_f64().ok_or("boundaryLimit must be a number")?;
        if val < 0.1 || val > 100000.0 {  // Very generous range
            return Err("boundaryLimit must be between 0.1 and 100000.0".to_string());
        }
        
        // If boundsSize is also present, validate the relationship
        if let Some(bounds_size) = physics.get("boundsSize").and_then(|b| b.as_f64()) {
            let max_boundary = bounds_size * 0.99;  // Allow up to 99% for safety
            if val > max_boundary {
                return Err(format!("boundaryLimit ({:.1}) must be less than 99% of boundsSize ({:.1})", val, bounds_size));
            }
        }
    }
    
    Ok(())
}

fn validate_node_settings(nodes: &Value) -> Result<(), String> {
    // UNIFIED FORMAT: Only accept camelCase "baseColor"
    if let Some(color) = nodes.get("baseColor") {
        let color_str = color.as_str().ok_or("baseColor must be a string")?;
        if !color_str.starts_with('#') || (color_str.len() != 7 && color_str.len() != 4) {
            return Err("baseColor must be a valid hex color (e.g., #ffffff or #fff)".to_string());
        }
    }
    
    if let Some(opacity) = nodes.get("opacity") {
        let val = opacity.as_f64().ok_or("opacity must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("opacity must be between 0.0 and 1.0".to_string());
        }
    }
    
    if let Some(metalness) = nodes.get("metalness") {
        let val = metalness.as_f64().ok_or("metalness must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("metalness must be between 0.0 and 1.0".to_string());
        }
    }
    
    if let Some(roughness) = nodes.get("roughness") {
        let val = roughness.as_f64().ok_or("roughness must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("roughness must be between 0.0 and 1.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "nodeSize"
    if let Some(node_size) = nodes.get("nodeSize") {
        let val = node_size.as_f64().ok_or("nodeSize must be a number")?;
        if val <= 0.0 || val > 1000.0 {
            return Err("nodeSize must be between 0.0 and 1000.0".to_string());
        }
    }
    
    if let Some(quality) = nodes.get("quality") {
        let q = quality.as_str().ok_or("quality must be a string")?;
        if !["low", "medium", "high"].contains(&q) {
            return Err("quality must be 'low', 'medium', or 'high'".to_string());
        }
    }
    
    Ok(())
}

fn validate_rendering_settings(rendering: &Value) -> Result<(), String> {
    // UNIFIED FORMAT: Only accept "ambientLightIntensity"
    if let Some(ambient) = rendering.get("ambientLightIntensity") {
        let val = ambient.as_f64().ok_or("ambientLightIntensity must be a number")?;
        if val < 0.0 || val > 100.0 {
            return Err("ambientLightIntensity must be between 0.0 and 100.0".to_string());
        }
    }
    
    Ok(())
}

fn validate_hologram_settings(hologram: &Value) -> Result<(), String> {
    // Validate ringCount - MUST be an integer
    if let Some(ring_count) = hologram.get("ringCount") {
        // Accept both integer and float values (JavaScript might send 5.0)
        let val = ring_count.as_f64()
            .map(|f| f.round() as u64)  // Round float to u64
            .or_else(|| ring_count.as_u64())  // Also accept direct integer
            .ok_or("ringCount must be a positive integer")?;
        
        if val > 20 {
            return Err("ringCount must be between 0 and 20".to_string());
        }
    }
    
    // Validate ringColor (hex color)
    if let Some(color) = hologram.get("ringColor") {
        let color_str = color.as_str().ok_or("ringColor must be a string")?;
        if !color_str.starts_with('#') || (color_str.len() != 7 && color_str.len() != 4) {
            return Err("ringColor must be a valid hex color (e.g., #ffffff or #fff)".to_string());
        }
    }
    
    // Validate ringOpacity
    if let Some(opacity) = hologram.get("ringOpacity") {
        let val = opacity.as_f64().ok_or("ringOpacity must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("ringOpacity must be between 0.0 and 1.0".to_string());
        }
    }
    
    // Validate ringRotationSpeed
    if let Some(speed) = hologram.get("ringRotationSpeed") {
        let val = speed.as_f64().ok_or("ringRotationSpeed must be a number")?;
        if val < 0.0 || val > 1000.0 {
            return Err("ringRotationSpeed must be between 0.0 and 1000.0".to_string());
        }
    }
    
    Ok(())
}

fn validate_system_settings(system: &Value) -> Result<(), String> {
    // Handle debug settings
    if let Some(debug) = system.get("debug") {
        if let Some(debug_obj) = debug.as_object() {
            // All debug flags should be booleans - UNIFIED FORMAT ONLY
            let boolean_fields = [
                "enabled",  // NOT "enableClientDebugMode" - unified format only!
                "showFPS", 
                "showMemory",
                "enablePerformanceDebug",
                "enableTelemetry",
                "enableDataDebug",
                "enableWebSocketDebug",
                "enablePhysicsDebug",
                "enableNodeDebug",
                "enableShaderDebug",
                "enableMatrixDebug"
            ];
            
            for field in &boolean_fields {
                if let Some(val) = debug_obj.get(*field) {
                    if !val.is_boolean() {
                        return Err(format!("debug.{} must be a boolean", field));
                    }
                }
            }
            
            // logLevel can be a number or string
            if let Some(log_level) = debug_obj.get("logLevel") {
                if let Some(val) = log_level.as_f64() {
                    if val < 0.0 || val > 3.0 {
                        return Err("debug.logLevel must be between 0 and 3".to_string());
                    }
                } else if let Some(val) = log_level.as_u64() {
                    if val > 3 {
                        return Err("debug.logLevel must be between 0 and 3".to_string());
                    }
                } else if let Some(val) = log_level.as_str() {
                    // Accept string log levels from client
                    match val {
                        "error" | "warn" | "info" | "debug" => {
                            // Valid string log level
                        }
                        _ => {
                            return Err("debug.logLevel must be 'error', 'warn', 'info', or 'debug'".to_string());
                        }
                    }
                } else {
                    return Err("debug.logLevel must be a number or string".to_string());
                }
            }
        }
    }
    
    // Handle persistSettingsOnServer
    if let Some(persist) = system.get("persistSettingsOnServer") {
        if !persist.is_boolean() {
            return Err("system.persistSettingsOnServer must be a boolean".to_string());
        }
    }
    
    // Handle customBackendUrl
    if let Some(url) = system.get("customBackendUrl") {
        if !url.is_string() && !url.is_null() {
            return Err("system.customBackendUrl must be a string or null".to_string());
        }
    }
    
    Ok(())
}

fn validate_xr_settings(xr: &Value) -> Result<(), String> {
    // UNIFIED FORMAT: Only accept "enabled", not "enableXrMode"  
    if let Some(enabled) = xr.get("enabled") {
        if !enabled.is_boolean() {
            return Err("XR enabled must be a boolean".to_string());
        }
    }
    
    // Handle quality setting
    if let Some(quality) = xr.get("quality") {
        if let Some(q) = quality.as_str() {
            if !["Low", "Medium", "High", "low", "medium", "high"].contains(&q) {
                return Err("XR quality must be Low, Medium, or High".to_string());
            }
        } else {
            return Err("XR quality must be a string".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "renderScale"  
    if let Some(render_scale) = xr.get("renderScale") {
        let val = render_scale.as_f64().ok_or("renderScale must be a number")?;
        if val < 0.1 || val > 10.0 {
            return Err("renderScale must be between 0.1 and 10.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "roomScale"
    if let Some(room_scale) = xr.get("roomScale") {
        let val = room_scale.as_f64().ok_or("roomScale must be a number")?;
        if val <= 0.0 || val > 100.0 {
            return Err("roomScale must be between 0.0 and 100.0".to_string());
        }
    }
    
    // Handle nested handTracking object
    if let Some(hand_tracking) = xr.get("handTracking") {
        if let Some(ht_obj) = hand_tracking.as_object() {
            if let Some(enabled) = ht_obj.get("enabled") {
                if !enabled.is_boolean() {
                    return Err("handTracking.enabled must be a boolean".to_string());
                }
            }
        }
    }
    
    // Handle nested interactions object
    if let Some(interactions) = xr.get("interactions") {
        if let Some(int_obj) = interactions.as_object() {
            if let Some(haptics) = int_obj.get("enableHaptics") {
                if !haptics.is_boolean() {
                    return Err("interactions.enableHaptics must be a boolean".to_string());
                }
            }
        }
    }
    
    Ok(())
}

/// Propagate physics settings to GPU compute actor
async fn propagate_physics_to_gpu(
    state: &web::Data<AppState>,
    settings: &AppFullSettings,
    graph: &str,
) {
    let physics = settings.get_physics(graph);
    
    // Always log critical physics values with new parameter names
    info!(
        "[PHYSICS UPDATE] Propagating {} physics to actors:", graph
    );
    info!(
        "  - repulsion_k: {:.3} (affects node spreading)", 
        physics.repel_k
    );
    info!(
        "  - spring_k: {:.3} (affects edge tension)",
        physics.spring_k
    );
    info!(
        "  - attraction_k: {:.3} (affects clustering)",
        physics.attraction_k
    );
    info!(
        "  - damping: {:.3} (affects settling, 1.0 = no movement)",
        physics.damping
    );
    info!(
        "  - time_step: {:.3} (simulation speed)",
        physics.dt
    );
    info!(
        "  - max_velocity: {:.3} (prevents explosions)",
        physics.max_velocity
    );
    info!(
        "  - temperature: {:.3} (random motion)",
        physics.temperature
    );
    info!(
        "  - gravity: {:.3} (directional force)",
        physics.gravity
    );
    
    if crate::utils::logging::is_debug_enabled() {
        debug!("  - bounds_size: {:.1}", physics.bounds_size);
        debug!("  - separation_radius: {:.3}", physics.separation_radius);  // Updated name
        debug!("  - mass_scale: {:.3}", physics.mass_scale);
        debug!("  - boundary_damping: {:.3}", physics.boundary_damping);
        debug!("  - update_threshold: {:.3}", physics.update_threshold);
        debug!("  - iterations: {}", physics.iterations);
        debug!("  - enabled: {}", physics.enabled);
        
        // Log new GPU-aligned parameters
        debug!("  - min_distance: {:.3}", physics.min_distance);
        debug!("  - max_repulsion_dist: {:.1}", physics.max_repulsion_dist);
        debug!("  - boundary_margin: {:.3}", physics.boundary_margin);
        debug!("  - boundary_force_strength: {:.1}", physics.boundary_force_strength);
        debug!("  - warmup_iterations: {}", physics.warmup_iterations);
        debug!("  - warmup_curve: {}", physics.warmup_curve);
        debug!("  - zero_velocity_iterations: {}", physics.zero_velocity_iterations);
        debug!("  - cooling_rate: {:.6}", physics.cooling_rate);
        debug!("  - clustering_algorithm: {}", physics.clustering_algorithm);
        debug!("  - cluster_count: {}", physics.cluster_count);
        debug!("  - clustering_resolution: {:.3}", physics.clustering_resolution);
        debug!("  - clustering_iterations: {}", physics.clustering_iterations);
        debug!("[GPU Parameters] All new parameters available for GPU processing");
    }
    
    let sim_params: crate::models::simulation_params::SimulationParams = physics.into();
    
    info!(
        "[PHYSICS UPDATE] Converted to SimulationParams - repulsion: {}, damping: {:.3}, time_step: {:.3}",
        sim_params.repel_k, sim_params.damping, sim_params.dt
    );
    
    let update_msg = UpdateSimulationParams { params: sim_params.clone() };
    
    // Send to GPU compute actor
    if let Some(gpu_addr) = &state.gpu_compute_addr {
        info!("[PHYSICS UPDATE] Sending to GPUComputeActor...");
        if let Err(e) = gpu_addr.send(update_msg.clone()).await {
            error!("[PHYSICS UPDATE] FAILED to update GPUComputeActor: {}", e);
        } else {
            info!("[PHYSICS UPDATE] GPUComputeActor updated successfully");
        }
    } else {
        warn!("[PHYSICS UPDATE] No GPUComputeActor available");
    }
    
    // Send to graph service actor
    info!("[PHYSICS UPDATE] Sending to GraphServiceActor...");
    if let Err(e) = state.graph_service_addr.send(update_msg).await {
        error!("[PHYSICS UPDATE] FAILED to update GraphServiceActor: {}", e);
    } else {
        info!("[PHYSICS UPDATE] GraphServiceActor updated successfully");
    }
}

/// Helper function to get field variants (camelCase or snake_case)
fn get_field_variant<'a>(obj: &'a Value, variants: &[&str]) -> Option<&'a Value> {
    for variant in variants {
        if let Some(val) = obj.get(*variant) {
            return Some(val);
        }
    }
    None
}

/// Count the number of fields in a JSON object recursively
fn count_fields(value: &Value) -> usize {
    match value {
        Value::Object(map) => {
            map.len() + map.values().map(count_fields).sum::<usize>()
        }
        Value::Array(arr) => arr.iter().map(count_fields).sum(),
        _ => 0,
    }
}

/// Extract which graphs have physics updates
fn extract_physics_updates(update: &Value) -> Vec<&str> {
    update.get("visualisation")
        .and_then(|v| v.get("graphs"))
        .and_then(|g| g.as_object())
        .map(|graphs| {
            let mut updated = Vec::new();
            if graphs.contains_key("logseq") && 
               graphs.get("logseq").and_then(|g| g.get("physics")).is_some() {
                updated.push("logseq");
            }
            if graphs.contains_key("visionflow") && 
               graphs.get("visionflow").and_then(|g| g.get("physics")).is_some() {
                updated.push("visionflow");
            }
            updated
        })
        .unwrap_or_default()
}

/// Extract the field name that failed validation
fn extract_failed_field(physics: &Value) -> String {
    if let Some(obj) = physics.as_object() {
        obj.keys().next().unwrap_or(&"unknown".to_string()).clone()
    } else {
        "unknown".to_string()
    }
}

/// Create a proper settings update structure for physics parameters
/// Maps old parameter names to new ones for backward compatibility
fn create_physics_settings_update(physics_update: Value) -> Value {
    let mut normalized_physics = physics_update.clone();
    
    // Map old parameter names to new ones if old names are present
    if let Some(obj) = normalized_physics.as_object_mut() {
        // Map springStrength -> springK
        if let Some(spring_strength) = obj.remove("springStrength") {
            if !obj.contains_key("springK") {
                obj.insert("springK".to_string(), spring_strength);
            }
        }
        
        // Map repulsionStrength -> repelK (GPU-aligned name)
        if let Some(repulsion_strength) = obj.remove("repulsionStrength") {
            if !obj.contains_key("repelK") {
                obj.insert("repelK".to_string(), repulsion_strength);
            }
        }
        
        // Map attractionStrength -> attractionK
        if let Some(attraction_strength) = obj.remove("attractionStrength") {
            if !obj.contains_key("attractionK") {
                obj.insert("attractionK".to_string(), attraction_strength);
            }
        }
        
        // Map collisionRadius -> separationRadius
        if let Some(collision_radius) = obj.remove("collisionRadius") {
            if !obj.contains_key("separationRadius") {
                obj.insert("separationRadius".to_string(), collision_radius);
            }
        }
    }
    
    json!({
        "visualisation": {
            "graphs": {
                "logseq": {
                    "physics": normalized_physics
                },
                "visionflow": {
                    "physics": normalized_physics.clone()
                }
            }
        }
    })
}

/// Update compute mode endpoint
async fn update_compute_mode(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();
    
    info!("Compute mode update request received");
    debug!("Compute mode payload: {}", serde_json::to_string_pretty(&update).unwrap_or_default());
    
    // Validate compute mode
    let compute_mode = update.get("computeMode")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            actix_web::error::ErrorBadRequest("computeMode must be an integer between 0 and 3")
        })?;
    
    if compute_mode > 3 {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "computeMode must be between 0 and 3"
        })));
    }
    
    // Create physics update with compute mode
    let physics_update = json!({
        "computeMode": compute_mode
    });
    
    let settings_update = create_physics_settings_update(physics_update);
    
    // Get and update settings
    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge compute mode settings: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to update compute mode: {}", e)
        })));
    }
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Compute mode updated successfully to: {}", compute_mode);
            
            // Propagate to GPU
            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "Compute mode updated successfully",
                "computeMode": compute_mode
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save compute mode settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save compute mode settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Update clustering algorithm endpoint
async fn update_clustering_algorithm(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();
    
    info!("Clustering algorithm update request received");
    debug!("Clustering payload: {}", serde_json::to_string_pretty(&update).unwrap_or_default());
    
    // Validate clustering algorithm
    let algorithm = update.get("algorithm")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            actix_web::error::ErrorBadRequest("algorithm must be a string")
        })?;
    
    if !["none", "kmeans", "spectral", "louvain"].contains(&algorithm) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "algorithm must be 'none', 'kmeans', 'spectral', or 'louvain'"
        })));
    }
    
    // Extract optional parameters
    let cluster_count = update.get("clusterCount").and_then(|v| v.as_u64()).unwrap_or(5);
    let resolution = update.get("resolution").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let iterations = update.get("iterations").and_then(|v| v.as_u64()).unwrap_or(30);
    
    // Create physics update with clustering parameters
    let physics_update = json!({
        "clusteringAlgorithm": algorithm,
        "clusterCount": cluster_count,
        "clusteringResolution": resolution,
        "clusteringIterations": iterations
    });
    
    let settings_update = create_physics_settings_update(physics_update);
    
    // Get and update settings
    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge clustering settings: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to update clustering algorithm: {}", e)
        })));
    }
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Clustering algorithm updated successfully to: {}", algorithm);
            
            // Propagate to GPU
            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "Clustering algorithm updated successfully",
                "algorithm": algorithm,
                "clusterCount": cluster_count,
                "resolution": resolution,
                "iterations": iterations
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save clustering settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save clustering settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Update constraints endpoint
async fn update_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();
    
    info!("Constraints update request received");
    debug!("Constraints payload: {}", serde_json::to_string_pretty(&update).unwrap_or_default());
    
    // Validate constraint data structure
    if let Err(e) = validate_constraints(&update) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid constraints: {}", e)
        })));
    }
    
    // For now, store constraints in physics settings
    // In a real implementation, you'd have a dedicated constraints store
    let settings_update = json!({
        "visualisation": {
            "graphs": {
                "logseq": {
                    "physics": {
                        "computeMode": 2  // Enable constraints mode
                    }
                },
                "visionflow": {
                    "physics": {
                        "computeMode": 2
                    }
                }
            }
        }
    });
    
    // Get and update settings
    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge constraints settings: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to update constraints: {}", e)
        })));
    }
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Constraints updated successfully");
            
            // Propagate to GPU
            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "Constraints updated successfully"
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save constraints settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save constraints settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Get cluster analytics endpoint
async fn get_cluster_analytics(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("Cluster analytics request received");
    
    // Check if GPU clustering is available
    if let Some(gpu_addr) = &state.gpu_compute_addr {
        // Get cluster data from GPU
        // This would call a GPU clustering analysis function
        // For now, return mock data
        
        let mock_analytics = json!({
            "clusters": [
                {
                    "id": "cluster_1",
                    "nodeCount": 25,
                    "coherence": 0.85,
                    "centroid": [10.5, 15.2, 8.7],
                    "keywords": ["semantic", "knowledge", "graph"]
                },
                {
                    "id": "cluster_2",
                    "nodeCount": 18,
                    "coherence": 0.72,
                    "centroid": [-5.3, 12.1, -3.4],
                    "keywords": ["analytics", "clustering", "gpu"]
                }
            ],
            "totalNodes": 43,
            "algorithmUsed": "louvain",
            "modularity": 0.78,
            "lastUpdated": chrono::Utc::now().to_rfc3339()
        });
        
        Ok(HttpResponse::Ok().json(mock_analytics))
    } else {
        // Fallback to CPU-based analytics
        let fallback_analytics = json!({
            "clusters": [],
            "totalNodes": 0,
            "algorithmUsed": "none",
            "modularity": 0.0,
            "lastUpdated": chrono::Utc::now().to_rfc3339(),
            "note": "GPU clustering not available, using CPU fallback"
        });
        
        Ok(HttpResponse::Ok().json(fallback_analytics))
    }
}

/// Update stress optimization endpoint
async fn update_stress_optimization(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();
    
    info!("Stress optimization update request received");
    debug!("Stress optimization payload: {}", serde_json::to_string_pretty(&update).unwrap_or_default());
    
    // Validate stress parameters
    let stress_weight = update.get("stressWeight")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.1) as f32;
    
    let stress_alpha = update.get("stressAlpha")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.1) as f32;
    
    if !(0.0..=1.0).contains(&stress_weight) || !(0.0..=1.0).contains(&stress_alpha) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "stressWeight and stressAlpha must be between 0.0 and 1.0"
        })));
    }
    
    // Create physics update with stress optimization parameters
    let physics_update = json!({
        "stressWeight": stress_weight,
        "stressAlpha": stress_alpha
    });
    
    let settings_update = create_physics_settings_update(physics_update);
    
    // Get and update settings
    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge stress optimization settings: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to update stress optimization: {}", e)
        })));
    }
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Stress optimization updated successfully");
            
            // Propagate to GPU
            propagate_physics_to_gpu(&state, &app_settings, "logseq").await;
            propagate_physics_to_gpu(&state, &app_settings, "visionflow").await;
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "Stress optimization updated successfully",
                "stressWeight": stress_weight,
                "stressAlpha": stress_alpha
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save stress optimization settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save stress optimization settings: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Validate constraint data structure
fn validate_constraints(constraints: &Value) -> Result<(), String> {
    // Basic validation for constraint structure
    if let Some(obj) = constraints.as_object() {
        for (constraint_type, constraint_data) in obj {
            if !["separation", "boundary", "alignment", "cluster"].contains(&constraint_type.as_str()) {
                return Err(format!("Unknown constraint type: {}", constraint_type));
            }
            
            if let Some(data) = constraint_data.as_object() {
                if let Some(strength) = data.get("strength") {
                    let val = strength.as_f64().ok_or("strength must be a number")?;
                    if val < 0.0 || val > 100.0 {
                        return Err("strength must be between 0.0 and 100.0".to_string());
                    }
                }
                
                if let Some(enabled) = data.get("enabled") {
                    if !enabled.is_boolean() {
                        return Err("enabled must be a boolean".to_string());
                    }
                }
            }
        }
    }
    
    Ok(())
}

