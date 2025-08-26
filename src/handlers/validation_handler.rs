use actix_web::{web, HttpResponse, HttpRequest, Result};
use serde_json::Value;
use crate::utils::validation::{ValidationContext, ValidationResult};
use crate::utils::validation::schemas::{ApiSchemas, ValidationSchema};
use crate::utils::validation::sanitization::Sanitizer;
use crate::utils::validation::errors::DetailedValidationError;
use crate::utils::validation::rate_limit::extract_client_id;
use log::{info, warn, debug};

/// Comprehensive validation service for all API endpoints
pub struct ValidationService {
    settings_schema: ValidationSchema,
    physics_schema: ValidationSchema,
    ragflow_schema: ValidationSchema,
    bots_schema: ValidationSchema,
    swarm_schema: ValidationSchema,
}

impl ValidationService {
    pub fn new() -> Self {
        Self {
            settings_schema: ApiSchemas::settings_update(),
            physics_schema: ApiSchemas::physics_params(),
            ragflow_schema: ApiSchemas::ragflow_chat(),
            bots_schema: ApiSchemas::bots_data(),
            swarm_schema: ApiSchemas::swarm_init(),
        }
    }

    /// Validate settings update payload
    pub fn validate_settings_update(&self, payload: &Value) -> ValidationResult<Value> {
        let mut ctx = ValidationContext::new();
        let mut sanitized_payload = payload.clone();
        
        // Sanitize first
        Sanitizer::sanitize_json(&mut sanitized_payload)?;
        
        // Validate against schema
        self.settings_schema.validate(&sanitized_payload, &mut ctx)?;
        
        // Additional custom validation for settings
        self.validate_settings_custom(&sanitized_payload)?;
        
        Ok(sanitized_payload)
    }

    /// Validate physics parameters
    pub fn validate_physics_params(&self, payload: &Value) -> ValidationResult<Value> {
        let mut ctx = ValidationContext::new();
        let mut sanitized_payload = payload.clone();
        
        // Sanitize first
        Sanitizer::sanitize_json(&mut sanitized_payload)?;
        
        // Validate against schema
        self.physics_schema.validate(&sanitized_payload, &mut ctx)?;
        
        // Additional physics validation
        self.validate_physics_custom(&sanitized_payload)?;
        
        Ok(sanitized_payload)
    }

    /// Validate RAGFlow chat request
    pub fn validate_ragflow_chat(&self, payload: &Value) -> ValidationResult<Value> {
        let mut ctx = ValidationContext::new();
        let mut sanitized_payload = payload.clone();
        
        // Sanitize first (especially important for chat messages)
        Sanitizer::sanitize_json(&mut sanitized_payload)?;
        
        // Validate against schema
        self.ragflow_schema.validate(&sanitized_payload, &mut ctx)?;
        
        Ok(sanitized_payload)
    }

    /// Validate bots data update
    pub fn validate_bots_data(&self, payload: &Value) -> ValidationResult<Value> {
        let mut ctx = ValidationContext::new();
        let mut sanitized_payload = payload.clone();
        
        // Sanitize first
        Sanitizer::sanitize_json(&mut sanitized_payload)?;
        
        // Validate against schema
        self.bots_schema.validate(&sanitized_payload, &mut ctx)?;
        
        Ok(sanitized_payload)
    }

    /// Validate swarm initialization
    pub fn validate_swarm_init(&self, payload: &Value) -> ValidationResult<Value> {
        let mut ctx = ValidationContext::new();
        let mut sanitized_payload = payload.clone();
        
        // Sanitize first
        Sanitizer::sanitize_json(&mut sanitized_payload)?;
        
        // Validate against schema
        self.swarm_schema.validate(&sanitized_payload, &mut ctx)?;
        
        Ok(sanitized_payload)
    }

    /// Custom validation for settings that goes beyond schema validation
    fn validate_settings_custom(&self, payload: &Value) -> ValidationResult<()> {
        // Validate nested relationships
        if let Some(vis) = payload.get("visualisation") {
            if let Some(graphs) = vis.get("graphs") {
                self.validate_graph_consistency(graphs)?;
            }
            
            // Validate rendering settings including bloom/glow field mapping
            if let Some(rendering) = vis.get("rendering") {
                self.validate_rendering_settings_custom(rendering)?;
            }
        }
        
        // Validate XR settings compatibility
        if let Some(xr) = payload.get("xr") {
            self.validate_xr_compatibility(xr)?;
        }
        
        Ok(())
    }

    /// Custom validation for physics parameters
    fn validate_physics_custom(&self, payload: &Value) -> ValidationResult<()> {
        // Validate parameter relationships
        if let Some(damping) = payload.get("damping").and_then(|v| v.as_f64()) {
            if let Some(max_velocity) = payload.get("maxVelocity").and_then(|v| v.as_f64()) {
                // High damping with high max velocity might cause instability
                if damping < 0.5 && max_velocity > 100.0 {
                    return Err(DetailedValidationError::new(
                        "physics.parameters",
                        "Low damping with high max velocity may cause instability",
                        "UNSTABLE_PARAMETERS"
                    ));
                }
            }
        }

        // Validate spring constants
        if let Some(spring_k) = payload.get("springK").and_then(|v| v.as_f64()) {
            if let Some(repel_k) = payload.get("repelK").and_then(|v| v.as_f64()) {
                // Spring force shouldn't be much stronger than repulsion
                if spring_k > repel_k * 10.0 {
                    return Err(DetailedValidationError::new(
                        "physics.forces",
                        "Spring force significantly stronger than repulsion may cause clustering issues",
                        "FORCE_IMBALANCE"
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate graph consistency across logseq and visionflow
    fn validate_graph_consistency(&self, graphs: &Value) -> ValidationResult<()> {
        let graphs_obj = graphs.as_object()
            .ok_or_else(|| DetailedValidationError::new("visualisation.graphs", "Must be an object", "INVALID_TYPE"))?;

        // Check for required graphs
        if !graphs_obj.contains_key("logseq") && !graphs_obj.contains_key("visionflow") {
            return Err(DetailedValidationError::new(
                "visualisation.graphs",
                "At least one graph (logseq or visionflow) must be specified",
                "MISSING_GRAPHS"
            ));
        }

        // Validate physics consistency between graphs
        if let (Some(logseq), Some(visionflow)) = (graphs_obj.get("logseq"), graphs_obj.get("visionflow")) {
            if let (Some(logseq_physics), Some(visionflow_physics)) = 
                (logseq.get("physics"), visionflow.get("physics")) {
                self.validate_physics_consistency(logseq_physics, visionflow_physics)?;
            }
        }

        Ok(())
    }

    /// Validate physics consistency between graphs
    fn validate_physics_consistency(&self, physics1: &Value, physics2: &Value) -> ValidationResult<()> {
        // Check if auto-balance settings are consistent
        let auto_balance1 = physics1.get("autoBalance").and_then(|v| v.as_bool()).unwrap_or(false);
        let auto_balance2 = physics2.get("autoBalance").and_then(|v| v.as_bool()).unwrap_or(false);

        if auto_balance1 != auto_balance2 {
            return Err(DetailedValidationError::new(
                "physics.autoBalance",
                "Auto-balance setting should be consistent across graphs",
                "INCONSISTENT_AUTO_BALANCE"
            ));
        }

        Ok(())
    }

    /// Validate XR compatibility
    fn validate_xr_compatibility(&self, xr: &Value) -> ValidationResult<()> {
        if let Some(enabled) = xr.get("enabled").and_then(|v| v.as_bool()) {
            if enabled {
                // Check for VR-compatible settings
                if let Some(render_scale) = xr.get("renderScale").and_then(|v| v.as_f64()) {
                    if render_scale > 2.0 {
                        return Err(DetailedValidationError::new(
                            "xr.renderScale",
                            "Render scale above 2.0 may cause performance issues in VR",
                            "PERFORMANCE_WARNING"
                        ));
                    }
                }

                // Check quality settings
                if let Some(quality) = xr.get("quality").and_then(|v| v.as_str()) {
                    if quality == "high" {
                        info!("High quality XR mode enabled - ensure adequate GPU performance");
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate rendering settings with bloom/glow field mapping support
    fn validate_rendering_settings_custom(&self, rendering: &Value) -> ValidationResult<()> {
        // Handle bloom/glow field mapping - frontend sends 'bloom', backend uses 'glow'
        let bloom_glow_field = rendering.get("bloom").or_else(|| rendering.get("glow"));
        if let Some(bloom_glow) = bloom_glow_field {
            self.validate_bloom_glow_effects(bloom_glow)?;
        }
        
        Ok(())
    }

    /// Validate bloom/glow effect settings
    fn validate_bloom_glow_effects(&self, bloom_glow: &Value) -> ValidationResult<()> {
        // Validate enabled flag
        if let Some(enabled) = bloom_glow.get("enabled") {
            if !enabled.is_boolean() {
                return Err(DetailedValidationError::new(
                    "rendering.bloom.enabled",
                    "Bloom/glow enabled must be a boolean",
                    "INVALID_TYPE"
                ));
            }
        }
        
        // Validate intensity/strength with generous range for effects
        for field_name in ["intensity", "strength"] {
            if let Some(intensity) = bloom_glow.get(field_name) {
                if let Some(val) = intensity.as_f64() {
                    if val < 0.0 || val > 10.0 {
                        return Err(DetailedValidationError::out_of_range(
                            &format!("rendering.bloom.{}", field_name),
                            val,
                            0.0,
                            10.0
                        ));
                    }
                } else {
                    return Err(DetailedValidationError::new(
                        &format!("rendering.bloom.{}", field_name),
                        "Must be a number",
                        "INVALID_TYPE"
                    ));
                }
            }
        }
        
        // Validate radius field
        if let Some(radius) = bloom_glow.get("radius") {
            if let Some(val) = radius.as_f64() {
                if val < 0.0 || val > 5.0 {
                    return Err(DetailedValidationError::out_of_range(
                        "rendering.bloom.radius",
                        val,
                        0.0,
                        5.0
                    ));
                }
            } else {
                return Err(DetailedValidationError::new(
                    "rendering.bloom.radius",
                    "Must be a number",
                    "INVALID_TYPE"
                ));
            }
        }
        
        // Validate threshold field
        if let Some(threshold) = bloom_glow.get("threshold") {
            if let Some(val) = threshold.as_f64() {
                if val < 0.0 || val > 2.0 {
                    return Err(DetailedValidationError::out_of_range(
                        "rendering.bloom.threshold",
                        val,
                        0.0,
                        2.0
                    ));
                }
            } else {
                return Err(DetailedValidationError::new(
                    "rendering.bloom.threshold",
                    "Must be a number",
                    "INVALID_TYPE"
                ));
            }
        }
        
        // Validate specific bloom strength fields (0.0 to 1.0 range)
        for field_name in ["edgeBloomStrength", "environmentBloomStrength", "nodeBloomStrength"] {
            if let Some(strength) = bloom_glow.get(field_name) {
                if let Some(val) = strength.as_f64() {
                    if val < 0.0 || val > 1.0 {
                        return Err(DetailedValidationError::out_of_range(
                            &format!("rendering.bloom.{}", field_name),
                            val,
                            0.0,
                            1.0
                        ));
                    }
                } else {
                    return Err(DetailedValidationError::new(
                        &format!("rendering.bloom.{}", field_name),
                        "Must be a number",
                        "INVALID_TYPE"
                    ));
                }
            }
        }
        
        Ok(())
    }
}

impl Default for ValidationService {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation endpoint for testing validation rules
pub async fn validate_payload(
    req: HttpRequest,
    payload: web::Json<Value>,
    validation_service: web::Data<ValidationService>,
) -> Result<HttpResponse> {
    let client_id = extract_client_id(&req);
    info!("Validation test request from client: {}", client_id);

    // Get validation type from query parameters
    let validation_type = req.match_info().get("type").unwrap_or("settings");

    let result = match validation_type {
        "settings" => validation_service.validate_settings_update(&payload),
        "physics" => validation_service.validate_physics_params(&payload),
        "ragflow" => validation_service.validate_ragflow_chat(&payload),
        "bots" => validation_service.validate_bots_data(&payload),
        "swarm" => validation_service.validate_swarm_init(&payload),
        _ => {
            return Ok(HttpResponse::BadRequest().json(serde_json::json!({
                "error": "invalid_validation_type",
                "message": "Supported types: settings, physics, ragflow, bots, swarm"
            })));
        }
    };

    match result {
        Ok(sanitized_payload) => {
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "status": "valid",
                "message": "Payload validation successful",
                "sanitized_payload": sanitized_payload,
                "validation_type": validation_type
            })))
        }
        Err(error) => {
            warn!("Validation failed for {}: {}", validation_type, error);
            Ok(error.to_http_response())
        }
    }
}

/// Get validation statistics
pub async fn get_validation_stats(req: HttpRequest) -> Result<HttpResponse> {
    let client_id = extract_client_id(&req);
    debug!("Validation stats request from client: {}", client_id);

    let stats = serde_json::json!({
        "validation_service": "active",
        "supported_endpoints": [
            "settings",
            "physics", 
            "ragflow",
            "bots",
            "swarm"
        ],
        "security_features": [
            "input_sanitization",
            "schema_validation", 
            "rate_limiting",
            "xss_prevention",
            "sql_injection_prevention",
            "path_traversal_prevention"
        ],
        "timestamp": chrono::Utc::now().to_rfc3339()
    });

    Ok(HttpResponse::Ok().json(stats))
}

/// Validation service configuration
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/validation")
            .route("/test/{type}", web::post().to(validate_payload))
            .route("/stats", web::get().to(get_validation_stats))
    );
}