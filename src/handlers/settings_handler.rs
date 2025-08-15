// Unified Settings Handler - Single source of truth: AppFullSettings
use actix_web::{web, Error, HttpResponse, HttpRequest};
use crate::app_state::AppState;
use crate::config::AppFullSettings;
use crate::actors::messages::{GetSettings, UpdateSettings, UpdateSimulationParams};
use log::{info, warn, error, debug};
use serde_json::{json, Value};

/// Configure routes for settings endpoints
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/settings")
            .route("", web::get().to(get_settings))
            .route("", web::post().to(update_settings))
            .route("/reset", web::post().to(reset_settings))
    )
    .service(
        web::scope("/physics")
            .route("/update", web::post().to(update_physics))
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
    
    // Merge the update
    if let Err(e) = app_settings.merge_update(update.clone()) {
        error!("Failed to merge settings: {}", e);
        if crate::utils::logging::is_debug_enabled() {
            error!("Update payload that caused error: {}", serde_json::to_string_pretty(&update).unwrap_or_else(|_| "Could not serialize".to_string()));
        }
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to merge settings: {}", e)
        })));
    }
    
    // Check which graphs had physics updated
    let updated_graphs = update.get("visualisation")
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
        .unwrap_or_default();
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Settings updated successfully");
            
            // Propagate physics updates to GPU for each updated graph
            for graph_name in updated_graphs {
                propagate_physics_to_gpu(&state, &app_settings, graph_name).await;
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
    
    // Validate all physics fields with proper ranges
    // NOTE: Client sends both formats - handle all variations
    if let Some(damping) = physics.get("damping") {
        let val = damping.as_f64().ok_or("damping must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("damping must be between 0.0 and 1.0".to_string());
        }
    }
    
    if let Some(iterations) = physics.get("iterations") {
        let val = iterations.as_u64().ok_or("iterations must be a positive integer")?;
        if val == 0 || val > 1000 {
            return Err("iterations must be between 1 and 1000".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "springStrength"
    if let Some(spring) = physics.get("springStrength") {
        let val = spring.as_f64().ok_or("spring must be a number")?;
        if !(0.0..=10.0).contains(&val) {
            return Err("spring must be between 0.0 and 10.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "repulsionStrength"
    if let Some(repulsion) = physics.get("repulsionStrength") {
        let val = repulsion.as_f64().ok_or("repulsion must be a number")?;
        if val < 0.0 || val > 10000.0 {
            return Err("repulsion must be between 0.0 and 10000.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "attractionStrength"
    if let Some(attraction) = physics.get("attractionStrength") {
        let val = attraction.as_f64().ok_or("attraction must be a number")?;
        if !(0.0..=10.0).contains(&val) {
            return Err("attraction must be between 0.0 and 10.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "boundsSize"
    if let Some(bounds) = physics.get("boundsSize") {
        let val = bounds.as_f64().ok_or("boundsSize must be a number")?;
        if val < 100.0 || val > 50000.0 {
            return Err("boundsSize must be between 100.0 and 50000.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "collisionRadius"
    if let Some(collision) = physics.get("collisionRadius") {
        let val = collision.as_f64().ok_or("collisionRadius must be a number")?;
        if val < 0.0 || val > 100.0 {
            return Err("collisionRadius must be between 0.0 and 100.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "maxVelocity"
    if let Some(max_vel) = physics.get("maxVelocity") {
        let val = max_vel.as_f64().ok_or("maxVelocity must be a number")?;
        if val < 0.0 || val > 1000.0 {
            return Err("maxVelocity must be between 0.0 and 1000.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "massScale"
    if let Some(mass) = physics.get("massScale") {
        let val = mass.as_f64().ok_or("massScale must be a number")?;
        if val <= 0.0 || val > 10.0 {
            return Err("massScale must be between 0.0 and 10.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "boundaryDamping"
    if let Some(boundary) = physics.get("boundaryDamping") {
        let val = boundary.as_f64().ok_or("boundaryDamping must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("boundaryDamping must be between 0.0 and 1.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "timeStep"
    if let Some(time_step) = physics.get("timeStep") {
        let val = time_step.as_f64().ok_or("timeStep must be a number")?;
        if val <= 0.0 || val > 1.0 {
            return Err("timeStep must be between 0.0 and 1.0".to_string());
        }
    }
    
    // Client sends "temperature" directly
    if let Some(temp) = physics.get("temperature") {
        let val = temp.as_f64().ok_or("temperature must be a number")?;
        if val < 0.0 || val > 10.0 {
            return Err("temperature must be between 0.0 and 10.0".to_string());
        }
    }
    
    // Client sends "gravity" directly
    if let Some(gravity) = physics.get("gravity") {
        let val = gravity.as_f64().ok_or("gravity must be a number")?;
        if val < -10.0 || val > 10.0 {
            return Err("gravity must be between -10.0 and 10.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "updateThreshold"
    if let Some(threshold) = physics.get("updateThreshold") {
        let val = threshold.as_f64().ok_or("updateThreshold must be a number")?;
        if val < 0.0 || val > 1.0 {
            return Err("updateThreshold must be between 0.0 and 1.0".to_string());
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
        if val <= 0.0 || val > 10.0 {
            return Err("nodeSize must be between 0.0 and 10.0".to_string());
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
        if val < 0.0 || val > 10.0 {
            return Err("ambientLightIntensity must be between 0.0 and 10.0".to_string());
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
            
            // logLevel should be a number
            if let Some(log_level) = debug_obj.get("logLevel") {
                if let Some(val) = log_level.as_f64() {
                    if val < 0.0 || val > 3.0 {
                        return Err("debug.logLevel must be between 0 and 3".to_string());
                    }
                } else if let Some(val) = log_level.as_u64() {
                    if val > 3 {
                        return Err("debug.logLevel must be between 0 and 3".to_string());
                    }
                } else {
                    return Err("debug.logLevel must be a number".to_string());
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
        if val < 0.5 || val > 2.0 {
            return Err("renderScale must be between 0.5 and 2.0".to_string());
        }
    }
    
    // UNIFIED FORMAT: Only accept "roomScale"
    if let Some(room_scale) = xr.get("roomScale") {
        let val = room_scale.as_f64().ok_or("roomScale must be a number")?;
        if val <= 0.0 || val > 10.0 {
            return Err("roomScale must be between 0.0 and 10.0".to_string());
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
    let sim_params = physics.into();
    
    info!(
        "Propagating {} physics to GPU - damping: {:.3}, spring: {:.3}, repulsion: {:.3}, timeStep: {:.3}", 
        graph, physics.damping, physics.spring_strength, physics.repulsion_strength, physics.time_step
    );
    
    let update_msg = UpdateSimulationParams { params: sim_params };
    
    // Send to GPU compute actor
    if let Some(gpu_addr) = &state.gpu_compute_addr {
        if let Err(e) = gpu_addr.send(update_msg.clone()).await {
            warn!("Failed to update GPU physics: {}", e);
        } else {
            info!("GPU physics updated successfully");
        }
    }
    
    // Send to graph service actor
    if let Err(e) = state.graph_service_addr.send(update_msg).await {
        warn!("Failed to update graph service physics: {}", e);
    } else {
        info!("Graph service physics updated successfully");
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
fn create_physics_settings_update(physics_update: Value) -> Value {
    json!({
        "visualisation": {
            "graphs": {
                "logseq": {
                    "physics": physics_update
                },
                "visionflow": {
                    "physics": physics_update.clone()
                }
            }
        }
    })
}

