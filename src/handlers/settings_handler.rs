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
        web::scope("/settings")  // Removed redundant /api prefix - already under /api scope
            .route("", web::get().to(get_settings))
            .route("", web::post().to(update_settings))
            .route("/reset", web::post().to(reset_settings))
            // Physics updates should go through the main update_settings endpoint
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

// Physics updates now go through the main update_settings endpoint
// The update_settings function already handles physics changes and GPU propagation

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
    
    Ok(())
}

fn validate_physics_settings(physics: &Value) -> Result<(), String> {
    // Validate all physics fields with proper ranges
    // NOTE: Validation happens on camelCase JSON from client (before conversion)
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
    
    // Check for both camelCase (from client) and snake_case (potential future use)
    let spring = physics.get("springStrength").or_else(|| physics.get("spring_strength"));
    if let Some(spring) = spring {
        let val = spring.as_f64().ok_or("springStrength must be a number")?;
        if !(0.0..=10.0).contains(&val) {
            return Err("springStrength must be between 0.0 and 10.0".to_string());
        }
    }
    
    let repulsion = physics.get("repulsionStrength").or_else(|| physics.get("repulsion_strength"));
    if let Some(repulsion) = repulsion {
        let val = repulsion.as_f64().ok_or("repulsionStrength must be a number")?;
        if val < 0.0 || val > 10000.0 {
            return Err("repulsionStrength must be between 0.0 and 10000.0".to_string());
        }
    }
    
    let attraction = physics.get("attractionStrength").or_else(|| physics.get("attraction_strength"));
    if let Some(attraction) = attraction {
        let val = attraction.as_f64().ok_or("attractionStrength must be a number")?;
        if !(0.0..=10.0).contains(&val) {
            return Err("attractionStrength must be between 0.0 and 10.0".to_string());
        }
    }
    
    let bounds = physics.get("boundsSize").or_else(|| physics.get("bounds_size"));
    if let Some(bounds) = bounds {
        let val = bounds.as_f64().ok_or("boundsSize must be a number")?;
        if val < 100.0 || val > 50000.0 {
            return Err("boundsSize must be between 100.0 and 50000.0".to_string());
        }
    }
    
    let collision = physics.get("collisionRadius").or_else(|| physics.get("collision_radius"));
    if let Some(collision) = collision {
        let val = collision.as_f64().ok_or("collisionRadius must be a number")?;
        if val < 0.0 || val > 100.0 {
            return Err("collisionRadius must be between 0.0 and 100.0".to_string());
        }
    }
    
    let max_vel = physics.get("maxVelocity").or_else(|| physics.get("max_velocity"));
    if let Some(max_vel) = max_vel {
        let val = max_vel.as_f64().ok_or("maxVelocity must be a number")?;
        if val < 0.0 || val > 1000.0 {
            return Err("maxVelocity must be between 0.0 and 1000.0".to_string());
        }
    }
    
    let mass = physics.get("massScale").or_else(|| physics.get("mass_scale"));
    if let Some(mass) = mass {
        let val = mass.as_f64().ok_or("massScale must be a number")?;
        if val <= 0.0 || val > 10.0 {
            return Err("massScale must be between 0.0 and 10.0".to_string());
        }
    }
    
    let boundary = physics.get("boundaryDamping").or_else(|| physics.get("boundary_damping"));
    if let Some(boundary) = boundary {
        let val = boundary.as_f64().ok_or("boundaryDamping must be a number")?;
        if !(0.0..=1.0).contains(&val) {
            return Err("boundaryDamping must be between 0.0 and 1.0".to_string());
        }
    }
    
    let time_step = physics.get("timeStep").or_else(|| physics.get("time_step"));
    if let Some(time_step) = time_step {
        let val = time_step.as_f64().ok_or("timeStep must be a number")?;
        if val <= 0.0 || val > 1.0 {
            return Err("timeStep must be between 0.0 and 1.0".to_string());
        }
    }
    
    if let Some(temp) = physics.get("temperature") {
        let val = temp.as_f64().ok_or("temperature must be a number")?;
        if val < 0.0 || val > 10.0 {
            return Err("temperature must be between 0.0 and 10.0".to_string());
        }
    }
    
    if let Some(gravity) = physics.get("gravity") {
        let val = gravity.as_f64().ok_or("gravity must be a number")?;
        if val < -10.0 || val > 10.0 {
            return Err("gravity must be between -10.0 and 10.0".to_string());
        }
    }
    
    let threshold = physics.get("updateThreshold").or_else(|| physics.get("update_threshold"));
    if let Some(threshold) = threshold {
        let val = threshold.as_f64().ok_or("updateThreshold must be a number")?;
        if val < 0.0 || val > 1.0 {
            return Err("updateThreshold must be between 0.0 and 1.0".to_string());
        }
    }
    
    Ok(())
}

fn validate_node_settings(nodes: &Value) -> Result<(), String> {
    // Validate color format - check both camelCase and snake_case
    let color = nodes.get("baseColor").or_else(|| nodes.get("base_color"));
    if let Some(color) = color {
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
    
    let node_size = nodes.get("nodeSize").or_else(|| nodes.get("node_size"));
    if let Some(node_size) = node_size {
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
    let ambient = rendering.get("ambientLightIntensity").or_else(|| rendering.get("ambient_light_intensity"));
    if let Some(ambient) = ambient {
        let val = ambient.as_f64().ok_or("ambientLightIntensity must be a number")?;
        if val < 0.0 || val > 10.0 {
            return Err("ambientLightIntensity must be between 0.0 and 10.0".to_string());
        }
    }
    
    Ok(())
}

fn validate_xr_settings(xr: &Value) -> Result<(), String> {
    let room_scale = xr.get("roomScale").or_else(|| xr.get("room_scale"));
    if let Some(room_scale) = room_scale {
        let val = room_scale.as_f64().ok_or("roomScale must be a number")?;
        if val <= 0.0 || val > 10.0 {
            return Err("roomScale must be between 0.0 and 10.0".to_string());
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
        "Propagating {} physics to GPU - damping: {}, spring: {}, repulsion: {}", 
        graph, physics.damping, physics.spring_strength, physics.repulsion_strength
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

