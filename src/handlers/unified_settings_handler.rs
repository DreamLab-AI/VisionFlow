// Simplified Settings Handler - Clean, maintainable, no redundant conversions
// Replaces 1200+ lines with ~200 lines of clear, focused code

use actix_web::{web, Error, HttpResponse, HttpRequest};
use crate::app_state::AppState;
use crate::config::unified::{UnifiedSettings, SettingsUpdate};
use crate::actors::messages::{GetSettings, UpdateSettings, UpdateSimulationParams};
use log::{info, warn, error, debug};
use serde_json::json;

/// Configure routes for settings endpoints
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/settings")
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings))
    )
    .service(
        web::resource("/settings/physics/{graph}")
            .route(web::post().to(update_physics))
    );
}

/// Get current settings
async fn get_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    // Get settings from actor
    let settings = match state.settings_addr.send(GetSettings).await {
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
    
    // Check if user wants specific graph settings
    if let Some(graph) = req.match_info().get("graph") {
        match graph {
            "logseq" => Ok(HttpResponse::Ok().json(&settings.graphs.logseq)),
            "visionflow" => Ok(HttpResponse::Ok().json(&settings.graphs.visionflow)),
            _ => Ok(HttpResponse::BadRequest().json(json!({
                "error": "Invalid graph name"
            })))
        }
    } else {
        // Return full settings
        Ok(HttpResponse::Ok().json(&settings))
    }
}

/// Update settings (partial update supported)
async fn update_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<SettingsUpdate>,
) -> Result<HttpResponse, Error> {
    let update = payload.into_inner();
    
    // Log what's being updated
    debug!("Settings update received: {:?}", update);
    
    // Get current settings
    let mut settings: UnifiedSettings = match state.settings_addr.send(GetSettings).await {
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
    
    // Check if physics is being updated
    let physics_updated = update.graphs.as_ref()
        .and_then(|g| g.logseq.as_ref())
        .and_then(|l| l.physics.as_ref())
        .is_some();
    
    // Merge the update
    settings.merge(update);
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: settings.clone() }).await {
        Ok(Ok(())) => {
            info!("Settings updated successfully");
            
            // If physics was updated, propagate to GPU
            if physics_updated {
                propagate_physics_to_gpu(&state, &settings, "logseq").await;
            }
            
            Ok(HttpResponse::Ok().json(&settings))
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

/// Direct physics update endpoint for specific graph
async fn update_physics(
    req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<crate::config::unified::PhysicsUpdate>,
) -> Result<HttpResponse, Error> {
    let graph = req.match_info().get("graph").unwrap_or("logseq");
    let physics_update = payload.into_inner();
    
    info!("Physics update for graph '{}': {:?}", graph, physics_update);
    
    // Get current settings
    let mut settings: UnifiedSettings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    // Update physics for the specified graph
    settings.update_physics(graph, physics_update);
    
    // Save settings
    match state.settings_addr.send(UpdateSettings { settings: settings.clone() }).await {
        Ok(Ok(())) => {
            // Propagate to GPU
            propagate_physics_to_gpu(&state, &settings, graph).await;
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "success",
                "message": format!("Physics updated for graph '{}'", graph)
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save physics: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save physics: {}", e)
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

/// Propagate physics settings to GPU compute actor
async fn propagate_physics_to_gpu(
    state: &web::Data<AppState>,
    settings: &UnifiedSettings,
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

// ============================================================================
// USER-SPECIFIC SETTINGS (if needed)
// ============================================================================

use crate::models::UserSettings;
use crate::config::feature_access::FeatureAccess;

/// Get user-specific settings
async fn get_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
) -> Result<HttpResponse, Error> {
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            return Ok(HttpResponse::BadRequest().json(json!({
                "error": "Missing authentication"
            })));
        }
    };
    
    if !feature_access.can_sync_settings(&pubkey) {
        return Ok(HttpResponse::Forbidden().json(json!({
            "error": "Settings sync not enabled"
        })));
    }
    
    // Power users get global settings
    if feature_access.is_power_user(&pubkey) {
        return get_settings(req, state).await;
    }
    
    // Regular users get their saved settings or defaults
    let user_settings = UserSettings::load(&pubkey)
        .unwrap_or_else(|| UserSettings::new(&pubkey, UnifiedSettings::default()));
    
    Ok(HttpResponse::Ok().json(&user_settings.settings))
}

/// Update user-specific settings
async fn update_user_settings(
    req: HttpRequest,
    state: web::Data<AppState>,
    feature_access: web::Data<FeatureAccess>,
    payload: web::Json<SettingsUpdate>,
) -> Result<HttpResponse, Error> {
    let pubkey = match req.headers().get("X-Nostr-Pubkey") {
        Some(value) => value.to_str().unwrap_or("").to_string(),
        None => {
            return Ok(HttpResponse::BadRequest().json(json!({
                "error": "Missing authentication"
            })));
        }
    };
    
    if !feature_access.can_sync_settings(&pubkey) {
        return Ok(HttpResponse::Forbidden().json(json!({
            "error": "Settings sync not enabled"
        })));
    }
    
    // Power users update global settings
    if feature_access.is_power_user(&pubkey) {
        info!("Power user {} updating global settings", pubkey);
        return update_settings(req, state, payload).await;
    }
    
    // Regular users update their own settings
    let mut user_settings = UserSettings::load(&pubkey)
        .unwrap_or_else(|| UserSettings::new(&pubkey, UnifiedSettings::default()));
    
    user_settings.settings.merge(payload.into_inner());
    user_settings.last_modified = chrono::Utc::now().timestamp();
    
    if let Err(e) = user_settings.save() {
        error!("Failed to save user settings: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": "Failed to save settings"
        })));
    }
    
    info!("User {} updated their settings", pubkey);
    Ok(HttpResponse::Ok().json(&user_settings.settings))
}