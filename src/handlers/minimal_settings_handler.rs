// Minimal settings handler - clean, simple, direct
use actix_web::{web, HttpResponse, Result};
use crate::config::minimal::SettingsUpdate;
use crate::actors::messages::UpdateSimulationParams;
use crate::AppState;
use log::{info, error};

// GET /api/settings - Return current settings
pub async fn get_settings(
    app_state: web::Data<AppState>,
) -> Result<HttpResponse> {
    // Get settings from the settings actor
    use crate::actors::messages::GetSettings;
    
    match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            // Convert to minimal settings if needed
            // For now, just return as JSON
            Ok(HttpResponse::Ok().json(&settings))
        }
        _ => {
            error!("Failed to get settings");
            Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "error": "Failed to retrieve settings"
            })))
        }
    }
}

// POST /api/settings/physics - Update physics settings
pub async fn update_physics(
    app_state: web::Data<AppState>,
    physics: web::Json<crate::config::minimal::PhysicsSettings>,
) -> Result<HttpResponse> {
    info!("Updating physics settings");
    
    // Convert to SimulationParams
    let physics_ref = physics.into_inner();
    let sim_params = physics_ref.to_simulation_params();
    
    // Send to GPU compute actor
    if let Some(gpu_addr) = &app_state.gpu_compute_addr {
        if let Err(e) = gpu_addr.send(UpdateSimulationParams { params: sim_params.clone() }).await {
            error!("Failed to update GPU physics: {}", e);
        } else {
            info!("Physics updated successfully in GPU");
        }
    }
    
    // Also update GraphServiceActor
    if let Err(e) = app_state.graph_service_addr.send(UpdateSimulationParams { params: sim_params }).await {
        error!("Failed to update graph physics: {}", e);
    }
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "message": "Physics settings updated"
    })))
}

// POST /api/settings - Update any settings section
pub async fn update_settings(
    app_state: web::Data<AppState>,
    update: web::Json<SettingsUpdate>,
) -> Result<HttpResponse> {
    info!("Updating settings section");
    
    // Apply update based on type
    match update.into_inner() {
        SettingsUpdate::Physics(physics) => {
            // Route to physics update
            let sim_params = physics.to_simulation_params();
            
            if let Some(gpu_addr) = &app_state.gpu_compute_addr {
                let _ = gpu_addr.send(UpdateSimulationParams { params: sim_params.clone() }).await;
            }
            let _ = app_state.graph_service_addr.send(UpdateSimulationParams { params: sim_params }).await;
        }
        SettingsUpdate::Nodes(_) | SettingsUpdate::Edges(_) | SettingsUpdate::Labels(_) => {
            // These affect rendering only - no backend update needed
            info!("Visual settings updated (client-side only)");
        }
        SettingsUpdate::Rendering(_) => {
            // Global rendering settings - no backend update needed
            info!("Rendering settings updated (client-side only)");
        }
        SettingsUpdate::Full(_) => {
            // Full settings replacement - handle all aspects
            info!("Full settings replacement");
        }
    }
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "message": "Settings updated"
    })))
}

// GET /api/settings/graphs - List available graphs
pub async fn list_graphs(
    _app_state: web::Data<AppState>,
) -> Result<HttpResponse> {
    // For now, return hardcoded list
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "graphs": ["logseq", "visionflow"],
        "active": "logseq"
    })))
}

// POST /api/settings/graph/{name} - Switch active graph
pub async fn switch_graph(
    _app_state: web::Data<AppState>,
    path: web::Path<String>,
) -> Result<HttpResponse> {
    let graph_name = path.into_inner();
    info!("Switching to graph: {}", graph_name);
    
    // TODO: Implement graph switching logic
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "message": format!("Switched to graph: {}", graph_name)
    })))
}

// Configure routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("/settings")
            .route(web::get().to(get_settings))
            .route(web::post().to(update_settings))
    )
    .service(
        web::resource("/settings/physics")
            .route(web::post().to(update_physics))
    )
    .service(
        web::resource("/settings/graphs")
            .route(web::get().to(list_graphs))
    )
    .service(
        web::resource("/settings/graph/{name}")
            .route(web::post().to(switch_graph))
    );
}