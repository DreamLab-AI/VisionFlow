//! Settings Management API Handlers
//!
//! Provides REST API endpoints for managing physics, constraint, and rendering
//! settings with persistence and profile management.

use actix_web::{web, HttpResponse, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::actors::messages::{GetSettings, UpdateSettings};
use crate::config::{ConstraintSettings, PhysicsSettings, RenderingSettings};
use crate::AppState;

///
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdatePhysicsRequest {
    pub gravity: Option<[f32; 3]>,
    pub damping: Option<f32>,
    pub repulsion_strength: Option<f32>,
    pub attraction_strength: Option<f32>,
    pub spring_stiffness: Option<f32>,
    pub spring_damping: Option<f32>,
    pub max_velocity: Option<f32>,
    pub dt: Option<f32>,
}

///
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateConstraintRequest {
    pub enabled: Option<bool>,
    pub max_iterations: Option<u32>,
    pub convergence_threshold: Option<f32>,
    pub constraint_strength_multiplier: Option<f32>,
    pub progressive_activation: Option<bool>,
}

///
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateRenderingRequest {
    pub ambient_light_intensity: Option<f32>,
    pub enable_ambient_occlusion: Option<bool>,
    pub background_color: Option<String>,
    pub node_scale: Option<f32>,
    pub edge_thickness: Option<f32>,
}

///
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SettingsProfile {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub physics: PhysicsSettings,
    pub constraints: ConstraintSettings,
    pub rendering: RenderingSettings,
    pub created_at: String,
}

///
pub async fn get_physics_settings(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/settings/physics");

    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            ok_json!(json!({
                "physics": settings.physics
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to fetch physics settings: {}", e);
            error_json!("Failed to fetch settings").expect("JSON serialization failed")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed").expect("JSON serialization failed")
        }
    }
}

///
pub async fn update_physics_settings(
    state: web::Data<AppState>,
    req: web::Json<UpdatePhysicsRequest>,
) -> impl Responder {
    info!("PUT /api/settings/physics");

    
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut settings)) => {
            
            if let Some(gravity) = req.gravity {
                settings.physics.gravity = gravity;
            }
            if let Some(damping) = req.damping {
                settings.physics.damping = damping;
            }
            if let Some(repulsion) = req.repulsion_strength {
                settings.physics.repulsion_strength = repulsion;
            }
            if let Some(attraction) = req.attraction_strength {
                settings.physics.attraction_strength = attraction;
            }
            if let Some(stiffness) = req.spring_stiffness {
                settings.physics.spring_stiffness = stiffness;
            }
            if let Some(spring_damping) = req.spring_damping {
                settings.physics.spring_damping = spring_damping;
            }
            if let Some(max_vel) = req.max_velocity {
                settings.physics.max_velocity = max_vel;
            }
            if let Some(dt) = req.dt {
                settings.physics.dt = dt;
            }

            
            match state.settings_addr.send(UpdateSettings { settings: settings.clone() }).await {
                Ok(Ok(())) => {
                    ok_json!(json!({
                        "success": true,
                        "physics": settings.physics
                    }))
                }
                Ok(Err(e)) => {
                    error!("Failed to update physics settings: {}", e);
                    error_json!("Failed to update settings").expect("JSON serialization failed")
                }
                Err(e) => {
                    error!("Actor mailbox error: {}", e);
                    error_json!("Actor communication failed").expect("JSON serialization failed")
                }
            }
        }
        Ok(Err(e)) => {
            error!("Failed to fetch current settings: {}", e);
            error_json!("Failed to fetch settings").expect("JSON serialization failed")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed").expect("JSON serialization failed")
        }
    }
}

///
pub async fn get_constraint_settings(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/settings/constraints");

    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            ok_json!(json!({
                "constraints": settings.constraints
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to fetch constraint settings: {}", e);
            error_json!("Failed to fetch settings").expect("JSON serialization failed")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed").expect("JSON serialization failed")
        }
    }
}

///
pub async fn update_constraint_settings(
    state: web::Data<AppState>,
    req: web::Json<UpdateConstraintRequest>,
) -> impl Responder {
    info!("PUT /api/settings/constraints");

    
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut settings)) => {
            
            if let Some(enabled) = req.enabled {
                settings.constraints.enabled = enabled;
            }
            if let Some(max_iter) = req.max_iterations {
                settings.constraints.max_iterations = max_iter;
            }
            if let Some(threshold) = req.convergence_threshold {
                settings.constraints.convergence_threshold = threshold;
            }
            if let Some(multiplier) = req.constraint_strength_multiplier {
                settings.constraints.constraint_strength_multiplier = multiplier;
            }
            if let Some(progressive) = req.progressive_activation {
                settings.constraints.progressive_activation = progressive;
            }

            
            match state.settings_addr.send(UpdateSettings { settings: settings.clone() }).await {
                Ok(Ok(())) => {
                    ok_json!(json!({
                        "success": true,
                        "constraints": settings.constraints
                    }))
                }
                Ok(Err(e)) => {
                    error!("Failed to update constraint settings: {}", e);
                    error_json!("Failed to update settings").expect("JSON serialization failed")
                }
                Err(e) => {
                    error!("Actor mailbox error: {}", e);
                    error_json!("Actor communication failed").expect("JSON serialization failed")
                }
            }
        }
        Ok(Err(e)) => {
            error!("Failed to fetch current settings: {}", e);
            error_json!("Failed to fetch settings").expect("JSON serialization failed")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed").expect("JSON serialization failed")
        }
    }
}

///
pub async fn get_rendering_settings(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/settings/rendering");

    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            ok_json!(json!({
                "rendering": settings.rendering
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to fetch rendering settings: {}", e);
            error_json!("Failed to fetch settings").expect("JSON serialization failed")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed").expect("JSON serialization failed")
        }
    }
}

///
pub async fn update_rendering_settings(
    state: web::Data<AppState>,
    req: web::Json<UpdateRenderingRequest>,
) -> impl Responder {
    info!("PUT /api/settings/rendering");

    
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut settings)) => {
            
            if let Some(ambient) = req.ambient_light_intensity {
                settings.rendering.ambient_light_intensity = ambient;
            }
            if let Some(ao) = req.enable_ambient_occlusion {
                settings.rendering.enable_ambient_occlusion = ao;
            }
            if let Some(ref bg) = req.background_color {
                settings.rendering.background_color = bg.clone();
            }
            if let Some(scale) = req.node_scale {
                settings.rendering.node_scale = scale;
            }
            if let Some(thickness) = req.edge_thickness {
                settings.rendering.edge_thickness = thickness;
            }

            
            match state.settings_addr.send(UpdateSettings { settings: settings.clone() }).await {
                Ok(Ok(())) => {
                    ok_json!(json!({
                        "success": true,
                        "rendering": settings.rendering
                    }))
                }
                Ok(Err(e)) => {
                    error!("Failed to update rendering settings: {}", e);
                    error_json!("Failed to update settings").expect("JSON serialization failed")
                }
                Err(e) => {
                    error!("Actor mailbox error: {}", e);
                    error_json!("Actor communication failed").expect("JSON serialization failed")
                }
            }
        }
        Ok(Err(e)) => {
            error!("Failed to fetch current settings: {}", e);
            error_json!("Failed to fetch settings").expect("JSON serialization failed")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed").expect("JSON serialization failed")
        }
    }
}

///
pub async fn save_profile(
    state: web::Data<AppState>,
    req: web::Json<serde_json::Value>,
) -> impl Responder {
    info!("POST /api/settings/profiles");

    let name = req.get("name").and_then(|v| v.as_str()).unwrap_or("Untitled Profile");
    let description = req.get("description").and_then(|v| v.as_str());

    
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            let profile = SettingsProfile {
                id: uuid::Uuid::new_v4().to_string(),
                name: name.to_string(),
                description: description.map(|s| s.to_string()),
                physics: settings.physics.clone(),
                constraints: settings.constraints.clone(),
                rendering: settings.rendering.clone(),
                created_at: chrono::Utc::now().to_rfc3339(),
            };

            
            created_json!(json!({
                "success": true,
                "profile": profile
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to fetch settings for profile: {}", e);
            error_json!("Failed to create profile").expect("JSON serialization failed")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed").expect("JSON serialization failed")
        }
    }
}

///
pub async fn list_profiles(_state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/settings/profiles");

    
    
    ok_json!(json!({
        "profiles": []
    }))
}

///
pub async fn load_profile(
    _state: web::Data<AppState>,
    profile_id: web::Path<String>,
) -> impl Responder {
    info!("GET /api/settings/profiles/{}", profile_id);

    
    not_found!("Profile not found").unwrap()
}

///
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/settings")
            .route("/physics", web::get().to(get_physics_settings))
            .route("/physics", web::put().to(update_physics_settings))
            .route("/constraints", web::get().to(get_constraint_settings))
            .route("/constraints", web::put().to(update_constraint_settings))
            .route("/rendering", web::get().to(get_rendering_settings))
            .route("/rendering", web::put().to(update_rendering_settings))
            .route("/profiles", web::post().to(save_profile))
            .route("/profiles", web::get().to(list_profiles))
            .route("/profiles/{id}", web::get().to(load_profile)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::{
    ok_json, created_json, error_json, bad_request, not_found,
    unauthorized, forbidden, conflict, no_content, accepted,
    too_many_requests, service_unavailable, payload_too_large
};


    #[test]
    fn test_settings_profile_serialization() {
        let profile = SettingsProfile {
            id: "test-123".to_string(),
            name: "Test Profile".to_string(),
            description: Some("Test description".to_string()),
            physics: PhysicsSettings::default(),
            constraints: ConstraintSettings::default(),
            rendering: RenderingSettings::default(),
            created_at: "2025-10-31T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&profile).unwrap();
        assert!(json.contains("test-123"));
        assert!(json.contains("Test Profile"));
    }
}
