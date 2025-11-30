// src/settings/api/settings_routes.rs
//! REST API endpoints for settings management

use actix::Addr;
use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use log::{error, info};

use crate::config::{PhysicsSettings, RenderingSettings};
use crate::settings::settings_actor::{
    SettingsActor, UpdatePhysicsSettings, GetPhysicsSettings,
    UpdateConstraintSettings, GetConstraintSettings,
    UpdateRenderingSettings, GetRenderingSettings,
    UpdateNodeFilterSettings, GetNodeFilterSettings,
    UpdateQualityGateSettings, GetQualityGateSettings,
    LoadProfile, SaveProfile, ListProfiles, DeleteProfile, GetAllSettings,
};
use crate::settings::models::{ConstraintSettings, NodeFilterSettings, QualityGateSettings};

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SaveProfileRequest {
    pub name: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProfileIdResponse {
    pub id: i64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorResponse {
    pub error: String,
}

// ============================================================================
// Physics Settings Routes
// ============================================================================

/// GET /api/settings/physics
pub async fn get_physics_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
) -> impl Responder {
    match settings_actor.send(GetPhysicsSettings).await {
        Ok(settings) => HttpResponse::Ok().json(settings),
        Err(e) => {
            error!("Failed to get physics settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get physics settings: {}", e),
            })
        }
    }
}

/// PUT /api/settings/physics
pub async fn update_physics_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<PhysicsSettings>,
) -> impl Responder {
    info!("Updating physics settings");

    match settings_actor.send(UpdatePhysicsSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Physics settings updated successfully");
            HttpResponse::Ok().finish()
        }
        Ok(Err(e)) => {
            error!("Failed to update physics settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to update physics settings: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

// ============================================================================
// Constraint Settings Routes
// ============================================================================

/// GET /api/settings/constraints
pub async fn get_constraint_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
) -> impl Responder {
    match settings_actor.send(GetConstraintSettings).await {
        Ok(settings) => HttpResponse::Ok().json(settings),
        Err(e) => {
            error!("Failed to get constraint settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get constraint settings: {}", e),
            })
        }
    }
}

/// PUT /api/settings/constraints
pub async fn update_constraint_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<ConstraintSettings>,
) -> impl Responder {
    info!("Updating constraint settings");

    match settings_actor.send(UpdateConstraintSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Constraint settings updated successfully");
            HttpResponse::Ok().finish()
        }
        Ok(Err(e)) => {
            error!("Failed to update constraint settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to update constraint settings: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

// ============================================================================
// Rendering Settings Routes
// ============================================================================

///
pub async fn get_rendering_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
) -> impl Responder {
    match settings_actor.send(GetRenderingSettings).await {
        Ok(settings) => HttpResponse::Ok().json(settings),
        Err(e) => {
            error!("Failed to get rendering settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get rendering settings: {}", e),
            })
        }
    }
}

///
pub async fn update_rendering_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<RenderingSettings>,
) -> impl Responder {
    info!("Updating rendering settings");

    match settings_actor.send(UpdateRenderingSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Rendering settings updated successfully");
            HttpResponse::Ok().finish()
        }
        Ok(Err(e)) => {
            error!("Failed to update rendering settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to update rendering settings: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

// ============================================================================
// Node Filter Settings Routes
// ============================================================================

/// GET /api/settings/node-filter
pub async fn get_node_filter_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
) -> impl Responder {
    match settings_actor.send(GetNodeFilterSettings).await {
        Ok(settings) => HttpResponse::Ok().json(settings),
        Err(e) => {
            error!("Failed to get node filter settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get node filter settings: {}", e),
            })
        }
    }
}

/// PUT /api/settings/node-filter
pub async fn update_node_filter_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<NodeFilterSettings>,
) -> impl Responder {
    info!("Updating node filter settings: enabled={}, threshold={}",
          body.enabled, body.quality_threshold);

    match settings_actor.send(UpdateNodeFilterSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Node filter settings updated successfully");
            HttpResponse::Ok().finish()
        }
        Ok(Err(e)) => {
            error!("Failed to update node filter settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to update node filter settings: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

// ============================================================================
// Quality Gate Settings Routes
// ============================================================================

/// GET /api/settings/quality-gates
pub async fn get_quality_gate_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
) -> impl Responder {
    match settings_actor.send(GetQualityGateSettings).await {
        Ok(settings) => HttpResponse::Ok().json(settings),
        Err(e) => {
            error!("Failed to get quality gate settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get quality gate settings: {}", e),
            })
        }
    }
}

/// PUT /api/settings/quality-gates
pub async fn update_quality_gate_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<QualityGateSettings>,
) -> impl Responder {
    info!("Updating quality gate settings: gpu={}, ontology={}, semantic={}",
          body.gpu_acceleration, body.ontology_physics, body.semantic_forces);

    match settings_actor.send(UpdateQualityGateSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Quality gate settings updated successfully");
            HttpResponse::Ok().finish()
        }
        Ok(Err(e)) => {
            error!("Failed to update quality gate settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to update quality gate settings: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

// ============================================================================
// All Settings Route
// ============================================================================

///
pub async fn get_all_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
) -> impl Responder {
    info!("📥 GET /api/settings/all endpoint called");
    match settings_actor.send(GetAllSettings).await {
        Ok(settings) => HttpResponse::Ok().json(settings),
        Err(e) => {
            error!("Failed to get all settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get all settings: {}", e),
            })
        }
    }
}

// ============================================================================
// Profile Management Routes
// ============================================================================

///
pub async fn save_profile(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<SaveProfileRequest>,
) -> impl Responder {
    info!("Saving settings profile: {}", body.name);

    match settings_actor.send(SaveProfile { name: body.name.clone() }).await {
        Ok(Ok(profile_id)) => {
            info!("Settings profile saved with ID: {}", profile_id);
            HttpResponse::Created().json(ProfileIdResponse { id: profile_id })
        }
        Ok(Err(e)) => {
            error!("Failed to save settings profile: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to save settings profile: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

///
pub async fn load_profile(
    settings_actor: web::Data<Addr<SettingsActor>>,
    path: web::Path<i64>,
) -> impl Responder {
    let profile_id = path.into_inner();
    info!("Loading settings profile: {}", profile_id);

    match settings_actor.send(LoadProfile(profile_id)).await {
        Ok(Ok(settings)) => {
            info!("Settings profile loaded: {}", profile_id);
            HttpResponse::Ok().json(settings)
        }
        Ok(Err(e)) => {
            error!("Failed to load settings profile: {}", e);
            HttpResponse::NotFound().json(ErrorResponse {
                error: format!("Settings profile not found: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

///
pub async fn list_profiles(
    settings_actor: web::Data<Addr<SettingsActor>>,
) -> impl Responder {
    match settings_actor.send(ListProfiles).await {
        Ok(Ok(profiles)) => HttpResponse::Ok().json(profiles),
        Ok(Err(e)) => {
            error!("Failed to list settings profiles: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to list settings profiles: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

///
pub async fn delete_profile(
    settings_actor: web::Data<Addr<SettingsActor>>,
    path: web::Path<i64>,
) -> impl Responder {
    let profile_id = path.into_inner();
    info!("Deleting settings profile: {}", profile_id);

    match settings_actor.send(DeleteProfile(profile_id)).await {
        Ok(Ok(())) => {
            info!("Settings profile deleted: {}", profile_id);
            HttpResponse::Ok().finish()
        }
        Ok(Err(e)) => {
            error!("Failed to delete settings profile: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to delete settings profile: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            })
        }
    }
}

// ============================================================================
// Route Configuration
// ============================================================================

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    log::info!("🔧 Configuring settings routes");
    log::info!("📍 Registering route: GET all");
    log::info!("📍 Registering route: GET physics");
    log::info!("📍 Registering route: PUT physics");
    log::info!("📍 Registering route: GET constraints");
    log::info!("📍 Registering route: PUT constraints");
    log::info!("📍 Registering route: GET rendering");
    log::info!("📍 Registering route: PUT rendering");
    log::info!("📍 Registering route: POST profiles");
    log::info!("📍 Registering route: GET profiles");
    log::info!("📍 Registering route: GET profiles/{{id}}");
    log::info!("📍 Registering route: DELETE profiles/{{id}}");

    cfg.route("physics", web::get().to(get_physics_settings))
        .route("physics", web::put().to(update_physics_settings))
        .route("constraints", web::get().to(get_constraint_settings))
        .route("constraints", web::put().to(update_constraint_settings))
        .route("rendering", web::get().to(get_rendering_settings))
        .route("rendering", web::put().to(update_rendering_settings))
        .route("node-filter", web::get().to(get_node_filter_settings))
        .route("node-filter", web::put().to(update_node_filter_settings))
        .route("quality-gates", web::get().to(get_quality_gate_settings))
        .route("quality-gates", web::put().to(update_quality_gate_settings))
        .route("all", web::get().to(get_all_settings))
        .route("profiles", web::post().to(save_profile))
        .route("profiles", web::get().to(list_profiles))
        .route("profiles/{id}", web::get().to(load_profile))
        .route("profiles/{id}", web::delete().to(delete_profile));

    log::info!("✅ Settings routes configuration complete");
}
