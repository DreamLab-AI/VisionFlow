// src/settings/api/settings_routes.rs
//! REST API endpoints for settings management

use actix::Addr;
use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use log::{error, info};
use std::sync::Arc;

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
use crate::settings::auth_extractor::{AuthenticatedUser, OptionalAuth};
use crate::adapters::neo4j_settings_repository::{Neo4jSettingsRepository, UserFilter};

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
/// Allows both authenticated and anonymous access (read-only for anonymous)
pub async fn get_physics_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    _auth: OptionalAuth,
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
/// Requires authentication
pub async fn update_physics_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<PhysicsSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating physics settings", auth.pubkey);

    match settings_actor.send(UpdatePhysicsSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Physics settings updated successfully by {}", auth.pubkey);
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
    _auth: OptionalAuth,
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
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating constraint settings", auth.pubkey);

    match settings_actor.send(UpdateConstraintSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Constraint settings updated successfully by {}", auth.pubkey);
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

/// GET /api/settings/rendering
pub async fn get_rendering_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    _auth: OptionalAuth,
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

/// PUT /api/settings/rendering
pub async fn update_rendering_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<RenderingSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating rendering settings", auth.pubkey);

    match settings_actor.send(UpdateRenderingSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Rendering settings updated successfully by {}", auth.pubkey);
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
    _auth: OptionalAuth,
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
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating node filter settings: enabled={}, threshold={}",
          auth.pubkey, body.enabled, body.quality_threshold);

    match settings_actor.send(UpdateNodeFilterSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Node filter settings updated successfully by {}", auth.pubkey);
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
    _auth: OptionalAuth,
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
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating quality gate settings: gpu={}, ontology={}, semantic={}",
          auth.pubkey, body.gpu_acceleration, body.ontology_physics, body.semantic_forces);

    match settings_actor.send(UpdateQualityGateSettings(body.into_inner())).await {
        Ok(Ok(())) => {
            info!("Quality gate settings updated successfully by {}", auth.pubkey);
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

/// GET /api/settings/all
/// Returns global settings for anonymous users, or user-specific settings for authenticated users
pub async fn get_all_settings(
    settings_actor: web::Data<Addr<SettingsActor>>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    auth: OptionalAuth,
) -> impl Responder {
    match auth.0 {
        Some(user) => {
            info!("📥 GET /api/settings/all for authenticated user: {}", user.pubkey);

            // Try to get user-specific settings first
            match neo4j_repo.get_user_settings(&user.pubkey).await {
                Ok(Some(user_settings)) => {
                    info!("Returning user-specific settings for: {}", user.pubkey);
                    return HttpResponse::Ok().json(user_settings);
                }
                Ok(None) => {
                    info!("No user-specific settings found, returning global settings for: {}", user.pubkey);
                }
                Err(e) => {
                    error!("Failed to query user settings: {}, falling back to global", e);
                }
            }

            // Fall back to global settings
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
        None => {
            info!("📥 GET /api/settings/all for anonymous user (read-only)");
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
    }
}

// ============================================================================
// User Filter Routes
// ============================================================================

/// GET /api/user/filter
/// Get user's personal filter settings
pub async fn get_user_filter(
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("📥 GET /api/user/filter for user: {}", auth.pubkey);

    match neo4j_repo.get_user_filter(&auth.pubkey).await {
        Ok(Some(filter)) => {
            info!("Returning user filter for: {}", auth.pubkey);
            HttpResponse::Ok().json(filter)
        }
        Ok(None) => {
            info!("No user filter found, returning defaults for: {}", auth.pubkey);
            HttpResponse::Ok().json(UserFilter::default())
        }
        Err(e) => {
            error!("Failed to query user filter: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get user filter: {}", e),
            })
        }
    }
}

/// PUT /api/user/filter
/// Update user's personal filter settings
pub async fn update_user_filter(
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    body: web::Json<UserFilter>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("📤 PUT /api/user/filter for user: {}", auth.pubkey);

    let mut filter = body.into_inner();
    filter.pubkey = auth.pubkey.clone();

    match neo4j_repo.save_user_filter(&auth.pubkey, &filter).await {
        Ok(()) => {
            info!("User filter saved successfully for: {}", auth.pubkey);
            HttpResponse::Ok().json(filter)
        }
        Err(e) => {
            error!("Failed to save user filter: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to save user filter: {}", e),
            })
        }
    }
}

// ============================================================================
// Profile Management Routes
// ============================================================================

/// POST /api/settings/profiles
pub async fn save_profile(
    settings_actor: web::Data<Addr<SettingsActor>>,
    body: web::Json<SaveProfileRequest>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} saving settings profile: {}", auth.pubkey, body.name);

    match settings_actor.send(SaveProfile { name: body.name.clone() }).await {
        Ok(Ok(profile_id)) => {
            info!("Settings profile saved with ID {} by {}", profile_id, auth.pubkey);
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

/// GET /api/settings/profiles/{id}
pub async fn load_profile(
    settings_actor: web::Data<Addr<SettingsActor>>,
    path: web::Path<i64>,
    _auth: OptionalAuth,
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

/// GET /api/settings/profiles
pub async fn list_profiles(
    settings_actor: web::Data<Addr<SettingsActor>>,
    _auth: OptionalAuth,
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

/// DELETE /api/settings/profiles/{id}
pub async fn delete_profile(
    settings_actor: web::Data<Addr<SettingsActor>>,
    path: web::Path<i64>,
    auth: AuthenticatedUser,
) -> impl Responder {
    let profile_id = path.into_inner();
    info!("User {} deleting settings profile: {}", auth.pubkey, profile_id);

    match settings_actor.send(DeleteProfile(profile_id)).await {
        Ok(Ok(())) => {
            info!("Settings profile {} deleted by {}", profile_id, auth.pubkey);
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

    // User-specific filter settings - requires authentication
    cfg.service(
        web::scope("/user")
            .route("/filter", web::get().to(get_user_filter))
            .route("/filter", web::put().to(update_user_filter))
    );

    log::info!("✅ Settings routes configuration complete with user filter support");
}
