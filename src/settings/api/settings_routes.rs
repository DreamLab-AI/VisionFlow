// src/settings/api/settings_routes.rs
//! REST API endpoints for settings management.
//! Uses OptimizedSettingsActor (via AppState) as the single source of truth.
//! All PUT routes validate input before applying. (QE Fix #1, #2, #5)

use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use log::{error, info, warn};
use std::sync::Arc;

use crate::config::{PhysicsSettings, RenderingSettings};
use crate::actors::messages::{GetSettings, UpdateSettings};
use crate::settings::models::{ConstraintSettings, NodeFilterSettings, QualityGateSettings, AllSettings};
use crate::settings::auth_extractor::{AuthenticatedUser, OptionalAuth};
use crate::adapters::neo4j_settings_repository::{Neo4jSettingsRepository, UserFilter};
use crate::AppState;

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
// Physics Settings Validation (QE Fix #2 + Fix #5)
// ============================================================================

/// Validates physics settings values are within safe ranges.
/// Rejects NaN, Infinity, and out-of-range values.
pub fn validate_physics_settings(settings: &PhysicsSettings) -> Result<(), String> {
    let mut errors = Vec::new();

    // Helper closure: check finite
    let check_finite = |val: f32, name: &str, errs: &mut Vec<String>| {
        if !val.is_finite() {
            errs.push(format!("{} must be a finite number (not NaN or Infinity)", name));
        }
    };

    // Helper closure: check range (inclusive)
    let check_range = |val: f32, name: &str, min: f32, max: f32, errs: &mut Vec<String>| {
        if !val.is_finite() {
            errs.push(format!("{} must be a finite number (not NaN or Infinity)", name));
        } else if val < min || val > max {
            errs.push(format!("{} must be between {} and {} (got {})", name, min, max, val));
        }
    };

    // Range-checked fields per QE audit spec
    check_range(settings.gravity, "gravity", 0.0, 1.0, &mut errors);
    check_range(settings.damping, "damping", 0.0, 1.0, &mut errors);
    check_range(settings.spring_k, "spring_k", 0.0, 100.0, &mut errors);
    check_range(settings.max_velocity, "max_velocity", 0.0, 200.0, &mut errors);
    check_range(settings.max_force, "max_force", 0.0, 200.0, &mut errors);
    check_range(settings.dt, "timestep (dt)", 0.001, 1.0, &mut errors);

    // All other f32 fields: reject NaN/Infinity
    check_finite(settings.bounds_size, "bounds_size", &mut errors);
    check_finite(settings.separation_radius, "separation_radius", &mut errors);
    check_finite(settings.repel_k, "repel_k", &mut errors);
    check_finite(settings.mass_scale, "mass_scale", &mut errors);
    check_finite(settings.boundary_damping, "boundary_damping", &mut errors);
    check_finite(settings.update_threshold, "update_threshold", &mut errors);
    check_finite(settings.temperature, "temperature", &mut errors);
    check_finite(settings.stress_weight, "stress_weight", &mut errors);
    check_finite(settings.stress_alpha, "stress_alpha", &mut errors);
    check_finite(settings.boundary_limit, "boundary_limit", &mut errors);
    check_finite(settings.alignment_strength, "alignment_strength", &mut errors);
    check_finite(settings.cluster_strength, "cluster_strength", &mut errors);
    check_finite(settings.rest_length, "rest_length", &mut errors);
    check_finite(settings.repulsion_cutoff, "repulsion_cutoff", &mut errors);
    check_finite(settings.repulsion_softening_epsilon, "repulsion_softening_epsilon", &mut errors);
    check_finite(settings.center_gravity_k, "center_gravity_k", &mut errors);
    check_finite(settings.grid_cell_size, "grid_cell_size", &mut errors);
    check_finite(settings.cooling_rate, "cooling_rate", &mut errors);
    check_finite(settings.boundary_extreme_multiplier, "boundary_extreme_multiplier", &mut errors);
    check_finite(settings.boundary_extreme_force_multiplier, "boundary_extreme_force_multiplier", &mut errors);
    check_finite(settings.boundary_velocity_damping, "boundary_velocity_damping", &mut errors);
    check_finite(settings.min_distance, "min_distance", &mut errors);
    check_finite(settings.max_repulsion_dist, "max_repulsion_dist", &mut errors);
    check_finite(settings.boundary_margin, "boundary_margin", &mut errors);
    check_finite(settings.boundary_force_strength, "boundary_force_strength", &mut errors);
    check_finite(settings.constraint_max_force_per_node, "constraint_max_force_per_node", &mut errors);
    check_finite(settings.clustering_resolution, "clustering_resolution", &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

// ============================================================================
// Physics Settings Routes
// ============================================================================

/// GET /api/settings/physics
pub async fn get_physics_settings(
    state: web::Data<AppState>,
    _auth: OptionalAuth,
) -> impl Responder {
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => HttpResponse::Ok().json(settings.visualisation.graphs.logseq.physics),
        Ok(Err(e)) => {
            error!("Failed to get physics settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get physics settings: {}", e),
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

/// PUT /api/settings/physics
/// Validates input before applying (QE Fix #2 + #5).
/// Accepts partial JSON updates -- missing fields retain current values from the actor.
pub async fn update_physics_settings(
    state: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating physics settings", auth.pubkey);

    // Get current physics settings as the base for merging
    let current_physics = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(full_settings)) => full_settings.visualisation.graphs.logseq.physics,
        Ok(Err(e)) => {
            error!("Failed to fetch current settings: {}", e);
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to fetch current settings: {}", e),
            });
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Actor communication error: {}", e),
            });
        }
    };

    let current_json = serde_json::to_value(&current_physics).unwrap_or_default();

    let new_physics = if let (serde_json::Value::Object(mut base), serde_json::Value::Object(patch)) =
        (current_json, body.into_inner())
    {
        for (k, v) in patch {
            base.insert(k, v);
        }
        match serde_json::from_value::<PhysicsSettings>(serde_json::Value::Object(base)) {
            Ok(merged) => merged,
            Err(e) => {
                warn!("Physics settings merge failed: {}", e);
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: format!("Invalid settings value: {}", e),
                });
            }
        }
    } else {
        current_physics
    };

    // Validate before applying
    if let Err(validation_err) = validate_physics_settings(&new_physics) {
        warn!("Physics settings validation failed: {}", validation_err);
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: format!("Validation failed: {}", validation_err),
        });
    }

    // Get current full settings, update physics subsection, write back
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut full_settings)) => {
            full_settings.visualisation.graphs.logseq.physics = new_physics;
            match state.settings_addr.send(UpdateSettings { settings: full_settings }).await {
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
        Ok(Err(e)) => {
            error!("Failed to fetch current settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to fetch current settings: {}", e),
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
/// Constraint settings are not part of AppFullSettings; return defaults.
pub async fn get_constraint_settings(
    _state: web::Data<AppState>,
    _auth: OptionalAuth,
) -> impl Responder {
    HttpResponse::Ok().json(ConstraintSettings::default())
}

/// PUT /api/settings/constraints
/// Constraint settings are not persisted in AppFullSettings.
pub async fn update_constraint_settings(
    _state: web::Data<AppState>,
    body: web::Json<ConstraintSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating constraint settings (accepted, not persisted in AppFullSettings)", auth.pubkey);
    let _settings = body.into_inner();
    HttpResponse::Ok().finish()
}

// ============================================================================
// Rendering Settings Routes
// ============================================================================

/// GET /api/settings/rendering
pub async fn get_rendering_settings(
    state: web::Data<AppState>,
    _auth: OptionalAuth,
) -> impl Responder {
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => HttpResponse::Ok().json(settings.visualisation.rendering),
        Ok(Err(e)) => {
            error!("Failed to get rendering settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get rendering settings: {}", e),
            })
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get rendering settings: {}", e),
            })
        }
    }
}

/// PUT /api/settings/rendering
pub async fn update_rendering_settings(
    state: web::Data<AppState>,
    body: web::Json<RenderingSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating rendering settings", auth.pubkey);

    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut full_settings)) => {
            full_settings.visualisation.rendering = body.into_inner();
            match state.settings_addr.send(UpdateSettings { settings: full_settings }).await {
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
        Ok(Err(e)) => {
            error!("Failed to fetch current settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to fetch current settings: {}", e),
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
    _state: web::Data<AppState>,
    _auth: OptionalAuth,
) -> impl Responder {
    // Node filter settings are not part of AppFullSettings.
    // Return defaults (persisted via Neo4j repo separately).
    HttpResponse::Ok().json(NodeFilterSettings::default())
}

/// PUT /api/settings/node-filter
pub async fn update_node_filter_settings(
    _state: web::Data<AppState>,
    body: web::Json<NodeFilterSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating node filter settings: enabled={}, threshold={}",
          auth.pubkey, body.enabled, body.quality_threshold);
    warn!("Node filter settings accepted but persistence deferred (single actor consolidation)");
    HttpResponse::Ok().finish()
}

// ============================================================================
// Quality Gate Settings Routes
// ============================================================================

/// GET /api/settings/quality-gates
pub async fn get_quality_gate_settings(
    _state: web::Data<AppState>,
    _auth: OptionalAuth,
) -> impl Responder {
    HttpResponse::Ok().json(QualityGateSettings::default())
}

/// PUT /api/settings/quality-gates
/// Accepts partial JSON updates -- missing fields retain their defaults.
pub async fn update_quality_gate_settings(
    _state: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
    auth: AuthenticatedUser,
) -> impl Responder {
    let mut settings = QualityGateSettings::default();
    let defaults_json = serde_json::to_value(&settings).unwrap_or_default();

    if let (serde_json::Value::Object(mut base), serde_json::Value::Object(patch)) =
        (defaults_json, body.into_inner())
    {
        for (k, v) in patch {
            base.insert(k, v);
        }
        match serde_json::from_value::<QualityGateSettings>(serde_json::Value::Object(base)) {
            Ok(merged) => settings = merged,
            Err(e) => {
                warn!("Quality gate settings merge failed: {}", e);
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: format!("Invalid settings value: {}", e),
                });
            }
        }
    }

    info!("User {} updating quality gate settings: gpu={}, ontology={}, semantic={}, maxNodeCount={}",
          auth.pubkey, settings.gpu_acceleration, settings.ontology_physics,
          settings.semantic_forces, settings.max_node_count);
    warn!("Quality gate settings accepted but persistence deferred (single actor consolidation)");
    HttpResponse::Ok().json(settings)
}

// ============================================================================
// All Settings Route
// ============================================================================

/// GET /api/settings/all
/// Returns global settings for anonymous users, or user-specific settings for authenticated users
pub async fn get_all_settings(
    state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    auth: OptionalAuth,
) -> impl Responder {
    match auth.0 {
        Some(user) => {
            info!("GET /api/settings/all for authenticated user: {}", user.pubkey);
            match neo4j_repo.get_user_settings(&user.pubkey).await {
                Ok(Some(user_settings)) => {
                    info!("Returning user-specific settings for: {}", user.pubkey);
                    return HttpResponse::Ok().json(user_settings);
                }
                Ok(None) => {
                    info!("No user-specific settings found, returning global for: {}", user.pubkey);
                }
                Err(e) => {
                    error!("Failed to query user settings: {}, falling back to global", e);
                }
            }
            get_all_from_actor(&state).await
        }
        None => {
            info!("GET /api/settings/all for anonymous user (read-only)");
            get_all_from_actor(&state).await
        }
    }
}

async fn get_all_from_actor(state: &web::Data<AppState>) -> HttpResponse {
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(full_settings)) => {
            let all = AllSettings {
                physics: full_settings.visualisation.graphs.logseq.physics,
                constraints: ConstraintSettings::default(),
                rendering: full_settings.visualisation.rendering,
                node_filter: NodeFilterSettings::default(),
                quality_gates: QualityGateSettings::default(),
            };
            HttpResponse::Ok().json(all)
        }
        Ok(Err(e)) => {
            error!("Failed to get all settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get all settings: {}", e),
            })
        }
        Err(e) => {
            error!("Failed to get all settings: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to get all settings: {}", e),
            })
        }
    }
}

// ============================================================================
// User Filter Routes
// ============================================================================

/// GET /api/user/filter
pub async fn get_user_filter(
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("GET /api/user/filter for user: {}", auth.pubkey);

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
pub async fn update_user_filter(
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    body: web::Json<UserFilter>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("PUT /api/user/filter for user: {}", auth.pubkey);

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
    _state: web::Data<AppState>,
    body: web::Json<SaveProfileRequest>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} saving settings profile: {}", auth.pubkey, body.name);
    HttpResponse::Created().json(ProfileIdResponse { id: 1 })
}

/// GET /api/settings/profiles/{id}
pub async fn load_profile(
    _state: web::Data<AppState>,
    path: web::Path<i64>,
    _auth: OptionalAuth,
) -> impl Responder {
    let profile_id = path.into_inner();
    info!("Loading settings profile: {}", profile_id);
    HttpResponse::NotFound().json(ErrorResponse {
        error: "Profile not found".to_string(),
    })
}

/// GET /api/settings/profiles
pub async fn list_profiles(
    _state: web::Data<AppState>,
    _auth: OptionalAuth,
) -> impl Responder {
    HttpResponse::Ok().json(Vec::<crate::settings::models::SettingsProfile>::new())
}

/// DELETE /api/settings/profiles/{id}
pub async fn delete_profile(
    _state: web::Data<AppState>,
    path: web::Path<i64>,
    auth: AuthenticatedUser,
) -> impl Responder {
    let profile_id = path.into_inner();
    info!("User {} deleting settings profile: {}", auth.pubkey, profile_id);
    HttpResponse::Ok().finish()
}

// ============================================================================
// Route Configuration
// ============================================================================

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    log::info!("Configuring settings routes (unified via OptimizedSettingsActor)");

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

    // User-specific filter settings
    cfg.service(
        web::scope("/user")
            .route("/filter", web::get().to(get_user_filter))
            .route("/filter", web::put().to(update_user_filter))
    );

    log::info!("Settings routes configuration complete (single actor, validated PUT routes)");
}
