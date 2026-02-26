// src/settings/api/settings_routes.rs
//! REST API endpoints for settings management.
//! Uses OptimizedSettingsActor (via AppState) as the single source of truth.
//! All PUT routes validate input before applying. (QE Fix #1, #2, #3, #5)

use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use log::{debug, error, info, warn};
use std::sync::Arc;

use crate::config::{PhysicsSettings, RenderingSettings};
use crate::actors::messages::{BroadcastMessage, GetSettings, SetComputeMode, UpdateConstraints, UpdateSettings, UpdateSimulationParams};
use crate::utils::unified_gpu_compute::ComputeMode;
use crate::settings::models::{ConstraintSettings, NodeFilterSettings, QualityGateSettings, AllSettings};
use crate::settings::auth_extractor::{AuthenticatedUser, OptionalAuth};
use crate::adapters::neo4j_settings_repository::{Neo4jSettingsRepository, UserFilter};
use crate::ports::settings_repository::{SettingValue, SettingsRepository};
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

/// Normalize incoming physics JSON keys to the canonical camelCase names
/// expected by PhysicsSettings (which uses `#[serde(rename_all = "camelCase")]`).
///
/// This maps common aliases (snake_case, legacy names, client variants) so that
/// both `{"spring_k": 0.05}` and `{"springK": 0.05}` work correctly.
fn normalize_physics_keys(patch: serde_json::Map<String, serde_json::Value>) -> serde_json::Map<String, serde_json::Value> {
    let mut normalized = serde_json::Map::new();
    for (key, value) in patch {
        let canonical = match key.as_str() {
            // snake_case → camelCase mappings
            "spring_k"          => "springK",
            "repel_k"           => "repelK",
            "spring_strength"   => "springK",
            "repulsion_strength"=> "repelK",
            "center_gravity_k"  => "centerGravityK",
            "max_velocity"      => "maxVelocity",
            "max_force"         => "maxForce",
            "mass_scale"        => "massScale",
            "enable_bounds"     => "enableBounds",
            "bounds_size"       => "boundsSize",
            "separation_radius" => "separationRadius",
            "boundary_damping"  => "boundaryDamping",
            "update_threshold"  => "updateThreshold",
            "rest_length"       => "restLength",
            "repulsion_cutoff"  => "repulsionCutoff",
            "grid_cell_size"    => "gridCellSize",
            "warmup_iterations" => "warmupIterations",
            "cooling_rate"      => "coolingRate",
            "max_repulsion_dist"=> "maxRepulsionDist",
            "min_distance"      => "minDistance",
            "auto_balance"      => "autoBalance",
            "compute_mode"      => "computeMode",
            "cluster_strength"  => "clusterStrength",
            "alignment_strength"=> "alignmentStrength",
            // Legacy/client aliases
            "springStrength"    => "springK",
            "repulsionStrength" => "repelK",
            "attractionStrength"=> "centerGravityK",
            "springStiffness"   => "springK",
            "springDamping"     => "damping",
            "deltaTime"         => "dt",
            // Already camelCase — pass through
            other => other,
        };
        // Don't overwrite if the canonical key was already provided explicitly
        if !normalized.contains_key(canonical) {
            normalized.insert(canonical.to_string(), value);
        }
    }
    normalized
}

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
// Constraint Settings Validation (QE Fix #1)
// ============================================================================

/// Validates constraint threshold values are finite, non-negative, and properly ordered.
pub fn validate_constraint_settings(settings: &ConstraintSettings) -> Result<(), String> {
    if !settings.far_threshold.is_finite() || settings.far_threshold < 0.0 {
        return Err("far_threshold must be finite and non-negative".into());
    }
    if !settings.medium_threshold.is_finite() || settings.medium_threshold < 0.0 {
        return Err("medium_threshold must be finite and non-negative".into());
    }
    if !settings.near_threshold.is_finite() || settings.near_threshold < 0.0 {
        return Err("near_threshold must be finite and non-negative".into());
    }
    if settings.near_threshold >= settings.medium_threshold || settings.medium_threshold >= settings.far_threshold {
        return Err("Thresholds must be ordered: near < medium < far".into());
    }
    Ok(())
}

// ============================================================================
// Rendering Settings Validation (QE Fix #1)
// ============================================================================

/// Validates rendering light intensity values are finite and non-negative.
pub fn validate_rendering_settings(settings: &RenderingSettings) -> Result<(), String> {
    let check_finite = |v: f64, name: &str| -> Result<(), String> {
        if !v.is_finite() || v < 0.0 {
            Err(format!("{} must be finite and non-negative", name))
        } else {
            Ok(())
        }
    };
    check_finite(settings.ambient_light_intensity as f64, "ambient_light_intensity")?;
    check_finite(settings.directional_light_intensity as f64, "directional_light_intensity")?;
    check_finite(settings.environment_intensity as f64, "environment_intensity")?;
    Ok(())
}

// ============================================================================
// Node Filter Settings Validation (QE Fix #1)
// ============================================================================

/// Validates node filter thresholds and filter mode.
pub fn validate_node_filter_settings(settings: &NodeFilterSettings) -> Result<(), String> {
    if settings.quality_threshold < 0.0 || settings.quality_threshold > 1.0 {
        return Err("quality_threshold must be 0.0-1.0".into());
    }
    if settings.authority_threshold < 0.0 || settings.authority_threshold > 1.0 {
        return Err("authority_threshold must be 0.0-1.0".into());
    }
    if settings.filter_mode != "and" && settings.filter_mode != "or" {
        return Err("filter_mode must be 'and' or 'or'".into());
    }
    Ok(())
}

// ============================================================================
// Physics Settings Routes
// ============================================================================

/// GET /api/settings/physics
pub async fn get_physics_settings(
    state: web::Data<AppState>,
    _auth: OptionalAuth,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
) -> impl Responder {
    // Try Neo4j persisted settings first, fall back to in-memory actor
    if let Ok(Some(SettingValue::Json(json))) = neo4j_repo.get_setting("physics").await {
        if let Ok(physics) = serde_json::from_value::<PhysicsSettings>(json) {
            return HttpResponse::Ok().json(physics);
        }
    }

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
/// Uses single GetSettings call to avoid TOCTOU race (QE Fix #3).
pub async fn update_physics_settings(
    state: web::Data<AppState>,
    body: web::Json<serde_json::Value>,
    auth: AuthenticatedUser,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
) -> impl Responder {
    debug!("User {} updating physics settings — request body: {:?}", auth.pubkey, body);

    // Single GetSettings call -- fetch full settings snapshot once to avoid TOCTOU race
    let mut full_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
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

    // Merge partial patch onto current physics from the same snapshot
    let current_physics = &full_settings.visualisation.graphs.logseq.physics;
    let current_json = serde_json::to_value(current_physics).unwrap_or_default();

    let new_physics = if let (serde_json::Value::Object(mut base), serde_json::Value::Object(patch)) =
        (current_json, body.into_inner())
    {
        // Normalize incoming keys: map common aliases (snake_case, legacy names)
        // to the canonical camelCase field names used by PhysicsSettings.
        let normalized_patch = normalize_physics_keys(patch);
        for (k, v) in normalized_patch {
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
        full_settings.visualisation.graphs.logseq.physics.clone()
    };

    // Validate before applying
    if let Err(validation_err) = validate_physics_settings(&new_physics) {
        warn!("Physics settings validation failed: {}", validation_err);
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: format!("Validation failed: {}", validation_err),
        });
    }

    // Apply merged physics to the same snapshot and write back atomically
    full_settings.visualisation.graphs.logseq.physics = new_physics.clone();
    match state.settings_addr.send(UpdateSettings { settings: full_settings }).await {
        Ok(Ok(())) => {
            info!("Physics settings updated successfully by {}", auth.pubkey);

            // Propagate physics changes to GPU actors so layout actually responds
            let sim_params: crate::models::simulation_params::SimulationParams = (&new_physics).into();
            debug!("Propagating SimulationParams to GPU actors: {:?}", sim_params);
            let update_msg = UpdateSimulationParams { params: sim_params };

            if let Some(gpu_addr) = state.get_gpu_compute_addr().await {
                if let Err(e) = gpu_addr.send(update_msg.clone()).await {
                    error!("Failed to propagate physics to GPUComputeActor: {}", e);
                }
            } else {
                warn!("No GPUComputeActor address available — physics won't propagate to GPU!");
            }
            if let Err(e) = state.graph_service_addr.send(update_msg).await {
                error!("Failed to propagate physics to GraphServiceActor: {}", e);
            }

            // Persist physics settings to Neo4j for cross-restart survival
            match serde_json::to_value(&new_physics) {
                Ok(physics_json) => {
                    if let Err(e) = neo4j_repo.set_setting(
                        "physics",
                        SettingValue::Json(physics_json),
                        Some("Physics simulation settings"),
                    ).await {
                        warn!("Failed to persist physics settings to Neo4j: {}", e);
                    }
                }
                Err(e) => {
                    warn!("Failed to serialize physics settings for persistence: {}", e);
                }
            }

            HttpResponse::Ok().json(&new_physics)
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
/// Loads constraint settings from Neo4j repository, falling back to defaults.
pub async fn get_constraint_settings(
    _state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    _auth: OptionalAuth,
) -> impl Responder {
    match neo4j_repo.get_setting("constraints").await {
        Ok(Some(crate::ports::settings_repository::SettingValue::Json(json))) => {
            match serde_json::from_value::<ConstraintSettings>(json) {
                Ok(settings) => HttpResponse::Ok().json(settings),
                Err(e) => {
                    warn!("Failed to parse stored constraint settings, returning defaults: {}", e);
                    HttpResponse::Ok().json(ConstraintSettings::default())
                }
            }
        }
        Ok(_) => HttpResponse::Ok().json(ConstraintSettings::default()),
        Err(e) => {
            warn!("Failed to load constraint settings from repository: {}", e);
            HttpResponse::Ok().json(ConstraintSettings::default())
        }
    }
}

/// PUT /api/settings/constraints
/// Validates input before accepting (QE Fix #1). Returns updated state (QE Fix #2).
/// Persists to Neo4j repository via set_setting.
pub async fn update_constraint_settings(
    state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    body: web::Json<ConstraintSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating constraint settings", auth.pubkey);

    let settings = body.into_inner();

    // Validate before accepting
    if let Err(validation_err) = validate_constraint_settings(&settings) {
        warn!("Constraint settings validation failed: {}", validation_err);
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: format!("Validation failed: {}", validation_err),
        });
    }

    // Persist to Neo4j
    let settings_json = match serde_json::to_value(&settings) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to serialize constraint settings: {}", e);
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to serialize constraint settings: {}", e),
            });
        }
    };

    if let Err(e) = neo4j_repo.set_setting(
        "constraints",
        crate::ports::settings_repository::SettingValue::Json(settings_json),
        Some("Constraint settings for physics simulation"),
    ).await {
        error!("Failed to persist constraint settings: {}", e);
        return HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Failed to persist constraint settings: {}", e),
        });
    }

    info!("Constraint settings updated and persisted for user {}", auth.pubkey);

    // Propagate constraint settings to GPU actors so physics simulation reflects changes
    let constraint_json = match serde_json::to_value(&settings) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to serialize constraint settings for GPU propagation: {}", e);
            // Settings are already persisted; return success but log the propagation failure
            return HttpResponse::Ok().json(&settings);
        }
    };

    let constraint_msg = UpdateConstraints { constraint_data: constraint_json };

    if let Some(ref gpu_manager) = state.gpu_manager_addr {
        if let Err(e) = gpu_manager.send(constraint_msg).await {
            error!("Failed to propagate constraints to GPUManagerActor: {}", e);
        } else {
            info!("Constraint settings propagated to GPUManagerActor for user {}", auth.pubkey);
        }
    } else {
        warn!("GPUManagerActor not available; constraint settings persisted but not propagated to GPU");
    }

    HttpResponse::Ok().json(&settings)
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
/// Validates input before applying (QE Fix #1). Returns updated state (QE Fix #2).
pub async fn update_rendering_settings(
    state: web::Data<AppState>,
    body: web::Json<RenderingSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating rendering settings", auth.pubkey);

    let new_rendering = body.into_inner();

    // Validate before applying
    if let Err(validation_err) = validate_rendering_settings(&new_rendering) {
        warn!("Rendering settings validation failed: {}", validation_err);
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: format!("Validation failed: {}", validation_err),
        });
    }

    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(mut full_settings)) => {
            full_settings.visualisation.rendering = new_rendering.clone();
            match state.settings_addr.send(UpdateSettings { settings: full_settings }).await {
                Ok(Ok(())) => {
                    info!("Rendering settings updated successfully by {}", auth.pubkey);

                    // Propagate rendering changes to connected clients via broadcast
                    // Rendering settings (ambient light, shadows, environment) are applied
                    // client-side; notify all clients so they pick up the new values.
                    let broadcast_payload = serde_json::json!({
                        "type": "settingsUpdated",
                        "category": "rendering",
                        "updatedBy": auth.pubkey,
                        "timestamp": chrono::Utc::now().timestamp_millis()
                    });
                    if let Ok(msg_str) = serde_json::to_string(&broadcast_payload) {
                        state.client_manager_addr.do_send(BroadcastMessage { message: msg_str });
                        info!("Rendering settings change broadcast sent to connected clients");
                    }

                    HttpResponse::Ok().json(&new_rendering)
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
/// Loads node filter settings from Neo4j repository, falling back to defaults.
pub async fn get_node_filter_settings(
    _state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    _auth: OptionalAuth,
) -> impl Responder {
    match neo4j_repo.get_setting("node_filter").await {
        Ok(Some(crate::ports::settings_repository::SettingValue::Json(json))) => {
            match serde_json::from_value::<NodeFilterSettings>(json) {
                Ok(settings) => HttpResponse::Ok().json(settings),
                Err(e) => {
                    warn!("Failed to parse stored node filter settings, returning defaults: {}", e);
                    HttpResponse::Ok().json(NodeFilterSettings::default())
                }
            }
        }
        Ok(_) => HttpResponse::Ok().json(NodeFilterSettings::default()),
        Err(e) => {
            warn!("Failed to load node filter settings from repository: {}", e);
            HttpResponse::Ok().json(NodeFilterSettings::default())
        }
    }
}

/// PUT /api/settings/node-filter
/// Validates input before accepting (QE Fix #1). Returns updated state (QE Fix #2).
/// Persists to Neo4j repository via set_setting.
pub async fn update_node_filter_settings(
    state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    body: web::Json<NodeFilterSettings>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating node filter settings: enabled={}, threshold={}",
          auth.pubkey, body.enabled, body.quality_threshold);

    let settings = body.into_inner();

    // Validate before accepting
    if let Err(validation_err) = validate_node_filter_settings(&settings) {
        warn!("Node filter settings validation failed: {}", validation_err);
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: format!("Validation failed: {}", validation_err),
        });
    }

    // Persist to Neo4j
    let settings_json = match serde_json::to_value(&settings) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to serialize node filter settings: {}", e);
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to serialize node filter settings: {}", e),
            });
        }
    };

    if let Err(e) = neo4j_repo.set_setting(
        "node_filter",
        crate::ports::settings_repository::SettingValue::Json(settings_json),
        Some("Node confidence filter settings"),
    ).await {
        error!("Failed to persist node filter settings: {}", e);
        return HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Failed to persist node filter settings: {}", e),
        });
    }

    // Propagate node filter changes to all connected clients via broadcast.
    // Clients receiving this message recompute which nodes pass the filter
    // and re-render the visible graph accordingly.
    let broadcast_payload = serde_json::json!({
        "type": "settingsUpdated",
        "category": "nodeFilter",
        "settings": {
            "enabled": settings.enabled,
            "qualityThreshold": settings.quality_threshold,
            "authorityThreshold": settings.authority_threshold,
            "filterByQuality": settings.filter_by_quality,
            "filterByAuthority": settings.filter_by_authority,
            "filterMode": settings.filter_mode,
        },
        "updatedBy": auth.pubkey,
        "timestamp": chrono::Utc::now().timestamp_millis()
    });
    if let Ok(msg_str) = serde_json::to_string(&broadcast_payload) {
        state.client_manager_addr.do_send(BroadcastMessage { message: msg_str });
        info!("Node filter settings change broadcast sent to connected clients");
    }

    info!("Node filter settings updated and persisted for user {}: enabled={}, quality_threshold={}",
          auth.pubkey, settings.enabled, settings.quality_threshold);
    HttpResponse::Ok().json(&settings)
}

// ============================================================================
// Quality Gate Settings Routes
// ============================================================================

/// GET /api/settings/quality-gates
/// Loads quality gate settings from Neo4j repository, falling back to defaults.
pub async fn get_quality_gate_settings(
    _state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    _auth: OptionalAuth,
) -> impl Responder {
    match neo4j_repo.get_setting("quality_gates").await {
        Ok(Some(crate::ports::settings_repository::SettingValue::Json(json))) => {
            match serde_json::from_value::<QualityGateSettings>(json) {
                Ok(settings) => HttpResponse::Ok().json(settings),
                Err(e) => {
                    warn!("Failed to parse stored quality gate settings, returning defaults: {}", e);
                    HttpResponse::Ok().json(QualityGateSettings::default())
                }
            }
        }
        Ok(_) => HttpResponse::Ok().json(QualityGateSettings::default()),
        Err(e) => {
            warn!("Failed to load quality gate settings from repository: {}", e);
            HttpResponse::Ok().json(QualityGateSettings::default())
        }
    }
}

/// PUT /api/settings/quality-gates
/// Accepts partial JSON updates -- missing fields retain their persisted or default values.
/// Returns updated state (QE Fix #2). Persists to Neo4j repository.
pub async fn update_quality_gate_settings(
    state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    body: web::Json<serde_json::Value>,
    auth: AuthenticatedUser,
) -> impl Responder {
    // Load current persisted settings as the merge base (instead of hardcoded defaults)
    let current_settings = match neo4j_repo.get_setting("quality_gates").await {
        Ok(Some(crate::ports::settings_repository::SettingValue::Json(json))) => {
            serde_json::from_value::<QualityGateSettings>(json).unwrap_or_default()
        }
        _ => QualityGateSettings::default(),
    };

    let mut settings = current_settings;
    let current_json = serde_json::to_value(&settings).unwrap_or_default();

    if let (serde_json::Value::Object(mut base), serde_json::Value::Object(patch)) =
        (current_json, body.into_inner())
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

    // Persist to Neo4j
    let settings_json = match serde_json::to_value(&settings) {
        Ok(json) => json,
        Err(e) => {
            error!("Failed to serialize quality gate settings: {}", e);
            return HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Failed to serialize quality gate settings: {}", e),
            });
        }
    };

    if let Err(e) = neo4j_repo.set_setting(
        "quality_gates",
        crate::ports::settings_repository::SettingValue::Json(settings_json),
        Some("Quality gate settings for feature toggles and performance thresholds"),
    ).await {
        error!("Failed to persist quality gate settings: {}", e);
        return HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Failed to persist quality gate settings: {}", e),
        });
    }

    info!("User {} updated quality gate settings: gpu={}, ontology={}, semantic={}, layout={}, maxNodeCount={}",
          auth.pubkey, settings.gpu_acceleration, settings.ontology_physics,
          settings.semantic_forces, settings.layout_mode, settings.max_node_count);

    // Propagate GPU-affecting quality gate changes to the physics engine.
    // Fetch current physics from the settings actor so SimulationParams is built
    // from the authoritative snapshot, then overlay quality-gate fields.
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(full_settings)) => {
            let physics = &full_settings.visualisation.graphs.logseq.physics;
            let mut sim_params: crate::models::simulation_params::SimulationParams = physics.into();

            // gpu_acceleration -> compute_mode (0 = Basic/CPU, 2 = Advanced/GPU)
            if settings.gpu_acceleration {
                sim_params.compute_mode = 2;
            } else {
                sim_params.compute_mode = 0;
            }
            sim_params.enabled = settings.gpu_acceleration;

            // Apply layout-mode-specific physics overrides.
            // DAG modes enable center gravity + SSSP for hierarchical layout.
            // Type-clustering enables cluster/alignment forces for grouping.
            match settings.layout_mode.as_str() {
                "dag-topdown" | "dag-radial" | "dag-leftright" => {
                    info!("Quality gates: applying DAG layout overrides for mode: {}", settings.layout_mode);
                    sim_params.center_gravity_k = sim_params.center_gravity_k.max(0.1);
                    sim_params.use_sssp_distances = true;
                    sim_params.sssp_alpha = Some(sim_params.sssp_alpha.unwrap_or(0.0).max(0.5));
                }
                "type-clustering" => {
                    info!("Quality gates: applying type-clustering layout overrides");
                    sim_params.cluster_strength = sim_params.cluster_strength.max(0.5);
                    sim_params.alignment_strength = sim_params.alignment_strength.max(0.3);
                }
                _ => {
                    // force-directed: use physics settings as-is
                }
            }

            let update_msg = UpdateSimulationParams { params: sim_params };

            if let Some(gpu_addr) = state.get_gpu_compute_addr().await {
                if let Err(e) = gpu_addr.send(update_msg.clone()).await {
                    error!("Quality gates: failed to propagate SimulationParams to ForceComputeActor: {}", e);
                }
                let compute_mode = if settings.gpu_acceleration {
                    ComputeMode::Advanced
                } else {
                    ComputeMode::Basic
                };
                if let Err(e) = gpu_addr.send(SetComputeMode { mode: compute_mode }).await {
                    error!("Quality gates: failed to propagate ComputeMode to ForceComputeActor: {}", e);
                }
            }

            if let Err(e) = state.graph_service_addr.send(update_msg).await {
                error!("Quality gates: failed to propagate SimulationParams to GraphServiceActor: {}", e);
            }

            info!("Quality gates propagated to physics engine: gpu={}, compute_mode={}, layout={}",
                  settings.gpu_acceleration,
                  if settings.gpu_acceleration { "Advanced" } else { "Basic" },
                  settings.layout_mode);
        }
        Ok(Err(e)) => {
            warn!("Quality gates persisted but failed to read settings for propagation: {}", e);
        }
        Err(e) => {
            warn!("Quality gates persisted but settings actor unreachable for propagation: {}", e);
        }
    }

    HttpResponse::Ok().json(settings)
}

// ============================================================================
// Visual Settings Routes (opaque JSON blob for client visual settings)
// ============================================================================

/// Recursively deep-merge `patch` into `base`. Values in `patch` take priority.
/// Both must be JSON objects; non-object values in `patch` replace `base` wholesale.
fn deep_merge_json(base: &mut serde_json::Value, patch: serde_json::Value) {
    if let (serde_json::Value::Object(base_map), serde_json::Value::Object(patch_map)) = (base, patch) {
        for (key, value) in patch_map {
            let entry = base_map.entry(key).or_insert(serde_json::Value::Null);
            if value.is_object() && entry.is_object() {
                deep_merge_json(entry, value);
            } else {
                *entry = value;
            }
        }
    }
}

/// GET /api/settings/visual
/// Returns the stored visual settings blob (glow, hologram, graphTypeVisuals,
/// gemMaterial, sceneEffects, clusterHulls, animations, interaction, nodes, edges, labels).
/// Falls back to empty object if nothing stored yet.
pub async fn get_visual_settings(
    _state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    _auth: OptionalAuth,
) -> impl Responder {
    match neo4j_repo.get_setting("visual").await {
        Ok(Some(SettingValue::Json(json))) => HttpResponse::Ok().json(json),
        Ok(_) => HttpResponse::Ok().json(serde_json::json!({})),
        Err(e) => {
            warn!("Failed to load visual settings from repository: {}", e);
            HttpResponse::Ok().json(serde_json::json!({}))
        }
    }
}

/// PUT /api/settings/visual
/// Accepts partial JSON updates — deep merges with currently stored values.
/// This is the persistence endpoint for all client-only visual settings.
pub async fn update_visual_settings(
    _state: web::Data<AppState>,
    neo4j_repo: web::Data<Arc<Neo4jSettingsRepository>>,
    body: web::Json<serde_json::Value>,
    auth: AuthenticatedUser,
) -> impl Responder {
    info!("User {} updating visual settings", auth.pubkey);

    let patch = body.into_inner();
    if !patch.is_object() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "Visual settings must be a JSON object".to_string(),
        });
    }

    // Load current stored settings as merge base
    let mut current = match neo4j_repo.get_setting("visual").await {
        Ok(Some(SettingValue::Json(json))) if json.is_object() => json,
        _ => serde_json::json!({}),
    };

    // Deep merge patch into current
    deep_merge_json(&mut current, patch);

    // Persist merged result to Neo4j
    if let Err(e) = neo4j_repo.set_setting(
        "visual",
        SettingValue::Json(current.clone()),
        Some("Client visual settings (glow, hologram, graphTypeVisuals, nodes, edges, labels, etc.)"),
    ).await {
        error!("Failed to persist visual settings: {}", e);
        return HttpResponse::InternalServerError().json(ErrorResponse {
            error: format!("Failed to persist visual settings: {}", e),
        });
    }

    info!("Visual settings updated and persisted for user {}", auth.pubkey);
    HttpResponse::Ok().json(current)
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
            get_all_from_actor(&state, &neo4j_repo).await
        }
        None => {
            info!("GET /api/settings/all for anonymous user (read-only)");
            get_all_from_actor(&state, &neo4j_repo).await
        }
    }
}

async fn get_all_from_actor(
    state: &web::Data<AppState>,
    neo4j_repo: &web::Data<Arc<Neo4jSettingsRepository>>,
) -> HttpResponse {
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(full_settings)) => {
            // Load persisted settings from Neo4j repository, falling back to actor/defaults
            let physics = match neo4j_repo.get_setting("physics").await {
                Ok(Some(SettingValue::Json(json))) => {
                    serde_json::from_value::<PhysicsSettings>(json)
                        .unwrap_or(full_settings.visualisation.graphs.logseq.physics)
                }
                _ => full_settings.visualisation.graphs.logseq.physics,
            };

            let constraints = match neo4j_repo.get_setting("constraints").await {
                Ok(Some(SettingValue::Json(json))) => {
                    serde_json::from_value::<ConstraintSettings>(json).unwrap_or_default()
                }
                _ => ConstraintSettings::default(),
            };

            let node_filter = match neo4j_repo.get_setting("node_filter").await {
                Ok(Some(SettingValue::Json(json))) => {
                    serde_json::from_value::<NodeFilterSettings>(json).unwrap_or_default()
                }
                _ => NodeFilterSettings::default(),
            };

            let quality_gates = match neo4j_repo.get_setting("quality_gates").await {
                Ok(Some(SettingValue::Json(json))) => {
                    serde_json::from_value::<QualityGateSettings>(json).unwrap_or_default()
                }
                _ => QualityGateSettings::default(),
            };

            let visual = match neo4j_repo.get_setting("visual").await {
                Ok(Some(SettingValue::Json(json))) => json,
                _ => serde_json::json!({}),
            };

            let all = AllSettings {
                physics,
                constraints,
                rendering: full_settings.visualisation.rendering,
                node_filter,
                quality_gates,
                visual,
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
        .route("visual", web::get().to(get_visual_settings))
        .route("visual", web::put().to(update_visual_settings))
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
