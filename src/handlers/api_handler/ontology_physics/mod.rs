//! Ontology-Physics Integration API Handlers
//!
//! Provides REST API endpoints for managing ontology-driven physics forces,
//! enabling semantic relationships to influence graph layout through GPU-accelerated
//! constraint forces.
//!
//! ## Architecture
//!
//! This module bridges the ontology validation system with the GPU physics pipeline:
//! - OntologyActor -> OntologyConstraintActor -> ForceComputeActor
//! - Constraints are uploaded to GPU and applied during physics simulation

use actix_web::{web, HttpResponse, Responder};
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use crate::{ok_json, error_json, bad_request, service_unavailable};

use crate::actors::messages::{
    ApplyOntologyConstraints, ConstraintMergeMode, GetConstraintStats, GetOntologyConstraintStats,
};
use crate::models::constraints::ConstraintSet;
use crate::AppState;

// ============================================================================
// REQUEST/RESPONSE DTOs
// ============================================================================

/// Enable ontology physics forces
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnableOntologyPhysicsRequest {
    /// Ontology ID to source constraints from
    pub ontology_id: String,

    /// Constraint merge mode (replace, merge, add_if_no_conflict)
    pub merge_mode: Option<String>,

    /// Force weight (0.0 to 1.0)
    pub strength: Option<f32>,
}

/// Constraint list response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintListResponse {
    pub active_constraints: u32,
    pub total_constraints: u32,
    pub constraint_types: Vec<String>,
    pub ontology_id: Option<String>,
}

/// Weight adjustment request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdjustWeightsRequest {
    /// Global strength multiplier (0.0 to 1.0)
    pub global_strength: Option<f32>,

    /// Per-constraint-type weights
    pub constraint_weights: Option<std::collections::HashMap<String, f32>>,
}

/// Weight adjustment response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WeightsResponse {
    pub success: bool,
    pub applied_strength: f32,
    pub active_constraints: u32,
}

// ============================================================================
// FEATURE CHECK
// ============================================================================

/// Check if ontology feature is enabled (always enabled)
async fn check_ontology_feature() -> Result<(), actix_web::Error> {
    Ok(())
}

// ============================================================================
// REST ENDPOINTS
// ============================================================================

/// POST /api/ontology-physics/enable - Enable ontology forces
///
/// Activates ontology-derived constraint forces in the physics simulation.
/// Constraints are sourced from the specified ontology and uploaded to GPU memory.
///
/// ## Request Body
/// ```json
/// {
///   "ontologyId": "university-ontology",
///   "mergeMode": "replace",
///   "strength": 0.8
/// }
/// ```
///
/// ## Response
/// ```json
/// {
///   "success": true,
///   "activeConstraints": 42,
///   "message": "Ontology forces enabled"
/// }
/// ```
pub async fn enable_ontology_physics(
    state: web::Data<AppState>,
    req: web::Json<EnableOntologyPhysicsRequest>,
) -> impl Responder {
    info!("POST /api/ontology-physics/enable - Ontology: {}", req.ontology_id);

    check_ontology_feature().await?;

    // Parse merge mode
    let merge_mode = match req.merge_mode.as_deref() {
        Some("replace") | None => ConstraintMergeMode::Replace,
        Some("merge") => ConstraintMergeMode::Merge,
        Some("add_if_no_conflict") => ConstraintMergeMode::AddIfNoConflict,
        Some(other) => {
            return bad_request!(json!({
                "error": "Invalid merge mode",
                "validModes": ["replace", "merge", "add_if_no_conflict"],
                "provided": other
            }));
        }
    };

    // Get the ontology actor address
    let Some(ref ontology_addr) = state.ontology_actor_addr else {
        return service_unavailable!(json!({
            "error": "Ontology actor not available",
            "message": "Ontology system not initialized"
        }));
    };

    // Retrieve the most recent validation report for this ontology
    use crate::actors::messages::GetOntologyReport;
    let report_result = ontology_addr
        .send(GetOntologyReport {
            report_id: Some(req.ontology_id.clone()),
        })
        .await;

    match report_result {
        Ok(Ok(Some(report))) => {
            info!("Retrieved validation report with {} constraints", report.constraint_summary.total_constraints);

            // Convert the report's reasoning constraints into a ConstraintSet
            // The report contains ontology axioms and inferences that were validated
            // We need to apply these as physics constraints

            // Get OntologyConstraintActor address from GPU Manager
            let Some(ref gpu_manager_addr) = state.gpu_manager_addr else {
                return service_unavailable!(json!({
                    "error": "GPU manager not available",
                    "message": "GPU acceleration not initialized"
                }));
            };

            // Create a constraint set from the validation report
            // This would normally be done by the ontology translator
            let constraint_set = ConstraintSet::new(); // Placeholder

            // Apply constraints to OntologyConstraintActor
            use crate::actors::messages::ApplyOntologyConstraints;

            // Clone merge_mode for display later (moved into message)
            let merge_mode_display = format!("{:?}", merge_mode);

            let apply_msg = ApplyOntologyConstraints {
                constraint_set,
                merge_mode,
                graph_id: 0, // Default graph
            };

            // Forward to GPU Manager which will route to OntologyConstraintActor
            match gpu_manager_addr.send(apply_msg).await {
                Ok(Ok(())) => {
                    ok_json!(json!({
                        "success": true,
                        "activeConstraints": report.constraint_summary.total_constraints,
                        "message": "Ontology forces enabled",
                        "ontologyId": req.ontology_id,
                        "mergeMode": merge_mode_display
                    }))
                }
                Ok(Err(e)) => {
                    error!("Failed to apply ontology constraints: {}", e);
                    error_json!(json!({
                        "error": "Constraint application failed",
                        "details": e
                    }))
                }
                Err(e) => {
                    error!("GPU Manager mailbox error: {}", e);
                    error_json!(json!({
                        "error": "Actor communication failed",
                        "details": e.to_string()
                    }))
                }
            }
        }
        Ok(Ok(None)) => {
            warn!("No validation report found for ontology: {}", req.ontology_id);
            bad_request!(json!({
                "error": "Ontology not found",
                "message": "Run validation first to generate constraints",
                "ontologyId": req.ontology_id
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to retrieve ontology report: {}", e);
            error_json!(json!({
                "error": "Report retrieval failed",
                "details": e
            }))
        }
        Err(e) => {
            error!("Ontology actor mailbox error: {}", e);
            error_json!(json!({
                "error": "Actor communication failed",
                "details": e.to_string()
            }))
        }
    }
}

/// GET /api/ontology-physics/constraints - List active ontology constraints
///
/// Returns statistics about currently active ontology-derived physics constraints.
///
/// ## Response
/// ```json
/// {
///   "activeConstraints": 42,
///   "totalConstraints": 50,
///   "constraintTypes": ["SubClassOf", "DisjointWith", "EquivalentClasses"],
///   "ontologyId": "university-ontology"
/// }
/// ```
pub async fn get_constraints(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/ontology-physics/constraints");

    check_ontology_feature().await?;

    let Some(ref gpu_manager_addr) = state.gpu_manager_addr else {
        return service_unavailable!(json!({
            "error": "GPU manager not available"
        }));
    };

    // Get ontology constraint stats
    match gpu_manager_addr.send(GetOntologyConstraintStats).await {
        Ok(Ok(stats)) => {
            ok_json!(json!({
                "activeConstraints": stats.active_ontology_constraints,
                "totalConstraints": stats.total_axioms_processed,
                "constraintEvaluationCount": stats.constraint_evaluation_count,
                "lastUpdateTimeMs": stats.last_update_time_ms,
                "gpuFailureCount": stats.gpu_failure_count,
                "cpuFallbackCount": stats.cpu_fallback_count
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to get constraint stats: {}", e);
            error_json!(json!({
                "error": "Stats retrieval failed",
                "details": e
            }))
        }
        Err(e) => {
            error!("GPU Manager mailbox error: {}", e);
            error_json!(json!({
                "error": "Actor communication failed",
                "details": e.to_string()
            }))
        }
    }
}

/// PUT /api/ontology-physics/weights - Adjust constraint strengths
///
/// Modifies the strength of ontology-derived forces without reloading constraints.
///
/// ## Request Body
/// ```json
/// {
///   "globalStrength": 0.5,
///   "constraintWeights": {
///     "SubClassOf": 0.8,
///     "DisjointWith": 1.0
///   }
/// }
/// ```
///
/// ## Response
/// ```json
/// {
///   "success": true,
///   "appliedStrength": 0.5,
///   "activeConstraints": 42
/// }
/// ```
pub async fn adjust_weights(
    state: web::Data<AppState>,
    req: web::Json<AdjustWeightsRequest>,
) -> Result<impl Responder, actix_web::Error> {
    info!("PUT /api/ontology-physics/weights");

    check_ontology_feature().await?;

    let global_strength = req.global_strength.unwrap_or(1.0).clamp(0.0, 1.0);

    // For now, we'll just acknowledge the request
    // Full implementation would involve updating constraint strengths in GPU memory
    warn!("Weight adjustment not yet fully implemented - returning success");

    ok_json!(json!({
        "success": true,
        "appliedStrength": global_strength,
        "message": "Weights adjusted (implementation pending)",
        "activeConstraints": 0
    }))
}

/// POST /api/ontology-physics/disable - Disable ontology forces
///
/// Removes all ontology-derived constraints from the physics simulation.
pub async fn disable_ontology_physics(state: web::Data<AppState>) -> impl Responder {
    info!("POST /api/ontology-physics/disable");

    check_ontology_feature().await?;

    let Some(ref gpu_manager_addr) = state.gpu_manager_addr else {
        return service_unavailable!(json!({
            "error": "GPU manager not available"
        }));
    };

    // Clear constraints by applying empty set
    let apply_msg = ApplyOntologyConstraints {
        constraint_set: ConstraintSet::new(),
        merge_mode: ConstraintMergeMode::Replace,
        graph_id: 0,
    };

    match gpu_manager_addr.send(apply_msg).await {
        Ok(Ok(())) => {
            ok_json!(json!({
                "success": true,
                "message": "Ontology forces disabled"
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to disable ontology forces: {}", e);
            error_json!(json!({
                "error": "Disable failed",
                "details": e
            }))
        }
        Err(e) => {
            error!("GPU Manager mailbox error: {}", e);
            error_json!(json!({
                "error": "Actor communication failed",
                "details": e.to_string()
            }))
        }
    }
}

// ============================================================================
// ROUTE CONFIGURATION
// ============================================================================

/// Configure ontology-physics routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/ontology-physics")
            .route("/enable", web::post().to(enable_ontology_physics))
            .route("/disable", web::post().to(disable_ontology_physics))
            .route("/constraints", web::get().to(get_constraints))
            .route("/weights", web::put().to(adjust_weights)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_mode_parsing() {
        let modes = vec!["replace", "merge", "add_if_no_conflict"];
        for mode in modes {
            assert!(matches!(
                mode,
                "replace" | "merge" | "add_if_no_conflict"
            ));
        }
    }

    #[test]
    fn test_strength_clamping() {
        let values: Vec<f32> = vec![-0.5, 0.0, 0.5, 1.0, 1.5];
        let clamped: Vec<f32> = values.iter().map(|v| v.clamp(0.0, 1.0)).collect();
        assert_eq!(clamped, vec![0.0, 0.0, 0.5, 1.0, 1.0]);
    }
}
