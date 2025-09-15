use actix_web::{web, Error, HttpResponse, HttpRequest};
use crate::app_state::AppState;
use crate::actors::messages::{GetSettings, UpdateSettings};
use log::{info, error, debug};
use serde_json::{json, Value};
use crate::config::{ConstraintSystem, LegacyConstraintData};
use crate::models::constraints::{Constraint, ConstraintSet, ConstraintKind};
use crate::handlers::settings_validation_fix::validate_constraint;

/// Configure constraint-specific routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/constraints")
            .route("/define", web::post().to(define_constraints))
            .route("/apply", web::post().to(apply_constraints))
            .route("/remove", web::post().to(remove_constraints))
            .route("/list", web::get().to(list_constraints))
            .route("/validate", web::post().to(validate_constraint_definition))
    );
}

/// Define new constraints
async fn define_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<ConstraintSystem>,
) -> Result<HttpResponse, Error> {
    let constraints = payload.into_inner();
    
    info!("Constraint definition request received");
    debug!("Constraints: {:?}", constraints);
    
    // Validate all constraints
    if let Err(e) = validate_constraint_system(&constraints) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid constraint system: {}", e)
        })));
    }
    
    // Enable constraints mode in physics settings
    let settings_update = json!({
        "visualisation": {
            "graphs": {
                "logseq": {
                    "physics": {
                        "computeMode": 2  // Constraints mode
                    }
                },
                "visionflow": {
                    "physics": {
                        "computeMode": 2
                    }
                }
            }
        }
    });
    
    // Get and update settings
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
    
    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge constraint settings: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to update constraint settings: {}", e)
        })));
    }
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings }).await {
        Ok(Ok(())) => {
            info!("Constraints defined successfully");
            
            // TODO: Send constraints to GPU when GPU constraint system is implemented
            if state.gpu_compute_addr.is_some() {
                info!("GPU available for future constraint processing");
            } else {
                info!("GPU constraints will be available when GPU actor is ready");
            }
            
            Ok(HttpResponse::Ok().json(json!({
                "status": "Constraints defined successfully",
                "constraints": constraints
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save constraint settings: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save constraint settings: {}", e)
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

/// Apply constraints to specific nodes
async fn apply_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let apply_request = payload.into_inner();
    
    info!("Constraint application request received");
    debug!("Apply request: {}", serde_json::to_string_pretty(&apply_request).unwrap_or_default());
    
    // Validate application request
    let constraint_type = apply_request.get("constraintType")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            actix_web::error::ErrorBadRequest("constraintType is required")
        })?;
    
    let node_ids = apply_request.get("nodeIds")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            actix_web::error::ErrorBadRequest("nodeIds array is required")
        })?;
    
    if !["separation", "boundary", "alignment", "cluster"].contains(&constraint_type) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "constraintType must be separation, boundary, alignment, or cluster"
        })));
    }
    
    // Convert node IDs
    let nodes: Result<Vec<u32>, _> = node_ids.iter()
        .map(|v| v.as_u64().map(|n| n as u32))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| "Invalid node IDs");
    
    let nodes = match nodes {
        Ok(n) => n,
        Err(e) => {
            return Ok(HttpResponse::BadRequest().json(json!({
                "error": e
            })));
        }
    };
    
    // For now, store constraint application request
    let strength = apply_request.get("strength")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;
    
    info!("Constraint application recorded: {} to {} nodes with strength {}", 
        constraint_type, nodes.len(), strength);
    
    Ok(HttpResponse::Ok().json(json!({
        "status": "Constraints recorded successfully",
        "constraintType": constraint_type,
        "nodeCount": nodes.len(),
        "strength": strength,
        "gpuAvailable": state.gpu_compute_addr.is_some(),
        "note": "Ready for GPU constraint processing integration"
    })))
}

/// Remove constraints from nodes
async fn remove_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let remove_request = payload.into_inner();
    
    info!("Constraint removal request received");
    debug!("Remove request: {}", serde_json::to_string_pretty(&remove_request).unwrap_or_default());
    
    let constraint_type = remove_request.get("constraintType")
        .and_then(|v| v.as_str());
    
    let node_ids = remove_request.get("nodeIds")
        .and_then(|v| v.as_array());
    
    // For now, record constraint removal request
    let removal_count = node_ids.map(|arr| arr.len()).unwrap_or(0);
    
    info!("Constraint removal recorded: {:?} affecting {} nodes", 
        constraint_type, removal_count);
    
    Ok(HttpResponse::Ok().json(json!({
        "status": "Constraint removal recorded successfully",
        "removedCount": removal_count,
        "gpuAvailable": state.gpu_compute_addr.is_some(),
        "note": "Ready for GPU constraint removal integration"
    })))
}

/// List all active constraints
async fn list_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("Constraint list request received");
    
    // Return empty constraints list for now - ready for GPU integration
    Ok(HttpResponse::Ok().json(json!({
        "constraints": [],
        "count": 0,
        "gpuAvailable": state.gpu_compute_addr.is_some(),
        "note": "Ready for GPU constraint system integration"
    })))
}

/// Validate constraint definition
async fn validate_constraint_definition(
    _req: HttpRequest,
    _state: web::Data<AppState>,
    payload: web::Json<LegacyConstraintData>,
) -> Result<HttpResponse, Error> {
    let constraint = payload.into_inner();

    info!("Constraint validation request received");
    debug!("Constraint to validate: {:?}", constraint);

    match validate_single_constraint(&constraint) {
        Ok(()) => {
            Ok(HttpResponse::Ok().json(json!({
                "valid": true,
                "message": "Constraint definition is valid"
            })))
        }
        Err(e) => {
            Ok(HttpResponse::BadRequest().json(json!({
                "valid": false,
                "error": e
            })))
        }
    }
}

/// Validate a complete constraint system
fn validate_constraint_system(system: &ConstraintSystem) -> Result<(), String> {
    validate_single_constraint(&system.separation)?;
    validate_single_constraint(&system.boundary)?;
    validate_single_constraint(&system.alignment)?;
    validate_single_constraint(&system.cluster)?;
    
    Ok(())
}

/// Validate a single constraint
fn validate_single_constraint(constraint: &LegacyConstraintData) -> Result<(), String> {
    // Use comprehensive validation that prevents GPU errors
    let constraint_json = serde_json::to_value(constraint).map_err(|e| e.to_string())?;
    validate_constraint(&constraint_json)?;

    // Additional type-specific validation
    // Validate constraint type
    if constraint.constraint_type < 0 || constraint.constraint_type > 4 {
        return Err("constraint_type must be between 0 and 4".to_string());
    }

    // Validate strength
    if constraint.strength < 0.0 || constraint.strength > 10.0 {
        return Err("strength must be between 0.0 and 10.0".to_string());
    }
    
    // Validate parameters based on constraint type
    match constraint.constraint_type {
        1 => { // Separation constraint
            if constraint.param1 <= 0.0 {
                return Err("separation distance (param1) must be positive".to_string());
            }
        }
        2 => { // Boundary constraint
            if constraint.param1 <= 0.0 || constraint.param2 <= 0.0 {
                return Err("boundary dimensions (param1, param2) must be positive".to_string());
            }
        }
        3 => { // Alignment constraint
            if constraint.param1 < 0.0 || constraint.param1 > 360.0 {
                return Err("alignment angle (param1) must be between 0 and 360 degrees".to_string());
            }
        }
        4 => { // Cluster constraint
            if constraint.param1.abs() > 1000.0 || constraint.param2.abs() > 1000.0 {
                return Err("cluster center coordinates must be within reasonable bounds".to_string());
            }
        }
        _ => {} // Type 0 (none) requires no validation
    }
    
    Ok(())
}