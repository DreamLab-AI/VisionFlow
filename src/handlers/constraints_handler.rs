use crate::actors::messages::{GetSettings, UpdateSettings};
use crate::app_state::AppState;
use crate::config::{ConstraintSystem, LegacyConstraintData};
use actix_web::{web, Error, HttpRequest, HttpResponse};
use log::{debug, error, info, warn};
use serde_json::{json, Value};
// Note: Constraint imports available but currently unused - keeping for future enhancements
use crate::handlers::settings_validation_fix::validate_constraint;

///
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/constraints")
            .route("/define", web::post().to(define_constraints))
            .route("/apply", web::post().to(apply_constraints))
            .route("/remove", web::post().to(remove_constraints))
            .route("/list", web::get().to(list_constraints))
            .route("/validate", web::post().to(validate_constraint_definition)),
    );
}

///
async fn define_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<ConstraintSystem>,
) -> Result<HttpResponse, Error> {
    let constraints = payload.into_inner();

    info!("Constraint definition request received");
    debug!("Constraints: {:?}", constraints);

    
    if let Err(e) = validate_constraint_system(&constraints) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid constraint system: {}", e)
        })));
    }

    
    let settings_update = json!({
        "visualisation": {
            "graphs": {
                "logseq": {
                    "physics": {
                        "computeMode": 2  
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

    
    match state
        .settings_addr
        .send(UpdateSettings {
            settings: app_settings,
        })
        .await
    {
        Ok(Ok(())) => {
            info!("Constraints defined successfully");

            
            if let Some(gpu_addr) = &state.gpu_compute_addr {
                info!("Sending constraints to GPU compute actor");

                
                use crate::actors::messages::UpdateConstraints;
                let gpu_constraints_json = serde_json::to_value(&constraints).unwrap_or_else(|e| {
                    error!("Failed to serialize constraints: {}", e);
                    json!({})
                });

                match gpu_addr
                    .send(UpdateConstraints {
                        constraint_data: gpu_constraints_json,
                    })
                    .await
                {
                    Ok(Ok(())) => {
                        info!("Successfully sent constraints to GPU compute actor");
                    }
                    Ok(Err(e)) => {
                        warn!("GPU compute actor failed to update constraints: {}", e);
                        
                    }
                    Err(e) => {
                        warn!("Failed to communicate with GPU compute actor: {}", e);
                        
                    }
                }
            } else {
                info!("GPU compute actor not available - constraints saved to settings only");
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

///
async fn apply_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let apply_request = payload.into_inner();

    info!("Constraint application request received");
    debug!(
        "Apply request: {}",
        serde_json::to_string_pretty(&apply_request).unwrap_or_default()
    );

    
    let constraint_type = apply_request
        .get("constraintType")
        .and_then(|v| v.as_str())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("constraintType is required"))?;

    let node_ids = apply_request
        .get("nodeIds")
        .and_then(|v| v.as_array())
        .ok_or_else(|| actix_web::error::ErrorBadRequest("nodeIds array is required"))?;

    if !["separation", "boundary", "alignment", "cluster"].contains(&constraint_type) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "constraintType must be separation, boundary, alignment, or cluster"
        })));
    }

    
    let nodes: Result<Vec<u32>, _> = node_ids
        .iter()
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

    
    let strength = apply_request
        .get("strength")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0) as f32;

    info!(
        "Constraint application recorded: {} to {} nodes with strength {}",
        constraint_type,
        nodes.len(),
        strength
    );

    Ok(HttpResponse::Ok().json(json!({
        "status": "Constraints recorded successfully",
        "constraintType": constraint_type,
        "nodeCount": nodes.len(),
        "strength": strength,
        "gpuAvailable": state.gpu_compute_addr.is_some(),
        "note": "Ready for GPU constraint processing integration"
    })))
}

///
async fn remove_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let remove_request = payload.into_inner();

    info!("Constraint removal request received");
    debug!(
        "Remove request: {}",
        serde_json::to_string_pretty(&remove_request).unwrap_or_default()
    );

    let constraint_type = remove_request
        .get("constraintType")
        .and_then(|v| v.as_str());

    let node_ids = remove_request.get("nodeIds").and_then(|v| v.as_array());

    
    let removal_count = node_ids.map(|arr| arr.len()).unwrap_or(0);

    info!(
        "Constraint removal recorded: {:?} affecting {} nodes",
        constraint_type, removal_count
    );

    Ok(HttpResponse::Ok().json(json!({
        "status": "Constraint removal recorded successfully",
        "removedCount": removal_count,
        "gpuAvailable": state.gpu_compute_addr.is_some(),
        "note": "Ready for GPU constraint removal integration"
    })))
}

///
async fn list_constraints(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("Constraint list request received");

    
    if let Some(gpu_addr) = &state.gpu_compute_addr {
        use crate::actors::messages::GetConstraints;
        match gpu_addr.send(GetConstraints).await {
            Ok(Ok(gpu_constraints)) => {
                info!(
                    "Retrieved {} constraints from GPU compute actor",
                    gpu_constraints.constraints.len()
                );
                return Ok(HttpResponse::Ok().json(json!({
                    "constraints": gpu_constraints,
                    "count": gpu_constraints.constraints.len(),
                    "data_source": "gpu_compute_actor",
                    "gpu_available": true
                })));
            }
            Ok(Err(e)) => {
                warn!("Failed to get constraints from GPU: {}", e);
            }
            Err(e) => {
                warn!("Failed to communicate with GPU compute actor: {}", e);
            }
        }
    }

    
    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            
            let mut constraints_list = Vec::new();

            
            let logseq_mode = settings.visualisation.graphs.logseq.physics.compute_mode;
            let visionflow_mode = settings
                .visualisation
                .graphs
                .visionflow
                .physics
                .compute_mode;

            if logseq_mode == 2 || visionflow_mode == 2 {
                constraints_list.push(json!({
                    "type": "physics_constraints",
                    "enabled": true,
                    "mode": "compute_mode_2",
                    "target_graphs": if logseq_mode == 2 && visionflow_mode == 2 {
                        vec!["logseq", "visionflow"]
                    } else if logseq_mode == 2 {
                        vec!["logseq"]
                    } else {
                        vec!["visionflow"]
                    }
                }));
            }

            Ok(HttpResponse::Ok().json(json!({
                "constraints": constraints_list,
                "count": constraints_list.len(),
                "data_source": "settings",
                "gpu_available": state.gpu_compute_addr.is_some(),
                "modes": {
                    "logseq_compute_mode": logseq_mode,
                    "visionflow_compute_mode": visionflow_mode
                }
            })))
        }
        _ => {
            error!("Failed to get settings for constraint listing");
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to retrieve constraint information",
                "constraints": [],
                "count": 0,
                "gpu_available": state.gpu_compute_addr.is_some()
            })))
        }
    }
}

///
async fn validate_constraint_definition(
    _req: HttpRequest,
    _state: web::Data<AppState>,
    payload: web::Json<LegacyConstraintData>,
) -> Result<HttpResponse, Error> {
    let constraint = payload.into_inner();

    info!("Constraint validation request received");
    debug!("Constraint to validate: {:?}", constraint);

    match validate_single_constraint(&constraint) {
        Ok(()) => Ok(HttpResponse::Ok().json(json!({
            "valid": true,
            "message": "Constraint definition is valid"
        }))),
        Err(e) => Ok(HttpResponse::BadRequest().json(json!({
            "valid": false,
            "error": e
        }))),
    }
}

///
fn validate_constraint_system(system: &ConstraintSystem) -> Result<(), String> {
    validate_single_constraint(&system.separation)?;
    validate_single_constraint(&system.boundary)?;
    validate_single_constraint(&system.alignment)?;
    validate_single_constraint(&system.cluster)?;

    Ok(())
}

///
fn validate_single_constraint(constraint: &LegacyConstraintData) -> Result<(), String> {
    
    let constraint_json = serde_json::to_value(constraint).map_err(|e| e.to_string())?;
    validate_constraint(&constraint_json)?;

    
    
    if constraint.constraint_type < 0 || constraint.constraint_type > 4 {
        return Err("constraint_type must be between 0 and 4".to_string());
    }

    
    if constraint.strength < 0.0 || constraint.strength > 10.0 {
        return Err("strength must be between 0.0 and 10.0".to_string());
    }

    
    match constraint.constraint_type {
        1 => {
            
            if constraint.param1 <= 0.0 {
                return Err("separation distance (param1) must be positive".to_string());
            }
        }
        2 => {
            
            if constraint.param1 <= 0.0 || constraint.param2 <= 0.0 {
                return Err("boundary dimensions (param1, param2) must be positive".to_string());
            }
        }
        3 => {
            
            if constraint.param1 < 0.0 || constraint.param1 > 360.0 {
                return Err(
                    "alignment angle (param1) must be between 0 and 360 degrees".to_string()
                );
            }
        }
        4 => {
            
            if constraint.param1.abs() > 1000.0 || constraint.param2.abs() > 1000.0 {
                return Err(
                    "cluster center coordinates must be within reasonable bounds".to_string(),
                );
            }
        }
        _ => {} 
    }

    Ok(())
}
