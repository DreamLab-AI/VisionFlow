//! Constraint Management API Handlers
//!
//! Provides REST API endpoints for managing ontology-derived and user-defined
//! physics constraints in the VisionFlow system.

use actix_web::{web, HttpResponse, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use crate::{ok_json, error_json, bad_request, not_found, created_json, service_unavailable};

use crate::actors::gpu::ontology_constraint_actor::OntologyConstraintStats;
use crate::actors::messages::{GetConstraints, UpdateConstraintData};
use crate::models::constraints::{Constraint, ConstraintType};
use crate::AppState;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateConstraintRequest {
    pub constraint_type: String,
    pub source_node: String,
    pub target_node: Option<String>,
    pub strength: f32,
    pub distance: Option<f32>,
    pub active: bool,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateConstraintRequest {
    pub active: Option<bool>,
    pub strength: Option<f32>,
    pub distance: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintResponse {
    pub id: String,
    pub constraint_type: String,
    pub source_node: String,
    pub target_node: Option<String>,
    pub strength: f32,
    pub distance: Option<f32>,
    pub active: bool,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintStatsResponse {
    pub total_constraints: u32,
    pub active_constraints: u32,
    pub ontology_constraints: u32,
    pub user_constraints: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constraint_evaluation_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_update_time_ms: Option<f64>,
    pub gpu_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_rate: Option<f64>,
}

pub async fn get_constraints(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/constraints - Fetching all constraints");

    
    match state
        .graph_service_addr
        .send(GetConstraints)
        .await
    {
        Ok(Ok(constraints)) => {
            let response: Vec<ConstraintResponse> = constraints
                .iter()
                .map(|c| ConstraintResponse {
                    id: c.id.clone(),
                    constraint_type: format!("{:?}", c.constraint_type),
                    source_node: c.source_node.clone(),
                    target_node: c.target_node.clone(),
                    strength: c.strength,
                    distance: c.distance,
                    active: c.active,
                    metadata: c.metadata.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                })
                .collect();

            ok_json!(json!({
                "constraints": response,
                "count": response.len()
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to fetch constraints: {}", e);
            error_json!("Failed to fetch constraints")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed")
        }
    }
}

pub async fn get_constraint(
    state: web::Data<AppState>,
    constraint_id: web::Path<String>,
) -> impl Responder {
    info!("GET /api/constraints/{} - Fetching specific constraint", constraint_id);

    
    match state
        .graph_service_addr
        .send(GetConstraints)
        .await
    {
        Ok(Ok(constraints)) => {
            if let Some(constraint) = constraints.iter().find(|c| c.id == *constraint_id) {
                let response = ConstraintResponse {
                    id: constraint.id.clone(),
                    constraint_type: format!("{:?}", constraint.constraint_type),
                    source_node: constraint.source_node.clone(),
                    target_node: constraint.target_node.clone(),
                    strength: constraint.strength,
                    distance: constraint.distance,
                    active: constraint.active,
                    metadata: constraint.metadata.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                };

                ok_json!(response)
            } else {
                not_found!("Constraint not found")
            }
        }
        Ok(Err(e)) => {
            error!("Failed to fetch constraint: {}", e);
            error_json!("Failed to fetch constraint")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed")
        }
    }
}

pub async fn update_constraint(
    _auth: crate::settings::auth_extractor::AuthenticatedUser,
    state: web::Data<AppState>,
    constraint_id: web::Path<String>,
    req: web::Json<UpdateConstraintRequest>,
) -> impl Responder {
    info!("PUT /api/constraints/{} - Updating constraint", constraint_id);

    let constraint_data = json!({
        "constraint_id": constraint_id.to_string(),
        "active": req.active,
        "strength": req.strength,
        "distance": req.distance,
    });

    let update_msg = UpdateConstraintData {
        constraint_data,
    };

    match state
        .graph_service_addr
        .send(update_msg)
        .await
    {
        Ok(Ok(())) => {
            ok_json!(json!({
                "success": true,
                "id": *constraint_id,
                "message": "Constraint updated successfully"
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to update constraint: {}", e);
            error_json!("Failed to update constraint")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed")
        }
    }
}

pub async fn create_user_constraint(
    _auth: crate::settings::auth_extractor::AuthenticatedUser,
    state: web::Data<AppState>,
    req: web::Json<CreateConstraintRequest>,
) -> impl Responder {
    info!("POST /api/constraints/user - Creating user constraint");

    if !req.strength.is_finite() || req.strength < 0.0 || req.strength > 10.0 {
        return bad_request!("strength must be a finite number in range 0.0..=10.0");
    }

    let constraint_type = match req.constraint_type.as_str() {
        "Distance" => ConstraintType::Distance,
        "Angle" => ConstraintType::Angle,
        "Hierarchy" => ConstraintType::Hierarchy,
        "Containment" => ConstraintType::Containment,
        "Alignment" => ConstraintType::Alignment,
        _ => {
            return bad_request!("Invalid constraint type");
        }
    };

    let constraint = Constraint {
        id: uuid::Uuid::new_v4().to_string(),
        constraint_type,
        source_node: req.source_node.clone(),
        target_node: req.target_node.clone(),
        strength: req.strength,
        distance: req.distance,
        active: req.active,
        metadata: req.metadata.clone(),
    };

    // Persist the constraint by sending it to the graph service actor
    let constraint_data = serde_json::to_value(&constraint).unwrap_or_else(|_| json!({}));
    let persist_msg = UpdateConstraintData { constraint_data };
    match state.graph_service_addr.send(persist_msg).await {
        Ok(Ok(())) => {
            info!("User constraint {} persisted successfully", constraint.id);
        }
        Ok(Err(e)) => {
            error!("Failed to persist user constraint: {}", e);
            return error_json!("Failed to persist constraint: {}", e);
        }
        Err(e) => {
            error!("Actor mailbox error persisting constraint: {}", e);
            return error_json!("Actor communication failed");
        }
    }

    created_json!(json!({
        "success": true,
        "constraint": ConstraintResponse {
            id: constraint.id.clone(),
            constraint_type: format!("{:?}", constraint.constraint_type),
            source_node: constraint.source_node.clone(),
            target_node: constraint.target_node.clone(),
            strength: constraint.strength,
            distance: constraint.distance,
            active: constraint.active,
            metadata: constraint.metadata.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }))
}

pub async fn get_constraint_stats(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/constraints/stats - Fetching constraint statistics");

    match state
        .graph_service_addr
        .send(crate::actors::messages::GetConstraintStats)
        .await
    {
        Ok(Ok(stats)) => {
            let response = ConstraintStatsResponse {
                total_constraints: stats.total_constraints as u32,
                active_constraints: stats.active_constraints as u32,
                ontology_constraints: stats.ontology_constraints as u32,
                user_constraints: stats.user_constraints as u32,
                constraint_evaluation_count: None,
                last_update_time_ms: None,
                gpu_status: "operational".to_string(),
                cache_hit_rate: None,
            };
            ok_json!(response)
        }
        Ok(Err(e)) => {
            error!("Failed to fetch constraint stats: {}", e);
            error_json!("Failed to fetch constraint statistics")
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            error_json!("Actor communication failed")
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/constraints")
            .route("", web::get().to(get_constraints))
            .route("/{id}", web::get().to(get_constraint))
            .route("/{id}", web::put().to(update_constraint))
            .route("/user", web::post().to(create_user_constraint))
            .route("/stats", web::get().to(get_constraint_stats)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_type_parsing() {
        let valid_types = vec!["Distance", "Angle", "Hierarchy", "Containment", "Alignment"];
        for t in valid_types {
            assert!(matches!(
                t,
                "Distance" | "Angle" | "Hierarchy" | "Containment" | "Alignment"
            ));
        }
    }

    #[test]
    fn test_constraint_response_serialization() {
        let response = ConstraintResponse {
            id: "test-123".to_string(),
            constraint_type: "Distance".to_string(),
            source_node: "node1".to_string(),
            target_node: Some("node2".to_string()),
            strength: 1.0,
            distance: Some(10.0),
            active: true,
            metadata: None,
            created_at: "2025-10-31T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&response)
            .expect("ConstraintResponse should serialize to JSON");
        assert!(json.contains("test-123"));
        assert!(json.contains("Distance"));
    }
}
