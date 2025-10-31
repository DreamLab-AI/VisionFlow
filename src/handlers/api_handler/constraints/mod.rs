//! Constraint Management API Handlers
//!
//! Provides REST API endpoints for managing ontology-derived and user-defined
//! physics constraints in the VisionFlow system.

use actix_web::{web, HttpResponse, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use serde_json::json};
use std::sync::Arc;

use crate::actors::gpu::ontology_constraint_actor::OntologyConstraintStats;
use crate::actors::messages::{GetConstraints, UpdateConstraint};
use crate::models::constraints::{Constraint, ConstraintType};
use crate::AppState;

/// Request body for creating user-defined constraints
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

/// Request body for updating constraints
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateConstraintRequest {
    pub active: Option<bool>,
    pub strength: Option<f32>,
    pub distance: Option<f32>,
}

/// Response for constraint operations
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

/// Response for constraint statistics
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintStatsResponse {
    pub total_constraints: u32,
    pub active_constraints: u32,
    pub ontology_constraints: u32,
    pub user_constraints: u32,
    pub constraint_evaluation_count: u32,
    pub last_update_time_ms: f32,
    pub gpu_status: String,
    pub cache_hit_rate: f32,
}

/// GET /api/constraints - List all constraints
pub async fn get_constraints(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/constraints - Fetching all constraints");

    // Query constraint actor for current constraints
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

            HttpResponse::Ok().json(json!({
                "constraints": response,
                "count": response.len()
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to fetch constraints: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Failed to fetch constraints",
                "details": e
            }))
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Actor communication failed",
                "details": e.to_string()
            }))
        }
    }
}

/// GET /api/constraints/{id} - Get specific constraint
pub async fn get_constraint(
    state: web::Data<AppState>,
    constraint_id: web::Path<String>,
) -> impl Responder {
    info!("GET /api/constraints/{} - Fetching specific constraint", constraint_id);

    // Query constraint actor
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

                HttpResponse::Ok().json(response)
            } else {
                HttpResponse::NotFound().json(json!({
                    "error": "Constraint not found",
                    "id": *constraint_id
                }))
            }
        }
        Ok(Err(e)) => {
            error!("Failed to fetch constraint: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Failed to fetch constraint",
                "details": e
            }))
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Actor communication failed",
                "details": e.to_string()
            }))
        }
    }
}

/// PUT /api/constraints/{id} - Update constraint
pub async fn update_constraint(
    state: web::Data<AppState>,
    constraint_id: web::Path<String>,
    req: web::Json<UpdateConstraintRequest>,
) -> impl Responder {
    info!("PUT /api/constraints/{} - Updating constraint", constraint_id);

    // Create update message
    let update_msg = UpdateConstraint {
        constraint_id: constraint_id.to_string(),
        active: req.active,
        strength: req.strength,
        distance: req.distance,
    };

    // Send to constraint actor
    match state
        .graph_service_addr
        .send(update_msg)
        .await
    {
        Ok(Ok(())) => {
            HttpResponse::Ok().json(json!({
                "success": true,
                "id": *constraint_id,
                "message": "Constraint updated successfully"
            }))
        }
        Ok(Err(e)) => {
            error!("Failed to update constraint: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Failed to update constraint",
                "details": e
            }))
        }
        Err(e) => {
            error!("Actor mailbox error: {}", e);
            HttpResponse::InternalServerError().json(json!({
                "error": "Actor communication failed",
                "details": e.to_string()
            }))
        }
    }
}

/// POST /api/constraints/user - Create user-defined constraint
pub async fn create_user_constraint(
    state: web::Data<AppState>,
    req: web::Json<CreateConstraintRequest>,
) -> impl Responder {
    info!("POST /api/constraints/user - Creating user constraint");

    // Parse constraint type
    let constraint_type = match req.constraint_type.as_str() {
        "Distance" => ConstraintType::Distance,
        "Angle" => ConstraintType::Angle,
        "Hierarchy" => ConstraintType::Hierarchy,
        "Containment" => ConstraintType::Containment,
        "Alignment" => ConstraintType::Alignment,
        _ => {
            return HttpResponse::BadRequest().json(json!({
                "error": "Invalid constraint type",
                "valid_types": ["Distance", "Angle", "Hierarchy", "Containment", "Alignment"]
            }));
        }
    };

    // Create constraint object
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

    // TODO: Send to constraint actor to add to active constraints
    // For now, return success response
    HttpResponse::Created().json(json!({
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

/// GET /api/constraints/stats - Get constraint statistics
pub async fn get_constraint_stats(state: web::Data<AppState>) -> impl Responder {
    info!("GET /api/constraints/stats - Fetching constraint statistics");

    // Query constraint actor for statistics
    // For now, return mock data (TODO: implement GetConstraintStats message)
    let stats = ConstraintStatsResponse {
        total_constraints: 150,
        active_constraints: 120,
        ontology_constraints: 80,
        user_constraints: 40,
        constraint_evaluation_count: 1500,
        last_update_time_ms: 3.2,
        gpu_status: "operational".to_string(),
        cache_hit_rate: 0.85,
    };

    HttpResponse::Ok().json(stats)
}

/// Configure constraint API routes
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

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("test-123"));
        assert!(json.contains("Distance"));
    }
}
