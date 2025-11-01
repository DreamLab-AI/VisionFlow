// src/handlers/inference_handler.rs
//! Inference HTTP Handlers
//!
//! REST API endpoints for ontology inference operations.

use actix_web::{web, HttpResponse, Responder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::application::inference_service::InferenceService;

///
#[derive(Debug, Deserialize)]
pub struct RunInferenceRequest {
    
    pub ontology_id: String,

    
    #[serde(default)]
    pub force: bool,
}

///
#[derive(Debug, Serialize)]
pub struct RunInferenceResponse {
    pub success: bool,
    pub ontology_id: String,
    pub inferred_axioms_count: usize,
    pub inference_time_ms: u64,
    pub reasoner_version: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

///
#[derive(Debug, Deserialize)]
pub struct BatchInferenceRequest {
    
    pub ontology_ids: Vec<String>,

    
    #[serde(default = "default_max_parallel")]
    pub max_parallel: usize,
}

fn default_max_parallel() -> usize {
    4
}

///
#[derive(Debug, Serialize)]
pub struct BatchInferenceResponse {
    pub success: bool,
    pub total_ontologies: usize,
    pub completed: usize,
    pub failed: usize,
    pub total_time_ms: u64,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub results: Option<Vec<RunInferenceResponse>>,
}

///
#[derive(Debug, Deserialize)]
pub struct ValidateOntologyRequest {
    pub ontology_id: String,
}

///
#[derive(Debug, Deserialize)]
pub struct GetExplanationRequest {
    pub axiom_id: String,
}

///
///
pub async fn run_inference(
    service: web::Data<Arc<RwLock<InferenceService>>>,
    req: web::Json<RunInferenceRequest>,
) -> impl Responder {
    info!("Inference request for ontology: {}", req.ontology_id);

    let service_lock = service.read().await;

    
    if req.force {
        service_lock.invalidate_cache(&req.ontology_id).await;
    }

    match service_lock.run_inference(&req.ontology_id).await {
        Ok(results) => {
            let response = RunInferenceResponse {
                success: true,
                ontology_id: req.ontology_id.clone(),
                inferred_axioms_count: results.inferred_axioms.len(),
                inference_time_ms: results.inference_time_ms,
                reasoner_version: results.reasoner_version,
                error: None,
            };

            HttpResponse::Ok().json(response)
        }
        Err(e) => {
            warn!("Inference failed: {:?}", e);

            let response = RunInferenceResponse {
                success: false,
                ontology_id: req.ontology_id.clone(),
                inferred_axioms_count: 0,
                inference_time_ms: 0,
                reasoner_version: String::new(),
                error: Some(format!("{:?}", e)),
            };

            HttpResponse::InternalServerError().json(response)
        }
    }
}

///
///
pub async fn batch_inference(
    service: web::Data<Arc<RwLock<InferenceService>>>,
    req: web::Json<BatchInferenceRequest>,
) -> impl Responder {
    info!("Batch inference request for {} ontologies", req.ontology_ids.len());
    let start = std::time::Instant::now();

    let service_lock = service.read().await;

    match service_lock.batch_inference(req.ontology_ids.clone()).await {
        Ok(results_map) => {
            let mut responses = Vec::new();
            let mut completed = 0;
            let mut failed = 0;

            for (ont_id, results) in results_map {
                completed += 1;
                responses.push(RunInferenceResponse {
                    success: true,
                    ontology_id: ont_id,
                    inferred_axioms_count: results.inferred_axioms.len(),
                    inference_time_ms: results.inference_time_ms,
                    reasoner_version: results.reasoner_version,
                    error: None,
                });
            }

            failed = req.ontology_ids.len() - completed;
            let total_time_ms = start.elapsed().as_millis() as u64;

            let response = BatchInferenceResponse {
                success: true,
                total_ontologies: req.ontology_ids.len(),
                completed,
                failed,
                total_time_ms,
                results: Some(responses),
            };

            HttpResponse::Ok().json(response)
        }
        Err(e) => {
            warn!("Batch inference failed: {:?}", e);

            let response = BatchInferenceResponse {
                success: false,
                total_ontologies: req.ontology_ids.len(),
                completed: 0,
                failed: req.ontology_ids.len(),
                total_time_ms: start.elapsed().as_millis() as u64,
                results: None,
            };

            HttpResponse::InternalServerError().json(response)
        }
    }
}

///
///
pub async fn validate_ontology(
    service: web::Data<Arc<RwLock<InferenceService>>>,
    req: web::Json<ValidateOntologyRequest>,
) -> impl Responder {
    info!("Validation request for ontology: {}", req.ontology_id);

    let service_lock = service.read().await;

    match service_lock.validate_ontology(&req.ontology_id).await {
        Ok(validation_result) => HttpResponse::Ok().json(validation_result),
        Err(e) => {
            warn!("Validation failed: {:?}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("{:?}", e)
            }))
        }
    }
}

///
///
pub async fn get_inference_results(
    service: web::Data<Arc<RwLock<InferenceService>>>,
    path: web::Path<String>,
) -> impl Responder {
    let ontology_id = path.into_inner();
    info!("Get inference results for: {}", ontology_id);

    let service_lock = service.read().await;

    
    match service_lock.run_inference(&ontology_id).await {
        Ok(results) => HttpResponse::Ok().json(results),
        Err(e) => {
            warn!("Failed to get results: {:?}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("{:?}", e)
            }))
        }
    }
}

///
///
pub async fn classify_ontology(
    service: web::Data<Arc<RwLock<InferenceService>>>,
    path: web::Path<String>,
) -> impl Responder {
    let ontology_id = path.into_inner();
    info!("Classification request for: {}", ontology_id);

    let service_lock = service.read().await;

    match service_lock.classify_ontology(&ontology_id).await {
        Ok(classification) => HttpResponse::Ok().json(classification),
        Err(e) => {
            warn!("Classification failed: {:?}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("{:?}", e)
            }))
        }
    }
}

///
///
pub async fn get_consistency_report(
    service: web::Data<Arc<RwLock<InferenceService>>>,
    path: web::Path<String>,
) -> impl Responder {
    let ontology_id = path.into_inner();
    info!("Consistency report request for: {}", ontology_id);

    let service_lock = service.read().await;

    match service_lock.get_consistency_report(&ontology_id).await {
        Ok(report) => HttpResponse::Ok().json(report),
        Err(e) => {
            warn!("Consistency check failed: {:?}", e);
            HttpResponse::InternalServerError().json(serde_json::json!({
                "success": false,
                "error": format!("{:?}", e)
            }))
        }
    }
}

///
///
pub async fn invalidate_cache(
    service: web::Data<Arc<RwLock<InferenceService>>>,
    path: web::Path<String>,
) -> impl Responder {
    let ontology_id = path.into_inner();
    info!("Cache invalidation request for: {}", ontology_id);

    let service_lock = service.read().await;
    service_lock.invalidate_cache(&ontology_id).await;

    HttpResponse::Ok().json(serde_json::json!({
        "success": true,
        "message": "Cache invalidated"
    }))
}

///
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/inference")
            .route("/run", web::post().to(run_inference))
            .route("/batch", web::post().to(batch_inference))
            .route("/validate", web::post().to(validate_ontology))
            .route("/results/{ontology_id}", web::get().to(get_inference_results))
            .route("/classify/{ontology_id}", web::get().to(classify_ontology))
            .route("/consistency/{ontology_id}", web::get().to(get_consistency_report))
            .route("/cache/{ontology_id}", web::delete().to(invalidate_cache)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    
}
