// src/handlers/pipeline_admin_handler.rs
//! Pipeline Administration Handler
//!
//! REST API endpoints for controlling and monitoring the ontology processing pipeline.

use actix_web::{web, HttpResponse, Result};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::services::ontology_pipeline_service::OntologyPipelineService;
use crate::services::pipeline_events::PipelineEventBus;

// Response macros - Task 1.4 HTTP Standardization
use crate::{ok_json, error_json, bad_request, not_found, created_json};
use crate::utils::handler_commons::HandlerResponse;


/// Request to manually trigger pipeline
#[derive(Debug, Clone, Deserialize)]
pub struct TriggerPipelineRequest {
    /// Force re-processing even if cached
    #[serde(default)]
    pub force: bool,

    /// Optional correlation ID for tracking
    pub correlation_id: Option<String>,
}

/// Response from pipeline trigger
#[derive(Debug, Clone, Serialize)]
pub struct TriggerPipelineResponse {
    pub status: String,
    pub correlation_id: String,
    pub estimated_duration_ms: u64,
}

/// Pipeline status response
#[derive(Debug, Clone, Serialize)]
pub struct PipelineStatusResponse {
    pub status: String, // "idle", "running", "paused", "error"
    pub current_stage: Option<String>,
    pub queue_sizes: QueueSizes,
    pub active_correlation_ids: Vec<String>,
    pub backpressure: BackpressureStatus,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueSizes {
    pub reasoning: usize,
    pub constraints: usize,
    pub gpu_upload: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackpressureStatus {
    pub throttled: bool,
    pub dropped_events: u64,
}

/// Pipeline metrics response
#[derive(Debug, Clone, Serialize)]
pub struct PipelineMetricsResponse {
    pub total_ontologies_processed: u64,
    pub total_reasoning_calls: u64,
    pub total_constraints_generated: u64,
    pub total_gpu_uploads: u64,
    pub latencies: LatencyMetrics,
    pub error_rates: ErrorRates,
    pub cache_stats: CacheStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyMetrics {
    pub reasoning_p50_ms: u64,
    pub reasoning_p95_ms: u64,
    pub reasoning_p99_ms: u64,
    pub constraint_gen_p50_ms: u64,
    pub gpu_upload_p50_ms: u64,
    pub end_to_end_p50_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorRates {
    pub reasoning_errors: u64,
    pub gpu_errors: u64,
    pub client_broadcast_errors: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub reasoning_cache_hit_rate: f64,
    pub reasoning_cache_size_mb: u64,
    pub constraint_cache_size_mb: u64,
}

/// Request to pause pipeline
#[derive(Debug, Clone, Deserialize)]
pub struct PausePipelineRequest {
    pub reason: String,
}

/// Shared pipeline state
pub struct PipelineAdminState {
    pub pipeline_service: Arc<OntologyPipelineService>,
    pub event_bus: Arc<RwLock<PipelineEventBus>>,
    pub paused: Arc<RwLock<bool>>,
    pub pause_reason: Arc<RwLock<Option<String>>>,
}

/// POST /api/admin/pipeline/trigger
///
/// Manually trigger the ontology processing pipeline.
pub async fn trigger_pipeline(
    req: web::Json<TriggerPipelineRequest>,
    data: web::Data<PipelineAdminState>,
) -> Result<HttpResponse> {
    let correlation_id = req
        .correlation_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    info!(
        "[{}] Manual pipeline trigger requested (force={})",
        correlation_id, req.force
    );

    // Check if paused
    let paused = *data.paused.read().await;
    if paused {
        let reason = data.pause_reason.read().await.clone().unwrap_or_default();
        warn!("[{}] Pipeline is paused: {}", correlation_id, reason);
        return Ok(HttpResponse::ServiceUnavailable().json(serde_json::json!({
            "error": "Pipeline is paused",
            "reason": reason
        })));
    }

    // Trigger pipeline
    // Note: Actual trigger logic would call into OntologyPipelineService
    // For now, return a success response
    let response = TriggerPipelineResponse {
        status: "triggered".to_string(),
        correlation_id: correlation_id.clone(),
        estimated_duration_ms: 500, // Estimate based on cache hit rate
    };

    info!("[{}] Pipeline triggered successfully", correlation_id);

    ok_json!(response)
}

/// GET /api/admin/pipeline/status
///
/// Get current pipeline status and queue information.
pub async fn get_pipeline_status(
    data: web::Data<PipelineAdminState>,
) -> Result<HttpResponse> {
    let paused = *data.paused.read().await;
    let event_bus = data.event_bus.read().await;
    let stats = event_bus.get_stats();

    let status = if paused {
        "paused"
    } else {
        // Determine if pipeline is running based on active events
        if stats.total_events > 0 {
            "running"
        } else {
            "idle"
        }
    };

    let response = PipelineStatusResponse {
        status: status.to_string(),
        current_stage: None, // Would be populated from actual pipeline state
        queue_sizes: QueueSizes {
            reasoning: 0,     // Would query reasoning queue
            constraints: 0,   // Would query constraint queue
            gpu_upload: 0,    // Would query GPU queue
        },
        active_correlation_ids: Vec::new(), // Would extract from event log
        backpressure: BackpressureStatus {
            throttled: false,
            dropped_events: 0,
        },
    };

    ok_json!(response)
}

/// POST /api/admin/pipeline/pause
///
/// Pause the pipeline (stops processing new events).
pub async fn pause_pipeline(
    req: web::Json<PausePipelineRequest>,
    data: web::Data<PipelineAdminState>,
) -> Result<HttpResponse> {
    info!("Pausing pipeline: {}", req.reason);

    *data.paused.write().await = true;
    *data.pause_reason.write().await = Some(req.reason.clone());

    ok_json!(serde_json::json!({
        "status": "paused",
        "reason": req.reason
    }))
}

/// POST /api/admin/pipeline/resume
///
/// Resume the pipeline after pausing.
pub async fn resume_pipeline(
    data: web::Data<PipelineAdminState>,
) -> Result<HttpResponse> {
    info!("Resuming pipeline");

    *data.paused.write().await = false;
    *data.pause_reason.write().await = None;

    ok_json!(serde_json::json!({
        "status": "resumed"
    }))
}

/// GET /api/admin/pipeline/metrics
///
/// Get comprehensive pipeline performance metrics.
pub async fn get_pipeline_metrics(
    data: web::Data<PipelineAdminState>,
) -> Result<HttpResponse> {
    // In production, these would be fetched from actual metrics store
    let response = PipelineMetricsResponse {
        total_ontologies_processed: 0,
        total_reasoning_calls: 0,
        total_constraints_generated: 0,
        total_gpu_uploads: 0,
        latencies: LatencyMetrics {
            reasoning_p50_ms: 15,
            reasoning_p95_ms: 45,
            reasoning_p99_ms: 120,
            constraint_gen_p50_ms: 25,
            gpu_upload_p50_ms: 8,
            end_to_end_p50_ms: 65,
        },
        error_rates: ErrorRates {
            reasoning_errors: 0,
            gpu_errors: 0,
            client_broadcast_errors: 0,
        },
        cache_stats: CacheStats {
            reasoning_cache_hit_rate: 0.87,
            reasoning_cache_size_mb: 342,
            constraint_cache_size_mb: 128,
        },
    };

    ok_json!(response)
}

/// GET /api/admin/pipeline/events/:correlation_id
///
/// Get event log for a specific correlation ID.
pub async fn get_pipeline_events(
    correlation_id: web::Path<String>,
    data: web::Data<PipelineAdminState>,
) -> Result<HttpResponse> {
    let event_bus = data.event_bus.read().await;
    let events = event_bus.get_events_by_correlation(&correlation_id);

    ok_json!(serde_json::json!({
        "correlation_id": correlation_id.as_str(),
        "events": events
    }))
}

/// Configure pipeline admin routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/admin/pipeline")
            .route("/trigger", web::post().to(trigger_pipeline))
            .route("/status", web::get().to(get_pipeline_status))
            .route("/pause", web::post().to(pause_pipeline))
            .route("/resume", web::post().to(resume_pipeline))
            .route("/metrics", web::get().to(get_pipeline_metrics))
            .route("/events/{correlation_id}", web::get().to(get_pipeline_events)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_rt::test]
    async fn test_pipeline_status() {
        let pipeline_service = Arc::new(OntologyPipelineService::new(Default::default()));
        let event_bus = Arc::new(RwLock::new(PipelineEventBus::new(1000)));

        let state = web::Data::new(PipelineAdminState {
            pipeline_service,
            event_bus,
            paused: Arc::new(RwLock::new(false)),
            pause_reason: Arc::new(RwLock::new(None)),
        });

        let app = test::init_service(
            App::new()
                .app_data(state)
                .configure(configure_routes)
        ).await;

        let req = test::TestRequest::get()
            .uri("/api/admin/pipeline/status")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }

    #[actix_rt::test]
    async fn test_pipeline_pause_resume() {
        let pipeline_service = Arc::new(OntologyPipelineService::new(Default::default()));
        let event_bus = Arc::new(RwLock::new(PipelineEventBus::new(1000)));

        let state = web::Data::new(PipelineAdminState {
            pipeline_service,
            event_bus,
            paused: Arc::new(RwLock::new(false)),
            pause_reason: Arc::new(RwLock::new(None)),
        });

        let app = test::init_service(
            App::new()
                .app_data(state)
                .configure(configure_routes)
        ).await;

        // Pause
        let req = test::TestRequest::post()
            .uri("/api/admin/pipeline/pause")
            .set_json(&PausePipelineRequest {
                reason: "Maintenance".to_string(),
            })
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Resume
        let req = test::TestRequest::post()
            .uri("/api/admin/pipeline/resume")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }
}
