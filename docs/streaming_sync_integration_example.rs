// Example: Integrating StreamingSyncService into VisionFlow Application
//
// This file demonstrates how to integrate the streaming sync service
// into your application state and create API endpoints for it.

use crate::services::streaming_sync_service::{StreamingSyncService, SyncProgress, SyncStatistics};
use actix_web::{web, HttpResponse, Result as ActixResult};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

// ============================================================================
// EXAMPLE 1: Integration into AppState
// ============================================================================

pub struct AppState {
    pub streaming_sync_service: Arc<StreamingSyncService>,
    // Progress tracking for current sync operation
    pub sync_progress: Arc<RwLock<Option<SyncProgress>>>,
    // ... other services
}

impl AppState {
    pub fn new(
        content_api: Arc<crate::services::github::content_enhanced::EnhancedContentAPI>,
        kg_repo: Arc<crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository>,
        onto_repo: Arc<crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository>,
    ) -> Self {
        // Create streaming sync service with 8 workers
        let streaming_sync_service = Arc::new(StreamingSyncService::new(
            content_api,
            kg_repo,
            onto_repo,
            Some(8), // 8 concurrent workers
        ));

        Self {
            streaming_sync_service,
            sync_progress: Arc::new(RwLock::new(None)),
        }
    }
}

// ============================================================================
// EXAMPLE 2: API Endpoint - Start Sync
// ============================================================================

/// POST /api/sync/streaming/start
/// Start a streaming sync operation
pub async fn start_streaming_sync(
    app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    // Create progress channel
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();

    // Clone service and configure progress channel
    let mut service = (*app_state.streaming_sync_service).clone();
    service.set_progress_channel(progress_tx);

    // Clone state for progress updates
    let sync_progress = Arc::clone(&app_state.sync_progress);

    // Spawn background task for progress monitoring
    tokio::spawn(async move {
        while let Some(progress) = progress_rx.recv().await {
            let mut guard = sync_progress.write().await;
            *guard = Some(progress);
        }
    });

    // Spawn sync operation
    let sync_handle = tokio::spawn(async move {
        service.sync_graphs_streaming().await
    });

    // Don't wait for completion - return immediately
    Ok(HttpResponse::Accepted().json(serde_json::json!({
        "status": "started",
        "message": "Streaming sync started in background"
    })))
}

// ============================================================================
// EXAMPLE 3: API Endpoint - Get Progress
// ============================================================================

/// GET /api/sync/streaming/progress
/// Get current sync progress
pub async fn get_sync_progress(
    app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let guard = app_state.sync_progress.read().await;

    match &*guard {
        Some(progress) => {
            let percentage = if progress.files_total > 0 {
                (progress.files_processed as f64 / progress.files_total as f64) * 100.0
            } else {
                0.0
            };

            Ok(HttpResponse::Ok().json(serde_json::json!({
                "status": "in_progress",
                "progress": {
                    "total_files": progress.files_total,
                    "processed": progress.files_processed,
                    "succeeded": progress.files_succeeded,
                    "failed": progress.files_failed,
                    "current_file": progress.current_file,
                    "percentage": format!("{:.1}%", percentage),
                    "metrics": {
                        "nodes_saved": progress.kg_nodes_saved,
                        "edges_saved": progress.kg_edges_saved,
                        "classes_saved": progress.onto_classes_saved,
                        "properties_saved": progress.onto_properties_saved,
                        "axioms_saved": progress.onto_axioms_saved,
                    },
                    "errors": progress.errors,
                }
            })))
        }
        None => {
            Ok(HttpResponse::Ok().json(serde_json::json!({
                "status": "idle",
                "message": "No sync operation in progress"
            })))
        }
    }
}

// ============================================================================
// EXAMPLE 4: API Endpoint - Synchronous Sync (Wait for Completion)
// ============================================================================

/// POST /api/sync/streaming/run
/// Run streaming sync and wait for completion
pub async fn run_streaming_sync(
    app_state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    log::info!("Starting synchronous streaming sync");

    match app_state
        .streaming_sync_service
        .sync_graphs_streaming()
        .await
    {
        Ok(stats) => {
            log::info!("Streaming sync completed: {:?}", stats);

            Ok(HttpResponse::Ok().json(serde_json::json!({
                "status": "completed",
                "statistics": {
                    "total_files": stats.total_files,
                    "kg_files_processed": stats.kg_files_processed,
                    "ontology_files_processed": stats.ontology_files_processed,
                    "skipped_files": stats.skipped_files,
                    "failed_files": stats.failed_files,
                    "duration_seconds": stats.duration.as_secs_f64(),
                    "metrics": {
                        "total_nodes": stats.total_nodes,
                        "total_edges": stats.total_edges,
                        "total_classes": stats.total_classes,
                        "total_properties": stats.total_properties,
                        "total_axioms": stats.total_axioms,
                    },
                    "errors": stats.errors,
                }
            })))
        }
        Err(e) => {
            log::error!("Streaming sync failed: {}", e);
            Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                "status": "error",
                "error": e
            })))
        }
    }
}

// ============================================================================
// EXAMPLE 5: Advanced - Sync with Custom Worker Count
// ============================================================================

use serde::Deserialize;

#[derive(Deserialize)]
pub struct StreamingSyncRequest {
    pub max_workers: Option<usize>,
}

/// POST /api/sync/streaming/custom
/// Run streaming sync with custom configuration
pub async fn run_custom_streaming_sync(
    app_state: web::Data<AppState>,
    req: web::Json<StreamingSyncRequest>,
) -> ActixResult<HttpResponse> {
    let worker_count = req.max_workers.unwrap_or(8).clamp(1, 16);

    log::info!("Starting streaming sync with {} workers", worker_count);

    // Create new service instance with custom worker count
    let service = StreamingSyncService::new(
        // Note: You'd need to store these in AppState
        // Arc::clone(&app_state.content_api),
        // Arc::clone(&app_state.kg_repo),
        // Arc::clone(&app_state.onto_repo),
        // Some(worker_count),
    );

    match service.sync_graphs_streaming().await {
        Ok(stats) => Ok(HttpResponse::Ok().json(serde_json::json!({
            "status": "completed",
            "worker_count": worker_count,
            "statistics": stats,
        }))),
        Err(e) => Ok(HttpResponse::InternalServerError().json(serde_json::json!({
            "status": "error",
            "error": e
        }))),
    }
}

// ============================================================================
// EXAMPLE 6: WebSocket Progress Streaming
// ============================================================================

use actix_web_actors::ws;

pub struct SyncProgressWebSocket {
    progress_rx: mpsc::UnboundedReceiver<SyncProgress>,
}

impl SyncProgressWebSocket {
    pub fn new(progress_rx: mpsc::UnboundedReceiver<SyncProgress>) -> Self {
        Self { progress_rx }
    }
}

// Note: Full WebSocket implementation omitted for brevity
// This would use actix-web-actors to stream progress updates to clients

// ============================================================================
// EXAMPLE 7: Background Sync Scheduler
// ============================================================================

use tokio::time::{interval, Duration};

/// Background task that runs streaming sync every N hours
pub async fn scheduled_sync_task(
    streaming_sync_service: Arc<StreamingSyncService>,
    interval_hours: u64,
) {
    let mut timer = interval(Duration::from_secs(interval_hours * 3600));

    loop {
        timer.tick().await;

        log::info!("Starting scheduled streaming sync");

        match streaming_sync_service.sync_graphs_streaming().await {
            Ok(stats) => {
                log::info!("Scheduled sync completed successfully");
                log::info!("  Files processed: {}", stats.total_files);
                log::info!("  Nodes saved: {}", stats.total_nodes);
                log::info!("  Duration: {:?}", stats.duration);

                if !stats.errors.is_empty() {
                    log::warn!("Sync completed with {} errors", stats.errors.len());
                }
            }
            Err(e) => {
                log::error!("Scheduled sync failed: {}", e);
            }
        }
    }
}

// ============================================================================
// EXAMPLE 8: Actix Web Route Configuration
// ============================================================================

use actix_web::web as actix_web;

pub fn configure_streaming_sync_routes(cfg: &mut actix_web::ServiceConfig) {
    cfg.service(
        actix_web::scope("/api/sync/streaming")
            .route("/start", actix_web::post().to(start_streaming_sync))
            .route("/run", actix_web::post().to(run_streaming_sync))
            .route("/progress", actix_web::get().to(get_sync_progress))
            .route("/custom", actix_web::post().to(run_custom_streaming_sync)),
    );
}

// ============================================================================
// EXAMPLE 9: CLI Command Integration
// ============================================================================

use clap::{Command, Arg};

pub async fn run_cli_sync(
    streaming_sync_service: Arc<StreamingSyncService>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create progress channel
    let (progress_tx, mut progress_rx) = mpsc::unbounded_channel();

    // Clone and configure service
    let mut service = (*streaming_sync_service).clone();
    service.set_progress_channel(progress_tx);

    // Spawn progress monitoring
    let progress_handle = tokio::spawn(async move {
        while let Some(progress) = progress_rx.recv().await {
            let percentage = if progress.files_total > 0 {
                (progress.files_processed as f64 / progress.files_total as f64) * 100.0
            } else {
                0.0
            };

            println!(
                "\rProgress: {}/{} files ({:.1}%) | Nodes: {} | Edges: {} | Errors: {}",
                progress.files_processed,
                progress.files_total,
                percentage,
                progress.kg_nodes_saved,
                progress.kg_edges_saved,
                progress.files_failed
            );
        }
    });

    // Run sync
    println!("Starting streaming GitHub sync...");
    let stats = service.sync_graphs_streaming().await?;

    // Wait for progress monitor
    progress_handle.await?;

    // Print final statistics
    println!("\n✅ Sync completed!");
    println!("  Total files: {}", stats.total_files);
    println!("  KG files: {}", stats.kg_files_processed);
    println!("  Ontology files: {}", stats.ontology_files_processed);
    println!("  Skipped: {}", stats.skipped_files);
    println!("  Failed: {}", stats.failed_files);
    println!("  Duration: {:?}", stats.duration);
    println!("\nMetrics:");
    println!("  Nodes: {}", stats.total_nodes);
    println!("  Edges: {}", stats.total_edges);
    println!("  Classes: {}", stats.total_classes);
    println!("  Properties: {}", stats.total_properties);
    println!("  Axioms: {}", stats.total_axioms);

    if !stats.errors.is_empty() {
        println!("\n⚠️  Errors ({}):", stats.errors.len());
        for error in &stats.errors {
            println!("  - {}", error);
        }
    }

    Ok(())
}

// ============================================================================
// EXAMPLE 10: Testing
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[actix_web::test]
    async fn test_streaming_sync_endpoint() {
        // Create test app state
        let app_state = web::Data::new(/* ... */);

        // Test the endpoint
        let resp = run_streaming_sync(app_state).await.unwrap();

        // Assert response
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_progress_tracking() {
        let (tx, mut rx) = mpsc::unbounded_channel();

        // Send test progress
        let progress = SyncProgress::new(100);
        tx.send(progress).unwrap();

        // Receive and verify
        let received = rx.recv().await.unwrap();
        assert_eq!(received.files_total, 100);
    }
}
