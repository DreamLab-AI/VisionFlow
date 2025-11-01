// src/handlers/admin_sync_handler.rs
//! Admin endpoint for triggering GitHub synchronization

use actix_web::{web, HttpResponse};
use log::{error, info};
use serde::Serialize;

use crate::services::github_sync_service::{GitHubSyncService, SyncStatistics};

#[derive(Serialize)]
pub struct SyncResponse {
    pub success: bool,
    pub message: String,
    pub statistics: Option<SyncStatisticsDto>,
}

#[derive(Serialize)]
pub struct SyncStatisticsDto {
    pub total_files: usize,
    pub kg_files_processed: usize,
    pub ontology_files_processed: usize,
    pub skipped_files: usize,
    pub errors: Vec<String>,
    pub duration_secs: f64,
    pub total_nodes: usize,
    pub total_edges: usize,
}

impl From<SyncStatistics> for SyncStatisticsDto {
    fn from(stats: SyncStatistics) -> Self {
        Self {
            total_files: stats.total_files,
            kg_files_processed: stats.kg_files_processed,
            ontology_files_processed: stats.ontology_files_processed,
            skipped_files: stats.skipped_files,
            errors: stats.errors,
            duration_secs: stats.duration.as_secs_f64(),
            total_nodes: stats.total_nodes,
            total_edges: stats.total_edges,
        }
    }
}

///
pub async fn trigger_sync(
    sync_service: web::Data<GitHubSyncService>,
) -> HttpResponse {
    info!("Admin sync endpoint triggered");

    match sync_service.sync_graphs().await {
        Ok(stats) => {
            info!(
                "Sync completed successfully: {} nodes, {} edges from {} files",
                stats.total_nodes, stats.total_edges, stats.total_files
            );

            HttpResponse::Ok().json(SyncResponse {
                success: true,
                message: format!(
                    "Sync completed: {} nodes, {} edges",
                    stats.total_nodes, stats.total_edges
                ),
                statistics: Some(stats.into()),
            })
        }
        Err(e) => {
            error!("Sync failed: {}", e);
            HttpResponse::InternalServerError().json(SyncResponse {
                success: false,
                message: format!("Sync failed: {}", e),
                statistics: None,
            })
        }
    }
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/admin/sync", web::post().to(trigger_sync));
}
