// src/handlers/admin_bridge_handler.rs
//! Admin endpoint for ontology-to-graph bridge synchronization

use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::services::ontology_graph_bridge::OntologyGraphBridge;

#[derive(Debug, Serialize)]
pub struct BridgeSyncResponse {
    pub success: bool,
    pub message: String,
    pub nodes_created: usize,
    pub edges_created: usize,
}

#[derive(Debug, Deserialize)]
pub struct BridgeSyncRequest {
    #[serde(default)]
    pub clear_graph: bool,
}

/// POST /api/admin/sync-ontology-to-graph
/// Synchronize ontology data to knowledge graph database
pub async fn sync_ontology_to_graph(
    bridge: web::Data<OntologyGraphBridge>,
    payload: web::Json<BridgeSyncRequest>,
) -> Result<HttpResponse> {
    info!("[AdminBridge] Received ontology→graph sync request (clear_graph: {})", payload.clear_graph);

    // Clear graph if requested
    if payload.clear_graph {
        match bridge.clear_graph().await {
            Ok(_) => info!("[AdminBridge] Graph cleared successfully"),
            Err(e) => {
                error!("[AdminBridge] Failed to clear graph: {}", e);
                return Ok(HttpResponse::InternalServerError().json(BridgeSyncResponse {
                    success: false,
                    message: format!("Failed to clear graph: {}", e),
                    nodes_created: 0,
                    edges_created: 0,
                }));
            }
        }
    }

    // Run synchronization
    match bridge.sync_ontology_to_graph().await {
        Ok(stats) => {
            info!(
                "[AdminBridge] Sync completed: {} nodes, {} edges",
                stats.nodes_created, stats.edges_created
            );

            Ok(HttpResponse::Ok().json(BridgeSyncResponse {
                success: true,
                message: format!(
                    "Successfully synced ontology to graph: {} nodes, {} edges",
                    stats.nodes_created, stats.edges_created
                ),
                nodes_created: stats.nodes_created,
                edges_created: stats.edges_created,
            }))
        }
        Err(e) => {
            error!("[AdminBridge] Sync failed: {}", e);
            Ok(HttpResponse::InternalServerError().json(BridgeSyncResponse {
                success: false,
                message: format!("Sync failed: {}", e),
                nodes_created: 0,
                edges_created: 0,
            }))
        }
    }
}

/// GET /api/admin/bridge-status
/// Get bridge status information
pub async fn get_bridge_status() -> Result<HttpResponse> {
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "service": "ontology_graph_bridge",
        "status": "ready"
    })))
}

/// Configure admin bridge routes
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api/admin")
            .route("/sync-ontology-to-graph", web::post().to(sync_ontology_to_graph))
            .route("/bridge-status", web::get().to(get_bridge_status)),
    );
}
