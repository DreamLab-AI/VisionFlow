use actix_web::{web, HttpResponse, Responder, Result};
use serde_json::json;
use std::sync::Arc;
use log::{info, warn};

use crate::services::mcp_session_bridge::McpSessionBridge;

/// GET /api/sessions/list - List all sessions with UUID and swarm_id mapping
pub async fn list_sessions(
    bridge: web::Data<Arc<McpSessionBridge>>,
) -> Result<impl Responder> {
    info!("Listing all monitored sessions");

    let sessions = bridge.list_monitored_sessions().await;

    Ok(HttpResponse::Ok().json(json!({
        "sessions": sessions,
        "count": sessions.len(),
    })))
}

/// GET /api/sessions/{uuid}/status - Get session status and swarm mapping
pub async fn get_session_status(
    path: web::Path<String>,
    bridge: web::Data<Arc<McpSessionBridge>>,
) -> Result<impl Responder> {
    let uuid = path.into_inner();
    info!("Getting status for session: {}", uuid);

    match bridge.get_session_status(&uuid).await {
        Ok(status) => Ok(HttpResponse::Ok().json(status)),
        Err(e) => {
            warn!("Failed to get session status for {}: {}", uuid, e);
            Ok(HttpResponse::NotFound().json(json!({
                "error": format!("Session not found: {}", e)
            })))
        }
    }
}

/// GET /api/sessions/{uuid}/telemetry - Get full telemetry via MCP
pub async fn get_session_telemetry(
    path: web::Path<String>,
    bridge: web::Data<Arc<McpSessionBridge>>,
) -> Result<impl Responder> {
    let uuid = path.into_inner();
    info!("Getting telemetry for session: {}", uuid);

    match bridge.query_session_telemetry(&uuid).await {
        Ok(telemetry) => Ok(HttpResponse::Ok().json(telemetry)),
        Err(e) => {
            warn!("Failed to get telemetry for {}: {}", uuid, e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to query telemetry: {}", e)
            })))
        }
    }
}

/// Configure session API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/sessions")
            .route("/list", web::get().to(list_sessions))
            .route("/{uuid}/status", web::get().to(get_session_status))
            .route("/{uuid}/telemetry", web::get().to(get_session_telemetry))
    );
}
