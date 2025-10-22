// LEGACY MODULE - Sessions API removed after McpSessionBridge deprecation
// All session management now handled by Management API (port 9090)

use actix_web::{web, HttpResponse, Responder, Result};
use log::warn;
use serde_json::json;

/// GET /api/sessions/list - List all sessions (DEPRECATED)
pub async fn list_sessions() -> Result<impl Responder> {
    warn!("Sessions API deprecated - use Management API at port 9090");
    Ok(HttpResponse::Gone().json(json!({
        "error": "Sessions API deprecated",
        "message": "Use Management API at agentic-workstation:9090/v1/tasks instead"
    })))
}

/// GET /api/sessions/{uuid}/status - Get session status (DEPRECATED)
pub async fn get_session_status(path: web::Path<String>) -> Result<impl Responder> {
    let uuid = path.into_inner();
    warn!("Sessions API deprecated - session {} not available", uuid);
    Ok(HttpResponse::Gone().json(json!({
        "error": "Sessions API deprecated",
        "message": format!("Use Management API at agentic-workstation:9090/v1/tasks/{} instead", uuid)
    })))
}

/// GET /api/sessions/{uuid}/telemetry - Get full telemetry (DEPRECATED)
pub async fn get_session_telemetry(path: web::Path<String>) -> Result<impl Responder> {
    let uuid = path.into_inner();
    warn!("Telemetry API deprecated - session {} not available", uuid);
    Ok(HttpResponse::Gone().json(json!({
        "error": "Telemetry API deprecated",
        "message": format!("Use Management API at agentic-workstation:9090/v1/tasks/{} instead", uuid)
    })))
}

/// Configure session API routes (DEPRECATED)
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/sessions")
            .route("/list", web::get().to(list_sessions))
            .route("/{uuid}/status", web::get().to(get_session_status))
            .route("/{uuid}/telemetry", web::get().to(get_session_telemetry)),
    );
}
