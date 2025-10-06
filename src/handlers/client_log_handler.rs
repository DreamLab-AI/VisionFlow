use actix_web::{web, HttpRequest, HttpResponse, Error};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Local;
use log::{debug, error, info};

use crate::services::session_correlation_bridge::{get_global_bridge, extract_session_id_from_header};
use crate::telemetry::agent_telemetry::CorrelationId;

#[derive(Debug, Deserialize, Serialize)]
pub struct LogEntry {
    level: String,
    namespace: String,
    message: String,
    timestamp: String,
    data: Option<serde_json::Value>,
    #[serde(rename = "userAgent")]
    user_agent: Option<String>,
    url: Option<String>,
    stack: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ClientLogsPayload {
    logs: Vec<LogEntry>,
    #[serde(rename = "sessionId")]
    session_id: String,
    timestamp: String,
}

/// Handler for receiving browser logs from Quest 3 and other remote clients
pub async fn handle_client_logs(
    req: HttpRequest,
    payload: web::Json<ClientLogsPayload>,
) -> Result<HttpResponse, Error> {
    let log_file_path = "/app/logs/client.log";

    // Register session correlation mapping if bridge is available
    if let Some(bridge) = get_global_bridge() {
        // Check for X-Session-ID header
        let header_session_id = req.headers()
            .get("X-Session-ID")
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string());

        // Use header session ID if present, otherwise use payload session ID
        let client_session_id = header_session_id
            .as_ref()
            .unwrap_or(&payload.session_id);

        // Generate a server correlation ID for this session
        let correlation_id = CorrelationId::from_session_uuid(client_session_id);

        // Register the mapping
        match bridge.register_session(client_session_id.clone(), correlation_id.clone()) {
            Ok(_) => {
                info!("Session correlation registered: {} ↔ {}", client_session_id, correlation_id);
            }
            Err(e) => {
                error!("Failed to register session correlation: {}", e);
            }
        }

        // Log telemetry event if telemetry logger is available
        if let Some(telemetry) = crate::telemetry::agent_telemetry::get_telemetry_logger() {
            let event = crate::telemetry::agent_telemetry::TelemetryEvent::new(
                correlation_id,
                crate::telemetry::agent_telemetry::LogLevel::INFO,
                "session_tracking",
                "client_logs_received",
                &format!("Received {} log entries from client session", payload.logs.len()),
                "client_log_handler"
            )
            .with_client_session_id(client_session_id)
            .with_metadata("log_count", serde_json::json!(payload.logs.len()))
            .with_metadata("has_x_session_id_header", serde_json::json!(header_session_id.is_some()));

            telemetry.log_event(event);
        }
    } else {
        debug!("Session correlation bridge not initialized, skipping mapping registration");
    }

    // Open the log file in append mode
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_file_path)
        .map_err(|e| {
            log::error!("Failed to open client.log: {}", e);
            actix_web::error::ErrorInternalServerError(format!("Failed to open log file: {}", e))
        })?;

    // Write each log entry to the file
    for entry in &payload.logs {
        let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

        // Format the log line
        let log_line = format!(
            "[{}] [{}] [{}] {} - {} | UA: {} | URL: {}{}\n",
            timestamp,
            entry.level.to_uppercase(),
            entry.namespace,
            payload.session_id,
            entry.message,
            entry.user_agent.as_ref().unwrap_or(&"unknown".to_string()),
            entry.url.as_ref().unwrap_or(&"unknown".to_string()),
            if let Some(data) = &entry.data {
                format!(" | Data: {}", serde_json::to_string(data).unwrap_or_default())
            } else {
                String::new()
            }
        );

        // Write to file
        file.write_all(log_line.as_bytes())
            .map_err(|e| {
                log::error!("Failed to write to client.log: {}", e);
                actix_web::error::ErrorInternalServerError(format!("Failed to write log: {}", e))
            })?;

        // Also log to server console for debugging
        match entry.level.as_str() {
            "error" => log::error!("[CLIENT] {} - {}", entry.namespace, entry.message),
            "warn" => log::warn!("[CLIENT] {} - {}", entry.namespace, entry.message),
            "info" => log::info!("[CLIENT] {} - {}", entry.namespace, entry.message),
            _ => log::debug!("[CLIENT] {} - {}", entry.namespace, entry.message),
        }

        // If there's a stack trace, write it separately
        if let Some(stack) = &entry.stack {
            let stack_line = format!("[{}] [STACK] {}\n{}\n", timestamp, payload.session_id, stack);
            file.write_all(stack_line.as_bytes())
                .map_err(|e| {
                    log::error!("Failed to write stack trace: {}", e);
                    actix_web::error::ErrorInternalServerError(format!("Failed to write stack: {}", e))
                })?;
        }
    }

    // Flush to ensure data is written
    file.flush()
        .map_err(|e| {
            log::error!("Failed to flush client.log: {}", e);
            actix_web::error::ErrorInternalServerError(format!("Failed to flush log file: {}", e))
        })?;

    log::debug!("Received {} log entries from client session {}", payload.logs.len(), payload.session_id);

    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": "success",
        "received": payload.logs.len()
    })))
}