use actix_web::{web, HttpResponse, Error};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use chrono::Local;

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
    payload: web::Json<ClientLogsPayload>,
) -> Result<HttpResponse, Error> {
    let log_file_path = "/app/logs/client.log";

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