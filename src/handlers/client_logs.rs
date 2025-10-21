use actix_web::{web, HttpResponse, Error};
use serde::Deserialize;
use log::{info, warn, error, debug};

#[derive(Deserialize)]
pub struct ClientLogEntry {
    level: String,
    message: String,
    timestamp: Option<String>,
    context: Option<String>,
}

/// Simple handler to receive client-side logs and forward to server logger
pub async fn post_client_logs(
    web::Json(payload): web::Json<ClientLogEntry>
) -> Result<HttpResponse, Error> {
    let prefix = "[CLIENT]";
    let msg = if let Some(ctx) = payload.context {
        format!("{} {}", payload.message, ctx)
    } else {
        payload.message
    };

    match payload.level.to_lowercase().as_str() {
        "error" => error!("{} {}", prefix, msg),
        "warn" => warn!("{} {}", prefix, msg),
        "debug" => debug!("{} {}", prefix, msg),
        _ => info!("{} {}", prefix, msg),
    }

    Ok(HttpResponse::Ok().json(serde_json::json!({"success": true})))
}
