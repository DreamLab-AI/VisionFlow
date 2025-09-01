// Temporary stub to handle legacy settings calls during refactor
use actix_web::{web, HttpResponse, Result};
use serde_json::json;

pub async fn legacy_settings_error() -> Result<HttpResponse> {
    Ok(HttpResponse::ServiceUnavailable().json(json!({
        "error": "Legacy settings endpoint temporarily unavailable during refactor",
        "message": "These endpoints are being updated to use granular path-based operations",
        "code": "LEGACY_REFACTOR"
    })))
}