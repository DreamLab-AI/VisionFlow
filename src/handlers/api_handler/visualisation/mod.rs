// Visualization API Handler - Legacy endpoints removed, use /api/settings instead
use actix_web::web;

// Legacy function kept for backward compatibility but redirects to new settings API
pub async fn get_visualisation_settings() -> actix_web::HttpResponse {
    actix_web::HttpResponse::MovedPermanently()
        .append_header(("Location", "/api/settings"))
        .finish()
}

pub fn config(cfg: &mut web::ServiceConfig) {
    // All settings endpoints have been moved to /api/settings
    // This scope is kept empty for backward compatibility
    cfg.service(
        web::scope("/visualisation")
            // Legacy routes removed - use /api/settings instead
    );
}