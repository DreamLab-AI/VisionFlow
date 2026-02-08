// Visualization API Handler - Legacy endpoints removed, use /api/settings instead

// Legacy function kept for backward compatibility but redirects to new settings API
pub async fn get_visualisation_settings() -> actix_web::HttpResponse {
    actix_web::HttpResponse::MovedPermanently()
        .append_header(("Location", "/api/settings"))
        .finish()
}

// Empty visualisation scope removed -- no routes registered.
// Use /api/settings for all settings operations.
