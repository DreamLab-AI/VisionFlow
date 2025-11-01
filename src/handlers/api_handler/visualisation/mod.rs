// Visualization API Handler - Legacy endpoints removed, use /api/settings instead
use actix_web::web;

// Legacy function kept for backward compatibility but redirects to new settings API
pub async fn get_visualisation_settings() -> actix_web::HttpResponse {
    actix_web::HttpResponse::MovedPermanently()
        .append_header(("Location", "/api/settings"))
        .finish()
}

pub fn config(cfg: &mut web::ServiceConfig) {
    
    
    cfg.service(
        web::scope("/visualisation"), 
    );
}
