use actix_web::{web, HttpResponse, Result};
use log::info;

use crate::ok_json;

use super::state::FEATURE_FLAGS;
use super::types::FeatureFlags;

pub async fn get_feature_flags() -> Result<HttpResponse> {
    let flags = FEATURE_FLAGS.lock().await;

    ok_json!(serde_json::json!({
        "success": true,
        "flags": *flags,
        "description": {
            "gpu_clustering": "Enable GPU-accelerated clustering algorithms",
            "gpu_anomaly_detection": "Enable GPU-accelerated anomaly detection",
            "real_time_insights": "Enable real-time AI insights generation",
            "advanced_visualizations": "Enable advanced visualization features",
            "performance_monitoring": "Enable detailed performance monitoring",
            "stress_majorization": "Enable stress majorization layout algorithm",
            "semantic_constraints": "Enable semantic constraint processing",
            "sssp_integration": "Enable single-source shortest path integration",
            "ontology_validation": "Enable ontology validation and inference operations"
        }
    }))
}

pub async fn update_feature_flags(
    _auth: crate::settings::auth_extractor::AuthenticatedUser,
    request: web::Json<FeatureFlags>,
) -> Result<HttpResponse> {
    info!("Updating analytics feature flags");

    let mut flags = FEATURE_FLAGS.lock().await;
    *flags = request.into_inner();

    ok_json!(serde_json::json!({
        "success": true,
        "message": "Feature flags updated successfully",
        "flags": *flags
    }))
}
