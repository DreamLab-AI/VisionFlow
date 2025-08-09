/*!
 * Visual Analytics Control API
 * 
 * REST API endpoints for controlling visual analytics parameters, constraints,
 * focus settings, and performance monitoring for the knowledge graph visualization.
 * 
 * Endpoints:
 * - GET /api/analytics/params - Get current visual analytics parameters
 * - POST /api/analytics/params - Update parameters
 * - GET /api/analytics/constraints - Get current constraints
 * - POST /api/analytics/constraints - Add/update constraints
 * - POST /api/analytics/focus - Set focus node/region
 * - GET /api/analytics/stats - Get performance statistics
 */

use actix_web::{web, HttpResponse, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use log::{debug, error, info, warn};

use crate::AppState;
use crate::actors::messages::{
    GetSettings, UpdateVisualAnalyticsParams, 
    GetConstraints, UpdateConstraints, GetPhysicsStats
};
use crate::gpu::visual_analytics::{VisualAnalyticsParams, PerformanceMetrics};
use crate::models::constraints::ConstraintSet;
use crate::actors::gpu_compute_actor::PhysicsStats;

/// Response for analytics parameter operations
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyticsParamsResponse {
    pub success: bool,
    pub params: Option<VisualAnalyticsParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Response for constraints operations
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConstraintsResponse {
    pub success: bool,
    pub constraints: Option<ConstraintSet>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Request payload for updating constraints
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateConstraintsRequest {
    pub constraint_set: Option<ConstraintSet>,
    pub constraint_data: Option<Value>,
    pub group_name: Option<String>,
    pub active: Option<bool>,
}

/// Request payload for setting focus
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SetFocusRequest {
    pub node_id: Option<i32>,
    pub region: Option<FocusRegion>,
    pub radius: Option<f32>,
    pub intensity: Option<f32>,
}

/// Focus region definition
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusRegion {
    pub center_x: f32,
    pub center_y: f32,
    pub center_z: f32,
    pub radius: f32,
}

/// Response for focus operations
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusResponse {
    pub success: bool,
    pub focus_node: Option<i32>,
    pub focus_region: Option<FocusRegion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Performance statistics response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StatsResponse {
    pub success: bool,
    pub physics_stats: Option<PhysicsStats>,
    pub visual_analytics_metrics: Option<PerformanceMetrics>,
    pub system_metrics: Option<SystemMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// System performance metrics
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemMetrics {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub gpu_utilization: f32,
    pub memory_usage_mb: f32,
    pub active_nodes: u32,
    pub active_edges: u32,
    pub render_time_ms: f32,
}

/// GET /api/analytics/params - Get current visual analytics parameters
pub async fn get_analytics_params(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    info!("Getting current visual analytics parameters");

    let settings = match app_state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => settings,
        Ok(Err(e)) => {
            error!("Failed to get settings for analytics params: {}", e);
            return Ok(HttpResponse::InternalServerError().json(AnalyticsParamsResponse {
                success: false,
                params: None,
                error: Some("Failed to retrieve settings".to_string()),
            }));
        }
        Err(e) => {
            error!("Settings actor mailbox error: {}", e);
            return Ok(HttpResponse::InternalServerError().json(AnalyticsParamsResponse {
                success: false,
                params: None,
                error: Some("Settings service unavailable".to_string()),
            }));
        }
    };

    // Extract or create default visual analytics parameters
    // For now, we'll create a default set since they might not be stored in settings yet
    let params = create_default_analytics_params(&settings);

    Ok(HttpResponse::Ok().json(AnalyticsParamsResponse {
        success: true,
        params: Some(params),
        error: None,
    }))
}

/// POST /api/analytics/params - Update parameters
pub async fn update_analytics_params(
    app_state: web::Data<AppState>,
    params: web::Json<VisualAnalyticsParams>,
) -> Result<HttpResponse> {
    info!("Updating visual analytics parameters");

    // Send update to GPU compute actor if available
    if let Some(gpu_addr) = app_state.gpu_compute_addr.as_ref() {
        match gpu_addr.send(UpdateVisualAnalyticsParams { params: params.clone() }).await {
            Ok(Ok(())) => {
                debug!("Visual analytics parameters updated successfully");
            }
            Ok(Err(e)) => {
                warn!("Failed to update GPU visual analytics params: {}", e);
            }
            Err(e) => {
                warn!("GPU compute actor mailbox error: {}", e);
            }
        }
    }

    Ok(HttpResponse::Ok().json(AnalyticsParamsResponse {
        success: true,
        params: Some(params.into_inner()),
        error: None,
    }))
}

/// GET /api/analytics/constraints - Get current constraints
pub async fn get_constraints(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    info!("Getting current constraint set");

    if let Some(gpu_addr) = app_state.gpu_compute_addr.as_ref() {
        match gpu_addr.send(GetConstraints).await {
            Ok(Ok(constraints)) => {
                return Ok(HttpResponse::Ok().json(ConstraintsResponse {
                    success: true,
                    constraints: Some(constraints),
                    error: None,
                }));
            }
            Ok(Err(e)) => {
                error!("Failed to get constraints from GPU actor: {}", e);
            }
            Err(e) => {
                error!("GPU compute actor mailbox error: {}", e);
            }
        }
    }

    // Fallback to empty constraint set
    Ok(HttpResponse::Ok().json(ConstraintsResponse {
        success: true,
        constraints: Some(ConstraintSet::default()),
        error: None,
    }))
}

/// POST /api/analytics/constraints - Add/update constraints
pub async fn update_constraints(
    app_state: web::Data<AppState>,
    request: web::Json<UpdateConstraintsRequest>,
) -> Result<HttpResponse> {
    info!("Updating constraint set");

    let constraint_data = if let Some(constraint_set) = &request.constraint_set {
        serde_json::to_value(constraint_set).unwrap_or_default()
    } else if let Some(data) = &request.constraint_data {
        data.clone()
    } else {
        return Ok(HttpResponse::BadRequest().json(ConstraintsResponse {
            success: false,
            constraints: None,
            error: Some("Either constraint_set or constraint_data must be provided".to_string()),
        }));
    };

    if let Some(gpu_addr) = app_state.gpu_compute_addr.as_ref() {
        match gpu_addr.send(UpdateConstraints { constraint_data }).await {
            Ok(Ok(())) => {
                debug!("Constraints updated successfully");
                
                // Get updated constraints to return
                if let Ok(Ok(updated_constraints)) = gpu_addr.send(GetConstraints).await {
                    return Ok(HttpResponse::Ok().json(ConstraintsResponse {
                        success: true,
                        constraints: Some(updated_constraints),
                        error: None,
                    }));
                }
            }
            Ok(Err(e)) => {
                error!("Failed to update constraints: {}", e);
                return Ok(HttpResponse::InternalServerError().json(ConstraintsResponse {
                    success: false,
                    constraints: None,
                    error: Some(format!("Failed to update constraints: {}", e)),
                }));
            }
            Err(e) => {
                error!("GPU compute actor mailbox error: {}", e);
                return Ok(HttpResponse::InternalServerError().json(ConstraintsResponse {
                    success: false,
                    constraints: None,
                    error: Some("GPU compute service unavailable".to_string()),
                }));
            }
        }
    }

    Ok(HttpResponse::ServiceUnavailable().json(ConstraintsResponse {
        success: false,
        constraints: None,
        error: Some("GPU compute service not available".to_string()),
    }))
}

/// POST /api/analytics/focus - Set focus node/region
pub async fn set_focus(
    _app_state: web::Data<AppState>,
    request: web::Json<SetFocusRequest>,
) -> Result<HttpResponse> {
    info!("Setting focus node/region");

    let focus_response = if let Some(node_id) = request.node_id {
        // Focus on specific node
        debug!("Setting focus on node {}", node_id);
        FocusResponse {
            success: true,
            focus_node: Some(node_id),
            focus_region: None,
            error: None,
        }
    } else if let Some(region) = &request.region {
        // Focus on specific region
        debug!("Setting focus on region center: ({}, {}, {}), radius: {}", 
               region.center_x, region.center_y, region.center_z, region.radius);
        FocusResponse {
            success: true,
            focus_node: None,
            focus_region: Some(FocusRegion {
                center_x: region.center_x,
                center_y: region.center_y,
                center_z: region.center_z,
                radius: region.radius,
            }),
            error: None,
        }
    } else {
        return Ok(HttpResponse::BadRequest().json(FocusResponse {
            success: false,
            focus_node: None,
            focus_region: None,
            error: Some("Either node_id or region must be specified".to_string()),
        }));
    };

    // TODO: Implement actual focus logic with GPU compute actor
    // This would involve updating visual analytics parameters with the focus information

    Ok(HttpResponse::Ok().json(focus_response))
}

/// GET /api/analytics/stats - Get performance statistics
pub async fn get_performance_stats(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    info!("Getting performance statistics");

    let physics_stats = if let Some(gpu_addr) = app_state.gpu_compute_addr.as_ref() {
        match gpu_addr.send(GetPhysicsStats).await {
            Ok(Ok(stats)) => Some(stats),
            Ok(Err(e)) => {
                warn!("Failed to get physics stats: {}", e);
                None
            }
            Err(e) => {
                warn!("GPU compute actor mailbox error: {}", e);
                None
            }
        }
    } else {
        None
    };

    // Create mock system metrics (in a real implementation, these would come from system monitoring)
    let system_metrics = SystemMetrics {
        fps: 60.0,
        frame_time_ms: 16.67,
        gpu_utilization: 45.0,
        memory_usage_mb: 512.0,
        active_nodes: physics_stats.as_ref().map(|s| s.num_nodes).unwrap_or(0),
        active_edges: physics_stats.as_ref().map(|s| s.num_edges).unwrap_or(0),
        render_time_ms: 12.5,
    };

    Ok(HttpResponse::Ok().json(StatsResponse {
        success: true,
        physics_stats,
        visual_analytics_metrics: None, // Would be populated from VisualAnalyticsGPU
        system_metrics: Some(system_metrics),
        error: None,
    }))
}

/// Helper function to create default analytics parameters
fn create_default_analytics_params(_settings: &crate::config::AppFullSettings) -> VisualAnalyticsParams {
    use crate::gpu::visual_analytics::VisualAnalyticsBuilder;
    
    VisualAnalyticsBuilder::new()
        .with_nodes(1000)
        .with_edges(2000)
        .with_focus(-1, 2.2)
        .with_temporal_decay(0.1)
        .build()
}

/// Configure analytics API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/analytics")
            .route("/params", web::get().to(get_analytics_params))
            .route("/params", web::post().to(update_analytics_params))
            .route("/constraints", web::get().to(get_constraints))
            .route("/constraints", web::post().to(update_constraints))
            .route("/focus", web::post().to(set_focus))
            .route("/stats", web::get().to(get_performance_stats))
    );
}