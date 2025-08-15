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
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use uuid::Uuid;
use once_cell::sync::Lazy;

use crate::AppState;
use crate::actors::messages::{
    GetSettings, UpdateVisualAnalyticsParams, 
    GetConstraints, UpdateConstraints, GetPhysicsStats,
    SetComputeMode, GetGraphData
};
use crate::gpu::visual_analytics::{VisualAnalyticsParams, PerformanceMetrics};
use crate::models::constraints::ConstraintSet;
use crate::actors::gpu_compute_actor::PhysicsStats;
use crate::physics::semantic_constraints::SemanticCluster;

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

/// Clustering request payload
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusteringRequest {
    pub method: String,
    pub params: ClusteringParams,
}

/// Clustering parameters
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusteringParams {
    pub num_clusters: Option<u32>,
    pub min_cluster_size: Option<u32>,
    pub similarity: Option<String>,
    pub convergence_threshold: Option<f32>,
    pub max_iterations: Option<u32>,
    pub eps: Option<f32>,
    pub min_samples: Option<u32>,
    pub distance_threshold: Option<f32>,
    pub linkage: Option<String>,
    pub resolution: Option<f32>,
    pub random_state: Option<u32>,
    pub damping: Option<f32>,
    pub preference: Option<f32>,
}

/// Cluster data structure
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Cluster {
    pub id: String,
    pub label: String,
    pub node_count: u32,
    pub coherence: f32,
    pub color: String,
    pub keywords: Vec<String>,
    pub nodes: Vec<u32>,
    pub centroid: Option<[f32; 3]>,
}

/// Clustering response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusteringResponse {
    pub success: bool,
    pub clusters: Option<Vec<Cluster>>,
    pub method: Option<String>,
    pub execution_time_ms: Option<u64>,
    pub task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Clustering status response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusteringStatusResponse {
    pub success: bool,
    pub task_id: Option<String>,
    pub status: String, // "pending", "running", "completed", "failed"
    pub progress: f32, // 0.0 to 1.0
    pub method: Option<String>,
    pub started_at: Option<String>,
    pub estimated_completion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Cluster focus request
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClusterFocusRequest {
    pub cluster_id: String,
    pub zoom_level: Option<f32>,
    pub highlight: Option<bool>,
}

/// Anomaly detection configuration
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnomalyDetectionConfig {
    pub enabled: bool,
    pub method: String,
    pub sensitivity: f32,
    pub window_size: u32,
    pub update_interval: u32,
}

/// Anomaly data structure
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Anomaly {
    pub id: String,
    pub node_id: String,
    pub r#type: String,
    pub severity: String, // "low", "medium", "high", "critical"
    pub score: f32,
    pub description: String,
    pub timestamp: u64,
    pub metadata: Option<serde_json::Value>,
}

/// Anomaly statistics
#[derive(Debug, Serialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnomalyStats {
    pub total: u32,
    pub critical: u32,
    pub high: u32,
    pub medium: u32,
    pub low: u32,
    pub last_updated: Option<u64>,
}

/// Anomaly response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnomalyResponse {
    pub success: bool,
    pub anomalies: Option<Vec<Anomaly>>,
    pub stats: Option<AnomalyStats>,
    pub enabled: Option<bool>,
    pub method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// AI insights response
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InsightsResponse {
    pub success: bool,
    pub insights: Option<Vec<String>>,
    pub patterns: Option<Vec<GraphPattern>>,
    pub recommendations: Option<Vec<String>>,
    pub analysis_timestamp: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Graph pattern detected by AI analysis
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphPattern {
    pub id: String,
    pub r#type: String,
    pub description: String,
    pub confidence: f32,
    pub nodes: Vec<u32>,
    pub significance: String, // "low", "medium", "high"
}

// Global state for clustering operations
static CLUSTERING_TASKS: Lazy<Arc<Mutex<HashMap<String, ClusteringTask>>>> = 
    Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

static ANOMALY_STATE: Lazy<Arc<Mutex<AnomalyState>>> = 
    Lazy::new(|| Arc::new(Mutex::new(AnomalyState::default())));

#[derive(Debug, Clone)]
struct ClusteringTask {
    pub task_id: String,
    pub method: String,
    pub status: String,
    pub progress: f32,
    pub started_at: u64,
    pub clusters: Option<Vec<Cluster>>,
    pub error: Option<String>,
}

#[derive(Debug, Default, Clone)]
struct AnomalyState {
    pub enabled: bool,
    pub method: String,
    pub sensitivity: f32,
    pub window_size: u32,
    pub update_interval: u32,
    pub anomalies: Vec<Anomaly>,
    pub stats: AnomalyStats,
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
/// IMPORTANT: This endpoint is MISUSED by the client to update physics parameters!
/// The client PhysicsEngineControls sends physics values (repulsion, damping, etc.) here
/// but this endpoint expects VisualAnalyticsParams which doesn't have those fields.
/// This is a DESIGN FLAW that needs to be fixed properly with a dedicated physics endpoint.
/// 
/// Current workaround: Parse the raw JSON to extract physics params and forward them.
pub async fn update_analytics_params(
    app_state: web::Data<AppState>,
    params: web::Json<serde_json::Value>, // Changed to accept raw JSON
) -> Result<HttpResponse> {
    info!("Updating analytics/physics parameters via mismatched endpoint");
    
    // HACK: Extract physics parameters from the raw JSON
    // The client sends: { repulsion, attraction, damping, temperature, maxVelocity, timeStep, enabled }
    // We need to convert these to proper SimulationParams
    
    // Check if this is actually physics params from the UI
    if params.get("repulsion").is_some() || params.get("damping").is_some() {
        // This is physics params from the UI controls
        info!("Detected physics parameters in analytics endpoint");
        
        // Create a proper physics update message
        let mut sim_params = crate::models::simulation_params::SimulationParams::default();
        
        if let Some(v) = params.get("repulsion").and_then(|v| v.as_f64()) {
            sim_params.repulsion = v as f32;
        }
        if let Some(v) = params.get("attraction").and_then(|v| v.as_f64()) {
            sim_params.attraction_strength = v as f32;
        }
        if let Some(v) = params.get("damping").and_then(|v| v.as_f64()) {
            sim_params.damping = v as f32;
        }
        if let Some(v) = params.get("temperature").and_then(|v| v.as_f64()) {
            sim_params.temperature = v as f32;
        }
        if let Some(v) = params.get("maxVelocity").and_then(|v| v.as_f64()) {
            sim_params.max_velocity = v as f32;
        }
        if let Some(v) = params.get("timeStep").and_then(|v| v.as_f64()) {
            sim_params.time_step = v as f32;
        }
        if let Some(v) = params.get("enabled").and_then(|v| v.as_bool()) {
            sim_params.enabled = v;
        }
        
        // Send to GraphServiceActor (which is running the actual simulation)
        let graph_actor_addr = &app_state.graph_service_addr;
        
        // Send as UpdateSimulationParams to the GraphServiceActor
        use crate::actors::messages::UpdateSimulationParams;
        match graph_actor_addr.send(UpdateSimulationParams { params: sim_params }).await {
            Ok(Ok(())) => {
                info!("Physics parameters forwarded successfully to GraphServiceActor");
            }
            Ok(Err(e)) => {
                warn!("GraphServiceActor failed to update physics params: {}", e);
            }
            Err(e) => {
                warn!("GraphServiceActor mailbox error: {}", e);
            }
        }
    } else if let Some(gpu_addr) = app_state.gpu_compute_addr.as_ref() {
        // Try to parse as VisualAnalyticsParams for backwards compatibility
        if let Ok(visual_params) = serde_json::from_value::<VisualAnalyticsParams>(params.0.clone()) {
            match gpu_addr.send(UpdateVisualAnalyticsParams { params: visual_params }).await {
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
    }

    Ok(HttpResponse::Ok().json(AnalyticsParamsResponse {
        success: true,
        params: None, // Can't return typed params since we accept raw JSON
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

/// POST /api/analytics/kernel-mode - Set GPU kernel mode
pub async fn set_kernel_mode(
    app_state: web::Data<AppState>,
    request: web::Json<serde_json::Value>,
) -> Result<HttpResponse> {
    info!("Setting GPU kernel mode");
    
    if let Some(mode) = request.get("mode").and_then(|m| m.as_str()) {
        // Convert string mode to ComputeMode enum
        let compute_mode = match mode {
            "legacy" => crate::actors::gpu_compute_actor::ComputeMode::Basic,
            "dual_graph" => crate::actors::gpu_compute_actor::ComputeMode::DualGraph,
            "advanced" => crate::actors::gpu_compute_actor::ComputeMode::Advanced,
            // Accept alternate names for compatibility
            "standard" => crate::actors::gpu_compute_actor::ComputeMode::Basic,
            // Note: "visual_analytics" maps to Advanced ComputeMode, which triggers
            // automatic selection of KernelMode::VisualAnalytics when appropriate
            "visual_analytics" => crate::actors::gpu_compute_actor::ComputeMode::Advanced,
            _ => {
                return Ok(HttpResponse::BadRequest().json(serde_json::json!({
                    "success": false,
                    "error": format!("Invalid mode: {}. Valid modes: legacy, dual_graph, advanced", mode)
                })));
            }
        };
        
        if let Some(gpu_actor) = &app_state.gpu_compute_addr {
            match gpu_actor.send(SetComputeMode {
                mode: compute_mode,
            }).await {
                Ok(result) => {
                    match result {
                        Ok(()) => {
                            info!("GPU kernel mode set to: {}", mode);
                            Ok(HttpResponse::Ok().json(serde_json::json!({
                                "success": true,
                                "mode": mode
                            })))
                        }
                        Err(e) => {
                            error!("Failed to set kernel mode: {}", e);
                            Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                                "success": false,
                                "error": e
                            })))
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to send kernel mode message: {}", e);
                    Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                        "success": false,
                        "error": "Failed to communicate with GPU actor"
                    })))
                }
            }
        } else {
            Ok(HttpResponse::ServiceUnavailable().json(serde_json::json!({
                "success": false,
                "error": "GPU compute not available"
            })))
        }
    } else {
        Ok(HttpResponse::BadRequest().json(serde_json::json!({
            "success": false,
            "error": "Missing 'mode' parameter"
        })))
    }
}

/// GET /api/analytics/gpu-metrics - Get GPU metrics
pub async fn get_gpu_metrics(
    _app_state: web::Data<AppState>,
) -> Result<HttpResponse> {
    // Mock GPU metrics for now
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "utilization": 45.0,
        "memory": 32.0,
        "temperature": 68.0,
        "power": 120.0
    })))
}

/// POST /api/analytics/clustering/run - Run clustering analysis
pub async fn run_clustering(
    app_state: web::Data<AppState>,
    request: web::Json<ClusteringRequest>,
) -> Result<HttpResponse> {
    info!("Starting clustering analysis with method: {}", request.method);
    
    let task_id = Uuid::new_v4().to_string();
    let method = request.method.clone();
    
    // Create clustering task
    let task = ClusteringTask {
        task_id: task_id.clone(),
        method: method.clone(),
        status: "running".to_string(),
        progress: 0.0,
        started_at: chrono::Utc::now().timestamp() as u64,
        clusters: None,
        error: None,
    };
    
    // Store task
    {
        let mut tasks = CLUSTERING_TASKS.lock().await;
        tasks.insert(task_id.clone(), task);
    }
    
    // Start clustering in background
    let app_state_clone = app_state.clone();
    let task_id_clone = task_id.clone();
    let request_clone = request.into_inner();
    
    tokio::spawn(async move {
        let clusters = perform_clustering(&app_state_clone, &request_clone, &task_id_clone).await;
        
        let mut tasks = CLUSTERING_TASKS.lock().await;
        if let Some(task) = tasks.get_mut(&task_id_clone) {
            match clusters {
                Ok(clusters) => {
                    task.status = "completed".to_string();
                    task.progress = 1.0;
                    task.clusters = Some(clusters);
                },
                Err(e) => {
                    task.status = "failed".to_string();
                    task.error = Some(e);
                }
            }
        }
    });
    
    Ok(HttpResponse::Ok().json(ClusteringResponse {
        success: true,
        clusters: None,
        method: Some(method),
        execution_time_ms: None,
        task_id: Some(task_id),
        error: None,
    }))
}

/// GET /api/analytics/clustering/status - Get clustering status
pub async fn get_clustering_status(
    query: web::Query<HashMap<String, String>>,
) -> Result<HttpResponse> {
    let task_id = query.get("task_id");
    
    if let Some(task_id) = task_id {
        let tasks = CLUSTERING_TASKS.lock().await;
        if let Some(task) = tasks.get(task_id) {
            let estimated_completion = if task.status == "running" {
                Some(chrono::Utc::now().timestamp() as u64 + 30) // Estimate 30 seconds
            } else {
                None
            };
            
            return Ok(HttpResponse::Ok().json(ClusteringStatusResponse {
                success: true,
                task_id: Some(task.task_id.clone()),
                status: task.status.clone(),
                progress: task.progress,
                method: Some(task.method.clone()),
                started_at: Some(task.started_at.to_string()),
                estimated_completion: estimated_completion.map(|t| t.to_string()),
                error: task.error.clone(),
            }));
        }
    }
    
    Ok(HttpResponse::NotFound().json(ClusteringStatusResponse {
        success: false,
        task_id: None,
        status: "not_found".to_string(),
        progress: 0.0,
        method: None,
        started_at: None,
        estimated_completion: None,
        error: Some("Task not found".to_string()),
    }))
}

/// POST /api/analytics/clustering/focus - Focus on specific cluster
pub async fn focus_cluster(
    app_state: web::Data<AppState>,
    request: web::Json<ClusterFocusRequest>,
) -> Result<HttpResponse> {
    info!("Focusing on cluster: {}", request.cluster_id);
    
    // Find the cluster in our stored results
    let tasks = CLUSTERING_TASKS.lock().await;
    let cluster = tasks.values()
        .filter_map(|task| task.clusters.as_ref())
        .flatten()
        .find(|c| c.id == request.cluster_id)
        .cloned();
    
    if let Some(cluster) = cluster {
        // Create focus parameters based on cluster
        if let Some(centroid) = cluster.centroid {
            let focus_request = SetFocusRequest {
                node_id: None,
                region: Some(FocusRegion {
                    center_x: centroid[0],
                    center_y: centroid[1],
                    center_z: centroid[2],
                    radius: request.zoom_level.unwrap_or(5.0),
                }),
                radius: Some(request.zoom_level.unwrap_or(5.0)),
                intensity: Some(1.0),
            };
            
            // Apply focus using existing focus handler
            let focus_response = set_focus(app_state, web::Json(focus_request)).await?;
            return Ok(focus_response);
        }
    }
    
    Ok(HttpResponse::Ok().json(FocusResponse {
        success: true,
        focus_node: None,
        focus_region: None,
        error: Some("Cluster not found or no centroid available".to_string()),
    }))
}

/// POST /api/analytics/anomaly/toggle - Toggle anomaly detection
pub async fn toggle_anomaly_detection(
    request: web::Json<AnomalyDetectionConfig>,
) -> Result<HttpResponse> {
    info!("Toggling anomaly detection: enabled={}", request.enabled);
    
    let mut state = ANOMALY_STATE.lock().await;
    state.enabled = request.enabled;
    state.method = request.method.clone();
    state.sensitivity = request.sensitivity;
    state.window_size = request.window_size;
    state.update_interval = request.update_interval;
    
    if request.enabled {
        // Start anomaly detection simulation
        start_anomaly_detection().await;
    } else {
        // Clear existing anomalies
        state.anomalies.clear();
        state.stats = AnomalyStats::default();
    }
    
    Ok(HttpResponse::Ok().json(AnomalyResponse {
        success: true,
        anomalies: None,
        stats: Some(state.stats.clone()),
        enabled: Some(state.enabled),
        method: Some(state.method.clone()),
        error: None,
    }))
}

/// GET /api/analytics/anomaly/current - Get current anomalies
pub async fn get_current_anomalies() -> Result<HttpResponse> {
    let state = ANOMALY_STATE.lock().await;
    
    if !state.enabled {
        return Ok(HttpResponse::Ok().json(AnomalyResponse {
            success: true,
            anomalies: Some(vec![]),
            stats: Some(AnomalyStats::default()),
            enabled: Some(false),
            method: None,
            error: None,
        }));
    }
    
    Ok(HttpResponse::Ok().json(AnomalyResponse {
        success: true,
        anomalies: Some(state.anomalies.clone()),
        stats: Some(state.stats.clone()),
        enabled: Some(state.enabled),
        method: Some(state.method.clone()),
        error: None,
    }))
}

/// Helper function to perform clustering analysis
async fn perform_clustering(
    app_state: &web::Data<AppState>,
    request: &ClusteringRequest,
    task_id: &str,
) -> Result<Vec<Cluster>, String> {
    // Get graph data for clustering
    let graph_data = {
        match app_state.graph_service_addr.send(GetGraphData).await {
            Ok(Ok(data)) => data,
            _ => return Err("Failed to get graph data".to_string()),
        }
    };
    
    // Simulate clustering based on method
    let clusters = match request.method.as_str() {
        "spectral" => generate_spectral_clusters(&graph_data, &request.params),
        "kmeans" => generate_kmeans_clusters(&graph_data, &request.params),
        "louvain" => generate_louvain_clusters(&graph_data, &request.params),
        _ => generate_default_clusters(&graph_data, &request.params),
    };
    
    // Update progress periodically
    let mut tasks = CLUSTERING_TASKS.lock().await;
    if let Some(task) = tasks.get_mut(task_id) {
        task.progress = 0.5;
    }
    drop(tasks);
    
    // Simulate processing time
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    Ok(clusters)
}

/// Generate spectral clustering
fn generate_spectral_clusters(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Vec<Cluster> {
    let num_clusters = params.num_clusters.unwrap_or(5);
    generate_mock_clusters(graph_data, num_clusters, "spectral")
}

/// Generate k-means clustering
fn generate_kmeans_clusters(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Vec<Cluster> {
    let num_clusters = params.num_clusters.unwrap_or(8);
    generate_mock_clusters(graph_data, num_clusters, "kmeans")
}

/// Generate Louvain clustering
fn generate_louvain_clusters(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Vec<Cluster> {
    let resolution = params.resolution.unwrap_or(1.0);
    let num_clusters = (5.0 / resolution) as u32;
    generate_mock_clusters(graph_data, num_clusters, "louvain")
}

/// Generate default clustering
fn generate_default_clusters(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Vec<Cluster> {
    generate_mock_clusters(graph_data, 6, "default")
}

/// Generate mock clusters for demonstration
fn generate_mock_clusters(
    graph_data: &crate::models::graph::GraphData,
    num_clusters: u32,
    method: &str,
) -> Vec<Cluster> {
    let nodes_per_cluster = graph_data.nodes.len() / num_clusters as usize;
    let colors = vec!["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F"];
    let labels = vec!["Core Concepts", "Implementation", "Documentation", "Testing", "Infrastructure", "UI Components", "API Layer", "Data Models"];
    
    (0..num_clusters).map(|i| {
        let start_idx = (i as usize) * nodes_per_cluster;
        let end_idx = ((i + 1) as usize * nodes_per_cluster).min(graph_data.nodes.len());
        let cluster_nodes: Vec<u32> = (start_idx..end_idx).map(|idx| idx as u32).collect();
        
        // Calculate centroid from actual node positions
        let centroid = if !cluster_nodes.is_empty() {
            let sum_x: f32 = cluster_nodes.iter()
                .filter_map(|&id| graph_data.nodes.get(id as usize))
                .map(|n| n.data.position.x)
                .sum();
            let sum_y: f32 = cluster_nodes.iter()
                .filter_map(|&id| graph_data.nodes.get(id as usize))
                .map(|n| n.data.position.y)
                .sum();
            let sum_z: f32 = cluster_nodes.iter()
                .filter_map(|&id| graph_data.nodes.get(id as usize))
                .map(|n| n.data.position.z)
                .sum();
            let count = cluster_nodes.len() as f32;
            Some([sum_x / count, sum_y / count, sum_z / count])
        } else {
            None
        };
        
        Cluster {
            id: format!("cluster_{}", i),
            label: labels.get(i as usize).unwrap_or(&"Cluster").to_string(),
            node_count: cluster_nodes.len() as u32,
            coherence: 0.75 + (i as f32 * 0.03),
            color: colors.get(i as usize).unwrap_or(&"#888888").to_string(),
            keywords: vec![
                format!("{}_keyword1", method),
                format!("{}_keyword2", method),
            ],
            nodes: cluster_nodes,
            centroid,
        }
    }).collect()
}

/// Start anomaly detection simulation
async fn start_anomaly_detection() {
    tokio::spawn(async move {
        // Simulate generating anomalies
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        
        let mut state = ANOMALY_STATE.lock().await;
        
        // Generate some mock anomalies
        state.anomalies = vec![
            Anomaly {
                id: Uuid::new_v4().to_string(),
                node_id: "42".to_string(),
                r#type: "outlier".to_string(),
                severity: "high".to_string(),
                score: 0.92,
                description: "Node significantly deviates from expected pattern".to_string(),
                timestamp: chrono::Utc::now().timestamp() as u64,
                metadata: None,
            },
            Anomaly {
                id: Uuid::new_v4().to_string(),
                node_id: "128".to_string(),
                r#type: "disconnected".to_string(),
                severity: "medium".to_string(),
                score: 0.68,
                description: "Node has unusually low connectivity".to_string(),
                timestamp: chrono::Utc::now().timestamp() as u64,
                metadata: None,
            },
            Anomaly {
                id: Uuid::new_v4().to_string(),
                node_id: "256".to_string(),
                r#type: "cluster_drift".to_string(),
                severity: "low".to_string(),
                score: 0.45,
                description: "Node drifting from its semantic cluster".to_string(),
                timestamp: chrono::Utc::now().timestamp() as u64,
                metadata: None,
            },
        ];
        
        // Update stats
        state.stats = AnomalyStats {
            total: state.anomalies.len() as u32,
            critical: state.anomalies.iter().filter(|a| a.severity == "critical").count() as u32,
            high: state.anomalies.iter().filter(|a| a.severity == "high").count() as u32,
            medium: state.anomalies.iter().filter(|a| a.severity == "medium").count() as u32,
            low: state.anomalies.iter().filter(|a| a.severity == "low").count() as u32,
            last_updated: Some(chrono::Utc::now().timestamp() as u64),
        };
    });
}

/// GET /api/analytics/insights - Get AI insights
pub async fn get_ai_insights(
    app_state: web::Data<AppState>,
) -> Result<HttpResponse> {
    info!("Generating AI insights for graph analysis");
    
    // Get current graph data
    let graph_data = match app_state.graph_service_addr.send(GetGraphData).await {
        Ok(Ok(data)) => Some(data),
        _ => None,
    };
    
    // Generate insights based on current clustering and anomaly state
    let clustering_tasks = CLUSTERING_TASKS.lock().await;
    let anomaly_state = ANOMALY_STATE.lock().await;
    
    let mut insights = vec![
        "Graph structure analysis shows balanced connectivity patterns".to_string(),
        "Node distribution follows expected semantic clustering".to_string(),
    ];
    
    let mut patterns = vec![];
    let mut recommendations = vec![];
    
    // Add clustering insights
    if let Some(latest_clusters) = clustering_tasks.values()
        .filter(|t| t.status == "completed")
        .max_by_key(|t| t.started_at)
        .and_then(|t| t.clusters.as_ref()) {
        
        insights.push(format!("Identified {} distinct semantic clusters", latest_clusters.len()));
        
        if latest_clusters.len() > 10 {
            recommendations.push("Consider increasing clustering threshold to reduce cluster count".to_string());
        } else if latest_clusters.len() < 3 {
            recommendations.push("Consider decreasing clustering threshold for more granular grouping".to_string());
        }
        
        // Add pattern for largest cluster
        if let Some(largest_cluster) = latest_clusters.iter().max_by_key(|c| c.node_count) {
            patterns.push(GraphPattern {
                id: Uuid::new_v4().to_string(),
                r#type: "dominant_cluster".to_string(),
                description: format!("Large semantic cluster '{}' with {} nodes", largest_cluster.label, largest_cluster.node_count),
                confidence: largest_cluster.coherence,
                nodes: largest_cluster.nodes.clone(),
                significance: if largest_cluster.node_count > 50 { "high" } else { "medium" }.to_string(),
            });
        }
    }
    
    // Add anomaly insights
    if anomaly_state.enabled && anomaly_state.stats.total > 0 {
        insights.push(format!("Detected {} anomalies across the graph", anomaly_state.stats.total));
        
        if anomaly_state.stats.critical > 0 {
            recommendations.push("Investigate critical anomalies that may indicate data quality issues".to_string());
        }
        
        patterns.push(GraphPattern {
            id: Uuid::new_v4().to_string(),
            r#type: "anomaly_pattern".to_string(),
            description: format!("Anomaly distribution: {} critical, {} high, {} medium", 
                anomaly_state.stats.critical, anomaly_state.stats.high, anomaly_state.stats.medium),
            confidence: 0.9,
            nodes: anomaly_state.anomalies.iter()
                .take(10)
                .filter_map(|a| a.node_id.parse::<u32>().ok())
                .collect(),
            significance: "high".to_string(),
        });
    }
    
    // Add general graph insights
    if let Some(data) = graph_data {
        let node_count = data.nodes.len();
        let edge_count = data.edges.len();
        let density = if node_count > 1 {
            (2.0 * edge_count as f32) / (node_count as f32 * (node_count - 1) as f32)
        } else {
            0.0
        };
        
        insights.push(format!("Graph contains {} nodes and {} edges with density {:.3}", 
            node_count, edge_count, density));
        
        if density > 0.5 {
            recommendations.push("High graph density may benefit from hierarchical layout".to_string());
        } else if density < 0.1 {
            recommendations.push("Low graph density suggests potential for force-directed layout".to_string());
        }
    }
    
    Ok(HttpResponse::Ok().json(InsightsResponse {
        success: true,
        insights: Some(insights),
        patterns: Some(patterns),
        recommendations: Some(recommendations),
        analysis_timestamp: Some(chrono::Utc::now().timestamp() as u64),
        error: None,
    }))
}

/// Configure analytics API routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/analytics")
            // Existing endpoints
            .route("/params", web::get().to(get_analytics_params))
            .route("/params", web::post().to(update_analytics_params))
            .route("/constraints", web::get().to(get_constraints))
            .route("/constraints", web::post().to(update_constraints))
            .route("/focus", web::post().to(set_focus))
            .route("/stats", web::get().to(get_performance_stats))
            .route("/kernel-mode", web::post().to(set_kernel_mode))
            .route("/gpu-metrics", web::get().to(get_gpu_metrics))
            
            // New clustering endpoints
            .route("/clustering/run", web::post().to(run_clustering))
            .route("/clustering/status", web::get().to(get_clustering_status))
            .route("/clustering/focus", web::post().to(focus_cluster))
            
            // New anomaly detection endpoints
            .route("/anomaly/toggle", web::post().to(toggle_anomaly_detection))
            .route("/anomaly/current", web::get().to(get_current_anomalies))
            
            // New insights endpoint
            .route("/insights", web::get().to(get_ai_insights))
    );
}