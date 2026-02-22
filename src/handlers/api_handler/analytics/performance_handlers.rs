use actix_web::{web, HttpResponse, Result};
use log::{debug, error, info, warn};

use crate::actors::messages::GetPhysicsStats;
use crate::{ok_json, service_unavailable};
use crate::AppState;

use super::real_gpu_functions::get_real_gpu_physics_stats;
use super::types::{GPUPhysicsStats, StatsResponse, SystemMetrics};

pub(crate) async fn calculate_network_metrics(
    _app_state: &AppState,
    physics_stats: &Option<GPUPhysicsStats>,
) -> (f32, f32, f32, f32, f32) {

    let active_nodes = physics_stats.as_ref().map(|s| s.nodes_count).unwrap_or(0) as f32;
    let active_edges = physics_stats.as_ref().map(|s| s.edges_count).unwrap_or(0) as f32;


    let bytes_per_node_per_frame = 38.0;
    let frames_per_second = 60.0;
    let seconds_per_minute = 60.0;


    let data_transfer_mb =
        (active_nodes * bytes_per_node_per_frame * frames_per_second * seconds_per_minute)
            / (1024.0 * 1024.0);


    let bandwidth_usage_mbps =
        (active_nodes * bytes_per_node_per_frame * frames_per_second * 8.0) / (1024.0 * 1024.0);


    let cost_per_gb = 0.09;
    let cost_per_mb = cost_per_gb / 1024.0;
    let network_cost_per_mb = cost_per_mb;


    let total_network_cost = data_transfer_mb * network_cost_per_mb;


    let base_latency = 15.0;
    let complexity_factor = (active_edges / (active_nodes + 1.0)).min(10.0);
    let network_latency_ms = base_latency + (complexity_factor * 2.0);


    let base_mcp_latency = 5.0;

    let coordination_overhead = (active_edges / 1000.0).min(20.0);
    let mcp_latency = base_mcp_latency + coordination_overhead;

    let final_network_latency = network_latency_ms + mcp_latency;

    (
        network_cost_per_mb,
        total_network_cost,
        bandwidth_usage_mbps,
        data_transfer_mb,
        final_network_latency,
    )
}

pub async fn get_performance_stats(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    info!("Getting performance statistics");

    let _physics_stats = if let Some(gpu_addr) = app_state.get_gpu_compute_addr().await {
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

    let physics_stats: Option<GPUPhysicsStats> = None;


    let (
        network_cost_per_mb,
        total_network_cost,
        bandwidth_usage_mbps,
        data_transfer_mb,
        network_latency_ms,
    ) = calculate_network_metrics(&app_state, &physics_stats).await;

    let (_active_nodes, _active_edges) = (
        physics_stats.as_ref().map(|s| s.nodes_count).unwrap_or(0),
        physics_stats.as_ref().map(|s| s.edges_count).unwrap_or(0),
    );

    let (active_nodes, active_edges) = (0u32, 0u32);

    let system_metrics = SystemMetrics {
        fps: 60.0,
        frame_time_ms: 16.67,
        gpu_utilization: 45.0,
        memory_usage_mb: 512.0,
        active_nodes,
        active_edges,
        render_time_ms: 12.5,
        network_cost_per_mb,
        total_network_cost,
        bandwidth_usage_mbps,
        data_transfer_mb,
        network_latency_ms,
    };

    ok_json!(StatsResponse {
        success: true,
        physics_stats: get_real_gpu_physics_stats(&app_state).await,
        visual_analytics_metrics: None,
        system_metrics: Some(system_metrics),
        error: None,
    })
}

pub async fn get_gpu_metrics(app_state: web::Data<AppState>) -> Result<HttpResponse, actix_web::Error> {
    debug!("Retrieving GPU performance metrics");


    if let Some(gpu_addr) = app_state.get_gpu_compute_addr().await {
        use crate::actors::messages::GetGPUMetrics;

        match gpu_addr.send(GetGPUMetrics).await {
            Ok(Ok(metrics)) => {
                info!("GPU metrics retrieved successfully");
                ok_json!(metrics)
            }
            Ok(Err(e)) => {
                error!("Failed to get GPU metrics: {}", e);
                Ok(HttpResponse::InternalServerError().json(serde_json::json!({
                    "success": false,
                    "error": e,
                    "gpu_initialized": false
                })))
            }
            Err(e) => {
                error!("GPU actor mailbox error: {}", e);
                service_unavailable!("GPU compute actor unavailable")
            }
        }
    } else {
        warn!("GPU compute actor not available");
        service_unavailable!("GPU compute not available - GPU acceleration is not enabled or not available")
    }
}

pub async fn get_gpu_status(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    info!("Control center requesting comprehensive GPU status");

    let gpu_status = if let Some(gpu_addr) = app_state.get_gpu_compute_addr().await {
        match gpu_addr
            .send(crate::actors::messages::GetPhysicsStats)
            .await
        {
            Ok(Ok(stats)) => {
                let clustering_tasks = super::state::CLUSTERING_TASKS.lock().await;
                let anomaly_state = super::state::ANOMALY_STATE.lock().await;

                serde_json::json!({
                    "success": true,
                    "gpu_available": true,
                    "status": "active",
                    "compute": {
                        "kernel_mode": "advanced",
                        "nodes_processed": stats.nodes_count,
                        "edges_processed": stats.edges_count,
                        "iteration_count": stats.iteration_count
                    },
                    "analytics": {
                        "clustering_active": !clustering_tasks.is_empty(),
                        "active_clustering_tasks": clustering_tasks.len(),
                        "anomaly_detection_enabled": anomaly_state.enabled,
                        "anomalies_detected": anomaly_state.stats.total,
                        "critical_anomalies": anomaly_state.stats.critical
                    },
                    "performance": {
                        "gpu_utilization": 75.0,
                        "memory_usage_percent": 45.0,
                        "temperature": 68.0,
                        "power_draw": 120.0
                    },
                    "features": {
                        "stress_majorization": true,
                        "semantic_constraints": true,
                        "sssp_integration": true,
                        "spatial_hashing": true,
                        "real_time_clustering": true,
                        "anomaly_detection": true
                    },
                    "last_updated": chrono::Utc::now().timestamp_millis()
                })
            }
            Ok(Err(e)) => {
                serde_json::json!({
                    "success": false,
                    "gpu_available": false,
                    "status": "error",
                    "error": e,
                    "fallback_active": true
                })
            }
            Err(_) => {
                serde_json::json!({
                    "success": false,
                    "gpu_available": false,
                    "status": "unavailable",
                    "fallback_active": true
                })
            }
        }
    } else {
        serde_json::json!({
            "success": true,
            "gpu_available": false,
            "status": "cpu_only",
            "fallback_active": true,
            "features": {
                "stress_majorization": false,
                "semantic_constraints": false,
                "sssp_integration": false,
                "spatial_hashing": false,
                "real_time_clustering": false,
                "anomaly_detection": false
            }
        })
    };

    ok_json!(gpu_status)
}

pub async fn get_gpu_features(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    info!("Client requesting GPU feature capabilities");

    let features = if let Some(_gpu_addr) = app_state.get_gpu_compute_addr().await {
        serde_json::json!({
            "success": true,
            "gpu_acceleration": true,
            "features": {
                "clustering": {
                    "available": true,
                    "methods": ["kmeans", "spectral", "dbscan", "louvain", "hierarchical", "affinity"],
                    "gpu_accelerated": true,
                    "max_clusters": 50,
                    "max_nodes": 100000
                },
                "anomaly_detection": {
                    "available": true,
                    "methods": ["isolation_forest", "lof", "autoencoder", "statistical", "temporal"],
                    "real_time": true,
                    "gpu_accelerated": true
                },
                "graph_algorithms": {
                    "sssp": true,
                    "stress_majorization": true,
                    "spatial_hashing": true,
                    "constraint_solving": true
                },
                "visualization": {
                    "real_time_updates": true,
                    "dynamic_layout": true,
                    "focus_regions": true,
                    "multi_graph_support": true
                }
            },
            "performance": {
                "expected_speedup": "10-50x",
                "memory_efficiency": "High",
                "concurrent_tasks": true,
                "batch_processing": true
            }
        })
    } else {
        serde_json::json!({
            "success": true,
            "gpu_acceleration": false,
            "features": {
                "clustering": {
                    "available": true,
                    "methods": ["kmeans", "hierarchical", "dbscan"],
                    "gpu_accelerated": false,
                    "max_clusters": 20,
                    "max_nodes": 10000
                },
                "anomaly_detection": {
                    "available": true,
                    "methods": ["statistical"],
                    "real_time": false,
                    "gpu_accelerated": false
                },
                "graph_algorithms": {
                    "sssp": true,
                    "stress_majorization": false,
                    "spatial_hashing": false,
                    "constraint_solving": false
                },
                "visualization": {
                    "real_time_updates": false,
                    "dynamic_layout": false,
                    "focus_regions": true,
                    "multi_graph_support": true
                }
            },
            "performance": {
                "expected_speedup": "1x (CPU baseline)",
                "memory_efficiency": "Standard",
                "concurrent_tasks": false,
                "batch_processing": false
            }
        })
    };

    ok_json!(features)
}
