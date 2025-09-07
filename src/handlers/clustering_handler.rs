use actix_web::{web, Error, HttpResponse, HttpRequest};
use crate::app_state::AppState;
use crate::actors::messages::{GetSettings, UpdateSettings};
use log::{info, error, debug};
use serde_json::{json, Value};
use crate::config::ClusteringConfiguration;

/// Configure clustering-specific routes
pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/clustering")
            .route("/configure", web::post().to(configure_clustering))
            .route("/start", web::post().to(start_clustering))
            .route("/status", web::get().to(get_clustering_status))
            .route("/results", web::get().to(get_clustering_results))
            .route("/export", web::post().to(export_cluster_assignments))
    );
}

/// Configure clustering parameters
async fn configure_clustering(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<ClusteringConfiguration>,
) -> Result<HttpResponse, Error> {
    let config = payload.into_inner();
    
    info!("Clustering configuration request: algorithm={}, clusters={}", 
        config.algorithm, config.num_clusters);
    
    // Validate configuration
    if let Err(e) = validate_clustering_config(&config) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": format!("Invalid clustering configuration: {}", e)
        })));
    }
    
    // Store configuration in settings
    let settings_update = json!({
        "visualisation": {
            "graphs": {
                "logseq": {
                    "physics": {
                        "clusteringAlgorithm": config.algorithm,
                        "clusterCount": config.num_clusters,
                        "clusteringResolution": config.resolution,
                        "clusteringIterations": config.iterations
                    }
                },
                "visionflow": {
                    "physics": {
                        "clusteringAlgorithm": config.algorithm,
                        "clusterCount": config.num_clusters,
                        "clusteringResolution": config.resolution,
                        "clusteringIterations": config.iterations
                    }
                }
            }
        }
    });
    
    // Get and update settings
    let mut app_settings = match state.settings_addr.send(GetSettings).await {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            error!("Failed to get current settings: {}", e);
            return Ok(HttpResponse::InternalServerError().json(json!({
                "error": "Failed to get current settings"
            })));
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            return Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })));
        }
    };
    
    if let Err(e) = app_settings.merge_update(settings_update) {
        error!("Failed to merge clustering configuration: {}", e);
        return Ok(HttpResponse::InternalServerError().json(json!({
            "error": format!("Failed to update clustering configuration: {}", e)
        })));
    }
    
    // Save updated settings
    match state.settings_addr.send(UpdateSettings { settings: app_settings }).await {
        Ok(Ok(())) => {
            info!("Clustering configuration saved successfully");
            Ok(HttpResponse::Ok().json(json!({
                "status": "Clustering configuration updated successfully",
                "config": config
            })))
        }
        Ok(Err(e)) => {
            error!("Failed to save clustering configuration: {}", e);
            Ok(HttpResponse::InternalServerError().json(json!({
                "error": format!("Failed to save clustering configuration: {}", e)
            })))
        }
        Err(e) => {
            error!("Settings actor error: {}", e);
            Ok(HttpResponse::ServiceUnavailable().json(json!({
                "error": "Settings service unavailable"
            })))
        }
    }
}

/// Start clustering analysis
async fn start_clustering(
    _req: HttpRequest,
    _state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let request = payload.into_inner();
    
    info!("Starting clustering analysis");
    debug!("Clustering request: {}", serde_json::to_string_pretty(&request).unwrap_or_default());
    
    // For now, return a mock clustering start response
    // TODO: Implement actual GPU clustering integration
    let algorithm = request.get("algorithm")
        .and_then(|v| v.as_str())
        .unwrap_or("louvain");
    
    let cluster_count = request.get("clusterCount")
        .and_then(|v| v.as_u64())
        .unwrap_or(5) as u32;
    
    let task_id = uuid::Uuid::new_v4().to_string();
    
    info!("Clustering started with algorithm: {}, clusters: {}", algorithm, cluster_count);
    
    Ok(HttpResponse::Ok().json(json!({
        "status": "Clustering started",
        "taskId": task_id,
        "algorithm": algorithm,
        "clusterCount": cluster_count,
        "note": "Clustering functionality ready for GPU integration"
    })))
}

/// Get clustering status
async fn get_clustering_status(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("Clustering status request");
    
    // Return mock status - ready for GPU integration
    Ok(HttpResponse::Ok().json(json!({
        "status": "idle",
        "algorithm": "none",
        "progress": 0.0,
        "clustersFound": 0,
        "lastRun": null,
        "gpuAvailable": state.gpu_compute_addr.is_some()
    })))
}

/// Get clustering results
async fn get_clustering_results(
    _req: HttpRequest,
    state: web::Data<AppState>,
) -> Result<HttpResponse, Error> {
    info!("Clustering results request");
    
    // Return empty results - ready for GPU integration
    Ok(HttpResponse::Ok().json(json!({
        "clusters": [],
        "totalNodes": 0,
        "algorithmUsed": "none",
        "modularity": 0.0,
        "lastUpdated": chrono::Utc::now().to_rfc3339(),
        "gpuAvailable": state.gpu_compute_addr.is_some()
    })))
}

/// Export cluster assignments
async fn export_cluster_assignments(
    _req: HttpRequest,
    state: web::Data<AppState>,
    payload: web::Json<Value>,
) -> Result<HttpResponse, Error> {
    let export_request = payload.into_inner();
    
    info!("Cluster assignment export request");
    debug!("Export request: {}", serde_json::to_string_pretty(&export_request).unwrap_or_default());
    
    let format = export_request.get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("json");
    
    if !["json", "csv", "graphml"].contains(&format) {
        return Ok(HttpResponse::BadRequest().json(json!({
            "error": "format must be 'json', 'csv', or 'graphml'"
        })));
    }
    
    // Return empty export - ready for GPU integration
    let empty_export = match format {
        "csv" => "node_id,cluster_id\n".to_string(),
        "graphml" => "<?xml version=\"1.0\"?><graphml></graphml>".to_string(),
        _ => json!({"clusters": [], "nodes": [], "gpuAvailable": state.gpu_compute_addr.is_some()}).to_string(),
    };
    
    let content_type = match format {
        "csv" => "text/csv",
        "graphml" => "application/xml",
        _ => "application/json",
    };
    
    Ok(HttpResponse::Ok()
        .content_type(content_type)
        .body(empty_export))
}

/// Validate clustering configuration
fn validate_clustering_config(config: &ClusteringConfiguration) -> Result<(), String> {
    // Validate algorithm
    if !["none", "kmeans", "spectral", "louvain", "hierarchical", "dbscan"].contains(&config.algorithm.as_str()) {
        return Err("algorithm must be 'none', 'kmeans', 'spectral', 'louvain', 'hierarchical', or 'dbscan'".to_string());
    }
    
    // Validate cluster count
    if config.num_clusters < 2 || config.num_clusters > 50 {
        return Err("num_clusters must be between 2 and 50".to_string());
    }
    
    // Validate resolution
    if config.resolution < 0.1 || config.resolution > 5.0 {
        return Err("resolution must be between 0.1 and 5.0".to_string());
    }
    
    // Validate iterations
    if config.iterations < 10 || config.iterations > 1000 {
        return Err("iterations must be between 10 and 1000".to_string());
    }
    
    Ok(())
}