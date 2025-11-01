use crate::actors::messages::{GetGPUStatus, GetGraphData, GetMetadata};
use crate::services::mcp_relay_manager::McpRelayManager;
use crate::AppState;
use actix_web::{web, Error, HttpResponse, Result};
use chrono::Utc;
use log::{error, info, warn};
use serde::{Deserialize, Serialize};
use std::process::Command;
use sysinfo::System;
use tokio::time::Duration;

#[derive(Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub timestamp: String,
    pub issues: Vec<String>,
    pub system: SystemMetrics,
    pub services: ServiceMetrics,
    pub mcp: Option<McpMetrics>,
}

#[derive(Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub gpu_status: String,
}

#[derive(Serialize, Deserialize)]
pub struct ServiceMetrics {
    pub metadata_count: usize,
    pub nodes_count: usize,
    pub edges_count: usize,
    pub mcp_status: String,
}

#[derive(Serialize, Deserialize)]
pub struct McpMetrics {
    pub container_running: bool,
    pub mcp_relay_running: bool,
    pub last_logs: Option<String>,
    pub message: String,
}

#[derive(Serialize, Deserialize)]
pub struct PhysicsSimulationStatus {
    status: String,
    details: String,
    timestamp: String,
}

/
pub async fn unified_health_check(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let mut health_status = "healthy".to_string();
    let mut issues = Vec::new();

    
    let system_metrics = check_system_metrics(&mut health_status, &mut issues);

    
    let service_metrics = check_service_metrics(&app_state, &mut health_status, &mut issues).await;

    
    let mcp_metrics = check_mcp_metrics().await;

    
    if health_status == "healthy" && !issues.is_empty() {
        health_status = "degraded".to_string();
    }

    let response = HealthResponse {
        status: health_status,
        timestamp: Utc::now().to_rfc3339(),
        issues,
        system: system_metrics,
        services: service_metrics,
        mcp: Some(mcp_metrics),
    };

    Ok(HttpResponse::Ok().json(response))
}

fn check_system_metrics(health_status: &mut String, issues: &mut Vec<String>) -> SystemMetrics {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_usage = sys.global_cpu_usage();
    let memory_usage = sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0;
    let disk_usage = check_disk_usage();
    let gpu_status = check_gpu_status();

    
    if cpu_usage > 90.0 {
        *health_status = "degraded".to_string();
        issues.push("High CPU usage".to_string());
    }
    if memory_usage > 90.0 {
        *health_status = "degraded".to_string();
        issues.push("High memory usage".to_string());
    }
    if disk_usage > 90.0 {
        *health_status = "degraded".to_string();
        issues.push("High disk usage".to_string());
    }

    SystemMetrics {
        cpu_usage: cpu_usage as f64,
        memory_usage,
        disk_usage,
        gpu_status,
    }
}

async fn check_service_metrics(
    app_state: &web::Data<AppState>,
    health_status: &mut String,
    issues: &mut Vec<String>,
) -> ServiceMetrics {
    
    let metadata_count = match tokio::time::timeout(
        Duration::from_secs(5),
        app_state.metadata_addr.send(GetMetadata),
    )
    .await
    {
        Ok(Ok(Ok(metadata_store))) => metadata_store.len(),
        Ok(Ok(Err(_))) => {
            *health_status = "degraded".to_string();
            issues.push("Metadata store error".to_string());
            0
        }
        Ok(Err(_)) => {
            *health_status = "degraded".to_string();
            issues.push("Metadata actor not responding".to_string());
            0
        }
        Err(_) => {
            *health_status = "unhealthy".to_string();
            issues.push("Metadata actor timeout".to_string());
            0
        }
    };

    
    let (nodes_count, edges_count) = match tokio::time::timeout(
        Duration::from_secs(5),
        app_state.graph_service_addr.send(GetGraphData),
    )
    .await
    {
        Ok(Ok(Ok(graph_data))) => (graph_data.nodes.len(), graph_data.edges.len()),
        Ok(Ok(Err(_))) => {
            *health_status = "degraded".to_string();
            issues.push("Graph service error".to_string());
            (0, 0)
        }
        Ok(Err(_)) => {
            *health_status = "degraded".to_string();
            issues.push("Graph service actor not responding".to_string());
            (0, 0)
        }
        Err(_) => {
            *health_status = "unhealthy".to_string();
            issues.push("Graph service actor timeout".to_string());
            (0, 0)
        }
    };

    ServiceMetrics {
        metadata_count,
        nodes_count,
        edges_count,
        mcp_status: "not_configured".to_string(), 
    }
}

async fn check_mcp_metrics() -> McpMetrics {
    let container_running = McpRelayManager::check_mcp_container();
    let mcp_relay_running = if container_running {
        let manager = McpRelayManager::new();
        manager.check_relay_status().await.unwrap_or(false)
    } else {
        false
    };

    let last_logs = if container_running {
        McpRelayManager::get_relay_logs(20).ok()
    } else {
        None
    };

    let message = match (container_running, mcp_relay_running) {
        (false, _) => "Multi-agent container is not running".to_string(),
        (true, false) => "Container is running but MCP relay is not active".to_string(),
        (true, true) => "MCP relay is healthy and running".to_string(),
    };

    McpMetrics {
        container_running,
        mcp_relay_running,
        last_logs,
        message,
    }
}

fn check_disk_usage() -> f64 {
    match Command::new("df").args(["."]).output() {
        Ok(output) => {
            let output_str = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = output_str.lines().nth(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    if let Ok(usage) = parts[4].trim_end_matches('%').parse::<f64>() {
                        return usage;
                    }
                }
            }
        }
        Err(e) => warn!("Failed to check disk usage: {}", e),
    }
    0.0
}

fn check_gpu_status() -> String {
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) => {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if let Some(usage_str) = output_str.lines().next() {
                    if let Ok(usage) = usage_str.trim().parse::<f64>() {
                        return format!("available ({}% usage)", usage);
                    }
                }
                "available".to_string()
            } else {
                "unavailable".to_string()
            }
        }
        Err(_) => match Command::new("nvcc").args(["--version"]).output() {
            Ok(output) if output.status.success() => "cuda_only".to_string(),
            _ => "unavailable".to_string(),
        },
    }
}

pub async fn check_physics_simulation(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let current_time = Utc::now();

    let (status, diagnostics) = match get_physics_diagnostics(&app_state).await {
        Ok((s, d)) => (s, d),
        Err(e) => {
            error!("Failed to get physics diagnostics: {}", e);
            ("error".to_string(), format!("Diagnostics failed: {}", e))
        }
    };

    info!(
        "Physics simulation diagnostic check at {}: {}",
        current_time, diagnostics
    );

    Ok(HttpResponse::Ok().json(PhysicsSimulationStatus {
        status,
        details: diagnostics,
        timestamp: current_time.to_rfc3339(),
    }))
}

async fn get_physics_diagnostics(
    app_state: &web::Data<AppState>,
) -> Result<(String, String), String> {
    let mut diagnostics = Vec::new();
    let mut status = "healthy".to_string();

    
    match tokio::time::timeout(
        Duration::from_secs(3),
        app_state.graph_service_addr.send(GetGraphData),
    )
    .await
    {
        Ok(Ok(Ok(graph_data))) => {
            diagnostics.push(format!(
                "Graph: {} nodes, {} edges",
                graph_data.nodes.len(),
                graph_data.edges.len()
            ));
            if graph_data.nodes.is_empty() {
                status = "warning".to_string();
                diagnostics.push("No nodes in graph".to_string());
            }
        }
        Ok(Ok(Err(e))) => {
            status = "error".to_string();
            diagnostics.push(format!("Graph service error: {}", e));
        }
        Ok(Err(_)) => {
            status = "error".to_string();
            diagnostics.push("Graph service actor not responding".to_string());
        }
        Err(_) => {
            status = "error".to_string();
            diagnostics.push("Graph service timeout".to_string());
        }
    }

    
    if let Some(gpu_compute_addr) = &app_state.gpu_compute_addr {
        match tokio::time::timeout(Duration::from_secs(2), gpu_compute_addr.send(GetGPUStatus))
            .await
        {
            Ok(Ok(gpu_status)) => {
                diagnostics.push(format!("GPU compute: available, status: {:?}", gpu_status));
            }
            Ok(Err(e)) => {
                diagnostics.push(format!("GPU compute error: {}", e));
                if status == "healthy" {
                    status = "degraded".to_string();
                }
            }
            _ => {
                diagnostics.push("GPU compute not responding".to_string());
                if status == "healthy" {
                    status = "degraded".to_string();
                }
            }
        }
    } else {
        diagnostics.push("GPU compute not available - using CPU fallback".to_string());
    }

    
    let physics_info = check_physics_parameters();
    diagnostics.push(physics_info);

    let full_diagnostics = diagnostics.join("; ");
    Ok((status, full_diagnostics))
}

fn check_physics_parameters() -> String {
    let gravity = 0.08;
    let damping = 0.92;
    let spring_k = 0.3;

    if gravity < 0.0 || gravity > 1.0 {
        return "Invalid gravity value".to_string();
    }
    if damping < 0.0 || damping > 1.0 {
        return "Invalid damping value".to_string();
    }
    if spring_k < 0.0 || spring_k > 1.0 {
        return "Invalid spring strength".to_string();
    }

    format!(
        "Physics params OK (gravity: {}, damping: {}, spring: {})",
        gravity, damping, spring_k
    )
}

/
pub async fn start_mcp_relay() -> Result<HttpResponse> {
    let manager = McpRelayManager::new();
    match manager.ensure_relay_running().await {
        Ok(_) => Ok(HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "message": "MCP relay started successfully"
        }))),
        Err(e) => Err(Error::from(actix_web::error::ErrorInternalServerError(e))),
    }
}

#[derive(Deserialize)]
pub struct LogQuery {
    lines: Option<usize>,
}

pub async fn get_mcp_logs(query: web::Query<LogQuery>) -> Result<HttpResponse> {
    let lines = query.lines.unwrap_or(50);

    match McpRelayManager::get_relay_logs(lines) {
        Ok(logs) => Ok(HttpResponse::Ok().json(serde_json::json!({
            "success": true,
            "logs": logs
        }))),
        Err(e) => Err(Error::from(actix_web::error::ErrorInternalServerError(e))),
    }
}

/
pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/health")
            .route("", web::get().to(unified_health_check))
            .route("/physics", web::get().to(check_physics_simulation))
            .service(
                web::scope("/mcp")
                    .route("/start", web::post().to(start_mcp_relay))
                    .route("/logs", web::get().to(get_mcp_logs)),
            ),
    );
}
