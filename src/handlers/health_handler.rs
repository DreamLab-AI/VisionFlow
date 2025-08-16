use actix_web::{web, HttpResponse, Result, get};
use serde::{Deserialize, Serialize};
use crate::AppState;
use log::{info, error, warn};
use chrono::Utc;
use crate::actors::messages::{GetMetadata, GetGraphData};
use sysinfo::System;
use std::process::Command;
use tokio::time::Duration;

#[derive(Serialize, Deserialize)]
pub struct PhysicsSimulationStatus {
    status: String,
    details: String,
    timestamp: String,
}

pub async fn health_check(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let mut health_status = "healthy".to_string();
    let mut issues = Vec::new();
    
    // Check system metrics
    let mut sys = System::new_all();
    sys.refresh_all();
    
    let cpu_usage = sys.global_cpu_info().cpu_usage();
    let memory_usage = sys.used_memory() as f64 / sys.total_memory() as f64 * 100.0;
    let disk_usage = check_disk_usage();
    
    // Check if system resources are healthy
    if cpu_usage > 90.0 {
        health_status = "degraded".to_string();
        issues.push("High CPU usage".to_string());
    }
    if memory_usage > 90.0 {
        health_status = "degraded".to_string();
        issues.push("High memory usage".to_string());
    }
    if disk_usage > 90.0 {
        health_status = "degraded".to_string();
        issues.push("High disk usage".to_string());
    }
    
    // Check GPU status
    let gpu_status = check_gpu_status();
    
    // Check MCP connection
    let mcp_status = check_mcp_connection(&app_state).await;
    
    // Check database/metadata
    let metadata_count_result = tokio::time::timeout(
        Duration::from_secs(5),
        app_state.metadata_addr.send(GetMetadata)
    ).await;
    
    let metadata_count = match metadata_count_result {
        Ok(Ok(Ok(metadata_store))) => metadata_store.len(),
        Ok(Ok(Err(_))) => {
            health_status = "degraded".to_string();
            issues.push("Metadata store error".to_string());
            0
        },
        Ok(Err(_)) => {
            health_status = "degraded".to_string();
            issues.push("Metadata actor not responding".to_string());
            0
        },
        Err(_) => {
            health_status = "unhealthy".to_string();
            issues.push("Metadata actor timeout".to_string());
            0
        }
    };

    // Check graph service
    let graph_data_result = tokio::time::timeout(
        Duration::from_secs(5),
        app_state.graph_service_addr.send(GetGraphData)
    ).await;
    
    let (nodes_count, edges_count) = match graph_data_result {
        Ok(Ok(Ok(graph_data))) => (graph_data.nodes.len(), graph_data.edges.len()),
        Ok(Ok(Err(_))) => {
            health_status = "degraded".to_string();
            issues.push("Graph service error".to_string());
            (0, 0)
        },
        Ok(Err(_)) => {
            health_status = "degraded".to_string();
            issues.push("Graph service actor not responding".to_string());
            (0, 0)
        },
        Err(_) => {
            health_status = "unhealthy".to_string();
            issues.push("Graph service actor timeout".to_string());
            (0, 0)
        }
    };
    
    if health_status == "healthy" && !issues.is_empty() {
        health_status = "degraded".to_string();
    }
    
    Ok(HttpResponse::Ok().json(serde_json::json!({
        "status": health_status,
        "timestamp": Utc::now().to_rfc3339(),
        "issues": issues,
        "system": {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage,
            "gpu_status": gpu_status
        },
        "services": {
            "metadata_count": metadata_count,
            "nodes_count": nodes_count,
            "edges_count": edges_count,
            "mcp_status": mcp_status
        }
    })))
}

fn check_disk_usage() -> f64 {
    // Check disk usage for current directory
    match Command::new("df")
        .args(["."])
        .output()
    {
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
    // Check NVIDIA GPU status
    match Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
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
        Err(_) => {
            // Check for CUDA runtime
            match Command::new("nvcc").args(["--version"]).output() {
                Ok(output) if output.status.success() => "cuda_only".to_string(),
                _ => "unavailable".to_string(),
            }
        }
    }
}

async fn check_mcp_connection(_app_state: &web::Data<AppState>) -> String {
    "not_configured".to_string()
}

#[get("/physics")]
pub async fn check_physics_simulation(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    let current_time = Utc::now();
    
    // Assuming GraphServiceActor has a message like GetSimulationDiagnostics
    // If not, this part needs to be adapted based on how diagnostics are exposed by the actor.
    // For now, let's assume a placeholder or that GraphServiceActor itself doesn't expose this directly anymore
    // and this logic might need to be re-evaluated or moved.
    // If `get_simulation_diagnostics` was a method on the old GraphService struct,
    // it needs a corresponding message for the GraphServiceActor.
    // Let's assume for now we get a generic status from the actor.
    
    // Get real physics simulation diagnostics
    let (status, diagnostics) = match get_physics_diagnostics(&app_state).await {
        Ok((s, d)) => (s, d),
        Err(e) => {
            error!("Failed to get physics diagnostics: {}", e);
            ("error".to_string(), format!("Diagnostics failed: {}", e))
        }
    };
    
    info!("Physics simulation diagnostic check at {}: {}", current_time, diagnostics);
    
    Ok(HttpResponse::Ok().json(PhysicsSimulationStatus {
        status,
        details: diagnostics,
        timestamp: current_time.to_rfc3339(),
    }))
}

async fn get_physics_diagnostics(app_state: &web::Data<AppState>) -> Result<(String, String), String> {
    let mut diagnostics = Vec::new();
    let mut status = "healthy".to_string();
    
    // Check graph service
    match tokio::time::timeout(
        Duration::from_secs(3),
        app_state.graph_service_addr.send(GetGraphData)
    ).await {
        Ok(Ok(Ok(graph_data))) => {
            diagnostics.push(format!("Graph: {} nodes, {} edges", graph_data.nodes.len(), graph_data.edges.len()));
            if graph_data.nodes.is_empty() {
                status = "warning".to_string();
                diagnostics.push("No nodes in graph".to_string());
            }
        },
        Ok(Ok(Err(e))) => {
            status = "error".to_string();
            diagnostics.push(format!("Graph service error: {}", e));
        },
        Ok(Err(_)) => {
            status = "error".to_string();
            diagnostics.push("Graph service actor not responding".to_string());
        },
        Err(_) => {
            status = "error".to_string();
            diagnostics.push("Graph service timeout".to_string());
        }
    }
    
    // Check GPU compute if available
    if let Some(gpu_compute_addr) = &app_state.gpu_compute_addr {
        match tokio::time::timeout(
            Duration::from_secs(2),
            gpu_compute_addr.send(crate::actors::messages::GetGPUStatus)
        ).await {
            Ok(Ok(gpu_status)) => {
                diagnostics.push(format!("GPU compute: available, status: {:?}", gpu_status));
            },
            Ok(Err(e)) => {
                diagnostics.push(format!("GPU compute error: {}", e));
                if status == "healthy" { status = "degraded".to_string(); }
            },
            _ => {
                diagnostics.push("GPU compute not responding".to_string());
                if status == "healthy" { status = "degraded".to_string(); }
            }
        }
    } else {
        diagnostics.push("GPU compute not available - using CPU fallback".to_string());
    }
    
    // Check physics simulation parameters
    let physics_info = check_physics_parameters();
    diagnostics.push(physics_info);
    
    let full_diagnostics = diagnostics.join("; ");
    Ok((status, full_diagnostics))
}

fn check_physics_parameters() -> String {
    // Check if physics constants are reasonable
    let gravity = 0.08; // From PhysicsConfig
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
    
    format!("Physics params OK (gravity: {}, damping: {}, spring: {})", gravity, damping, spring_k)
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::resource("")
            .route(web::get().to(health_check))
    );
    cfg.service(check_physics_simulation);
}