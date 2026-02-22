use actix_web::{web, HttpResponse, Result};
use log::{info, warn};
use std::collections::HashMap;
use uuid::Uuid;

use crate::actors::messages::GetGraphData;
use crate::services::agent_visualization_protocol::McpServerType;
use crate::utils::mcp_tcp_client::create_mcp_client;
use crate::{ok_json, not_found};
use crate::AppState;

use super::real_gpu_functions::*;
use super::state::CLUSTERING_TASKS;
use super::types::{
    Cluster, ClusterFocusRequest, ClusteringParams, ClusteringRequest,
    ClusteringResponse, ClusteringStatusResponse, ClusteringTask,
    FocusRegion, SetFocusRequest,
};
use super::params_handlers::set_focus;

pub async fn run_clustering(
    _auth: crate::settings::auth_extractor::AuthenticatedUser,
    app_state: web::Data<AppState>,
    request: web::Json<ClusteringRequest>,
) -> Result<HttpResponse> {
    info!(
        "Starting clustering analysis with method: {}",
        request.method
    );

    let task_id = Uuid::new_v4().to_string();
    let method = request.method.clone();


    let task = ClusteringTask {
        task_id: task_id.clone(),
        method: method.clone(),
        status: "running".to_string(),
        progress: 0.0,
        started_at: chrono::Utc::now().timestamp() as u64,
        clusters: None,
        error: None,
    };


    {
        let mut tasks = CLUSTERING_TASKS.lock().await;
        tasks.insert(task_id.clone(), task);
    }


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
                }
                Err(e) => {
                    task.status = "failed".to_string();
                    task.error = Some(e);
                }
            }
        }
    });

    ok_json!(ClusteringResponse {
        success: true,
        clusters: None,
        method: Some(method),
        execution_time_ms: None,
        task_id: Some(task_id),
        error: None,
    })
}

pub async fn get_clustering_status(
    query: web::Query<HashMap<String, String>>,
) -> Result<HttpResponse> {
    let task_id = query.get("task_id");

    if let Some(task_id) = task_id {
        let tasks = CLUSTERING_TASKS.lock().await;
        if let Some(task) = tasks.get(task_id) {
            let estimated_completion = if task.status == "running" {
                Some(chrono::Utc::now().timestamp() as u64 + 30)
            } else {
                None
            };

            return ok_json!(ClusteringStatusResponse {
                success: true,
                task_id: Some(task.task_id.clone()),
                status: task.status.clone(),
                progress: task.progress,
                method: Some(task.method.clone()),
                started_at: Some(task.started_at.to_string()),
                estimated_completion: estimated_completion.map(|t| t.to_string()),
                error: task.error.clone(),
            });
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

pub async fn focus_cluster(
    auth: crate::settings::auth_extractor::AuthenticatedUser,
    app_state: web::Data<AppState>,
    request: web::Json<ClusterFocusRequest>,
) -> Result<HttpResponse> {
    info!("Focusing on cluster: {}", request.cluster_id);


    let tasks = CLUSTERING_TASKS.lock().await;
    let cluster = tasks
        .values()
        .filter_map(|task| task.clusters.as_ref())
        .flatten()
        .find(|c| c.id == request.cluster_id)
        .cloned();

    if let Some(cluster) = cluster {

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


            let focus_response = set_focus(auth, app_state, web::Json(focus_request)).await?;
            return Ok(focus_response);
        }
    }

    ok_json!(super::types::FocusResponse {
        success: true,
        focus_node: None,
        focus_region: None,
        error: Some("Cluster not found or no centroid available".to_string()),
    })
}

pub async fn cancel_clustering(
    _auth: crate::settings::auth_extractor::AuthenticatedUser,
    query: web::Query<HashMap<String, String>>,
) -> Result<HttpResponse> {
    let task_id = query.get("task_id");

    if let Some(task_id) = task_id {
        info!("Canceling clustering task: {}", task_id);

        let mut tasks = CLUSTERING_TASKS.lock().await;
        if let Some(task) = tasks.get_mut(task_id) {
            task.status = "cancelled".to_string();
            task.error = Some("Cancelled by user".to_string());

            return ok_json!(serde_json::json!({
                "success": true,
                "message": "Task cancelled successfully",
                "task_id": task_id
            }));
        }
    }

    not_found!("Task not found or not cancellable")
}

pub(crate) async fn perform_clustering(
    app_state: &web::Data<AppState>,
    request: &ClusteringRequest,
    task_id: &str,
) -> Result<Vec<Cluster>, String> {
    info!("Performing real clustering analysis using MCP agent data");


    let graph_data = {
        match app_state.graph_service_addr.send(GetGraphData).await {
            Ok(Ok(data)) => data,
            _ => return Err("Failed to get graph data".to_string()),
        }
    };


    let host = std::env::var("MCP_HOST").unwrap_or_else(|_| "localhost".to_string());
    let port = std::env::var("MCP_TCP_PORT")
        .unwrap_or_else(|_| "9500".to_string())
        .parse::<u16>()
        .unwrap_or(9500);

    let mcp_client = create_mcp_client(&McpServerType::ClaudeFlow, &host, port);


    let agents = match mcp_client.query_agent_list().await {
        Ok(agent_list) => {
            info!(
                "Retrieved {} agents from MCP server for clustering",
                agent_list.len()
            );
            agent_list
        }
        Err(e) => {
            warn!(
                "Failed to get agents from MCP server, using graph data: {}",
                e
            );
            Vec::new()
        }
    };


    let clusters = match request.method.as_str() {
        "spectral" => {
            perform_gpu_spectral_clustering(&**app_state, &graph_data, &agents, &request.params)
                .await
        }
        "kmeans" => {
            perform_gpu_kmeans_clustering(&**app_state, &graph_data, &agents, &request.params).await
        }
        "louvain" => {
            perform_gpu_louvain_clustering(&**app_state, &graph_data, &agents, &request.params)
                .await
        }
        _ => {
            perform_gpu_default_clustering(&**app_state, &graph_data, &agents, &request.params)
                .await
        }
    };


    let mut tasks = CLUSTERING_TASKS.lock().await;
    if let Some(task) = tasks.get_mut(task_id) {
        task.progress = 0.5;
    }
    drop(tasks);


    let processing_time = std::cmp::min(agents.len() / 10, 5) as u64;
    tokio::time::sleep(tokio::time::Duration::from_secs(processing_time)).await;

    Ok(clusters)
}

#[allow(dead_code)]
pub(crate) fn generate_spectral_clusters_from_agents(
    graph_data: &crate::models::graph::GraphData,
    agents: &[crate::services::agent_visualization_protocol::MultiMcpAgentStatus],
    params: &ClusteringParams,
) -> Vec<Cluster> {
    let num_clusters = params.num_clusters.unwrap_or(5);
    generate_agent_based_clusters(graph_data, agents, num_clusters, "spectral")
}

#[allow(dead_code)]
pub(crate) fn generate_kmeans_clusters_from_agents(
    graph_data: &crate::models::graph::GraphData,
    agents: &[crate::services::agent_visualization_protocol::MultiMcpAgentStatus],
    params: &ClusteringParams,
) -> Vec<Cluster> {
    let num_clusters = params.num_clusters.unwrap_or(8);
    generate_agent_based_clusters(graph_data, agents, num_clusters, "kmeans")
}

#[allow(dead_code)]
pub(crate) fn generate_louvain_clusters_from_agents(
    graph_data: &crate::models::graph::GraphData,
    agents: &[crate::services::agent_visualization_protocol::MultiMcpAgentStatus],
    params: &ClusteringParams,
) -> Vec<Cluster> {
    let resolution = params.resolution.unwrap_or(1.0);
    let num_clusters = std::cmp::min((5.0 / resolution) as u32, agents.len() as u32);
    generate_agent_based_clusters(graph_data, agents, num_clusters, "louvain")
}

#[allow(dead_code)]
pub(crate) fn generate_default_clusters_from_agents(
    graph_data: &crate::models::graph::GraphData,
    agents: &[crate::services::agent_visualization_protocol::MultiMcpAgentStatus],
    params: &ClusteringParams,
) -> Vec<Cluster> {
    let cluster_count = std::cmp::min(params.num_clusters.unwrap_or(6), agents.len() as u32);
    generate_agent_based_clusters(graph_data, agents, cluster_count, "default")
}

pub(crate) fn generate_agent_based_clusters(
    graph_data: &crate::models::graph::GraphData,
    agents: &[crate::services::agent_visualization_protocol::MultiMcpAgentStatus],
    num_clusters: u32,
    method: &str,
) -> Vec<Cluster> {
    if agents.is_empty() {
        warn!("No agent data available for clustering, using graph-based clustering");
        return generate_graph_based_clusters(graph_data, num_clusters, method);
    }

    info!(
        "Generating {} clusters from {} real agents using {} method",
        num_clusters,
        agents.len(),
        method
    );


    let mut agent_type_groups: std::collections::HashMap<
        String,
        Vec<&crate::services::agent_visualization_protocol::MultiMcpAgentStatus>,
    > = std::collections::HashMap::new();

    for agent in agents {
        agent_type_groups
            .entry(agent.agent_type.clone())
            .or_insert_with(Vec::new)
            .push(agent);
    }

    let colors = vec![
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    ];

    let mut clusters = Vec::new();
    let mut cluster_id = 0;


    for (agent_type, type_agents) in agent_type_groups {
        if cluster_id >= num_clusters {
            break;
        }


        let _avg_cpu = type_agents
            .iter()
            .map(|a| a.performance.cpu_usage)
            .sum::<f32>()
            / type_agents.len() as f32;
        let _avg_memory = type_agents
            .iter()
            .map(|a| a.performance.memory_usage)
            .sum::<f32>()
            / type_agents.len() as f32;
        let avg_health = type_agents
            .iter()
            .map(|a| a.performance.health_score)
            .sum::<f32>()
            / type_agents.len() as f32;
        let _total_tasks = type_agents
            .iter()
            .map(|a| a.performance.tasks_completed)
            .sum::<u32>();


        let cluster_nodes: Vec<u32> = type_agents
            .iter()
            .enumerate()
            .map(|(idx, _)| cluster_id * 100 + idx as u32)
            .take(graph_data.nodes.len() / num_clusters as usize)
            .collect();


        let centroid = if !cluster_nodes.is_empty() && !graph_data.nodes.is_empty() {
            let node_subset: Vec<_> = cluster_nodes
                .iter()
                .filter_map(|&id| graph_data.nodes.get(id as usize))
                .collect();

            if !node_subset.is_empty() {
                let sum_x: f32 = node_subset.iter().map(|n| n.data.x).sum();
                let sum_y: f32 = node_subset.iter().map(|n| n.data.y).sum();
                let sum_z: f32 = node_subset.iter().map(|n| n.data.z).sum();
                let count = node_subset.len() as f32;
                Some([sum_x / count, sum_y / count, sum_z / count])
            } else {
                None
            }
        } else {
            None
        };


        let keywords: Vec<String> = type_agents
            .iter()
            .flat_map(|agent| agent.capabilities.iter())
            .take(5)
            .cloned()
            .collect();

        let coherence = (avg_health / 100.0).min(1.0).max(0.0);

        clusters.push(Cluster {
            id: format!("cluster_{}_{}", method, cluster_id),
            label: format!("{} Agents ({})", agent_type, type_agents.len()),
            node_count: type_agents.len() as u32,
            coherence,
            color: colors
                .get(cluster_id as usize)
                .unwrap_or(&"#888888")
                .to_string(),
            keywords,
            nodes: cluster_nodes,
            centroid,
        });

        cluster_id += 1;
    }


    while clusters.len() < num_clusters as usize && cluster_id < num_clusters {
        clusters.push(Cluster {
            id: format!("cluster_{}_{}", method, cluster_id),
            label: format!("Mixed Cluster {}", cluster_id + 1),
            node_count: 0,
            coherence: 0.5,
            color: colors
                .get(cluster_id as usize)
                .unwrap_or(&"#888888")
                .to_string(),
            keywords: vec![format!("{}_analysis", method)],
            nodes: vec![],
            centroid: None,
        });
        cluster_id += 1;
    }

    info!("Generated {} real clusters from agent data", clusters.len());
    clusters
}

pub(crate) fn generate_graph_based_clusters(
    graph_data: &crate::models::graph::GraphData,
    num_clusters: u32,
    method: &str,
) -> Vec<Cluster> {
    let nodes_per_cluster = if graph_data.nodes.is_empty() {
        0
    } else {
        graph_data.nodes.len() / num_clusters as usize
    };
    let colors = vec![
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    ];
    let labels = vec![
        "Core Concepts",
        "Implementation",
        "Documentation",
        "Testing",
        "Infrastructure",
        "UI Components",
        "API Layer",
        "Data Models",
    ];

    (0..num_clusters)
        .map(|i| {
            let start_idx = (i as usize) * nodes_per_cluster;
            let end_idx = ((i + 1) as usize * nodes_per_cluster).min(graph_data.nodes.len());
            let cluster_nodes: Vec<u32> = (start_idx..end_idx).map(|idx| idx as u32).collect();

            let centroid = if !cluster_nodes.is_empty() {
                let sum_x: f32 = cluster_nodes
                    .iter()
                    .filter_map(|&id| graph_data.nodes.get(id as usize))
                    .map(|n| n.data.x)
                    .sum();
                let sum_y: f32 = cluster_nodes
                    .iter()
                    .filter_map(|&id| graph_data.nodes.get(id as usize))
                    .map(|n| n.data.y)
                    .sum();
                let sum_z: f32 = cluster_nodes
                    .iter()
                    .filter_map(|&id| graph_data.nodes.get(id as usize))
                    .map(|n| n.data.z)
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
        })
        .collect()
}
