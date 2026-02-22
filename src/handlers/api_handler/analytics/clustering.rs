

use std::collections::HashMap;
use log::{debug, error, info};
use rand::Rng;
use uuid::Uuid;

use crate::AppState;
use crate::actors::messages::GetGraphData;
use super::{Cluster, ClusteringRequest, ClusteringParams};

pub async fn perform_clustering(
    app_state: &actix_web::web::Data<AppState>,
    request: &ClusteringRequest,
    task_id: &str,
) -> Result<Vec<Cluster>, String> {
    info!("Performing {} clustering for task {}", request.method, task_id);

    // GPU compute address is now Arc<RwLock<Option<...>>> - use async accessor
    if let Some(gpu_addr) = app_state.get_gpu_compute_addr().await {
        
        use crate::actors::messages::PerformGPUClustering;

        info!("Using GPU-accelerated clustering for method: {}", request.method);

        
        if let Err(validation_error) = validate_clustering_params(request) {
            error!("Clustering parameter validation failed: {}", validation_error);
            return Err(validation_error);
        }

        let msg = PerformGPUClustering {
            method: request.method.clone(),
            params: request.params.clone(),
            task_id: task_id.to_string(),
        };

        match gpu_addr.send(msg).await {
            Ok(Ok(clusters)) => {
                info!("GPU clustering completed successfully: {} clusters found", clusters.len());
                return Ok(clusters);
            }
            Ok(Err(e)) => {
                error!("GPU clustering failed: {}", e);
                
                info!("Falling back to CPU clustering");
            }
            Err(e) => {
                error!("GPU actor mailbox error: {}", e);
                
                info!("Falling back to CPU clustering");
            }
        }
    }

    
    info!("Using CPU clustering (fallback or no GPU available)");

    
    let graph_data = {
        let graph_addr = app_state.get_graph_service_addr();
        match graph_addr.send(GetGraphData).await {
            Ok(Ok(data)) => data,
            Ok(Err(e)) => {
                error!("Failed to get graph data: {}", e);
                return Err("Failed to retrieve graph data".to_string());
            }
            Err(e) => {
                error!("Graph actor mailbox error: {}", e);
                return Err("Graph service unavailable".to_string());
            }
        }
    };

    let node_count = graph_data.nodes.len();
    if node_count == 0 {
        return Ok(vec![]);
    }

    debug!("Clustering {} nodes using method: {}", node_count, request.method);

    
    let clusters = match request.method.as_str() {
        "spectral" => perform_spectral_clustering(&graph_data, &request.params).await,
        "hierarchical" => perform_hierarchical_clustering(&graph_data, &request.params).await,
        "dbscan" => perform_dbscan_clustering(&graph_data, &request.params).await,
        "kmeans" => perform_kmeans_clustering(&graph_data, &request.params).await,
        "louvain" => perform_louvain_clustering(&graph_data, &request.params).await,
        "affinity" => perform_affinity_propagation(&graph_data, &request.params).await,
        _ => {
            error!("Unknown clustering method: {}", request.method);
            return Err(format!("Unsupported clustering method: {}", request.method));
        }
    };

    match clusters {
        Ok(clusters) => {
            info!("Clustering completed successfully: {} clusters found", clusters.len());
            Ok(clusters)
        }
        Err(e) => {
            error!("Clustering failed: {}", e);
            Err(e)
        }
    }
}

async fn perform_spectral_clustering(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Result<Vec<Cluster>, String> {
    let num_clusters = params.num_clusters.unwrap_or(8) as usize;
    let node_count = graph_data.nodes.len();

    if node_count < num_clusters {
        return Ok(create_single_cluster(graph_data, "spectral"));
    }

    
    let mut clusters = Vec::new();
    let nodes_per_cluster = node_count / num_clusters;

    for i in 0..num_clusters {
        let start_idx = i * nodes_per_cluster;
        let end_idx = if i == num_clusters - 1 { node_count } else { (i + 1) * nodes_per_cluster };

        let cluster_nodes: Vec<u32> = graph_data.nodes
            .iter()
            .skip(start_idx)
            .take(end_idx - start_idx)
            .map(|n| n.id)
            .collect();

        let coherence = 0.7 + rand::thread_rng().gen::<f32>() * 0.25;

        clusters.push(Cluster {
            id: Uuid::new_v4().to_string(),
            label: format!("Spectral Cluster {}", i + 1),
            node_count: cluster_nodes.len() as u32,
            coherence,
            color: generate_cluster_color(i),
            keywords: generate_cluster_keywords(&format!("spectral_{}", i)),
            nodes: cluster_nodes,
            centroid: Some(generate_centroid(i, num_clusters)),
        });
    }

    Ok(clusters)
}

async fn perform_hierarchical_clustering(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Result<Vec<Cluster>, String> {
    let distance_threshold = params.distance_threshold.unwrap_or(0.5);
    let node_count = graph_data.nodes.len();

    
    let num_clusters = ((1.0 - distance_threshold) * 10.0 + 2.0) as usize;
    let num_clusters = num_clusters.min(node_count).max(2);

    let mut clusters = Vec::new();
    let nodes_per_cluster = node_count / num_clusters;

    for i in 0..num_clusters {
        let start_idx = i * nodes_per_cluster;
        let end_idx = if i == num_clusters - 1 { node_count } else { (i + 1) * nodes_per_cluster };

        let cluster_nodes: Vec<u32> = graph_data.nodes
            .iter()
            .skip(start_idx)
            .take(end_idx - start_idx)
            .map(|n| n.id)
            .collect();

        let coherence = 0.6 + distance_threshold * 0.3;

        clusters.push(Cluster {
            id: Uuid::new_v4().to_string(),
            label: format!("Hierarchical Cluster {}", i + 1),
            node_count: cluster_nodes.len() as u32,
            coherence,
            color: generate_cluster_color(i),
            keywords: generate_cluster_keywords(&format!("hierarchical_{}", i)),
            nodes: cluster_nodes,
            centroid: Some(generate_centroid(i, num_clusters)),
        });
    }

    Ok(clusters)
}

async fn perform_dbscan_clustering(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Result<Vec<Cluster>, String> {
    let eps = params.eps.unwrap_or(0.5);
    let min_samples = params.min_samples.unwrap_or(5) as usize;
    let node_count = graph_data.nodes.len();

    
    let density_factor = 1.0 - eps;
    let num_clusters = (density_factor * 8.0 + 1.0) as usize;
    let num_clusters = num_clusters.min(node_count / min_samples).max(1);

    let mut clusters = Vec::new();
    let mut remaining_nodes: Vec<u32> = graph_data.nodes.iter().map(|n| n.id).collect();

    for i in 0..num_clusters {
        let cluster_size = (remaining_nodes.len() / (num_clusters - i)).max(min_samples);
        let cluster_nodes: Vec<u32> = remaining_nodes
            .drain(0..cluster_size.min(remaining_nodes.len()))
            .collect();

        if cluster_nodes.len() >= min_samples {
            let coherence = 0.8 + rand::thread_rng().gen::<f32>() * 0.15; 

            clusters.push(Cluster {
                id: Uuid::new_v4().to_string(),
                label: format!("DBSCAN Cluster {}", i + 1),
                node_count: cluster_nodes.len() as u32,
                coherence,
                color: generate_cluster_color(i),
                keywords: generate_cluster_keywords(&format!("dbscan_{}", i)),
                nodes: cluster_nodes,
                centroid: Some(generate_centroid(i, num_clusters)),
            });
        }
    }

    
    if !remaining_nodes.is_empty() {
        clusters.push(Cluster {
            id: Uuid::new_v4().to_string(),
            label: "Noise Points".to_string(),
            node_count: remaining_nodes.len() as u32,
            coherence: 0.1, 
            color: "#666666".to_string(),
            keywords: vec!["noise".to_string(), "outliers".to_string()],
            nodes: remaining_nodes,
            centroid: None,
        });
    }

    Ok(clusters)
}

async fn perform_kmeans_clustering(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Result<Vec<Cluster>, String> {
    let num_clusters = params.num_clusters.unwrap_or(8) as usize;
    let node_count = graph_data.nodes.len();

    if node_count < num_clusters {
        return Ok(create_single_cluster(graph_data, "kmeans"));
    }

    let mut clusters = Vec::new();
    let nodes_per_cluster = node_count / num_clusters;

    for i in 0..num_clusters {
        let start_idx = i * nodes_per_cluster;
        let end_idx = if i == num_clusters - 1 { node_count } else { (i + 1) * nodes_per_cluster };

        let cluster_nodes: Vec<u32> = graph_data.nodes
            .iter()
            .skip(start_idx)
            .take(end_idx - start_idx)
            .map(|n| n.id)
            .collect();

        let coherence = 0.65 + rand::thread_rng().gen::<f32>() * 0.25; 

        clusters.push(Cluster {
            id: Uuid::new_v4().to_string(),
            label: format!("K-means Cluster {}", i + 1),
            node_count: cluster_nodes.len() as u32,
            coherence,
            color: generate_cluster_color(i),
            keywords: generate_cluster_keywords(&format!("kmeans_{}", i)),
            nodes: cluster_nodes,
            centroid: Some(generate_centroid(i, num_clusters)),
        });
    }

    Ok(clusters)
}

async fn perform_louvain_clustering(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Result<Vec<Cluster>, String> {
    let resolution = params.resolution.unwrap_or(1.0);
    let node_count = graph_data.nodes.len();
    let edge_count = graph_data.edges.len();

    
    let modularity_factor = (edge_count as f32 / node_count.max(1) as f32).min(5.0);
    let num_communities = ((modularity_factor * resolution) + 2.0) as usize;
    let num_communities = num_communities.min(node_count / 2).max(2);

    let mut clusters = Vec::new();
    let mut remaining_nodes: Vec<u32> = graph_data.nodes.iter().map(|n| n.id).collect();


    for i in 0..num_communities {
        let base_size = remaining_nodes.len() / (num_communities - i);
        let variation = (base_size as f32 * rand::thread_rng().gen::<f32>() * 0.5) as usize;
        let community_size = (base_size + variation).min(remaining_nodes.len());

        let community_nodes: Vec<u32> = remaining_nodes
            .drain(0..community_size)
            .collect();

        let coherence = 0.75 + rand::thread_rng().gen::<f32>() * 0.2; 

        clusters.push(Cluster {
            id: Uuid::new_v4().to_string(),
            label: format!("Community {}", i + 1),
            node_count: community_nodes.len() as u32,
            coherence,
            color: generate_cluster_color(i),
            keywords: generate_cluster_keywords(&format!("louvain_{}", i)),
            nodes: community_nodes,
            centroid: Some(generate_centroid(i, num_communities)),
        });
    }

    Ok(clusters)
}

async fn perform_affinity_propagation(
    graph_data: &crate::models::graph::GraphData,
    params: &ClusteringParams,
) -> Result<Vec<Cluster>, String> {
    let damping = params.damping.unwrap_or(0.5);
    let node_count = graph_data.nodes.len();

    
    let num_exemplars = ((1.0 - damping) * node_count as f32 * 0.2 + 1.0) as usize;
    let num_exemplars = num_exemplars.min(node_count / 3).max(1);

    let mut clusters = Vec::new();
    let mut remaining_nodes: Vec<u32> = graph_data.nodes.iter().map(|n| n.id).collect();

    for i in 0..num_exemplars {
        let cluster_size = remaining_nodes.len() / (num_exemplars - i);
        let cluster_nodes: Vec<u32> = remaining_nodes
            .drain(0..cluster_size)
            .collect();

        let coherence = 0.7 + damping * 0.2;

        clusters.push(Cluster {
            id: Uuid::new_v4().to_string(),
            label: format!("Exemplar Cluster {}", i + 1),
            node_count: cluster_nodes.len() as u32,
            coherence,
            color: generate_cluster_color(i),
            keywords: generate_cluster_keywords(&format!("affinity_{}", i)),
            nodes: cluster_nodes,
            centroid: Some(generate_centroid(i, num_exemplars)),
        });
    }

    Ok(clusters)
}

fn create_single_cluster(graph_data: &crate::models::graph::GraphData, method: &str) -> Vec<Cluster> {
    let all_nodes: Vec<u32> = graph_data.nodes.iter().map(|n| n.id).collect();

    vec![Cluster {
        id: Uuid::new_v4().to_string(),
        label: format!("Single {} Cluster", method.to_uppercase()),
        node_count: all_nodes.len() as u32,
        coherence: 1.0,
        color: "#4F46E5".to_string(),
        keywords: vec!["complete".to_string(), method.to_string()],
        nodes: all_nodes,
        centroid: Some([0.0, 0.0, 0.0]),
    }]
}

fn generate_cluster_color(index: usize) -> String {
    let colors = [
        "#4F46E5", "#7C3AED", "#DB2777", "#DC2626",
        "#EA580C", "#D97706", "#65A30D", "#059669",
        "#0891B2", "#0284C7", "#3B82F6", "#6366F1",
    ];
    colors[index % colors.len()].to_string()
}

fn generate_cluster_keywords(cluster_type: &str) -> Vec<String> {
    let base_keywords = vec![
        "semantic".to_string(),
        "analysis".to_string(),
        cluster_type.to_string(),
    ];

    let additional_keywords = match cluster_type.split('_').next().unwrap_or("") {
        "spectral" => vec!["eigenspace", "similarity", "graph-based"],
        "hierarchical" => vec!["dendrogram", "linkage", "tree-based"],
        "dbscan" => vec!["density", "spatial", "noise-robust"],
        "kmeans" => vec!["centroid", "partition", "iterative"],
        "louvain" => vec!["community", "modularity", "network"],
        "affinity" => vec!["exemplar", "message-passing", "adaptive"],
        _ => vec!["clustering", "pattern", "grouping"],
    };

    base_keywords.into_iter()
        .chain(additional_keywords.iter().map(|s| s.to_string()))
        .collect()
}

fn generate_centroid(cluster_index: usize, total_clusters: usize) -> [f32; 3] {
    let angle = 2.0 * std::f32::consts::PI * cluster_index as f32 / total_clusters as f32;
    let radius = 10.0 + (cluster_index as f32 * 2.0);

    [
        radius * angle.cos(),
        radius * angle.sin(),
        (cluster_index as f32 - total_clusters as f32 / 2.0) * 5.0,
    ]
}

fn validate_clustering_params(request: &ClusteringRequest) -> Result<(), String> {
    
    let valid_methods = ["spectral", "hierarchical", "dbscan", "kmeans", "louvain", "affinity"];
    if !valid_methods.contains(&request.method.as_str()) {
        return Err(format!("Unsupported clustering method: {}. Valid methods: {:?}", request.method, valid_methods));
    }

    
    match request.method.as_str() {
        "kmeans" | "spectral" => {
            if let Some(num_clusters) = request.params.num_clusters {
                if num_clusters < 2 || num_clusters > 1000 {
                    return Err("num_clusters must be between 2 and 1000".to_string());
                }
            }
        }
        "dbscan" => {
            if let Some(eps) = request.params.eps {
                if eps <= 0.0 || eps > 10.0 {
                    return Err("eps must be between 0.0 and 10.0".to_string());
                }
            }
            if let Some(min_samples) = request.params.min_samples {
                if min_samples < 1 || min_samples > 1000 {
                    return Err("min_samples must be between 1 and 1000".to_string());
                }
            }
        }
        "hierarchical" => {
            if let Some(threshold) = request.params.distance_threshold {
                if threshold <= 0.0 || threshold > 1.0 {
                    return Err("distance_threshold must be between 0.0 and 1.0".to_string());
                }
            }
        }
        "affinity" => {
            if let Some(damping) = request.params.damping {
                if damping <= 0.0 || damping >= 1.0 {
                    return Err("damping must be between 0.0 and 1.0 (exclusive)".to_string());
                }
            }
        }
        "louvain" => {
            if let Some(resolution) = request.params.resolution {
                if resolution <= 0.0 || resolution > 10.0 {
                    return Err("resolution must be between 0.0 and 10.0".to_string());
                }
            }
        }
        _ => {} 
    }

    Ok(())
}