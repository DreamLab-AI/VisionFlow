/*!
 * Community Detection Implementation for Analytics API
 * 
 * GPU-accelerated community detection algorithms for network analysis.
 * Supports label propagation algorithm with modularity optimization.
 */

use std::collections::HashMap;
use log::{debug, error, info};
use uuid::Uuid;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::AppState;
use crate::actors::messages::{RunCommunityDetection, CommunityDetectionParams, CommunityDetectionAlgorithm};

/// Community detection request structure
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunityDetectionRequest {
    pub algorithm: String,
    pub max_iterations: Option<u32>,
    pub convergence_tolerance: Option<f32>,
    pub synchronous: Option<bool>,
    pub seed: Option<u32>,
}

/// Community structure for API response
#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Community {
    pub id: String,
    pub label: String,
    pub nodes: Vec<u32>,
    pub size: u32,
    pub modularity_contribution: f32,
    pub color: String,
    pub center_node: Option<u32>, // Representative node (highest degree in community)
}

/// Community detection result for API
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunityDetectionResponse {
    pub success: bool,
    pub communities: Vec<Community>,
    pub total_communities: usize,
    pub modularity: f32,
    pub iterations: u32,
    pub converged: bool,
    pub algorithm: String,
    pub processing_time_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Run GPU-accelerated community detection
pub async fn run_gpu_community_detection(
    app_state: &actix_web::web::Data<AppState>,
    request: &CommunityDetectionRequest,
) -> Result<CommunityDetectionResponse, String> {
    let start_time = std::time::Instant::now();
    info!("Running GPU community detection with algorithm: {}", request.algorithm);
    
    // Check if GPU compute actor is available
    let gpu_addr = app_state.gpu_compute_addr.as_ref()
        .ok_or_else(|| "GPU compute actor not available".to_string())?;
    
    // Parse algorithm
    let algorithm = match request.algorithm.as_str() {
        "label_propagation" | "lp" => CommunityDetectionAlgorithm::LabelPropagation,
        _ => return Err(format!("Unsupported community detection algorithm: {}", request.algorithm)),
    };
    
    // Set default parameters
    let params = CommunityDetectionParams {
        algorithm: algorithm.clone(),
        max_iterations: request.max_iterations.unwrap_or(100),
        convergence_tolerance: request.convergence_tolerance.unwrap_or(0.001),
        synchronous: request.synchronous.unwrap_or(true),
        seed: request.seed.unwrap_or(42),
    };
    
    // Validate parameters
    validate_community_params(&params)?;
    
    let msg = RunCommunityDetection { params };
    
    match gpu_addr.send(msg).await {
        Ok(Ok(result)) => {
            let processing_time = start_time.elapsed().as_millis() as u64;
            info!("GPU community detection completed: {} communities found with modularity {:.4} in {} iterations", 
                  result.num_communities, result.modularity, result.iterations);
            
            let communities = convert_gpu_result_to_communities(result.clone())?;
            
            Ok(CommunityDetectionResponse {
                success: true,
                communities,
                total_communities: result.num_communities,
                modularity: result.modularity,
                iterations: result.iterations,
                converged: result.converged,
                algorithm: request.algorithm.clone(),
                processing_time_ms: processing_time,
                error: None,
            })
        }
        Ok(Err(e)) => {
            error!("GPU community detection failed: {}", e);
            Ok(CommunityDetectionResponse {
                success: false,
                communities: vec![],
                total_communities: 0,
                modularity: 0.0,
                iterations: 0,
                converged: false,
                algorithm: request.algorithm.clone(),
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                error: Some(e),
            })
        }
        Err(e) => {
            error!("GPU actor mailbox error: {}", e);
            Err(format!("Failed to communicate with GPU actor: {}", e))
        }
    }
}

/// Validate community detection parameters
fn validate_community_params(params: &CommunityDetectionParams) -> Result<(), String> {
    if params.max_iterations == 0 || params.max_iterations > 10000 {
        return Err("max_iterations must be between 1 and 10000".to_string());
    }
    
    if params.convergence_tolerance <= 0.0 || params.convergence_tolerance > 1.0 {
        return Err("convergence_tolerance must be between 0.0 and 1.0".to_string());
    }
    
    Ok(())
}

/// Convert GPU community detection result to API communities
fn convert_gpu_result_to_communities(
    result: crate::actors::messages::CommunityDetectionResult,
) -> Result<Vec<Community>, String> {
    let mut communities = Vec::new();
    let mut community_nodes: HashMap<i32, Vec<u32>> = HashMap::new();
    
    // Group nodes by community label
    for (node_id, &label) in result.node_labels.iter().enumerate() {
        community_nodes.entry(label)
            .or_insert_with(Vec::new)
            .push(node_id as u32);
    }
    
    // Create community objects
    for (community_id, nodes) in community_nodes {
        let size = nodes.len() as u32;
        
        // Calculate modularity contribution (approximation)
        let modularity_contribution = if result.num_communities > 0 {
            result.modularity / result.num_communities as f32
        } else {
            0.0
        };
        
        // Find center node (for now, use first node - in real implementation, 
        // this would be the node with highest degree within the community)
        let center_node = nodes.first().cloned();
        
        communities.push(Community {
            id: Uuid::new_v4().to_string(),
            label: format!("Community {}", community_id),
            nodes: nodes.clone(),
            size,
            modularity_contribution,
            color: generate_community_color(community_id as usize),
            center_node,
        });
    }
    
    // Sort by size (largest first)
    communities.sort_by(|a, b| b.size.cmp(&a.size));
    
    Ok(communities)
}

/// Generate community color based on ID
fn generate_community_color(community_id: usize) -> String {
    let colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57",
        "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43",
        "#C44569", "#40407A", "#706FD3", "#F97F51", "#833471",
        "#A55EEA", "#26D0CE", "#FD79A8", "#FDCB6E", "#6C5CE7"
    ];
    colors[community_id % colors.len()].to_string()
}

/// Get community statistics
pub fn get_community_statistics(communities: &[Community]) -> HashMap<String, serde_json::Value> {
    let mut stats = HashMap::new();
    
    if communities.is_empty() {
        return stats;
    }
    
    // Basic statistics
    let total_nodes: u32 = communities.iter().map(|c| c.size).sum();
    let avg_size = total_nodes as f32 / communities.len() as f32;
    let max_size = communities.iter().map(|c| c.size).max().unwrap_or(0);
    let min_size = communities.iter().map(|c| c.size).min().unwrap_or(0);
    
    // Size distribution
    let mut size_counts = HashMap::new();
    for community in communities {
        let size_range = match community.size {
            1..=5 => "small",
            6..=20 => "medium", 
            21..=100 => "large",
            _ => "very_large",
        };
        *size_counts.entry(size_range).or_insert(0) += 1;
    }
    
    stats.insert("total_communities".to_string(), serde_json::Value::Number(
        serde_json::Number::from(communities.len())
    ));
    stats.insert("total_nodes".to_string(), serde_json::Value::Number(
        serde_json::Number::from(total_nodes)
    ));
    stats.insert("avg_community_size".to_string(), serde_json::Value::Number(
        serde_json::Number::from_f64(avg_size as f64).unwrap()
    ));
    stats.insert("max_community_size".to_string(), serde_json::Value::Number(
        serde_json::Number::from(max_size)
    ));
    stats.insert("min_community_size".to_string(), serde_json::Value::Number(
        serde_json::Number::from(min_size)
    ));
    
    // Convert size_counts to JSON
    let size_dist: HashMap<String, serde_json::Value> = size_counts.into_iter()
        .map(|(k, v)| (k.to_string(), serde_json::Value::Number(serde_json::Number::from(v))))
        .collect();
    stats.insert("size_distribution".to_string(), serde_json::Value::Object(size_dist.into_iter().collect()));
    
    stats
}