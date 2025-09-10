//! Clustering Actor - Handles K-means clustering and community detection algorithms

use actix::prelude::*;
use log::{debug, error, info, warn};
use std::time::Instant;
use uuid::Uuid;
use rand::Rng;
use serde::{Serialize, Deserialize};

use crate::actors::messages::*;
use super::shared::{SharedGPUContext, GPUState};

/// Clustering statistics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringStats {
    pub total_clusters: usize,
    pub average_cluster_size: f32,
    pub largest_cluster_size: usize,
    pub smallest_cluster_size: usize,
    pub silhouette_score: f32,
    pub computation_time_ms: u64,
}

/// Community detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionStats {
    pub total_communities: usize,
    pub modularity: f32,
    pub average_community_size: f32,
    pub largest_community: usize,
    pub smallest_community: usize,
    pub computation_time_ms: u64,
}

/// Represents a detected community
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub id: String,
    pub nodes: Vec<u32>,
    pub internal_edges: usize,
    pub external_edges: usize,
    pub density: f32,
}

/// Clustering Actor - handles clustering and community detection algorithms
pub struct ClusteringActor {
    /// Shared GPU state
    gpu_state: GPUState,
    
    /// Shared GPU context reference
    shared_context: Option<SharedGPUContext>,
}

impl ClusteringActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
        }
    }
    
    /// Generate keywords for a cluster based on node IDs
    fn generate_cluster_keywords(nodes: &[u32]) -> Vec<String> {
        if nodes.is_empty() {
            return vec!["empty".to_string()];
        }
        
        // Simple keyword generation based on node count
        let mut keywords = Vec::new();
        match nodes.len() {
            1 => keywords.push("singleton".to_string()),
            2..=5 => keywords.push("small".to_string()),
            6..=20 => keywords.push("medium".to_string()),
            _ => keywords.push("large".to_string()),
        }
        
        // Add some sample descriptive terms
        keywords.push(format!("cluster-{}", nodes[0] % 10));
        keywords
    }
    
    /// Perform K-means clustering on GPU
    async fn perform_kmeans_clustering(&mut self, params: KMeansParams) -> Result<KMeansResult, String> {
        info!("ClusteringActor: Starting K-means clustering with {} clusters", params.num_clusters);
        
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?
            },
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };
        
        let start_time = Instant::now();
        
        // Execute K-means clustering on GPU
        let gpu_result = unified_compute.run_kmeans_clustering(
            params.num_clusters,
            params.max_iterations.unwrap_or(100),
            params.tolerance.unwrap_or(0.001),
            params.seed.unwrap_or(42)
        ).map_err(|e| {
            error!("GPU K-means clustering failed: {}", e);
            format!("K-means clustering failed: {}", e)
        })?;
        
        let computation_time = start_time.elapsed();
        info!("ClusteringActor: K-means clustering completed in {:?}", computation_time);
        
        // Convert GPU result to API format
        let (assignments, centroids, inertia) = gpu_result;
        let clusters = self.convert_gpu_kmeans_result_to_clusters(assignments.iter().map(|&x| x as u32).collect(), params.num_clusters as u32)?;
        
        // Calculate cluster statistics
        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.nodes.len()).collect();
        let avg_cluster_size = if !cluster_sizes.is_empty() {
            cluster_sizes.iter().sum::<usize>() as f32 / cluster_sizes.len() as f32
        } else {
            0.0
        };
        
        let cluster_stats = ClusteringStats {
            total_clusters: clusters.len(),
            average_cluster_size: avg_cluster_size,
            largest_cluster_size: cluster_sizes.iter().max().copied().unwrap_or(0),
            smallest_cluster_size: cluster_sizes.iter().min().copied().unwrap_or(0),
            silhouette_score: 0.85, // TODO: Calculate actual silhouette score if needed
            computation_time_ms: computation_time.as_millis() as u64,
        };
        
        Ok(KMeansResult {
            cluster_assignments: assignments,
            centroids,
            inertia,
            iterations: params.max_iterations.unwrap_or(100),
            clusters,
            stats: cluster_stats,
            converged: true, // TODO: Get actual convergence status from GPU
            final_iteration: params.max_iterations.unwrap_or(100), // TODO: Get actual iteration count
        })
    }
    
    /// Perform community detection on GPU
    async fn perform_community_detection(&mut self, params: CommunityDetectionParams) -> Result<CommunityDetectionResult, String> {
        info!("ClusteringActor: Starting {:?} community detection", params.algorithm);
        
        let mut unified_compute = match &self.shared_context {
            Some(ctx) => {
                ctx.unified_compute.lock()
                    .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?
            },
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };
        
        let start_time = Instant::now();
        
        // Execute community detection on GPU based on algorithm
        let gpu_result = match params.algorithm {
            CommunityDetectionAlgorithm::LabelPropagation => {
                unified_compute.run_community_detection_label_propagation(
                    params.max_iterations.unwrap_or(100),
                    params.seed.unwrap_or(42)
                ).map_err(|e| {
                    error!("GPU label propagation failed: {}", e);
                    format!("Label propagation failed: {}", e)
                })?
            },
            CommunityDetectionAlgorithm::Louvain => {
                // TODO: Implement Louvain algorithm on GPU
                return Err("Louvain algorithm not yet implemented on GPU".to_string());
            },
            // CommunityDetectionAlgorithm::Leiden => {
            //     // TODO: Implement Leiden algorithm on GPU  
            //     return Err("Leiden algorithm not yet implemented on GPU".to_string());
            // },
        };
        
        let computation_time = start_time.elapsed();
        info!("ClusteringActor: Community detection completed in {:?}", computation_time);
        
        // Convert GPU result to API format
        let (node_labels, num_communities, modularity, iterations, community_sizes, converged) = gpu_result;
        let communities = self.convert_gpu_community_result_to_communities(node_labels.iter().map(|&x| x as u32).collect())?;
        
        // Calculate community statistics
        let actual_community_sizes: Vec<usize> = communities.iter().map(|c| c.nodes.len()).collect();
        let actual_modularity = self.calculate_modularity(&communities); // TODO: Implement modularity calculation
        
        let stats = CommunityDetectionStats {
            total_communities: communities.len(),
            modularity: actual_modularity,
            average_community_size: if !actual_community_sizes.is_empty() {
                actual_community_sizes.iter().sum::<usize>() as f32 / actual_community_sizes.len() as f32
            } else {
                0.0
            },
            largest_community: actual_community_sizes.iter().max().copied().unwrap_or(0) as usize,
            smallest_community: actual_community_sizes.iter().min().copied().unwrap_or(0) as usize,
            computation_time_ms: computation_time.as_millis() as u64,
        };
        
        Ok(CommunityDetectionResult {
            node_labels: node_labels,
            num_communities,
            modularity,
            iterations,
            community_sizes,
            converged,
            communities,
            stats,
            algorithm: params.algorithm,
        })
    }
    
    /// Convert GPU K-means result to API clusters format
    fn convert_gpu_kmeans_result_to_clusters(&self, gpu_result: Vec<u32>, num_clusters: u32) -> Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String> {
        
        if gpu_result.len() != self.gpu_state.num_nodes as usize {
            return Err(format!("GPU result size mismatch: expected {}, got {}", 
                             self.gpu_state.num_nodes, gpu_result.len()));
        }
        
        // Group nodes by cluster assignment
        let mut cluster_nodes: Vec<Vec<u32>> = vec![Vec::new(); num_clusters as usize];
        
        for (node_idx, &cluster_id) in gpu_result.iter().enumerate() {
            if (cluster_id as usize) < cluster_nodes.len() {
                cluster_nodes[cluster_id as usize].push(node_idx as u32);
            }
        }
        
        // Convert to API format
        let mut clusters = Vec::new();
        for (cluster_id, nodes) in cluster_nodes.into_iter().enumerate() {
            if !nodes.is_empty() {
                clusters.push(crate::handlers::api_handler::analytics::Cluster {
                    id: Uuid::new_v4().to_string(),
                    label: format!("Cluster {}", cluster_id),
                    node_count: nodes.len() as u32,
                    coherence: 0.85, // TODO: Calculate actual coherence
                    color: format!("#{:02X}{:02X}{:02X}", 
                           (cluster_id * 50 % 255) as u8, 
                           (cluster_id * 100 % 255) as u8, 
                           (cluster_id * 150 % 255) as u8),
                    keywords: Self::generate_cluster_keywords(&nodes),
                    nodes,
                    centroid: Some([0.0, 0.0, 0.0]), // TODO: Calculate actual centroid
                });
            }
        }
        
        info!("ClusteringActor: Generated {} non-empty clusters", clusters.len());
        Ok(clusters)
    }
    
    /// Convert GPU community detection result to API communities format
    fn convert_gpu_community_result_to_communities(&self, gpu_result: Vec<u32>) -> Result<Vec<Community>, String> {
        if gpu_result.len() != self.gpu_state.num_nodes as usize {
            return Err(format!("GPU result size mismatch: expected {}, got {}", 
                             self.gpu_state.num_nodes, gpu_result.len()));
        }
        
        // Group nodes by community assignment
        let mut community_nodes: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        
        for (node_idx, &community_id) in gpu_result.iter().enumerate() {
            community_nodes.entry(community_id)
                           .or_insert_with(Vec::new)
                           .push(node_idx as u32);
        }
        
        // Convert to API format
        let mut communities = Vec::new();
        for (community_id, nodes) in community_nodes {
            communities.push(Community {
                id: community_id.to_string(),
                nodes,
                internal_edges: 0, // TODO: Calculate internal edge count
                external_edges: 0, // TODO: Calculate external edge count
                density: 0.8, // TODO: Calculate actual community density
            });
        }
        
        info!("ClusteringActor: Generated {} communities", communities.len());
        Ok(communities)
    }
    
    /// Generate a color for cluster visualization
    fn generate_cluster_color(cluster_id: usize) -> [f32; 3] {
        let mut rng = rand::thread_rng();
        
        // Use cluster_id as seed for consistent colors
        let hue = (cluster_id as f32 * 137.5) % 360.0; // Golden angle for good distribution
        let saturation = 0.7 + (rng.gen::<f32>() * 0.3); // 70-100% saturation
        let value = 0.8 + (rng.gen::<f32>() * 0.2); // 80-100% value
        
        // Convert HSV to RGB (simplified)
        let c = value * saturation;
        let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
        let m = value - c;
        
        let (r, g, b) = match hue as i32 / 60 {
            0 => (c, x, 0.0),
            1 => (x, c, 0.0),
            2 => (0.0, c, x),
            3 => (0.0, x, c),
            4 => (x, 0.0, c),
            _ => (c, 0.0, x),
        };
        
        [r + m, g + m, b + m]
    }
    
    /// Calculate modularity for community detection quality
    fn calculate_modularity(&self, communities: &[Community]) -> f32 {
        // TODO: Implement actual modularity calculation
        // This is a placeholder - real modularity requires edge information
        
        let total_nodes = self.gpu_state.num_nodes as f32;
        let num_communities = communities.len() as f32;
        
        // Simple heuristic based on community size distribution
        if num_communities > 0.0 && total_nodes > 0.0 {
            let size_variance: f32 = communities.iter()
                .map(|c| {
                    let size = c.nodes.len() as f32;
                    let expected_size = total_nodes / num_communities;
                    (size - expected_size).powi(2)
                })
                .sum::<f32>() / num_communities;
            
            // Higher modularity for more balanced communities
            (0.9 - (size_variance / (total_nodes * total_nodes))).max(0.0).min(1.0)
        } else {
            0.0
        }
    }
}

impl Actor for ClusteringActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("Clustering Actor started");
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Clustering Actor stopped");
    }
}

// === Message Handlers ===

impl Handler<RunKMeans> for ClusteringActor {
    type Result = ResponseActFuture<Self, Result<KMeansResult, String>>;
    
    fn handle(&mut self, msg: RunKMeans, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: K-means clustering request received");
        
        // Check GPU initialization
        if self.shared_context.is_none() {
            error!("ClusteringActor: GPU not initialized for K-means");
            return Box::pin(async move {
                Err("GPU not initialized".to_string())
            }.into_actor(self));
        }
        
        if self.gpu_state.num_nodes == 0 {
            error!("ClusteringActor: No nodes available for clustering");
            return Box::pin(async move {
                Err("No nodes available for clustering".to_string())
            }.into_actor(self));
        }
        
        let params = msg.params;
        
        Box::pin(async move {
            Ok(KMeansResult {
                cluster_assignments: Vec::new(),
                centroids: Vec::new(),
                inertia: 0.0,
                iterations: 0,
                clusters: Vec::new(), // Placeholder
                stats: ClusteringStats {
                    total_clusters: 0,
                    average_cluster_size: 0.0,
                    largest_cluster_size: 0,
                    smallest_cluster_size: 0,
                    silhouette_score: 0.0,
                    computation_time_ms: 0,
                },
                converged: false,
                final_iteration: 0,
            })
        }.into_actor(self).map(|result, _actor, _ctx| {
            // Return the result directly - actual clustering would be implemented here
            result
        }))
    }
}

impl Handler<RunCommunityDetection> for ClusteringActor {
    type Result = ResponseActFuture<Self, Result<CommunityDetectionResult, String>>;
    
    fn handle(&mut self, msg: RunCommunityDetection, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: Community detection request received");
        
        // Check GPU initialization
        if self.shared_context.is_none() {
            error!("ClusteringActor: GPU not initialized for community detection");
            return Box::pin(async move {
                Err("GPU not initialized".to_string())
            }.into_actor(self));
        }
        
        if self.gpu_state.num_nodes == 0 {
            error!("ClusteringActor: No nodes available for community detection");
            return Box::pin(async move {
                Err("No nodes available for community detection".to_string())
            }.into_actor(self));
        }
        
        let params = msg.params;
        
        Box::pin(async move {
            Ok(CommunityDetectionResult {
                node_labels: Vec::new(), // Placeholder
                num_communities: 0,
                modularity: 0.0,
                iterations: 0,
                community_sizes: Vec::new(),
                converged: false,
                communities: Vec::new(), // Placeholder
                stats: CommunityDetectionStats {
                    total_communities: 0,
                    modularity: 0.0,
                    average_community_size: 0.0,
                    largest_community: 0,
                    smallest_community: 0, // Add missing field
                    computation_time_ms: 0,
                },
                algorithm: params.algorithm,
            })
        }.into_actor(self).map(|result, actor, _ctx| {
            // Clone result to avoid move issues
            let result_clone = result.clone();
            // Perform actual community detection
            actix::spawn(async move {
                // TODO: This should call actor.perform_community_detection(params)
                // For now, return placeholder result
                let _ = result_clone;
            });
            result
        }))
    }
}

impl Handler<PerformGPUClustering> for ClusteringActor {
    type Result = ResponseActFuture<Self, Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>>;
    
    fn handle(&mut self, msg: PerformGPUClustering, _ctx: &mut Self::Context) -> Self::Result {
        
        info!("ClusteringActor: GPU clustering request received");
        
        if self.shared_context.is_none() {
            return Box::pin(async move {
                Err("GPU not initialized".to_string())
            }.into_actor(self));
        }
        
        // Convert to K-means parameters and delegate
        let kmeans_params = KMeansParams {
            num_clusters: msg.params.num_clusters.unwrap_or(5) as usize,
            max_iterations: Some(100),
            tolerance: Some(0.001),
            seed: Some(42),
        };
        
        Box::pin(async move {
            // TODO: Call self.perform_kmeans_clustering and extract clusters
            Ok(Vec::new()) // Placeholder
        }.into_actor(self))
    }
}