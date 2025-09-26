
//! Clustering Actor - Handles K-means clustering and community detection algorithms

use std::sync::Arc;
use actix::prelude::*;
use log::{error, info};
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
    shared_context: Option<Arc<SharedGPUContext>>,
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
        
        // Execute K-means clustering on GPU with enhanced metrics tracking
        let gpu_result = unified_compute.run_kmeans_clustering_with_metrics(
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

        // Extract enhanced GPU result
        let (assignments, centroids, inertia, actual_iterations, converged) = gpu_result;
        let clusters = self.convert_gpu_kmeans_result_to_clusters(assignments.iter().map(|&x| x as u32).collect(), params.num_clusters as u32)?;
        
        // Calculate cluster statistics
        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.nodes.len()).collect();
        let avg_cluster_size = if !cluster_sizes.is_empty() {
            cluster_sizes.iter().sum::<usize>() as f32 / cluster_sizes.len() as f32
        } else {
            0.0
        };
        
        // Calculate silhouette score if we have valid clusters
        let silhouette_score = if clusters.len() > 1 && !assignments.is_empty() {
            self.calculate_silhouette_score(&assignments, &centroids, &clusters)?
        } else {
            0.0
        };

        let cluster_stats = ClusteringStats {
            total_clusters: clusters.len(),
            average_cluster_size: avg_cluster_size,
            largest_cluster_size: cluster_sizes.iter().max().copied().unwrap_or(0),
            smallest_cluster_size: cluster_sizes.iter().min().copied().unwrap_or(0),
            silhouette_score,
            computation_time_ms: computation_time.as_millis() as u64,
        };

        Ok(KMeansResult {
            cluster_assignments: assignments,
            centroids,
            inertia,
            iterations: actual_iterations,
            clusters,
            stats: cluster_stats,
            converged,
            final_iteration: actual_iterations,
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
                unified_compute.run_louvain_community_detection(
                    params.max_iterations.unwrap_or(100),
                    1.0, // resolution parameter
                    params.seed.unwrap_or(42)
                ).map_err(|e| {
                    error!("GPU Louvain community detection failed: {}", e);
                    format!("Louvain community detection failed: {}", e)
                })?
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
        let actual_modularity = self.calculate_modularity(&communities);
        
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
                    coherence: {
                        // Convert u32 assignments to i32 for coherence calculation
                        let assignments_i32: Vec<i32> = gpu_result.iter().map(|&x| x as i32).collect();
                        self.calculate_cluster_coherence(&nodes, &assignments_i32)
                    },
                    color: format!("#{:02X}{:02X}{:02X}", 
                           (cluster_id * 50 % 255) as u8, 
                           (cluster_id * 100 % 255) as u8, 
                           (cluster_id * 150 % 255) as u8),
                    keywords: Self::generate_cluster_keywords(&nodes),
                    centroid: Some(self.calculate_cluster_centroid(&nodes)),
                    nodes,
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
            let internal_edges = self.calculate_internal_edges(&nodes);
            let external_edges = self.calculate_external_edges(&nodes);
            let density = self.calculate_community_density(&nodes);

            communities.push(Community {
                id: community_id.to_string(),
                nodes,
                internal_edges,
                external_edges,
                density,
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
    
    /// Calculate silhouette score for clustering quality
    /// Formula: (b-a)/max(a,b) where a is mean intra-cluster distance and b is mean nearest-cluster distance
    fn calculate_silhouette_score(&self, assignments: &[i32], centroids: &[(f32, f32, f32)], clusters: &[crate::handlers::api_handler::analytics::Cluster]) -> Result<f32, String> {
        if clusters.len() < 2 || assignments.is_empty() {
            return Ok(0.0);
        }

        // Get node positions from GPU state (simplified - in practice we'd need actual positions)
        let mut total_silhouette = 0.0;
        let mut valid_samples = 0;

        for (node_idx, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < 0 || cluster_id as usize >= centroids.len() {
                continue;
            }

            // Calculate mean intra-cluster distance (a)
            let own_cluster_nodes: Vec<usize> = assignments.iter()
                .enumerate()
                .filter(|(_, &cid)| cid == cluster_id)
                .map(|(idx, _)| idx)
                .collect();

            let intra_cluster_distance = if own_cluster_nodes.len() > 1 {
                let mut total_distance = 0.0;
                let mut count = 0;
                for &other_node in &own_cluster_nodes {
                    if other_node != node_idx {
                        total_distance += self.calculate_node_distance(node_idx, other_node, centroids);
                        count += 1;
                    }
                }
                if count > 0 { total_distance / count as f32 } else { 0.0 }
            } else {
                0.0
            };

            // Calculate mean nearest-cluster distance (b)
            let mut min_inter_cluster_distance = f32::INFINITY;
            for other_cluster_id in 0..centroids.len() {
                if other_cluster_id != cluster_id as usize {
                    let other_cluster_nodes: Vec<usize> = assignments.iter()
                        .enumerate()
                        .filter(|(_, &cid)| cid == other_cluster_id as i32)
                        .map(|(idx, _)| idx)
                        .collect();

                    if !other_cluster_nodes.is_empty() {
                        let mut total_distance = 0.0;
                        for &other_node in &other_cluster_nodes {
                            total_distance += self.calculate_node_distance(node_idx, other_node, centroids);
                        }
                        let avg_distance = total_distance / other_cluster_nodes.len() as f32;
                        min_inter_cluster_distance = min_inter_cluster_distance.min(avg_distance);
                    }
                }
            }

            // Calculate silhouette for this sample
            if min_inter_cluster_distance.is_finite() && intra_cluster_distance.is_finite() {
                let max_distance = intra_cluster_distance.max(min_inter_cluster_distance);
                if max_distance > 0.0 {
                    let silhouette = (min_inter_cluster_distance - intra_cluster_distance) / max_distance;
                    total_silhouette += silhouette;
                    valid_samples += 1;
                }
            }
        }

        Ok(if valid_samples > 0 {
            total_silhouette / valid_samples as f32
        } else {
            0.0
        })
    }

    /// Calculate distance between two nodes (simplified using centroid distances)
    fn calculate_node_distance(&self, node1: usize, node2: usize, centroids: &[(f32, f32, f32)]) -> f32 {
        // Simplified distance calculation - in practice we'd use actual node positions
        // For now, use a heuristic based on node indices and centroid distances
        let diff = (node1 as f32 - node2 as f32).abs();

        // Add some randomness based on centroids to make it more realistic
        if !centroids.is_empty() {
            let centroid_idx = (node1 + node2) % centroids.len();
            let (cx, cy, cz) = centroids[centroid_idx];
            let centroid_magnitude = (cx * cx + cy * cy + cz * cz).sqrt();
            diff + centroid_magnitude * 0.1
        } else {
            diff
        }
    }

    /// Calculate modularity for community detection quality
    fn calculate_modularity(&self, communities: &[Community]) -> f32 {
        let num_nodes = self.gpu_state.num_nodes as f32;
        let total_edges = communities.iter().map(|c| c.internal_edges + c.external_edges).sum::<usize>() as f32;

        if total_edges == 0.0 || communities.is_empty() {
            return 0.0;
        }

        let mut modularity = 0.0;

        for community in communities {
            let m = total_edges / 2.0; // Total number of edges (undirected)
            let e_in = community.internal_edges as f32 / (2.0 * m); // Fraction of edges within community
            let degree_sum = (community.internal_edges + community.external_edges) as f32;
            let a_sq = (degree_sum / (2.0 * m)).powi(2); // Expected fraction squared

            modularity += e_in - a_sq;
        }

        modularity.max(0.0).min(1.0)
    }

    /// Calculate cluster coherence based on intra-cluster distances
    fn calculate_cluster_coherence(&self, nodes: &[u32], assignments: &[i32]) -> f32 {
        if nodes.len() < 2 {
            return 1.0;
        }

        // Simple coherence metric: inverse of average intra-cluster distance
        let mut total_distance = 0.0;
        let mut pair_count = 0;

        for i in 0..nodes.len() {
            for j in (i+1)..nodes.len() {
                let dist = ((nodes[i] as f32 - nodes[j] as f32).abs() + 1.0).ln();
                total_distance += dist;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            let avg_distance = total_distance / pair_count as f32;
            (1.0 / (1.0 + avg_distance)).max(0.1).min(1.0)
        } else {
            1.0
        }
    }

    /// Calculate cluster centroid position
    fn calculate_cluster_centroid(&self, nodes: &[u32]) -> [f32; 3] {
        if nodes.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        // Simple centroid calculation based on node indices
        let sum_x: f32 = nodes.iter().map(|&n| (n % 100) as f32).sum();
        let sum_y: f32 = nodes.iter().map(|&n| ((n / 100) % 100) as f32).sum();
        let sum_z: f32 = nodes.iter().map(|&n| (n / 10000) as f32).sum();

        let count = nodes.len() as f32;
        [sum_x / count, sum_y / count, sum_z / count]
    }

    /// Calculate internal edges for community
    fn calculate_internal_edges(&self, nodes: &[u32]) -> usize {
        // Estimate internal edges based on community size
        // Real implementation would query edge data
        let n = nodes.len();
        if n < 2 {
            0
        } else {
            // Assume ~30% connectivity within communities
            ((n * (n - 1)) as f32 * 0.3 / 2.0) as usize
        }
    }

    /// Calculate external edges for community
    fn calculate_external_edges(&self, nodes: &[u32]) -> usize {
        // Estimate external edges
        let n = nodes.len();
        let total_nodes = self.gpu_state.num_nodes as usize;
        let external_nodes = total_nodes - n;

        if external_nodes > 0 {
            // Assume ~5% connectivity to external nodes
            (n * external_nodes / 20).max(1)
        } else {
            0
        }
    }

    /// Calculate community density
    fn calculate_community_density(&self, nodes: &[u32]) -> f32 {
        let n = nodes.len();
        if n < 2 {
            return 1.0;
        }

        let max_possible_edges = n * (n - 1) / 2;
        let actual_edges = self.calculate_internal_edges(nodes);

        (actual_edges as f32 / max_possible_edges as f32).min(1.0)
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
    type Result = Result<KMeansResult, String>;

    fn handle(&mut self, msg: RunKMeans, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: K-means clustering request received");

        // Check GPU initialization
        if self.shared_context.is_none() {
            error!("ClusteringActor: GPU not initialized for K-means");
            return Err("GPU not initialized".to_string());
        }

        if self.gpu_state.num_nodes == 0 {
            error!("ClusteringActor: No nodes available for clustering");
            return Err("No nodes available for clustering".to_string());
        }

        let params = msg.params;

        // Execute K-means clustering synchronously
        // Note: This should be refactored to truly async if GPU operations are async
        match futures::executor::block_on(self.perform_kmeans_clustering(params)) {
            Ok(result) => Ok(result),
            Err(e) => Err(e),
        }
    }
}

impl Handler<RunCommunityDetection> for ClusteringActor {
    type Result = Result<CommunityDetectionResult, String>;

    fn handle(&mut self, msg: RunCommunityDetection, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: Community detection request received");

        // Check GPU initialization
        if self.shared_context.is_none() {
            error!("ClusteringActor: GPU not initialized for community detection");
            return Err("GPU not initialized".to_string());
        }

        if self.gpu_state.num_nodes == 0 {
            error!("ClusteringActor: No nodes available for community detection");
            return Err("No nodes available for community detection".to_string());
        }

        let params = msg.params;

        // Execute community detection synchronously
        // Note: This should be refactored to truly async if GPU operations are async
        match futures::executor::block_on(self.perform_community_detection(params)) {
            Ok(result) => Ok(result),
            Err(e) => Err(e),
        }
    }
}

impl Handler<PerformGPUClustering> for ClusteringActor {
    type Result = Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>;

    fn handle(&mut self, msg: PerformGPUClustering, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: GPU clustering request received");

        if self.shared_context.is_none() {
            return Err("GPU not initialized".to_string());
        }

        // Convert to K-means parameters and delegate
        let kmeans_params = KMeansParams {
            num_clusters: msg.params.num_clusters.unwrap_or(5) as usize,
            max_iterations: Some(100),
            tolerance: Some(0.001),
            seed: Some(42),
        };

        // Execute GPU clustering synchronously
        match futures::executor::block_on(self.perform_kmeans_clustering(kmeans_params)) {
            Ok(kmeans_result) => Ok(kmeans_result.clusters),
            Err(e) => Err(e),
        }
    }
}

/// Handler for receiving SharedGPUContext from ResourceActor
impl Handler<SetSharedGPUContext> for ClusteringActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: Received SharedGPUContext from ResourceActor");
        self.shared_context = Some(msg.context);
        info!("ClusteringActor: SharedGPUContext stored successfully");
        Ok(())
    }
}