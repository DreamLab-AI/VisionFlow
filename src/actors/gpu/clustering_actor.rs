//! Clustering Actor - Handles K-means clustering and community detection algorithms

use actix::prelude::*;
use log::{error, info, warn};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use super::shared::{GPUState, SharedGPUContext};
use crate::actors::messages::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringStats {
    pub total_clusters: usize,
    pub average_cluster_size: f32,
    pub largest_cluster_size: usize,
    pub smallest_cluster_size: usize,
    pub silhouette_score: f32,
    pub computation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityDetectionStats {
    pub total_communities: usize,
    pub modularity: f32,
    pub average_community_size: f32,
    pub largest_community: usize,
    pub smallest_community: usize,
    pub computation_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Community {
    pub id: String,
    pub nodes: Vec<u32>,
    pub internal_edges: usize,
    pub external_edges: usize,
    pub density: f32,
}

pub struct ClusteringActor {
    
    gpu_state: GPUState,

    
    shared_context: Option<Arc<SharedGPUContext>>,
}

impl ClusteringActor {
    pub fn new() -> Self {
        Self {
            gpu_state: GPUState::default(),
            shared_context: None,
        }
    }

    
    fn generate_cluster_keywords(nodes: &[u32]) -> Vec<String> {
        if nodes.is_empty() {
            return vec!["empty".to_string()];
        }

        
        let mut keywords = Vec::new();
        match nodes.len() {
            1 => keywords.push("singleton".to_string()),
            2..=5 => keywords.push("small".to_string()),
            6..=20 => keywords.push("medium".to_string()),
            _ => keywords.push("large".to_string()),
        }

        
        keywords.push(format!("cluster-{}", nodes[0] % 10));
        keywords
    }

    
    async fn perform_kmeans_clustering(
        &mut self,
        params: KMeansParams,
    ) -> Result<KMeansResult, String> {
        info!(
            "ClusteringActor: Starting K-means clustering with {} clusters",
            params.num_clusters
        );

        let unified_compute_arc = match &self.shared_context {
            Some(ctx) => Arc::clone(&ctx.unified_compute),
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };

        let num_clusters = params.num_clusters;
        let max_iterations = params.max_iterations.unwrap_or(100);
        let tolerance = params.tolerance.unwrap_or(0.001);
        let seed = params.seed.unwrap_or(42);

        // Move blocking GPU operations to dedicated blocking thread pool
        // This prevents std::sync::Mutex::lock() from blocking Tokio worker threads
        let blocking_result = tokio::task::spawn_blocking(move || -> Result<_, String> {
            let mut unified_compute = match unified_compute_arc.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    warn!("ClusteringActor: GPU mutex was poisoned, recovering");
                    poisoned.into_inner()
                }
            };

            let start_time = Instant::now();

            let gpu_result = unified_compute
                .run_kmeans_clustering_with_metrics(
                    num_clusters,
                    max_iterations,
                    tolerance,
                    seed,
                )
                .map_err(|e| {
                    error!("GPU K-means clustering failed: {}", e);
                    format!("K-means clustering failed: {}", e)
                })?;

            let computation_time = start_time.elapsed();
            info!(
                "ClusteringActor: K-means clustering completed in {:?}",
                computation_time
            );

            Ok((gpu_result, computation_time))
        }).await;

        let (gpu_result, computation_time) = match blocking_result {
            Ok(inner_result) => inner_result?,
            Err(join_err) => return Err(format!("GPU blocking task panicked: {}", join_err)),
        };

        
        let (assignments, centroids, inertia, actual_iterations, converged) = gpu_result;
        let clusters = self.convert_gpu_kmeans_result_to_clusters(
            assignments.iter().map(|&x| x as u32).collect(),
            params.num_clusters as u32,
        )?;

        
        let cluster_sizes: Vec<usize> = clusters.iter().map(|c| c.nodes.len()).collect();
        let avg_cluster_size = if !cluster_sizes.is_empty() {
            cluster_sizes.iter().sum::<usize>() as f32 / cluster_sizes.len() as f32
        } else {
            0.0
        };

        
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

    
    async fn perform_community_detection(
        &mut self,
        params: CommunityDetectionParams,
    ) -> Result<CommunityDetectionResult, String> {
        info!(
            "ClusteringActor: Starting {:?} community detection",
            params.algorithm
        );

        let unified_compute_arc = match &self.shared_context {
            Some(ctx) => Arc::clone(&ctx.unified_compute),
            None => {
                return Err("GPU context not initialized".to_string());
            }
        };

        let algorithm = params.algorithm.clone();
        let max_iterations = params.max_iterations.unwrap_or(100);
        let seed = params.seed.unwrap_or(42);

        // Move blocking GPU operations to dedicated blocking thread pool
        // This prevents std::sync::Mutex::lock() from blocking Tokio worker threads
        let blocking_result = tokio::task::spawn_blocking(move || -> Result<_, String> {
            let mut unified_compute = match unified_compute_arc.lock() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    warn!("ClusteringActor: GPU mutex was poisoned, recovering");
                    poisoned.into_inner()
                }
            };

            let start_time = Instant::now();

            let gpu_result = match algorithm {
                CommunityDetectionAlgorithm::LabelPropagation => unified_compute
                    .run_community_detection_label_propagation(
                        max_iterations,
                        seed,
                    )
                    .map_err(|e| {
                        error!("GPU label propagation failed: {}", e);
                        format!("Label propagation failed: {}", e)
                    })?,
                CommunityDetectionAlgorithm::Louvain => {
                    unified_compute
                        .run_louvain_community_detection(
                            max_iterations,
                            1.0,
                            seed,
                        )
                        .map_err(|e| {
                            error!("GPU Louvain community detection failed: {}", e);
                            format!("Louvain community detection failed: {}", e)
                        })?
                }
            };

            let computation_time = start_time.elapsed();
            info!(
                "ClusteringActor: Community detection completed in {:?}",
                computation_time
            );

            Ok((gpu_result, computation_time))
        }).await;

        let (gpu_result, computation_time) = match blocking_result {
            Ok(inner_result) => inner_result?,
            Err(join_err) => return Err(format!("GPU blocking task panicked: {}", join_err)),
        };

        
        let (node_labels, num_communities, modularity, iterations, community_sizes, converged) =
            gpu_result;
        let communities = self.convert_gpu_community_result_to_communities(
            node_labels.iter().map(|&x| x as u32).collect(),
        )?;

        
        let actual_community_sizes: Vec<usize> =
            communities.iter().map(|c| c.nodes.len()).collect();
        let actual_modularity = self.calculate_modularity(&communities);

        let stats = CommunityDetectionStats {
            total_communities: communities.len(),
            modularity: actual_modularity,
            average_community_size: if !actual_community_sizes.is_empty() {
                actual_community_sizes.iter().sum::<usize>() as f32
                    / actual_community_sizes.len() as f32
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

    
    fn convert_gpu_kmeans_result_to_clusters(
        &self,
        gpu_result: Vec<u32>,
        num_clusters: u32,
    ) -> Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String> {
        if gpu_result.len() != self.gpu_state.num_nodes as usize {
            return Err(format!(
                "GPU result size mismatch: expected {}, got {}",
                self.gpu_state.num_nodes,
                gpu_result.len()
            ));
        }

        
        let mut cluster_nodes: Vec<Vec<u32>> = vec![Vec::new(); num_clusters as usize];

        for (node_idx, &cluster_id) in gpu_result.iter().enumerate() {
            if (cluster_id as usize) < cluster_nodes.len() {
                cluster_nodes[cluster_id as usize].push(node_idx as u32);
            }
        }

        
        let mut clusters = Vec::new();
        for (cluster_id, nodes) in cluster_nodes.into_iter().enumerate() {
            if !nodes.is_empty() {
                clusters.push(crate::handlers::api_handler::analytics::Cluster {
                    id: Uuid::new_v4().to_string(),
                    label: format!("Cluster {}", cluster_id),
                    node_count: nodes.len() as u32,
                    coherence: {
                        
                        let assignments_i32: Vec<i32> =
                            gpu_result.iter().map(|&x| x as i32).collect();
                        self.calculate_cluster_coherence(&nodes, &assignments_i32)
                    },
                    color: format!(
                        "#{:02X}{:02X}{:02X}",
                        (cluster_id * 50 % 255) as u8,
                        (cluster_id * 100 % 255) as u8,
                        (cluster_id * 150 % 255) as u8
                    ),
                    keywords: Self::generate_cluster_keywords(&nodes),
                    centroid: Some(self.calculate_cluster_centroid(&nodes)),
                    nodes,
                });
            }
        }

        info!(
            "ClusteringActor: Generated {} non-empty clusters",
            clusters.len()
        );
        Ok(clusters)
    }

    
    fn convert_gpu_community_result_to_communities(
        &self,
        gpu_result: Vec<u32>,
    ) -> Result<Vec<Community>, String> {
        if gpu_result.len() != self.gpu_state.num_nodes as usize {
            return Err(format!(
                "GPU result size mismatch: expected {}, got {}",
                self.gpu_state.num_nodes,
                gpu_result.len()
            ));
        }

        
        let mut community_nodes: std::collections::HashMap<u32, Vec<u32>> =
            std::collections::HashMap::new();

        for (node_idx, &community_id) in gpu_result.iter().enumerate() {
            community_nodes
                .entry(community_id)
                .or_insert_with(Vec::new)
                .push(node_idx as u32);
        }

        
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

        info!(
            "ClusteringActor: Generated {} communities",
            communities.len()
        );
        Ok(communities)
    }

    
    #[allow(dead_code)]
    fn generate_cluster_color(cluster_id: usize) -> [f32; 3] {
        let mut rng = rand::thread_rng();

        
        let hue = (cluster_id as f32 * 137.5) % 360.0; 
        let saturation = 0.7 + (rng.gen::<f32>() * 0.3); 
        let value = 0.8 + (rng.gen::<f32>() * 0.2); 

        
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

    
    
    fn calculate_silhouette_score(
        &self,
        assignments: &[i32],
        centroids: &[(f32, f32, f32)],
        clusters: &[crate::handlers::api_handler::analytics::Cluster],
    ) -> Result<f32, String> {
        if clusters.len() < 2 || assignments.is_empty() {
            return Ok(0.0);
        }

        
        let mut total_silhouette = 0.0;
        let mut valid_samples = 0;

        for (node_idx, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < 0 || cluster_id as usize >= centroids.len() {
                continue;
            }

            
            let own_cluster_nodes: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|(_, &cid)| cid == cluster_id)
                .map(|(idx, _)| idx)
                .collect();

            let intra_cluster_distance = if own_cluster_nodes.len() > 1 {
                let mut total_distance = 0.0;
                let mut count = 0;
                for &other_node in &own_cluster_nodes {
                    if other_node != node_idx {
                        total_distance +=
                            self.calculate_node_distance(node_idx, other_node, centroids);
                        count += 1;
                    }
                }
                if count > 0 {
                    total_distance / count as f32
                } else {
                    0.0
                }
            } else {
                0.0
            };

            
            let mut min_inter_cluster_distance = f32::INFINITY;
            for other_cluster_id in 0..centroids.len() {
                if other_cluster_id != cluster_id as usize {
                    let other_cluster_nodes: Vec<usize> = assignments
                        .iter()
                        .enumerate()
                        .filter(|(_, &cid)| cid == other_cluster_id as i32)
                        .map(|(idx, _)| idx)
                        .collect();

                    if !other_cluster_nodes.is_empty() {
                        let mut total_distance = 0.0;
                        for &other_node in &other_cluster_nodes {
                            total_distance +=
                                self.calculate_node_distance(node_idx, other_node, centroids);
                        }
                        let avg_distance = total_distance / other_cluster_nodes.len() as f32;
                        min_inter_cluster_distance = min_inter_cluster_distance.min(avg_distance);
                    }
                }
            }

            
            if min_inter_cluster_distance.is_finite() && intra_cluster_distance.is_finite() {
                let max_distance = intra_cluster_distance.max(min_inter_cluster_distance);
                if max_distance > 0.0 {
                    let silhouette =
                        (min_inter_cluster_distance - intra_cluster_distance) / max_distance;
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

    /// Compute Euclidean distance between two nodes using actual GPU positions.
    /// Falls back to centroid Euclidean distance if positions are unavailable.
    fn calculate_node_distance(
        &self,
        node1: usize,
        node2: usize,
        centroids: &[(f32, f32, f32)],
    ) -> f32 {
        // Try to use actual spatial positions from GPU
        if let Some(ref ctx) = self.shared_context {
            if let Ok(uc) = ctx.unified_compute.lock() {
                let n = uc.num_nodes;
                if node1 < n && node2 < n {
                    let mut px = vec![0.0f32; n];
                    let mut py = vec![0.0f32; n];
                    let mut pz = vec![0.0f32; n];
                    if uc.download_positions(&mut px, &mut py, &mut pz).is_ok() {
                        let dx = px[node1] - px[node2];
                        let dy = py[node1] - py[node2];
                        let dz = pz[node1] - pz[node2];
                        return (dx * dx + dy * dy + dz * dz).sqrt();
                    }
                }
            }
        }

        // Fallback: Euclidean distance between cluster centroids
        if centroids.len() >= 2 {
            let c1 = node1 % centroids.len();
            let c2 = node2 % centroids.len();
            let (x1, y1, z1) = centroids[c1];
            let (x2, y2, z2) = centroids[c2];
            let dx = x1 - x2;
            let dy = y1 - y2;
            let dz = z1 - z2;
            (dx * dx + dy * dy + dz * dz).sqrt()
        } else {
            1.0
        }
    }

    
    fn calculate_modularity(&self, communities: &[Community]) -> f32 {
        let _num_nodes = self.gpu_state.num_nodes as f32;
        let total_edges = communities
            .iter()
            .map(|c| c.internal_edges + c.external_edges)
            .sum::<usize>() as f32;

        if total_edges == 0.0 || communities.is_empty() {
            return 0.0;
        }

        let mut modularity = 0.0;

        for community in communities {
            let m = total_edges / 2.0; 
            let e_in = community.internal_edges as f32 / (2.0 * m); 
            let degree_sum = (community.internal_edges + community.external_edges) as f32;
            let a_sq = (degree_sum / (2.0 * m)).powi(2); 

            modularity += e_in - a_sq;
        }

        modularity.max(0.0).min(1.0)
    }

    /// Compute cluster coherence using actual Euclidean distances between
    /// node positions from the GPU layout. High coherence means nodes in
    /// this cluster are spatially close together.
    fn calculate_cluster_coherence(&self, nodes: &[u32], _assignments: &[i32]) -> f32 {
        if nodes.len() < 2 {
            return 1.0;
        }

        // Try to load actual positions from GPU for Euclidean distance
        let positions: Option<(Vec<f32>, Vec<f32>, Vec<f32>)> =
            self.shared_context.as_ref().and_then(|ctx| {
                ctx.unified_compute.lock().ok().and_then(|uc| {
                    let n = uc.num_nodes;
                    let mut px = vec![0.0f32; n];
                    let mut py = vec![0.0f32; n];
                    let mut pz = vec![0.0f32; n];
                    uc.download_positions(&mut px, &mut py, &mut pz)
                        .ok()
                        .map(|_| (px, py, pz))
                })
            });

        let mut total_distance = 0.0f32;
        let mut pair_count = 0u64;

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let dist = if let Some((ref px, ref py, ref pz)) = positions {
                    let ni = nodes[i] as usize;
                    let nj = nodes[j] as usize;
                    if ni < px.len() && nj < px.len() {
                        let dx = px[ni] - px[nj];
                        let dy = py[ni] - py[nj];
                        let dz = pz[ni] - pz[nj];
                        (dx * dx + dy * dy + dz * dz).sqrt()
                    } else {
                        1.0
                    }
                } else {
                    // No GPU positions available; use constant to avoid fake ordering
                    1.0
                };
                total_distance += dist;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            let avg_distance = total_distance / pair_count as f32;
            // Inverse relationship: smaller avg distance = higher coherence
            (1.0 / (1.0 + avg_distance)).max(0.1).min(1.0)
        } else {
            1.0
        }
    }

    /// Compute cluster centroid from actual GPU node positions.
    /// Falls back to origin if positions are unavailable.
    fn calculate_cluster_centroid(&self, nodes: &[u32]) -> [f32; 3] {
        if nodes.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        // Try actual positions from GPU
        if let Some(ref ctx) = self.shared_context {
            if let Ok(uc) = ctx.unified_compute.lock() {
                let n = uc.num_nodes;
                let mut px = vec![0.0f32; n];
                let mut py = vec![0.0f32; n];
                let mut pz = vec![0.0f32; n];
                if uc.download_positions(&mut px, &mut py, &mut pz).is_ok() {
                    let mut sx = 0.0f32;
                    let mut sy = 0.0f32;
                    let mut sz = 0.0f32;
                    let mut count = 0u32;
                    for &nid in nodes {
                        let idx = nid as usize;
                        if idx < n {
                            sx += px[idx];
                            sy += py[idx];
                            sz += pz[idx];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        let c = count as f32;
                        return [sx / c, sy / c, sz / c];
                    }
                }
            }
        }

        // Fallback: return origin
        [0.0, 0.0, 0.0]
    }

    
    fn calculate_internal_edges(&self, nodes: &[u32]) -> usize {
        
        
        let n = nodes.len();
        if n < 2 {
            0
        } else {
            
            ((n * (n - 1)) as f32 * 0.3 / 2.0) as usize
        }
    }

    
    fn calculate_external_edges(&self, nodes: &[u32]) -> usize {
        
        let n = nodes.len();
        let total_nodes = self.gpu_state.num_nodes as usize;
        let external_nodes = total_nodes - n;

        if external_nodes > 0 {
            
            (n * external_nodes / 20).max(1)
        } else {
            0
        }
    }

    
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

impl Handler<SetSharedGPUContext> for ClusteringActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: Received SharedGPUContext from ResourceActor");
        self.shared_context = Some(msg.context);
        
        info!("ClusteringActor: SharedGPUContext stored successfully");
        Ok(())
    }
}

impl Handler<RunKMeans> for ClusteringActor {
    type Result = actix::ResponseFuture<Result<KMeansResult, String>>;

    fn handle(&mut self, msg: RunKMeans, _ctx: &mut Self::Context) -> Self::Result {
        info!(
            "ClusteringActor: Received RunKMeans request with {} clusters",
            msg.params.num_clusters
        );

        
        let mut actor_clone = Self {
            gpu_state: self.gpu_state.clone(),
            shared_context: self.shared_context.clone(),
        };

        Box::pin(async move { actor_clone.perform_kmeans_clustering(msg.params).await })
    }
}

impl Handler<RunCommunityDetection> for ClusteringActor {
    type Result = actix::ResponseFuture<Result<CommunityDetectionResult, String>>;

    fn handle(&mut self, msg: RunCommunityDetection, _ctx: &mut Self::Context) -> Self::Result {
        info!("ClusteringActor: Received RunCommunityDetection request");

        
        let mut actor_clone = Self {
            gpu_state: self.gpu_state.clone(),
            shared_context: self.shared_context.clone(),
        };

        Box::pin(async move { actor_clone.perform_community_detection(msg.params).await })
    }
}

impl Handler<PerformGPUClustering> for ClusteringActor {
    type Result = actix::ResponseFuture<
        Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>,
    >;

    fn handle(&mut self, msg: PerformGPUClustering, _ctx: &mut Self::Context) -> Self::Result {
        info!(
            "ClusteringActor: Received PerformGPUClustering request with method: {}",
            msg.method
        );

        
        let mut actor_clone = Self {
            gpu_state: self.gpu_state.clone(),
            shared_context: self.shared_context.clone(),
        };

        Box::pin(async move {
            
            let params = KMeansParams {
                num_clusters: msg.params.num_clusters.unwrap_or(5) as usize,
                max_iterations: msg.params.max_iterations,
                tolerance: msg.params.convergence_threshold,
                seed: None,
            };

            
            let result = actor_clone.perform_kmeans_clustering(params).await?;

            
            Ok(result.clusters)
        })
    }
}
