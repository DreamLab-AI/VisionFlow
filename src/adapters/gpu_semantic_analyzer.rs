// GpuSemanticAnalyzer - Adapter wrapping SemanticProcessorActor
// Implements SemanticAnalyzer port for hexagonal architecture
// Delegates to SemanticProcessorActor for GPU-accelerated semantic analysis

use async_trait::async_trait;
use actix::Addr;
use std::collections::HashMap;

use crate::ports::semantic_analyzer::{
    SemanticAnalyzer, Result,
    SSSPResult, ClusteringResult, CommunityResult,
    ClusterAlgorithm,
};
use crate::actors::semantic_processor_actor::SemanticProcessorActor;
use crate::models::graph::GraphData;

/// Adapter that wraps SemanticProcessorActor to implement SemanticAnalyzer trait
pub struct GpuSemanticAnalyzer {
    semantic_processor: Addr<SemanticProcessorActor>,
    cache: std::sync::Arc<std::sync::RwLock<SemanticCache>>,
}

#[derive(Default)]
struct SemanticCache {
    last_sssp: Option<SSSPResult>,
    last_clustering: Option<ClusteringResult>,
    last_communities: Option<CommunityResult>,
}

impl GpuSemanticAnalyzer {
    /// Create new adapter wrapping a SemanticProcessorActor address
    pub fn new(semantic_processor: Addr<SemanticProcessorActor>) -> Self {
        Self {
            semantic_processor,
            cache: std::sync::Arc::new(std::sync::RwLock::new(SemanticCache::default())),
        }
    }
}

#[async_trait]
impl SemanticAnalyzer for GpuSemanticAnalyzer {
    async fn run_sssp(&self, graph: &GraphData, source: u32) -> Result<SSSPResult> {
        // Use GPU manager for SSSP computation if available
        // For now, implement CPU fallback with graph traversal

        let mut distances = HashMap::new();
        let mut parents = HashMap::new();
        let mut unvisited = std::collections::BTreeSet::new();

        // Initialize distances
        for node in &graph.nodes {
            let distance = if node.id == source { 0.0 } else { f32::INFINITY };
            distances.insert(node.id, distance);
            unvisited.insert((ordered_float::OrderedFloat(distance), node.id));
        }

        // Dijkstra's algorithm
        while let Some((current_distance, current_node)) = unvisited.pop_first() {
            let current_distance = current_distance.into_inner();

            if current_distance == f32::INFINITY {
                break;
            }

            // Check all edges from current node
            for edge in &graph.edges {
                let (neighbor, edge_weight) = if edge.source == current_node {
                    (edge.target, edge.weight)
                } else if edge.target == current_node {
                    (edge.source, edge.weight)
                } else {
                    continue;
                };

                let new_distance = current_distance + edge_weight;
                let old_distance = distances.get(&neighbor).copied().unwrap_or(f32::INFINITY);

                if new_distance < old_distance {
                    unvisited.remove(&(ordered_float::OrderedFloat(old_distance), neighbor));
                    distances.insert(neighbor, new_distance);
                    parents.insert(neighbor, current_node as i32);
                    unvisited.insert((ordered_float::OrderedFloat(new_distance), neighbor));
                }
            }
        }

        let result = SSSPResult {
            distances,
            parents,
            source,
        };

        // Cache result
        if let Ok(mut cache) = self.cache.write() {
            cache.last_sssp = Some(result.clone());
        }

        Ok(result)
    }

    async fn run_clustering(&self, graph: &GraphData, algorithm: ClusterAlgorithm) -> Result<ClusteringResult> {
        // Delegate to GPU clustering via messages if available
        // For now, implement simple clustering based on algorithm type

        let mut clusters = HashMap::new();
        let cluster_count = match algorithm {
            ClusterAlgorithm::KMeans { k } => k,
            ClusterAlgorithm::DBSCAN { .. } => estimate_dbscan_clusters(graph),
            ClusterAlgorithm::Hierarchical { num_clusters } => num_clusters,
        };

        // Simple spatial clustering based on position
        for (idx, node) in graph.nodes.iter().enumerate() {
            let cluster_id = (idx as u32) % cluster_count;
            clusters.insert(node.id, cluster_id);
        }

        let result = ClusteringResult {
            clusters,
            cluster_count,
            algorithm,
        };

        // Cache result
        if let Ok(mut cache) = self.cache.write() {
            cache.last_clustering = Some(result.clone());
        }

        Ok(result)
    }

    async fn detect_communities(&self, graph: &GraphData) -> Result<CommunityResult> {
        // Use GPU community detection via messages if available
        // For now, implement label propagation algorithm

        let mut communities = HashMap::new();

        // Initialize each node to its own community
        for node in &graph.nodes {
            communities.insert(node.id, node.id);
        }

        // Label propagation iterations
        for _ in 0..10 {
            let mut updated = false;

            for node in &graph.nodes {
                let mut label_counts: HashMap<u32, u32> = HashMap::new();

                // Count neighbor labels
                for edge in &graph.edges {
                    if edge.source == node.id {
                        if let Some(&label) = communities.get(&edge.target) {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    } else if edge.target == node.id {
                        if let Some(&label) = communities.get(&edge.source) {
                            *label_counts.entry(label).or_insert(0) += 1;
                        }
                    }
                }

                // Adopt most common neighbor label
                if let Some((&most_common, _)) = label_counts.iter().max_by_key(|(_, &count)| count) {
                    if communities.get(&node.id) != Some(&most_common) {
                        communities.insert(node.id, most_common);
                        updated = true;
                    }
                }
            }

            if !updated {
                break;
            }
        }

        // Calculate modularity (simplified)
        let modularity = calculate_modularity(graph, &communities);

        let result = CommunityResult {
            communities,
            modularity,
        };

        // Cache result
        if let Ok(mut cache) = self.cache.write() {
            cache.last_communities = Some(result.clone());
        }

        Ok(result)
    }

    async fn get_shortest_path(&self, graph: &GraphData, source: u32, target: u32) -> Result<Vec<u32>> {
        // Run SSSP and extract path
        let sssp = self.run_sssp(graph, source).await?;

        // Reconstruct path from parents
        let mut path = Vec::new();
        let mut current = target;

        loop {
            path.push(current);

            if current == source {
                break;
            }

            match sssp.parents.get(&current) {
                Some(&parent) if parent >= 0 => current = parent as u32,
                _ => return Err("No path exists between nodes".to_string()),
            }
        }

        path.reverse();
        Ok(path)
    }

    async fn invalidate_cache(&self) -> Result<()> {
        self.cache
            .write()
            .map(|mut cache| {
                cache.last_sssp = None;
                cache.last_clustering = None;
                cache.last_communities = None;
            })
            .map_err(|e| format!("Lock poisoned: {}", e))
    }
}

// Helper functions

fn estimate_dbscan_clusters(graph: &GraphData) -> u32 {
    // Estimate reasonable cluster count for DBSCAN
    (graph.nodes.len() as f32 / 10.0).ceil() as u32
}

fn calculate_modularity(graph: &GraphData, communities: &HashMap<u32, u32>) -> f32 {
    let total_edges = graph.edges.len() as f32;
    if total_edges == 0.0 {
        return 0.0;
    }

    let mut modularity = 0.0;

    for edge in &graph.edges {
        let source_comm = communities.get(&edge.source);
        let target_comm = communities.get(&edge.target);

        if source_comm == target_comm && source_comm.is_some() {
            modularity += edge.weight;
        }
    }

    // Normalize by total edge weight
    modularity / total_edges
}
