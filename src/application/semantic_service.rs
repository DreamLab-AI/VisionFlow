// src/application/semantic_service.rs
//! Semantic Analysis Service
//!
//! Application service that integrates actor-based semantic analysis
//! through hexagonal architecture ports. Handles GPU-accelerated
//! graph algorithms and community detection.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::events::event_bus::EventBus;
use crate::models::constraints::ConstraintSet;
use crate::models::graph::GraphData;
use crate::ports::gpu_semantic_analyzer::{
    ClusteringAlgorithm, CommunityDetectionResult, GpuSemanticAnalyzer, ImportanceAlgorithm,
    OptimizationResult, PathfindingResult, Result as SemanticResult, SemanticConstraintConfig,
    SemanticStatistics,
};

/// Community detection request
#[derive(Debug, Clone)]
pub struct CommunityDetectionRequest {
    pub algorithm: ClusteringAlgorithm,
    pub min_cluster_size: Option<usize>,
}

/// Centrality analysis request
#[derive(Debug, Clone)]
pub struct CentralityRequest {
    pub algorithm: ImportanceAlgorithm,
    pub top_k: Option<usize>,
}

/// Shortest path request
#[derive(Debug, Clone)]
pub struct ShortestPathRequest {
    pub source_node_id: u32,
    pub target_node_id: Option<u32>,
    pub include_path: bool,
}

/// Semantic service for managing GPU-accelerated semantic analysis
pub struct SemanticService {
    semantic_adapter: Arc<RwLock<dyn GpuSemanticAnalyzer>>,
    event_bus: Arc<RwLock<EventBus>>,
}

impl SemanticService {
    /// Create new semantic service
    pub fn new(
        semantic_adapter: Arc<RwLock<dyn GpuSemanticAnalyzer>>,
        event_bus: Arc<RwLock<EventBus>>,
    ) -> Self {
        Self {
            semantic_adapter,
            event_bus,
        }
    }

    /// Initialize semantic analyzer with graph
    pub async fn initialize(&self, graph: Arc<GraphData>) -> SemanticResult<()> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.initialize(graph).await
    }

    /// Detect communities in the graph
    pub async fn detect_communities(
        &self,
        request: CommunityDetectionRequest,
    ) -> SemanticResult<CommunityDetectionResult> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.detect_communities(request.algorithm).await
    }

    /// Compute node centrality scores
    pub async fn compute_centrality(
        &self,
        request: CentralityRequest,
    ) -> SemanticResult<HashMap<u32, f32>> {
        let mut adapter = self.semantic_adapter.write().await;
        let scores = adapter.analyze_node_importance(request.algorithm).await?;

        // Filter top-k if requested
        if let Some(k) = request.top_k {
            let mut sorted: Vec<_> = scores.into_iter().collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            sorted.truncate(k);
            Ok(sorted.into_iter().collect())
        } else {
            Ok(scores)
        }
    }

    /// Compute shortest paths from source
    pub async fn compute_shortest_paths(
        &self,
        request: ShortestPathRequest,
    ) -> SemanticResult<PathfindingResult> {
        let mut adapter = self.semantic_adapter.write().await;

        if request.include_path {
            adapter.compute_shortest_paths(request.source_node_id).await
        } else {
            // Optimized version without path reconstruction
            let distances = adapter
                .compute_sssp_distances(request.source_node_id)
                .await?;
            Ok(PathfindingResult {
                source_node: request.source_node_id,
                distances: distances
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| (i as u32, d))
                    .collect(),
                paths: HashMap::new(),
                computation_time_ms: 0.0,
            })
        }
    }

    /// Compute all-pairs shortest paths
    pub async fn compute_all_pairs_shortest_paths(
        &self,
    ) -> SemanticResult<HashMap<(u32, u32), Vec<u32>>> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.compute_all_pairs_shortest_paths().await
    }

    /// Compute landmark-based approximate APSP
    pub async fn compute_landmark_apsp(
        &self,
        num_landmarks: usize,
    ) -> SemanticResult<Vec<Vec<f32>>> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.compute_landmark_apsp(num_landmarks).await
    }

    /// Generate semantic constraints for layout
    pub async fn generate_semantic_constraints(
        &self,
        config: SemanticConstraintConfig,
    ) -> SemanticResult<ConstraintSet> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.generate_semantic_constraints(config).await
    }

    /// Optimize layout with constraints
    pub async fn optimize_layout(
        &self,
        constraints: &ConstraintSet,
        max_iterations: usize,
    ) -> SemanticResult<OptimizationResult> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.optimize_layout(constraints, max_iterations).await
    }

    /// Analyze node importance with PageRank
    pub async fn compute_pagerank(
        &self,
        damping: f32,
        max_iterations: usize,
    ) -> SemanticResult<HashMap<u32, f32>> {
        let algorithm = ImportanceAlgorithm::PageRank {
            damping,
            max_iterations,
        };
        let mut adapter = self.semantic_adapter.write().await;
        adapter.analyze_node_importance(algorithm).await
    }

    /// Compute betweenness centrality
    pub async fn compute_betweenness_centrality(&self) -> SemanticResult<HashMap<u32, f32>> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter
            .analyze_node_importance(ImportanceAlgorithm::Betweenness)
            .await
    }

    /// Compute closeness centrality
    pub async fn compute_closeness_centrality(&self) -> SemanticResult<HashMap<u32, f32>> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter
            .analyze_node_importance(ImportanceAlgorithm::Closeness)
            .await
    }

    /// Update graph data
    pub async fn update_graph_data(&self, graph: Arc<GraphData>) -> SemanticResult<()> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.update_graph_data(graph).await
    }

    /// Invalidate pathfinding cache
    pub async fn invalidate_cache(&self) -> SemanticResult<()> {
        let mut adapter = self.semantic_adapter.write().await;
        adapter.invalidate_pathfinding_cache().await
    }

    /// Get semantic analysis statistics
    pub async fn get_statistics(&self) -> SemanticResult<SemanticStatistics> {
        let adapter = self.semantic_adapter.read().await;
        adapter.get_statistics().await
    }

    /// Detect communities with Louvain algorithm
    pub async fn detect_communities_louvain(&self) -> SemanticResult<CommunityDetectionResult> {
        self.detect_communities(CommunityDetectionRequest {
            algorithm: ClusteringAlgorithm::Louvain,
            min_cluster_size: None,
        })
        .await
    }

    /// Detect communities with label propagation
    pub async fn detect_communities_label_propagation(
        &self,
    ) -> SemanticResult<CommunityDetectionResult> {
        self.detect_communities(CommunityDetectionRequest {
            algorithm: ClusteringAlgorithm::LabelPropagation,
            min_cluster_size: None,
        })
        .await
    }

    /// Find connected components
    pub async fn find_connected_components(&self) -> SemanticResult<CommunityDetectionResult> {
        self.detect_communities(CommunityDetectionRequest {
            algorithm: ClusteringAlgorithm::ConnectedComponents,
            min_cluster_size: None,
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::event_bus::EventBus;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    struct MockSemanticAnalyzer;

    #[async_trait]
    impl GpuSemanticAnalyzer for MockSemanticAnalyzer {
        async fn initialize(&mut self, _graph: Arc<GraphData>) -> Result<()> {
            Ok(())
        }

        async fn detect_communities(
            &mut self,
            _algorithm: ClusteringAlgorithm,
        ) -> Result<CommunityDetectionResult> {
            Ok(CommunityDetectionResult {
                clusters: HashMap::new(),
                cluster_sizes: HashMap::new(),
                modularity: 0.5,
                computation_time_ms: 10.0,
            })
        }

        async fn compute_shortest_paths(
            &mut self,
            source_node_id: u32,
        ) -> Result<PathfindingResult> {
            Ok(PathfindingResult {
                source_node: source_node_id,
                distances: HashMap::new(),
                paths: HashMap::new(),
                computation_time_ms: 5.0,
            })
        }

        async fn compute_sssp_distances(
            &mut self,
            _source_node_id: u32,
        ) -> Result<Vec<f32>> {
            Ok(vec![])
        }

        async fn compute_all_pairs_shortest_paths(
            &mut self,
        ) -> Result<HashMap<(u32, u32), Vec<u32>>> {
            Ok(HashMap::new())
        }

        async fn compute_landmark_apsp(
            &mut self,
            _num_landmarks: usize,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(vec![])
        }

        async fn generate_semantic_constraints(
            &mut self,
            _config: SemanticConstraintConfig,
        ) -> Result<ConstraintSet> {
            Ok(ConstraintSet::default())
        }

        async fn optimize_layout(
            &mut self,
            _constraints: &ConstraintSet,
            _max_iterations: usize,
        ) -> Result<OptimizationResult> {
            Ok(OptimizationResult {
                converged: true,
                iterations: 100,
                final_stress: 0.01,
                convergence_delta: 0.001,
                computation_time_ms: 50.0,
            })
        }

        async fn analyze_node_importance(
            &mut self,
            _algorithm: ImportanceAlgorithm,
        ) -> Result<HashMap<u32, f32>> {
            Ok(HashMap::new())
        }

        async fn update_graph_data(&mut self, _graph: Arc<GraphData>) -> SemanticResult<()> {
            Ok(())
        }

        async fn get_statistics(&self) -> SemanticResult<SemanticStatistics> {
            Ok(SemanticStatistics {
                total_analyses: 50,
                average_clustering_time_ms: 15.0,
                average_pathfinding_time_ms: 8.0,
                cache_hit_rate: 0.75,
                gpu_memory_used_mb: 512.0,
            })
        }

        async fn invalidate_pathfinding_cache(&mut self) -> SemanticResult<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_semantic_service_creation() {
        let adapter = Arc::new(RwLock::new(MockSemanticAnalyzer));
        let event_bus = Arc::new(RwLock::new(EventBus::new()));
        let service = SemanticService::new(adapter, event_bus);

        let stats = service.get_statistics().await.unwrap();
        assert_eq!(stats.total_analyses, 50);
    }

    #[tokio::test]
    async fn test_detect_communities() {
        let adapter = Arc::new(RwLock::new(MockSemanticAnalyzer));
        let event_bus = Arc::new(RwLock::new(EventBus::new()));
        let service = SemanticService::new(adapter, event_bus);

        let result = service.detect_communities_louvain().await.unwrap();
        assert_eq!(result.modularity, 0.5);
    }
}
