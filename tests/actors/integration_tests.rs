// tests/actors/integration_tests.rs
//! Integration Tests for Actor System with Hexagonal Architecture
//!
//! Tests the integration of Actix actors through adapter layer.

use std::sync::Arc;
use tokio::sync::RwLock;

use visionflow::actors::lifecycle::{ActorLifecycleManager, SupervisionStrategy, SupervisionDecision};
use visionflow::application::physics_service::{PhysicsService, SimulationParams};
use visionflow::application::semantic_service::{SemanticService, CommunityDetectionRequest};
use visionflow::events::event_bus::EventBus;
use visionflow::models::graph::GraphData;
use visionflow::models::node::Node;
use visionflow::models::edge::Edge;
use visionflow::ports::gpu_physics_adapter::{GpuPhysicsAdapter, PhysicsParameters};
use visionflow::ports::gpu_semantic_analyzer::{GpuSemanticAnalyzer, ClusteringAlgorithm};

// Mock implementations for testing
mod mocks {
    use super::*;
    use async_trait::async_trait;
    use std::collections::HashMap;

    use visionflow::models::constraints::ConstraintSet;
    use visionflow::ports::gpu_physics_adapter::*;
    use visionflow::ports::gpu_semantic_analyzer::*;

    pub struct MockPhysicsAdapter {
        pub initialized: Arc<RwLock<bool>>,
    }

    impl Default for MockPhysicsAdapter {
        fn default() -> Self {
            Self {
                initialized: Arc::new(RwLock::new(false)),
            }
        }
    }

    #[async_trait]
    impl GpuPhysicsAdapter for MockPhysicsAdapter {
        async fn initialize(&mut self, _graph: Arc<GraphData>, _params: PhysicsParameters) -> Result<()> {
            *self.initialized.write().await = true;
            Ok(())
        }

        async fn compute_forces(&mut self) -> Result<Vec<NodeForce>> {
            Ok(vec![
                NodeForce {
                    node_id: 0,
                    force_x: 1.0,
                    force_y: 2.0,
                    force_z: 0.0,
                }
            ])
        }

        async fn update_positions(&mut self, _forces: &[NodeForce]) -> Result<Vec<(u32, f32, f32, f32)>> {
            Ok(vec![(0, 10.0, 20.0, 0.0)])
        }

        async fn step(&mut self) -> Result<PhysicsStepResult> {
            Ok(PhysicsStepResult {
                nodes_updated: 1,
                total_energy: 0.5,
                max_displacement: 1.0,
                converged: false,
                computation_time_ms: 5.0,
            })
        }

        async fn simulate_until_convergence(&mut self) -> Result<PhysicsStepResult> {
            Ok(PhysicsStepResult {
                nodes_updated: 1,
                total_energy: 0.01,
                max_displacement: 0.1,
                converged: true,
                computation_time_ms: 50.0,
            })
        }

        async fn apply_external_forces(&mut self, _forces: Vec<(u32, f32, f32, f32)>) -> Result<()> {
            Ok(())
        }

        async fn pin_nodes(&mut self, _nodes: Vec<(u32, f32, f32, f32)>) -> Result<()> {
            Ok(())
        }

        async fn unpin_nodes(&mut self, _node_ids: Vec<u32>) -> Result<()> {
            Ok(())
        }

        async fn update_parameters(&mut self, _params: PhysicsParameters) -> Result<()> {
            Ok(())
        }

        async fn update_graph_data(&mut self, _graph: Arc<GraphData>) -> Result<()> {
            Ok(())
        }

        async fn get_gpu_status(&self) -> Result<GpuDeviceInfo> {
            Ok(GpuDeviceInfo {
                device_id: 0,
                device_name: "Mock GPU".to_string(),
                compute_capability: (7, 5),
                total_memory_mb: 8192,
                free_memory_mb: 4096,
                multiprocessor_count: 40,
                warp_size: 32,
                max_threads_per_block: 1024,
            })
        }

        async fn get_statistics(&self) -> Result<PhysicsStatistics> {
            Ok(PhysicsStatistics {
                total_steps: 100,
                average_step_time_ms: 1.5,
                average_energy: 0.5,
                gpu_memory_used_mb: 256.0,
                cache_hit_rate: 0.8,
                last_convergence_iterations: 50,
            })
        }

        async fn reset(&mut self) -> Result<()> {
            *self.initialized.write().await = false;
            Ok(())
        }

        async fn cleanup(&mut self) -> Result<()> {
            Ok(())
        }
    }

    pub struct MockSemanticAnalyzer;

    #[async_trait]
    impl GpuSemanticAnalyzer for MockSemanticAnalyzer {
        async fn initialize(&mut self, _graph: Arc<GraphData>) -> SemanticResult<()> {
            Ok(())
        }

        async fn detect_communities(
            &mut self,
            _algorithm: ClusteringAlgorithm,
        ) -> SemanticResult<CommunityDetectionResult> {
            let mut clusters = HashMap::new();
            clusters.insert(0, 0);
            clusters.insert(1, 0);
            clusters.insert(2, 1);

            let mut cluster_sizes = HashMap::new();
            cluster_sizes.insert(0, 2);
            cluster_sizes.insert(1, 1);

            Ok(CommunityDetectionResult {
                clusters,
                cluster_sizes,
                modularity: 0.5,
                computation_time_ms: 10.0,
            })
        }

        async fn compute_shortest_paths(&mut self, source_node_id: u32) -> SemanticResult<PathfindingResult> {
            let mut distances = HashMap::new();
            distances.insert(source_node_id, 0.0);

            Ok(PathfindingResult {
                source_node: source_node_id,
                distances,
                paths: HashMap::new(),
                computation_time_ms: 5.0,
            })
        }

        async fn compute_sssp_distances(&mut self, _source_node_id: u32) -> SemanticResult<Vec<f32>> {
            Ok(vec![0.0, 1.0, 2.0])
        }

        async fn compute_all_pairs_shortest_paths(&mut self) -> SemanticResult<HashMap<(u32, u32), Vec<u32>>> {
            Ok(HashMap::new())
        }

        async fn compute_landmark_apsp(&mut self, _num_landmarks: usize) -> SemanticResult<Vec<Vec<f32>>> {
            Ok(vec![vec![0.0, 1.0], vec![1.0, 0.0]])
        }

        async fn generate_semantic_constraints(
            &mut self,
            _config: SemanticConstraintConfig,
        ) -> SemanticResult<ConstraintSet> {
            Ok(ConstraintSet::default())
        }

        async fn optimize_layout(
            &mut self,
            _constraints: &ConstraintSet,
            _max_iterations: usize,
        ) -> SemanticResult<OptimizationResult> {
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
        ) -> SemanticResult<HashMap<u32, f32>> {
            let mut scores = HashMap::new();
            scores.insert(0, 0.5);
            scores.insert(1, 0.3);
            scores.insert(2, 0.2);
            Ok(scores)
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
}

fn create_test_graph() -> GraphData {
    GraphData {
        nodes: vec![
            Node {
                id: "0".to_string(),
                label: "Node 0".to_string(),
                node_type: "test".to_string(),
                x: Some(0.0),
                y: Some(0.0),
                z: Some(0.0),
                properties: Default::default(),
            },
            Node {
                id: "1".to_string(),
                label: "Node 1".to_string(),
                node_type: "test".to_string(),
                x: Some(1.0),
                y: Some(1.0),
                z: Some(0.0),
                properties: Default::default(),
            },
        ],
        edges: vec![
            Edge {
                id: "e0".to_string(),
                source: "0".to_string(),
                target: "1".to_string(),
                edge_type: "test".to_string(),
                weight: 1.0,
                properties: Default::default(),
            },
        ],
    }
}

#[tokio::test]
async fn test_physics_service_integration() {
    let adapter = Arc::new(RwLock::new(mocks::MockPhysicsAdapter::default()));
    let event_bus = Arc::new(RwLock::new(EventBus::new()));
    let service = PhysicsService::new(adapter.clone(), event_bus);

    let graph = create_test_graph();
    let params = SimulationParams::default();

    let result = service.start_simulation(Arc::new(graph), params).await;
    assert!(result.is_ok());

    let initialized = adapter.read().await.initialized.read().await;
    assert!(*initialized);
}

#[tokio::test]
async fn test_semantic_service_integration() {
    let adapter = Arc::new(RwLock::new(mocks::MockSemanticAnalyzer));
    let event_bus = Arc::new(RwLock::new(EventBus::new()));
    let service = SemanticService::new(adapter, event_bus);

    let graph = create_test_graph();
    service.initialize(Arc::new(graph)).await.unwrap();

    let result = service.detect_communities_louvain().await;
    assert!(result.is_ok());

    let communities = result.unwrap();
    assert_eq!(communities.clusters.len(), 3);
    assert_eq!(communities.modularity, 0.5);
}

#[tokio::test]
async fn test_physics_simulation_lifecycle() {
    let adapter = Arc::new(RwLock::new(mocks::MockPhysicsAdapter::default()));
    let event_bus = Arc::new(RwLock::new(EventBus::new()));
    let service = PhysicsService::new(adapter, event_bus);

    let graph = create_test_graph();

    // Start simulation
    let sim_id = service.start_simulation(Arc::new(graph), SimulationParams::default())
        .await.unwrap();
    assert!(!sim_id.is_empty());
    assert!(service.is_running().await);

    // Perform step
    let step_result = service.step().await.unwrap();
    assert_eq!(step_result.nodes_updated, 1);

    // Stop simulation
    service.stop_simulation().await.unwrap();
    assert!(!service.is_running().await);
}

#[tokio::test]
async fn test_semantic_centrality_computation() {
    let adapter = Arc::new(RwLock::new(mocks::MockSemanticAnalyzer));
    let event_bus = Arc::new(RwLock::new(EventBus::new()));
    let service = SemanticService::new(adapter, event_bus);

    let scores = service.compute_pagerank(0.85, 100).await.unwrap();
    assert!(!scores.is_empty());
    assert_eq!(scores.len(), 3);
}

#[tokio::test]
async fn test_actor_lifecycle_manager() {
    let mut manager = ActorLifecycleManager::new();
    assert!(!manager.is_healthy());

    // Note: Full lifecycle test would require actual actor system
    // This is a simplified version
}

#[tokio::test]
async fn test_supervision_strategy() {
    let strategy = SupervisionStrategy::default();

    let decision1 = strategy.handle_failure("test_actor", 0).await;
    assert_eq!(decision1, SupervisionDecision::Restart);

    let decision2 = strategy.handle_failure("test_actor", 3).await;
    assert_eq!(decision2, SupervisionDecision::Stop);
}

#[tokio::test]
async fn test_event_driven_coordination() {
    // Test would require full event bus setup
    // Placeholder for actual implementation
}

#[tokio::test]
async fn test_backward_compatibility() {
    use visionflow::actors::backward_compat::LegacyActorCompat;

    std::env::set_var("VISIONFLOW_LEGACY_ACTORS", "true");
    assert!(LegacyActorCompat::legacy_mode_enabled());
}

#[tokio::test]
async fn test_physics_gpu_status() {
    let adapter = Arc::new(RwLock::new(mocks::MockPhysicsAdapter::default()));
    let event_bus = Arc::new(RwLock::new(EventBus::new()));
    let service = PhysicsService::new(adapter, event_bus);

    let status = service.get_gpu_status().await.unwrap();
    assert_eq!(status.device_name, "Mock GPU");
    assert_eq!(status.compute_capability, (7, 5));
}

#[tokio::test]
async fn test_semantic_statistics() {
    let adapter = Arc::new(RwLock::new(mocks::MockSemanticAnalyzer));
    let event_bus = Arc::new(RwLock::new(EventBus::new()));
    let service = SemanticService::new(adapter, event_bus);

    let stats = service.get_statistics().await.unwrap();
    assert_eq!(stats.total_analyses, 50);
    assert_eq!(stats.cache_hit_rate, 0.75);
}
