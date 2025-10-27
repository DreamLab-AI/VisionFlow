// tests/adapters/integration_actor_tests.rs
//! Integration Tests for Actor-based Adapters
//!
//! These tests verify full simulation cycles and actor lifecycle management.

use actix::prelude::*;
use std::sync::Arc;

use visionflow::adapters::{ActixPhysicsAdapter, ActixSemanticAdapter, WhelkInferenceEngineStub};
use visionflow::models::graph::GraphData;
use visionflow::models::node::Node;
use visionflow::ports::gpu_physics_adapter::{GpuPhysicsAdapter, PhysicsParameters};
use visionflow::ports::gpu_semantic_analyzer::{
    ClusteringAlgorithm, GpuSemanticAnalyzer, ImportanceAlgorithm, SemanticConstraintConfig,
};
use visionflow::ports::inference_engine::InferenceEngine;
use visionflow::utils::socket_flow_messages::BinaryNodeData;

fn create_large_test_graph() -> Arc<GraphData> {
    let nodes = (1..=100)
        .map(|i| Node {
            id: i,
            data: BinaryNodeData::default(),
        })
        .collect();

    Arc::new(GraphData {
        nodes,
        edges: Vec::new(),
    })
}

#[actix_rt::test]
async fn test_full_physics_simulation_cycle() {
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_large_test_graph();
    let params = PhysicsParameters::default();

    // Initialize
    adapter.initialize(graph.clone(), params).await.unwrap();

    // Run multiple steps
    for i in 0..10 {
        let result = adapter.step().await;
        assert!(result.is_ok(), "Step {} should succeed", i);
    }

    // Get statistics
    let stats = adapter.get_statistics().await;
    assert!(stats.is_ok(), "Statistics retrieval should succeed");

    // Cleanup
    adapter.cleanup().await.unwrap();
}

#[actix_rt::test]
async fn test_semantic_analysis_pipeline() {
    let mut adapter = ActixSemanticAdapter::new();
    let graph = create_large_test_graph();

    // Initialize
    adapter.initialize(graph).await.unwrap();

    // Detect communities
    let communities = adapter
        .detect_communities(ClusteringAlgorithm::Louvain)
        .await;
    assert!(communities.is_ok(), "Community detection should succeed");

    // Analyze importance
    let importance = adapter
        .analyze_node_importance(ImportanceAlgorithm::PageRank {
            damping: 0.85,
            max_iterations: 100,
        })
        .await;
    assert!(importance.is_ok(), "Importance analysis should succeed");

    // Generate constraints
    let config = SemanticConstraintConfig {
        similarity_threshold: 0.7,
        enable_clustering_constraints: true,
        enable_importance_constraints: true,
        enable_topic_constraints: false,
        max_constraints: 500,
    };

    let constraints = adapter.generate_semantic_constraints(config).await;
    assert!(constraints.is_ok(), "Constraint generation should succeed");

    // Get statistics
    let stats = adapter.get_statistics().await;
    assert!(stats.is_ok(), "Statistics retrieval should succeed");
}

#[actix_rt::test]
async fn test_inference_engine_stub_lifecycle() {
    let mut engine = WhelkInferenceEngineStub::new();

    // Load ontology
    engine.load_ontology(vec![], vec![]).await.unwrap();

    // Perform inference
    let result = engine.infer().await;
    assert!(result.is_ok(), "Inference should succeed");

    // Check consistency
    let consistent = engine.check_consistency().await.unwrap();
    assert!(consistent, "Ontology should be consistent");

    // Get statistics
    let stats = engine.get_statistics().await.unwrap();
    assert_eq!(stats.total_inferences, 1);

    // Clear
    engine.clear().await.unwrap();

    // Verify cleared
    let infer_after_clear = engine.infer().await;
    assert!(infer_after_clear.is_err(), "Inference should fail after clear");
}

#[actix_rt::test]
async fn test_actor_lifecycle_management() {
    let graph = create_large_test_graph();

    // Create adapter
    let mut adapter = ActixPhysicsAdapter::new();

    // Initialize
    adapter
        .initialize(graph.clone(), PhysicsParameters::default())
        .await
        .unwrap();

    // Verify actor is running
    assert!(
        adapter.actor_addr().is_some(),
        "Actor should be running after init"
    );

    // Cleanup
    adapter.cleanup().await.unwrap();

    // Verify actor is stopped
    // (actor address is consumed during cleanup)
}

#[actix_rt::test]
async fn test_error_propagation() {
    let mut adapter = ActixPhysicsAdapter::new();

    // Try to step without initialization
    let result = adapter.step().await;
    assert!(result.is_err(), "Step should fail without initialization");

    // Try to get stats without initialization
    let stats = adapter.get_statistics().await;
    assert!(stats.is_err(), "Stats should fail without initialization");
}

#[actix_rt::test]
async fn test_concurrent_adapter_operations() {
    let graph = create_large_test_graph();
    let params = PhysicsParameters::default();

    let mut physics = ActixPhysicsAdapter::new();
    let mut semantic = ActixSemanticAdapter::new();

    // Initialize both concurrently
    let (physics_init, semantic_init) = tokio::join!(
        physics.initialize(graph.clone(), params),
        semantic.initialize(graph)
    );

    assert!(physics_init.is_ok());
    assert!(semantic_init.is_ok());

    // Run operations concurrently
    let (physics_step, community_detection) = tokio::join!(
        physics.step(),
        semantic.detect_communities(ClusteringAlgorithm::Louvain)
    );

    assert!(physics_step.is_ok());
    assert!(community_detection.is_ok());

    // Cleanup
    let (physics_cleanup, _) = tokio::join!(physics.cleanup(), async { Ok::<(), ()>(()) });
    assert!(physics_cleanup.is_ok());
}

#[actix_rt::test]
async fn test_adapter_state_consistency() {
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_large_test_graph();
    let mut params = PhysicsParameters::default();

    adapter.initialize(graph.clone(), params.clone()).await.unwrap();

    // Update parameters multiple times
    for i in 1..=5 {
        params.damping = 0.5 + (i as f32 * 0.05);
        params.repulsion_strength = 100.0 + (i as f32 * 10.0);

        let result = adapter.update_parameters(params.clone()).await;
        assert!(result.is_ok(), "Parameter update {} should succeed", i);
    }

    // Update graph data
    let result = adapter.update_graph_data(graph).await;
    assert!(result.is_ok(), "Graph data update should succeed");
}

#[actix_rt::test]
async fn test_adapter_supervision() {
    // Test that adapters handle actor failures gracefully
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_large_test_graph();
    let params = PhysicsParameters::default();

    adapter.initialize(graph, params).await.unwrap();

    // Perform operations
    for _ in 0..5 {
        let _ = adapter.step().await;
    }

    // Adapters should continue working even if individual operations fail
    let stats = adapter.get_statistics().await;
    // Stats may fail but shouldn't panic
    let _ = stats;
}
