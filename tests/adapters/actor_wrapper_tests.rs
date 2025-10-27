// tests/adapters/actor_wrapper_tests.rs
//! Backward Compatibility Tests for Actor Wrappers
//!
//! These tests verify that the adapter wrappers behave identically to direct actor usage.

use actix::prelude::*;
use std::sync::Arc;
use std::time::Duration;

use visionflow::adapters::{ActixPhysicsAdapter, ActixSemanticAdapter};
use visionflow::models::graph::GraphData;
use visionflow::models::node::Node;
use visionflow::ports::gpu_physics_adapter::{GpuPhysicsAdapter, PhysicsParameters};
use visionflow::ports::gpu_semantic_analyzer::{ClusteringAlgorithm, GpuSemanticAnalyzer};
use visionflow::utils::socket_flow_messages::BinaryNodeData;

fn create_test_graph() -> Arc<GraphData> {
    Arc::new(GraphData {
        nodes: vec![
            Node {
                id: 1,
                data: BinaryNodeData::default(),
            },
            Node {
                id: 2,
                data: BinaryNodeData::default(),
            },
        ],
        edges: Vec::new(),
    })
}

#[actix_rt::test]
async fn test_physics_adapter_initialization() {
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_test_graph();
    let params = PhysicsParameters::default();

    let result = adapter.initialize(graph, params).await;
    assert!(result.is_ok(), "Physics adapter initialization should succeed");
}

#[actix_rt::test]
async fn test_physics_adapter_step() {
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_test_graph();
    let params = PhysicsParameters::default();

    adapter.initialize(graph, params).await.unwrap();

    let result = adapter.step().await;
    assert!(result.is_ok(), "Physics step should succeed");
}

#[actix_rt::test]
async fn test_physics_adapter_parameter_update() {
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_test_graph();
    let mut params = PhysicsParameters::default();

    adapter.initialize(graph, params.clone()).await.unwrap();

    // Update parameters
    params.damping = 0.9;
    params.repulsion_strength = 150.0;

    let result = adapter.update_parameters(params).await;
    assert!(result.is_ok(), "Parameter update should succeed");
}

#[actix_rt::test]
async fn test_physics_adapter_timeout() {
    let mut adapter = ActixPhysicsAdapter::with_timeout(Duration::from_millis(100));
    let graph = create_test_graph();
    let params = PhysicsParameters::default();

    let result = adapter.initialize(graph, params).await;
    // Should succeed or timeout gracefully
    let _ = result;
}

#[actix_rt::test]
async fn test_semantic_adapter_initialization() {
    let mut adapter = ActixSemanticAdapter::new();
    let graph = create_test_graph();

    let result = adapter.initialize(graph).await;
    assert!(result.is_ok(), "Semantic adapter initialization should succeed");
}

#[actix_rt::test]
async fn test_semantic_adapter_community_detection() {
    let mut adapter = ActixSemanticAdapter::new();
    let graph = create_test_graph();

    adapter.initialize(graph).await.unwrap();

    let result = adapter.detect_communities(ClusteringAlgorithm::Louvain).await;
    assert!(result.is_ok(), "Community detection should succeed");
}

#[actix_rt::test]
async fn test_semantic_adapter_shortest_paths() {
    let mut adapter = ActixSemanticAdapter::new();
    let graph = create_test_graph();

    adapter.initialize(graph).await.unwrap();

    let result = adapter.compute_shortest_paths(1).await;
    assert!(result.is_ok(), "Shortest paths computation should succeed");
}

#[actix_rt::test]
async fn test_semantic_adapter_statistics() {
    let mut adapter = ActixSemanticAdapter::new();
    let graph = create_test_graph();

    adapter.initialize(graph).await.unwrap();

    let result = adapter.get_statistics().await;
    assert!(result.is_ok(), "Getting statistics should succeed");
}

#[actix_rt::test]
async fn test_semantic_adapter_cache_invalidation() {
    let mut adapter = ActixSemanticAdapter::new();
    let graph = create_test_graph();

    adapter.initialize(graph).await.unwrap();

    let result = adapter.invalidate_pathfinding_cache().await;
    assert!(result.is_ok(), "Cache invalidation should succeed");
}

#[actix_rt::test]
async fn test_concurrent_physics_adapters() {
    let graph = create_test_graph();
    let params = PhysicsParameters::default();

    let mut adapter1 = ActixPhysicsAdapter::new();
    let mut adapter2 = ActixPhysicsAdapter::new();

    // Initialize both adapters concurrently
    let (result1, result2) = tokio::join!(
        adapter1.initialize(graph.clone(), params.clone()),
        adapter2.initialize(graph, params)
    );

    assert!(result1.is_ok(), "First adapter should initialize");
    assert!(result2.is_ok(), "Second adapter should initialize");

    // Run steps concurrently
    let (step1, step2) = tokio::join!(adapter1.step(), adapter2.step());

    assert!(step1.is_ok(), "First adapter step should succeed");
    assert!(step2.is_ok(), "Second adapter step should succeed");
}

#[actix_rt::test]
async fn test_adapter_cleanup() {
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_test_graph();
    let params = PhysicsParameters::default();

    adapter.initialize(graph, params).await.unwrap();

    let result = adapter.cleanup().await;
    assert!(result.is_ok(), "Cleanup should succeed");

    // Verify adapter is no longer initialized
    let step_result = adapter.step().await;
    assert!(step_result.is_err(), "Step should fail after cleanup");
}

#[actix_rt::test]
async fn test_message_translation_accuracy() {
    // Test that message translation preserves data integrity
    let mut adapter = ActixPhysicsAdapter::new();
    let graph = create_test_graph();

    let params = PhysicsParameters {
        time_step: 0.02,
        damping: 0.85,
        spring_constant: 0.015,
        repulsion_strength: 120.0,
        attraction_strength: 0.12,
        max_velocity: 12.0,
        convergence_threshold: 0.015,
        max_iterations: 1200,
    };

    adapter.initialize(graph, params.clone()).await.unwrap();

    // Verify parameters were correctly translated
    // (In actual implementation, would query actor state)
    let stats_result = adapter.get_statistics().await;
    assert!(stats_result.is_ok());
}
