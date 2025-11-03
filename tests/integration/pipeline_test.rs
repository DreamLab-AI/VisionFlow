// tests/integration/pipeline_test.rs
//! End-to-End Pipeline Integration Tests
//!
//! Tests the complete data flow from GitHub OWL upload to client WebSocket delivery.

use actix::Actor;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use visionflow::actors::gpu::ontology_constraint_actor::OntologyConstraintActor;
use visionflow::actors::messages::*;
use visionflow::reasoning::custom_reasoner::{CustomReasoner, OWLClass, Ontology};
use visionflow::reasoning::reasoning_actor::ReasoningActor;
use visionflow::repositories::unified_ontology_repository::UnifiedOntologyRepository;
use visionflow::services::github_sync_service::GitHubSyncService;
use visionflow::services::ontology_pipeline_service::{
    OntologyPipelineService, SemanticPhysicsConfig,
};
use visionflow::services::pipeline_events::{
    OntologyModifiedEvent, PipelineEventBus, PipelineEvent,
};

/// Test helper to create a simple ontology
fn create_test_ontology() -> Ontology {
    let mut ontology = Ontology::default();

    // Add classes
    ontology.classes.insert(
        "http://example.org/Person".to_string(),
        OWLClass {
            iri: "http://example.org/Person".to_string(),
            label: Some("Person".to_string()),
            parent_class_iri: None,
        },
    );

    ontology.classes.insert(
        "http://example.org/Student".to_string(),
        OWLClass {
            iri: "http://example.org/Student".to_string(),
            label: Some("Student".to_string()),
            parent_class_iri: Some("http://example.org/Person".to_string()),
        },
    );

    // Add subclass relationship
    ontology
        .subclass_of
        .entry("http://example.org/Student".to_string())
        .or_insert_with(std::collections::HashSet::new)
        .insert("http://example.org/Person".to_string());

    ontology
}

/// Test end-to-end pipeline: Ontology upload → Reasoning → Constraints → GPU
#[actix_rt::test]
async fn test_pipeline_end_to_end() {
    // Setup
    let temp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("reasoning_cache.db");

    // Initialize actors
    let reasoning_actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let constraint_actor = OntologyConstraintActor::new().start();

    // Initialize pipeline service
    let config = SemanticPhysicsConfig {
        auto_trigger_reasoning: true,
        auto_generate_constraints: true,
        use_gpu_constraints: false, // CPU fallback for testing
        constraint_strength: 1.0,
        max_reasoning_depth: 10,
        cache_inferences: true,
    };

    let mut pipeline_service = OntologyPipelineService::new(config);
    pipeline_service.set_reasoning_actor(reasoning_actor.clone());
    pipeline_service.set_constraint_actor(constraint_actor.clone());

    let pipeline_service = Arc::new(pipeline_service);

    // Create test ontology
    let ontology = create_test_ontology();
    let correlation_id = uuid::Uuid::new_v4().to_string();

    // Trigger pipeline
    let start = Instant::now();
    let result = pipeline_service
        .on_ontology_modified(1, ontology.clone(), correlation_id.clone())
        .await;

    // Verify pipeline completed successfully
    assert!(result.is_ok());
    let stats = result.unwrap();

    assert!(stats.reasoning_triggered);
    assert!(stats.inferred_axioms_count > 0);
    assert!(stats.constraints_generated > 0);

    // Verify timing is reasonable
    assert!(stats.total_time_ms < 5000); // Should complete in under 5 seconds

    println!("Pipeline completed in {}ms", stats.total_time_ms);
    println!("Inferred axioms: {}", stats.inferred_axioms_count);
    println!("Constraints generated: {}", stats.constraints_generated);
}

/// Test reasoning cache hit performance
#[actix_rt::test]
async fn test_reasoning_cache_hit() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("reasoning_cache.db");

    let reasoning_actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let ontology = create_test_ontology();

    // First call (cache miss)
    let start = Instant::now();
    let result1 = reasoning_actor
        .send(TriggerReasoning {
            ontology_id: 1,
            ontology: ontology.clone(),
        })
        .await
        .unwrap();
    let duration1 = start.elapsed();

    assert!(result1.is_ok());
    println!("Cache miss duration: {:?}", duration1);

    // Second call (cache hit)
    let start = Instant::now();
    let result2 = reasoning_actor
        .send(TriggerReasoning {
            ontology_id: 1,
            ontology: ontology.clone(),
        })
        .await
        .unwrap();
    let duration2 = start.elapsed();

    assert!(result2.is_ok());
    println!("Cache hit duration: {:?}", duration2);

    // Cache hit should be significantly faster
    assert!(duration2 < duration1 / 5);

    // Results should be identical
    let axioms1 = result1.unwrap();
    let axioms2 = result2.unwrap();
    assert_eq!(axioms1.len(), axioms2.len());
}

/// Test event bus event tracking
#[tokio::test]
async fn test_event_bus_tracking() {
    let mut event_bus = PipelineEventBus::new(1000);

    let ontology = create_test_ontology();
    let correlation_id = uuid::Uuid::new_v4().to_string();

    // Publish event
    let event = OntologyModifiedEvent {
        ontology_id: 1,
        ontology,
        source: "test".to_string(),
        correlation_id: correlation_id.clone(),
        timestamp: chrono::Utc::now(),
    };

    event_bus.publish(&event).await.unwrap();

    // Verify event was logged
    let events = event_bus.get_events_by_correlation(&correlation_id);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].event_type, "OntologyModified");
    assert_eq!(events[0].correlation_id, correlation_id);
}

/// Test pipeline error handling
#[actix_rt::test]
async fn test_pipeline_error_handling() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("reasoning_cache.db");

    let reasoning_actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let config = SemanticPhysicsConfig::default();
    let mut pipeline_service = OntologyPipelineService::new(config);
    pipeline_service.set_reasoning_actor(reasoning_actor);

    // Create invalid ontology (empty)
    let ontology = Ontology::default();
    let correlation_id = uuid::Uuid::new_v4().to_string();

    // Pipeline should handle gracefully
    let result = pipeline_service
        .on_ontology_modified(1, ontology, correlation_id)
        .await;

    // Should complete without panic
    assert!(result.is_ok());

    // Stats should reflect no axioms inferred
    let stats = result.unwrap();
    assert_eq!(stats.inferred_axioms_count, 0);
    assert_eq!(stats.constraints_generated, 0);
}

/// Test constraint generation from axioms
#[actix_rt::test]
async fn test_constraint_generation() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("reasoning_cache.db");

    let reasoning_actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let constraint_actor = OntologyConstraintActor::new().start();

    let config = SemanticPhysicsConfig {
        auto_trigger_reasoning: true,
        auto_generate_constraints: true,
        use_gpu_constraints: false,
        constraint_strength: 2.0, // Higher strength for testing
        max_reasoning_depth: 10,
        cache_inferences: true,
    };

    let mut pipeline_service = OntologyPipelineService::new(config);
    pipeline_service.set_reasoning_actor(reasoning_actor);
    pipeline_service.set_constraint_actor(constraint_actor.clone());

    let pipeline_service = Arc::new(pipeline_service);

    // Create ontology with multiple classes
    let mut ontology = create_test_ontology();

    // Add more classes for richer constraints
    ontology.classes.insert(
        "http://example.org/Teacher".to_string(),
        OWLClass {
            iri: "http://example.org/Teacher".to_string(),
            label: Some("Teacher".to_string()),
            parent_class_iri: Some("http://example.org/Person".to_string()),
        },
    );

    // Trigger pipeline
    let correlation_id = uuid::Uuid::new_v4().to_string();
    let result = pipeline_service
        .on_ontology_modified(1, ontology, correlation_id)
        .await;

    assert!(result.is_ok());
    let stats = result.unwrap();

    // Verify constraints were generated
    assert!(stats.constraints_generated > 0);
    println!("Generated {} constraints", stats.constraints_generated);

    // Query constraint actor for stats
    let constraint_stats = constraint_actor
        .send(GetOntologyConstraintStats)
        .await
        .unwrap();

    assert!(constraint_stats.is_ok());
}

/// Test pipeline timeout handling
#[actix_rt::test]
async fn test_pipeline_timeout() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("reasoning_cache.db");

    let reasoning_actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let config = SemanticPhysicsConfig::default();
    let mut pipeline_service = OntologyPipelineService::new(config);
    pipeline_service.set_reasoning_actor(reasoning_actor);

    let pipeline_service = Arc::new(pipeline_service);

    let ontology = create_test_ontology();
    let correlation_id = uuid::Uuid::new_v4().to_string();

    // Set timeout
    let timeout = Duration::from_secs(10);
    let start = Instant::now();

    // Trigger pipeline with timeout
    let result = tokio::time::timeout(
        timeout,
        pipeline_service.on_ontology_modified(1, ontology, correlation_id),
    )
    .await;

    // Should complete within timeout
    assert!(result.is_ok());
    assert!(start.elapsed() < timeout);
}

/// Test concurrent pipeline executions
#[actix_rt::test]
async fn test_concurrent_pipeline_executions() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("reasoning_cache.db");

    let reasoning_actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let config = SemanticPhysicsConfig::default();
    let mut pipeline_service = OntologyPipelineService::new(config);
    pipeline_service.set_reasoning_actor(reasoning_actor);

    let pipeline_service = Arc::new(pipeline_service);

    // Spawn multiple concurrent pipeline executions
    let mut handles = Vec::new();

    for i in 0..5 {
        let pipeline = pipeline_service.clone();
        let ontology = create_test_ontology();
        let correlation_id = format!("concurrent-{}", i);

        let handle = tokio::spawn(async move {
            pipeline
                .on_ontology_modified(i as i64, ontology, correlation_id)
                .await
        });

        handles.push(handle);
    }

    // Wait for all to complete
    let results = futures::future::join_all(handles).await;

    // All should succeed
    for result in results {
        assert!(result.is_ok());
        let stats = result.unwrap().unwrap();
        assert!(stats.reasoning_triggered);
    }
}

#[actix_rt::test]
async fn test_pipeline_metrics_collection() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("reasoning_cache.db");

    let reasoning_actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let config = SemanticPhysicsConfig::default();
    let mut pipeline_service = OntologyPipelineService::new(config);
    pipeline_service.set_reasoning_actor(reasoning_actor);

    let pipeline_service = Arc::new(pipeline_service);

    let ontology = create_test_ontology();
    let correlation_id = uuid::Uuid::new_v4().to_string();

    // Execute pipeline
    let result = pipeline_service
        .on_ontology_modified(1, ontology, correlation_id.clone())
        .await;

    assert!(result.is_ok());
    let stats = result.unwrap();

    // Verify metrics are populated
    assert!(stats.total_time_ms > 0);
    assert!(stats.reasoning_triggered);

    println!("Pipeline metrics:");
    println!("  Total time: {}ms", stats.total_time_ms);
    println!("  Inferred axioms: {}", stats.inferred_axioms_count);
    println!("  Constraints generated: {}", stats.constraints_generated);
    println!("  GPU upload: {}", stats.gpu_upload_success);
}
