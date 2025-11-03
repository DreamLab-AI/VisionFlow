// tests/ontology_reasoning_integration_test.rs
//! Integration tests for OntologyReasoningService
//!
//! Tests the complete reasoning pipeline from ontology loading to inference.

#![cfg(test)]

use std::collections::HashMap;
use std::sync::Arc;

use webxr::adapters::whelk_inference_engine::WhelkInferenceEngine;
use webxr::ports::ontology_repository::{AxiomType, OntologyRepository, OwlAxiom, OwlClass};
use webxr::repositories::unified_ontology_repository::UnifiedOntologyRepository;
use webxr::services::ontology_reasoning_service::OntologyReasoningService;

#[tokio::test]
async fn test_reasoning_service_initialization() {
    let engine = Arc::new(WhelkInferenceEngine::new());
    let repo = Arc::new(
        UnifiedOntologyRepository::new(":memory:").expect("Failed to create repository"),
    );

    let service = OntologyReasoningService::new(engine, repo);

    // Service should initialize without errors
    service.clear_cache().await;
}

#[tokio::test]
async fn test_infer_axioms_simple_hierarchy() {
    // Setup
    let engine = Arc::new(WhelkInferenceEngine::new());
    let repo = Arc::new(
        UnifiedOntologyRepository::new(":memory:").expect("Failed to create repository"),
    );
    let service = OntologyReasoningService::new(engine.clone(), repo.clone());

    // Create simple class hierarchy: Employee -> Person -> Thing
    let classes = vec![
        OwlClass {
            iri: "http://example.org/Thing".to_string(),
            label: Some("Thing".to_string()),
            description: Some("Top-level class".to_string()),
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
        OwlClass {
            iri: "http://example.org/Person".to_string(),
            label: Some("Person".to_string()),
            description: Some("A person".to_string()),
            parent_classes: vec!["http://example.org/Thing".to_string()],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
        OwlClass {
            iri: "http://example.org/Employee".to_string(),
            label: Some("Employee".to_string()),
            description: Some("An employee".to_string()),
            parent_classes: vec!["http://example.org/Person".to_string()],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
    ];

    let axioms = vec![
        OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "http://example.org/Person".to_string(),
            object: "http://example.org/Thing".to_string(),
            annotations: HashMap::new(),
        },
        OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "http://example.org/Employee".to_string(),
            object: "http://example.org/Person".to_string(),
            annotations: HashMap::new(),
        },
    ];

    // Save ontology
    repo.save_ontology(&classes, &[], &axioms)
        .await
        .expect("Failed to save ontology");

    // Test inference (this will use whelk-rs)
    // Note: whelk may infer transitive relationships like Employee -> Thing
    let inferred = service
        .infer_axioms("default")
        .await
        .expect("Failed to infer axioms");

    // Verify we got some inferences (exact count depends on reasoner)
    println!("Inferred {} axioms", inferred.len());

    // Verify all inferred axioms are valid
    for axiom in &inferred {
        assert!(!axiom.subject_iri.is_empty());
        assert!(axiom.object_iri.is_some());
        assert_eq!(axiom.user_defined, false);
        assert!(axiom.confidence > 0.0 && axiom.confidence <= 1.0);
    }
}

#[tokio::test]
async fn test_class_hierarchy_computation() {
    let engine = Arc::new(WhelkInferenceEngine::new());
    let repo = Arc::new(
        UnifiedOntologyRepository::new(":memory:").expect("Failed to create repository"),
    );
    let service = OntologyReasoningService::new(engine, repo.clone());

    // Create multi-level hierarchy
    let classes = vec![
        OwlClass {
            iri: "root".to_string(),
            label: Some("Root".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
        OwlClass {
            iri: "child1".to_string(),
            label: Some("Child 1".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
        OwlClass {
            iri: "child2".to_string(),
            label: Some("Child 2".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
        OwlClass {
            iri: "grandchild".to_string(),
            label: Some("Grandchild".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
    ];

    let axioms = vec![
        OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "child1".to_string(),
            object: "root".to_string(),
            annotations: HashMap::new(),
        },
        OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "child2".to_string(),
            object: "root".to_string(),
            annotations: HashMap::new(),
        },
        OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "grandchild".to_string(),
            object: "child1".to_string(),
            annotations: HashMap::new(),
        },
    ];

    repo.save_ontology(&classes, &[], &axioms)
        .await
        .expect("Failed to save ontology");

    // Get hierarchy
    let hierarchy = service
        .get_class_hierarchy("default")
        .await
        .expect("Failed to get hierarchy");

    // Verify structure
    assert_eq!(hierarchy.root_classes.len(), 1);
    assert_eq!(hierarchy.root_classes[0], "root");

    assert_eq!(hierarchy.hierarchy.len(), 4);

    // Check root node
    let root = hierarchy.hierarchy.get("root").unwrap();
    assert_eq!(root.children_iris.len(), 2);
    assert_eq!(root.depth, 0);
    assert_eq!(root.parent_iri, None);

    // Check child1 node
    let child1 = hierarchy.hierarchy.get("child1").unwrap();
    assert_eq!(child1.children_iris.len(), 1);
    assert_eq!(child1.depth, 1);
    assert_eq!(child1.parent_iri, Some("root".to_string()));

    // Check grandchild node
    let grandchild = hierarchy.hierarchy.get("grandchild").unwrap();
    assert_eq!(grandchild.children_iris.len(), 0);
    assert_eq!(grandchild.depth, 2);
    assert_eq!(grandchild.parent_iri, Some("child1".to_string()));
}

#[tokio::test]
async fn test_disjoint_classes_detection() {
    let engine = Arc::new(WhelkInferenceEngine::new());
    let repo = Arc::new(
        UnifiedOntologyRepository::new(":memory:").expect("Failed to create repository"),
    );
    let service = OntologyReasoningService::new(engine, repo.clone());

    // Create classes with disjoint axiom
    let classes = vec![
        OwlClass {
            iri: "http://example.org/Cat".to_string(),
            label: Some("Cat".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
        OwlClass {
            iri: "http://example.org/Dog".to_string(),
            label: Some("Dog".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
    ];

    let axioms = vec![OwlAxiom {
        id: None,
        axiom_type: AxiomType::DisjointWith,
        subject: "http://example.org/Cat".to_string(),
        object: "http://example.org/Dog".to_string(),
        annotations: HashMap::new(),
    }];

    repo.save_ontology(&classes, &[], &axioms)
        .await
        .expect("Failed to save ontology");

    // Get disjoint pairs
    let disjoint = service
        .get_disjoint_classes("default")
        .await
        .expect("Failed to get disjoint classes");

    // Verify
    assert_eq!(disjoint.len(), 1);
    assert_eq!(disjoint[0].class_a, "http://example.org/Cat");
    assert_eq!(disjoint[0].class_b, "http://example.org/Dog");
    assert!(!disjoint[0].reason.is_empty());
}

#[tokio::test]
async fn test_cache_invalidation() {
    let engine = Arc::new(WhelkInferenceEngine::new());
    let repo = Arc::new(
        UnifiedOntologyRepository::new(":memory:").expect("Failed to create repository"),
    );
    let service = OntologyReasoningService::new(engine, repo.clone());

    // Create initial ontology
    let classes = vec![OwlClass {
        iri: "test".to_string(),
        label: Some("Test".to_string()),
        description: None,
        parent_classes: vec![],
        properties: HashMap::new(),
        source_file: None,
        markdown_content: None,
        file_sha1: None,
        last_synced: None,
    }];

    repo.save_ontology(&classes, &[], &[])
        .await
        .expect("Failed to save ontology");

    // First inference (cache miss)
    let result1 = service.infer_axioms("default").await;
    assert!(result1.is_ok());

    // Second inference (should use cache)
    let result2 = service.infer_axioms("default").await;
    assert!(result2.is_ok());

    // Modify ontology
    let classes2 = vec![
        OwlClass {
            iri: "test".to_string(),
            label: Some("Test".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
        OwlClass {
            iri: "test2".to_string(),
            label: Some("Test 2".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        },
    ];

    repo.save_ontology(&classes2, &[], &[])
        .await
        .expect("Failed to save ontology");

    // Third inference (cache should be invalidated)
    let result3 = service.infer_axioms("default").await;
    assert!(result3.is_ok());

    // Clear cache manually
    service.clear_cache().await;

    // Fourth inference (cache miss after clear)
    let result4 = service.infer_axioms("default").await;
    assert!(result4.is_ok());
}
