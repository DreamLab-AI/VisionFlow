// Test disabled - references deprecated/removed modules (webxr::reasoning::inference_cache, reasoning_actor)
// The reasoning module structure has changed per ADR-001 architecture changes
/*
/// Integration tests for reasoning module
///
/// Tests:
/// - Custom reasoner with real ontology
/// - Inference caching performance
/// - Reasoning actor integration

use webxr::reasoning::{
    custom_reasoner::{CustomReasoner, Ontology, OWLClass, OntologyReasoner, AxiomType},
    inference_cache::InferenceCache,
    reasoning_actor::{ReasoningActor, TriggerReasoning},
};
use std::collections::{HashMap, HashSet};
use tempfile::TempDir;
use actix::Actor;

/// Create biological ontology for testing
fn create_biological_ontology() -> Ontology {
    let mut ontology = Ontology::default();

    // Entity -> MaterialEntity -> Cell -> (Neuron, Astrocyte)
    ontology.classes.insert("Entity".to_string(), OWLClass {
        iri: "http://purl.obolibrary.org/obo/BFO_0000001".to_string(),
        label: Some("Entity".to_string()),
        parent_class_iri: None,
    });

    ontology.classes.insert("MaterialEntity".to_string(), OWLClass {
        iri: "http://purl.obolibrary.org/obo/BFO_0000040".to_string(),
        label: Some("Material Entity".to_string()),
        parent_class_iri: Some("http://purl.obolibrary.org/obo/BFO_0000001".to_string()),
    });

    ontology.classes.insert("Cell".to_string(), OWLClass {
        iri: "http://purl.obolibrary.org/obo/CL_0000000".to_string(),
        label: Some("Cell".to_string()),
        parent_class_iri: Some("http://purl.obolibrary.org/obo/BFO_0000040".to_string()),
    });

    ontology.classes.insert("Neuron".to_string(), OWLClass {
        iri: "http://purl.obolibrary.org/obo/CL_0000540".to_string(),
        label: Some("Neuron".to_string()),
        parent_class_iri: Some("http://purl.obolibrary.org/obo/CL_0000000".to_string()),
    });

    ontology.classes.insert("Astrocyte".to_string(), OWLClass {
        iri: "http://purl.obolibrary.org/obo/CL_0000127".to_string(),
        label: Some("Astrocyte".to_string()),
        parent_class_iri: Some("http://purl.obolibrary.org/obo/CL_0000000".to_string()),
    });

    // Add SubClassOf relationships
    ontology.subclass_of.insert(
        "http://purl.obolibrary.org/obo/BFO_0000040".to_string(),
        vec!["http://purl.obolibrary.org/obo/BFO_0000001".to_string()].into_iter().collect()
    );

    ontology.subclass_of.insert(
        "http://purl.obolibrary.org/obo/CL_0000000".to_string(),
        vec!["http://purl.obolibrary.org/obo/BFO_0000040".to_string()].into_iter().collect()
    );

    ontology.subclass_of.insert(
        "http://purl.obolibrary.org/obo/CL_0000540".to_string(),
        vec!["http://purl.obolibrary.org/obo/CL_0000000".to_string()].into_iter().collect()
    );

    ontology.subclass_of.insert(
        "http://purl.obolibrary.org/obo/CL_0000127".to_string(),
        vec!["http://purl.obolibrary.org/obo/CL_0000000".to_string()].into_iter().collect()
    );

    // Add DisjointClasses: Neuron and Astrocyte
    ontology.disjoint_classes.push(vec![
        "http://purl.obolibrary.org/obo/CL_0000540".to_string(),
        "http://purl.obolibrary.org/obo/CL_0000127".to_string(),
    ].into_iter().collect());

    ontology
}

#[test]
fn test_biological_ontology_reasoning() {
    let ontology = create_biological_ontology();
    let reasoner = CustomReasoner::new();

    let inferred = reasoner.infer_axioms(&ontology).unwrap();

    // Should infer: Neuron SubClassOf MaterialEntity
    assert!(inferred.iter().any(|axiom|
        axiom.axiom_type == AxiomType::SubClassOf
        && axiom.subject.contains("CL_0000540") // Neuron
        && axiom.object.as_ref().unwrap().contains("BFO_0000040") // MaterialEntity
    ));

    // Should infer: Neuron SubClassOf Entity
    assert!(inferred.iter().any(|axiom|
        axiom.axiom_type == AxiomType::SubClassOf
        && axiom.subject.contains("CL_0000540") // Neuron
        && axiom.object.as_ref().unwrap().contains("BFO_0000001") // Entity
    ));

    println!("Inferred {} axioms from biological ontology", inferred.len());
    for axiom in &inferred {
        println!("  {:?}: {} -> {}",
            axiom.axiom_type,
            axiom.subject,
            axiom.object.as_ref().unwrap_or(&"None".to_string())
        );
    }
}

#[test]
fn test_inference_cache_performance() {
    let temp_dir = TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("cache.db");

    let cache = InferenceCache::new(&cache_path).unwrap();
    let reasoner = CustomReasoner::new();
    let ontology = create_biological_ontology();

    // Cold start (cache miss)
    let start = std::time::Instant::now();
    let result1 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
    let cold_duration = start.elapsed();

    // Warm start (cache hit)
    let start = std::time::Instant::now();
    let result2 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
    let warm_duration = start.elapsed();

    assert_eq!(result1, result2);

    println!("Cache performance:");
    println!("  Cold (miss): {:?}", cold_duration);
    println!("  Warm (hit):  {:?}", warm_duration);

    // Cache hit should be at least 5x faster
    assert!(warm_duration < cold_duration / 5);

    // Target: cache hit < 20ms
    assert!(warm_duration.as_millis() < 20,
        "Cache hit took {}ms, expected <20ms", warm_duration.as_millis());
}

#[test]
fn test_large_ontology_performance() {
    // Create larger ontology with 1000 classes
    let mut ontology = Ontology::default();

    for i in 0..1000 {
        ontology.classes.insert(format!("Class{}", i), OWLClass {
            iri: format!("http://example.org/Class{}", i),
            label: Some(format!("Class {}", i)),
            parent_class_iri: if i > 0 {
                Some(format!("http://example.org/Class{}", i - 1))
            } else {
                None
            },
        });

        if i > 0 {
            ontology.subclass_of.insert(
                format!("http://example.org/Class{}", i),
                vec![format!("http://example.org/Class{}", i - 1)].into_iter().collect()
            );
        }
    }

    let reasoner = CustomReasoner::new();

    let start = std::time::Instant::now();
    let inferred = reasoner.infer_axioms(&ontology).unwrap();
    let duration = start.elapsed();

    println!("Large ontology reasoning:");
    println!("  Classes: 1000");
    println!("  Inferred axioms: {}", inferred.len());
    println!("  Duration: {:?}", duration);

    // Target: <100ms for 1000 classes
    assert!(duration.as_millis() < 100,
        "Reasoning took {}ms for 1000 classes, expected <100ms", duration.as_millis());
}

#[actix_rt::test]
async fn test_reasoning_actor_integration() {
    let temp_dir = TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("cache.db");

    let actor = ReasoningActor::new(cache_path.to_str().unwrap())
        .unwrap()
        .start();

    let ontology = create_biological_ontology();

    // Trigger reasoning
    let result = actor.send(TriggerReasoning {
        ontology_id: 1,
        ontology,
    }).await;

    assert!(result.is_ok());
    let axioms = result.unwrap().unwrap();

    println!("Reasoning actor produced {} axioms", axioms.len());
    assert!(!axioms.is_empty());
}

#[test]
fn test_checksum_computation() {
    let temp_dir = TempDir::new().unwrap();
    let cache_path = temp_dir.path().join("cache.db");
    let cache = InferenceCache::new(&cache_path).unwrap();

    let ontology1 = create_biological_ontology();
    let mut ontology2 = create_biological_ontology();

    // Same ontology should have same checksum
    let reasoner = CustomReasoner::new();
    let result1 = cache.get_or_compute(1, &reasoner, &ontology1).unwrap();
    let result2 = cache.get_or_compute(1, &reasoner, &ontology1).unwrap();
    assert_eq!(result1, result2);

    // Modified ontology should have different checksum
    ontology2.classes.insert("NewClass".to_string(), OWLClass {
        iri: "http://example.org/NewClass".to_string(),
        label: Some("New Class".to_string()),
        parent_class_iri: None,
    });

    let result3 = cache.get_or_compute(2, &reasoner, &ontology2).unwrap();
    assert_ne!(result1.len(), result3.len());
}
*/
