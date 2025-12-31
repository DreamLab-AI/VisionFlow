// tests/ports/test_ontology_repository.rs
//! Contract tests for OntologyRepository port

use super::mocks::MockOntologyRepository;
use std::collections::HashMap;
use webxr::ports::ontology_repository::*;
use webxr::ports::OntologyRepository;

fn create_test_class(iri: &str, label: &str) -> OwlClass {
    OwlClass {
        iri: iri.to_string(),
        label: Some(label.to_string()),
        ..Default::default()
    }
}

fn create_test_axiom(subject: &str, object: &str) -> OwlAxiom {
    OwlAxiom {
        id: None,
        axiom_type: AxiomType::SubClassOf,
        subject: subject.to_string(),
        object: object.to_string(),
        annotations: HashMap::new(),
    }
}

#[tokio::test]
async fn test_add_get_owl_class() {
    let repo = MockOntologyRepository::new();

    let class = create_test_class("http://example.org/Person", "Person");
    let iri = repo.add_owl_class(&class).await.unwrap();

    assert_eq!(iri, "http://example.org/Person");

    let loaded = repo.get_owl_class(&iri).await.unwrap().unwrap();
    assert_eq!(loaded.label, Some("Person".to_string()));
}

#[tokio::test]
async fn test_list_owl_classes() {
    let repo = MockOntologyRepository::new();

    repo.add_owl_class(&create_test_class("http://example.org/Person", "Person"))
        .await
        .unwrap();
    repo.add_owl_class(&create_test_class("http://example.org/Student", "Student"))
        .await
        .unwrap();

    let classes = repo.list_owl_classes().await.unwrap();
    assert_eq!(classes.len(), 2);
}

#[tokio::test]
async fn test_add_get_axiom() {
    let repo = MockOntologyRepository::new();

    let axiom = create_test_axiom("http://example.org/Student", "http://example.org/Person");
    let axiom_id = repo.add_axiom(&axiom).await.unwrap();

    assert!(axiom_id > 0);

    let axioms = repo
        .get_class_axioms("http://example.org/Student")
        .await
        .unwrap();
    assert_eq!(axioms.len(), 1);
}

#[tokio::test]
async fn test_save_ontology_batch() {
    let repo = MockOntologyRepository::new();

    let classes = vec![
        create_test_class("http://example.org/Person", "Person"),
        create_test_class("http://example.org/Student", "Student"),
    ];

    let properties = vec![OwlProperty {
        iri: "http://example.org/hasName".to_string(),
        label: Some("has name".to_string()),
        property_type: PropertyType::DataProperty,
        domain: vec!["http://example.org/Person".to_string()],
        range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
    }];

    let axioms = vec![create_test_axiom(
        "http://example.org/Student",
        "http://example.org/Person",
    )];

    repo.save_ontology(&classes, &properties, &axioms)
        .await
        .unwrap();

    let loaded_classes = repo.list_owl_classes().await.unwrap();
    assert_eq!(loaded_classes.len(), 2);

    let loaded_properties = repo.list_owl_properties().await.unwrap();
    assert_eq!(loaded_properties.len(), 1);
}

#[tokio::test]
async fn test_validate_ontology() {
    let repo = MockOntologyRepository::new();

    let report = repo.validate_ontology().await.unwrap();
    assert!(report.is_valid);
    assert_eq!(report.errors.len(), 0);
}

#[tokio::test]
async fn test_get_metrics() {
    let repo = MockOntologyRepository::new();

    repo.add_owl_class(&create_test_class("http://example.org/Person", "Person"))
        .await
        .unwrap();
    repo.add_axiom(&create_test_axiom(
        "http://example.org/Student",
        "http://example.org/Person",
    ))
    .await
    .unwrap();

    let metrics = repo.get_metrics().await.unwrap();
    assert_eq!(metrics.class_count, 1);
    assert_eq!(metrics.axiom_count, 1);
}
