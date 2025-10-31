// tests/test_ontology_schema_fixes.rs
//! Tests to verify ontology database schema fixes
//! Tests that INSERT operations work with correct column names

use webxr::repositories::UnifiedOntologyRepository;
use webxr::ports::ontology_repository::{OntologyRepository, OwlClass, OwlProperty, PropertyType};

#[tokio::test]
async fn test_schema_creates_correctly() {
    // Create a temporary database
    let temp_db = tempfile::NamedTempFile::new().unwrap();
    let db_path = temp_db.path().to_str().unwrap();

    // Initialize repository - this creates the schema
    let repo = UnifiedOntologyRepository::new(db_path).expect("Failed to create repository");

    // Verify we can get metrics without errors
    let metrics = repo.get_metrics().await.expect("Failed to get metrics");
    assert_eq!(metrics.class_count, 0, "Should start with 0 classes");
}

#[tokio::test]
async fn test_add_owl_class_with_correct_schema() {
    let temp_db = tempfile::NamedTempFile::new().unwrap();
    let db_path = temp_db.path().to_str().unwrap();
    let repo = UnifiedOntologyRepository::new(db_path).expect("Failed to create repository");

    // Create a test class
    let test_class = OwlClass {
        iri: "http://example.org/TestClass".to_string(),
        label: Some("Test Class".to_string()),
        description: Some("A test class".to_string()),
        parent_classes: vec!["http://www.w3.org/2002/07/owl#Thing".to_string()],
        properties: std::collections::HashMap::new(),
        source_file: None,
        markdown_content: None,
        file_sha1: Some("abc123".to_string()),
        last_synced: None,
    };

    // Add the class - this tests INSERT with correct column names
    let result = repo.add_owl_class(&test_class).await;
    assert!(result.is_ok(), "Failed to add owl class: {:?}", result.err());

    // Verify we can retrieve it
    let retrieved = repo.get_owl_class("http://example.org/TestClass").await
        .expect("Failed to retrieve class");
    assert!(retrieved.is_some(), "Class should be retrievable");

    let retrieved_class = retrieved.unwrap();
    assert_eq!(retrieved_class.iri, test_class.iri);
    assert_eq!(retrieved_class.label, test_class.label);
    assert_eq!(retrieved_class.description, test_class.description);
    assert_eq!(retrieved_class.file_sha1, test_class.file_sha1);
}

#[tokio::test]
async fn test_add_owl_property_with_correct_schema() {
    let temp_db = tempfile::NamedTempFile::new().unwrap();
    let db_path = temp_db.path().to_str().unwrap();
    let repo = UnifiedOntologyRepository::new(db_path).expect("Failed to create repository");

    // Create a test property
    let test_property = OwlProperty {
        iri: "http://example.org/testProperty".to_string(),
        label: Some("Test Property".to_string()),
        property_type: PropertyType::ObjectProperty,
        domain: vec!["http://example.org/TestClass".to_string()],
        range: vec!["http://www.w3.org/2001/XMLSchema#string".to_string()],
    };

    // Add the property - tests INSERT with correct column names
    let result = repo.add_owl_property(&test_property).await;
    assert!(result.is_ok(), "Failed to add owl property: {:?}", result.err());

    // Verify we can retrieve it
    let retrieved = repo.get_owl_property("http://example.org/testProperty").await
        .expect("Failed to retrieve property");
    assert!(retrieved.is_some(), "Property should be retrievable");

    let retrieved_property = retrieved.unwrap();
    assert_eq!(retrieved_property.iri, test_property.iri);
    assert_eq!(retrieved_property.label, test_property.label);
}

#[tokio::test]
async fn test_class_hierarchy_foreign_keys() {
    let temp_db = tempfile::NamedTempFile::new().unwrap();
    let db_path = temp_db.path().to_str().unwrap();
    let repo = UnifiedOntologyRepository::new(db_path).expect("Failed to create repository");

    // Create parent class
    let parent_class = OwlClass {
        iri: "http://example.org/ParentClass".to_string(),
        label: Some("Parent Class".to_string()),
        description: None,
        parent_classes: vec![],
        properties: std::collections::HashMap::new(),
        source_file: None,
        markdown_content: None,
        file_sha1: None,
        last_synced: None,
    };

    // Create child class
    let child_class = OwlClass {
        iri: "http://example.org/ChildClass".to_string(),
        label: Some("Child Class".to_string()),
        description: None,
        parent_classes: vec!["http://example.org/ParentClass".to_string()],
        properties: std::collections::HashMap::new(),
        source_file: None,
        markdown_content: None,
        file_sha1: None,
        last_synced: None,
    };

    // Add both classes
    repo.add_owl_class(&parent_class).await.expect("Failed to add parent class");
    repo.add_owl_class(&child_class).await.expect("Failed to add child class");

    // Verify hierarchy was created correctly
    let retrieved_child = repo.get_owl_class("http://example.org/ChildClass").await
        .expect("Failed to retrieve child class");
    assert!(retrieved_child.is_some());

    let child = retrieved_child.unwrap();
    assert_eq!(child.parent_classes.len(), 1);
    assert_eq!(child.parent_classes[0], "http://example.org/ParentClass");
}

#[tokio::test]
async fn test_save_ontology_bulk() {
    let temp_db = tempfile::NamedTempFile::new().unwrap();
    let db_path = temp_db.path().to_str().unwrap();
    let repo = UnifiedOntologyRepository::new(db_path).expect("Failed to create repository");

    // Create test data
    let classes = vec![
        OwlClass {
            iri: "http://example.org/Class1".to_string(),
            label: Some("Class 1".to_string()),
            description: None,
            parent_classes: vec![],
            properties: std::collections::HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: Some("hash1".to_string()),
            last_synced: None,
        },
        OwlClass {
            iri: "http://example.org/Class2".to_string(),
            label: Some("Class 2".to_string()),
            description: None,
            parent_classes: vec!["http://example.org/Class1".to_string()],
            properties: std::collections::HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: Some("hash2".to_string()),
            last_synced: None,
        },
    ];

    let properties = vec![
        OwlProperty {
            iri: "http://example.org/property1".to_string(),
            label: Some("Property 1".to_string()),
            property_type: PropertyType::ObjectProperty,
            domain: vec![],
            range: vec![],
        },
    ];

    // Save ontology - tests bulk INSERT with correct column names
    let result = repo.save_ontology(&classes, &properties, &[]).await;
    assert!(result.is_ok(), "Failed to save ontology: {:?}", result.err());

    // Verify data was saved
    let metrics = repo.get_metrics().await.expect("Failed to get metrics");
    assert_eq!(metrics.class_count, 2, "Should have 2 classes");
    assert_eq!(metrics.property_count, 1, "Should have 1 property");
}
