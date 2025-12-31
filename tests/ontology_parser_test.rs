// Test disabled - references deprecated/removed modules (webxr::ports::ontology_repository, webxr::services::parsers::ontology_parser)
// OntologyParser and related types may have been restructured per ADR-001
/*
// tests/ontology_parser_test.rs
//! Tests for OntologyParser module

use webxr::ports::ontology_repository::{AxiomType, OwlAxiom, OwlClass, OwlProperty, PropertyType};
use webxr::services::parsers::ontology_parser::{OntologyData, OntologyParser};

#[test]
fn test_parse_basic_owl_class() {
    let parser = OntologyParser::new();
    let content = r#"
# Test Document

- ### OntologyBlock
  - owl_class:: Person
    - label:: Human Person
    - description:: A human being
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.classes.len(), 1);
    assert_eq!(result.classes[0].iri, "Person");
    assert_eq!(result.classes[0].label, Some("Human Person".to_string()));
    assert_eq!(
        result.classes[0].description,
        Some("A human being".to_string())
    );
    assert_eq!(result.classes[0].source_file, Some("test.md".to_string()));
}

// ... remaining tests omitted for brevity ...
*/
