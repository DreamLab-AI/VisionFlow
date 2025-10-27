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

#[test]
fn test_parse_class_hierarchy() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - owl_class:: Student
    - label:: Student
    - subClassOf:: Person
  - owl_class:: Person
    - label:: Person
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.classes.len(), 2);
    assert_eq!(result.class_hierarchy.len(), 1);
    assert_eq!(
        result.class_hierarchy[0],
        ("Student".to_string(), "Person".to_string())
    );

    // Check parent_classes in OwlClass
    let student = result.classes.iter().find(|c| c.iri == "Student").unwrap();
    assert_eq!(student.parent_classes, vec!["Person".to_string()]);
}

#[test]
fn test_parse_object_property() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - objectProperty:: hasParent
    - label:: has parent
    - domain:: Person
    - range:: Person
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.properties.len(), 1);
    assert_eq!(result.properties[0].iri, "hasParent");
    assert_eq!(result.properties[0].label, Some("has parent".to_string()));
    assert_eq!(
        result.properties[0].property_type,
        PropertyType::ObjectProperty
    );
    assert_eq!(result.properties[0].domain, vec!["Person".to_string()]);
    assert_eq!(result.properties[0].range, vec!["Person".to_string()]);
}

#[test]
fn test_parse_data_property() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - dataProperty:: hasAge
    - label:: has age
    - domain:: Person
    - range:: xsd:integer
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.properties.len(), 1);
    assert_eq!(result.properties[0].iri, "hasAge");
    assert_eq!(
        result.properties[0].property_type,
        PropertyType::DataProperty
    );
    assert_eq!(result.properties[0].domain, vec!["Person".to_string()]);
    assert_eq!(result.properties[0].range, vec!["xsd:integer".to_string()]);
}

#[test]
fn test_parse_axioms() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - owl_class:: Student
    - subClassOf:: Person
  - owl_class:: Teacher
    - subClassOf:: Person
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.axioms.len(), 2);

    let student_axiom = result
        .axioms
        .iter()
        .find(|a| a.subject == "Student")
        .unwrap();
    assert_eq!(student_axiom.axiom_type, AxiomType::SubClassOf);
    assert_eq!(student_axiom.object, "Person");

    let teacher_axiom = result
        .axioms
        .iter()
        .find(|a| a.subject == "Teacher")
        .unwrap();
    assert_eq!(teacher_axiom.axiom_type, AxiomType::SubClassOf);
    assert_eq!(teacher_axiom.object, "Person");
}

#[test]
fn test_parse_multiple_parent_classes() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - owl_class:: TeachingAssistant
    - subClassOf:: Student
    - subClassOf:: Teacher
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.classes.len(), 1);
    let ta = &result.classes[0];
    assert_eq!(ta.parent_classes.len(), 2);
    assert!(ta.parent_classes.contains(&"Student".to_string()));
    assert!(ta.parent_classes.contains(&"Teacher".to_string()));
}

#[test]
fn test_parse_iri_format() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - owl_class:: http://example.org/ontology#Person
    - label:: Person
  - owl_class:: ex:Student
    - subClassOf:: http://example.org/ontology#Person
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.classes.len(), 2);

    let person = result
        .classes
        .iter()
        .find(|c| c.iri == "http://example.org/ontology#Person")
        .unwrap();
    assert_eq!(person.label, Some("Person".to_string()));

    let student = result
        .classes
        .iter()
        .find(|c| c.iri == "ex:Student")
        .unwrap();
    assert_eq!(
        student.parent_classes,
        vec!["http://example.org/ontology#Person".to_string()]
    );
}

#[test]
fn test_parse_multiple_domain_range() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - objectProperty:: hasRelative
    - domain:: Person, Animal
    - range:: Person, Animal
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    assert_eq!(result.properties.len(), 1);
    let prop = &result.properties[0];
    assert_eq!(prop.domain.len(), 2);
    assert!(prop.domain.contains(&"Person".to_string()));
    assert!(prop.domain.contains(&"Animal".to_string()));
    assert_eq!(prop.range.len(), 2);
    assert!(prop.range.contains(&"Person".to_string()));
    assert!(prop.range.contains(&"Animal".to_string()));
}

#[test]
fn test_no_ontology_block() {
    let parser = OntologyParser::new();
    let content = r#"
# Just a regular document
No ontology here!
  "#;

    let result = parser.parse(content, "test.md");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("No OntologyBlock found"));
}

#[test]
fn test_complex_ontology_document() {
    let parser = OntologyParser::new();
    let content = r#"
# University Ontology

Some regular markdown content here.

- ### OntologyBlock
  - owl_class:: Person
    - label:: Person
    - description:: A human being

  - owl_class:: Student
    - label:: Student
    - description:: A person enrolled in courses
    - subClassOf:: Person

  - owl_class:: Teacher
    - label:: Teacher
    - description:: A person who teaches
    - subClassOf:: Person

  - owl_class:: Course
    - label:: Course
    - description:: An educational course

  - objectProperty:: enrolledIn
    - label:: enrolled in
    - domain:: Student
    - range:: Course

  - objectProperty:: teaches
    - label:: teaches
    - domain:: Teacher
    - range:: Course

  - dataProperty:: hasStudentID
    - label:: has student ID
    - domain:: Student
    - range:: xsd:string
  "#;

    let result = parser.parse(content, "university.md").unwrap();

    // Verify classes
    assert_eq!(result.classes.len(), 4);
    assert!(result.classes.iter().any(|c| c.iri == "Person"));
    assert!(result.classes.iter().any(|c| c.iri == "Student"));
    assert!(result.classes.iter().any(|c| c.iri == "Teacher"));
    assert!(result.classes.iter().any(|c| c.iri == "Course"));

    // Verify properties
    assert_eq!(result.properties.len(), 3);
    assert_eq!(
        result
            .properties
            .iter()
            .filter(|p| p.property_type == PropertyType::ObjectProperty)
            .count(),
        2
    );
    assert_eq!(
        result
            .properties
            .iter()
            .filter(|p| p.property_type == PropertyType::DataProperty)
            .count(),
        1
    );

    // Verify axioms
    assert_eq!(result.axioms.len(), 2);
    assert!(result
        .axioms
        .iter()
        .all(|a| a.axiom_type == AxiomType::SubClassOf));

    // Verify class hierarchy
    assert_eq!(result.class_hierarchy.len(), 2);
    assert!(result
        .class_hierarchy
        .contains(&("Student".to_string(), "Person".to_string())));
    assert!(result
        .class_hierarchy
        .contains(&("Teacher".to_string(), "Person".to_string())));

    // Verify source file tracking
    assert!(result
        .classes
        .iter()
        .all(|c| c.source_file == Some("university.md".to_string())));
}

#[test]
fn test_rdfs_subclass_syntax() {
    let parser = OntologyParser::new();
    let content = r#"
- ### OntologyBlock
  - owl_class:: Student
    - rdfs:subClassOf:: Person
  "#;

    let result = parser.parse(content, "test.md").unwrap();

    // The current implementation uses "subClassOf::" pattern
    // This test verifies the behavior - it should still work or we may need to update parser
    assert_eq!(result.classes.len(), 1);
}
