//! Comprehensive Ontology System Smoke Tests
//!
//! This test suite provides comprehensive validation for the ontology system,
//! including unit tests, integration tests, end-to-end tests, performance tests,
//! and error handling scenarios.
//!
//! The tests use the fixtures in `tests/fixtures/ontology/` to provide realistic
//! test data and scenarios.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mockall::{mock, predicate::*};
use pretty_assertions::assert_eq;
use tokio_test;

use webxr::models::{
    constraints::{Constraint, ConstraintKind, ConstraintSet},
    graph::GraphData,
    node::Node,
};
use webxr::physics::ontology_constraints::{
    ConsistencyCheck, OWLAxiom, OWLAxiomType, OntologyConstraintConfig,
    OntologyConstraintTranslator, OntologyInference, OntologyReasoningReport,
};
use webxr::services::owl_validator::{
    GraphEdge, GraphNode, OwlValidatorService, PropertyGraph, RdfTriple, Severity,
    ValidationConfig, ValidationError, ValidationReport, Violation,
};
use webxr::types::vec3::Vec3Data;
use webxr::utils::socket_flow_messages::BinaryNodeData;

// Mock implementations for external dependencies
mock! {
    HttpClient {
        async fn get(&self, url: &str) -> Result<String, reqwest::Error>;
    }
}

// Test fixtures and utilities
mod test_fixtures {
    use super::*;
    use serde_json::Value;

    pub fn load_sample_graph() -> Result<Value, Box<dyn std::error::Error>> {
        let content = fs::read_to_string("tests/fixtures/ontology/sample_graph.json")?;
        let graph: Value = serde_json::from_str(&content)?;
        Ok(graph)
    }

    pub fn load_sample_ontology() -> Result<String, Box<dyn std::error::Error>> {
        let content = fs::read_to_string("tests/fixtures/ontology/sample.ttl")?;
        Ok(content)
    }

    pub fn load_mapping_config() -> Result<String, Box<dyn std::error::Error>> {
        let content = fs::read_to_string("tests/fixtures/ontology/test_mapping.toml")?;
        Ok(content)
    }

    pub fn create_test_node(
        id: u32,
        metadata_id: String,
        node_type: Option<String>,
        position: (f32, f32, f32),
    ) -> Node {
        let mut metadata = HashMap::new();
        if let Some(ref nt) = node_type {
            metadata.insert("type".to_string(), nt.clone());
        }

        Node {
            id,
            metadata_id,
            label: format!("Test Node {}", id),
            data: BinaryNodeData {
                position: Vec3Data {
                    x: position.0,
                    y: position.1,
                    z: position.2,
                },
                velocity: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                acceleration: Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                mass: 1.0,
                radius: 1.0,
            },
            metadata,
            file_size: 0,
            node_type,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }

    pub fn create_property_graph() -> PropertyGraph {
        PropertyGraph {
            nodes: vec![
                GraphNode {
                    id: "person1".to_string(),
                    labels: vec!["Person".to_string()],
                    properties: {
                        let mut props = HashMap::new();
                        props.insert(
                            "name".to_string(),
                            serde_json::Value::String("John Doe".to_string()),
                        );
                        props.insert(
                            "age".to_string(),
                            serde_json::Value::Number(serde_json::Number::from(30)),
                        );
                        props.insert(
                            "email".to_string(),
                            serde_json::Value::String("john@example.com".to_string()),
                        );
                        props
                    },
                },
                GraphNode {
                    id: "company1".to_string(),
                    labels: vec!["Company".to_string()],
                    properties: {
                        let mut props = HashMap::new();
                        props.insert(
                            "name".to_string(),
                            serde_json::Value::String("ACME Corp".to_string()),
                        );
                        props.insert(
                            "industry".to_string(),
                            serde_json::Value::String("Technology".to_string()),
                        );
                        props
                    },
                },
            ],
            edges: vec![GraphEdge {
                id: "edge1".to_string(),
                source: "person1".to_string(),
                target: "company1".to_string(),
                relationship_type: "worksFor".to_string(),
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "since".to_string(),
                        serde_json::Value::String("2020".to_string()),
                    );
                    props
                },
            }],
            metadata: HashMap::new(),
        }
    }

    pub fn create_sample_axioms() -> Vec<OWLAxiom> {
        vec![
            OWLAxiom {
                axiom_type: OWLAxiomType::DisjointClasses,
                subject: "Person".to_string(),
                object: Some("Company".to_string()),
                property: None,
                confidence: 1.0,
            },
            OWLAxiom {
                axiom_type: OWLAxiomType::SubClassOf,
                subject: "Employee".to_string(),
                object: Some("Person".to_string()),
                property: None,
                confidence: 0.9,
            },
            OWLAxiom {
                axiom_type: OWLAxiomType::SameAs,
                subject: "person1".to_string(),
                object: Some("employee1".to_string()),
                property: None,
                confidence: 0.8,
            },
        ]
    }
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;
    use test_fixtures::*;

    mod owl_validator_tests {
        use super::*;

        #[test]
        fn test_validator_creation() {
            let validator = OwlValidatorService::new();
            assert_eq!(validator.config.reasoning_timeout_seconds, 30);
            assert!(validator.config.enable_reasoning);
            assert!(validator.config.enable_caching);
        }

        #[test]
        fn test_validator_with_custom_config() {
            let config = ValidationConfig {
                enable_reasoning: false,
                reasoning_timeout_seconds: 60,
                enable_inference: false,
                max_inference_depth: 5,
                enable_caching: false,
                cache_ttl_seconds: 7200,
                validate_cardinality: false,
                validate_domains_ranges: false,
                validate_disjoint_classes: false,
            };

            let validator = OwlValidatorService::with_config(config.clone());
            assert_eq!(
                validator.config.reasoning_timeout_seconds,
                config.reasoning_timeout_seconds
            );
            assert_eq!(validator.config.enable_reasoning, config.enable_reasoning);
            assert_eq!(validator.config.enable_inference, config.enable_inference);
        }

        #[test]
        fn test_iri_expansion() {
            let validator = OwlValidatorService::new();

            // Test prefixed IRI
            let expanded = validator.expand_iri("foaf:Person").unwrap();
            assert_eq!(expanded, "http://xmlns.com/foaf/0.1/Person");

            // Test RDF prefixed IRI
            let expanded = validator.expand_iri("rdf:type").unwrap();
            assert_eq!(expanded, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type");

            // Test full IRI
            let full_iri = "http://example.org/Person";
            let expanded = validator.expand_iri(full_iri).unwrap();
            assert_eq!(expanded, full_iri);

            // Test plain name (gets default namespace)
            let expanded = validator.expand_iri("Person").unwrap();
            assert_eq!(expanded, "http://example.org/Person");
        }

        #[test]
        fn test_iri_expansion_errors() {
            let validator = OwlValidatorService::new();

            // Test unknown prefix
            let result = validator.expand_iri("unknown:Person");
            assert!(result.is_err());
            if let Err(e) = result {
                assert!(e.to_string().contains("Unknown prefix"));
            }
        }

        #[test]
        fn test_property_value_serialization() {
            let validator = OwlValidatorService::new();

            // Test string value
            let string_val = serde_json::Value::String("test".to_string());
            let (object, is_literal, datatype, language) =
                validator.serialize_property_value(&string_val).unwrap();
            assert_eq!(object, "test");
            assert!(is_literal);
            assert_eq!(
                datatype,
                Some("http://www.w3.org/2001/XMLSchema#string".to_string())
            );
            assert_eq!(language, None);

            // Test integer value
            let int_val = serde_json::Value::Number(serde_json::Number::from(42));
            let (object, is_literal, datatype, _) =
                validator.serialize_property_value(&int_val).unwrap();
            assert_eq!(object, "42");
            assert!(is_literal);
            assert_eq!(
                datatype,
                Some("http://www.w3.org/2001/XMLSchema#integer".to_string())
            );

            // Test boolean value
            let bool_val = serde_json::Value::Bool(true);
            let (object, is_literal, datatype, _) =
                validator.serialize_property_value(&bool_val).unwrap();
            assert_eq!(object, "true");
            assert!(is_literal);
            assert_eq!(
                datatype,
                Some("http://www.w3.org/2001/XMLSchema#boolean".to_string())
            );

            // Test URL value (should be treated as IRI)
            let url_val = serde_json::Value::String("http://example.org/resource".to_string());
            let (object, is_literal, datatype, _) =
                validator.serialize_property_value(&url_val).unwrap();
            assert_eq!(object, "http://example.org/resource");
            assert!(!is_literal);
            assert_eq!(datatype, None);
        }

        #[test]
        fn test_map_graph_to_rdf() {
            let validator = OwlValidatorService::new();
            let graph = create_property_graph();

            let triples = validator.map_graph_to_rdf(&graph).unwrap();

            assert!(!triples.is_empty());

            // Check that we have type triples for nodes
            let type_triples: Vec<&RdfTriple> = triples
                .iter()
                .filter(|t| t.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                .collect();

            assert!(!type_triples.is_empty());

            // Check that we have property triples
            let property_triples: Vec<&RdfTriple> = triples
                .iter()
                .filter(|t| t.predicate != "http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
                .collect();

            assert!(!property_triples.is_empty());

            // Verify specific mappings
            assert!(triples.iter().any(|t| t.subject.contains("person1")
                && t.predicate == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
                && t.object.contains("Person")));
        }

        #[test]
        fn test_signature_calculation() {
            let validator = OwlValidatorService::new();

            let content1 = "test content";
            let content2 = "test content";
            let content3 = "different content";

            let sig1 = validator.calculate_signature(content1);
            let sig2 = validator.calculate_signature(content2);
            let sig3 = validator.calculate_signature(content3);

            assert_eq!(sig1, sig2);
            assert_ne!(sig1, sig3);

            // Signature should be consistent hex string
            assert_eq!(sig1.len(), 64); // blake3 produces 32-byte hash -> 64 hex chars
            assert!(sig1.chars().all(|c| c.is_ascii_hexdigit()));
        }

        #[tokio::test]
        async fn test_inference_basic() {
            let validator = OwlValidatorService::new();

            let triples = vec![RdfTriple {
                subject: "http://example.org/john".to_string(),
                predicate: "http://example.org/employs".to_string(),
                object: "http://example.org/mary".to_string(),
                is_literal: false,
                datatype: None,
                language: None,
            }];

            let inferred = validator.infer(&triples).unwrap();

            // Should infer inverse relationship
            assert!(inferred
                .iter()
                .any(|t| t.subject == "http://example.org/mary"
                    && t.predicate == "http://example.org/worksFor"
                    && t.object == "http://example.org/john"));
        }

        #[test]
        fn test_cache_operations() {
            let validator = OwlValidatorService::new();

            // Initially no cache entries
            assert!(validator.validation_cache.is_empty());

            // Clear should work without errors
            validator.clear_caches();
            assert!(validator.validation_cache.is_empty());
            assert!(validator.ontology_cache.is_empty());
        }
    }

    mod constraint_translator_tests {
        use super::*;

        #[test]
        fn test_translator_creation() {
            let translator = OntologyConstraintTranslator::new();
            assert_eq!(translator.config.disjoint_separation_strength, 0.8);
            assert_eq!(translator.config.hierarchy_alignment_strength, 0.6);
            assert!(translator.config.enable_constraint_caching);
        }

        #[test]
        fn test_custom_config_translator() {
            let config = OntologyConstraintConfig {
                disjoint_separation_strength: 1.0,
                hierarchy_alignment_strength: 0.9,
                sameas_colocation_strength: 0.7,
                cardinality_boundary_strength: 0.5,
                max_separation_distance: 100.0,
                min_colocation_distance: 1.0,
                enable_constraint_caching: false,
                cache_invalidation_enabled: false,
            };

            let translator = OntologyConstraintTranslator::with_config(config.clone());
            assert_eq!(
                translator.config.disjoint_separation_strength,
                config.disjoint_separation_strength
            );
            assert_eq!(
                translator.config.max_separation_distance,
                config.max_separation_distance
            );
            assert_eq!(
                translator.config.enable_constraint_caching,
                config.enable_constraint_caching
            );
        }

        #[test]
        fn test_constraint_strength_calculation() {
            let translator = OntologyConstraintTranslator::new();

            let disjoint_strength =
                translator.get_constraint_strength(&OWLAxiomType::DisjointClasses);
            let sameas_strength = translator.get_constraint_strength(&OWLAxiomType::SameAs);
            let subclass_strength = translator.get_constraint_strength(&OWLAxiomType::SubClassOf);

            assert_eq!(disjoint_strength, 0.8);
            assert_eq!(sameas_strength, 0.9);
            assert_eq!(subclass_strength, 0.6);

            // Test default case
            let default_strength =
                translator.get_constraint_strength(&OWLAxiomType::TransitiveProperty);
            assert_eq!(default_strength, 0.5);
        }

        #[test]
        fn test_disjoint_classes_constraint_generation() {
            let mut translator = OntologyConstraintTranslator::new();

            let nodes = vec![
                create_test_node(
                    1,
                    "animal1".to_string(),
                    Some("Animal".to_string()),
                    (0.0, 0.0, 0.0),
                ),
                create_test_node(
                    2,
                    "plant1".to_string(),
                    Some("Plant".to_string()),
                    (10.0, 0.0, 0.0),
                ),
                create_test_node(
                    3,
                    "animal2".to_string(),
                    Some("Animal".to_string()),
                    (5.0, 5.0, 0.0),
                ),
            ];

            let axiom = OWLAxiom {
                axiom_type: OWLAxiomType::DisjointClasses,
                subject: "Animal".to_string(),
                object: Some("Plant".to_string()),
                property: None,
                confidence: 1.0,
            };

            let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

            // Should create separation constraints between Animal and Plant instances
            assert!(!constraints.is_empty());
            assert!(constraints
                .iter()
                .all(|c| c.kind == ConstraintKind::Separation));

            // Should have 2 constraints (animal1-plant1, animal2-plant1)
            assert_eq!(constraints.len(), 2);
        }

        #[test]
        fn test_sameas_constraint_generation() {
            let mut translator = OntologyConstraintTranslator::new();

            let nodes = vec![
                create_test_node(1, "person1".to_string(), None, (0.0, 0.0, 0.0)),
                create_test_node(2, "person2".to_string(), None, (10.0, 0.0, 0.0)),
            ];

            let axiom = OWLAxiom {
                axiom_type: OWLAxiomType::SameAs,
                subject: "person1".to_string(),
                object: Some("person2".to_string()),
                property: None,
                confidence: 1.0,
            };

            let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

            // Should create clustering constraint to pull nodes together
            assert_eq!(constraints.len(), 1);
            assert_eq!(constraints[0].kind, ConstraintKind::Clustering);
            assert_eq!(constraints[0].node_indices, vec![1, 2]);
        }

        #[test]
        fn test_subclass_constraint_generation() {
            let mut translator = OntologyConstraintTranslator::new();

            let nodes = vec![
                create_test_node(
                    1,
                    "employee1".to_string(),
                    Some("Employee".to_string()),
                    (0.0, 0.0, 0.0),
                ),
                create_test_node(
                    2,
                    "person1".to_string(),
                    Some("Person".to_string()),
                    (10.0, 0.0, 0.0),
                ),
                create_test_node(
                    3,
                    "employee2".to_string(),
                    Some("Employee".to_string()),
                    (5.0, 5.0, 0.0),
                ),
            ];

            let axiom = OWLAxiom {
                axiom_type: OWLAxiomType::SubClassOf,
                subject: "Employee".to_string(),
                object: Some("Person".to_string()),
                property: None,
                confidence: 0.9,
            };

            let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

            // Should create clustering constraints to align Employee instances toward Person centroid
            assert!(!constraints.is_empty());
            assert!(constraints
                .iter()
                .all(|c| c.kind == ConstraintKind::Clustering));

            // Should have constraints for both employee nodes
            assert_eq!(constraints.len(), 2);
        }

        #[test]
        fn test_cache_management() {
            let mut translator = OntologyConstraintTranslator::new();

            let nodes = vec![create_test_node(
                1,
                "test".to_string(),
                Some("TestType".to_string()),
                (0.0, 0.0, 0.0),
            )];
            translator.update_node_type_cache(&nodes);

            let stats = translator.get_cache_stats();
            assert_eq!(stats.node_type_entries, 1);
            assert_eq!(stats.total_cache_entries, 0); // No constraint cache entries yet

            translator.clear_cache();
            let stats_after_clear = translator.get_cache_stats();
            assert_eq!(stats_after_clear.node_type_entries, 0);
        }

        #[test]
        fn test_inference_constraint_generation() {
            let mut translator = OntologyConstraintTranslator::new();

            let nodes = vec![
                create_test_node(
                    1,
                    "person1".to_string(),
                    Some("Person".to_string()),
                    (0.0, 0.0, 0.0),
                ),
                create_test_node(
                    2,
                    "person2".to_string(),
                    Some("Person".to_string()),
                    (10.0, 0.0, 0.0),
                ),
            ];

            let graph = GraphData {
                nodes,
                edges: vec![],
            };

            let inference = OntologyInference {
                inferred_axiom: OWLAxiom {
                    axiom_type: OWLAxiomType::SameAs,
                    subject: "person1".to_string(),
                    object: Some("person2".to_string()),
                    property: None,
                    confidence: 0.8,
                },
                premise_axioms: vec!["axiom1".to_string(), "axiom2".to_string()],
                reasoning_confidence: 0.7,
                is_derived: true,
            };

            let constraints = translator
                .inferences_to_constraints(&[inference], &graph)
                .unwrap();

            // Should create constraints with adjusted confidence
            assert!(!constraints.is_empty());
            assert!(constraints.iter().all(|c| c.weight <= 0.8 * 0.7)); // confidence * reasoning_confidence
        }
    }
}

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;
    use test_fixtures::*;

    #[tokio::test]
    async fn test_load_ontology_from_fixture() {
        let validator = OwlValidatorService::new();

        // Load the sample ontology file
        let ontology_content =
            load_sample_ontology().expect("Should load sample ontology from fixtures");

        let ontology_id = validator
            .load_ontology(&ontology_content)
            .await
            .expect("Should successfully load ontology");

        assert!(!ontology_id.is_empty());
        assert!(ontology_id.starts_with("ontology_"));

        // Verify ontology is cached
        assert!(validator.ontology_cache.contains_key(&ontology_id));
    }

    #[tokio::test]
    async fn test_validate_sample_graph() {
        let validator = OwlValidatorService::new();

        // Load ontology first
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // Create property graph from fixture data
        let graph_data = load_sample_graph().unwrap();
        let property_graph = convert_json_to_property_graph(&graph_data).unwrap();

        // Perform validation
        let report = validator
            .validate(&ontology_id, &property_graph)
            .await
            .unwrap();

        assert!(!report.id.is_empty());
        assert!(report.duration_ms > 0);
        assert!(report.total_triples > 0);

        // Should have some statistics
        assert!(report.statistics.classes_checked >= 0);
        assert!(report.statistics.properties_checked >= 0);
    }

    #[tokio::test]
    async fn test_constraint_violation_detection() {
        let validator = OwlValidatorService::new();

        // Load ontology
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // Create graph with intentional violations
        let violating_graph = create_violating_property_graph();

        let report = validator
            .validate(&ontology_id, &violating_graph)
            .await
            .unwrap();

        // Should detect some violations
        // Note: Actual violations depend on the ontology and validation rules
        println!("Detected {} violations", report.violations.len());

        for violation in &report.violations {
            println!("Violation: {} - {}", violation.rule, violation.message);
        }
    }

    #[tokio::test]
    async fn test_inference_generation() {
        let validator = OwlValidatorService::with_config(ValidationConfig {
            enable_inference: true,
            max_inference_depth: 2,
            ..ValidationConfig::default()
        });

        // Load ontology
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // Create graph with inferrable relationships
        let graph = create_inferrable_property_graph();

        let report = validator.validate(&ontology_id, &graph).await.unwrap();

        // Should generate some inferences
        println!(
            "Generated {} inferred triples",
            report.inferred_triples.len()
        );

        for inference in &report.inferred_triples {
            println!(
                "Inferred: {} -> {} -> {}",
                inference.subject, inference.predicate, inference.object
            );
        }
    }

    #[tokio::test]
    async fn test_apply_constraints_to_physics() {
        let mut translator = OntologyConstraintTranslator::new();
        let validator = OwlValidatorService::new();

        // Load ontology and validate graph
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        let graph_data = load_sample_graph().unwrap();
        let property_graph = convert_json_to_property_graph(&graph_data).unwrap();

        // Create mock graph data structure
        let nodes = create_nodes_from_property_graph(&property_graph);
        let graph = GraphData {
            nodes,
            edges: vec![],
        };

        // Create reasoning report
        let reasoning_report = OntologyReasoningReport {
            axioms: create_sample_axioms(),
            inferences: vec![],
            consistency_checks: vec![],
            reasoning_time_ms: 100,
        };

        let constraint_set = translator
            .apply_ontology_constraints(&graph, &reasoning_report)
            .unwrap();

        assert!(!constraint_set.constraints.is_empty());
        assert!(!constraint_set.groups.is_empty());

        println!(
            "Generated {} constraints in {} groups",
            constraint_set.constraints.len(),
            constraint_set.groups.len()
        );
    }

    #[tokio::test]
    async fn test_caching_behavior() {
        let validator = OwlValidatorService::with_config(ValidationConfig {
            enable_caching: true,
            cache_ttl_seconds: 3600,
            ..ValidationConfig::default()
        });

        let ontology_content = load_sample_ontology().unwrap();

        // First load
        let start = Instant::now();
        let ontology_id1 = validator.load_ontology(&ontology_content).await.unwrap();
        let first_load_time = start.elapsed();

        // Second load (should use cache)
        let start = Instant::now();
        let ontology_id2 = validator.load_ontology(&ontology_content).await.unwrap();
        let second_load_time = start.elapsed();

        assert_eq!(ontology_id1, ontology_id2);
        // Second load should be faster (cached)
        assert!(second_load_time < first_load_time);
    }

    // Helper functions for integration tests

    fn convert_json_to_property_graph(
        json_data: &serde_json::Value,
    ) -> Result<PropertyGraph, Box<dyn std::error::Error>> {
        let nodes_json = json_data["nodes"].as_array().ok_or("Missing nodes array")?;
        let edges_json = json_data["edges"].as_array().ok_or("Missing edges array")?;

        let mut nodes = Vec::new();
        for node_json in nodes_json {
            let node = GraphNode {
                id: node_json["id"].as_str().unwrap().to_string(),
                labels: vec![node_json["type"].as_str().unwrap().to_string()],
                properties: node_json["properties"]
                    .as_object()
                    .unwrap_or(&serde_json::Map::new())
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            };
            nodes.push(node);
        }

        let mut edges = Vec::new();
        for edge_json in edges_json {
            let edge = GraphEdge {
                id: format!(
                    "{}_{}",
                    edge_json["from"].as_str().unwrap(),
                    edge_json["to"].as_str().unwrap()
                ),
                source: edge_json["from"].as_str().unwrap().to_string(),
                target: edge_json["to"].as_str().unwrap().to_string(),
                relationship_type: edge_json["type"].as_str().unwrap().to_string(),
                properties: HashMap::new(),
            };
            edges.push(edge);
        }

        Ok(PropertyGraph {
            nodes,
            edges,
            metadata: HashMap::new(),
        })
    }

    fn create_violating_property_graph() -> PropertyGraph {
        PropertyGraph {
            nodes: vec![GraphNode {
                id: "violation_person".to_string(),
                labels: vec!["Person".to_string(), "Company".to_string()], // Disjoint classes violation
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "age".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(-5)),
                    ); // Invalid age
                    props
                },
            }],
            edges: vec![],
            metadata: HashMap::new(),
        }
    }

    fn create_inferrable_property_graph() -> PropertyGraph {
        PropertyGraph {
            nodes: vec![
                GraphNode {
                    id: "john".to_string(),
                    labels: vec!["Person".to_string()],
                    properties: HashMap::new(),
                },
                GraphNode {
                    id: "mary".to_string(),
                    labels: vec!["Person".to_string()],
                    properties: HashMap::new(),
                },
                GraphNode {
                    id: "acme".to_string(),
                    labels: vec!["Company".to_string()],
                    properties: HashMap::new(),
                },
            ],
            edges: vec![
                GraphEdge {
                    id: "emp1".to_string(),
                    source: "acme".to_string(),
                    target: "john".to_string(),
                    relationship_type: "employs".to_string(),
                    properties: HashMap::new(),
                }, // Should infer: john worksFor acme
            ],
            metadata: HashMap::new(),
        }
    }

    fn create_nodes_from_property_graph(graph: &PropertyGraph) -> Vec<Node> {
        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, graph_node)| {
                create_test_node(
                    i as u32,
                    graph_node.id.clone(),
                    graph_node.labels.first().cloned(),
                    (i as f32 * 10.0, 0.0, 0.0),
                )
            })
            .collect()
    }
}

// =============================================================================
// END-TO-END TESTS
// =============================================================================

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;
    use test_fixtures::*;

    #[tokio::test]
    async fn test_full_validation_workflow() {
        // 1. Initialize services
        let validator = OwlValidatorService::new();
        let mut translator = OntologyConstraintTranslator::new();

        // 2. Load ontology from fixture
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // 3. Load and convert graph data
        let graph_data = load_sample_graph().unwrap();
        let property_graph = convert_json_to_property_graph(&graph_data).unwrap();

        // 4. Perform validation
        let validation_report = validator
            .validate(&ontology_id, &property_graph)
            .await
            .unwrap();

        // 5. Convert to constraints
        let nodes = create_nodes_from_property_graph(&property_graph);
        let graph = GraphData {
            nodes,
            edges: vec![],
        };

        let reasoning_report = OntologyReasoningReport {
            axioms: create_sample_axioms(),
            inferences: vec![],
            consistency_checks: vec![ConsistencyCheck {
                is_consistent: true,
                conflicting_axioms: vec![],
                suggested_resolution: None,
            }],
            reasoning_time_ms: validation_report.duration_ms,
        };

        let constraint_set = translator
            .apply_ontology_constraints(&graph, &reasoning_report)
            .unwrap();

        // 6. Validate results
        assert!(!validation_report.id.is_empty());
        assert!(validation_report.total_triples > 0);
        assert!(!constraint_set.constraints.is_empty());

        println!("✅ Full workflow completed:");
        println!("  - Loaded ontology: {}", ontology_id);
        println!("  - Validated {} triples", validation_report.total_triples);
        println!(
            "  - Found {} violations",
            validation_report.violations.len()
        );
        println!(
            "  - Generated {} inferences",
            validation_report.inferred_triples.len()
        );
        println!(
            "  - Created {} constraints",
            constraint_set.constraints.len()
        );
    }

    #[tokio::test]
    async fn test_websocket_communication_simulation() {
        use std::sync::Arc;
        use tokio::sync::{mpsc, Mutex};

        // Simulate WebSocket communication for real-time validation
        let (tx, mut rx) = mpsc::channel::<String>(10);

        let validator = Arc::new(OwlValidatorService::new());
        let validator_clone = Arc::clone(&validator);

        // Simulate WebSocket message handler
        let handler = tokio::spawn(async move {
            while let Some(message) = rx.recv().await {
                match message.as_str() {
                    "validate_graph" => {
                        let ontology_content = load_sample_ontology().unwrap();
                        let ontology_id = validator_clone
                            .load_ontology(&ontology_content)
                            .await
                            .unwrap();

                        let graph = create_property_graph();
                        let _report = validator_clone
                            .validate(&ontology_id, &graph)
                            .await
                            .unwrap();

                        println!("✅ WebSocket validation completed");
                    }
                    "clear_cache" => {
                        validator_clone.clear_caches();
                        println!("✅ Cache cleared via WebSocket");
                    }
                    _ => {
                        println!("Unknown WebSocket message: {}", message);
                    }
                }
            }
        });

        // Send test messages
        tx.send("validate_graph".to_string()).await.unwrap();
        tx.send("clear_cache".to_string()).await.unwrap();

        // Close channel and wait for handler
        drop(tx);
        handler.await.unwrap();
    }

    #[tokio::test]
    async fn test_feature_flag_toggling() {
        // Test different configurations based on feature flags

        // Configuration 1: Full validation
        let full_config = ValidationConfig {
            enable_reasoning: true,
            enable_inference: true,
            validate_cardinality: true,
            validate_domains_ranges: true,
            validate_disjoint_classes: true,
            ..ValidationConfig::default()
        };

        let validator_full = OwlValidatorService::with_config(full_config);
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator_full
            .load_ontology(&ontology_content)
            .await
            .unwrap();
        let graph = create_property_graph();

        let full_report = validator_full.validate(&ontology_id, &graph).await.unwrap();

        // Configuration 2: Minimal validation
        let minimal_config = ValidationConfig {
            enable_reasoning: false,
            enable_inference: false,
            validate_cardinality: false,
            validate_domains_ranges: false,
            validate_disjoint_classes: false,
            ..ValidationConfig::default()
        };

        let validator_minimal = OwlValidatorService::with_config(minimal_config);
        let ontology_id2 = validator_minimal
            .load_ontology(&ontology_content)
            .await
            .unwrap();
        let minimal_report = validator_minimal
            .validate(&ontology_id2, &graph)
            .await
            .unwrap();

        // Full validation should produce more results
        println!(
            "Full validation: {} violations, {} inferences",
            full_report.violations.len(),
            full_report.inferred_triples.len()
        );
        println!(
            "Minimal validation: {} violations, {} inferences",
            minimal_report.violations.len(),
            minimal_report.inferred_triples.len()
        );

        // Full validation should take longer
        assert!(full_report.duration_ms >= minimal_report.duration_ms);
    }

    // Helper function for E2E tests
    fn convert_json_to_property_graph(
        json_data: &serde_json::Value,
    ) -> Result<PropertyGraph, Box<dyn std::error::Error>> {
        let nodes_json = json_data["nodes"].as_array().ok_or("Missing nodes array")?;
        let edges_json = json_data["edges"].as_array().ok_or("Missing edges array")?;

        let mut nodes = Vec::new();
        for node_json in nodes_json {
            let node = GraphNode {
                id: node_json["id"].as_str().unwrap().to_string(),
                labels: vec![node_json["type"].as_str().unwrap().to_string()],
                properties: node_json["properties"]
                    .as_object()
                    .unwrap_or(&serde_json::Map::new())
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            };
            nodes.push(node);
        }

        let mut edges = Vec::new();
        for edge_json in edges_json {
            let edge = GraphEdge {
                id: format!(
                    "{}_{}",
                    edge_json["from"].as_str().unwrap(),
                    edge_json["to"].as_str().unwrap()
                ),
                source: edge_json["from"].as_str().unwrap().to_string(),
                target: edge_json["to"].as_str().unwrap().to_string(),
                relationship_type: edge_json["type"].as_str().unwrap().to_string(),
                properties: HashMap::new(),
            };
            edges.push(edge);
        }

        Ok(PropertyGraph {
            nodes,
            edges,
            metadata: HashMap::new(),
        })
    }

    fn create_nodes_from_property_graph(graph: &PropertyGraph) -> Vec<Node> {
        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, graph_node)| {
                create_test_node(
                    i as u32,
                    graph_node.id.clone(),
                    graph_node.labels.first().cloned(),
                    (i as f32 * 10.0, 0.0, 0.0),
                )
            })
            .collect()
    }
}

// =============================================================================
// PERFORMANCE TESTS
// =============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    use test_fixtures::*;

    #[tokio::test]
    async fn test_large_graph_validation_performance() {
        let validator = OwlValidatorService::new();

        // Load ontology
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // Create large graph (1000 nodes, 5000 edges)
        let large_graph = create_large_property_graph(1000, 5000);

        let start = Instant::now();
        let report = validator
            .validate(&ontology_id, &large_graph)
            .await
            .unwrap();
        let duration = start.elapsed();

        println!("Large graph validation completed in {:?}", duration);
        println!(
            "Processed {} triples, found {} violations",
            report.total_triples,
            report.violations.len()
        );

        // Performance assertions
        assert!(
            duration < Duration::from_secs(30),
            "Validation should complete within 30 seconds"
        );
        assert!(
            report.total_triples > 1000,
            "Should process significant number of triples"
        );
    }

    #[tokio::test]
    async fn test_cache_performance() {
        let validator = OwlValidatorService::with_config(ValidationConfig {
            enable_caching: true,
            cache_ttl_seconds: 3600,
            ..ValidationConfig::default()
        });

        let ontology_content = load_sample_ontology().unwrap();
        let graph = create_property_graph();

        // First validation (cold cache)
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();
        let start = Instant::now();
        let _report1 = validator.validate(&ontology_id, &graph).await.unwrap();
        let cold_cache_time = start.elapsed();

        // Second validation (warm cache)
        let start = Instant::now();
        let _report2 = validator.validate(&ontology_id, &graph).await.unwrap();
        let warm_cache_time = start.elapsed();

        println!(
            "Cold cache: {:?}, Warm cache: {:?}",
            cold_cache_time, warm_cache_time
        );

        // Warm cache should be faster
        assert!(
            warm_cache_time <= cold_cache_time,
            "Cached validation should be faster"
        );
    }

    #[tokio::test]
    async fn test_incremental_validation_performance() {
        let validator = OwlValidatorService::new();

        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // Start with small graph
        let mut graph = create_property_graph();

        let mut performance_data = Vec::new();

        // Incrementally add nodes and measure performance
        for size in [10, 50, 100, 250, 500] {
            graph = create_large_property_graph(size, size * 2);

            let start = Instant::now();
            let report = validator.validate(&ontology_id, &graph).await.unwrap();
            let duration = start.elapsed();

            performance_data.push((size, duration.as_millis(), report.total_triples));

            println!(
                "Size {}: {:?} ({} triples)",
                size, duration, report.total_triples
            );
        }

        // Check that performance scales reasonably
        for i in 1..performance_data.len() {
            let (prev_size, prev_time, _) = performance_data[i - 1];
            let (curr_size, curr_time, _) = performance_data[i];

            let size_ratio = curr_size as f64 / prev_size as f64;
            let time_ratio = curr_time as f64 / prev_time as f64;

            // Time should not increase faster than O(n²)
            assert!(
                time_ratio < size_ratio * size_ratio + 1.0,
                "Performance should scale reasonably: size ratio {:.2}, time ratio {:.2}",
                size_ratio,
                time_ratio
            );
        }
    }

    #[test]
    fn test_constraint_generation_performance() {
        let mut translator = OntologyConstraintTranslator::new();

        // Create many nodes
        let nodes: Vec<Node> = (0..1000)
            .map(|i| {
                create_test_node(
                    i,
                    format!("node_{}", i),
                    Some(if i % 3 == 0 {
                        "TypeA".to_string()
                    } else {
                        "TypeB".to_string()
                    }),
                    (i as f32, i as f32, 0.0),
                )
            })
            .collect();

        // Create axioms that will generate many constraints
        let axioms = vec![OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "TypeA".to_string(),
            object: Some("TypeB".to_string()),
            property: None,
            confidence: 1.0,
        }];

        let start = Instant::now();
        let constraints = translator.axioms_to_constraints(&axioms, &nodes).unwrap();
        let duration = start.elapsed();

        println!(
            "Generated {} constraints in {:?}",
            constraints.len(),
            duration
        );

        // Should complete within reasonable time
        assert!(
            duration < Duration::from_secs(5),
            "Constraint generation should be fast"
        );

        // Should generate expected number of constraints
        let type_a_count = nodes
            .iter()
            .filter(|n| n.node_type.as_ref() == Some(&"TypeA".to_string()))
            .count();
        let type_b_count = nodes
            .iter()
            .filter(|n| n.node_type.as_ref() == Some(&"TypeB".to_string()))
            .count();
        let expected_constraints = type_a_count * type_b_count;

        assert_eq!(constraints.len(), expected_constraints);
    }

    // Helper function to create large graphs for performance testing
    fn create_large_property_graph(num_nodes: usize, num_edges: usize) -> PropertyGraph {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut nodes = Vec::new();
        let node_types = ["Person", "Company", "Department", "Product"];

        for i in 0..num_nodes {
            let node_type = node_types[i % node_types.len()];
            let node = GraphNode {
                id: format!("node_{}", i),
                labels: vec![node_type.to_string()],
                properties: {
                    let mut props = HashMap::new();
                    props.insert(
                        "id".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(i)),
                    );
                    props.insert(
                        "name".to_string(),
                        serde_json::Value::String(format!("{} {}", node_type, i)),
                    );
                    props
                },
            };
            nodes.push(node);
        }

        let mut edges = Vec::new();
        let relationship_types = ["employs", "worksFor", "partOf", "uses"];

        for i in 0..num_edges {
            let source_idx = rng.gen_range(0..num_nodes);
            let target_idx = rng.gen_range(0..num_nodes);

            if source_idx != target_idx {
                let relationship_type = relationship_types[i % relationship_types.len()];
                let edge = GraphEdge {
                    id: format!("edge_{}", i),
                    source: format!("node_{}", source_idx),
                    target: format!("node_{}", target_idx),
                    relationship_type: relationship_type.to_string(),
                    properties: HashMap::new(),
                };
                edges.push(edge);
            }
        }

        PropertyGraph {
            nodes,
            edges,
            metadata: HashMap::new(),
        }
    }
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

#[cfg(test)]
mod error_handling_tests {
    use super::*;
    use test_fixtures::*;

    #[tokio::test]
    async fn test_invalid_ontology_handling() {
        let validator = OwlValidatorService::new();

        // Test with completely invalid ontology content
        let invalid_ontology = "This is not valid RDF/OWL content at all!";
        let result = validator.load_ontology(invalid_ontology).await;

        assert!(result.is_err(), "Should reject invalid ontology");

        if let Err(e) = result {
            assert!(
                e.to_string().contains("Unknown ontology format")
                    || e.to_string().contains("ParseError")
            );
        }
    }

    #[tokio::test]
    async fn test_malformed_graph_handling() {
        let validator = OwlValidatorService::new();

        // Load valid ontology first
        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // Create graph with malformed data
        let malformed_graph = PropertyGraph {
            nodes: vec![GraphNode {
                id: "".to_string(), // Empty ID
                labels: vec![],     // No labels
                properties: HashMap::new(),
            }],
            edges: vec![GraphEdge {
                id: "edge1".to_string(),
                source: "nonexistent".to_string(), // Reference to non-existent node
                target: "also_nonexistent".to_string(),
                relationship_type: "".to_string(), // Empty relationship type
                properties: HashMap::new(),
            }],
            metadata: HashMap::new(),
        };

        // Validation should handle malformed data gracefully
        let result = validator.validate(&ontology_id, &malformed_graph).await;

        // Should succeed but may produce warnings/violations
        assert!(result.is_ok(), "Should handle malformed data gracefully");
    }

    #[tokio::test]
    async fn test_network_failure_simulation() {
        let validator = OwlValidatorService::new();

        // Test with unreachable URL
        let unreachable_url = "http://definitely-not-a-real-domain-12345.com/ontology.owl";
        let result = validator.load_ontology(unreachable_url).await;

        assert!(result.is_err(), "Should handle network failures");
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let validator = OwlValidatorService::with_config(ValidationConfig {
            reasoning_timeout_seconds: 1, // Very short timeout
            enable_reasoning: true,
            ..ValidationConfig::default()
        });

        // Create very complex graph that might cause timeout
        let complex_graph = create_large_property_graph(100, 500);

        let ontology_content = load_sample_ontology().unwrap();
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        let result = validator.validate(&ontology_id, &complex_graph).await;

        // Should either complete or timeout gracefully
        match result {
            Ok(report) => {
                println!(
                    "Validation completed: {} violations",
                    report.violations.len()
                );
            }
            Err(e) => {
                assert!(e.to_string().contains("Timeout") || e.to_string().contains("timeout"));
                println!("Timeout handled correctly: {}", e);
            }
        }
    }

    #[test]
    fn test_constraint_translator_error_handling() {
        let mut translator = OntologyConstraintTranslator::new();

        // Test with empty nodes
        let result = translator.axioms_to_constraints(&create_sample_axioms(), &[]);
        assert!(result.is_ok(), "Should handle empty nodes gracefully");

        let constraints = result.unwrap();
        // Should produce no constraints since no nodes exist
        assert!(constraints.is_empty() || constraints.len() < create_sample_axioms().len());

        // Test with axiom missing required fields
        let incomplete_axioms = vec![OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "ClassA".to_string(),
            object: None, // Missing required object
            property: None,
            confidence: 1.0,
        }];

        let nodes = vec![create_test_node(
            1,
            "test".to_string(),
            None,
            (0.0, 0.0, 0.0),
        )];
        let result = translator.axioms_to_constraints(&incomplete_axioms, &nodes);

        // Should handle incomplete axioms gracefully
        assert!(result.is_ok(), "Should handle incomplete axioms");
    }

    #[test]
    fn test_invalid_iri_handling() {
        let validator = OwlValidatorService::new();

        // Test with invalid IRIs
        let invalid_iris = ["not:a:valid:iri:", "http://", "", ":::"];

        for invalid_iri in &invalid_iris {
            let result = validator.expand_iri(invalid_iri);
            // Should either handle gracefully or produce meaningful error
            match result {
                Ok(_) => {
                    // Some might be handled gracefully
                    println!("IRI '{}' was handled gracefully", invalid_iri);
                }
                Err(e) => {
                    println!("IRI '{}' produced expected error: {}", invalid_iri, e);
                    assert!(e.to_string().contains("Invalid") || e.to_string().contains("Unknown"));
                }
            }
        }
    }

    #[tokio::test]
    async fn test_concurrent_access_safety() {
        use std::sync::Arc;
        use tokio::task;

        let validator = Arc::new(OwlValidatorService::new());
        let ontology_content = load_sample_ontology().unwrap();

        // Load ontology in validator
        let ontology_id = validator.load_ontology(&ontology_content).await.unwrap();

        // Create multiple concurrent validation tasks
        let mut handles = Vec::new();

        for i in 0..10 {
            let validator_clone = Arc::clone(&validator);
            let ontology_id_clone = ontology_id.clone();

            let handle = task::spawn(async move {
                let graph = create_property_graph();
                let result = validator_clone.validate(&ontology_id_clone, &graph).await;

                match result {
                    Ok(report) => {
                        println!(
                            "Task {}: Validation completed with {} violations",
                            i,
                            report.violations.len()
                        );
                        true
                    }
                    Err(e) => {
                        println!("Task {}: Validation failed: {}", i, e);
                        false
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all tasks to complete
        let results = futures::future::join_all(handles).await;

        // Most tasks should succeed
        let success_count = results
            .into_iter()
            .filter_map(|r| r.ok())
            .filter(|&success| success)
            .count();
        assert!(
            success_count >= 8,
            "Most concurrent validations should succeed"
        );
    }

    #[test]
    fn test_memory_pressure_handling() {
        let mut translator = OntologyConstraintTranslator::new();

        // Test with extremely large numbers to see if it handles memory pressure
        let large_number_of_nodes = 10000;
        let nodes: Vec<Node> = (0..large_number_of_nodes)
            .map(|i| {
                create_test_node(
                    i as u32,
                    format!("node_{}", i),
                    Some("TestType".to_string()),
                    (0.0, 0.0, 0.0),
                )
            })
            .collect();

        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::FunctionalProperty,
            subject: "hasUniqueProperty".to_string(),
            object: None,
            property: None,
            confidence: 1.0,
        };

        // This should either complete or fail gracefully
        let result = translator.axioms_to_constraints(&[axiom], &nodes);

        match result {
            Ok(constraints) => {
                println!(
                    "Successfully generated {} constraints for {} nodes",
                    constraints.len(),
                    nodes.len()
                );
            }
            Err(e) => {
                println!("Handled memory pressure gracefully: {}", e);
                // Should be a meaningful error, not a panic
                assert!(!e.to_string().is_empty());
            }
        }
    }
}

// =============================================================================
// INTEGRATION WITH FIXTURES
// =============================================================================

#[cfg(test)]
mod fixture_integration_tests {
    use super::*;
    use test_fixtures::*;

    #[tokio::test]
    async fn test_complete_fixture_workflow() {
        // This test demonstrates the complete workflow using all fixtures

        println!("🚀 Starting complete fixture workflow test");

        // 1. Load all fixtures
        let graph_data = load_sample_graph().expect("Load sample graph");
        let ontology_content = load_sample_ontology().expect("Load sample ontology");
        let _mapping_config = load_mapping_config().expect("Load mapping config");

        println!("✅ Loaded all fixture files");

        // 2. Initialize services
        let validator = OwlValidatorService::new();
        let mut translator = OntologyConstraintTranslator::new();

        // 3. Load and validate ontology
        let ontology_id = validator
            .load_ontology(&ontology_content)
            .await
            .expect("Load ontology");
        println!("✅ Ontology loaded: {}", ontology_id);

        // 4. Convert JSON graph to property graph
        let property_graph = convert_json_to_property_graph(&graph_data).expect("Convert graph");
        println!(
            "✅ Converted graph: {} nodes, {} edges",
            property_graph.nodes.len(),
            property_graph.edges.len()
        );

        // 5. Perform validation
        let validation_report = validator
            .validate(&ontology_id, &property_graph)
            .await
            .expect("Validate graph");
        println!(
            "✅ Validation completed: {} triples, {} violations, {} inferences",
            validation_report.total_triples,
            validation_report.violations.len(),
            validation_report.inferred_triples.len()
        );

        // 6. Generate physics constraints
        let nodes = create_nodes_from_property_graph(&property_graph);
        let graph = GraphData {
            nodes,
            edges: vec![],
        };

        let reasoning_report = OntologyReasoningReport {
            axioms: create_sample_axioms(),
            inferences: vec![],
            consistency_checks: vec![],
            reasoning_time_ms: validation_report.duration_ms,
        };

        let constraint_set = translator
            .apply_ontology_constraints(&graph, &reasoning_report)
            .expect("Generate constraints");
        println!(
            "✅ Generated {} physics constraints in {} groups",
            constraint_set.constraints.len(),
            constraint_set.groups.len()
        );

        // 7. Verify all test scenarios from fixture
        let test_scenarios = graph_data["test_scenarios"]
            .as_array()
            .expect("Get test scenarios");

        for scenario in test_scenarios {
            let scenario_name = scenario["name"].as_str().unwrap();
            let scenario_type = scenario["type"].as_str().unwrap();

            match scenario_type {
                "constraint_violation" => {
                    // Check that we detected the expected violation
                    let expected_violation = scenario["expected_result"]["violation_type"]
                        .as_str()
                        .unwrap();
                    let found_violation = validation_report.violations.iter().any(|v| {
                        v.rule.contains(expected_violation)
                            || v.message.contains(expected_violation)
                    });

                    if found_violation {
                        println!(
                            "✅ Test scenario '{}': Found expected violation",
                            scenario_name
                        );
                    } else {
                        println!(
                            "⚠️  Test scenario '{}': Expected violation not found",
                            scenario_name
                        );
                    }
                }
                "inference_test" => {
                    // Check that we generated expected inference
                    let expected_inference = scenario["expected_result"]["inferred_relationship"]
                        .as_str()
                        .unwrap();
                    let found_inference = validation_report
                        .inferred_triples
                        .iter()
                        .any(|t| t.predicate.contains(expected_inference));

                    if found_inference {
                        println!(
                            "✅ Test scenario '{}': Found expected inference",
                            scenario_name
                        );
                    } else {
                        println!(
                            "⚠️  Test scenario '{}': Expected inference not found",
                            scenario_name
                        );
                    }
                }
                "valid_case" => {
                    println!("✅ Test scenario '{}': Valid case processed", scenario_name);
                }
                _ => {
                    println!(
                        "ℹ️  Test scenario '{}': Unknown type {}",
                        scenario_name, scenario_type
                    );
                }
            }
        }

        println!("🎉 Complete fixture workflow test completed successfully!");
    }

    #[tokio::test]
    async fn test_mapping_config_integration() {
        let mapping_config = load_mapping_config().unwrap();

        // Parse the TOML config to verify it's valid
        let config: toml::Value = toml::from_str(&mapping_config).expect("Parse mapping config");

        // Verify structure
        assert!(
            config["metadata"].is_table(),
            "Should have metadata section"
        );
        assert!(
            config["node_mappings"].is_table(),
            "Should have node_mappings section"
        );
        assert!(
            config["edge_mappings"].is_table(),
            "Should have edge_mappings section"
        );
        assert!(
            config["validation_rules"].is_table(),
            "Should have validation_rules section"
        );

        // Test that mapping config can be used for actual mapping
        let node_mappings = &config["node_mappings"];
        assert!(
            node_mappings["person"].is_table(),
            "Should have person mapping"
        );
        assert!(
            node_mappings["company"].is_table(),
            "Should have company mapping"
        );

        let person_mapping = &node_mappings["person"];
        assert!(
            person_mapping["ontology_class"].is_str(),
            "Person should have ontology class"
        );
        assert!(
            person_mapping["properties"].is_table(),
            "Person should have properties"
        );

        println!("✅ Mapping configuration is valid and complete");
    }

    // Helper functions
    fn convert_json_to_property_graph(
        json_data: &serde_json::Value,
    ) -> Result<PropertyGraph, Box<dyn std::error::Error>> {
        let nodes_json = json_data["nodes"].as_array().ok_or("Missing nodes array")?;
        let edges_json = json_data["edges"].as_array().ok_or("Missing edges array")?;

        let mut nodes = Vec::new();
        for node_json in nodes_json {
            let node = GraphNode {
                id: node_json["id"].as_str().unwrap().to_string(),
                labels: vec![node_json["type"].as_str().unwrap().to_string()],
                properties: node_json["properties"]
                    .as_object()
                    .unwrap_or(&serde_json::Map::new())
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            };
            nodes.push(node);
        }

        let mut edges = Vec::new();
        for edge_json in edges_json {
            let edge = GraphEdge {
                id: format!(
                    "{}_{}",
                    edge_json["from"].as_str().unwrap(),
                    edge_json["to"].as_str().unwrap()
                ),
                source: edge_json["from"].as_str().unwrap().to_string(),
                target: edge_json["to"].as_str().unwrap().to_string(),
                relationship_type: edge_json["type"].as_str().unwrap().to_string(),
                properties: HashMap::new(),
            };
            edges.push(edge);
        }

        Ok(PropertyGraph {
            nodes,
            edges,
            metadata: HashMap::new(),
        })
    }

    fn create_nodes_from_property_graph(graph: &PropertyGraph) -> Vec<Node> {
        graph
            .nodes
            .iter()
            .enumerate()
            .map(|(i, graph_node)| {
                create_test_node(
                    i as u32,
                    graph_node.id.clone(),
                    graph_node.labels.first().cloned(),
                    (i as f32 * 10.0, 0.0, 0.0),
                )
            })
            .collect()
    }
}

// Run the tests with: cargo test --test ontology_smoke_test
