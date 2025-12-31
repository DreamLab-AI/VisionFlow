//! GPU kernel tests for ontology constraint translation
//!
//! Tests cover:
//! - Each of the 5 constraint kernels (separation, alignment, clustering, boundary, identity)
//! - Multi-graph support
//! - Memory alignment
//! - Performance benchmarks
//!
//! NOTE: These tests are disabled because:
//! 1. physics::ontology_constraints module doesn't exist
//! 2. OWLAxiom, OntologyConstraintTranslator types not implemented
//! 3. BinaryNodeData::default() doesn't exist
//!
//! To re-enable:
//! 1. Implement physics::ontology_constraints module
//! 2. Add Default to BinaryNodeData
//! 3. Uncomment the code below

/*
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    use webxr::physics::ontology_constraints::{
        ConsistencyCheck, OWLAxiom, OWLAxiomType, OntologyConstraintConfig,
        OntologyConstraintTranslator, OntologyInference, OntologyReasoningReport,
    };

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    use webxr::models::{
        constraints::{Constraint, ConstraintKind, ConstraintSet},
        graph::GraphData,
        node::Node,
    };

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    use webxr::utils::socket_flow_messages::BinaryNodeData;
    #[cfg(all(feature = "ontology", feature = "gpu"))]
    use webxr::models::metadata::MetadataStore;

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    fn create_test_node(id: u32, metadata_id: String, node_type: Option<String>) -> Node {
        Node {
            id,
            metadata_id: metadata_id.clone(),
            label: format!("Test Node {}", id),
            data: BinaryNodeData {
                node_id: id,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            },
            x: Some(0.0),
            y: Some(0.0),
            z: Some(0.0),
            vx: Some(0.0),
            vy: Some(0.0),
            vz: Some(0.0),
            mass: Some(1.0),
            owl_class_iri: None,
            metadata: HashMap::new(),
            file_size: 0,
            node_type,
            size: None,
            color: None,
            weight: None,
            group: None,
            user_data: None,
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    fn create_test_graph(nodes: Vec<Node>) -> GraphData {
        GraphData {
            nodes,
            edges: vec![],
            metadata: MetadataStore::new(),
            id_to_metadata: HashMap::new(),
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_separation_constraint_kernel() {
        let mut translator = OntologyConstraintTranslator::new();

        // Create nodes of disjoint types
        let nodes = vec![
            create_test_node(1, "person1".to_string(), Some("Person".to_string())),
            create_test_node(2, "person2".to_string(), Some("Person".to_string())),
            create_test_node(3, "company1".to_string(), Some("Company".to_string())),
            create_test_node(4, "company2".to_string(), Some("Company".to_string())),
        ];

        // Create disjoint classes axiom
        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "Person".to_string(),
            object: Some("Company".to_string()),
            property: None,
            confidence: 1.0,
        };

        let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

        // Should create separation constraints between all Person-Company pairs
        // 2 persons * 2 companies = 4 separation constraints
        assert_eq!(
            constraints.len(),
            4,
            "Should create 4 separation constraints"
        );

        for constraint in &constraints {
            assert_eq!(constraint.kind, ConstraintKind::Separation);
            assert_eq!(constraint.node_indices.len(), 2);
            assert!(constraint.weight > 0.0);
            assert!(constraint.active);
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_alignment_constraint_kernel() {
        let mut translator = OntologyConstraintTranslator::new();

        // Create class hierarchy: Employee subClassOf Person
        let nodes = vec![
            create_test_node(1, "person1".to_string(), Some("Person".to_string())),
            create_test_node(2, "person2".to_string(), Some("Person".to_string())),
            create_test_node(3, "employee1".to_string(), Some("Employee".to_string())),
            create_test_node(4, "employee2".to_string(), Some("Employee".to_string())),
        ];

        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::SubClassOf,
            subject: "Employee".to_string(),
            object: Some("Person".to_string()),
            property: None,
            confidence: 1.0,
        };

        let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

        // Should create clustering constraints for employees toward person centroid
        assert!(
            !constraints.is_empty(),
            "Should create alignment constraints"
        );

        for constraint in &constraints {
            assert_eq!(constraint.kind, ConstraintKind::Clustering);
            assert!(constraint.weight > 0.0);
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_clustering_constraint_kernel() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes = vec![
            create_test_node(1, "entity1".to_string(), None),
            create_test_node(2, "entity2".to_string(), None),
        ];

        // SameAs axiom creates clustering constraint
        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::SameAs,
            subject: "entity1".to_string(),
            object: Some("entity2".to_string()),
            property: None,
            confidence: 1.0,
        };

        let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].kind, ConstraintKind::Clustering);
        assert_eq!(constraints[0].node_indices.len(), 2);
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_boundary_constraint_kernel() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes = vec![
            create_test_node(1, "person1".to_string(), Some("Person".to_string())),
            create_test_node(2, "person2".to_string(), Some("Person".to_string())),
        ];

        // Functional property creates boundary constraints
        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::FunctionalProperty,
            subject: "hasSSN".to_string(),
            object: None,
            property: None,
            confidence: 1.0,
        };

        let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

        // Functional properties may create boundary constraints
        for constraint in &constraints {
            if constraint.kind == ConstraintKind::Boundary {
                assert_eq!(constraint.params.len(), 6); // x_min, x_max, y_min, y_max, z_min, z_max
            }
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_identity_constraint_kernel() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes = vec![
            create_test_node(1, "entity1".to_string(), None),
            create_test_node(2, "entity2".to_string(), None),
        ];

        // SameAs creates identity/co-location constraints
        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::SameAs,
            subject: "entity1".to_string(),
            object: Some("entity2".to_string()),
            property: None,
            confidence: 1.0,
        };

        let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

        assert_eq!(constraints.len(), 1);
        // SameAs creates strong clustering constraint (identity)
        assert!(constraints[0].weight > 0.8);
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_multi_graph_support() {
        let mut translator = OntologyConstraintTranslator::new();

        // Create multiple disconnected node groups
        let nodes = vec![
            create_test_node(1, "graph1_person1".to_string(), Some("Person".to_string())),
            create_test_node(2, "graph1_person2".to_string(), Some("Person".to_string())),
            create_test_node(
                3,
                "graph2_company1".to_string(),
                Some("Company".to_string()),
            ),
            create_test_node(
                4,
                "graph2_company2".to_string(),
                Some("Company".to_string()),
            ),
        ];

        let axioms = vec![OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "Person".to_string(),
            object: Some("Company".to_string()),
            property: None,
            confidence: 1.0,
        }];

        let constraints = translator.axioms_to_constraints(&axioms, &nodes).unwrap();

        // Verify constraints span across different node groups
        assert!(!constraints.is_empty());

        for constraint in &constraints {
            assert!(constraint.node_indices.len() >= 1);
            // Verify node IDs are valid
            for &node_id in &constraint.node_indices {
                assert!(node_id <= 4);
            }
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_memory_alignment() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes: Vec<Node> = (0..100)
            .map(|i| create_test_node(i, format!("node{}", i), Some("TestType".to_string())))
            .collect();

        let axioms = vec![OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "TestType".to_string(),
            object: Some("OtherType".to_string()),
            property: None,
            confidence: 1.0,
        }];

        let constraints = translator.axioms_to_constraints(&axioms, &nodes).unwrap();

        // Verify all constraints have properly aligned data
        for constraint in &constraints {
            // Check that constraint data is properly structured
            assert!(constraint.node_indices.len() > 0);
            assert!(!constraint.params.is_empty() || constraint.kind == ConstraintKind::Separation);

            // Verify memory alignment for GPU transfer
            // Node indices should be valid u32 values
            for &node_id in &constraint.node_indices {
                assert!(node_id < u32::MAX);
            }

            // Parameters should be valid f32 values
            for &param in &constraint.params {
                assert!(param.is_finite());
            }
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_constraint_strength_calculation() {
        let translator = OntologyConstraintTranslator::new();

        // Test strength for different axiom types
        let disjoint_strength = translator.get_constraint_strength(&OWLAxiomType::DisjointClasses);
        let subclass_strength = translator.get_constraint_strength(&OWLAxiomType::SubClassOf);
        let sameas_strength = translator.get_constraint_strength(&OWLAxiomType::SameAs);
        let functional_strength =
            translator.get_constraint_strength(&OWLAxiomType::FunctionalProperty);

        // Verify reasonable strength values
        assert!(disjoint_strength > 0.0 && disjoint_strength <= 1.0);
        assert!(subclass_strength > 0.0 && subclass_strength <= 1.0);
        assert!(sameas_strength > 0.0 && sameas_strength <= 1.0);
        assert!(functional_strength > 0.0 && functional_strength <= 1.0);

        // SameAs should have highest strength
        assert!(sameas_strength >= disjoint_strength);
        assert!(sameas_strength >= subclass_strength);
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_constraint_grouping() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes = vec![
            create_test_node(1, "person1".to_string(), Some("Person".to_string())),
            create_test_node(2, "company1".to_string(), Some("Company".to_string())),
            create_test_node(3, "employee1".to_string(), Some("Employee".to_string())),
        ];

        let axioms = vec![
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
                confidence: 1.0,
            },
        ];

        let graph = create_test_graph(nodes);
        let reasoning_report = OntologyReasoningReport {
            axioms,
            inferences: vec![],
            consistency_checks: vec![],
            reasoning_time_ms: 0,
        };

        let constraint_set = translator
            .apply_ontology_constraints(&graph, &reasoning_report)
            .unwrap();

        // Verify constraint groups exist
        assert!(!constraint_set.constraints.is_empty());
        assert!(!constraint_set.groups.is_empty());

        // Check for expected group names
        let has_separation_group = constraint_set.groups.contains_key("ontology_separation");
        let has_alignment_group = constraint_set.groups.contains_key("ontology_alignment");

        assert!(
            has_separation_group || has_alignment_group,
            "Should have at least one ontology constraint group"
        );
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_inference_to_constraints() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes = vec![
            create_test_node(1, "entity1".to_string(), None),
            create_test_node(2, "entity2".to_string(), None),
        ];

        let graph = create_test_graph(nodes);

        let inferences = vec![OntologyInference {
            inferred_axiom: OWLAxiom {
                axiom_type: OWLAxiomType::SameAs,
                subject: "entity1".to_string(),
                object: Some("entity2".to_string()),
                property: None,
                confidence: 0.8,
            },
            premise_axioms: vec!["axiom1".to_string()],
            reasoning_confidence: 0.8,
            is_derived: true,
        }];

        let constraints = translator
            .inferences_to_constraints(&inferences, &graph)
            .unwrap();

        assert!(!constraints.is_empty());

        // Inferred constraints should have adjusted weights
        for constraint in &constraints {
            assert!(constraint.weight <= 1.0);
            // Inferred axiom confidence (0.8) * reasoning confidence (0.8) * base strength
            assert!(constraint.weight < 1.0);
        }
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_cache_functionality() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes = vec![
            create_test_node(1, "node1".to_string(), Some("Type1".to_string())),
            create_test_node(2, "node2".to_string(), Some("Type2".to_string())),
        ];

        // Update node type cache
        translator.update_node_type_cache(&nodes);

        let stats = translator.get_cache_stats();
        assert_eq!(stats.node_type_entries, 2);

        // Clear cache
        translator.clear_cache();
        let stats_after = translator.get_cache_stats();
        assert_eq!(stats_after.node_type_entries, 0);
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_performance_large_graph() {
        use std::time::Instant;

        let mut translator = OntologyConstraintTranslator::new();

        // Create a large graph
        let nodes: Vec<Node> = (0..1000)
            .map(|i| {
                let node_type = if i % 2 == 0 { "TypeA" } else { "TypeB" };
                create_test_node(i, format!("node{}", i), Some(node_type.to_string()))
            })
            .collect();

        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::DisjointClasses,
            subject: "TypeA".to_string(),
            object: Some("TypeB".to_string()),
            property: None,
            confidence: 1.0,
        };

        let start = Instant::now();
        let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();
        let duration = start.elapsed();

        println!(
            "Generated {} constraints in {:?}",
            constraints.len(),
            duration
        );

        // Should complete in reasonable time
        assert!(
            duration.as_secs() < 5,
            "Performance test took too long: {:?}",
            duration
        );
        assert!(!constraints.is_empty());
    }

    #[cfg(all(feature = "ontology", feature = "gpu"))]
    #[test]
    fn test_different_from_constraint() {
        let mut translator = OntologyConstraintTranslator::new();

        let nodes = vec![
            create_test_node(1, "entity1".to_string(), None),
            create_test_node(2, "entity2".to_string(), None),
        ];

        let axiom = OWLAxiom {
            axiom_type: OWLAxiomType::DifferentFrom,
            subject: "entity1".to_string(),
            object: Some("entity2".to_string()),
            property: None,
            confidence: 1.0,
        };

        let constraints = translator.axioms_to_constraints(&[axiom], &nodes).unwrap();

        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].kind, ConstraintKind::Separation);
    }

    #[cfg(not(all(feature = "ontology", feature = "gpu")))]
    #[test]
    fn test_ontology_gpu_features_disabled() {
        println!("Ontology GPU tests skipped - features not enabled");
        assert!(true);
    }
}

*/
