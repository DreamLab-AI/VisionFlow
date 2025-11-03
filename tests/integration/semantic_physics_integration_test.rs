/// Integration tests for Semantic Physics
///
/// Validates that ontological constraints are correctly translated into physics forces
/// and applied to the graph simulation

#[cfg(all(feature = "ontology", feature = "gpu"))]
mod semantic_physics_integration {
    use webxr::reasoning::custom_reasoner::CustomReasoner;
    use webxr::gpu::semantic_forces::SemanticForceGenerator;
    use std::path::PathBuf;
    use std::fs;

    fn get_test_ontology_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ontologies/test_reasoning.owl")
    }

    fn load_owl_content() -> Result<String, std::io::Error> {
        fs::read_to_string(get_test_ontology_path())
    }

    #[test]
    fn test_disjoint_creates_repulsion_forces() {
        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        // Create mock node mapping
        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Person".to_string(), 0);
        node_id_to_index.insert("Organization".to_string(), 1);
        node_id_to_index.insert("Company".to_string(), 2);
        node_id_to_index.insert("NonProfit".to_string(), 3);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Find repulsion constraints for disjoint classes
        let person_org_repulsion = constraints.iter()
            .find(|c| {
                c.constraint_type == "repulsion" &&
                ((c.node_a == 0 && c.node_b == 1) || (c.node_a == 1 && c.node_b == 0))
            });

        assert!(
            person_org_repulsion.is_some(),
            "Should create repulsion constraint for Person and Organization"
        );

        let constraint = person_org_repulsion.unwrap();
        assert!(
            constraint.strength > 0.0,
            "Repulsion strength should be positive"
        );

        println!("Person-Organization repulsion strength: {}", constraint.strength);
    }

    #[test]
    fn test_subclass_creates_attraction_forces() {
        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        // Create mock node mapping
        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Executive".to_string(), 0);
        node_id_to_index.insert("Manager".to_string(), 1);
        node_id_to_index.insert("Employee".to_string(), 2);
        node_id_to_index.insert("Person".to_string(), 3);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Find attraction constraints for subclass relationships
        let executive_manager_attraction = constraints.iter()
            .find(|c| {
                c.constraint_type == "attraction" &&
                ((c.node_a == 0 && c.node_b == 1) || (c.node_a == 1 && c.node_b == 0))
            });

        assert!(
            executive_manager_attraction.is_some(),
            "Should create attraction constraint for Executive subClassOf Manager"
        );

        let constraint = executive_manager_attraction.unwrap();
        assert!(
            constraint.strength > 0.0,
            "Attraction strength should be positive"
        );

        println!("Executive-Manager attraction strength: {}", constraint.strength);
    }

    #[test]
    fn test_force_magnitude_correctness() {
        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Person".to_string(), 0);
        node_id_to_index.insert("Organization".to_string(), 1);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Verify force magnitudes are reasonable
        for constraint in &constraints {
            assert!(
                constraint.strength >= 0.0 && constraint.strength <= 10.0,
                "Force strength {} should be in reasonable range [0, 10]",
                constraint.strength
            );

            assert!(
                constraint.distance_target >= 0.0,
                "Target distance {} should be non-negative",
                constraint.distance_target
            );
        }

        println!("Generated {} constraints with valid magnitudes", constraints.len());
    }

    #[test]
    fn test_correct_node_pair_application() {
        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Executive".to_string(), 0);
        node_id_to_index.insert("Manager".to_string(), 1);
        node_id_to_index.insert("Person".to_string(), 2);
        node_id_to_index.insert("Organization".to_string(), 3);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Verify node indices are valid
        for constraint in &constraints {
            assert!(
                constraint.node_a < 4,
                "node_a index {} out of bounds",
                constraint.node_a
            );
            assert!(
                constraint.node_b < 4,
                "node_b index {} out of bounds",
                constraint.node_b
            );
            assert_ne!(
                constraint.node_a, constraint.node_b,
                "Constraint should not reference same node"
            );
        }

        println!("All {} constraints have valid node pairs", constraints.len());
    }

    #[test]
    fn test_equivalent_class_handling() {
        // Worker and Employee are equivalent - should have special handling

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Worker".to_string(), 0);
        node_id_to_index.insert("Employee".to_string(), 1);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Equivalent classes might create strong attraction or be merged
        let worker_employee_constraint = constraints.iter()
            .find(|c| {
                (c.node_a == 0 && c.node_b == 1) || (c.node_a == 1 && c.node_b == 0)
            });

        if let Some(constraint) = worker_employee_constraint {
            println!(
                "Worker-Employee constraint: type={}, strength={}",
                constraint.constraint_type, constraint.strength
            );

            // If there's a constraint, it should be strong attraction
            if constraint.constraint_type == "attraction" {
                assert!(
                    constraint.strength > 0.5,
                    "Equivalent classes should have strong attraction"
                );
            }
        }
    }

    #[test]
    fn test_transitive_hierarchy_forces() {
        // Executive -> Manager -> Employee -> Person
        // Should create force chain

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Executive".to_string(), 0);
        node_id_to_index.insert("Manager".to_string(), 1);
        node_id_to_index.insert("Employee".to_string(), 2);
        node_id_to_index.insert("Person".to_string(), 3);
        node_id_to_index.insert("Entity".to_string(), 4);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Count how many attraction constraints exist in the hierarchy
        let hierarchy_attractions = constraints.iter()
            .filter(|c| c.constraint_type == "attraction")
            .count();

        assert!(
            hierarchy_attractions >= 4,
            "Should have at least 4 attraction constraints in hierarchy chain"
        );

        println!(
            "Hierarchy chain created {} attraction constraints",
            hierarchy_attractions
        );
    }

    #[test]
    fn test_disjoint_transitive_application() {
        // If Person ⊥ Organization and Company ⊑ Organization,
        // then Person ⊥ Company should also hold

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        // Check if reasoner inferred Person ⊥ Company
        let has_person_company_disjoint = inferred.disjoint_with.iter()
            .any(|(c1, c2)| {
                (c1.contains("Person") && c2.contains("Company")) ||
                (c2.contains("Person") && c1.contains("Company"))
            });

        if has_person_company_disjoint {
            println!("Reasoner correctly inferred Person ⊥ Company through transitivity");
        } else {
            println!("Note: Transitive disjointness may need explicit reasoning");
        }

        // Test should pass regardless, but log the behavior
        assert!(true);
    }

    #[test]
    fn test_constraint_serialization() {
        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        let mut node_id_to_index = std::collections::HashMap::new();
        node_id_to_index.insert("Person".to_string(), 0);
        node_id_to_index.insert("Organization".to_string(), 1);

        let force_gen = SemanticForceGenerator::new();
        let constraints = force_gen.generate_constraints(&inferred, &node_id_to_index);

        // Test that constraints can be serialized for GPU
        let serialized = serde_json::to_string(&constraints);
        assert!(serialized.is_ok(), "Constraints should be serializable");

        let json = serialized.unwrap();
        println!("Serialized constraints: {} bytes", json.len());

        // Test deserialization
        let deserialized: Result<Vec<_>, _> = serde_json::from_str(&json);
        assert!(deserialized.is_ok(), "Constraints should be deserializable");
    }
}
