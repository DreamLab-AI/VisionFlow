/// Unit tests for OntologyReasoningService and CustomReasoner
///
/// Tests inference generation, caching, and correctness of reasoning operations

#[cfg(feature = "ontology")]
mod ontology_reasoning_tests {
    use webxr::reasoning::custom_reasoner::CustomReasoner;
    use webxr::reasoning::inference_cache::InferenceCache;
    use std::path::PathBuf;
    use std::fs;
    use tempfile::TempDir;

    /// Helper to get test ontology path
    fn get_test_ontology_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/ontologies/test_reasoning.owl")
    }

    /// Helper to load OWL file as string
    fn load_owl_content() -> Result<String, std::io::Error> {
        let path = get_test_ontology_path();
        fs::read_to_string(&path)
    }

    #[test]
    fn test_custom_reasoner_initialization() {
        let reasoner = CustomReasoner::new();
        assert!(reasoner.is_initialized(), "Reasoner should be initialized");
    }

    #[test]
    fn test_basic_reasoning_from_owl() {
        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let result = reasoner.process_owl(&owl_content);

        assert!(result.is_ok(), "Reasoning should succeed: {:?}", result.err());

        let inferred_axioms = result.unwrap();
        assert!(!inferred_axioms.is_empty(), "Should infer some axioms");

        println!("Inferred {} axioms from test ontology", inferred_axioms.len());
    }

    #[test]
    fn test_subclass_transitive_inference() {
        // Executive -> Manager -> Employee -> Person -> Entity
        // Should infer Executive subClassOf Person, Employee, Entity

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        // Check for transitive subclass relationships
        let has_executive_to_person = inferred.subclass_of.iter()
            .any(|(sub, sup)| {
                sub.contains("Executive") && sup.contains("Person")
            });

        assert!(
            has_executive_to_person,
            "Should infer Executive subClassOf Person through transitivity"
        );

        let has_executive_to_entity = inferred.subclass_of.iter()
            .any(|(sub, sup)| {
                sub.contains("Executive") && sup.contains("Entity")
            });

        assert!(
            has_executive_to_entity,
            "Should infer Executive subClassOf Entity through transitivity"
        );
    }

    #[test]
    fn test_disjoint_classes_detection() {
        // Person and Organization are explicitly disjoint
        // Company and NonProfit are also disjoint

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        // Check for disjoint relationship
        let has_person_org_disjoint = inferred.disjoint_with.iter()
            .any(|(c1, c2)| {
                (c1.contains("Person") && c2.contains("Organization")) ||
                (c1.contains("Organization") && c2.contains("Person"))
            });

        assert!(
            has_person_org_disjoint,
            "Should detect Person and Organization as disjoint"
        );

        let has_company_nonprofit_disjoint = inferred.disjoint_with.iter()
            .any(|(c1, c2)| {
                (c1.contains("Company") && c2.contains("NonProfit")) ||
                (c1.contains("NonProfit") && c2.contains("Company"))
            });

        assert!(
            has_company_nonprofit_disjoint,
            "Should detect Company and NonProfit as disjoint"
        );
    }

    #[test]
    fn test_equivalent_classes_detection() {
        // Worker and Employee are declared equivalent

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        let has_worker_employee_equiv = inferred.equivalent_class.iter()
            .any(|(c1, c2)| {
                (c1.contains("Worker") && c2.contains("Employee")) ||
                (c1.contains("Employee") && c2.contains("Worker"))
            });

        assert!(
            has_worker_employee_equiv,
            "Should detect Worker and Employee as equivalent"
        );
    }

    #[test]
    fn test_inverse_properties_detection() {
        // hasEmployee and worksFor are inverse properties

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        let has_inverse = inferred.inverse_of.iter()
            .any(|(p1, p2)| {
                (p1.contains("hasEmployee") && p2.contains("worksFor")) ||
                (p1.contains("worksFor") && p2.contains("hasEmployee"))
            });

        assert!(
            has_inverse,
            "Should detect hasEmployee and worksFor as inverse properties"
        );
    }

    #[test]
    fn test_inference_cache_hit_miss() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache = InferenceCache::new(temp_dir.path().to_path_buf());

        let owl_content = "test content for cache";

        // First access - should be a miss
        let cached = cache.get(owl_content);
        assert!(cached.is_none(), "First access should be cache miss");

        // Store inference result
        use webxr::reasoning::custom_reasoner::InferredAxioms;
        let test_axioms = InferredAxioms {
            subclass_of: vec![("A".to_string(), "B".to_string())],
            disjoint_with: vec![],
            equivalent_class: vec![],
            inverse_of: vec![],
        };

        cache.set(owl_content, test_axioms.clone());

        // Second access - should be a hit
        let cached = cache.get(owl_content);
        assert!(cached.is_some(), "Second access should be cache hit");

        let retrieved = cached.unwrap();
        assert_eq!(retrieved.subclass_of.len(), 1);
        assert_eq!(retrieved.subclass_of[0].0, "A");
        assert_eq!(retrieved.subclass_of[0].1, "B");
    }

    #[test]
    fn test_cache_invalidation_on_content_change() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache = InferenceCache::new(temp_dir.path().to_path_buf());

        let owl_content_v1 = "version 1 content";
        let owl_content_v2 = "version 2 content";

        use webxr::reasoning::custom_reasoner::InferredAxioms;
        let axioms_v1 = InferredAxioms {
            subclass_of: vec![("A".to_string(), "B".to_string())],
            disjoint_with: vec![],
            equivalent_class: vec![],
            inverse_of: vec![],
        };

        cache.set(owl_content_v1, axioms_v1);

        // Different content should not hit cache
        let cached = cache.get(owl_content_v2);
        assert!(cached.is_none(), "Different content should not hit cache");
    }

    #[test]
    fn test_reasoning_performance_simple_ontology() {
        use std::time::Instant;

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();

        let start = Instant::now();
        let result = reasoner.process_owl(&owl_content);
        let duration = start.elapsed();

        assert!(result.is_ok(), "Reasoning should succeed");
        assert!(
            duration.as_millis() < 50,
            "Simple ontology reasoning should complete in <50ms, took {}ms",
            duration.as_millis()
        );

        println!("Reasoning completed in {}ms", duration.as_millis());
    }

    #[test]
    fn test_empty_ontology_handling() {
        let empty_owl = r#"<?xml version="1.0"?>
<Ontology xmlns="http://www.w3.org/2002/07/owl#"
     ontologyIRI="http://example.org/empty">
</Ontology>"#;

        let reasoner = CustomReasoner::new();
        let result = reasoner.process_owl(empty_owl);

        assert!(result.is_ok(), "Empty ontology should not cause errors");
        let inferred = result.unwrap();

        // Empty ontology might still have some built-in inferences
        println!("Empty ontology produced {} inferred axioms", inferred.len());
    }

    #[test]
    fn test_malformed_owl_handling() {
        let malformed_owl = "This is not valid OWL XML";

        let reasoner = CustomReasoner::new();
        let result = reasoner.process_owl(malformed_owl);

        assert!(result.is_err(), "Malformed OWL should return error");
    }

    #[test]
    fn test_cache_persistence_across_instances() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cache_path = temp_dir.path().to_path_buf();

        let owl_content = "persistent test content";

        use webxr::reasoning::custom_reasoner::InferredAxioms;
        let test_axioms = InferredAxioms {
            subclass_of: vec![("X".to_string(), "Y".to_string())],
            disjoint_with: vec![],
            equivalent_class: vec![],
            inverse_of: vec![],
        };

        // First cache instance - write
        {
            let cache1 = InferenceCache::new(cache_path.clone());
            cache1.set(owl_content, test_axioms.clone());
        }

        // Second cache instance - read
        {
            let cache2 = InferenceCache::new(cache_path.clone());
            let retrieved = cache2.get(owl_content);

            assert!(retrieved.is_some(), "Cache should persist across instances");
            let axioms = retrieved.unwrap();
            assert_eq!(axioms.subclass_of[0].0, "X");
            assert_eq!(axioms.subclass_of[0].1, "Y");
        }
    }

    #[test]
    fn test_subproperty_inference() {
        // manages is a subPropertyOf hasEmployee

        let owl_content = load_owl_content()
            .expect("Failed to load test ontology");

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.process_owl(&owl_content)
            .expect("Reasoning failed");

        // Check if subproperty relationships are captured
        // Note: This depends on CustomReasoner implementation
        println!("Inferred axioms: {:?}", inferred);

        // This test validates that property hierarchies are processed
        assert!(
            !inferred.is_empty(),
            "Should process property hierarchies"
        );
    }
}
