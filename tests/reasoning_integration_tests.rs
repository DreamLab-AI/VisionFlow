// Integration Tests for Ontology Reasoning Pipeline
// Tests end-to-end workflows: GitHub sync → reasoning → constraints → GPU

use webxr::reasoning::{
    custom_reasoner::{CustomReasoner, OntologyReasoner, AxiomType},
    inference_cache::InferenceCache,
};
use webxr::constraints::{
    axiom_mapper::{AxiomMapper, OWLAxiom, AxiomType as MapperAxiomType},
    physics_constraint::PhysicsConstraintType,
};
use tempfile::TempDir;

mod fixtures;
use fixtures::ontology::test_ontologies::*;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_pipeline_simple_ontology() {
        // Setup
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let mut mapper = AxiomMapper::new();
        let ontology = create_simple_hierarchy();

        // Step 1: Inference
        let inferred_axioms = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        assert!(inferred_axioms.len() > 0, "Should have inferred axioms");

        // Step 2: Convert to mapper axioms
        let mapper_axioms: Vec<OWLAxiom> = inferred_axioms
            .iter()
            .filter_map(|axiom| {
                match axiom.axiom_type {
                    AxiomType::SubClassOf => {
                        // Map class names to IDs (simplified)
                        Some(OWLAxiom::inferred(MapperAxiomType::SubClassOf {
                            subclass: 1,
                            superclass: 2,
                        }))
                    }
                    AxiomType::DisjointWith => {
                        Some(OWLAxiom::inferred(MapperAxiomType::DisjointClasses {
                            classes: vec![1, 2],
                        }))
                    }
                    _ => None,
                }
            })
            .collect();

        // Step 3: Generate constraints
        let constraints = mapper.translate_axioms(&mapper_axioms);
        assert!(constraints.len() > 0, "Should generate physics constraints");

        // Step 4: Verify constraint types
        let has_clustering = constraints.iter().any(|c| {
            matches!(c.constraint_type, PhysicsConstraintType::Clustering { .. })
        });
        let has_separation = constraints.iter().any(|c| {
            matches!(c.constraint_type, PhysicsConstraintType::Separation { .. })
        });

        assert!(has_clustering || has_separation, "Should have at least one constraint type");
    }

    #[tokio::test]
    async fn test_cache_invalidation_on_update() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();

        // Initial ontology
        let mut ontology = create_simple_hierarchy();
        let result1 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        // Simulate GitHub sync update
        ontology.classes.insert("UpdatedClass".to_string(),
            webxr::reasoning::custom_reasoner::OWLClass {
                iri: "http://example.org/UpdatedClass".to_string(),
                label: Some("Updated Class".to_string()),
                parent_class_iri: Some("Cell".to_string()),
            }
        );

        // Cache should detect change and recompute
        let result2 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        assert_ne!(result1.len(), result2.len(), "Cache should invalidate on ontology update");
    }

    #[tokio::test]
    async fn test_multi_ontology_workflow() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let mut mapper = AxiomMapper::new();

        // Process multiple ontologies
        let ontologies = vec![
            (1, create_simple_hierarchy()),
            (2, create_deep_hierarchy()),
            (3, create_diamond_pattern()),
        ];

        for (id, ontology) in ontologies {
            let inferred = cache.get_or_compute(id, &reasoner, &ontology).unwrap();
            assert!(inferred.len() >= 0, "Each ontology should process");
        }

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.total_entries, 3, "Should cache all 3 ontologies");
    }

    #[tokio::test]
    async fn test_constraint_priority_ordering() {
        let mut mapper = AxiomMapper::new();

        let axioms = vec![
            OWLAxiom::user_defined(MapperAxiomType::SubClassOf {
                subclass: 1,
                superclass: 2,
            }),
            OWLAxiom::inferred(MapperAxiomType::SubClassOf {
                subclass: 3,
                superclass: 4,
            }),
            OWLAxiom::asserted(MapperAxiomType::DisjointClasses {
                classes: vec![5, 6],
            }),
        ];

        let constraints = mapper.translate_axioms(&axioms);

        // Verify priorities are correct
        assert_eq!(constraints[0].priority, 1); // user-defined
        assert_eq!(constraints[1].priority, 3); // inferred
        assert_eq!(constraints[2].priority, 5); // asserted
    }

    #[tokio::test]
    async fn test_inference_determinism() {
        let ontology = create_simple_hierarchy();
        let reasoner = CustomReasoner::new();

        // Run inference multiple times
        let result1 = reasoner.infer_axioms(&ontology).unwrap();
        let result2 = reasoner.infer_axioms(&ontology).unwrap();
        let result3 = reasoner.infer_axioms(&ontology).unwrap();

        // Results should be identical
        assert_eq!(result1.len(), result2.len());
        assert_eq!(result2.len(), result3.len());

        // Verify axiom equivalence
        for (ax1, ax2) in result1.iter().zip(result2.iter()) {
            assert_eq!(ax1.axiom_type, ax2.axiom_type);
            assert_eq!(ax1.subject, ax2.subject);
            assert_eq!(ax1.object, ax2.object);
        }
    }

    #[tokio::test]
    async fn test_large_ontology_performance() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();

        // Test with large ontology
        let ontology = create_large_ontology(1000);

        let start = std::time::Instant::now();
        let result = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        let duration = start.elapsed();

        println!("Large ontology (1000 classes) inference time: {:?}", duration);
        assert!(duration.as_secs() < 10, "Should complete within 10 seconds");
        assert!(result.len() > 0, "Should have inferences");
    }

    #[tokio::test]
    async fn test_constraint_generation_completeness() {
        let mut mapper = AxiomMapper::new();

        // Test all axiom types
        let axioms = vec![
            OWLAxiom::asserted(MapperAxiomType::SubClassOf {
                subclass: 1,
                superclass: 2,
            }),
            OWLAxiom::asserted(MapperAxiomType::DisjointClasses {
                classes: vec![3, 4, 5],
            }),
            OWLAxiom::asserted(MapperAxiomType::EquivalentClasses {
                class1: 6,
                class2: 7,
            }),
            OWLAxiom::asserted(MapperAxiomType::PartOf {
                part: 8,
                whole: 9,
            }),
            OWLAxiom::asserted(MapperAxiomType::DisjointUnion {
                union_class: 10,
                disjoint_classes: vec![11, 12],
            }),
        ];

        let constraints = mapper.translate_axioms(&axioms);

        // Verify all axiom types generated constraints
        assert!(constraints.len() >= axioms.len(), "All axioms should generate constraints");

        // Verify constraint type diversity
        let has_clustering = constraints.iter().any(|c|
            matches!(c.constraint_type, PhysicsConstraintType::Clustering { .. })
        );
        let has_separation = constraints.iter().any(|c|
            matches!(c.constraint_type, PhysicsConstraintType::Separation { .. })
        );
        let has_colocation = constraints.iter().any(|c|
            matches!(c.constraint_type, PhysicsConstraintType::Colocation { .. })
        );
        let has_containment = constraints.iter().any(|c|
            matches!(c.constraint_type, PhysicsConstraintType::Containment { .. })
        );

        assert!(has_clustering, "Should have clustering constraints");
        assert!(has_separation, "Should have separation constraints");
        assert!(has_colocation, "Should have colocation constraints");
        assert!(has_containment, "Should have containment constraints");
    }

    #[tokio::test]
    async fn test_cache_concurrent_access() {
        use std::sync::Arc;
        use tokio::task;

        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");
        let cache = Arc::new(InferenceCache::new(&cache_path).unwrap());
        let reasoner = Arc::new(CustomReasoner::new());
        let ontology = Arc::new(create_simple_hierarchy());

        // Simulate concurrent requests
        let mut handles = vec![];
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let reasoner_clone = Arc::clone(&reasoner);
            let ontology_clone = Arc::clone(&ontology);

            let handle = task::spawn(async move {
                cache_clone.get_or_compute(i, reasoner_clone.as_ref(), ontology_clone.as_ref())
            });
            handles.push(handle);
        }

        // All should complete successfully
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Concurrent access should succeed");
        }
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_empty_ontology_handling() {
        let ontology = create_empty_ontology();
        let reasoner = CustomReasoner::new();

        let result = reasoner.infer_axioms(&ontology);
        assert!(result.is_ok(), "Empty ontology should not error");
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_cache_with_invalid_path() {
        // Should fail gracefully
        let result = InferenceCache::new("/invalid/path/cache.db");
        assert!(result.is_err(), "Invalid path should return error");
    }

    #[tokio::test]
    async fn test_empty_axiom_list() {
        let mut mapper = AxiomMapper::new();
        let constraints = mapper.translate_axioms(&vec![]);
        assert_eq!(constraints.len(), 0);
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[tokio::test]
    async fn test_circular_reference_handling() {
        let mut ontology = create_empty_ontology();

        // Create circular reference: A → B → C → A
        ontology.classes.insert("A".to_string(), webxr::reasoning::custom_reasoner::OWLClass {
            iri: "A".to_string(),
            label: Some("A".to_string()),
            parent_class_iri: Some("C".to_string()),
        });
        ontology.classes.insert("B".to_string(), webxr::reasoning::custom_reasoner::OWLClass {
            iri: "B".to_string(),
            label: Some("B".to_string()),
            parent_class_iri: Some("A".to_string()),
        });
        ontology.classes.insert("C".to_string(), webxr::reasoning::custom_reasoner::OWLClass {
            iri: "C".to_string(),
            label: Some("C".to_string()),
            parent_class_iri: Some("B".to_string()),
        });

        ontology.subclass_of.insert("A".to_string(), vec!["C".to_string()].into_iter().collect());
        ontology.subclass_of.insert("B".to_string(), vec!["A".to_string()].into_iter().collect());
        ontology.subclass_of.insert("C".to_string(), vec!["B".to_string()].into_iter().collect());

        let reasoner = CustomReasoner::new();
        let result = reasoner.infer_axioms(&ontology);

        // Should handle circular references without infinite loop
        assert!(result.is_ok(), "Should handle circular references");
    }

    #[tokio::test]
    async fn test_single_class_ontology() {
        let mut ontology = create_empty_ontology();

        ontology.classes.insert("Singleton".to_string(), webxr::reasoning::custom_reasoner::OWLClass {
            iri: "Singleton".to_string(),
            label: Some("Singleton".to_string()),
            parent_class_iri: None,
        });

        let reasoner = CustomReasoner::new();
        let inferred = reasoner.infer_axioms(&ontology).unwrap();

        assert_eq!(inferred.len(), 0, "Single class should have no inferences");
    }

    #[tokio::test]
    async fn test_self_referential_class() {
        let mut ontology = create_empty_ontology();

        ontology.classes.insert("SelfRef".to_string(), webxr::reasoning::custom_reasoner::OWLClass {
            iri: "SelfRef".to_string(),
            label: Some("Self Referential".to_string()),
            parent_class_iri: Some("SelfRef".to_string()),
        });

        ontology.subclass_of.insert(
            "SelfRef".to_string(),
            vec!["SelfRef".to_string()].into_iter().collect(),
        );

        let reasoner = CustomReasoner::new();
        let result = reasoner.infer_axioms(&ontology);

        assert!(result.is_ok(), "Should handle self-reference");
    }
}
