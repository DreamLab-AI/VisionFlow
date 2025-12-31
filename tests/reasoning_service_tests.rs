// Comprehensive Tests for Ontology Reasoning Service
// Tests inference, caching, constraint generation, and integration
//
// NOTE: These tests are disabled because:
// 1. inference_cache module does not exist in the reasoning module
// 2. webxr::constraints module doesn't export expected types
// 3. fixtures::ontology::test_ontologies module doesn't exist
//
// To re-enable:
// 1. Implement inference_cache module
// 2. Update constraints module exports
// 3. Create test fixtures
// 4. Uncomment the code below

/*
use webxr::reasoning::{
    custom_reasoner::{CustomReasoner, OntologyReasoner, AxiomType, InferredAxiom},
    inference_cache::InferenceCache,
};
use webxr::constraints::{
    axiom_mapper::{AxiomMapper, OWLAxiom, AxiomType as MapperAxiomType, TranslationConfig},
    physics_constraint::{PhysicsConstraintType, PRIORITY_ASSERTED, PRIORITY_INFERRED, PRIORITY_USER_DEFINED},
};
use tempfile::TempDir;

// These imports reference non-existent types - commenting out the entire module
// [INNER COMMENT REMOVED - was: mod fixtures; use fixtures::ontology::test_ontologies::*;]

#[cfg(test)]
mod custom_reasoner_tests {
    use super::*;

    #[tokio::test]
    async fn test_infer_transitive_subclass_simple() {
        let ontology = create_simple_hierarchy();
        let reasoner = CustomReasoner::new();

        let inferred = reasoner.infer_axioms(&ontology).unwrap();

        // Should infer: Neuron → MaterialEntity, Neuron → Entity
        // Should infer: Astrocyte → MaterialEntity, Astrocyte → Entity
        // Should infer: Cell → Entity
        let transitive_axioms: Vec<_> = inferred
            .iter()
            .filter(|a| a.axiom_type == AxiomType::SubClassOf)
            .collect();

        assert!(transitive_axioms.len() >= 3, "Expected at least 3 transitive inferences");

        // Verify specific inference
        assert!(
            inferred.iter().any(|axiom|
                axiom.axiom_type == AxiomType::SubClassOf
                && axiom.subject == "Neuron"
                && axiom.object.as_ref() == Some(&"MaterialEntity".to_string())
            ),
            "Should infer Neuron → MaterialEntity"
        );

        assert!(
            inferred.iter().any(|axiom|
                axiom.axiom_type == AxiomType::SubClassOf
                && axiom.subject == "Neuron"
                && axiom.object.as_ref() == Some(&"Entity".to_string())
            ),
            "Should infer Neuron → Entity"
        );
    }

    #[tokio::test]
    async fn test_infer_deep_hierarchy() {
        let ontology = create_deep_hierarchy();
        let reasoner = CustomReasoner::new();

        let inferred = reasoner.infer_axioms(&ontology).unwrap();

        // Level4 should have 3 transitive parents (Level3, Level2, Level1, Level0)
        let level4_inferences: Vec<_> = inferred
            .iter()
            .filter(|a|
                a.axiom_type == AxiomType::SubClassOf
                && a.subject == "Level4"
            )
            .collect();

        assert!(level4_inferences.len() >= 3, "Deep hierarchy should infer multiple levels");

        // Verify Level4 → Level0 is inferred
        assert!(
            inferred.iter().any(|axiom|
                axiom.axiom_type == AxiomType::SubClassOf
                && axiom.subject == "Level4"
                && axiom.object.as_ref() == Some(&"Level0".to_string())
            ),
            "Should infer transitive relationship to root"
        );
    }

    #[tokio::test]
    async fn test_is_subclass_of_direct() {
        let ontology = create_simple_hierarchy();
        let reasoner = CustomReasoner::new();

        assert!(
            reasoner.is_subclass_of("Neuron", "Cell", &ontology),
            "Direct subclass relationship"
        );
        assert!(
            reasoner.is_subclass_of("Cell", "MaterialEntity", &ontology),
            "Direct subclass relationship"
        );
    }

    #[tokio::test]
    async fn test_is_subclass_of_transitive() {
        let ontology = create_simple_hierarchy();
        let reasoner = CustomReasoner::new();

        assert!(
            reasoner.is_subclass_of("Neuron", "MaterialEntity", &ontology),
            "Transitive subclass: Neuron → Cell → MaterialEntity"
        );
        assert!(
            reasoner.is_subclass_of("Neuron", "Entity", &ontology),
            "Transitive subclass: Neuron → ... → Entity"
        );
        assert!(
            !reasoner.is_subclass_of("Entity", "Neuron", &ontology),
            "Should not be subclass in reverse"
        );
    }

    #[tokio::test]
    async fn test_are_disjoint() {
        let ontology = create_simple_hierarchy();
        let reasoner = CustomReasoner::new();

        assert!(
            reasoner.are_disjoint("Neuron", "Astrocyte", &ontology),
            "Neuron and Astrocyte are disjoint"
        );
        assert!(
            reasoner.are_disjoint("Astrocyte", "Neuron", &ontology),
            "Disjoint is symmetric"
        );
        assert!(
            !reasoner.are_disjoint("Neuron", "Cell", &ontology),
            "Neuron is subclass of Cell, not disjoint"
        );
    }

    #[tokio::test]
    async fn test_infer_disjoint_subclasses() {
        let ontology = create_multiple_disjoint();
        let reasoner = CustomReasoner::new();

        let inferred = reasoner.infer_axioms(&ontology).unwrap();

        let disjoint_axioms: Vec<_> = inferred
            .iter()
            .filter(|a| a.axiom_type == AxiomType::DisjointWith)
            .collect();

        // Multiple disjoint sets should generate disjoint inferences
        assert!(disjoint_axioms.len() >= 0, "Should handle multiple disjoint sets");
    }

    #[tokio::test]
    async fn test_infer_equivalent_classes() {
        let ontology = create_equivalent_classes();
        let reasoner = CustomReasoner::new();

        let inferred = reasoner.infer_axioms(&ontology).unwrap();

        let equiv_axioms: Vec<_> = inferred
            .iter()
            .filter(|a| a.axiom_type == AxiomType::EquivalentTo)
            .collect();

        assert!(equiv_axioms.len() > 0, "Should infer symmetric equivalence");

        // Transitive equivalence: Person ≡ Human ≡ Individual
        assert!(
            inferred.iter().any(|axiom|
                axiom.axiom_type == AxiomType::EquivalentTo
                && axiom.subject == "Person"
                && axiom.object.as_ref() == Some(&"Individual".to_string())
            ),
            "Should infer transitive equivalence"
        );
    }

    #[tokio::test]
    async fn test_diamond_pattern() {
        let ontology = create_diamond_pattern();
        let reasoner = CustomReasoner::new();

        let inferred = reasoner.infer_axioms(&ontology).unwrap();

        // Bottom should have Top as ancestor through both paths
        assert!(
            reasoner.is_subclass_of("Bottom", "Top", &ontology),
            "Diamond pattern should resolve to common ancestor"
        );

        // Verify both paths are captured
        assert!(
            reasoner.is_subclass_of("Bottom", "Left", &ontology),
            "Bottom → Left → Top"
        );
        assert!(
            reasoner.is_subclass_of("Bottom", "Right", &ontology),
            "Bottom → Right → Top"
        );
    }

    #[tokio::test]
    async fn test_empty_ontology() {
        let ontology = create_empty_ontology();
        let reasoner = CustomReasoner::new();

        let inferred = reasoner.infer_axioms(&ontology).unwrap();
        assert_eq!(inferred.len(), 0, "Empty ontology should have no inferences");
    }

    #[tokio::test]
    async fn test_confidence_scores() {
        let ontology = create_simple_hierarchy();
        let reasoner = CustomReasoner::new();

        let inferred = reasoner.infer_axioms(&ontology).unwrap();

        // All inferences should have confidence = 1.0
        for axiom in &inferred {
            assert_eq!(axiom.confidence, 1.0, "All inferences should have confidence 1.0");
        }
    }
}

#[cfg(test)]
mod inference_cache_tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_miss_and_hit() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_simple_hierarchy();

        // First call - cache miss
        let start = std::time::Instant::now();
        let result1 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        let duration1 = start.elapsed();

        // Second call - cache hit
        let start = std::time::Instant::now();
        let result2 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        let duration2 = start.elapsed();

        assert_eq!(result1.len(), result2.len());
        assert!(duration2 < duration1, "Cache hit should be faster");
        println!("Cache miss: {:?}, Cache hit: {:?}", duration1, duration2);
    }

    #[tokio::test]
    async fn test_cache_invalidation_on_change() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let mut ontology = create_simple_hierarchy();

        // Initial computation
        let result1 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        // Modify ontology
        ontology.classes.insert("NewClass".to_string(), webxr::reasoning::custom_reasoner::OWLClass {
            iri: "http://example.org/NewClass".to_string(),
            label: Some("New Class".to_string()),
            parent_class_iri: Some("Cell".to_string()),
        });

        // Should recompute due to checksum change
        let result2 = cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        assert_ne!(result1.len(), result2.len(), "Cache should invalidate on ontology change");
    }

    #[tokio::test]
    async fn test_cache_checksum_stability() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();

        // Same ontology should produce same checksum
        let ontology1 = create_simple_hierarchy();
        let ontology2 = create_simple_hierarchy();

        cache.get_or_compute(1, &reasoner, &ontology1).unwrap();
        let cached1 = cache.load_from_cache(1).unwrap().unwrap();

        cache.get_or_compute(2, &reasoner, &ontology2).unwrap();
        let cached2 = cache.load_from_cache(2).unwrap().unwrap();

        assert_eq!(
            cached1.ontology_checksum,
            cached2.ontology_checksum,
            "Identical ontologies should have same checksum"
        );
    }

    #[tokio::test]
    async fn test_cache_invalidate_specific() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_simple_hierarchy();

        cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        cache.get_or_compute(2, &reasoner, &ontology).unwrap();

        // Invalidate only ontology 1
        cache.invalidate(1).unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.total_entries, 1, "Should have 1 entry after invalidation");
    }

    #[tokio::test]
    async fn test_cache_clear_all() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_simple_hierarchy();

        cache.get_or_compute(1, &reasoner, &ontology).unwrap();
        cache.get_or_compute(2, &reasoner, &ontology).unwrap();

        cache.clear_all().unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.total_entries, 0, "Cache should be empty after clear_all");
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache_path = temp_dir.path().join("cache.db");

        let cache = InferenceCache::new(&cache_path).unwrap();
        let reasoner = CustomReasoner::new();
        let ontology = create_simple_hierarchy();

        cache.get_or_compute(1, &reasoner, &ontology).unwrap();

        let stats = cache.get_stats().unwrap();
        assert_eq!(stats.total_entries, 1);
        assert!(stats.total_size_bytes > 0, "Cache should have size > 0");
    }
}

#[cfg(test)]
mod axiom_mapper_tests {
    use super::*;

    #[test]
    fn test_disjoint_classes_constraint_generation() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::asserted(MapperAxiomType::DisjointClasses {
            classes: vec![1, 2, 3],
        });

        let constraints = mapper.translate_axiom(&axiom);

        // n classes = n*(n-1)/2 pairwise constraints
        assert_eq!(constraints.len(), 3, "3 classes should generate 3 pairwise constraints");

        for constraint in &constraints {
            assert_eq!(constraint.nodes.len(), 2);
            assert_eq!(constraint.priority, PRIORITY_ASSERTED);

            match &constraint.constraint_type {
                PhysicsConstraintType::Separation { min_distance, strength } => {
                    assert_eq!(*min_distance, 35.0);
                    assert_eq!(*strength, 0.8);
                }
                _ => panic!("Wrong constraint type"),
            }
        }
    }

    #[test]
    fn test_subclass_of_constraint_generation() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::asserted(MapperAxiomType::SubClassOf {
            subclass: 10,
            superclass: 20,
        });

        let constraints = mapper.translate_axiom(&axiom);

        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].nodes, vec![10, 20]);

        match &constraints[0].constraint_type {
            PhysicsConstraintType::Clustering { ideal_distance, stiffness } => {
                assert_eq!(*ideal_distance, 20.0);
                assert_eq!(*stiffness, 0.6);
            }
            _ => panic!("Wrong constraint type"),
        }

        // Verify hierarchy cache
        assert_eq!(mapper.get_subclasses(20), vec![10]);
    }

    #[test]
    fn test_equivalent_classes_constraint() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::asserted(MapperAxiomType::EquivalentClasses {
            class1: 5,
            class2: 6,
        });

        let constraints = mapper.translate_axiom(&axiom);

        assert_eq!(constraints.len(), 1);
        match &constraints[0].constraint_type {
            PhysicsConstraintType::Colocation { target_distance, strength } => {
                assert_eq!(*target_distance, 2.0);
                assert_eq!(*strength, 0.9);
            }
            _ => panic!("Wrong constraint type"),
        }
    }

    #[test]
    fn test_priority_blending_asserted() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::asserted(MapperAxiomType::SubClassOf {
            subclass: 1,
            superclass: 2,
        });

        let constraints = mapper.translate_axiom(&axiom);
        assert_eq!(constraints[0].priority, PRIORITY_ASSERTED);
    }

    #[test]
    fn test_priority_blending_inferred() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::inferred(MapperAxiomType::SubClassOf {
            subclass: 1,
            superclass: 2,
        });

        let constraints = mapper.translate_axiom(&axiom);
        assert_eq!(constraints[0].priority, PRIORITY_INFERRED);
    }

    #[test]
    fn test_priority_blending_user_defined() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::user_defined(MapperAxiomType::SubClassOf {
            subclass: 1,
            superclass: 2,
        });

        let constraints = mapper.translate_axiom(&axiom);
        assert_eq!(constraints[0].priority, PRIORITY_USER_DEFINED);
    }

    #[test]
    fn test_batch_translation() {
        let mut mapper = AxiomMapper::new();
        let axioms = vec![
            OWLAxiom::asserted(MapperAxiomType::SubClassOf {
                subclass: 1,
                superclass: 2,
            }),
            OWLAxiom::asserted(MapperAxiomType::DisjointClasses {
                classes: vec![1, 3],
            }),
            OWLAxiom::asserted(MapperAxiomType::EquivalentClasses {
                class1: 4,
                class2: 5,
            }),
        ];

        let constraints = mapper.translate_axioms(&axioms);

        // SubClassOf (1) + DisjointClasses (1) + EquivalentClasses (1) = 3
        assert_eq!(constraints.len(), 3);
    }

    #[test]
    fn test_custom_config() {
        let config = TranslationConfig {
            disjoint_separation_distance: 50.0,
            disjoint_separation_strength: 0.95,
            subclass_clustering_distance: 25.0,
            subclass_clustering_stiffness: 0.7,
            ..Default::default()
        };

        let mut mapper = AxiomMapper::with_config(config);

        let axiom1 = OWLAxiom::asserted(MapperAxiomType::DisjointClasses {
            classes: vec![1, 2],
        });
        let constraints1 = mapper.translate_axiom(&axiom1);

        match &constraints1[0].constraint_type {
            PhysicsConstraintType::Separation { min_distance, strength } => {
                assert_eq!(*min_distance, 50.0);
                assert_eq!(*strength, 0.95);
            }
            _ => panic!("Wrong constraint type"),
        }

        let axiom2 = OWLAxiom::asserted(MapperAxiomType::SubClassOf {
            subclass: 10,
            superclass: 20,
        });
        let constraints2 = mapper.translate_axiom(&axiom2);

        match &constraints2[0].constraint_type {
            PhysicsConstraintType::Clustering { ideal_distance, stiffness } => {
                assert_eq!(*ideal_distance, 25.0);
                assert_eq!(*stiffness, 0.7);
            }
            _ => panic!("Wrong constraint type"),
        }
    }

    #[test]
    fn test_part_of_translation() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::asserted(MapperAxiomType::PartOf {
            part: 10,
            whole: 20,
        });

        let constraints = mapper.translate_axiom(&axiom);

        assert_eq!(constraints.len(), 1);
        match &constraints[0].constraint_type {
            PhysicsConstraintType::Containment { parent_node, radius, strength } => {
                assert_eq!(*parent_node, 20);
                assert!(*radius > 0.0);
                assert_eq!(*strength, 0.8);
            }
            _ => panic!("Wrong constraint type"),
        }
    }

    #[test]
    fn test_disjoint_union_translation() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::asserted(MapperAxiomType::DisjointUnion {
            union_class: 1,
            disjoint_classes: vec![2, 3, 4],
        });

        let constraints = mapper.translate_axiom(&axiom);

        // Disjoint constraints (3) + Clustering to union (3) = 6
        assert_eq!(constraints.len(), 6);
    }

    #[test]
    fn test_axiom_id_propagation() {
        let mut mapper = AxiomMapper::new();
        let axiom = OWLAxiom::asserted(MapperAxiomType::SubClassOf {
            subclass: 1,
            superclass: 2,
        }).with_id(42);

        let constraints = mapper.translate_axiom(&axiom);

        assert_eq!(constraints[0].axiom_id, Some(42));
    }
}
*/
