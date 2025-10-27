// tests/inference/classification_tests.rs
//! Classification Integration Tests

#[cfg(test)]
#[cfg(feature = "ontology")]
mod tests {
    use webxr::adapters::whelk_inference_engine::WhelkInferenceEngine;
    use webxr::ports::inference_engine::InferenceEngine;
    use webxr::ports::ontology_repository::{OwlClass, OwlAxiom, AxiomType};
    use std::collections::HashMap;

    async fn create_test_ontology() -> (Vec<OwlClass>, Vec<OwlAxiom>) {
        let classes = vec![
            OwlClass {
                id: None,
                iri: "http://example.com/Animal".to_string(),
                label: Some("Animal".to_string()),
                description: None,
                parent_iri: None,
                deprecated: false,
            },
            OwlClass {
                id: None,
                iri: "http://example.com/Mammal".to_string(),
                label: Some("Mammal".to_string()),
                description: None,
                parent_iri: Some("http://example.com/Animal".to_string()),
                deprecated: false,
            },
            OwlClass {
                id: None,
                iri: "http://example.com/Dog".to_string(),
                label: Some("Dog".to_string()),
                description: None,
                parent_iri: Some("http://example.com/Mammal".to_string()),
                deprecated: false,
            },
        ];

        let axioms = vec![
            OwlAxiom {
                id: None,
                axiom_type: AxiomType::SubClassOf,
                subject: "http://example.com/Mammal".to_string(),
                object: "http://example.com/Animal".to_string(),
                annotations: HashMap::new(),
            },
            OwlAxiom {
                id: None,
                axiom_type: AxiomType::SubClassOf,
                subject: "http://example.com/Dog".to_string(),
                object: "http://example.com/Mammal".to_string(),
                annotations: HashMap::new(),
            },
        ];

        (classes, axioms)
    }

    #[tokio::test]
    async fn test_classification_basic() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_test_ontology().await;

        let load_result = engine.load_ontology(classes, axioms).await;
        assert!(load_result.is_ok());

        let infer_result = engine.infer().await;
        assert!(infer_result.is_ok());

        let inference_results = infer_result.unwrap();
        assert!(inference_results.inferred_axioms.len() > 0);
    }

    #[tokio::test]
    async fn test_get_subclass_hierarchy() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_test_ontology().await;

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        let hierarchy = engine.get_subclass_hierarchy().await;
        assert!(hierarchy.is_ok());

        let hierarchy_vec = hierarchy.unwrap();
        assert!(hierarchy_vec.len() > 0);

        // Should include transitive subsumptions (Dog -> Animal)
        let has_dog_animal = hierarchy_vec.iter().any(|(child, parent)| {
            child.contains("Dog") && parent.contains("Animal")
        });

        assert!(has_dog_animal, "Should infer transitive subsumption");
    }

    #[tokio::test]
    async fn test_classify_instance() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_test_ontology().await;

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        let dog_classes = engine
            .classify_instance("http://example.com/Dog")
            .await;

        assert!(dog_classes.is_ok());

        let classes = dog_classes.unwrap();
        assert!(classes.iter().any(|c| c.contains("Mammal")));
        assert!(classes.iter().any(|c| c.contains("Animal")));
    }

    #[tokio::test]
    async fn test_classification_caching() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_test_ontology().await;

        engine.load_ontology(classes.clone(), axioms.clone()).await.unwrap();

        // First inference
        let result1 = engine.infer().await.unwrap();
        let time1 = result1.inference_time_ms;

        // Second inference (should use cache)
        engine.load_ontology(classes, axioms).await.unwrap();
        let result2 = engine.infer().await.unwrap();
        let time2 = result2.inference_time_ms;

        // Cached inference should be faster
        assert!(time2 <= time1);
    }

    #[tokio::test]
    async fn test_statistics() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_test_ontology().await;

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        let stats = engine.get_statistics().await;
        assert!(stats.is_ok());

        let statistics = stats.unwrap();
        assert_eq!(statistics.loaded_classes, 3);
        assert_eq!(statistics.loaded_axioms, 2);
        assert!(statistics.inferred_axioms > 0);
        assert!(statistics.last_inference_time_ms > 0);
    }
}
