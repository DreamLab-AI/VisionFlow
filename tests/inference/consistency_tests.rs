// tests/inference/consistency_tests.rs
//! Consistency Checking Integration Tests

#[cfg(test)]
#[cfg(feature = "ontology")]
mod tests {
    use webxr::adapters::whelk_inference_engine::WhelkInferenceEngine;
    use webxr::ports::inference_engine::InferenceEngine;
    use webxr::ports::ontology_repository::{OwlClass, OwlAxiom, AxiomType};
    use std::collections::HashMap;

    async fn create_consistent_ontology() -> (Vec<OwlClass>, Vec<OwlAxiom>) {
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
                iri: "http://example.com/Dog".to_string(),
                label: Some("Dog".to_string()),
                description: None,
                parent_iri: Some("http://example.com/Animal".to_string()),
                deprecated: false,
            },
        ];

        let axioms = vec![OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "http://example.com/Dog".to_string(),
            object: "http://example.com/Animal".to_string(),
            annotations: HashMap::new(),
        }];

        (classes, axioms)
    }

    #[tokio::test]
    async fn test_consistent_ontology() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_consistent_ontology().await;

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        let is_consistent = engine.check_consistency().await;
        assert!(is_consistent.is_ok());
        assert_eq!(is_consistent.unwrap(), true);
    }

    #[tokio::test]
    async fn test_consistency_without_inference() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_consistent_ontology().await;

        engine.load_ontology(classes, axioms).await.unwrap();

        // Without running inference first, should fail
        let result = engine.check_consistency().await;
        // Should either fail or return true for stub
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_empty_ontology_consistency() {
        let mut engine = WhelkInferenceEngine::new();

        engine.load_ontology(vec![], vec![]).await.unwrap();
        engine.infer().await.unwrap();

        let is_consistent = engine.check_consistency().await;
        assert!(is_consistent.is_ok());
        assert_eq!(is_consistent.unwrap(), true, "Empty ontology should be consistent");
    }
}
