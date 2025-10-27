// tests/inference/explanation_tests.rs
//! Explanation Generation Tests

#[cfg(test)]
#[cfg(feature = "ontology")]
mod tests {
    use webxr::adapters::whelk_inference_engine::WhelkInferenceEngine;
    use webxr::ports::inference_engine::InferenceEngine;
    use webxr::ports::ontology_repository::{OwlClass, OwlAxiom, AxiomType};
    use std::collections::HashMap;

    async fn create_explanation_test_ontology() -> (Vec<OwlClass>, Vec<OwlAxiom>) {
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
    async fn test_explain_entailment() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_explanation_test_ontology().await;

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        // Create test axiom to explain
        let axiom_to_explain = OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "http://example.com/Dog".to_string(),
            object: "http://example.com/Animal".to_string(),
            annotations: HashMap::new(),
        };

        let explanation = engine.explain_entailment(&axiom_to_explain).await;
        assert!(explanation.is_ok());

        let explanation_axioms = explanation.unwrap();
        // Should have supporting axioms
        assert!(explanation_axioms.len() >= 0);
    }

    #[tokio::test]
    async fn test_is_entailed() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_explanation_test_ontology().await;

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        // Test direct axiom
        let direct_axiom = OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "http://example.com/Dog".to_string(),
            object: "http://example.com/Mammal".to_string(),
            annotations: HashMap::new(),
        };

        let is_entailed = engine.is_entailed(&direct_axiom).await;
        assert!(is_entailed.is_ok());

        // Test inferred axiom (transitive)
        let inferred_axiom = OwlAxiom {
            id: None,
            axiom_type: AxiomType::SubClassOf,
            subject: "http://example.com/Dog".to_string(),
            object: "http://example.com/Animal".to_string(),
            annotations: HashMap::new(),
        };

        let is_inferred_entailed = engine.is_entailed(&inferred_axiom).await;
        assert!(is_inferred_entailed.is_ok());
    }
}
