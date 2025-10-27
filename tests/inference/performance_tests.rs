// tests/inference/performance_tests.rs
//! Performance Benchmarks for Inference

#[cfg(test)]
#[cfg(feature = "ontology")]
mod tests {
    use webxr::adapters::whelk_inference_engine::WhelkInferenceEngine;
    use webxr::ports::inference_engine::InferenceEngine;
    use webxr::ports::ontology_repository::{OwlClass, OwlAxiom, AxiomType};
    use std::collections::HashMap;

    fn create_large_ontology(class_count: usize, depth: usize) -> (Vec<OwlClass>, Vec<OwlAxiom>) {
        let mut classes = Vec::new();
        let mut axioms = Vec::new();

        // Create hierarchy
        for i in 0..class_count {
            let class = OwlClass {
                id: None,
                iri: format!("http://example.com/Class{}", i),
                label: Some(format!("Class {}", i)),
                description: None,
                parent_iri: None,
                deprecated: false,
            };
            classes.push(class);

            // Create subsumption axioms
            if i > 0 && i % (class_count / depth) != 0 {
                let parent_idx = i - 1;
                axioms.push(OwlAxiom {
                    id: None,
                    axiom_type: AxiomType::SubClassOf,
                    subject: format!("http://example.com/Class{}", i),
                    object: format!("http://example.com/Class{}", parent_idx),
                    annotations: HashMap::new(),
                });
            }
        }

        (classes, axioms)
    }

    #[tokio::test]
    async fn test_inference_performance_small() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_large_ontology(10, 3);

        let start = std::time::Instant::now();

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        let elapsed = start.elapsed();
        println!("Small ontology (10 classes) inference time: {:?}", elapsed);

        // Should complete in reasonable time
        assert!(elapsed.as_secs() < 5);
    }

    #[tokio::test]
    async fn test_inference_performance_medium() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_large_ontology(100, 5);

        let start = std::time::Instant::now();

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        let elapsed = start.elapsed();
        println!("Medium ontology (100 classes) inference time: {:?}", elapsed);

        assert!(elapsed.as_secs() < 10);
    }

    #[tokio::test]
    async fn test_cache_performance() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_large_ontology(50, 4);

        // First run (cold)
        let start = std::time::Instant::now();
        engine.load_ontology(classes.clone(), axioms.clone()).await.unwrap();
        engine.infer().await.unwrap();
        let cold_time = start.elapsed();

        // Second run (cached)
        let start = std::time::Instant::now();
        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();
        let cached_time = start.elapsed();

        println!("Cold inference: {:?}, Cached inference: {:?}", cold_time, cached_time);

        // Cached should be faster or similar
        assert!(cached_time <= cold_time);
    }

    #[tokio::test]
    async fn test_statistics_overhead() {
        let mut engine = WhelkInferenceEngine::new();
        let (classes, axioms) = create_large_ontology(20, 3);

        engine.load_ontology(classes, axioms).await.unwrap();
        engine.infer().await.unwrap();

        let start = std::time::Instant::now();

        for _ in 0..100 {
            let _ = engine.get_statistics().await;
        }

        let elapsed = start.elapsed();
        println!("100 statistics calls: {:?}", elapsed);

        // Should be very fast
        assert!(elapsed.as_millis() < 100);
    }
}
