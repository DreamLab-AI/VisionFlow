// src/adapters/whelk_inference_stub.rs
//! Whelk Inference Engine Stub
//!
//! Stub implementation of the InferenceEngine port for Phase 2.2.
//! This allows compilation and integration testing while Phase 7 implements
//! the full whelk-rs integration for OWL ontology reasoning.

use async_trait::async_trait;
use log::{debug, warn};

use crate::ports::inference_engine::{InferenceEngine, InferenceStatistics, Result as PortResult};
use crate::ports::ontology_repository::{InferenceResults, OwlAxiom, OwlClass};
use crate::utils::time;

///
///
///
///
pub struct WhelkInferenceEngineStub {
    
    loaded_classes: Vec<OwlClass>,

    
    loaded_axioms: Vec<OwlAxiom>,

    
    is_loaded: bool,

    
    total_inferences: u64,
}

impl WhelkInferenceEngineStub {
    
    pub fn new() -> Self {
        warn!("WhelkInferenceEngineStub: Using stub implementation - Phase 7 will implement full reasoning");
        Self {
            loaded_classes: Vec::new(),
            loaded_axioms: Vec::new(),
            is_loaded: false,
            total_inferences: 0,
        }
    }
}

impl Default for WhelkInferenceEngineStub {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl InferenceEngine for WhelkInferenceEngineStub {
    
    async fn load_ontology(
        &mut self,
        classes: Vec<OwlClass>,
        axioms: Vec<OwlAxiom>,
    ) -> PortResult<()> {
        debug!(
            "WhelkInferenceEngineStub: Loading {} classes and {} axioms (stub)",
            classes.len(),
            axioms.len()
        );

        self.loaded_classes = classes;
        self.loaded_axioms = axioms;
        self.is_loaded = true;

        Ok(())
    }

    
    
    
    async fn infer(&mut self) -> PortResult<InferenceResults> {
        debug!("WhelkInferenceEngineStub: Performing inference (stub - returns empty)");

        if !self.is_loaded {
            return Err(crate::ports::inference_engine::InferenceEngineError::OntologyNotLoaded);
        }

        self.total_inferences += 1;

        Ok(InferenceResults {
            timestamp: time::now(),
            inferred_axioms: Vec::new(),
            inference_time_ms: 0,
            reasoner_version: "whelk-stub".to_string(),
        })
    }

    
    async fn is_entailed(&self, axiom: &OwlAxiom) -> PortResult<bool> {
        debug!("WhelkInferenceEngineStub: Checking entailment (stub - returns false)");

        if !self.is_loaded {
            return Err(crate::ports::inference_engine::InferenceEngineError::OntologyNotLoaded);
        }

        
        Ok(false)
    }

    
    async fn get_subclass_hierarchy(&self) -> PortResult<Vec<(String, String)>> {
        debug!("WhelkInferenceEngineStub: Getting subclass hierarchy (stub - returns empty)");

        if !self.is_loaded {
            return Err(crate::ports::inference_engine::InferenceEngineError::OntologyNotLoaded);
        }

        Ok(Vec::new())
    }

    
    async fn classify_instance(&self, instance_iri: &str) -> PortResult<Vec<String>> {
        debug!(
            "WhelkInferenceEngineStub: Classifying instance {} (stub - returns empty)",
            instance_iri
        );

        if !self.is_loaded {
            return Err(crate::ports::inference_engine::InferenceEngineError::OntologyNotLoaded);
        }

        Ok(Vec::new())
    }

    
    async fn check_consistency(&self) -> PortResult<bool> {
        debug!("WhelkInferenceEngineStub: Checking consistency (stub - returns true)");

        if !self.is_loaded {
            return Err(crate::ports::inference_engine::InferenceEngineError::OntologyNotLoaded);
        }

        
        Ok(true)
    }

    
    async fn explain_entailment(&self, axiom: &OwlAxiom) -> PortResult<Vec<OwlAxiom>> {
        debug!("WhelkInferenceEngineStub: Explaining entailment (stub - returns empty)");

        if !self.is_loaded {
            return Err(crate::ports::inference_engine::InferenceEngineError::OntologyNotLoaded);
        }

        Ok(Vec::new())
    }

    
    async fn clear(&mut self) -> PortResult<()> {
        debug!("WhelkInferenceEngineStub: Clearing ontology");

        self.loaded_classes.clear();
        self.loaded_axioms.clear();
        self.is_loaded = false;

        Ok(())
    }

    
    async fn get_statistics(&self) -> PortResult<InferenceStatistics> {
        debug!("WhelkInferenceEngineStub: Getting statistics");

        Ok(InferenceStatistics {
            loaded_classes: self.loaded_classes.len(),
            loaded_axioms: self.loaded_axioms.len(),
            inferred_axioms: 0, 
            last_inference_time_ms: 0,
            total_inferences: self.total_inferences,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stub_creation() {
        let engine = WhelkInferenceEngineStub::new();
        assert!(!engine.is_loaded);
        assert_eq!(engine.total_inferences, 0);
    }

    #[tokio::test]
    async fn test_stub_load_ontology() {
        let mut engine = WhelkInferenceEngineStub::new();

        let classes = vec![];
        let axioms = vec![];

        let result = engine.load_ontology(classes, axioms).await;
        assert!(result.is_ok());
        assert!(engine.is_loaded);
    }

    #[tokio::test]
    async fn test_stub_infer_without_load() {
        let mut engine = WhelkInferenceEngineStub::new();

        let result = engine.infer().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_stub_infer_with_load() {
        let mut engine = WhelkInferenceEngineStub::new();
        engine.load_ontology(vec![], vec![]).await.unwrap();

        let result = engine.infer().await;
        assert!(result.is_ok());
        assert_eq!(engine.total_inferences, 1);

        let inference_result = result.unwrap();
        assert_eq!(inference_result.inferred_axioms.len(), 0);
    }

    #[tokio::test]
    async fn test_stub_consistency_check() {
        let mut engine = WhelkInferenceEngineStub::new();
        engine.load_ontology(vec![], vec![]).await.unwrap();

        let result = engine.check_consistency().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), true);
    }

    #[tokio::test]
    async fn test_stub_statistics() {
        let mut engine = WhelkInferenceEngineStub::new();
        engine.load_ontology(vec![], vec![]).await.unwrap();
        engine.infer().await.unwrap();
        engine.infer().await.unwrap();

        let stats = engine.get_statistics().await.unwrap();
        assert_eq!(stats.total_inferences, 2);
        assert_eq!(stats.inferred_axioms, 0);
    }

    #[tokio::test]
    async fn test_stub_clear() {
        let mut engine = WhelkInferenceEngineStub::new();
        engine.load_ontology(vec![], vec![]).await.unwrap();

        let result = engine.clear().await;
        assert!(result.is_ok());
        assert!(!engine.is_loaded);
    }
}
