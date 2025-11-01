// src/ports/inference_engine.rs
//! Inference Engine Port
//!
//! Provides ontology reasoning and inference capabilities using whelk-rs or similar reasoners.
//! This port abstracts the specific reasoning engine implementation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::ports::ontology_repository::{InferenceResults, OwlAxiom, OwlClass};

pub type Result<T> = std::result::Result<T, InferenceEngineError>;

#[derive(Debug, thiserror::Error)]
pub enum InferenceEngineError {
    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Ontology not loaded")]
    OntologyNotLoaded,

    #[error("Inconsistent ontology: {0}")]
    InconsistentOntology(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Reasoner error: {0}")]
    ReasonerError(String),
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStatistics {
    pub loaded_classes: usize,
    pub loaded_axioms: usize,
    pub inferred_axioms: usize,
    pub last_inference_time_ms: u64,
    pub total_inferences: u64,
}

/
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    
    async fn load_ontology(&mut self, classes: Vec<OwlClass>, axioms: Vec<OwlAxiom>) -> Result<()>;

    
    async fn infer(&mut self) -> Result<InferenceResults>;

    
    async fn is_entailed(&self, axiom: &OwlAxiom) -> Result<bool>;

    
    
    async fn get_subclass_hierarchy(&self) -> Result<Vec<(String, String)>>;

    
    
    async fn classify_instance(&self, instance_iri: &str) -> Result<Vec<String>>;

    
    async fn check_consistency(&self) -> Result<bool>;

    
    
    async fn explain_entailment(&self, axiom: &OwlAxiom) -> Result<Vec<OwlAxiom>>;

    
    async fn clear(&mut self) -> Result<()>;

    
    async fn get_statistics(&self) -> Result<InferenceStatistics>;
}
