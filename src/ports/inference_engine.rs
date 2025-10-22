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

/// Inference engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStatistics {
    pub loaded_classes: usize,
    pub loaded_axioms: usize,
    pub inferred_axioms: usize,
    pub last_inference_time_ms: u64,
    pub total_inferences: u64,
}

/// Port for ontology inference operations
#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Load ontology for reasoning
    async fn load_ontology(&mut self, classes: Vec<OwlClass>, axioms: Vec<OwlAxiom>) -> Result<()>;

    /// Perform inference to derive new axioms
    async fn infer(&mut self) -> Result<InferenceResults>;

    /// Check if a specific axiom is entailed by the ontology
    async fn is_entailed(&self, axiom: &OwlAxiom) -> Result<bool>;

    /// Get all inferred subclass relationships
    /// Returns Vec<(child_iri, parent_iri)>
    async fn get_subclass_hierarchy(&self) -> Result<Vec<(String, String)>>;

    /// Classify instances into classes
    /// Returns class IRIs that the instance belongs to
    async fn classify_instance(&self, instance_iri: &str) -> Result<Vec<String>>;

    /// Check ontology consistency
    async fn check_consistency(&self) -> Result<bool>;

    /// Explain why an axiom is entailed
    /// Returns the axioms that support the entailment
    async fn explain_entailment(&self, axiom: &OwlAxiom) -> Result<Vec<OwlAxiom>>;

    /// Clear loaded ontology
    async fn clear(&mut self) -> Result<()>;

    /// Get inference engine statistics
    async fn get_statistics(&self) -> Result<InferenceStatistics>;
}
