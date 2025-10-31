/// Reasoning module for OWL ontology inference and constraint translation
///
/// This module provides:
/// - Custom OWL reasoner with hash-based class hierarchy (O(log n) lookups)
/// - Horned-OWL integration for advanced reasoning
/// - Inference caching with checksum-based invalidation
/// - Actix actor for background reasoning tasks

pub mod custom_reasoner;
pub mod horned_integration;
pub mod inference_cache;
pub mod reasoning_actor;

// Re-export main types
pub use custom_reasoner::{CustomReasoner, InferredAxiom, OntologyReasoner};
pub use horned_integration::HornedOwlReasoner;
pub use inference_cache::{InferenceCache, CachedInference};
pub use reasoning_actor::{ReasoningActor, ReasoningMessage, TriggerReasoning, GetInferredAxioms};

/// Result type for reasoning operations
pub type ReasoningResult<T> = Result<T, ReasoningError>;

/// Reasoning error types
#[derive(Debug, thiserror::Error)]
pub enum ReasoningError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Ontology parsing error: {0}")]
    Parsing(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Cache error: {0}")]
    Cache(String),

    #[error("Actor error: {0}")]
    Actor(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
