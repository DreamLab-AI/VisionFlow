//! Error types for knowledge graph construction

use std::fmt;

/// Result type alias for KG construction operations
pub type Result<T> = std::result::Result<T, KgConstructionError>;

/// Error types that can occur during knowledge graph construction
#[derive(Debug)]
pub enum KgConstructionError {
    /// IO related errors
    Io(std::io::Error),

    /// CSV parsing errors
    Csv(csv::Error),

    /// JSON serialization/deserialization errors
    Json(serde_json::Error),

    /// Invalid node type
    InvalidNodeType(String),

    /// LLM generation errors
    LlmError(String),

    /// Graph processing errors
    GraphError(String),

    /// Configuration errors
    ConfigError(String),

    /// Threading/concurrency errors
    ThreadingError(String),

    /// Data validation errors
    ValidationError(String),

    /// Memory allocation errors
    MemoryError(String),

    /// Triple extraction specific errors
    ChunkingError(String),

    /// Dataset processing errors
    DatasetError(String),

    /// Data loader errors
    LoaderError(String),

    /// Output parsing errors
    ParsingError(String),

    /// ML inference errors
    InferenceError(String),

    /// Serialization errors
    SerializationError(String),

    /// IO errors with context
    IoError(String),

    /// Generic error with message
    Other(String),
}

impl fmt::Display for KgConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KgConstructionError::Io(err) => write!(f, "IO error: {}", err),
            KgConstructionError::Csv(err) => write!(f, "CSV error: {}", err),
            KgConstructionError::Json(err) => write!(f, "JSON error: {}", err),
            KgConstructionError::InvalidNodeType(node_type) => {
                write!(f, "Invalid node type: {}", node_type)
            }
            KgConstructionError::LlmError(msg) => write!(f, "LLM error: {}", msg),
            KgConstructionError::GraphError(msg) => write!(f, "Graph error: {}", msg),
            KgConstructionError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            KgConstructionError::ThreadingError(msg) => write!(f, "Threading error: {}", msg),
            KgConstructionError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            KgConstructionError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            KgConstructionError::ChunkingError(msg) => write!(f, "Text chunking error: {}", msg),
            KgConstructionError::DatasetError(msg) => write!(f, "Dataset processing error: {}", msg),
            KgConstructionError::LoaderError(msg) => write!(f, "Data loader error: {}", msg),
            KgConstructionError::ParsingError(msg) => write!(f, "Output parsing error: {}", msg),
            KgConstructionError::InferenceError(msg) => write!(f, "ML inference error: {}", msg),
            KgConstructionError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            KgConstructionError::IoError(msg) => write!(f, "IO error: {}", msg),
            KgConstructionError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for KgConstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            KgConstructionError::Io(err) => Some(err),
            KgConstructionError::Csv(err) => Some(err),
            KgConstructionError::Json(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for KgConstructionError {
    fn from(err: std::io::Error) -> Self {
        KgConstructionError::Io(err)
    }
}

impl From<csv::Error> for KgConstructionError {
    fn from(err: csv::Error) -> Self {
        KgConstructionError::Csv(err)
    }
}

impl From<serde_json::Error> for KgConstructionError {
    fn from(err: serde_json::Error) -> Self {
        KgConstructionError::Json(err)
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for KgConstructionError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        KgConstructionError::Other(err.to_string())
    }
}