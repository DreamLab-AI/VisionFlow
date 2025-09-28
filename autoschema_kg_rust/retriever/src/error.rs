//! Error handling for the retriever module
//!
//! Provides comprehensive error types for all retrieval operations

use thiserror::Error;

/// Result type alias for retriever operations
pub type Result<T> = std::result::Result<T, RetrieverError>;

/// Comprehensive error types for retrieval operations
#[derive(Error, Debug)]
pub enum RetrieverError {
    #[error("Configuration error: {message}")]
    Config { message: String },

    #[error("Vector index error: {message}")]
    VectorIndex { message: String },

    #[error("Embedding generation failed: {message}")]
    Embedding { message: String },

    #[error("Graph traversal error: {message}")]
    GraphTraversal { message: String },

    #[error("Query processing error: {message}")]
    QueryProcessing { message: String },

    #[error("Search operation failed: {message}")]
    SearchFailed { message: String },

    #[error("Ranking error: {message}")]
    Ranking { message: String },

    #[error("Cache operation failed: {message}")]
    Cache { message: String },

    #[error("Context management error: {message}")]
    Context { message: String },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Timeout occurred during operation: {operation}")]
    Timeout { operation: String },

    #[error("Resource not found: {resource}")]
    NotFound { resource: String },

    #[error("Concurrent access error: {message}")]
    Concurrency { message: String },

    #[error("Memory allocation failed: {message}")]
    Memory { message: String },

    #[error("Model loading error: {model_path}")]
    ModelLoad { model_path: String },

    #[error("Unknown error: {message}")]
    Unknown { message: String },
}

impl RetrieverError {
    /// Create a configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::Config { message: message.into() }
    }

    /// Create a vector index error
    pub fn vector_index<S: Into<String>>(message: S) -> Self {
        Self::VectorIndex { message: message.into() }
    }

    /// Create an embedding error
    pub fn embedding<S: Into<String>>(message: S) -> Self {
        Self::Embedding { message: message.into() }
    }

    /// Create a graph traversal error
    pub fn graph_traversal<S: Into<String>>(message: S) -> Self {
        Self::GraphTraversal { message: message.into() }
    }

    /// Create a query processing error
    pub fn query_processing<S: Into<String>>(message: S) -> Self {
        Self::QueryProcessing { message: message.into() }
    }

    /// Create a search failure error
    pub fn search_failed<S: Into<String>>(message: S) -> Self {
        Self::SearchFailed { message: message.into() }
    }

    /// Create a ranking error
    pub fn ranking<S: Into<String>>(message: S) -> Self {
        Self::Ranking { message: message.into() }
    }

    /// Create a cache error
    pub fn cache<S: Into<String>>(message: S) -> Self {
        Self::Cache { message: message.into() }
    }

    /// Create a context management error
    pub fn context<S: Into<String>>(message: S) -> Self {
        Self::Context { message: message.into() }
    }

    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInput { message: message.into() }
    }

    /// Create a timeout error
    pub fn timeout<S: Into<String>>(operation: S) -> Self {
        Self::Timeout { operation: operation.into() }
    }

    /// Create a not found error
    pub fn not_found<S: Into<String>>(resource: S) -> Self {
        Self::NotFound { resource: resource.into() }
    }

    /// Create a concurrency error
    pub fn concurrency<S: Into<String>>(message: S) -> Self {
        Self::Concurrency { message: message.into() }
    }

    /// Create a memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory { message: message.into() }
    }

    /// Create a model loading error
    pub fn model_load<S: Into<String>>(model_path: S) -> Self {
        Self::ModelLoad { model_path: model_path.into() }
    }

    /// Create an unknown error
    pub fn unknown<S: Into<String>>(message: S) -> Self {
        Self::Unknown { message: message.into() }
    }

    /// Check if the error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::Timeout { .. } |
            Self::Cache { .. } |
            Self::Concurrency { .. } |
            Self::Io(_)
        )
    }

    /// Get error category for metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config",
            Self::VectorIndex { .. } => "vector_index",
            Self::Embedding { .. } => "embedding",
            Self::GraphTraversal { .. } => "graph_traversal",
            Self::QueryProcessing { .. } => "query_processing",
            Self::SearchFailed { .. } => "search",
            Self::Ranking { .. } => "ranking",
            Self::Cache { .. } => "cache",
            Self::Context { .. } => "context",
            Self::Io(_) => "io",
            Self::Serialization(_) => "serialization",
            Self::InvalidInput { .. } => "input",
            Self::Timeout { .. } => "timeout",
            Self::NotFound { .. } => "not_found",
            Self::Concurrency { .. } => "concurrency",
            Self::Memory { .. } => "memory",
            Self::ModelLoad { .. } => "model_load",
            Self::Unknown { .. } => "unknown",
        }
    }
}

/// Trait for converting errors into RetrieverError
pub trait IntoRetrieverError<T> {
    fn into_retriever_error(self) -> Result<T>;
}

impl<T, E> IntoRetrieverError<T> for std::result::Result<T, E>
where
    E: Into<RetrieverError>,
{
    fn into_retriever_error(self) -> Result<T> {
        self.map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = RetrieverError::config("test message");
        assert!(matches!(err, RetrieverError::Config { .. }));
        assert_eq!(err.category(), "config");
    }

    #[test]
    fn test_error_recoverability() {
        let timeout_err = RetrieverError::timeout("search");
        assert!(timeout_err.is_recoverable());

        let config_err = RetrieverError::config("invalid");
        assert!(!config_err.is_recoverable());
    }

    #[test]
    fn test_error_categories() {
        let errors = vec![
            RetrieverError::config("test"),
            RetrieverError::vector_index("test"),
            RetrieverError::embedding("test"),
            RetrieverError::graph_traversal("test"),
        ];

        let categories: Vec<_> = errors.iter().map(|e| e.category()).collect();
        assert_eq!(categories, vec!["config", "vector_index", "embedding", "graph_traversal"]);
    }
}