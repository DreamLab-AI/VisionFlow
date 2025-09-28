//! Error types for the vectorstore crate

use thiserror::Error;

/// Result type alias for vectorstore operations
pub type Result<T> = std::result::Result<T, VectorError>;

/// Comprehensive error types for vector operations
#[derive(Error, Debug)]
pub enum VectorError {
    #[error("Embedding model error: {message}")]
    EmbeddingError { message: String },

    #[error("Index error: {message}")]
    IndexError { message: String },

    #[error("Storage error: {message}")]
    StorageError { message: String },

    #[error("Search error: {message}")]
    SearchError { message: String },

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Vector not found: {id}")]
    VectorNotFound { id: String },

    #[error("Invalid configuration: {message}")]
    ConfigError { message: String },

    #[error("GPU error: {message}")]
    GpuError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },

    #[error("Bincode error: {source}")]
    BincodeError {
        #[from]
        source: bincode::Error,
    },

    #[error("Async runtime error: {message}")]
    RuntimeError { message: String },

    #[error("Memory allocation error: {message}")]
    MemoryError { message: String },

    #[error("Concurrent access error: {message}")]
    ConcurrencyError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },
}

impl VectorError {
    pub fn embedding_error<S: Into<String>>(message: S) -> Self {
        Self::EmbeddingError {
            message: message.into(),
        }
    }

    pub fn index_error<S: Into<String>>(message: S) -> Self {
        Self::IndexError {
            message: message.into(),
        }
    }

    pub fn storage_error<S: Into<String>>(message: S) -> Self {
        Self::StorageError {
            message: message.into(),
        }
    }

    pub fn search_error<S: Into<String>>(message: S) -> Self {
        Self::SearchError {
            message: message.into(),
        }
    }

    pub fn config_error<S: Into<String>>(message: S) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    pub fn gpu_error<S: Into<String>>(message: S) -> Self {
        Self::GpuError {
            message: message.into(),
        }
    }

    pub fn serialization_error<S: Into<String>>(message: S) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    pub fn runtime_error<S: Into<String>>(message: S) -> Self {
        Self::RuntimeError {
            message: message.into(),
        }
    }

    pub fn memory_error<S: Into<String>>(message: S) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    pub fn concurrency_error<S: Into<String>>(message: S) -> Self {
        Self::ConcurrencyError {
            message: message.into(),
        }
    }

    pub fn network_error<S: Into<String>>(message: S) -> Self {
        Self::NetworkError {
            message: message.into(),
        }
    }
}

// Conversions from external crates
#[cfg(feature = "faiss-backend")]
impl From<faiss::Error> for VectorError {
    fn from(err: faiss::Error) -> Self {
        Self::IndexError {
            message: format!("FAISS error: {}", err),
        }
    }
}

#[cfg(feature = "gpu")]
impl From<cuda::CudaError> for VectorError {
    fn from(err: cuda::CudaError) -> Self {
        Self::GpuError {
            message: format!("CUDA error: {:?}", err),
        }
    }
}

impl From<tokio::task::JoinError> for VectorError {
    fn from(err: tokio::task::JoinError) -> Self {
        Self::RuntimeError {
            message: format!("Task join error: {}", err),
        }
    }
}

impl From<anyhow::Error> for VectorError {
    fn from(err: anyhow::Error) -> Self {
        Self::RuntimeError {
            message: err.to_string(),
        }
    }
}

impl From<fs_extra::error::Error> for VectorError {
    fn from(err: fs_extra::error::Error) -> Self {
        Self::StorageError {
            message: format!("File operation error: {}", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = VectorError::embedding_error("Test error");
        assert!(matches!(err, VectorError::EmbeddingError { .. }));
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = VectorError::DimensionMismatch {
            expected: 768,
            actual: 512,
        };
        assert_eq!(
            err.to_string(),
            "Dimension mismatch: expected 768, got 512"
        );
    }
}