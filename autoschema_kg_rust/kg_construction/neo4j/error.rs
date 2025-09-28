//! Error handling for Neo4j operations

use thiserror::Error;

/// Neo4j specific error types
#[derive(Error, Debug)]
pub enum Neo4jError {
    #[error("Database connection error: {0}")]
    ConnectionError(#[from] neo4rs::Error),
    
    #[error("Connection pool error: {0}")]
    PoolError(String),
    
    #[error("Query execution error: {message}")]
    QueryError { message: String },
    
    #[error("Transaction error: {0}")]
    TransactionError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    
    #[error("Timeout error: {operation}")]
    TimeoutError { operation: String },
    
    #[error("Resource not found: {resource_type} with id {id}")]
    NotFound { resource_type: String, id: String },
    
    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),
    
    #[error("Index operation failed: {0}")]
    IndexError(String),
    
    #[error("Batch operation failed: {failed_count} out of {total_count} operations")]
    BatchError { failed_count: usize, total_count: usize },
    
    #[error("URL parsing error: {0}")]
    UrlError(#[from] url::ParseError),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Generic error: {0}")]
    Other(#[from] anyhow::Error),
}

/// Type alias for Neo4j results
pub type Result<T> = std::result::Result<T, Neo4jError>;

impl Neo4jError {
    pub fn query_error<S: Into<String>>(message: S) -> Self {
        Self::QueryError {
            message: message.into(),
        }
    }
    
    pub fn timeout_error<S: Into<String>>(operation: S) -> Self {
        Self::TimeoutError {
            operation: operation.into(),
        }
    }
    
    pub fn not_found<S: Into<String>>(resource_type: S, id: S) -> Self {
        Self::NotFound {
            resource_type: resource_type.into(),
            id: id.into(),
        }
    }
    
    pub fn batch_error(failed_count: usize, total_count: usize) -> Self {
        Self::BatchError {
            failed_count,
            total_count,
        }
    }
}
