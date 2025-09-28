//! Comprehensive error handling for AutoSchemaKG
//!
//! This module provides a unified error type that can represent all kinds of errors
//! that might occur in the AutoSchemaKG system.

use std::fmt;
use thiserror::Error;

/// The main error type for AutoSchemaKG operations
#[derive(Error, Debug)]
pub enum AutoSchemaError {
    /// IO operation failed
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization failed
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// CSV processing failed
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// Text processing error
    #[error("Text processing error: {message}")]
    TextProcessing { message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Validation error
    #[error("Validation error: {field}: {message}")]
    Validation { field: String, message: String },

    /// Network request failed
    #[error("Network error: {0}")]
    Network(String),

    /// Database operation failed
    #[error("Database error: {0}")]
    Database(String),

    /// LLM generation failed
    #[error("LLM generation error: {0}")]
    LlmGeneration(String),

    /// Vector operation failed
    #[error("Vector operation error: {0}")]
    VectorOperation(String),

    /// Knowledge graph construction failed
    #[error("KG construction error: {0}")]
    KgConstruction(String),

    /// Retrieval failed
    #[error("Retrieval error: {0}")]
    Retrieval(String),

    /// Authentication/authorization failed
    #[error("Auth error: {0}")]
    Auth(String),

    /// Rate limiting error
    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    /// Timeout error
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Resource not found
    #[error("Resource not found: {resource_type} '{id}'")]
    NotFound { resource_type: String, id: String },

    /// Resource already exists
    #[error("Resource already exists: {resource_type} '{id}'")]
    AlreadyExists { resource_type: String, id: String },

    /// Generic error with context
    #[error("Error: {message}")]
    Generic { message: String },
}

impl AutoSchemaError {
    /// Create a new text processing error
    pub fn text_processing<S: Into<String>>(message: S) -> Self {
        Self::TextProcessing {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new validation error
    pub fn validation<S: Into<String>>(field: S, message: S) -> Self {
        Self::Validation {
            field: field.into(),
            message: message.into(),
        }
    }

    /// Create a new network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::Network(message.into())
    }

    /// Create a new database error
    pub fn database<S: Into<String>>(message: S) -> Self {
        Self::Database(message.into())
    }

    /// Create a new LLM generation error
    pub fn llm_generation<S: Into<String>>(message: S) -> Self {
        Self::LlmGeneration(message.into())
    }

    /// Create a new vector operation error
    pub fn vector_operation<S: Into<String>>(message: S) -> Self {
        Self::VectorOperation(message.into())
    }

    /// Create a new knowledge graph construction error
    pub fn kg_construction<S: Into<String>>(message: S) -> Self {
        Self::KgConstruction(message.into())
    }

    /// Create a new retrieval error
    pub fn retrieval<S: Into<String>>(message: S) -> Self {
        Self::Retrieval(message.into())
    }

    /// Create a not found error
    pub fn not_found<S: Into<String>>(resource_type: S, id: S) -> Self {
        Self::NotFound {
            resource_type: resource_type.into(),
            id: id.into(),
        }
    }

    /// Create an already exists error
    pub fn already_exists<S: Into<String>>(resource_type: S, id: S) -> Self {
        Self::AlreadyExists {
            resource_type: resource_type.into(),
            id: id.into(),
        }
    }

    /// Create a generic error
    pub fn generic<S: Into<String>>(message: S) -> Self {
        Self::Generic {
            message: message.into(),
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Network(_) | Self::Timeout(_) | Self::RateLimit(_) | Self::Database(_)
        )
    }

    /// Get the error category for metrics/logging
    pub fn category(&self) -> &'static str {
        match self {
            Self::Io(_) => "io",
            Self::Json(_) => "serialization",
            Self::Csv(_) => "csv",
            Self::TextProcessing { .. } => "text_processing",
            Self::Configuration { .. } => "configuration",
            Self::Validation { .. } => "validation",
            Self::Network(_) => "network",
            Self::Database(_) => "database",
            Self::LlmGeneration(_) => "llm_generation",
            Self::VectorOperation(_) => "vector_operation",
            Self::KgConstruction(_) => "kg_construction",
            Self::Retrieval(_) => "retrieval",
            Self::Auth(_) => "auth",
            Self::RateLimit(_) => "rate_limit",
            Self::Timeout(_) => "timeout",
            Self::NotFound { .. } => "not_found",
            Self::AlreadyExists { .. } => "already_exists",
            Self::Generic { .. } => "generic",
        }
    }
}

/// Convenient Result type alias
pub type Result<T> = std::result::Result<T, AutoSchemaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = AutoSchemaError::text_processing("test message");
        assert_eq!(err.category(), "text_processing");
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_retryable_errors() {
        let network_err = AutoSchemaError::network("connection failed");
        assert!(network_err.is_retryable());

        let validation_err = AutoSchemaError::validation("field", "invalid");
        assert!(!validation_err.is_retryable());
    }
}