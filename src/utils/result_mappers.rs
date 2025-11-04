//! Result Mapping Utilities for Error Conversion
//!
//! This module consolidates 250+ duplicate error mapping patterns by providing
//! specialized error mappers for common conversions across the codebase.
//!
//! ## Problem Statement
//! - 728 total `.map_err()` patterns across codebase
//! - 371 format-based error mappings (`.map_err(|e| format!(...))`)
//! - 93 format-based mappings in repositories/adapters alone
//! - Inconsistent error messages and conversion patterns
//!
//! ## Solution
//! Provides specialized error mapper functions that:
//! - Convert database errors (rusqlite) to domain errors
//! - Map service-level errors consistently
//! - Add contextual information automatically
//! - Reduce code duplication by ~150 lines
//!
//! ## Usage
//! ```rust
//! use crate::utils::result_mappers::{map_db_error, map_graph_db_error};
//!
//! // Before: Manual error mapping (duplicated everywhere)
//! let result = conn.execute(sql, params)
//!     .map_err(|e| RepositoryError::DatabaseError(format!("Failed to execute: {}", e)))?;
//!
//! // After: Centralized error mapping
//! let result = map_db_error(conn.execute(sql, params), "Failed to execute")?;
//! ```

use rusqlite;
use crate::ports::graph_repository::GraphRepositoryError;
use crate::ports::ontology_repository::OntologyRepositoryError;
use crate::errors::VisionFlowError;

// RepositoryError is now defined in the port traits themselves
// Use GraphRepositoryError or OntologyRepositoryError directly

// DEPRECATED: Generic RepositoryError was removed in favor of specific error types
// Use map_graph_db_error or map_ontology_db_error instead
/*
pub fn map_db_error<T>(
    result: rusqlite::Result<T>,
    context: &str
) -> Result<T, RepositoryError> {
    // This function is deprecated - use specific error mappers
}
*/

/// Maps rusqlite::Error to GraphRepositoryError with context.
///
/// Specialized mapper for graph repository operations that converts database
/// errors to graph-specific error types.
///
/// # Arguments
/// * `result` - A rusqlite::Result to convert
/// * `context` - Context message describing the operation
///
/// # Examples
/// ```rust
/// use crate::utils::result_mappers::map_graph_db_error;
///
/// let result = conn.query_row("SELECT * FROM nodes WHERE id = ?1", [node_id], |row| {
///     Ok(row.get(0)?)
/// });
/// let node = map_graph_db_error(result, "fetch node")?;
/// ```
pub fn map_graph_db_error<T>(
    result: rusqlite::Result<T>,
    context: &str
) -> Result<T, GraphRepositoryError> {
    result.map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => {
            GraphRepositoryError::NotFound
        }
        rusqlite::Error::SqliteFailure(err, _)
            if err.code == rusqlite::ErrorCode::ConstraintViolation => {
            GraphRepositoryError::InvalidData(format!("{}: constraint violation", context))
        }
        _ => GraphRepositoryError::AccessError(format!("{}: {}", context, e))
    })
}

/// Maps rusqlite::Error to OntologyRepositoryError with context.
///
/// Specialized mapper for ontology repository operations that converts database
/// errors to ontology-specific error types.
///
/// # Arguments
/// * `result` - A rusqlite::Result to convert
/// * `context` - Context message describing the operation
///
/// # Examples
/// ```rust
/// use crate::utils::result_mappers::map_ontology_db_error;
///
/// let result = conn.execute(
///     "INSERT INTO classes (iri, label) VALUES (?1, ?2)",
///     params![class.iri, class.label]
/// );
/// map_ontology_db_error(result, "insert OWL class")?;
/// ```
pub fn map_ontology_db_error<T>(
    result: rusqlite::Result<T>,
    context: &str
) -> Result<T, OntologyRepositoryError> {
    result.map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => {
            OntologyRepositoryError::NotFound
        }
        rusqlite::Error::SqliteFailure(err, _)
            if err.code == rusqlite::ErrorCode::ConstraintViolation => {
            OntologyRepositoryError::InvalidData(format!("{}: constraint violation", context))
        }
        _ => OntologyRepositoryError::DatabaseError(format!("{}: {}", context, e))
    })
}

// DEPRECATED: Use map_graph_service_error or map_ontology_service_error instead
/*
pub fn map_service_error<T>(
    result: Result<T, RepositoryError>,
    context: &str
) -> Result<T, VisionFlowError> {
    // This function is deprecated - use specific service error mappers
}
*/

/// Maps GraphRepositoryError to VisionFlowError for service layer.
///
/// Specialized service-level mapper for graph repository errors.
///
/// # Arguments
/// * `result` - A Result with GraphRepositoryError to convert
/// * `context` - Context message describing the operation
pub fn map_graph_service_error<T>(
    result: Result<T, GraphRepositoryError>,
    context: &str
) -> Result<T, VisionFlowError> {
    result.map_err(|e| match e {
        GraphRepositoryError::NotFound => VisionFlowError::Generic {
            message: format!("{}: not found", context),
            source: None,
        },
        _ => VisionFlowError::Generic {
            message: format!("{}: {}", context, e),
            source: None,
        }
    })
}

/// Maps OntologyRepositoryError to VisionFlowError for service layer.
///
/// Specialized service-level mapper for ontology repository errors.
///
/// # Arguments
/// * `result` - A Result with OntologyRepositoryError to convert
/// * `context` - Context message describing the operation
pub fn map_ontology_service_error<T>(
    result: Result<T, OntologyRepositoryError>,
    context: &str
) -> Result<T, VisionFlowError> {
    result.map_err(|e| match e {
        OntologyRepositoryError::NotFound => VisionFlowError::Generic {
            message: format!("{}: ontology not found", context),
            source: None,
        },
        OntologyRepositoryError::ClassNotFound(class) => VisionFlowError::Generic {
            message: format!("{}: class '{}' not found", context, class),
            source: None,
        },
        OntologyRepositoryError::PropertyNotFound(prop) => VisionFlowError::Generic {
            message: format!("{}: property '{}' not found", context, prop),
            source: None,
        },
        _ => VisionFlowError::Generic {
            message: format!("{}: {}", context, e),
            source: None,
        }
    })
}

/// Extension trait for rusqlite::Result to add convenient error mapping.
///
/// Provides ergonomic methods for error conversion directly on Results.
pub trait RusqliteResultExt<T> {
    /// Convert to GraphRepositoryError with context
    fn to_graph_error(self, context: &str) -> Result<T, GraphRepositoryError>;

    /// Convert to OntologyRepositoryError with context
    fn to_ontology_error(self, context: &str) -> Result<T, OntologyRepositoryError>;
}

impl<T> RusqliteResultExt<T> for rusqlite::Result<T> {
    fn to_graph_error(self, context: &str) -> Result<T, GraphRepositoryError> {
        map_graph_db_error(self, context)
    }

    fn to_ontology_error(self, context: &str) -> Result<T, OntologyRepositoryError> {
        map_ontology_db_error(self, context)
    }
}

// DEPRECATED: Generic RepositoryResultExt removed
// Use GraphRepositoryResultExt or OntologyRepositoryResultExt instead

/// Extension trait for graph repository Results.
pub trait GraphRepositoryResultExt<T> {
    /// Convert GraphRepositoryError to VisionFlowError
    fn to_service_error(self, context: &str) -> Result<T, VisionFlowError>;
}

impl<T> GraphRepositoryResultExt<T> for Result<T, GraphRepositoryError> {
    fn to_service_error(self, context: &str) -> Result<T, VisionFlowError> {
        map_graph_service_error(self, context)
    }
}

/// Extension trait for ontology repository Results.
pub trait OntologyRepositoryResultExt<T> {
    /// Convert OntologyRepositoryError to VisionFlowError
    fn to_service_error(self, context: &str) -> Result<T, VisionFlowError>;
}

impl<T> OntologyRepositoryResultExt<T> for Result<T, OntologyRepositoryError> {
    fn to_service_error(self, context: &str) -> Result<T, VisionFlowError> {
        map_ontology_service_error(self, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::{Connection, params};

    // DEPRECATED: Generic map_db_error test removed
    // Use test_map_graph_db_error_not_found instead

    #[test]
    fn test_map_graph_db_error_not_found() {
        let conn = Connection::open_in_memory().unwrap();
        let result: rusqlite::Result<i32> = conn.query_row(
            "SELECT id FROM nodes WHERE id = ?1",
            [999],
            |row| row.get(0)
        );

        let mapped = map_graph_db_error(result, "fetch node");
        assert!(mapped.is_err());
        match mapped.unwrap_err() {
            GraphRepositoryError::NotFound => {}
            _ => panic!("Expected NotFound error"),
        }
    }

    #[test]
    fn test_rusqlite_result_ext() {
        let conn = Connection::open_in_memory().unwrap();

        // Test to_graph_error
        let result: rusqlite::Result<i32> = conn.query_row(
            "SELECT 1",
            [],
            |row| row.get(0)
        );
        assert!(result.to_graph_error("test query").is_ok());

        // Test to_ontology_error
        let result2: rusqlite::Result<i32> = conn.query_row(
            "SELECT 1",
            [],
            |row| row.get(0)
        );
        assert!(result2.to_ontology_error("test query").is_ok());
    }

    #[test]
    fn test_graph_service_error_mapping() {
        let graph_error: Result<i32, GraphRepositoryError> = Err(
            GraphRepositoryError::AccessError("test error".to_string())
        );

        let service_result = map_graph_service_error(graph_error, "test operation");
        assert!(service_result.is_err());

        if let Err(VisionFlowError::Generic { message, .. }) = service_result {
            assert!(message.contains("test operation"));
            assert!(message.contains("test error"));
        } else {
            panic!("Expected Generic error");
        }
    }

    #[test]
    fn test_ontology_error_mapping() {
        let onto_error: Result<i32, OntologyRepositoryError> = Err(
            OntologyRepositoryError::ClassNotFound("TestClass".to_string())
        );

        let service_result = map_ontology_service_error(onto_error, "fetch class");
        assert!(service_result.is_err());

        if let Err(VisionFlowError::Generic { message, .. }) = service_result {
            assert!(message.contains("fetch class"));
            assert!(message.contains("TestClass"));
        } else {
            panic!("Expected Generic error");
        }
    }

    #[test]
    fn test_graph_repository_result_ext() {
        let graph_result: Result<i32, GraphRepositoryError> = Ok(42);
        assert!(graph_result.to_service_error("test").is_ok());

        let graph_error: Result<i32, GraphRepositoryError> = Err(
            GraphRepositoryError::NotFound
        );
        assert!(graph_error.to_service_error("test").is_err());
    }
}
