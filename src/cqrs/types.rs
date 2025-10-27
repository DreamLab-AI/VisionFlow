// src/cqrs/types.rs
//! CQRS Base Types and Traits
//!
//! Defines the fundamental traits for the Command Query Responsibility Segregation pattern.
//! Commands represent write operations, Queries represent read operations.

use async_trait::async_trait;
use std::fmt::Debug;

/// Result type using anyhow for flexible error handling
pub type Result<T> = anyhow::Result<T>;

/// Command trait - represents a write operation that modifies system state
///
/// Commands should be immutable structs that contain all data needed to perform
/// the operation. They should be validated before execution.
///
/// # Example
/// ```
/// use async_trait::async_trait;
/// use crate::cqrs::types::{Command, Result};
///
/// #[derive(Debug, Clone)]
/// pub struct AddNodeCommand {
///     pub label: String,
///     pub x: f32,
///     pub y: f32,
/// }
///
/// impl Command for AddNodeCommand {
///     type Result = u32; // Returns node ID
///
///     fn name(&self) -> &'static str {
///         "AddNode"
///     }
///
///     fn validate(&self) -> Result<()> {
///         if self.label.is_empty() {
///             return Err(anyhow::anyhow!("Node label cannot be empty"));
///         }
///         Ok(())
///     }
/// }
/// ```
pub trait Command: Send + Sync + Debug {
    /// The type returned when this command is executed
    type Result: Send;

    /// Name of the command for logging and metrics
    fn name(&self) -> &'static str;

    /// Validate command data before execution
    ///
    /// This is called before the handler executes the command.
    /// Return an error if the command data is invalid.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Query trait - represents a read operation that does not modify system state
///
/// Queries should be immutable structs that specify what data to retrieve.
/// Queries should never modify system state.
///
/// # Example
/// ```
/// use async_trait::async_trait;
/// use crate::cqrs::types::{Query, Result};
///
/// #[derive(Debug, Clone)]
/// pub struct GetNodeQuery {
///     pub node_id: u32,
/// }
///
/// impl Query for GetNodeQuery {
///     type Result = Option<Node>;
///
///     fn name(&self) -> &'static str {
///         "GetNode"
///     }
/// }
/// ```
pub trait Query: Send + Sync + Debug {
    /// The type returned when this query is executed
    type Result: Send;

    /// Name of the query for logging and metrics
    fn name(&self) -> &'static str;

    /// Validate query parameters before execution
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Handler for executing a command
///
/// Handlers contain the business logic for executing commands.
/// They typically depend on repositories and adapters.
#[async_trait]
pub trait CommandHandler<C: Command>: Send + Sync {
    /// Execute the command and return the result
    ///
    /// # Errors
    /// Returns an error if command validation fails or execution fails
    async fn handle(&self, command: C) -> Result<C::Result>;
}

/// Handler for executing a query
///
/// Handlers contain the logic for retrieving data.
/// They typically depend on repositories.
#[async_trait]
pub trait QueryHandler<Q: Query>: Send + Sync {
    /// Execute the query and return the result
    ///
    /// # Errors
    /// Returns an error if query validation fails or execution fails
    async fn handle(&self, query: Q) -> Result<Q::Result>;
}

/// Middleware for command execution pipeline
///
/// Middleware can intercept commands before and after execution
/// for cross-cutting concerns like logging, metrics, and transactions.
#[async_trait]
pub trait CommandMiddleware: Send + Sync {
    /// Called before command execution
    async fn before_execute(&self, command_name: &str) -> Result<()> {
        Ok(())
    }

    /// Called after successful command execution
    async fn after_execute(&self, command_name: &str) -> Result<()> {
        Ok(())
    }

    /// Called if command execution fails
    async fn on_error(&self, command_name: &str, error: &anyhow::Error) -> Result<()> {
        Ok(())
    }
}

/// Middleware for query execution pipeline
#[async_trait]
pub trait QueryMiddleware: Send + Sync {
    /// Called before query execution
    async fn before_execute(&self, query_name: &str) -> Result<()> {
        Ok(())
    }

    /// Called after successful query execution
    async fn after_execute(&self, query_name: &str) -> Result<()> {
        Ok(())
    }

    /// Called if query execution fails
    async fn on_error(&self, query_name: &str, error: &anyhow::Error) -> Result<()> {
        Ok(())
    }
}

/// Logging middleware for audit trail
pub struct LoggingMiddleware;

#[async_trait]
impl CommandMiddleware for LoggingMiddleware {
    async fn before_execute(&self, command_name: &str) -> Result<()> {
        tracing::info!(command = %command_name, "Executing command");
        Ok(())
    }

    async fn after_execute(&self, command_name: &str) -> Result<()> {
        tracing::info!(command = %command_name, "Command executed successfully");
        Ok(())
    }

    async fn on_error(&self, command_name: &str, error: &anyhow::Error) -> Result<()> {
        tracing::error!(command = %command_name, error = %error, "Command execution failed");
        Ok(())
    }
}

#[async_trait]
impl QueryMiddleware for LoggingMiddleware {
    async fn before_execute(&self, query_name: &str) -> Result<()> {
        tracing::debug!(query = %query_name, "Executing query");
        Ok(())
    }

    async fn after_execute(&self, query_name: &str) -> Result<()> {
        tracing::debug!(query = %query_name, "Query executed successfully");
        Ok(())
    }

    async fn on_error(&self, query_name: &str, error: &anyhow::Error) -> Result<()> {
        tracing::warn!(query = %query_name, error = %error, "Query execution failed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone)]
    struct TestCommand {
        value: String,
    }

    impl Command for TestCommand {
        type Result = ();

        fn name(&self) -> &'static str {
            "TestCommand"
        }

        fn validate(&self) -> Result<()> {
            if self.value.is_empty() {
                return Err(anyhow::anyhow!("Value cannot be empty"));
            }
            Ok(())
        }
    }

    #[derive(Debug, Clone)]
    struct TestQuery {
        id: u32,
    }

    impl Query for TestQuery {
        type Result = String;

        fn name(&self) -> &'static str {
            "TestQuery"
        }
    }

    #[test]
    fn test_command_validation() {
        let valid = TestCommand {
            value: "test".to_string(),
        };
        assert!(valid.validate().is_ok());

        let invalid = TestCommand {
            value: "".to_string(),
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_command_name() {
        let cmd = TestCommand {
            value: "test".to_string(),
        };
        assert_eq!(cmd.name(), "TestCommand");
    }

    #[test]
    fn test_query_name() {
        let query = TestQuery { id: 1 };
        assert_eq!(query.name(), "TestQuery");
    }
}
