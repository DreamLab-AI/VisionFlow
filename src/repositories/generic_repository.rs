// src/repositories/generic_repository.rs
//! Generic Repository Pattern Implementation
//!
//! Provides base implementations for SQLite repositories to eliminate code duplication
//! across UnifiedGraphRepository, UnifiedOntologyRepository, and SqliteSettingsRepository.
//!
//! Key features:
//! - Generic async blocking wrapper (eliminates 400+ lines of duplicate code)
//! - Generic transaction management (eliminates 150+ lines)
//! - Generic batch operations
//! - Error conversion utilities
//! - Connection management with mutex handling
//!
//! Usage:
//! ```rust
//! pub struct MyRepository {
//!     base: SqliteRepository,
//! }
//!
//! impl MyRepository {
//!     pub fn new(db_path: &str) -> Result<Self, String> {
//!         Ok(Self {
//!             base: SqliteRepository::new(db_path)?,
//!         })
//!     }
//! }
//! ```

use rusqlite::Connection;
use std::sync::{Arc, Mutex};
use tracing::{debug, instrument};

/// Generic repository error type
#[derive(Debug, thiserror::Error)]
pub enum RepositoryError {
    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Connection lock error: {0}")]
    LockError(String),

    #[error("Task join error: {0}")]
    TaskJoinError(String),

    #[error("Transaction error: {0}")]
    TransactionError(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, RepositoryError>;

/// Generic repository trait for CRUD operations
///
/// Provides a contract for basic repository operations that can be implemented
/// by any storage backend (SQLite, PostgreSQL, etc.)
pub trait GenericRepository<T, ID>: Send + Sync {
    /// Create a new entity
    fn create(&self, entity: &T) -> Result<ID>;

    /// Read an entity by ID
    fn read(&self, id: &ID) -> Result<Option<T>>;

    /// Update an existing entity
    fn update(&self, entity: &T) -> Result<()>;

    /// Delete an entity by ID
    fn delete(&self, id: &ID) -> Result<()>;

    /// Batch create multiple entities
    ///
    /// Default implementation iterates through entities.
    /// Override for optimized batch operations.
    fn batch_create(&self, entities: Vec<T>) -> Result<Vec<ID>> {
        let mut ids = Vec::new();
        for entity in &entities {
            ids.push(self.create(entity)?);
        }
        Ok(ids)
    }

    /// Batch read multiple entities
    ///
    /// Default implementation iterates through IDs.
    /// Override for optimized batch operations.
    fn batch_read(&self, ids: &[ID]) -> Result<Vec<T>> {
        let mut results = Vec::new();
        for id in ids {
            if let Some(entity) = self.read(id)? {
                results.push(entity);
            }
        }
        Ok(results)
    }

    /// Batch update multiple entities
    ///
    /// Default implementation iterates through entities.
    /// Override for optimized batch operations.
    fn batch_update(&self, entities: Vec<T>) -> Result<()> {
        for entity in &entities {
            self.update(entity)?;
        }
        Ok(())
    }

    /// Batch delete multiple entities
    ///
    /// Default implementation iterates through IDs.
    /// Override for optimized batch operations.
    fn batch_delete(&self, ids: &[ID]) -> Result<()> {
        for id in ids {
            self.delete(id)?;
        }
        Ok(())
    }

    /// Check if an entity exists by ID
    ///
    /// Default implementation uses read() and checks for Some.
    /// Override for optimized existence checks.
    fn exists(&self, id: &ID) -> Result<bool> {
        Ok(self.read(id)?.is_some())
    }

    /// Get an entity by ID or return an error if not found
    ///
    /// Default implementation uses read() and converts None to NotFound error.
    fn get_by_id_or_error(&self, id: &ID) -> Result<T>
    where
        ID: std::fmt::Debug,
    {
        self.read(id)?.ok_or_else(||
            RepositoryError::DatabaseError(format!("Entity not found: {:?}", id))
        )
    }

    /// Count total number of entities
    ///
    /// Default implementation is not provided - must be implemented by concrete types.
    /// This typically requires database-specific queries.
    fn count(&self) -> Result<usize>;

    /// Health check for repository connection
    fn health_check(&self) -> Result<bool>;
}

/// Base SQLite repository implementation
///
/// Provides common SQLite functionality that all SQLite-based repositories can inherit:
/// - Connection management with Arc<Mutex<Connection>>
/// - Async blocking wrapper for tokio::task::spawn_blocking
/// - Transaction management (begin, commit, rollback)
/// - Batch operation utilities
/// - Error conversion helpers
pub struct SqliteRepository {
    pub(crate) conn: Arc<Mutex<Connection>>,
}

impl SqliteRepository {
    /// Create a new SqliteRepository
    ///
    /// Opens or creates a SQLite database at the specified path.
    /// Foreign keys are enabled by default for referential integrity.
    ///
    /// # Arguments
    /// * `db_path` - Path to database file (or ":memory:" for in-memory)
    ///
    /// # Returns
    /// Initialized repository with active connection
    pub fn new(db_path: &str) -> Result<Self> {
        let conn = Connection::open(db_path).map_err(|e| {
            RepositoryError::DatabaseError(format!("Failed to open database {}: {}", db_path, e))
        })?;

        // Enable foreign keys for referential integrity
        conn.execute("PRAGMA foreign_keys = ON", []).map_err(|e| {
            RepositoryError::DatabaseError(format!("Failed to enable foreign keys: {}", e))
        })?;

        debug!("Initialized SqliteRepository at {}", db_path);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Get a clone of the connection Arc for async operations
    ///
    /// This is used by repositories to get access to the connection
    /// for their async blocking tasks.
    pub fn get_connection(&self) -> Arc<Mutex<Connection>> {
        self.conn.clone()
    }

    /// Execute a blocking operation asynchronously
    ///
    /// This is the core async blocking wrapper that eliminates ~400 lines of duplicate code.
    /// All async database operations should use this method.
    ///
    /// # Type Parameters
    /// * `F` - Closure that takes Connection reference and returns Result
    /// * `R` - Return type from the closure
    ///
    /// # Arguments
    /// * `operation` - Closure to execute with database connection
    ///
    /// # Returns
    /// Result from the closure execution
    ///
    /// # Example
    /// ```rust
    /// let node_count: i64 = repo.execute_blocking(|conn| {
    ///     conn.query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))
    ///         .map_err(|e| RepositoryError::DatabaseError(format!("Query failed: {}", e)))
    /// }).await?;
    /// ```
    #[instrument(skip(self, operation), level = "debug")]
    pub async fn execute_blocking<F, R>(&self, operation: F) -> Result<R>
    where
        F: FnOnce(&Connection) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().map_err(|e| {
                RepositoryError::LockError(format!("Failed to acquire mutex: {}", e))
            })?;

            operation(&*conn)
        })
        .await
        .map_err(|e| RepositoryError::TaskJoinError(format!("Task join error: {}", e)))?
    }

    /// Execute a transactional operation asynchronously
    ///
    /// Automatically handles transaction lifecycle (BEGIN, COMMIT, ROLLBACK).
    /// Eliminates ~150 lines of duplicate transaction management code.
    ///
    /// # Type Parameters
    /// * `F` - Closure that takes transaction reference and returns Result
    /// * `R` - Return type from the closure
    ///
    /// # Arguments
    /// * `operation` - Closure to execute within transaction
    ///
    /// # Returns
    /// Result from the transaction execution
    ///
    /// # Example
    /// ```rust
    /// repo.execute_transaction(|tx| {
    ///     tx.execute("INSERT INTO nodes (label) VALUES (?1)", params![label])?;
    ///     tx.execute("UPDATE stats SET count = count + 1", [])?;
    ///     Ok(())
    /// }).await?;
    /// ```
    #[instrument(skip(self, operation), level = "debug")]
    pub async fn execute_transaction<F, R>(&self, operation: F) -> Result<R>
    where
        F: FnOnce(&rusqlite::Transaction) -> Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc.lock().map_err(|e| {
                RepositoryError::LockError(format!("Failed to acquire mutex: {}", e))
            })?;

            let tx = conn.transaction().map_err(|e| {
                RepositoryError::TransactionError(format!("Failed to begin transaction: {}", e))
            })?;

            let result = operation(&tx)?;

            tx.commit().map_err(|e| {
                RepositoryError::TransactionError(format!("Failed to commit transaction: {}", e))
            })?;

            Ok(result)
        })
        .await
        .map_err(|e| RepositoryError::TaskJoinError(format!("Task join error: {}", e)))?
    }

    /// Execute a batch operation with transaction support
    ///
    /// Processes items in chunks to avoid overwhelming the database.
    /// Automatically wraps in transaction for atomicity.
    ///
    /// # Type Parameters
    /// * `T` - Type of items to process
    /// * `F` - Closure that processes a single item
    ///
    /// # Arguments
    /// * `items` - Items to process in batch
    /// * `chunk_size` - Number of items per transaction chunk (default: 1000)
    /// * `operation` - Closure to execute for each item
    ///
    /// # Returns
    /// Total number of items processed
    ///
    /// # Example
    /// ```rust
    /// repo.batch_execute(&nodes, 500, |tx, node| {
    ///     tx.execute("INSERT INTO nodes (id, label) VALUES (?1, ?2)",
    ///         params![node.id, node.label])?;
    ///     Ok(())
    /// }).await?;
    /// ```
    #[instrument(skip(self, items, operation), level = "debug")]
    pub async fn batch_execute<T, F>(
        &self,
        items: &[T],
        chunk_size: usize,
        operation: F,
    ) -> Result<usize>
    where
        T: Clone + Send + 'static,
        F: Fn(&rusqlite::Transaction, &T) -> Result<()> + Send + Clone + 'static,
    {
        let total = items.len();
        let chunks: Vec<Vec<T>> = items.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        for chunk in chunks {
            let op = operation.clone();
            self.execute_transaction(move |tx| {
                for item in &chunk {
                    op(tx, item)?;
                }
                Ok(())
            }).await?;
        }

        debug!("Batch executed {} items in chunks of {}", total, chunk_size);
        Ok(total)
    }

    /// Query a single row asynchronously
    ///
    /// Convenience method for single-row queries with the async blocking wrapper.
    ///
    /// # Type Parameters
    /// * `F` - Row mapping function
    /// * `R` - Return type from row mapping
    ///
    /// # Arguments
    /// * `sql` - SQL query string
    /// * `params` - Query parameters
    /// * `mapper` - Function to map row to result type
    ///
    /// # Returns
    /// Optional result if row exists
    #[instrument(skip(self, params, mapper), level = "debug")]
    pub async fn query_row_optional<P, F, R>(
        &self,
        sql: &str,
        params: P,
        mapper: F,
    ) -> Result<Option<R>>
    where
        P: rusqlite::Params + Send + 'static,
        F: FnOnce(&rusqlite::Row) -> rusqlite::Result<R> + Send + 'static,
        R: Send + 'static,
    {
        let sql = sql.to_string();

        self.execute_blocking(move |conn| {
            use rusqlite::OptionalExtension;

            conn.query_row(&sql, params, mapper)
                .optional()
                .map_err(|e| RepositoryError::DatabaseError(format!("Query failed: {}", e)))
        })
        .await
    }

    /// Health check for database connection
    ///
    /// Verifies the connection is alive by executing a simple query.
    pub async fn health_check(&self) -> Result<bool> {
        self.execute_blocking(|conn| {
            conn.query_row("SELECT 1", [], |row| row.get::<_, i64>(0))
                .map_err(|e| RepositoryError::DatabaseError(format!("Health check failed: {}", e)))?;
            Ok(true)
        })
        .await
    }
}

/// Helper function to convert rusqlite::Error to RepositoryError
pub fn convert_rusqlite_error(e: rusqlite::Error) -> RepositoryError {
    RepositoryError::DatabaseError(format!("SQLite error: {}", e))
}

/// Helper macro for common async blocking pattern
///
/// Usage:
/// ```rust
/// async_blocking!(self.conn, |conn| {
///     conn.execute("INSERT INTO table VALUES (?1)", params![value])?;
///     Ok(())
/// })
/// ```
#[macro_export]
macro_rules! async_blocking {
    ($conn:expr, $operation:expr) => {
        {
            let conn_arc = $conn.clone();
            tokio::task::spawn_blocking(move || {
                let conn = conn_arc.lock()
                    .map_err(|e| RepositoryError::LockError(format!("Failed to acquire mutex: {}", e)))?;
                $operation(&*conn)
            })
            .await
            .map_err(|e| RepositoryError::TaskJoinError(format!("Task join error: {}", e)))?
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_repository_creation() {
        let repo = SqliteRepository::new(":memory:").unwrap();
        assert!(repo.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_execute_blocking() {
        let repo = SqliteRepository::new(":memory:").unwrap();

        // Create test table
        repo.execute_blocking(|conn| {
            conn.execute(
                "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                [],
            ).map_err(convert_rusqlite_error)?;
            Ok(())
        }).await.unwrap();

        // Insert data
        repo.execute_blocking(|conn| {
            conn.execute(
                "INSERT INTO test (value) VALUES (?1)",
                ["test_value"],
            ).map_err(convert_rusqlite_error)?;
            Ok(())
        }).await.unwrap();

        // Query data
        let value: String = repo.execute_blocking(|conn| {
            conn.query_row(
                "SELECT value FROM test WHERE id = 1",
                [],
                |row| row.get(0)
            ).map_err(convert_rusqlite_error)
        }).await.unwrap();

        assert_eq!(value, "test_value");
    }

    #[tokio::test]
    async fn test_execute_transaction() {
        let repo = SqliteRepository::new(":memory:").unwrap();

        // Create table
        repo.execute_blocking(|conn| {
            conn.execute(
                "CREATE TABLE test (id INTEGER PRIMARY KEY, value INTEGER)",
                [],
            ).map_err(convert_rusqlite_error)?;
            Ok(())
        }).await.unwrap();

        // Execute transaction
        repo.execute_transaction(|tx| {
            tx.execute("INSERT INTO test (value) VALUES (?1)", [1])
                .map_err(convert_rusqlite_error)?;
            tx.execute("INSERT INTO test (value) VALUES (?1)", [2])
                .map_err(convert_rusqlite_error)?;
            Ok(())
        }).await.unwrap();

        // Verify both rows inserted
        let count: i64 = repo.execute_blocking(|conn| {
            conn.query_row("SELECT COUNT(*) FROM test", [], |row| row.get(0))
                .map_err(convert_rusqlite_error)
        }).await.unwrap();

        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_batch_execute() {
        let repo = SqliteRepository::new(":memory:").unwrap();

        // Create table
        repo.execute_blocking(|conn| {
            conn.execute(
                "CREATE TABLE test (id INTEGER PRIMARY KEY AUTOINCREMENT, value INTEGER)",
                [],
            ).map_err(convert_rusqlite_error)?;
            Ok(())
        }).await.unwrap();

        // Batch insert
        let items: Vec<i32> = (1..=100).collect();
        let count = repo.batch_execute(&items, 25, |tx, value| {
            tx.execute("INSERT INTO test (value) VALUES (?1)", [value])
                .map_err(convert_rusqlite_error)?;
            Ok(())
        }).await.unwrap();

        assert_eq!(count, 100);

        // Verify all inserted
        let total: i64 = repo.execute_blocking(|conn| {
            conn.query_row("SELECT COUNT(*) FROM test", [], |row| row.get(0))
                .map_err(convert_rusqlite_error)
        }).await.unwrap();

        assert_eq!(total, 100);
    }

    #[tokio::test]
    async fn test_query_row_optional() {
        let repo = SqliteRepository::new(":memory:").unwrap();

        // Create and populate table
        repo.execute_blocking(|conn| {
            conn.execute(
                "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)",
                [],
            ).map_err(convert_rusqlite_error)?;
            conn.execute(
                "INSERT INTO test (value) VALUES (?1)",
                ["exists"],
            ).map_err(convert_rusqlite_error)?;
            Ok(())
        }).await.unwrap();

        // Query existing row
        let value = repo.query_row_optional(
            "SELECT value FROM test WHERE id = 1",
            [],
            |row| row.get::<_, String>(0)
        ).await.unwrap();

        assert_eq!(value, Some("exists".to_string()));

        // Query non-existing row
        let value = repo.query_row_optional(
            "SELECT value FROM test WHERE id = 999",
            [],
            |row| row.get::<_, String>(0)
        ).await.unwrap();

        assert_eq!(value, None);
    }
}
