// src/repositories/query_builder.rs
//! Type-Safe SQL Query Builder
//!
//! Provides a fluent API for constructing SQL queries with parameter binding
//! to eliminate duplicate SQL construction patterns and prevent SQL injection.
//!
//! Key features:
//! - Fluent API for SELECT, INSERT, UPDATE, DELETE
//! - Automatic parameter binding with type safety
//! - Batch operation support
//! - WHERE clause chaining
//! - ORDER BY, LIMIT, OFFSET support
//! - SQL injection prevention via parameterized queries
//!
//! Usage:
//! ```rust
//! let (sql, params) = QueryBuilder::select("nodes")
//!     .columns(&["id", "label", "owl_class_iri"])
//!     .where_clause("owl_class_iri = ?")
//!     .order_by("id ASC")
//!     .limit(100)
//!     .build();
//! ```

use rusqlite::params_from_iter;
use std::fmt;

/// SQL query builder with fluent API
///
/// Builds parameterized SQL queries to prevent injection attacks
/// and eliminate duplicate query construction code.
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    table: String,
    operation: Operation,
    columns: Vec<String>,
    where_clauses: Vec<String>,
    order_by: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
    values: Vec<Vec<SqlValue>>,
}

#[derive(Debug, Clone)]
enum Operation {
    Select,
    Insert,
    Update,
    Delete,
}

/// Represents a SQL value for parameter binding
#[derive(Debug, Clone)]
pub enum SqlValue {
    Null,
    Integer(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

impl fmt::Display for SqlValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SqlValue::Null => write!(f, "NULL"),
            SqlValue::Integer(i) => write!(f, "{}", i),
            SqlValue::Real(r) => write!(f, "{}", r),
            SqlValue::Text(s) => write!(f, "'{}'", s.replace('\'', "''")),
            SqlValue::Blob(_) => write!(f, "<BLOB>"),
        }
    }
}

impl QueryBuilder {
    /// Create a new SELECT query builder
    ///
    /// # Arguments
    /// * `table` - Table name to select from
    ///
    /// # Example
    /// ```rust
    /// let builder = QueryBuilder::select("graph_nodes");
    /// ```
    pub fn select(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            operation: Operation::Select,
            columns: Vec::new(),
            where_clauses: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
            values: Vec::new(),
        }
    }

    /// Create a new INSERT query builder
    ///
    /// # Arguments
    /// * `table` - Table name to insert into
    ///
    /// # Example
    /// ```rust
    /// let builder = QueryBuilder::insert("graph_nodes");
    /// ```
    pub fn insert(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            operation: Operation::Insert,
            columns: Vec::new(),
            where_clauses: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
            values: Vec::new(),
        }
    }

    /// Create a new UPDATE query builder
    ///
    /// # Arguments
    /// * `table` - Table name to update
    ///
    /// # Example
    /// ```rust
    /// let builder = QueryBuilder::update("graph_nodes");
    /// ```
    pub fn update(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            operation: Operation::Update,
            columns: Vec::new(),
            where_clauses: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
            values: Vec::new(),
        }
    }

    /// Create a new DELETE query builder
    ///
    /// # Arguments
    /// * `table` - Table name to delete from
    ///
    /// # Example
    /// ```rust
    /// let builder = QueryBuilder::delete("graph_nodes");
    /// ```
    pub fn delete(table: impl Into<String>) -> Self {
        Self {
            table: table.into(),
            operation: Operation::Delete,
            columns: Vec::new(),
            where_clauses: Vec::new(),
            order_by: None,
            limit: None,
            offset: None,
            values: Vec::new(),
        }
    }

    /// Specify columns for SELECT or INSERT/UPDATE operations
    ///
    /// For SELECT: columns to retrieve
    /// For INSERT/UPDATE: columns to set
    ///
    /// # Arguments
    /// * `cols` - Column names
    ///
    /// # Example
    /// ```rust
    /// builder.columns(&["id", "label", "owl_class_iri"])
    /// ```
    pub fn columns(mut self, cols: &[&str]) -> Self {
        self.columns = cols.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a WHERE clause condition
    ///
    /// Multiple WHERE clauses are combined with AND.
    /// Use ? placeholders for parameter binding.
    ///
    /// # Arguments
    /// * `condition` - SQL condition with ? placeholders
    ///
    /// # Example
    /// ```rust
    /// builder.where_clause("owl_class_iri = ?")
    ///        .where_clause("id > ?")
    /// ```
    pub fn where_clause(mut self, condition: impl Into<String>) -> Self {
        self.where_clauses.push(condition.into());
        self
    }

    /// Add ORDER BY clause
    ///
    /// # Arguments
    /// * `ordering` - ORDER BY expression (e.g., "id ASC", "label DESC")
    ///
    /// # Example
    /// ```rust
    /// builder.order_by("id ASC")
    /// ```
    pub fn order_by(mut self, ordering: impl Into<String>) -> Self {
        self.order_by = Some(ordering.into());
        self
    }

    /// Add LIMIT clause
    ///
    /// # Arguments
    /// * `count` - Maximum number of rows to return
    ///
    /// # Example
    /// ```rust
    /// builder.limit(100)
    /// ```
    pub fn limit(mut self, count: usize) -> Self {
        self.limit = Some(count);
        self
    }

    /// Add OFFSET clause
    ///
    /// # Arguments
    /// * `count` - Number of rows to skip
    ///
    /// # Example
    /// ```rust
    /// builder.offset(50)
    /// ```
    pub fn offset(mut self, count: usize) -> Self {
        self.offset = Some(count);
        self
    }

    /// Add values for INSERT operation
    ///
    /// For single INSERT, call once with one row of values.
    /// For batch INSERT, call multiple times with each row.
    ///
    /// # Arguments
    /// * `vals` - Values corresponding to columns
    ///
    /// # Example
    /// ```rust
    /// builder.columns(&["id", "label"])
    ///        .values(vec![SqlValue::Integer(1), SqlValue::Text("Node".into())])
    /// ```
    pub fn values(mut self, vals: Vec<SqlValue>) -> Self {
        self.values.push(vals);
        self
    }

    /// Add SET clause for UPDATE operations
    ///
    /// Sets column=value pairs for UPDATE.
    ///
    /// # Arguments
    /// * `assignments` - Map of column names to values
    ///
    /// # Example
    /// ```rust
    /// builder.set(vec![
    ///     ("label", SqlValue::Text("Updated".into())),
    ///     ("x", SqlValue::Real(10.0))
    /// ])
    /// ```
    pub fn set(mut self, assignments: Vec<(&str, SqlValue)>) -> Self {
        for (col, val) in assignments {
            self.columns.push(col.to_string());
            if self.values.is_empty() {
                self.values.push(Vec::new());
            }
            self.values[0].push(val);
        }
        self
    }

    /// Build the SQL query string
    ///
    /// Returns the SQL query with ? placeholders for parameters.
    /// Use with rusqlite's params! or prepared statements.
    ///
    /// # Returns
    /// SQL query string
    ///
    /// # Example
    /// ```rust
    /// let sql = builder.build();
    /// // "SELECT id, label FROM nodes WHERE id > ? ORDER BY id ASC LIMIT 100"
    /// ```
    pub fn build(&self) -> String {
        match self.operation {
            Operation::Select => self.build_select(),
            Operation::Insert => self.build_insert(),
            Operation::Update => self.build_update(),
            Operation::Delete => self.build_delete(),
        }
    }

    fn build_select(&self) -> String {
        let mut query = String::from("SELECT ");

        // Columns
        if self.columns.is_empty() {
            query.push('*');
        } else {
            query.push_str(&self.columns.join(", "));
        }

        // FROM
        query.push_str(&format!(" FROM {}", self.table));

        // WHERE
        if !self.where_clauses.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&self.where_clauses.join(" AND "));
        }

        // ORDER BY
        if let Some(ref order) = self.order_by {
            query.push_str(&format!(" ORDER BY {}", order));
        }

        // LIMIT
        if let Some(limit) = self.limit {
            query.push_str(&format!(" LIMIT {}", limit));
        }

        // OFFSET
        if let Some(offset) = self.offset {
            query.push_str(&format!(" OFFSET {}", offset));
        }

        query
    }

    fn build_insert(&self) -> String {
        if self.values.is_empty() {
            return format!("INSERT INTO {} DEFAULT VALUES", self.table);
        }

        let mut query = format!("INSERT INTO {}", self.table);

        // Columns
        if !self.columns.is_empty() {
            query.push_str(&format!(" ({})", self.columns.join(", ")));
        }

        // VALUES
        query.push_str(" VALUES ");

        let placeholders: Vec<String> = self
            .values
            .iter()
            .map(|row| {
                let params = (0..row.len()).map(|_| "?").collect::<Vec<_>>().join(", ");
                format!("({})", params)
            })
            .collect();

        query.push_str(&placeholders.join(", "));

        query
    }

    fn build_update(&self) -> String {
        let mut query = format!("UPDATE {}", self.table);

        // SET
        if !self.columns.is_empty() {
            let assignments: Vec<String> = self
                .columns
                .iter()
                .map(|col| format!("{} = ?", col))
                .collect();
            query.push_str(&format!(" SET {}", assignments.join(", ")));
        }

        // WHERE
        if !self.where_clauses.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&self.where_clauses.join(" AND "));
        }

        query
    }

    fn build_delete(&self) -> String {
        let mut query = format!("DELETE FROM {}", self.table);

        // WHERE
        if !self.where_clauses.is_empty() {
            query.push_str(" WHERE ");
            query.push_str(&self.where_clauses.join(" AND "));
        }

        query
    }

    /// Build query with REPLACE instead of INSERT (SQLite specific)
    ///
    /// # Example
    /// ```rust
    /// let sql = builder.build_replace();
    /// // "INSERT OR REPLACE INTO ..."
    /// ```
    pub fn build_replace(&self) -> String {
        self.build_insert().replace("INSERT", "INSERT OR REPLACE")
    }

    /// Get flattened parameter values for binding
    ///
    /// Returns all values in the order they should be bound to ? placeholders.
    ///
    /// # Returns
    /// Vector of all parameter values
    pub fn get_params(&self) -> Vec<&SqlValue> {
        self.values.iter().flat_map(|row| row.iter()).collect()
    }
}

/// Batch query builder for bulk operations
///
/// Efficiently handles large batch inserts and updates.
pub struct BatchQueryBuilder {
    table: String,
    columns: Vec<String>,
    batch_size: usize,
}

impl BatchQueryBuilder {
    /// Create a new batch query builder
    ///
    /// # Arguments
    /// * `table` - Table name
    /// * `columns` - Column names for batch operation
    /// * `batch_size` - Number of rows per batch (default: 1000)
    pub fn new(table: impl Into<String>, columns: Vec<String>, batch_size: usize) -> Self {
        Self {
            table: table.into(),
            columns,
            batch_size,
        }
    }

    /// Build batch INSERT query for a chunk of rows
    ///
    /// # Arguments
    /// * `num_rows` - Number of rows in this batch
    ///
    /// # Returns
    /// SQL query with ? placeholders for batch insert
    pub fn build_batch_insert(&self, num_rows: usize) -> String {
        let mut query = format!("INSERT INTO {} ({})", self.table, self.columns.join(", "));

        query.push_str(" VALUES ");

        let row_placeholders = (0..self.columns.len())
            .map(|_| "?")
            .collect::<Vec<_>>()
            .join(", ");

        let value_groups: Vec<String> = (0..num_rows)
            .map(|_| format!("({})", row_placeholders))
            .collect();

        query.push_str(&value_groups.join(", "));

        query
    }

    /// Build batch UPDATE query
    ///
    /// Updates multiple rows with same column values.
    ///
    /// # Returns
    /// SQL query for batch update
    pub fn build_batch_update(&self, where_column: &str) -> String {
        let assignments: Vec<String> = self
            .columns
            .iter()
            .map(|col| format!("{} = ?", col))
            .collect();

        format!(
            "UPDATE {} SET {} WHERE {} = ?",
            self.table,
            assignments.join(", "),
            where_column
        )
    }

    /// Get optimal batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_basic() {
        let sql = QueryBuilder::select("nodes").build();
        assert_eq!(sql, "SELECT * FROM nodes");
    }

    #[test]
    fn test_select_with_columns() {
        let sql = QueryBuilder::select("nodes")
            .columns(&["id", "label", "owl_class_iri"])
            .build();
        assert_eq!(sql, "SELECT id, label, owl_class_iri FROM nodes");
    }

    #[test]
    fn test_select_with_where() {
        let sql = QueryBuilder::select("nodes")
            .columns(&["id", "label"])
            .where_clause("owl_class_iri = ?")
            .where_clause("id > ?")
            .build();
        assert_eq!(sql, "SELECT id, label FROM nodes WHERE owl_class_iri = ? AND id > ?");
    }

    #[test]
    fn test_select_with_order_limit_offset() {
        let sql = QueryBuilder::select("nodes")
            .order_by("id ASC")
            .limit(100)
            .offset(50)
            .build();
        assert_eq!(sql, "SELECT * FROM nodes ORDER BY id ASC LIMIT 100 OFFSET 50");
    }

    #[test]
    fn test_insert_basic() {
        let sql = QueryBuilder::insert("nodes")
            .columns(&["id", "label"])
            .values(vec![
                SqlValue::Integer(1),
                SqlValue::Text("Test".to_string()),
            ])
            .build();
        assert_eq!(sql, "INSERT INTO nodes (id, label) VALUES (?, ?)");
    }

    #[test]
    fn test_insert_batch() {
        let sql = QueryBuilder::insert("nodes")
            .columns(&["id", "label"])
            .values(vec![SqlValue::Integer(1), SqlValue::Text("A".to_string())])
            .values(vec![SqlValue::Integer(2), SqlValue::Text("B".to_string())])
            .build();
        assert_eq!(sql, "INSERT INTO nodes (id, label) VALUES (?, ?), (?, ?)");
    }

    #[test]
    fn test_update_basic() {
        let sql = QueryBuilder::update("nodes")
            .set(vec![
                ("label", SqlValue::Text("Updated".to_string())),
                ("x", SqlValue::Real(10.0)),
            ])
            .where_clause("id = ?")
            .build();
        assert_eq!(sql, "UPDATE nodes SET label = ?, x = ? WHERE id = ?");
    }

    #[test]
    fn test_delete_basic() {
        let sql = QueryBuilder::delete("nodes")
            .where_clause("id = ?")
            .build();
        assert_eq!(sql, "DELETE FROM nodes WHERE id = ?");
    }

    #[test]
    fn test_replace() {
        let sql = QueryBuilder::insert("nodes")
            .columns(&["id", "label"])
            .values(vec![SqlValue::Integer(1), SqlValue::Text("Test".to_string())])
            .build_replace();
        assert_eq!(sql, "INSERT OR REPLACE INTO nodes (id, label) VALUES (?, ?)");
    }

    #[test]
    fn test_batch_query_builder() {
        let batch = BatchQueryBuilder::new("nodes", vec!["id".to_string(), "label".to_string()], 1000);
        let sql = batch.build_batch_insert(3);
        assert_eq!(sql, "INSERT INTO nodes (id, label) VALUES (?, ?), (?, ?), (?, ?)");
    }

    #[test]
    fn test_batch_update() {
        let batch = BatchQueryBuilder::new(
            "nodes",
            vec!["label".to_string(), "x".to_string()],
            1000,
        );
        let sql = batch.build_batch_update("id");
        assert_eq!(sql, "UPDATE nodes SET label = ?, x = ? WHERE id = ?");
    }

    #[test]
    fn test_sql_value_display() {
        assert_eq!(SqlValue::Null.to_string(), "NULL");
        assert_eq!(SqlValue::Integer(42).to_string(), "42");
        assert_eq!(SqlValue::Real(3.14).to_string(), "3.14");
        assert_eq!(SqlValue::Text("test".to_string()).to_string(), "'test'");
        assert_eq!(SqlValue::Text("it's".to_string()).to_string(), "'it''s'");
    }
}
