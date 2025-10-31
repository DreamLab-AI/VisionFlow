# DatabaseService Generic Methods

## Overview

Added three core generic database methods to `DatabaseService` for unified database operations across all three SQLite databases (Settings, KnowledgeGraph, Ontology).

## New Public API

### 1. `execute()` - Execute SQL Statements

Execute INSERT, UPDATE, or DELETE statements with parameters.

**Signature:**
```rust
pub fn execute(
    &self,
    target: DatabaseTarget,
    query: &str,
    params: &[&dyn rusqlite::ToSql],
) -> SqliteResult<usize>
```

**Example:**
```rust
use rusqlite::params;
use webxr::services::database_service::{DatabaseService, DatabaseTarget};

// Insert data
let rows_affected = db_service.execute(
    DatabaseTarget::Settings,
    "INSERT INTO users (name, email) VALUES (?1, ?2)",
    params!["John Doe", "john@example.com"]
)?;

// Update data
let rows_affected = db_service.execute(
    DatabaseTarget::Settings,
    "UPDATE users SET email = ?1 WHERE name = ?2",
    params!["new@example.com", "John Doe"]
)?;

// Delete data
let rows_affected = db_service.execute(
    DatabaseTarget::Settings,
    "DELETE FROM users WHERE name = ?1",
    params!["John Doe"]
)?;
```

### 2. `query_one()` - Query Single Row

Query a single row from the database, returns `Option<T>`.

**Signature:**
```rust
pub fn query_one<T, F>(
    &self,
    target: DatabaseTarget,
    query: &str,
    params: &[&dyn rusqlite::ToSql],
    mapper: F,
) -> SqliteResult<Option<T>>
where
    F: FnOnce(&rusqlite::Row<'_>) -> SqliteResult<T>
```

**Example:**
```rust
// Query single user
let user: Option<User> = db_service.query_one(
    DatabaseTarget::Settings,
    "SELECT id, name, email FROM users WHERE id = ?1",
    params![42],
    |row| Ok(User {
        id: row.get(0)?,
        name: row.get(1)?,
        email: row.get(2)?,
    })
)?;

// Query single value
let count: Option<i64> = db_service.query_one(
    DatabaseTarget::Settings,
    "SELECT COUNT(*) FROM users",
    &[],
    |row| row.get(0)
)?;
```

### 3. `query_all()` - Query Multiple Rows

Query multiple rows from the database, returns `Vec<T>`.

**Signature:**
```rust
pub fn query_all<T, F>(
    &self,
    target: DatabaseTarget,
    query: &str,
    params: &[&dyn rusqlite::ToSql],
    mapper: F,
) -> SqliteResult<Vec<T>>
where
    F: FnMut(&rusqlite::Row<'_>) -> SqliteResult<T>
```

**Example:**
```rust
// Query multiple users
let users: Vec<User> = db_service.query_all(
    DatabaseTarget::Settings,
    "SELECT id, name, email FROM users WHERE active = ?1",
    params![true],
    |row| Ok(User {
        id: row.get(0)?,
        name: row.get(1)?,
        email: row.get(2)?,
    })
)?;

// Query with filtering
let names: Vec<String> = db_service.query_all(
    DatabaseTarget::Settings,
    "SELECT name FROM users WHERE email LIKE ?1 ORDER BY name",
    params!["%@example.com"],
    |row| row.get(0)
)?;
```

## DatabaseTarget Enum

New public enum for selecting which database to operate on:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatabaseTarget {
    Settings,
    KnowledgeGraph,
    Ontology,
}
```

## Implementation Details

### Connection Management
- All methods use the existing connection pool infrastructure
- Connections are automatically returned to the pool after use
- Thread-safe concurrent access via r2d2 connection pooling

### Error Handling
- All methods return `SqliteResult<T>` for consistent error handling
- Connection errors are wrapped with descriptive messages
- SQL errors propagate directly from rusqlite

### Type Safety
- Generic `T` parameter allows type-safe result mapping
- Mapper functions provide compile-time type checking
- Works with any type that can be constructed from a row

### Performance
- Zero-copy parameter binding via `&[&dyn ToSql]`
- Prepared statements used internally for query optimization
- Connection pooling prevents connection overhead

## Use Cases

### Execute Operations
- Inserting new records
- Updating existing data
- Deleting records
- Executing DDL statements (CREATE, ALTER, DROP)

### Query One Operations
- Finding a single record by ID
- Checking if a record exists
- Retrieving aggregate values (COUNT, MAX, MIN, AVG)
- Getting configuration values

### Query All Operations
- Listing records with filters
- Batch data retrieval
- Reporting and analytics
- Data export operations

## Testing

Comprehensive integration tests in `/home/devuser/workspace/project/tests/database_service_methods_test.rs`:

- ✅ Execute INSERT operations
- ✅ Execute UPDATE operations
- ✅ Execute DELETE operations
- ✅ Query single row (found)
- ✅ Query single row (not found)
- ✅ Query multiple rows
- ✅ Query with empty results
- ✅ All database targets (Settings, KnowledgeGraph, Ontology)
- ✅ Error handling for invalid queries
- ✅ Transaction safety across pool connections
- ✅ Different data types (string, integer, float, boolean, JSON)

## Migration Guide

### Before (Direct Connection Access)
```rust
let conn = db_service.get_settings_connection()?;
conn.execute(
    "INSERT INTO settings (key, value) VALUES (?1, ?2)",
    params!["key", "value"]
)?;
```

### After (Using Generic Methods)
```rust
db_service.execute(
    DatabaseTarget::Settings,
    "INSERT INTO settings (key, value) VALUES (?1, ?2)",
    params!["key", "value"]
)?;
```

## Benefits

1. **Unified Interface**: Single API for all three databases
2. **Type Safety**: Compile-time type checking via generics
3. **Error Handling**: Consistent error propagation
4. **Testability**: Easy to mock and test
5. **Performance**: Leverages connection pooling
6. **Maintainability**: Reduces boilerplate code
7. **Flexibility**: Custom mapper functions for any result type

## Future Enhancements

Potential improvements for future versions:

- [ ] Async/await support for non-blocking operations
- [ ] Transaction support across multiple operations
- [ ] Batch insert/update helpers
- [ ] Query builder integration
- [ ] Automatic retry logic for transient errors
- [ ] Query result caching
- [ ] Performance metrics collection
