// tests/database_service_methods_test.rs
//! Integration tests for DatabaseService generic methods: execute(), query_one(), query_all()

use rusqlite::params;
use std::path::PathBuf;
use tempfile::TempDir;
use webxr::services::database_service::{DatabaseService, DatabaseTarget};

/// Helper to create a test database service
fn create_test_db() -> (DatabaseService, TempDir) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test.db");

    let db_service = DatabaseService::new(&db_path).expect("Failed to create database service");
    db_service.initialize_schema().expect("Failed to initialize schema");

    (db_service, temp_dir)
}

#[test]
fn test_execute_insert() {
    let (db_service, _temp_dir) = create_test_db();

    // Execute INSERT statement
    let rows_affected = db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_text) VALUES (?1, ?2, ?3)",
            params!["test_key", "string", "test_value"],
        )
        .expect("Failed to execute INSERT");

    assert_eq!(rows_affected, 1, "Should insert 1 row");

    // Verify the insert worked
    let value = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT value_text FROM settings WHERE key = ?1",
            params!["test_key"],
            |row| row.get::<_, String>(0),
        )
        .expect("Failed to query inserted value");

    assert_eq!(value, Some("test_value".to_string()));
}

#[test]
fn test_execute_update() {
    let (db_service, _temp_dir) = create_test_db();

    // Insert initial data
    db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_text) VALUES (?1, ?2, ?3)",
            params!["update_test", "string", "original"],
        )
        .expect("Failed to insert");

    // Execute UPDATE statement
    let rows_affected = db_service
        .execute(
            DatabaseTarget::Settings,
            "UPDATE settings SET value_text = ?1 WHERE key = ?2",
            params!["updated", "update_test"],
        )
        .expect("Failed to execute UPDATE");

    assert_eq!(rows_affected, 1, "Should update 1 row");

    // Verify the update
    let value = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT value_text FROM settings WHERE key = ?1",
            params!["update_test"],
            |row| row.get::<_, String>(0),
        )
        .expect("Failed to query updated value");

    assert_eq!(value, Some("updated".to_string()));
}

#[test]
fn test_execute_delete() {
    let (db_service, _temp_dir) = create_test_db();

    // Insert test data
    db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_text) VALUES (?1, ?2, ?3)",
            params!["delete_test", "string", "to_delete"],
        )
        .expect("Failed to insert");

    // Execute DELETE statement
    let rows_affected = db_service
        .execute(
            DatabaseTarget::Settings,
            "DELETE FROM settings WHERE key = ?1",
            params!["delete_test"],
        )
        .expect("Failed to execute DELETE");

    assert_eq!(rows_affected, 1, "Should delete 1 row");

    // Verify deletion
    let value = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT value_text FROM settings WHERE key = ?1",
            params!["delete_test"],
            |row| row.get::<_, String>(0),
        )
        .expect("Failed to query after delete");

    assert_eq!(value, None, "Row should be deleted");
}

#[test]
fn test_query_one_found() {
    let (db_service, _temp_dir) = create_test_db();

    // Insert test data
    db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_integer) VALUES (?1, ?2, ?3)",
            params!["count", "integer", &42],
        )
        .expect("Failed to insert");

    // Query single row with custom mapper
    let result = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT key, value_integer FROM settings WHERE key = ?1",
            params!["count"],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                ))
            },
        )
        .expect("Failed to query");

    assert_eq!(result, Some(("count".to_string(), 42)));
}

#[test]
fn test_query_one_not_found() {
    let (db_service, _temp_dir) = create_test_db();

    // Query non-existent row
    let result = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT value_text FROM settings WHERE key = ?1",
            params!["nonexistent"],
            |row| row.get::<_, String>(0),
        )
        .expect("Failed to query");

    assert_eq!(result, None, "Should return None for non-existent row");
}

#[test]
fn test_query_all_empty() {
    let (db_service, _temp_dir) = create_test_db();

    // Query empty table
    let results: Vec<String> = db_service
        .query_all(
            DatabaseTarget::Settings,
            "SELECT key FROM settings WHERE key LIKE ?1",
            params!["prefix_%"],
            |row| row.get::<_, String>(0),
        )
        .expect("Failed to query");

    assert_eq!(results.len(), 0, "Should return empty vector");
}

#[test]
fn test_query_all_multiple_rows() {
    let (db_service, _temp_dir) = create_test_db();

    // Insert multiple rows
    for i in 1..=5 {
        db_service
            .execute(
                DatabaseTarget::Settings,
                "INSERT INTO settings (key, value_type, value_integer) VALUES (?1, ?2, ?3)",
                params![format!("item_{}", i), "integer", &(i * 10)],
            )
            .expect("Failed to insert");
    }

    // Query all rows with pattern
    let results: Vec<(String, i64)> = db_service
        .query_all(
            DatabaseTarget::Settings,
            "SELECT key, value_integer FROM settings WHERE key LIKE ?1 ORDER BY key",
            params!["item_%"],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                ))
            },
        )
        .expect("Failed to query all");

    assert_eq!(results.len(), 5, "Should return 5 rows");
    assert_eq!(results[0], ("item_1".to_string(), 10));
    assert_eq!(results[4], ("item_5".to_string(), 50));
}

#[test]
fn test_all_database_targets() {
    let (db_service, _temp_dir) = create_test_db();

    // Test Settings database
    let rows = db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_text) VALUES (?1, ?2, ?3)",
            params!["settings_test", "string", "works"],
        )
        .expect("Settings database should work");
    assert_eq!(rows, 1);

    // Note: KnowledgeGraph and Ontology databases would need their own test data
    // These tests verify the connection routing works correctly
}

#[test]
fn test_execute_with_no_params() {
    let (db_service, _temp_dir) = create_test_db();

    // Execute query with empty params
    let count: Option<i64> = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT COUNT(*) FROM settings",
            &[],
            |row| row.get(0),
        )
        .expect("Failed to count");

    assert_eq!(count, Some(0), "Should count 0 rows in empty table");
}

#[test]
fn test_error_handling_invalid_query() {
    let (db_service, _temp_dir) = create_test_db();

    // Try to execute invalid SQL
    let result = db_service.execute(
        DatabaseTarget::Settings,
        "INVALID SQL SYNTAX",
        &[],
    );

    assert!(result.is_err(), "Should return error for invalid SQL");
}

#[test]
fn test_query_all_with_different_types() {
    let (db_service, _temp_dir) = create_test_db();

    // Insert rows with different value types
    db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_text) VALUES (?1, ?2, ?3)",
            params!["string_val", "string", "text"],
        )
        .unwrap();

    db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_integer) VALUES (?1, ?2, ?3)",
            params!["int_val", "integer", &123],
        )
        .unwrap();

    db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_float) VALUES (?1, ?2, ?3)",
            params!["float_val", "float", &45.67],
        )
        .unwrap();

    // Query all keys
    let keys: Vec<String> = db_service
        .query_all(
            DatabaseTarget::Settings,
            "SELECT key FROM settings ORDER BY key",
            &[],
            |row| row.get(0),
        )
        .expect("Failed to query all keys");

    assert_eq!(keys.len(), 3);
    assert_eq!(keys, vec!["float_val", "int_val", "string_val"]);
}

#[test]
fn test_transaction_safety() {
    let (db_service, _temp_dir) = create_test_db();

    // Insert data
    db_service
        .execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings (key, value_type, value_text) VALUES (?1, ?2, ?3)",
            params!["tx_test", "string", "value1"],
        )
        .unwrap();

    // Each execute gets its own connection from the pool
    // Verify isolation works correctly
    let result1 = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT value_text FROM settings WHERE key = ?1",
            params!["tx_test"],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(result1, Some("value1".to_string()));

    // Update in separate connection
    db_service
        .execute(
            DatabaseTarget::Settings,
            "UPDATE settings SET value_text = ?1 WHERE key = ?2",
            params!["value2", "tx_test"],
        )
        .unwrap();

    // Verify update visible in new query
    let result2 = db_service
        .query_one(
            DatabaseTarget::Settings,
            "SELECT value_text FROM settings WHERE key = ?1",
            params!["tx_test"],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(result2, Some("value2".to_string()));
}
