//! Integration tests for database migration system

use rusqlite::Connection;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

use webxr::migrations::{MigrationRunner, RollbackManager, VersionTracker};

fn setup_test_env() -> (TempDir, PathBuf, PathBuf) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let migrations_dir = temp_dir.path().join("migrations");
    fs::create_dir_all(&migrations_dir).unwrap();

    (temp_dir, db_path, migrations_dir)
}

fn create_migration(dir: &PathBuf, version: i32, name: &str, up_sql: &str, down_sql: &str) {
    let filename = format!("{:03}_{}.sql", version, name.replace(' ', "_"));
    let content = format!(
        "-- Migration: {}\n-- === UP MIGRATION ===\n{}\n-- === DOWN MIGRATION ===\n{}\n",
        filename, up_sql, down_sql
    );
    fs::write(dir.join(filename), content).unwrap();
}

#[test]
fn test_migration_up() {
    let (_temp, db_path, migrations_dir) = setup_test_env();

    create_migration(
        &migrations_dir,
        1,
        "create_users",
        "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
        "DROP TABLE users;",
    );

    let mut conn = Connection::open(&db_path).unwrap();
    let tracker = VersionTracker::new(&conn);
    tracker.initialize().unwrap();

    let runner = MigrationRunner::new(&migrations_dir);
    let count = runner.migrate_up(&mut conn).unwrap();

    assert_eq!(count, 1);
    assert_eq!(tracker.current_version().unwrap(), Some(1));

    // Verify table exists
    let exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='users'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(exists);
}

#[test]
fn test_migration_version_tracking() {
    let (_temp, db_path, migrations_dir) = setup_test_env();

    create_migration(
        &migrations_dir,
        1,
        "first",
        "CREATE TABLE first (id INTEGER);",
        "DROP TABLE first;",
    );

    let mut conn = Connection::open(&db_path).unwrap();
    let tracker = VersionTracker::new(&conn);
    tracker.initialize().unwrap();

    let runner = MigrationRunner::new(&migrations_dir);
    runner.migrate_up(&mut conn).unwrap();

    let applied = tracker.get_all_applied().unwrap();
    assert_eq!(applied.len(), 1);
    assert_eq!(applied[0].version, 1);
    assert_eq!(applied[0].name, "first");
    assert!(applied[0].applied_at.is_some());
    assert!(applied[0].execution_time_ms.is_some());
}
