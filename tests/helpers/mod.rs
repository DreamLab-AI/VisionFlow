// Test helpers and fixtures module
// Provides common utilities for testing

use rusqlite::Connection;
use std::sync::Arc;
use tempfile::TempDir;

/// Create a temporary test database with standard schema
pub fn create_test_db() -> Connection {
    let conn = Connection::open_in_memory().expect("Failed to create test database");

    // Create standard test schema
    conn.execute(
        "CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            value TEXT
        )",
        [],
    ).expect("Failed to create test schema");

    conn
}

/// Create a temporary directory for test files
pub fn create_test_dir() -> TempDir {
    TempDir::new().expect("Failed to create temporary directory")
}

/// Generate test graph data
pub fn generate_test_graph(node_count: usize, edge_ratio: f32) -> (Vec<TestNode>, Vec<TestEdge>) {
    let nodes: Vec<TestNode> = (0..node_count)
        .map(|i| TestNode {
            id: format!("node_{}", i),
            label: format!("Test Node {}", i),
            x: rand::random::<f32>() * 100.0,
            y: rand::random::<f32>() * 100.0,
            z: rand::random::<f32>() * 100.0,
        })
        .collect();

    let edge_count = (node_count as f32 * edge_ratio) as usize;
    let edges: Vec<TestEdge> = (0..edge_count)
        .map(|i| {
            let source_idx = i % node_count;
            let target_idx = (i + 1) % node_count;
            TestEdge {
                source: format!("node_{}", source_idx),
                target: format!("node_{}", target_idx),
                label: format!("edge_{}", i),
            }
        })
        .collect();

    (nodes, edges)
}

#[derive(Debug, Clone)]
pub struct TestNode {
    pub id: String,
    pub label: String,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone)]
pub struct TestEdge {
    pub source: String,
    pub target: String,
    pub label: String,
}

/// Assert that two floating point numbers are approximately equal
pub fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "Values not approximately equal: {} vs {} (epsilon: {})",
        a, b, epsilon
    );
}

/// Measure execution time of a function
pub fn measure_time<F, R>(f: F) -> (R, std::time::Duration)
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let duration = start.elapsed();
    (result, duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_test_db() {
        let conn = create_test_db();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM test_data", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_generate_test_graph() {
        let (nodes, edges) = generate_test_graph(10, 1.5);
        assert_eq!(nodes.len(), 10);
        assert_eq!(edges.len(), 15); // 10 * 1.5
    }

    #[test]
    fn test_assert_approx_eq() {
        assert_approx_eq(1.0, 1.0001, 0.001);
    }

    #[test]
    #[should_panic]
    fn test_assert_approx_eq_fails() {
        assert_approx_eq(1.0, 2.0, 0.001);
    }

    #[test]
    fn test_measure_time() {
        let (result, duration) = measure_time(|| {
            std::thread::sleep(std::time::Duration::from_millis(100));
            42
        });
        assert_eq!(result, 42);
        assert!(duration >= std::time::Duration::from_millis(100));
    }
}
