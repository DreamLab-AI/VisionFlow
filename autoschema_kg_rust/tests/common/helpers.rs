//! Test helper functions and utilities

use std::path::Path;
use std::fs;
use tempfile::TempDir;
use serde_json::Value;

/// Create a temporary directory with test files
pub fn create_test_dir_with_files(files: &[(&str, &str)]) -> TempDir {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");

    for (filename, content) in files {
        let file_path = temp_dir.path().join(filename);
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create parent directories");
        }
        fs::write(file_path, content).expect("Failed to write test file");
    }

    temp_dir
}

/// Compare two JSON values with tolerance for floating point numbers
pub fn json_eq_with_tolerance(a: &Value, b: &Value, tolerance: f64) -> bool {
    match (a, b) {
        (Value::Number(a_num), Value::Number(b_num)) => {
            if let (Some(a_f64), Some(b_f64)) = (a_num.as_f64(), b_num.as_f64()) {
                (a_f64 - b_f64).abs() < tolerance
            } else {
                a_num == b_num
            }
        }
        (Value::Array(a_arr), Value::Array(b_arr)) => {
            a_arr.len() == b_arr.len() &&
            a_arr.iter().zip(b_arr.iter()).all(|(a, b)| json_eq_with_tolerance(a, b, tolerance))
        }
        (Value::Object(a_obj), Value::Object(b_obj)) => {
            a_obj.len() == b_obj.len() &&
            a_obj.iter().all(|(key, a_val)| {
                b_obj.get(key).map_or(false, |b_val| json_eq_with_tolerance(a_val, b_val, tolerance))
            })
        }
        _ => a == b
    }
}

/// Calculate cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        0.0
    } else {
        dot_product / (magnitude_a * magnitude_b)
    }
}

/// Normalize a vector to unit length
pub fn normalize_vector(vector: &mut [f32]) {
    let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for x in vector.iter_mut() {
            *x /= magnitude;
        }
    }
}

/// Generate a random vector with given dimensions
pub fn random_vector(dimensions: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..dimensions).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Wait for a condition to be true with timeout
pub async fn wait_for_condition<F>(mut condition: F, timeout_ms: u64) -> bool
where
    F: FnMut() -> bool,
{
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_millis(timeout_ms);

    while start.elapsed() < timeout {
        if condition() {
            return true;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    false
}

/// Assert that two vectors are approximately equal
pub fn assert_vectors_eq(a: &[f32], b: &[f32], tolerance: f32) {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tolerance,
            "Vectors differ at index {}: {} vs {} (tolerance: {})",
            i, x, y, tolerance
        );
    }
}

/// Create a test configuration
pub fn create_test_config() -> Value {
    serde_json::json!({
        "test_mode": true,
        "timeout_ms": 5000,
        "retry_attempts": 3,
        "batch_size": 10
    })
}

/// Mock sleep function for testing time-dependent code
pub async fn mock_sleep(duration: std::time::Duration) {
    // In tests, we can make this instant or use tokio-test's advance time
    if std::env::var("TEST_MODE").is_ok() {
        // Don't actually sleep in tests
        return;
    }
    tokio::time::sleep(duration).await;
}

/// Generate test data with specific patterns
pub fn generate_test_pattern(pattern: &str, count: usize) -> Vec<String> {
    (0..count)
        .map(|i| format!("{}{:04}", pattern, i))
        .collect()
}

/// Validate JSON schema for test data
pub fn validate_test_json(value: &Value, required_fields: &[&str]) -> Result<(), String> {
    let obj = value.as_object().ok_or("Expected JSON object")?;

    for field in required_fields {
        if !obj.contains_key(*field) {
            return Err(format!("Missing required field: {}", field));
        }
    }

    Ok(())
}

/// Create a deterministic hash for testing
pub fn test_hash(input: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Setup logging for tests
pub fn setup_test_logging() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();
}

/// Benchmark helper for measuring execution time
pub async fn benchmark_async<F, Fut, T>(name: &str, f: F) -> T
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let start = std::time::Instant::now();
    let result = f().await;
    let duration = start.elapsed();
    println!("{}: {:?}", name, duration);
    result
}

/// Create a mock server for testing HTTP requests
#[cfg(feature = "wiremock")]
pub async fn create_mock_server() -> wiremock::MockServer {
    wiremock::MockServer::start().await
}