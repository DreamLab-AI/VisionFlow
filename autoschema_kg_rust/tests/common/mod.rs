//! Common testing utilities and shared test infrastructure

pub mod fixtures;
pub mod generators;
pub mod helpers;
pub mod mocks;

use std::path::PathBuf;
use tempfile::TempDir;

/// Test configuration and setup utilities
pub struct TestConfig {
    pub temp_dir: TempDir,
    pub data_dir: PathBuf,
}

impl TestConfig {
    pub fn new() -> Self {
        let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
        let data_dir = temp_dir.path().join("test_data");
        std::fs::create_dir_all(&data_dir).expect("Failed to create data directory");

        Self { temp_dir, data_dir }
    }

    pub fn data_path(&self, filename: &str) -> PathBuf {
        self.data_dir.join(filename)
    }
}

/// Initialize test environment with logging
pub fn init_test_env() {
    let _ = env_logger::builder()
        .is_test(true)
        .filter_level(log::LevelFilter::Debug)
        .try_init();
}

/// Assert that a result is Ok and return the value
#[macro_export]
macro_rules! assert_ok {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(err) => panic!("Expected Ok, got Err: {:?}", err),
        }
    };
}

/// Assert that a result is Err
#[macro_export]
macro_rules! assert_err {
    ($expr:expr) => {
        match $expr {
            Ok(val) => panic!("Expected Err, got Ok: {:?}", val),
            Err(_) => (),
        }
    };
}

/// Async test timeout wrapper
#[macro_export]
macro_rules! timeout_test {
    ($duration:expr, $test:expr) => {
        tokio::time::timeout($duration, $test)
            .await
            .expect("Test timed out")
    };
}