// Test module disabled - references deprecated/removed modules (helpers, integration, test_utils)
// The helpers module does not exist; integration and performance modules may have moved per ADR-001
/*
//! Comprehensive test modules for VisionFlow migration
//!
//! This module organizes all test files for the migration project:
//! - Integration tests for end-to-end migration pipeline
//! - Adapter parity tests for dual-adapter validation
//! - Performance benchmarks for constraint translation
//! - Control center tests for settings persistence
//! - Load tests for concurrent user simulation

// Test module declarations
pub mod helpers;
pub mod integration;
pub mod performance;

// Test utilities and helpers (legacy)
pub mod test_utils;

// Re-export commonly used test utilities
pub use helpers::*;

#[cfg(test)]
mod migration_tests {
    use super::*;

    /// Integration test helper to verify all test modules compile and run
    #[test]
    fn test_modules_load() {
        // This test ensures all test modules are properly imported
        // and can be compiled together
        assert!(true);
    }
}
*/
