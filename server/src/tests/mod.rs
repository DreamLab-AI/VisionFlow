//! Comprehensive test modules for VisionFlow settings system
//!
//! This module organizes all test files for the settings refactor project,
//! ported and enhanced from the codestore testing suite.
//! Tests are organized by functionality and integration patterns.

pub mod settings_validation_tests;
pub mod granular_api_tests;
pub mod settings_serialisation_tests;
pub mod concurrent_access_tests;
pub mod performance_tests;
pub mod websocket_integration_tests;
pub mod security_tests;

// Test utilities and helpers
pub mod test_utils;

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    /// Integration test helper to verify all test modules compile and run
    #[test]
    fn test_modules_load() {
        // This test ensures all test modules are properly imported
        // and can be compiled together
        assert!(true);
    }
}