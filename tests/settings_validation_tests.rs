//! Comprehensive validation tests for VisionFlow settings refactor
//!
//! Tests input validation, constraint checking, type safety, and error handling
//! for the new settings system with granular updates and camelCase serialization
//! Ported and enhanced from codestore testing suite

use serde_json::{json, Value};
use std::collections::HashMap;
use std::thread;
use std::time::{Duration, Instant};

use crate::tests::test_utils::{
    contains_dangerous_content, is_valid_hex_color, validate_path_update, PerformanceTimer,
    TestAppSettings, TestValidationError,
};

#[cfg(test)]
mod settings_validation_tests {
    use super::*;

    #[test]
    fn test_numeric_range_validation() {
        let mut settings = TestAppSettings::new();

        // Test valid numeric ranges
        let valid_updates = vec![
            ("visualisation.glow.nodeGlowStrength", json!(1.5)),
            ("visualisation.glow.edgeGlowStrength", json!(2.0)),
            ("visualisation.graphs.logseq.physics.springK", json!(0.1)),
            ("system.maxConnections", json!(100)),
        ];

        for (path, value) in valid_updates {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_ok(),
                "Valid update for path '{}' should succeed: {:?}",
                path,
                result
            );
        }

        // Test invalid numeric ranges
        let invalid_updates = vec![
            ("visualisation.glow.nodeGlowStrength", json!(-1.0)), // Negative not allowed
            ("visualisation.glow.nodeGlowStrength", json!(100.0)), // Too high
            ("system.maxConnections", json!(-5)),                 // Negative connections
            ("system.maxConnections", json!(100000)),             // Unreasonably high
            ("visualisation.graphs.logseq.physics.springK", json!(0.0)), // Zero physics
        ];

        for (path, value) in invalid_updates {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_err(),
                "Invalid update for path '{}' should fail",
                path
            );
        }
    }

    #[test]
    fn test_string_validation() {
        let mut settings = TestAppSettings::new();

        // Test valid string values
        let valid_string_updates = vec![
            ("visualisation.glow.baseColor", json!("#ff0000")),
            ("visualisation.glow.baseColor", json!("#00FF00")),
            ("xr.locomotionMethod", json!("teleport")),
            ("xr.locomotionMethod", json!("smooth")),
        ];

        for (path, value) in valid_string_updates {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_ok(),
                "Valid string update for path '{}' should succeed: {:?}",
                path,
                result
            );
        }

        // Test invalid string values
        let invalid_string_updates = vec![
            ("visualisation.glow.baseColor", json!("invalid_color")),
            ("visualisation.glow.baseColor", json!("#gggggg")),
            ("visualisation.glow.baseColor", json!("#ff00")),
            ("xr.locomotionMethod", json!("flying")),
        ];

        for (path, value) in invalid_string_updates {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_err(),
                "Invalid string update for path '{}' should fail",
                path
            );
        }
    }

    #[test]
    fn test_boolean_validation() {
        let mut settings = TestAppSettings::new();

        // Test valid boolean values
        let valid_bool_updates = vec![
            ("system.debugMode", json!(true)),
            ("system.debugMode", json!(false)),
        ];

        for (path, value) in valid_bool_updates {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_ok(),
                "Valid boolean update for path '{}' should succeed: {:?}",
                path,
                result
            );
        }

        // Test invalid boolean values
        let invalid_bool_updates = vec![
            ("system.debugMode", json!(1)),
            ("system.debugMode", json!("true")),
            ("system.debugMode", json!(null)),
        ];

        for (path, value) in invalid_bool_updates {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_err(),
                "Type mismatch should be rejected for path '{}'",
                path
            );
        }
    }

    #[test]
    fn test_physics_parameter_constraints() {
        let mut settings = TestAppSettings::new();

        // Test valid physics parameters
        let valid_physics_tests = vec![
            ("visualisation.graphs.logseq.physics.springK", json!(0.1)),
            ("visualisation.graphs.logseq.physics.springK", json!(5.0)),
        ];

        for (path, value) in valid_physics_tests {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_ok(),
                "Valid physics parameter '{}' should succeed: {:?}",
                path,
                result
            );
        }

        // Test invalid physics parameter limits
        let invalid_physics_tests = vec![
            ("visualisation.graphs.logseq.physics.springK", json!(0.0)), // Zero spring
            ("visualisation.graphs.logseq.physics.springK", json!(-10.0)), // Negative spring
        ];

        for (path, value) in invalid_physics_tests {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_err(),
                "Invalid physics parameter '{}' should fail",
                path
            );
        }
    }

    #[test]
    fn test_cross_field_validation() {
        let mut settings = TestAppSettings::new();

        // First set debugMode to true
        let result = validate_path_update(&mut settings, "system.debugMode", &json!(true));
        assert!(result.is_ok());
        assert!(settings.system.debug_mode);

        // Test XR-specific validations
        let xr_tests = vec![
            ("xr.locomotionMethod", json!("teleport")),
            ("xr.locomotionMethod", json!("smooth")),
            ("xr.locomotionMethod", json!("dash")),
        ];

        for (path, value) in xr_tests {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_ok(),
                "XR setting '{}' should be valid: {:?}",
                path,
                result
            );
        }

        // Test invalid XR values
        let invalid_xr_tests = vec![
            ("xr.locomotionMethod", json!("flying")),
            ("xr.locomotionMethod", json!("invalid_method")),
        ];

        for (path, value) in invalid_xr_tests {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(result.is_err(), "Invalid XR setting '{}' should fail", path);
        }
    }

    #[test]
    fn test_type_coercion_validation() {
        let mut settings = TestAppSettings::new();

        // Test strict type checking - no coercion allowed
        let type_mismatch_tests = vec![
            ("visualisation.glow.nodeGlowStrength", json!("1.5"), false),
            ("system.maxConnections", json!("100"), false),
            ("system.debugMode", json!("true"), false),
            ("system.debugMode", json!(1), false),
        ];

        for (path, value, _should_succeed) in type_mismatch_tests {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(result.is_err(), "Type mismatch for '{}' should fail", path);

            if let Err(error) = result {
                assert_eq!(error, TestValidationError::TypeMismatch);
            }
        }
    }

    #[test]
    fn test_validation_error_messages() {
        let mut settings = TestAppSettings::new();

        let error_tests = vec![
            (
                "visualisation.glow.nodeGlowStrength",
                json!(-1.0),
                TestValidationError::OutOfRange,
            ),
            (
                "system.maxConnections",
                json!(-5),
                TestValidationError::OutOfRange,
            ),
            (
                "visualisation.glow.baseColor",
                json!("#invalid"),
                TestValidationError::InvalidFormat,
            ),
            (
                "system.debugMode",
                json!("true"),
                TestValidationError::TypeMismatch,
            ),
        ];

        for (path, value, expected_error) in error_tests {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(result.is_err(), "Error test for '{}' should fail", path);

            if let Err(error) = result {
                assert_eq!(
                    error, expected_error,
                    "Error type mismatch for path '{}'",
                    path
                );
            }
        }
    }

    #[test]
    fn test_concurrent_validation() {
        use std::sync::{Arc, Mutex};

        let settings = Arc::new(Mutex::new(TestAppSettings::new()));
        let mut handles = vec![];

        // Test concurrent validation calls
        for i in 0..10 {
            let settings_clone = settings.clone();
            let handle = thread::spawn(move || {
                let mut settings_guard = settings_clone.lock().unwrap();
                let path = "visualisation.glow.nodeGlowStrength";
                let value = json!(1.0 + (i as f64) * 0.1);
                validate_path_update(&mut *settings_guard, path, &value)
            });
            handles.push(handle);
        }

        // Wait for all threads and check results
        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.join().unwrap();
            assert!(
                result.is_ok(),
                "Concurrent validation {} should succeed: {:?}",
                i,
                result
            );
        }
    }

    #[test]
    fn test_memory_safety_validation() {
        let mut settings = TestAppSettings::new();

        // Test with very large values
        let large_string = "x".repeat(10000);
        let result = validate_path_update(
            &mut settings,
            "visualisation.glow.baseColor",
            &json!(large_string),
        );

        // Should fail due to invalid hex color format
        assert!(
            result.is_err(),
            "Large invalid color string should be rejected"
        );

        // Test deeply nested path (should be handled gracefully)
        let deep_path = (0..100)
            .map(|i| format!("level{}", i))
            .collect::<Vec<_>>()
            .join(".");
        let result = validate_path_update(&mut settings, &deep_path, &json!("test"));

        // Should fail as path not found, but not crash
        assert!(
            result.is_err(),
            "Deep unknown path should be rejected gracefully"
        );

        if let Err(error) = result {
            assert_eq!(error, TestValidationError::PathNotFound);
        }
    }

    #[test]
    fn test_security_validation() {
        let mut settings = TestAppSettings::new();

        // Test potentially dangerous values
        let security_tests = vec![
            ("unknown.path", json!("<script>alert('xss')</script>")),
            ("unknown.path", json!("'; DROP TABLE users; --")),
            ("unknown.path", json!("../../../etc/passwd")),
            ("unknown.path", json!("javascript:alert(1)")),
        ];

        for (path, value) in security_tests {
            let result = validate_path_update(&mut settings, path, &value);
            assert!(
                result.is_err(),
                "Security validation should reject dangerous value for '{}'",
                path
            );

            if let Err(error) = result {
                // Should either be SecurityViolation or PathNotFound (both are acceptable)
                assert!(
                    error == TestValidationError::SecurityViolation
                        || error == TestValidationError::PathNotFound,
                    "Expected SecurityViolation or PathNotFound, got: {:?}",
                    error
                );
            }
        }
    }

    #[test]
    fn test_performance_validation() {
        let mut settings = TestAppSettings::new();

        // Test validation performance
        let timer = PerformanceTimer::new();

        for i in 0..1000 {
            let path = "visualisation.glow.nodeGlowStrength";
            let value = json!(1.0 + (i as f64) * 0.001);
            let _ = validate_path_update(&mut settings, path, &value);
        }

        let duration = timer.elapsed();
        assert!(
            duration < Duration::from_secs(1),
            "1000 validations should complete within 1 second, took: {:?}",
            duration
        );

        // Test complex validation performance
        let complex_update_timer = PerformanceTimer::new();

        let complex_paths = vec![
            ("visualisation.glow.nodeGlowStrength", json!(2.0)),
            ("visualisation.glow.baseColor", json!("#ff0000")),
            ("system.debugMode", json!(true)),
            ("system.maxConnections", json!(150)),
            ("xr.locomotionMethod", json!("teleport")),
        ];

        for (path, value) in complex_paths {
            let _ = validate_path_update(&mut settings, path, &value);
        }

        let complex_duration = complex_update_timer.elapsed();
        assert!(
            complex_duration < Duration::from_millis(10),
            "Complex validation should be fast, took: {:?}",
            complex_duration
        );
    }

    #[test]
    fn test_boundary_conditions() {
        let mut settings = TestAppSettings::new();

        // Test exact boundary values
        let boundary_tests = vec![
            // Minimum valid values
            ("visualisation.glow.nodeGlowStrength", json!(0.0), false), // Should fail (0 not allowed)
            ("visualisation.glow.nodeGlowStrength", json!(0.001), true), // Should pass
            // Maximum valid values
            ("visualisation.glow.nodeGlowStrength", json!(10.0), true), // Should pass
            ("visualisation.glow.nodeGlowStrength", json!(10.001), false), // Should fail
            // Connection boundaries
            ("system.maxConnections", json!(1), true), // Minimum valid
            ("system.maxConnections", json!(0), false), // Below minimum
            ("system.maxConnections", json!(10000), true), // Maximum valid
            ("system.maxConnections", json!(10001), false), // Above maximum
        ];

        for (path, value, should_succeed) in boundary_tests {
            let result = validate_path_update(&mut settings, path, &value);
            if should_succeed {
                assert!(
                    result.is_ok(),
                    "Boundary value {} for '{}' should succeed: {:?}",
                    value,
                    path,
                    result
                );
            } else {
                assert!(
                    result.is_err(),
                    "Boundary value {} for '{}' should fail",
                    value,
                    path
                );
            }
        }
    }

    #[test]
    fn test_edge_case_strings() {
        let mut settings = TestAppSettings::new();

        // Test edge case string values
        let string_edge_cases = vec![
            // Empty strings
            ("visualisation.glow.baseColor", json!(""), false),
            // Very long valid hex colors
            ("visualisation.glow.baseColor", json!("#abcdef"), true),
            ("visualisation.glow.baseColor", json!("#ABCDEF"), true),
            // Case sensitivity in locomotion methods
            ("xr.locomotionMethod", json!("TELEPORT"), false), // Should be lowercase
            ("xr.locomotionMethod", json!("Teleport"), false), // Should be lowercase
            ("xr.locomotionMethod", json!("teleport"), true),  // Correct
            // Unicode and special characters
            ("visualisation.glow.baseColor", json!("#café00"), false), // Non-hex chars
        ];

        for (path, value, should_succeed) in string_edge_cases {
            let result = validate_path_update(&mut settings, path, &value);
            if should_succeed {
                assert!(
                    result.is_ok(),
                    "String edge case {} for '{}' should succeed: {:?}",
                    value,
                    path,
                    result
                );
            } else {
                assert!(
                    result.is_err(),
                    "String edge case {} for '{}' should fail",
                    value,
                    path
                );
            }
        }
    }

    #[test]
    fn test_validation_state_consistency() {
        let mut settings = TestAppSettings::new();

        // Record initial state
        let initial_glow = settings.visualisation.glow.node_glow_strength;
        let initial_debug = settings.system.debug_mode;

        // Apply valid update
        let result = validate_path_update(
            &mut settings,
            "visualisation.glow.nodeGlowStrength",
            &json!(5.0),
        );
        assert!(result.is_ok());
        assert_eq!(settings.visualisation.glow.node_glow_strength, 5.0);

        // Try invalid update
        let result = validate_path_update(
            &mut settings,
            "visualisation.glow.nodeGlowStrength",
            &json!(-1.0),
        );
        assert!(result.is_err());

        // State should remain unchanged after failed validation
        assert_eq!(settings.visualisation.glow.node_glow_strength, 5.0);
        assert_eq!(settings.system.debug_mode, initial_debug);

        // Apply another valid update to different field
        let result = validate_path_update(&mut settings, "system.debugMode", &json!(true));
        assert!(result.is_ok());
        assert_eq!(settings.system.debug_mode, true);

        // Previous field should remain unchanged
        assert_eq!(settings.visualisation.glow.node_glow_strength, 5.0);
    }

    #[test]
    fn test_validation_with_null_and_special_values() {
        let mut settings = TestAppSettings::new();

        // Test null values
        let null_tests = vec![
            ("visualisation.glow.nodeGlowStrength", json!(null), false),
            ("system.debugMode", json!(null), false),
            ("visualisation.glow.baseColor", json!(null), false),
        ];

        for (path, value, should_succeed) in null_tests {
            let result = validate_path_update(&mut settings, path, &value);
            if should_succeed {
                assert!(
                    result.is_ok(),
                    "Null value for '{}' should be accepted",
                    path
                );
            } else {
                assert!(
                    result.is_err(),
                    "Null value for '{}' should be rejected",
                    path
                );
            }
        }
    }
}

#[cfg(test)]
mod validation_helper_tests {
    use super::*;

    #[test]
    fn test_hex_color_validation_comprehensive() {
        // Valid hex colors
        let valid_colors = vec![
            "#000000", "#ffffff", "#FF0000", "#00ff00", "#0000FF", "#123456", "#abcdef", "#ABCDEF",
            "#a1b2c3", "#A1B2C3",
        ];

        for color in valid_colors {
            assert!(
                is_valid_hex_color(color),
                "Color '{}' should be valid",
                color
            );
        }

        // Invalid hex colors
        let invalid_colors = vec![
            "000000",    // Missing #
            "#00000",    // Too short
            "#0000000",  // Too long
            "#gggggg",   // Invalid hex chars
            "#",         // Just #
            "",          // Empty
            "red",       // Color name
            "#xyz123",   // Invalid chars
            "#12 34 56", // Spaces
        ];

        for color in invalid_colors {
            assert!(
                !is_valid_hex_color(color),
                "Color '{}' should be invalid",
                color
            );
        }
    }

    #[test]
    fn test_dangerous_content_detection_comprehensive() {
        // Dangerous content
        let dangerous_inputs = vec![
            "<script>alert('xss')</script>",
            "<SCRIPT>ALERT('XSS')</SCRIPT>",
            "'; DROP TABLE users; --",
            "\"; DROP TABLE users; --",
            "javascript:alert(1)",
            "vbscript:msgbox(1)",
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config",
            "<?php echo 'hack'; ?>",
            "<% evil_code %>",
            "%3Cscript%3Ealert(1)%3C/script%3E",
        ];

        for input in dangerous_inputs {
            assert!(
                contains_dangerous_content(input),
                "Input '{}' should be detected as dangerous",
                input
            );
        }

        // Safe content
        let safe_inputs = vec![
            "normal text",
            "user@domain.com",
            "file.txt",
            "https://example.com",
            "This is a long sentence with normal content.",
            "Special chars: !@#$%^&*()",
            "Numbers: 123456789",
            "Mixed: Text123!@#",
        ];

        for input in safe_inputs {
            assert!(
                !contains_dangerous_content(input),
                "Input '{}' should be detected as safe",
                input
            );
        }
    }
}
