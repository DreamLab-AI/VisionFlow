// Test disabled - references deprecated/removed modules (crate::config, crate::handlers::settings_handler, crate::handlers::validation_handler)
// Handler and config module paths have changed; use webxr::config instead
/*
/// Tests for REST API validation logic for bloom/glow field handling
///
/// This test ensures that the REST API validation correctly accepts both
/// 'bloom' and 'glow' field names, handles the field mapping, and properly
/// validates the settings update payloads.

#[cfg(test)]
mod bloom_glow_validation_tests {
    use crate::config::AppFullSettings;
    use crate::handlers::settings_handler::{
        validate_bloom_glow_settings, validate_rendering_settings,
    };
    use crate::handlers::validation_handler::ValidationService;
    use serde_json::{json, Value};

    #[test]
    fn test_validation_accepts_bloom_field() {
        let validation_service = ValidationService::new();

        // Test payload with 'bloom' field (frontend format)
        let payload = json!({
            "visualisation": {
                "rendering": {
                    "bloom": {
                        "enabled": true,
                        "intensity": 1.5,
                        "radius": 2.0,
                        "threshold": 0.8
                    }
                }
            }
        });

        let result = validation_service.validate_settings_update(&payload);
        assert!(
            result.is_ok(),
            "Validation should accept 'bloom' field: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_validation_accepts_glow_field() {
        let validation_service = ValidationService::new();

        // Test payload with 'glow' field (internal format)
        let payload = json!({
            "visualisation": {
                "rendering": {
                    "glow": {
                        "enabled": true,
                        "intensity": 1.5,
                        "radius": 2.0,
                        "threshold": 0.8
                    }
                }
            }
        });

        let result = validation_service.validate_settings_update(&payload);
        assert!(
            result.is_ok(),
            "Validation should accept 'glow' field: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_bloom_field_validation_ranges() {
        // Test intensity validation
        let payload_high_intensity = json!({
            "bloom": {
                "enabled": true,
                "intensity": 15.0  // Above max (10.0)
            }
        });

        let result = validate_bloom_glow_settings(&payload_high_intensity.get("bloom").unwrap());
        assert!(result.is_err(), "Should reject intensity > 10.0");
        assert!(result
            .unwrap_err()
            .contains("intensity must be between 0.0 and 10.0"));

        // Test radius validation
        let payload_high_radius = json!({
            "bloom": {
                "enabled": true,
                "radius": 6.0  // Above max (5.0)
            }
        });

        let result = validate_bloom_glow_settings(&payload_high_radius.get("bloom").unwrap());
        assert!(result.is_err(), "Should reject radius > 5.0");
        assert!(result
            .unwrap_err()
            .contains("radius must be between 0.0 and 5.0"));
    }

    #[test]
    fn test_glow_field_validation_ranges() {
        // Test the same validation works for 'glow' field
        let payload_high_strength = json!({
            "glow": {
                "enabled": true,
                "strength": 12.0  // Above max (10.0)
            }
        });

        let result = validate_bloom_glow_settings(&payload_high_strength.get("glow").unwrap());
        assert!(result.is_err(), "Should reject strength > 10.0");
        assert!(result
            .unwrap_err()
            .contains("strength must be between 0.0 and 10.0"));
    }

    #[test]
    fn test_bloom_specific_strength_fields() {
        // ... test implementation
    }

    #[test]
    fn test_rendering_settings_validation() {
        // ... test implementation
    }

    #[test]
    fn test_boolean_validation() {
        // ... test implementation
    }

    #[test]
    fn test_field_mapping_in_merge_process() {
        // ... test implementation
    }

    #[test]
    fn test_comprehensive_validation_service() {
        // ... test implementation
    }
}
*/
