/// Tests for REST API validation logic for bloom/glow field handling
/// 
/// This test ensures that the REST API validation correctly accepts both
/// 'bloom' and 'glow' field names, handles the field mapping, and properly
/// validates the settings update payloads.

#[cfg(test)]
mod bloom_glow_validation_tests {
    use crate::handlers::validation_handler::ValidationService;
    use crate::handlers::settings_handler::{validate_rendering_settings, validate_bloom_glow_settings};
    use crate::config::AppFullSettings;
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
        assert!(result.is_ok(), "Validation should accept 'bloom' field: {:?}", result.err());
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
        assert!(result.is_ok(), "Validation should accept 'glow' field: {:?}", result.err());
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
        assert!(result.unwrap_err().contains("intensity must be between 0.0 and 10.0"));
        
        // Test radius validation
        let payload_high_radius = json!({
            "bloom": {
                "enabled": true,
                "radius": 6.0  // Above max (5.0)
            }
        });
        
        let result = validate_bloom_glow_settings(&payload_high_radius.get("bloom").unwrap());
        assert!(result.is_err(), "Should reject radius > 5.0");
        assert!(result.unwrap_err().contains("radius must be between 0.0 and 5.0"));
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
        assert!(result.unwrap_err().contains("strength must be between 0.0 and 10.0"));
    }

    #[test] 
    fn test_bloom_specific_strength_fields() {
        let payload = json!({
            "bloom": {
                "enabled": true,
                "edgeBloomStrength": 0.8,
                "environmentBloomStrength": 0.6,
                "nodeBloomStrength": 0.9
            }
        });
        
        let result = validate_bloom_glow_settings(&payload.get("bloom").unwrap());
        assert!(result.is_ok(), "Should accept valid bloom strength fields: {:?}", result.err());
        
        // Test invalid strength value
        let invalid_payload = json!({
            "bloom": {
                "enabled": true,
                "edgeBloomStrength": 1.5  // Above max (1.0)
            }
        });
        
        let result = validate_bloom_glow_settings(&invalid_payload.get("bloom").unwrap());
        assert!(result.is_err(), "Should reject edgeBloomStrength > 1.0");
        assert!(result.unwrap_err().contains("edgeBloomStrength must be between 0.0 and 1.0"));
    }

    #[test]
    fn test_rendering_settings_validation() {
        // Test rendering validation accepts both bloom and glow
        let rendering_with_bloom = json!({
            "ambientLightIntensity": 50.0,
            "bloom": {
                "enabled": true,
                "intensity": 2.0
            }
        });
        
        let result = validate_rendering_settings(&rendering_with_bloom);
        assert!(result.is_ok(), "Should accept rendering with bloom field: {:?}", result.err());
        
        let rendering_with_glow = json!({
            "ambientLightIntensity": 50.0,
            "glow": {
                "enabled": true,
                "intensity": 2.0
            }
        });
        
        let result = validate_rendering_settings(&rendering_with_glow);
        assert!(result.is_ok(), "Should accept rendering with glow field: {:?}", result.err());
    }

    #[test]
    fn test_boolean_validation() {
        // Test that enabled field must be boolean
        let invalid_enabled = json!({
            "bloom": {
                "enabled": "true"  // String instead of boolean
            }
        });
        
        let result = validate_bloom_glow_settings(&invalid_enabled.get("bloom").unwrap());
        assert!(result.is_err(), "Should reject non-boolean enabled field");
        assert!(result.unwrap_err().contains("enabled must be a boolean"));
    }

    #[test]
    fn test_field_mapping_in_merge_process() {
        // Test that bloom field gets converted to glow during processing
        let mut settings = AppFullSettings::new().expect("Should load default settings");
        
        let bloom_update = json!({
            "visualisation": {
                "rendering": {
                    "bloom": {
                        "enabled": true,
                        "intensity": 2.5,
                        "radius": 1.8
                    }
                }
            }
        });
        
        // This should not fail - the bloom field should be mapped to glow internally
        let result = settings.merge_update(bloom_update);
        assert!(result.is_ok(), "Should successfully merge bloom update: {:?}", result.err());
        
        // Verify the field is accessible as glow internally
        let serialized = serde_json::to_value(&settings).expect("Should serialize");
        let glow_field = serialized
            .pointer("/visualisation/rendering/glow")
            .or_else(|| serialized.pointer("/visualisation/glow"));
        
        assert!(glow_field.is_some(), "Should find glow field after bloom update");
        
        // When converting to camelCase for client, it should be 'bloom'
        let camel_case = settings.to_camel_case_json().expect("Should convert to camelCase");
        let bloom_field = camel_case
            .pointer("/visualisation/rendering/bloom")
            .or_else(|| camel_case.pointer("/visualisation/bloom"));
        
        assert!(bloom_field.is_some(), "Should find bloom field in client JSON");
    }

    #[test]
    fn test_comprehensive_validation_service() {
        let validation_service = ValidationService::new();
        
        // Complex payload with bloom effects
        let complex_payload = json!({
            "visualisation": {
                "rendering": {
                    "ambientLightIntensity": 75.0,
                    "bloom": {
                        "enabled": true,
                        "intensity": 3.2,
                        "radius": 2.1,
                        "threshold": 0.6,
                        "edgeBloomStrength": 0.7,
                        "environmentBloomStrength": 0.5,
                        "nodeBloomStrength": 0.8
                    }
                },
                "graphs": {
                    "logseq": {
                        "physics": {
                            "enabled": true,
                            "autoBalance": false
                        }
                    }
                }
            },
            "system": {
                "debug": {
                    "enabled": true
                }
            }
        });
        
        let result = validation_service.validate_settings_update(&complex_payload);
        assert!(result.is_ok(), "Should validate complex payload with bloom effects: {:?}", result.err());
        
        if let Ok(sanitized) = result {
            // Ensure the bloom field is preserved in the sanitized output
            let bloom_field = sanitized
                .pointer("/visualisation/rendering/bloom");
            assert!(bloom_field.is_some(), "Bloom field should be preserved in sanitized output");
        }
    }
}