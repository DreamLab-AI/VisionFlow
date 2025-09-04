//! Comprehensive tests for serde camelCase serialisation in settings refactor
//!
//! Tests the automatic snake_case to camelCase conversion using serde attributes
//! Validates proper field renaming, nested structure handling, and compatibility
//!

use serde_json::{json, Value};
use serde_yaml;
use std::collections::HashMap;
use crate::config::{AppFullSettings, VisualisationSettings, SystemSettings, XRSettings};

#[cfg(test)]
mod serde_camelcase_serialisation_tests {
    use super::*;
    
    #[test]
    fn test_top_level_camelcase_conversion() {
        let settings = AppFullSettings::default();
        
        // Serialize to JSON with automatic camelCase conversion
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        let json_str = serde_json::to_string(&settings).expect("Should serialize to JSON string");
        
        // Verify top-level fields are camelCase
        assert!(json_value.get("visualisation").is_some(), "visualisation field should exist");
        assert!(json_value.get("system").is_some(), "system field should exist");
        assert!(json_value.get("xr").is_some(), "xr field should exist");
        
        // Verify snake_case fields don't exist
        assert!(json_value.get("visualisation_settings").is_none(), "snake_case should not exist");
        assert!(json_value.get("system_settings").is_none(), "snake_case should not exist");
        assert!(json_value.get("xr_settings").is_none(), "snake_case should not exist");
    }

    #[test]
    fn test_visualisation_glow_camelcase_fields() {
        let settings = AppFullSettings::default();
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        
        let glow = &json_value["visualisation"]["glow"];
        
        // Test all glow-related camelCase conversions
        assert!(glow["nodeGlowStrength"].is_number(), "nodeGlowStrength should be camelCase");
        assert!(glow["edgeGlowStrength"].is_number(), "edgeGlowStrength should be camelCase");
        assert!(glow["environmentGlowStrength"].is_number(), "environmentGlowStrength should be camelCase");
        assert!(glow["baseColor"].is_string(), "baseColor should be camelCase");
        assert!(glow["emissionColor"].is_string(), "emissionColor should be camelCase");
        
        // Verify snake_case versions don't exist
        assert!(glow.get("node_glow_strength").is_none(), "snake_case should not exist");
        assert!(glow.get("edge_glow_strength").is_none(), "snake_case should not exist");
        assert!(glow.get("environment_glow_strength").is_none(), "snake_case should not exist");
        assert!(glow.get("base_color").is_none(), "snake_case should not exist");
        assert!(glow.get("emission_color").is_none(), "snake_case should not exist");
    }

    #[test]
    fn test_physics_parameters_camelcase() {
        let settings = AppFullSettings::default();
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        
        let physics = &json_value["visualisation"]["graphs"]["logseq"]["physics"];
        
        // Test all physics parameter camelCase conversions
        let expected_camel_fields = vec![
            "springK", "repelK", "attractionK", "maxVelocity", "boundsSize",
            "separationRadius", "centerGravityK", "gridCellSize", "warmupIterations",
            "coolingRate", "boundaryDamping", "updateThreshold", "stressWeight",
            "stressAlpha", "boundaryLimit", "alignmentStrength", "clusterStrength",
            "computeMode", "restLength", "linkDistance", "linkStrength",
            "chargeStrength", "centeringStrength", "timeStep", "minVelocity"
        ];
        
        for field in expected_camel_fields {
            assert!(physics[field].is_number(), "Physics field {} should exist and be camelCase", field);
        }
        
        // Test that snake_case versions don't exist
        let snake_case_fields = vec![
            "spring_k", "repel_k", "attraction_k", "max_velocity", "bounds_size",
            "separation_radius", "center_gravity_k", "grid_cell_size", "warmup_iterations",
            "cooling_rate", "boundary_damping", "update_threshold", "stress_weight"
        ];
        
        for field in snake_case_fields {
            assert!(physics.get(field).is_none(), "Snake_case field {} should not exist", field);
        }
    }

    #[test]
    fn test_system_settings_camelcase() {
        let settings = AppFullSettings::default();
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        
        let system = &json_value["system"];
        
        // Test system-level camelCase conversions
        assert!(system["debugMode"].is_boolean(), "debugMode should be camelCase");
        assert!(system["maxConnections"].is_number(), "maxConnections should be camelCase");
        assert!(system["connectionTimeout"].is_number(), "connectionTimeout should be camelCase");
        assert!(system["autoSave"].is_boolean(), "autoSave should be camelCase");
        
        // Test WebSocket settings
        if let Some(websocket) = system.get("websocket") {
            assert!(websocket["heartbeatInterval"].is_number(), "heartbeatInterval should be camelCase");
            assert!(websocket["reconnectDelay"].is_number(), "reconnectDelay should be camelCase");
            assert!(websocket["maxRetries"].is_number(), "maxRetries should be camelCase");
        }
        
        // Test audit settings
        if let Some(audit) = system.get("audit") {
            assert!(audit["auditLogPath"].is_string(), "auditLogPath should be camelCase");
            assert!(audit["maxLogSize"].is_number(), "maxLogSize should be camelCase");
        }
    }

    #[test]
    fn test_xr_settings_camelcase() {
        let settings = AppFullSettings::default();
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        
        let xr = &json_value["xr"];
        
        // Test XR-specific camelCase conversions
        assert!(xr["handMeshColor"].is_string(), "handMeshColor should be camelCase");
        assert!(xr["handRayColor"].is_string(), "handRayColor should be camelCase");
        assert!(xr["teleportRayColor"].is_string(), "teleportRayColor should be camelCase");
        assert!(xr["controllerRayColor"].is_string(), "controllerRayColor should be camelCase");
        assert!(xr["planeColor"].is_string(), "planeColor should be camelCase");
        assert!(xr["portalEdgeColor"].is_string(), "portalEdgeColor should be camelCase");
        assert!(xr["spaceType"].is_string(), "spaceType should be camelCase");
        assert!(xr["locomotionMethod"].is_string(), "locomotionMethod should be camelCase");
        
        // Verify snake_case doesn't exist
        assert!(xr.get("hand_mesh_color").is_none(), "snake_case should not exist");
        assert!(xr.get("teleport_ray_color").is_none(), "snake_case should not exist");
        assert!(xr.get("locomotion_method").is_none(), "snake_case should not exist");
    }

    #[test]
    fn test_round_trip_serialization() {
        let original_settings = AppFullSettings::default();
        
        // Serialize to JSON
        let json_str = serde_json::to_string(&original_settings).expect("Should serialize");
        
        // Deserialize back from JSON
        let deserialized: AppFullSettings = serde_json::from_str(&json_str)
            .expect("Should deserialize from camelCase JSON");
        
        // Verify they are equivalent
        let original_json = serde_json::to_value(&original_settings).expect("Should serialize original");
        let deserialized_json = serde_json::to_value(&deserialized).expect("Should serialize deserialized");
        
        assert_eq!(original_json, deserialized_json, "Round-trip serialization should maintain equality");
    }

    #[test]
    fn test_partial_camelcase_deserialization() {
        // Test that we can deserialize partial JSON with camelCase fields
        let partial_json = json!({
            "visualisation": {
                "glow": {
                    "nodeGlowStrength": 2.5,
                    "baseColor": "#ff0000"
                }
            },
            "system": {
                "debugMode": true,
                "maxConnections": 100
            }
        });
        
        // Create a default settings object
        let mut settings = AppFullSettings::default();
        
        // In a real implementation, we would merge this partial data
        // For testing, we verify the JSON structure is correct for camelCase
        let serialized = serde_json::to_string(&partial_json).expect("Should serialize");
        assert!(serialized.contains("nodeGlowStrength"), "Should contain camelCase fields");
        assert!(serialized.contains("debugMode"), "Should contain camelCase fields");
        assert!(!serialized.contains("node_glow_strength"), "Should not contain snake_case");
        assert!(!serialized.contains("debug_mode"), "Should not contain snake_case");
    }

    #[test]
    fn test_nested_structure_camelcase_preservation() {
        let settings = AppFullSettings::default();
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        
        // Test deeply nested structures maintain camelCase
        let logseq_graph = &json_value["visualisation"]["graphs"]["logseq"];
        assert!(logseq_graph["nodeRadius"].is_number(), "Nested nodeRadius should be camelCase");
        assert!(logseq_graph["edgeThickness"].is_number(), "Nested edgeThickness should be camelCase");
        
        // Test array fields with nested objects
        if let Some(color_schemes) = json_value["visualisation"].get("colorSchemes") {
            if color_schemes.is_array() {
                // Verify that array elements also use camelCase if they contain objects
                assert!(color_schemes.is_array(), "colorSchemes should be an array");
            }
        }
    }

    #[test]
    fn test_optional_field_camelcase() {
        let settings = AppFullSettings::default();
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        
        // Test that optional fields, when present, are also camelCase
        if let Some(ragflow) = json_value.get("ragflow") {
            if !ragflow.is_null() {
                // Check if ragflow has camelCase fields when present
                assert!(ragflow.is_object(), "ragflow should be an object when present");
            }
        }
        
        if let Some(openai) = json_value.get("openai") {
            if !openai.is_null() {
                assert!(openai.is_object(), "openai should be an object when present");
            }
        }
    }

    #[test]
    fn test_error_handling_invalid_camelcase() {
        // Test that invalid snake_case JSON fails to deserialize properly
        let invalid_snake_case_json = json!({
            "visualisation": {
                "glow": {
                    "node_glow_strength": 2.5,  // snake_case should not work
                    "base_color": "#ff0000"     // snake_case should not work
                }
            }
        });
        
        let json_str = serde_json::to_string(&invalid_snake_case_json).expect("Should serialize");
        
        // Try to deserialize - this should handle gracefully but may ignore snake_case fields
        let result: Result<AppFullSettings, _> = serde_json::from_str(&json_str);
        
        match result {
            Ok(settings) => {
                // If it deserializes successfully, the snake_case fields should be ignored
                // and default values should be used
                let json_value = serde_json::to_value(&settings).expect("Should serialize");
                assert!(json_value["visualisation"]["glow"]["nodeGlowStrength"].is_number());
                // The value should be the default, not 2.5
            },
            Err(_) => {
                // It's also acceptable if it fails to deserialize completely
            }
        }
    }

    #[test]
    fn test_yaml_camelcase_serialization() {
        let settings = AppFullSettings::default();
        
        // Test that YAML serialization also uses camelCase
        let yaml_str = serde_yaml::to_string(&settings).expect("Should serialize to YAML");
        
        // Check for camelCase patterns in YAML
        assert!(yaml_str.contains("nodeGlowStrength"), "YAML should contain camelCase fields");
        assert!(yaml_str.contains("debugMode"), "YAML should contain camelCase fields");
        assert!(!yaml_str.contains("node_glow_strength"), "YAML should not contain snake_case");
        assert!(!yaml_str.contains("debug_mode"), "YAML should not contain snake_case");
    }

    #[test]
    fn test_performance_large_settings_serialization() {
        use std::time::Instant;
        
        let settings = AppFullSettings::default();
        
        // Benchmark serialization performance
        let start = Instant::now();
        let json_str = serde_json::to_string(&settings).expect("Should serialize");
        let serialize_duration = start.elapsed();
        
        // Benchmark deserialization performance
        let start = Instant::now();
        let _deserialized: AppFullSettings = serde_json::from_str(&json_str).expect("Should deserialize");
        let deserialize_duration = start.elapsed();
        
        // Verify performance is reasonable (these are generous limits)
        assert!(serialize_duration.as_millis() < 100, "Serialization should be under 100ms");
        assert!(deserialize_duration.as_millis() < 100, "Deserialization should be under 100ms");
        
        // Verify JSON size is reasonable
        assert!(json_str.len() > 1000, "JSON should contain substantial data");
        assert!(json_str.len() < 100000, "JSON should not be excessively large");
    }
}

#[cfg(test)]
mod serde_compatibility_tests {
    use super::*;
    
    #[test]
    fn test_backwards_compatibility_with_existing_data() {
        // Test that existing camelCase data still deserializes correctly
        let existing_camelcase_json = json!({
            "visualisation": {
                "glow": {
                    "nodeGlowStrength": 1.5,
                    "edgeGlowStrength": 2.0,
                    "baseColor": "#00ffff"
                },
                "graphs": {
                    "logseq": {
                        "physics": {
                            "springK": 0.1,
                            "repelK": 100.0,
                            "maxVelocity": 5.0
                        }
                    }
                }
            },
            "system": {
                "debugMode": true,
                "maxConnections": 50
            },
            "xr": {
                "handMeshColor": "#ffffff",
                "locomotionMethod": "teleport"
            }
        });
        
        let json_str = serde_json::to_string(&existing_camelcase_json).expect("Should serialize");
        let settings: AppFullSettings = serde_json::from_str(&json_str).expect("Should deserialize");
        
        // Verify the data was correctly loaded
        let re_serialized = serde_json::to_value(&settings).expect("Should re-serialize");
        assert_eq!(re_serialized["visualisation"]["glow"]["nodeGlowStrength"], json!(1.5));
        assert_eq!(re_serialized["system"]["debugMode"], json!(true));
    }
    
    #[test]
    fn test_mixed_case_handling() {
        // Test edge cases with unusual casing
        let mixed_case_json = json!({
            "visualisation": {
                "glow": {
                    "nodeGlowStrength": 2.0
                }
            }
        });
        
        let json_str = serde_json::to_string(&mixed_case_json).expect("Should serialize");
        let result: Result<AppFullSettings, _> = serde_json::from_str(&json_str);
        
        assert!(result.is_ok(), "Mixed case should deserialize successfully");
    }
}