//! Comprehensive tests for settings refactor - serde serialization with rename_all
//!
//! Tests the automatic snake_case to camelCase conversion using serde attributes
//! Validates backwards compatibility and proper serialization/deserialization
//!

use serde_json::{json, Value};
use serde_yaml;

use webxr::config::AppFullSettings;

#[cfg(test)]
mod serde_camelcase_tests {
    use super::*;
    
    #[test]
    fn test_automatic_camelcase_serialization() {
        let settings = AppFullSettings::default();
        
        // Serialize to JSON with automatic camelCase conversion
        let json_value = serde_json::to_value(&settings).expect("Should serialize to JSON");
        
        // Verify key camelCase conversions
        assert!(json_value["visualisation"]["glow"]["nodeGlowStrength"].is_number());
        assert!(json_value["visualisation"]["glow"]["edgeGlowStrength"].is_number());
        assert!(json_value["visualisation"]["glow"]["environmentGlowStrength"].is_number());
        assert!(json_value["visualisation"]["glow"]["baseColor"].is_string());
        assert!(json_value["visualisation"]["glow"]["emissionColor"].is_string());
        
        // Verify physics settings camelCase
        let physics = &json_value["visualisation"]["graphs"]["logseq"]["physics"];
        assert!(physics["springK"].is_number());
        assert!(physics["repelK"].is_number());
        assert!(physics["attractionK"].is_number());
        assert!(physics["maxVelocity"].is_number());
        assert!(physics["boundsSize"].is_number());
        assert!(physics["separationRadius"].is_number());
        assert!(physics["centerGravityK"].is_number());
        assert!(physics["gridCellSize"].is_number());
        assert!(physics["warmupIterations"].is_number());
        assert!(physics["coolingRate"].is_number());
        assert!(physics["boundaryDamping"].is_number());
        assert!(physics["updateThreshold"].is_number());
        assert!(physics["stressWeight"].is_number());
        assert!(physics["stressAlpha"].is_number());
        assert!(physics["boundaryLimit"].is_number());
        assert!(physics["alignmentStrength"].is_number());
        assert!(physics["clusterStrength"].is_number());
        assert!(physics["computeMode"].is_number());
        assert!(physics["restLength"].is_number());
        assert!(physics["repulsionCutoff"].is_number());
        assert!(physics["repulsionSofteningEpsilon"].is_number());
        
        // Verify system settings camelCase
        assert!(json_value["system"]["network"]["bindAddress"].is_string());
        assert!(json_value["system"]["network"]["enableHttp2"].is_boolean());
        assert!(json_value["system"]["network"]["enableRateLimiting"].is_boolean());
        assert!(json_value["system"]["network"]["enableTls"].is_boolean());
        assert!(json_value["system"]["network"]["maxRequestSize"].is_number());
        assert!(json_value["system"]["network"]["minTlsVersion"].is_string());
        assert!(json_value["system"]["network"]["rateLimitRequests"].is_number());
        assert!(json_value["system"]["network"]["rateLimitWindow"].is_number());
        assert!(json_value["system"]["network"]["tunnelId"].is_string());
        assert!(json_value["system"]["network"]["apiClientTimeout"].is_number());
        
        println!("✅ Automatic camelCase serialization working correctly");
    }
    
    #[test]
    fn test_camelcase_deserialization() {
        // Create camelCase JSON input
        let camelcase_json = json!({
            "visualisation": {
                "glow": {
                    "enabled": true,
                    "intensity": 2.5,
                    "nodeGlowStrength": 3.0,
                    "edgeGlowStrength": 2.8,
                    "environmentGlowStrength": 1.5,
                    "baseColor": "#ff0000",
                    "emissionColor": "#00ff00",
                    "pulseSpeed": 1.2,
                    "flowSpeed": 0.8
                },
                "graphs": {
                    "logseq": {
                        "physics": {
                            "springK": 0.01,
                            "repelK": 100.0,
                            "attractionK": 0.0001,
                            "maxVelocity": 5.0,
                            "boundsSize": 1000.0,
                            "separationRadius": 3.0,
                            "centerGravityK": 0.05,
                            "gridCellSize": 100.0,
                            "warmupIterations": 200,
                            "coolingRate": 0.002,
                            "boundaryDamping": 0.9,
                            "updateThreshold": 0.02,
                            "stressWeight": 0.2,
                            "stressAlpha": 0.2,
                            "boundaryLimit": 950.0,
                            "alignmentStrength": 0.1,
                            "clusterStrength": 0.1,
                            "computeMode": 1,
                            "restLength": 100.0,
                            "repulsionCutoff": 100.0,
                            "repulsionSofteningEpsilon": 0.0002
                        }
                    }
                }
            },
            "system": {
                "network": {
                    "bindAddress": "127.0.0.1",
                    "enableHttp2": true,
                    "enableRateLimiting": true,
                    "enableTls": true,
                    "maxRequestSize": 2048576,
                    "minTlsVersion": "1.3",
                    "rateLimitRequests": 200,
                    "rateLimitWindow": 30,
                    "tunnelId": "test-tunnel-123",
                    "apiClientTimeout": 60
                },
                "debug": {
                    "enabled": true
                },
                "persistSettings": true
            },
            "xr": {
                "roomScale": 2.0,
                "spaceType": "bounded-floor",
                "quality": "high",
                "interactionDistance": 2.0,
                "locomotionMethod": "smooth",
                "teleportRayColor": "#0000ff",
                "controllerRayColor": "#ff00ff",
                "enableHandTracking": true,
                "handMeshEnabled": true,
                "handMeshColor": "#ffff00",
                "handMeshOpacity": 0.8,
                "handPointSize": 0.02,
                "handRayEnabled": true,
                "handRayColor": "#00ffff",
                "handRayWidth": 0.02,
                "gesturesmoothing": 0.7,
                "enableHaptics": true,
                "hapticIntensity": 0.8
            },
            "auth": {
                "enabled": true,
                "provider": "nostr",
                "required": true
            }
        });
        
        // Deserialize from camelCase JSON
        let settings: AppFullSettings = serde_json::from_value(camelcase_json)
            .expect("Should deserialize camelCase JSON");
        
        // Verify values were correctly mapped to snake_case fields
        assert_eq!(settings.visualisation.glow.enabled, true);
        assert_eq!(settings.visualisation.glow.intensity, 2.5);
        assert_eq!(settings.visualisation.glow.node_glow_strength, 3.0);
        assert_eq!(settings.visualisation.glow.edge_glow_strength, 2.8);
        assert_eq!(settings.visualisation.glow.environment_glow_strength, 1.5);
        assert_eq!(settings.visualisation.glow.base_color, "#ff0000");
        assert_eq!(settings.visualisation.glow.emission_color, "#00ff00");
        assert_eq!(settings.visualisation.glow.pulse_speed, 1.2);
        assert_eq!(settings.visualisation.glow.flow_speed, 0.8);
        
        // Verify physics settings
        let physics = &settings.visualisation.graphs.logseq.physics;
        assert_eq!(physics.spring_k, 0.01);
        assert_eq!(physics.repel_k, 100.0);
        assert_eq!(physics.attraction_k, 0.0001);
        assert_eq!(physics.max_velocity, 5.0);
        assert_eq!(physics.bounds_size, 1000.0);
        assert_eq!(physics.separation_radius, 3.0);
        assert_eq!(physics.center_gravity_k, 0.05);
        assert_eq!(physics.grid_cell_size, 100.0);
        assert_eq!(physics.warmup_iterations, 200);
        assert_eq!(physics.cooling_rate, 0.002);
        assert_eq!(physics.boundary_damping, 0.9);
        assert_eq!(physics.update_threshold, 0.02);
        
        // Verify system settings
        assert_eq!(settings.system.network.bind_address, "127.0.0.1");
        assert_eq!(settings.system.network.enable_http2, true);
        assert_eq!(settings.system.network.enable_rate_limiting, true);
        assert_eq!(settings.system.network.enable_tls, true);
        assert_eq!(settings.system.network.max_request_size, 2048576);
        assert_eq!(settings.system.network.min_tls_version, "1.3");
        assert_eq!(settings.system.network.rate_limit_requests, 200);
        assert_eq!(settings.system.network.rate_limit_window, 30);
        assert_eq!(settings.system.network.tunnel_id, "test-tunnel-123");
        assert_eq!(settings.system.network.api_client_timeout, 60);
        assert_eq!(settings.system.debug.enabled, true);
        assert_eq!(settings.system.persist_settings, true);
        
        // Verify XR settings
        assert_eq!(settings.xr.room_scale, 2.0);
        assert_eq!(settings.xr.space_type, "bounded-floor");
        assert_eq!(settings.xr.quality, "high");
        assert_eq!(settings.xr.interaction_distance, 2.0);
        assert_eq!(settings.xr.locomotion_method, "smooth");
        assert_eq!(settings.xr.teleport_ray_color, "#0000ff");
        assert_eq!(settings.xr.controller_ray_color, "#ff00ff");
        assert_eq!(settings.xr.enable_hand_tracking, true);
        assert_eq!(settings.xr.hand_mesh_enabled, true);
        assert_eq!(settings.xr.hand_mesh_color, "#ffff00");
        assert_eq!(settings.xr.hand_mesh_opacity, 0.8);
        assert_eq!(settings.xr.hand_point_size, 0.02);
        assert_eq!(settings.xr.hand_ray_enabled, true);
        assert_eq!(settings.xr.hand_ray_color, "#00ffff");
        assert_eq!(settings.xr.hand_ray_width, 0.02);
        assert_eq!(settings.xr.gesture_smoothing, 0.7);
        assert_eq!(settings.xr.enable_haptics, true);
        assert_eq!(settings.xr.haptic_intensity, 0.8);
        
        // Verify auth settings
        assert_eq!(settings.auth.enabled, true);
        assert_eq!(settings.auth.provider, "nostr");
        assert_eq!(settings.auth.required, true);
        
        println!("✅ Automatic camelCase deserialization working correctly");
    }
    
    #[test]
    fn test_roundtrip_serialization_consistency() {
        // Create settings with specific values
        let mut original_settings = AppFullSettings::default();
        original_settings.visualisation.glow.intensity = 1.5;
        original_settings.visualisation.glow.node_glow_strength = 2.0;
        original_settings.visualisation.glow.base_color = "#ff0000".to_string();
        original_settings.visualisation.graphs.logseq.physics.spring_k = 0.02;
        original_settings.visualisation.graphs.logseq.physics.repel_k = 200.0;
        original_settings.system.network.bind_address = "0.0.0.0".to_string();
        original_settings.system.network.port = 8080;
        original_settings.system.persist_settings = true;
        
        // Serialize to JSON (should be camelCase)
        let json_value = serde_json::to_value(&original_settings)
            .expect("Should serialize to JSON");
        
        // Deserialize back to struct (should handle camelCase)
        let deserialized_settings: AppFullSettings = serde_json::from_value(json_value)
            .expect("Should deserialize from JSON");
        
        // Verify values match
        assert_eq!(original_settings.visualisation.glow.intensity, deserialized_settings.visualisation.glow.intensity);
        assert_eq!(original_settings.visualisation.glow.node_glow_strength, deserialized_settings.visualisation.glow.node_glow_strength);
        assert_eq!(original_settings.visualisation.glow.base_color, deserialized_settings.visualisation.glow.base_color);
        assert_eq!(original_settings.visualisation.graphs.logseq.physics.spring_k, deserialized_settings.visualisation.graphs.logseq.physics.spring_k);
        assert_eq!(original_settings.visualisation.graphs.logseq.physics.repel_k, deserialized_settings.visualisation.graphs.logseq.physics.repel_k);
        assert_eq!(original_settings.system.network.bind_address, deserialized_settings.system.network.bind_address);
        assert_eq!(original_settings.system.network.port, deserialized_settings.system.network.port);
        assert_eq!(original_settings.system.persist_settings, deserialized_settings.system.persist_settings);
        
        println!("✅ Roundtrip serialization maintains consistency");
    }
    
    #[test]
    fn test_backwards_compatibility_bloom_to_glow() {
        // Test YAML with old 'bloom' field name
        let yaml_with_bloom = r#"
visualisation:
  bloom:
    enabled: true
    intensity: 1.8
    node_glow_strength: 2.5
    edge_glow_strength: 3.0
    base_color: '#00ffff'
    emission_color: '#ffffff'
system:
  persist_settings: true
"#;
        
        // Should deserialize successfully using alias
        let settings: AppFullSettings = serde_yaml::from_str(yaml_with_bloom)
            .expect("Should deserialize YAML with bloom field");
        
        assert_eq!(settings.visualisation.glow.enabled, true);
        assert_eq!(settings.visualisation.glow.intensity, 1.8);
        assert_eq!(settings.visualisation.glow.node_glow_strength, 2.5);
        assert_eq!(settings.visualisation.glow.edge_glow_strength, 3.0);
        assert_eq!(settings.visualisation.glow.base_color, "#00ffff");
        assert_eq!(settings.visualisation.glow.emission_color, "#ffffff");
        
        println!("✅ Backwards compatibility with bloom field working");
    }
    
    #[test]
    fn test_complex_nested_camelcase_conversion() {
        // Test deeply nested settings with complex field names
        let complex_json = json!({
            "visualisation": {
                "graphs": {
                    "logseq": {
                        "physics": {
                            "autoBalanceConfig": {
                                "stabilityVarianceThreshold": 150.0,
                                "stabilityFrameCount": 200,
                                "clusteringDistanceThreshold": 25.0,
                                "bouncingNodePercentage": 0.4,
                                "boundaryMinDistance": 100.0,
                                "boundaryMaxDistance": 110.0,
                                "extremeDistanceThreshold": 1200.0,
                                "explosionDistanceThreshold": 12000.0,
                                "spreadingDistanceThreshold": 600.0,
                                "oscillationDetectionFrames": 12,
                                "oscillationChangeThreshold": 6.0,
                                "minOscillationChanges": 6,
                                "gridCellSizeMin": 2.0,
                                "gridCellSizeMax": 60.0,
                                "repulsionCutoffMin": 6.0,
                                "repulsionCutoffMax": 250.0,
                                "repulsionSofteningMin": 0.000002,
                                "repulsionSofteningMax": 2.0,
                                "centerGravityMin": 0.0,
                                "centerGravityMax": 0.15,
                                "spatialHashEfficiencyThreshold": 0.4,
                                "clusterDensityThreshold": 60.0,
                                "numericalInstabilityThreshold": 0.002
                            }
                        }
                    }
                }
            }
        });
        
        let settings: AppFullSettings = serde_json::from_value(complex_json)
            .expect("Should deserialize complex nested camelCase");
        
        let auto_balance = &settings.visualisation.graphs.logseq.physics.auto_balance_config;
        assert_eq!(auto_balance.stability_variance_threshold, 150.0);
        assert_eq!(auto_balance.stability_frame_count, 200);
        assert_eq!(auto_balance.clustering_distance_threshold, 25.0);
        assert_eq!(auto_balance.bouncing_node_percentage, 0.4);
        assert_eq!(auto_balance.boundary_min_distance, 100.0);
        assert_eq!(auto_balance.boundary_max_distance, 110.0);
        assert_eq!(auto_balance.extreme_distance_threshold, 1200.0);
        assert_eq!(auto_balance.explosion_distance_threshold, 12000.0);
        assert_eq!(auto_balance.spreading_distance_threshold, 600.0);
        assert_eq!(auto_balance.oscillation_detection_frames, 12);
        assert_eq!(auto_balance.oscillation_change_threshold, 6.0);
        assert_eq!(auto_balance.min_oscillation_changes, 6);
        assert_eq!(auto_balance.grid_cell_size_min, 2.0);
        assert_eq!(auto_balance.grid_cell_size_max, 60.0);
        assert_eq!(auto_balance.repulsion_cutoff_min, 6.0);
        assert_eq!(auto_balance.repulsion_cutoff_max, 250.0);
        assert_eq!(auto_balance.repulsion_softening_min, 0.000002);
        assert_eq!(auto_balance.repulsion_softening_max, 2.0);
        assert_eq!(auto_balance.center_gravity_min, 0.0);
        assert_eq!(auto_balance.center_gravity_max, 0.15);
        assert_eq!(auto_balance.spatial_hash_efficiency_threshold, 0.4);
        assert_eq!(auto_balance.cluster_density_threshold, 60.0);
        assert_eq!(auto_balance.numerical_instability_threshold, 0.002);
        
        println!("✅ Complex nested camelCase conversion working correctly");
    }
    
    #[test]
    fn test_serialization_excludes_snake_case() {
        let settings = AppFullSettings::default();
        
        let json_str = serde_json::to_string(&settings)
            .expect("Should serialize to JSON string");
        
        // Verify JSON contains camelCase, not snake_case
        assert!(json_str.contains("nodeGlowStrength"));
        assert!(!json_str.contains("node_glow_strength"));
        
        assert!(json_str.contains("edgeGlowStrength"));
        assert!(!json_str.contains("edge_glow_strength"));
        
        assert!(json_str.contains("baseColor"));
        assert!(!json_str.contains("base_color"));
        
        assert!(json_str.contains("springK"));
        assert!(!json_str.contains("spring_k"));
        
        assert!(json_str.contains("bindAddress"));
        assert!(!json_str.contains("bind_address"));
        
        assert!(json_str.contains("persistSettings"));
        assert!(!json_str.contains("persist_settings"));
        
        println!("✅ Serialized JSON uses camelCase exclusively");
    }
}

#[cfg(test)]
mod performance_serde_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_serialization_performance() {
        let settings = AppFullSettings::default();
        let start = Instant::now();
        
        // Serialize 1000 times
        for _ in 0..1000 {
            let _json = serde_json::to_value(&settings).expect("Should serialize");
        }
        
        let duration = start.elapsed();
        println!("1000 serializations took: {:?}", duration);
        assert!(duration.as_millis() < 1000, "Serialization should be fast");
    }
    
    #[test]
    fn test_deserialization_performance() {
        let settings = AppFullSettings::default();
        let json_value = serde_json::to_value(&settings).expect("Should serialize");
        
        let start = Instant::now();
        
        // Deserialize 1000 times
        for _ in 0..1000 {
            let _settings: AppFullSettings = serde_json::from_value(json_value.clone())
                .expect("Should deserialize");
        }
        
        let duration = start.elapsed();
        println!("1000 deserializations took: {:?}", duration);
        assert!(duration.as_millis() < 1000, "Deserialization should be fast");
    }
}