//! End-to-End Settings Validation Tests
//!
//! Comprehensive validation suite for settings sync functionality,
//! focusing on the robustness of the REST API and proper handling
//! of bloom/glow field validation that was mentioned as brittle.

use actix_web::{test, web, App, http::StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

// Import project modules
use visionflow::app_state::AppState;
use visionflow::handlers::settings_handler::{config, EnhancedSettingsHandler};
use visionflow::config::AppFullSettings;
use visionflow::actors::settings_actor::SettingsActor;
use visionflow::utils::validation::ValidationService;

/// Test fixture for creating a fully configured test server
struct TestServer {
    app: Box<dyn actix_web::dev::Service<
        actix_web::dev::ServiceRequest,
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
    >>,
}

impl TestServer {
    async fn new() -> Self {
        let app_settings = AppFullSettings::new().expect("Failed to create default settings");
        let settings_actor = Arc::new(Mutex::new(SettingsActor::new(app_settings)));
        let app_state = AppState::new_test_state(settings_actor);
        
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .app_data(web::Data::new(EnhancedSettingsHandler::new()))
                .configure(config)
        ).await;
        
        Self { app: Box::new(app) }
    }
    
    async fn get(&self, path: &str) -> actix_web::dev::ServiceResponse {
        let req = test::TestRequest::get().uri(path).to_request();
        test::call_service(&*self.app, req).await
    }
    
    async fn post_json(&self, path: &str, json: Value) -> actix_web::dev::ServiceResponse {
        let req = test::TestRequest::post()
            .uri(path)
            .set_json(&json)
            .to_request();
        test::call_service(&*self.app, req).await
    }
    
    async fn post(&self, path: &str) -> actix_web::dev::ServiceResponse {
        let req = test::TestRequest::post().uri(path).to_request();
        test::call_service(&*self.app, req).await
    }
}

/// Comprehensive validation test data
mod validation_test_data {
    use super::*;
    
    pub struct BloomFieldTest {
        pub name: &'static str,
        pub settings: Value,
        pub should_pass: bool,
        pub expected_error_pattern: Option<&'static str>,
    }
    
    pub fn bloom_field_tests() -> Vec<BloomFieldTest> {
        vec![
            // Valid bloom settings - should pass
            BloomFieldTest {
                name: "Valid complete bloom settings",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "enabled": true,
                            "intensity": 2.0,
                            "radius": 0.85,
                            "threshold": 0.15,
                            "diffuseStrength": 1.5,
                            "atmosphericDensity": 0.8,
                            "volumetricIntensity": 1.2,
                            "baseColor": "#00ffff",
                            "emissionColor": "#ffffff",
                            "opacity": 0.9,
                            "pulseSpeed": 1.0,
                            "flowSpeed": 0.8,
                            "nodeGlowStrength": 3.0,
                            "edgeGlowStrength": 3.5,
                            "environmentGlowStrength": 3.0
                        }
                    }
                }),
                should_pass: true,
                expected_error_pattern: None,
            },
            
            // Edge cases - valid
            BloomFieldTest {
                name: "Minimum valid intensity",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "intensity": 0.0
                        }
                    }
                }),
                should_pass: true,
                expected_error_pattern: None,
            },
            
            BloomFieldTest {
                name: "Maximum valid radius",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "radius": 10.0
                        }
                    }
                }),
                should_pass: true,
                expected_error_pattern: None,
            },
            
            // Color validation tests
            BloomFieldTest {
                name: "Valid short hex color",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "baseColor": "#fff"
                        }
                    }
                }),
                should_pass: true,
                expected_error_pattern: None,
            },
            
            BloomFieldTest {
                name: "Valid long hex color",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "baseColor": "#abcdef"
                        }
                    }
                }),
                should_pass: true,
                expected_error_pattern: None,
            },
            
            // Invalid cases - should fail
            BloomFieldTest {
                name: "Negative intensity",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "intensity": -1.0
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("intensity"),
            },
            
            BloomFieldTest {
                name: "Invalid color format",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "baseColor": "not-a-color"
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("color"),
            },
            
            BloomFieldTest {
                name: "RGB color format (should fail)",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "baseColor": "rgb(255,255,255)"
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("color"),
            },
            
            BloomFieldTest {
                name: "Named color (should fail)",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "baseColor": "blue"
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("color"),
            },
            
            BloomFieldTest {
                name: "Out of range opacity",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "opacity": 1.5
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("opacity"),
            },
            
            BloomFieldTest {
                name: "Negative radius",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "radius": -0.5
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("radius"),
            },
            
            BloomFieldTest {
                name: "Extremely large intensity",
                settings: json!({
                    "visualisation": {
                        "glow": {
                            "intensity": 999999.0
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("intensity"),
            },
        ]
    }
    
    pub fn physics_field_tests() -> Vec<BloomFieldTest> {
        vec![
            // Valid physics settings
            BloomFieldTest {
                name: "Valid physics parameters",
                settings: json!({
                    "visualisation": {
                        "graphs": {
                            "logseq": {
                                "physics": {
                                    "enabled": true,
                                    "springK": 0.1,
                                    "repelK": 2.0,
                                    "attractionK": 0.01,
                                    "damping": 0.85,
                                    "maxVelocity": 5.0,
                                    "dt": 0.016,
                                    "iterations": 50
                                }
                            }
                        }
                    }
                }),
                should_pass: true,
                expected_error_pattern: None,
            },
            
            // Invalid physics settings
            BloomFieldTest {
                name: "Damping > 1.0",
                settings: json!({
                    "visualisation": {
                        "graphs": {
                            "logseq": {
                                "physics": {
                                    "damping": 1.5
                                }
                            }
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("damping"),
            },
            
            BloomFieldTest {
                name: "Negative iterations",
                settings: json!({
                    "visualisation": {
                        "graphs": {
                            "logseq": {
                                "physics": {
                                    "iterations": -10
                                }
                            }
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("iterations"),
            },
            
            BloomFieldTest {
                name: "Zero iterations",
                settings: json!({
                    "visualisation": {
                        "graphs": {
                            "logseq": {
                                "physics": {
                                    "iterations": 0
                                }
                            }
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("iterations"),
            },
            
            BloomFieldTest {
                name: "Negative spring constant",
                settings: json!({
                    "visualisation": {
                        "graphs": {
                            "logseq": {
                                "physics": {
                                    "springK": -0.1
                                }
                            }
                        }
                    }
                }),
                should_pass: false,
                expected_error_pattern: Some("springK"),
            },
        ]
    }
    
    pub fn stress_test_data() -> Vec<Value> {
        vec![
            // Large nested structure
            json!({
                "visualisation": {
                    "glow": {
                        "enabled": true,
                        "intensity": 2.0,
                        "radius": 0.85,
                        "threshold": 0.15,
                        "diffuseStrength": 1.5,
                        "atmosphericDensity": 0.8,
                        "volumetricIntensity": 1.2,
                        "baseColor": "#00ffff",
                        "emissionColor": "#ffffff",
                        "opacity": 0.9,
                        "pulseSpeed": 1.0,
                        "flowSpeed": 0.8,
                        "nodeGlowStrength": 3.0,
                        "edgeGlowStrength": 3.5,
                        "environmentGlowStrength": 3.0
                    },
                    "graphs": {
                        "logseq": {
                            "physics": {
                                "enabled": true,
                                "springK": 0.1,
                                "repelK": 2.0,
                                "attractionK": 0.01,
                                "gravity": 0.0001,
                                "damping": 0.85,
                                "maxVelocity": 5.0,
                                "dt": 0.016,
                                "temperature": 0.01,
                                "iterations": 50,
                                "boundsSize": 1000.0,
                                "separationRadius": 2.0
                            }
                        },
                        "visionflow": {
                            "physics": {
                                "enabled": true,
                                "springK": 0.1,
                                "repelK": 2.0,
                                "attractionK": 0.01,
                                "gravity": 0.0001,
                                "damping": 0.85,
                                "maxVelocity": 5.0,
                                "dt": 0.016,
                                "temperature": 0.01,
                                "iterations": 50,
                                "boundsSize": 1000.0,
                                "separationRadius": 2.0
                            }
                        }
                    }
                }
            }),
        ]
    }
}

/// Comprehensive API endpoint tests
#[cfg(test)]
mod api_endpoint_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_get_settings_structure() {
        let server = TestServer::new().await;
        let response = server.get("/api/settings").await;
        
        assert_eq!(response.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(response).await;
        
        // Verify core structure
        assert!(body.get("visualisation").is_some(), "Missing visualisation section");
        assert!(body.get("system").is_some(), "Missing system section");
        assert!(body.get("xr").is_some(), "Missing xr section");
        
        // Verify bloom/glow structure is present
        let glow = body.get("visualisation")
            .and_then(|v| v.get("glow"))
            .expect("Missing glow settings");
            
        assert!(glow.get("enabled").is_some(), "Missing glow.enabled");
        assert!(glow.get("intensity").is_some(), "Missing glow.intensity");
        assert!(glow.get("nodeGlowStrength").is_some(), "Missing glow.nodeGlowStrength");
        assert!(glow.get("edgeGlowStrength").is_some(), "Missing glow.edgeGlowStrength");
        
        // Verify physics structure for both graphs
        let graphs = body.get("visualisation")
            .and_then(|v| v.get("graphs"))
            .expect("Missing graphs section");
            
        let logseq_physics = graphs.get("logseq")
            .and_then(|g| g.get("physics"))
            .expect("Missing logseq physics");
            
        let visionflow_physics = graphs.get("visionflow")
            .and_then(|g| g.get("physics"))
            .expect("Missing visionflow physics");
            
        // Verify key physics parameters exist
        for physics in [logseq_physics, visionflow_physics] {
            assert!(physics.get("enabled").is_some(), "Missing physics.enabled");
            assert!(physics.get("springK").is_some(), "Missing physics.springK");
            assert!(physics.get("repelK").is_some(), "Missing physics.repelK");
            assert!(physics.get("damping").is_some(), "Missing physics.damping");
        }
    }
    
    #[tokio::test]
    async fn test_bloom_field_validation_comprehensive() {
        let server = TestServer::new().await;
        
        for test_case in validation_test_data::bloom_field_tests() {
            let response = server.post_json("/api/settings", test_case.settings).await;
            
            if test_case.should_pass {
                assert_eq!(
                    response.status(), 
                    StatusCode::OK,
                    "Test '{}' should have passed but got status: {}",
                    test_case.name,
                    response.status()
                );
            } else {
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "Test '{}' should have failed but got status: {}",
                    test_case.name,
                    response.status()
                );
                
                if let Some(expected_pattern) = test_case.expected_error_pattern {
                    let body: Value = test::read_body_json(response).await;
                    let error_message = body.get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("");
                        
                    assert!(
                        error_message.to_lowercase().contains(expected_pattern),
                        "Test '{}' error message '{}' should contain '{}'",
                        test_case.name,
                        error_message,
                        expected_pattern
                    );
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_physics_field_validation_comprehensive() {
        let server = TestServer::new().await;
        
        for test_case in validation_test_data::physics_field_tests() {
            let response = server.post_json("/api/settings", test_case.settings).await;
            
            if test_case.should_pass {
                assert_eq!(
                    response.status(), 
                    StatusCode::OK,
                    "Physics test '{}' should have passed but got status: {}",
                    test_case.name,
                    response.status()
                );
            } else {
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "Physics test '{}' should have failed but got status: {}",
                    test_case.name,
                    response.status()
                );
                
                if let Some(expected_pattern) = test_case.expected_error_pattern {
                    let body: Value = test::read_body_json(response).await;
                    let error_message = body.get("error")
                        .and_then(|e| e.as_str())
                        .unwrap_or("");
                        
                    assert!(
                        error_message.to_lowercase().contains(expected_pattern),
                        "Physics test '{}' error message '{}' should contain '{}'",
                        test_case.name,
                        error_message,
                        expected_pattern
                    );
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_settings_reset_robustness() {
        let server = TestServer::new().await;
        
        // First, modify settings with bloom data
        let bloom_update = json!({
            "visualisation": {
                "glow": {
                    "intensity": 5.0,
                    "baseColor": "#ff0000"
                }
            }
        });
        
        let update_response = server.post_json("/api/settings", bloom_update).await;
        assert_eq!(update_response.status(), StatusCode::OK);
        
        // Reset to defaults
        let reset_response = server.post("/api/settings/reset").await;
        assert_eq!(reset_response.status(), StatusCode::OK);
        
        let reset_body: Value = test::read_body_json(reset_response).await;
        
        // Verify defaults were restored
        let glow = reset_body.get("visualisation")
            .and_then(|v| v.get("glow"))
            .expect("Missing glow settings after reset");
            
        let intensity = glow.get("intensity")
            .and_then(|i| i.as_f64())
            .expect("Missing intensity after reset");
            
        assert_ne!(intensity, 5.0, "Intensity should have been reset from 5.0");
        
        let base_color = glow.get("baseColor")
            .and_then(|c| c.as_str())
            .expect("Missing baseColor after reset");
            
        assert_ne!(base_color, "#ff0000", "Color should have been reset from #ff0000");
    }
    
    #[tokio::test]
    async fn test_validation_stats_endpoint() {
        let server = TestServer::new().await;
        let response = server.get("/api/settings/validation/stats").await;
        
        assert_eq!(response.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(response).await;
        
        // Verify stats structure
        assert!(body.get("validation_service").is_some(), "Missing validation_service");
        assert!(body.get("rate_limiting").is_some(), "Missing rate_limiting");
        assert!(body.get("security_features").is_some(), "Missing security_features");
        assert!(body.get("endpoints_protected").is_some(), "Missing endpoints_protected");
        
        let security_features = body.get("security_features")
            .and_then(|f| f.as_array())
            .expect("security_features should be an array");
            
        let feature_names: Vec<&str> = security_features
            .iter()
            .filter_map(|f| f.as_str())
            .collect();
            
        assert!(feature_names.contains(&"comprehensive_input_validation"));
        assert!(feature_names.contains(&"rate_limiting"));
        assert!(feature_names.contains(&"request_size_validation"));
    }
}

/// Stress and performance tests
#[cfg(test)]
mod stress_tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_large_settings_payload() {
        let server = TestServer::new().await;
        
        for large_payload in validation_test_data::stress_test_data() {
            let response = server.post_json("/api/settings", large_payload).await;
            
            // Should handle large payloads gracefully
            assert!(
                response.status() == StatusCode::OK || 
                response.status() == StatusCode::BAD_REQUEST ||
                response.status() == StatusCode::PAYLOAD_TOO_LARGE,
                "Unexpected status for large payload: {}",
                response.status()
            );
        }
    }
    
    #[tokio::test]
    async fn test_response_time_performance() {
        let server = TestServer::new().await;
        
        let start = Instant::now();
        let response = server.get("/api/settings").await;
        let duration = start.elapsed();
        
        assert_eq!(response.status(), StatusCode::OK);
        assert!(duration.as_millis() < 500, "Settings fetch should be fast: {:?}", duration);
    }
    
    #[tokio::test]
    async fn test_concurrent_requests_handling() {
        let server = Arc::new(TestServer::new().await);
        let mut handles = Vec::new();
        
        // Launch 10 concurrent requests
        for i in 0..10 {
            let server_clone = Arc::clone(&server);
            let handle = tokio::spawn(async move {
                let update = json!({
                    "visualisation": {
                        "glow": {
                            "intensity": i as f64 * 0.1
                        }
                    }
                });
                
                server_clone.post_json("/api/settings", update).await.status()
            });
            
            handles.push(handle);
        }
        
        let results: Vec<StatusCode> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
            
        // All requests should be handled properly (success or controlled failure)
        for (i, status) in results.iter().enumerate() {
            assert!(
                !status.is_server_error(),
                "Request {} resulted in server error: {}",
                i, status
            );
        }
    }
}

/// Edge case and error handling tests
#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_malformed_json_handling() {
        let server = TestServer::new().await;
        
        let req = test::TestRequest::post()
            .uri("/api/settings")
            .set_payload("{ invalid json }")
            .insert_header(("content-type", "application/json"))
            .to_request();
            
        let response = test::call_service(&*server.app, req).await;
        
        // Should return 400 for malformed JSON
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    
    #[tokio::test]
    async fn test_empty_json_object() {
        let server = TestServer::new().await;
        let response = server.post_json("/api/settings", json!({})).await;
        
        // Empty object should be valid (no changes)
        assert_eq!(response.status(), StatusCode::OK);
    }
    
    #[tokio::test]
    async fn test_partial_bloom_updates() {
        let server = TestServer::new().await;
        
        // Test updating just intensity
        let partial_update = json!({
            "visualisation": {
                "glow": {
                    "intensity": 1.5
                }
            }
        });
        
        let response = server.post_json("/api/settings", partial_update).await;
        assert_eq!(response.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(response).await;
        let updated_intensity = body["visualisation"]["glow"]["intensity"]
            .as_f64()
            .expect("Should have updated intensity");
            
        assert_eq!(updated_intensity, 1.5);
    }
    
    #[tokio::test]
    async fn test_deep_nested_bloom_validation() {
        let server = TestServer::new().await;
        
        // Test deeply nested invalid value
        let deep_invalid = json!({
            "visualisation": {
                "glow": {
                    "intensity": {
                        "invalid": "should be number"
                    }
                }
            }
        });
        
        let response = server.post_json("/api/settings", deep_invalid).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
    
    #[tokio::test] 
    async fn test_auto_balance_sync_robustness() {
        let server = TestServer::new().await;
        
        // Test auto-balance setting synchronization across graphs
        let auto_balance_update = json!({
            "visualisation": {
                "graphs": {
                    "logseq": {
                        "physics": {
                            "autoBalance": true
                        }
                    }
                }
            }
        });
        
        let response = server.post_json("/api/settings", auto_balance_update).await;
        assert_eq!(response.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(response).await;
        
        // Verify auto-balance was applied to both graphs
        let logseq_auto_balance = body["visualisation"]["graphs"]["logseq"]["physics"]["autoBalance"]
            .as_bool()
            .expect("logseq should have autoBalance");
            
        let visionflow_auto_balance = body["visualisation"]["graphs"]["visionflow"]["physics"]["autoBalance"]
            .as_bool()
            .expect("visionflow should have autoBalance");
            
        assert!(logseq_auto_balance, "logseq autoBalance should be true");
        assert!(visionflow_auto_balance, "visionflow autoBalance should be true (synced)");
    }
}

/// Integration with external services
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_physics_endpoint_integration() {
        let server = TestServer::new().await;
        
        let physics_update = json!({
            "springK": 0.2,
            "repelK": 3.0,
            "damping": 0.95,
            "maxVelocity": 10.0,
            "iterations": 100
        });
        
        let response = server.post_json("/api/physics/update", physics_update).await;
        assert_eq!(response.status(), StatusCode::OK);
        
        // Verify the update was applied by fetching settings
        let get_response = server.get("/api/settings").await;
        assert_eq!(get_response.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(get_response).await;
        
        // Check both graphs received the physics update
        let logseq_spring_k = body["visualisation"]["graphs"]["logseq"]["physics"]["springK"]
            .as_f64()
            .expect("logseq should have updated springK");
            
        let visionflow_spring_k = body["visualisation"]["graphs"]["visionflow"]["physics"]["springK"]
            .as_f64()
            .expect("visionflow should have updated springK");
            
        assert_eq!(logseq_spring_k, 0.2);
        assert_eq!(visionflow_spring_k, 0.2);
    }
    
    #[tokio::test]
    async fn test_clustering_endpoint_integration() {
        let server = TestServer::new().await;
        
        let clustering_update = json!({
            "algorithm": "louvain",
            "clusterCount": 10,
            "resolution": 1.5,
            "iterations": 100
        });
        
        let response = server.post_json("/api/clustering/algorithm", clustering_update).await;
        assert_eq!(response.status(), StatusCode::OK);
        
        // Verify clustering settings were applied
        let get_response = server.get("/api/settings").await;
        let body: Value = test::read_body_json(get_response).await;
        
        let algorithm = body["visualisation"]["graphs"]["logseq"]["physics"]["clusteringAlgorithm"]
            .as_str()
            .expect("Should have clustering algorithm");
            
        assert_eq!(algorithm, "louvain");
    }
    
    #[tokio::test]
    async fn test_stress_optimization_endpoint() {
        let server = TestServer::new().await;
        
        let stress_update = json!({
            "stressWeight": 0.2,
            "stressAlpha": 0.3
        });
        
        let response = server.post_json("/api/stress/optimization", stress_update).await;
        assert_eq!(response.status(), StatusCode::OK);
        
        // Verify stress optimization settings were applied
        let get_response = server.get("/api/settings").await;
        let body: Value = test::read_body_json(get_response).await;
        
        let stress_weight = body["visualisation"]["graphs"]["logseq"]["physics"]["stressWeight"]
            .as_f64()
            .expect("Should have stress weight");
            
        assert_eq!(stress_weight, 0.2);
    }
}
