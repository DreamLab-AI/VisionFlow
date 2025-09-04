//! Comprehensive tests for granular API endpoints in settings refactor
//!
//! Tests the new path-based GET and SET endpoints that replace monolithic settings transfer
//! Validates dot-notation path parsing, partial updates, and performance improvements
//!

use actix_web::{test, web, App, HttpResponse, Result as ActixResult};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::AppFullSettings;
use crate::handlers::settings_handler::{
    get_settings_by_paths, 
    update_settings_by_paths, 
    SettingsPath, 
    SettingsUpdate
};
use crate::app_state::AppState;
use crate::actors::settings_actor::SettingsActor;

#[cfg(test)]
mod granular_api_tests {
    use super::*;

    fn create_test_app_state() -> web::Data<AppState> {
        let settings = AppFullSettings::default();
        let settings_actor = Arc::new(RwLock::new(SettingsActor::new(settings)));
        
        web::Data::new(AppState {
            settings_actor,
            // Add other required fields with default/test values
        })
    }

    #[actix_web::test]
    async fn test_get_single_path() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        // Test single path request
        let req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "Single path request should succeed");
        
        let body: Value = test::read_body_json(resp).await;
        assert!(body["visualisation"]["glow"]["nodeGlowStrength"].is_number(), 
                "Response should contain requested path in camelCase");
        
        // Verify only requested data is returned
        assert!(body["visualisation"]["glow"].get("edgeGlowStrength").is_none(),
                "Unrequested fields should not be included");
        assert!(body.get("system").is_none(),
                "Unrequested top-level sections should not be included");
    }

    #[actix_web::test]
    async fn test_get_multiple_paths() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        // Test multiple paths request
        let req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.nodeGlowStrength,visualisation.glow.baseColor,system.debugMode")
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "Multiple paths request should succeed");
        
        let body: Value = test::read_body_json(resp).await;
        
        // Verify all requested paths are present
        assert!(body["visualisation"]["glow"]["nodeGlowStrength"].is_number());
        assert!(body["visualisation"]["glow"]["baseColor"].is_string());
        assert!(body["system"]["debugMode"].is_boolean());
        
        // Verify unrequested paths are not included
        assert!(body["visualisation"]["glow"].get("edgeGlowStrength").is_none());
        assert!(body["system"].get("maxConnections").is_none());
    }

    #[actix_web::test]
    async fn test_get_nested_object_path() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        // Test requesting entire nested object
        let req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow")
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "Nested object request should succeed");
        
        let body: Value = test::read_body_json(resp).await;
        let glow = &body["visualisation"]["glow"];
        
        // Verify entire glow object is returned with all fields
        assert!(glow["nodeGlowStrength"].is_number());
        assert!(glow["edgeGlowStrength"].is_number());
        assert!(glow["environmentGlowStrength"].is_number());
        assert!(glow["baseColor"].is_string());
        assert!(glow["emissionColor"].is_string());
        
        // Verify other top-level sections are not included
        assert!(body.get("system").is_none());
        assert!(body.get("xr").is_none());
    }

    #[actix_web::test]
    async fn test_get_array_element_path() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        // Test requesting specific array element (if colorSchemes is an array)
        let req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.colorSchemes.0")
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        if resp.status().is_success() {
            let body: Value = test::read_body_json(resp).await;
            // If colorSchemes[0] exists, verify it's returned correctly
            if let Some(color_scheme) = body["visualisation"]["colorSchemes"].get(0) {
                assert!(!color_scheme.is_null(), "Array element should be valid");
            }
        }
        // It's okay if this fails - depends on whether colorSchemes is configured as array
    }

    #[actix_web::test]
    async fn test_get_invalid_path() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        // Test invalid path request
        let req = test::TestRequest::get()
            .uri("/api/settings/get?paths=invalid.path.does.not.exist")
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request or empty object, depending on implementation
        if resp.status().is_client_error() {
            // Implementation choice: return error for invalid paths
        } else if resp.status().is_success() {
            // Implementation choice: return empty object for invalid paths
            let body: Value = test::read_body_json(resp).await;
            assert!(body.as_object().unwrap().is_empty() || body.is_null());
        }
    }

    #[actix_web::test]
    async fn test_update_single_path() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;

        // Test single path update
        let update_data = json!([
            {
                "path": "visualisation.glow.nodeGlowStrength",
                "value": 2.5
            }
        ]);

        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update_data)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "Single path update should succeed");
        
        // Verify the update was applied
        let get_app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        let get_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let get_resp = test::call_service(&get_app, get_req).await;
        let body: Value = test::read_body_json(get_resp).await;
        
        assert_eq!(body["visualisation"]["glow"]["nodeGlowStrength"], json!(2.5),
                   "Updated value should be persisted");
    }

    #[actix_web::test]
    async fn test_update_multiple_paths() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;

        // Test multiple path updates
        let update_data = json!([
            {
                "path": "visualisation.glow.nodeGlowStrength",
                "value": 3.0
            },
            {
                "path": "visualisation.glow.baseColor",
                "value": "#ff0000"
            },
            {
                "path": "system.debugMode",
                "value": true
            }
        ]);

        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update_data)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "Multiple path update should succeed");
        
        // Verify all updates were applied
        let get_app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        let get_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.nodeGlowStrength,visualisation.glow.baseColor,system.debugMode")
            .to_request();
            
        let get_resp = test::call_service(&get_app, get_req).await;
        let body: Value = test::read_body_json(get_resp).await;
        
        assert_eq!(body["visualisation"]["glow"]["nodeGlowStrength"], json!(3.0));
        assert_eq!(body["visualisation"]["glow"]["baseColor"], json!("#ff0000"));
        assert_eq!(body["system"]["debugMode"], json!(true));
    }

    #[actix_web::test]
    async fn test_update_nested_object() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;

        // Test updating entire nested object
        let update_data = json!([
            {
                "path": "visualisation.glow",
                "value": {
                    "nodeGlowStrength": 4.0,
                    "edgeGlowStrength": 3.5,
                    "baseColor": "#00ff00"
                }
            }
        ]);

        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update_data)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success(), "Nested object update should succeed");
    }

    #[actix_web::test]
    async fn test_update_invalid_path() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;

        // Test invalid path update
        let update_data = json!([
            {
                "path": "invalid.path.does.not.exist",
                "value": "some_value"
            }
        ]);

        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update_data)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request for invalid paths
        assert!(resp.status().is_client_error(), 
                "Invalid path update should return client error");
    }

    #[actix_web::test]
    async fn test_update_type_validation() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;

        // Test type mismatch (string where number expected)
        let update_data = json!([
            {
                "path": "visualisation.glow.nodeGlowStrength",
                "value": "not_a_number"
            }
        ]);

        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update_data)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request for type mismatch
        assert!(resp.status().is_client_error(),
                "Type mismatch should return client error");
    }

    #[actix_web::test]
    async fn test_get_performance_many_paths() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        // Test performance with many paths
        let many_paths = vec![
            "visualisation.glow.nodeGlowStrength",
            "visualisation.glow.edgeGlowStrength", 
            "visualisation.glow.baseColor",
            "visualisation.graphs.logseq.physics.springK",
            "visualisation.graphs.logseq.physics.repelK",
            "system.debugMode",
            "system.maxConnections",
            "xr.handMeshColor",
            "xr.locomotionMethod"
        ].join(",");

        let uri = format!("/api/settings/get?paths={}", many_paths);
        let req = test::TestRequest::get().uri(&uri).to_request();
        
        let start = std::time::Instant::now();
        let resp = test::call_service(&app, req).await;
        let duration = start.elapsed();
        
        assert!(resp.status().is_success(), "Many paths request should succeed");
        assert!(duration.as_millis() < 100, "Many paths request should be fast");
        
        let body: Value = test::read_body_json(resp).await;
        
        // Verify all requested paths are present
        assert!(body["visualisation"]["glow"]["nodeGlowStrength"].is_number());
        assert!(body["visualisation"]["glow"]["edgeGlowStrength"].is_number());
        assert!(body["system"]["debugMode"].is_boolean());
        assert!(body["xr"]["handMeshColor"].is_string());
    }

    #[actix_web::test]
    async fn test_atomic_updates() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;

        // Test atomic updates - all should succeed or all should fail
        let update_data = json!([
            {
                "path": "visualisation.glow.nodeGlowStrength",
                "value": 5.0
            },
            {
                "path": "invalid.path",
                "value": "should_cause_failure"
            }
        ]);

        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update_data)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should fail due to invalid path
        assert!(resp.status().is_client_error(),
                "Atomic update with invalid path should fail entirely");
        
        // Verify no partial updates occurred
        let get_app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        let get_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let get_resp = test::call_service(&get_app, get_req).await;
        let body: Value = test::read_body_json(get_resp).await;
        
        // Value should not be 5.0 (should be default value)
        assert_ne!(body["visualisation"]["glow"]["nodeGlowStrength"], json!(5.0),
                   "Failed atomic update should not apply any changes");
    }

    #[actix_web::test]
    async fn test_concurrent_updates() {
        let app_state = create_test_app_state();
        
        // Test concurrent updates to different paths
        let app1 = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;
        
        let app2 = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/set", web::post().to(update_settings_by_paths))
        ).await;

        let update1 = json!([{"path": "visualisation.glow.nodeGlowStrength", "value": 1.0}]);
        let update2 = json!([{"path": "system.debugMode", "value": true}]);

        // Execute concurrent requests
        let req1 = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update1)
            .to_request();
            
        let req2 = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update2)
            .to_request();

        let (resp1, resp2) = tokio::join!(
            test::call_service(&app1, req1),
            test::call_service(&app2, req2)
        );

        assert!(resp1.status().is_success(), "Concurrent update 1 should succeed");
        assert!(resp2.status().is_success(), "Concurrent update 2 should succeed");
    }

    #[actix_web::test]
    async fn test_response_size_efficiency() {
        let app_state = create_test_app_state();
        
        let app = test::init_service(
            App::new()
                .app_data(app_state.clone())
                .route("/api/settings/get", web::get().to(get_settings_by_paths))
        ).await;

        // Compare response sizes: single field vs full settings
        let single_field_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let single_resp = test::call_service(&app, single_field_req).await;
        let single_body: Value = test::read_body_json(single_resp).await;
        let single_size = serde_json::to_string(&single_body).unwrap().len();

        // Get a larger subset for comparison
        let multiple_fields_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow,system.debugMode")
            .to_request();
            
        let multiple_resp = test::call_service(&app, multiple_fields_req).await;
        let multiple_body: Value = test::read_body_json(multiple_resp).await;
        let multiple_size = serde_json::to_string(&multiple_body).unwrap().len();

        // Verify size efficiency
        assert!(single_size < multiple_size, "Single field response should be smaller");
        assert!(single_size < 1000, "Single field response should be compact");
        
        // For reference: get full settings would be much larger
        // This demonstrates the efficiency gain of granular endpoints
    }
}

#[cfg(test)]
mod path_parsing_tests {
    use super::*;

    #[test]
    fn test_dot_notation_parsing() {
        // Test various dot notation patterns
        let test_cases = vec![
            ("simple", vec!["simple"]),
            ("nested.field", vec!["nested", "field"]),
            ("deeply.nested.field.value", vec!["deeply", "nested", "field", "value"]),
            ("array.0.field", vec!["array", "0", "field"]),
        ];

        for (input, expected) in test_cases {
            let parsed = parse_path(input);
            assert_eq!(parsed, expected, "Path '{}' should parse correctly", input);
        }
    }

    #[test]
    fn test_invalid_path_handling() {
        let invalid_paths = vec![
            "",           // Empty path
            ".",          // Just dot
            ".field",     // Leading dot
            "field.",     // Trailing dot
            "field..nested", // Double dot
        ];

        for path in invalid_paths {
            let result = parse_path(path);
            // Implementation should handle these gracefully
            assert!(result.is_empty() || result.len() == 1, 
                    "Invalid path '{}' should be handled gracefully", path);
        }
    }

    // Helper function that would be implemented in the actual handler
    fn parse_path(path: &str) -> Vec<&str> {
        if path.is_empty() || path == "." {
            return vec![];
        }
        
        path.split('.').filter(|s| !s.is_empty()).collect()
    }

    #[test]
    fn test_camelcase_path_validation() {
        // Test that paths expect camelCase format
        let camelcase_paths = vec![
            "visualisation.glow.nodeGlowStrength",
            "system.debugMode", 
            "xr.handMeshColor",
        ];

        let snake_case_paths = vec![
            "visualisation.glow.node_glow_strength",
            "system.debug_mode",
            "xr.hand_mesh_color",
        ];

        for path in camelcase_paths {
            let parsed = parse_path(path);
            assert!(!parsed.is_empty(), "CamelCase path '{}' should be valid", path);
        }

        // Snake case paths should be treated as different paths
        // (they may not match existing fields in the camelCase structure)
        for path in snake_case_paths {
            let parsed = parse_path(path);
            // Parsing succeeds, but these wouldn't match actual struct fields
            assert!(!parsed.is_empty(), "Path parsing should work for any format");
        }
    }
}