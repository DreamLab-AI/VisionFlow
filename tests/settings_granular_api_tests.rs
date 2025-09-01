//! Comprehensive tests for granular settings API endpoints
//!
//! Tests the new GET /api/settings/get and POST /api/settings/set endpoints
//! for efficient, partial settings management
//!

use actix_web::{test, web, App, http::StatusCode};
use serde_json::{json, Value};
use std::collections::HashMap;

// Import project modules
use webxr::{app_state::AppState, handlers::settings_handler};
use webxr::config::AppFullSettings;

#[cfg(test)]
mod granular_api_tests {
    use super::*;
    
    /// Helper to create test app with settings handler
    async fn create_test_app() -> impl actix_web::dev::Service<
        actix_web::dev::ServiceRequest,
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
    > {
        let app_settings = AppFullSettings::default();
        let app_state = AppState::new_test_state(app_settings);
        
        test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .configure(settings_handler::config)
        ).await
    }
    
    #[tokio::test]
    async fn test_get_specific_settings_paths() {
        let app = create_test_app().await;
        
        // Test getting specific settings using dot notation paths
        let paths = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.nodeGlowStrength",
            "visualisation.graphs.logseq.physics.springK",
            "system.network.port",
            "system.persistSettings"
        ];
        
        let query = paths.join(",");
        let req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", query))
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(resp).await;
        
        // Verify only requested paths are returned
        assert!(body["visualisation"]["glow"]["intensity"].is_number());
        assert!(body["visualisation"]["glow"]["nodeGlowStrength"].is_number());
        assert!(body["visualisation"]["graphs"]["logseq"]["physics"]["springK"].is_number());
        assert!(body["system"]["network"]["port"].is_number());
        assert!(body["system"]["persistSettings"].is_boolean());
        
        // Verify other paths are not included
        assert!(body["visualisation"]["glow"]["radius"].is_null() || !body["visualisation"]["glow"].as_object().unwrap().contains_key("radius"));
        assert!(body["system"]["network"]["bindAddress"].is_null() || !body["system"]["network"].as_object().unwrap().contains_key("bindAddress"));
        
        println!("✅ Granular GET endpoint returns only requested paths");
    }
    
    #[tokio::test]
    async fn test_set_specific_settings_paths() {
        let app = create_test_app().await;
        
        // Test setting specific settings using path-value pairs
        let update_data = json!([
            {"path": "visualisation.glow.intensity", "value": 2.5},
            {"path": "visualisation.glow.nodeGlowStrength", "value": 3.0},
            {"path": "visualisation.graphs.logseq.physics.springK", "value": 0.02},
            {"path": "system.network.port", "value": 8080},
            {"path": "system.persistSettings", "value": true}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&update_data)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(resp).await;
        
        // Verify updated values
        assert_eq!(body["updated_paths"].as_array().unwrap().len(), 5);
        assert!(body["updated_paths"].as_array().unwrap().contains(&json!("visualisation.glow.intensity")));
        assert!(body["updated_paths"].as_array().unwrap().contains(&json!("visualisation.glow.nodeGlowStrength")));
        assert!(body["updated_paths"].as_array().unwrap().contains(&json!("visualisation.graphs.logseq.physics.springK")));
        assert!(body["updated_paths"].as_array().unwrap().contains(&json!("system.network.port")));
        assert!(body["updated_paths"].as_array().unwrap().contains(&json!("system.persistSettings")));
        
        // Verify the changes were applied by getting the settings
        let get_req = test::TestRequest::get()
            .uri("/api/settings")
            .to_request();
            
        let get_resp = test::call_service(&app, get_req).await;
        let get_body: Value = test::read_body_json(get_resp).await;
        
        assert_eq!(get_body["visualisation"]["glow"]["intensity"], 2.5);
        assert_eq!(get_body["visualisation"]["glow"]["nodeGlowStrength"], 3.0);
        assert_eq!(get_body["visualisation"]["graphs"]["logseq"]["physics"]["springK"], 0.02);
        assert_eq!(get_body["system"]["network"]["port"], 8080);
        assert_eq!(get_body["system"]["persistSettings"], true);
        
        println!("✅ Granular SET endpoint updates only specified paths");
    }
    
    #[tokio::test]
    async fn test_nested_path_access() {
        let app = create_test_app().await;
        
        // Test deeply nested path access
        let paths = vec![
            "visualisation.graphs.logseq.physics.autoBalanceConfig.stabilityVarianceThreshold",
            "visualisation.graphs.visionflow.physics.autoBalanceConfig.clusteringDistanceThreshold",
            "system.websocket.binaryChunkSize",
            "system.security.cookieHttponly",
            "xr.movementAxes.horizontal"
        ];
        
        let query = paths.join(",");
        let req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", query))
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(resp).await;
        
        // Verify nested access works
        assert!(body["visualisation"]["graphs"]["logseq"]["physics"]["autoBalanceConfig"]["stabilityVarianceThreshold"].is_number());
        assert!(body["visualisation"]["graphs"]["visionflow"]["physics"]["autoBalanceConfig"]["clusteringDistanceThreshold"].is_number());
        assert!(body["system"]["websocket"]["binaryChunkSize"].is_number());
        assert!(body["system"]["security"]["cookieHttponly"].is_boolean());
        assert!(body["xr"]["movementAxes"]["horizontal"].is_number());
        
        println!("✅ Deeply nested path access working");
    }
    
    #[tokio::test]
    async fn test_invalid_paths_handling() {
        let app = create_test_app().await;
        
        // Test with invalid/non-existent paths
        let invalid_paths = vec![
            "nonexistent.path",
            "visualisation.invalidField",
            "system.network.nonExistentProperty"
        ];
        
        let query = invalid_paths.join(",");
        let req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", query))
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request for invalid paths
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"].is_string());
        assert!(body["invalid_paths"].is_array());
        assert_eq!(body["invalid_paths"].as_array().unwrap().len(), 3);
        
        println!("✅ Invalid paths are handled gracefully");
    }
    
    #[tokio::test]
    async fn test_type_validation_in_set() {
        let app = create_test_app().await;
        
        // Test setting values with wrong types
        let invalid_updates = json!([
            {"path": "visualisation.glow.intensity", "value": "not_a_number"},
            {"path": "visualisation.glow.enabled", "value": "not_a_boolean"},
            {"path": "system.network.port", "value": "not_an_integer"},
            {"path": "visualisation.glow.baseColor", "value": 123}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&invalid_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request for type mismatches
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"].is_string());
        assert!(body["validation_errors"].is_array());
        
        let validation_errors = body["validation_errors"].as_array().unwrap();
        assert_eq!(validation_errors.len(), 4);
        
        println!("✅ Type validation working for SET endpoint");
    }
    
    #[tokio::test]
    async fn test_atomic_updates() {
        let app = create_test_app().await;
        
        // Test that either all updates succeed or none do (atomic behavior)
        let mixed_updates = json!([
            {"path": "visualisation.glow.intensity", "value": 1.5},
            {"path": "nonexistent.path", "value": "invalid"},
            {"path": "visualisation.glow.nodeGlowStrength", "value": 2.0}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&mixed_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should fail completely due to invalid path
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        // Verify no changes were applied
        let get_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let get_resp = test::call_service(&app, get_req).await;
        let get_body: Value = test::read_body_json(get_resp).await;
        
        // Values should still be defaults (not 1.5 and 2.0)
        let default_settings = AppFullSettings::default();
        assert_eq!(get_body["visualisation"]["glow"]["intensity"], default_settings.visualisation.glow.intensity);
        assert_eq!(get_body["visualisation"]["glow"]["nodeGlowStrength"], default_settings.visualisation.glow.node_glow_strength);
        
        println!("✅ Atomic updates working - partial failures prevent all changes");
    }
    
    #[tokio::test]
    async fn test_bulk_path_retrieval_performance() {
        let app = create_test_app().await;
        
        // Test retrieving many paths at once
        let many_paths = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.nodeGlowStrength",
            "visualisation.glow.edgeGlowStrength",
            "visualisation.glow.environmentGlowStrength",
            "visualisation.glow.baseColor",
            "visualisation.glow.emissionColor",
            "visualisation.graphs.logseq.physics.springK",
            "visualisation.graphs.logseq.physics.repelK",
            "visualisation.graphs.logseq.physics.attractionK",
            "visualisation.graphs.logseq.physics.gravity",
            "visualisation.graphs.logseq.physics.damping",
            "visualisation.graphs.logseq.physics.maxVelocity",
            "system.network.port",
            "system.network.bindAddress",
            "system.network.enableTls",
            "system.network.enableHttp2",
            "system.websocket.maxConnections",
            "system.websocket.heartbeatInterval",
            "xr.roomScale",
            "xr.quality",
            "xr.interactionDistance",
            "auth.enabled",
            "auth.provider",
            "auth.required"
        ];
        
        let query = many_paths.join(",");
        
        let start = std::time::Instant::now();
        let req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", query))
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        let duration = start.elapsed();
        
        assert_eq!(resp.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(resp).await;
        
        // Verify all requested paths are present
        assert!(body["visualisation"]["glow"]["intensity"].is_number());
        assert!(body["visualisation"]["graphs"]["logseq"]["physics"]["springK"].is_number());
        assert!(body["system"]["network"]["port"].is_number());
        assert!(body["xr"]["roomScale"].is_number());
        assert!(body["auth"]["enabled"].is_boolean());
        
        // Performance should be reasonable
        assert!(duration.as_millis() < 100, "Bulk retrieval should be fast: {:?}", duration);
        
        println!("✅ Bulk path retrieval performance: {:?}", duration);
    }
    
    #[tokio::test]
    async fn test_concurrent_granular_updates() {
        let app = std::sync::Arc::new(create_test_app().await);
        
        let mut handles = Vec::new();
        
        // Launch concurrent granular updates
        for i in 0..10 {
            let app_clone = std::sync::Arc::clone(&app);
            let handle = tokio::spawn(async move {
                let update_data = json!([
                    {"path": "visualisation.glow.intensity", "value": i as f64}
                ]);
                
                let req = test::TestRequest::post()
                    .uri("/api/settings/set")
                    .set_json(&update_data)
                    .to_request();
                    
                test::call_service(&*app_clone, req).await.status()
            });
            
            handles.push(handle);
        }
        
        let results: Vec<StatusCode> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        // All should succeed
        for status in results {
            assert_eq!(status, StatusCode::OK);
        }
        
        println!("✅ Concurrent granular updates handled successfully");
    }
    
    #[tokio::test]
    async fn test_partial_object_updates() {
        let app = create_test_app().await;
        
        // Test updating only part of a nested object
        let partial_update = json!([
            {"path": "visualisation.glow.intensity", "value": 2.0},
            {"path": "visualisation.glow.baseColor", "value": "#ff0000"}
            // Not updating other glow properties
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&partial_update)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        // Get all glow settings to verify partial update
        let get_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow")
            .to_request();
            
        let get_resp = test::call_service(&app, get_req).await;
        let get_body: Value = test::read_body_json(get_resp).await;
        
        let glow = &get_body["visualisation"]["glow"];
        
        // Updated values should be changed
        assert_eq!(glow["intensity"], 2.0);
        assert_eq!(glow["baseColor"], "#ff0000");
        
        // Non-updated values should remain at defaults
        let default_settings = AppFullSettings::default();
        assert_eq!(glow["nodeGlowStrength"], default_settings.visualisation.glow.node_glow_strength);
        assert_eq!(glow["edgeGlowStrength"], default_settings.visualisation.glow.edge_glow_strength);
        assert_eq!(glow["emissionColor"], default_settings.visualisation.glow.emission_color);
        
        println!("✅ Partial object updates preserve unchanged fields");
    }
    
    #[tokio::test]
    async fn test_path_normalization() {
        let app = create_test_app().await;
        
        // Test various path formats
        let paths_with_variations = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.intensity ",  // trailing space
            " visualisation.glow.intensity",  // leading space
            "visualisation . glow . intensity", // spaces around dots
        ];
        
        for path in paths_with_variations {
            let req = test::TestRequest::get()
                .uri(&format!("/api/settings/get?paths={}", path))
                .to_request();
                
            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            
            let body: Value = test::read_body_json(resp).await;
            assert!(body["visualisation"]["glow"]["intensity"].is_number());
        }
        
        println!("✅ Path normalization handles various formats");
    }
}

#[cfg(test)]
mod granular_api_edge_cases {
    use super::*;
    
    #[tokio::test]
    async fn test_empty_paths_query() {
        let app = create_test_app().await;
        
        let req = test::TestRequest::get()
            .uri("/api/settings/get?paths=")
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request for empty paths
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"].is_string());
    }
    
    #[tokio::test]
    async fn test_missing_paths_query() {
        let app = create_test_app().await;
        
        let req = test::TestRequest::get()
            .uri("/api/settings/get")
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request when paths parameter is missing
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
    
    #[tokio::test]
    async fn test_empty_updates_array() {
        let app = create_test_app().await;
        
        let empty_updates = json!([]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&empty_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should handle empty updates gracefully
        assert_eq!(resp.status(), StatusCode::OK);
        
        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["updated_paths"].as_array().unwrap().len(), 0);
    }
    
    #[tokio::test]
    async fn test_malformed_update_objects() {
        let app = create_test_app().await;
        
        let malformed_updates = json!([
            {"path": "visualisation.glow.intensity"},  // missing value
            {"value": 2.0},  // missing path
            {"path": "visualisation.glow.baseColor", "value": "#ff0000", "extra": "field"}  // extra field
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&malformed_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should return 400 Bad Request for malformed objects
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"].is_string());
    }
}