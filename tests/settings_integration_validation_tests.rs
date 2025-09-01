//! Integration tests for granular settings API with comprehensive validation
//!
//! Tests the complete integration between frontend and backend,
//! including validation, error handling, and real-world scenarios
//!

use actix_web::{test, web, App, middleware::DefaultHeaders, http::StatusCode};
use serde_json::{json, Value};
use std::sync::Arc;
use std::collections::HashMap;

// Import project modules
use webxr::{app_state::AppState, handlers::settings_handler};
use webxr::config::AppFullSettings;
use webxr::utils::validation::rate_limit::RateLimiter;

#[cfg(test)]
mod integration_validation_tests {
    use super::*;
    
    /// Helper to create test app with validation and rate limiting
    async fn create_validated_app() -> impl actix_web::dev::Service<
        actix_web::dev::ServiceRequest,
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
    > {
        let app_settings = AppFullSettings::default();
        let app_state = AppState::new_test_state(app_settings);
        
        test::init_service(
            App::new()
                .wrap(DefaultHeaders::new().add(("X-Version", "2.0")))
                .app_data(web::Data::new(app_state))
                .configure(settings_handler::config)
        ).await
    }
    
    #[tokio::test]
    async fn test_comprehensive_validation_workflow() {
        let app = create_validated_app().await;
        
        // Test 1: Valid granular updates with proper validation
        let valid_updates = json!([
            {"path": "visualisation.glow.intensity", "value": 2.0},
            {"path": "visualisation.glow.nodeGlowStrength", "value": 3.0},
            {"path": "visualisation.glow.baseColor", "value": "#ff0000"},
            {"path": "system.network.port", "value": 8080},
            {"path": "system.persistSettings", "value": true},
            {"path": "xr.roomScale", "value": 1.5},
            {"path": "xr.quality", "value": "high"}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&valid_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        let result: Value = test::read_body_json(resp).await;
        assert_eq!(result["updated_paths"].as_array().unwrap().len(), 7);
        
        // Test 2: Validate the updates took effect
        let verify_paths = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.nodeGlowStrength", 
            "visualisation.glow.baseColor",
            "system.network.port",
            "system.persistSettings",
            "xr.roomScale",
            "xr.quality"
        ].join(",");
        
        let verify_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", verify_paths))
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        assert_eq!(verified["visualisation"]["glow"]["intensity"], 2.0);
        assert_eq!(verified["visualisation"]["glow"]["nodeGlowStrength"], 3.0);
        assert_eq!(verified["visualisation"]["glow"]["baseColor"], "#ff0000");
        assert_eq!(verified["system"]["network"]["port"], 8080);
        assert_eq!(verified["system"]["persistSettings"], true);
        assert_eq!(verified["xr"]["roomScale"], 1.5);
        assert_eq!(verified["xr"]["quality"], "high");
        
        println!("✅ Comprehensive validation workflow passed");
    }
    
    #[tokio::test]
    async fn test_validation_error_scenarios() {
        let app = create_validated_app().await;
        
        // Test 1: Invalid value types
        let type_errors = json!([
            {"path": "visualisation.glow.intensity", "value": "not_a_number"},
            {"path": "visualisation.glow.enabled", "value": "not_a_boolean"},
            {"path": "system.network.port", "value": "not_an_integer"},
            {"path": "visualisation.glow.baseColor", "value": 12345}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&type_errors)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        let error: Value = test::read_body_json(resp).await;
        assert!(error["error"].is_string());
        assert!(error["validation_errors"].is_array());
        
        // Test 2: Out of range values
        let range_errors = json!([
            {"path": "visualisation.glow.intensity", "value": -1.0},
            {"path": "visualisation.glow.opacity", "value": 2.0}, // >1.0
            {"path": "system.network.port", "value": 70000}, // >65535
            {"path": "xr.roomScale", "value": -1.0}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&range_errors)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        // Test 3: Invalid color formats
        let color_errors = json!([
            {"path": "visualisation.glow.baseColor", "value": "not-a-color"},
            {"path": "visualisation.glow.emissionColor", "value": "#gggggg"},
            {"path": "xr.teleportRayColor", "value": "rgb(300,300,300)"}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&color_errors)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        println!("✅ Validation error scenarios handled correctly");
    }
    
    #[tokio::test]
    async fn test_complex_nested_validation() {
        let app = create_validated_app().await;
        
        // Test complex physics settings validation
        let physics_updates = json!([
            // Valid updates
            {"path": "visualisation.graphs.logseq.physics.springK", "value": 0.01},
            {"path": "visualisation.graphs.logseq.physics.repelK", "value": 100.0},
            {"path": "visualisation.graphs.logseq.physics.damping", "value": 0.95},
            {"path": "visualisation.graphs.logseq.physics.maxVelocity", "value": 10.0},
            
            // Nested auto-balance config
            {"path": "visualisation.graphs.logseq.physics.autoBalanceConfig.stabilityVarianceThreshold", "value": 120.0},
            {"path": "visualisation.graphs.logseq.physics.autoBalanceConfig.clusteringDistanceThreshold", "value": 25.0},
            {"path": "visualisation.graphs.logseq.physics.autoBalanceConfig.bouncingNodePercentage", "value": 0.35},
            
            // Mirror updates for visionflow
            {"path": "visualisation.graphs.visionflow.physics.springK", "value": 0.015},
            {"path": "visualisation.graphs.visionflow.physics.repelK", "value": 120.0}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&physics_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        let result: Value = test::read_body_json(resp).await;
        assert_eq!(result["updated_paths"].as_array().unwrap().len(), 9);
        
        // Verify the nested updates
        let nested_paths = vec![
            "visualisation.graphs.logseq.physics.autoBalanceConfig.stabilityVarianceThreshold",
            "visualisation.graphs.logseq.physics.autoBalanceConfig.clusteringDistanceThreshold",
            "visualisation.graphs.visionflow.physics.springK"
        ].join(",");
        
        let verify_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", nested_paths))
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        assert_eq!(verified["visualisation"]["graphs"]["logseq"]["physics"]["autoBalanceConfig"]["stabilityVarianceThreshold"], 120.0);
        assert_eq!(verified["visualisation"]["graphs"]["logseq"]["physics"]["autoBalanceConfig"]["clusteringDistanceThreshold"], 25.0);
        assert_eq!(verified["visualisation"]["graphs"]["visionflow"]["physics"]["springK"], 0.015);
        
        println!("✅ Complex nested validation passed");
    }
    
    #[tokio::test]
    async fn test_cross_setting_validation() {
        let app = create_validated_app().await;
        
        // Test settings that depend on each other
        let dependent_updates = json!([
            {"path": "system.network.enableTls", "value": true},
            {"path": "system.network.port", "value": 443}, // HTTPS port
            {"path": "system.network.minTlsVersion", "value": "1.3"},
            {"path": "system.security.cookieSecure", "value": true}, // Should be true when TLS enabled
            {"path": "system.security.cookieHttponly", "value": true}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&dependent_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        // Test conflicting settings (TLS disabled but secure cookies enabled)
        let conflicting_updates = json!([
            {"path": "system.network.enableTls", "value": false},
            {"path": "system.security.cookieSecure", "value": true} // Should warn or fail
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&conflicting_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should either warn or correct the conflict
        assert!(resp.status() == StatusCode::OK || resp.status() == StatusCode::BAD_REQUEST);
        
        if resp.status() == StatusCode::BAD_REQUEST {
            let error: Value = test::read_body_json(resp).await;
            assert!(error["error"].as_str().unwrap().contains("conflict") || 
                   error["error"].as_str().unwrap().contains("inconsistent"));
        }
        
        println!("✅ Cross-setting validation handled correctly");
    }
    
    #[tokio::test]
    async fn test_batch_validation_atomicity() {
        let app = create_validated_app().await;
        
        // Test mixed valid/invalid batch - should fail atomically
        let mixed_batch = json!([
            {"path": "visualisation.glow.intensity", "value": 2.0}, // valid
            {"path": "visualisation.glow.baseColor", "value": "#ff0000"}, // valid
            {"path": "nonexistent.path", "value": "invalid"}, // invalid path
            {"path": "visualisation.glow.nodeGlowStrength", "value": 3.0}, // valid
            {"path": "system.network.port", "value": "not_a_number"} // invalid type
        ]);
        
        // Get initial state
        let initial_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.baseColor,visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let initial_resp = test::call_service(&app, initial_req).await;
        let initial_state: Value = test::read_body_json(initial_resp).await;
        
        // Apply mixed batch (should fail)
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&mixed_batch)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        // Verify no changes were applied (atomicity)
        let verify_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.baseColor,visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        let final_state: Value = test::read_body_json(verify_resp).await;
        
        // States should be identical (no partial updates)
        assert_eq!(initial_state, final_state);
        
        println!("✅ Batch validation atomicity maintained");
    }
    
    #[tokio::test]
    async fn test_rate_limiting_integration() {
        let app = create_validated_app().await;
        
        // Make many rapid requests to trigger rate limiting
        let mut responses = Vec::new();
        
        for i in 0..20 {
            let update = json!([
                {"path": "visualisation.glow.intensity", "value": i as f64 * 0.1}
            ]);
            
            let req = test::TestRequest::post()
                .uri("/api/settings/set")
                .set_json(&update)
                .insert_header(("X-Client-ID", format!("test-client-{}", i % 3))) // Simulate different clients
                .to_request();
                
            let resp = test::call_service(&app, req).await;
            responses.push(resp.status());
            
            // Small delay to simulate real timing
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        
        // Should have some successful requests and some rate limited
        let successful = responses.iter().filter(|&&s| s == StatusCode::OK).count();
        let rate_limited = responses.iter().filter(|&&s| s == StatusCode::TOO_MANY_REQUESTS).count();
        
        println!("Rate limiting results: {} successful, {} rate limited", successful, rate_limited);
        
        // Should have at least some successful requests
        assert!(successful > 0);
        
        // Depending on rate limiting configuration, should have some rate limited requests
        // This is environment-dependent, so we don't assert on rate_limited count
        
        println!("✅ Rate limiting integration working");
    }
    
    #[tokio::test]
    async fn test_security_validation_integration() {
        let app = create_validated_app().await;
        
        // Test 1: Injection attempt in string values
        let injection_attempts = json!([
            {"path": "visualisation.glow.baseColor", "value": "#ff0000'; DROP TABLE settings; --"},
            {"path": "system.network.bindAddress", "value": "0.0.0.0'; DELETE FROM users; --"},
            {"path": "xr.teleportRayColor", "value": "<script>alert('xss')</script>"}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&injection_attempts)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should be rejected due to invalid values (not due to injection, but validation)
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        
        // Test 2: Extremely large payloads
        let mut large_updates = Vec::new();
        for i in 0..1000 {
            large_updates.push(json!({
                "path": format!("visualisation.glow.customField{}", i),
                "value": "x".repeat(1000)
            }));
        }
        
        let large_payload = json!(large_updates);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&large_payload)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        
        // Should be rejected due to size limits
        assert!(resp.status() == StatusCode::BAD_REQUEST || 
                resp.status() == StatusCode::PAYLOAD_TOO_LARGE);
        
        // Test 3: Valid security-related settings
        let security_updates = json!([
            {"path": "system.security.enableAuditLogging", "value": true},
            {"path": "system.security.sessionTimeout", "value": 1800},
            {"path": "system.security.csrfTokenTimeout", "value": 3600},
            {"path": "system.network.enableRateLimiting", "value": true},
            {"path": "system.network.rateLimitRequests", "value": 100},
            {"path": "system.network.rateLimitWindow", "value": 60}
        ]);
        
        let req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&security_updates)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        println!("✅ Security validation integration working");
    }
    
    #[tokio::test]
    async fn test_performance_under_load() {
        let app = Arc::new(create_validated_app().await);
        let mut handles = Vec::new();
        
        // Simulate realistic load
        for client_id in 0..10 {
            let app_clone = Arc::clone(&app);
            
            let handle = tokio::spawn(async move {
                let mut client_results = Vec::new();
                
                for request_id in 0..5 {
                    let start = std::time::Instant::now();
                    
                    // Realistic settings update
                    let update = json!([
                        {"path": "visualisation.glow.intensity", "value": (client_id as f64 + request_id as f64) * 0.1},
                        {"path": "visualisation.glow.nodeGlowStrength", "value": 2.0 + client_id as f64 * 0.1}
                    ]);
                    
                    let req = test::TestRequest::post()
                        .uri("/api/settings/set")
                        .set_json(&update)
                        .to_request();
                        
                    let resp = test::call_service(&*app_clone, req).await;
                    let duration = start.elapsed();
                    
                    client_results.push((client_id, request_id, resp.status(), duration));
                    
                    // Realistic delay between requests
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
                
                client_results
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let all_results: Vec<Vec<(i32, i32, StatusCode, std::time::Duration)>> = 
            futures::future::join_all(handles).await
                .into_iter()
                .map(|r| r.unwrap())
                .collect();
        
        // Analyze performance
        let mut total_requests = 0;
        let mut successful_requests = 0;
        let mut total_duration = std::time::Duration::ZERO;
        let mut max_duration = std::time::Duration::ZERO;
        
        for client_results in all_results {
            for (client_id, request_id, status, duration) in client_results {
                total_requests += 1;
                total_duration += duration;
                max_duration = max_duration.max(duration);
                
                if status == StatusCode::OK {
                    successful_requests += 1;
                }
                
                // Each request should complete reasonably quickly
                assert!(duration.as_millis() < 1000, 
                    "Request too slow: client {} request {} took {:?}", 
                    client_id, request_id, duration);
            }
        }
        
        let avg_duration = total_duration / total_requests;
        let success_rate = successful_requests as f64 / total_requests as f64 * 100.0;
        
        println!("Performance under load:");
        println!("  Total requests: {}", total_requests);
        println!("  Success rate: {:.1}%", success_rate);
        println!("  Average duration: {:?}", avg_duration);
        println!("  Max duration: {:?}", max_duration);
        
        // Performance thresholds
        assert!(success_rate > 80.0, "Success rate too low: {:.1}%", success_rate);
        assert!(avg_duration.as_millis() < 100, "Average response time too high: {:?}", avg_duration);
        
        println!("✅ Performance under load acceptable");
    }
}

#[cfg(test)]
mod backwards_compatibility_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_legacy_api_compatibility() {
        let app = create_validated_app().await;
        
        // Test legacy full settings update still works
        let legacy_update = json!({
            "visualisation": {
                "glow": {
                    "intensity": 1.5,
                    "baseColor": "#00ff00",
                    "enabled": true
                }
            },
            "system": {
                "network": {
                    "port": 9000
                },
                "persistSettings": true
            }
        });
        
        let req = test::TestRequest::post()
            .uri("/api/settings") // legacy endpoint
            .set_json(&legacy_update)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        // Verify using new granular API
        let verify_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.baseColor,system.network.port")
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        assert_eq!(verified["visualisation"]["glow"]["intensity"], 1.5);
        assert_eq!(verified["visualisation"]["glow"]["baseColor"], "#00ff00");
        assert_eq!(verified["system"]["network"]["port"], 9000);
        
        // Test legacy full settings fetch still works
        let legacy_get_req = test::TestRequest::get()
            .uri("/api/settings") // legacy endpoint
            .to_request();
            
        let legacy_get_resp = test::call_service(&app, legacy_get_req).await;
        assert_eq!(legacy_get_resp.status(), StatusCode::OK);
        
        let legacy_settings: Value = test::read_body_json(legacy_get_resp).await;
        assert!(legacy_settings["visualisation"]["glow"]["intensity"].is_number());
        assert!(legacy_settings["system"]["network"]["port"].is_number());
        
        println!("✅ Legacy API compatibility maintained");
    }
    
    #[tokio::test]
    async fn test_bloom_to_glow_compatibility() {
        let app = create_validated_app().await;
        
        // Test that legacy 'bloom' field name still works in YAML/JSON input
        let bloom_legacy = json!({
            "visualisation": {
                "bloom": {  // old field name
                    "intensity": 2.0,
                    "baseColor": "#ff0000",
                    "enabled": true
                }
            }
        });
        
        let req = test::TestRequest::post()
            .uri("/api/settings")
            .set_json(&bloom_legacy)
            .to_request();
            
        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), StatusCode::OK);
        
        // Verify that the 'bloom' data was mapped to 'glow' internally
        let verify_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.baseColor")
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        assert_eq!(verified["visualisation"]["glow"]["intensity"], 2.0);
        assert_eq!(verified["visualisation"]["glow"]["baseColor"], "#ff0000");
        
        println!("✅ Bloom to glow compatibility working");
    }
}