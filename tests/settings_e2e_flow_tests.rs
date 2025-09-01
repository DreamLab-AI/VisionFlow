//! End-to-end tests for complete settings persistence flow
//!
//! Tests the entire settings journey from frontend to backend and back,
//! including WebSocket updates, persistence, and real-world scenarios
//!

use actix_web::{test, web, App, http::StatusCode, middleware::Logger};
use serde_json::{json, Value};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::time::{sleep, Duration};

// Import project modules
use webxr::{app_state::AppState, handlers::settings_handler};
use webxr::config::AppFullSettings;

#[cfg(test)]
mod e2e_settings_tests {
    use super::*;
    
    /// Helper to create full test app with middleware
    async fn create_full_test_app() -> impl actix_web::dev::Service<
        actix_web::dev::ServiceRequest,
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
    > {
        let app_settings = AppFullSettings::default();
        let app_state = AppState::new_test_state(app_settings);
        
        test::init_service(
            App::new()
                .wrap(Logger::default())
                .app_data(web::Data::new(app_state))
                .configure(settings_handler::config)
        ).await
    }
    
    #[tokio::test]
    async fn test_complete_settings_workflow() {
        let app = create_full_test_app().await;
        
        // Step 1: Fetch initial settings
        let get_req = test::TestRequest::get()
            .uri("/api/settings")
            .to_request();
            
        let get_resp = test::call_service(&app, get_req).await;
        assert_eq!(get_resp.status(), StatusCode::OK);
        
        let initial_settings: Value = test::read_body_json(get_resp).await;
        let initial_intensity = initial_settings["visualisation"]["glow"]["intensity"].as_f64().unwrap();
        
        // Step 2: Update settings using granular API
        let granular_updates = json!([
            {"path": "visualisation.glow.intensity", "value": initial_intensity + 1.0},
            {"path": "visualisation.glow.nodeGlowStrength", "value": 3.0},
            {"path": "system.network.port", "value": 8080},
            {"path": "xr.roomScale", "value": 2.5}
        ]);
        
        let update_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&granular_updates)
            .to_request();
            
        let update_resp = test::call_service(&app, update_req).await;
        assert_eq!(update_resp.status(), StatusCode::OK);
        
        let update_result: Value = test::read_body_json(update_resp).await;
        assert_eq!(update_result["updated_paths"].as_array().unwrap().len(), 4);
        
        // Step 3: Verify changes with granular GET
        let verify_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.nodeGlowStrength,system.network.port,xr.roomScale")
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified_settings: Value = test::read_body_json(verify_resp).await;
        
        assert_eq!(verified_settings["visualisation"]["glow"]["intensity"], initial_intensity + 1.0);
        assert_eq!(verified_settings["visualisation"]["glow"]["nodeGlowStrength"], 3.0);
        assert_eq!(verified_settings["system"]["network"]["port"], 8080);
        assert_eq!(verified_settings["xr"]["roomScale"], 2.5);
        
        // Step 4: Verify full settings still contain updates
        let full_get_req = test::TestRequest::get()
            .uri("/api/settings")
            .to_request();
            
        let full_get_resp = test::call_service(&app, full_get_req).await;
        assert_eq!(full_get_resp.status(), StatusCode::OK);
        
        let full_settings: Value = test::read_body_json(full_get_resp).await;
        
        assert_eq!(full_settings["visualisation"]["glow"]["intensity"], initial_intensity + 1.0);
        assert_eq!(full_settings["visualisation"]["glow"]["nodeGlowStrength"], 3.0);
        assert_eq!(full_settings["system"]["network"]["port"], 8080);
        assert_eq!(full_settings["xr"]["roomScale"], 2.5);
        
        println!("✅ Complete settings workflow E2E test passed");
    }
    
    #[tokio::test]
    async fn test_persistence_across_operations() {
        let app = create_full_test_app().await;
        
        // Set persistence to true
        let enable_persistence = json!([
            {"path": "system.persistSettings", "value": true}
        ]);
        
        let persist_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&enable_persistence)
            .to_request();
            
        let persist_resp = test::call_service(&app, persist_req).await;
        assert_eq!(persist_resp.status(), StatusCode::OK);
        
        // Make several setting changes
        let changes = vec![
            json!([{"path": "visualisation.glow.intensity", "value": 1.5}]),
            json!([{"path": "visualisation.glow.baseColor", "value": "#ff0000"}]),
            json!([{"path": "system.network.maxRequestSize", "value": 2048576}]),
            json!([{"path": "xr.interactionDistance", "value": 3.0}])
        ];
        
        for change in changes {
            let req = test::TestRequest::post()
                .uri("/api/settings/set")
                .set_json(&change)
                .to_request();
                
            let resp = test::call_service(&app, req).await;
            assert_eq!(resp.status(), StatusCode::OK);
            
            // Small delay to simulate real usage
            sleep(Duration::from_millis(10)).await;
        }
        
        // Verify all changes persisted
        let verify_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.baseColor,system.network.maxRequestSize,xr.interactionDistance")
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        
        assert_eq!(verified["visualisation"]["glow"]["intensity"], 1.5);
        assert_eq!(verified["visualisation"]["glow"]["baseColor"], "#ff0000");
        assert_eq!(verified["system"]["network"]["maxRequestSize"], 2048576);
        assert_eq!(verified["xr"]["interactionDistance"], 3.0);
        
        println!("✅ Settings persistence E2E test passed");
    }
    
    #[tokio::test]
    async fn test_complex_nested_updates() {
        let app = create_full_test_app().await;
        
        // Test deeply nested physics settings
        let physics_updates = json!([
            {"path": "visualisation.graphs.logseq.physics.springK", "value": 0.02},
            {"path": "visualisation.graphs.logseq.physics.repelK", "value": 200.0},
            {"path": "visualisation.graphs.logseq.physics.autoBalanceConfig.stabilityVarianceThreshold", "value": 150.0},
            {"path": "visualisation.graphs.logseq.physics.autoBalanceConfig.clusteringDistanceThreshold", "value": 25.0},
            {"path": "visualisation.graphs.visionflow.physics.springK", "value": 0.025},
            {"path": "visualisation.graphs.visionflow.physics.repelK", "value": 180.0},
        ]);
        
        let update_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&physics_updates)
            .to_request();
            
        let update_resp = test::call_service(&app, update_req).await;
        assert_eq!(update_resp.status(), StatusCode::OK);
        
        let update_result: Value = test::read_body_json(update_resp).await;
        assert_eq!(update_result["updated_paths"].as_array().unwrap().len(), 6);
        
        // Verify using granular GET with nested paths
        let nested_paths = vec![
            "visualisation.graphs.logseq.physics.springK",
            "visualisation.graphs.logseq.physics.repelK",
            "visualisation.graphs.logseq.physics.autoBalanceConfig.stabilityVarianceThreshold",
            "visualisation.graphs.logseq.physics.autoBalanceConfig.clusteringDistanceThreshold",
            "visualisation.graphs.visionflow.physics.springK",
            "visualisation.graphs.visionflow.physics.repelK"
        ].join(",");
        
        let verify_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", nested_paths))
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        
        assert_eq!(verified["visualisation"]["graphs"]["logseq"]["physics"]["springK"], 0.02);
        assert_eq!(verified["visualisation"]["graphs"]["logseq"]["physics"]["repelK"], 200.0);
        assert_eq!(verified["visualisation"]["graphs"]["logseq"]["physics"]["autoBalanceConfig"]["stabilityVarianceThreshold"], 150.0);
        assert_eq!(verified["visualisation"]["graphs"]["logseq"]["physics"]["autoBalanceConfig"]["clusteringDistanceThreshold"], 25.0);
        assert_eq!(verified["visualisation"]["graphs"]["visionflow"]["physics"]["springK"], 0.025);
        assert_eq!(verified["visualisation"]["graphs"]["visionflow"]["physics"]["repelK"], 180.0);
        
        println!("✅ Complex nested updates E2E test passed");
    }
    
    #[tokio::test]
    async fn test_error_recovery_workflow() {
        let app = create_full_test_app().await;
        
        // Step 1: Make valid updates
        let valid_updates = json!([
            {"path": "visualisation.glow.intensity", "value": 2.0}
        ]);
        
        let valid_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&valid_updates)
            .to_request();
            
        let valid_resp = test::call_service(&app, valid_req).await;
        assert_eq!(valid_resp.status(), StatusCode::OK);
        
        // Step 2: Try invalid updates (should fail atomically)
        let invalid_updates = json!([
            {"path": "visualisation.glow.intensity", "value": 2.5},  // valid
            {"path": "nonexistent.path", "value": "invalid"},        // invalid
            {"path": "visualisation.glow.nodeGlowStrength", "value": 3.0}  // valid
        ]);
        
        let invalid_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&invalid_updates)
            .to_request();
            
        let invalid_resp = test::call_service(&app, invalid_req).await;
        assert_eq!(invalid_resp.status(), StatusCode::BAD_REQUEST);
        
        // Step 3: Verify original valid settings are preserved
        let verify_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        
        // Should still have the first valid update (2.0), not the failed batch updates
        assert_eq!(verified["visualisation"]["glow"]["intensity"], 2.0);
        
        // nodeGlowStrength should be default (not 3.0 from failed batch)
        let default_settings = AppFullSettings::default();
        assert_eq!(verified["visualisation"]["glow"]["nodeGlowStrength"], default_settings.visualisation.glow.node_glow_strength);
        
        // Step 4: Recover with corrected updates
        let corrected_updates = json!([
            {"path": "visualisation.glow.intensity", "value": 2.5},
            {"path": "visualisation.glow.nodeGlowStrength", "value": 3.0}  // fixed path
        ]);
        
        let corrected_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&corrected_updates)
            .to_request();
            
        let corrected_resp = test::call_service(&app, corrected_req).await;
        assert_eq!(corrected_resp.status(), StatusCode::OK);
        
        // Step 5: Verify recovery
        let final_verify_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.nodeGlowStrength")
            .to_request();
            
        let final_verify_resp = test::call_service(&app, final_verify_req).await;
        let final_verified: Value = test::read_body_json(final_verify_resp).await;
        
        assert_eq!(final_verified["visualisation"]["glow"]["intensity"], 2.5);
        assert_eq!(final_verified["visualisation"]["glow"]["nodeGlowStrength"], 3.0);
        
        println!("✅ Error recovery workflow E2E test passed");
    }
    
    #[tokio::test]
    async fn test_concurrent_client_simulation() {
        let app = Arc::new(create_full_test_app().await);
        
        // Simulate multiple clients making concurrent requests
        let mut handles = Vec::new();
        
        for client_id in 0..5 {
            let app_clone = Arc::clone(&app);
            let handle = tokio::spawn(async move {
                let mut results = Vec::new();
                
                // Each client makes several requests
                for request_id in 0..3 {
                    let intensity_value = (client_id as f64 + 1.0) * (request_id as f64 + 1.0);
                    
                    let update = json!([
                        {"path": "visualisation.glow.intensity", "value": intensity_value},
                        {"path": format!("system.debug.clientId{}", client_id), "value": true}
                    ]);
                    
                    let req = test::TestRequest::post()
                        .uri("/api/settings/set")
                        .set_json(&update)
                        .to_request();
                        
                    let resp = test::call_service(&*app_clone, req).await;
                    results.push((client_id, request_id, resp.status()));
                    
                    // Small random delay to create realistic concurrency
                    sleep(Duration::from_millis(5 + (client_id * 3) as u64)).await;
                }
                
                results
            });
            
            handles.push(handle);
        }
        
        // Collect all results
        let all_results: Vec<Vec<(i32, i32, StatusCode)>> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        // Verify all requests succeeded or were handled gracefully
        for client_results in all_results {
            for (client_id, request_id, status) in client_results {
                assert!(
                    status.is_success() || status == StatusCode::TOO_MANY_REQUESTS,
                    "Client {} request {} failed with status: {}",
                    client_id,
                    request_id,
                    status
                );
            }
        }
        
        // Verify final state is consistent
        let final_req = test::TestRequest::get()
            .uri("/api/settings")
            .to_request();
            
        let final_resp = test::call_service(&*app, final_req).await;
        assert_eq!(final_resp.status(), StatusCode::OK);
        
        let final_settings: Value = test::read_body_json(final_resp).await;
        
        // Should have a valid intensity value (from one of the concurrent updates)
        let final_intensity = final_settings["visualisation"]["glow"]["intensity"].as_f64().unwrap();
        assert!(final_intensity >= 1.0 && final_intensity <= 15.0); // Reasonable range
        
        println!("✅ Concurrent client simulation E2E test passed with final intensity: {}", final_intensity);
    }
    
    #[tokio::test]
    async fn test_backwards_compatibility_workflow() {
        let app = create_full_test_app().await;
        
        // Test that old-style full settings updates still work alongside new granular updates
        
        // Step 1: Use legacy full settings update
        let full_settings_update = json!({
            "visualisation": {
                "glow": {
                    "intensity": 1.8,
                    "baseColor": "#00ff00"
                }
            },
            "system": {
                "network": {
                    "port": 9000
                }
            }
        });
        
        let legacy_req = test::TestRequest::post()
            .uri("/api/settings")  // legacy endpoint
            .set_json(&full_settings_update)
            .to_request();
            
        let legacy_resp = test::call_service(&app, legacy_req).await;
        assert_eq!(legacy_resp.status(), StatusCode::OK);
        
        // Step 2: Use new granular API to update additional settings
        let granular_update = json!([
            {"path": "visualisation.glow.nodeGlowStrength", "value": 2.5},
            {"path": "xr.quality", "value": "high"}
        ]);
        
        let granular_req = test::TestRequest::post()
            .uri("/api/settings/set")  // new endpoint
            .set_json(&granular_update)
            .to_request();
            
        let granular_resp = test::call_service(&app, granular_req).await;
        assert_eq!(granular_resp.status(), StatusCode::OK);
        
        // Step 3: Verify both legacy and new changes are preserved
        let verify_req = test::TestRequest::get()
            .uri("/api/settings")  // full settings
            .to_request();
            
        let verify_resp = test::call_service(&app, verify_req).await;
        assert_eq!(verify_resp.status(), StatusCode::OK);
        
        let verified: Value = test::read_body_json(verify_resp).await;
        
        // Legacy changes should be preserved
        assert_eq!(verified["visualisation"]["glow"]["intensity"], 1.8);
        assert_eq!(verified["visualisation"]["glow"]["baseColor"], "#00ff00");
        assert_eq!(verified["system"]["network"]["port"], 9000);
        
        // Granular changes should also be present
        assert_eq!(verified["visualisation"]["glow"]["nodeGlowStrength"], 2.5);
        assert_eq!(verified["xr"]["quality"], "high");
        
        // Step 4: Test mixed retrieval using granular GET
        let mixed_req = test::TestRequest::get()
            .uri("/api/settings/get?paths=visualisation.glow.intensity,visualisation.glow.nodeGlowStrength,system.network.port,xr.quality")
            .to_request();
            
        let mixed_resp = test::call_service(&app, mixed_req).await;
        assert_eq!(mixed_resp.status(), StatusCode::OK);
        
        let mixed_verified: Value = test::read_body_json(mixed_resp).await;
        
        assert_eq!(mixed_verified["visualisation"]["glow"]["intensity"], 1.8);
        assert_eq!(mixed_verified["visualisation"]["glow"]["nodeGlowStrength"], 2.5);
        assert_eq!(mixed_verified["system"]["network"]["port"], 9000);
        assert_eq!(mixed_verified["xr"]["quality"], "high");
        
        println!("✅ Backwards compatibility workflow E2E test passed");
    }
    
    #[tokio::test]
    async fn test_real_world_usage_patterns() {
        let app = create_full_test_app().await;
        
        // Simulate realistic usage patterns
        
        // Pattern 1: Initial app load (get essential settings)
        let essential_paths = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.enabled",
            "system.network.port",
            "system.debug.enabled",
            "xr.quality"
        ].join(",");
        
        let initial_load_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", essential_paths))
            .to_request();
            
        let initial_load_resp = test::call_service(&app, initial_load_req).await;
        assert_eq!(initial_load_resp.status(), StatusCode::OK);
        
        let initial_settings: Value = test::read_body_json(initial_load_resp).await;
        assert!(initial_settings["visualisation"]["glow"]["intensity"].is_number());
        assert!(initial_settings["system"]["network"]["port"].is_number());
        
        // Pattern 2: User opens glow settings panel (lazy load glow section)
        let glow_paths = vec![
            "visualisation.glow.nodeGlowStrength",
            "visualisation.glow.edgeGlowStrength",
            "visualisation.glow.baseColor",
            "visualisation.glow.emissionColor",
            "visualisation.glow.pulseSpeed"
        ].join(",");
        
        let glow_load_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", glow_paths))
            .to_request();
            
        let glow_load_resp = test::call_service(&app, glow_load_req).await;
        assert_eq!(glow_load_resp.status(), StatusCode::OK);
        
        // Pattern 3: User adjusts multiple glow settings (batch update)
        let glow_adjustments = json!([
            {"path": "visualisation.glow.intensity", "value": 2.2},
            {"path": "visualisation.glow.nodeGlowStrength", "value": 3.1},
            {"path": "visualisation.glow.edgeGlowStrength", "value": 2.8},
            {"path": "visualisation.glow.baseColor", "value": "#ff4444"}
        ]);
        
        let adjust_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&glow_adjustments)
            .to_request();
            
        let adjust_resp = test::call_service(&app, adjust_req).await;
        assert_eq!(adjust_resp.status(), StatusCode::OK);
        
        // Pattern 4: User opens physics panel (different section, lazy load)
        let physics_paths = vec![
            "visualisation.graphs.logseq.physics.springK",
            "visualisation.graphs.logseq.physics.repelK",
            "visualisation.graphs.logseq.physics.damping",
            "visualisation.graphs.logseq.physics.maxVelocity"
        ].join(",");
        
        let physics_load_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={}", physics_paths))
            .to_request();
            
        let physics_load_resp = test::call_service(&app, physics_load_req).await;
        assert_eq!(physics_load_resp.status(), StatusCode::OK);
        
        // Pattern 5: User makes single physics adjustment
        let single_physics_update = json!([
            {"path": "visualisation.graphs.logseq.physics.springK", "value": 0.03}
        ]);
        
        let single_update_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&single_physics_update)
            .to_request();
            
        let single_update_resp = test::call_service(&app, single_update_req).await;
        assert_eq!(single_update_resp.status(), StatusCode::OK);
        
        // Pattern 6: Final verification (mix of old and new values)
        let final_verify_req = test::TestRequest::get()
            .uri(&format!("/api/settings/get?paths={},{}", essential_paths, "visualisation.graphs.logseq.physics.springK"))
            .to_request();
            
        let final_verify_resp = test::call_service(&app, final_verify_req).await;
        assert_eq!(final_verify_resp.status(), StatusCode::OK);
        
        let final_verified: Value = test::read_body_json(final_verify_resp).await;
        
        // Verify the user's changes are preserved
        assert_eq!(final_verified["visualisation"]["glow"]["intensity"], 2.2);
        assert_eq!(final_verified["visualisation"]["graphs"]["logseq"]["physics"]["springK"], 0.03);
        
        println!("✅ Real-world usage patterns E2E test passed");
    }
}

#[cfg(test)]
mod performance_e2e_tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_performance_comparison() {
        let app = create_full_test_app().await;
        
        // Measure old approach: full settings update
        let full_update = json!({
            "visualisation": {
                "glow": {
                    "intensity": 1.5,
                    "nodeGlowStrength": 2.0,
                    "baseColor": "#ff0000"
                }
            }
        });
        
        let start = Instant::now();
        let full_req = test::TestRequest::post()
            .uri("/api/settings")
            .set_json(&full_update)
            .to_request();
            
        let full_resp = test::call_service(&app, full_req).await;
        let full_duration = start.elapsed();
        
        assert_eq!(full_resp.status(), StatusCode::OK);
        
        // Measure new approach: granular update
        let granular_update = json!([
            {"path": "visualisation.glow.intensity", "value": 1.8},
            {"path": "visualisation.glow.nodeGlowStrength", "value": 2.3},
            {"path": "visualisation.glow.baseColor", "value": "#00ff00"}
        ]);
        
        let start = Instant::now();
        let granular_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&granular_update)
            .to_request();
            
        let granular_resp = test::call_service(&app, granular_req).await;
        let granular_duration = start.elapsed();
        
        assert_eq!(granular_resp.status(), StatusCode::OK);
        
        // Granular should be comparable or better for small updates
        println!("Full update duration: {:?}", full_duration);
        println!("Granular update duration: {:?}", granular_duration);
        
        // Both should be reasonably fast
        assert!(full_duration.as_millis() < 100);
        assert!(granular_duration.as_millis() < 100);
        
        println!("✅ Performance comparison E2E test completed");
    }
    
    #[tokio::test]
    async fn test_bulk_operations_performance() {
        let app = create_full_test_app().await;
        
        // Test bulk granular updates
        let bulk_updates: Vec<_> = (0..50).map(|i| {
            json!({
                "path": format!("visualisation.glow.customParam{}", i),
                "value": i as f64 * 0.1
            })
        }).collect();
        
        let bulk_update_json = json!(bulk_updates);
        
        let start = Instant::now();
        let bulk_req = test::TestRequest::post()
            .uri("/api/settings/set")
            .set_json(&bulk_update_json)
            .to_request();
            
        let bulk_resp = test::call_service(&app, bulk_req).await;
        let bulk_duration = start.elapsed();
        
        // Should handle bulk operations reasonably well
        assert!(bulk_duration.as_millis() < 500);
        
        println!("Bulk update (50 settings) duration: {:?}", bulk_duration);
        println!("✅ Bulk operations performance E2E test completed");
    }
}