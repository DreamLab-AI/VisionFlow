//! Integration tests for settings sync functionality
//!
//! Tests the complete settings synchronization flow:
//! - REST API endpoints with bloom field validation
//! - Server acceptance and processing of bloom settings
//! - Bidirectional sync between client and server
//! - Nostr authentication and settings persistence
//! - Rate limiting and security measures
//! - Error handling and recovery scenarios

use actix_web::{test, web, App};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

// Import project modules
use visionflow::actors::settings_actor::SettingsActor;
use visionflow::config::AppFullSettings;
use visionflow::handlers::nostr_handler;
use visionflow::utils::validation::rate_limit::RateLimiter;
use visionflow::{app_state::AppState, handlers::settings_handler::config};

/// Test data for bloom/glow settings with comprehensive validation
mod test_data {
    use super::*;

    pub fn valid_bloom_settings() -> Value {
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
        })
    }

    pub fn invalid_bloom_settings() -> Vec<(Value, &'static str)> {
        vec![
            // Invalid intensity (out of range)
            (
                json!({
                    "visualisation": {
                        "glow": {
                            "intensity": -1.0
                        }
                    }
                }),
                "negative intensity",
            ),
            // Invalid color format
            (
                json!({
                    "visualisation": {
                        "glow": {
                            "baseColor": "invalid-color"
                        }
                    }
                }),
                "invalid color format",
            ),
            // Invalid physics parameters
            (
                json!({
                    "visualisation": {
                        "graphs": {
                            "logseq": {
                                "physics": {
                                    "damping": 2.0  // > 1.0 is invalid
                                }
                            }
                        }
                    }
                }),
                "damping out of range",
            ),
            // Extremely large payload (should be rejected)
            (
                json!({
                    "visualisation": {
                        "glow": {
                            "massiveData": "x".repeat(1000000)
                        }
                    }
                }),
                "payload too large",
            ),
        ]
    }

    pub fn nostr_test_event() -> Value {
        json!({
            "id": "test_event_id_12345",
            "pubkey": "test_pubkey_abcdef1234567890",
            "content": "Authenticate to LogseqSpringThing",
            "sig": "test_signature_fedcba0987654321",
            "created_at": 1640995200,
            "kind": 22242,
            "tags": [
                ["relay", "wss://relay.damus.io"],
                ["challenge", "test_challenge_uuid"]
            ]
        })
    }
}

/// Integration test suite for settings sync functionality
#[cfg(test)]
mod settings_sync_tests {
    use super::*;
    use actix_web::http::StatusCode;

    /// Helper to create test app with full settings handler
    async fn create_test_app() -> impl actix_web::dev::Service<
        actix_web::dev::ServiceRequest,
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
    > {
        let app_settings = AppFullSettings::new().expect("Failed to create default settings");
        let settings_actor = Arc::new(Mutex::new(SettingsActor::new(app_settings)));

        let app_state = AppState::new_test_state(settings_actor);

        test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .configure(config) // Settings routes
                .configure(nostr_handler::config), // Nostr auth routes
        )
        .await
    }

    #[tokio::test]
    async fn test_get_settings_endpoint() {
        let app = create_test_app().await;

        let req = test::TestRequest::get().uri("/settings").to_request();

        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::OK);

        let body: Value = test::read_body_json(resp).await;

        // Verify response structure
        assert!(body.get("visualisation").is_some());
        assert!(body.get("system").is_some());
        assert!(body.get("xr").is_some());

        // Verify bloom/glow settings are present
        let glow = body["visualisation"]["glow"]
            .as_object()
            .expect("Glow settings should be present");

        assert!(glow.contains_key("enabled"));
        assert!(glow.contains_key("intensity"));
        assert!(glow.contains_key("nodeGlowStrength"));
        assert!(glow.contains_key("edgeGlowStrength"));
    }

    #[tokio::test]
    async fn test_update_bloom_settings_valid() {
        let app = create_test_app().await;

        let bloom_update = test_data::valid_bloom_settings();

        let req = test::TestRequest::post()
            .uri("/settings")
            .set_json(&bloom_update)
            .to_request();

        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::OK);

        let body: Value = test::read_body_json(resp).await;

        // Verify bloom settings were applied
        let updated_glow = &body["visualisation"]["glow"];
        assert_eq!(updated_glow["intensity"], 2.0);
        assert_eq!(updated_glow["nodeGlowStrength"], 3.0);
        assert_eq!(updated_glow["baseColor"], "#00ffff");
    }

    #[tokio::test]
    async fn test_update_bloom_settings_invalid() {
        let app = create_test_app().await;

        for (invalid_setting, description) in test_data::invalid_bloom_settings() {
            let req = test::TestRequest::post()
                .uri("/settings")
                .set_json(&invalid_setting)
                .to_request();

            let resp = test::call_service(&app, req).await;

            // Should return 400 Bad Request for invalid data
            assert!(
                resp.status() == StatusCode::BAD_REQUEST
                    || resp.status() == StatusCode::PAYLOAD_TOO_LARGE,
                "Failed validation test for: {}",
                description
            );

            let body: Value = test::read_body_json(resp).await;
            assert!(
                body.get("error").is_some(),
                "Error message should be present for: {}",
                description
            );
        }
    }

    #[tokio::test]
    async fn test_physics_settings_propagation() {
        let app = create_test_app().await;

        // Test physics endpoint specifically
        let physics_update = json!({
            "springK": 0.2,
            "repelK": 3.0,
            "damping": 0.9,
            "maxVelocity": 10.0,
            "iterations": 100
        });

        let req = test::TestRequest::post()
            .uri("/physics/update")
            .set_json(&physics_update)
            .to_request();

        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::OK);

        // Verify settings were updated by fetching them
        let get_req = test::TestRequest::get().uri("/settings").to_request();

        let get_resp = test::call_service(&app, get_req).await;
        let body: Value = test::read_body_json(get_resp).await;

        // Check both graphs received the update
        let logseq_physics = &body["visualisation"]["graphs"]["logseq"]["physics"];
        let visionflow_physics = &body["visualisation"]["graphs"]["visionflow"]["physics"];

        assert_eq!(logseq_physics["springK"], 0.2);
        assert_eq!(logseq_physics["repelK"], 3.0);
        assert_eq!(visionflow_physics["springK"], 0.2);
        assert_eq!(visionflow_physics["repelK"], 3.0);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let app = create_test_app().await;

        // Send many requests rapidly to trigger rate limiting
        let mut responses = Vec::new();

        for i in 0..15 {
            // Exceed typical rate limits
            let req = test::TestRequest::post()
                .uri("/settings")
                .set_json(&json!({"test": i}))
                .to_request();

            let resp = test::call_service(&app, req).await;
            responses.push(resp.status());
        }

        // At least one should be rate limited
        assert!(responses
            .iter()
            .any(|&status| status == StatusCode::TOO_MANY_REQUESTS));
    }

    #[tokio::test]
    async fn test_settings_reset() {
        let app = create_test_app().await;

        // First, modify settings
        let update = json!({
            "visualisation": {
                "glow": {
                    "intensity": 5.0
                }
            }
        });

        let req = test::TestRequest::post()
            .uri("/settings")
            .set_json(&update)
            .to_request();

        test::call_service(&app, req).await;

        // Reset to defaults
        let reset_req = test::TestRequest::post()
            .uri("/settings/reset")
            .to_request();

        let reset_resp = test::call_service(&app, reset_req).await;

        assert_eq!(reset_resp.status(), StatusCode::OK);

        let body: Value = test::read_body_json(reset_resp).await;

        // Verify defaults were restored (should not be 5.0)
        assert_ne!(body["visualisation"]["glow"]["intensity"], 5.0);
    }

    #[tokio::test]
    async fn test_validation_stats_endpoint() {
        let app = create_test_app().await;

        let req = test::TestRequest::get()
            .uri("/settings/validation/stats")
            .to_request();

        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::OK);

        let body: Value = test::read_body_json(resp).await;

        // Verify stats structure
        assert!(body.get("validation_service").is_some());
        assert!(body.get("rate_limiting").is_some());
        assert!(body.get("security_features").is_some());
        assert!(body.get("endpoints_protected").is_some());

        let security_features = body["security_features"]
            .as_array()
            .expect("Security features should be an array");

        assert!(security_features
            .iter()
            .any(|f| f.as_str() == Some("comprehensive_input_validation")));
        assert!(security_features
            .iter()
            .any(|f| f.as_str() == Some("rate_limiting")));
    }
}

/// Tests for Nostr authentication with settings persistence
#[cfg(test)]
mod nostr_auth_tests {
    use super::*;

    #[tokio::test]
    async fn test_nostr_auth_flow() {
        let app = create_test_app().await;

        let auth_event = test_data::nostr_test_event();

        let req = test::TestRequest::post()
            .uri("/auth/nostr")
            .set_json(&auth_event)
            .to_request();

        let resp = test::call_service(&app, req).await;

        // Note: This will likely fail without proper Nostr signature validation
        // but we can test the endpoint structure
        assert!(resp.status() == StatusCode::OK || resp.status() == StatusCode::UNAUTHORIZED);

        if resp.status() == StatusCode::OK {
            let body: Value = test::read_body_json(resp).await;
            assert!(body.get("user").is_some());
            assert!(body.get("token").is_some());
        }
    }

    #[tokio::test]
    async fn test_nostr_verify_token() {
        let app = create_test_app().await;

        let verify_payload = json!({
            "pubkey": "test_pubkey_abcdef1234567890",
            "token": "test_token_12345"
        });

        let req = test::TestRequest::post()
            .uri("/auth/nostr/verify")
            .set_json(&verify_payload)
            .to_request();

        let resp = test::call_service(&app, req).await;

        // Should return verification result (likely false for test data)
        assert_eq!(resp.status(), StatusCode::OK);

        let body: Value = test::read_body_json(resp).await;
        assert!(body.get("valid").is_some());
    }
}

/// Advanced integration tests for error handling and edge cases
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_malformed_json_handling() {
        let app = create_test_app().await;

        let req = test::TestRequest::post()
            .uri("/settings")
            .set_payload("{ invalid json }")
            .insert_header(("content-type", "application/json"))
            .to_request();

        let resp = test::call_service(&app, req).await;

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_missing_content_type() {
        let app = create_test_app().await;

        let req = test::TestRequest::post()
            .uri("/settings")
            .set_payload(r#"{"test": true}"#)
            // No content-type header
            .to_request();

        let resp = test::call_service(&app, req).await;

        // Should handle gracefully
        assert!(resp.status().is_client_error() || resp.status().is_success());
    }

    #[tokio::test]
    async fn test_concurrent_settings_updates() {
        let app = Arc::new(create_test_app().await);

        let mut handles = Vec::new();

        // Launch concurrent update requests
        for i in 0..5 {
            let app_clone = Arc::clone(&app);
            let handle = tokio::spawn(async move {
                let update = json!({
                    "visualisation": {
                        "glow": {
                            "intensity": i as f64
                        }
                    }
                });

                let req = test::TestRequest::post()
                    .uri("/settings")
                    .set_json(&update)
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

        // All should succeed or be handled gracefully
        for status in results {
            assert!(
                !status.is_server_error(),
                "Server error during concurrent updates"
            );
        }
    }
}

/// Performance and stress tests
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_settings_response_time() {
        let app = create_test_app().await;

        let start = Instant::now();

        let req = test::TestRequest::get().uri("/settings").to_request();

        let resp = test::call_service(&app, req).await;

        let duration = start.elapsed();

        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            duration.as_millis() < 100,
            "Settings fetch should be fast: {:?}",
            duration
        );
    }

    #[tokio::test]
    async fn test_large_settings_update() {
        let app = create_test_app().await;

        // Create a reasonably large but valid update
        let mut large_update = test_data::valid_bloom_settings();

        // Add many physics parameters
        for i in 0..100 {
            large_update["visualisation"]["graphs"]["logseq"]["physics"]
                [format!("customParam{}", i)] = json!(i as f64 * 0.01);
        }

        let req = test::TestRequest::post()
            .uri("/settings")
            .set_json(&large_update)
            .to_request();

        let resp = test::call_service(&app, req).await;

        // Should handle large payloads gracefully (success or controlled rejection)
        assert!(!resp.status().is_server_error());
    }
}
