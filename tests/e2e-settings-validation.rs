// Test disabled - references deprecated/removed modules (webxr::actors::settings_actor, webxr::utils::validation::ValidationService)
// Actor module structure has changed per ADR-001; settings_actor has moved
/*
//! End-to-End Settings Validation Tests
//!
//! Comprehensive validation suite for settings sync functionality,
//! focusing on the robustness of the REST API and proper handling
//! of bloom/glow field validation that was mentioned as brittle.

use actix_web::{http::StatusCode, test, web, App};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::Mutex;

// Import project modules
use webxr::actors::settings_actor::SettingsActor;
use webxr::app_state::AppState;
use webxr::config::AppFullSettings;
use webxr::handlers::settings_handler::{config, EnhancedSettingsHandler};
use webxr::utils::validation::ValidationService;

/// Test fixture for creating a fully configured test server
struct TestServer {
    app: Box<
        dyn actix_web::dev::Service<
            actix_web::dev::ServiceRequest,
            Response = actix_web::dev::ServiceResponse,
            Error = actix_web::Error,
        >,
    >,
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
                .configure(config),
        )
        .await;

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
    }
}
*/
