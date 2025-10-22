//! Comprehensive Settings API Test Module
//! Tests all CRUD operations, field conversions, batch operations, and edge cases

#[cfg(test)]
mod settings_api_tests {
    use super::*;
    use crate::actors::settings_actor::SettingsActor;
    use crate::app_state::AppState;
    use crate::config::AppFullSettings;
    use crate::handlers::settings_paths::{
        batch_read_settings_by_path, batch_update_settings_by_path, get_settings_by_path,
        get_settings_schema, update_settings_by_path, BatchPathReadRequest, BatchPathUpdateRequest,
        PathQuery, PathUpdateRequest,
    };
    use actix::Actor;
    use actix_web::{test, web, App, HttpResponse, Result as ActixResult};
    use serde_json::{json, Value};
    use std::sync::Arc;

    /// Create test app state with settings actor
    async fn create_test_app_state() -> web::Data<AppState> {
        let settings = AppFullSettings::default();
        let settings_actor = SettingsActor::new(settings).start();

        // Create minimal app state for testing
        let app_state = AppState {
            settings_addr: settings_actor,
        };

        web::Data::new(app_state)
    }

    /// Helper function to create test app with settings routes
    async fn create_test_app() -> App<
        impl actix_web::dev::ServiceFactory<
            actix_web::dev::ServiceRequest,
            Config = (),
            Response = actix_web::dev::ServiceResponse,
            Error = actix_web::Error,
            InitError = (),
        >,
    > {
        let app_state = create_test_app_state().await;

        test::init_service(
            App::new().app_data(app_state).service(
                web::scope("/api/settings")
                    .route("/path", web::get().to(get_settings_by_path))
                    .route("/path", web::put().to(update_settings_by_path))
                    .route("/batch", web::post().to(batch_read_settings_by_path))
                    .route("/batch", web::put().to(batch_update_settings_by_path))
                    .route("/schema", web::get().to(get_settings_schema)),
            ),
        )
        .await
    }

    #[actix_web::test]
    async fn test_get_settings_by_path() {
        let app = create_test_app().await;

        // Test getting a simple path
        let req = test::TestRequest::get()
            .uri("/api/settings/path?path=visualisation.physics.damping")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["path"], "visualisation.physics.damping");
        assert!(body["value"].is_number());
        assert_eq!(body["success"], true);
    }

    #[actix_web::test]
    async fn test_get_settings_by_path_not_found() {
        let app = create_test_app().await;

        // Test getting a non-existent path
        let req = test::TestRequest::get()
            .uri("/api/settings/path?path=nonexistent.path")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 404);

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["success"], false);
        assert!(body["error"].as_str().unwrap().contains("Path not found"));
    }

    #[actix_web::test]
    async fn test_get_settings_empty_path() {
        let app = create_test_app().await;

        // Test getting with empty path
        let req = test::TestRequest::get()
            .uri("/api/settings/path?path=")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["error"], "Path cannot be empty");
    }

    #[actix_web::test]
    async fn test_update_settings_by_path() {
        let app = create_test_app().await;

        // Test updating a simple physics parameter
        let update_data = json!({
            "path": "visualisation.physics.damping",
            "value": 0.95
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["success"], true);
        assert_eq!(body["path"], "visualisation.physics.damping");
        assert_eq!(body["value"], 0.95);
    }

    #[actix_web::test]
    async fn test_field_conversion_base_color() {
        let app = create_test_app().await;

        // Test the problematic baseColor field that was causing duplicate field errors
        let update_data = json!({
            "path": "visualisation.graphs.logseq.nodes.baseColor",
            "value": "#FF5733"
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(
            resp.status().is_success(),
            "baseColor update should succeed after normalization fix"
        );

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["success"], true);
        assert_eq!(body["value"], "#FF5733");
    }

    #[actix_web::test]
    async fn test_field_conversion_ambient_light_intensity() {
        let app = create_test_app().await;

        // Test the problematic ambientLightIntensity field
        let update_data = json!({
            "path": "visualisation.rendering.ambientLightIntensity",
            "value": 0.8
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(
            resp.status().is_success(),
            "ambientLightIntensity update should succeed after normalization fix"
        );

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["success"], true);
        assert_eq!(body["value"], 0.8);
    }

    #[actix_web::test]
    async fn test_field_conversion_emission_color() {
        let app = create_test_app().await;

        // Test emissionColor field conversion
        let update_data = json!({
            "path": "visualisation.graphs.logseq.nodes.emissionColor",
            "value": "#00FF00"
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(
            resp.status().is_success(),
            "emissionColor update should succeed"
        );
    }

    #[actix_web::test]
    async fn test_complex_nested_settings_update() {
        let app = create_test_app().await;

        // Test updating deeply nested physics auto-balance settings
        let update_data = json!({
            "path": "visualisation.graphs.logseq.physics.autoBalance.stabilityVarianceThreshold",
            "value": 0.05
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(
            resp.status().is_success(),
            "Complex nested field update should succeed"
        );

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["success"], true);
        assert_eq!(body["value"], 0.05);
    }

    #[actix_web::test]
    async fn test_batch_read_settings() {
        let app = create_test_app().await;

        // Test batch reading multiple settings paths
        let batch_data = json!({
            "paths": [
                "visualisation.physics.damping",
                "visualisation.physics.gravity",
                "visualisation.graphs.logseq.nodes.baseColor",
                "visualisation.rendering.ambientLightIntensity"
            ]
        });

        let req = test::TestRequest::post()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let body: Value = test::read_body_json(resp).await;

        // Verify all requested paths are present in response
        assert!(body["visualisation.physics.damping"].is_number());
        assert!(body["visualisation.physics.gravity"].is_number());
        assert!(body["visualisation.graphs.logseq.nodes.baseColor"].is_string());
        assert!(body["visualisation.rendering.ambientLightIntensity"].is_number());
    }

    #[actix_web::test]
    async fn test_batch_read_empty_paths() {
        let app = create_test_app().await;

        // Test batch read with no paths
        let batch_data = json!({
            "paths": []
        });

        let req = test::TestRequest::post()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["error"], "No paths provided");
    }

    #[actix_web::test]
    async fn test_batch_read_too_many_paths() {
        let app = create_test_app().await;

        // Test batch read with more than 50 paths
        let mut paths = Vec::new();
        for i in 0..51 {
            paths.push(format!("fake.path.{}", i));
        }

        let batch_data = json!({
            "paths": paths
        });

        let req = test::TestRequest::post()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);

        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("exceeds maximum of 50"));
    }

    #[actix_web::test]
    async fn test_batch_update_settings() {
        let app = create_test_app().await;

        // Test batch updating multiple settings with field conversion
        let batch_data = json!({
            "updates": [
                {
                    "path": "visualisation.physics.damping",
                    "value": 0.98
                },
                {
                    "path": "visualisation.graphs.logseq.nodes.baseColor",
                    "value": "#FF0000"
                },
                {
                    "path": "visualisation.rendering.ambientLightIntensity",
                    "value": 0.7
                }
            ]
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(
            resp.status().is_success(),
            "Batch update should succeed with field normalization"
        );

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["success"], true);

        let results = body["results"].as_array().unwrap();
        assert_eq!(results.len(), 3);

        // Verify all updates succeeded
        for result in results {
            assert_eq!(result["success"], true);
        }
    }

    #[actix_web::test]
    async fn test_batch_update_empty_updates() {
        let app = create_test_app().await;

        // Test batch update with no updates
        let batch_data = json!({
            "updates": []
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["error"], "No updates provided");
    }

    #[actix_web::test]
    async fn test_batch_update_too_many_updates() {
        let app = create_test_app().await;

        // Test batch update with more than 50 updates
        let mut updates = Vec::new();
        for i in 0..51 {
            updates.push(json!({
                "path": format!("fake.path.{}", i),
                "value": i
            }));
        }

        let batch_data = json!({
            "updates": updates
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(resp.status(), 400);

        let body: Value = test::read_body_json(resp).await;
        assert!(body["error"]
            .as_str()
            .unwrap()
            .contains("exceeds maximum of 50"));
    }

    #[actix_web::test]
    async fn test_get_settings_schema() {
        let app = create_test_app().await;

        // Test getting schema for physics settings
        let req = test::TestRequest::get()
            .uri("/api/settings/schema?path=visualisation.physics")
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["path"], "visualisation.physics");
        assert!(body["schema"]["type"] == "object");
        assert!(body["schema"]["properties"].is_object());
        assert_eq!(body["success"], true);
    }

    #[actix_web::test]
    async fn test_field_normalization_edge_cases() {
        let app = create_test_app().await;

        // Test various edge case field names that could cause conflicts
        let test_cases = vec![
            ("visualisation.graphs.logseq.edges.arrowSize", 0.5),
            ("visualisation.graphs.logseq.edges.baseWidth", 1.2),
            ("visualisation.graphs.logseq.physics.enableBounds", true),
            ("visualisation.graphs.logseq.physics.maxVelocity", 10.0),
            ("visualisation.graphs.logseq.physics.repelK", 1000.0),
            ("visualisation.graphs.logseq.physics.springK", 0.002),
            (
                "visualisation.graphs.logseq.physics.autoPause.equilibriumVelocityThreshold",
                0.01,
            ),
            (
                "visualisation.graphs.logseq.physics.autoBalance.stabilityFrameCount",
                30u32,
            ),
        ];

        for (path, value) in test_cases {
            let update_data = json!({
                "path": path,
                "value": value
            });

            let req = test::TestRequest::put()
                .uri("/api/settings/path")
                .set_json(&update_data)
                .to_request();

            let resp = test::call_service(&app, req).await;
            assert!(
                resp.status().is_success(),
                "Field normalization should work for path: {}",
                path
            );

            let body: Value = test::read_body_json(resp).await;
            assert_eq!(
                body["success"], true,
                "Update should succeed for path: {}",
                path
            );
        }
    }

    #[actix_web::test]
    async fn test_validation_errors() {
        let app = create_test_app().await;

        // Test validation error for invalid hex color
        let update_data = json!({
            "path": "visualisation.graphs.logseq.nodes.baseColor",
            "value": "invalid-color"
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status(),
            400,
            "Invalid color should trigger validation error"
        );

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["error"], "Validation failed");
        assert!(body["validationErrors"].is_object());
    }

    #[actix_web::test]
    async fn test_validation_range_errors() {
        let app = create_test_app().await;

        // Test validation error for out of range opacity
        let update_data = json!({
            "path": "visualisation.graphs.logseq.nodes.opacity",
            "value": 1.5  // Invalid: should be 0.0-1.0
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert_eq!(
            resp.status(),
            400,
            "Out of range value should trigger validation error"
        );
    }

    #[actix_web::test]
    async fn test_all_problematic_fields() {
        let app = create_test_app().await;

        // Test all the fields mentioned in the task that were causing duplicate field errors
        let problematic_fields = vec![
            ("visualisation.graphs.logseq.nodes.baseColor", "#FF5733"),
            ("visualisation.graphs.visionflow.nodes.baseColor", "#33FF57"),
            ("visualisation.rendering.ambientLightIntensity", 0.8),
            ("visualisation.rendering.backgroundColor", "#000000"),
            ("visualisation.rendering.directionalLightIntensity", 1.0),
            ("visualisation.rendering.environmentIntensity", 0.5),
        ];

        for (path, value) in problematic_fields {
            let update_data = json!({
                "path": path,
                "value": value
            });

            let req = test::TestRequest::put()
                .uri("/api/settings/path")
                .set_json(&update_data)
                .to_request();

            let resp = test::call_service(&app, req).await;
            assert!(
                resp.status().is_success(),
                "Problematic field should now work: {}",
                path
            );

            let body: Value = test::read_body_json(resp).await;
            assert_eq!(
                body["success"], true,
                "Update should succeed for problematic field: {}",
                path
            );
        }
    }

    #[actix_web::test]
    async fn test_concurrent_field_updates() {
        let app = create_test_app().await;

        // Test concurrent updates using batch operation with potential conflicts
        let batch_data = json!({
            "updates": [
                {
                    "path": "visualisation.graphs.logseq.nodes.baseColor",
                    "value": "#FF0000"
                },
                {
                    "path": "visualisation.graphs.logseq.nodes.opacity",
                    "value": 0.8
                },
                {
                    "path": "visualisation.rendering.ambientLightIntensity",
                    "value": 0.9
                },
                {
                    "path": "visualisation.rendering.directionalLightIntensity",
                    "value": 1.2
                }
            ]
        });

        let req = test::TestRequest::put()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();

        let resp = test::call_service(&app, req).await;
        assert!(
            resp.status().is_success(),
            "Concurrent field updates should succeed"
        );

        let body: Value = test::read_body_json(resp).await;
        assert_eq!(body["success"], true);

        // Verify all updates succeeded
        let results = body["results"].as_array().unwrap();
        for result in results {
            assert_eq!(
                result["success"], true,
                "Each concurrent update should succeed"
            );
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use actix_web::{test, web, App};
    use serde_json::json;

    /// Integration test that verifies the complete settings API workflow
    #[actix_web::test]
    async fn test_complete_settings_workflow() {
        let app_state = super::settings_api_tests::create_test_app_state().await;
        let app = test::init_service(
            App::new().app_data(app_state).service(
                web::scope("/api/settings")
                    .route("/path", web::get().to(super::get_settings_by_path))
                    .route("/path", web::put().to(super::update_settings_by_path))
                    .route("/batch", web::post().to(super::batch_read_settings_by_path))
                    .route(
                        "/batch",
                        web::put().to(super::batch_update_settings_by_path),
                    )
                    .route("/schema", web::get().to(super::get_settings_schema)),
            ),
        )
        .await;

        // Step 1: Get initial settings
        let req = test::TestRequest::get()
            .uri("/api/settings/path?path=visualisation.graphs.logseq.nodes.baseColor")
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Step 2: Update the baseColor field (this was the problematic one)
        let update_data = json!({
            "path": "visualisation.graphs.logseq.nodes.baseColor",
            "value": "#NEW123"
        });
        let req = test::TestRequest::put()
            .uri("/api/settings/path")
            .set_json(&update_data)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Step 3: Verify the update took effect
        let req = test::TestRequest::get()
            .uri("/api/settings/path?path=visualisation.graphs.logseq.nodes.baseColor")
            .to_request();
        let resp = test::call_service(&app, req).await;
        let body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(body["value"], "#NEW123");

        // Step 4: Test batch operations with the fixed field
        let batch_data = json!({
            "updates": [
                {
                    "path": "visualisation.graphs.logseq.nodes.baseColor",
                    "value": "#BATCH1"
                },
                {
                    "path": "visualisation.rendering.ambientLightIntensity",
                    "value": 0.75
                }
            ]
        });
        let req = test::TestRequest::put()
            .uri("/api/settings/batch")
            .set_json(&batch_data)
            .to_request();
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());

        // Step 5: Verify batch updates took effect
        let batch_read = json!({
            "paths": [
                "visualisation.graphs.logseq.nodes.baseColor",
                "visualisation.rendering.ambientLightIntensity"
            ]
        });
        let req = test::TestRequest::post()
            .uri("/api/settings/batch")
            .set_json(&batch_read)
            .to_request();
        let resp = test::call_service(&app, req).await;
        let body: serde_json::Value = test::read_body_json(resp).await;
        assert_eq!(
            body["visualisation.graphs.logseq.nodes.baseColor"],
            "#BATCH1"
        );
        assert_eq!(body["visualisation.rendering.ambientLightIntensity"], 0.75);
    }
}
