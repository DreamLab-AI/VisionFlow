#[cfg(test)]
mod batch_update_integration_tests {
    use actix_web::{test, web, App};
    use serde_json::json;
    use webxr::app_state::AppState;
    use webxr::handlers::settings_handler;

    #[actix_rt::test]
    async fn test_batch_update_endpoint_with_camel_case() {
        // This test simulates the exact scenario that was failing:
        // Client sends batch updates with camelCase paths

        let app_state = AppState::new_for_testing().await;
        let mut app = test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .configure(settings_handler::config),
        )
        .await;

        // Prepare batch update request with camelCase paths
        let batch_request = json!({
            "updates": [
                {
                    "path": "visualisation.enableHologram",
                    "value": true
                },
                {
                    "path": "visualisation.hologramSettings.ringCount",
                    "value": 5
                },
                {
                    "path": "visualisation.nodes.baseColor",
                    "value": "#00FF00"
                },
                {
                    "path": "visualisation.animations.enableMotionBlur",
                    "value": true
                }
            ]
        });

        let req = test::TestRequest::put()
            .uri("/settings/batch")
            .set_json(&batch_request)
            .to_request();

        let resp = test::call_service(&mut app, req).await;
        assert!(
            resp.status().is_success(),
            "Batch update failed with status: {}",
            resp.status()
        );

        let body: serde_json::Value = test::read_body_json(resp).await;

        // Verify response structure
        assert_eq!(body["success"], true);
        assert!(body["message"]
            .as_str()
            .unwrap()
            .contains("Successfully processed"));

        // Verify each update result
        let values = body["values"].as_array().unwrap();
        assert_eq!(values.len(), 4);

        for value in values {
            assert_eq!(
                value["success"], true,
                "Update failed for path: {}",
                value["path"]
            );
            assert!(
                value["value"].is_object()
                    || value["value"].is_bool()
                    || value["value"].is_string()
                    || value["value"].is_number()
            );
        }

        // Now verify the settings were actually updated by getting them
        let get_req = test::TestRequest::get()
            .uri("/settings/path?path=visualisation.enableHologram")
            .to_request();

        let get_resp = test::call_service(&mut app, get_req).await;
        assert!(get_resp.status().is_success());

        let get_body: serde_json::Value = test::read_body_json(get_resp).await;
        assert_eq!(get_body["value"], true);
    }

    #[actix_rt::test]
    async fn test_batch_update_with_invalid_paths() {
        let app_state = AppState::new_for_testing().await;
        let mut app = test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .configure(settings_handler::config),
        )
        .await;

        // Mix of valid and invalid paths
        let batch_request = json!({
            "updates": [
                {
                    "path": "visualisation.enableHologram",
                    "value": true
                },
                {
                    "path": "visualisation.nonExistentField",
                    "value": "should fail"
                },
                {
                    "path": "visualisation.nodes.baseColor",
                    "value": "#FF0000"
                }
            ]
        });

        let req = test::TestRequest::put()
            .uri("/settings/batch")
            .set_json(&batch_request)
            .to_request();

        let resp = test::call_service(&mut app, req).await;
        assert!(resp.status().is_success());

        let body: serde_json::Value = test::read_body_json(resp).await;
        let values = body["values"].as_array().unwrap();

        // First update should succeed
        assert_eq!(values[0]["success"], true);
        assert_eq!(values[0]["path"], "visualisation.enableHologram");

        // Second update should fail
        assert_eq!(values[1]["success"], false);
        assert_eq!(values[1]["path"], "visualisation.nonExistentField");
        assert!(values[1]["error"]
            .as_str()
            .unwrap()
            .contains("does not exist"));

        // Third update should succeed
        assert_eq!(values[2]["success"], true);
        assert_eq!(values[2]["path"], "visualisation.nodes.baseColor");
    }

    #[actix_rt::test]
    async fn test_batch_update_with_type_mismatches() {
        let app_state = AppState::new_for_testing().await;
        let mut app = test::init_service(
            App::new()
                .app_data(web::Data::new(app_state))
                .configure(settings_handler::config),
        )
        .await;

        let batch_request = json!({
            "updates": [
                {
                    "path": "visualisation.hologramSettings.ringCount",
                    "value": "not a number" // Should fail - expects number
                },
                {
                    "path": "visualisation.enableHologram",
                    "value": 123 // Should fail - expects boolean
                },
                {
                    "path": "visualisation.nodes.baseColor",
                    "value": true // Should fail - expects string
                }
            ]
        });

        let req = test::TestRequest::put()
            .uri("/settings/batch")
            .set_json(&batch_request)
            .to_request();

        let resp = test::call_service(&mut app, req).await;
        assert!(resp.status().is_success());

        let body: serde_json::Value = test::read_body_json(resp).await;
        let values = body["values"].as_array().unwrap();

        // All updates should fail due to type mismatches
        for value in values {
            assert_eq!(value["success"], false);
            assert!(value["error"].as_str().unwrap().contains("Type mismatch"));
            assert!(value.get("expectedType").is_some());
        }
    }
}
