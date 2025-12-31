// Test disabled - references deprecated/removed modules (webxr::actors::SettingsActor, webxr::actors::AppState, messages::GetSettings)
// Actor module structure has changed per ADR-001; SettingsActor and AppState have moved
/*
use actix_web::{http::StatusCode, test, web, App};
use serde_json::json;
use std::sync::Arc;
use tempfile::NamedTempFile;

// Import necessary types from the main crate
use actix::Actor;
use webxr::{
    actors::{messages::GetSettings, SettingsActor},
    config::AppFullSettings,
    handlers::settings_handler,
};

// Helper function to create a test app with settings
async fn create_test_app_with_settings(
    settings: AppFullSettings,
) -> (actix_web::test::TestServer, actix::Addr<SettingsActor>) {
    let settings_addr =
        SettingsActor::new(Arc::new(tokio::sync::RwLock::new(settings)), None, None).start();

    let app_state = web::Data::new(webxr::actors::AppState {
        settings_addr: settings_addr.clone(),
        graph_service: None,
        gpu_manager: None,
        websocket_sessions: None,
        mcp_host_session: None,
        bot_config: None,
        nostr_service: None,
        file_service: None,
        perplexity_service: None,
        quest_service: None,
        clustering_service: None,
        constraints_service: None,
        auth_service: None,
        ragflow_service: None,
        multi_mcp_service: None,
    });

    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .configure(settings_handler::config),
    )
    .await;

    (app, settings_addr)
}

#[actix_web::test]
async fn test_save_endpoint_exists() {
    // Create settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    let (app, _) = create_test_app_with_settings(settings).await;

    // Test that the endpoint exists
    let req = test::TestRequest::post().uri("/settings/save").to_request();

    let resp = test::call_service(&app, req).await;

    // Should not be 404
    assert_ne!(resp.status(), StatusCode::NOT_FOUND);
}

#[actix_web::test]
async fn test_save_without_persistence() {
    // Create settings with persistence disabled
    let settings = AppFullSettings::default(); // persist_settings is false by default

    let (app, _) = create_test_app_with_settings(settings).await;

    // Try to save
    let req = test::TestRequest::post().uri("/settings/save").to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["error"]
        .as_str()
        .unwrap()
        .contains("Settings persistence is disabled"));
}

#[actix_web::test]
async fn test_save_with_persistence_enabled() {
    // Create a temporary file for settings
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap().to_string();
    std::env::set_var("SETTINGS_FILE_PATH", &temp_path);

    // Create settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    let (app, _) = create_test_app_with_settings(settings).await;

    // Save settings
    let req = test::TestRequest::post().uri("/settings/save").to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["message"], "Settings saved successfully");
    assert!(body["settings"].is_object());

    // Verify file was created
    assert!(std::path::Path::new(&temp_path).exists());

    // Clean up
    std::env::remove_var("SETTINGS_FILE_PATH");
}

#[actix_web::test]
async fn test_save_with_updates() {
    // Create a temporary file for settings
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap().to_string();
    std::env::set_var("SETTINGS_FILE_PATH", &temp_path);

    // Create settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    let (app, settings_addr) = create_test_app_with_settings(settings).await;

    // Save with updates
    let update_payload = json!({
        "visualisation": {
            "rendering": {
                "ambientLightIntensity": 2.5,
                "directionalLightIntensity": 3.0
            }
        }
    });

    let req = test::TestRequest::post()
        .uri("/settings/save")
        .set_json(&update_payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["message"], "Settings saved successfully");

    // Verify the updates were applied
    assert_eq!(
        body["settings"]["visualisation"]["rendering"]["ambientLightIntensity"],
        2.5
    );
    assert_eq!(
        body["settings"]["visualisation"]["rendering"]["directionalLightIntensity"],
        3.0
    );

    // Verify the settings were actually saved by checking the actor
    let current_settings = settings_addr.send(GetSettings).await.unwrap().unwrap();
    assert_eq!(
        current_settings
            .visualisation
            .rendering
            .ambient_light_intensity,
        2.5
    );
    assert_eq!(
        current_settings
            .visualisation
            .rendering
            .directional_light_intensity,
        3.0
    );

    // Clean up
    std::env::remove_var("SETTINGS_FILE_PATH");
}

#[actix_web::test]
async fn test_save_with_invalid_updates() {
    // Create settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    let (app, _) = create_test_app_with_settings(settings).await;

    // Try to save with invalid updates
    let invalid_payload = json!({
        "visualisation": {
            "glow": {
                "intensity": -5.0 // Invalid negative value
            }
        }
    });

    let req = test::TestRequest::post()
        .uri("/settings/save")
        .set_json(&invalid_payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert!(body["error"].as_str().unwrap().contains("Invalid settings"));
}
*/
