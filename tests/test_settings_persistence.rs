use actix::Actor;
use actix_web::{http::StatusCode, test, web, App};
use serde_json::json;
use std::sync::Arc;
use tempfile::NamedTempFile;
use visionflow::actors::{AppState, SettingsActor};
use visionflow::config::AppFullSettings;
use visionflow::handlers::settings_handler;

#[actix_web::test]
async fn test_save_settings_endpoint() {
    // Create a temporary settings file
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap().to_string();

    // Set the environment variable to use our temp file
    std::env::set_var("SETTINGS_FILE_PATH", &temp_path);

    // Create default settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    // Create settings actor
    let settings_addr =
        SettingsActor::new(Arc::new(tokio::sync::RwLock::new(settings)), None, None).start();

    // Create app state
    let app_state = web::Data::new(AppState {
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

    // Create test app
    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .configure(settings_handler::config),
    )
    .await;

    // Test save endpoint without payload
    let req = test::TestRequest::post().uri("/settings/save").to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    let body: serde_json::Value = test::read_body_json(resp).await;
    assert_eq!(body["message"], "Settings saved successfully");
    assert!(body["settings"].is_object());

    // Verify file was written
    let saved_content = std::fs::read_to_string(&temp_path).unwrap();
    assert!(saved_content.contains("visualisation:"));
    assert!(saved_content.contains("system:"));
}

#[actix_web::test]
async fn test_save_settings_with_update() {
    // Create a temporary settings file
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap().to_string();

    // Set the environment variable to use our temp file
    std::env::set_var("SETTINGS_FILE_PATH", &temp_path);

    // Create default settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    // Create settings actor
    let settings_addr =
        SettingsActor::new(Arc::new(tokio::sync::RwLock::new(settings)), None, None).start();

    // Create app state
    let app_state = web::Data::new(AppState {
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

    // Create test app
    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .configure(settings_handler::config),
    )
    .await;

    // Test save endpoint with payload
    let update_payload = json!({
        "visualisation": {
            "glow": {
                "intensity": 2.5,
                "radius": 0.75
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

    // Verify the update was applied in the response
    assert_eq!(body["settings"]["visualisation"]["glow"]["intensity"], 2.5);
    assert_eq!(body["settings"]["visualisation"]["glow"]["radius"], 0.75);

    // Verify file was updated
    let saved_content = std::fs::read_to_string(&temp_path).unwrap();
    assert!(saved_content.contains("intensity: 2.5"));
    assert!(saved_content.contains("radius: 0.75"));
}

#[actix_web::test]
async fn test_save_settings_persistence_disabled() {
    // Create default settings with persistence disabled
    let settings = AppFullSettings::default(); // persist_settings is false by default

    // Create settings actor
    let settings_addr =
        SettingsActor::new(Arc::new(tokio::sync::RwLock::new(settings)), None, None).start();

    // Create app state
    let app_state = web::Data::new(AppState {
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

    // Create test app
    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .configure(settings_handler::config),
    )
    .await;

    // Test save endpoint
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
async fn test_save_settings_validation_error() {
    // Create default settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    // Create settings actor
    let settings_addr =
        SettingsActor::new(Arc::new(tokio::sync::RwLock::new(settings)), None, None).start();

    // Create app state
    let app_state = web::Data::new(AppState {
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

    // Create test app
    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .configure(settings_handler::config),
    )
    .await;

    // Test save endpoint with invalid payload
    let invalid_payload = json!({
        "visualisation": {
            "glow": {
                "intensity": -5.0, // Invalid: negative intensity
                "radius": 100.0    // Invalid: too large
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

#[actix_web::test]
async fn test_update_and_save_workflow() {
    // Create a temporary settings file
    let temp_file = NamedTempFile::new().unwrap();
    let temp_path = temp_file.path().to_str().unwrap().to_string();

    // Set the environment variable to use our temp file
    std::env::set_var("SETTINGS_FILE_PATH", &temp_path);

    // Create default settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;

    // Create settings actor
    let settings_addr =
        SettingsActor::new(Arc::new(tokio::sync::RwLock::new(settings)), None, None).start();

    // Create app state
    let app_state = web::Data::new(AppState {
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

    // Create test app
    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .configure(settings_handler::config),
    )
    .await;

    // First, update settings using the regular update endpoint
    let update_payload = json!({
        "visualisation": {
            "rendering": {
                "ambientLightIntensity": 1.8
            }
        }
    });

    let req = test::TestRequest::post()
        .uri("/settings")
        .set_json(&update_payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify the file was automatically saved (since persist_settings is true)
    let saved_content = std::fs::read_to_string(&temp_path).unwrap();
    assert!(saved_content.contains("ambient_light_intensity: 1.8"));

    // Now explicitly save with additional changes
    let save_payload = json!({
        "visualisation": {
            "rendering": {
                "directionalLightIntensity": 2.0
            }
        }
    });

    let req = test::TestRequest::post()
        .uri("/settings/save")
        .set_json(&save_payload)
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify both changes are in the file
    let final_content = std::fs::read_to_string(&temp_path).unwrap();
    assert!(final_content.contains("ambient_light_intensity: 1.8"));
    assert!(final_content.contains("directional_light_intensity: 2"));
}
