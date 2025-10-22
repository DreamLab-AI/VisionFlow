// Settings Sync Integration Test
// Tests that settings synchronization works properly between client and server

use serde_json::{json, Value};
use webxr::config::path_access::JsonPathAccessible;
use webxr::config::AppFullSettings;

#[tokio::test]
async fn test_settings_json_serialization() {
    // Create default settings without loading from file
    let settings = AppFullSettings::default();

    // Serialize to JSON (this should use camelCase)
    let json_value = serde_json::to_value(&settings).expect("Failed to serialize to JSON");

    // Check that specific fields are in camelCase format
    let rendering = json_value
        .get("visualisation")
        .and_then(|v| v.get("rendering"))
        .expect("Missing rendering settings");

    // These should be camelCase in JSON
    assert!(
        rendering.get("ambientLightIntensity").is_some(),
        "ambientLightIntensity should be in camelCase"
    );
    assert!(
        rendering.get("directionalLightIntensity").is_some(),
        "directionalLightIntensity should be in camelCase"
    );
    assert!(
        rendering.get("enableAmbientOcclusion").is_some(),
        "enableAmbientOcclusion should be in camelCase"
    );
    assert!(
        rendering.get("enableAntialiasing").is_some(),
        "enableAntialiasing should be in camelCase"
    );

    // These should NOT exist in snake_case
    assert!(
        rendering.get("ambient_light_intensity").is_none(),
        "ambient_light_intensity should not exist (should be camelCase)"
    );
    assert!(
        rendering.get("directional_light_intensity").is_none(),
        "directional_light_intensity should not exist (should be camelCase)"
    );

    println!("✅ Settings JSON serialization uses camelCase correctly");
}

#[tokio::test]
async fn test_json_path_access() {
    let settings = AppFullSettings::default();

    // Test getting values with camelCase paths (as they appear in JSON)
    let ambient_light = settings
        .get_json_by_path("visualisation.rendering.ambientLightIntensity")
        .expect("Failed to get ambientLightIntensity");

    let enable_ao = settings
        .get_json_by_path("visualisation.rendering.enableAmbientOcclusion")
        .expect("Failed to get enableAmbientOcclusion");

    // Verify values are correct types
    assert!(
        ambient_light.is_f64(),
        "ambientLightIntensity should be a number"
    );
    assert!(
        enable_ao.is_boolean(),
        "enableAmbientOcclusion should be a boolean"
    );

    println!("✅ JSON path access works with camelCase paths");
}

#[tokio::test]
async fn test_json_path_set_and_get() {
    let mut settings = AppFullSettings::default();

    // Set a value using camelCase path
    let new_intensity = json!(0.8);
    settings
        .set_json_by_path(
            "visualisation.rendering.ambientLightIntensity",
            new_intensity.clone(),
        )
        .expect("Failed to set ambientLightIntensity");

    // Get the value back
    let retrieved_value = settings
        .get_json_by_path("visualisation.rendering.ambientLightIntensity")
        .expect("Failed to get ambientLightIntensity");

    assert_eq!(
        retrieved_value, new_intensity,
        "Set and retrieved values should match"
    );

    // Verify it's also reflected in the actual struct field
    assert_eq!(
        settings.visualisation.rendering.ambient_light_intensity, 0.8,
        "Struct field should be updated"
    );

    println!("✅ JSON path set and get work correctly with camelCase");
}

#[tokio::test]
async fn test_physics_parameter_sync() {
    let mut settings = AppFullSettings::default();

    // Set physics parameters using camelCase paths (as client would send)
    let physics_updates = vec![
        ("visualisation.graphs.logseq.physics.springK", json!(1.5)),
        ("visualisation.graphs.logseq.physics.repelK", json!(2.0)),
        ("visualisation.graphs.logseq.physics.damping", json!(0.95)),
    ];

    for (path, value) in physics_updates {
        settings
            .set_json_by_path(path, value.clone())
            .expect(&format!("Failed to set {}", path));

        // Verify we can get it back
        let retrieved = settings
            .get_json_by_path(path)
            .expect(&format!("Failed to get {}", path));
        assert_eq!(retrieved, value, "Physics parameter {} should match", path);
    }

    // Verify the actual struct fields are updated
    let physics = &settings.visualisation.graphs.logseq.physics;
    assert_eq!(physics.spring_k, 1.5, "springK should be updated in struct");
    assert_eq!(physics.repel_k, 2.0, "repelK should be updated in struct");
    assert_eq!(physics.damping, 0.95, "damping should be updated in struct");

    println!("✅ Physics parameter sync works correctly");
}

#[tokio::test]
async fn test_batch_settings_update() {
    let mut settings = AppFullSettings::default();

    // Simulate batch update from client (all camelCase)
    let batch_updates = vec![
        ("visualisation.rendering.ambientLightIntensity", json!(0.7)),
        (
            "visualisation.rendering.directionalLightIntensity",
            json!(1.2),
        ),
        (
            "visualisation.rendering.enableAmbientOcclusion",
            json!(false),
        ),
        ("system.websocket.updateRate", json!(30)),
    ];

    // Apply all updates
    for (path, value) in &batch_updates {
        settings
            .set_json_by_path(path, value.clone())
            .expect(&format!("Failed to set {}", path));
    }

    // Verify all values are set correctly
    for (path, expected_value) in &batch_updates {
        let actual_value = settings
            .get_json_by_path(path)
            .expect(&format!("Failed to get {}", path));
        assert_eq!(
            &actual_value, expected_value,
            "Batch update failed for {}",
            path
        );
    }

    // Verify struct fields are updated
    assert_eq!(
        settings.visualisation.rendering.ambient_light_intensity,
        0.7
    );
    assert_eq!(
        settings.visualisation.rendering.directional_light_intensity,
        1.2
    );
    assert_eq!(
        settings.visualisation.rendering.enable_ambient_occlusion,
        false
    );
    assert_eq!(settings.system.websocket.update_rate, 30);

    println!("✅ Batch settings update works correctly");
}

#[test]
fn test_client_server_serialization_compatibility() {
    // This test ensures that the client (camelCase JSON) is compatible with server (snake_case structs)

    // Start with default settings and then update specific fields
    let mut settings = AppFullSettings::default();
    settings.visualisation.rendering.ambient_light_intensity = 0.6;
    settings.visualisation.rendering.directional_light_intensity = 1.1;
    settings.visualisation.rendering.enable_ambient_occlusion = true;
    settings.visualisation.rendering.enable_antialiasing = false;
    settings.visualisation.glow.enabled = true;
    settings.visualisation.glow.intensity = 0.8;
    settings.visualisation.glow.radius = 2.0;
    settings.visualisation.glow.threshold = 0.5;

    // Test serializing to JSON (should be camelCase)
    let server_json = serde_json::to_value(&settings).expect("Should serialize to JSON");

    // Check that it's camelCase in the output
    let rendering = server_json
        .get("visualisation")
        .and_then(|v| v.get("rendering"))
        .expect("Should have rendering");

    assert!(
        rendering.get("ambientLightIntensity").is_some(),
        "Should have camelCase ambientLightIntensity"
    );
    assert!(
        rendering.get("enableAmbientOcclusion").is_some(),
        "Should have camelCase enableAmbientOcclusion"
    );
    assert!(
        rendering.get("ambient_light_intensity").is_none(),
        "Should NOT have snake_case ambient_light_intensity"
    );
    assert!(
        rendering.get("enable_ambient_occlusion").is_none(),
        "Should NOT have snake_case enable_ambient_occlusion"
    );

    // Verify the serialized values match what we set (with some tolerance for floats)
    assert!(
        (rendering
            .get("ambientLightIntensity")
            .unwrap()
            .as_f64()
            .unwrap()
            - 0.6)
            .abs()
            < 0.01
    );
    assert!(
        (rendering
            .get("directionalLightIntensity")
            .unwrap()
            .as_f64()
            .unwrap()
            - 1.1)
            .abs()
            < 0.01
    );
    assert_eq!(
        rendering
            .get("enableAmbientOcclusion")
            .unwrap()
            .as_bool()
            .unwrap(),
        true
    );
    assert_eq!(
        rendering
            .get("enableAntialiasing")
            .unwrap()
            .as_bool()
            .unwrap(),
        false
    );

    let glow = server_json
        .get("visualisation")
        .and_then(|v| v.get("glow"))
        .expect("Should have glow");

    assert_eq!(glow.get("enabled").unwrap().as_bool().unwrap(), true);
    assert!((glow.get("intensity").unwrap().as_f64().unwrap() - 0.8).abs() < 0.01);
    assert!((glow.get("radius").unwrap().as_f64().unwrap() - 2.0).abs() < 0.01);
    assert!((glow.get("threshold").unwrap().as_f64().unwrap() - 0.5).abs() < 0.01);

    println!("✅ Client-server serialization compatibility confirmed");
}
