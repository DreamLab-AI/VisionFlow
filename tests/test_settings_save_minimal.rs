// Minimal test to verify settings save functionality without full app setup
use std::fs;
use std::path::Path;
use webxr::config::AppFullSettings;

#[test]
fn test_settings_save_to_file() {
    // Create a temporary directory for the test
    let test_dir = format!("/tmp/visionflow_test_{}", std::process::id());
    fs::create_dir_all(&test_dir).unwrap();
    let settings_path = format!("{}/test_settings.yaml", test_dir);
    
    // Set the environment variable
    std::env::set_var("SETTINGS_FILE_PATH", &settings_path);
    
    // Create settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;
    
    // Modify some values to test
    settings.visualisation.glow.intensity = 2.5;
    settings.visualisation.glow.radius = 0.75;
    settings.visualisation.rendering.ambient_light_intensity = 3.0;
    
    // Save the settings
    let result = settings.save();
    assert!(result.is_ok(), "Failed to save settings: {:?}", result.err());
    
    // Verify the file was created
    assert!(Path::new(&settings_path).exists(), "Settings file was not created");
    
    // Read the file and verify content
    let saved_content = fs::read_to_string(&settings_path).unwrap();
    assert!(saved_content.contains("intensity: 2.5"), "Intensity not found in saved file");
    assert!(saved_content.contains("radius: 0.75"), "Radius not found in saved file");
    assert!(saved_content.contains("ambient_light_intensity: 3"), "Ambient light intensity not found in saved file");
    
    // Clean up
    fs::remove_dir_all(&test_dir).ok();
    std::env::remove_var("SETTINGS_FILE_PATH");
}

#[test]
fn test_settings_save_disabled() {
    // Create a temporary directory for the test
    let test_dir = format!("/tmp/visionflow_test_{}", std::process::id());
    fs::create_dir_all(&test_dir).unwrap();
    let settings_path = format!("{}/test_settings_disabled.yaml", test_dir);
    
    // Set the environment variable
    std::env::set_var("SETTINGS_FILE_PATH", &settings_path);
    
    // Create settings with persistence disabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = false; // Disabled
    
    // Try to save
    let result = settings.save();
    assert!(result.is_ok(), "Save should return Ok even when disabled");
    
    // Verify the file was NOT created (since persist_settings is false)
    assert!(!Path::new(&settings_path).exists(), "Settings file should not be created when persist_settings is false");
    
    // Clean up
    fs::remove_dir_all(&test_dir).ok();
    std::env::remove_var("SETTINGS_FILE_PATH");
}

#[test]
fn test_settings_merge_and_save() {
    use serde_json::json;
    
    // Create a temporary directory for the test
    let test_dir = format!("/tmp/visionflow_test_{}", std::process::id());
    fs::create_dir_all(&test_dir).unwrap();
    let settings_path = format!("{}/test_settings_merge.yaml", test_dir);
    
    // Set the environment variable
    std::env::set_var("SETTINGS_FILE_PATH", &settings_path);
    
    // Create settings with persistence enabled
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;
    
    // Create an update
    let update = json!({
        "visualisation": {
            "glow": {
                "intensity": 3.5,
                "threshold": 0.9
            },
            "rendering": {
                "directionalLightIntensity": 2.0
            }
        }
    });
    
    // Merge the update
    let merge_result = settings.merge_update(update);
    assert!(merge_result.is_ok(), "Failed to merge update: {:?}", merge_result.err());
    
    // Save the settings
    let save_result = settings.save();
    assert!(save_result.is_ok(), "Failed to save merged settings: {:?}", save_result.err());
    
    // Verify the file contains merged values
    let saved_content = fs::read_to_string(&settings_path).unwrap();
    assert!(saved_content.contains("intensity: 3.5"), "Merged intensity not found");
    assert!(saved_content.contains("threshold: 0.9"), "Merged threshold not found");
    assert!(saved_content.contains("directional_light_intensity: 2"), "Merged directional light not found");
    
    // Clean up
    fs::remove_dir_all(&test_dir).ok();
    std::env::remove_var("SETTINGS_FILE_PATH");
}

#[test]
fn test_settings_io_error_handling() {
    // Try to save to a non-existent directory
    let invalid_path = "/non/existent/directory/settings.yaml";
    std::env::set_var("SETTINGS_FILE_PATH", invalid_path);
    
    let mut settings = AppFullSettings::default();
    settings.system.persist_settings = true;
    
    let result = settings.save();
    assert!(result.is_err(), "Save should fail with invalid path");
    let error_msg = result.unwrap_err();
    assert!(error_msg.contains("No such file or directory") || 
            error_msg.contains("cannot find the path"), 
            "Error message should indicate path issue: {}", error_msg);
    
    std::env::remove_var("SETTINGS_FILE_PATH");
}

#[test]
fn test_settings_validation_boundaries() {
    use serde_json::json;
    
    let mut settings = AppFullSettings::default();
    
    // Test edge cases for glow settings
    let valid_edge_cases = json!({
        "visualisation": {
            "glow": {
                "intensity": 10.0,  // Max allowed
                "radius": 10.0,     // Max allowed
                "threshold": 1.0,   // Max allowed
                "opacity": 1.0      // Max allowed
            }
        }
    });
    
    let result = settings.merge_update(valid_edge_cases);
    assert!(result.is_ok(), "Valid edge cases should be accepted");
    
    // Verify the values were set correctly
    assert_eq!(settings.visualisation.glow.intensity, 10.0);
    assert_eq!(settings.visualisation.glow.radius, 10.0);
    assert_eq!(settings.visualisation.glow.threshold, 1.0);
    assert_eq!(settings.visualisation.glow.opacity, 1.0);
}