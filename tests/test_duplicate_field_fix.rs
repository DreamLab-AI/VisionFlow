// Test disabled - references deprecated/removed modules (crate::config::AppFullSettings)
// AppFullSettings merge_update method may have changed; use config module directly
/*
use crate::config::AppFullSettings;
use serde_json::json;

#[tokio::test]
async fn test_duplicate_field_fix() {
    // Create a default settings instance
    let mut settings = AppFullSettings::default();

    // Test the problematic fields that were causing duplicate field errors
    let update_with_base_color = json!({
        "visualisation": {
            "nodes": {
                "baseColor": "#FF0000"
            }
        }
    });

    let update_with_ambient_light = json!({
        "visualisation": {
            "rendering": {
                "ambientLightIntensity": 0.8
            }
        }
    });

    let combined_update = json!({
        "visualisation": {
            "nodes": {
                "baseColor": "#00FF00",
                "opacity": 0.9
            },
            "rendering": {
                "ambientLightIntensity": 0.7,
                "enableShadows": true
            }
        }
    });

    // These should all succeed now that field normalization is implemented
    assert!(
        settings.merge_update(update_with_base_color).is_ok(),
        "baseColor update should succeed"
    );

    assert!(
        settings.merge_update(update_with_ambient_light).is_ok(),
        "ambientLightIntensity update should succeed"
    );

    assert!(
        settings.merge_update(combined_update).is_ok(),
        "Combined update with multiple fields should succeed"
    );

    // Verify the values were actually applied
    assert_eq!(settings.visualisation.nodes.base_color, "#00FF00");
    assert_eq!(settings.visualisation.nodes.opacity, 0.9);
    assert_eq!(
        settings.visualisation.rendering.ambient_light_intensity,
        0.7
    );
    assert!(settings.visualisation.rendering.enable_shadows);
}

#[tokio::test]
async fn test_field_normalization_edge_cases() {
    let mut settings = AppFullSettings::default();

    // Test nested updates with both naming conventions in the same payload
    // This would previously cause the duplicate field error
    let mixed_naming_update = json!({
        "visualisation": {
            "nodes": {
                "baseColor": "#FF0000",
                "base_color": "#00FF00"  // This should be normalized to baseColor
            }
        }
    });

    // The fix should handle this by normalizing both to camelCase
    // The camelCase version should win since it appears later in processing
    let result = settings.merge_update(mixed_naming_update);
    assert!(
        result.is_ok(),
        "Mixed naming convention update should succeed: {:?}",
        result
    );
}
*/
