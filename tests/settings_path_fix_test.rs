#[cfg(test)]
mod settings_path_fix_tests {
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};

    // Import the path access trait (assuming it's in the webxr crate)
    use webxr::config::path_access::JsonPathAccessible;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct TestSettings {
        visualisation: VisualisationSettings,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct VisualisationSettings {
        enable_hologram: bool,
        hologram_settings: HologramSettings,
        nodes: NodeSettings,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct HologramSettings {
        ring_count: u32,
        ring_color: String,
        ring_opacity: f32,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct NodeSettings {
        base_color: String,
        metalness: f32,
        opacity: f32,
    }

    #[test]
    fn test_batch_update_with_camel_case_paths() {
        let mut settings = TestSettings {
            visualisation: VisualisationSettings {
                enable_hologram: false,
                hologram_settings: HologramSettings {
                    ring_count: 3,
                    ring_color: "#FF0000".to_string(),
                    ring_opacity: 0.8,
                },
                nodes: NodeSettings {
                    base_color: "#FFFFFF".to_string(),
                    metalness: 0.5,
                    opacity: 1.0,
                },
            },
        };

        // Simulate batch updates from client with camelCase paths
        let updates = vec![
            ("visualisation.enableHologram", json!(true)),
            ("visualisation.hologramSettings.ringCount", json!(5)),
            ("visualisation.hologramSettings.ringColor", json!("#00FF00")),
            ("visualisation.hologramSettings.ringOpacity", json!(0.6)),
            ("visualisation.nodes.baseColor", json!("#0000FF")),
            ("visualisation.nodes.metalness", json!(0.8)),
        ];

        // Apply all updates
        for (path, value) in &updates {
            let result = settings.set_json_by_path(path, value.clone());
            assert!(
                result.is_ok(),
                "Failed to update path '{}': {:?}",
                path,
                result
            );
        }

        // Verify all updates were applied correctly
        assert_eq!(settings.visualisation.enable_hologram, true);
        assert_eq!(settings.visualisation.hologram_settings.ring_count, 5);
        assert_eq!(
            settings.visualisation.hologram_settings.ring_color,
            "#00FF00"
        );
        assert_eq!(settings.visualisation.hologram_settings.ring_opacity, 0.6);
        assert_eq!(settings.visualisation.nodes.base_color, "#0000FF");
        assert_eq!(settings.visualisation.nodes.metalness, 0.8);

        // Verify serialization round-trip works
        let json = serde_json::to_value(&settings).unwrap();
        let deserialized: TestSettings = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(settings, deserialized, "Round-trip serialization failed");

        // Verify the JSON has correct camelCase field names
        let json_str = serde_json::to_string_pretty(&json).unwrap();
        assert!(json_str.contains("\"enableHologram\": true"));
        assert!(json_str.contains("\"ringCount\": 5"));
        assert!(json_str.contains("\"baseColor\": \"#0000FF\""));
    }

    #[test]
    fn test_invalid_path_rejection() {
        let mut settings = TestSettings {
            visualisation: VisualisationSettings {
                enable_hologram: false,
                hologram_settings: HologramSettings {
                    ring_count: 3,
                    ring_color: "#FF0000".to_string(),
                    ring_opacity: 0.8,
                },
                nodes: NodeSettings {
                    base_color: "#FFFFFF".to_string(),
                    metalness: 0.5,
                    opacity: 1.0,
                },
            },
        };

        // Test invalid path
        let result = settings.set_json_by_path("visualisation.nonExistentField", json!(true));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));

        // Test path to non-existent nested field
        let result =
            settings.set_json_by_path("visualisation.hologramSettings.nonExistent", json!(42));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not exist"));
    }

    #[test]
    fn test_type_mismatch_rejection() {
        let mut settings = TestSettings {
            visualisation: VisualisationSettings {
                enable_hologram: false,
                hologram_settings: HologramSettings {
                    ring_count: 3,
                    ring_color: "#FF0000".to_string(),
                    ring_opacity: 0.8,
                },
                nodes: NodeSettings {
                    base_color: "#FFFFFF".to_string(),
                    metalness: 0.5,
                    opacity: 1.0,
                },
            },
        };

        // Try to set boolean where number is expected
        let result =
            settings.set_json_by_path("visualisation.hologramSettings.ringCount", json!(true));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Type mismatch"));

        // Try to set number where string is expected
        let result = settings.set_json_by_path("visualisation.nodes.baseColor", json!(123));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Type mismatch"));

        // Try to set string where boolean is expected
        let result = settings.set_json_by_path("visualisation.enableHologram", json!("yes"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Type mismatch"));
    }

    #[test]
    fn test_case_insensitive_path_navigation() {
        let mut settings = TestSettings {
            visualisation: VisualisationSettings {
                enable_hologram: false,
                hologram_settings: HologramSettings {
                    ring_count: 3,
                    ring_color: "#FF0000".to_string(),
                    ring_opacity: 0.8,
                },
                nodes: NodeSettings {
                    base_color: "#FFFFFF".to_string(),
                    metalness: 0.5,
                    opacity: 1.0,
                },
            },
        };

        // Test that we can use either camelCase or snake_case
        let result1 = settings.set_json_by_path("visualisation.enableHologram", json!(true));
        assert!(result1.is_ok());

        // Reset
        settings.visualisation.enable_hologram = false;

        // Try with snake_case (should work due to case conversion)
        let result2 = settings.set_json_by_path("visualisation.enable_hologram", json!(true));
        assert!(result2.is_ok());
        assert_eq!(settings.visualisation.enable_hologram, true);
    }

    #[test]
    fn test_get_json_by_path() {
        let settings = TestSettings {
            visualisation: VisualisationSettings {
                enable_hologram: true,
                hologram_settings: HologramSettings {
                    ring_count: 5,
                    ring_color: "#00FF00".to_string(),
                    ring_opacity: 0.75,
                },
                nodes: NodeSettings {
                    base_color: "#0000FF".to_string(),
                    metalness: 0.9,
                    opacity: 0.95,
                },
            },
        };

        // Test getting values with camelCase paths
        let value = settings
            .get_json_by_path("visualisation.enableHologram")
            .unwrap();
        assert_eq!(value, json!(true));

        let value = settings
            .get_json_by_path("visualisation.hologramSettings.ringCount")
            .unwrap();
        assert_eq!(value, json!(5));

        let value = settings
            .get_json_by_path("visualisation.nodes.metalness")
            .unwrap();
        assert_eq!(value, json!(0.9));

        // Test invalid path
        let result = settings.get_json_by_path("visualisation.nonExistent");
        assert!(result.is_err());
    }
}
