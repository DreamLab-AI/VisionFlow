use serde_yaml;
use std::fs;

// Copy the exact structs from the config module for testing
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct RenderingSettings {
    pub ambient_light_intensity: f32,
    pub background_color: String,
    pub directional_light_intensity: f32,
    pub enable_ambient_occlusion: bool,
    pub enable_antialiasing: bool,
    pub enable_shadows: bool,
    pub environment_intensity: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_map_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shadow_bias: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

fn main() {
    println!("Testing YAML configuration loading...");
    
    // Test just the rendering section first
    let yaml_content = r#"
rendering:
  ambient_light_intensity: 1.2
  background_color: '#0a0e1a'
  directional_light_intensity: 1.5
  enable_ambient_occlusion: false
  enable_antialiasing: true
  enable_shadows: true
  environment_intensity: 0.7
  shadow_map_size: '2048'
  shadow_bias: 0.0001
  context: desktop
"#;

    // Test direct serde_yaml deserialization
    println!("\nTesting direct serde_yaml deserialization:");
    match serde_yaml::from_str::<std::collections::HashMap<String, RenderingSettings>>(yaml_content) {
        Ok(settings) => {
            println!("SUCCESS: Direct YAML deserialization worked!");
            println!("Parsed settings: {:?}", settings);
        }
        Err(e) => {
            println!("FAILED: Direct YAML deserialization failed: {}", e);
        }
    }
    
    // Test with actual file content
    println!("\nTesting with actual settings.yaml file:");
    match fs::read_to_string("data/settings.yaml") {
        Ok(file_content) => {
            // Try to parse just the visualisation.rendering section
            match serde_yaml::from_str::<serde_yaml::Value>(&file_content) {
                Ok(parsed_yaml) => {
                    println!("SUCCESS: Full YAML parsed as Value");
                    
                    if let Some(visualisation) = parsed_yaml.get("visualisation") {
                        if let Some(rendering) = visualisation.get("rendering") {
                            println!("Found rendering section: {:?}", rendering);
                            
                            // Try to deserialize just the rendering section
                            match serde_yaml::from_value::<RenderingSettings>(rendering.clone()) {
                                Ok(r) => println!("SUCCESS: Rendering section deserialized: {:?}", r),
                                Err(e) => println!("FAILED: Rendering section deserialization failed: {}", e),
                            }
                        } else {
                            println!("No rendering section found in visualisation");
                        }
                    } else {
                        println!("No visualisation section found");
                    }
                }
                Err(e) => {
                    println!("FAILED: Could not parse YAML as Value: {}", e);
                }
            }
        }
        Err(e) => {
            println!("FAILED: Could not read settings.yaml: {}", e);
        }
    }
}