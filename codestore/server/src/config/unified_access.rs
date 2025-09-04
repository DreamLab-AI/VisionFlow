// Unified PathAccessible implementation - no case conversion needed!
// This uses serde's built-in capabilities to handle both snake_case and camelCase automatically

use serde_json::Value;

/// Generic PathAccessible implementation that works with any Serialize/Deserialize type
/// This leverages serde's renaming capabilities, so we don't need manual case conversion
pub trait UnifiedPathAccessible: serde::Serialize + serde::de::DeserializeOwned {
    fn unified_get_by_path(&self, path: &str) -> Option<Value> {
        // Serialize self to JSON Value
        let root = serde_json::to_value(self).ok()?;
        
        // Navigate the path
        navigate_path(&root, path)
    }
    
    fn unified_set_by_path(&mut self, path: &str, value: Value) -> Result<(), String> {
        // Serialize current state to JSON
        let mut root = serde_json::to_value(&*self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        
        // Set the value at the path
        set_at_path(&mut root, path, value)?;
        
        // Deserialize back to self
        *self = serde_json::from_value(root)
            .map_err(|e| format!("Failed to deserialize: {}", e))?;
        
        Ok(())
    }
}

// Implement for all types that have Serialize + DeserializeOwned
impl<T: serde::Serialize + serde::de::DeserializeOwned> UnifiedPathAccessible for T {}

/// Navigate to a value by dot-notation path
fn navigate_path(root: &Value, path: &str) -> Option<Value> {
    if path.is_empty() {
        return Some(root.clone());
    }
    
    let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
    let mut current = root;
    
    for segment in segments {
        match current {
            Value::Object(map) => {
                current = map.get(segment)?;
            }
            _ => return None,
        }
    }
    
    Some(current.clone())
}

/// Set a value at a dot-notation path
fn set_at_path(root: &mut Value, path: &str, value: Value) -> Result<(), String> {
    if path.is_empty() {
        *root = value;
        return Ok(());
    }
    
    let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
    
    if segments.is_empty() {
        return Err("Invalid empty path".to_string());
    }
    
    // Navigate to the parent of the target
    let mut current = root;
    
    for (i, segment) in segments.iter().enumerate() {
        if i == segments.len() - 1 {
            // Last segment - set the value
            match current {
                Value::Object(map) => {
                    map.insert(segment.to_string(), value);
                    return Ok(());
                }
                _ => return Err(format!("Parent of '{}' is not an object", segment)),
            }
        } else {
            // Navigate deeper
            match current {
                Value::Object(map) => {
                    // Create intermediate objects if needed
                    if !map.contains_key(*segment) {
                        map.insert(segment.to_string(), Value::Object(serde_json::Map::new()));
                    }
                    current = map.get_mut(*segment).unwrap();
                }
                _ => return Err(format!("Cannot navigate through non-object at '{}'", segment)),
            }
        }
    }
    
    Ok(())
}

/// Macro to implement PathAccessible using the unified approach
#[macro_export]
macro_rules! impl_unified_path_accessible {
    ($struct_type:ty) => {
        impl $crate::config::path_access::PathAccessible for $struct_type {
            fn get_by_path(&self, path: &str) -> Option<serde_json::Value> {
                use $crate::config::unified_access::UnifiedPathAccessible;
                self.unified_get_by_path(path)
            }
            
            fn set_by_path(&mut self, path: &str, value: serde_json::Value) -> Result<(), String> {
                use $crate::config::unified_access::UnifiedPathAccessible;
                self.unified_set_by_path(path, value)
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]  // This makes all fields use camelCase in JSON
    struct TestSettings {
        enable_hologram: bool,  // Serializes as "enableHologram"
        max_velocity: f32,      // Serializes as "maxVelocity"
        auto_balance: bool,     // Serializes as "autoBalance"
    }
    
    impl_unified_path_accessible!(TestSettings);
    
    #[test]
    fn test_unified_get() {
        let settings = TestSettings {
            enable_hologram: true,
            max_velocity: 10.0,
            auto_balance: false,
        };
        
        // The path uses camelCase because of serde(rename_all = "camelCase")
        let value = settings.get_by_path("enableHologram").unwrap();
        assert_eq!(value, Value::Bool(true));
        
        let value = settings.get_by_path("maxVelocity").unwrap();
        assert_eq!(value, Value::Number(serde_json::Number::from_f64(10.0).unwrap()));
    }
    
    #[test]
    fn test_unified_set() {
        let mut settings = TestSettings {
            enable_hologram: true,
            max_velocity: 10.0,
            auto_balance: false,
        };
        
        // Set using camelCase path
        settings.set_by_path("enableHologram", Value::Bool(false)).unwrap();
        assert_eq!(settings.enable_hologram, false);
        
        settings.set_by_path("autoBalance", Value::Bool(true)).unwrap();
        assert_eq!(settings.auto_balance, true);
    }
    
    #[test]
    fn test_nested_path() {
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct NestedSettings {
            visualisation: VisualisationPart,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct VisualisationPart {
            enable_hologram: bool,
            nodes: NodesPart,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct NodesPart {
            max_size: f32,
        }
        
        impl_unified_path_accessible!(NestedSettings);
        
        let settings = NestedSettings {
            visualisation: VisualisationPart {
                enable_hologram: true,
                nodes: NodesPart {
                    max_size: 100.0,
                },
            },
        };
        
        // Access nested path
        let value = settings.get_by_path("visualisation.enableHologram").unwrap();
        assert_eq!(value, Value::Bool(true));
        
        let value = settings.get_by_path("visualisation.nodes.maxSize").unwrap();
        assert_eq!(value, Value::Number(serde_json::Number::from_f64(100.0).unwrap()));
    }
}