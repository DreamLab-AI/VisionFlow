use serde_json::Value;
use std::any::Any;

/// Core trait for direct field access without JSON serialization overhead
/// This eliminates the performance bottleneck caused by converting entire settings 
/// structures to/from JSON for single field updates.
pub trait PathAccessible {
    /// Get a value by dot-notation path (e.g., "visualisation.nodes.baseColor")
    fn get_by_path(&self, path: &str) -> Result<Box<dyn Any>, String>;
    
    /// Set a value by dot-notation path with type checking
    fn set_by_path(&mut self, path: &str, value: Box<dyn Any>) -> Result<(), String>;
}

/// Enhanced trait that works with JSON values and leverages serde capabilities
/// This provides better integration with API endpoints and automatic case conversion
pub trait JsonPathAccessible: serde::Serialize + serde::de::DeserializeOwned {
    /// Get a JSON value by dot-notation path with automatic camelCase conversion
    fn get_json_by_path(&self, path: &str) -> Result<Value, String> {
        // Serialize self to JSON Value (this respects serde rename attributes)
        let root = serde_json::to_value(self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        
        // Navigate the path
        navigate_json_path(&root, path)
            .ok_or_else(|| format!("Path '{}' not found", path))
    }
    
    /// Set a JSON value by dot-notation path with validation
    fn set_json_by_path(&mut self, path: &str, value: Value) -> Result<(), String> {
        // Serialize current state to JSON
        let mut root = serde_json::to_value(&*self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        
        // Set the value at the path
        set_json_at_path(&mut root, path, value)?;
        
        // Deserialize back to self (this validates the structure)
        *self = serde_json::from_value(root)
            .map_err(|e| format!("Failed to deserialize: {}", e))?;
        
        Ok(())
    }
}

// Implement JsonPathAccessible for all types that have Serialize + DeserializeOwned
impl<T: serde::Serialize + serde::de::DeserializeOwned> JsonPathAccessible for T {}

/// Parse dot-notation path into segments
/// Returns error for empty paths or empty segments
pub fn parse_path(path: &str) -> Result<Vec<&str>, String> {
    if path.is_empty() {
        return Err("Path cannot be empty".to_string());
    }
    
    let segments: Vec<&str> = path.split('.').collect();
    
    // Validate no empty segments
    if segments.iter().any(|s| s.is_empty()) {
        return Err("Path segments cannot be empty".to_string());
    }
    
    Ok(segments)
}

/// Macro to implement common field access patterns
/// Reduces boilerplate code for implementing PathAccessible
#[allow(unused_macros)]
macro_rules! impl_field_access {
    ($struct_name:ident, {
        $($field:ident => $field_type:ty),*
    }) => {
        impl PathAccessible for $struct_name {
            fn get_by_path(&self, path: &str) -> Result<Box<dyn Any>, String> {
                let segments = parse_path(path)?;
                
                match segments[0] {
                    $(
                        stringify!($field) => {
                            if segments.len() == 1 {
                                Ok(Box::new(self.$field.clone()))
                            } else {
                                // Handle nested field access recursively
                                let remaining = segments[1..].join(".");
                                self.$field.get_by_path(&remaining)
                            }
                        }
                    )*
                    _ => Err(format!("Unknown field: {}", segments[0]))
                }
            }
            
            fn set_by_path(&mut self, path: &str, value: Box<dyn Any>) -> Result<(), String> {
                let segments = parse_path(path)?;
                
                match segments[0] {
                    $(
                        stringify!($field) => {
                            if segments.len() == 1 {
                                match value.downcast::<$field_type>() {
                                    Ok(v) => {
                                        self.$field = *v;
                                        Ok(())
                                    }
                                    Err(_) => Err(format!("Type mismatch for field {}", segments[0]))
                                }
                            } else {
                                // Handle nested field updates recursively
                                let remaining = segments[1..].join(".");
                                self.$field.set_by_path(&remaining, value)
                            }
                        }
                    )*
                    _ => Err(format!("Unknown field: {}", segments[0]))
                }
            }
        }
    };
}

// Make the macro available to other modules
#[allow(unused_imports)]
pub(crate) use impl_field_access;

/// Navigate to a JSON value by dot-notation path
/// Supports both camelCase and snake_case field names automatically
fn navigate_json_path(root: &Value, path: &str) -> Option<Value> {
    if path.is_empty() {
        return Some(root.clone());
    }
    
    let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
    let mut current = root;
    
    for segment in segments {
        match current {
            Value::Object(map) => {
                // Try the segment as-is first, then try snake_case conversion if camelCase fails
                current = map.get(segment)
                    .or_else(|| map.get(&camel_to_snake_case(segment)))
                    .or_else(|| map.get(&snake_to_camel_case(segment)))?;
            }
            _ => return None,
        }
    }
    
    Some(current.clone())
}

/// Set a JSON value at a dot-notation path
fn set_json_at_path(root: &mut Value, path: &str, value: Value) -> Result<(), String> {
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
                    // Use the exact segment name (preserving camelCase from API)
                    map.insert(segment.to_string(), value);
                    return Ok(());
                }
                _ => return Err(format!("Parent of '{}' is not an object", segment)),
            }
        } else {
            // Navigate deeper
            match current {
                Value::Object(map) => {
                    // Try to find existing field with flexible naming
                    let field_key = find_field_key(map, segment)
                        .unwrap_or_else(|| segment.to_string());
                    
                    // Create intermediate objects if needed
                    if !map.contains_key(&field_key) {
                        map.insert(field_key.clone(), Value::Object(serde_json::Map::new()));
                    }
                    current = map.get_mut(&field_key).unwrap();
                }
                _ => return Err(format!("Cannot navigate through non-object at '{}'", segment)),
            }
        }
    }
    
    Ok(())
}

/// Find the actual field key in a map, trying different case variants
fn find_field_key(map: &serde_json::Map<String, Value>, segment: &str) -> Option<String> {
    // Try exact match first
    if map.contains_key(segment) {
        return Some(segment.to_string());
    }
    
    // Try camelCase conversion
    let camel_case = snake_to_camel_case(segment);
    if map.contains_key(&camel_case) {
        return Some(camel_case);
    }
    
    // Try snake_case conversion
    let snake_case = camel_to_snake_case(segment);
    if map.contains_key(&snake_case) {
        return Some(snake_case);
    }
    
    None
}

/// Convert snake_case to camelCase
fn snake_to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    
    for ch in s.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    
    result
}

/// Convert camelCase to snake_case
fn camel_to_snake_case(s: &str) -> String {
    let mut result = String::new();
    
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[test]
    fn test_parse_path() {
        assert_eq!(parse_path("a.b.c").unwrap(), vec!["a", "b", "c"]);
        assert_eq!(parse_path("single").unwrap(), vec!["single"]);
        assert!(parse_path("").is_err());
        assert!(parse_path("a..b").is_err());
    }
    
    #[test]
    fn test_case_conversion() {
        assert_eq!(snake_to_camel_case("max_velocity"), "maxVelocity");
        assert_eq!(snake_to_camel_case("enable_hologram"), "enableHologram");
        assert_eq!(camel_to_snake_case("maxVelocity"), "max_velocity");
        assert_eq!(camel_to_snake_case("enableHologram"), "enable_hologram");
    }

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]  // This makes all fields use camelCase in JSON
    struct TestSettings {
        enable_hologram: bool,  // Serializes as "enableHologram"
        max_velocity: f32,      // Serializes as "maxVelocity"
        auto_balance: bool,     // Serializes as "autoBalance"
    }
    
    #[test]
    fn test_json_path_get() {
        let settings = TestSettings {
            enable_hologram: true,
            max_velocity: 10.0,
            auto_balance: false,
        };
        
        // Test camelCase paths (as they appear in JSON)
        let value = settings.get_json_by_path("enableHologram").unwrap();
        assert_eq!(value, Value::Bool(true));
        
        let value = settings.get_json_by_path("maxVelocity").unwrap();
        assert_eq!(value, Value::Number(serde_json::Number::from_f64(10.0).unwrap()));
    }
    
    #[test]
    fn test_json_path_set() {
        let mut settings = TestSettings {
            enable_hologram: true,
            max_velocity: 10.0,
            auto_balance: false,
        };
        
        // Set using camelCase path
        settings.set_json_by_path("enableHologram", Value::Bool(false)).unwrap();
        assert_eq!(settings.enable_hologram, false);
        
        settings.set_json_by_path("autoBalance", Value::Bool(true)).unwrap();
        assert_eq!(settings.auto_balance, true);
        
        settings.set_json_by_path("maxVelocity", Value::Number(serde_json::Number::from_f64(25.5).unwrap())).unwrap();
        assert_eq!(settings.max_velocity, 25.5);
    }
    
    #[test]
    fn test_nested_json_path() {
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        #[serde(rename_all = "camelCase")]
        struct NestedSettings {
            visualisation: VisualisationPart,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        #[serde(rename_all = "camelCase")]
        struct VisualisationPart {
            enable_hologram: bool,
            physics: PhysicsPart,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        #[serde(rename_all = "camelCase")]
        struct PhysicsPart {
            max_velocity: f32,
        }
        
        let mut settings = NestedSettings {
            visualisation: VisualisationPart {
                enable_hologram: true,
                physics: PhysicsPart {
                    max_velocity: 100.0,
                },
            },
        };
        
        // Access nested path with camelCase
        let value = settings.get_json_by_path("visualisation.enableHologram").unwrap();
        assert_eq!(value, Value::Bool(true));
        
        let value = settings.get_json_by_path("visualisation.physics.maxVelocity").unwrap();
        assert_eq!(value, Value::Number(serde_json::Number::from_f64(100.0).unwrap()));
        
        // Set nested path
        settings.set_json_by_path("visualisation.physics.maxVelocity", 
                                 Value::Number(serde_json::Number::from_f64(200.0).unwrap())).unwrap();
        assert_eq!(settings.visualisation.physics.max_velocity, 200.0);
    }
}