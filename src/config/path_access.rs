// Direct field access trait to eliminate JSON serialization bottleneck
// This trait provides efficient path-based field access without converting
// the entire struct to JSON for every operation.

use serde_json::Value;

/// Trait for accessing struct fields by dot-notation paths without JSON conversion
pub trait PathAccessible {
    /// Get a field value by path (e.g., "visualisation.graphs.logseq.physics.damping")
    fn get_by_path(&self, path: &str) -> Option<Value>;
    
    /// Set a field value by path (e.g., "visualisation.graphs.logseq.physics.damping")
    fn set_by_path(&mut self, path: &str, value: Value) -> Result<(), String>;
}

/// Helper function to parse and validate paths
pub fn parse_path(path: &str) -> Result<Vec<&str>, String> {
    if path.is_empty() {
        return Err("Empty path".to_string());
    }
    
    let parts: Vec<&str> = path.split('.').collect();
    if parts.iter().any(|part| part.is_empty()) {
        return Err("Path contains empty segments".to_string());
    }
    
    Ok(parts)
}

/// Macro to implement common field access patterns
macro_rules! impl_field_access {
    ($struct_name:ty, {
        $($field_name:literal => $field:ident: $field_type:ty),*
    }) => {
        impl PathAccessible for $struct_name {
            fn get_by_path(&self, path: &str) -> Option<Value> {
                let parts: Vec<&str> = path.split('.').collect();
                if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
                    return None;
                }
                
                match parts[0] {
                    $(
                        $field_name => {
                            if parts.len() == 1 {
                                // Return the field value directly
                                serde_json::to_value(&self.$field).ok()
                            } else {
                                // Delegate to nested struct
                                let remaining_path = parts[1..].join(".");
                                self.$field.get_by_path(&remaining_path)
                            }
                        }
                    )*
                    _ => None,
                }
            }
            
            fn set_by_path(&mut self, path: &str, value: Value) -> Result<(), String> {
                let parts: Vec<&str> = path.split('.').collect();
                if parts.is_empty() || parts.iter().any(|part| part.is_empty()) {
                    return Err("Invalid path".to_string());
                }
                
                match parts[0] {
                    $(
                        $field_name => {
                            if parts.len() == 1 {
                                // Set the field value directly
                                self.$field = serde_json::from_value(value)
                                    .map_err(|e| format!("Failed to deserialize value for {}: {}", $field_name, e))?;
                                Ok(())
                            } else {
                                // Delegate to nested struct
                                let remaining_path = parts[1..].join(".");
                                self.$field.set_by_path(&remaining_path, value)
                            }
                        }
                    )*
                    _ => Err(format!("Unknown field: {}", parts[0])),
                }
            }
        }
    };
}

pub(crate) use impl_field_access;