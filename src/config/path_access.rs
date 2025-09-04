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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_path() {
        assert_eq!(parse_path("a.b.c").unwrap(), vec!["a", "b", "c"]);
        assert_eq!(parse_path("single").unwrap(), vec!["single"]);
        assert!(parse_path("").is_err());
        assert!(parse_path("a..b").is_err());
    }
}