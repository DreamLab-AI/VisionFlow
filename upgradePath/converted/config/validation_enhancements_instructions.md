# Config Validation Enhancements Instructions

## Overview
Add comprehensive validation system with camelCase support for frontend compatibility. This replaces manual validation code with compile-time validation attributes and centralized validation patterns.

## File: src/config/mod.rs - Enhanced Validation System

### Required Imports

Add these imports to enable validation and type generation:

```rust
use specta::Type;
use validator::{Validate, ValidationError};
use regex::Regex;
use lazy_static::lazy_static;
```

### Centralized Validation Patterns

Add these regex patterns for common validation needs:

```rust
lazy_static! {
    /// Validates hex color format (#RRGGBB or #RRGGBBAA)
    static ref HEX_COLOR_REGEX: Regex = Regex::new(r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$").unwrap();
    
    /// Validates URL format (http/https)
    static ref URL_REGEX: Regex = Regex::new(r"^https?://[^\s/$.?#].[^\s]*$").unwrap();
    
    /// Validates file path format (Unix/Windows compatible)
    static ref FILE_PATH_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9._/\\-]+$").unwrap();
    
    /// Validates domain name format
    static ref DOMAIN_REGEX: Regex = Regex::new(r"^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*$").unwrap();
}
```

### Custom Validation Functions

Add these validation functions for specific business logic:

```rust
/// Validates hex color format (#RRGGBB or #RRGGBBAA)
pub fn validate_hex_color(color: &str) -> Result<(), ValidationError> {
    if !HEX_COLOR_REGEX.is_match(color) {
        return Err(ValidationError::new("invalid_hex_color"));
    }
    Ok(())
}

/// Validates width range has exactly 2 elements with proper min/max order
pub fn validate_width_range(range: &[f64]) -> Result<(), ValidationError> {
    if range.len() != 2 {
        return Err(ValidationError::new("width_range_length"));
    }
    if range[0] >= range[1] {
        return Err(ValidationError::new("width_range_order"));
    }
    Ok(())
}

/// Validates port number is in valid range (1-65535)
pub fn validate_port(port: u16) -> Result<(), ValidationError> {
    if port == 0 {
        return Err(ValidationError::new("invalid_port"));
    }
    Ok(())
}

/// Validates percentage is between 0 and 100
pub fn validate_percentage(value: f64) -> Result<(), ValidationError> {
    if !(0.0..=100.0).contains(value) {
        return Err(ValidationError::new("invalid_percentage"));
    }
    Ok(())
}
```

### Enhanced Struct Validation Attributes

Update existing config structs with validation attributes:

#### NodeSettings Validation

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct NodeSettings {
    #[validate(custom(function = "validate_hex_color"))]
    pub fill_color: String,
    
    #[validate(custom(function = "validate_hex_color"))]
    pub stroke_color: String,
    
    #[validate(range(min = 1.0, max = 50.0))]
    pub stroke_width: f64,
    
    #[validate(range(min = 0.1, max = 10.0))]
    pub opacity: f64,
    
    #[validate(range(min = 1, max = 100))]
    pub size: u32,
    
    pub enabled: bool,
}
```

#### EdgeSettings Validation

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct EdgeSettings {
    #[validate(custom(function = "validate_hex_color"))]
    pub color: String,
    
    #[validate(custom(function = "validate_width_range"))]
    pub width_range: Vec<f64>,
    
    #[validate(range(min = 0.0, max = 1.0))]
    pub opacity: f64,
    
    #[validate(range(min = 0.0, max = 5.0))]
    pub curve: f64,
    
    pub enabled: bool,
}
```

#### MovementAxes Validation

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct MovementAxes {
    #[validate(range(min = -100.0, max = 100.0))]
    pub x: f64,
    
    #[validate(range(min = -100.0, max = 100.0))]
    pub y: f64,
    
    #[validate(range(min = -100.0, max = 100.0))]
    pub z: f64,
    
    pub enabled: bool,
}
```

#### ServerSettings Validation

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct ServerSettings {
    #[validate(custom(function = "validate_port"))]
    pub port: u16,
    
    #[validate(regex(path = "DOMAIN_REGEX"))]
    pub host: String,
    
    #[validate(url)]
    pub api_endpoint: Option<String>,
    
    pub ssl_enabled: bool,
    
    #[validate(custom(function = "validate_file_path"))]
    pub ssl_cert_path: Option<String>,
}
```

### Validation Implementation for AppFullSettings

Add validation method that supports camelCase field names:

```rust
impl AppFullSettings {
    /// Validates the entire configuration with camelCase field names for frontend compatibility
    pub fn validate_config_camel_case(&self) -> Result<(), validator::ValidationErrors> {
        // Validate the entire struct
        self.validate()?;
        
        // Additional cross-field validation
        self.validate_cross_field_constraints()?;
        
        Ok(())
    }
    
    /// Validates constraints that span multiple fields
    fn validate_cross_field_constraints(&self) -> Result<(), validator::ValidationErrors> {
        let mut errors = validator::ValidationErrors::new();
        
        // Example: Check that physics simulation is enabled if physics settings are configured
        if self.physics.gravity != 0.0 && !self.physics.enabled {
            errors.add("physics", ValidationError::new("physics_enabled_required"));
        }
        
        // Example: Check that visualization is enabled if any visual settings are non-default
        if self.visualisation.nodes.enabled && !self.visualisation.enabled {
            errors.add("visualisation", ValidationError::new("visualisation_enabled_required"));
        }
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}
```

### Custom File Path Validation

```rust
/// Validates file path format and existence (optional)
pub fn validate_file_path(path: &str) -> Result<(), ValidationError> {
    if !FILE_PATH_REGEX.is_match(path) {
        return Err(ValidationError::new("invalid_file_path"));
    }
    
    // Optional: Check if file exists (for critical paths)
    // if !std::path::Path::new(path).exists() {
    //     return Err(ValidationError::new("file_not_found"));
    // }
    
    Ok(())
}
```

### Error Message Customization

Add custom error messages for better user experience:

```rust
impl AppFullSettings {
    /// Gets user-friendly error messages in camelCase format
    pub fn get_validation_errors_camel_case(
        errors: &validator::ValidationErrors
    ) -> std::collections::HashMap<String, Vec<String>> {
        let mut result = std::collections::HashMap::new();
        
        for (field, field_errors) in errors.field_errors() {
            let camel_case_field = to_camel_case(field);
            let messages: Vec<String> = field_errors
                .iter()
                .map(|error| match error.code.as_ref() {
                    "invalid_hex_color" => "Must be a valid hex color (#RRGGBB or #RRGGBBAA)".to_string(),
                    "width_range_length" => "Width range must have exactly 2 values".to_string(),
                    "width_range_order" => "Width range minimum must be less than maximum".to_string(),
                    "invalid_port" => "Port must be between 1 and 65535".to_string(),
                    "invalid_percentage" => "Value must be between 0 and 100".to_string(),
                    "invalid_file_path" => "Invalid file path format".to_string(),
                    _ => format!("Invalid value for {}", camel_case_field),
                })
                .collect();
            
            result.insert(camel_case_field, messages);
        }
        
        result
    }
}

/// Converts snake_case to camelCase
fn to_camel_case(snake_str: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    
    for ch in snake_str.chars() {
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
```

## Legacy Code Removal

Remove the following deprecated functions from mod.rs:

```rust
// REMOVE: This entire function (~40 lines)
pub fn convert_empty_strings_to_null(value: &mut serde_json::Value) {
    // ... entire implementation should be removed
}
```

**Reason**: Replaced by proper validation attributes and serde configuration.

## Key Benefits

1. **Compile-time Validation**: Attributes catch errors at build time
2. **Frontend Compatibility**: Automatic camelCase field name conversion
3. **Centralized Patterns**: Reusable regex patterns reduce duplication
4. **User-friendly Errors**: Clear error messages for API consumers
5. **Cross-field Validation**: Support for complex business rules
6. **Type Safety**: Maintains Rust's type safety guarantees

## Usage Example

```rust
use crate::config::{AppFullSettings, validate_hex_color};

let mut settings = AppFullSettings::default();

// Validation happens automatically during deserialization
match serde_json::from_str::<AppFullSettings>(&json_data) {
    Ok(settings) => {
        // Additional validation if needed
        match settings.validate_config_camel_case() {
            Ok(_) => println!("Settings valid"),
            Err(errors) => {
                let friendly_errors = AppFullSettings::get_validation_errors_camel_case(&errors);
                // Return friendly_errors to frontend
            }
        }
    }
    Err(e) => println!("Deserialization error: {}", e)
}
```

## Integration Requirements

- Must work with PathAccessible trait for direct field access
- Should integrate with new settings actor message handlers
- Error responses must be in camelCase format for frontend compatibility
- Validation should occur before any path-based updates are committed