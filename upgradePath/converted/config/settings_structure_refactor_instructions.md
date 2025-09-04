# Settings Structure Refactoring Instructions

## Overview
Refactor the settings module structure to support path-based operations, validation, and improved organization. This involves updating mod.rs exports, adding new traits, and restructuring for better maintainability.

## File: src/config/mod.rs - Core Module Structure

### Updated Module Declaration and Exports

```rust
// Core configuration structures
pub mod app_settings;
pub mod ui_settings;
pub mod visualisation_settings;
pub mod physics_settings;
pub mod server_settings;

// New modules for enhanced functionality
pub mod path_access;
pub mod validation;

// Re-export main structures
pub use app_settings::*;
pub use ui_settings::*;
pub use visualisation_settings::*;
pub use physics_settings::*;
pub use server_settings::*;

// Re-export traits and utilities
pub use path_access::{PathAccessible, parse_path};
pub use validation::*;
```

### Core Imports and Dependencies

```rust
use serde::{Deserialize, Serialize};
use specta::Type;
use validator::{Validate, ValidationError};
use std::collections::HashMap;

// For TypeScript generation
use specta::Type;

// For validation patterns
use regex::Regex;
use lazy_static::lazy_static;
```

### AppFullSettings Structure Enhancement

Update the main settings structure to support the new functionality:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AppFullSettings {
    /// UI and theme settings
    #[validate]
    pub ui: UiSettings,
    
    /// Data visualization settings
    #[validate]
    pub visualisation: VisualisationSettings,
    
    /// Physics simulation settings
    #[validate]
    pub physics: PhysicsSettings,
    
    /// Server configuration settings
    #[validate]
    pub server: ServerSettings,
    
    /// Metadata about settings version and updates
    pub metadata: SettingsMetadata,
}

impl Default for AppFullSettings {
    fn default() -> Self {
        Self {
            ui: UiSettings::default(),
            visualisation: VisualisationSettings::default(),
            physics: PhysicsSettings::default(),
            server: ServerSettings::default(),
            metadata: SettingsMetadata::default(),
        }
    }
}
```

### Settings Metadata Structure

Add metadata tracking for settings management:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SettingsMetadata {
    /// Version of settings schema
    pub schema_version: String,
    
    /// Timestamp of last update
    pub last_updated: Option<i64>,
    
    /// Hash of current settings for change detection
    pub settings_hash: Option<String>,
    
    /// Validation status
    pub is_valid: bool,
    
    /// Last validation errors (if any)
    pub last_validation_errors: Option<HashMap<String, Vec<String>>>,
}

impl Default for SettingsMetadata {
    fn default() -> Self {
        Self {
            schema_version: "1.0.0".to_string(),
            last_updated: None,
            settings_hash: None,
            is_valid: true,
            last_validation_errors: None,
        }
    }
}
```

### PathAccessible Implementation for AppFullSettings

Implement the trait for the main settings structure:

```rust
impl crate::config::path_access::PathAccessible for AppFullSettings {
    fn get_by_path(&self, path: &str) -> Result<Box<dyn std::any::Any>, String> {
        let segments = crate::config::path_access::parse_path(path)?;
        
        match segments[0] {
            "ui" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.ui.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.ui.get_by_path(&remaining)
                }
            }
            "visualisation" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.visualisation.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.visualisation.get_by_path(&remaining)
                }
            }
            "physics" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.physics.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.physics.get_by_path(&remaining)
                }
            }
            "server" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.server.clone()))
                } else {
                    let remaining = segments[1..].join(".");
                    self.server.get_by_path(&remaining)
                }
            }
            "metadata" => {
                if segments.len() == 1 {
                    Ok(Box::new(self.metadata.clone()))
                } else {
                    Err("Metadata fields are not accessible via path".to_string())
                }
            }
            _ => Err(format!("Unknown top-level field: {}", segments[0]))
        }
    }
    
    fn set_by_path(&mut self, path: &str, value: Box<dyn std::any::Any>) -> Result<(), String> {
        let segments = crate::config::path_access::parse_path(path)?;
        
        match segments[0] {
            "ui" => {
                if segments.len() == 1 {
                    match value.downcast::<UiSettings>() {
                        Ok(v) => {
                            self.ui = *v;
                            self.update_metadata();
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for ui field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    let result = self.ui.set_by_path(&remaining, value);
                    if result.is_ok() {
                        self.update_metadata();
                    }
                    result
                }
            }
            "visualisation" => {
                if segments.len() == 1 {
                    match value.downcast::<VisualisationSettings>() {
                        Ok(v) => {
                            self.visualisation = *v;
                            self.update_metadata();
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for visualisation field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    let result = self.visualisation.set_by_path(&remaining, value);
                    if result.is_ok() {
                        self.update_metadata();
                    }
                    result
                }
            }
            "physics" => {
                if segments.len() == 1 {
                    match value.downcast::<PhysicsSettings>() {
                        Ok(v) => {
                            self.physics = *v;
                            self.update_metadata();
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for physics field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    let result = self.physics.set_by_path(&remaining, value);
                    if result.is_ok() {
                        self.update_metadata();
                    }
                    result
                }
            }
            "server" => {
                if segments.len() == 1 {
                    match value.downcast::<ServerSettings>() {
                        Ok(v) => {
                            self.server = *v;
                            self.update_metadata();
                            Ok(())
                        }
                        Err(_) => Err("Type mismatch for server field".to_string())
                    }
                } else {
                    let remaining = segments[1..].join(".");
                    let result = self.server.set_by_path(&remaining, value);
                    if result.is_ok() {
                        self.update_metadata();
                    }
                    result
                }
            }
            "metadata" => Err("Metadata fields cannot be modified via path".to_string()),
            _ => Err(format!("Unknown top-level field: {}", segments[0]))
        }
    }
}
```

### Metadata Update Methods

Add methods for maintaining settings metadata:

```rust
impl AppFullSettings {
    /// Updates metadata after settings changes
    fn update_metadata(&mut self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        self.metadata.last_updated = Some(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64
        );
        
        // Update settings hash for change detection
        self.metadata.settings_hash = Some(self.calculate_hash());
        
        // Validate and update status
        match self.validate_config_camel_case() {
            Ok(_) => {
                self.metadata.is_valid = true;
                self.metadata.last_validation_errors = None;
            }
            Err(errors) => {
                self.metadata.is_valid = false;
                self.metadata.last_validation_errors = Some(
                    Self::get_validation_errors_camel_case(&errors)
                );
            }
        }
    }
    
    /// Calculates hash of current settings for change detection
    fn calculate_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash the main settings (excluding metadata to avoid circular hashing)
        format!("{:?}{:?}{:?}{:?}", self.ui, self.visualisation, self.physics, self.server)
            .hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }
    
    /// Checks if settings have changed since last hash
    pub fn has_changed(&self) -> bool {
        match &self.metadata.settings_hash {
            Some(stored_hash) => &self.calculate_hash() != stored_hash,
            None => true // No stored hash means it's new/changed
        }
    }
    
    /// Gets a summary of current settings state
    pub fn get_settings_summary(&self) -> SettingsSummary {
        SettingsSummary {
            total_sections: 4, // ui, visualisation, physics, server
            valid_sections: self.count_valid_sections(),
            last_updated: self.metadata.last_updated,
            has_errors: !self.metadata.is_valid,
            schema_version: self.metadata.schema_version.clone(),
        }
    }
    
    /// Counts how many sections are valid
    fn count_valid_sections(&self) -> u32 {
        let mut count = 0;
        
        if self.ui.validate().is_ok() { count += 1; }
        if self.visualisation.validate().is_ok() { count += 1; }
        if self.physics.validate().is_ok() { count += 1; }
        if self.server.validate().is_ok() { count += 1; }
        
        count
    }
}
```

### Settings Summary Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct SettingsSummary {
    pub total_sections: u32,
    pub valid_sections: u32,
    pub last_updated: Option<i64>,
    pub has_errors: bool,
    pub schema_version: String,
}
```

### Utility Functions for Settings Management

```rust
/// Utility functions for settings management
impl AppFullSettings {
    /// Creates a new settings instance with validation
    pub fn new_with_validation() -> Result<Self, validator::ValidationErrors> {
        let settings = Self::default();
        settings.validate_config_camel_case()?;
        Ok(settings)
    }
    
    /// Merges settings from another instance with validation
    pub fn merge_settings(&mut self, other: &AppFullSettings) -> Result<(), String> {
        // Create a temporary copy for validation
        let mut temp = self.clone();
        
        // Merge each section
        temp.ui = other.ui.clone();
        temp.visualisation = other.visualisation.clone();
        temp.physics = other.physics.clone();
        temp.server = other.server.clone();
        
        // Validate the merged result
        match temp.validate_config_camel_case() {
            Ok(_) => {
                *self = temp;
                self.update_metadata();
                Ok(())
            }
            Err(errors) => Err(format!("Validation failed: {:?}", errors))
        }
    }
    
    /// Exports settings to JSON with camelCase formatting
    pub fn to_json_camel_case(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// Imports settings from JSON with validation
    pub fn from_json_with_validation(json: &str) -> Result<Self, String> {
        let settings: Self = serde_json::from_str(json)
            .map_err(|e| format!("JSON parsing error: {}", e))?;
        
        settings.validate_config_camel_case()
            .map_err(|e| format!("Validation error: {:?}", e))?;
        
        Ok(settings)
    }
}
```

## File Organization Requirements

After implementing these changes, ensure the following file structure:

```
src/config/
├── mod.rs              # Main module with AppFullSettings and utilities
├── path_access.rs      # PathAccessible trait and helpers
├── validation.rs       # Validation functions and patterns
├── app_settings.rs     # Individual setting structures (optional split)
├── ui_settings.rs      # UI-specific settings (optional split)
├── visualisation_settings.rs  # Visualization settings (optional split)
├── physics_settings.rs # Physics settings (optional split)
└── server_settings.rs  # Server settings (optional split)
```

## Key Benefits

1. **Modular Structure**: Clear separation of concerns with dedicated modules
2. **Path-Based Access**: Direct field access without JSON serialization
3. **Automatic Validation**: Compile-time and runtime validation with clear error messages
4. **Metadata Tracking**: Change detection and validation status tracking
5. **Frontend Compatibility**: Automatic camelCase conversion for API responses
6. **Type Safety**: Maintains Rust's type safety while enabling dynamic access
7. **Extensibility**: Easy to add new setting sections and validation rules

## Migration Notes

- Existing code using bulk settings operations will need to be updated
- API consumers should use the new granular endpoints
- Database schema may need updates to store metadata
- Frontend code should handle the new camelCase field names
- Error handling should account for the new validation system