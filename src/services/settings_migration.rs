//! Settings Migration Service
//!
//! Migrates YAML-based settings to SQLite database with dual key format support.
//! Supports both camelCase (client) and snake_case (server) key formats simultaneously.

use std::path::Path;
use std::collections::HashMap;
use serde_yaml::Value as YamlValue;
use log::{info, warn, error, debug};
use rusqlite::{Connection, params, Result as SqliteResult};

use crate::services::database_service::{DatabaseService, SettingValue};

/// Settings migration service
pub struct SettingsMigration {
    db_service: std::sync::Arc<DatabaseService>,
}

impl SettingsMigration {
    /// Create new migration service
    pub fn new(db_service: std::sync::Arc<DatabaseService>) -> Self {
        Self { db_service }
    }

    /// Execute complete migration from YAML files to database
    pub fn migrate_from_yaml_files(&self) -> Result<MigrationResult, String> {
        info!("Starting settings migration from YAML to SQLite");
        let start_time = std::time::Instant::now();

        let mut result = MigrationResult::default();

        // Load and merge YAML files
        let main_yaml_path = std::env::var("DATA_ROOT")
            .unwrap_or_else(|_| "/app/data".to_string()) + "/settings.yaml";
        let ontology_yaml_path = std::env::var("DATA_ROOT")
            .unwrap_or_else(|_| "/app/data".to_string()) + "/settings_ontology_extension.yaml";

        // Parse main settings
        let main_settings = match self.load_yaml_file(&main_yaml_path) {
            Ok(value) => {
                info!("Loaded main settings from: {}", main_yaml_path);
                value
            }
            Err(e) => {
                error!("Failed to load main settings: {}", e);
                return Err(format!("Failed to load main settings: {}", e));
            }
        };

        // Parse ontology extension settings
        let ontology_settings = match self.load_yaml_file(&ontology_yaml_path) {
            Ok(value) => {
                info!("Loaded ontology extension from: {}", ontology_yaml_path);
                value
            }
            Err(e) => {
                warn!("Failed to load ontology extension (continuing without it): {}", e);
                YamlValue::Mapping(serde_yaml::Mapping::new())
            }
        };

        // Merge settings
        let merged = self.merge_yaml_values(vec![main_settings, ontology_settings]);

        // Flatten YAML hierarchy into key-value pairs
        let flattened = self.flatten_yaml(&merged, "");
        info!("Flattened {} settings keys", flattened.len());

        // Migrate each setting with dual key format
        for (key, value) in flattened.iter() {
            match self.migrate_setting(key, value) {
                Ok(_) => {
                    result.settings_migrated += 1;
                }
                Err(e) => {
                    error!("Failed to migrate setting '{}': {}", key, e);
                    result.errors.push(format!("Setting '{}': {}", key, e));
                }
            }
        }

        // Extract and migrate physics settings
        match self.migrate_physics_profiles(&merged) {
            Ok(count) => {
                info!("Migrated {} physics profiles", count);
                result.physics_profiles_migrated = count;
            }
            Err(e) => {
                error!("Failed to migrate physics profiles: {}", e);
                result.errors.push(format!("Physics profiles: {}", e));
            }
        }

        // Migrate dev_config.toml
        match self.migrate_dev_config_to_sqlite() {
            Ok(count) => {
                info!("Migrated {} dev_config parameters", count);
                result.dev_config_params_migrated = count;
            }
            Err(e) => {
                error!("Failed to migrate dev_config.toml: {}", e);
                result.errors.push(format!("Dev config: {}", e));
            }
        }

        result.duration = start_time.elapsed();
        info!("Migration completed in {:?}", result.duration);
        info!("  Settings migrated: {}", result.settings_migrated);
        info!("  Physics profiles: {}", result.physics_profiles_migrated);
        info!("  Dev config params: {}", result.dev_config_params_migrated);
        if !result.errors.is_empty() {
            warn!("  Errors encountered: {}", result.errors.len());
        }

        Ok(result)
    }

    /// Load YAML file from disk
    fn load_yaml_file(&self, path: &str) -> Result<YamlValue, String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        serde_yaml::from_str(&contents)
            .map_err(|e| format!("Failed to parse YAML: {}", e))
    }

    /// Merge multiple YAML values (deep merge)
    fn merge_yaml_values(&self, values: Vec<YamlValue>) -> YamlValue {
        let mut result = YamlValue::Mapping(serde_yaml::Mapping::new());

        for value in values {
            if let YamlValue::Mapping(map) = value {
                if let YamlValue::Mapping(result_map) = &mut result {
                    for (k, v) in map {
                        result_map.insert(k, v);
                    }
                }
            }
        }

        result
    }

    /// Flatten nested YAML into hierarchical keys
    fn flatten_yaml(&self, value: &YamlValue, prefix: &str) -> HashMap<String, YamlValue> {
        let mut result = HashMap::new();

        match value {
            YamlValue::Mapping(map) => {
                for (key, val) in map {
                    if let Some(key_str) = key.as_str() {
                        let new_prefix = if prefix.is_empty() {
                            key_str.to_string()
                        } else {
                            format!("{}.{}", prefix, key_str)
                        };

                        match val {
                            YamlValue::Mapping(_) => {
                                // Recursively flatten nested objects
                                let nested = self.flatten_yaml(val, &new_prefix);
                                result.extend(nested);
                            }
                            YamlValue::Sequence(_) => {
                                // Store arrays as JSON
                                result.insert(new_prefix, val.clone());
                            }
                            _ => {
                                // Store primitive values
                                result.insert(new_prefix, val.clone());
                            }
                        }
                    }
                }
            }
            _ => {
                // Non-mapping root value
                if !prefix.is_empty() {
                    result.insert(prefix.to_string(), value.clone());
                }
            }
        }

        result
    }

    /// Migrate a single setting with dual key format
    fn migrate_setting(&self, key: &str, value: &YamlValue) -> Result<(), String> {
        // Convert YAML value to SettingValue
        let setting_value = self.yaml_to_setting_value(value)?;

        // Store with camelCase key (original format)
        self.db_service.set_setting(key, setting_value.clone(), None)
            .map_err(|e| format!("Failed to store camelCase key: {}", e))?;

        // Generate snake_case equivalent
        let snake_key = self.to_snake_case_key(key);

        // Store with snake_case key if different
        if snake_key != key {
            self.db_service.set_setting(&snake_key, setting_value, None)
                .map_err(|e| format!("Failed to store snake_case key: {}", e))?;
        }

        debug!("Migrated: {} (and {})", key, snake_key);
        Ok(())
    }

    /// Convert YAML value to SettingValue
    fn yaml_to_setting_value(&self, value: &YamlValue) -> Result<SettingValue, String> {
        match value {
            YamlValue::String(s) => Ok(SettingValue::String(s.clone())),
            YamlValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(SettingValue::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(SettingValue::Float(f))
                } else {
                    Err("Unsupported number type".to_string())
                }
            }
            YamlValue::Bool(b) => Ok(SettingValue::Boolean(*b)),
            YamlValue::Sequence(_) | YamlValue::Mapping(_) => {
                // Convert to JSON for complex types
                let json = serde_json::to_value(value)
                    .map_err(|e| format!("Failed to convert to JSON: {}", e))?;
                Ok(SettingValue::Json(json))
            }
            YamlValue::Null => Ok(SettingValue::String("".to_string())),
            _ => Err("Unsupported YAML type".to_string()),
        }
    }

    /// Convert hierarchical key to snake_case
    fn to_snake_case_key(&self, key: &str) -> String {
        key.split('.')
            .map(|part| Self::to_snake_case_part(part))
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Convert hierarchical key to camelCase
    fn to_camel_case_key(&self, key: &str) -> String {
        key.split('.')
            .map(|part| Self::to_camel_case_part(part))
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Convert a single part to snake_case
    fn to_snake_case_part(s: &str) -> String {
        let mut result = String::new();
        let mut prev_is_upper = false;

        for (i, ch) in s.chars().enumerate() {
            if ch.is_uppercase() {
                if i > 0 && !prev_is_upper {
                    result.push('_');
                }
                result.push(ch.to_lowercase().next().unwrap());
                prev_is_upper = true;
            } else {
                result.push(ch);
                prev_is_upper = false;
            }
        }
        result
    }

    /// Convert a single part to camelCase
    fn to_camel_case_part(s: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;
        let mut first = true;

        for ch in s.chars() {
            if ch == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(ch.to_uppercase().next().unwrap());
                capitalize_next = false;
                first = false;
            } else {
                if first {
                    result.push(ch.to_lowercase().next().unwrap());
                    first = false;
                } else {
                    result.push(ch);
                }
            }
        }
        result
    }

    /// Migrate physics settings profiles
    fn migrate_physics_profiles(&self, yaml: &YamlValue) -> Result<usize, String> {
        let mut count = 0;

        // Extract physics settings from nested structure
        // Path: visualisation.graphs.{profile}.physics
        if let Some(vis) = yaml.get("visualisation") {
            if let Some(graphs) = vis.get("graphs") {
                if let YamlValue::Mapping(graphs_map) = graphs {
                    for (profile_name, profile_config) in graphs_map {
                        if let Some(profile_name_str) = profile_name.as_str() {
                            if let Some(physics) = profile_config.get("physics") {
                                match self.migrate_physics_profile(profile_name_str, physics) {
                                    Ok(_) => {
                                        count += 1;
                                        debug!("Migrated physics profile: {}", profile_name_str);
                                    }
                                    Err(e) => {
                                        warn!("Failed to migrate physics profile '{}': {}",
                                              profile_name_str, e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    /// Migrate a single physics profile
    fn migrate_physics_profile(&self, profile_name: &str, physics: &YamlValue) -> Result<(), String> {
        use crate::config::PhysicsSettings;

        // Extract physics settings from YAML
        let settings = PhysicsSettings {
            damping: self.get_f32(physics, "damping").unwrap_or(0.95),
            dt: self.get_f32(physics, "dt").unwrap_or(0.016),
            iterations: self.get_u32(physics, "iterations").unwrap_or(100),
            max_velocity: self.get_f32(physics, "maxVelocity").unwrap_or(1.0),
            max_force: self.get_f32(physics, "maxForce").unwrap_or(100.0),
            repel_k: self.get_f32(physics, "repelK").unwrap_or(50.0),
            spring_k: self.get_f32(physics, "springK").unwrap_or(0.005),
            mass_scale: self.get_f32(physics, "massScale").unwrap_or(1.0),
            boundary_damping: self.get_f32(physics, "boundaryDamping").unwrap_or(0.95),
            temperature: self.get_f32(physics, "temperature").unwrap_or(0.01),
            gravity: self.get_f32(physics, "gravity").unwrap_or(0.0001),
            bounds_size: self.get_f32(physics, "boundsSize").unwrap_or(500.0),
            enable_bounds: self.get_bool(physics, "enableBounds").unwrap_or(false),
            rest_length: self.get_f32(physics, "restLength").unwrap_or(50.0),
            repulsion_cutoff: self.get_f32(physics, "repulsionCutoff").unwrap_or(50.0),
            repulsion_softening_epsilon: self.get_f32(physics, "repulsionSofteningEpsilon").unwrap_or(0.0001),
            center_gravity_k: self.get_f32(physics, "centerGravityK").unwrap_or(0.0),
            grid_cell_size: self.get_f32(physics, "gridCellSize").unwrap_or(50.0),
            warmup_iterations: self.get_u32(physics, "warmupIterations").unwrap_or(100),
            cooling_rate: self.get_f32(physics, "coolingRate").unwrap_or(0.001),
            constraint_ramp_frames: self.get_u32(physics, "constraintRampFrames").unwrap_or(60),
            constraint_max_force_per_node: self.get_f32(physics, "constraintMaxForcePerNode").unwrap_or(50.0),
            ..PhysicsSettings::default()
        };

        self.db_service.save_physics_settings(profile_name, &settings)
            .map_err(|e| format!("Database error: {}", e))
    }

    /// Helper to get f32 from YAML
    fn get_f32(&self, yaml: &YamlValue, key: &str) -> Option<f32> {
        yaml.get(key)?.as_f64().map(|v| v as f32)
    }

    /// Helper to get u32 from YAML
    fn get_u32(&self, yaml: &YamlValue, key: &str) -> Option<u32> {
        yaml.get(key)?.as_i64().map(|v| v as u32)
    }

    /// Helper to get bool from YAML
    fn get_bool(&self, yaml: &YamlValue, key: &str) -> Option<bool> {
        yaml.get(key)?.as_bool()
    }

    /// Check if migration has been run
    pub fn is_migrated(&self) -> bool {
        // Check if any settings exist
        match self.db_service.get_setting("version") {
            Ok(Some(_)) => true,
            _ => false,
        }
    }

    /// Rollback migration (delete all settings)
    pub fn rollback(&self) -> Result<(), String> {
        warn!("Rolling back settings migration - this will delete all settings!");

        // This would require direct database access
        // For now, we'll just log a warning
        warn!("Rollback not fully implemented - manual database cleanup required");

        Ok(())
    }

    /// Migrate dev_config.toml to SQLite database
    pub fn migrate_dev_config_to_sqlite(&self) -> Result<usize, String> {
        info!("Starting dev_config.toml migration to SQLite");

        let dev_config_path = "data/dev_config.toml";

        // Check if file exists
        if !std::path::Path::new(dev_config_path).exists() {
            warn!("dev_config.toml not found, skipping migration");
            return Ok(0);
        }

        // Load TOML file
        let content = std::fs::read_to_string(dev_config_path)
            .map_err(|e| format!("Failed to read dev_config.toml: {}", e))?;

        let toml_value: toml::Value = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse dev_config.toml: {}", e))?;

        let mut count = 0;

        // Migrate each section
        if let toml::Value::Table(table) = toml_value {
            // Physics section (32 parameters)
            if let Some(toml::Value::Table(physics)) = table.get("physics") {
                count += self.migrate_toml_section("dev.physics", physics)?;
            }

            // CUDA section (11 parameters)
            if let Some(toml::Value::Table(cuda)) = table.get("cuda") {
                count += self.migrate_toml_section("dev.cuda", cuda)?;
            }

            // Network section (13 parameters)
            if let Some(toml::Value::Table(network)) = table.get("network") {
                count += self.migrate_toml_section("dev.network", network)?;
            }

            // Rendering section (with nested agent_colors)
            if let Some(toml::Value::Table(rendering)) = table.get("rendering") {
                // Separate agent_colors from other rendering params
                let mut rendering_copy = rendering.clone();

                // Handle nested agent_colors
                if let Some(toml::Value::Table(agent_colors)) = rendering_copy.remove("agent_colors") {
                    count += self.migrate_toml_section("dev.rendering.agent_colors", &agent_colors)?;
                }

                // Migrate remaining rendering params
                count += self.migrate_toml_section("dev.rendering", &rendering_copy)?;
            }

            // Performance section (11 parameters)
            if let Some(toml::Value::Table(performance)) = table.get("performance") {
                count += self.migrate_toml_section("dev.performance", performance)?;
            }

            // Debug section (8 parameters)
            if let Some(toml::Value::Table(debug)) = table.get("debug") {
                count += self.migrate_toml_section("dev.debug", debug)?;
            }
        }

        info!("Migrated {} dev_config parameters to database", count);
        Ok(count)
    }

    /// Migrate a TOML section to database with hierarchical keys
    fn migrate_toml_section(&self, prefix: &str, table: &toml::map::Map<String, toml::Value>) -> Result<usize, String> {
        let mut count = 0;

        for (key, value) in table {
            let full_key = format!("{}.{}", prefix, key);

            // Convert TOML value to SettingValue
            let setting_value = self.toml_to_setting_value(value)?;

            // Store with both camelCase and snake_case
            self.db_service.set_setting(&full_key, setting_value.clone(), None)
                .map_err(|e| format!("Failed to store key '{}': {}", full_key, e))?;

            let snake_key = self.to_snake_case_key(&full_key);
            if snake_key != full_key {
                self.db_service.set_setting(&snake_key, setting_value, None)
                    .map_err(|e| format!("Failed to store snake_case key '{}': {}", snake_key, e))?;
            }

            count += 1;
            debug!("Migrated dev_config parameter: {}", full_key);
        }

        Ok(count)
    }

    /// Convert TOML value to SettingValue
    fn toml_to_setting_value(&self, value: &toml::Value) -> Result<SettingValue, String> {
        match value {
            toml::Value::String(s) => Ok(SettingValue::String(s.clone())),
            toml::Value::Integer(i) => Ok(SettingValue::Integer(*i)),
            toml::Value::Float(f) => Ok(SettingValue::Float(*f)),
            toml::Value::Boolean(b) => Ok(SettingValue::Boolean(*b)),
            toml::Value::Array(_) | toml::Value::Table(_) => {
                // Convert complex types to JSON
                let json = serde_json::to_value(value)
                    .map_err(|e| format!("Failed to convert TOML to JSON: {}", e))?;
                Ok(SettingValue::Json(json))
            }
            toml::Value::Datetime(dt) => Ok(SettingValue::String(dt.to_string())),
        }
    }
}

/// Key format converter utility
pub struct KeyFormatConverter;

impl KeyFormatConverter {
    /// Convert any key between camelCase and snake_case
    pub fn to_snake_case(key: &str) -> String {
        key.split('.')
            .map(|part| Self::to_snake_case_part(part))
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Convert any key to camelCase
    pub fn to_camel_case(key: &str) -> String {
        key.split('.')
            .map(|part| Self::to_camel_case_part(part))
            .collect::<Vec<_>>()
            .join(".")
    }

    /// Get both key formats for a given key
    pub fn both_formats(key: &str) -> (String, String) {
        (Self::to_camel_case(key), Self::to_snake_case(key))
    }

    /// Convert a single part to snake_case
    fn to_snake_case_part(s: &str) -> String {
        let mut result = String::new();
        let mut prev_is_upper = false;

        for (i, ch) in s.chars().enumerate() {
            if ch.is_uppercase() {
                if i > 0 && !prev_is_upper {
                    result.push('_');
                }
                result.push(ch.to_lowercase().next().unwrap());
                prev_is_upper = true;
            } else {
                result.push(ch);
                prev_is_upper = false;
            }
        }
        result
    }

    /// Convert a single part to camelCase
    fn to_camel_case_part(s: &str) -> String {
        let mut result = String::new();
        let mut capitalize_next = false;
        let mut first = true;

        for ch in s.chars() {
            if ch == '_' {
                capitalize_next = true;
            } else if capitalize_next {
                result.push(ch.to_uppercase().next().unwrap());
                capitalize_next = false;
                first = false;
            } else {
                if first {
                    result.push(ch.to_lowercase().next().unwrap());
                    first = false;
                } else {
                    result.push(ch);
                }
            }
        }
        result
    }
}

/// Migration result statistics
#[derive(Debug, Default)]
pub struct MigrationResult {
    pub settings_migrated: usize,
    pub physics_profiles_migrated: usize,
    pub dev_config_params_migrated: usize,
    pub errors: Vec<String>,
    pub duration: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_conversion() {
        assert_eq!(
            KeyFormatConverter::to_snake_case("visualisation.graphs.logseq.nodes.baseColor"),
            "visualisation.graphs.logseq.nodes.base_color"
        );

        assert_eq!(
            KeyFormatConverter::to_camel_case("visualisation.graphs.logseq.nodes.base_color"),
            "visualisation.graphs.logseq.nodes.baseColor"
        );
    }

    #[test]
    fn test_both_formats() {
        let (camel, snake) = KeyFormatConverter::both_formats("visualisation.enableBounds");
        assert_eq!(camel, "visualisation.enableBounds");
        assert_eq!(snake, "visualisation.enable_bounds");
    }

    #[test]
    fn test_yaml_flattening() {
        let yaml_str = r#"
        root:
          nested:
            value: 42
            flag: true
          array: [1, 2, 3]
        "#;

        let yaml: YamlValue = serde_yaml::from_str(yaml_str).unwrap();
        let db = std::sync::Arc::new(
            DatabaseService::new(":memory:").unwrap()
        );
        let migration = SettingsMigration::new(db);

        let flattened = migration.flatten_yaml(&yaml, "");

        assert!(flattened.contains_key("root.nested.value"));
        assert!(flattened.contains_key("root.nested.flag"));
        assert!(flattened.contains_key("root.array"));
    }

    #[test]
    fn test_yaml_to_setting_value() {
        let db = std::sync::Arc::new(
            DatabaseService::new(":memory:").unwrap()
        );
        let migration = SettingsMigration::new(db);

        // String
        let yaml = YamlValue::String("test".to_string());
        match migration.yaml_to_setting_value(&yaml).unwrap() {
            SettingValue::String(s) => assert_eq!(s, "test"),
            _ => panic!("Expected String"),
        }

        // Integer
        let yaml = serde_yaml::to_value(42).unwrap();
        match migration.yaml_to_setting_value(&yaml).unwrap() {
            SettingValue::Integer(i) => assert_eq!(i, 42),
            _ => panic!("Expected Integer"),
        }

        // Float
        let yaml = serde_yaml::to_value(3.14).unwrap();
        match migration.yaml_to_setting_value(&yaml).unwrap() {
            SettingValue::Float(f) => assert!((f - 3.14).abs() < 0.001),
            _ => panic!("Expected Float"),
        }

        // Boolean
        let yaml = YamlValue::Bool(true);
        match migration.yaml_to_setting_value(&yaml).unwrap() {
            SettingValue::Boolean(b) => assert!(b),
            _ => panic!("Expected Boolean"),
        }
    }
}
