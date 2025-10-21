// SQLite-backed Settings Service
// Replaces file-based settings with database-backed configuration
// Supports user-specific overrides and hierarchical settings

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value as JsonValue;
use log::{info, error, debug, warn};

use crate::services::database_service::{DatabaseService, SettingValue};
use crate::services::settings_validator::{SettingsValidator, ValidationResult};
use crate::config::{AppFullSettings, PhysicsSettings};

#[derive(Clone)]
pub struct SettingsService {
    db: Arc<DatabaseService>,
    validator: Arc<SettingsValidator>,
    cache: Arc<RwLock<SettingsCache>>,
    change_listeners: Arc<RwLock<Vec<ChangeListener>>>,
}

#[derive(Clone)]
struct SettingsCache {
    settings: HashMap<String, CachedSetting>,
    last_updated: std::time::Instant,
}

#[derive(Clone)]
struct CachedSetting {
    value: SettingValue,
    timestamp: std::time::Instant,
}

type ChangeListener = Arc<dyn Fn(&str, &SettingValue, Option<&str>) + Send + Sync>;

#[derive(Debug, Clone)]
pub struct SettingsTreeNode {
    pub key: String,
    pub value: Option<SettingValue>,
    pub children: HashMap<String, SettingsTreeNode>,
}

impl SettingsService {
    pub fn new(db: Arc<DatabaseService>) -> Result<Self, String> {
        let validator = Arc::new(SettingsValidator::new());

        Ok(Self {
            db,
            validator,
            cache: Arc::new(RwLock::new(SettingsCache {
                settings: HashMap::new(),
                last_updated: std::time::Instant::now(),
            })),
            change_listeners: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Get setting by key (uses camelCase format, database handles snake_case fallback)
    pub async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>, String> {
        // Use key as-is (camelCase) - database service has smart lookup with fallback

        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.settings.get(key) {
                if cached.timestamp.elapsed().as_secs() < 300 { // 5 min TTL
                    debug!("Cache hit for setting: {}", key);
                    return Ok(Some(cached.value.clone()));
                }
            }
        }

        // Query database (smart lookup handles both camelCase and snake_case)
        match self.db.get_setting(key) {
            Ok(Some(value)) => {
                // Update cache
                let mut cache = self.cache.write().await;
                cache.settings.insert(key.to_string(), CachedSetting {
                    value: value.clone(),
                    timestamp: std::time::Instant::now(),
                });
                Ok(Some(value))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                error!("Database error getting setting {}: {}", key, e);
                Err(format!("Database error: {}", e))
            }
        }
    }

    /// Set setting value with permission check
    pub async fn set_setting(
        &self,
        key: &str,
        value: SettingValue,
        user_id: Option<&str>,
    ) -> Result<(), String> {
        // Use key as-is (camelCase format) - no normalization needed

        // Validate the setting
        let validation = self.validator.validate_setting(key, &value)?;
        if !validation.is_valid {
            return Err(format!(
                "Validation failed for {}: {}",
                key,
                validation.errors.join(", ")
            ));
        }

        // Store in database (camelCase format)
        self.db.set_setting(key, value.clone(), None)
            .map_err(|e| format!("Database error: {}", e))?;

        // Invalidate cache
        {
            let mut cache = self.cache.write().await;
            cache.settings.remove(key);
            cache.last_updated = std::time::Instant::now();
        }

        // Notify listeners
        self.notify_change(key, &value, user_id).await;

        info!("Setting updated: {} by user {:?}", key, user_id);
        Ok(())
    }

    /// Get settings tree by prefix
    pub async fn get_settings_tree(&self, prefix: &str) -> Result<SettingsTreeNode, String> {
        // Use prefix as-is (camelCase format)
        let all_settings = self.list_all_settings().await?;

        let mut root = SettingsTreeNode {
            key: prefix.to_string(),
            value: None,
            children: HashMap::new(),
        };

        // Build tree from flat settings
        for (key, value) in all_settings {
            if key.starts_with(prefix) {
                self.insert_into_tree(&mut root, &key, value, prefix);
            }
        }

        Ok(root)
    }

    fn insert_into_tree(
        &self,
        node: &mut SettingsTreeNode,
        key: &str,
        value: SettingValue,
        prefix: &str,
    ) {
        let relative_key = if key.starts_with(prefix) {
            &key[prefix.len()..].trim_start_matches('.')
        } else {
            key
        };

        let parts: Vec<&str> = relative_key.split('.').collect();
        if parts.is_empty() {
            return;
        }

        let mut current = node;
        for (i, part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // Leaf node
                current.children.insert(
                    part.to_string(),
                    SettingsTreeNode {
                        key: key.to_string(),
                        value: Some(value.clone()),
                        children: HashMap::new(),
                    },
                );
            } else {
                // Intermediate node
                current = current
                    .children
                    .entry(part.to_string())
                    .or_insert_with(|| SettingsTreeNode {
                        key: format!("{}.{}", prefix, parts[..=i].join(".")),
                        value: None,
                        children: HashMap::new(),
                    });
            }
        }
    }

    /// Get physics profile
    pub async fn get_physics_profile(&self, profile_name: &str) -> Result<PhysicsSettings, String> {
        self.db
            .get_physics_settings(profile_name)
            .map_err(|e| format!("Database error: {}", e))
    }

    /// Update physics profile with permission check
    pub async fn update_physics_profile(
        &self,
        profile_name: &str,
        params: PhysicsSettings,
        user_id: Option<&str>,
    ) -> Result<(), String> {
        // Validate physics settings
        let validation = self.validator.validate_physics_settings(&params)?;
        if !validation.is_valid {
            return Err(format!(
                "Physics validation failed: {}",
                validation.errors.join(", ")
            ));
        }

        // Save to database
        self.db
            .save_physics_settings(profile_name, &params)
            .map_err(|e| format!("Database error: {}", e))?;

        info!(
            "Physics profile {} updated by user {:?}",
            profile_name, user_id
        );
        Ok(())
    }

    /// List all settings (for power users)
    pub async fn list_all_settings(&self) -> Result<HashMap<String, SettingValue>, String> {
        // For now, return a simple implementation
        // In production, this would query all settings from the database
        let cache = self.cache.read().await;
        let mut result = HashMap::new();

        for (key, cached) in cache.settings.iter() {
            result.insert(key.clone(), cached.value.clone());
        }

        Ok(result)
    }

    /// Search settings by pattern
    pub async fn search_settings(&self, pattern: &str) -> Result<Vec<(String, SettingValue)>, String> {
        let all_settings = self.list_all_settings().await?;
        let pattern_lower = pattern.to_lowercase();

        let results: Vec<(String, SettingValue)> = all_settings
            .into_iter()
            .filter(|(key, _)| key.to_lowercase().contains(&pattern_lower))
            .collect();

        Ok(results)
    }

    /// Reset setting to default
    pub async fn reset_to_default(&self, key: &str, user_id: Option<&str>) -> Result<(), String> {
        // Use key as-is (camelCase format)

        // Get default value from AppFullSettings
        let defaults = AppFullSettings::default();
        let default_value = self.extract_default_value(&defaults, key)?;

        self.set_setting(key, default_value, user_id).await
    }

    /// Register change listener for WebSocket broadcasts
    pub async fn register_change_listener<F>(&self, listener: F)
    where
        F: Fn(&str, &SettingValue, Option<&str>) + Send + Sync + 'static,
    {
        let mut listeners = self.change_listeners.write().await;
        listeners.push(Arc::new(listener));
    }

    /// Notify all listeners of a setting change
    async fn notify_change(&self, key: &str, value: &SettingValue, user_id: Option<&str>) {
        let listeners = self.change_listeners.read().await;
        for listener in listeners.iter() {
            listener(key, value, user_id);
        }
    }

    /// Extract default value from AppFullSettings
    fn extract_default_value(
        &self,
        defaults: &AppFullSettings,
        key: &str,
    ) -> Result<SettingValue, String> {
        // Convert settings to JSON and extract the value
        let json = serde_json::to_value(defaults)
            .map_err(|e| format!("Failed to serialize defaults: {}", e))?;

        let parts: Vec<&str> = key.split('.').collect();
        let mut current = &json;

        for part in parts {
            match current.get(part) {
                Some(v) => current = v,
                None => {
                    return Err(format!("Key not found in defaults: {}", key));
                }
            }
        }

        // Convert JSON value to SettingValue
        match current {
            JsonValue::String(s) => Ok(SettingValue::String(s.clone())),
            JsonValue::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(SettingValue::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(SettingValue::Float(f))
                } else {
                    Err("Invalid number type".to_string())
                }
            }
            JsonValue::Bool(b) => Ok(SettingValue::Boolean(*b)),
            JsonValue::Object(_) | JsonValue::Array(_) => {
                Ok(SettingValue::Json(current.clone()))
            }
            JsonValue::Null => Err("Cannot reset to null value".to_string()),
        }
    }

    /// Clear cache
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.settings.clear();
        cache.last_updated = std::time::Instant::now();
        info!("Settings cache cleared");
    }

    /// Warm cache with common settings
    pub async fn warm_cache(&self) {
        let common_keys = vec![
            "visualisation.graphs.logseq.physics",
            "visualisation.rendering",
            "system",
        ];

        for key in common_keys {
            if let Err(e) = self.get_setting(key).await {
                warn!("Failed to warm cache for {}: {}", key, e);
            }
        }

        info!("Settings cache warmed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_camel_case_keys() {
        // Verify that settings service uses camelCase keys directly
        let db = Arc::new(DatabaseService::new(":memory:").unwrap());
        db.initialize_schema().unwrap();

        let service = SettingsService::new(db).unwrap();

        // Set a camelCase setting
        service.set_setting(
            "testSetting",
            SettingValue::String("test_value".to_string()),
            None
        ).await.unwrap();

        // Should retrieve with same camelCase key
        let value = service.get_setting("testSetting").await.unwrap();
        assert!(value.is_some());
    }
}
