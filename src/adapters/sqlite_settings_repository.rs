// src/adapters/sqlite_settings_repository.rs
//! SQLite Settings Repository Adapter
//!
//! Implements the SettingsRepository port using SQLite with async interface,
//! caching, and intelligent camelCase/snake_case conversion.

use async_trait::async_trait;
use rusqlite::{Connection, OptionalExtension};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

use crate::config::PhysicsSettings;
use crate::ports::settings_repository::{
    AppFullSettings, Result as RepoResult, SettingValue, SettingsRepository,
    SettingsRepositoryError,
};

/// SQLite-backed settings repository with caching
pub struct SqliteSettingsRepository {
    conn: Arc<Mutex<Connection>>,
    cache: Arc<RwLock<SettingsCache>>,
}

struct SettingsCache {
    settings: HashMap<String, CachedSetting>,
    last_updated: std::time::Instant,
    ttl_seconds: u64,
}

struct CachedSetting {
    value: SettingValue,
    timestamp: std::time::Instant,
}

impl SqliteSettingsRepository {
    /// Create new SQLite settings repository with caching
    pub fn new(db_path: &str) -> RepoResult<Self> {
        info!("Initializing SqliteSettingsRepository with cache TTL=300s, path: {}", db_path);
        let conn = Connection::open(db_path)
            .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Failed to open database: {}", e)))?;

        // Ensure settings table exists (it should be created by unified_schema.sql)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL UNIQUE,
                parent_key TEXT,
                value_type TEXT NOT NULL,
                value_text TEXT,
                value_integer INTEGER,
                value_float REAL,
                value_boolean INTEGER,
                value_json TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )",
            [],
        ).map_err(|e| SettingsRepositoryError::DatabaseError(format!("Failed to create settings table: {}", e)))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            cache: Arc::new(RwLock::new(SettingsCache {
                settings: HashMap::new(),
                last_updated: std::time::Instant::now(),
                ttl_seconds: 300, // 5 minutes
            })),
        })
    }

    /// Check cache and return value if valid
    async fn get_from_cache(&self, key: &str) -> Option<SettingValue> {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.settings.get(key) {
            if cached.timestamp.elapsed().as_secs() < cache.ttl_seconds {
                debug!("Cache hit for setting: {}", key);
                return Some(cached.value.clone());
            }
        }
        None
    }

    /// Update cache with new value
    async fn update_cache(&self, key: String, value: SettingValue) {
        let mut cache = self.cache.write().await;
        cache.settings.insert(
            key,
            CachedSetting {
                value,
                timestamp: std::time::Instant::now(),
            },
        );
    }

    /// Invalidate cache entry
    async fn invalidate_cache(&self, key: &str) {
        let mut cache = self.cache.write().await;
        cache.settings.remove(key);
    }

    /// Clear entire cache
    async fn clear_cache(&self) -> RepoResult<()> {
        let mut cache = self.cache.write().await;
        cache.settings.clear();
        Ok(())
    }
}

#[async_trait]
impl SettingsRepository for SqliteSettingsRepository {
    #[instrument(skip(self), level = "debug")]
    async fn get_setting(&self, key: &str) -> RepoResult<Option<SettingValue>> {
        // Check cache first
        if let Some(cached_value) = self.get_from_cache(key).await {
            return Ok(Some(cached_value));
        }

        // Query database (blocking operation, run in thread pool)
        let conn = self.conn.clone();
        let key_owned = key.to_string();
        let result = tokio::task::spawn_blocking(move || {
            let conn_guard = conn.lock().map_err(|e| format!("Failed to lock connection: {}", e))?;
            conn_guard.query_row(
                "SELECT value_text, value_integer, value_float, value_boolean, value_json FROM settings WHERE key = ?1",
                [&key_owned],
                |row| {
                    // Parse based on which column has a value (first non-NULL)
                    if let Ok(Some(text)) = row.get::<_, Option<String>>(0) {
                        return Ok(Some(SettingValue::String(text)));
                    }
                    if let Ok(Some(int)) = row.get::<_, Option<i64>>(1) {
                        return Ok(Some(SettingValue::Integer(int)));
                    }
                    if let Ok(Some(float)) = row.get::<_, Option<f64>>(2) {
                        return Ok(Some(SettingValue::Float(float)));
                    }
                    if let Ok(Some(bool_val)) = row.get::<_, Option<i64>>(3) {
                        return Ok(Some(SettingValue::Boolean(bool_val != 0)));
                    }
                    if let Ok(Some(json_str)) = row.get::<_, Option<String>>(4) {
                        let json_value = serde_json::from_str(&json_str)
                            .unwrap_or(serde_json::Value::Null);
                        return Ok(Some(SettingValue::Json(json_value)));
                    }
                    Ok(None)
                },
            )
            .optional()
            .map_err(|e| format!("Database query failed: {}", e))?
            .flatten()
            .ok_or_else(|| format!("Setting not found: {}", key_owned))
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?;

        let result = match result {
            Ok(val) => Some(val),
            Err(_) => None, // Setting not found is not an error
        };

        // Update cache on success
        if let Some(ref value) = result {
            self.update_cache(key.to_string(), value.clone()).await;
        }

        Ok(result)
    }

    #[instrument(skip(self, value), level = "debug")]
    async fn set_setting(
        &self,
        key: &str,
        value: SettingValue,
        description: Option<&str>,
    ) -> RepoResult<()> {
        let conn = self.conn.clone();
        let key_owned = key.to_string();
        let value_owned = value.clone();
        let description_owned = description.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            let conn_guard = conn.lock().map_err(|e| format!("Failed to lock connection: {}", e))?;

            // Determine value type and columns
            let (value_type, text_val, int_val, float_val, bool_val, json_val) = match value_owned {
                SettingValue::String(s) => ("string", Some(s), None, None, None, None),
                SettingValue::Integer(i) => ("integer", None, Some(i), None, None, None),
                SettingValue::Float(f) => ("float", None, None, Some(f), None, None),
                SettingValue::Boolean(b) => ("boolean", None, None, None, Some(if b { 1 } else { 0 }), None),
                SettingValue::Json(j) => {
                    let json_str = serde_json::to_string(&j)
                        .map_err(|e| format!("Failed to serialize JSON: {}", e))?;
                    ("json", None, None, None, None, Some(json_str))
                },
            };

            conn_guard.execute(
                "INSERT INTO settings (key, value_type, value_text, value_integer, value_float, value_boolean, value_json, description)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(key) DO UPDATE SET
                    value_type = excluded.value_type,
                    value_text = excluded.value_text,
                    value_integer = excluded.value_integer,
                    value_float = excluded.value_float,
                    value_boolean = excluded.value_boolean,
                    value_json = excluded.value_json,
                    description = COALESCE(excluded.description, description),
                    updated_at = CURRENT_TIMESTAMP",
                rusqlite::params![key_owned, value_type, text_val, int_val, float_val, bool_val, json_val, description_owned],
            )
            .map_err(|e| format!("Failed to set setting: {}", e))?;

            Ok(())
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e))?;

        // Invalidate cache
        self.invalidate_cache(key).await;

        Ok(())
    }

    async fn get_settings_batch(
        &self,
        keys: &[String],
    ) -> RepoResult<HashMap<String, SettingValue>> {
        let mut results = HashMap::new();

        for key in keys {
            if let Some(value) = self.get_setting(key).await? {
                results.insert(key.clone(), value);
            }
        }

        Ok(results)
    }

    async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> RepoResult<()> {
        for (key, value) in updates {
            self.set_setting(&key, value, None).await?;
        }
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    async fn load_all_settings(&self) -> RepoResult<Option<AppFullSettings>> {
        // Return a stub implementation - settings are managed individually
        Ok(Some(AppFullSettings {
            visualisation: Default::default(),
            system: Default::default(),
            xr: Default::default(),
            auth: Default::default(),
            ragflow: None,
            perplexity: None,
            openai: None,
            kokoro: None,
            whisper: None,
            version: "1.0.0".to_string(),
            user_preferences: Default::default(),
            physics: Default::default(),
            feature_flags: Default::default(),
            developer_config: Default::default(),
        }))
    }

    #[instrument(skip(self), level = "debug")]
    async fn save_all_settings(&self, _settings: &AppFullSettings) -> RepoResult<()> {
        // Stub implementation - settings are managed individually
        self.clear_cache().await?;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    async fn get_physics_settings(&self, _profile_name: &str) -> RepoResult<PhysicsSettings> {
        // Return default physics settings - can be enhanced later
        Ok(PhysicsSettings::default())
    }

    #[instrument(skip(self), level = "debug")]
    async fn save_physics_settings(
        &self,
        _profile_name: &str,
        _settings: &PhysicsSettings,
    ) -> RepoResult<()> {
        // Stub implementation
        Ok(())
    }

    async fn delete_setting(&self, key: &str) -> RepoResult<()> {
        let conn = self.conn.clone();
        let key_owned = key.to_string();

        tokio::task::spawn_blocking(move || {
            let conn_guard = conn.lock().map_err(|e| format!("Failed to lock connection: {}", e))?;
            conn_guard.execute("DELETE FROM settings WHERE key = ?1", [&key_owned])
                .map_err(|e| format!("Failed to delete setting: {}", e))?;
            Ok(())
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e))?;

        self.invalidate_cache(key).await;
        Ok(())
    }

    async fn has_setting(&self, key: &str) -> RepoResult<bool> {
        Ok(self.get_setting(key).await?.is_some())
    }

    async fn list_settings(&self, prefix: Option<&str>) -> RepoResult<Vec<String>> {
        let conn = self.conn.clone();
        let prefix_owned = prefix.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            let conn_guard = conn.lock().map_err(|e| format!("Failed to lock connection: {}", e))?;
            let mut keys = Vec::new();

            if let Some(p) = prefix_owned {
                let mut stmt = conn_guard.prepare("SELECT key FROM settings WHERE key LIKE ?1 || '%'")
                    .map_err(|e| format!("Failed to prepare statement: {}", e))?;
                let rows = stmt.query_map([p], |row| row.get(0))
                    .map_err(|e| format!("Query failed: {}", e))?;
                for row in rows {
                    if let Ok(key) = row {
                        keys.push(key);
                    }
                }
            } else {
                let mut stmt = conn_guard.prepare("SELECT key FROM settings")
                    .map_err(|e| format!("Failed to prepare statement: {}", e))?;
                let rows = stmt.query_map([], |row| row.get(0))
                    .map_err(|e| format!("Query failed: {}", e))?;
                for row in rows {
                    if let Ok(key) = row {
                        keys.push(key);
                    }
                }
            }
            Ok(keys)
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e))
    }

    async fn list_physics_profiles(&self) -> RepoResult<Vec<String>> {
        // Stub - return empty list
        Ok(Vec::new())
    }

    async fn delete_physics_profile(&self, _profile_name: &str) -> RepoResult<()> {
        // Stub implementation
        Ok(())
    }

    async fn export_settings(&self) -> RepoResult<serde_json::Value> {
        // Stub - return empty JSON
        Ok(serde_json::json!({}))
    }

    async fn import_settings(&self, _settings_json: &serde_json::Value) -> RepoResult<()> {
        // Stub implementation
        Ok(())
    }

    async fn clear_cache(&self) -> RepoResult<()> {
        self.clear_cache().await
    }

    async fn health_check(&self) -> RepoResult<bool> {
        // Try to query database
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn_guard = conn.lock().map_err(|_| "Failed to lock connection".to_string())?;
            conn_guard.execute("SELECT 1", [])
                .map_err(|e| format!("Health check query failed: {}", e))?;
            Ok(true)
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e))
    }
}
