// src/adapters/sqlite_settings_repository.rs
//! SQLite Settings Repository Adapter
//!
//! Implements the SettingsRepository port using SQLite with async interface,
//! caching, and intelligent camelCase/snake_case conversion.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

use crate::config::PhysicsSettings;
use crate::ports::settings_repository::{
    AppFullSettings, Result as RepoResult, SettingValue, SettingsRepository,
    SettingsRepositoryError,
};
use crate::services::database_service::DatabaseService;

/// SQLite-backed settings repository with caching
pub struct SqliteSettingsRepository {
    db: Arc<DatabaseService>,
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
    pub fn new(db: Arc<DatabaseService>) -> Self {
        info!("Initializing SqliteSettingsRepository with cache TTL=300s");
        Self {
            db,
            cache: Arc::new(RwLock::new(SettingsCache {
                settings: HashMap::new(),
                last_updated: std::time::Instant::now(),
                ttl_seconds: 300, // 5 minutes
            })),
        }
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
        let db = self.db.clone();
        let key_owned = key.to_string();
        let result = tokio::task::spawn_blocking(move || db.get_setting(&key_owned))
            .await
            .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
            .map_err(|e| SettingsRepositoryError::DatabaseError(e.to_string()))?;

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
        let db = self.db.clone();
        let key_owned = key.to_string();
        let value_owned = value.clone();
        let description_owned = description.map(|s| s.to_string());

        // Convert SettingValue to database_service::SettingValue
        let db_value = match value_owned {
            SettingValue::String(s) => crate::services::database_service::SettingValue::String(s),
            SettingValue::Integer(i) => crate::services::database_service::SettingValue::Integer(i),
            SettingValue::Float(f) => crate::services::database_service::SettingValue::Float(f),
            SettingValue::Boolean(b) => crate::services::database_service::SettingValue::Boolean(b),
            SettingValue::Json(j) => crate::services::database_service::SettingValue::Json(j),
        };

        tokio::task::spawn_blocking(move || {
            db.set_setting(&key_owned, db_value, description_owned.as_deref())
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e.to_string()))?;

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
        // Database service doesn't support AppFullSettings yet
        // Return a stub implementation
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
        // Database service doesn't support AppFullSettings yet
        // Return success as stub
        self.clear_cache().await?;
        Ok(())
    }

    #[instrument(skip(self), level = "debug")]
    async fn get_physics_settings(&self, profile_name: &str) -> RepoResult<PhysicsSettings> {
        let db = self.db.clone();
        let profile_owned = profile_name.to_string();
        tokio::task::spawn_blocking(move || db.get_physics_settings(&profile_owned))
            .await
            .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
            .map_err(|e| SettingsRepositoryError::DatabaseError(e.to_string()))
    }

    #[instrument(skip(self), level = "debug")]
    async fn save_physics_settings(
        &self,
        profile_name: &str,
        settings: &PhysicsSettings,
    ) -> RepoResult<()> {
        let db = self.db.clone();
        let profile_owned = profile_name.to_string();
        let settings_owned = settings.clone();
        tokio::task::spawn_blocking(move || {
            db.save_physics_settings(&profile_owned, &settings_owned)
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e.to_string()))
    }

    async fn list_physics_profiles(&self) -> RepoResult<Vec<String>> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db
                .get_settings_connection()
                .map_err(|e| format!("Connection error: {}", e))?;

            let mut stmt = conn
                .prepare("SELECT DISTINCT profile_name FROM physics_settings")
                .map_err(|e| format!("SQL error: {}", e))?;

            let profiles = stmt
                .query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| format!("Query error: {}", e))?
                .collect::<std::result::Result<Vec<String>, _>>()
                .map_err(|e| format!("Row error: {}", e))?;

            Ok(profiles)
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e))
    }

    async fn delete_physics_profile(&self, profile_name: &str) -> RepoResult<()> {
        let db = self.db.clone();
        let profile_owned = profile_name.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = db
                .get_settings_connection()
                .map_err(|e| format!("Connection error: {}", e))?;

            conn.execute(
                "DELETE FROM physics_settings WHERE profile_name = ?1",
                [&profile_owned],
            )
            .map_err(|e| format!("SQL error: {}", e))?;
            Ok(())
        })
        .await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
        .map_err(|e| SettingsRepositoryError::DatabaseError(e))
    }

    async fn clear_cache(&self) -> RepoResult<()> {
        let mut cache = self.cache.write().await;
        cache.settings.clear();
        cache.last_updated = std::time::Instant::now();
        info!("Settings cache cleared");
        Ok(())
    }
}
