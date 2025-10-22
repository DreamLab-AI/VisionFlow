// src/services/settings_service.rs
//! Settings service providing high-level API over DatabaseService
//!
//! This service:
//! - Provides async API for settings access
//! - Handles validation before database writes
//! - Manages in-memory cache with TTL
//! - Broadcasts change notifications to listeners
//! - Uses camelCase format (database handles snake_case fallback)

use log::{debug, error, info};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::{AppFullSettings, PhysicsSettings};
use crate::services::database_service::{DatabaseService, SettingValue};

#[derive(Clone)]
pub struct SettingsService {
    db: Arc<DatabaseService>,
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

type ChangeListener = Arc<dyn Fn(&str, &SettingValue) + Send + Sync>;

impl SettingsService {
    /// Create new settings service
    pub fn new(db: Arc<DatabaseService>) -> Result<Self, String> {
        Ok(Self {
            db,
            cache: Arc::new(RwLock::new(SettingsCache {
                settings: HashMap::new(),
                last_updated: std::time::Instant::now(),
            })),
            change_listeners: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Get setting by key (uses camelCase format, database handles snake_case fallback)
    pub async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>, String> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(cached) = cache.settings.get(key) {
                if cached.timestamp.elapsed().as_secs() < 300 {
                    // 5 min TTL
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
                cache.settings.insert(
                    key.to_string(),
                    CachedSetting {
                        value: value.clone(),
                        timestamp: std::time::Instant::now(),
                    },
                );
                Ok(Some(value))
            }
            Ok(None) => Ok(None),
            Err(e) => {
                error!("Database error getting setting {}: {}", key, e);
                Err(format!("Database error: {}", e))
            }
        }
    }

    /// Set setting value
    pub async fn set_setting(&self, key: &str, value: SettingValue) -> Result<(), String> {
        // Store in database (camelCase format)
        self.db
            .set_setting(key, value.clone(), None)
            .map_err(|e| format!("Database error: {}", e))?;

        // Invalidate cache
        {
            let mut cache = self.cache.write().await;
            cache.settings.remove(key);
        }

        // Notify listeners
        self.notify_change(key, &value).await;

        Ok(())
    }

    /// Get batch of settings by keys
    pub async fn get_settings_batch(
        &self,
        keys: &[String],
    ) -> Result<HashMap<String, SettingValue>, String> {
        let mut results = HashMap::new();

        for key in keys {
            if let Some(value) = self.get_setting(key).await? {
                results.insert(key.clone(), value);
            }
        }

        Ok(results)
    }

    /// Set batch of settings atomically
    pub async fn set_settings_batch(
        &self,
        updates: HashMap<String, SettingValue>,
    ) -> Result<(), String> {
        for (key, value) in updates {
            self.set_setting(&key, value).await?;
        }

        Ok(())
    }

    /// Load complete settings from database
    pub fn load_all_settings(&self) -> Result<Option<AppFullSettings>, String> {
        self.db
            .load_all_settings()
            .map_err(|e| format!("Database error: {}", e))
    }

    /// Save complete settings to database
    /// IMPORTANT: This preserves separate graph configurations (logseq, visionflow)
    /// DO NOT conflate graphs - each maintains its own settings
    pub fn save_all_settings(&self, settings: &AppFullSettings) -> Result<(), String> {
        self.db
            .save_all_settings(settings)
            .map_err(|e| format!("Database error: {}", e))
    }

    /// Get physics settings for a specific graph profile
    /// CRITICAL: Maintains separation between logseq and visionflow physics
    pub fn get_physics_settings(&self, graph_name: &str) -> Result<PhysicsSettings, String> {
        self.db
            .get_physics_settings(graph_name)
            .map_err(|e| format!("Database error: {}", e))
    }

    /// Save physics settings for a specific graph profile
    /// CRITICAL: Ensures graph-specific physics settings don't leak across graphs
    pub fn save_physics_settings(
        &self,
        graph_name: &str,
        settings: &PhysicsSettings,
    ) -> Result<(), String> {
        info!("Saving physics settings for graph: {}", graph_name);
        self.db
            .save_physics_settings(graph_name, settings)
            .map_err(|e| format!("Database error: {}", e))
    }

    /// Clear all cached settings
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.settings.clear();
        cache.last_updated = std::time::Instant::now();
        info!("Settings cache cleared");
    }

    /// Add change listener for settings updates
    pub async fn add_change_listener<F>(&self, listener: F)
    where
        F: Fn(&str, &SettingValue) + Send + Sync + 'static,
    {
        let mut listeners = self.change_listeners.write().await;
        listeners.push(Arc::new(listener));
    }

    /// Notify all listeners of a setting change
    async fn notify_change(&self, key: &str, value: &SettingValue) {
        let listeners = self.change_listeners.read().await;
        for listener in listeners.iter() {
            listener(key, value);
        }
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        CacheStats {
            entries: cache.settings.len(),
            last_updated: cache.last_updated.elapsed().as_secs(),
        }
    }
}

#[derive(Debug)]
pub struct CacheStats {
    pub entries: usize,
    pub last_updated: u64,
}
