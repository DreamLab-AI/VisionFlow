// src/ports/settings_repository.rs
//! Settings Repository Port
//!
//! Provides access to application, user, and developer configuration settings.
//! This port abstracts database operations for all settings management.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::config::PhysicsSettings;

pub type Result<T> = std::result::Result<T, SettingsRepositoryError>;

#[derive(Debug, thiserror::Error)]
pub enum SettingsRepositoryError {
    #[error("Setting not found: {0}")]
    NotFound(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Cache error: {0}")]
    CacheError(String),
}

/// Setting value that can be various types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(serde_json::Value),
}

impl SettingValue {
    pub fn as_string(&self) -> Option<&str> {
        match self {
            SettingValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            SettingValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            SettingValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SettingValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            SettingValue::Json(j) => Some(j),
            _ => None,
        }
    }
}

// Re-export AppFullSettings from config module (single source of truth)
pub use crate::config::AppFullSettings;

/// Port for settings repository operations
#[async_trait]
pub trait SettingsRepository: Send + Sync {
    /// Get a single setting by key (supports both camelCase and snake_case)
    async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>>;

    /// Set a single setting by key with optional description
    async fn set_setting(
        &self,
        key: &str,
        value: SettingValue,
        description: Option<&str>,
    ) -> Result<()>;

    /// Get batch of settings by keys
    async fn get_settings_batch(&self, keys: &[String]) -> Result<HashMap<String, SettingValue>>;

    /// Set batch of settings atomically
    async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> Result<()>;

    /// Load complete application settings
    async fn load_all_settings(&self) -> Result<Option<AppFullSettings>>;

    /// Save complete application settings
    async fn save_all_settings(&self, settings: &AppFullSettings) -> Result<()>;

    /// Get physics settings for a specific profile (e.g., "logseq", "ontology")
    async fn get_physics_settings(&self, profile_name: &str) -> Result<PhysicsSettings>;

    /// Save physics settings for a specific profile
    async fn save_physics_settings(
        &self,
        profile_name: &str,
        settings: &PhysicsSettings,
    ) -> Result<()>;

    /// List all available physics profiles
    async fn list_physics_profiles(&self) -> Result<Vec<String>>;

    /// Delete a physics profile
    async fn delete_physics_profile(&self, profile_name: &str) -> Result<()>;

    /// Clear cache (for implementations with caching)
    async fn clear_cache(&self) -> Result<()>;
}
