// src/cqrs/queries/settings_queries.rs
//! Settings Queries
//!
//! Read operations for application settings repository.

use crate::config::{AppFullSettings, PhysicsSettings};
use crate::cqrs::types::{Query, Result};
use crate::ports::settings_repository::SettingValue;
use std::collections::HashMap;

/// Get a single setting by key
#[derive(Debug, Clone)]
pub struct GetSettingQuery {
    pub key: String,
}

impl Query for GetSettingQuery {
    type Result = Option<SettingValue>;

    fn name(&self) -> &'static str {
        "GetSetting"
    }

    fn validate(&self) -> Result<()> {
        if self.key.is_empty() {
            return Err(anyhow::anyhow!("Setting key cannot be empty"));
        }
        Ok(())
    }
}

/// Get multiple settings by keys
#[derive(Debug, Clone)]
pub struct GetBatchSettingsQuery {
    pub keys: Vec<String>,
}

impl Query for GetBatchSettingsQuery {
    type Result = HashMap<String, SettingValue>;

    fn name(&self) -> &'static str {
        "GetBatchSettings"
    }

    fn validate(&self) -> Result<()> {
        if self.keys.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one key"));
        }
        Ok(())
    }
}

/// Get all application settings
#[derive(Debug, Clone)]
pub struct GetAllSettingsQuery;

impl Query for GetAllSettingsQuery {
    type Result = Option<AppFullSettings>;

    fn name(&self) -> &'static str {
        "GetAllSettings"
    }
}

/// List all setting keys (optionally filtered by prefix)
#[derive(Debug, Clone)]
pub struct ListSettingsQuery {
    pub prefix: Option<String>,
}

impl Query for ListSettingsQuery {
    type Result = Vec<String>;

    fn name(&self) -> &'static str {
        "ListSettings"
    }
}

/// Check if a setting exists
#[derive(Debug, Clone)]
pub struct HasSettingQuery {
    pub key: String,
}

impl Query for HasSettingQuery {
    type Result = bool;

    fn name(&self) -> &'static str {
        "HasSetting"
    }

    fn validate(&self) -> Result<()> {
        if self.key.is_empty() {
            return Err(anyhow::anyhow!("Setting key cannot be empty"));
        }
        Ok(())
    }
}

/// Get physics settings for a specific profile
#[derive(Debug, Clone)]
pub struct GetPhysicsSettingsQuery {
    pub profile_name: String,
}

impl Query for GetPhysicsSettingsQuery {
    type Result = PhysicsSettings;

    fn name(&self) -> &'static str {
        "GetPhysicsSettings"
    }

    fn validate(&self) -> Result<()> {
        if self.profile_name.is_empty() {
            return Err(anyhow::anyhow!("Profile name cannot be empty"));
        }
        Ok(())
    }
}

/// List all available physics profiles
#[derive(Debug, Clone)]
pub struct ListPhysicsProfilesQuery;

impl Query for ListPhysicsProfilesQuery {
    type Result = Vec<String>;

    fn name(&self) -> &'static str {
        "ListPhysicsProfiles"
    }
}

/// Export settings to JSON
#[derive(Debug, Clone)]
pub struct ExportSettingsQuery;

impl Query for ExportSettingsQuery {
    type Result = serde_json::Value;

    fn name(&self) -> &'static str {
        "ExportSettings"
    }
}

/// Check settings repository health
#[derive(Debug, Clone)]
pub struct SettingsHealthCheckQuery;

impl Query for SettingsHealthCheckQuery {
    type Result = bool;

    fn name(&self) -> &'static str {
        "SettingsHealthCheck"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_setting_validation() {
        let query = GetSettingQuery {
            key: "test.key".to_string(),
        };
        assert!(query.validate().is_ok());

        let query = GetSettingQuery {
            key: "".to_string(),
        };
        assert!(query.validate().is_err());
    }

    #[test]
    fn test_get_batch_validation() {
        let query = GetBatchSettingsQuery {
            keys: vec!["key1".to_string()],
        };
        assert!(query.validate().is_ok());

        let query = GetBatchSettingsQuery { keys: vec![] };
        assert!(query.validate().is_err());
    }

    #[test]
    fn test_get_physics_validation() {
        let query = GetPhysicsSettingsQuery {
            profile_name: "default".to_string(),
        };
        assert!(query.validate().is_ok());

        let query = GetPhysicsSettingsQuery {
            profile_name: "".to_string(),
        };
        assert!(query.validate().is_err());
    }
}
