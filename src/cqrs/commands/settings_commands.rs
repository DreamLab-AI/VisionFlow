// src/cqrs/commands/settings_commands.rs
//! Settings Commands
//!
//! Write operations for application settings repository.

use crate::config::{AppFullSettings, PhysicsSettings};
use crate::cqrs::types::{Command, Result};
use crate::ports::settings_repository::SettingValue;
use std::collections::HashMap;

/// Update a single setting
#[derive(Debug, Clone)]
pub struct UpdateSettingCommand {
    pub key: String,
    pub value: SettingValue,
    pub description: Option<String>,
}

impl Command for UpdateSettingCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateSetting"
    }

    fn validate(&self) -> Result<()> {
        if self.key.is_empty() {
            return Err(anyhow::anyhow!("Setting key cannot be empty"));
        }
        Ok(())
    }
}

/// Update multiple settings atomically
#[derive(Debug, Clone)]
pub struct UpdateBatchSettingsCommand {
    pub updates: HashMap<String, SettingValue>,
}

impl Command for UpdateBatchSettingsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateBatchSettings"
    }

    fn validate(&self) -> Result<()> {
        if self.updates.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one setting update"));
        }
        for key in self.updates.keys() {
            if key.is_empty() {
                return Err(anyhow::anyhow!("Setting keys cannot be empty"));
            }
        }
        Ok(())
    }
}

/// Delete a setting
#[derive(Debug, Clone)]
pub struct DeleteSettingCommand {
    pub key: String,
}

impl Command for DeleteSettingCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "DeleteSetting"
    }

    fn validate(&self) -> Result<()> {
        if self.key.is_empty() {
            return Err(anyhow::anyhow!("Setting key cannot be empty"));
        }
        Ok(())
    }
}

/// Save complete application settings
#[derive(Debug, Clone)]
pub struct SaveAllSettingsCommand {
    pub settings: AppFullSettings,
}

impl Command for SaveAllSettingsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "SaveAllSettings"
    }
}

/// Save physics settings for a specific profile
#[derive(Debug, Clone)]
pub struct SavePhysicsSettingsCommand {
    pub profile_name: String,
    pub settings: PhysicsSettings,
}

impl Command for SavePhysicsSettingsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "SavePhysicsSettings"
    }

    fn validate(&self) -> Result<()> {
        if self.profile_name.is_empty() {
            return Err(anyhow::anyhow!("Profile name cannot be empty"));
        }
        // Validate physics settings
        if self.settings.time_step <= 0.0 {
            return Err(anyhow::anyhow!("Time step must be positive"));
        }
        if self.settings.damping < 0.0 || self.settings.damping > 1.0 {
            return Err(anyhow::anyhow!("Damping must be between 0 and 1"));
        }
        Ok(())
    }
}

/// Delete a physics profile
#[derive(Debug, Clone)]
pub struct DeletePhysicsProfileCommand {
    pub profile_name: String,
}

impl Command for DeletePhysicsProfileCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "DeletePhysicsProfile"
    }

    fn validate(&self) -> Result<()> {
        if self.profile_name.is_empty() {
            return Err(anyhow::anyhow!("Profile name cannot be empty"));
        }
        Ok(())
    }
}

/// Import settings from JSON
#[derive(Debug, Clone)]
pub struct ImportSettingsCommand {
    pub settings_json: serde_json::Value,
}

impl Command for ImportSettingsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "ImportSettings"
    }

    fn validate(&self) -> Result<()> {
        if !self.settings_json.is_object() {
            return Err(anyhow::anyhow!("Settings JSON must be an object"));
        }
        Ok(())
    }
}

/// Clear settings cache
#[derive(Debug, Clone)]
pub struct ClearSettingsCacheCommand;

impl Command for ClearSettingsCacheCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "ClearSettingsCache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_setting_validation() {
        let cmd = UpdateSettingCommand {
            key: "test.key".to_string(),
            value: SettingValue::String("value".to_string()),
            description: None,
        };
        assert!(cmd.validate().is_ok());

        let cmd = UpdateSettingCommand {
            key: "".to_string(),
            value: SettingValue::String("value".to_string()),
            description: None,
        };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_update_batch_validation() {
        let mut updates = HashMap::new();
        updates.insert("key1".to_string(), SettingValue::Boolean(true));
        let cmd = UpdateBatchSettingsCommand { updates };
        assert!(cmd.validate().is_ok());

        let cmd = UpdateBatchSettingsCommand {
            updates: HashMap::new(),
        };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_save_physics_validation() {
        let settings = PhysicsSettings::default();
        let cmd = SavePhysicsSettingsCommand {
            profile_name: "test".to_string(),
            settings,
        };
        assert!(cmd.validate().is_ok());

        let cmd = SavePhysicsSettingsCommand {
            profile_name: "".to_string(),
            settings: PhysicsSettings::default(),
        };
        assert!(cmd.validate().is_err());
    }
}
