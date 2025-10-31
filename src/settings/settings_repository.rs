// src/settings/settings_repository.rs
//! Settings Repository - Database persistence layer

use anyhow::{Result, Context};
use std::sync::Arc;
use crate::config::{PhysicsSettings, RenderingSettings};
use crate::services::database_service::DatabaseService;
use super::models::{ConstraintSettings, AllSettings, SettingsProfile};

pub struct SettingsRepository {
    db: Arc<DatabaseService>,
}

impl SettingsRepository {
    pub fn new(db: Arc<DatabaseService>) -> Self {
        Self { db }
    }

    /// Save physics settings to database
    pub async fn save_physics_settings(&self, settings: &PhysicsSettings) -> Result<()> {
        let settings_json = serde_json::to_string(settings)
            .context("Failed to serialize physics settings")?;

        self.db.execute(
            "INSERT OR REPLACE INTO physics_settings
             (id, settings_json, updated_at)
             VALUES (1, ?, datetime('now'))",
            &[&settings_json]
        ).await
        .context("Failed to save physics settings")?;

        Ok(())
    }

    /// Load physics settings from database
    pub async fn load_physics_settings(&self) -> Result<PhysicsSettings> {
        let result = self.db.query_one(
            "SELECT settings_json FROM physics_settings WHERE id = 1",
            &[]
        ).await;

        match result {
            Ok(row) => {
                let settings_json: String = row.get("settings_json")
                    .context("Failed to get settings_json column")?;
                serde_json::from_str(&settings_json)
                    .context("Failed to deserialize physics settings")
            }
            Err(_) => Ok(PhysicsSettings::default()),
        }
    }

    /// Save constraint settings to database
    pub async fn save_constraint_settings(&self, settings: &ConstraintSettings) -> Result<()> {
        let settings_json = serde_json::to_string(settings)
            .context("Failed to serialize constraint settings")?;

        self.db.execute(
            "INSERT OR REPLACE INTO constraint_settings
             (id, settings_json, updated_at)
             VALUES (1, ?, datetime('now'))",
            &[&settings_json]
        ).await
        .context("Failed to save constraint settings")?;

        Ok(())
    }

    /// Load constraint settings from database
    pub async fn load_constraint_settings(&self) -> Result<ConstraintSettings> {
        let result = self.db.query_one(
            "SELECT settings_json FROM constraint_settings WHERE id = 1",
            &[]
        ).await;

        match result {
            Ok(row) => {
                let settings_json: String = row.get("settings_json")
                    .context("Failed to get settings_json column")?;
                serde_json::from_str(&settings_json)
                    .context("Failed to deserialize constraint settings")
            }
            Err(_) => Ok(ConstraintSettings::default()),
        }
    }

    /// Save rendering settings to database
    pub async fn save_rendering_settings(&self, settings: &RenderingSettings) -> Result<()> {
        let settings_json = serde_json::to_string(settings)
            .context("Failed to serialize rendering settings")?;

        self.db.execute(
            "INSERT OR REPLACE INTO rendering_settings
             (id, settings_json, updated_at)
             VALUES (1, ?, datetime('now'))",
            &[&settings_json]
        ).await
        .context("Failed to save rendering settings")?;

        Ok(())
    }

    /// Load rendering settings from database
    pub async fn load_rendering_settings(&self) -> Result<RenderingSettings> {
        let result = self.db.query_one(
            "SELECT settings_json FROM rendering_settings WHERE id = 1",
            &[]
        ).await;

        match result {
            Ok(row) => {
                let settings_json: String = row.get("settings_json")
                    .context("Failed to get settings_json column")?;
                serde_json::from_str(&settings_json)
                    .context("Failed to deserialize rendering settings")
            }
            Err(_) => Ok(RenderingSettings::default()),
        }
    }

    /// Save a settings profile (all settings combined)
    pub async fn save_profile(&self, name: &str, settings: &AllSettings) -> Result<i64> {
        let physics_json = serde_json::to_string(&settings.physics)?;
        let constraints_json = serde_json::to_string(&settings.constraints)?;
        let rendering_json = serde_json::to_string(&settings.rendering)?;

        self.db.execute(
            "INSERT INTO settings_profiles
             (name, physics_json, constraints_json, rendering_json, created_at, updated_at)
             VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
            &[name, &physics_json, &constraints_json, &rendering_json]
        ).await
        .context("Failed to save settings profile")?;

        // Get the last insert ID
        let row = self.db.query_one(
            "SELECT last_insert_rowid() as id",
            &[]
        ).await?;

        let id: i64 = row.get("id")?;
        Ok(id)
    }

    /// Load a settings profile by ID
    pub async fn load_profile(&self, id: i64) -> Result<AllSettings> {
        let row = self.db.query_one(
            "SELECT physics_json, constraints_json, rendering_json
             FROM settings_profiles WHERE id = ?",
            &[&id.to_string()]
        ).await
        .context("Failed to load settings profile")?;

        let physics: PhysicsSettings = serde_json::from_str(
            &row.get::<String>("physics_json")?
        )?;
        let constraints: ConstraintSettings = serde_json::from_str(
            &row.get::<String>("constraints_json")?
        )?;
        let rendering: RenderingSettings = serde_json::from_str(
            &row.get::<String>("rendering_json")?
        )?;

        Ok(AllSettings {
            physics,
            constraints,
            rendering,
        })
    }

    /// List all settings profiles
    pub async fn list_profiles(&self) -> Result<Vec<SettingsProfile>> {
        let rows = self.db.query_all(
            "SELECT id, name, created_at, updated_at
             FROM settings_profiles
             ORDER BY updated_at DESC",
            &[]
        ).await
        .context("Failed to list settings profiles")?;

        let profiles = rows.into_iter().map(|row| {
            Ok(SettingsProfile {
                id: row.get("id")?,
                name: row.get("name")?,
                created_at: row.get("created_at")?,
                updated_at: row.get("updated_at")?,
            })
        }).collect::<Result<Vec<_>>>()?;

        Ok(profiles)
    }

    /// Delete a settings profile
    pub async fn delete_profile(&self, id: i64) -> Result<()> {
        self.db.execute(
            "DELETE FROM settings_profiles WHERE id = ?",
            &[&id.to_string()]
        ).await
        .context("Failed to delete settings profile")?;

        Ok(())
    }

    /// Load all current settings
    pub async fn load_all_settings(&self) -> Result<AllSettings> {
        let physics = self.load_physics_settings().await?;
        let constraints = self.load_constraint_settings().await?;
        let rendering = self.load_rendering_settings().await?;

        Ok(AllSettings {
            physics,
            constraints,
            rendering,
        })
    }

    /// Save all current settings
    pub async fn save_all_settings(&self, settings: &AllSettings) -> Result<()> {
        self.save_physics_settings(&settings.physics).await?;
        self.save_constraint_settings(&settings.constraints).await?;
        self.save_rendering_settings(&settings.rendering).await?;
        Ok(())
    }
}
