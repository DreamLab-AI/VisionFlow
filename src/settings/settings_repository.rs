// src/settings/settings_repository.rs
//! Settings Repository - Database persistence layer

use anyhow::{Result, Context};
use std::sync::Arc;
use crate::config::{PhysicsSettings, RenderingSettings};
use crate::services::database_service::{DatabaseService, DatabaseTarget};
use super::models::{ConstraintSettings, AllSettings, SettingsProfile};

pub struct SettingsRepository {
    db: Arc<DatabaseService>,
}

impl SettingsRepository {
    pub fn new(db: Arc<DatabaseService>) -> Self {
        Self { db }
    }

    /// Save physics settings to database
    pub fn save_physics_settings(&self, settings: &PhysicsSettings) -> Result<()> {
        let settings_json = serde_json::to_string(settings)
            .context("Failed to serialize physics settings")?;

        self.db.execute(
            DatabaseTarget::Settings,
            "INSERT OR REPLACE INTO physics_settings
             (id, settings_json, updated_at)
             VALUES (1, ?, datetime('now'))",
            &[&settings_json as &dyn rusqlite::ToSql]
        )
        .context("Failed to save physics settings")?;

        Ok(())
    }

    /// Load physics settings from database
    pub fn load_physics_settings(&self) -> Result<PhysicsSettings> {
        let result = self.db.query_one(
            DatabaseTarget::Settings,
            "SELECT settings_json FROM physics_settings WHERE id = 1",
            &[],
            |row| {
                let settings_json: String = row.get(0)?;
                Ok(settings_json)
            }
        );

        match result {
            Ok(Some(settings_json)) => {
                serde_json::from_str(&settings_json)
                    .context("Failed to deserialize physics settings")
            }
            Ok(None) | Err(_) => Ok(PhysicsSettings::default()),
        }
    }

    /// Save constraint settings to database
    pub fn save_constraint_settings(&self, settings: &ConstraintSettings) -> Result<()> {
        let settings_json = serde_json::to_string(settings)
            .context("Failed to serialize constraint settings")?;

        self.db.execute(
            DatabaseTarget::Settings,
            "INSERT OR REPLACE INTO constraint_settings
             (id, settings_json, updated_at)
             VALUES (1, ?, datetime('now'))",
            &[&settings_json as &dyn rusqlite::ToSql]
        )
        .context("Failed to save constraint settings")?;

        Ok(())
    }

    /// Load constraint settings from database
    pub fn load_constraint_settings(&self) -> Result<ConstraintSettings> {
        let result = self.db.query_one(
            DatabaseTarget::Settings,
            "SELECT settings_json FROM constraint_settings WHERE id = 1",
            &[],
            |row| {
                let settings_json: String = row.get(0)?;
                Ok(settings_json)
            }
        );

        match result {
            Ok(Some(settings_json)) => {
                serde_json::from_str(&settings_json)
                    .context("Failed to deserialize constraint settings")
            }
            Ok(None) | Err(_) => Ok(ConstraintSettings::default()),
        }
    }

    /// Save rendering settings to database
    pub fn save_rendering_settings(&self, settings: &RenderingSettings) -> Result<()> {
        let settings_json = serde_json::to_string(settings)
            .context("Failed to serialize rendering settings")?;

        self.db.execute(
            DatabaseTarget::Settings,
            "INSERT OR REPLACE INTO rendering_settings
             (id, settings_json, updated_at)
             VALUES (1, ?, datetime('now'))",
            &[&settings_json as &dyn rusqlite::ToSql]
        )
        .context("Failed to save rendering settings")?;

        Ok(())
    }

    /// Load rendering settings from database
    pub fn load_rendering_settings(&self) -> Result<RenderingSettings> {
        let result = self.db.query_one(
            DatabaseTarget::Settings,
            "SELECT settings_json FROM rendering_settings WHERE id = 1",
            &[],
            |row| {
                let settings_json: String = row.get(0)?;
                Ok(settings_json)
            }
        );

        match result {
            Ok(Some(settings_json)) => {
                serde_json::from_str(&settings_json)
                    .context("Failed to deserialize rendering settings")
            }
            Ok(None) | Err(_) => Ok(RenderingSettings::default()),
        }
    }

    /// Save a settings profile (all settings combined)
    pub fn save_profile(&self, name: &str, settings: &AllSettings) -> Result<i64> {
        let physics_json = serde_json::to_string(&settings.physics)?;
        let constraints_json = serde_json::to_string(&settings.constraints)?;
        let rendering_json = serde_json::to_string(&settings.rendering)?;

        self.db.execute(
            DatabaseTarget::Settings,
            "INSERT INTO settings_profiles
             (name, physics_json, constraints_json, rendering_json, created_at, updated_at)
             VALUES (?, ?, ?, ?, datetime('now'), datetime('now'))",
            &[
                &name as &dyn rusqlite::ToSql,
                &physics_json as &dyn rusqlite::ToSql,
                &constraints_json as &dyn rusqlite::ToSql,
                &rendering_json as &dyn rusqlite::ToSql
            ]
        )
        .context("Failed to save settings profile")?;

        // Get the last insert ID
        let id = self.db.query_one(
            DatabaseTarget::Settings,
            "SELECT last_insert_rowid() as id",
            &[],
            |row| row.get::<_, i64>(0)
        )
        .context("Failed to get last insert ID")?
        .ok_or_else(|| anyhow::anyhow!("Failed to retrieve last insert ID"))?;

        Ok(id)
    }

    /// Load a settings profile by ID
    pub fn load_profile(&self, id: i64) -> Result<AllSettings> {
        let row = self.db.query_one(
            DatabaseTarget::Settings,
            "SELECT physics_json, constraints_json, rendering_json
             FROM settings_profiles WHERE id = ?",
            &[&id as &dyn rusqlite::ToSql],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?
                ))
            }
        )
        .context("Failed to load settings profile")?
        .ok_or_else(|| anyhow::anyhow!("Settings profile not found"))?;

        let physics: PhysicsSettings = serde_json::from_str(&row.0)?;
        let constraints: ConstraintSettings = serde_json::from_str(&row.1)?;
        let rendering: RenderingSettings = serde_json::from_str(&row.2)?;

        Ok(AllSettings {
            physics,
            constraints,
            rendering,
        })
    }

    /// List all settings profiles
    pub fn list_profiles(&self) -> Result<Vec<SettingsProfile>> {
        let rows = self.db.query_all(
            DatabaseTarget::Settings,
            "SELECT id, name, created_at, updated_at
             FROM settings_profiles
             ORDER BY updated_at DESC",
            &[],
            |row| {
                Ok(SettingsProfile {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                })
            }
        )
        .context("Failed to list settings profiles")?;

        Ok(rows)
    }

    /// Delete a settings profile
    pub fn delete_profile(&self, id: i64) -> Result<()> {
        self.db.execute(
            DatabaseTarget::Settings,
            "DELETE FROM settings_profiles WHERE id = ?",
            &[&id as &dyn rusqlite::ToSql]
        )
        .context("Failed to delete settings profile")?;

        Ok(())
    }

    /// Load all current settings
    pub fn load_all_settings(&self) -> Result<AllSettings> {
        let physics = self.load_physics_settings()?;
        let constraints = self.load_constraint_settings()?;
        let rendering = self.load_rendering_settings()?;

        Ok(AllSettings {
            physics,
            constraints,
            rendering,
        })
    }

    /// Save all current settings
    pub fn save_all_settings(&self, settings: &AllSettings) -> Result<()> {
        self.save_physics_settings(&settings.physics)?;
        self.save_constraint_settings(&settings.constraints)?;
        self.save_rendering_settings(&settings.rendering)?;
        Ok(())
    }
}
