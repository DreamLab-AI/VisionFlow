// src/services/database_service.rs
//! Database service for SQLite-based ontology and metadata storage
//!
//! Provides centralized database access for:
//! - Application settings and physics configuration
//! - Ontology framework metadata (OWL classes, properties, axioms)
//! - Markdown file metadata with ontology blocks
//! - Validation reports and inference results
//! - Physics constraint generation

use rusqlite::{Connection, OptionalExtension, params, Result as SqliteResult};
use std::path::Path;
use std::sync::{Arc, Mutex};
use serde_json::Value as JsonValue;

use crate::config::PhysicsSettings;

/// Setting value types for database storage
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(JsonValue),
}

/// Main database service providing thread-safe access to SQLite
pub struct DatabaseService {
    conn: Arc<Mutex<Connection>>,
}

impl DatabaseService {
    /// Create new database service, initializing schema if needed
    pub fn new<P: AsRef<Path>>(db_path: P) -> SqliteResult<Self> {
        let conn = Connection::open(db_path)?;

        // Configure SQLite for optimal performance
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "cache_size", 10000)?;
        conn.pragma_update(None, "foreign_keys", true)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Execute schema initialization from SQL file
    pub fn execute_schema(&self, schema_sql: &str) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(schema_sql)?;
        Ok(())
    }

    /// Get current schema version
    pub fn get_schema_version(&self) -> SqliteResult<i32> {
        let conn = self.conn.lock().unwrap();
        let version: i32 = conn.query_row(
            "SELECT version FROM schema_version WHERE id = 1",
            [],
            |row| row.get(0)
        )?;
        Ok(version)
    }

    /// Initialize database schema from embedded SQL file
    pub fn initialize_schema(&self) -> SqliteResult<()> {
        // Load schema from embedded SQL file
        const SCHEMA_SQL: &str = include_str!("../../schema/ontology_db.sql");
        self.execute_schema(SCHEMA_SQL)?;
        Ok(())
    }
}

// ================================================================
// SETTINGS QUERIES
// ================================================================

impl DatabaseService {
    /// Convert snake_case to camelCase
    /// Examples: "spring_k" -> "springK", "max_velocity" -> "maxVelocity"
    fn to_camel_case(s: &str) -> String {
        let parts: Vec<&str> = s.split('_').collect();
        if parts.len() == 1 {
            return s.to_string();
        }

        let mut result = parts[0].to_string();
        for part in &parts[1..] {
            if !part.is_empty() {
                let mut chars = part.chars();
                if let Some(first) = chars.next() {
                    result.push(first.to_ascii_uppercase());
                    result.push_str(chars.as_str());
                }
            }
        }
        result
    }

    /// Get setting value with exact key match (no fallback)
    fn get_setting_exact(&self, key: &str) -> SqliteResult<Option<SettingValue>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT value_type, value_text, value_integer, value_float, value_boolean, value_json
             FROM settings WHERE key = ?1"
        )?;

        stmt.query_row(params![key], |row| {
            let value_type: String = row.get(0)?;
            let value = match value_type.as_str() {
                "string" => SettingValue::String(row.get(1)?),
                "integer" => SettingValue::Integer(row.get(2)?),
                "float" => SettingValue::Float(row.get(3)?),
                "boolean" => SettingValue::Boolean(row.get::<_, i32>(4)? == 1),
                "json" => {
                    let json_str: String = row.get(5)?;
                    SettingValue::Json(serde_json::from_str(&json_str).unwrap_or(JsonValue::Null))
                },
                _ => SettingValue::String(String::new()),
            };
            Ok(value)
        }).optional()
    }

    /// Get hierarchical settings by key path with intelligent camelCase/snake_case fallback
    ///
    /// This method provides smart lookup:
    /// 1. First tries exact match with the provided key
    /// 2. If not found and key contains underscore, converts to camelCase and retries
    ///
    /// Examples:
    /// - Database has "springK" = 150.0
    /// - `get_setting("springK")` -> Direct hit, returns 150.0
    /// - `get_setting("spring_k")` -> Converts to "springK", returns 150.0
    pub fn get_setting(&self, key: &str) -> SqliteResult<Option<SettingValue>> {
        // Try exact match first
        if let Some(value) = self.get_setting_exact(key)? {
            return Ok(Some(value));
        }

        // If not found and key contains underscore, try camelCase conversion
        if key.contains('_') {
            let camel_key = Self::to_camel_case(key);
            if let Some(value) = self.get_setting_exact(&camel_key)? {
                return Ok(Some(value));
            }
        }

        // Not found with either key format
        Ok(None)
    }

    /// Set a setting value
    pub fn set_setting(&self, key: &str, value: SettingValue, description: Option<&str>) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        let (value_type, text, int, float, bool_val, json) = match value {
            SettingValue::String(s) => ("string", Some(s), None, None, None, None),
            SettingValue::Integer(i) => ("integer", None, Some(i), None, None, None),
            SettingValue::Float(f) => ("float", None, None, Some(f), None, None),
            SettingValue::Boolean(b) => ("boolean", None, None, None, Some(if b { 1 } else { 0 }), None),
            SettingValue::Json(j) => ("json", None, None, None, None, Some(j.to_string())),
        };

        conn.execute(
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
            params![key, value_type, text, int, float, bool_val, json, description]
        )?;

        Ok(())
    }

    /// Get physics settings for a specific profile
    pub fn get_physics_settings(&self, profile_name: &str) -> SqliteResult<PhysicsSettings> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT damping, dt, iterations, max_velocity, max_force, repel_k, spring_k,
                    mass_scale, boundary_damping, temperature, gravity, bounds_size, enable_bounds,
                    rest_length, repulsion_cutoff, repulsion_softening_epsilon, center_gravity_k,
                    grid_cell_size, warmup_iterations, cooling_rate, constraint_ramp_frames,
                    constraint_max_force_per_node
             FROM physics_settings WHERE profile_name = ?1"
        )?;

        stmt.query_row(params![profile_name], |row| {
            Ok(PhysicsSettings {
                damping: row.get(0)?,
                dt: row.get(1)?,
                iterations: row.get(2)?,
                max_velocity: row.get(3)?,
                max_force: row.get(4)?,
                repel_k: row.get(5)?,
                spring_k: row.get(6)?,
                mass_scale: row.get(7)?,
                boundary_damping: row.get(8)?,
                temperature: row.get(9)?,
                gravity: row.get(10)?,
                bounds_size: row.get(11)?,
                enable_bounds: row.get::<_, i32>(12)? == 1,
                rest_length: row.get(13)?,
                repulsion_cutoff: row.get(14)?,
                repulsion_softening_epsilon: row.get(15)?,
                center_gravity_k: row.get(16)?,
                grid_cell_size: row.get(17)?,
                warmup_iterations: row.get(18)?,
                cooling_rate: row.get(19)?,
                constraint_ramp_frames: row.get(20)?,
                constraint_max_force_per_node: row.get(21)?,
                ..PhysicsSettings::default()
            })
        })
    }

    /// Save physics settings profile
    pub fn save_physics_settings(&self, profile_name: &str, settings: &PhysicsSettings) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO physics_settings (
                profile_name, damping, dt, iterations, max_velocity, max_force,
                repel_k, spring_k, mass_scale, boundary_damping, temperature, gravity,
                bounds_size, enable_bounds, rest_length, repulsion_cutoff,
                repulsion_softening_epsilon, center_gravity_k, grid_cell_size,
                warmup_iterations, cooling_rate, constraint_ramp_frames,
                constraint_max_force_per_node
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)
             ON CONFLICT(profile_name) DO UPDATE SET
                damping = excluded.damping,
                dt = excluded.dt,
                iterations = excluded.iterations,
                max_velocity = excluded.max_velocity,
                max_force = excluded.max_force,
                repel_k = excluded.repel_k,
                spring_k = excluded.spring_k,
                updated_at = CURRENT_TIMESTAMP",
            params![
                profile_name, settings.damping, settings.dt, settings.iterations,
                settings.max_velocity, settings.max_force, settings.repel_k, settings.spring_k,
                settings.mass_scale, settings.boundary_damping, settings.temperature,
                settings.gravity, settings.bounds_size, if settings.enable_bounds { 1 } else { 0 },
                settings.rest_length, settings.repulsion_cutoff, settings.repulsion_softening_epsilon,
                settings.center_gravity_k, settings.grid_cell_size, settings.warmup_iterations,
                settings.cooling_rate, settings.constraint_ramp_frames, settings.constraint_max_force_per_node
            ]
        )?;

        Ok(())
    }

    /// Save complete settings to database as JSON
    /// Stores all settings categories (rendering, XR, system, auth, etc.)
    pub fn save_all_settings(&self, settings: &crate::config::AppFullSettings) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        // Serialize all settings as JSON
        let settings_json = serde_json::to_string(settings)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        // Save as a single JSON blob for simplicity and atomicity
        conn.execute(
            "INSERT INTO settings (key, value_type, value_json, description)
             VALUES ('app_full_settings', 'json', ?1, 'Complete application settings')
             ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP",
            params![settings_json]
        )?;

        // Also save physics settings to dedicated table for fast access
        // Inline the physics save to avoid deadlock from double-locking
        let physics = &settings.visualisation.graphs.logseq.physics;
        conn.execute(
            "INSERT INTO physics_settings (
                profile_name, damping, dt, iterations, max_velocity, max_force,
                repel_k, spring_k, mass_scale, boundary_damping, temperature, gravity,
                bounds_size, enable_bounds, rest_length, repulsion_cutoff,
                repulsion_softening_epsilon, center_gravity_k, grid_cell_size,
                warmup_iterations, cooling_rate, constraint_ramp_frames,
                constraint_max_force_per_node
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23)
             ON CONFLICT(profile_name) DO UPDATE SET
                damping = excluded.damping,
                dt = excluded.dt,
                iterations = excluded.iterations,
                max_velocity = excluded.max_velocity,
                max_force = excluded.max_force,
                repel_k = excluded.repel_k,
                spring_k = excluded.spring_k,
                updated_at = CURRENT_TIMESTAMP",
            params![
                "default", physics.damping, physics.dt, physics.iterations,
                physics.max_velocity, physics.max_force, physics.repel_k, physics.spring_k,
                physics.mass_scale, physics.boundary_damping, physics.temperature,
                physics.gravity, physics.bounds_size, if physics.enable_bounds { 1 } else { 0 },
                physics.rest_length, physics.repulsion_cutoff, physics.repulsion_softening_epsilon,
                physics.center_gravity_k, physics.grid_cell_size, physics.warmup_iterations,
                physics.cooling_rate, physics.constraint_ramp_frames, physics.constraint_max_force_per_node
            ]
        )?;

        Ok(())
    }

    /// Load complete settings from database
    /// Returns None if settings not found in database
    pub fn load_all_settings(&self) -> SqliteResult<Option<crate::config::AppFullSettings>> {
        let conn = self.conn.lock().unwrap();

        let settings_json: Option<String> = conn.query_row(
            "SELECT value_json FROM settings WHERE key = 'app_full_settings'",
            [],
            |row| row.get(0)
        ).optional()?;

        if let Some(json_str) = settings_json {
            let settings: crate::config::AppFullSettings = serde_json::from_str(&json_str)
                .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                    0,
                    rusqlite::types::Type::Text,
                    Box::new(e)
                ))?;
            Ok(Some(settings))
        } else {
            Ok(None)
        }
    }
}
