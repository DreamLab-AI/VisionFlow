// src/services/database_service.rs
//! Database service for SQLite-based storage across three separate databases
//!
//! Provides centralized database access for:
//! - Application settings and physics configuration (settings.db)
//! - Knowledge graph nodes, edges, and file metadata (knowledge_graph.db)
//! - Ontology framework metadata (OWL classes, properties, axioms) (ontology.db)

use log::info;
use r2d2::{Pool, PooledConnection};
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::{params, Connection, OptionalExtension, Result as SqliteResult};
use serde_json::Value as JsonValue;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::config::PhysicsSettings;

// Re-export SettingValue for backward compatibility
pub use crate::ports::settings_repository::SettingValue;

/// Health status for a single database
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatabaseHealth {
    pub name: String,
    pub is_connected: bool,
    pub pool_size: u32,
    pub idle_connections: u32,
    pub schema_version: Option<i32>,
    pub last_error: Option<String>,
}

/// Overall health status for all databases
#[derive(Debug, Clone, serde::Serialize)]
pub struct OverallDatabaseHealth {
    pub settings: DatabaseHealth,
    pub knowledge_graph: DatabaseHealth,
    pub ontology: DatabaseHealth,
    pub all_healthy: bool,
}

/// Main database service providing thread-safe access to three SQLite databases
pub struct DatabaseService {
    settings_pool: Pool<SqliteConnectionManager>,
    knowledge_graph_pool: Pool<SqliteConnectionManager>,
    ontology_pool: Pool<SqliteConnectionManager>,
    base_path: PathBuf,
}

impl DatabaseService {
    /// Create new database service with three separate databases
    pub fn new<P: AsRef<Path>>(base_path: P) -> SqliteResult<Self> {
        let base_path = base_path.as_ref().to_path_buf();

        // Ensure base directory exists
        if let Some(parent) = base_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
                    Some(format!("Failed to create directory: {}", e)),
                )
            })?;
        }

        info!("[DatabaseService] Initializing three-database architecture");
        info!("[DatabaseService] Base path: {}", base_path.display());

        // Create connection pools for each database
        let settings_path = base_path.with_file_name("settings.db");
        let knowledge_graph_path = base_path.with_file_name("knowledge_graph.db");
        let ontology_path = base_path.with_file_name("ontology.db");

        info!("[DatabaseService] Settings DB: {}", settings_path.display());
        info!(
            "[DatabaseService] Knowledge Graph DB: {}",
            knowledge_graph_path.display()
        );
        info!("[DatabaseService] Ontology DB: {}", ontology_path.display());

        let settings_pool = Self::create_pool(&settings_path)?;
        let knowledge_graph_pool = Self::create_pool(&knowledge_graph_path)?;
        let ontology_pool = Self::create_pool(&ontology_path)?;

        Ok(Self {
            settings_pool,
            knowledge_graph_pool,
            ontology_pool,
            base_path,
        })
    }

    /// Create a connection pool for a database
    fn create_pool(db_path: &Path) -> SqliteResult<Pool<SqliteConnectionManager>> {
        let manager = SqliteConnectionManager::file(db_path).with_init(|conn| {
            // Configure SQLite for optimal performance
            conn.pragma_update(None, "journal_mode", "WAL")?;
            conn.pragma_update(None, "synchronous", "NORMAL")?;
            conn.pragma_update(None, "cache_size", 10000)?;
            conn.pragma_update(None, "foreign_keys", true)?;
            conn.pragma_update(None, "temp_store", "MEMORY")?;
            Ok(())
        });

        Pool::builder()
            .max_size(10) // Increased pool size to prevent startup deadlocks
            .min_idle(Some(2))
            .connection_timeout(Duration::from_secs(10)) // Increased timeout for safety
            .build(manager)
            .map_err(|e| {
                rusqlite::Error::SqliteFailure(
                    rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                    Some(format!("Failed to create pool: {}", e)),
                )
            })
    }

    /// Get a connection to the settings database
    pub fn get_settings_connection(
        &self,
    ) -> Result<PooledConnection<SqliteConnectionManager>, String> {
        self.settings_pool
            .get()
            .map_err(|e| format!("Failed to get settings connection: {}", e))
    }

    /// Get a connection to the knowledge graph database
    pub fn get_knowledge_graph_connection(
        &self,
    ) -> Result<PooledConnection<SqliteConnectionManager>, String> {
        self.knowledge_graph_pool
            .get()
            .map_err(|e| format!("Failed to get knowledge graph connection: {}", e))
    }

    /// Get a connection to the ontology database
    pub fn get_ontology_connection(
        &self,
    ) -> Result<PooledConnection<SqliteConnectionManager>, String> {
        self.ontology_pool
            .get()
            .map_err(|e| format!("Failed to get ontology connection: {}", e))
    }

    /// Execute schema SQL on a specific database
    fn execute_schema(conn: &Connection, schema_sql: &str, db_name: &str) -> SqliteResult<()> {
        info!("[DatabaseService] Executing schema for {}", db_name);
        let start = Instant::now();

        conn.execute_batch(schema_sql)?;

        let duration = start.elapsed();
        info!(
            "[DatabaseService] Schema execution for {} completed in {:?}",
            db_name, duration
        );
        Ok(())
    }

    /// Initialize all database schemas from embedded SQL files
    pub fn initialize_schema(&self) -> SqliteResult<()> {
        info!("[DatabaseService] Initializing all database schemas");

        // Load schemas from embedded SQL files
        const SETTINGS_SCHEMA: &str = include_str!("../../schema/settings_db.sql");
        const KNOWLEDGE_GRAPH_SCHEMA: &str = include_str!("../../schema/knowledge_graph_db.sql");
        const ONTOLOGY_SCHEMA: &str = include_str!("../../schema/ontology_metadata_db.sql");

        // Initialize settings database
        let settings_conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;
        Self::execute_schema(&settings_conn, SETTINGS_SCHEMA, "settings")?;

        // Initialize knowledge graph database
        let kg_conn = self.get_knowledge_graph_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;
        Self::execute_schema(&kg_conn, KNOWLEDGE_GRAPH_SCHEMA, "knowledge_graph")?;

        // Initialize ontology database
        let ontology_conn = self.get_ontology_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;
        Self::execute_schema(&ontology_conn, ONTOLOGY_SCHEMA, "ontology")?;

        info!("[DatabaseService] All schemas initialized successfully");
        Ok(())
    }

    /// Migrate all databases (execute schema updates)
    pub fn migrate_all(&self) -> SqliteResult<()> {
        info!("[DatabaseService] Running migrations for all databases");

        // For now, migrations are handled by schema initialization
        // In the future, we can add migration logic here based on schema_version table

        let settings_version = self.get_schema_version("settings")?;
        let kg_version = self.get_schema_version("knowledge_graph")?;
        let ontology_version = self.get_schema_version("ontology")?;

        info!(
            "[DatabaseService] Current schema versions - Settings: {}, KG: {}, Ontology: {}",
            settings_version, kg_version, ontology_version
        );

        Ok(())
    }

    /// Get schema version for a specific database
    fn get_schema_version(&self, db_name: &str) -> SqliteResult<i32> {
        let conn = match db_name {
            "settings" => self.get_settings_connection(),
            "knowledge_graph" => self.get_knowledge_graph_connection(),
            "ontology" => self.get_ontology_connection(),
            _ => return Err(rusqlite::Error::InvalidQuery),
        }
        .map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        let version: i32 = conn.query_row(
            "SELECT version FROM schema_version WHERE id = 1",
            [],
            |row| row.get(0),
        )?;

        Ok(version)
    }

    /// Perform health check on all databases
    pub fn health_check(&self) -> Result<OverallDatabaseHealth, String> {
        let settings_health = self.check_database_health("settings", &self.settings_pool);
        let kg_health = self.check_database_health("knowledge_graph", &self.knowledge_graph_pool);
        let ontology_health = self.check_database_health("ontology", &self.ontology_pool);

        let all_healthy =
            settings_health.is_connected && kg_health.is_connected && ontology_health.is_connected;

        Ok(OverallDatabaseHealth {
            settings: settings_health,
            knowledge_graph: kg_health,
            ontology: ontology_health,
            all_healthy,
        })
    }

    /// Check health of a single database
    fn check_database_health(
        &self,
        name: &str,
        pool: &Pool<SqliteConnectionManager>,
    ) -> DatabaseHealth {
        let state = pool.state();

        let (is_connected, schema_version, last_error) = match pool.get() {
            Ok(conn) => {
                let version = conn
                    .query_row(
                        "SELECT version FROM schema_version WHERE id = 1",
                        [],
                        |row| row.get::<_, i32>(0),
                    )
                    .ok();
                (true, version, None)
            }
            Err(e) => (false, None, Some(e.to_string())),
        };

        DatabaseHealth {
            name: name.to_string(),
            is_connected,
            pool_size: state.connections,
            idle_connections: state.idle_connections,
            schema_version,
            last_error,
        }
    }

    /// Gracefully close all database connections
    pub fn close(&self) -> Result<(), String> {
        info!("[DatabaseService] Closing all database connections");

        // Connection pools will be automatically dropped and cleaned up
        // We just need to log the action

        info!("[DatabaseService] All database connections closed");
        Ok(())
    }
}

// ================================================================
// SETTINGS QUERIES (Settings Database)
// ================================================================

impl DatabaseService {
    /// Convert snake_case to camelCase
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
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        let mut stmt = conn.prepare(
            "SELECT value_type, value_text, value_integer, value_float, value_boolean, value_json
             FROM settings WHERE key = ?1",
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
                }
                _ => SettingValue::String(String::new()),
            };
            Ok(value)
        })
        .optional()
    }

    /// Get setting value with intelligent camelCase/snake_case fallback
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

        Ok(None)
    }

    /// Set a setting value
    pub fn set_setting(
        &self,
        key: &str,
        value: SettingValue,
        description: Option<&str>,
    ) -> SqliteResult<()> {
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        let (value_type, text, int, float, bool_val, json) = match value {
            SettingValue::String(s) => ("string", Some(s), None, None, None, None),
            SettingValue::Integer(i) => ("integer", None, Some(i), None, None, None),
            SettingValue::Float(f) => ("float", None, None, Some(f), None, None),
            SettingValue::Boolean(b) => (
                "boolean",
                None,
                None,
                None,
                Some(if b { 1 } else { 0 }),
                None,
            ),
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
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        let mut stmt = conn.prepare(
            "SELECT damping, dt, iterations, max_velocity, max_force, repel_k, spring_k,
                    mass_scale, boundary_damping, temperature, gravity, bounds_size, enable_bounds,
                    rest_length, repulsion_cutoff, repulsion_softening_epsilon, center_gravity_k,
                    grid_cell_size, warmup_iterations, cooling_rate, constraint_ramp_frames,
                    constraint_max_force_per_node
             FROM physics_settings WHERE profile_name = ?1",
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
    pub fn save_physics_settings(
        &self,
        profile_name: &str,
        settings: &PhysicsSettings,
    ) -> SqliteResult<()> {
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

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
    pub fn save_all_settings(&self, settings: &crate::config::AppFullSettings) -> SqliteResult<()> {
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        // Serialize all settings as JSON
        let settings_json = serde_json::to_string(settings)
            .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

        conn.execute(
            "INSERT INTO settings (key, value_type, value_json, description)
             VALUES ('app_full_settings', 'json', ?1, 'Complete application settings')
             ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP",
            params![settings_json],
        )?;

        // Also save physics settings to dedicated table
        let physics = &settings.visualisation.graphs.logseq.physics;
        self.save_physics_settings("default", physics)?;

        Ok(())
    }

    /// Load complete settings from database
    pub fn load_all_settings(&self) -> SqliteResult<Option<crate::config::AppFullSettings>> {
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        let settings_json: Option<String> = conn
            .query_row(
                "SELECT value_json FROM settings WHERE key = 'app_full_settings'",
                [],
                |row| row.get(0),
            )
            .optional()?;

        if let Some(json_str) = settings_json {
            let settings: crate::config::AppFullSettings = serde_json::from_str(&json_str)
                .map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        0,
                        rusqlite::types::Type::Text,
                        Box::new(e),
                    )
                })?;
            Ok(Some(settings))
        } else {
            Ok(None)
        }
    }

    /// Delete a setting by key
    pub fn delete_setting(&self, key: &str) -> SqliteResult<()> {
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        conn.execute("DELETE FROM settings WHERE key = ?1", params![key])?;
        Ok(())
    }

    /// List all setting keys, optionally filtered by prefix
    pub fn list_settings(&self, prefix: Option<&str>) -> SqliteResult<Vec<String>> {
        let conn = self.get_settings_connection().map_err(|e| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_ERROR),
                Some(e),
            )
        })?;

        let mut keys = Vec::new();

        if let Some(prefix_str) = prefix {
            let pattern = format!("{}%", prefix_str);
            let mut stmt =
                conn.prepare("SELECT key FROM settings WHERE key LIKE ?1 ORDER BY key")?;
            let key_iter = stmt.query_map(params![pattern], |row| row.get::<_, String>(0))?;

            for key_result in key_iter {
                keys.push(key_result?);
            }
        } else {
            let mut stmt = conn.prepare("SELECT key FROM settings ORDER BY key")?;
            let key_iter = stmt.query_map([], |row| row.get::<_, String>(0))?;

            for key_result in key_iter {
                keys.push(key_result?);
            }
        }

        Ok(keys)
    }
}
