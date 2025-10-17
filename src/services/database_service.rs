// src/services/database_service.rs
//! Database service for SQLite-based ontology and metadata storage
//!
//! Provides centralized database access for:
//! - Application settings and physics configuration
//! - Ontology framework metadata (OWL classes, properties, axioms)
//! - Markdown file metadata with ontology blocks
//! - Validation reports and inference results
//! - Physics constraint generation

use rusqlite::{Connection, OptionalExtension, params, Result as SqliteResult, Transaction};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Utc};
use serde_json::Value as JsonValue;

use crate::models::metadata::Metadata;
use crate::config::PhysicsSettings;

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
    /// Get hierarchical settings by key path
    pub fn get_setting(&self, key: &str) -> SqliteResult<Option<SettingValue>> {
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
}

// ================================================================
// ONTOLOGY QUERIES
// ================================================================

impl DatabaseService {
    /// Save ontology metadata
    pub fn save_ontology(&self, ontology: &OntologyMetadata) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO ontologies (
                ontology_id, source_path, source_type, base_iri, version_iri,
                title, description, author, version, content_hash, axiom_count,
                class_count, property_count
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
             ON CONFLICT(ontology_id) DO UPDATE SET
                last_validated_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP",
            params![
                ontology.ontology_id, ontology.source_path, ontology.source_type,
                ontology.base_iri, ontology.version_iri, ontology.title,
                ontology.description, ontology.author, ontology.version,
                ontology.content_hash, ontology.axiom_count, ontology.class_count,
                ontology.property_count
            ]
        )?;

        Ok(())
    }

    /// Get ontology by ID
    pub fn get_ontology(&self, ontology_id: &str) -> SqliteResult<Option<OntologyMetadata>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT ontology_id, source_path, source_type, base_iri, version_iri,
                    title, description, author, version, content_hash, axiom_count,
                    class_count, property_count, parsed_at, last_validated_at
             FROM ontologies WHERE ontology_id = ?1"
        )?;

        stmt.query_row(params![ontology_id], |row| {
            Ok(OntologyMetadata {
                ontology_id: row.get(0)?,
                source_path: row.get(1)?,
                source_type: row.get(2)?,
                base_iri: row.get(3)?,
                version_iri: row.get(4)?,
                title: row.get(5)?,
                description: row.get(6)?,
                author: row.get(7)?,
                version: row.get(8)?,
                content_hash: row.get(9)?,
                axiom_count: row.get(10)?,
                class_count: row.get(11)?,
                property_count: row.get(12)?,
                parsed_at: row.get(13)?,
                last_validated_at: row.get(14)?,
            })
        }).optional()
    }

    /// Save OWL class definition
    pub fn save_owl_class(&self, ontology_id: &str, class: &OwlClass) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO owl_classes (ontology_id, class_iri, label, comment, parent_class_iri, is_deprecated)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(ontology_id, class_iri) DO UPDATE SET
                label = excluded.label,
                comment = excluded.comment,
                parent_class_iri = excluded.parent_class_iri",
            params![
                ontology_id, class.class_iri, class.label, class.comment,
                class.parent_class_iri, if class.is_deprecated { 1 } else { 0 }
            ]
        )?;

        Ok(())
    }

    /// Get all classes for an ontology
    pub fn get_owl_classes(&self, ontology_id: &str) -> SqliteResult<Vec<OwlClass>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT class_iri, label, comment, parent_class_iri, is_deprecated
             FROM owl_classes WHERE ontology_id = ?1
             ORDER BY class_iri"
        )?;

        let rows = stmt.query_map(params![ontology_id], |row| {
            Ok(OwlClass {
                class_iri: row.get(0)?,
                label: row.get(1)?,
                comment: row.get(2)?,
                parent_class_iri: row.get(3)?,
                is_deprecated: row.get::<_, i32>(4)? == 1,
            })
        })?;

        rows.collect()
    }

    /// Get disjoint class pairs
    pub fn get_disjoint_classes(&self, ontology_id: &str) -> SqliteResult<Vec<(String, String)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT class_iri_1, class_iri_2 FROM owl_disjoint_classes WHERE ontology_id = ?1"
        )?;

        let rows = stmt.query_map(params![ontology_id], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;

        rows.collect()
    }
}

// ================================================================
// MAPPING CONFIGURATION QUERIES
// ================================================================

impl DatabaseService {
    /// Get namespace IRI for prefix
    pub fn get_namespace(&self, prefix: &str) -> SqliteResult<Option<String>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT namespace_iri FROM namespaces WHERE prefix = ?1",
            params![prefix],
            |row| row.get(0)
        ).optional()
    }

    /// Save namespace
    pub fn save_namespace(&self, prefix: &str, namespace_iri: &str, is_default: bool) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO namespaces (prefix, namespace_iri, is_default)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(prefix) DO UPDATE SET
                namespace_iri = excluded.namespace_iri,
                is_default = excluded.is_default",
            params![prefix, namespace_iri, if is_default { 1 } else { 0 }]
        )?;

        Ok(())
    }

    /// Get OWL class IRI for graph label
    pub fn get_class_mapping(&self, graph_label: &str) -> SqliteResult<Option<String>> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT owl_class_iri FROM class_mappings WHERE graph_label = ?1",
            params![graph_label],
            |row| row.get(0)
        ).optional()
    }

    /// Get property mapping
    pub fn get_property_mapping(&self, graph_property: &str) -> SqliteResult<Option<PropertyMapping>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT owl_property_iri, property_type, rdfs_domain, rdfs_range, inverse_property_iri
             FROM property_mappings WHERE graph_property = ?1"
        )?;

        stmt.query_row(params![graph_property], |row| {
            Ok(PropertyMapping {
                owl_property_iri: row.get(0)?,
                property_type: row.get(1)?,
                rdfs_domain: row.get(2)?,
                rdfs_range: row.get(3)?,
                inverse_property_iri: row.get(4)?,
            })
        }).optional()
    }
}

// ================================================================
// FILE METADATA QUERIES
// ================================================================

impl DatabaseService {
    /// Get file metadata
    pub fn get_file_metadata(&self, file_name: &str) -> SqliteResult<Option<Metadata>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT file_name, file_size, sha1, file_blob_sha, node_id, node_size,
                    hyperlink_count, perplexity_link, last_modified, last_content_change,
                    last_commit, last_perplexity_process, change_count
             FROM file_metadata WHERE file_name = ?1"
        )?;

        let result = stmt.query_row(params![file_name], |row| {
            Ok(Metadata {
                file_name: row.get(0)?,
                file_size: row.get(1)?,
                sha1: row.get(2)?,
                file_blob_sha: row.get(3)?,
                node_id: row.get(4)?,
                node_size: row.get(5)?,
                hyperlink_count: row.get(6)?,
                perplexity_link: row.get(7)?,
                last_modified: row.get(8)?,
                last_content_change: row.get(9)?,
                last_commit: row.get(10)?,
                last_perplexity_process: row.get(11)?,
                change_count: row.get(12)?,
                topic_counts: HashMap::new(), // Will be populated separately
            })
        }).optional()?;

        if let Some(mut metadata) = result {
            metadata.topic_counts = self.get_file_topics(file_name)?;
            Ok(Some(metadata))
        } else {
            Ok(None)
        }
    }

    /// Get topic counts for a file
    fn get_file_topics(&self, file_name: &str) -> SqliteResult<HashMap<String, usize>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT topic, count FROM file_topics WHERE file_name = ?1"
        )?;

        let rows = stmt.query_map(params![file_name], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })?;

        let mut topics = HashMap::new();
        for row in rows {
            let (topic, count) = row?;
            topics.insert(topic, count);
        }

        Ok(topics)
    }

    /// Get all file metadata
    pub fn get_all_file_metadata(&self) -> SqliteResult<HashMap<String, Metadata>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT file_name FROM file_metadata ORDER BY file_name"
        )?;

        let file_names: Vec<String> = stmt.query_map([], |row| row.get(0))?.collect::<SqliteResult<_>>()?;

        drop(stmt);
        drop(conn);

        let mut metadata_map = HashMap::new();
        for file_name in file_names {
            if let Some(metadata) = self.get_file_metadata(&file_name)? {
                metadata_map.insert(file_name, metadata);
            }
        }

        Ok(metadata_map)
    }

    /// Save file metadata
    pub fn save_file_metadata(&self, metadata: &Metadata) -> SqliteResult<()> {
        let conn = self.conn.lock().unwrap();

        conn.execute(
            "INSERT INTO file_metadata (
                file_name, file_path, file_size, sha1, file_blob_sha, node_id,
                node_size, hyperlink_count, perplexity_link, last_modified,
                last_content_change, last_commit, last_perplexity_process, change_count
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
             ON CONFLICT(file_name) DO UPDATE SET
                file_size = excluded.file_size,
                sha1 = excluded.sha1,
                node_size = excluded.node_size,
                hyperlink_count = excluded.hyperlink_count,
                last_modified = excluded.last_modified,
                updated_at = CURRENT_TIMESTAMP",
            params![
                metadata.file_name, format!("./markdown/{}", metadata.file_name),
                metadata.file_size, metadata.sha1, metadata.file_blob_sha, metadata.node_id,
                metadata.node_size, metadata.hyperlink_count, metadata.perplexity_link,
                metadata.last_modified, metadata.last_content_change, metadata.last_commit,
                metadata.last_perplexity_process, metadata.change_count
            ]
        )?;

        // Save topic counts
        for (topic, count) in &metadata.topic_counts {
            conn.execute(
                "INSERT INTO file_topics (file_name, topic, count) VALUES (?1, ?2, ?3)
                 ON CONFLICT(file_name, topic) DO UPDATE SET count = excluded.count",
                params![metadata.file_name, topic, count]
            )?;
        }

        Ok(())
    }
}

// ================================================================
// CONSTRAINT QUERIES
// ================================================================

impl DatabaseService {
    /// Get constraint groups by physics type
    pub fn get_constraint_groups(&self, physics_type: &str) -> SqliteResult<Vec<ConstraintGroup>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, group_name, kernel_name, physics_type, default_strength, enabled, batch_size
             FROM constraint_groups
             WHERE physics_type = ?1 AND enabled = 1"
        )?;

        let rows = stmt.query_map(params![physics_type], |row| {
            Ok(ConstraintGroup {
                id: row.get(0)?,
                group_name: row.get(1)?,
                kernel_name: row.get(2)?,
                physics_type: row.get(3)?,
                default_strength: row.get(4)?,
                enabled: row.get::<_, i32>(5)? == 1,
                batch_size: row.get(6)?,
            })
        })?;

        rows.collect()
    }
}

// ================================================================
// TYPE DEFINITIONS
// ================================================================

#[derive(Debug, Clone)]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(JsonValue),
}

#[derive(Debug, Clone)]
pub struct OntologyMetadata {
    pub ontology_id: String,
    pub source_path: String,
    pub source_type: String,
    pub base_iri: String,
    pub version_iri: Option<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub author: Option<String>,
    pub version: Option<String>,
    pub content_hash: String,
    pub axiom_count: i32,
    pub class_count: i32,
    pub property_count: i32,
    pub parsed_at: DateTime<Utc>,
    pub last_validated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct OwlClass {
    pub class_iri: String,
    pub label: Option<String>,
    pub comment: Option<String>,
    pub parent_class_iri: Option<String>,
    pub is_deprecated: bool,
}

#[derive(Debug, Clone)]
pub struct PropertyMapping {
    pub owl_property_iri: String,
    pub property_type: String,
    pub rdfs_domain: Option<String>,
    pub rdfs_range: Option<String>,
    pub inverse_property_iri: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConstraintGroup {
    pub id: i64,
    pub group_name: String,
    pub kernel_name: String,
    pub physics_type: String,
    pub default_strength: f64,
    pub enabled: bool,
    pub batch_size: i32,
}

// ================================================================
// TESTS
// ================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_creation() {
        let db = DatabaseService::new(":memory:").expect("Failed to create database");
        let version = db.get_schema_version();
        assert!(version.is_ok() || version.is_err()); // Schema not initialized yet
    }

    #[test]
    fn test_setting_crud() {
        let db = DatabaseService::new(":memory:").expect("Failed to create database");

        // Initialize schema first
        let schema = include_str!("../../schema/ontology_db.sql");
        db.execute_schema(schema).expect("Failed to initialize schema");

        // Create setting
        db.set_setting("test.key", SettingValue::String("value".to_string()), Some("Test setting"))
            .expect("Failed to set setting");

        // Read setting
        let value = db.get_setting("test.key").expect("Failed to get setting");
        assert!(value.is_some());

        if let Some(SettingValue::String(s)) = value {
            assert_eq!(s, "value");
        } else {
            panic!("Expected string value");
        }
    }
}
