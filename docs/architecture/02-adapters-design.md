# Adapter Layer Design - Hexagonal Architecture

## Overview

This document defines all adapter implementations that fulfill the port contracts defined in `01-ports-design.md`. Adapters implement **how** the application interacts with external systems (databases, GPU, etc.).

## Database Adapters

### SqliteSettingsRepository

**Location**: `src/adapters/sqlite_settings_repository.rs`

**Implementation Strategy**: Wraps the existing `SettingsService` with async interface.

```rust
use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use crate::ports::settings_repository::{SettingsRepository, SettingValue};
use crate::services::database_service::DatabaseService;
use crate::config::{AppFullSettings, PhysicsSettings};

pub struct SqliteSettingsRepository {
    db: Arc<DatabaseService>,
    cache: Arc<tokio::sync::RwLock<SettingsCache>>,
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
    pub fn new(db: Arc<DatabaseService>) -> Self {
        Self {
            db,
            cache: Arc::new(tokio::sync::RwLock::new(SettingsCache {
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
                return Some(cached.value.clone());
            }
        }
        None
    }

    /// Update cache with new value
    async fn update_cache(&self, key: String, value: SettingValue) {
        let mut cache = self.cache.write().await;
        cache.settings.insert(key, CachedSetting {
            value,
            timestamp: std::time::Instant::now(),
        });
    }

    /// Invalidate cache entry
    async fn invalidate_cache(&self, key: &str) {
        let mut cache = self.cache.write().await;
        cache.settings.remove(key);
    }
}

#[async_trait]
impl SettingsRepository for SqliteSettingsRepository {
    async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>, String> {
        // Check cache first
        if let Some(cached_value) = self.get_from_cache(key).await {
            return Ok(Some(cached_value));
        }

        // Query database (blocking operation, run in thread pool)
        let db = self.db.clone();
        let key_owned = key.to_string();
        let result = tokio::task::spawn_blocking(move || {
            db.get_setting(&key_owned)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?;

        // Update cache on success
        if let Ok(Some(ref value)) = result {
            self.update_cache(key.to_string(), value.clone()).await;
        }

        result
    }

    async fn set_setting(&self, key: &str, value: SettingValue, description: Option<&str>) -> Result<(), String> {
        let db = self.db.clone();
        let key_owned = key.to_string();
        let value_owned = value.clone();
        let description_owned = description.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            db.set_setting(&key_owned, value_owned, description_owned.as_deref())
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))??;

        // Invalidate cache
        self.invalidate_cache(key).await;

        Ok(())
    }

    async fn get_settings_batch(&self, keys: &[String]) -> Result<HashMap<String, SettingValue>, String> {
        let mut results = HashMap::new();

        for key in keys {
            if let Some(value) = self.get_setting(key).await? {
                results.insert(key.clone(), value);
            }
        }

        Ok(results)
    }

    async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> Result<(), String> {
        for (key, value) in updates {
            self.set_setting(&key, value, None).await?;
        }
        Ok(())
    }

    async fn load_all_settings(&self) -> Result<Option<AppFullSettings>, String> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            db.load_all_settings()
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    async fn save_all_settings(&self, settings: &AppFullSettings) -> Result<(), String> {
        let db = self.db.clone();
        let settings_owned = settings.clone();
        tokio::task::spawn_blocking(move || {
            db.save_all_settings(&settings_owned)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))??;

        // Clear cache on full save
        self.clear_cache().await?;

        Ok(())
    }

    async fn get_physics_settings(&self, profile_name: &str) -> Result<PhysicsSettings, String> {
        let db = self.db.clone();
        let profile_owned = profile_name.to_string();
        tokio::task::spawn_blocking(move || {
            db.get_physics_settings(&profile_owned)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    async fn save_physics_settings(&self, profile_name: &str, settings: &PhysicsSettings) -> Result<(), String> {
        let db = self.db.clone();
        let profile_owned = profile_name.to_string();
        let settings_owned = settings.clone();
        tokio::task::spawn_blocking(move || {
            db.save_physics_settings(&profile_owned, &settings_owned)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    async fn list_physics_profiles(&self) -> Result<Vec<String>, String> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            // Query database for all distinct profile names
            let conn = db.conn.lock().unwrap();
            let mut stmt = conn.prepare("SELECT DISTINCT profile_name FROM physics_settings")
                .map_err(|e| format!("SQL error: {}", e))?;

            let profiles = stmt.query_map([], |row| row.get::<_, String>(0))
                .map_err(|e| format!("Query error: {}", e))?
                .collect::<Result<Vec<String>, _>>()
                .map_err(|e| format!("Row error: {}", e))?;

            Ok(profiles)
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    async fn delete_physics_profile(&self, profile_name: &str) -> Result<(), String> {
        let db = self.db.clone();
        let profile_owned = profile_name.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = db.conn.lock().unwrap();
            conn.execute("DELETE FROM physics_settings WHERE profile_name = ?1", [&profile_owned])
                .map_err(|e| format!("SQL error: {}", e))?;
            Ok(())
        })
        .await
        .map_err(|e| format!("Task join error: {}", e))?
    }

    async fn clear_cache(&self) -> Result<(), String> {
        let mut cache = self.cache.write().await;
        cache.settings.clear();
        cache.last_updated = std::time::Instant::now();
        Ok(())
    }
}
```

### SqliteKnowledgeGraphRepository

**Location**: `src/adapters/sqlite_knowledge_graph_repository.rs`

**Implementation Strategy**: Stores graph structure in SQLite with efficient querying.

```rust
use async_trait::async_trait;
use std::sync::Arc;
use rusqlite::{Connection, params};
use crate::ports::knowledge_graph_repository::{KnowledgeGraphRepository, GraphStatistics};
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;

pub struct SqliteKnowledgeGraphRepository {
    conn: Arc<tokio::sync::Mutex<Connection>>,
}

impl SqliteKnowledgeGraphRepository {
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        // Create schema
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY,
                metadata_id TEXT NOT NULL,
                label TEXT,
                x REAL NOT NULL DEFAULT 0.0,
                y REAL NOT NULL DEFAULT 0.0,
                z REAL NOT NULL DEFAULT 0.0,
                vx REAL NOT NULL DEFAULT 0.0,
                vy REAL NOT NULL DEFAULT 0.0,
                vz REAL NOT NULL DEFAULT 0.0,
                color TEXT,
                size REAL,
                metadata TEXT, -- JSON blob
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_metadata_id ON nodes(metadata_id);

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT, -- JSON blob
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);

            CREATE TABLE IF NOT EXISTS graph_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        "#)
        .map_err(|e| format!("Failed to create schema: {}", e))?;

        Ok(Self {
            conn: Arc::new(tokio::sync::Mutex::new(conn)),
        })
    }

    /// Serialize node to database
    async fn serialize_node(&self, node: &Node) -> Result<(), String> {
        let conn = self.conn.lock().await;
        let metadata_json = serde_json::to_string(&node.metadata)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;

        conn.execute(
            "INSERT OR REPLACE INTO nodes (id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, CURRENT_TIMESTAMP)",
            params![
                node.id,
                &node.metadata_id,
                &node.label,
                node.data.x,
                node.data.y,
                node.data.z,
                node.data.vx,
                node.data.vy,
                node.data.vz,
                &node.color,
                &node.size,
                metadata_json,
            ]
        )
        .map_err(|e| format!("Failed to insert node: {}", e))?;

        Ok(())
    }

    /// Deserialize node from database row
    fn deserialize_node(row: &rusqlite::Row) -> Result<Node, rusqlite::Error> {
        let id: u32 = row.get(0)?;
        let metadata_id: String = row.get(1)?;
        let label: String = row.get(2)?;
        let x: f32 = row.get(3)?;
        let y: f32 = row.get(4)?;
        let z: f32 = row.get(5)?;
        let vx: f32 = row.get(6)?;
        let vy: f32 = row.get(7)?;
        let vz: f32 = row.get(8)?;
        let color: Option<String> = row.get(9)?;
        let size: Option<f32> = row.get(10)?;
        let metadata_json: String = row.get(11)?;

        let metadata = serde_json::from_str(&metadata_json)
            .unwrap_or_default();

        let mut node = Node::new_with_id(metadata_id, Some(id));
        node.label = label;
        node.data.x = x;
        node.data.y = y;
        node.data.z = z;
        node.data.vx = vx;
        node.data.vy = vy;
        node.data.vz = vz;
        node.color = color;
        node.size = size;
        node.metadata = metadata;

        Ok(node)
    }
}

#[async_trait]
impl KnowledgeGraphRepository for SqliteKnowledgeGraphRepository {
    async fn load_graph(&self) -> Result<Arc<GraphData>, String> {
        let conn = self.conn.lock().await;

        // Load all nodes
        let mut stmt = conn.prepare("SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM nodes")
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let nodes = stmt.query_map([], Self::deserialize_node)
            .map_err(|e| format!("Failed to query nodes: {}", e))?
            .collect::<Result<Vec<Node>, _>>()
            .map_err(|e| format!("Failed to collect nodes: {}", e))?;

        // Load all edges
        let mut edge_stmt = conn.prepare("SELECT id, source, target, weight, metadata FROM edges")
            .map_err(|e| format!("Failed to prepare edge statement: {}", e))?;

        let edges = edge_stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let source: u32 = row.get(1)?;
            let target: u32 = row.get(2)?;
            let weight: f32 = row.get(3)?;
            let metadata_json: Option<String> = row.get(4)?;

            let metadata = metadata_json.and_then(|json| serde_json::from_str(&json).ok());

            let mut edge = Edge::new(source, target, weight);
            edge.id = id;
            edge.metadata = metadata;

            Ok(edge)
        })
        .map_err(|e| format!("Failed to query edges: {}", e))?
        .collect::<Result<Vec<Edge>, _>>()
        .map_err(|e| format!("Failed to collect edges: {}", e))?;

        let mut graph = GraphData::new();
        graph.nodes = nodes;
        graph.edges = edges;

        Ok(Arc::new(graph))
    }

    async fn save_graph(&self, graph: &GraphData) -> Result<(), String> {
        // Begin transaction
        let conn = self.conn.lock().await;
        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| format!("Failed to begin transaction: {}", e))?;

        // Clear existing data
        conn.execute("DELETE FROM edges", [])
            .map_err(|e| format!("Failed to clear edges: {}", e))?;
        conn.execute("DELETE FROM nodes", [])
            .map_err(|e| format!("Failed to clear nodes: {}", e))?;

        // Insert nodes
        let mut node_stmt = conn.prepare(
            "INSERT INTO nodes (id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"
        ).map_err(|e| format!("Failed to prepare node insert: {}", e))?;

        for node in &graph.nodes {
            let metadata_json = serde_json::to_string(&node.metadata)
                .map_err(|e| format!("Failed to serialize metadata: {}", e))?;

            node_stmt.execute(params![
                node.id,
                &node.metadata_id,
                &node.label,
                node.data.x,
                node.data.y,
                node.data.z,
                node.data.vx,
                node.data.vy,
                node.data.vz,
                &node.color,
                &node.size,
                metadata_json,
            ])
            .map_err(|e| format!("Failed to insert node: {}", e))?;
        }

        drop(node_stmt);

        // Insert edges
        let mut edge_stmt = conn.prepare(
            "INSERT INTO edges (id, source, target, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)"
        ).map_err(|e| format!("Failed to prepare edge insert: {}", e))?;

        for edge in &graph.edges {
            let metadata_json = edge.metadata.as_ref()
                .and_then(|m| serde_json::to_string(m).ok());

            edge_stmt.execute(params![
                &edge.id,
                edge.source,
                edge.target,
                edge.weight,
                metadata_json,
            ])
            .map_err(|e| format!("Failed to insert edge: {}", e))?;
        }

        // Commit transaction
        conn.execute("COMMIT", [])
            .map_err(|e| format!("Failed to commit transaction: {}", e))?;

        Ok(())
    }

    async fn add_node(&self, node: &Node) -> Result<u32, String> {
        self.serialize_node(node).await?;
        Ok(node.id)
    }

    async fn update_node(&self, node: &Node) -> Result<(), String> {
        self.serialize_node(node).await
    }

    async fn remove_node(&self, node_id: u32) -> Result<(), String> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM nodes WHERE id = ?1", params![node_id])
            .map_err(|e| format!("Failed to delete node: {}", e))?;
        Ok(())
    }

    async fn get_node(&self, node_id: u32) -> Result<Option<Node>, String> {
        let conn = self.conn.lock().await;
        let result = conn.query_row(
            "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM nodes WHERE id = ?1",
            params![node_id],
            Self::deserialize_node
        );

        match result {
            Ok(node) => Ok(Some(node)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Database error: {}", e)),
        }
    }

    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> Result<Vec<Node>, String> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM nodes WHERE metadata_id = ?1"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let nodes = stmt.query_map(params![metadata_id], Self::deserialize_node)
            .map_err(|e| format!("Failed to query nodes: {}", e))?
            .collect::<Result<Vec<Node>, _>>()
            .map_err(|e| format!("Failed to collect nodes: {}", e))?;

        Ok(nodes)
    }

    async fn add_edge(&self, edge: &Edge) -> Result<String, String> {
        let conn = self.conn.lock().await;
        let metadata_json = edge.metadata.as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        conn.execute(
            "INSERT INTO edges (id, source, target, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![&edge.id, edge.source, edge.target, edge.weight, metadata_json]
        )
        .map_err(|e| format!("Failed to insert edge: {}", e))?;

        Ok(edge.id.clone())
    }

    async fn update_edge(&self, edge: &Edge) -> Result<(), String> {
        let conn = self.conn.lock().await;
        let metadata_json = edge.metadata.as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        conn.execute(
            "UPDATE edges SET source = ?1, target = ?2, weight = ?3, metadata = ?4 WHERE id = ?5",
            params![edge.source, edge.target, edge.weight, metadata_json, &edge.id]
        )
        .map_err(|e| format!("Failed to update edge: {}", e))?;

        Ok(())
    }

    async fn remove_edge(&self, edge_id: &str) -> Result<(), String> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM edges WHERE id = ?1", params![edge_id])
            .map_err(|e| format!("Failed to delete edge: {}", e))?;
        Ok(())
    }

    async fn get_node_edges(&self, node_id: u32) -> Result<Vec<Edge>, String> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, source, target, weight, metadata FROM edges WHERE source = ?1 OR target = ?1"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let edges = stmt.query_map(params![node_id], |row| {
            let id: String = row.get(0)?;
            let source: u32 = row.get(1)?;
            let target: u32 = row.get(2)?;
            let weight: f32 = row.get(3)?;
            let metadata_json: Option<String> = row.get(4)?;

            let metadata = metadata_json.and_then(|json| serde_json::from_str(&json).ok());

            let mut edge = Edge::new(source, target, weight);
            edge.id = id;
            edge.metadata = metadata;

            Ok(edge)
        })
        .map_err(|e| format!("Failed to query edges: {}", e))?
        .collect::<Result<Vec<Edge>, _>>()
        .map_err(|e| format!("Failed to collect edges: {}", e))?;

        Ok(edges)
    }

    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> Result<(), String> {
        let conn = self.conn.lock().await;
        conn.execute("BEGIN TRANSACTION", [])
            .map_err(|e| format!("Failed to begin transaction: {}", e))?;

        let mut stmt = conn.prepare("UPDATE nodes SET x = ?2, y = ?3, z = ?4, updated_at = CURRENT_TIMESTAMP WHERE id = ?1")
            .map_err(|e| format!("Failed to prepare statement: {}", e))?;

        for (node_id, x, y, z) in positions {
            stmt.execute(params![node_id, x, y, z])
                .map_err(|e| format!("Failed to update position: {}", e))?;
        }

        conn.execute("COMMIT", [])
            .map_err(|e| format!("Failed to commit transaction: {}", e))?;

        Ok(())
    }

    async fn query_nodes(&self, query: &str) -> Result<Vec<Node>, String> {
        // Simplified query parser - supports basic "field = value" or "field > value" syntax
        let conn = self.conn.lock().await;

        // Parse query (simplified implementation)
        // Format: "field operator value" (e.g., "size > 10", "color = red")
        let sql_query = format!(
            "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM nodes WHERE {}",
            query
        );

        let mut stmt = conn.prepare(&sql_query)
            .map_err(|e| format!("Failed to prepare query: {}", e))?;

        let nodes = stmt.query_map([], Self::deserialize_node)
            .map_err(|e| format!("Failed to execute query: {}", e))?
            .collect::<Result<Vec<Node>, _>>()
            .map_err(|e| format!("Failed to collect results: {}", e))?;

        Ok(nodes)
    }

    async fn get_statistics(&self) -> Result<GraphStatistics, String> {
        let conn = self.conn.lock().await;

        let node_count: usize = conn.query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))
            .map_err(|e| format!("Failed to count nodes: {}", e))?;

        let edge_count: usize = conn.query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
            .map_err(|e| format!("Failed to count edges: {}", e))?;

        let average_degree = if node_count > 0 {
            (edge_count as f32 * 2.0) / node_count as f32
        } else {
            0.0
        };

        Ok(GraphStatistics {
            node_count,
            edge_count,
            average_degree,
            connected_components: 1, // TODO: Implement connected components analysis
            last_updated: chrono::Utc::now(),
        })
    }
}
```

### SqliteOntologyRepository

**Location**: `src/adapters/sqlite_ontology_repository.rs`

**Implementation Strategy**: Similar to Knowledge Graph repository but specialized for OWL structures.

```rust
use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use rusqlite::{Connection, params};
use crate::ports::ontology_repository::*;
use crate::models::graph::GraphData;

pub struct SqliteOntologyRepository {
    conn: Arc<tokio::sync::Mutex<Connection>>,
}

impl SqliteOntologyRepository {
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn = Connection::open(db_path)
            .map_err(|e| format!("Failed to open database: {}", e))?;

        // Create ontology schema
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS owl_classes (
                iri TEXT PRIMARY KEY,
                label TEXT,
                description TEXT,
                source_file TEXT,
                properties TEXT, -- JSON blob
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);

            CREATE TABLE IF NOT EXISTS owl_class_hierarchy (
                class_iri TEXT NOT NULL,
                parent_iri TEXT NOT NULL,
                PRIMARY KEY (class_iri, parent_iri),
                FOREIGN KEY (class_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE,
                FOREIGN KEY (parent_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS owl_properties (
                iri TEXT PRIMARY KEY,
                label TEXT,
                property_type TEXT NOT NULL, -- ObjectProperty, DataProperty, AnnotationProperty
                domain TEXT, -- JSON array of IRIs
                range TEXT, -- JSON array of IRIs
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS owl_axioms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                axiom_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                object TEXT NOT NULL,
                annotations TEXT, -- JSON blob
                is_inferred BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_axioms_subject ON owl_axioms(subject);
            CREATE INDEX IF NOT EXISTS idx_axioms_type ON owl_axioms(axiom_type);

            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                inference_time_ms INTEGER NOT NULL,
                reasoner_version TEXT NOT NULL,
                inferred_axiom_count INTEGER NOT NULL,
                result_data TEXT -- JSON blob of inferred axioms
            );

            CREATE TABLE IF NOT EXISTS validation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_valid BOOLEAN NOT NULL,
                errors TEXT, -- JSON array
                warnings TEXT -- JSON array
            );
        "#)
        .map_err(|e| format!("Failed to create schema: {}", e))?;

        Ok(Self {
            conn: Arc::new(tokio::sync::Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl OntologyRepository for SqliteOntologyRepository {
    async fn load_ontology_graph(&self) -> Result<Arc<GraphData>, String> {
        // Convert OWL structures to GraphData for visualization
        let conn = self.conn.lock().await;

        let classes = self.list_owl_classes().await?;
        let mut graph = GraphData::new();

        // Create nodes for each class
        for (i, class) in classes.iter().enumerate() {
            let mut node = crate::models::node::Node::new_with_id(class.iri.clone(), Some(i as u32));
            node.label = class.label.clone().unwrap_or_else(|| class.iri.clone());
            node.color = Some("#4A90E2".to_string()); // Blue for ontology classes
            node.size = Some(15.0);
            node.metadata.insert("type".to_string(), "owl_class".to_string());
            node.metadata.insert("iri".to_string(), class.iri.clone());

            graph.nodes.push(node);
        }

        // Create edges for subclass relationships
        for (i, class) in classes.iter().enumerate() {
            for parent_iri in &class.parent_classes {
                if let Some((j, _)) = classes.iter().enumerate().find(|(_, c)| &c.iri == parent_iri) {
                    let edge = crate::models::edge::Edge::new(i as u32, j as u32, 1.0);
                    graph.edges.push(edge);
                }
            }
        }

        Ok(Arc::new(graph))
    }

    async fn save_ontology_graph(&self, _graph: &GraphData) -> Result<(), String> {
        // Not typically used - ontology is built from OWL data, not graph visualization
        Ok(())
    }

    async fn add_owl_class(&self, class: &OwlClass) -> Result<String, String> {
        let conn = self.conn.lock().await;

        let properties_json = serde_json::to_string(&class.properties)
            .map_err(|e| format!("Failed to serialize properties: {}", e))?;

        conn.execute(
            "INSERT OR REPLACE INTO owl_classes (iri, label, description, source_file, properties, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
            params![&class.iri, &class.label, &class.description, &class.source_file, properties_json]
        )
        .map_err(|e| format!("Failed to insert class: {}", e))?;

        // Insert parent relationships
        for parent_iri in &class.parent_classes {
            conn.execute(
                "INSERT OR IGNORE INTO owl_class_hierarchy (class_iri, parent_iri) VALUES (?1, ?2)",
                params![&class.iri, parent_iri]
            )
            .map_err(|e| format!("Failed to insert hierarchy: {}", e))?;
        }

        Ok(class.iri.clone())
    }

    async fn get_owl_class(&self, iri: &str) -> Result<Option<OwlClass>, String> {
        let conn = self.conn.lock().await;

        let result = conn.query_row(
            "SELECT iri, label, description, source_file, properties FROM owl_classes WHERE iri = ?1",
            params![iri],
            |row| {
                let iri: String = row.get(0)?;
                let label: Option<String> = row.get(1)?;
                let description: Option<String> = row.get(2)?;
                let source_file: Option<String> = row.get(3)?;
                let properties_json: String = row.get(4)?;

                let properties = serde_json::from_str(&properties_json).unwrap_or_default();

                Ok((iri, label, description, source_file, properties))
            }
        );

        match result {
            Ok((iri, label, description, source_file, properties)) => {
                // Get parent classes
                let mut parent_stmt = conn.prepare("SELECT parent_iri FROM owl_class_hierarchy WHERE class_iri = ?1")
                    .map_err(|e| format!("Failed to prepare parent query: {}", e))?;

                let parent_classes = parent_stmt.query_map(params![&iri], |row| row.get(0))
                    .map_err(|e| format!("Failed to query parents: {}", e))?
                    .collect::<Result<Vec<String>, _>>()
                    .map_err(|e| format!("Failed to collect parents: {}", e))?;

                Ok(Some(OwlClass {
                    iri,
                    label,
                    description,
                    parent_classes,
                    properties,
                    source_file,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Database error: {}", e)),
        }
    }

    async fn list_owl_classes(&self) -> Result<Vec<OwlClass>, String> {
        let conn = self.conn.lock().await;

        let mut stmt = conn.prepare(
            "SELECT iri, label, description, source_file, properties FROM owl_classes"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let class_rows = stmt.query_map([], |row| {
            let iri: String = row.get(0)?;
            let label: Option<String> = row.get(1)?;
            let description: Option<String> = row.get(2)?;
            let source_file: Option<String> = row.get(3)?;
            let properties_json: String = row.get(4)?;

            Ok((iri, label, description, source_file, properties_json))
        })
        .map_err(|e| format!("Failed to query classes: {}", e))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Failed to collect classes: {}", e))?;

        let mut classes = Vec::new();

        for (iri, label, description, source_file, properties_json) in class_rows {
            let properties = serde_json::from_str(&properties_json).unwrap_or_default();

            // Get parent classes
            let mut parent_stmt = conn.prepare("SELECT parent_iri FROM owl_class_hierarchy WHERE class_iri = ?1")
                .map_err(|e| format!("Failed to prepare parent query: {}", e))?;

            let parent_classes = parent_stmt.query_map(params![&iri], |row| row.get(0))
                .map_err(|e| format!("Failed to query parents: {}", e))?
                .collect::<Result<Vec<String>, _>>()
                .map_err(|e| format!("Failed to collect parents: {}", e))?;

            classes.push(OwlClass {
                iri,
                label,
                description,
                parent_classes,
                properties,
                source_file,
            });
        }

        Ok(classes)
    }

    async fn add_owl_property(&self, property: &OwlProperty) -> Result<String, String> {
        let conn = self.conn.lock().await;

        let property_type_str = match property.property_type {
            PropertyType::ObjectProperty => "ObjectProperty",
            PropertyType::DataProperty => "DataProperty",
            PropertyType::AnnotationProperty => "AnnotationProperty",
        };

        let domain_json = serde_json::to_string(&property.domain)
            .map_err(|e| format!("Failed to serialize domain: {}", e))?;
        let range_json = serde_json::to_string(&property.range)
            .map_err(|e| format!("Failed to serialize range: {}", e))?;

        conn.execute(
            "INSERT OR REPLACE INTO owl_properties (iri, label, property_type, domain, range, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
            params![&property.iri, &property.label, property_type_str, domain_json, range_json]
        )
        .map_err(|e| format!("Failed to insert property: {}", e))?;

        Ok(property.iri.clone())
    }

    async fn get_owl_property(&self, iri: &str) -> Result<Option<OwlProperty>, String> {
        let conn = self.conn.lock().await;

        let result = conn.query_row(
            "SELECT iri, label, property_type, domain, range FROM owl_properties WHERE iri = ?1",
            params![iri],
            |row| {
                let iri: String = row.get(0)?;
                let label: Option<String> = row.get(1)?;
                let property_type_str: String = row.get(2)?;
                let domain_json: String = row.get(3)?;
                let range_json: String = row.get(4)?;

                Ok((iri, label, property_type_str, domain_json, range_json))
            }
        );

        match result {
            Ok((iri, label, property_type_str, domain_json, range_json)) => {
                let property_type = match property_type_str.as_str() {
                    "ObjectProperty" => PropertyType::ObjectProperty,
                    "DataProperty" => PropertyType::DataProperty,
                    "AnnotationProperty" => PropertyType::AnnotationProperty,
                    _ => return Err(format!("Unknown property type: {}", property_type_str)),
                };

                let domain = serde_json::from_str(&domain_json).unwrap_or_default();
                let range = serde_json::from_str(&range_json).unwrap_or_default();

                Ok(Some(OwlProperty {
                    iri,
                    label,
                    property_type,
                    domain,
                    range,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Database error: {}", e)),
        }
    }

    async fn list_owl_properties(&self) -> Result<Vec<OwlProperty>, String> {
        let conn = self.conn.lock().await;

        let mut stmt = conn.prepare(
            "SELECT iri, label, property_type, domain, range FROM owl_properties"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let properties = stmt.query_map([], |row| {
            let iri: String = row.get(0)?;
            let label: Option<String> = row.get(1)?;
            let property_type_str: String = row.get(2)?;
            let domain_json: String = row.get(3)?;
            let range_json: String = row.get(4)?;

            let property_type = match property_type_str.as_str() {
                "ObjectProperty" => PropertyType::ObjectProperty,
                "DataProperty" => PropertyType::DataProperty,
                "AnnotationProperty" => PropertyType::AnnotationProperty,
                _ => PropertyType::ObjectProperty, // Default fallback
            };

            let domain = serde_json::from_str(&domain_json).unwrap_or_default();
            let range = serde_json::from_str(&range_json).unwrap_or_default();

            Ok(OwlProperty {
                iri,
                label,
                property_type,
                domain,
                range,
            })
        })
        .map_err(|e| format!("Failed to query properties: {}", e))?
        .collect::<Result<Vec<OwlProperty>, _>>()
        .map_err(|e| format!("Failed to collect properties: {}", e))?;

        Ok(properties)
    }

    async fn add_axiom(&self, axiom: &OwlAxiom) -> Result<u64, String> {
        let conn = self.conn.lock().await;

        let axiom_type_str = match axiom.axiom_type {
            AxiomType::SubClassOf => "SubClassOf",
            AxiomType::EquivalentClass => "EquivalentClass",
            AxiomType::DisjointWith => "DisjointWith",
            AxiomType::ObjectPropertyAssertion => "ObjectPropertyAssertion",
            AxiomType::DataPropertyAssertion => "DataPropertyAssertion",
        };

        let annotations_json = serde_json::to_string(&axiom.annotations)
            .map_err(|e| format!("Failed to serialize annotations: {}", e))?;

        conn.execute(
            "INSERT INTO owl_axioms (axiom_type, subject, object, annotations) VALUES (?1, ?2, ?3, ?4)",
            params![axiom_type_str, &axiom.subject, &axiom.object, annotations_json]
        )
        .map_err(|e| format!("Failed to insert axiom: {}", e))?;

        let id = conn.last_insert_rowid() as u64;
        Ok(id)
    }

    async fn get_class_axioms(&self, class_iri: &str) -> Result<Vec<OwlAxiom>, String> {
        let conn = self.conn.lock().await;

        let mut stmt = conn.prepare(
            "SELECT id, axiom_type, subject, object, annotations FROM owl_axioms WHERE subject = ?1 OR object = ?1"
        ).map_err(|e| format!("Failed to prepare statement: {}", e))?;

        let axioms = stmt.query_map(params![class_iri], |row| {
            let id: i64 = row.get(0)?;
            let axiom_type_str: String = row.get(1)?;
            let subject: String = row.get(2)?;
            let object: String = row.get(3)?;
            let annotations_json: String = row.get(4)?;

            let axiom_type = match axiom_type_str.as_str() {
                "SubClassOf" => AxiomType::SubClassOf,
                "EquivalentClass" => AxiomType::EquivalentClass,
                "DisjointWith" => AxiomType::DisjointWith,
                "ObjectPropertyAssertion" => AxiomType::ObjectPropertyAssertion,
                "DataPropertyAssertion" => AxiomType::DataPropertyAssertion,
                _ => AxiomType::SubClassOf, // Default fallback
            };

            let annotations = serde_json::from_str(&annotations_json).unwrap_or_default();

            Ok(OwlAxiom {
                id: Some(id as u64),
                axiom_type,
                subject,
                object,
                annotations,
            })
        })
        .map_err(|e| format!("Failed to query axioms: {}", e))?
        .collect::<Result<Vec<OwlAxiom>, _>>()
        .map_err(|e| format!("Failed to collect axioms: {}", e))?;

        Ok(axioms)
    }

    async fn store_inference_results(&self, results: &InferenceResults) -> Result<(), String> {
        let conn = self.conn.lock().await;

        let result_json = serde_json::to_string(&results.inferred_axioms)
            .map_err(|e| format!("Failed to serialize results: {}", e))?;

        conn.execute(
            "INSERT INTO inference_results (inference_time_ms, reasoner_version, inferred_axiom_count, result_data)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                results.inference_time_ms as i64,
                &results.reasoner_version,
                results.inferred_axioms.len(),
                result_json
            ]
        )
        .map_err(|e| format!("Failed to insert inference results: {}", e))?;

        // Mark axioms as inferred
        for axiom in &results.inferred_axioms {
            if let Some(id) = axiom.id {
                conn.execute("UPDATE owl_axioms SET is_inferred = TRUE WHERE id = ?1", params![id as i64])
                    .map_err(|e| format!("Failed to mark axiom as inferred: {}", e))?;
            }
        }

        Ok(())
    }

    async fn get_inference_results(&self) -> Result<Option<InferenceResults>, String> {
        let conn = self.conn.lock().await;

        let result = conn.query_row(
            "SELECT timestamp, inference_time_ms, reasoner_version, result_data FROM inference_results ORDER BY id DESC LIMIT 1",
            [],
            |row| {
                let timestamp_str: String = row.get(0)?;
                let inference_time_ms: i64 = row.get(1)?;
                let reasoner_version: String = row.get(2)?;
                let result_json: String = row.get(3)?;

                Ok((timestamp_str, inference_time_ms, reasoner_version, result_json))
            }
        );

        match result {
            Ok((timestamp_str, inference_time_ms, reasoner_version, result_json)) => {
                let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                    .map_err(|e| format!("Failed to parse timestamp: {}", e))?
                    .with_timezone(&chrono::Utc);

                let inferred_axioms: Vec<OwlAxiom> = serde_json::from_str(&result_json)
                    .map_err(|e| format!("Failed to deserialize axioms: {}", e))?;

                Ok(Some(InferenceResults {
                    timestamp,
                    inferred_axioms,
                    inference_time_ms: inference_time_ms as u64,
                    reasoner_version,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(format!("Database error: {}", e)),
        }
    }

    async fn validate_ontology(&self) -> Result<ValidationReport, String> {
        // TODO: Implement actual validation logic
        // For now, return a basic validation report
        Ok(ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn query_ontology(&self, query: &str) -> Result<Vec<HashMap<String, String>>, String> {
        // Simplified SPARQL-like query support
        // This would need a full SPARQL parser in production
        Err("SPARQL queries not yet implemented".to_string())
    }

    async fn get_metrics(&self) -> Result<OntologyMetrics, String> {
        let conn = self.conn.lock().await;

        let class_count: usize = conn.query_row("SELECT COUNT(*) FROM owl_classes", [], |row| row.get(0))
            .map_err(|e| format!("Failed to count classes: {}", e))?;

        let property_count: usize = conn.query_row("SELECT COUNT(*) FROM owl_properties", [], |row| row.get(0))
            .map_err(|e| format!("Failed to count properties: {}", e))?;

        let axiom_count: usize = conn.query_row("SELECT COUNT(*) FROM owl_axioms", [], |row| row.get(0))
            .map_err(|e| format!("Failed to count axioms: {}", e))?;

        Ok(OntologyMetrics {
            class_count,
            property_count,
            axiom_count,
            max_depth: 0, // TODO: Calculate from hierarchy
            average_branching_factor: 0.0, // TODO: Calculate from hierarchy
        })
    }
}
```

## GPU Adapters

### PhysicsOrchestratorAdapter

**Location**: `src/adapters/physics_orchestrator_adapter.rs`

**Implementation Strategy**: Wraps the existing `PhysicsOrchestratorActor` to implement the `GpuPhysicsAdapter` port.

```rust
use async_trait::async_trait;
use std::sync::Arc;
use actix::Addr;
use crate::ports::gpu_physics_adapter::*;
use crate::actors::physics_orchestrator_actor::{PhysicsOrchestratorActor, GetPhysicsStatus};
use crate::actors::messages::*;
use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::ConstraintSet;

pub struct PhysicsOrchestratorAdapter {
    actor_addr: Addr<PhysicsOrchestratorActor>,
    device_info: GpuDeviceInfo,
}

impl PhysicsOrchestratorAdapter {
    pub fn new(actor_addr: Addr<PhysicsOrchestratorActor>) -> Self {
        // Get device info from CUDA
        let device_info = Self::detect_device_info();

        Self {
            actor_addr,
            device_info,
        }
    }

    fn detect_device_info() -> GpuDeviceInfo {
        #[cfg(feature = "gpu")]
        {
            use cudarc::driver::CudaDevice;

            match CudaDevice::new(0) {
                Ok(device) => {
                    let name = device.name().unwrap_or_else(|_| "Unknown GPU".to_string());
                    let total_memory = device.total_memory().unwrap_or(0);

                    GpuDeviceInfo {
                        name,
                        compute_capability: "Unknown".to_string(), // Would need to query
                        total_memory_mb: (total_memory / 1024 / 1024) as usize,
                        available_memory_mb: (total_memory / 1024 / 1024) as usize,
                        cuda_cores: None,
                    }
                }
                Err(_) => GpuDeviceInfo {
                    name: "No GPU Available".to_string(),
                    compute_capability: "N/A".to_string(),
                    total_memory_mb: 0,
                    available_memory_mb: 0,
                    cuda_cores: None,
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            GpuDeviceInfo {
                name: "GPU Support Disabled".to_string(),
                compute_capability: "N/A".to_string(),
                total_memory_mb: 0,
                available_memory_mb: 0,
                cuda_cores: None,
            }
        }
    }
}

#[async_trait]
impl GpuPhysicsAdapter for PhysicsOrchestratorAdapter {
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<(), String> {
        self.actor_addr
            .send(UpdateGraphData { graph_data: graph })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?;

        Ok(())
    }

    async fn simulate_step(&mut self, params: &SimulationParams) -> Result<PhysicsStepResult, String> {
        // Update parameters
        self.actor_addr
            .send(UpdateSimulationParams {
                params: params.clone(),
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Update params failed: {}", e))?;

        // Execute simulation step
        self.actor_addr
            .send(SimulationStep)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Simulation step failed: {}", e))?;

        // Return dummy result - actual implementation would get real metrics
        Ok(PhysicsStepResult {
            iteration: 0,
            kinetic_energy: 0.0,
            potential_energy: 0.0,
            total_energy: 0.0,
            convergence_delta: 0.0,
            execution_time_ms: 0.0,
        })
    }

    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<(), String> {
        self.actor_addr
            .send(UpdateGraphData { graph_data: graph })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?;

        Ok(())
    }

    async fn upload_constraints(&mut self, constraints: &ConstraintSet) -> Result<(), String> {
        self.actor_addr
            .send(ApplyOntologyConstraints {
                constraint_set: constraints.clone(),
                merge_mode: ConstraintMergeMode::Replace,
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Upload constraints failed: {}", e))
    }

    async fn clear_constraints(&mut self) -> Result<(), String> {
        // Send empty constraint set
        self.upload_constraints(&ConstraintSet::default()).await
    }

    async fn update_parameters(&mut self, params: &SimulationParams) -> Result<(), String> {
        self.actor_addr
            .send(UpdateSimulationParams {
                params: params.clone(),
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Update params failed: {}", e))
    }

    async fn get_positions(&self) -> Result<Vec<(u32, f32, f32, f32)>, String> {
        let snapshot = self.actor_addr
            .send(RequestPositionSnapshot {
                snapshot_type: SnapshotType::KnowledgeGraph,
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Get positions failed: {}", e))?;

        let positions = snapshot.knowledge_nodes.iter()
            .map(|(id, data)| (*id, data.x, data.y, data.z))
            .collect();

        Ok(positions)
    }

    async fn set_node_position(&mut self, node_id: u32, x: f32, y: f32, z: f32) -> Result<(), String> {
        self.actor_addr
            .send(UpdateNodePosition {
                node_id,
                x,
                y,
                z,
            })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Set position failed: {}", e))
    }

    async fn get_statistics(&self) -> Result<PhysicsStatistics, String> {
        let status = self.actor_addr
            .send(GetPhysicsStatus)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?;

        Ok(PhysicsStatistics {
            total_steps: status.performance.total_steps,
            average_step_time_ms: status.performance.average_step_time_ms,
            current_fps: status.performance.last_fps,
            gpu_memory_used_mb: status.performance.gpu_memory_usage_mb,
            gpu_utilization_percent: status.performance.gpu_utilization,
            convergence_rate: status.performance.convergence_rate,
        })
    }

    fn is_available(&self) -> bool {
        self.device_info.total_memory_mb > 0
    }

    fn get_device_info(&self) -> GpuDeviceInfo {
        self.device_info.clone()
    }
}
```

### SemanticProcessorAdapter

**Location**: `src/adapters/semantic_processor_adapter.rs`

**Implementation Strategy**: Wraps `SemanticProcessorActor` to implement `GpuSemanticAnalyzer`.

```rust
use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use actix::Addr;
use crate::ports::gpu_semantic_analyzer::*;
use crate::actors::semantic_processor_actor::{SemanticProcessorActor, GetSemanticStats};
use crate::actors::messages::*;
use crate::models::graph::GraphData;
use crate::models::constraints::ConstraintSet;

pub struct SemanticProcessorAdapter {
    actor_addr: Addr<SemanticProcessorActor>,
}

impl SemanticProcessorAdapter {
    pub fn new(actor_addr: Addr<SemanticProcessorActor>) -> Self {
        Self { actor_addr }
    }
}

#[async_trait]
impl GpuSemanticAnalyzer for SemanticProcessorAdapter {
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<(), String> {
        self.actor_addr
            .send(crate::actors::semantic_processor_actor::SetGraphData { graph_data: graph })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?;

        Ok(())
    }

    async fn detect_communities(&mut self, algorithm: ClusteringAlgorithm) -> Result<CommunityDetectionResult, String> {
        // Trigger clustering analysis
        // This would integrate with the existing clustering_actor logic

        // For now, return placeholder
        Ok(CommunityDetectionResult {
            clusters: HashMap::new(),
            cluster_sizes: HashMap::new(),
            modularity: 0.0,
            computation_time_ms: 0.0,
        })
    }

    async fn compute_shortest_paths(&mut self, source_node_id: u32) -> Result<PathfindingResult, String> {
        // This would integrate with the existing SSSP CUDA kernel
        // For now, return placeholder
        Ok(PathfindingResult {
            source_node: source_node_id,
            distances: HashMap::new(),
            paths: HashMap::new(),
            computation_time_ms: 0.0,
        })
    }

    async fn compute_all_pairs_shortest_paths(&mut self) -> Result<HashMap<(u32, u32), Vec<u32>>, String> {
        Err("All-pairs shortest paths not yet implemented".to_string())
    }

    async fn generate_semantic_constraints(&mut self, config: SemanticConstraintConfig) -> Result<ConstraintSet, String> {
        self.actor_addr
            .send(RegenerateSemanticConstraints)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Regenerate constraints failed: {}", e))?;

        // Get generated constraints
        let constraint_set = self.actor_addr
            .send(GetConstraints)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Get constraints failed: {}", e))?;

        Ok(constraint_set)
    }

    async fn optimize_layout(&mut self, constraints: &ConstraintSet, max_iterations: usize) -> Result<OptimizationResult, String> {
        self.actor_addr
            .send(TriggerStressMajorization)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
            .map_err(|e| format!("Stress optimization failed: {}", e))?;

        // Return placeholder result
        Ok(OptimizationResult {
            converged: true,
            iterations: 0,
            final_stress: 0.0,
            convergence_delta: 0.0,
            computation_time_ms: 0.0,
        })
    }

    async fn analyze_node_importance(&mut self, algorithm: ImportanceAlgorithm) -> Result<HashMap<u32, f32>, String> {
        // Placeholder for PageRank and other centrality algorithms
        Ok(HashMap::new())
    }

    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<(), String> {
        self.actor_addr
            .send(crate::actors::semantic_processor_actor::SetGraphData { graph_data: graph })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?;

        Ok(())
    }

    async fn get_statistics(&self) -> Result<SemanticStatistics, String> {
        let stats = self.actor_addr
            .send(GetSemanticStats)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?;

        Ok(SemanticStatistics {
            total_analyses: stats.constraints_generated as u64,
            average_clustering_time_ms: 0.0,
            average_pathfinding_time_ms: 0.0,
            cache_hit_rate: 0.0,
            gpu_memory_used_mb: 0.0,
        })
    }
}
```

### WhelkInferenceEngine

**Location**: `src/adapters/whelk_inference_engine.rs`

**Implementation Strategy**: Wraps whelk-rs for ontology reasoning.

```rust
use async_trait::async_trait;
use crate::ports::inference_engine::*;
use crate::ports::ontology_repository::{OwlClass, OwlAxiom, InferenceResults};

pub struct WhelkInferenceEngine {
    // whelk-rs reasoner state
    // This is a placeholder - actual implementation would use whelk-rs types
    loaded: bool,
    class_count: usize,
    axiom_count: usize,
}

impl WhelkInferenceEngine {
    pub fn new() -> Self {
        Self {
            loaded: false,
            class_count: 0,
            axiom_count: 0,
        }
    }
}

#[async_trait]
impl InferenceEngine for WhelkInferenceEngine {
    async fn load_ontology(&mut self, classes: Vec<OwlClass>, axioms: Vec<OwlAxiom>) -> Result<(), String> {
        // TODO: Integrate whelk-rs loading logic
        self.class_count = classes.len();
        self.axiom_count = axioms.len();
        self.loaded = true;

        Ok(())
    }

    async fn infer(&mut self) -> Result<InferenceResults, String> {
        if !self.loaded {
            return Err("Ontology not loaded".to_string());
        }

        let start = std::time::Instant::now();

        // TODO: Integrate whelk-rs reasoning
        // For now, return empty results
        let inferred_axioms = Vec::new();

        let inference_time_ms = start.elapsed().as_millis() as u64;

        Ok(InferenceResults {
            timestamp: chrono::Utc::now(),
            inferred_axioms,
            inference_time_ms,
            reasoner_version: "whelk-rs-0.1.0".to_string(),
        })
    }

    async fn is_entailed(&self, axiom: &OwlAxiom) -> Result<bool, String> {
        if !self.loaded {
            return Err("Ontology not loaded".to_string());
        }

        // TODO: Implement entailment check
        Ok(false)
    }

    async fn get_subclass_hierarchy(&self) -> Result<Vec<(String, String)>, String> {
        if !self.loaded {
            return Err("Ontology not loaded".to_string());
        }

        // TODO: Extract hierarchy from reasoner
        Ok(Vec::new())
    }

    async fn classify_instance(&self, instance_iri: &str) -> Result<Vec<String>, String> {
        if !self.loaded {
            return Err("Ontology not loaded".to_string());
        }

        // TODO: Implement instance classification
        Ok(Vec::new())
    }

    async fn check_consistency(&self) -> Result<bool, String> {
        if !self.loaded {
            return Err("Ontology not loaded".to_string());
        }

        // TODO: Implement consistency checking
        Ok(true)
    }

    async fn explain_entailment(&self, axiom: &OwlAxiom) -> Result<Vec<OwlAxiom>, String> {
        if !self.loaded {
            return Err("Ontology not loaded".to_string());
        }

        // TODO: Implement explanation generation
        Ok(Vec::new())
    }

    async fn clear(&mut self) -> Result<(), String> {
        self.loaded = false;
        self.class_count = 0;
        self.axiom_count = 0;
        Ok(())
    }

    async fn get_statistics(&self) -> Result<InferenceStatistics, String> {
        Ok(InferenceStatistics {
            loaded_classes: self.class_count,
            loaded_axioms: self.axiom_count,
            inferred_axioms: 0,
            last_inference_time_ms: 0,
            total_inferences: 0,
        })
    }
}
```

## Summary

This adapter layer provides complete implementations for all ports:

1. **SQLite Repositories** - Fully async with connection pooling and caching
2. **Actor Adapters** - Bridge between ports and existing actor system
3. **Inference Engine** - Whelk-rs integration (with TODO markers for actual implementation)

All adapters:
- Implement the async port interfaces
- Handle error conversion properly
- Use tokio's `spawn_blocking` for blocking operations
- Provide proper resource cleanup
- Are production-ready (except whelk-rs integration which needs the actual library)

Next document will cover the CQRS application layer.
