// src/repositories/unified_graph_repository.rs
//! Unified Graph Repository Adapter
//!
//! Implements KnowledgeGraphRepository trait using unified.db schema.
//! This is the unified database implementation that combines graph and ontology data
//! in a single database schema, replacing the legacy separate SQLite adapters.
//!
//! CRITICAL: This maintains identical interface to preserve CUDA kernel compatibility.

use async_trait::async_trait;
use log::{debug, info, warn};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::instrument;

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::knowledge_graph_repository::{
    GraphStatistics, KnowledgeGraphRepository, KnowledgeGraphRepositoryError,
    Result as RepoResult,
};

/// Unified database repository implementing KnowledgeGraphRepository
///
/// Uses unified.db schema with graph_nodes table that includes owl_class_iri
/// linkage to ontology data for semantic integration.
pub struct UnifiedGraphRepository {
    conn: Arc<Mutex<Connection>>,
    #[allow(dead_code)]
    metrics: Arc<RepositoryMetrics>,
}

/// Repository metrics for monitoring
#[derive(Debug, Default)]
pub struct RepositoryMetrics {
    pub load_count: std::sync::atomic::AtomicU64,
    pub save_count: std::sync::atomic::AtomicU64,
    pub query_count: std::sync::atomic::AtomicU64,
    pub error_count: std::sync::atomic::AtomicU64,
}

impl UnifiedGraphRepository {
    /// Create new unified graph repository
    ///
    /// # Arguments
    /// * `db_path` - Path to unified.db database
    ///
    /// # Schema
    /// Uses unified schema with graph_nodes table that includes:
    /// - id, metadata_id, label (identity)
    /// - x, y, z, vx, vy, vz (physics - UNCHANGED for CUDA)
    /// - mass, charge (physics properties)
    /// - owl_class_iri (ontology linkage - NEW)
    /// - color, size, metadata (rendering)
    pub fn new(db_path: &str) -> Result<Self, KnowledgeGraphRepositoryError> {
        let conn = Connection::open(db_path).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!(
                "Failed to open unified database: {}",
                e
            ))
        })?;

        // Enable foreign keys for referential integrity
        conn.execute("PRAGMA foreign_keys = ON", []).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!(
                "Failed to enable foreign keys: {}",
                e
            ))
        })?;

        // Create unified schema
        Self::create_schema(&conn)?;

        info!("Initialized UnifiedGraphRepository at {}", db_path);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            metrics: Arc::new(RepositoryMetrics::default()),
        })
    }

    /// Create unified database schema
    ///
    /// CRITICAL: x,y,z,vx,vy,vz columns MUST match old schema for CUDA compatibility
    fn create_schema(conn: &Connection) -> Result<(), KnowledgeGraphRepositoryError> {
        conn.execute_batch(
            r#"
            -- Unified graph nodes table (combines knowledge graph + ontology linkage)
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata_id TEXT UNIQUE NOT NULL,
                label TEXT NOT NULL,

                -- Physics state (UNCHANGED - CUDA compatibility)
                x REAL NOT NULL DEFAULT 0.0,
                y REAL NOT NULL DEFAULT 0.0,
                z REAL NOT NULL DEFAULT 0.0,
                vx REAL NOT NULL DEFAULT 0.0,
                vy REAL NOT NULL DEFAULT 0.0,
                vz REAL NOT NULL DEFAULT 0.0,

                -- Physics properties
                mass REAL NOT NULL DEFAULT 1.0,
                charge REAL NOT NULL DEFAULT 0.0,

                -- NEW: Ontology linkage (NULL for non-ontology nodes)
                owl_class_iri TEXT,

                -- Rendering properties
                color TEXT,
                size REAL,
                node_type TEXT,
                weight REAL,
                group_name TEXT,
                metadata TEXT, -- JSON

                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                -- Foreign key to ontology (if exists)
                FOREIGN KEY (owl_class_iri) REFERENCES owl_classes(iri)
                    ON DELETE SET NULL
                    ON UPDATE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_graph_nodes_metadata_id
                ON graph_nodes(metadata_id);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_owl_class
                ON graph_nodes(owl_class_iri);
            CREATE INDEX IF NOT EXISTS idx_graph_nodes_label
                ON graph_nodes(label);

            -- Unified graph edges table
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                edge_type TEXT,
                metadata TEXT, -- JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (source) REFERENCES graph_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES graph_nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_source_target
                ON graph_edges(source, target);

            -- Graph statistics cache
            CREATE TABLE IF NOT EXISTS graph_statistics (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                node_count INTEGER NOT NULL,
                edge_count INTEGER NOT NULL,
                average_degree REAL NOT NULL,
                connected_components INTEGER NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            "#,
        )
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!(
                "Failed to create unified schema: {}",
                e
            ))
        })?;

        Ok(())
    }

    /// Deserialize node from database row (unified schema)
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
        let _mass: f32 = row.get(9)?; // Available but not used in Node struct
        let _charge: f32 = row.get(10)?; // Available but not used in Node struct
        let owl_class_iri: Option<String> = row.get(11)?;
        let color: Option<String> = row.get(12)?;
        let size: Option<f32> = row.get(13)?;
        let node_type: Option<String> = row.get(14)?;
        let weight: Option<f32> = row.get(15)?;
        let group_name: Option<String> = row.get(16)?;
        let metadata_json: Option<String> = row.get(17)?;

        let mut metadata: HashMap<String, String> = metadata_json
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();

        // Add owl_class_iri to metadata if present
        if let Some(iri) = owl_class_iri {
            metadata.insert("owl_class_iri".to_string(), iri);
        }

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
        node.node_type = node_type;
        node.weight = weight;
        node.group = group_name;
        node.metadata = metadata;

        Ok(node)
    }

    /// Serialize node metadata to JSON
    fn serialize_metadata(metadata: &HashMap<String, String>) -> String {
        serde_json::to_string(metadata).unwrap_or_else(|_| "{}".to_string())
    }

    /// Update graph statistics cache
    fn update_statistics_cache(&self, conn: &Connection) -> RepoResult<()> {
        let node_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to count nodes: {}",
                    e
                ))
            })?;

        let edge_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0))
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to count edges: {}",
                    e
                ))
            })?;

        let average_degree = if node_count > 0 {
            (edge_count as f32 * 2.0) / node_count as f32
        } else {
            0.0
        };

        conn.execute(
            r#"
            INSERT OR REPLACE INTO graph_statistics
                (id, node_count, edge_count, average_degree, connected_components, last_updated)
            VALUES (1, ?1, ?2, ?3, 1, CURRENT_TIMESTAMP)
            "#,
            params![node_count, edge_count, average_degree],
        )
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!(
                "Failed to update statistics: {}",
                e
            ))
        })?;

        Ok(())
    }
}

#[async_trait]
impl KnowledgeGraphRepository for UnifiedGraphRepository {
    #[instrument(skip(self), level = "debug")]
    async fn load_graph(&self) -> RepoResult<Arc<GraphData>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            // Load all nodes
            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                           mass, charge, owl_class_iri, color, size,
                           node_type, weight, group_name, metadata
                    FROM graph_nodes
                    "#,
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare node query: {}",
                        e
                    ))
                })?;

            let nodes = stmt
                .query_map([], Self::deserialize_node)
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query nodes: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect nodes: {}",
                        e
                    ))
                })?;

            debug!("Loaded {} nodes from unified database", nodes.len());

            // Load all edges
            let mut edge_stmt = conn
                .prepare("SELECT id, source, target, weight, edge_type, metadata FROM graph_edges")
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare edge query: {}",
                        e
                    ))
                })?;

            let edges = edge_stmt
                .query_map([], |row| {
                    let id: String = row.get(0)?;
                    let source: u32 = row.get(1)?;
                    let target: u32 = row.get(2)?;
                    let weight: f32 = row.get(3)?;
                    let edge_type: Option<String> = row.get(4)?;
                    let metadata_json: Option<String> = row.get(5)?;

                    let metadata =
                        metadata_json.and_then(|json| serde_json::from_str(&json).ok());

                    let mut edge = Edge::new(source, target, weight);
                    edge.id = id;
                    edge.edge_type = edge_type;
                    edge.metadata = metadata;

                    Ok(edge)
                })
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query edges: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Edge>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect edges: {}",
                        e
                    ))
                })?;

            debug!("Loaded {} edges from unified database", edges.len());

            let mut graph = GraphData::new();
            graph.nodes = nodes;
            graph.edges = edges;

            Ok(Arc::new(graph))
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn save_graph(&self, graph: &GraphData) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let nodes = graph.nodes.clone();
        let edges = graph.edges.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            // Clear existing data
            tx.execute("DELETE FROM graph_edges", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to clear edges: {}",
                    e
                ))
            })?;

            tx.execute("DELETE FROM graph_nodes", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to clear nodes: {}",
                    e
                ))
            })?;

            // Insert nodes
            let mut node_stmt = tx
                .prepare(
                    r#"
                    INSERT INTO graph_nodes
                        (id, metadata_id, label, x, y, z, vx, vy, vz, mass, charge,
                         owl_class_iri, color, size, node_type, weight, group_name, metadata)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
                    "#,
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare node insert: {}",
                        e
                    ))
                })?;

            for node in &nodes {
                let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
                let metadata_json = Self::serialize_metadata(&node.metadata);

                node_stmt
                    .execute(params![
                        node.id,
                        node.metadata_id,
                        node.label,
                        node.data.x,
                        node.data.y,
                        node.data.z,
                        node.data.vx,
                        node.data.vy,
                        node.data.vz,
                        1.0, // mass default
                        0.0, // charge default
                        owl_class_iri,
                        node.color,
                        node.size,
                        node.node_type,
                        node.weight,
                        node.group,
                        metadata_json,
                    ])
                    .map_err(|e| {
                        KnowledgeGraphRepositoryError::DatabaseError(format!(
                            "Failed to insert node {}: {}",
                            node.id, e
                        ))
                    })?;
            }

            // Insert edges
            let mut edge_stmt = tx
                .prepare(
                    "INSERT INTO graph_edges (id, source, target, weight, edge_type, metadata)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare edge insert: {}",
                        e
                    ))
                })?;

            for edge in &edges {
                let metadata_json = edge
                    .metadata
                    .as_ref()
                    .and_then(|m| serde_json::to_string(m).ok());

                edge_stmt
                    .execute(params![
                        edge.id,
                        edge.source,
                        edge.target,
                        edge.weight,
                        edge.edge_type,
                        metadata_json,
                    ])
                    .map_err(|e| {
                        KnowledgeGraphRepositoryError::DatabaseError(format!(
                            "Failed to insert edge {}: {}",
                            edge.id, e
                        ))
                    })?;
            }

            // Drop statements before commit to avoid borrow checker issues
            drop(node_stmt);
            drop(edge_stmt);

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            info!(
                "Saved graph to unified database: {} nodes, {} edges",
                nodes.len(),
                edges.len()
            );

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn add_node(&self, node: &Node) -> RepoResult<u32> {
        let conn_arc = self.conn.clone();
        let node = node.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
            let metadata_json = Self::serialize_metadata(&node.metadata);

            conn.execute(
                r#"
                INSERT INTO graph_nodes
                    (metadata_id, label, x, y, z, vx, vy, vz, mass, charge,
                     owl_class_iri, color, size, node_type, weight, group_name, metadata)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
                "#,
                params![
                    node.metadata_id,
                    node.label,
                    node.data.x,
                    node.data.y,
                    node.data.z,
                    node.data.vx,
                    node.data.vy,
                    node.data.vz,
                    1.0,
                    0.0,
                    owl_class_iri,
                    node.color,
                    node.size,
                    node.node_type,
                    node.weight,
                    node.group,
                    metadata_json,
                ],
            )
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to insert node: {}",
                    e
                ))
            })?;

            let node_id = conn.last_insert_rowid() as u32;
            debug!("Added node {} to unified database", node_id);

            Ok(node_id)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn batch_add_nodes(&self, nodes: Vec<Node>) -> RepoResult<Vec<u32>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            let mut stmt = tx
                .prepare(
                    r#"
                    INSERT INTO graph_nodes
                        (metadata_id, label, x, y, z, vx, vy, vz, mass, charge,
                         owl_class_iri, color, size, node_type, weight, group_name, metadata)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
                    "#,
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let mut node_ids = Vec::new();

            for node in &nodes {
                let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
                let metadata_json = Self::serialize_metadata(&node.metadata);

                stmt.execute(params![
                    node.metadata_id,
                    node.label,
                    node.data.x,
                    node.data.y,
                    node.data.z,
                    node.data.vx,
                    node.data.vy,
                    node.data.vz,
                    1.0,
                    0.0,
                    owl_class_iri,
                    node.color,
                    node.size,
                    node.node_type,
                    node.weight,
                    node.group,
                    metadata_json,
                ])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to insert node: {}",
                        e
                    ))
                })?;

                node_ids.push(tx.last_insert_rowid() as u32);
            }

            // Drop statement before commit to avoid borrow checker issues
            drop(stmt);

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            debug!("Batch added {} nodes to unified database", nodes.len());

            Ok(node_ids)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn update_node(&self, node: &Node) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let node = node.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
            let metadata_json = Self::serialize_metadata(&node.metadata);

            let rows = conn
                .execute(
                    r#"
                    UPDATE graph_nodes
                    SET metadata_id = ?1, label = ?2, x = ?3, y = ?4, z = ?5,
                        vx = ?6, vy = ?7, vz = ?8, owl_class_iri = ?9,
                        color = ?10, size = ?11, node_type = ?12, weight = ?13,
                        group_name = ?14, metadata = ?15, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?16
                    "#,
                    params![
                        node.metadata_id,
                        node.label,
                        node.data.x,
                        node.data.y,
                        node.data.z,
                        node.data.vx,
                        node.data.vy,
                        node.data.vz,
                        owl_class_iri,
                        node.color,
                        node.size,
                        node.node_type,
                        node.weight,
                        node.group,
                        metadata_json,
                        node.id,
                    ],
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to update node: {}",
                        e
                    ))
                })?;

            if rows == 0 {
                return Err(KnowledgeGraphRepositoryError::NodeNotFound(node.id));
            }

            debug!("Updated node {} in unified database", node.id);

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn batch_update_nodes(&self, nodes: Vec<Node>) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            let mut stmt = tx
                .prepare(
                    r#"
                    UPDATE graph_nodes
                    SET metadata_id = ?1, label = ?2, x = ?3, y = ?4, z = ?5,
                        vx = ?6, vy = ?7, vz = ?8, owl_class_iri = ?9,
                        color = ?10, size = ?11, node_type = ?12, weight = ?13,
                        group_name = ?14, metadata = ?15, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?16
                    "#,
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            for node in &nodes {
                let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
                let metadata_json = Self::serialize_metadata(&node.metadata);

                stmt.execute(params![
                    node.metadata_id,
                    node.label,
                    node.data.x,
                    node.data.y,
                    node.data.z,
                    node.data.vx,
                    node.data.vy,
                    node.data.vz,
                    owl_class_iri,
                    node.color,
                    node.size,
                    node.node_type,
                    node.weight,
                    node.group,
                    metadata_json,
                    node.id,
                ])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to update node {}: {}",
                        node.id, e
                    ))
                })?;
            }

            // Drop statement before commit to avoid borrow checker issues
            drop(stmt);

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            debug!("Batch updated {} nodes in unified database", nodes.len());

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn remove_node(&self, node_id: u32) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let rows = conn
                .execute("DELETE FROM graph_nodes WHERE id = ?1", params![node_id])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to delete node: {}",
                        e
                    ))
                })?;

            if rows == 0 {
                return Err(KnowledgeGraphRepositoryError::NodeNotFound(node_id));
            }

            debug!("Removed node {} from unified database", node_id);

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn batch_remove_nodes(&self, node_ids: Vec<u32>) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            let mut stmt = tx
                .prepare("DELETE FROM graph_nodes WHERE id = ?1")
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            for node_id in &node_ids {
                stmt.execute(params![node_id]).map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to delete node {}: {}",
                        node_id, e
                    ))
                })?;
            }

            // Drop statement before commit to avoid borrow checker issues
            drop(stmt);

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            debug!(
                "Batch removed {} nodes from unified database",
                node_ids.len()
            );

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn get_node(&self, node_id: u32) -> RepoResult<Option<Node>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let node = conn
                .query_row(
                    r#"
                    SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                           mass, charge, owl_class_iri, color, size,
                           node_type, weight, group_name, metadata
                    FROM graph_nodes
                    WHERE id = ?1
                    "#,
                    params![node_id],
                    Self::deserialize_node,
                )
                .optional()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to get node: {}",
                        e
                    ))
                })?;

            Ok(node)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn get_nodes(&self, node_ids: Vec<u32>) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let placeholders = node_ids
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(",");

            let query = format!(
                r#"
                SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                       mass, charge, owl_class_iri, color, size,
                       node_type, weight, group_name, metadata
                FROM graph_nodes
                WHERE id IN ({})
                "#,
                placeholders
            );

            let mut stmt = conn.prepare(&query).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to prepare statement: {}",
                    e
                ))
            })?;

            let params: Vec<&dyn rusqlite::ToSql> =
                node_ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

            let nodes = stmt
                .query_map(&params[..], Self::deserialize_node)
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query nodes: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect nodes: {}",
                        e
                    ))
                })?;

            Ok(nodes)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();
        let metadata_id = metadata_id.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                           mass, charge, owl_class_iri, color, size,
                           node_type, weight, group_name, metadata
                    FROM graph_nodes
                    WHERE metadata_id = ?1
                    "#,
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let nodes = stmt
                .query_map(params![metadata_id], Self::deserialize_node)
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query nodes: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect nodes: {}",
                        e
                    ))
                })?;

            Ok(nodes)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn search_nodes_by_label(&self, label: &str) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();
        let search_pattern = format!("%{}%", label);

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                           mass, charge, owl_class_iri, color, size,
                           node_type, weight, group_name, metadata
                    FROM graph_nodes
                    WHERE label LIKE ?1
                    "#,
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let nodes = stmt
                .query_map(params![search_pattern], Self::deserialize_node)
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query nodes: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect nodes: {}",
                        e
                    ))
                })?;

            Ok(nodes)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn add_edge(&self, edge: &Edge) -> RepoResult<String> {
        let conn_arc = self.conn.clone();
        let edge = edge.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let metadata_json = edge
                .metadata
                .as_ref()
                .and_then(|m| serde_json::to_string(m).ok());

            conn.execute(
                "INSERT INTO graph_edges (id, source, target, weight, edge_type, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    edge.id,
                    edge.source,
                    edge.target,
                    edge.weight,
                    edge.edge_type,
                    metadata_json,
                ],
            )
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to insert edge: {}", e))
            })?;

            debug!("Added edge {} to unified database", edge.id);

            Ok(edge.id.clone())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn batch_add_edges(&self, edges: Vec<Edge>) -> RepoResult<Vec<String>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            let mut stmt = tx
                .prepare(
                    "INSERT INTO graph_edges (id, source, target, weight, edge_type, metadata)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let mut edge_ids = Vec::new();

            for edge in &edges {
                let metadata_json = edge
                    .metadata
                    .as_ref()
                    .and_then(|m| serde_json::to_string(m).ok());

                stmt.execute(params![
                    edge.id,
                    edge.source,
                    edge.target,
                    edge.weight,
                    edge.edge_type,
                    metadata_json,
                ])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to insert edge: {}",
                        e
                    ))
                })?;

                edge_ids.push(edge.id.clone());
            }

            // Drop statement before commit to avoid borrow checker issues
            drop(stmt);

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            debug!("Batch added {} edges to unified database", edges.len());

            Ok(edge_ids)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn update_edge(&self, edge: &Edge) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let edge = edge.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let metadata_json = edge
                .metadata
                .as_ref()
                .and_then(|m| serde_json::to_string(m).ok());

            let rows = conn
                .execute(
                    "UPDATE graph_edges
                     SET source = ?1, target = ?2, weight = ?3, edge_type = ?4, metadata = ?5
                     WHERE id = ?6",
                    params![
                        edge.source,
                        edge.target,
                        edge.weight,
                        edge.edge_type,
                        metadata_json,
                        edge.id,
                    ],
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to update edge: {}",
                        e
                    ))
                })?;

            if rows == 0 {
                return Err(KnowledgeGraphRepositoryError::EdgeNotFound(
                    edge.id.clone(),
                ));
            }

            debug!("Updated edge {} in unified database", edge.id);

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn remove_edge(&self, edge_id: &str) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let edge_id = edge_id.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let rows = conn
                .execute("DELETE FROM graph_edges WHERE id = ?1", params![edge_id])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to delete edge: {}",
                        e
                    ))
                })?;

            if rows == 0 {
                return Err(KnowledgeGraphRepositoryError::EdgeNotFound(edge_id.clone()));
            }

            debug!("Removed edge {} from unified database", edge_id);

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn batch_remove_edges(&self, edge_ids: Vec<String>) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            let mut stmt = tx
                .prepare("DELETE FROM graph_edges WHERE id = ?1")
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            for edge_id in &edge_ids {
                stmt.execute(params![edge_id]).map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to delete edge {}: {}",
                        edge_id, e
                    ))
                })?;
            }

            // Drop statement before commit to avoid borrow checker issues
            drop(stmt);

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            debug!(
                "Batch removed {} edges from unified database",
                edge_ids.len()
            );

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn get_node_edges(&self, node_id: u32) -> RepoResult<Vec<Edge>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let mut stmt = conn
                .prepare(
                    "SELECT id, source, target, weight, edge_type, metadata
                     FROM graph_edges
                     WHERE source = ?1 OR target = ?1",
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let edges = stmt
                .query_map(params![node_id], |row| {
                    let id: String = row.get(0)?;
                    let source: u32 = row.get(1)?;
                    let target: u32 = row.get(2)?;
                    let weight: f32 = row.get(3)?;
                    let edge_type: Option<String> = row.get(4)?;
                    let metadata_json: Option<String> = row.get(5)?;

                    let metadata =
                        metadata_json.and_then(|json| serde_json::from_str(&json).ok());

                    let mut edge = Edge::new(source, target, weight);
                    edge.id = id;
                    edge.edge_type = edge_type;
                    edge.metadata = metadata;

                    Ok(edge)
                })
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query edges: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Edge>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect edges: {}",
                        e
                    ))
                })?;

            Ok(edges)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn get_edges_between(&self, source_id: u32, target_id: u32) -> RepoResult<Vec<Edge>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let mut stmt = conn
                .prepare(
                    "SELECT id, source, target, weight, edge_type, metadata
                     FROM graph_edges
                     WHERE (source = ?1 AND target = ?2) OR (source = ?2 AND target = ?1)",
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let edges = stmt
                .query_map(params![source_id, target_id], |row| {
                    let id: String = row.get(0)?;
                    let source: u32 = row.get(1)?;
                    let target: u32 = row.get(2)?;
                    let weight: f32 = row.get(3)?;
                    let edge_type: Option<String> = row.get(4)?;
                    let metadata_json: Option<String> = row.get(5)?;

                    let metadata =
                        metadata_json.and_then(|json| serde_json::from_str(&json).ok());

                    let mut edge = Edge::new(source, target, weight);
                    edge.id = id;
                    edge.edge_type = edge_type;
                    edge.metadata = metadata;

                    Ok(edge)
                })
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query edges: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Edge>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect edges: {}",
                        e
                    ))
                })?;

            Ok(edges)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    /// CRITICAL: Batch update positions - UNCHANGED interface for CUDA compatibility
    ///
    /// This method MUST maintain identical interface to legacy adapter
    /// Format: Vec<(node_id, x, y, z)>
    ///
    /// CUDA kernels call this after force computation with updated positions.
    /// The x,y,z,vx,vy,vz column names MUST NOT change.
    async fn batch_update_positions(
        &self,
        positions: Vec<(u32, f32, f32, f32)>,
    ) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            // CRITICAL: Column names x,y,z,vx,vy,vz MUST match CUDA expectations
            let mut stmt = tx
                .prepare(
                    "UPDATE graph_nodes
                     SET x = ?1, y = ?2, z = ?3, updated_at = CURRENT_TIMESTAMP
                     WHERE id = ?4",
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            // Batch update in chunks of 10,000 for performance (same as legacy)
            for chunk in positions.chunks(10_000) {
                for (node_id, x, y, z) in chunk {
                    stmt.execute(params![x, y, z, node_id]).map_err(|e| {
                        KnowledgeGraphRepositoryError::DatabaseError(format!(
                            "Failed to update position for node {}: {}",
                            node_id, e
                        ))
                    })?;
                }
            }

            // Drop statement before commit to avoid borrow checker issues
            drop(stmt);

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            debug!(
                "Batch updated {} positions in unified database",
                positions.len()
            );

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn query_nodes(&self, _query: &str) -> RepoResult<Vec<Node>> {
        // TODO: Implement query DSL parsing
        warn!("query_nodes not yet implemented for unified repository");
        Ok(Vec::new())
    }

    async fn get_neighbors(&self, node_id: u32) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let mut stmt = conn
                .prepare(
                    r#"
                    SELECT DISTINCT n.id, n.metadata_id, n.label, n.x, n.y, n.z,
                           n.vx, n.vy, n.vz, n.mass, n.charge, n.owl_class_iri,
                           n.color, n.size, n.node_type, n.weight, n.group_name, n.metadata
                    FROM graph_nodes n
                    JOIN graph_edges e ON (e.source = ?1 AND e.target = n.id)
                                       OR (e.target = ?1 AND e.source = n.id)
                    "#,
                )
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let nodes = stmt
                .query_map(params![node_id], Self::deserialize_node)
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to query neighbors: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect neighbors: {}",
                        e
                    ))
                })?;

            Ok(nodes)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn get_statistics(&self) -> RepoResult<GraphStatistics> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            // Try to get cached statistics
            let cached: Option<GraphStatistics> = conn
                .query_row(
                    "SELECT node_count, edge_count, average_degree, connected_components, last_updated
                     FROM graph_statistics WHERE id = 1",
                    [],
                    |row| {
                        let timestamp: i64 = row.get(4)?;
                        Ok(GraphStatistics {
                            node_count: row.get::<_, i64>(0)? as usize,
                            edge_count: row.get::<_, i64>(1)? as usize,
                            average_degree: row.get(2)?,
                            connected_components: row.get::<_, i64>(3)? as usize,
                            last_updated: chrono::DateTime::from_timestamp(timestamp, 0)
                                .unwrap_or_else(chrono::Utc::now),
                        })
                    },
                )
                .optional()
                .ok()
                .flatten();

            if let Some(stats) = cached {
                return Ok(stats);
            }

            // Compute fresh statistics
            let node_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
                .unwrap_or(0);

            let edge_count: i64 = conn
                .query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0))
                .unwrap_or(0);

            let average_degree = if node_count > 0 {
                (edge_count as f32 * 2.0) / node_count as f32
            } else {
                0.0
            };

            let stats = GraphStatistics {
                node_count: node_count as usize,
                edge_count: edge_count as usize,
                average_degree,
                connected_components: 1, // TODO: Compute actual connected components
                last_updated: chrono::Utc::now(),
            };

            Ok(stats)
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn clear_graph(&self) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let mut conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            let tx = conn.transaction().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            tx.execute("DELETE FROM graph_edges", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to clear edges: {}",
                    e
                ))
            })?;

            tx.execute("DELETE FROM graph_nodes", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to clear nodes: {}",
                    e
                ))
            })?;

            tx.execute("DELETE FROM graph_statistics", [])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to clear statistics: {}",
                        e
                    ))
                })?;

            tx.commit().map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            info!("Cleared all graph data from unified database");

            Ok(())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }

    async fn begin_transaction(&self) -> RepoResult<()> {
        // Note: rusqlite doesn't support explicit transaction management across async boundaries
        // Transactions are managed per-operation in this implementation
        Ok(())
    }

    async fn commit_transaction(&self) -> RepoResult<()> {
        // Note: rusqlite doesn't support explicit transaction management across async boundaries
        Ok(())
    }

    async fn rollback_transaction(&self) -> RepoResult<()> {
        // Note: rusqlite doesn't support explicit transaction management across async boundaries
        Ok(())
    }

    async fn health_check(&self) -> RepoResult<bool> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified repository mutex");

            // Simple health check: try to query the database
            let result: Result<i64, _> = conn.query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0));

            Ok(result.is_ok())
        })
        .await
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Task join error: {}", e))
        })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unified_repository_creation() {
        let repo = UnifiedGraphRepository::new(":memory:").unwrap();
        assert!(repo.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_add_and_get_node() {
        let repo = UnifiedGraphRepository::new(":memory:").unwrap();

        let mut node = Node::new("test-node".to_string());
        node.label = "Test Node".to_string();

        let node_id = repo.add_node(&node).await.unwrap();
        assert!(node_id > 0);

        let retrieved = repo.get_node(node_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().label, "Test Node");
    }

    #[tokio::test]
    async fn test_batch_update_positions() {
        let repo = UnifiedGraphRepository::new(":memory:").unwrap();

        // Add test nodes
        let mut node1 = Node::new("node1".to_string());
        node1.label = "Node 1".to_string();
        let id1 = repo.add_node(&node1).await.unwrap();

        let mut node2 = Node::new("node2".to_string());
        node2.label = "Node 2".to_string();
        let id2 = repo.add_node(&node2).await.unwrap();

        // Update positions
        let positions = vec![(id1, 10.0, 20.0, 30.0), (id2, 40.0, 50.0, 60.0)];

        repo.batch_update_positions(positions).await.unwrap();

        // Verify positions were updated
        let updated1 = repo.get_node(id1).await.unwrap().unwrap();
        assert_eq!(updated1.data.x, 10.0);
        assert_eq!(updated1.data.y, 20.0);
        assert_eq!(updated1.data.z, 30.0);

        let updated2 = repo.get_node(id2).await.unwrap().unwrap();
        assert_eq!(updated2.data.x, 40.0);
        assert_eq!(updated2.data.y, 50.0);
        assert_eq!(updated2.data.z, 60.0);
    }
}
