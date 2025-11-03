// src/repositories/unified_graph_repository.rs
//! Unified Graph Repository Adapter (Refactored)
//!
//! Implements KnowledgeGraphRepository trait using unified.db schema.
//! Uses SqliteRepository base to eliminate code duplication.
//!
//! Database schema:
//! - graph_nodes: Nodes with owl_class_iri for ontology linkage, physics state (x,y,z,vx,vy,vz)
//! - graph_edges: Edges with optional owl_property_iri for semantic typing
//! - graph_statistics: Cached statistics (node/edge counts, average degree)
//! - file_metadata: SHA1 hashing for incremental GitHub sync

use async_trait::async_trait;
use log::{debug, info};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use tracing::instrument;

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::knowledge_graph_repository::{
    GraphStatistics, KnowledgeGraphRepository, KnowledgeGraphRepositoryError,
    Result as RepoResult,
};
use crate::repositories::generic_repository::{convert_rusqlite_error, SqliteRepository};
use crate::utils::result_mappers::{map_graph_db_error, RusqliteResultExt};

/// Repository for knowledge graph data in unified.db
///
/// Provides high-performance batch operations for GPU physics integration.
/// Uses SqliteRepository base to eliminate ~540 lines of duplicate code.
pub struct UnifiedGraphRepository {
    base: SqliteRepository,
    #[allow(dead_code)]
    metrics: std::sync::Arc<RepositoryMetrics>,
}

///
#[derive(Debug, Default)]
pub struct RepositoryMetrics {
    pub load_count: std::sync::atomic::AtomicU64,
    pub save_count: std::sync::atomic::AtomicU64,
    pub query_count: std::sync::atomic::AtomicU64,
    pub error_count: std::sync::atomic::AtomicU64,
}

impl UnifiedGraphRepository {
    /// Create a new UnifiedGraphRepository
    pub fn new(db_path: &str) -> Result<Self, KnowledgeGraphRepositoryError> {
        let base = SqliteRepository::new(db_path).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to create base repository: {}", e))
        })?;

        // Create schema using base connection
        let conn = base.get_connection();
        let conn_guard = conn.lock().map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to lock connection: {}", e))
        })?;
        Self::create_schema(&conn_guard)?;
        drop(conn_guard);

        info!("Initialized UnifiedGraphRepository at {}", db_path);

        Ok(Self {
            base,
            metrics: std::sync::Arc::new(RepositoryMetrics::default()),
        })
    }

    /// Create unified.db schema for graph storage
    fn create_schema(conn: &Connection) -> Result<(), KnowledgeGraphRepositoryError> {
        info!("Creating unified graph database schema...");

        // Create graph_nodes table
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metadata_id TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                x REAL NOT NULL DEFAULT 0.0,
                y REAL NOT NULL DEFAULT 0.0,
                z REAL NOT NULL DEFAULT 0.0,
                vx REAL NOT NULL DEFAULT 0.0,
                vy REAL NOT NULL DEFAULT 0.0,
                vz REAL NOT NULL DEFAULT 0.0,
                mass REAL NOT NULL DEFAULT 1.0,
                charge REAL NOT NULL DEFAULT 0.0,
                owl_class_iri TEXT,
                color TEXT,
                size REAL DEFAULT 10.0,
                node_type TEXT,
                weight REAL DEFAULT 1.0,
                group_name TEXT,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#,
            [],
        ).to_graph_error("create graph_nodes table")?;

        // Create graph_edges table
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                relation_type TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES graph_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES graph_nodes(id) ON DELETE CASCADE
            )
            "#,
            [],
        ).to_graph_error("create graph_edges table")?;

        // Create graph_statistics table
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS graph_statistics (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                node_count INTEGER NOT NULL DEFAULT 0,
                edge_count INTEGER NOT NULL DEFAULT 0,
                average_degree REAL NOT NULL DEFAULT 0.0,
                connected_components INTEGER NOT NULL DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#,
            [],
        ).to_graph_error("create graph_statistics table")?;

        // Create file_metadata table (used for sync tracking)
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_name TEXT PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                file_blob_sha TEXT,
                github_node_id TEXT,
                sha1 TEXT,
                content_hash TEXT,
                last_modified DATETIME,
                last_content_change DATETIME,
                last_commit DATETIME,
                change_count INTEGER DEFAULT 0,
                processing_status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            "#,
            [],
        ).to_graph_error("create file_metadata table")?;

        // Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_nodes_metadata_id ON graph_nodes(metadata_id)", [])
            .to_graph_error("create metadata_id index")?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id)", [])
            .to_graph_error("create source_id index")?;
        conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id)", [])
            .to_graph_error("create target_id index")?;

        info!("✅ Unified graph database schema created successfully");
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
        let _mass: f32 = row.get(9)?;
        let _charge: f32 = row.get(10)?;
        let owl_class_iri: Option<String> = row.get(11)?;
        let color: Option<String> = row.get(12)?;
        let size: Option<f32> = row.get(13)?;
        let node_type: Option<String> = row.get(14)?;
        let weight: Option<f32> = row.get(15)?;
        let group_name: Option<String> = row.get(16)?;
        let metadata_json: Option<String> = row.get(17)?;

        let mut metadata: HashMap<String, String> = metadata_json
            .and_then(|json| from_json(&json).ok())
            .unwrap_or_default();

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

    /// Serialize metadata to JSON string
    fn serialize_metadata(metadata: &HashMap<String, String>) -> String {
        to_json(metadata).unwrap_or_else(|_| "{}".to_string())
    }

    /// Update statistics cache
    fn update_statistics_cache(&self, conn: &Connection) -> RepoResult<()> {
        let node_count: i64 = conn.query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0))
            .to_graph_error("count nodes")?;

        let edge_count: i64 = conn.query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0))
            .to_graph_error("count edges")?;

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
        ).to_graph_error("update statistics")?;

        Ok(())
    }

    /// Get connection for advanced operations
    pub fn get_connection(&self) -> std::sync::Arc<std::sync::Mutex<Connection>> {
        self.base.get_connection()
    }
}

#[async_trait]
impl KnowledgeGraphRepository for UnifiedGraphRepository {
    #[instrument(skip(self), level = "debug")]
    async fn load_graph(&self) -> RepoResult<std::sync::Arc<GraphData>> {
        self.base.execute_blocking(|conn| {
            let mut stmt = conn.prepare(
                r#"
                SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                       mass, charge, owl_class_iri, color, size,
                       node_type, weight, group_name, metadata
                FROM graph_nodes
                "#,
            ).to_graph_error("prepare node query")?;

            let nodes = stmt.query_map([], Self::deserialize_node)
                .to_graph_error("query nodes")?
                .collect::<Result<Vec<Node>, _>>()
                .to_graph_error("collect nodes")?;

            debug!("Loaded {} nodes from unified database", nodes.len());

            let mut edge_stmt = conn.prepare("SELECT id, source_id, target_id, weight, relation_type, metadata FROM graph_edges")
                .to_graph_error("prepare edge query")?;

            let edges = edge_stmt.query_map([], |row| {
                let id: String = row.get(0)?;
                let source: u32 = row.get(1)?;
                let target: u32 = row.get(2)?;
                let weight: f32 = row.get(3)?;
                let edge_type: Option<String> = row.get(4)?;
                let metadata_json: Option<String> = row.get(5)?;
                let metadata = metadata_json.and_then(|json| from_json(&json).ok());

                let mut edge = Edge::new(source, target, weight);
                edge.id = id;
                edge.edge_type = edge_type;
                edge.metadata = metadata;
                Ok(edge)
            }).to_graph_error("query edges")?
            .collect::<Result<Vec<Edge>, _>>()
            .to_graph_error("collect edges")?;

            debug!("Loaded {} edges from unified database", edges.len());

            let mut graph = GraphData::new();
            graph.nodes = nodes;
            graph.edges = edges;
            Ok(std::sync::Arc::new(graph))
        }).await.map_err(|e| match e {
            crate::repositories::generic_repository::RepositoryError::DatabaseError(msg) => KnowledgeGraphRepositoryError::DatabaseError(msg),
            _ => KnowledgeGraphRepositoryError::DatabaseError(format!("Repository error: {}", e)),
        })
    }

    async fn save_graph(&self, graph: &GraphData) -> RepoResult<()> {
        let nodes = graph.nodes.clone();
        let edges = graph.edges.clone();

        self.base.execute_transaction(|tx| {
            // Check if initial sync
            let metadata_count: i64 = tx.query_row("SELECT COUNT(*) FROM file_metadata", [], |row| row.get(0)).unwrap_or(0);

            if metadata_count == 0 {
                info!("Initial sync detected - clearing existing graph data");
                tx.execute("DELETE FROM graph_edges", []).map_err(convert_rusqlite_error)?;
                tx.execute("DELETE FROM graph_nodes", []).map_err(convert_rusqlite_error)?;
            } else {
                info!("Incremental sync - preserving existing nodes");
            }

            // Insert nodes
            let mut node_stmt = tx.prepare(
                r#"
                INSERT OR REPLACE INTO graph_nodes
                    (id, metadata_id, label, x, y, z, vx, vy, vz, mass, charge,
                     owl_class_iri, color, size, node_type, weight, group_name, metadata)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18)
                "#,
            ).map_err(convert_rusqlite_error)?;

            for node in &nodes {
                let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
                let metadata_json = Self::serialize_metadata(&node.metadata);

                node_stmt.execute(params![
                    node.id, node.metadata_id, node.label,
                    node.data.x, node.data.y, node.data.z,
                    node.data.vx, node.data.vy, node.data.vz,
                    1.0, 0.0, owl_class_iri,
                    node.color, node.size, node.node_type,
                    node.weight, node.group, metadata_json,
                ]).map_err(convert_rusqlite_error)?;
            }

            // Insert edges
            let mut edge_stmt = tx.prepare(
                "INSERT OR REPLACE INTO graph_edges (id, source_id, target_id, weight, relation_type, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            ).map_err(convert_rusqlite_error)?;

            for edge in &edges {
                let metadata_json = edge.metadata.as_ref().and_then(|m| to_json(m).ok());
                edge_stmt.execute(params![edge.id, edge.source, edge.target, edge.weight, edge.edge_type, metadata_json])
                    .map_err(convert_rusqlite_error)?;
            }

            info!("Saved graph to unified database: {} nodes, {} edges", nodes.len(), edges.len());
            Ok(())
        }).await.map_err(|e| match e {
            crate::repositories::generic_repository::RepositoryError::DatabaseError(msg) => KnowledgeGraphRepositoryError::DatabaseError(msg),
            crate::repositories::generic_repository::RepositoryError::TransactionError(msg) => KnowledgeGraphRepositoryError::DatabaseError(msg),
            _ => KnowledgeGraphRepositoryError::DatabaseError(format!("Repository error: {}", e)),
        })
    }

    async fn add_node(&self, node: &Node) -> RepoResult<u32> {
        let node = node.clone();
        self.base.execute_blocking(move |conn| {
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
                    node.metadata_id, node.label,
                    node.data.x, node.data.y, node.data.z,
                    node.data.vx, node.data.vy, node.data.vz,
                    1.0, 0.0, owl_class_iri,
                    node.color, node.size, node.node_type,
                    node.weight, node.group, metadata_json,
                ],
            ).map_err(convert_rusqlite_error)?;

            let node_id = conn.last_insert_rowid() as u32;
            debug!("Added node {} to unified database", node_id);
            Ok(node_id)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn batch_add_nodes(&self, nodes: Vec<Node>) -> RepoResult<Vec<u32>> {
        let mut node_ids = Vec::new();

        self.base.execute_transaction(|tx| {
            let mut stmt = tx.prepare(
                r#"
                INSERT INTO graph_nodes
                    (metadata_id, label, x, y, z, vx, vy, vz, mass, charge,
                     owl_class_iri, color, size, node_type, weight, group_name, metadata)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)
                "#,
            ).map_err(convert_rusqlite_error)?;

            for node in &nodes {
                let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
                let metadata_json = Self::serialize_metadata(&node.metadata);

                stmt.execute(params![
                    node.metadata_id, node.label,
                    node.data.x, node.data.y, node.data.z,
                    node.data.vx, node.data.vy, node.data.vz,
                    1.0, 0.0, owl_class_iri,
                    node.color, node.size, node.node_type,
                    node.weight, node.group, metadata_json,
                ]).map_err(convert_rusqlite_error)?;

                node_ids.push(tx.last_insert_rowid() as u32);
            }

            debug!("Batch added {} nodes to unified database", nodes.len());
            Ok(node_ids)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn update_node(&self, node: &Node) -> RepoResult<()> {
        let node = node.clone();
        self.base.execute_blocking(move |conn| {
            let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
            let metadata_json = Self::serialize_metadata(&node.metadata);

            let rows = conn.execute(
                r#"
                UPDATE graph_nodes
                SET metadata_id = ?1, label = ?2, x = ?3, y = ?4, z = ?5,
                    vx = ?6, vy = ?7, vz = ?8, owl_class_iri = ?9,
                    color = ?10, size = ?11, node_type = ?12, weight = ?13,
                    group_name = ?14, metadata = ?15, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?16
                "#,
                params![
                    node.metadata_id, node.label,
                    node.data.x, node.data.y, node.data.z,
                    node.data.vx, node.data.vy, node.data.vz,
                    owl_class_iri, node.color, node.size,
                    node.node_type, node.weight, node.group,
                    metadata_json, node.id,
                ],
            ).map_err(convert_rusqlite_error)?;

            if rows == 0 {
                return Err(crate::repositories::generic_repository::RepositoryError::InvalidData(
                    format!("Node not found: {}", node.id)
                ));
            }

            debug!("Updated node {} in unified database", node.id);
            Ok(())
        }).await.map_err(|e| match e {
            crate::repositories::generic_repository::RepositoryError::InvalidData(_) => KnowledgeGraphRepositoryError::NodeNotFound(node.id),
            _ => KnowledgeGraphRepositoryError::DatabaseError(format!("{}", e)),
        })
    }

    async fn batch_update_nodes(&self, nodes: Vec<Node>) -> RepoResult<()> {
        self.base.execute_transaction(|tx| {
            let mut stmt = tx.prepare(
                r#"
                UPDATE graph_nodes
                SET metadata_id = ?1, label = ?2, x = ?3, y = ?4, z = ?5,
                    vx = ?6, vy = ?7, vz = ?8, owl_class_iri = ?9,
                    color = ?10, size = ?11, node_type = ?12, weight = ?13,
                    group_name = ?14, metadata = ?15, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?16
                "#,
            ).map_err(convert_rusqlite_error)?;

            for node in &nodes {
                let owl_class_iri = node.metadata.get("owl_class_iri").cloned();
                let metadata_json = Self::serialize_metadata(&node.metadata);

                stmt.execute(params![
                    node.metadata_id, node.label,
                    node.data.x, node.data.y, node.data.z,
                    node.data.vx, node.data.vy, node.data.vz,
                    owl_class_iri, node.color, node.size,
                    node.node_type, node.weight, node.group,
                    metadata_json, node.id,
                ]).map_err(convert_rusqlite_error)?;
            }

            debug!("Batch updated {} nodes in unified database", nodes.len());
            Ok(())
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn remove_node(&self, node_id: u32) -> RepoResult<()> {
        self.base.execute_blocking(move |conn| {
            let rows = conn.execute("DELETE FROM graph_nodes WHERE id = ?1", params![node_id])
                .map_err(convert_rusqlite_error)?;

            if rows == 0 {
                return Err(crate::repositories::generic_repository::RepositoryError::InvalidData(
                    format!("Node not found: {}", node_id)
                ));
            }

            debug!("Removed node {} from unified database", node_id);
            Ok(())
        }).await.map_err(|e| match e {
            crate::repositories::generic_repository::RepositoryError::InvalidData(_) => KnowledgeGraphRepositoryError::NodeNotFound(node_id),
            _ => KnowledgeGraphRepositoryError::DatabaseError(format!("{}", e)),
        })
    }

    async fn batch_remove_nodes(&self, node_ids: Vec<u32>) -> RepoResult<()> {
        self.base.execute_transaction(|tx| {
            let mut stmt = tx.prepare("DELETE FROM graph_nodes WHERE id = ?1")
                .map_err(convert_rusqlite_error)?;

            for node_id in &node_ids {
                stmt.execute(params![node_id]).map_err(convert_rusqlite_error)?;
            }

            debug!("Batch removed {} nodes from unified database", node_ids.len());
            Ok(())
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn get_node(&self, node_id: u32) -> RepoResult<Option<Node>> {
        self.base.execute_blocking(move |conn| {
            conn.query_row(
                r#"
                SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                       mass, charge, owl_class_iri, color, size,
                       node_type, weight, group_name, metadata
                FROM graph_nodes
                WHERE id = ?1
                "#,
                params![node_id],
                Self::deserialize_node,
            ).optional()
            .map_err(|e| crate::repositories::generic_repository::RepositoryError::DatabaseError(e.to_string()))
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn get_nodes(&self, node_ids: Vec<u32>) -> RepoResult<Vec<Node>> {
        self.base.execute_blocking(move |conn| {
            let placeholders = node_ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
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

            let mut stmt = conn.prepare(&query).map_err(convert_rusqlite_error)?;
            let params: Vec<&dyn rusqlite::ToSql> = node_ids.iter().map(|id| id as &dyn rusqlite::ToSql).collect();

            let nodes = stmt.query_map(&params[..], Self::deserialize_node)
                .map_err(convert_rusqlite_error)?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(convert_rusqlite_error)?;

            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> RepoResult<Vec<Node>> {
        let metadata_id = metadata_id.to_string();
        self.base.execute_blocking(move |conn| {
            let mut stmt = conn.prepare(
                r#"
                SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                       mass, charge, owl_class_iri, color, size,
                       node_type, weight, group_name, metadata
                FROM graph_nodes
                WHERE metadata_id = ?1
                "#,
            ).map_err(convert_rusqlite_error)?;

            let nodes = stmt.query_map(params![metadata_id], Self::deserialize_node)
                .map_err(convert_rusqlite_error)?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(convert_rusqlite_error)?;

            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn search_nodes_by_label(&self, label: &str) -> RepoResult<Vec<Node>> {
        let search_pattern = format!("%{}%", label);
        self.base.execute_blocking(move |conn| {
            let mut stmt = conn.prepare(
                r#"
                SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                       mass, charge, owl_class_iri, color, size,
                       node_type, weight, group_name, metadata
                FROM graph_nodes
                WHERE label LIKE ?1
                "#,
            ).map_err(convert_rusqlite_error)?;

            let nodes = stmt.query_map(params![search_pattern], Self::deserialize_node)
                .map_err(convert_rusqlite_error)?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(convert_rusqlite_error)?;

            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    /// Domain-specific method: Get all nodes with a specific OWL class IRI
    async fn get_nodes_by_owl_class_iri(&self, owl_class_iri: &str) -> RepoResult<Vec<Node>> {
        let owl_class_iri = owl_class_iri.to_string();
        self.base.execute_blocking(move |conn| {
            let mut stmt = conn.prepare(
                r#"
                SELECT id, metadata_id, label, x, y, z, vx, vy, vz,
                       mass, charge, owl_class_iri, color, size,
                       node_type, weight, group_name, metadata
                FROM graph_nodes
                WHERE owl_class_iri = ?1
                "#,
            ).map_err(convert_rusqlite_error)?;

            let nodes = stmt.query_map(params![owl_class_iri], Self::deserialize_node)
                .map_err(convert_rusqlite_error)?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(convert_rusqlite_error)?;

            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    // Edge operations - using base repository methods
    async fn add_edge(&self, edge: &Edge) -> RepoResult<String> {
        let edge = edge.clone();
        self.base.execute_blocking(move |conn| {
            let metadata_json = edge.metadata.as_ref().and_then(|m| to_json(m).ok());

            conn.execute(
                "INSERT INTO graph_edges (id, source_id, target_id, weight, relation_type, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![edge.id, edge.source, edge.target, edge.weight, edge.edge_type, metadata_json],
            ).map_err(convert_rusqlite_error)?;

            debug!("Added edge {} to unified database", edge.id);
            Ok(edge.id.clone())
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn batch_add_edges(&self, edges: Vec<Edge>) -> RepoResult<Vec<String>> {
        let mut edge_ids = Vec::new();

        self.base.execute_transaction(|tx| {
            let mut stmt = tx.prepare(
                "INSERT INTO graph_edges (id, source_id, target_id, weight, relation_type, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            ).map_err(convert_rusqlite_error)?;

            for edge in &edges {
                let metadata_json = edge.metadata.as_ref().and_then(|m| to_json(m).ok());
                stmt.execute(params![edge.id, edge.source, edge.target, edge.weight, edge.edge_type, metadata_json])
                    .map_err(convert_rusqlite_error)?;
                edge_ids.push(edge.id.clone());
            }

            debug!("Batch added {} edges to unified database", edges.len());
            Ok(edge_ids)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn update_edge(&self, edge: &Edge) -> RepoResult<()> {
        let edge = edge.clone();
        self.base.execute_blocking(move |conn| {
            let metadata_json = edge.metadata.as_ref().and_then(|m| to_json(m).ok());

            let rows = conn.execute(
                "UPDATE graph_edges
                 SET source_id = ?1, target_id = ?2, weight = ?3, relation_type = ?4, metadata = ?5
                 WHERE id = ?6",
                params![edge.source, edge.target, edge.weight, edge.edge_type, metadata_json, edge.id],
            ).map_err(convert_rusqlite_error)?;

            if rows == 0 {
                return Err(crate::repositories::generic_repository::RepositoryError::InvalidData(
                    format!("Edge not found: {}", edge.id)
                ));
            }

            debug!("Updated edge {} in unified database", edge.id);
            Ok(())
        }).await.map_err(|e| match e {
            crate::repositories::generic_repository::RepositoryError::InvalidData(msg) => KnowledgeGraphRepositoryError::EdgeNotFound(msg),
            _ => KnowledgeGraphRepositoryError::DatabaseError(format!("{}", e)),
        })
    }

    async fn remove_edge(&self, edge_id: &str) -> RepoResult<()> {
        let edge_id = edge_id.to_string();
        self.base.execute_blocking(move |conn| {
            let rows = conn.execute("DELETE FROM graph_edges WHERE id = ?1", params![edge_id])
                .map_err(convert_rusqlite_error)?;

            if rows == 0 {
                return Err(crate::repositories::generic_repository::RepositoryError::InvalidData(
                    format!("Edge not found: {}", edge_id)
                ));
            }

            debug!("Removed edge {} from unified database", edge_id);
            Ok(())
        }).await.map_err(|e| match e {
            crate::repositories::generic_repository::RepositoryError::InvalidData(msg) => KnowledgeGraphRepositoryError::EdgeNotFound(msg),
            _ => KnowledgeGraphRepositoryError::DatabaseError(format!("{}", e)),
        })
    }

    async fn batch_remove_edges(&self, edge_ids: Vec<String>) -> RepoResult<()> {
        self.base.execute_transaction(|tx| {
            let mut stmt = tx.prepare("DELETE FROM graph_edges WHERE id = ?1")
                .map_err(convert_rusqlite_error)?;

            for edge_id in &edge_ids {
                stmt.execute(params![edge_id]).map_err(convert_rusqlite_error)?;
            }

            debug!("Batch removed {} edges from unified database", edge_ids.len());
            Ok(())
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn get_node_edges(&self, node_id: u32) -> RepoResult<Vec<Edge>> {
        self.base.execute_blocking(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, source_id, target_id, weight, relation_type, metadata
                 FROM graph_edges
                 WHERE source_id = ?1 OR target_id = ?1",
            ).map_err(convert_rusqlite_error)?;

            let edges = stmt.query_map(params![node_id], |row| {
                let id: String = row.get(0)?;
                let source: u32 = row.get(1)?;
                let target: u32 = row.get(2)?;
                let weight: f32 = row.get(3)?;
                let edge_type: Option<String> = row.get(4)?;
                let metadata_json: Option<String> = row.get(5)?;
                let metadata = metadata_json.and_then(|json| from_json(&json).ok());

                let mut edge = Edge::new(source, target, weight);
                edge.id = id;
                edge.edge_type = edge_type;
                edge.metadata = metadata;
                Ok(edge)
            }).map_err(convert_rusqlite_error)?
            .collect::<Result<Vec<Edge>, _>>()
            .map_err(convert_rusqlite_error)?;

            Ok(edges)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn get_edges_between(&self, source_id: u32, target_id: u32) -> RepoResult<Vec<Edge>> {
        self.base.execute_blocking(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, source_id, target_id, weight, relation_type, metadata
                 FROM graph_edges
                 WHERE (source_id = ?1 AND target_id = ?2) OR (source_id = ?2 AND target_id = ?1)",
            ).map_err(convert_rusqlite_error)?;

            let edges = stmt.query_map(params![source_id, target_id], |row| {
                let id: String = row.get(0)?;
                let source: u32 = row.get(1)?;
                let target: u32 = row.get(2)?;
                let weight: f32 = row.get(3)?;
                let edge_type: Option<String> = row.get(4)?;
                let metadata_json: Option<String> = row.get(5)?;
                let metadata = metadata_json.and_then(|json| from_json(&json).ok());

                let mut edge = Edge::new(source, target, weight);
                edge.id = id;
                edge.edge_type = edge_type;
                edge.metadata = metadata;
                Ok(edge)
            }).map_err(convert_rusqlite_error)?
            .collect::<Result<Vec<Edge>, _>>()
            .map_err(convert_rusqlite_error)?;

            Ok(edges)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    // Batch position update - using base repository transaction
    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> RepoResult<()> {
        self.base.execute_transaction(|tx| {
            let mut stmt = tx.prepare(
                "UPDATE graph_nodes
                 SET x = ?1, y = ?2, z = ?3, updated_at = CURRENT_TIMESTAMP
                 WHERE id = ?4",
            ).map_err(convert_rusqlite_error)?;

            for chunk in positions.chunks(10_000) {
                for (node_id, x, y, z) in chunk {
                    stmt.execute(params![x, y, z, node_id]).map_err(convert_rusqlite_error)?;
                }
            }

            debug!("Batch updated {} positions in unified database", positions.len());
            Ok(())
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn query_nodes(&self, _query: &str) -> RepoResult<Vec<Node>> {
        log::warn!("query_nodes not yet implemented for unified repository");
        Ok(Vec::new())
    }

    async fn get_neighbors(&self, node_id: u32) -> RepoResult<Vec<Node>> {
        self.base.execute_blocking(move |conn| {
            let mut stmt = conn.prepare(
                r#"
                SELECT DISTINCT n.id, n.metadata_id, n.label, n.x, n.y, n.z,
                       n.vx, n.vy, n.vz, n.mass, n.charge, n.owl_class_iri,
                       n.color, n.size, n.node_type, n.weight, n.group_name, n.metadata
                FROM graph_nodes n
                JOIN graph_edges e ON (e.source_id = ?1 AND e.target_id = n.id)
                                   OR (e.target_id = ?1 AND e.source_id = n.id)
                "#,
            ).map_err(convert_rusqlite_error)?;

            let nodes = stmt.query_map(params![node_id], Self::deserialize_node)
                .map_err(convert_rusqlite_error)?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(convert_rusqlite_error)?;

            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn get_statistics(&self) -> RepoResult<GraphStatistics> {
        self.base.execute_blocking(|conn| {
            // Try cache first
            let cached: Option<GraphStatistics> = conn.query_row(
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
                        last_updated: chrono::DateTime::from_timestamp(timestamp, 0).unwrap_or_else(chrono::Utc::now),
                    })
                },
            ).optional().ok().flatten();

            if let Some(stats) = cached {
                return Ok(stats);
            }

            // Calculate if not cached
            let node_count: i64 = conn.query_row("SELECT COUNT(*) FROM graph_nodes", [], |row| row.get(0)).unwrap_or(0);
            let edge_count: i64 = conn.query_row("SELECT COUNT(*) FROM graph_edges", [], |row| row.get(0)).unwrap_or(0);
            let average_degree = if node_count > 0 {
                (edge_count as f32 * 2.0) / node_count as f32
            } else {
                0.0
            };

            Ok(GraphStatistics {
                node_count: node_count as usize,
                edge_count: edge_count as usize,
                average_degree,
                connected_components: 1,
                last_updated: chrono::Utc::now(),
            })
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn clear_graph(&self) -> RepoResult<()> {
        self.base.execute_transaction(|tx| {
            tx.execute("DELETE FROM graph_edges", []).map_err(convert_rusqlite_error)?;
            tx.execute("DELETE FROM graph_nodes", []).map_err(convert_rusqlite_error)?;
            tx.execute("DELETE FROM graph_statistics", []).map_err(convert_rusqlite_error)?;
            info!("Cleared all graph data from unified database");
            Ok(())
        }).await.map_err(|e| KnowledgeGraphRepositoryError::AccessError(e.to_string()))
    }

    async fn health_check(&self) -> RepoResult<bool> {
        self.base.health_check().await
            .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::utils::json::{from_json, to_json};

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

        let mut node1 = Node::new("node1".to_string());
        node1.label = "Node 1".to_string();
        let id1 = repo.add_node(&node1).await.unwrap();

        let mut node2 = Node::new("node2".to_string());
        node2.label = "Node 2".to_string();
        let id2 = repo.add_node(&node2).await.unwrap();

        let positions = vec![(id1, 10.0, 20.0, 30.0), (id2, 40.0, 50.0, 60.0)];
        repo.batch_update_positions(positions).await.unwrap();

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
