// src/adapters/sqlite_knowledge_graph_repository.rs
//! SQLite Knowledge Graph Repository Adapter
//!
//! Implements the KnowledgeGraphRepository port using SQLite with batch operations
//! and efficient graph structure storage.

use async_trait::async_trait;
use log::{error, info};
use rusqlite::{params, Connection};
use std::sync::{Arc, Mutex};
use tracing::{debug, instrument};

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::knowledge_graph_repository::{
    GraphStatistics, KnowledgeGraphRepository, KnowledgeGraphRepositoryError, Result as RepoResult,
};

/// SQLite-backed knowledge graph repository
pub struct SqliteKnowledgeGraphRepository {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteKnowledgeGraphRepository {
    /// Create new SQLite knowledge graph repository
    pub fn new(db_path: &str) -> Result<Self, KnowledgeGraphRepositoryError> {
        let conn = Connection::open(db_path).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to open database: {}", e))
        })?;

        // Create schema
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS kg_nodes (
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
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_kg_nodes_metadata_id ON kg_nodes(metadata_id);

            CREATE TABLE IF NOT EXISTS kg_edges (
                id TEXT PRIMARY KEY,
                source INTEGER NOT NULL,
                target INTEGER NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source) REFERENCES kg_nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES kg_nodes(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source);
            CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target);

            CREATE TABLE IF NOT EXISTS kg_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        "#,
        )
        .map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to create schema: {}", e))
        })?;

        info!("Initialized SqliteKnowledgeGraphRepository at {}", db_path);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
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

        let metadata = serde_json::from_str(&metadata_json).unwrap_or_default();

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
    #[instrument(skip(self), level = "debug")]
    async fn load_graph(&self) -> RepoResult<Arc<GraphData>> {
        let conn_arc = self.conn.clone();

        // Move all synchronous database work to blocking thread pool
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");

            // Load all nodes
            let mut stmt = conn.prepare("SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes")
                .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;

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

            debug!("Loaded {} nodes from database", nodes.len());

            // Load all edges
            let mut edge_stmt = conn
                .prepare("SELECT id, source, target, weight, metadata FROM kg_edges")
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to prepare edge statement: {}",
                        e
                    ))
                })?;

            let edges = edge_stmt
                .query_map([], |row| {
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

            debug!("Loaded {} edges from database", edges.len());

            let mut graph = GraphData::new();
            graph.nodes = nodes;
            graph.edges = edges;

            Ok(Arc::new(graph))
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    #[instrument(skip(self, graph), fields(nodes = graph.nodes.len(), edges = graph.edges.len()), level = "debug")]
    async fn save_graph(&self, graph: &GraphData) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let graph_clone = graph.clone();

        tokio::task::spawn_blocking(move || {
            println!("=== SAVE_GRAPH START ===");
            println!("Input: {} nodes, {} edges", graph_clone.nodes.len(), graph_clone.edges.len());

            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");

            // Begin transaction
            println!("Beginning transaction...");
            conn.execute("BEGIN TRANSACTION", []).map_err(|e| {
                println!("ERROR: Failed to begin transaction: {}", e);
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;
            println!("Transaction started");

            // Clear existing data
            println!("Clearing edges...");
            let edges_deleted = conn.execute("DELETE FROM kg_edges", []).map_err(|e| {
                println!("ERROR: Failed to clear edges: {}", e);
                KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to clear edges: {}", e))
            })?;
            println!("Deleted {} edges", edges_deleted);

            println!("Clearing nodes...");
            let nodes_deleted = conn.execute("DELETE FROM kg_nodes", []).map_err(|e| {
                println!("ERROR: Failed to clear nodes: {}", e);
                KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to clear nodes: {}", e))
            })?;
            println!("Deleted {} nodes", nodes_deleted);

            // Insert nodes (use INSERT OR REPLACE to handle any duplicates)
            println!("Preparing node insert statement...");
            let mut node_stmt = conn.prepare(
                "INSERT OR REPLACE INTO kg_nodes (id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"
            ).map_err(|e| {
                println!("ERROR: Failed to prepare node insert: {}", e);
                KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare node insert: {}", e))
            })?;
            println!("Node statement prepared");

            println!("Inserting {} nodes...", graph_clone.nodes.len());
            for (idx, node) in graph_clone.nodes.iter().enumerate() {
                let metadata_json = serde_json::to_string(&node.metadata).map_err(|e| {
                    println!("ERROR: Failed to serialize metadata for node {}: {}", idx, e);
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to serialize metadata: {}",
                        e
                    ))
                })?;

                match node_stmt.execute(params![
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
                ]) {
                    Ok(_) => {},
                    Err(e) => {
                        println!("ERROR: Failed to insert node {} (id={}, metadata_id={})", idx, node.id, node.metadata_id);
                        println!("  Node details: label='{}', x={}, y={}, z={}", node.label, node.data.x, node.data.y, node.data.z);
                        println!("  Error: {}", e);

                        // Extract detailed SQLite error information
                        match &e {
                            rusqlite::Error::SqliteFailure(err, msg) => {
                                println!("  SQLite Error Code: {:?}", err.code);
                                println!("  SQLite Extended Code: {:?}", err.extended_code);
                                if let Some(m) = msg {
                                    println!("  SQLite Message: {}", m);
                                }
                            }
                            _ => {
                                println!("  Error Type: {:?}", e);
                            }
                        }

                        // Check if this is a UNIQUE constraint failure
                        let err_str = format!("{}", e);
                        if err_str.contains("UNIQUE constraint failed") {
                            println!("  >>> UNIQUE CONSTRAINT VIOLATION <<<");
                            println!("  >>> Duplicate node ID {} <<<", node.id);
                        }

                        return Err(KnowledgeGraphRepositoryError::DatabaseError(format!(
                            "Failed to insert node {}: {}",
                            idx, e
                        )));
                    }
                }
            }
            println!("All nodes inserted successfully");

            drop(node_stmt);

            // Insert edges
            println!("Preparing edge insert statement...");
            let mut edge_stmt = conn.prepare(
                "INSERT INTO kg_edges (id, source, target, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)"
            ).map_err(|e| {
                println!("ERROR: Failed to prepare edge insert: {}", e);
                KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare edge insert: {}", e))
            })?;
            println!("Edge statement prepared");

            println!("Inserting {} edges...", graph_clone.edges.len());
            for (idx, edge) in graph_clone.edges.iter().enumerate() {
                let metadata_json = edge
                    .metadata
                    .as_ref()
                    .and_then(|m| serde_json::to_string(m).ok());

                match edge_stmt.execute(params![
                    &edge.id,
                    edge.source,
                    edge.target,
                    edge.weight,
                    metadata_json,
                ]) {
                    Ok(_) => {},
                    Err(e) => {
                        println!("ERROR: Failed to insert edge {} (id={}, source={}, target={})",
                            idx, edge.id, edge.source, edge.target);
                        println!("  Edge details: source={}, target={}, weight={}", edge.source, edge.target, edge.weight);
                        println!("  Error: {}", e);

                        // Extract detailed SQLite error information
                        match &e {
                            rusqlite::Error::SqliteFailure(err, msg) => {
                                println!("  SQLite Error Code: {:?}", err.code);
                                println!("  SQLite Extended Code: {:?}", err.extended_code);
                                if let Some(m) = msg {
                                    println!("  SQLite Message: {}", m);
                                }
                            }
                            rusqlite::Error::QueryReturnedNoRows => {
                                println!("  Error Type: QueryReturnedNoRows");
                            }
                            rusqlite::Error::InvalidColumnIndex(i) => {
                                println!("  Error Type: InvalidColumnIndex({})", i);
                            }
                            rusqlite::Error::InvalidColumnName(name) => {
                                println!("  Error Type: InvalidColumnName({})", name);
                            }
                            _ => {
                                println!("  Error Type: {:?}", e);
                            }
                        }

                        // Check if this is a foreign key constraint failure
                        let err_str = format!("{}", e);
                        if err_str.contains("FOREIGN KEY constraint failed") {
                            println!("  >>> FOREIGN KEY CONSTRAINT VIOLATION <<<");
                            println!("  >>> Attempting to reference non-existent node <<<");
                            println!("  >>> Source node {} or target node {} doesn't exist <<<", edge.source, edge.target);
                        }

                        return Err(KnowledgeGraphRepositoryError::DatabaseError(format!(
                            "Failed to insert edge {}: {}",
                            idx, e
                        )));
                    }
                }
            }
            println!("All edges inserted successfully");

            // Commit transaction
            println!("Committing transaction...");
            conn.execute("COMMIT", []).map_err(|e| {
                println!("ERROR: Failed to commit transaction: {}", e);
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;
            println!("Transaction committed");

            info!(
                "Saved graph with {} nodes and {} edges",
                graph_clone.nodes.len(),
                graph_clone.edges.len()
            );

            println!("=== SAVE_GRAPH COMPLETE ===");

            Ok(())
        })
        .await
        .map_err(|e| {
            println!("ERROR: Join error in save_graph: {}", e);
            KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e))
        })?
    }

    async fn add_node(&self, node: &Node) -> RepoResult<u32> {
        let conn_arc = self.conn.clone();
        let node_clone = node.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let metadata_json = serde_json::to_string(&node_clone.metadata).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to serialize metadata: {}",
                    e
                ))
            })?;

            conn.execute(
                "INSERT OR REPLACE INTO kg_nodes (id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, CURRENT_TIMESTAMP)",
                params![
                    node_clone.id,
                    &node_clone.metadata_id,
                    &node_clone.label,
                    node_clone.data.x,
                    node_clone.data.y,
                    node_clone.data.z,
                    node_clone.data.vx,
                    node_clone.data.vy,
                    node_clone.data.vz,
                    &node_clone.color,
                    &node_clone.size,
                    metadata_json,
                ]
            )
            .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to insert node: {}", e)))?;

            Ok(node_clone.id)
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn update_node(&self, node: &Node) -> RepoResult<()> {
        self.add_node(node).await?;
        Ok(())
    }

    async fn remove_node(&self, node_id: u32) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            conn.execute("DELETE FROM kg_nodes WHERE id = ?1", params![node_id])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to delete node: {}",
                        e
                    ))
                })?;
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_node(&self, node_id: u32) -> RepoResult<Option<Node>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let result = conn.query_row(
                "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes WHERE id = ?1",
                params![node_id],
                Self::deserialize_node
            );

            match result {
                Ok(node) => Ok(Some(node)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Database error: {}",
                    e
                ))),
            }
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();
        let metadata_id_owned = metadata_id.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut stmt = conn.prepare(
                "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes WHERE metadata_id = ?1"
            ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;

            let nodes = stmt
                .query_map(params![metadata_id_owned], Self::deserialize_node)
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
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn add_edge(&self, edge: &Edge) -> RepoResult<String> {
        let conn_arc = self.conn.clone();
        let edge_clone = edge.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let metadata_json = edge_clone
                .metadata
                .as_ref()
                .and_then(|m| serde_json::to_string(m).ok());

            conn.execute(
                "INSERT INTO kg_edges (id, source, target, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![&edge_clone.id, edge_clone.source, edge_clone.target, edge_clone.weight, metadata_json]
            )
            .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to insert edge: {}", e)))?;

            Ok(edge_clone.id.clone())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn update_edge(&self, edge: &Edge) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let edge_clone = edge.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let metadata_json = edge_clone
                .metadata
                .as_ref()
                .and_then(|m| serde_json::to_string(m).ok());

            conn.execute(
                "UPDATE kg_edges SET source = ?1, target = ?2, weight = ?3, metadata = ?4 WHERE id = ?5",
                params![edge_clone.source, edge_clone.target, edge_clone.weight, metadata_json, &edge_clone.id]
            )
            .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to update edge: {}", e)))?;

            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn remove_edge(&self, edge_id: &str) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let edge_id_owned = edge_id.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            conn.execute("DELETE FROM kg_edges WHERE id = ?1", params![edge_id_owned])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to delete edge: {}",
                        e
                    ))
                })?;
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_node_edges(&self, node_id: u32) -> RepoResult<Vec<Edge>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut stmt = conn.prepare(
                "SELECT id, source, target, weight, metadata FROM kg_edges WHERE source = ?1 OR target = ?1"
            ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;

            let edges = stmt
                .query_map(params![node_id], |row| {
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
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    #[instrument(skip(self, positions), fields(count = positions.len()), level = "debug")]
    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> RepoResult<()> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            conn.execute("BEGIN TRANSACTION", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            let mut stmt = conn.prepare("UPDATE kg_nodes SET x = ?2, y = ?3, z = ?4, updated_at = CURRENT_TIMESTAMP WHERE id = ?1")
                .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;

            for (node_id, x, y, z) in positions {
                stmt.execute(params![node_id, x, y, z]).map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to update position: {}",
                        e
                    ))
                })?;
            }

            conn.execute("COMMIT", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;

            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn query_nodes(&self, query: &str) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();
        let query_owned = query.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");

            let sql_query = format!(
                "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes WHERE {}",
                query_owned
            );

            let mut stmt = conn.prepare(&sql_query).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare query: {}", e))
            })?;

            let nodes = stmt
                .query_map([], Self::deserialize_node)
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to execute query: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to collect results: {}",
                        e
                    ))
                })?;

            Ok(nodes)
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_statistics(&self) -> RepoResult<GraphStatistics> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");

            let node_count: usize = conn
                .query_row("SELECT COUNT(*) FROM kg_nodes", [], |row| row.get(0))
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to count nodes: {}",
                        e
                    ))
                })?;

            let edge_count: usize = conn
                .query_row("SELECT COUNT(*) FROM kg_edges", [], |row| row.get(0))
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

            Ok(GraphStatistics {
                node_count,
                edge_count,
                average_degree,
                connected_components: 1, // TODO: Implement connected components analysis
                last_updated: chrono::Utc::now(),
            })
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn batch_add_nodes(&self, nodes: Vec<Node>) -> RepoResult<Vec<u32>> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut ids = Vec::with_capacity(nodes.len());
            for node in nodes {
                let metadata_json = serde_json::to_string(&node.metadata)
                    .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to serialize metadata: {}", e)))?;
                conn.execute(
                    "INSERT OR REPLACE INTO kg_nodes (id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, CURRENT_TIMESTAMP)",
                    params![node.id, &node.metadata_id, &node.label, node.data.x, node.data.y, node.data.z,
                            node.data.vx, node.data.vy, node.data.vz, &node.color, &node.size, metadata_json]
                ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to insert node: {}", e)))?;
                ids.push(node.id);
            }
            Ok(ids)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn batch_update_nodes(&self, nodes: Vec<Node>) -> RepoResult<()> {
        self.batch_add_nodes(nodes).await?;
        Ok(())
    }

    async fn batch_remove_nodes(&self, node_ids: Vec<u32>) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            for node_id in node_ids {
                conn.execute("DELETE FROM kg_nodes WHERE id = ?1", params![node_id])
                    .map_err(|e| {
                        KnowledgeGraphRepositoryError::DatabaseError(format!(
                            "Failed to delete node: {}",
                            e
                        ))
                    })?;
            }
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_nodes(&self, node_ids: Vec<u32>) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut nodes = Vec::new();
            for node_id in node_ids {
                match conn.query_row(
                    "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes WHERE id = ?1",
                    params![node_id], Self::deserialize_node
                ) {
                    Ok(node) => nodes.push(node),
                    Err(rusqlite::Error::QueryReturnedNoRows) => continue,
                    Err(e) => return Err(KnowledgeGraphRepositoryError::DatabaseError(format!("Database error: {}", e))),
                }
            }
            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn search_nodes_by_label(&self, label: &str) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();
        let label_owned = label.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut stmt = conn.prepare(
                "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes WHERE label LIKE ?1"
            ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;
            let nodes = stmt.query_map(params![format!("%{}%", label_owned)], Self::deserialize_node)
                .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to query nodes: {}", e)))?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to collect nodes: {}", e)))?;
            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn batch_add_edges(&self, edges: Vec<Edge>) -> RepoResult<Vec<String>> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut ids = Vec::with_capacity(edges.len());
            for edge in edges {
                let metadata_json = edge.metadata.as_ref().and_then(|m| serde_json::to_string(m).ok());
                conn.execute(
                    "INSERT INTO kg_edges (id, source, target, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![&edge.id, edge.source, edge.target, edge.weight, metadata_json]
                ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to insert edge: {}", e)))?;
                ids.push(edge.id.clone());
            }
            Ok(ids)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn batch_remove_edges(&self, edge_ids: Vec<String>) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            for edge_id in edge_ids {
                conn.execute("DELETE FROM kg_edges WHERE id = ?1", params![edge_id])
                    .map_err(|e| {
                        KnowledgeGraphRepositoryError::DatabaseError(format!(
                            "Failed to delete edge: {}",
                            e
                        ))
                    })?;
            }
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_edges_between(&self, source_id: u32, target_id: u32) -> RepoResult<Vec<Edge>> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut stmt = conn.prepare(
                "SELECT id, source, target, weight, metadata FROM kg_edges WHERE source = ?1 AND target = ?2"
            ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;
            let edges = stmt.query_map(params![source_id, target_id], |row| {
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
            }).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to query edges: {}", e)))?
            .collect::<Result<Vec<Edge>, _>>()
            .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to collect edges: {}", e)))?;
            Ok(edges)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_neighbors(&self, node_id: u32) -> RepoResult<Vec<Node>> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire knowledge graph repository mutex");
            let mut stmt = conn.prepare(
                "SELECT DISTINCT n.id, n.metadata_id, n.label, n.x, n.y, n.z, n.vx, n.vy, n.vz, n.color, n.size, n.metadata
                 FROM kg_nodes n JOIN kg_edges e ON (n.id = e.target AND e.source = ?1) OR (n.id = e.source AND e.target = ?1)"
            ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;
            let nodes = stmt.query_map(params![node_id], Self::deserialize_node)
                .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to query neighbors: {}", e)))?
                .collect::<Result<Vec<Node>, _>>()
                .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to collect neighbors: {}", e)))?;
            Ok(nodes)
        }).await.map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn clear_graph(&self) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            conn.execute("DELETE FROM kg_edges", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to clear edges: {}",
                    e
                ))
            })?;
            conn.execute("DELETE FROM kg_nodes", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to clear nodes: {}",
                    e
                ))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn begin_transaction(&self) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            conn.execute("BEGIN TRANSACTION", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn commit_transaction(&self) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            conn.execute("COMMIT", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to commit transaction: {}",
                    e
                ))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn rollback_transaction(&self) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            conn.execute("ROLLBACK", []).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to rollback transaction: {}",
                    e
                ))
            })?;
            Ok(())
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn health_check(&self) -> RepoResult<bool> {
        let conn_arc = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire knowledge graph repository mutex");
            match conn.query_row("SELECT 1", [], |_| Ok(())) {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        })
        .await
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }
}
