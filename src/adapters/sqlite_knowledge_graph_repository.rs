// src/adapters/sqlite_knowledge_graph_repository.rs
//! SQLite Knowledge Graph Repository Adapter
//!
//! Implements the KnowledgeGraphRepository port using SQLite with batch operations
//! and efficient graph structure storage.

use async_trait::async_trait;
use rusqlite::{params, Connection};
use std::sync::Arc;
use tracing::{debug, info, instrument};

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::knowledge_graph_repository::{
    GraphStatistics, KnowledgeGraphRepository, KnowledgeGraphRepositoryError, Result as RepoResult,
};

/// SQLite-backed knowledge graph repository
pub struct SqliteKnowledgeGraphRepository {
    conn: Arc<tokio::sync::Mutex<Connection>>,
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
            conn: Arc::new(tokio::sync::Mutex::new(conn)),
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
        let conn = self.conn.lock().await;

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
    }

    #[instrument(skip(self, graph), fields(nodes = graph.nodes.len(), edges = graph.edges.len()), level = "debug")]
    async fn save_graph(&self, graph: &GraphData) -> RepoResult<()> {
        let conn = self.conn.lock().await;

        // Begin transaction
        conn.execute("BEGIN TRANSACTION", []).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!(
                "Failed to begin transaction: {}",
                e
            ))
        })?;

        // Clear existing data
        conn.execute("DELETE FROM kg_edges", []).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to clear edges: {}", e))
        })?;
        conn.execute("DELETE FROM kg_nodes", []).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to clear nodes: {}", e))
        })?;

        // Insert nodes
        let mut node_stmt = conn.prepare(
            "INSERT INTO kg_nodes (id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"
        ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare node insert: {}", e)))?;

        for node in &graph.nodes {
            let metadata_json = serde_json::to_string(&node.metadata).map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to serialize metadata: {}",
                    e
                ))
            })?;

            node_stmt
                .execute(params![
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
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to insert node: {}",
                        e
                    ))
                })?;
        }

        drop(node_stmt);

        // Insert edges
        let mut edge_stmt = conn.prepare(
            "INSERT INTO kg_edges (id, source, target, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)"
        ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare edge insert: {}", e)))?;

        for edge in &graph.edges {
            let metadata_json = edge
                .metadata
                .as_ref()
                .and_then(|m| serde_json::to_string(m).ok());

            edge_stmt
                .execute(params![
                    &edge.id,
                    edge.source,
                    edge.target,
                    edge.weight,
                    metadata_json,
                ])
                .map_err(|e| {
                    KnowledgeGraphRepositoryError::DatabaseError(format!(
                        "Failed to insert edge: {}",
                        e
                    ))
                })?;
        }

        // Commit transaction
        conn.execute("COMMIT", []).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!(
                "Failed to commit transaction: {}",
                e
            ))
        })?;

        info!(
            "Saved graph with {} nodes and {} edges",
            graph.nodes.len(),
            graph.edges.len()
        );

        Ok(())
    }

    async fn add_node(&self, node: &Node) -> RepoResult<u32> {
        let conn = self.conn.lock().await;
        let metadata_json = serde_json::to_string(&node.metadata).map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!(
                "Failed to serialize metadata: {}",
                e
            ))
        })?;

        conn.execute(
            "INSERT OR REPLACE INTO kg_nodes (id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata, updated_at)
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
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to insert node: {}", e)))?;

        Ok(node.id)
    }

    async fn update_node(&self, node: &Node) -> RepoResult<()> {
        self.add_node(node).await?;
        Ok(())
    }

    async fn remove_node(&self, node_id: u32) -> RepoResult<()> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM kg_nodes WHERE id = ?1", params![node_id])
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to delete node: {}",
                    e
                ))
            })?;
        Ok(())
    }

    async fn get_node(&self, node_id: u32) -> RepoResult<Option<Node>> {
        let conn = self.conn.lock().await;
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
    }

    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> RepoResult<Vec<Node>> {
        let conn = self.conn.lock().await;
        let mut stmt = conn.prepare(
            "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes WHERE metadata_id = ?1"
        ).map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;

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
    }

    async fn add_edge(&self, edge: &Edge) -> RepoResult<String> {
        let conn = self.conn.lock().await;
        let metadata_json = edge
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        conn.execute(
            "INSERT INTO kg_edges (id, source, target, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![&edge.id, edge.source, edge.target, edge.weight, metadata_json]
        )
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to insert edge: {}", e)))?;

        Ok(edge.id.clone())
    }

    async fn update_edge(&self, edge: &Edge) -> RepoResult<()> {
        let conn = self.conn.lock().await;
        let metadata_json = edge
            .metadata
            .as_ref()
            .and_then(|m| serde_json::to_string(m).ok());

        conn.execute(
            "UPDATE kg_edges SET source = ?1, target = ?2, weight = ?3, metadata = ?4 WHERE id = ?5",
            params![edge.source, edge.target, edge.weight, metadata_json, &edge.id]
        )
        .map_err(|e| KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to update edge: {}", e)))?;

        Ok(())
    }

    async fn remove_edge(&self, edge_id: &str) -> RepoResult<()> {
        let conn = self.conn.lock().await;
        conn.execute("DELETE FROM kg_edges WHERE id = ?1", params![edge_id])
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to delete edge: {}",
                    e
                ))
            })?;
        Ok(())
    }

    async fn get_node_edges(&self, node_id: u32) -> RepoResult<Vec<Edge>> {
        let conn = self.conn.lock().await;
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
    }

    #[instrument(skip(self, positions), fields(count = positions.len()), level = "debug")]
    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> RepoResult<()> {
        let conn = self.conn.lock().await;
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
    }

    async fn query_nodes(&self, query: &str) -> RepoResult<Vec<Node>> {
        let conn = self.conn.lock().await;

        let sql_query = format!(
            "SELECT id, metadata_id, label, x, y, z, vx, vy, vz, color, size, metadata FROM kg_nodes WHERE {}",
            query
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
    }

    async fn get_statistics(&self) -> RepoResult<GraphStatistics> {
        let conn = self.conn.lock().await;

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
    }
}
