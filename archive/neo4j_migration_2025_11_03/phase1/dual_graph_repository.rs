// src/adapters/dual_graph_repository.rs
//! Dual Graph Repository
//!
//! Writes to both SQLite (unified.db) and Neo4j simultaneously for:
//! - SQLite: Fast local queries, physics state persistence
//! - Neo4j: Complex graph traversals, multi-hop reasoning, Cypher queries
//!
//! Strategy:
//! - Primary: SQLite (unified.db) - source of truth for positions/velocities
//! - Secondary: Neo4j - graph analytics and semantic queries
//! - Partial failures: Log errors but don't fail the operation
//! - Sync status: Track last sync timestamp

use async_trait::async_trait;
use log::{debug, error, info, warn};
use std::sync::Arc;
use tracing::instrument;

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::knowledge_graph_repository::{
    GraphStatistics, KnowledgeGraphRepository, KnowledgeGraphRepositoryError,
    Result as RepoResult,
};
use crate::repositories::unified_graph_repository::UnifiedGraphRepository;
use super::neo4j_adapter::Neo4jAdapter;

/// Dual-write repository wrapping SQLite and Neo4j
///
/// Ensures data consistency across both databases while providing
/// the best of both worlds:
/// - SQLite: High-speed local queries and physics integration
/// - Neo4j: Advanced graph analytics and semantic reasoning
pub struct DualGraphRepository {
    /// Primary repository (SQLite unified.db)
    primary: Arc<UnifiedGraphRepository>,

    /// Secondary repository (Neo4j)
    secondary: Option<Arc<Neo4jAdapter>>,

    /// Fail on Neo4j errors (default: false - log and continue)
    strict_mode: bool,
}

impl DualGraphRepository {
    /// Create a new DualGraphRepository
    ///
    /// # Arguments
    /// * `primary` - UnifiedGraphRepository (SQLite)
    /// * `secondary` - Optional Neo4jAdapter
    /// * `strict_mode` - If true, fail on Neo4j errors; if false, log and continue
    pub fn new(
        primary: Arc<UnifiedGraphRepository>,
        secondary: Option<Arc<Neo4jAdapter>>,
        strict_mode: bool,
    ) -> Self {
        if secondary.is_some() {
            info!("🔗 DualGraphRepository initialized with Neo4j support");
            info!("   Strict mode: {}", strict_mode);
        } else {
            info!("📦 DualGraphRepository initialized (SQLite only)");
        }

        Self {
            primary,
            secondary,
            strict_mode,
        }
    }

    /// Execute operation on both repositories
    ///
    /// Strategy:
    /// 1. Execute on primary (SQLite) - MUST succeed
    /// 2. Execute on secondary (Neo4j) - MAY fail in non-strict mode
    /// 3. Log any secondary failures
    async fn dual_write<F, T>(
        &self,
        operation_name: &str,
        primary_op: F,
        secondary_op: impl std::future::Future<Output = RepoResult<T>>,
    ) -> RepoResult<T>
    where
        F: std::future::Future<Output = RepoResult<T>>,
    {
        // Execute primary operation (MUST succeed)
        let primary_result = primary_op.await?;

        // Execute secondary operation if Neo4j is configured
        if self.secondary.is_some() {
            match secondary_op.await {
                Ok(_) => {
                    debug!("✅ {}: synced to Neo4j", operation_name);
                }
                Err(e) => {
                    if self.strict_mode {
                        error!("❌ {}: Neo4j failed in strict mode: {}", operation_name, e);
                        return Err(e);
                    } else {
                        warn!("⚠️  {}: Neo4j failed (non-strict mode): {}", operation_name, e);
                        // Continue without failing
                    }
                }
            }
        }

        Ok(primary_result)
    }
}

#[async_trait]
impl KnowledgeGraphRepository for DualGraphRepository {
    #[instrument(skip(self), level = "debug")]
    async fn load_graph(&self) -> RepoResult<Arc<GraphData>> {
        // Always load from primary (SQLite)
        self.primary.load_graph().await
    }

    async fn save_graph(&self, graph: &GraphData) -> RepoResult<()> {
        let graph_clone = graph.clone();

        self.dual_write(
            "save_graph",
            self.primary.save_graph(graph),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.save_graph(&graph_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn add_node(&self, node: &Node) -> RepoResult<u32> {
        let node_clone = node.clone();

        self.dual_write(
            "add_node",
            self.primary.add_node(node),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.add_node(&node_clone).await
                } else {
                    Ok(0)
                }
            },
        ).await
    }

    async fn batch_add_nodes(&self, nodes: Vec<Node>) -> RepoResult<Vec<u32>> {
        let nodes_clone = nodes.clone();

        self.dual_write(
            "batch_add_nodes",
            self.primary.batch_add_nodes(nodes),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.batch_add_nodes(nodes_clone).await
                } else {
                    Ok(Vec::new())
                }
            },
        ).await
    }

    async fn update_node(&self, node: &Node) -> RepoResult<()> {
        let node_clone = node.clone();

        self.dual_write(
            "update_node",
            self.primary.update_node(node),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.update_node(&node_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn batch_update_nodes(&self, nodes: Vec<Node>) -> RepoResult<()> {
        let nodes_clone = nodes.clone();

        self.dual_write(
            "batch_update_nodes",
            self.primary.batch_update_nodes(nodes),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.batch_update_nodes(nodes_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn remove_node(&self, node_id: u32) -> RepoResult<()> {
        self.dual_write(
            "remove_node",
            self.primary.remove_node(node_id),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.remove_node(node_id).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn batch_remove_nodes(&self, node_ids: Vec<u32>) -> RepoResult<()> {
        let node_ids_clone = node_ids.clone();

        self.dual_write(
            "batch_remove_nodes",
            self.primary.batch_remove_nodes(node_ids),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.batch_remove_nodes(node_ids_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn get_node(&self, node_id: u32) -> RepoResult<Option<Node>> {
        // Always read from primary (SQLite)
        self.primary.get_node(node_id).await
    }

    async fn get_nodes(&self, node_ids: Vec<u32>) -> RepoResult<Vec<Node>> {
        // Always read from primary (SQLite)
        self.primary.get_nodes(node_ids).await
    }

    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> RepoResult<Vec<Node>> {
        // Always read from primary (SQLite)
        self.primary.get_nodes_by_metadata_id(metadata_id).await
    }

    async fn search_nodes_by_label(&self, label: &str) -> RepoResult<Vec<Node>> {
        // Always read from primary (SQLite)
        self.primary.search_nodes_by_label(label).await
    }

    async fn add_edge(&self, edge: &Edge) -> RepoResult<String> {
        let edge_clone = edge.clone();

        self.dual_write(
            "add_edge",
            self.primary.add_edge(edge),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.add_edge(&edge_clone).await
                } else {
                    Ok(String::new())
                }
            },
        ).await
    }

    async fn batch_add_edges(&self, edges: Vec<Edge>) -> RepoResult<Vec<String>> {
        let edges_clone = edges.clone();

        self.dual_write(
            "batch_add_edges",
            self.primary.batch_add_edges(edges),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.batch_add_edges(edges_clone).await
                } else {
                    Ok(Vec::new())
                }
            },
        ).await
    }

    async fn update_edge(&self, edge: &Edge) -> RepoResult<()> {
        let edge_clone = edge.clone();

        self.dual_write(
            "update_edge",
            self.primary.update_edge(edge),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.update_edge(&edge_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn remove_edge(&self, edge_id: &str) -> RepoResult<()> {
        let edge_id_clone = edge_id.to_string();

        self.dual_write(
            "remove_edge",
            self.primary.remove_edge(edge_id),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.remove_edge(&edge_id_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn batch_remove_edges(&self, edge_ids: Vec<String>) -> RepoResult<()> {
        let edge_ids_clone = edge_ids.clone();

        self.dual_write(
            "batch_remove_edges",
            self.primary.batch_remove_edges(edge_ids),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.batch_remove_edges(edge_ids_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn get_node_edges(&self, node_id: u32) -> RepoResult<Vec<Edge>> {
        // Always read from primary (SQLite)
        self.primary.get_node_edges(node_id).await
    }

    async fn get_edges_between(&self, source_id: u32, target_id: u32) -> RepoResult<Vec<Edge>> {
        // Always read from primary (SQLite)
        self.primary.get_edges_between(source_id, target_id).await
    }

    async fn batch_update_positions(
        &self,
        positions: Vec<(u32, f32, f32, f32)>,
    ) -> RepoResult<()> {
        let positions_clone = positions.clone();

        self.dual_write(
            "batch_update_positions",
            self.primary.batch_update_positions(positions),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.batch_update_positions(positions_clone).await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn query_nodes(&self, query: &str) -> RepoResult<Vec<Node>> {
        // Always query from primary (SQLite)
        self.primary.query_nodes(query).await
    }

    async fn get_neighbors(&self, node_id: u32) -> RepoResult<Vec<Node>> {
        // Always read from primary (SQLite)
        self.primary.get_neighbors(node_id).await
    }

    async fn get_statistics(&self) -> RepoResult<GraphStatistics> {
        // Always read from primary (SQLite)
        self.primary.get_statistics().await
    }

    async fn clear_graph(&self) -> RepoResult<()> {
        self.dual_write(
            "clear_graph",
            self.primary.clear_graph(),
            async {
                if let Some(ref neo4j) = self.secondary {
                    neo4j.clear_graph().await
                } else {
                    Ok(())
                }
            },
        ).await
    }

    async fn begin_transaction(&self) -> RepoResult<()> {
        self.primary.begin_transaction().await
    }

    async fn commit_transaction(&self) -> RepoResult<()> {
        self.primary.commit_transaction().await
    }

    async fn rollback_transaction(&self) -> RepoResult<()> {
        self.primary.rollback_transaction().await
    }

    async fn health_check(&self) -> RepoResult<bool> {
        let primary_ok = self.primary.health_check().await?;

        if let Some(ref neo4j) = self.secondary {
            let secondary_ok = neo4j.health_check().await.unwrap_or(false);

            if !secondary_ok {
                warn!("⚠️  Neo4j health check failed");
            }

            // In non-strict mode, primary health is sufficient
            Ok(primary_ok && (secondary_ok || !self.strict_mode))
        } else {
            Ok(primary_ok)
        }
    }

    async fn get_nodes_by_owl_class_iri(&self, owl_class_iri: &str) -> RepoResult<Vec<Node>> {
        // Read operations only use primary (SQLite) for consistency
        self.primary.get_nodes_by_owl_class_iri(owl_class_iri).await
    }
}
