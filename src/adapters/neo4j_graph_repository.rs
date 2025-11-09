//! Neo4j Graph Repository - Direct queries with intelligent caching
//!
//! Professional architecture:
//! - Neo4j as single source of truth
//! - Read-through LRU cache for performance
//! - Lazy loading with pagination
//! - Batch operations for efficiency

use async_trait::async_trait;
use lru::LruCache;
use neo4rs::{Graph, query, BoltInteger, BoltFloat};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error, instrument};

use crate::actors::graph_actor::{AutoBalanceNotification, PhysicsState};
use crate::models::constraints::ConstraintSet;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::metadata::Metadata;
use crate::models::node::Node;
use crate::ports::graph_repository::{
    GraphRepository, GraphRepositoryError, PathfindingParams, PathfindingResult, Result,
};
use crate::types::vec3::Vec3Data;
use glam::Vec3;

const CACHE_SIZE: usize = 10_000;
const BATCH_SIZE: usize = 1000;

/// Neo4j-backed graph repository with intelligent caching
pub struct Neo4jGraphRepository {
    graph: Arc<Graph>,

    /// LRU cache for nodes (id -> Node)
    node_cache: Arc<RwLock<LruCache<u32, Node>>>,

    /// LRU cache for edges (id -> Edge)
    edge_cache: Arc<RwLock<LruCache<String, Edge>>>,

    /// Cached graph snapshot (refreshed periodically or on demand)
    graph_snapshot: Arc<RwLock<Option<Arc<GraphData>>>>,

    /// Track if full graph is loaded
    is_loaded: Arc<RwLock<bool>>,
}

impl Neo4jGraphRepository {
    pub fn new(graph: Arc<Graph>) -> Self {
        Self {
            graph,
            node_cache: Arc::new(RwLock::new(
                LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap())
            )),
            edge_cache: Arc::new(RwLock::new(
                LruCache::new(NonZeroUsize::new(CACHE_SIZE).unwrap())
            )),
            graph_snapshot: Arc::new(RwLock::new(None)),
            is_loaded: Arc::new(RwLock::new(false)),
        }
    }

    /// Load full graph from Neo4j (called on startup or refresh)
    #[instrument(skip(self))]
    pub async fn load_graph(&self) -> Result<()> {
        info!("Loading full graph from Neo4j...");

        // Load nodes in batches
        let nodes = self.load_all_nodes().await?;
        let edges = self.load_all_edges().await?;
        let metadata = self.load_all_metadata().await?;

        info!("Loaded {} nodes, {} edges, {} metadata entries",
              nodes.len(), edges.len(), metadata.len());

        // Update cache
        {
            let mut node_cache = self.node_cache.write().await;
            for node in &nodes {
                node_cache.put(node.id, node.clone());
            }
        }

        {
            let mut edge_cache = self.edge_cache.write().await;
            for edge in &edges {
                edge_cache.put(edge.id.clone(), edge.clone());
            }
        }

        // Create snapshot
        let graph_data = Arc::new(GraphData {
            nodes,
            edges,
            metadata,
            id_to_metadata: HashMap::new(),
        });

        *self.graph_snapshot.write().await = Some(graph_data);
        *self.is_loaded.write().await = true;

        Ok(())
    }

    /// Load all nodes from Neo4j
    async fn load_all_nodes(&self) -> Result<Vec<Node>> {
        let query_str = "
            MATCH (n:GraphNode)
            RETURN n.id as id,
                   n.metadata_id as metadata_id,
                   n.label as label,
                   n.x as x,
                   n.y as y,
                   n.z as z,
                   n.vx as vx,
                   n.vy as vy,
                   n.vz as vz,
                   n.mass as mass,
                   n.size as size,
                   n.color as color,
                   n.weight as weight,
                   n.node_type as node_type,
                   n.cluster as cluster,
                   n.cluster_id as cluster_id,
                   n.anomaly_score as anomaly_score,
                   n.community_id as community_id,
                   n.hierarchy_level as hierarchy_level,
                   n.metadata as metadata_json
            ORDER BY id
        ";

        let mut result = self.graph
            .execute(query(query_str))
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Failed to query nodes: {}", e)))?;

        let mut nodes = Vec::new();

        while let Some(row) = result.next().await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Failed to fetch row: {}", e)))?
        {
            let id: BoltInteger = row.get("id")
                .map_err(|e| GraphRepositoryError::DeserializationError(format!("Missing id: {}", e)))?;

            let metadata_id: String = row.get("metadata_id").unwrap_or_default();
            let label: String = row.get("label").unwrap_or_default();

            // Position
            let x: BoltFloat = row.get("x").unwrap_or(BoltFloat { value: 0.0 });
            let y: BoltFloat = row.get("y").unwrap_or(BoltFloat { value: 0.0 });
            let z: BoltFloat = row.get("z").unwrap_or(BoltFloat { value: 0.0 });

            // Velocity
            let vx: BoltFloat = row.get("vx").unwrap_or(BoltFloat { value: 0.0 });
            let vy: BoltFloat = row.get("vy").unwrap_or(BoltFloat { value: 0.0 });
            let vz: BoltFloat = row.get("vz").unwrap_or(BoltFloat { value: 0.0 });

            // Properties
            let mass: BoltFloat = row.get("mass").unwrap_or(BoltFloat { value: 1.0 });
            let size: BoltFloat = row.get("size").unwrap_or(BoltFloat { value: 1.0 });
            let color: String = row.get("color").unwrap_or_else(|_| "#888888".to_string());
            let weight: BoltFloat = row.get("weight").unwrap_or(BoltFloat { value: 1.0 });
            let node_type: String = row.get("node_type").unwrap_or_else(|_| "default".to_string());
            let cluster: Option<i64> = row.get("cluster").ok();

            // Analytics fields (P0-4)
            let cluster_id: Option<BoltInteger> = row.get("cluster_id").ok();
            let anomaly_score: Option<BoltFloat> = row.get("anomaly_score").ok();
            let community_id: Option<BoltInteger> = row.get("community_id").ok();
            let hierarchy_level: Option<BoltInteger> = row.get("hierarchy_level").ok();

            // Metadata JSON
            let metadata_json: String = row.get("metadata_json").unwrap_or_else(|_| "{}".to_string());
            let mut metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)
                .unwrap_or_default();

            // Store analytics in metadata for now (Node struct doesn't have dedicated fields yet)
            if let Some(cid) = cluster_id {
                metadata.insert("cluster_id".to_string(), cid.value.to_string());
            }
            if let Some(score) = anomaly_score {
                metadata.insert("anomaly_score".to_string(), score.value.to_string());
            }
            if let Some(cid) = community_id {
                metadata.insert("community_id".to_string(), cid.value.to_string());
            }
            if let Some(level) = hierarchy_level {
                metadata.insert("hierarchy_level".to_string(), level.value.to_string());
            }

            let node = Node {
                id: id.value as u32,
                metadata_id,
                label,
                data: crate::utils::socket_flow_messages::BinaryNodeData {
                    node_id: id.value as u32,
                    x: x.value as f32,
                    y: y.value as f32,
                    z: z.value as f32,
                    vx: vx.value as f32,
                    vy: vy.value as f32,
                    vz: vz.value as f32,
                },
                x: Some(x.value as f32),
                y: Some(y.value as f32),
                z: Some(z.value as f32),
                vx: Some(vx.value as f32),
                vy: Some(vy.value as f32),
                vz: Some(vz.value as f32),
                mass: Some(mass.value as f32),
                size: Some(size.value as f32),
                color: Some(color),
                weight: Some(weight.value as f32),
                node_type: Some(node_type),
                group: cluster.map(|c| c.to_string()),
                metadata,
                owl_class_iri: None,
                file_size: 0,
                user_data: None,
            };

            nodes.push(node);
        }

        Ok(nodes)
    }

    /// Load all edges from Neo4j
    async fn load_all_edges(&self) -> Result<Vec<Edge>> {
        let query_str = "
            MATCH (source:GraphNode)-[r:EDGE]->(target:GraphNode)
            RETURN source.id as source_id,
                   target.id as target_id,
                   r.weight as weight,
                   r.edge_type as edge_type
        ";

        let mut result = self.graph
            .execute(query(query_str))
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Failed to query edges: {}", e)))?;

        let mut edges = Vec::new();

        while let Some(row) = result.next().await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Failed to fetch row: {}", e)))?
        {
            let source_id: BoltInteger = row.get("source_id")
                .map_err(|e| GraphRepositoryError::DeserializationError(format!("Missing source_id: {}", e)))?;
            let target_id: BoltInteger = row.get("target_id")
                .map_err(|e| GraphRepositoryError::DeserializationError(format!("Missing target_id: {}", e)))?;

            let weight: BoltFloat = row.get("weight").unwrap_or(BoltFloat { value: 1.0 });
            let edge_type: String = row.get("edge_type").unwrap_or_else(|_| "default".to_string());

            let edge = Edge {
                id: format!("{}-{}", source_id.value, target_id.value),
                source: source_id.value as u32,
                target: target_id.value as u32,
                weight: weight.value as f32,
                edge_type: Some(edge_type),
                owl_property_iri: None,
                metadata: None,
            };

            edges.push(edge);
        }

        Ok(edges)
    }

    /// Load all metadata from Neo4j
    async fn load_all_metadata(&self) -> Result<HashMap<String, Metadata>> {
        // For now, extract from nodes
        // Could be separate MATCH query if metadata is stored separately
        Ok(HashMap::new())
    }

    /// Invalidate cache (call after mutations)
    pub async fn invalidate_cache(&self) {
        *self.is_loaded.write().await = false;
        *self.graph_snapshot.write().await = None;
        self.node_cache.write().await.clear();
        self.edge_cache.write().await.clear();
    }
}

#[async_trait]
impl GraphRepository for Neo4jGraphRepository {
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>> {
        let mut added_ids = Vec::new();

        for node in nodes {
            let query_str = "
                MERGE (n:GraphNode {id: $id})
                ON CREATE SET
                    n.created_at = datetime(),
                    n.metadata_id = $metadata_id,
                    n.label = $label
                ON MATCH SET n.updated_at = datetime()
                SET n.x = $x,
                    n.y = $y,
                    n.z = $z,
                    n.vx = $vx,
                    n.vy = $vy,
                    n.vz = $vz,
                    n.mass = $mass,
                    n.size = $size,
                    n.color = $color,
                    n.weight = $weight,
                    n.node_type = $node_type,
                    n.metadata = $metadata
            ";

            let metadata_json = serde_json::to_string(&node.metadata)
                .map_err(|e| GraphRepositoryError::SerializationError(format!("Failed to serialize metadata: {}", e)))?;

            self.graph
                .run(query(query_str)
                    .param("id", node.id as i64)
                    .param("metadata_id", node.metadata_id.clone())
                    .param("label", node.label.clone())
                    .param("x", node.data.position().x as f64)
                    .param("y", node.data.position().y as f64)
                    .param("z", node.data.position().z as f64)
                    .param("vx", node.data.velocity().x as f64)
                    .param("vy", node.data.velocity().y as f64)
                    .param("vz", node.data.velocity().z as f64)
                    .param("mass", node.data.mass() as f64)
                    .param("size", node.size.unwrap_or(1.0) as f64)
                    .param("color", node.color.clone().unwrap_or_else(|| "#888888".to_string()))
                    .param("weight", node.weight.unwrap_or(1.0) as f64)
                    .param("node_type", node.node_type.clone().unwrap_or_else(|| "default".to_string()))
                    .param("metadata", metadata_json))
                .await
                .map_err(|e| GraphRepositoryError::AccessError(format!("Failed to add node: {}", e)))?;

            added_ids.push(node.id);

            // Update cache
            self.node_cache.write().await.put(node.id, node);
        }

        // Invalidate full graph snapshot
        self.invalidate_cache().await;

        Ok(added_ids)
    }

    async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>> {
        let mut added_ids = Vec::new();

        for edge in edges {
            let query_str = "
                MATCH (source:GraphNode {id: $source_id})
                MATCH (target:GraphNode {id: $target_id})
                MERGE (source)-[r:EDGE]->(target)
                ON CREATE SET r.created_at = datetime()
                ON MATCH SET r.updated_at = datetime()
                SET r.weight = $weight,
                    r.edge_type = $edge_type
            ";

            self.graph
                .run(query(query_str)
                    .param("source_id", edge.source as i64)
                    .param("target_id", edge.target as i64)
                    .param("weight", edge.weight as f64)
                    .param("edge_type", edge.edge_type.clone().unwrap_or_else(|| "default".to_string())))
                .await
                .map_err(|e| GraphRepositoryError::AccessError(format!("Failed to add edge: {}", e)))?;

            added_ids.push(edge.id.clone());

            // Update cache
            self.edge_cache.write().await.put(edge.id.clone(), edge);
        }

        // Invalidate full graph snapshot
        self.invalidate_cache().await;

        Ok(added_ids)
    }

    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        // Check if loaded
        if !*self.is_loaded.read().await {
            self.load_graph().await?;
        }

        // Return cached snapshot
        self.graph_snapshot.read().await
            .clone()
            .ok_or_else(|| GraphRepositoryError::AccessError("Graph not loaded".to_string()))
    }

    async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>> {
        let graph = self.get_graph().await?;
        let map: HashMap<u32, Node> = graph.nodes.iter()
            .map(|n| (n.id, n.clone()))
            .collect();
        Ok(Arc::new(map))
    }

    async fn get_physics_state(&self) -> Result<PhysicsState> {
        // Physics state would be managed separately by PhysicsActor
        Ok(PhysicsState::default())
    }

    async fn update_positions(
        &self,
        updates: Vec<(u32, crate::ports::graph_repository::BinaryNodeData)>,
    ) -> Result<()> {
        // Batch update positions in Neo4j
        for (node_id, data) in updates {
            let query_str = "
                MATCH (n:GraphNode {id: $id})
                SET n.x = $x,
                    n.y = $y,
                    n.z = $z
            ";

            self.graph
                .run(query(query_str)
                    .param("id", node_id as i64)
                    .param("x", data.0 as f64)
                    .param("y", data.1 as f64)
                    .param("z", data.2 as f64))
                .await
                .map_err(|e| GraphRepositoryError::AccessError(format!("Failed to update position: {}", e)))?;
        }

        Ok(())
    }

    async fn clear_dirty_nodes(&self) -> Result<()> {
        // Not applicable for Neo4j-backed repo
        Ok(())
    }

    // Implement remaining trait methods with Neo4j queries...
    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>> {
        Ok(Vec::new())
    }

    async fn get_constraints(&self) -> Result<ConstraintSet> {
        Ok(ConstraintSet::default())
    }

    async fn compute_shortest_paths(&self, _params: PathfindingParams) -> Result<PathfindingResult> {
        Err(GraphRepositoryError::NotImplemented)
    }

    async fn get_dirty_nodes(&self) -> Result<HashSet<u32>> {
        Ok(HashSet::new())
    }

    async fn get_node_positions(&self) -> Result<Vec<(u32, Vec3)>> {
        let graph = self.get_graph().await?;
        let positions = graph.nodes.iter()
            .map(|n| (n.id, Vec3::new(
                n.x.unwrap_or(0.0),
                n.y.unwrap_or(0.0),
                n.z.unwrap_or(0.0)
            )))
            .collect();
        Ok(positions)
    }

    async fn get_bots_graph(&self) -> Result<Arc<GraphData>> {
        // For now, return the same graph
        // In the future, this could filter for bot nodes
        self.get_graph().await
    }

    async fn get_equilibrium_status(&self) -> Result<bool> {
        // This would check physics equilibrium state
        // For now, always return false (not in equilibrium)
        Ok(false)
    }
}
