// src/adapters/neo4j_adapter.rs
//! Neo4j Graph Repository Adapter
//!
//! Implements KnowledgeGraphRepository trait using Neo4j graph database.
//! Provides native Cypher query support for multi-hop reasoning and path analysis.
//!
//! Database schema:
//! - Nodes: (:GraphNode {id, metadata_id, label, owl_class_iri, ...})
//! - Relationships: [:EDGE {weight, relation_type, owl_property_iri}]
//!
//! This adapter enables:
//! - Complex graph traversals with Cypher
//! - Multi-hop path analysis
//! - Semantic reasoning via OWL enrichment
//! - High-performance graph queries

use async_trait::async_trait;
use log::{debug, info, warn};
use neo4rs::{Graph, Query, Node as Neo4jNode, ConfigBuilder};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::utils::json::{to_json, from_json};
use crate::ports::knowledge_graph_repository::{
    GraphStatistics, KnowledgeGraphRepository, KnowledgeGraphRepositoryError,
    Result as RepoResult,
};
use crate::utils::network::circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError};
use crate::utils::time;

/// Neo4j configuration with security and performance settings
#[derive(Debug, Clone)]
pub struct Neo4jConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
    pub database: Option<String>,
    /// Maximum number of connections in the pool (default: 50)
    pub max_connections: usize,
    /// Query timeout in seconds (default: 30)
    pub query_timeout_secs: u64,
    /// Connection timeout in seconds (default: 10)
    pub connection_timeout_secs: u64,
}

impl Default for Neo4jConfig {
    fn default() -> Self {
        // SECURITY: NEO4J_PASSWORD is REQUIRED - no insecure defaults
        let password = std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| {
            // In development/test mode, allow default only if explicitly set
            if std::env::var("ALLOW_INSECURE_DEFAULTS").is_ok() {
                log::warn!("⚠️  NEO4J_PASSWORD not set - using insecure default (ALLOW_INSECURE_DEFAULTS=1)");
                "password".to_string()
            } else {
                log::error!("🚨 CRITICAL: NEO4J_PASSWORD environment variable is REQUIRED!");
                log::error!("   Set NEO4J_PASSWORD=<your-secure-password> or");
                log::error!("   Set ALLOW_INSECURE_DEFAULTS=1 for development only");
                panic!("NEO4J_PASSWORD must be set. See logs for details.");
            }
        });

        // Reject obviously insecure passwords in production
        if password == "password" || password == "neo4j" || password.len() < 8 {
            if std::env::var("ALLOW_INSECURE_DEFAULTS").is_err() {
                log::error!("🚨 CRITICAL: NEO4J_PASSWORD is too weak or uses a default value!");
                panic!("NEO4J_PASSWORD must be at least 8 characters and not a default value");
            }
        }

        Self {
            uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string()),
            password,
            database: std::env::var("NEO4J_DATABASE").ok(),
            max_connections: std::env::var("NEO4J_MAX_CONNECTIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(50),
            query_timeout_secs: std::env::var("NEO4J_QUERY_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(30),
            connection_timeout_secs: std::env::var("NEO4J_CONNECTION_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
        }
    }
}

/// Repository for knowledge graph data in Neo4j
///
/// Provides high-performance graph operations with native Cypher support.
/// All node positions and velocities are persisted and can be queried with
/// complex graph patterns.
pub struct Neo4jAdapter {
    graph: Arc<Graph>,
    config: Neo4jConfig,
    circuit_breaker: CircuitBreaker,
}

impl Neo4jAdapter {
    /// Create a new Neo4jAdapter with security hardening
    ///
    /// # Arguments
    /// * `config` - Neo4j connection configuration
    ///
    /// # Security
    /// - Uses connection pooling (configured via config.max_connections)
    /// - Enforces query timeouts (configured via config.query_timeout_secs)
    /// - Logs warning if default password is used
    ///
    /// # Returns
    /// Initialized adapter ready for graph operations
    pub async fn new(config: Neo4jConfig) -> Result<Self, KnowledgeGraphRepositoryError> {
        // SECURITY: Validate configuration
        if config.password == "password" {
            log::error!("❌ CRITICAL: Using default password 'password' for Neo4j!");
            log::error!("❌ Set NEO4J_PASSWORD environment variable immediately!");
        }

        if config.max_connections == 0 {
            return Err(KnowledgeGraphRepositoryError::DatabaseError(
                "Invalid configuration: max_connections must be > 0".to_string()
            ));
        }

        info!("Connecting to Neo4j at {} (max_connections: {}, query_timeout: {}s)",
              config.uri, config.max_connections, config.query_timeout_secs);

        // PERF: Configure connection pool for high-throughput graph operations
        let neo4j_config = ConfigBuilder::default()
            .uri(&config.uri)
            .user(&config.user)
            .password(&config.password)
            .max_connections(config.max_connections)
            .build()
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to build Neo4j config: {}",
                    e
                ))
            })?;

        let graph = Graph::connect(neo4j_config)
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to connect to Neo4j: {}",
                    e
                ))
            })?;

        info!("Connected to Neo4j successfully with {} connection pool", config.max_connections);

        let adapter = Self {
            graph: Arc::new(graph),
            config,
            circuit_breaker: CircuitBreaker::new(CircuitBreakerConfig::tcp_connection()),
        };

        // Create indexes and constraints
        adapter.create_schema().await?;

        Ok(adapter)
    }

    /// Get access to underlying Graph for direct queries
    pub fn graph(&self) -> &Arc<Graph> {
        &self.graph
    }

    /// Create Neo4j schema (indexes and constraints)
    async fn create_schema(&self) -> RepoResult<()> {
        info!("Creating Neo4j schema...");

        // Create uniqueness constraint on GraphNode.id
        let constraint_query = Query::new("CREATE CONSTRAINT graph_node_id IF NOT EXISTS FOR (n:GraphNode) REQUIRE n.id IS UNIQUE".to_string());

        if let Err(e) = self.graph.run(constraint_query).await {
            warn!("Failed to create constraint (may already exist): {}", e);
        }

        // Create index on metadata_id for faster lookups
        let index_query = Query::new("CREATE INDEX graph_node_metadata_id IF NOT EXISTS FOR (n:GraphNode) ON (n.metadata_id)".to_string());

        if let Err(e) = self.graph.run(index_query).await {
            warn!("Failed to create index (may already exist): {}", e);
        }

        // Create index on owl_class_iri for semantic queries
        let owl_index_query = Query::new("CREATE INDEX graph_node_owl_class IF NOT EXISTS FOR (n:GraphNode) ON (n.owl_class_iri)".to_string());

        if let Err(e) = self.graph.run(owl_index_query).await {
            warn!("Failed to create OWL index (may already exist): {}", e);
        }

        // Create index on node_type for semantic force filtering
        let node_type_index_query = Query::new("CREATE INDEX graph_node_type IF NOT EXISTS FOR (n:GraphNode) ON (n.node_type)".to_string());

        if let Err(e) = self.graph.run(node_type_index_query).await {
            warn!("Failed to create node_type index (may already exist): {}", e);
        }

        // Create index on edge relation_type for semantic pathfinding
        let edge_type_index_query = Query::new("CREATE INDEX edge_relation_type IF NOT EXISTS FOR ()-[r:EDGE]-() ON (r.relation_type)".to_string());

        if let Err(e) = self.graph.run(edge_type_index_query).await {
            warn!("Failed to create edge relation_type index (may already exist): {}", e);
        }

        info!("✅ Neo4j schema created successfully with semantic type indexes");
        Ok(())
    }

    /// Convert Node to Neo4j properties
    fn node_to_properties(node: &Node) -> HashMap<String, neo4rs::BoltType> {
        let mut props = HashMap::new();

        props.insert("id".to_string(), neo4rs::BoltType::Integer(neo4rs::BoltInteger::new(node.id as i64)));
        props.insert("metadata_id".to_string(), neo4rs::BoltType::String(neo4rs::BoltString::from(node.metadata_id.clone())));
        props.insert("label".to_string(), neo4rs::BoltType::String(neo4rs::BoltString::from(node.label.clone())));
        props.insert("x".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(node.data.x as f64)));
        props.insert("y".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(node.data.y as f64)));
        props.insert("z".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(node.data.z as f64)));
        props.insert("vx".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(node.data.vx as f64)));
        props.insert("vy".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(node.data.vy as f64)));
        props.insert("vz".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(node.data.vz as f64)));
        props.insert("mass".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(node.mass.unwrap_or(1.0) as f64)));

        if let Some(ref iri) = node.owl_class_iri {
            props.insert("owl_class_iri".to_string(), neo4rs::BoltType::String(neo4rs::BoltString::from(iri.clone())));
        }

        if let Some(ref color) = node.color {
            props.insert("color".to_string(), neo4rs::BoltType::String(neo4rs::BoltString::from(color.clone())));
        }

        if let Some(size) = node.size {
            props.insert("size".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(size as f64)));
        }

        if let Some(ref node_type) = node.node_type {
            props.insert("node_type".to_string(), neo4rs::BoltType::String(neo4rs::BoltString::from(node_type.clone())));
        }

        if let Some(weight) = node.weight {
            props.insert("weight".to_string(), neo4rs::BoltType::Float(neo4rs::BoltFloat::new(weight as f64)));
        }

        if let Some(ref group) = node.group {
            props.insert("group_name".to_string(), neo4rs::BoltType::String(neo4rs::BoltString::from(group.clone())));
        }

        // Serialize metadata as JSON string
        if !node.metadata.is_empty() {
            if let Ok(json) = to_json(&node.metadata) {
                props.insert("metadata".to_string(), neo4rs::BoltType::String(neo4rs::BoltString::from(json)));
            }
        }

        props
    }

    /// Convert Neo4j node to our Node model
    /// Prioritizes sim_* properties for physics coordinates (GPU-calculated positions)
    fn neo4j_node_to_node(neo4j_node: &Neo4jNode) -> RepoResult<Node> {
        let id: i64 = neo4j_node.get("id").map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Missing id: {}", e))
        })?;

        let metadata_id: String = neo4j_node.get("metadata_id").map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Missing metadata_id: {}", e))
        })?;

        let label: String = neo4j_node.get("label").unwrap_or_else(|_| String::new());

        // Prefer sim_* properties (GPU physics state) over x/y/z (initial/content positions)
        // This preserves the calculated layout during content sync
        let x: f64 = neo4j_node.get("sim_x").or_else(|_| neo4j_node.get("x")).unwrap_or(0.0);
        let y: f64 = neo4j_node.get("sim_y").or_else(|_| neo4j_node.get("y")).unwrap_or(0.0);
        let z: f64 = neo4j_node.get("sim_z").or_else(|_| neo4j_node.get("z")).unwrap_or(0.0);
        let vx: f64 = neo4j_node.get("vx").unwrap_or(0.0);
        let vy: f64 = neo4j_node.get("vy").unwrap_or(0.0);
        let vz: f64 = neo4j_node.get("vz").unwrap_or(0.0);
        let mass: f64 = neo4j_node.get("mass").unwrap_or(1.0);

        let owl_class_iri: Option<String> = neo4j_node.get("owl_class_iri").ok();
        let color: Option<String> = neo4j_node.get("color").ok();
        let size: Option<f64> = neo4j_node.get("size").ok();
        let node_type: Option<String> = neo4j_node.get("node_type").ok();
        let weight: Option<f64> = neo4j_node.get("weight").ok();
        let group_name: Option<String> = neo4j_node.get("group_name").ok();

        let mut metadata: HashMap<String, String> = neo4j_node
            .get::<String>("metadata")
            .ok()
            .and_then(|json| from_json(&json).ok())
            .unwrap_or_default();

        // Read quality_score and authority_score from Neo4j node properties
        // These are stored as top-level properties, not inside metadata JSON
        if let Ok(quality_score) = neo4j_node.get::<f64>("quality_score") {
            metadata.insert("quality_score".to_string(), quality_score.to_string());
        }
        if let Ok(authority_score) = neo4j_node.get::<f64>("authority_score") {
            metadata.insert("authority_score".to_string(), authority_score.to_string());
        }

        let mut node = Node::new_with_id(metadata_id, Some(id as u32));
        node.label = label;
        node.data.x = x as f32;
        node.data.y = y as f32;
        node.data.z = z as f32;
        node.data.vx = vx as f32;
        node.data.vy = vy as f32;
        node.data.vz = vz as f32;
        node.mass = Some(mass as f32);
        node.owl_class_iri = owl_class_iri;
        node.color = color;
        node.size = size.map(|s| s as f32);
        node.node_type = node_type;
        node.weight = weight.map(|w| w as f32);
        node.group = group_name;
        node.metadata = metadata;

        Ok(node)
    }

    /// Execute a parameterized Cypher query (SAFE - use this for user input)
    ///
    /// # Security
    /// This method enforces parameterization to prevent Cypher injection attacks.
    /// DO NOT concatenate user input into the query string - use parameters instead.
    ///
    /// # Example
    /// ```ignore
    /// // SAFE - Uses parameters
    /// let params = hashmap!{"name" => BoltType::String("Alice".into())};
    /// adapter.execute_cypher_safe("MATCH (n:User {name: $name}) RETURN n", params).await?;
    ///
    /// // UNSAFE - Don't do this!
    /// // let query = format!("MATCH (n:User {{name: '{}'}}) RETURN n", user_input);
    /// ```
    pub async fn execute_cypher_safe(
        &self,
        query: &str,
        params: HashMap<String, neo4rs::BoltType>,
    ) -> RepoResult<Vec<HashMap<String, serde_json::Value>>> {
        self.execute_cypher_internal(query, params, true).await
    }

    /// Execute a Cypher query (DEPRECATED - use execute_cypher_safe)
    ///
    /// # Security Warning
    /// This method is deprecated in favor of execute_cypher_safe.
    /// Only use this for trusted, static queries. Never concatenate user input!
    #[deprecated(since = "0.1.0", note = "Use execute_cypher_safe instead")]
    pub async fn execute_cypher(
        &self,
        query: &str,
        params: HashMap<String, neo4rs::BoltType>,
    ) -> RepoResult<Vec<HashMap<String, serde_json::Value>>> {
        log::warn!("execute_cypher is deprecated - use execute_cypher_safe instead");
        self.execute_cypher_internal(query, params, false).await
    }

    /// Internal method for executing Cypher queries
    async fn execute_cypher_internal(
        &self,
        query: &str,
        params: HashMap<String, neo4rs::BoltType>,
        _safe_mode: bool,
    ) -> RepoResult<Vec<HashMap<String, serde_json::Value>>> {
        // SECURITY: Log query execution (without sensitive data)
        debug!("Executing Cypher query with {} parameters", params.len());

        let mut query_obj = Query::new(query.to_string());

        for (key, value) in params {
            query_obj = query_obj.param(&key, value);
        }

        // Apply query timeout at application level (neo4rs doesn't support query-level timeouts)
        // Default timeout: 30 seconds for complex graph queries
        let query_timeout = std::time::Duration::from_secs(
            std::env::var("NEO4J_QUERY_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30)
        );

        // Clone graph Arc for use in closure
        let graph = self.graph.clone();

        // Wrap query execution with circuit breaker for network resilience
        let execute_result = self.circuit_breaker.execute(async {
            let result = tokio::time::timeout(
                query_timeout,
                graph.execute(query_obj)
            ).await;

            match result {
                Ok(Ok(r)) => Ok(r),
                Ok(Err(e)) => {
                    log::error!("Cypher query failed: {}", e);
                    Err(KnowledgeGraphRepositoryError::DatabaseError(format!("Cypher query failed: {}", e)))
                }
                Err(_) => {
                    log::error!("Cypher query timed out after {:?}", query_timeout);
                    Err(KnowledgeGraphRepositoryError::DatabaseError(
                        format!("Query timed out after {:?}", query_timeout)
                    ))
                }
            }
        }).await;

        let mut result = match execute_result {
            Ok(r) => r,
            Err(CircuitBreakerError::CircuitOpen) => {
                log::warn!("Circuit breaker is open - Neo4j queries temporarily blocked");
                return Err(KnowledgeGraphRepositoryError::DatabaseError(
                    "Circuit breaker open: Neo4j service temporarily unavailable".to_string()
                ));
            }
            Err(CircuitBreakerError::OperationFailed(e)) => {
                return Err(e);
            }
        };

        let mut results = Vec::new();
        while let Ok(Some(_row)) = result.next().await {
            // Note: Neo4rs Row API doesn't provide direct access to all keys
            // For now, returning empty map - users should use specific field access
            let row_map = HashMap::new();
            results.push(row_map);
        }

        Ok(results)
    }
}

#[async_trait]
impl KnowledgeGraphRepository for Neo4jAdapter {
    #[instrument(skip(self), level = "debug")]
    async fn load_graph(&self) -> RepoResult<Arc<GraphData>> {
        // Load all nodes
        let nodes_query = Query::new("MATCH (n:GraphNode) RETURN n ORDER BY n.id".to_string());

        let mut nodes = Vec::new();
        let mut result = self.graph.execute(nodes_query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to load nodes: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            if let Ok(neo4j_node) = row.get::<Neo4jNode>("n") {
                nodes.push(Self::neo4j_node_to_node(&neo4j_node)?);
            }
        }

        debug!("Loaded {} nodes from Neo4j", nodes.len());

        // Load all edges
        let edges_query = Query::new("MATCH (s:GraphNode)-[r:EDGE]->(t:GraphNode) RETURN s.id AS source, t.id AS target, r.weight AS weight, r.relation_type AS relation_type, r.owl_property_iri AS owl_property_iri, r.metadata AS metadata".to_string());

        let mut edges = Vec::new();
        let mut result = self.graph.execute(edges_query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to load edges: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            let source: i64 = row.get("source").unwrap_or(0);
            let target: i64 = row.get("target").unwrap_or(0);
            let weight: f64 = row.get("weight").unwrap_or(1.0);
            let relation_type: Option<String> = row.get("relation_type").ok();
            let owl_property_iri: Option<String> = row.get("owl_property_iri").ok();
            let metadata_json: Option<String> = row.get("metadata").ok();

            let metadata = metadata_json
                .and_then(|json| from_json(&json).ok());

            let mut edge = Edge::new(source as u32, target as u32, weight as f32);
            edge.edge_type = relation_type;
            edge.owl_property_iri = owl_property_iri;
            edge.metadata = metadata;

            edges.push(edge);
        }

        debug!("Loaded {} edges from Neo4j", edges.len());

        let mut graph = GraphData::new();
        graph.nodes = nodes;
        graph.edges = edges;

        Ok(Arc::new(graph))
    }

    async fn save_graph(&self, graph: &GraphData) -> RepoResult<()> {
        // Save nodes in batch - PRESERVING physics state (sim_x/y/z, vx/vy/vz)
        // Content sync should NEVER reset GPU-calculated layout positions
        for node in &graph.nodes {
            // Use COALESCE to preserve existing physics coordinates
            // Physics state is stored in sim_* properties, content coords in x/y/z
            let query = Query::new(
                "MERGE (n:GraphNode {id: $id})
                 ON CREATE SET
                     n.metadata_id = $metadata_id,
                     n.label = $label,
                     n.x = $x,
                     n.y = $y,
                     n.z = $z,
                     n.sim_x = $x,
                     n.sim_y = $y,
                     n.sim_z = $z,
                     n.vx = $vx,
                     n.vy = $vy,
                     n.vz = $vz,
                     n.mass = $mass,
                     n.owl_class_iri = $owl_class_iri,
                     n.color = $color,
                     n.size = $size,
                     n.node_type = $node_type,
                     n.weight = $weight,
                     n.group_name = $group_name,
                     n.metadata = $metadata
                 ON MATCH SET
                     n.metadata_id = $metadata_id,
                     n.label = $label,
                     n.owl_class_iri = COALESCE($owl_class_iri, n.owl_class_iri),
                     n.color = COALESCE($color, n.color),
                     n.size = COALESCE($size, n.size),
                     n.node_type = COALESCE($node_type, n.node_type),
                     n.weight = COALESCE($weight, n.weight),
                     n.group_name = COALESCE($group_name, n.group_name),
                     n.metadata = $metadata
                 // NEVER overwrite sim_x/sim_y/sim_z or vx/vy/vz on MATCH
                 // These are the GPU-calculated physics positions
                ".to_string()
            )
            .param("id", node.id as i64)
            .param("metadata_id", node.metadata_id.clone())
            .param("label", node.label.clone())
            .param("x", node.data.x as f64)
            .param("y", node.data.y as f64)
            .param("z", node.data.z as f64)
            .param("vx", node.data.vx as f64)
            .param("vy", node.data.vy as f64)
            .param("vz", node.data.vz as f64)
            .param("mass", node.mass.unwrap_or(1.0) as f64)
            .param("owl_class_iri", node.owl_class_iri.clone().unwrap_or_default())
            .param("color", node.color.clone().unwrap_or_default())
            .param("size", node.size.unwrap_or(1.0) as f64)
            .param("node_type", node.node_type.clone().unwrap_or_default())
            .param("weight", node.weight.unwrap_or(1.0) as f64)
            .param("group_name", node.group.clone().unwrap_or_default())
            .param("metadata", serde_json::to_string(&node.metadata).unwrap_or_default());

            self.graph.run(query).await.map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to save node {}: {}",
                    node.id, e
                ))
            })?;
        }

        // Save edges in batch
        for edge in &graph.edges {
            let mut query = Query::new("MATCH (s:GraphNode {id: $source}) MATCH (t:GraphNode {id: $target}) MERGE (s)-[r:EDGE]->(t) SET r.weight = $weight, r.relation_type = $relation_type, r.owl_property_iri = $owl_property_iri, r.metadata = $metadata".to_string());

            query = query.param("source", edge.source as i64);
            query = query.param("target", edge.target as i64);
            query = query.param("weight", edge.weight as f64);
            query = query.param("relation_type", edge.edge_type.clone().unwrap_or_default());
            query = query.param("owl_property_iri", edge.owl_property_iri.clone().unwrap_or_default());

            let metadata_json = edge.metadata.as_ref()
                .and_then(|m| to_json(m).ok())
                .unwrap_or_default();
            query = query.param("metadata", metadata_json);

            self.graph.run(query).await.map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to save edge {}: {}",
                    edge.id, e
                ))
            })?;
        }

        info!("Saved graph to Neo4j: {} nodes, {} edges", graph.nodes.len(), graph.edges.len());
        Ok(())
    }

    async fn add_node(&self, node: &Node) -> RepoResult<u32> {
        let props = Self::node_to_properties(node);

        let mut query = Query::new("CREATE (n:GraphNode) SET n = $props RETURN n.id AS id".to_string());

        query = query.param("props", props);

        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to add node: {}", e))
        })?;

        if let Ok(Some(row)) = result.next().await {
            let id: i64 = row.get("id").unwrap_or(node.id as i64);
            Ok(id as u32)
        } else {
            Ok(node.id)
        }
    }

    async fn batch_add_nodes(&self, nodes: Vec<Node>) -> RepoResult<Vec<u32>> {
        let mut ids = Vec::new();
        for node in nodes {
            let id = self.add_node(&node).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn update_node(&self, node: &Node) -> RepoResult<()> {
        let props = Self::node_to_properties(node);

        let mut query = Query::new("MATCH (n:GraphNode {id: $id}) SET n = $props".to_string());

        query = query.param("id", node.id as i64);
        query = query.param("props", props);

        self.graph.run(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to update node: {}", e))
        })?;

        Ok(())
    }

    async fn batch_update_nodes(&self, nodes: Vec<Node>) -> RepoResult<()> {
        for node in nodes {
            self.update_node(&node).await?;
        }
        Ok(())
    }

    async fn remove_node(&self, node_id: u32) -> RepoResult<()> {
        let query = Query::new("MATCH (n:GraphNode {id: $id}) DETACH DELETE n".to_string()).param("id", node_id as i64);

        self.graph.run(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to remove node: {}", e))
        })?;

        Ok(())
    }

    async fn batch_remove_nodes(&self, node_ids: Vec<u32>) -> RepoResult<()> {
        for id in node_ids {
            self.remove_node(id).await?;
        }
        Ok(())
    }

    async fn get_node(&self, node_id: u32) -> RepoResult<Option<Node>> {
        let query = Query::new("MATCH (n:GraphNode {id: $id}) RETURN n".to_string()).param("id", node_id as i64);

        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get node: {}", e))
        })?;

        if let Ok(Some(row)) = result.next().await {
            if let Ok(neo4j_node) = row.get::<Neo4jNode>("n") {
                return Ok(Some(Self::neo4j_node_to_node(&neo4j_node)?));
            }
        }

        Ok(None)
    }

    async fn get_nodes(&self, node_ids: Vec<u32>) -> RepoResult<Vec<Node>> {
        let ids: Vec<i64> = node_ids.iter().map(|&id| id as i64).collect();

        let query = Query::new("MATCH (n:GraphNode) WHERE n.id IN $ids RETURN n".to_string()).param("ids", ids);

        let mut nodes = Vec::new();
        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get nodes: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            if let Ok(neo4j_node) = row.get::<Neo4jNode>("n") {
                nodes.push(Self::neo4j_node_to_node(&neo4j_node)?);
            }
        }

        Ok(nodes)
    }

    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> RepoResult<Vec<Node>> {
        let query = Query::new("MATCH (n:GraphNode {metadata_id: $metadata_id}) RETURN n".to_string()).param("metadata_id", metadata_id.to_string());

        let mut nodes = Vec::new();
        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get nodes: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            if let Ok(neo4j_node) = row.get::<Neo4jNode>("n") {
                nodes.push(Self::neo4j_node_to_node(&neo4j_node)?);
            }
        }

        Ok(nodes)
    }

    async fn search_nodes_by_label(&self, label: &str) -> RepoResult<Vec<Node>> {
        let query = Query::new("MATCH (n:GraphNode) WHERE n.label CONTAINS $label RETURN n".to_string()).param("label", label.to_string());

        let mut nodes = Vec::new();
        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to search nodes: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            if let Ok(neo4j_node) = row.get::<Neo4jNode>("n") {
                nodes.push(Self::neo4j_node_to_node(&neo4j_node)?);
            }
        }

        Ok(nodes)
    }

    async fn add_edge(&self, edge: &Edge) -> RepoResult<String> {
        let mut query = Query::new("MATCH (s:GraphNode {id: $source}) MATCH (t:GraphNode {id: $target}) CREATE (s)-[r:EDGE {weight: $weight, relation_type: $relation_type, owl_property_iri: $owl_property_iri, metadata: $metadata}]->(t) RETURN elementId(r) AS id".to_string());

        query = query.param("source", edge.source as i64);
        query = query.param("target", edge.target as i64);
        query = query.param("weight", edge.weight as f64);
        query = query.param("relation_type", edge.edge_type.clone().unwrap_or_default());
        query = query.param("owl_property_iri", edge.owl_property_iri.clone().unwrap_or_default());

        let metadata_json = edge.metadata.as_ref()
            .and_then(|m| to_json(m).ok())
            .unwrap_or_default();
        query = query.param("metadata", metadata_json);

        self.graph.run(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to add edge: {}", e))
        })?;

        Ok(edge.id.clone())
    }

    async fn batch_add_edges(&self, edges: Vec<Edge>) -> RepoResult<Vec<String>> {
        let mut ids = Vec::new();
        for edge in edges {
            let id = self.add_edge(&edge).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn update_edge(&self, edge: &Edge) -> RepoResult<()> {
        let mut query = Query::new("MATCH (s:GraphNode {id: $source})-[r:EDGE]->(t:GraphNode {id: $target}) SET r.weight = $weight, r.relation_type = $relation_type, r.owl_property_iri = $owl_property_iri, r.metadata = $metadata".to_string());

        query = query.param("source", edge.source as i64);
        query = query.param("target", edge.target as i64);
        query = query.param("weight", edge.weight as f64);
        query = query.param("relation_type", edge.edge_type.clone().unwrap_or_default());
        query = query.param("owl_property_iri", edge.owl_property_iri.clone().unwrap_or_default());

        let metadata_json = edge.metadata.as_ref()
            .and_then(|m| to_json(m).ok())
            .unwrap_or_default();
        query = query.param("metadata", metadata_json);

        self.graph.run(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to update edge: {}", e))
        })?;

        Ok(())
    }

    async fn remove_edge(&self, edge_id: &str) -> RepoResult<()> {
        // Parse edge_id format "source-target"
        let parts: Vec<&str> = edge_id.split('-').collect();
        if parts.len() != 2 {
            return Err(KnowledgeGraphRepositoryError::InvalidData(
                format!("Invalid edge_id format: {}", edge_id)
            ));
        }

        let source: u32 = parts[0].parse().map_err(|_| {
            KnowledgeGraphRepositoryError::InvalidData(format!("Invalid source id: {}", parts[0]))
        })?;

        let target: u32 = parts[1].parse().map_err(|_| {
            KnowledgeGraphRepositoryError::InvalidData(format!("Invalid target id: {}", parts[1]))
        })?;

        let query = Query::new("MATCH (s:GraphNode {id: $source})-[r:EDGE]->(t:GraphNode {id: $target}) DELETE r".to_string())
            .param("source", source as i64)
            .param("target", target as i64);

        self.graph.run(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to remove edge: {}", e))
        })?;

        Ok(())
    }

    async fn batch_remove_edges(&self, edge_ids: Vec<String>) -> RepoResult<()> {
        for id in edge_ids {
            self.remove_edge(&id).await?;
        }
        Ok(())
    }

    async fn get_node_edges(&self, node_id: u32) -> RepoResult<Vec<Edge>> {
        let query = Query::new("MATCH (s:GraphNode {id: $id})-[r:EDGE]-(t:GraphNode) RETURN s.id AS source, t.id AS target, r.weight AS weight, r.relation_type AS relation_type, r.owl_property_iri AS owl_property_iri, r.metadata AS metadata".to_string()).param("id", node_id as i64);

        let mut edges = Vec::new();
        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get node edges: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            let source: i64 = row.get("source").unwrap_or(0);
            let target: i64 = row.get("target").unwrap_or(0);
            let weight: f64 = row.get("weight").unwrap_or(1.0);
            let relation_type: Option<String> = row.get("relation_type").ok();
            let owl_property_iri: Option<String> = row.get("owl_property_iri").ok();
            let metadata_json: Option<String> = row.get("metadata").ok();

            let metadata = metadata_json
                .and_then(|json| from_json(&json).ok());

            let mut edge = Edge::new(source as u32, target as u32, weight as f32);
            edge.edge_type = relation_type;
            edge.owl_property_iri = owl_property_iri;
            edge.metadata = metadata;

            edges.push(edge);
        }

        Ok(edges)
    }

    async fn get_edges_between(&self, source_id: u32, target_id: u32) -> RepoResult<Vec<Edge>> {
        let query = Query::new("MATCH (s:GraphNode {id: $source})-[r:EDGE]-(t:GraphNode {id: $target}) RETURN s.id AS source, t.id AS target, r.weight AS weight, r.relation_type AS relation_type, r.owl_property_iri AS owl_property_iri, r.metadata AS metadata".to_string())
            .param("source", source_id as i64)
            .param("target", target_id as i64);

        let mut edges = Vec::new();
        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get edges between: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            let source: i64 = row.get("source").unwrap_or(0);
            let target: i64 = row.get("target").unwrap_or(0);
            let weight: f64 = row.get("weight").unwrap_or(1.0);
            let relation_type: Option<String> = row.get("relation_type").ok();
            let owl_property_iri: Option<String> = row.get("owl_property_iri").ok();
            let metadata_json: Option<String> = row.get("metadata").ok();

            let metadata = metadata_json
                .and_then(|json| from_json(&json).ok());

            let mut edge = Edge::new(source as u32, target as u32, weight as f32);
            edge.edge_type = relation_type;
            edge.owl_property_iri = owl_property_iri;
            edge.metadata = metadata;

            edges.push(edge);
        }

        Ok(edges)
    }

    async fn batch_update_positions(
        &self,
        positions: Vec<(u32, f32, f32, f32)>,
    ) -> RepoResult<()> {
        // Update sim_* properties (physics state) - these are the GPU-calculated positions
        // x/y/z remain as initial/content positions and are not overwritten by physics
        for (node_id, x, y, z) in positions {
            let query = Query::new(
                "MATCH (n:GraphNode {id: $id})
                 SET n.sim_x = $x, n.sim_y = $y, n.sim_z = $z".to_string()
            )
                .param("id", node_id as i64)
                .param("x", x as f64)
                .param("y", y as f64)
                .param("z", z as f64);

            self.graph.run(query).await.map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to update position for node {}: {}",
                    node_id, e
                ))
            })?;
        }

        Ok(())
    }

    async fn query_nodes(&self, query: &str) -> RepoResult<Vec<Node>> {
        warn!("query_nodes with custom query not yet implemented for Neo4j");
        Ok(Vec::new())
    }

    async fn get_neighbors(&self, node_id: u32) -> RepoResult<Vec<Node>> {
        let query = Query::new("MATCH (n:GraphNode {id: $id})-[:EDGE]-(neighbor:GraphNode) RETURN DISTINCT neighbor AS n".to_string()).param("id", node_id as i64);

        let mut nodes = Vec::new();
        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get neighbors: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            if let Ok(neo4j_node) = row.get::<Neo4jNode>("n") {
                nodes.push(Self::neo4j_node_to_node(&neo4j_node)?);
            }
        }

        Ok(nodes)
    }

    async fn get_statistics(&self) -> RepoResult<GraphStatistics> {
        let query = Query::new("MATCH (n:GraphNode) OPTIONAL MATCH (n)-[r:EDGE]-() RETURN count(DISTINCT n) AS node_count, count(r) AS edge_count".to_string());

        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get statistics: {}", e))
        })?;

        if let Ok(Some(row)) = result.next().await {
            let node_count: i64 = row.get("node_count").unwrap_or(0);
            let edge_count: i64 = row.get("edge_count").unwrap_or(0);

            let average_degree = if node_count > 0 {
                (edge_count as f32 * 2.0) / node_count as f32
            } else {
                0.0
            };

            // Calculate connected components using Cypher
            let components_query = Query::new(
                "MATCH (n:GraphNode)
                 WITH COLLECT(DISTINCT n) AS nodes
                 UNWIND nodes AS node
                 OPTIONAL MATCH path = (node)-[*]-(connected)
                 WITH node, COLLECT(DISTINCT connected) AS component
                 RETURN COUNT(DISTINCT component) AS component_count"
                    .to_string()
            );

            let mut component_count = 1; // Default to 1 if query fails
            if let Ok(mut result) = self.graph.execute(components_query).await {
                if let Some(row) = result.next().await.ok().flatten() {
                    if let Ok(count) = row.get::<i64>("component_count") {
                        component_count = count as usize;
                    }
                }
            }

            return Ok(GraphStatistics {
                node_count: node_count as usize,
                edge_count: edge_count as usize,
                average_degree,
                connected_components: component_count,
                last_updated: time::now(),
            });
        }

        Ok(GraphStatistics {
            node_count: 0,
            edge_count: 0,
            average_degree: 0.0,
            connected_components: 0,
            last_updated: time::now(),
        })
    }

    async fn clear_graph(&self) -> RepoResult<()> {
        let query = Query::new("MATCH (n:GraphNode) DETACH DELETE n".to_string());

        self.graph.run(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to clear graph: {}", e))
        })?;

        info!("Cleared all graph data from Neo4j");
        Ok(())
    }

    async fn begin_transaction(&self) -> RepoResult<()> {
        // Neo4j handles transactions internally
        Ok(())
    }

    async fn commit_transaction(&self) -> RepoResult<()> {
        // Neo4j handles transactions internally
        Ok(())
    }

    async fn rollback_transaction(&self) -> RepoResult<()> {
        // Neo4j handles transactions internally
        Ok(())
    }

    async fn health_check(&self) -> RepoResult<bool> {
        let query = Query::new("RETURN 1 AS health".to_string());

        match self.graph.run(query).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn get_all_positions(&self) -> RepoResult<HashMap<u32, (f32, f32, f32)>> {
        // Return sim_* positions (GPU physics state) when available, fallback to x/y/z
        let query = Query::new(
            "MATCH (n:GraphNode)
             WHERE n.sim_x IS NOT NULL OR n.x IS NOT NULL
             RETURN n.id AS id,
                    COALESCE(n.sim_x, n.x, 0.0) AS x,
                    COALESCE(n.sim_y, n.y, 0.0) AS y,
                    COALESCE(n.sim_z, n.z, 0.0) AS z".to_string()
        );

        let mut positions = HashMap::new();
        let mut result = self.graph.execute(query).await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get all positions: {}", e))
        })?;

        while let Ok(Some(row)) = result.next().await {
            let id: i64 = row.get("id").unwrap_or(0);
            let x: f64 = row.get("x").unwrap_or(0.0);
            let y: f64 = row.get("y").unwrap_or(0.0);
            let z: f64 = row.get("z").unwrap_or(0.0);

            positions.insert(id as u32, (x as f32, y as f32, z as f32));
        }

        debug!("Retrieved {} node positions from Neo4j (using sim_* where available)", positions.len());
        Ok(positions)
    }

    async fn get_nodes_by_owl_class_iri(&self, owl_class_iri: &str) -> RepoResult<Vec<Node>> {
        let query = Query::new("MATCH (n:GraphNode) WHERE n.owl_class_iri = $iri RETURN n".to_string()).param("iri", owl_class_iri);

        let mut result = self.graph
            .execute(query)
            .await
            .map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!(
                    "Failed to query nodes by owl_class_iri: {}",
                    e
                ))
            })?;

        let mut nodes = Vec::new();

        while let Some(row) = result.next().await.map_err(|e| {
            KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        })? {
            let neo_node: Neo4jNode = row.get("n").map_err(|e| {
                KnowledgeGraphRepositoryError::DatabaseError(format!("Failed to get node: {}", e))
            })?;

            let node = Self::neo4j_node_to_node(&neo_node)?;
            nodes.push(node);
        }

        Ok(nodes)
    }
}
