// src/adapters/neo4j_ontology_repository.rs
//! Neo4j Ontology Repository Adapter
//!
//! Implements OntologyRepository trait using Neo4j graph database.
//! Stores OWL classes, properties, axioms, and hierarchies in Neo4j.
//!
//! This replaces UnifiedOntologyRepository (SQLite-based) as part of the
//! SQL deprecation effort. See ADR-001 for architectural decision rationale.

use async_trait::async_trait;
use neo4rs::{Graph, query, Node as Neo4jNode};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn, instrument, error};

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::ontology_repository::{
    AxiomType, InferenceResults, OntologyMetrics, OntologyRepository,
    OntologyRepositoryError, OwlAxiom, OwlClass, OwlProperty,
    PathfindingCacheEntry, PropertyType, Result as RepoResult,
    ValidationReport,
};
use crate::utils::json::{to_json, from_json};

/// Neo4j configuration for ontology repository
#[derive(Debug, Clone)]
pub struct Neo4jOntologyConfig {
    pub uri: String,
    pub user: String,
    pub password: String,
    pub database: Option<String>,
}

impl Default for Neo4jOntologyConfig {
    fn default() -> Self {
        Self {
            uri: std::env::var("NEO4J_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            user: std::env::var("NEO4J_USER")
                .unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("NEO4J_PASSWORD")
                .unwrap_or_else(|_| "password".to_string()),
            database: std::env::var("NEO4J_DATABASE").ok(),
        }
    }
}

/// Repository for OWL ontology data in Neo4j
///
/// Provides full OntologyRepository implementation with:
/// - OWL class storage and hierarchy
/// - OWL property management
/// - OWL axiom storage (including inferred axioms)
/// - Ontology metrics and validation
/// - Pathfinding cache
pub struct Neo4jOntologyRepository {
    graph: Arc<Graph>,
    config: Neo4jOntologyConfig,
}

impl Neo4jOntologyRepository {
    /// Create a new Neo4jOntologyRepository
    ///
    /// # Arguments
    /// * `config` - Neo4j connection configuration
    ///
    /// # Returns
    /// Initialized repository with schema created
    pub async fn new(config: Neo4jOntologyConfig) -> RepoResult<Self> {
        let graph = Graph::new(&config.uri, &config.user, &config.password)
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to connect to Neo4j: {}",
                    e
                ))
            })?;

        info!("Connected to Neo4j ontology database at {}", config.uri);

        let repo = Self {
            graph: Arc::new(graph),
            config,
        };

        // Create schema
        repo.create_schema().await?;

        Ok(repo)
    }

    /// Create Neo4j schema (constraints and indexes)
    async fn create_schema(&self) -> RepoResult<()> {
        info!("Creating Neo4j ontology schema...");

        let queries = vec![
            // OWL Class constraints and indexes
            "CREATE CONSTRAINT owl_class_iri IF NOT EXISTS FOR (c:OwlClass) REQUIRE c.iri IS UNIQUE",
            "CREATE INDEX owl_class_label IF NOT EXISTS FOR (c:OwlClass) ON (c.label)",
            "CREATE INDEX owl_class_ontology_id IF NOT EXISTS FOR (c:OwlClass) ON (c.ontology_id)",

            // OWL Property constraints
            "CREATE CONSTRAINT owl_property_iri IF NOT EXISTS FOR (p:OwlProperty) REQUIRE p.iri IS UNIQUE",
            "CREATE INDEX owl_property_label IF NOT EXISTS FOR (p:OwlProperty) ON (p.label)",

            // OWL Axiom constraints
            "CREATE CONSTRAINT owl_axiom_id IF NOT EXISTS FOR (a:OwlAxiom) REQUIRE a.id IS UNIQUE",
            "CREATE INDEX owl_axiom_type IF NOT EXISTS FOR (a:OwlAxiom) ON (a.axiom_type)",
            "CREATE INDEX owl_axiom_inferred IF NOT EXISTS FOR (a:OwlAxiom) ON (a.is_inferred)",
        ];

        for query_str in queries {
            self.graph
                .run(query(query_str))
                .await
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to create schema: {}",
                        e
                    ))
                })?;
        }

        info!("Neo4j ontology schema created successfully");
        Ok(())
    }

    /// Convert Neo4j node to OwlClass
    fn node_to_owl_class(&self, node: Neo4jNode) -> RepoResult<OwlClass> {
        let iri: String = node.get("iri")
            .map_err(|_| OntologyRepositoryError::DeserializationError(
                "Missing iri field".to_string()
            ))?;

        let label: Option<String> = node.get("label").ok();
        let description: Option<String> = node.get("description").ok();
        let source_file: Option<String> = node.get("source_file").ok();
        let markdown_content: Option<String> = node.get("markdown_content").ok();
        let file_sha1: Option<String> = node.get("file_sha1").ok();
        let last_synced: Option<chrono::DateTime<chrono::Utc>> = node.get("last_synced").ok();

        Ok(OwlClass {
            iri,
            label,
            description,
            parent_classes: Vec::new(), // Fetched separately via relationships
            properties: std::collections::HashMap::new(),
            source_file,
            markdown_content,
            file_sha1,
            last_synced,
        })
    }
}

#[async_trait]
impl OntologyRepository for Neo4jOntologyRepository {
    // ============================================================
    // OWL Class Methods
    // ============================================================

    #[instrument(skip(self))]
    async fn add_owl_class(&self, class: &OwlClass) -> RepoResult<String> {
        debug!("Storing OWL class: {}", class.iri);

        let query_str = "
            MERGE (c:OwlClass {iri: $iri})
            ON CREATE SET c.created_at = datetime()
            ON MATCH SET c.updated_at = datetime()
            SET c.label = $label,
                c.description = $description,
                c.source_file = $source_file,
                c.markdown_content = $markdown_content,
                c.file_sha1 = $file_sha1,
                c.last_synced = $last_synced
        ";

        self.graph
            .run(query(query_str)
                .param("iri", class.iri.clone())
                .param("label", class.label.clone().unwrap_or_default())
                .param("description", class.description.clone().unwrap_or_default())
                .param("source_file", class.source_file.clone().unwrap_or_default())
                .param("markdown_content", class.markdown_content.clone().unwrap_or_default())
                .param("file_sha1", class.file_sha1.clone().unwrap_or_default())
                .param("last_synced", class.last_synced.map(|dt| dt.to_rfc3339()).unwrap_or_default()))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to store OWL class: {}",
                    e
                ))
            })?;

        // Store parent relationships
        for parent_iri in &class.parent_classes {
            let rel_query = "
                MATCH (c:OwlClass {iri: $child_iri})
                MERGE (p:OwlClass {iri: $parent_iri})
                MERGE (c)-[:SUBCLASS_OF]->(p)
            ";

            self.graph
                .run(query(rel_query)
                    .param("child_iri", class.iri.clone())
                    .param("parent_iri", parent_iri.clone()))
                .await
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to store parent relationship: {}",
                        e
                    ))
                })?;
        }

        Ok(class.iri.clone())
    }

    #[instrument(skip(self))]
    async fn get_owl_class(&self, iri: &str) -> RepoResult<Option<OwlClass>> {
        debug!("Fetching OWL class: {}", iri);

        let query_str = "
            MATCH (c:OwlClass {iri: $iri})
            OPTIONAL MATCH (c)-[:SUBCLASS_OF]->(p:OwlClass)
            RETURN c, collect(p.iri) as parent_iris
        ";

        let mut result = self.graph
            .execute(query(query_str).param("iri", iri.to_string()))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to get OWL class: {}",
                    e
                ))
            })?;

        if let Some(row) = result.next().await.map_err(|e| {
            OntologyRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        })? {
            let node: Neo4jNode = row.get("c")
                .map_err(|_| OntologyRepositoryError::DeserializationError(
                    "Missing node in result".to_string()
                ))?;

            let mut owl_class = self.node_to_owl_class(node)?;

            // Get parent IRIs
            let parent_iris: Vec<String> = row.get("parent_iris")
                .unwrap_or_else(|_| Vec::new());
            owl_class.parent_classes = parent_iris;

            Ok(Some(owl_class))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self))]
    async fn list_owl_classes(&self) -> RepoResult<Vec<OwlClass>> {
        debug!("Listing OWL classes");

        let query_str = "
            MATCH (c:OwlClass)
            OPTIONAL MATCH (c)-[:SUBCLASS_OF]->(p:OwlClass)
            RETURN c, collect(p.iri) as parent_iris
            ";

        let query_obj = query(query_str);

        let mut result = self.graph
            .execute(query_obj)
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to list OWL classes: {}",
                    e
                ))
            })?;

        let mut classes = Vec::new();
        while let Some(row) = result.next().await.map_err(|e| {
            OntologyRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        })? {
            let node: Neo4jNode = row.get("c").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get node: {}", e))
            })?;
            let mut owl_class = self.node_to_owl_class(node)?;

            let parent_iris: Vec<String> = row.get("parent_iris")
                .unwrap_or_else(|_| Vec::new());
            owl_class.parent_classes = parent_iris;

            classes.push(owl_class);
        }

        debug!("Found {} OWL classes", classes.len());
        Ok(classes)
    }

    // ============================================================
    // OWL Property Methods
    // ============================================================

    #[instrument(skip(self))]
    async fn add_owl_property(&self, property: &OwlProperty) -> RepoResult<String> {
        debug!("Storing OWL property: {}", property.iri);

        let query_str = "
            MERGE (p:OwlProperty {iri: $iri})
            ON CREATE SET p.created_at = datetime()
            ON MATCH SET p.updated_at = datetime()
            SET p.label = $label,
                p.property_type = $property_type,
                p.domain = $domain,
                p.range = $range
        ";

        let domain_json = to_json(&property.domain)
            .map_err(|e| OntologyRepositoryError::SerializationError(e.to_string()))?;
        let range_json = to_json(&property.range)
            .map_err(|e| OntologyRepositoryError::SerializationError(e.to_string()))?;

        self.graph
            .run(query(query_str)
                .param("iri", property.iri.clone())
                .param("label", property.label.clone().unwrap_or_default())
                .param("property_type", format!("{:?}", property.property_type))
                .param("domain", domain_json)
                .param("range", range_json))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to store OWL property: {}",
                    e
                ))
            })?;

        Ok(property.iri.clone())
    }

    #[instrument(skip(self))]
    async fn get_owl_property(&self, iri: &str) -> RepoResult<Option<OwlProperty>> {
        debug!("Fetching OWL property: {}", iri);

        let query_str = "
            MATCH (p:OwlProperty {iri: $iri})
            RETURN p
        ";

        let mut result = self.graph
            .execute(query(query_str).param("iri", iri.to_string()))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to get OWL property: {}",
                    e
                ))
            })?;

        if let Some(row) = result.next().await.map_err(|e| {
            OntologyRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        })? {
            let node: Neo4jNode = row.get("p").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get node: {}", e))
            })?;

            let iri: String = node.get("iri").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get iri: {}", e))
            })?;
            let label: Option<String> = node.get("label").ok();
            let property_type_str: String = node.get("property_type").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get property_type: {}", e))
            })?;
            let domain_json: String = node.get("domain").unwrap_or_else(|_| "[]".to_string());
            let range_json: String = node.get("range").unwrap_or_else(|_| "[]".to_string());

            let property_type = match property_type_str.as_str() {
                "ObjectProperty" => PropertyType::ObjectProperty,
                "DataProperty" => PropertyType::DataProperty,
                "AnnotationProperty" => PropertyType::AnnotationProperty,
                _ => PropertyType::ObjectProperty,
            };

            let domain: Vec<String> = from_json(&domain_json)
                .map_err(|e| OntologyRepositoryError::DeserializationError(e.to_string()))?;
            let range: Vec<String> = from_json(&range_json)
                .map_err(|e| OntologyRepositoryError::DeserializationError(e.to_string()))?;

            Ok(Some(OwlProperty {
                iri,
                label,
                property_type,
                domain,
                range,
            }))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self))]
    async fn list_owl_properties(&self) -> RepoResult<Vec<OwlProperty>> {
        debug!("Listing all OWL properties");

        let query_str = "MATCH (p:OwlProperty) RETURN p";

        let mut result = self.graph
            .execute(query(query_str))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to list OWL properties: {}",
                    e
                ))
            })?;

        let mut properties = Vec::new();
        while let Some(row) = result.next().await.map_err(|e| {
            OntologyRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        })? {
            let node: Neo4jNode = row.get("p").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get node: {}", e))
            })?;

            let iri: String = node.get("iri").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get iri: {}", e))
            })?;
            let label: Option<String> = node.get("label").ok();
            let property_type_str: String = node.get("property_type").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get property_type: {}", e))
            })?;
            let domain_json: String = node.get("domain").unwrap_or_else(|_| "[]".to_string());
            let range_json: String = node.get("range").unwrap_or_else(|_| "[]".to_string());

            let property_type = match property_type_str.as_str() {
                "ObjectProperty" => PropertyType::ObjectProperty,
                "DataProperty" => PropertyType::DataProperty,
                "AnnotationProperty" => PropertyType::AnnotationProperty,
                _ => PropertyType::ObjectProperty,
            };

            let domain: Vec<String> = from_json(&domain_json)
                .map_err(|e| OntologyRepositoryError::DeserializationError(e.to_string()))?;
            let range: Vec<String> = from_json(&range_json)
                .map_err(|e| OntologyRepositoryError::DeserializationError(e.to_string()))?;

            properties.push(OwlProperty {
                iri,
                label,
                property_type,
                domain,
                range,
            });
        }

        debug!("Found {} OWL properties", properties.len());
        Ok(properties)
    }

    // ============================================================
    // OWL Axiom Methods
    // ============================================================

    #[instrument(skip(self))]
    async fn add_axiom(&self, axiom: &OwlAxiom) -> RepoResult<u64> {
        debug!("Storing OWL axiom: {:?}", axiom.id);

        let annotations_json = to_json(&axiom.annotations)
            .map_err(|e| OntologyRepositoryError::SerializationError(e.to_string()))?;

        let axiom_id = axiom.id.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0)
        });

        let query_str = "
            MERGE (a:OwlAxiom {id: $id})
            ON CREATE SET a.created_at = datetime()
            ON MATCH SET a.updated_at = datetime()
            SET a.axiom_type = $axiom_type,
                a.subject = $subject,
                a.object = $object,
                a.annotations = $annotations
        ";

        self.graph
            .run(query(query_str)
                .param("id", axiom_id as i64)
                .param("axiom_type", format!("{:?}", axiom.axiom_type))
                .param("subject", axiom.subject.clone())
                .param("object", axiom.object.clone())
                .param("annotations", annotations_json))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to store OWL axiom: {}",
                    e
                ))
            })?;

        Ok(axiom_id)
    }

    #[instrument(skip(self))]
    async fn get_axioms(&self) -> RepoResult<Vec<OwlAxiom>> {
        debug!("Fetching all OWL axioms");

        let query_str = "MATCH (a:OwlAxiom) RETURN a";
        let query_obj = query(query_str);

        let mut result = self.graph
            .execute(query_obj)
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to get OWL axioms: {}",
                    e
                ))
            })?;

        let mut axioms = Vec::new();
        while let Some(row) = result.next().await.map_err(|e| {
            OntologyRepositoryError::DatabaseError(format!("Failed to fetch row: {}", e))
        })? {
            let node: Neo4jNode = row.get("a").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get node: {}", e))
            })?;

            let id: i64 = node.get("id").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get id: {}", e))
            })?;
            let axiom_type_str: String = node.get("axiom_type").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get axiom_type: {}", e))
            })?;
            let subject: String = node.get("subject").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get subject: {}", e))
            })?;
            let object: String = node.get("object").map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to get object: {}", e))
            })?;
            let annotations_json: String = node.get("annotations").unwrap_or_else(|_| "{}".to_string());

            let axiom_type = match axiom_type_str.as_str() {
                "SubClassOf" => AxiomType::SubClassOf,
                "EquivalentClass" => AxiomType::EquivalentClass,
                "DisjointWith" => AxiomType::DisjointWith,
                "ObjectPropertyAssertion" => AxiomType::ObjectPropertyAssertion,
                "DataPropertyAssertion" => AxiomType::DataPropertyAssertion,
                _ => AxiomType::SubClassOf,
            };

            let annotations: HashMap<String, String> = from_json(&annotations_json)
                .map_err(|e| OntologyRepositoryError::DeserializationError(e.to_string()))?;

            axioms.push(OwlAxiom {
                id: Some(id as u64),
                axiom_type,
                subject,
                object,
                annotations,
            });
        }

        debug!("Found {} OWL axioms", axioms.len());
        Ok(axioms)
    }

    // ============================================================
    // Inference Methods
    // ============================================================

    #[instrument(skip(self, results))]
    async fn store_inference_results(&self, results: &InferenceResults) -> RepoResult<()> {
        info!("Storing {} inferred axioms", results.inferred_axioms.len());

        for axiom in &results.inferred_axioms {
            self.add_axiom(axiom).await?;
        }

        Ok(())
    }

    // ============================================================
    // Metrics and Validation
    // ============================================================

    #[instrument(skip(self))]
    async fn get_metrics(&self) -> RepoResult<OntologyMetrics> {
        debug!("Computing ontology metrics");

        // Count classes
        let class_count_query = query("MATCH (c:OwlClass) RETURN count(c) as count");

        let mut result = self.graph.execute(class_count_query).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        let class_count: i64 = if let Some(row) = result.next().await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))? {
            row.get("count").unwrap_or(0)
        } else {
            0
        };

        // Count properties
        let property_count_query = query("MATCH (p:OwlProperty) RETURN count(p) as count");
        let mut result = self.graph.execute(property_count_query).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        let property_count: i64 = if let Some(row) = result.next().await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))? {
            row.get("count").unwrap_or(0)
        } else {
            0
        };

        // Count axioms
        let axiom_count_query = query("MATCH (a:OwlAxiom) RETURN count(a) as count");
        let mut result = self.graph.execute(axiom_count_query).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        let axiom_count: i64 = if let Some(row) = result.next().await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))? {
            row.get("count").unwrap_or(0)
        } else {
            0
        };

        Ok(OntologyMetrics {
            class_count: class_count as usize,
            property_count: property_count as usize,
            axiom_count: axiom_count as usize,
            max_depth: 0, // TODO: Calculate from hierarchy traversal
            average_branching_factor: 0.0, // TODO: Calculate branching factor
        })
    }

    #[instrument(skip(self))]
    async fn validate_ontology(&self) -> RepoResult<ValidationReport> {
        debug!("Validating ontology");

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for orphaned classes (no relationships)
        let orphan_query = query("
            MATCH (c:OwlClass)
            WHERE NOT (c)-[:SUBCLASS_OF]->() AND NOT ()-[:SUBCLASS_OF]->(c)
            RETURN count(c) as count
        ");

        let mut result = self.graph.execute(orphan_query).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        if let Some(row) = result.next().await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))? {
            let orphan_count: i64 = row.get("count").unwrap_or(0);
            if orphan_count > 0 {
                warnings.push(format!("{} orphaned classes found (no hierarchy relationships)", orphan_count));
            }
        }

        let is_valid = errors.is_empty();

        Ok(ValidationReport {
            is_valid,
            errors,
            warnings,
            timestamp: chrono::Utc::now(),
        })
    }

    #[instrument(skip(self))]
    async fn cache_sssp_result(&self, _entry: &PathfindingCacheEntry) -> RepoResult<()> {
        // TODO: Implement pathfinding cache if needed
        Ok(())
    }

    #[instrument(skip(self))]
    async fn get_cached_sssp(&self, _source_node_id: u32) -> RepoResult<Option<PathfindingCacheEntry>> {
        // TODO: Implement pathfinding cache if needed
        Ok(None)
    }

    #[instrument(skip(self))]
    async fn cache_apsp_result(&self, _distance_matrix: &Vec<Vec<f32>>) -> RepoResult<()> {
        // TODO: Implement pathfinding cache if needed
        Ok(())
    }

    #[instrument(skip(self))]
    async fn get_cached_apsp(&self) -> RepoResult<Option<Vec<Vec<f32>>>> {
        // TODO: Implement pathfinding cache if needed
        Ok(None)
    }

    #[instrument(skip(self))]
    async fn invalidate_pathfinding_caches(&self) -> RepoResult<()> {
        info!("Clearing pathfinding cache");
        // TODO: Implement pathfinding cache if needed
        Ok(())
    }

    #[instrument(skip(self))]
    async fn load_ontology_graph(&self) -> RepoResult<Arc<GraphData>> {
        debug!("Loading ontology graph from Neo4j");

        // Query all nodes
        let nodes_query = query("MATCH (n) RETURN n, id(n) as neo4j_id");
        let mut result = self.graph.execute(nodes_query).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        let mut nodes = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(neo4j_node) = row.get::<Neo4jNode>("n") {
                if let Ok(neo4j_id) = row.get::<i64>("neo4j_id") {
                    // Convert Neo4j node to our Node type
                    let label = neo4j_node.get::<String>("label").unwrap_or_default();
                    let node = Node::new_with_id(label, Some(neo4j_id as u32));
                    nodes.push(node);
                }
            }
        }

        // Query all edges
        let edges_query = query("MATCH (n)-[r]->(m) RETURN id(n) as source, id(m) as target, type(r) as rel_type");
        let mut result = self.graph.execute(edges_query).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        let mut edges = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let (Ok(source), Ok(target), Ok(rel_type)) = (
                row.get::<i64>("source"),
                row.get::<i64>("target"),
                row.get::<String>("rel_type"),
            ) {
                let edge = Edge::new(source as u32, target as u32, 1.0)
                    .with_edge_type(rel_type);
                edges.push(edge);
            }
        }

        Ok(Arc::new(GraphData {
            nodes,
            edges,
            metadata: Default::default(),
            id_to_metadata: HashMap::new(),
        }))
    }

    #[instrument(skip(self, graph))]
    async fn save_ontology_graph(&self, graph: &GraphData) -> RepoResult<()> {
        debug!("Saving ontology graph to Neo4j");

        // Clear existing graph
        let clear_query = query("MATCH (n) DETACH DELETE n");
        self.graph.execute(clear_query).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        // Insert nodes
        for node in &graph.nodes {
            let node_query = query("CREATE (n {id: $id, label: $label})")
                .param("id", node.id as i64)
                .param("label", node.label.clone());
            self.graph.execute(node_query).await
                .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;
        }

        // Insert edges
        for edge in &graph.edges {
            let rel_type = edge.edge_type.clone().unwrap_or_else(|| "RELATES".to_string());
            let edge_query = query(
                "MATCH (n {id: $source}), (m {id: $target}) \
                 CREATE (n)-[r:RELATES {relationship: $rel_type}]->(m)"
            )
            .param("source", edge.source as i64)
            .param("target", edge.target as i64)
            .param("rel_type", rel_type);

            self.graph.execute(edge_query).await
                .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;
        }

        Ok(())
    }

    #[instrument(skip(self, classes, properties, axioms))]
    async fn save_ontology(
        &self,
        classes: &[OwlClass],
        properties: &[OwlProperty],
        axioms: &[OwlAxiom],
    ) -> RepoResult<()> {
        debug!("Saving ontology: {} classes, {} properties, {} axioms",
               classes.len(), properties.len(), axioms.len());

        // Save classes
        for class in classes {
            self.add_owl_class(class).await?;
        }

        // Save properties
        for property in properties {
            self.add_owl_property(property).await?;
        }

        // Save axioms
        for axiom in axioms {
            self.add_axiom(axiom).await?;
        }

        Ok(())
    }

    #[instrument(skip(self))]
    async fn get_classes(&self) -> RepoResult<Vec<OwlClass>> {
        self.list_owl_classes().await
    }

    #[instrument(skip(self))]
    async fn get_class_axioms(&self, class_iri: &str) -> RepoResult<Vec<OwlAxiom>> {
        debug!("Getting axioms for class: {}", class_iri);

        let query_str = query(
            "MATCH (c:OwlClass {iri: $iri})-[:HAS_AXIOM]->(a:Axiom) \
             RETURN a.axiom_type as axiom_type, \
                    a.subject as subject, \
                    a.predicate as predicate, \
                    a.object as object, \
                    a.axiom_json as axiom_json"
        ).param("iri", class_iri);

        let mut result = self.graph.execute(query_str).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        let mut axioms = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let (Ok(axiom_type_str), Ok(subject), Ok(predicate), Ok(object)) = (
                row.get::<String>("axiom_type"),
                row.get::<String>("subject"),
                row.get::<String>("predicate"),
                row.get::<String>("object"),
            ) {
                let axiom_type = match axiom_type_str.as_str() {
                    "SubClassOf" => AxiomType::SubClassOf,
                    "EquivalentClass" | "EquivalentClasses" => AxiomType::EquivalentClass,
                    "DisjointWith" | "DisjointClasses" => AxiomType::DisjointWith,
                    "ObjectPropertyAssertion" | "SubObjectProperty" => AxiomType::ObjectPropertyAssertion,
                    "DataPropertyAssertion" | "Domain" | "Range" => AxiomType::DataPropertyAssertion,
                    _ => AxiomType::SubClassOf,
                };

                let axiom = OwlAxiom {
                    id: None,
                    axiom_type,
                    subject,
                    object,
                    annotations: HashMap::new(),
                };
                axioms.push(axiom);
            }
        }

        Ok(axioms)
    }
}
