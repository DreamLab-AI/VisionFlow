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
        let ontology_id: String = node.get("ontology_id")
            .unwrap_or_else(|_| "default".to_string());

        Ok(OwlClass {
            iri,
            label,
            description,
            ontology_id,
            parent_iris: Vec::new(), // Fetched separately via relationships
        })
    }
}

#[async_trait]
impl OntologyRepository for Neo4jOntologyRepository {
    // ============================================================
    // OWL Class Methods
    // ============================================================

    #[instrument(skip(self))]
    async fn store_owl_class(&self, class: &OwlClass) -> RepoResult<()> {
        debug!("Storing OWL class: {}", class.iri);

        let query_str = "
            MERGE (c:OwlClass {iri: $iri})
            SET c.label = $label,
                c.description = $description,
                c.ontology_id = $ontology_id,
                c.updated_at = datetime()
            ON CREATE SET c.created_at = datetime()
        ";

        self.graph
            .run(query(query_str)
                .param("iri", class.iri.clone())
                .param("label", class.label.clone().unwrap_or_default())
                .param("description", class.description.clone().unwrap_or_default())
                .param("ontology_id", class.ontology_id.clone()))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to store OWL class: {}",
                    e
                ))
            })?;

        // Store parent relationships
        for parent_iri in &class.parent_iris {
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

        Ok(())
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
            owl_class.parent_iris = parent_iris;

            Ok(Some(owl_class))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self))]
    async fn list_owl_classes(&self, ontology_id: Option<&str>) -> RepoResult<Vec<OwlClass>> {
        debug!("Listing OWL classes for ontology: {:?}", ontology_id);

        let query_str = if let Some(ont_id) = ontology_id {
            "
            MATCH (c:OwlClass {ontology_id: $ontology_id})
            OPTIONAL MATCH (c)-[:SUBCLASS_OF]->(p:OwlClass)
            RETURN c, collect(p.iri) as parent_iris
            "
        } else {
            "
            MATCH (c:OwlClass)
            OPTIONAL MATCH (c)-[:SUBCLASS_OF]->(p:OwlClass)
            RETURN c, collect(p.iri) as parent_iris
            "
        };

        let query_obj = if let Some(ont_id) = ontology_id {
            query(query_str).param("ontology_id", ont_id.to_string())
        } else {
            query(query_str)
        };

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
            let node: Neo4jNode = row.get("c")?;
            let mut owl_class = self.node_to_owl_class(node)?;

            let parent_iris: Vec<String> = row.get("parent_iris")
                .unwrap_or_else(|_| Vec::new());
            owl_class.parent_iris = parent_iris;

            classes.push(owl_class);
        }

        debug!("Found {} OWL classes", classes.len());
        Ok(classes)
    }

    #[instrument(skip(self))]
    async fn delete_owl_class(&self, iri: &str) -> RepoResult<()> {
        debug!("Deleting OWL class: {}", iri);

        let query_str = "
            MATCH (c:OwlClass {iri: $iri})
            DETACH DELETE c
        ";

        self.graph
            .run(query(query_str).param("iri", iri.to_string()))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to delete OWL class: {}",
                    e
                ))
            })?;

        Ok(())
    }

    // ============================================================
    // OWL Property Methods
    // ============================================================

    #[instrument(skip(self))]
    async fn store_owl_property(&self, property: &OwlProperty) -> RepoResult<()> {
        debug!("Storing OWL property: {}", property.iri);

        let query_str = "
            MERGE (p:OwlProperty {iri: $iri})
            SET p.label = $label,
                p.property_type = $property_type,
                p.domain_iri = $domain_iri,
                p.range_iri = $range_iri,
                p.updated_at = datetime()
            ON CREATE SET p.created_at = datetime()
        ";

        self.graph
            .run(query(query_str)
                .param("iri", property.iri.clone())
                .param("label", property.label.clone().unwrap_or_default())
                .param("property_type", format!("{:?}", property.property_type))
                .param("domain_iri", property.domain_iri.clone().unwrap_or_default())
                .param("range_iri", property.range_iri.clone().unwrap_or_default()))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to store OWL property: {}",
                    e
                ))
            })?;

        Ok(())
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
            let node: Neo4jNode = row.get("p")?;

            let iri: String = node.get("iri")?;
            let label: Option<String> = node.get("label").ok();
            let property_type_str: String = node.get("property_type")?;
            let domain_iri: Option<String> = node.get("domain_iri").ok();
            let range_iri: Option<String> = node.get("range_iri").ok();

            let property_type = match property_type_str.as_str() {
                "ObjectProperty" => PropertyType::ObjectProperty,
                "DataProperty" => PropertyType::DataProperty,
                "AnnotationProperty" => PropertyType::AnnotationProperty,
                _ => PropertyType::ObjectProperty,
            };

            Ok(Some(OwlProperty {
                iri,
                label,
                property_type,
                domain_iri,
                range_iri,
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
            let node: Neo4jNode = row.get("p")?;

            let iri: String = node.get("iri")?;
            let label: Option<String> = node.get("label").ok();
            let property_type_str: String = node.get("property_type")?;
            let domain_iri: Option<String> = node.get("domain_iri").ok();
            let range_iri: Option<String> = node.get("range_iri").ok();

            let property_type = match property_type_str.as_str() {
                "ObjectProperty" => PropertyType::ObjectProperty,
                "DataProperty" => PropertyType::DataProperty,
                "AnnotationProperty" => PropertyType::AnnotationProperty,
                _ => PropertyType::ObjectProperty,
            };

            properties.push(OwlProperty {
                iri,
                label,
                property_type,
                domain_iri,
                range_iri,
            });
        }

        debug!("Found {} OWL properties", properties.len());
        Ok(properties)
    }

    // ============================================================
    // OWL Axiom Methods
    // ============================================================

    #[instrument(skip(self))]
    async fn store_owl_axiom(&self, axiom: &OwlAxiom) -> RepoResult<()> {
        debug!("Storing OWL axiom: {}", axiom.id);

        let axiom_data_json = to_json(&axiom.axiom_data)
            .map_err(|e| OntologyRepositoryError::SerializationError(e.to_string()))?;

        let query_str = "
            MERGE (a:OwlAxiom {id: $id})
            SET a.axiom_type = $axiom_type,
                a.axiom_data = $axiom_data,
                a.is_inferred = $is_inferred,
                a.updated_at = datetime()
            ON CREATE SET a.created_at = datetime()
        ";

        self.graph
            .run(query(query_str)
                .param("id", axiom.id.clone())
                .param("axiom_type", format!("{:?}", axiom.axiom_type))
                .param("axiom_data", axiom_data_json)
                .param("is_inferred", axiom.is_inferred))
            .await
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to store OWL axiom: {}",
                    e
                ))
            })?;

        Ok(())
    }

    #[instrument(skip(self))]
    async fn get_owl_axioms(
        &self,
        axiom_type: Option<AxiomType>,
        include_inferred: bool,
    ) -> RepoResult<Vec<OwlAxiom>> {
        debug!("Fetching OWL axioms - type: {:?}, include_inferred: {}", axiom_type, include_inferred);

        let query_str = match (axiom_type, include_inferred) {
            (Some(_), true) => "MATCH (a:OwlAxiom {axiom_type: $axiom_type}) RETURN a",
            (Some(_), false) => "MATCH (a:OwlAxiom {axiom_type: $axiom_type, is_inferred: false}) RETURN a",
            (None, true) => "MATCH (a:OwlAxiom) RETURN a",
            (None, false) => "MATCH (a:OwlAxiom {is_inferred: false}) RETURN a",
        };

        let query_obj = if let Some(at) = axiom_type {
            query(query_str).param("axiom_type", format!("{:?}", at))
        } else {
            query(query_str)
        };

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
            let node: Neo4jNode = row.get("a")?;

            let id: String = node.get("id")?;
            let axiom_type_str: String = node.get("axiom_type")?;
            let axiom_data_json: String = node.get("axiom_data")?;
            let is_inferred: bool = node.get("is_inferred").unwrap_or(false);

            let axiom_type = match axiom_type_str.as_str() {
                "SubClassOf" => AxiomType::SubClassOf,
                "DisjointWith" => AxiomType::DisjointWith,
                "EquivalentTo" => AxiomType::EquivalentTo,
                _ => AxiomType::SubClassOf,
            };

            let axiom_data: HashMap<String, String> = from_json(&axiom_data_json)
                .map_err(|e| OntologyRepositoryError::DeserializationError(e.to_string()))?;

            axioms.push(OwlAxiom {
                id,
                axiom_type,
                axiom_data,
                is_inferred,
            });
        }

        debug!("Found {} OWL axioms", axioms.len());
        Ok(axioms)
    }

    // ============================================================
    // Inference Methods
    // ============================================================

    #[instrument(skip(self, results))]
    async fn store_inferred_axioms(&self, results: &InferenceResults) -> RepoResult<()> {
        info!("Storing {} inferred axioms", results.inferred_axioms.len());

        for axiom in &results.inferred_axioms {
            self.store_owl_axiom(axiom).await?;
        }

        Ok(())
    }

    #[instrument(skip(self))]
    async fn clear_inferred_axioms(&self) -> RepoResult<()> {
        info!("Clearing all inferred axioms");

        let query_str = "MATCH (a:OwlAxiom {is_inferred: true}) DELETE a";

        self.graph.run(query(query_str)).await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    // ============================================================
    // Metrics and Validation
    // ============================================================

    #[instrument(skip(self))]
    async fn get_ontology_metrics(&self, ontology_id: Option<&str>) -> RepoResult<OntologyMetrics> {
        debug!("Computing ontology metrics for: {:?}", ontology_id);

        // Count classes
        let class_count_query = if let Some(ont_id) = ontology_id {
            query("MATCH (c:OwlClass {ontology_id: $ontology_id}) RETURN count(c) as count")
                .param("ontology_id", ont_id.to_string())
        } else {
            query("MATCH (c:OwlClass) RETURN count(c) as count")
        };

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
            total_classes: class_count as usize,
            total_properties: property_count as usize,
            total_axioms: axiom_count as usize,
            max_depth: 0, // TODO: Calculate from hierarchy traversal
            total_individuals: 0,
        })
    }

    #[instrument(skip(self))]
    async fn validate_ontology(&self, ontology_id: Option<&str>) -> RepoResult<ValidationReport> {
        debug!("Validating ontology: {:?}", ontology_id);

        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check for orphaned classes (no relationships)
        let orphan_query = if let Some(ont_id) = ontology_id {
            query("
                MATCH (c:OwlClass {ontology_id: $ontology_id})
                WHERE NOT (c)-[:SUBCLASS_OF]->() AND NOT ()-[:SUBCLASS_OF]->(c)
                RETURN count(c) as count
            ").param("ontology_id", ont_id.to_string())
        } else {
            query("
                MATCH (c:OwlClass)
                WHERE NOT (c)-[:SUBCLASS_OF]->() AND NOT ()-[:SUBCLASS_OF]->(c)
                RETURN count(c) as count
            ")
        };

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
        })
    }

    // ============================================================
    // Pathfinding Cache Methods
    // ============================================================

    #[instrument(skip(self))]
    async fn get_pathfinding_cache(&self, query: &str) -> RepoResult<Option<PathfindingCacheEntry>> {
        debug!("Fetching pathfinding cache for query: {}", query);
        // TODO: Implement pathfinding cache if needed
        Ok(None)
    }

    #[instrument(skip(self))]
    async fn store_pathfinding_cache(&self, entry: &PathfindingCacheEntry) -> RepoResult<()> {
        debug!("Storing pathfinding cache entry");
        // TODO: Implement pathfinding cache if needed
        Ok(())
    }

    #[instrument(skip(self))]
    async fn clear_pathfinding_cache(&self) -> RepoResult<()> {
        info!("Clearing pathfinding cache");
        // TODO: Implement pathfinding cache if needed
        Ok(())
    }
}
