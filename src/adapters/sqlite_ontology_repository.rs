// src/adapters/sqlite_ontology_repository.rs
//! SQLite Ontology Repository Adapter with Rich Metadata Support
//!
//! Implements OntologyRepository trait using SQLite with comprehensive metadata.
//! Supports all fields from the ontology_parser.py schema including:
//! - Core identification (term_id, preferred_term)
//! - Classification (source_domain, version, type)
//! - Quality metrics (quality_score, authority_score, status, maturity)
//! - OWL2 properties (owl_physicality, owl_role)
//! - Domain relationships (belongs_to_domain, bridges_to_domain)
//! - Semantic relationships (has-part, uses, enables, requires, etc.)
//!
//! Schema version: 2 (see migrations/002_rich_ontology_metadata.sql)

use async_trait::async_trait;
use log::{debug, info, warn};
use rusqlite::{params, Connection, OptionalExtension, Row};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::instrument;

use crate::models::graph::GraphData;
use crate::ports::ontology_repository::{
    AxiomType, InferenceResults, OntologyMetrics, OntologyRepository,
    OntologyRepositoryError, OwlAxiom, OwlClass, OwlProperty,
    PathfindingCacheEntry, PropertyType, Result as RepoResult,
    ValidationReport,
};
use crate::repositories::generic_repository::{convert_rusqlite_error, SqliteRepository};
use crate::utils::json::{from_json, to_json};

/// Extended OwlClass with rich metadata support
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwlClassExtended {
    // Core identification
    pub iri: String,
    pub term_id: Option<String>,
    pub preferred_term: Option<String>,

    // Basic metadata
    pub label: Option<String>,
    pub description: Option<String>,
    pub parent_classes: Vec<String>,

    // Classification
    pub source_domain: Option<String>,
    pub version: Option<String>,
    pub class_type: Option<String>,

    // Quality metrics
    pub status: Option<String>,
    pub maturity: Option<String>,
    pub quality_score: Option<f32>,
    pub authority_score: Option<f32>,
    pub public_access: Option<bool>,
    pub content_status: Option<String>,

    // OWL2 properties
    pub owl_physicality: Option<String>,
    pub owl_role: Option<String>,

    // Domain relationships
    pub belongs_to_domain: Option<String>,
    pub bridges_to_domain: Option<String>,

    // Source tracking
    pub source_file: Option<String>,
    pub file_sha1: Option<String>,
    pub markdown_content: Option<String>,
    pub last_synced: Option<chrono::DateTime<chrono::Utc>>,

    // Extensibility
    pub properties: HashMap<String, String>,
    pub additional_metadata: Option<String>, // JSON
}

/// Semantic relationship between classes
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwlRelationship {
    pub source_class_iri: String,
    pub relationship_type: String,
    pub target_class_iri: String,
    pub confidence: f32,
    pub is_inferred: bool,
}

/// SQLite-based ontology repository with rich metadata support
pub struct SqliteOntologyRepository {
    base: SqliteRepository,
    default_ontology_id: String,
}

impl SqliteOntologyRepository {
    /// Create a new SqliteOntologyRepository
    pub fn new(db_path: &str) -> Result<Self, OntologyRepositoryError> {
        let base = SqliteRepository::new(db_path).map_err(|e| {
            OntologyRepositoryError::DatabaseError(format!("Failed to create base repository: {}", e))
        })?;

        // Ensure schema is created
        let conn = base.get_connection();
        let conn_guard = conn.lock().map_err(|e| {
            OntologyRepositoryError::DatabaseError(format!("Failed to lock connection: {}", e))
        })?;
        Self::ensure_schema(&conn_guard)?;
        drop(conn_guard);

        info!("Initialized SqliteOntologyRepository at {}", db_path);

        Ok(Self {
            base,
            default_ontology_id: "default".to_string(),
        })
    }

    /// Ensure schema exists (idempotent)
    fn ensure_schema(conn: &Connection) -> RepoResult<()> {
        // Check if schema exists
        let schema_version: Option<i64> = conn
            .query_row(
                "SELECT version FROM schema_version WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .ok()
            .flatten();

        if let Some(version) = schema_version {
            info!("Schema version {} detected", version);
            if version < 2 {
                warn!("Schema version {} is outdated. Run migration 002_rich_ontology_metadata.sql", version);
            }
            return Ok(());
        }

        // Schema doesn't exist, create it
        warn!("Schema not found, creating basic schema. Consider running full migration.");
        Self::create_basic_schema(conn)?;
        Ok(())
    }

    /// Create basic schema (backward compatibility)
    fn create_basic_schema(conn: &Connection) -> RepoResult<()> {
        info!("Creating basic ontology schema...");

        // Create ontologies table
        conn.execute(
            r#"
            CREATE TABLE IF NOT EXISTS ontologies (
                ontology_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_type TEXT NOT NULL CHECK (source_type IN ('file', 'url', 'embedded')),
                base_iri TEXT,
                version_iri TEXT,
                title TEXT,
                description TEXT,
                author TEXT,
                version TEXT,
                content_hash TEXT NOT NULL,
                axiom_count INTEGER DEFAULT 0,
                class_count INTEGER DEFAULT 0,
                property_count INTEGER DEFAULT 0,
                parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_validated_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
            [],
        )
        .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        // Insert default ontology
        conn.execute(
            r#"
            INSERT OR IGNORE INTO ontologies (
                ontology_id, source_path, source_type, content_hash, title
            ) VALUES ('default', 'default', 'embedded', 'default', 'Default Ontology')
            "#,
            [],
        )
        .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))?;

        info!("✅ Basic ontology schema created");
        Ok(())
    }

    /// Convert database row to OwlClass
    fn row_to_owl_class(row: &Row) -> Result<OwlClass, rusqlite::Error> {
        let iri: String = row.get(0)?;
        let label: Option<String> = row.get(1)?;
        let description: Option<String> = row.get(2)?;
        let parent_class_iri: Option<String> = row.get(3)?;
        let source_file: Option<String> = row.get(4)?;
        let file_sha1: Option<String> = row.get(5)?;
        let markdown_content: Option<String> = row.get(6)?;
        let last_synced: Option<chrono::DateTime<chrono::Utc>> = row.get(7)?;

        let parent_classes = if let Some(parent) = parent_class_iri {
            vec![parent]
        } else {
            Vec::new()
        };

        Ok(OwlClass {
            iri,
            label,
            description,
            parent_classes,
            properties: HashMap::new(),
            source_file,
            markdown_content,
            file_sha1,
            last_synced,
        })
    }

    /// Convert database row to OwlClassExtended (with all rich metadata)
    fn row_to_owl_class_extended(row: &Row) -> Result<OwlClassExtended, rusqlite::Error> {
        let iri: String = row.get("class_iri")?;
        let term_id: Option<String> = row.get("term_id")?;
        let preferred_term: Option<String> = row.get("preferred_term")?;
        let label: Option<String> = row.get("label")?;
        let description: Option<String> = row.get("comment")?;
        let parent_class_iri: Option<String> = row.get("parent_class_iri")?;

        let source_domain: Option<String> = row.get("source_domain")?;
        let version: Option<String> = row.get("version")?;
        let class_type: Option<String> = row.get("type")?;

        let status: Option<String> = row.get("status")?;
        let maturity: Option<String> = row.get("maturity")?;
        let quality_score: Option<f32> = row.get("quality_score")?;
        let authority_score: Option<f32> = row.get("authority_score")?;
        let public_access: Option<i32> = row.get("public_access")?;
        let content_status: Option<String> = row.get("content_status")?;

        let owl_physicality: Option<String> = row.get("owl_physicality")?;
        let owl_role: Option<String> = row.get("owl_role")?;

        let belongs_to_domain: Option<String> = row.get("belongs_to_domain")?;
        let bridges_to_domain: Option<String> = row.get("bridges_to_domain")?;

        let source_file: Option<String> = row.get("source_file")?;
        let file_sha1: Option<String> = row.get("file_sha1")?;
        let markdown_content: Option<String> = row.get("markdown_content")?;
        let last_synced: Option<chrono::DateTime<chrono::Utc>> = row.get("last_synced")?;

        let additional_metadata: Option<String> = row.get("additional_metadata")?;

        let parent_classes = if let Some(parent) = parent_class_iri {
            vec![parent]
        } else {
            Vec::new()
        };

        Ok(OwlClassExtended {
            iri,
            term_id,
            preferred_term,
            label,
            description,
            parent_classes,
            source_domain,
            version,
            class_type,
            status,
            maturity,
            quality_score,
            authority_score,
            public_access: public_access.map(|v| v != 0),
            content_status,
            owl_physicality,
            owl_role,
            belongs_to_domain,
            bridges_to_domain,
            source_file,
            file_sha1,
            markdown_content,
            last_synced,
            properties: HashMap::new(),
            additional_metadata,
        })
    }

    /// Convert OwlClass to OwlClassExtended (for backward compatibility)
    fn owl_class_to_extended(class: &OwlClass) -> OwlClassExtended {
        OwlClassExtended {
            iri: class.iri.clone(),
            term_id: None,
            preferred_term: None,
            label: class.label.clone(),
            description: class.description.clone(),
            parent_classes: class.parent_classes.clone(),
            source_domain: None,
            version: None,
            class_type: None,
            status: None,
            maturity: None,
            quality_score: None,
            authority_score: None,
            public_access: None,
            content_status: None,
            owl_physicality: None,
            owl_role: None,
            belongs_to_domain: None,
            bridges_to_domain: None,
            source_file: class.source_file.clone(),
            file_sha1: class.file_sha1.clone(),
            markdown_content: class.markdown_content.clone(),
            last_synced: class.last_synced,
            properties: class.properties.clone(),
            additional_metadata: None,
        }
    }

    /// Convert OwlClassExtended to OwlClass (for backward compatibility)
    fn extended_to_owl_class(extended: &OwlClassExtended) -> OwlClass {
        OwlClass {
            iri: extended.iri.clone(),
            label: extended.label.clone(),
            description: extended.description.clone(),
            parent_classes: extended.parent_classes.clone(),
            properties: extended.properties.clone(),
            source_file: extended.source_file.clone(),
            markdown_content: extended.markdown_content.clone(),
            file_sha1: extended.file_sha1.clone(),
            last_synced: extended.last_synced,
        }
    }

    /// Add relationship between classes
    pub async fn add_relationship(&self, relationship: &OwlRelationship) -> RepoResult<()> {
        let rel = relationship.clone();
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                conn.execute(
                    r#"
                    INSERT OR REPLACE INTO owl_relationships
                        (ontology_id, source_class_iri, relationship_type, target_class_iri, confidence, is_inferred)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                    "#,
                    params![
                        ontology_id,
                        rel.source_class_iri,
                        rel.relationship_type,
                        rel.target_class_iri,
                        rel.confidence,
                        if rel.is_inferred { 1 } else { 0 }
                    ],
                )
                .map_err(convert_rusqlite_error)?;

                debug!(
                    "Added relationship: {} --[{}]--> {}",
                    rel.source_class_iri, rel.relationship_type, rel.target_class_iri
                );
                Ok(())
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    /// Get relationships for a class
    pub async fn get_relationships(&self, class_iri: &str) -> RepoResult<Vec<OwlRelationship>> {
        let iri = class_iri.to_string();
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                let mut stmt = conn
                    .prepare(
                        r#"
                        SELECT source_class_iri, relationship_type, target_class_iri, confidence, is_inferred
                        FROM owl_relationships
                        WHERE ontology_id = ?1 AND (source_class_iri = ?2 OR target_class_iri = ?2)
                        "#,
                    )
                    .map_err(convert_rusqlite_error)?;

                let relationships = stmt
                    .query_map(params![ontology_id, iri], |row| {
                        Ok(OwlRelationship {
                            source_class_iri: row.get(0)?,
                            relationship_type: row.get(1)?,
                            target_class_iri: row.get(2)?,
                            confidence: row.get(3)?,
                            is_inferred: row.get::<_, i32>(4)? != 0,
                        })
                    })
                    .map_err(convert_rusqlite_error)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(convert_rusqlite_error)?;

                Ok(relationships)
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    /// Query classes by quality score
    pub async fn query_classes_by_quality(&self, min_score: f32) -> RepoResult<Vec<OwlClassExtended>> {
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                let mut stmt = conn
                    .prepare(
                        r#"
                        SELECT class_iri, term_id, preferred_term, label, comment, parent_class_iri,
                               source_domain, version, type, status, maturity,
                               quality_score, authority_score, public_access, content_status,
                               owl_physicality, owl_role, belongs_to_domain, bridges_to_domain,
                               source_file, file_sha1, markdown_content, last_synced, additional_metadata
                        FROM owl_classes
                        WHERE ontology_id = ?1 AND quality_score >= ?2
                        ORDER BY quality_score DESC
                        "#,
                    )
                    .map_err(convert_rusqlite_error)?;

                let classes = stmt
                    .query_map(params![ontology_id, min_score], Self::row_to_owl_class_extended)
                    .map_err(convert_rusqlite_error)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(convert_rusqlite_error)?;

                Ok(classes)
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    /// Get connection for advanced operations
    pub fn get_connection(&self) -> Arc<std::sync::Mutex<Connection>> {
        self.base.get_connection()
    }
}

#[async_trait]
impl OntologyRepository for SqliteOntologyRepository {
    async fn load_ontology_graph(&self) -> RepoResult<Arc<GraphData>> {
        // Load graph data from ontology
        warn!("load_ontology_graph not yet fully implemented");
        Ok(Arc::new(GraphData::new()))
    }

    async fn save_ontology_graph(&self, _graph: &GraphData) -> RepoResult<()> {
        warn!("save_ontology_graph not yet fully implemented");
        Ok(())
    }

    async fn save_ontology(
        &self,
        classes: &[OwlClass],
        properties: &[OwlProperty],
        axioms: &[OwlAxiom],
    ) -> RepoResult<()> {
        let classes_vec = classes.to_vec();
        let properties_vec = properties.to_vec();
        let axioms_vec = axioms.to_vec();
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_transaction(|tx| {
                // Insert classes
                let mut class_stmt = tx
                    .prepare(
                        r#"
                        INSERT OR REPLACE INTO owl_classes
                            (ontology_id, class_iri, label, comment, parent_class_iri,
                             source_file, file_sha1, markdown_content, last_synced)
                        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                        "#,
                    )
                    .map_err(convert_rusqlite_error)?;

                for class in &classes_vec {
                    let parent = class.parent_classes.first().cloned();
                    class_stmt
                        .execute(params![
                            ontology_id,
                            class.iri,
                            class.label,
                            class.description,
                            parent,
                            class.source_file,
                            class.file_sha1,
                            class.markdown_content,
                            class.last_synced,
                        ])
                        .map_err(convert_rusqlite_error)?;
                }

                // Insert properties
                let mut prop_stmt = tx
                    .prepare(
                        r#"
                        INSERT OR REPLACE INTO owl_properties
                            (ontology_id, property_iri, property_type, label, comment,
                             domain_class_iri, range_class_iri)
                        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                        "#,
                    )
                    .map_err(convert_rusqlite_error)?;

                for prop in &properties_vec {
                    let prop_type_str = match prop.property_type {
                        PropertyType::ObjectProperty => "ObjectProperty",
                        PropertyType::DataProperty => "DataProperty",
                        PropertyType::AnnotationProperty => "AnnotationProperty",
                    };

                    let domain = prop.domain.first().cloned();
                    let range = prop.range.first().cloned();

                    prop_stmt
                        .execute(params![
                            ontology_id,
                            prop.iri,
                            prop_type_str,
                            prop.label,
                            None::<String>, // comment
                            domain,
                            range,
                        ])
                        .map_err(convert_rusqlite_error)?;
                }

                info!(
                    "Saved ontology: {} classes, {} properties, {} axioms",
                    classes_vec.len(),
                    properties_vec.len(),
                    axioms_vec.len()
                );
                Ok(())
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    #[instrument(skip(self, class), level = "debug")]
    async fn add_owl_class(&self, class: &OwlClass) -> RepoResult<String> {
        let class = class.clone();
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                let parent = class.parent_classes.first().cloned();

                conn.execute(
                    r#"
                    INSERT OR REPLACE INTO owl_classes
                        (ontology_id, class_iri, label, comment, parent_class_iri,
                         source_file, file_sha1, markdown_content, last_synced)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                    "#,
                    params![
                        ontology_id,
                        class.iri,
                        class.label,
                        class.description,
                        parent,
                        class.source_file,
                        class.file_sha1,
                        class.markdown_content,
                        class.last_synced,
                    ],
                )
                .map_err(convert_rusqlite_error)?;

                debug!("Added OWL class: {}", class.iri);
                Ok(class.iri.clone())
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    async fn get_owl_class(&self, iri: &str) -> RepoResult<Option<OwlClass>> {
        let iri = iri.to_string();
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                conn.query_row(
                    r#"
                    SELECT class_iri, label, comment, parent_class_iri,
                           source_file, file_sha1, markdown_content, last_synced
                    FROM owl_classes
                    WHERE ontology_id = ?1 AND class_iri = ?2
                    "#,
                    params![ontology_id, iri],
                    Self::row_to_owl_class,
                )
                .optional()
                .map_err(|e| {
                    crate::repositories::generic_repository::RepositoryError::DatabaseError(
                        e.to_string(),
                    )
                })
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    async fn list_owl_classes(&self) -> RepoResult<Vec<OwlClass>> {
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                let mut stmt = conn
                    .prepare(
                        r#"
                        SELECT class_iri, label, comment, parent_class_iri,
                               source_file, file_sha1, markdown_content, last_synced
                        FROM owl_classes
                        WHERE ontology_id = ?1
                        "#,
                    )
                    .map_err(convert_rusqlite_error)?;

                let classes = stmt
                    .query_map(params![ontology_id], Self::row_to_owl_class)
                    .map_err(convert_rusqlite_error)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(convert_rusqlite_error)?;

                Ok(classes)
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    async fn add_owl_property(&self, property: &OwlProperty) -> RepoResult<String> {
        let property = property.clone();
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                let prop_type_str = match property.property_type {
                    PropertyType::ObjectProperty => "ObjectProperty",
                    PropertyType::DataProperty => "DataProperty",
                    PropertyType::AnnotationProperty => "AnnotationProperty",
                };

                let domain = property.domain.first().cloned();
                let range = property.range.first().cloned();

                conn.execute(
                    r#"
                    INSERT OR REPLACE INTO owl_properties
                        (ontology_id, property_iri, property_type, label, comment,
                         domain_class_iri, range_class_iri)
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                    "#,
                    params![
                        ontology_id,
                        property.iri,
                        prop_type_str,
                        property.label,
                        None::<String>,
                        domain,
                        range,
                    ],
                )
                .map_err(convert_rusqlite_error)?;

                debug!("Added OWL property: {}", property.iri);
                Ok(property.iri.clone())
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    async fn get_owl_property(&self, iri: &str) -> RepoResult<Option<OwlProperty>> {
        let iri = iri.to_string();
        let ontology_id = self.default_ontology_id.clone();

        self.base
            .execute_blocking(move |conn| {
                conn.query_row(
                    r#"
                    SELECT property_iri, label, property_type, domain_class_iri, range_class_iri
                    FROM owl_properties
                    WHERE ontology_id = ?1 AND property_iri = ?2
                    "#,
                    params![ontology_id, iri],
                    |row| {
                        let property_type_str: String = row.get(2)?;
                        let property_type = match property_type_str.as_str() {
                            "ObjectProperty" => PropertyType::ObjectProperty,
                            "DataProperty" => PropertyType::DataProperty,
                            "AnnotationProperty" => PropertyType::AnnotationProperty,
                            _ => PropertyType::ObjectProperty,
                        };

                        let domain: Option<String> = row.get(3)?;
                        let range: Option<String> = row.get(4)?;

                        Ok(OwlProperty {
                            iri: row.get(0)?,
                            label: row.get(1)?,
                            property_type,
                            domain: domain.map(|d| vec![d]).unwrap_or_default(),
                            range: range.map(|r| vec![r]).unwrap_or_default(),
                        })
                    },
                )
                .optional()
                .map_err(|e| {
                    crate::repositories::generic_repository::RepositoryError::DatabaseError(
                        e.to_string(),
                    )
                })
            })
            .await
            .map_err(|e| OntologyRepositoryError::DatabaseError(e.to_string()))
    }

    async fn list_owl_properties(&self) -> RepoResult<Vec<OwlProperty>> {
        warn!("list_owl_properties not yet fully implemented");
        Ok(Vec::new())
    }

    async fn get_classes(&self) -> RepoResult<Vec<OwlClass>> {
        self.list_owl_classes().await
    }

    async fn get_axioms(&self) -> RepoResult<Vec<OwlAxiom>> {
        warn!("get_axioms not yet fully implemented");
        Ok(Vec::new())
    }

    async fn add_axiom(&self, _axiom: &OwlAxiom) -> RepoResult<u64> {
        warn!("add_axiom not yet fully implemented");
        Ok(0)
    }

    async fn get_class_axioms(&self, _class_iri: &str) -> RepoResult<Vec<OwlAxiom>> {
        warn!("get_class_axioms not yet fully implemented");
        Ok(Vec::new())
    }

    async fn get_metrics(&self) -> RepoResult<OntologyMetrics> {
        // Get counts from database
        let class_count = self.list_owl_classes().await?.len();
        let property_count = self.list_owl_properties().await?.len();
        let axiom_count = self.get_axioms().await?.len();

        Ok(OntologyMetrics {
            class_count,
            property_count,
            axiom_count,
            max_depth: 0, // TODO: Calculate actual depth
            average_branching_factor: 0.0, // TODO: Calculate actual branching factor
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_ontology_repository_creation() {
        let repo = SqliteOntologyRepository::new(":memory:").unwrap();
        let classes = repo.list_owl_classes().await.unwrap();
        assert_eq!(classes.len(), 0);
    }

    #[tokio::test]
    async fn test_add_and_get_owl_class() {
        let repo = SqliteOntologyRepository::new(":memory:").unwrap();

        let class = OwlClass {
            iri: "http://example.com/TestClass".to_string(),
            label: Some("Test Class".to_string()),
            description: Some("A test class".to_string()),
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        };

        repo.add_owl_class(&class).await.unwrap();

        let retrieved = repo.get_owl_class(&class.iri).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().label, Some("Test Class".to_string()));
    }

    #[tokio::test]
    async fn test_add_relationship() {
        let repo = SqliteOntologyRepository::new(":memory:").unwrap();

        let class1 = OwlClass {
            iri: "http://example.com/Class1".to_string(),
            label: Some("Class 1".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        };

        let class2 = OwlClass {
            iri: "http://example.com/Class2".to_string(),
            label: Some("Class 2".to_string()),
            description: None,
            parent_classes: vec![],
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        };

        repo.add_owl_class(&class1).await.unwrap();
        repo.add_owl_class(&class2).await.unwrap();

        let relationship = OwlRelationship {
            source_class_iri: class1.iri.clone(),
            relationship_type: "has-part".to_string(),
            target_class_iri: class2.iri.clone(),
            confidence: 1.0,
            is_inferred: false,
        };

        repo.add_relationship(&relationship).await.unwrap();

        let relationships = repo.get_relationships(&class1.iri).await.unwrap();
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].relationship_type, "has-part");
    }
}
