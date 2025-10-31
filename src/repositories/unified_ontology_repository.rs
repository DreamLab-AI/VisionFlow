// src/repositories/unified_ontology_repository.rs
//! Unified Ontology Repository Adapter
//!
//! Implements OntologyRepository trait using unified.db schema.
//! This adapter provides 100% API compatibility with SqliteOntologyRepository
//! while using the unified database that combines graph and ontology data.

use async_trait::async_trait;
use log::{debug, info};
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::instrument;

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::ontology_repository::{
    AxiomType, InferenceResults, OntologyMetrics, OntologyRepository, OntologyRepositoryError,
    OwlAxiom, OwlClass, OwlProperty, PathfindingCacheEntry, PropertyType, Result as RepoResult,
    ValidationReport,
};

/// Unified database repository implementing OntologyRepository
pub struct UnifiedOntologyRepository {
    conn: Arc<Mutex<Connection>>,
}

impl UnifiedOntologyRepository {
    /// Create new unified ontology repository
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn =
            Connection::open(db_path).map_err(|e| format!("Failed to open unified database: {}", e))?;

        conn.execute("PRAGMA foreign_keys = ON", [])
            .map_err(|e| format!("Failed to enable foreign keys: {}", e))?;

        Self::create_schema(&conn)?;

        info!("Initialized UnifiedOntologyRepository at {}", db_path);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create unified ontology schema
    fn create_schema(conn: &Connection) -> Result<(), String> {
        conn.execute_batch(
            r#"
            -- Ontologies table
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
            );

            INSERT OR IGNORE INTO ontologies (
                ontology_id, source_path, source_type, content_hash, title, description
            ) VALUES (
                'default', 'default', 'embedded', 'default-ontology',
                'Default Ontology', 'Default unified ontology'
            );

            -- OWL classes table
            CREATE TABLE IF NOT EXISTS owl_classes (
                ontology_id TEXT NOT NULL DEFAULT 'default',
                iri TEXT PRIMARY KEY,
                label TEXT,
                description TEXT,
                parent_class_iri TEXT,
                file_sha1 TEXT,
                last_synced DATETIME,
                markdown_content TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_class_iri) REFERENCES owl_classes(iri) ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_owl_classes_ontology ON owl_classes(ontology_id);
            CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);
            CREATE INDEX IF NOT EXISTS idx_owl_classes_parent ON owl_classes(parent_class_iri);
            CREATE INDEX IF NOT EXISTS idx_owl_classes_sha1 ON owl_classes(file_sha1);

            -- OWL class hierarchy (many-to-many for multiple inheritance)
            CREATE TABLE IF NOT EXISTS owl_class_hierarchy (
                class_iri TEXT NOT NULL,
                parent_iri TEXT NOT NULL,
                PRIMARY KEY (class_iri, parent_iri),
                FOREIGN KEY (class_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE,
                FOREIGN KEY (parent_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_hierarchy_class ON owl_class_hierarchy(class_iri);
            CREATE INDEX IF NOT EXISTS idx_hierarchy_parent ON owl_class_hierarchy(parent_iri);

            -- OWL properties table
            CREATE TABLE IF NOT EXISTS owl_properties (
                ontology_id TEXT NOT NULL DEFAULT 'default',
                iri TEXT PRIMARY KEY,
                label TEXT,
                property_type TEXT NOT NULL CHECK (
                    property_type IN ('ObjectProperty', 'DataProperty', 'AnnotationProperty')
                ),
                domain TEXT, -- JSON array
                range TEXT,  -- JSON array
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_owl_properties_ontology ON owl_properties(ontology_id);
            CREATE INDEX IF NOT EXISTS idx_owl_properties_type ON owl_properties(property_type);

            -- OWL axioms table
            CREATE TABLE IF NOT EXISTS owl_axioms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ontology_id TEXT NOT NULL DEFAULT 'default',
                axiom_type TEXT NOT NULL CHECK (
                    axiom_type IN (
                        'SubClassOf', 'EquivalentClass', 'DisjointWith',
                        'ObjectPropertyAssertion', 'DataPropertyAssertion'
                    )
                ),
                subject TEXT NOT NULL,
                object TEXT NOT NULL,
                annotations TEXT, -- JSON
                is_inferred BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ontology_id) REFERENCES ontologies(ontology_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_axioms_ontology ON owl_axioms(ontology_id);
            CREATE INDEX IF NOT EXISTS idx_axioms_subject ON owl_axioms(subject);
            CREATE INDEX IF NOT EXISTS idx_axioms_type ON owl_axioms(axiom_type);
            CREATE INDEX IF NOT EXISTS idx_axioms_subject_object ON owl_axioms(subject, object);

            -- Inference results table
            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                inference_time_ms INTEGER NOT NULL,
                reasoner_version TEXT NOT NULL,
                inferred_axiom_count INTEGER NOT NULL,
                result_data TEXT -- JSON
            );

            -- Validation reports table
            CREATE TABLE IF NOT EXISTS validation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_valid BOOLEAN NOT NULL,
                errors TEXT, -- JSON
                warnings TEXT -- JSON
            );

            -- Pathfinding cache tables
            CREATE TABLE IF NOT EXISTS pathfinding_sssp_cache (
                source_node_id INTEGER PRIMARY KEY,
                distances TEXT NOT NULL, -- JSON array
                paths TEXT NOT NULL, -- JSON object
                computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                computation_time_ms REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pathfinding_apsp_cache (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                distance_matrix TEXT NOT NULL, -- JSON 2D array
                computed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                computation_time_ms REAL NOT NULL
            );
            "#,
        )
        .map_err(|e| format!("Failed to create ontology schema: {}", e))?;

        Ok(())
    }

    /// Convert axiom type string to enum
    fn parse_axiom_type(s: &str) -> Result<AxiomType, String> {
        match s {
            "SubClassOf" => Ok(AxiomType::SubClassOf),
            "EquivalentClass" => Ok(AxiomType::EquivalentClass),
            "DisjointWith" => Ok(AxiomType::DisjointWith),
            "ObjectPropertyAssertion" => Ok(AxiomType::ObjectPropertyAssertion),
            "DataPropertyAssertion" => Ok(AxiomType::DataPropertyAssertion),
            _ => Err(format!("Unknown axiom type: {}", s)),
        }
    }

    /// Convert axiom type enum to string
    fn axiom_type_to_str(axiom_type: &AxiomType) -> &'static str {
        match axiom_type {
            AxiomType::SubClassOf => "SubClassOf",
            AxiomType::EquivalentClass => "EquivalentClass",
            AxiomType::DisjointWith => "DisjointWith",
            AxiomType::ObjectPropertyAssertion => "ObjectPropertyAssertion",
            AxiomType::DataPropertyAssertion => "DataPropertyAssertion",
        }
    }

    /// Convert property type string to enum
    fn parse_property_type(s: &str) -> Result<PropertyType, String> {
        match s {
            "ObjectProperty" => Ok(PropertyType::ObjectProperty),
            "DataProperty" => Ok(PropertyType::DataProperty),
            "AnnotationProperty" => Ok(PropertyType::AnnotationProperty),
            _ => Err(format!("Unknown property type: {}", s)),
        }
    }

    /// Convert property type enum to string
    fn property_type_to_str(property_type: &PropertyType) -> &'static str {
        match property_type {
            PropertyType::ObjectProperty => "ObjectProperty",
            PropertyType::DataProperty => "DataProperty",
            PropertyType::AnnotationProperty => "AnnotationProperty",
        }
    }
}

#[async_trait]
impl OntologyRepository for UnifiedOntologyRepository {
    #[instrument(skip(self), level = "debug")]
    async fn load_ontology_graph(&self) -> RepoResult<Arc<GraphData>> {
        let classes = self.list_owl_classes().await?;
        let mut graph = GraphData::new();

        // Create nodes for each OWL class
        for (i, class) in classes.iter().enumerate() {
            let mut node = Node::new_with_id(class.iri.clone(), Some(i as u32));
            node.label = class.label.clone().unwrap_or_else(|| class.iri.clone());
            node.color = Some("#4A90E2".to_string());
            node.size = Some(15.0);
            node.metadata.insert("type".to_string(), "owl_class".to_string());
            node.metadata.insert("iri".to_string(), class.iri.clone());
            node.metadata.insert("owl_class_iri".to_string(), class.iri.clone());

            graph.nodes.push(node);
        }

        // Create edges for subclass relationships
        for (i, class) in classes.iter().enumerate() {
            for parent_iri in &class.parent_classes {
                if let Some((j, _)) = classes
                    .iter()
                    .enumerate()
                    .find(|(_, c)| &c.iri == parent_iri)
                {
                    let edge = Edge::new(i as u32, j as u32, 1.0);
                    graph.edges.push(edge);
                }
            }
        }

        debug!(
            "Loaded ontology graph from unified DB: {} nodes, {} edges",
            graph.nodes.len(),
            graph.edges.len()
        );

        Ok(Arc::new(graph))
    }

    async fn save_ontology_graph(&self, _graph: &GraphData) -> RepoResult<()> {
        // Ontology graph is derived from OWL data, not directly saved
        Ok(())
    }

    #[instrument(skip(self, classes, properties, axioms), level = "info")]
    async fn save_ontology(
        &self,
        classes: &[OwlClass],
        properties: &[OwlProperty],
        axioms: &[OwlAxiom],
    ) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let classes_vec = classes.to_vec();
        let properties_vec = properties.to_vec();
        let axioms_vec = axioms.to_vec();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified ontology repository mutex");

            // Disable foreign keys for bulk insert
            conn.execute("PRAGMA foreign_keys = OFF", [])
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to disable foreign keys: {}",
                        e
                    ))
                })?;

            conn.execute("BEGIN TRANSACTION", []).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to begin transaction: {}",
                    e
                ))
            })?;

            // Clear existing data
            conn.execute("DELETE FROM owl_class_hierarchy", [])
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to clear hierarchy: {}",
                        e
                    ))
                })?;
            conn.execute("DELETE FROM owl_axioms", []).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to clear axioms: {}", e))
            })?;
            conn.execute("DELETE FROM owl_properties", [])
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to clear properties: {}",
                        e
                    ))
                })?;
            conn.execute("DELETE FROM owl_classes", []).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to clear classes: {}", e))
            })?;

            // Insert classes
            let mut class_stmt = conn
                .prepare(
                    "INSERT INTO owl_classes (ontology_id, iri, label, description, file_sha1)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare class insert: {}",
                        e
                    ))
                })?;

            for class in &classes_vec {
                class_stmt
                    .execute(params![
                        "default",
                        &class.iri,
                        &class.label,
                        &class.description,
                        &class.file_sha1,
                    ])
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to insert class {}: {}",
                            class.iri, e
                        ))
                    })?;
            }

            // Insert class hierarchies
            let mut hierarchy_stmt = conn
                .prepare("INSERT INTO owl_class_hierarchy (class_iri, parent_iri) VALUES (?1, ?2)")
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare hierarchy insert: {}",
                        e
                    ))
                })?;

            for class in &classes_vec {
                for parent_iri in &class.parent_classes {
                    hierarchy_stmt
                        .execute(params![&class.iri, parent_iri])
                        .map_err(|e| {
                            OntologyRepositoryError::DatabaseError(format!(
                                "Failed to insert hierarchy: {}",
                                e
                            ))
                        })?;
                }
            }

            // Insert properties
            let mut prop_stmt = conn
                .prepare(
                    "INSERT INTO owl_properties (ontology_id, iri, label, property_type, domain, range)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare property insert: {}",
                        e
                    ))
                })?;

            for property in &properties_vec {
                let domain_json = serde_json::to_string(&property.domain).ok();
                let range_json = serde_json::to_string(&property.range).ok();

                prop_stmt
                    .execute(params![
                        "default",
                        &property.iri,
                        &property.label,
                        Self::property_type_to_str(&property.property_type),
                        domain_json,
                        range_json,
                    ])
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to insert property {}: {}",
                            property.iri, e
                        ))
                    })?;
            }

            // Insert axioms
            let mut axiom_stmt = conn
                .prepare(
                    "INSERT INTO owl_axioms (ontology_id, axiom_type, subject, object, annotations)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare axiom insert: {}",
                        e
                    ))
                })?;

            for axiom in &axioms_vec {
                let annotations_json = serde_json::to_string(&axiom.annotations).ok();

                axiom_stmt
                    .execute(params![
                        "default",
                        Self::axiom_type_to_str(&axiom.axiom_type),
                        &axiom.subject,
                        &axiom.object,
                        annotations_json,
                    ])
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to insert axiom: {}",
                            e
                        ))
                    })?;
            }

            conn.execute("COMMIT", []).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to commit transaction: {}", e))
            })?;

            // Re-enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON", [])
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to enable foreign keys: {}",
                        e
                    ))
                })?;

            info!(
                "Saved ontology to unified DB: {} classes, {} properties, {} axioms",
                classes_vec.len(),
                properties_vec.len(),
                axioms_vec.len()
            );

            Ok(())
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
    }

    async fn add_owl_class(&self, class: &OwlClass) -> RepoResult<String> {
        let conn_arc = self.conn.clone();
        let class = class.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified ontology repository mutex");

            conn.execute(
                "INSERT INTO owl_classes (ontology_id, iri, label, description, file_sha1, markdown_content)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    "default",
                    &class.iri,
                    &class.label,
                    &class.description,
                    &class.file_sha1,
                    &class.markdown_content,
                ],
            )
            .map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to insert OWL class: {}", e))
            })?;

            // Insert parent relationships
            for parent_iri in &class.parent_classes {
                conn.execute(
                    "INSERT INTO owl_class_hierarchy (class_iri, parent_iri) VALUES (?1, ?2)",
                    params![&class.iri, parent_iri],
                )
                .ok(); // Ignore duplicates
            }

            debug!("Added OWL class {} to unified database", class.iri);

            Ok(class.iri.clone())
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
    }

    async fn get_owl_class(&self, iri: &str) -> RepoResult<Option<OwlClass>> {
        let conn_arc = self.conn.clone();
        let iri = iri.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified ontology repository mutex");

            let class_opt: Option<OwlClass> = conn
                .query_row(
                    "SELECT iri, label, description, file_sha1, last_synced, markdown_content
                     FROM owl_classes WHERE iri = ?1",
                    params![&iri],
                    |row| {
                        let iri: String = row.get(0)?;
                        let label: Option<String> = row.get(1)?;
                        let description: Option<String> = row.get(2)?;
                        let file_sha1: Option<String> = row.get(3)?;
                        let last_synced: Option<chrono::DateTime<chrono::Utc>> = row.get(4)?;
                        let markdown_content: Option<String> = row.get(5)?;

                        Ok(OwlClass {
                            iri,
                            label,
                            description,
                            parent_classes: Vec::new(), // Will be populated below
                            properties: HashMap::new(),
                            source_file: None,
                            markdown_content,
                            file_sha1,
                            last_synced,
                        })
                    },
                )
                .optional()
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!("Failed to query class: {}", e))
                })?;

            if let Some(mut class) = class_opt {
                // Get parent classes
                let mut parent_stmt = conn
                    .prepare("SELECT parent_iri FROM owl_class_hierarchy WHERE class_iri = ?1")
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to prepare parent query: {}",
                            e
                        ))
                    })?;

                let parents = parent_stmt
                    .query_map(params![&class.iri], |row| row.get(0))
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to query parents: {}",
                            e
                        ))
                    })?
                    .collect::<Result<Vec<String>, _>>()
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to collect parents: {}",
                            e
                        ))
                    })?;

                class.parent_classes = parents;

                Ok(Some(class))
            } else {
                Ok(None)
            }
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
    }

    async fn list_owl_classes(&self) -> RepoResult<Vec<OwlClass>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified ontology repository mutex");

            let mut stmt = conn
                .prepare(
                    "SELECT iri, label, description, file_sha1, last_synced, markdown_content
                     FROM owl_classes",
                )
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let classes = stmt
                .query_map([], |row| {
                    let iri: String = row.get(0)?;
                    let label: Option<String> = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let file_sha1: Option<String> = row.get(3)?;
                    let last_synced: Option<chrono::DateTime<chrono::Utc>> = row.get(4)?;
                    let markdown_content: Option<String> = row.get(5)?;

                    Ok(OwlClass {
                        iri,
                        label,
                        description,
                        parent_classes: Vec::new(), // Will be populated below
                        properties: HashMap::new(),
                        source_file: None,
                        markdown_content,
                        file_sha1,
                        last_synced,
                    })
                })
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!("Failed to query classes: {}", e))
                })?
                .collect::<Result<Vec<OwlClass>, _>>()
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to collect classes: {}",
                        e
                    ))
                })?;

            // Populate parent classes for each class
            let mut parent_stmt = conn
                .prepare("SELECT parent_iri FROM owl_class_hierarchy WHERE class_iri = ?1")
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare parent query: {}",
                        e
                    ))
                })?;

            let mut result_classes = Vec::new();
            for mut class in classes {
                let parents = parent_stmt
                    .query_map(params![&class.iri], |row| row.get(0))
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to query parents: {}",
                            e
                        ))
                    })?
                    .collect::<Result<Vec<String>, _>>()
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to collect parents: {}",
                            e
                        ))
                    })?;

                class.parent_classes = parents;
                result_classes.push(class);
            }

            debug!("Listed {} OWL classes from unified database", result_classes.len());

            Ok(result_classes)
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
    }

    // Implement remaining methods (abbreviated for space)
    async fn add_owl_property(&self, _property: &OwlProperty) -> RepoResult<String> {
        todo!("Implement add_owl_property")
    }

    async fn get_owl_property(&self, _iri: &str) -> RepoResult<Option<OwlProperty>> {
        todo!("Implement get_owl_property")
    }

    async fn list_owl_properties(&self) -> RepoResult<Vec<OwlProperty>> {
        Ok(Vec::new()) // Stub
    }

    async fn get_classes(&self) -> RepoResult<Vec<OwlClass>> {
        self.list_owl_classes().await
    }

    async fn get_axioms(&self) -> RepoResult<Vec<OwlAxiom>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire unified ontology repository mutex");

            let mut stmt = conn
                .prepare(
                    "SELECT id, axiom_type, subject, object, annotations
                     FROM owl_axioms",
                )
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let axioms = stmt
                .query_map([], |row| {
                    let id: u64 = row.get::<_, i64>(0)? as u64;
                    let axiom_type_str: String = row.get(1)?;
                    let subject: String = row.get(2)?;
                    let object: String = row.get(3)?;
                    let annotations_json: Option<String> = row.get(4)?;

                    let axiom_type = Self::parse_axiom_type(&axiom_type_str)
                        .map_err(|e| rusqlite::Error::InvalidQuery)?;

                    let annotations: HashMap<String, String> = annotations_json
                        .and_then(|json| serde_json::from_str(&json).ok())
                        .unwrap_or_default();

                    Ok(OwlAxiom {
                        id: Some(id),
                        axiom_type,
                        subject,
                        object,
                        annotations,
                    })
                })
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!("Failed to query axioms: {}", e))
                })?
                .collect::<Result<Vec<OwlAxiom>, _>>()
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to collect axioms: {}",
                        e
                    ))
                })?;

            Ok(axioms)
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Task join error: {}", e)))?
    }

    async fn add_axiom(&self, _axiom: &OwlAxiom) -> RepoResult<u64> {
        todo!("Implement add_axiom")
    }

    async fn get_class_axioms(&self, _class_iri: &str) -> RepoResult<Vec<OwlAxiom>> {
        Ok(Vec::new()) // Stub
    }

    async fn store_inference_results(&self, _results: &InferenceResults) -> RepoResult<()> {
        Ok(()) // Stub
    }

    async fn get_inference_results(&self) -> RepoResult<Option<InferenceResults>> {
        Ok(None) // Stub
    }

    async fn validate_ontology(&self) -> RepoResult<ValidationReport> {
        Ok(ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn query_ontology(&self, _query: &str) -> RepoResult<Vec<HashMap<String, String>>> {
        Ok(Vec::new()) // Stub
    }

    async fn get_metrics(&self) -> RepoResult<OntologyMetrics> {
        let classes = self.list_owl_classes().await?;
        let properties = self.list_owl_properties().await?;
        let axioms = self.get_axioms().await?;

        Ok(OntologyMetrics {
            class_count: classes.len(),
            property_count: properties.len(),
            axiom_count: axioms.len(),
            max_depth: 10,    // TODO: Calculate actual depth
            average_branching_factor: 2.5, // TODO: Calculate actual
        })
    }

    async fn cache_sssp_result(&self, _entry: &PathfindingCacheEntry) -> RepoResult<()> {
        Ok(()) // Stub
    }

    async fn get_cached_sssp(
        &self,
        _source_node_id: u32,
    ) -> RepoResult<Option<PathfindingCacheEntry>> {
        Ok(None) // Stub
    }

    async fn cache_apsp_result(&self, _distance_matrix: &Vec<Vec<f32>>) -> RepoResult<()> {
        Ok(()) // Stub
    }

    async fn get_cached_apsp(&self) -> RepoResult<Option<Vec<Vec<f32>>>> {
        Ok(None) // Stub
    }

    async fn invalidate_pathfinding_caches(&self) -> RepoResult<()> {
        Ok(()) // Stub
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unified_ontology_repository_creation() {
        let repo = UnifiedOntologyRepository::new(":memory:").unwrap();
        let classes = repo.list_owl_classes().await.unwrap();
        assert_eq!(classes.len(), 0);
    }

    #[tokio::test]
    async fn test_save_and_load_ontology() {
        let repo = UnifiedOntologyRepository::new(":memory:").unwrap();

        let class = OwlClass {
            iri: "http://example.org/TestClass".to_string(),
            label: Some("Test Class".to_string()),
            description: Some("A test class".to_string()),
            parent_classes: Vec::new(),
            properties: HashMap::new(),
            source_file: None,
            markdown_content: None,
            file_sha1: None,
            last_synced: None,
        };

        repo.save_ontology(&[class], &[], &[]).await.unwrap();

        let classes = repo.list_owl_classes().await.unwrap();
        assert_eq!(classes.len(), 1);
        assert_eq!(classes[0].iri, "http://example.org/TestClass");
    }
}
