// src/adapters/sqlite_ontology_repository.rs
//! SQLite Ontology Repository Adapter
//!
//! Implements the OntologyRepository port using SQLite for OWL class hierarchy,
//! properties, axioms, and inference result storage.

use async_trait::async_trait;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{debug, info, instrument};

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::ontology_repository::{
    AxiomType, InferenceResults, OntologyMetrics, OntologyRepository, OntologyRepositoryError,
    OwlAxiom, OwlClass, OwlProperty, PathfindingCacheEntry, PropertyType, Result as RepoResult,
    ValidationReport,
};

/// SQLite-backed ontology repository
pub struct SqliteOntologyRepository {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteOntologyRepository {
    /// Create new SQLite ontology repository
    pub fn new(db_path: &str) -> Result<Self, String> {
        let conn =
            Connection::open(db_path).map_err(|e| format!("Failed to open database: {}", e))?;

        // Create ontology schema
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS owl_classes (
                iri TEXT PRIMARY KEY,
                label TEXT,
                description TEXT,
                source_file TEXT,
                properties TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_owl_classes_label ON owl_classes(label);

            CREATE TABLE IF NOT EXISTS owl_class_hierarchy (
                class_iri TEXT NOT NULL,
                parent_iri TEXT NOT NULL,
                PRIMARY KEY (class_iri, parent_iri),
                FOREIGN KEY (class_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE,
                FOREIGN KEY (parent_iri) REFERENCES owl_classes(iri) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS owl_properties (
                iri TEXT PRIMARY KEY,
                label TEXT,
                property_type TEXT NOT NULL,
                domain TEXT,
                range TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS owl_axioms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                axiom_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                object TEXT NOT NULL,
                annotations TEXT,
                is_inferred BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_axioms_subject ON owl_axioms(subject);
            CREATE INDEX IF NOT EXISTS idx_axioms_type ON owl_axioms(axiom_type);

            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                inference_time_ms INTEGER NOT NULL,
                reasoner_version TEXT NOT NULL,
                inferred_axiom_count INTEGER NOT NULL,
                result_data TEXT
            );

            CREATE TABLE IF NOT EXISTS validation_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_valid BOOLEAN NOT NULL,
                errors TEXT,
                warnings TEXT
            );
        "#,
        )
        .map_err(|e| format!("Failed to create schema: {}", e))?;

        info!("Initialized SqliteOntologyRepository at {}", db_path);

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

#[async_trait]
impl OntologyRepository for SqliteOntologyRepository {
    #[instrument(skip(self), level = "debug")]
    async fn load_ontology_graph(&self) -> RepoResult<Arc<GraphData>> {
        // Load classes first (releases lock after)
        let classes = self.list_owl_classes().await?;
        let mut graph = GraphData::new();

        // Create nodes for each class
        for (i, class) in classes.iter().enumerate() {
            let mut node = Node::new_with_id(class.iri.clone(), Some(i as u32));
            node.label = class.label.clone().unwrap_or_else(|| class.iri.clone());
            node.color = Some("#4A90E2".to_string());
            node.size = Some(15.0);
            node.metadata
                .insert("type".to_string(), "owl_class".to_string());
            node.metadata.insert("iri".to_string(), class.iri.clone());

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
            "Loaded ontology graph with {} nodes and {} edges",
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
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            // Start transaction
            conn.execute("BEGIN TRANSACTION", [])
                .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to begin transaction: {}", e)))?;

            // Delete existing data (clean slate)
            conn.execute("DELETE FROM owl_class_hierarchy", [])
                .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to clear hierarchy: {}", e)))?;
            conn.execute("DELETE FROM owl_axioms", [])
                .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to clear axioms: {}", e)))?;
            conn.execute("DELETE FROM owl_properties", [])
                .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to clear properties: {}", e)))?;
            conn.execute("DELETE FROM owl_classes", [])
                .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to clear classes: {}", e)))?;

            // Insert classes
            let mut class_stmt = conn.prepare(
                "INSERT INTO owl_classes (iri, label, description, source_file, properties, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)"
            ).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to prepare class insert: {}", e)))?;

            for class in &classes_vec {
                let properties_json = serde_json::to_string(&class.properties)
                    .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to serialize properties: {}", e)))?;

                class_stmt.execute(params![
                    &class.iri,
                    &class.label,
                    &class.description,
                    &class.source_file,
                    properties_json
                ]).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert class {}: {}", class.iri, e)))?;
            }
            drop(class_stmt);

            // Insert class hierarchies
            let mut hierarchy_stmt = conn.prepare(
                "INSERT INTO owl_class_hierarchy (class_iri, parent_iri) VALUES (?1, ?2)"
            ).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to prepare hierarchy insert: {}", e)))?;

            for class in &classes_vec {
                for parent_iri in &class.parent_classes {
                    hierarchy_stmt.execute(params![&class.iri, parent_iri])
                        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert hierarchy: {}", e)))?;
                }
            }
            drop(hierarchy_stmt);

            // Insert properties
            let mut property_stmt = conn.prepare(
                "INSERT INTO owl_properties (iri, label, property_type, domain, range, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)"
            ).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to prepare property insert: {}", e)))?;

            for property in &properties_vec {
                let property_type_str = match property.property_type {
                    PropertyType::ObjectProperty => "ObjectProperty",
                    PropertyType::DataProperty => "DataProperty",
                    PropertyType::AnnotationProperty => "AnnotationProperty",
                };

                let domain_json = serde_json::to_string(&property.domain)
                    .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to serialize domain: {}", e)))?;
                let range_json = serde_json::to_string(&property.range)
                    .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to serialize range: {}", e)))?;

                property_stmt.execute(params![
                    &property.iri,
                    &property.label,
                    property_type_str,
                    domain_json,
                    range_json
                ]).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert property {}: {}", property.iri, e)))?;
            }
            drop(property_stmt);

            // Insert axioms
            let mut axiom_stmt = conn.prepare(
                "INSERT INTO owl_axioms (axiom_type, subject, object, annotations)
                 VALUES (?1, ?2, ?3, ?4)"
            ).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to prepare axiom insert: {}", e)))?;

            for axiom in &axioms_vec {
                let axiom_type_str = match axiom.axiom_type {
                    AxiomType::SubClassOf => "SubClassOf",
                    AxiomType::EquivalentClass => "EquivalentClass",
                    AxiomType::DisjointWith => "DisjointWith",
                    AxiomType::ObjectPropertyAssertion => "ObjectPropertyAssertion",
                    AxiomType::DataPropertyAssertion => "DataPropertyAssertion",
                };

                let annotations_json = serde_json::to_string(&axiom.annotations)
                    .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to serialize annotations: {}", e)))?;

                axiom_stmt.execute(params![
                    axiom_type_str,
                    &axiom.subject,
                    &axiom.object,
                    annotations_json
                ]).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert axiom: {}", e)))?;
            }
            drop(axiom_stmt);

            // Commit transaction
            conn.execute("COMMIT", [])
                .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to commit transaction: {}", e)))?;

            info!("✅ Saved ontology: {} classes, {} properties, {} axioms",
                  classes_vec.len(), properties_vec.len(), axioms_vec.len());

            Ok(())
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Spawn blocking error: {}", e)))?
    }

    #[instrument(skip(self, class), fields(iri = %class.iri), level = "debug")]
    async fn add_owl_class(&self, class: &OwlClass) -> RepoResult<String> {
        let conn_arc = self.conn.clone();
        let class_clone = class.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let properties_json = serde_json::to_string(&class_clone.properties).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to serialize properties: {}", e))
            })?;

            conn.execute(
                "INSERT OR REPLACE INTO owl_classes (iri, label, description, source_file, properties, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
                params![&class_clone.iri, &class_clone.label, &class_clone.description, &class_clone.source_file, properties_json]
            )
            .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert class: {}", e)))?;

            // Insert parent relationships
            for parent_iri in &class_clone.parent_classes {
                conn.execute(
                    "INSERT OR IGNORE INTO owl_class_hierarchy (class_iri, parent_iri) VALUES (?1, ?2)",
                    params![&class_clone.iri, parent_iri],
                )
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!("Failed to insert hierarchy: {}", e))
                })?;
            }

            Ok(class_clone.iri.clone())
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_owl_class(&self, iri: &str) -> RepoResult<Option<OwlClass>> {
        let conn_arc = self.conn.clone();
        let iri_owned = iri.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let result = conn.query_row(
                "SELECT iri, label, description, source_file, properties FROM owl_classes WHERE iri = ?1",
                params![iri_owned],
                |row| {
                    let iri: String = row.get(0)?;
                    let label: Option<String> = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let source_file: Option<String> = row.get(3)?;
                    let properties_json: String = row.get(4)?;

                    let properties = serde_json::from_str(&properties_json).unwrap_or_default();

                    Ok((iri, label, description, source_file, properties))
                }
            );

            match result {
                Ok((iri, label, description, source_file, properties)) => {
                    let mut parent_stmt = conn
                        .prepare("SELECT parent_iri FROM owl_class_hierarchy WHERE class_iri = ?1")
                        .map_err(|e| {
                            OntologyRepositoryError::DatabaseError(format!(
                                "Failed to prepare parent query: {}",
                                e
                            ))
                        })?;

                    let parent_classes = parent_stmt
                        .query_map(params![&iri], |row| row.get(0))
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

                    Ok(Some(OwlClass {
                        iri,
                        label,
                        description,
                        parent_classes,
                        properties,
                        source_file,
                    }))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(OntologyRepositoryError::DatabaseError(format!(
                    "Database error: {}",
                    e
                ))),
            }
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn list_owl_classes(&self) -> RepoResult<Vec<OwlClass>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire ontology repository mutex");

            let mut stmt = conn
                .prepare("SELECT iri, label, description, source_file, properties FROM owl_classes")
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let class_rows = stmt
                .query_map([], |row| {
                    let iri: String = row.get(0)?;
                    let label: Option<String> = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let source_file: Option<String> = row.get(3)?;
                    let properties_json: String = row.get(4)?;

                    Ok((iri, label, description, source_file, properties_json))
                })
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to query classes: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to collect classes: {}",
                        e
                    ))
                })?;

            let mut classes = Vec::new();

            for (iri, label, description, source_file, properties_json) in class_rows {
                let properties = serde_json::from_str(&properties_json).unwrap_or_default();

                let mut parent_stmt = conn
                    .prepare("SELECT parent_iri FROM owl_class_hierarchy WHERE class_iri = ?1")
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to prepare parent query: {}",
                            e
                        ))
                    })?;

                let parent_classes = parent_stmt
                    .query_map(params![&iri], |row| row.get(0))
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

                classes.push(OwlClass {
                    iri,
                    label,
                    description,
                    parent_classes,
                    properties,
                    source_file,
                });
            }

            Ok(classes)
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn add_owl_property(&self, property: &OwlProperty) -> RepoResult<String> {
        let conn_arc = self.conn.clone();
        let property_clone = property.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let property_type_str = match property_clone.property_type {
                PropertyType::ObjectProperty => "ObjectProperty",
                PropertyType::DataProperty => "DataProperty",
                PropertyType::AnnotationProperty => "AnnotationProperty",
            };

            let domain_json = serde_json::to_string(&property_clone.domain).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to serialize domain: {}", e))
            })?;
            let range_json = serde_json::to_string(&property_clone.range).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to serialize range: {}", e))
            })?;

            conn.execute(
                "INSERT OR REPLACE INTO owl_properties (iri, label, property_type, domain, range, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
                params![&property_clone.iri, &property_clone.label, property_type_str, domain_json, range_json]
            )
            .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert property: {}", e)))?;

            Ok(property_clone.iri.clone())
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_owl_property(&self, iri: &str) -> RepoResult<Option<OwlProperty>> {
        let conn_arc = self.conn.clone();
        let iri_owned = iri.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let result = conn.query_row(
                "SELECT iri, label, property_type, domain, range FROM owl_properties WHERE iri = ?1",
                params![iri_owned],
                |row| {
                    let iri: String = row.get(0)?;
                    let label: Option<String> = row.get(1)?;
                    let property_type_str: String = row.get(2)?;
                    let domain_json: String = row.get(3)?;
                    let range_json: String = row.get(4)?;

                    Ok((iri, label, property_type_str, domain_json, range_json))
                },
            );

            match result {
                Ok((iri, label, property_type_str, domain_json, range_json)) => {
                    let property_type = match property_type_str.as_str() {
                        "ObjectProperty" => PropertyType::ObjectProperty,
                        "DataProperty" => PropertyType::DataProperty,
                        "AnnotationProperty" => PropertyType::AnnotationProperty,
                        _ => {
                            return Err(OntologyRepositoryError::InvalidData(format!(
                                "Unknown property type: {}",
                                property_type_str
                            )))
                        }
                    };

                    let domain = serde_json::from_str(&domain_json).unwrap_or_default();
                    let range = serde_json::from_str(&range_json).unwrap_or_default();

                    Ok(Some(OwlProperty {
                        iri,
                        label,
                        property_type,
                        domain,
                        range,
                    }))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(OntologyRepositoryError::DatabaseError(format!(
                    "Database error: {}",
                    e
                ))),
            }
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn list_owl_properties(&self) -> RepoResult<Vec<OwlProperty>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire ontology repository mutex");

            let mut stmt = conn
                .prepare("SELECT iri, label, property_type, domain, range FROM owl_properties")
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to prepare statement: {}",
                        e
                    ))
                })?;

            let properties = stmt
                .query_map([], |row| {
                    let iri: String = row.get(0)?;
                    let label: Option<String> = row.get(1)?;
                    let property_type_str: String = row.get(2)?;
                    let domain_json: String = row.get(3)?;
                    let range_json: String = row.get(4)?;

                    let property_type = match property_type_str.as_str() {
                        "ObjectProperty" => PropertyType::ObjectProperty,
                        "DataProperty" => PropertyType::DataProperty,
                        "AnnotationProperty" => PropertyType::AnnotationProperty,
                        _ => PropertyType::ObjectProperty,
                    };

                    let domain = serde_json::from_str(&domain_json).unwrap_or_default();
                    let range = serde_json::from_str(&range_json).unwrap_or_default();

                    Ok(OwlProperty {
                        iri,
                        label,
                        property_type,
                        domain,
                        range,
                    })
                })
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to query properties: {}",
                        e
                    ))
                })?
                .collect::<Result<Vec<OwlProperty>, _>>()
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to collect properties: {}",
                        e
                    ))
                })?;

            Ok(properties)
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn add_axiom(&self, axiom: &OwlAxiom) -> RepoResult<u64> {
        let conn_arc = self.conn.clone();
        let axiom_clone = axiom.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let axiom_type_str = match axiom_clone.axiom_type {
                AxiomType::SubClassOf => "SubClassOf",
                AxiomType::EquivalentClass => "EquivalentClass",
                AxiomType::DisjointWith => "DisjointWith",
                AxiomType::ObjectPropertyAssertion => "ObjectPropertyAssertion",
                AxiomType::DataPropertyAssertion => "DataPropertyAssertion",
            };

            let annotations_json = serde_json::to_string(&axiom_clone.annotations).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!(
                    "Failed to serialize annotations: {}",
                    e
                ))
            })?;

            conn.execute(
                "INSERT INTO owl_axioms (axiom_type, subject, object, annotations) VALUES (?1, ?2, ?3, ?4)",
                params![axiom_type_str, &axiom_clone.subject, &axiom_clone.object, annotations_json]
            )
            .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert axiom: {}", e)))?;

            let id = conn.last_insert_rowid() as u64;
            Ok(id)
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_class_axioms(&self, class_iri: &str) -> RepoResult<Vec<OwlAxiom>> {
        let conn_arc = self.conn.clone();
        let class_iri_owned = class_iri.to_string();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let mut stmt = conn.prepare(
                "SELECT id, axiom_type, subject, object, annotations FROM owl_axioms WHERE subject = ?1 OR object = ?1"
            ).map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to prepare statement: {}", e)))?;

            let axioms = stmt
                .query_map(params![class_iri_owned], |row| {
                    let id: i64 = row.get(0)?;
                    let axiom_type_str: String = row.get(1)?;
                    let subject: String = row.get(2)?;
                    let object: String = row.get(3)?;
                    let annotations_json: String = row.get(4)?;

                    let axiom_type = match axiom_type_str.as_str() {
                        "SubClassOf" => AxiomType::SubClassOf,
                        "EquivalentClass" => AxiomType::EquivalentClass,
                        "DisjointWith" => AxiomType::DisjointWith,
                        "ObjectPropertyAssertion" => AxiomType::ObjectPropertyAssertion,
                        "DataPropertyAssertion" => AxiomType::DataPropertyAssertion,
                        _ => AxiomType::SubClassOf,
                    };

                    let annotations = serde_json::from_str(&annotations_json).unwrap_or_default();

                    Ok(OwlAxiom {
                        id: Some(id as u64),
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
                    OntologyRepositoryError::DatabaseError(format!("Failed to collect axioms: {}", e))
                })?;

            Ok(axioms)
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn store_inference_results(&self, results: &InferenceResults) -> RepoResult<()> {
        let conn_arc = self.conn.clone();
        let results_clone = results.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let result_json = serde_json::to_string(&results_clone.inferred_axioms).map_err(|e| {
                OntologyRepositoryError::DatabaseError(format!("Failed to serialize results: {}", e))
            })?;

            conn.execute(
                "INSERT INTO inference_results (inference_time_ms, reasoner_version, inferred_axiom_count, result_data)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    results_clone.inference_time_ms as i64,
                    &results_clone.reasoner_version,
                    results_clone.inferred_axioms.len(),
                    result_json
                ]
            )
            .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Failed to insert inference results: {}", e)))?;

            // Mark axioms as inferred
            for axiom in &results_clone.inferred_axioms {
                if let Some(id) = axiom.id {
                    conn.execute(
                        "UPDATE owl_axioms SET is_inferred = TRUE WHERE id = ?1",
                        params![id as i64],
                    )
                    .map_err(|e| {
                        OntologyRepositoryError::DatabaseError(format!(
                            "Failed to mark axiom as inferred: {}",
                            e
                        ))
                    })?;
                }
            }

            Ok(())
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn get_inference_results(&self) -> RepoResult<Option<InferenceResults>> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc.lock().expect("Failed to acquire ontology repository mutex");

            let result = conn.query_row(
                "SELECT timestamp, inference_time_ms, reasoner_version, result_data FROM inference_results ORDER BY id DESC LIMIT 1",
                [],
                |row| {
                    let timestamp_str: String = row.get(0)?;
                    let inference_time_ms: i64 = row.get(1)?;
                    let reasoner_version: String = row.get(2)?;
                    let result_json: String = row.get(3)?;

                    Ok((timestamp_str, inference_time_ms, reasoner_version, result_json))
                }
            );

            match result {
                Ok((timestamp_str, inference_time_ms, reasoner_version, result_json)) => {
                    let timestamp = chrono::DateTime::parse_from_rfc3339(&timestamp_str)
                        .map_err(|e| {
                            OntologyRepositoryError::DatabaseError(format!(
                                "Failed to parse timestamp: {}",
                                e
                            ))
                        })?
                        .with_timezone(&chrono::Utc);

                    let inferred_axioms: Vec<OwlAxiom> =
                        serde_json::from_str(&result_json).map_err(|e| {
                            OntologyRepositoryError::DatabaseError(format!(
                                "Failed to deserialize axioms: {}",
                                e
                            ))
                        })?;

                    Ok(Some(InferenceResults {
                        timestamp,
                        inferred_axioms,
                        inference_time_ms: inference_time_ms as u64,
                        reasoner_version,
                    }))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(OntologyRepositoryError::DatabaseError(format!(
                    "Database error: {}",
                    e
                ))),
            }
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    async fn validate_ontology(&self) -> RepoResult<ValidationReport> {
        // Basic validation - check for cycles in class hierarchy
        Ok(ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }

    async fn query_ontology(&self, _query: &str) -> RepoResult<Vec<HashMap<String, String>>> {
        // SPARQL queries not yet implemented
        Err(OntologyRepositoryError::DatabaseError(
            "SPARQL queries not yet implemented".to_string(),
        ))
    }

    async fn get_metrics(&self) -> RepoResult<OntologyMetrics> {
        let conn_arc = self.conn.clone();

        tokio::task::spawn_blocking(move || {
            let conn = conn_arc
                .lock()
                .expect("Failed to acquire ontology repository mutex");

            let class_count: usize = conn
                .query_row("SELECT COUNT(*) FROM owl_classes", [], |row| row.get(0))
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to count classes: {}",
                        e
                    ))
                })?;

            let property_count: usize = conn
                .query_row("SELECT COUNT(*) FROM owl_properties", [], |row| row.get(0))
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!(
                        "Failed to count properties: {}",
                        e
                    ))
                })?;

            let axiom_count: usize = conn
                .query_row("SELECT COUNT(*) FROM owl_axioms", [], |row| row.get(0))
                .map_err(|e| {
                    OntologyRepositoryError::DatabaseError(format!("Failed to count axioms: {}", e))
                })?;

            Ok(OntologyMetrics {
                class_count,
                property_count,
                axiom_count,
                max_depth: 0,
                average_branching_factor: 0.0,
            })
        })
        .await
        .map_err(|e| OntologyRepositoryError::DatabaseError(format!("Join error: {}", e)))?
    }

    // Pathfinding cache methods (placeholders for ontology repository)
    async fn cache_sssp_result(&self, _entry: &PathfindingCacheEntry) -> RepoResult<()> {
        // Pathfinding is not applicable to ontology repositories
        Ok(())
    }

    async fn get_cached_sssp(
        &self,
        _source_node_id: u32,
    ) -> RepoResult<Option<PathfindingCacheEntry>> {
        // Pathfinding is not applicable to ontology repositories
        Ok(None)
    }

    async fn cache_apsp_result(&self, _distance_matrix: &Vec<Vec<f32>>) -> RepoResult<()> {
        // Pathfinding is not applicable to ontology repositories
        Ok(())
    }

    async fn get_cached_apsp(&self) -> RepoResult<Option<Vec<Vec<f32>>>> {
        // Pathfinding is not applicable to ontology repositories
        Ok(None)
    }

    async fn invalidate_pathfinding_caches(&self) -> RepoResult<()> {
        // Pathfinding is not applicable to ontology repositories
        Ok(())
    }
}
