// src/ports/ontology_repository.rs
//! Ontology Repository Port
//!
//! Manages the ontology graph structure parsed from GitHub markdown files,
//! including OWL classes, properties, axioms, and inference results.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::graph::GraphData;

pub type Result<T> = std::result::Result<T, OntologyRepositoryError>;

#[derive(Debug, thiserror::Error)]
pub enum OntologyRepositoryError {
    #[error("Ontology not found")]
    NotFound,

    #[error("OWL class not found: {0}")]
    ClassNotFound(String),

    #[error("OWL property not found: {0}")]
    PropertyNotFound(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Invalid OWL data: {0}")]
    InvalidData(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlClass {
    pub iri: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub parent_classes: Vec<String>,
    pub properties: HashMap<String, String>,
    pub source_file: Option<String>,
    
    pub markdown_content: Option<String>,
    
    pub file_sha1: Option<String>,
    
    pub last_synced: Option<chrono::DateTime<chrono::Utc>>,
}

/
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PropertyType {
    ObjectProperty,
    DataProperty,
    AnnotationProperty,
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlProperty {
    pub iri: String,
    pub label: Option<String>,
    pub property_type: PropertyType,
    pub domain: Vec<String>,
    pub range: Vec<String>,
}

/
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AxiomType {
    SubClassOf,
    EquivalentClass,
    DisjointWith,
    ObjectPropertyAssertion,
    DataPropertyAssertion,
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlAxiom {
    pub id: Option<u64>,
    pub axiom_type: AxiomType,
    pub subject: String,
    pub object: String,
    pub annotations: HashMap<String, String>,
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResults {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub inferred_axioms: Vec<OwlAxiom>,
    pub inference_time_ms: u64,
    pub reasoner_version: String,
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyMetrics {
    pub class_count: usize,
    pub property_count: usize,
    pub axiom_count: usize,
    pub max_depth: usize,
    pub average_branching_factor: f32,
}

/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathfindingCacheEntry {
    pub source_node_id: u32,
    pub target_node_id: Option<u32>,
    pub distances: Vec<f32>,
    pub paths: HashMap<u32, Vec<u32>>,
    pub computed_at: chrono::DateTime<chrono::Utc>,
    pub computation_time_ms: f32,
}

/
#[async_trait]
pub trait OntologyRepository: Send + Sync {
    
    async fn load_ontology_graph(&self) -> Result<Arc<GraphData>>;

    
    async fn save_ontology_graph(&self, graph: &GraphData) -> Result<()>;

    
    
    
    async fn save_ontology(
        &self,
        classes: &[OwlClass],
        properties: &[OwlProperty],
        axioms: &[OwlAxiom],
    ) -> Result<()>;

    
    
    async fn add_owl_class(&self, class: &OwlClass) -> Result<String>;

    
    async fn get_owl_class(&self, iri: &str) -> Result<Option<OwlClass>>;

    
    async fn list_owl_classes(&self) -> Result<Vec<OwlClass>>;

    
    
    async fn add_owl_property(&self, property: &OwlProperty) -> Result<String>;

    
    async fn get_owl_property(&self, iri: &str) -> Result<Option<OwlProperty>>;

    
    async fn list_owl_properties(&self) -> Result<Vec<OwlProperty>>;

    
    async fn get_classes(&self) -> Result<Vec<OwlClass>>;

    
    async fn get_axioms(&self) -> Result<Vec<OwlAxiom>>;

    
    
    async fn add_axiom(&self, axiom: &OwlAxiom) -> Result<u64>;

    
    async fn get_class_axioms(&self, class_iri: &str) -> Result<Vec<OwlAxiom>>;

    
    async fn store_inference_results(&self, results: &InferenceResults) -> Result<()>;

    
    async fn get_inference_results(&self) -> Result<Option<InferenceResults>>;

    
    async fn validate_ontology(&self) -> Result<ValidationReport>;

    
    async fn query_ontology(&self, query: &str) -> Result<Vec<HashMap<String, String>>>;

    
    async fn get_metrics(&self) -> Result<OntologyMetrics>;

    

    
    async fn cache_sssp_result(&self, entry: &PathfindingCacheEntry) -> Result<()>;

    
    async fn get_cached_sssp(&self, source_node_id: u32) -> Result<Option<PathfindingCacheEntry>>;

    
    async fn cache_apsp_result(&self, distance_matrix: &Vec<Vec<f32>>) -> Result<()>;

    
    async fn get_cached_apsp(&self) -> Result<Option<Vec<Vec<f32>>>>;

    
    async fn invalidate_pathfinding_caches(&self) -> Result<()>;
}
