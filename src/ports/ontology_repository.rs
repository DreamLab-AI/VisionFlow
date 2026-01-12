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

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),
}

/// OWL Class with rich metadata support (Schema V2)
/// Supports comprehensive ontology metadata including:
/// - Core identification (term_id, preferred_term)
/// - Classification (source_domain, version, type)
/// - Quality metrics (quality_score, authority_score, status, maturity)
/// - OWL2 properties (owl_physicality, owl_role)
/// - Domain relationships (belongs_to_domain, bridges_to_domain)
/// - Source tracking (file_sha1, markdown_content, last_synced)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlClass {
    // Core identification
    pub iri: String,
    pub term_id: Option<String>,
    pub preferred_term: Option<String>,

    // Basic metadata
    pub label: Option<String>,
    pub description: Option<String>,
    pub parent_classes: Vec<String>,

    // Classification metadata
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

    // Additional metadata (JSON for extensibility)
    pub properties: HashMap<String, String>,
    pub additional_metadata: Option<String>,
}

impl Default for OwlClass {
    fn default() -> Self {
        Self {
            iri: String::new(),
            term_id: None,
            preferred_term: None,
            label: None,
            description: None,
            parent_classes: Vec::new(),
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
            source_file: None,
            file_sha1: None,
            markdown_content: None,
            last_synced: None,
            properties: HashMap::new(),
            additional_metadata: None,
        }
    }
}

/// Semantic relationship between OWL classes
/// Supports relationship types: has-part, uses, enables, requires, subclass-of, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlRelationship {
    pub source_class_iri: String,
    pub relationship_type: String,
    pub target_class_iri: String,
    pub confidence: f32,
    pub is_inferred: bool,
}

/// Cross-reference (e.g., WikiLink) from an OWL class to external resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlCrossReference {
    pub source_class_iri: String,
    pub target_reference: String,
    pub reference_type: String, // wiki, external, doi, etc.
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PropertyType {
    ObjectProperty,
    DataProperty,
    AnnotationProperty,
}

/// OWL Property with quality metrics (Schema V2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlProperty {
    pub iri: String,
    pub label: Option<String>,
    pub property_type: PropertyType,
    pub domain: Vec<String>,
    pub range: Vec<String>,

    // Quality metrics (Schema V2)
    pub quality_score: Option<f32>,
    pub authority_score: Option<f32>,

    // Source tracking
    pub source_file: Option<String>,
}

impl Default for OwlProperty {
    fn default() -> Self {
        Self {
            iri: String::new(),
            label: None,
            property_type: PropertyType::ObjectProperty,
            domain: Vec::new(),
            range: Vec::new(),
            quality_score: None,
            authority_score: None,
            source_file: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AxiomType {
    SubClassOf,
    EquivalentClass,
    DisjointWith,
    ObjectPropertyAssertion,
    DataPropertyAssertion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwlAxiom {
    pub id: Option<u64>,
    pub axiom_type: AxiomType,
    pub subject: String,
    pub object: String,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResults {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub inferred_axioms: Vec<OwlAxiom>,
    pub inference_time_ms: u64,
    pub reasoner_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyMetrics {
    pub class_count: usize,
    pub property_count: usize,
    pub axiom_count: usize,
    pub max_depth: usize,
    pub average_branching_factor: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathfindingCacheEntry {
    pub source_node_id: u32,
    pub target_node_id: Option<u32>,
    pub distances: Vec<f32>,
    pub paths: HashMap<u32, Vec<u32>>,
    pub computed_at: chrono::DateTime<chrono::Utc>,
    pub computation_time_ms: f32,
}

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


    /// Default: No-op (not all implementations support inference)
    async fn store_inference_results(&self, _results: &InferenceResults) -> Result<()> {
        Ok(())
    }


    /// Default: None (not all implementations support inference)
    async fn get_inference_results(&self) -> Result<Option<InferenceResults>> {
        Ok(None)
    }


    /// Default: Valid report (override for actual validation)
    async fn validate_ontology(&self) -> Result<ValidationReport> {
        Ok(ValidationReport {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            timestamp: chrono::Utc::now(),
        })
    }


    /// Default: Empty results (override when query support added)
    async fn query_ontology(&self, _query: &str) -> Result<Vec<HashMap<String, String>>> {
        Ok(Vec::new())
    }


    async fn get_metrics(&self) -> Result<OntologyMetrics>;




    /// Default: No-op (not all implementations support caching)
    async fn cache_sssp_result(&self, _entry: &PathfindingCacheEntry) -> Result<()> {
        Ok(())
    }


    /// Default: None (not all implementations support caching)
    async fn get_cached_sssp(&self, _source_node_id: u32) -> Result<Option<PathfindingCacheEntry>> {
        Ok(None)
    }


    /// Default: No-op (not all implementations support caching)
    async fn cache_apsp_result(&self, _distance_matrix: &Vec<Vec<f32>>) -> Result<()> {
        Ok(())
    }


    /// Default: None (not all implementations support caching)
    async fn get_cached_apsp(&self) -> Result<Option<Vec<Vec<f32>>>> {
        Ok(None)
    }


    /// Default: No-op (not all implementations support caching)
    async fn invalidate_pathfinding_caches(&self) -> Result<()> {
        Ok(())
    }
}
