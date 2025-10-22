// src/application/ontology/queries.rs
//! Ontology Domain - Read Operations (Queries)
//!
//! All queries for reading ontology state following CQRS patterns.

use hexser::{HexResult, Hexserror, QueryHandler};
use std::collections::HashMap;
use std::sync::Arc;

use crate::models::graph::GraphData;
use crate::ports::ontology_repository::{
    InferenceResults, OntologyMetrics, OntologyRepository, OwlAxiom, OwlClass, OwlProperty,
    ValidationReport,
};

// ============================================================================
// LOAD ONTOLOGY GRAPH
// ============================================================================

#[derive(Debug, Clone)]
pub struct LoadOntologyGraph;

pub struct LoadOntologyGraphHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl LoadOntologyGraphHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<LoadOntologyGraph, Arc<GraphData>> for LoadOntologyGraphHandler {
    fn handle(&self, _query: LoadOntologyGraph) -> HexResult<Arc<GraphData>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing LoadOntologyGraph query");

            self.repository
                .load_ontology_graph()
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// GET OWL CLASS
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetOwlClass {
    pub iri: String,
}

pub struct GetOwlClassHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl GetOwlClassHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetOwlClass, Option<OwlClass>> for GetOwlClassHandler {
    fn handle(&self, query: GetOwlClass) -> HexResult<Option<OwlClass>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing GetOwlClass query: iri={}", query.iri);

            self.repository
                .get_owl_class(&query.iri)
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// LIST OWL CLASSES
// ============================================================================

#[derive(Debug, Clone)]
pub struct ListOwlClasses;

pub struct ListOwlClassesHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl ListOwlClassesHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<ListOwlClasses, Vec<OwlClass>> for ListOwlClassesHandler {
    fn handle(&self, _query: ListOwlClasses) -> HexResult<Vec<OwlClass>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing ListOwlClasses query");

            self.repository
                .list_owl_classes()
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// GET OWL PROPERTY
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetOwlProperty {
    pub iri: String,
}

pub struct GetOwlPropertyHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl GetOwlPropertyHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetOwlProperty, Option<OwlProperty>> for GetOwlPropertyHandler {
    fn handle(&self, query: GetOwlProperty) -> HexResult<Option<OwlProperty>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing GetOwlProperty query: iri={}", query.iri);

            self.repository
                .get_owl_property(&query.iri)
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// LIST OWL PROPERTIES
// ============================================================================

#[derive(Debug, Clone)]
pub struct ListOwlProperties;

pub struct ListOwlPropertiesHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl ListOwlPropertiesHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<ListOwlProperties, Vec<OwlProperty>> for ListOwlPropertiesHandler {
    fn handle(&self, _query: ListOwlProperties) -> HexResult<Vec<OwlProperty>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing ListOwlProperties query");

            self.repository
                .list_owl_properties()
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// GET CLASS AXIOMS
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetClassAxioms {
    pub class_iri: String,
}

pub struct GetClassAxiomsHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl GetClassAxiomsHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetClassAxioms, Vec<OwlAxiom>> for GetClassAxiomsHandler {
    fn handle(&self, query: GetClassAxioms) -> HexResult<Vec<OwlAxiom>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!(
                "Executing GetClassAxioms query: class_iri={}",
                query.class_iri
            );

            self.repository
                .get_class_axioms(&query.class_iri)
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// GET INFERENCE RESULTS
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetInferenceResults;

pub struct GetInferenceResultsHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl GetInferenceResultsHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetInferenceResults, Option<InferenceResults>> for GetInferenceResultsHandler {
    fn handle(&self, _query: GetInferenceResults) -> HexResult<Option<InferenceResults>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing GetInferenceResults query");

            self.repository
                .get_inference_results()
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// VALIDATE ONTOLOGY
// ============================================================================

#[derive(Debug, Clone)]
pub struct ValidateOntology;

pub struct ValidateOntologyHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl ValidateOntologyHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<ValidateOntology, ValidationReport> for ValidateOntologyHandler {
    fn handle(&self, _query: ValidateOntology) -> HexResult<ValidationReport> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing ValidateOntology query");

            self.repository
                .validate_ontology()
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// QUERY ONTOLOGY
// ============================================================================

#[derive(Debug, Clone)]
pub struct QueryOntology {
    pub query: String,
}

pub struct QueryOntologyHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl QueryOntologyHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<QueryOntology, Vec<HashMap<String, String>>> for QueryOntologyHandler {
    fn handle(&self, query: QueryOntology) -> HexResult<Vec<HashMap<String, String>>> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing QueryOntology query: query={}", query.query);

            self.repository
                .query_ontology(&query.query)
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}

// ============================================================================
// GET ONTOLOGY METRICS
// ============================================================================

#[derive(Debug, Clone)]
pub struct GetOntologyMetrics;

pub struct GetOntologyMetricsHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl GetOntologyMetricsHandler {
    pub fn new(repository: Arc<dyn OntologyRepository>) -> Self {
        Self { repository }
    }
}

impl QueryHandler<GetOntologyMetrics, OntologyMetrics> for GetOntologyMetricsHandler {
    fn handle(&self, _query: GetOntologyMetrics) -> HexResult<OntologyMetrics> {
        tokio::runtime::Handle::current().block_on(async {
            log::debug!("Executing GetOntologyMetrics query");

            self.repository
                .get_metrics()
                .await
                .map_err(|e| Hexserror::port("E_REPO_001", &format!("{}", e)))
        })
    }
}
