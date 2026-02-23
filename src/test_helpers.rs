//! Shared test helpers for Neo4j-dependent tests
//!
//! Provides:
//! - `MockOntologyRepository`: In-memory implementation of `OntologyRepository` for unit tests
//! - `Neo4jTestConfig`: Connection config for integration tests against a real Neo4j instance
//! - `neo4j_available()`: Async check for Neo4j availability (skips tests in CI without Neo4j)
//! - `create_test_ontology_repo()`: Convenience factory for mock repos pre-loaded with test data

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::models::graph::GraphData;
use crate::ports::ontology_repository::{
    AxiomType, InferenceResults, OntologyMetrics, OntologyRepository, OntologyRepositoryError,
    OwlAxiom, OwlClass, OwlProperty, PathfindingCacheEntry, PropertyType, ValidationReport,
    Result as OntResult,
};

// ---------------------------------------------------------------------------
// Neo4j test connection config (for integration tests)
// ---------------------------------------------------------------------------

/// Neo4j test connection configuration.
/// Reads from environment variables with sensible defaults for local dev.
pub struct Neo4jTestConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
}

impl Default for Neo4jTestConfig {
    fn default() -> Self {
        Self {
            uri: std::env::var("NEO4J_TEST_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            username: std::env::var("NEO4J_TEST_USER")
                .unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("NEO4J_TEST_PASSWORD")
                .unwrap_or_else(|_| "testpassword".to_string()),
        }
    }
}

/// Check if a Neo4j test instance is reachable.
/// Returns `false` when unavailable so tests can be skipped gracefully.
pub async fn neo4j_available() -> bool {
    let config = Neo4jTestConfig::default();
    match neo4rs::Graph::new(&config.uri, &config.username, &config.password) {
        Ok(graph) => graph.run(neo4rs::query("RETURN 1")).await.is_ok(),
        Err(_) => false,
    }
}

/// Skip the current test if Neo4j is not available.
/// Usage: `skip_without_neo4j!();` at the top of an async test.
#[macro_export]
macro_rules! skip_without_neo4j {
    () => {
        if !crate::test_helpers::neo4j_available().await {
            eprintln!("SKIPPED: Neo4j test instance not available");
            return;
        }
    };
}

// ---------------------------------------------------------------------------
// MockOntologyRepository -- in-memory impl for unit tests
// ---------------------------------------------------------------------------

/// In-memory `OntologyRepository` implementation for unit testing.
/// No Neo4j connection required.
pub struct MockOntologyRepository {
    classes: RwLock<HashMap<String, OwlClass>>,
    properties: RwLock<HashMap<String, OwlProperty>>,
    axioms: RwLock<Vec<OwlAxiom>>,
    next_axiom_id: RwLock<u64>,
    graph: RwLock<Option<GraphData>>,
    inference_results: RwLock<Option<InferenceResults>>,
}

impl MockOntologyRepository {
    pub fn new() -> Self {
        Self {
            classes: RwLock::new(HashMap::new()),
            properties: RwLock::new(HashMap::new()),
            axioms: RwLock::new(Vec::new()),
            next_axiom_id: RwLock::new(1),
            graph: RwLock::new(None),
            inference_results: RwLock::new(None),
        }
    }
}

#[async_trait]
impl OntologyRepository for MockOntologyRepository {
    async fn load_ontology_graph(&self) -> OntResult<Arc<GraphData>> {
        let g = self.graph.read().await;
        match &*g {
            Some(graph) => Ok(Arc::new(graph.clone())),
            None => Ok(Arc::new(GraphData::default())),
        }
    }

    async fn save_ontology_graph(&self, graph: &GraphData) -> OntResult<()> {
        let mut g = self.graph.write().await;
        *g = Some(graph.clone());
        Ok(())
    }

    async fn save_ontology(
        &self,
        classes: &[OwlClass],
        properties: &[OwlProperty],
        axioms: &[OwlAxiom],
    ) -> OntResult<()> {
        {
            let mut c = self.classes.write().await;
            for class in classes {
                c.insert(class.iri.clone(), class.clone());
            }
        }
        {
            let mut p = self.properties.write().await;
            for prop in properties {
                p.insert(prop.iri.clone(), prop.clone());
            }
        }
        {
            let mut a = self.axioms.write().await;
            for axiom in axioms {
                a.push(axiom.clone());
            }
        }
        Ok(())
    }

    async fn add_owl_class(&self, class: &OwlClass) -> OntResult<String> {
        let mut c = self.classes.write().await;
        c.insert(class.iri.clone(), class.clone());
        Ok(class.iri.clone())
    }

    async fn get_owl_class(&self, iri: &str) -> OntResult<Option<OwlClass>> {
        let c = self.classes.read().await;
        Ok(c.get(iri).cloned())
    }

    async fn list_owl_classes(&self) -> OntResult<Vec<OwlClass>> {
        let c = self.classes.read().await;
        Ok(c.values().cloned().collect())
    }

    async fn add_owl_property(&self, property: &OwlProperty) -> OntResult<String> {
        let mut p = self.properties.write().await;
        p.insert(property.iri.clone(), property.clone());
        Ok(property.iri.clone())
    }

    async fn get_owl_property(&self, iri: &str) -> OntResult<Option<OwlProperty>> {
        let p = self.properties.read().await;
        Ok(p.get(iri).cloned())
    }

    async fn list_owl_properties(&self) -> OntResult<Vec<OwlProperty>> {
        let p = self.properties.read().await;
        Ok(p.values().cloned().collect())
    }

    async fn get_classes(&self) -> OntResult<Vec<OwlClass>> {
        self.list_owl_classes().await
    }

    async fn get_axioms(&self) -> OntResult<Vec<OwlAxiom>> {
        let a = self.axioms.read().await;
        Ok(a.clone())
    }

    async fn add_axiom(&self, axiom: &OwlAxiom) -> OntResult<u64> {
        let mut id_lock = self.next_axiom_id.write().await;
        let id = *id_lock;
        *id_lock += 1;

        let mut stored = axiom.clone();
        stored.id = Some(id);

        let mut a = self.axioms.write().await;
        a.push(stored);
        Ok(id)
    }

    async fn get_class_axioms(&self, class_iri: &str) -> OntResult<Vec<OwlAxiom>> {
        let a = self.axioms.read().await;
        Ok(a.iter()
            .filter(|ax| ax.subject == class_iri || ax.object == class_iri)
            .cloned()
            .collect())
    }

    async fn store_inference_results(&self, results: &InferenceResults) -> OntResult<()> {
        let mut ir = self.inference_results.write().await;
        *ir = Some(results.clone());
        Ok(())
    }

    async fn get_inference_results(&self) -> OntResult<Option<InferenceResults>> {
        let ir = self.inference_results.read().await;
        Ok(ir.clone())
    }

    async fn remove_owl_class(&self, iri: &str) -> OntResult<()> {
        let mut c = self.classes.write().await;
        c.remove(iri);
        Ok(())
    }

    async fn remove_axiom(&self, axiom_id: u64) -> OntResult<()> {
        let mut a = self.axioms.write().await;
        a.retain(|ax| ax.id != Some(axiom_id));
        Ok(())
    }

    async fn get_metrics(&self) -> OntResult<OntologyMetrics> {
        let c = self.classes.read().await;
        let p = self.properties.read().await;
        let a = self.axioms.read().await;
        Ok(OntologyMetrics {
            class_count: c.len(),
            property_count: p.len(),
            axiom_count: a.len(),
            max_depth: 0,
            average_branching_factor: 0.0,
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience factories
// ---------------------------------------------------------------------------

/// Create a `MockOntologyRepository` pre-loaded with standard MindVault ontology classes.
/// Includes: mv:Person, mv:Company, mv:Technology, mv:Event, mv:Location, mv:Organization.
pub fn create_test_ontology_repo() -> Arc<MockOntologyRepository> {
    let repo = MockOntologyRepository::new();

    // Pre-populate with common test classes synchronously via direct lock
    let classes = vec![
        ("mv:Person", "Person"),
        ("mv:Company", "Company"),
        ("mv:Technology", "Technology"),
        ("mv:Event", "Event"),
        ("mv:Location", "Location"),
        ("mv:Organization", "Organization"),
    ];

    // Use blocking lock since this is called during test setup (not in async context)
    let mut class_map = repo.classes.blocking_write();
    for (iri, label) in classes {
        class_map.insert(
            iri.to_string(),
            OwlClass {
                iri: iri.to_string(),
                label: Some(label.to_string()),
                ..OwlClass::default()
            },
        );
    }
    drop(class_map);

    Arc::new(repo)
}

/// Create an `OntologyReasoner` backed by a mock repository for unit testing.
pub fn create_test_reasoner() -> crate::services::ontology_reasoner::OntologyReasoner {
    let engine = Arc::new(crate::adapters::whelk_inference_engine::WhelkInferenceEngine::new());
    let repo = create_test_ontology_repo();
    crate::services::ontology_reasoner::OntologyReasoner::new(engine, repo)
}

/// Create an `OntologyEnrichmentService` backed by mock implementations for unit testing.
pub fn create_test_enrichment_service() -> crate::services::ontology_enrichment_service::OntologyEnrichmentService {
    let reasoner = Arc::new(create_test_reasoner());
    let classifier = Arc::new(crate::services::edge_classifier::EdgeClassifier::new());
    crate::services::ontology_enrichment_service::OntologyEnrichmentService::new(reasoner, classifier)
}

/// Create an `OntologyReasoningService` backed by a mock repository for unit testing.
pub fn create_test_reasoning_service() -> crate::services::ontology_reasoning_service::OntologyReasoningService {
    let engine = Arc::new(crate::adapters::whelk_inference_engine::WhelkInferenceEngine::new());
    let repo = create_test_ontology_repo();
    crate::services::ontology_reasoning_service::OntologyReasoningService::new(engine, repo)
}
