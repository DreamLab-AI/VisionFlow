// src/cqrs/queries/ontology_queries.rs
//! Ontology Queries
//!
//! Read operations for ontology repository.

use crate::cqrs::types::{Query, Result};
use crate::models::graph::GraphData;
use crate::ports::ontology_repository::{
    InferenceResults, OntologyMetrics, OwlAxiom, OwlClass, OwlProperty, PathfindingCacheEntry,
    ValidationReport,
};
use std::collections::HashMap;
use std::sync::Arc;

/// Get an OWL class by IRI
#[derive(Debug, Clone)]
pub struct GetClassQuery {
    pub iri: String,
}

impl Query for GetClassQuery {
    type Result = Option<OwlClass>;

    fn name(&self) -> &'static str {
        "GetClass"
    }

    fn validate(&self) -> Result<()> {
        if self.iri.is_empty() {
            return Err(anyhow::anyhow!("Class IRI cannot be empty"));
        }
        Ok(())
    }
}

/// List all OWL classes
#[derive(Debug, Clone)]
pub struct ListClassesQuery;

impl Query for ListClassesQuery {
    type Result = Vec<OwlClass>;

    fn name(&self) -> &'static str {
        "ListClasses"
    }
}

/// Get class hierarchy tree
#[derive(Debug, Clone)]
pub struct GetClassHierarchyQuery {
    pub root_iri: Option<String>, // If None, get entire hierarchy
}

impl Query for GetClassHierarchyQuery {
    type Result = Vec<OwlClass>; // Returns classes in hierarchical order

    fn name(&self) -> &'static str {
        "GetClassHierarchy"
    }
}

/// Get an OWL property by IRI
#[derive(Debug, Clone)]
pub struct GetPropertyQuery {
    pub iri: String,
}

impl Query for GetPropertyQuery {
    type Result = Option<OwlProperty>;

    fn name(&self) -> &'static str {
        "GetProperty"
    }

    fn validate(&self) -> Result<()> {
        if self.iri.is_empty() {
            return Err(anyhow::anyhow!("Property IRI cannot be empty"));
        }
        Ok(())
    }
}

/// List all OWL properties
#[derive(Debug, Clone)]
pub struct ListPropertiesQuery;

impl Query for ListPropertiesQuery {
    type Result = Vec<OwlProperty>;

    fn name(&self) -> &'static str {
        "ListProperties"
    }
}

/// Get all axioms for a class
#[derive(Debug, Clone)]
pub struct GetAxiomsForClassQuery {
    pub class_iri: String,
}

impl Query for GetAxiomsForClassQuery {
    type Result = Vec<OwlAxiom>;

    fn name(&self) -> &'static str {
        "GetAxiomsForClass"
    }

    fn validate(&self) -> Result<()> {
        if self.class_iri.is_empty() {
            return Err(anyhow::anyhow!("Class IRI cannot be empty"));
        }
        Ok(())
    }
}

/// Get latest inference results
#[derive(Debug, Clone)]
pub struct GetInferenceResultsQuery;

impl Query for GetInferenceResultsQuery {
    type Result = Option<InferenceResults>;

    fn name(&self) -> &'static str {
        "GetInferenceResults"
    }
}

/// Validate ontology consistency
#[derive(Debug, Clone)]
pub struct ValidateOntologyQuery;

impl Query for ValidateOntologyQuery {
    type Result = ValidationReport;

    fn name(&self) -> &'static str {
        "ValidateOntology"
    }
}

/// Query ontology using SPARQL-like syntax
#[derive(Debug, Clone)]
pub struct QueryOntologyQuery {
    pub query: String,
}

impl Query for QueryOntologyQuery {
    type Result = Vec<HashMap<String, String>>;

    fn name(&self) -> &'static str {
        "QueryOntology"
    }

    fn validate(&self) -> Result<()> {
        if self.query.is_empty() {
            return Err(anyhow::anyhow!("Query string cannot be empty"));
        }
        Ok(())
    }
}

/// Get ontology metrics
#[derive(Debug, Clone)]
pub struct GetOntologyMetricsQuery;

impl Query for GetOntologyMetricsQuery {
    type Result = OntologyMetrics;

    fn name(&self) -> &'static str {
        "GetOntologyMetrics"
    }
}

/// Load ontology graph structure
#[derive(Debug, Clone)]
pub struct LoadOntologyGraphQuery;

impl Query for LoadOntologyGraphQuery {
    type Result = Arc<GraphData>;

    fn name(&self) -> &'static str {
        "LoadOntologyGraph"
    }
}

/// Export ontology to OWL XML
#[derive(Debug, Clone)]
pub struct ExportOntologyQuery;

impl Query for ExportOntologyQuery {
    type Result = String; // OWL XML string

    fn name(&self) -> &'static str {
        "ExportOntology"
    }
}

/// Get cached SSSP result
#[derive(Debug, Clone)]
pub struct GetCachedSsspQuery {
    pub source_node_id: u32,
}

impl Query for GetCachedSsspQuery {
    type Result = Option<PathfindingCacheEntry>;

    fn name(&self) -> &'static str {
        "GetCachedSssp"
    }
}

/// Get cached APSP distance matrix
#[derive(Debug, Clone)]
pub struct GetCachedApspQuery;

impl Query for GetCachedApspQuery {
    type Result = Option<Vec<Vec<f32>>>;

    fn name(&self) -> &'static str {
        "GetCachedApsp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_class_validation() {
        let query = GetClassQuery {
            iri: "http://example.org/Class1".to_string(),
        };
        assert!(query.validate().is_ok());

        let query = GetClassQuery {
            iri: "".to_string(),
        };
        assert!(query.validate().is_err());
    }

    #[test]
    fn test_query_ontology_validation() {
        let query = QueryOntologyQuery {
            query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
        };
        assert!(query.validate().is_ok());

        let query = QueryOntologyQuery {
            query: "".to_string(),
        };
        assert!(query.validate().is_err());
    }
}
