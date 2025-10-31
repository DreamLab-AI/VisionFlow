use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Edge structure representing connections between nodes
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Edge {
    pub id: String, // Added ID field
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,

    // OWL Ontology linkage (matches unified_schema.sql graph_edges table)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owl_property_iri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl Edge {
    pub fn new(source: u32, target: u32, weight: f32) -> Self {
        // Generate a simple unique ID for the edge
        let id = format!("{}-{}", source, target);
        Self {
            id,
            source,
            target,
            weight,
            edge_type: None,
            owl_property_iri: None,
            metadata: None,
        }
    }

    /// Create an edge with an OWL property IRI
    pub fn with_owl_property_iri(mut self, iri: String) -> Self {
        self.owl_property_iri = Some(iri);
        self
    }

    /// Create an edge with a specific type
    pub fn with_edge_type(mut self, edge_type: String) -> Self {
        self.edge_type = Some(edge_type);
        self
    }

    /// Create an edge with metadata
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Add a single metadata entry
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        if let Some(ref mut map) = self.metadata {
            map.insert(key, value);
        } else {
            let mut map = HashMap::new();
            map.insert(key, value);
            self.metadata = Some(map);
        }
        self
    }
}
