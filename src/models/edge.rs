use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Edge {
    pub id: String, 
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,

    
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owl_property_iri: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl Edge {
    pub fn new(source: u32, target: u32, weight: f32) -> Self {
        
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

    
    pub fn with_owl_property_iri(mut self, iri: String) -> Self {
        self.owl_property_iri = Some(iri);
        self
    }

    
    pub fn with_edge_type(mut self, edge_type: String) -> Self {
        self.edge_type = Some(edge_type);
        self
    }

    
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    
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
