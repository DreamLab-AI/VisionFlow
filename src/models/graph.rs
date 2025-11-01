use super::edge::Edge;
use super::metadata::MetadataStore;
use crate::models::node::Node;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/
/
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GraphData {
    
    pub nodes: Vec<Node>,
    
    pub edges: Vec<Edge>,
    
    pub metadata: MetadataStore,
    
    #[serde(skip)]
    pub id_to_metadata: HashMap<String, String>,
}

impl GraphData {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: MetadataStore::new(),
            id_to_metadata: HashMap::new(),
        }
    }
}
