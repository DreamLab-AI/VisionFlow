//! Data models for Neo4j graph operations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Represents a triple in the knowledge graph (subject, predicate, object)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: Option<f64>,
    pub source: Option<String>,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Triple {
    pub fn new<S: Into<String>>(subject: S, predicate: S, object: S) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            confidence: None,
            source: None,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }
    
    pub fn with_source<S: Into<String>>(mut self, source: S) -> Self {
        self.source = Some(source.into());
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Represents a node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: Option<String>,
    pub labels: Vec<String>,
    pub properties: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            id: Some(Uuid::new_v4().to_string()),
            labels: Vec::new(),
            properties: HashMap::new(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }
    
    pub fn with_id<S: Into<String>>(mut self, id: S) -> Self {
        self.id = Some(id.into());
        self
    }
    
    pub fn with_label<S: Into<String>>(mut self, label: S) -> Self {
        self.labels.push(label.into());
        self
    }
    
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = labels;
        self
    }
    
    pub fn with_property<S: Into<String>>(mut self, key: S, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
    
    pub fn with_properties(mut self, properties: HashMap<String, serde_json::Value>) -> Self {
        self.properties = properties;
        self
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a relationship in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: Option<String>,
    pub relationship_type: String,
    pub start_node_id: String,
    pub end_node_id: String,
    pub properties: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl Relationship {
    pub fn new<S: Into<String>>(relationship_type: S, start_node_id: S, end_node_id: S) -> Self {
        Self {
            id: Some(Uuid::new_v4().to_string()),
            relationship_type: relationship_type.into(),
            start_node_id: start_node_id.into(),
            end_node_id: end_node_id.into(),
            properties: HashMap::new(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }
    
    pub fn with_id<S: Into<String>>(mut self, id: S) -> Self {
        self.id = Some(id.into());
        self
    }
    
    pub fn with_property<S: Into<String>>(mut self, key: S, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }
    
    pub fn with_properties(mut self, properties: HashMap<String, serde_json::Value>) -> Self {
        self.properties = properties;
        self
    }
}

/// Container for graph data import/export
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub relationships: Vec<Relationship>,
    pub triples: Vec<Triple>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl GraphData {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn add_node(mut self, node: Node) -> Self {
        self.nodes.push(node);
        self
    }
    
    pub fn add_relationship(mut self, relationship: Relationship) -> Self {
        self.relationships.push(relationship);
        self
    }
    
    pub fn add_triple(mut self, triple: Triple) -> Self {
        self.triples.push(triple);
        self
    }
    
    pub fn with_metadata<S: Into<String>>(mut self, key: S, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
    
    /// Convert triples to nodes and relationships
    pub fn triples_to_graph(&mut self) {
        let mut node_map: HashMap<String, Node> = HashMap::new();
        
        for triple in &self.triples {
            // Create subject node if not exists
            if !node_map.contains_key(&triple.subject) {
                let mut node = Node::new()
                    .with_id(triple.subject.clone())
                    .with_label("Entity")
                    .with_property("name", serde_json::Value::String(triple.subject.clone()));
                
                if let Some(ref source) = triple.source {
                    node = node.with_property("source", serde_json::Value::String(source.clone()));
                }
                
                node_map.insert(triple.subject.clone(), node);
            }
            
            // Create object node if not exists
            if !node_map.contains_key(&triple.object) {
                let mut node = Node::new()
                    .with_id(triple.object.clone())
                    .with_label("Entity")
                    .with_property("name", serde_json::Value::String(triple.object.clone()));
                
                if let Some(ref source) = triple.source {
                    node = node.with_property("source", serde_json::Value::String(source.clone()));
                }
                
                node_map.insert(triple.object.clone(), node);
            }
            
            // Create relationship
            let mut relationship = Relationship::new(
                triple.predicate.clone(),
                triple.subject.clone(),
                triple.object.clone(),
            );
            
            if let Some(confidence) = triple.confidence {
                relationship = relationship.with_property(
                    "confidence",
                    serde_json::Value::Number(serde_json::Number::from_f64(confidence).unwrap()),
                );
            }
            
            if let Some(ref source) = triple.source {
                relationship = relationship.with_property(
                    "source",
                    serde_json::Value::String(source.clone()),
                );
            }
            
            relationship = relationship.with_property(
                "created_at",
                serde_json::Value::String(triple.created_at.to_rfc3339()),
            );
            
            self.relationships.push(relationship);
        }
        
        // Add all nodes to the graph
        self.nodes.extend(node_map.into_values());
    }
    
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty() && self.relationships.is_empty() && self.triples.is_empty()
    }
    
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn relationship_count(&self) -> usize {
        self.relationships.len()
    }
    
    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }
}

/// Query parameters for graph operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub order_by: Option<String>,
    pub filters: HashMap<String, serde_json::Value>,
}

impl QueryParams {
    pub fn new() -> Self {
        Self {
            limit: None,
            offset: None,
            order_by: None,
            filters: HashMap::new(),
        }
    }
    
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }
    
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
    
    pub fn with_order_by<S: Into<String>>(mut self, order_by: S) -> Self {
        self.order_by = Some(order_by.into());
        self
    }
    
    pub fn with_filter<S: Into<String>>(mut self, key: S, value: serde_json::Value) -> Self {
        self.filters.insert(key.into(), value);
        self
    }
}

impl Default for QueryParams {
    fn default() -> Self {
        Self::new()
    }
}
