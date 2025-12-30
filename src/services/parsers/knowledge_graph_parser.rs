// src/services/parsers/knowledge_graph_parser.rs
//! Knowledge Graph Parser
//!
//! Parses markdown files marked with `public:: true` to extract:
//! - Nodes (pages, concepts)
//! - Edges (links, relationships)
//! - Metadata (properties, tags)

use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::metadata::MetadataStore;
use crate::models::node::Node;
use crate::utils::socket_flow_messages::BinaryNodeData;
use log::{debug, info};
use std::collections::HashMap;

/// Knowledge graph parser with position preservation support
pub struct KnowledgeGraphParser {
    /// Existing positions from database (node_id -> (x, y, z))
    existing_positions: Option<HashMap<u32, (f32, f32, f32)>>,
}

impl KnowledgeGraphParser {
    pub fn new() -> Self {
        Self {
            existing_positions: None,
        }
    }

    /// Create parser with existing positions from database
    /// These positions will be used instead of generating random ones
    pub fn with_positions(existing_positions: HashMap<u32, (f32, f32, f32)>) -> Self {
        Self {
            existing_positions: Some(existing_positions),
        }
    }

    /// Set existing positions for position preservation
    pub fn set_positions(&mut self, positions: HashMap<u32, (f32, f32, f32)>) {
        self.existing_positions = Some(positions);
    }

    /// Get position for a node ID, using existing position or generating random
    fn get_position(&self, node_id: u32) -> (f32, f32, f32) {
        if let Some(ref positions) = self.existing_positions {
            if let Some(&(x, y, z)) = positions.get(&node_id) {
                return (x, y, z);
            }
        }
        // Generate random position only if no existing position found
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (
            rng.gen_range(-100.0..100.0),
            rng.gen_range(-100.0..100.0),
            rng.gen_range(-100.0..100.0),
        )
    }

    
    pub fn parse(&self, content: &str, filename: &str) -> Result<GraphData, String> {
        info!("Parsing knowledge graph file: {}", filename);

        
        let page_name = filename.strip_suffix(".md").unwrap_or(filename).to_string();

        
        let mut nodes = vec![self.create_page_node(&page_name, content)];
        let mut id_to_metadata = HashMap::new();
        id_to_metadata.insert(nodes[0].id.to_string(), page_name.clone());

        
        
        
        let (linked_nodes, file_edges) = self.extract_links(content, &nodes[0].id);
        for node in &linked_nodes {
            id_to_metadata.insert(node.id.to_string(), node.metadata_id.clone());
        }
        nodes.extend(linked_nodes);

        
        let metadata = self.extract_metadata_store(content);

        debug!(
            "Parsed {}: {} nodes, {} edges (linked nodes will be filtered)",
            filename,
            nodes.len(),
            file_edges.len()
        );

        Ok(GraphData {
            nodes,
            edges: file_edges,
            metadata,
            id_to_metadata,
        })
    }

    /// Create a page node, preserving existing position if available
    fn create_page_node(&self, page_name: &str, content: &str) -> Node {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "page".to_string());
        metadata.insert("source_file".to_string(), format!("{}.md", page_name));
        metadata.insert("public".to_string(), "true".to_string());

        let tags = self.extract_tags(content);
        if !tags.is_empty() {
            metadata.insert("tags".to_string(), tags.join(", "));
        }

        let id = self.page_name_to_id(page_name);

        // Use existing position or generate random (position preservation)
        let (x, y, z) = self.get_position(id);
        let data = BinaryNodeData {
            node_id: id,
            x,
            y,
            z,
            vx: 0.0,
            vy: 0.0,
            vz: 0.0,
        };

        Node {
            id,
            metadata_id: page_name.to_string(),
            label: page_name.to_string(),
            data,
            metadata,
            file_size: 0,
            node_type: Some("page".to_string()),
            color: Some("#4A90E2".to_string()),
            size: Some(1.0),
            weight: Some(1.0),
            group: None,
            user_data: None,
            mass: Some(1.0),
            x: Some(data.x),
            y: Some(data.y),
            z: Some(data.z),
            vx: Some(0.0),
            vy: Some(0.0),
            vz: Some(0.0),
            owl_class_iri: None,
        }
    }

    /// Extract links from content, preserving existing positions
    fn extract_links(&self, content: &str, source_id: &u32) -> (Vec<Node>, Vec<Edge>) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        let link_pattern = regex::Regex::new(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]").expect("Invalid regex pattern");

        for cap in link_pattern.captures_iter(content) {
            if let Some(link_match) = cap.get(1) {
                let target_page = link_match.as_str().trim().to_string();
                let target_id = self.page_name_to_id(&target_page);

                let mut metadata = HashMap::new();
                metadata.insert("type".to_string(), "linked_page".to_string());

                // Use existing position or generate random (position preservation)
                let (x, y, z) = self.get_position(target_id);
                let data = BinaryNodeData {
                    node_id: target_id,
                    x,
                    y,
                    z,
                    vx: 0.0,
                    vy: 0.0,
                    vz: 0.0,
                };

                nodes.push(Node {
                    id: target_id,
                    metadata_id: target_page.clone(),
                    label: target_page.clone(),
                    data,
                    metadata,
                    file_size: 0,
                    node_type: Some("linked_page".to_string()),
                    color: Some("#7C3AED".to_string()),
                    size: Some(0.8),
                    weight: Some(0.8),
                    group: None,
                    user_data: None,
                    mass: Some(1.0),
                    x: Some(data.x),
                    y: Some(data.y),
                    z: Some(data.z),
                    vx: Some(0.0),
                    vy: Some(0.0),
                    vz: Some(0.0),
                    owl_class_iri: None,
                });

                edges.push(Edge {
                    id: format!("{}_{}", source_id, target_id),
                    source: *source_id,
                    target: target_id,
                    weight: 1.0,
                    edge_type: Some("link".to_string()),
                    metadata: Some(HashMap::new()),
                    owl_property_iri: None,
                });
            }
        }

        (nodes, edges)
    }

    
    fn extract_metadata_store(&self, content: &str) -> MetadataStore {
        let store = MetadataStore::new();

        
        let prop_pattern = regex::Regex::new(r"([a-zA-Z_]+)::\s*(.+)").expect("Invalid regex pattern");

        
        let mut properties = HashMap::new();
        for cap in prop_pattern.captures_iter(content) {
            if let (Some(key), Some(value)) = (cap.get(1), cap.get(2)) {
                let key_str = key.as_str().to_string();
                let value_str = value.as_str().trim().to_string();

                
                properties.insert(key_str, value_str);
            }
        }

        
        
        store
    }

    
    fn extract_tags(&self, content: &str) -> Vec<String> {
        let mut tags = Vec::new();

        
        let tag_pattern =
            regex::Regex::new(r"#([a-zA-Z0-9_-]+)|tag::\s*#?([a-zA-Z0-9_-]+)").expect("Invalid regex pattern");

        for cap in tag_pattern.captures_iter(content) {
            if let Some(tag) = cap.get(1).or_else(|| cap.get(2)) {
                tags.push(tag.as_str().to_string());
            }
        }

        tags.dedup();
        tags
    }

    
    fn page_name_to_id(&self, page_name: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        page_name.hash(&mut hasher);
        let hash_val = hasher.finish();
        
        ((hash_val % 999999) as u32) + 1
    }
}

impl Default for KnowledgeGraphParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_preservation() {
        let mut positions = HashMap::new();
        positions.insert(12345u32, (10.0f32, 20.0f32, 30.0f32));

        let parser = KnowledgeGraphParser::with_positions(positions);
        let pos = parser.get_position(12345);

        assert_eq!(pos, (10.0, 20.0, 30.0));
    }

    #[test]
    fn test_fallback_to_random() {
        let parser = KnowledgeGraphParser::new();
        let pos = parser.get_position(99999);

        // Should be within random range
        assert!(pos.0 >= -100.0 && pos.0 <= 100.0);
        assert!(pos.1 >= -100.0 && pos.1 <= 100.0);
        assert!(pos.2 >= -100.0 && pos.2 <= 100.0);
    }
}
