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

pub struct KnowledgeGraphParser;

impl KnowledgeGraphParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse markdown content into graph data
    pub fn parse(&self, content: &str, filename: &str) -> Result<GraphData, String> {
        info!("Parsing knowledge graph file: {}", filename);

        // Extract the page name from filename (remove .md extension)
        let page_name = filename.strip_suffix(".md").unwrap_or(filename).to_string();

        // Create the main node for this page
        let mut nodes = vec![self.create_page_node(&page_name, content)];
        let mut id_to_metadata = HashMap::new();
        id_to_metadata.insert(nodes[0].id.to_string(), page_name.clone());

        // Extract linked nodes and edges
        // ⚠️ NOTE: Linked nodes will be filtered in github_sync_service
        // Only linked pages with public:: true will be retained
        let (linked_nodes, file_edges) = self.extract_links(content, &nodes[0].id);
        for node in &linked_nodes {
            id_to_metadata.insert(node.id.to_string(), node.metadata_id.clone());
        }
        nodes.extend(linked_nodes);

        // Extract metadata
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

    /// Create a node for the page itself
    fn create_page_node(&self, page_name: &str, content: &str) -> Node {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "page".to_string());
        metadata.insert("source_file".to_string(), format!("{}.md", page_name));
        metadata.insert("public".to_string(), "true".to_string()); // ✅ FIX: Mark as public

        // Extract tags if present
        let tags = self.extract_tags(content);
        if !tags.is_empty() {
            metadata.insert("tags".to_string(), tags.join(", "));
        }

        // Generate a deterministic ID from the page name
        let id = self.page_name_to_id(page_name);

        // Create BinaryNodeData with random position
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data = BinaryNodeData {
            node_id: id,
            x: rng.gen_range(-100.0..100.0),
            y: rng.gen_range(-100.0..100.0),
            z: rng.gen_range(-100.0..100.0),
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
            color: Some("#4A90E2".to_string()), // Default blue color
            size: Some(1.0),
            weight: Some(1.0),
            group: None,
            user_data: None,
        }
    }

    /// Extract links and create nodes/edges for linked pages
    fn extract_links(&self, content: &str, source_id: &u32) -> (Vec<Node>, Vec<Edge>) {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Regex pattern for [[Link]] or [[Link|Display Text]]
        let link_pattern = regex::Regex::new(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]").unwrap();

        for cap in link_pattern.captures_iter(content) {
            if let Some(link_match) = cap.get(1) {
                let target_page = link_match.as_str().trim().to_string();
                let target_id = self.page_name_to_id(&target_page);

                // Create node for linked page
                let mut metadata = HashMap::new();
                metadata.insert("type".to_string(), "linked_page".to_string());

                use rand::Rng;
                let mut rng = rand::thread_rng();
                let data = BinaryNodeData {
                    node_id: target_id,
                    x: rng.gen_range(-100.0..100.0),
                    y: rng.gen_range(-100.0..100.0),
                    z: rng.gen_range(-100.0..100.0),
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
                    color: Some("#7C3AED".to_string()), // Purple for linked pages
                    size: Some(0.8),
                    weight: Some(0.8),
                    group: None,
                    user_data: None,
                });

                // Create edge from source to target
                edges.push(Edge {
                    id: format!("{}_{}", source_id, target_id),
                    source: *source_id,
                    target: target_id,
                    weight: 1.0,
                    edge_type: Some("link".to_string()),
                    metadata: Some(HashMap::new()),
                });
            }
        }

        (nodes, edges)
    }

    /// Extract metadata from markdown properties
    fn extract_metadata_store(&self, content: &str) -> MetadataStore {
        let mut store = MetadataStore::new();

        // Pattern: property:: value
        let prop_pattern = regex::Regex::new(r"([a-zA-Z_]+)::\s*(.+)").unwrap();

        // Collect properties first
        let mut properties = HashMap::new();
        for cap in prop_pattern.captures_iter(content) {
            if let (Some(key), Some(value)) = (cap.get(1), cap.get(2)) {
                let key_str = key.as_str().to_string();
                let value_str = value.as_str().trim().to_string();

                // ✅ FIX: Store ALL properties including "public" for verification
                properties.insert(key_str, value_str);
            }
        }

        // For now, return empty MetadataStore - full implementation would create Metadata structs
        // This is a simplified version for the initial GitHub sync
        store
    }

    /// Extract tags from content
    fn extract_tags(&self, content: &str) -> Vec<String> {
        let mut tags = Vec::new();

        // Pattern: #tag or tag::
        let tag_pattern =
            regex::Regex::new(r"#([a-zA-Z0-9_-]+)|tag::\s*#?([a-zA-Z0-9_-]+)").unwrap();

        for cap in tag_pattern.captures_iter(content) {
            if let Some(tag) = cap.get(1).or_else(|| cap.get(2)) {
                tags.push(tag.as_str().to_string());
            }
        }

        tags.dedup();
        tags
    }

    /// Convert page name to deterministic numeric ID
    fn page_name_to_id(&self, page_name: &str) -> u32 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        page_name.hash(&mut hasher);
        let hash_val = hasher.finish();
        // Keep IDs in a reasonable range and avoid 0
        ((hash_val % 999999) as u32) + 1
    }
}

impl Default for KnowledgeGraphParser {
    fn default() -> Self {
        Self::new()
    }
}
