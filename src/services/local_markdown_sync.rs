// src/services/local_markdown_sync.rs
//! Local Markdown Sync Service
//!
//! Reads markdown files from local directory and populates knowledge_graph.db

use crate::models::edge::Edge;
use crate::models::node::Node;
use crate::services::parsers::knowledge_graph_parser::KnowledgeGraphParser;
use log::{debug, info};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

pub struct LocalMarkdownSync;

impl LocalMarkdownSync {
    pub fn new() -> Self {
        Self
    }

    /// Sync from local markdown directory
    pub fn sync_from_directory(&self, dir_path: &str) -> Result<LocalSyncResult, String> {
        info!("Starting local markdown sync from: {}", dir_path);

        let path = Path::new(dir_path);
        if !path.exists() {
            return Err(format!("Directory does not exist: {}", dir_path));
        }

        let parser = KnowledgeGraphParser::new();
        let mut accumulated_nodes: HashMap<u32, Node> = HashMap::new();
        let mut accumulated_edges: HashMap<String, Edge> = HashMap::new();
        let mut public_page_names: HashSet<String> = HashSet::new();

        let mut total_files = 0;
        let mut processed_files = 0;
        let mut skipped_files = 0;

        // Read all markdown files
        let entries = fs::read_dir(path)
            .map_err(|e| format!("Failed to read directory: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let file_path = entry.path();

            if !file_path.is_file() {
                continue;
            }

            let filename = file_path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if !filename.ends_with(".md") {
                continue;
            }

            total_files += 1;

            // Read file content
            let content = match fs::read_to_string(&file_path) {
                Ok(c) => c,
                Err(e) => {
                    debug!("Failed to read {}: {}", filename, e);
                    skipped_files += 1;
                    continue;
                }
            };

            // Check for public:: true
            if !content.contains("public:: true") {
                debug!("Skipping {} - no public:: true marker", filename);
                skipped_files += 1;
                continue;
            }

            // Add to public page names (strip .md extension)
            let page_name = filename.strip_suffix(".md").unwrap_or(filename);
            public_page_names.insert(page_name.to_string());

            // Parse file
            match parser.parse(&content, filename) {
                Ok(graph_data) => {
                    // Accumulate ALL nodes (no filtering yet)
                    for node in graph_data.nodes {
                        accumulated_nodes.insert(node.id, node);
                    }

                    // Accumulate edges (deduplication via HashMap)
                    for edge in graph_data.edges {
                        accumulated_edges.insert(edge.id.clone(), edge);
                    }

                    processed_files += 1;
                    if processed_files % 10 == 0 {
                        info!("Progress: {}/{} files processed", processed_files, total_files);
                    }
                },
                Err(e) => {
                    debug!("Failed to parse {}: {}", filename, e);
                    skipped_files += 1;
                }
            }
        }

        info!("File processing complete. Total: {}, Processed: {}, Skipped: {}",
              total_files, processed_files, skipped_files);
        info!("Public page names collected: {}", public_page_names.len());

        // ✅ TWO-PASS FILTERING: Now filter linked_page nodes
        info!("Filtering linked_page nodes against {} public pages", public_page_names.len());
        let node_count_before_filter = accumulated_nodes.len();
        accumulated_nodes.retain(|_id, node| {
            match node.metadata.get("type").map(|s| s.as_str()) {
                Some("page") => true, // Keep all page nodes
                Some("linked_page") => {
                    let is_public = public_page_names.contains(&node.metadata_id);
                    if !is_public {
                        debug!("Filtered out linked_page '{}' - not in public pages", node.metadata_id);
                    }
                    is_public
                },
                _ => true,
            }
        });
        let nodes_filtered = node_count_before_filter - accumulated_nodes.len();
        info!("Filtered {} linked_page nodes (kept {} of {} total nodes)",
              nodes_filtered, accumulated_nodes.len(), node_count_before_filter);

        // Filter orphan edges
        let edge_count_before_filter = accumulated_edges.len();
        accumulated_edges.retain(|_id, edge| {
            accumulated_nodes.contains_key(&edge.source) && accumulated_nodes.contains_key(&edge.target)
        });
        let edges_filtered = edge_count_before_filter - accumulated_edges.len();
        info!("Filtered {} orphan edges (kept {} of {} total edges)",
              edges_filtered, accumulated_edges.len(), edge_count_before_filter);

        // Convert to vectors
        let nodes: Vec<Node> = accumulated_nodes.into_values().collect();
        let edges: Vec<Edge> = accumulated_edges.into_values().collect();

        info!("Local sync complete: {} nodes, {} edges", nodes.len(), edges.len());

        Ok(LocalSyncResult {
            total_files,
            processed_files,
            skipped_files,
            nodes,
            edges,
        })
    }
}

#[derive(Debug)]
pub struct LocalSyncResult {
    pub total_files: usize,
    pub processed_files: usize,
    pub skipped_files: usize,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}
