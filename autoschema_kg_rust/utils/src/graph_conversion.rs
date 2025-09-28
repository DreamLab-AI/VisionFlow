//! Graph conversion utilities for transforming data between different graph formats

use crate::{Result, UtilsError};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Configuration for graph conversion operations
#[derive(Debug, Clone)]
pub struct GraphConfig {
    pub directed: bool,
    pub include_attributes: bool,
    pub merge_duplicate_edges: bool,
    pub validate_structure: bool,
    pub max_nodes: usize,
    pub max_edges: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            directed: false,
            include_attributes: true,
            merge_duplicate_edges: true,
            validate_structure: true,
            max_nodes: 1_000_000,
            max_edges: 10_000_000,
        }
    }
}

/// Represents a graph node with attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: String,
    pub label: Option<String>,
    pub attributes: IndexMap<String, String>,
    pub node_type: Option<String>,
}

/// Represents a graph edge with attributes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub label: Option<String>,
    pub weight: Option<f64>,
    pub attributes: IndexMap<String, String>,
    pub edge_type: Option<String>,
}

/// Represents a complete graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub directed: bool,
    pub metadata: IndexMap<String, String>,
}

/// Graph statistics and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub is_directed: bool,
    pub is_connected: bool,
    pub has_cycles: bool,
    pub average_degree: f64,
    pub density: f64,
    pub components: usize,
}

impl Graph {
    /// Create a new empty graph
    pub fn new(directed: bool) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            directed,
            metadata: IndexMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) -> Result<()> {
        // Check for duplicate node IDs
        if self.nodes.iter().any(|n| n.id == node.id) {
            return Err(UtilsError::Custom(format!("Node with ID '{}' already exists", node.id)));
        }

        self.nodes.push(node);
        Ok(())
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) -> Result<()> {
        // Validate that source and target nodes exist
        let node_ids: HashSet<&String> = self.nodes.iter().map(|n| &n.id).collect();

        if !node_ids.contains(&edge.source) {
            return Err(UtilsError::Custom(format!("Source node '{}' does not exist", edge.source)));
        }

        if !node_ids.contains(&edge.target) {
            return Err(UtilsError::Custom(format!("Target node '{}' does not exist", edge.target)));
        }

        self.edges.push(edge);
        Ok(())
    }

    /// Get nodes connected to a specific node
    pub fn get_neighbors(&self, node_id: &str) -> Vec<&GraphNode> {
        let mut neighbors = Vec::new();
        let neighbor_ids: HashSet<&String> = self.edges.iter()
            .filter_map(|edge| {
                if edge.source == node_id {
                    Some(&edge.target)
                } else if !self.directed && edge.target == node_id {
                    Some(&edge.source)
                } else {
                    None
                }
            })
            .collect();

        for node in &self.nodes {
            if neighbor_ids.contains(&node.id) {
                neighbors.push(node);
            }
        }

        neighbors
    }

    /// Calculate graph statistics
    pub fn calculate_stats(&self) -> GraphStats {
        let node_count = self.nodes.len();
        let edge_count = self.edges.len();

        let average_degree = if node_count > 0 {
            if self.directed {
                (edge_count * 2) as f64 / node_count as f64
            } else {
                edge_count as f64 / node_count as f64
            }
        } else {
            0.0
        };

        let max_possible_edges = if self.directed {
            node_count * (node_count - 1)
        } else {
            node_count * (node_count - 1) / 2
        };

        let density = if max_possible_edges > 0 {
            edge_count as f64 / max_possible_edges as f64
        } else {
            0.0
        };

        let components = self.count_connected_components();
        let is_connected = components <= 1;
        let has_cycles = self.has_cycles();

        GraphStats {
            node_count,
            edge_count,
            is_directed: self.directed,
            is_connected,
            has_cycles,
            average_degree,
            density,
            components,
        }
    }

    /// Check if the graph has cycles
    fn has_cycles(&self) -> bool {
        if !self.directed {
            // For undirected graphs, use DFS to detect cycles
            let mut visited = HashSet::new();
            let mut parent_map = HashMap::new();

            for node in &self.nodes {
                if !visited.contains(&node.id) {
                    if self.has_cycle_undirected_dfs(&node.id, None, &mut visited, &mut parent_map) {
                        return true;
                    }
                }
            }
        } else {
            // For directed graphs, use topological sort approach
            let mut in_degree = HashMap::new();
            let mut adj_list = HashMap::new();

            // Initialize
            for node in &self.nodes {
                in_degree.insert(node.id.clone(), 0);
                adj_list.insert(node.id.clone(), Vec::new());
            }

            // Build adjacency list and in-degree count
            for edge in &self.edges {
                adj_list.get_mut(&edge.source).unwrap().push(edge.target.clone());
                *in_degree.get_mut(&edge.target).unwrap() += 1;
            }

            // Kahn's algorithm for cycle detection
            let mut queue = Vec::new();
            for (node, &degree) in &in_degree {
                if degree == 0 {
                    queue.push(node.clone());
                }
            }

            let mut processed = 0;
            while let Some(node) = queue.pop() {
                processed += 1;
                if let Some(neighbors) = adj_list.get(&node) {
                    for neighbor in neighbors {
                        let degree = in_degree.get_mut(neighbor).unwrap();
                        *degree -= 1;
                        if *degree == 0 {
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }

            return processed != self.nodes.len();
        }

        false
    }

    fn has_cycle_undirected_dfs(
        &self,
        node: &str,
        parent: Option<&str>,
        visited: &mut HashSet<String>,
        parent_map: &mut HashMap<String, String>,
    ) -> bool {
        visited.insert(node.to_string());

        for edge in &self.edges {
            let neighbor = if edge.source == node {
                Some(&edge.target)
            } else if edge.target == node {
                Some(&edge.source)
            } else {
                None
            };

            if let Some(neighbor) = neighbor {
                if Some(neighbor.as_str()) == parent {
                    continue;
                }

                if visited.contains(neighbor) {
                    return true;
                }

                parent_map.insert(neighbor.clone(), node.to_string());
                if self.has_cycle_undirected_dfs(neighbor, Some(node), visited, parent_map) {
                    return true;
                }
            }
        }

        false
    }

    /// Count connected components
    fn count_connected_components(&self) -> usize {
        let mut visited = HashSet::new();
        let mut components = 0;

        for node in &self.nodes {
            if !visited.contains(&node.id) {
                self.dfs(&node.id, &mut visited);
                components += 1;
            }
        }

        components
    }

    fn dfs(&self, node_id: &str, visited: &mut HashSet<String>) {
        visited.insert(node_id.to_string());

        for edge in &self.edges {
            let next_node = if edge.source == node_id {
                Some(&edge.target)
            } else if !self.directed && edge.target == node_id {
                Some(&edge.source)
            } else {
                None
            };

            if let Some(next_node) = next_node {
                if !visited.contains(next_node) {
                    self.dfs(next_node, visited);
                }
            }
        }
    }
}

/// Convert graph to GraphML format
pub fn graph_to_graphml<P: AsRef<Path>>(
    graph: &Graph,
    output_file: P,
    config: &GraphConfig,
) -> Result<()> {
    use quick_xml::events::{Event, BytesEnd, BytesStart, BytesText};
    use quick_xml::Writer;

    let file = File::create(output_file)?;
    let mut writer = Writer::new(BufWriter::new(file));

    // Write XML declaration
    writer.write_event(Event::Decl(quick_xml::events::BytesDecl::new("1.0", Some("UTF-8"), None)))?;

    // Start GraphML
    let mut graphml_start = BytesStart::new("graphml");
    graphml_start.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
    graphml_start.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
    graphml_start.push_attribute(("xsi:schemaLocation", "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"));
    writer.write_event(Event::Start(graphml_start))?;

    // Define attribute keys
    if config.include_attributes {
        // Node attributes
        let mut node_attrs = IndexMap::new();
        for node in &graph.nodes {
            for (key, _) in &node.attributes {
                node_attrs.insert(key.clone(), "string");
            }
        }

        for (attr_name, attr_type) in node_attrs {
            let mut key_start = BytesStart::new("key");
            key_start.push_attribute(("id", attr_name.as_str()));
            key_start.push_attribute(("for", "node"));
            key_start.push_attribute(("attr.name", attr_name.as_str()));
            key_start.push_attribute(("attr.type", attr_type));
            writer.write_event(Event::Empty(key_start))?;
        }

        // Edge attributes
        let mut edge_attrs = IndexMap::new();
        for edge in &graph.edges {
            for (key, _) in &edge.attributes {
                edge_attrs.insert(key.clone(), "string");
            }
        }

        for (attr_name, attr_type) in edge_attrs {
            let mut key_start = BytesStart::new("key");
            key_start.push_attribute(("id", attr_name.as_str()));
            key_start.push_attribute(("for", "edge"));
            key_start.push_attribute(("attr.name", attr_name.as_str()));
            key_start.push_attribute(("attr.type", attr_type));
            writer.write_event(Event::Empty(key_start))?;
        }
    }

    // Start graph
    let mut graph_start = BytesStart::new("graph");
    graph_start.push_attribute(("id", "G"));
    graph_start.push_attribute(("edgedefault", if graph.directed { "directed" } else { "undirected" }));
    writer.write_event(Event::Start(graph_start))?;

    // Write nodes
    for node in &graph.nodes {
        let mut node_start = BytesStart::new("node");
        node_start.push_attribute(("id", node.id.as_str()));
        writer.write_event(Event::Start(node_start))?;

        // Write node attributes
        if config.include_attributes {
            for (key, value) in &node.attributes {
                let mut data_start = BytesStart::new("data");
                data_start.push_attribute(("key", key.as_str()));
                writer.write_event(Event::Start(data_start))?;
                writer.write_event(Event::Text(BytesText::new(value)))?;
                writer.write_event(Event::End(BytesEnd::new("data")))?;
            }
        }

        writer.write_event(Event::End(BytesEnd::new("node")))?;
    }

    // Write edges
    for (edge_idx, edge) in graph.edges.iter().enumerate() {
        let mut edge_start = BytesStart::new("edge");
        edge_start.push_attribute(("id", format!("e{}", edge_idx).as_str()));
        edge_start.push_attribute(("source", edge.source.as_str()));
        edge_start.push_attribute(("target", edge.target.as_str()));
        writer.write_event(Event::Start(edge_start))?;

        // Write edge attributes
        if config.include_attributes {
            for (key, value) in &edge.attributes {
                let mut data_start = BytesStart::new("data");
                data_start.push_attribute(("key", key.as_str()));
                writer.write_event(Event::Start(data_start))?;
                writer.write_event(Event::Text(BytesText::new(value)))?;
                writer.write_event(Event::End(BytesEnd::new("data")))?;
            }
        }

        writer.write_event(Event::End(BytesEnd::new("edge")))?;
    }

    writer.write_event(Event::End(BytesEnd::new("graph")))?;
    writer.write_event(Event::End(BytesEnd::new("graphml")))?;

    Ok(())
}

/// Load graph from GraphML format
pub fn graphml_to_graph<P: AsRef<Path>>(
    graphml_file: P,
    config: &GraphConfig,
) -> Result<Graph> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let file = File::open(graphml_file)?;
    let mut reader = Reader::from_reader(BufReader::new(file));
    reader.trim_text(true);

    let mut graph = Graph::new(false); // Will be set from graph element
    let mut buf = Vec::new();
    let mut current_node: Option<GraphNode> = None;
    let mut current_edge: Option<GraphEdge> = None;
    let mut current_data_key = String::new();
    let mut in_data = false;

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(ref e)) => {
                match e.name().as_ref() {
                    b"graph" => {
                        for attr in e.attributes() {
                            let attr = attr?;
                            if attr.key.as_ref() == b"edgedefault" {
                                let value = std::str::from_utf8(&attr.value)?;
                                graph.directed = value == "directed";
                            }
                        }
                    }
                    b"node" => {
                        let mut node = GraphNode {
                            id: String::new(),
                            label: None,
                            attributes: IndexMap::new(),
                            node_type: None,
                        };

                        for attr in e.attributes() {
                            let attr = attr?;
                            if attr.key.as_ref() == b"id" {
                                node.id = std::str::from_utf8(&attr.value)?.to_string();
                            }
                        }

                        current_node = Some(node);
                    }
                    b"edge" => {
                        let mut edge = GraphEdge {
                            source: String::new(),
                            target: String::new(),
                            label: None,
                            weight: None,
                            attributes: IndexMap::new(),
                            edge_type: None,
                        };

                        for attr in e.attributes() {
                            let attr = attr?;
                            match attr.key.as_ref() {
                                b"source" => edge.source = std::str::from_utf8(&attr.value)?.to_string(),
                                b"target" => edge.target = std::str::from_utf8(&attr.value)?.to_string(),
                                _ => {}
                            }
                        }

                        current_edge = Some(edge);
                    }
                    b"data" => {
                        for attr in e.attributes() {
                            let attr = attr?;
                            if attr.key.as_ref() == b"key" {
                                current_data_key = std::str::from_utf8(&attr.value)?.to_string();
                                in_data = true;
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                match e.name().as_ref() {
                    b"node" => {
                        if let Some(node) = current_node.take() {
                            graph.add_node(node)?;
                        }
                    }
                    b"edge" => {
                        if let Some(edge) = current_edge.take() {
                            graph.add_edge(edge)?;
                        }
                    }
                    b"data" => {
                        in_data = false;
                        current_data_key.clear();
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(e)) => {
                if in_data {
                    let value = e.unescape()?.to_string();
                    if let Some(node) = &mut current_node {
                        node.attributes.insert(current_data_key.clone(), value);
                    } else if let Some(edge) = &mut current_edge {
                        edge.attributes.insert(current_data_key.clone(), value);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(UtilsError::Custom(format!("XML parsing error: {}", e))),
            _ => {}
        }

        buf.clear();
    }

    Ok(graph)
}

/// Convert graph to adjacency matrix representation
pub fn graph_to_adjacency_matrix(graph: &Graph) -> (Vec<String>, Vec<Vec<f64>>) {
    let node_ids: Vec<String> = graph.nodes.iter().map(|n| n.id.clone()).collect();
    let n = node_ids.len();
    let mut matrix = vec![vec![0.0; n]; n];

    let id_to_index: HashMap<&String, usize> = node_ids.iter().enumerate().map(|(i, id)| (id, i)).collect();

    for edge in &graph.edges {
        if let (Some(&source_idx), Some(&target_idx)) = (id_to_index.get(&edge.source), id_to_index.get(&edge.target)) {
            let weight = edge.weight.unwrap_or(1.0);
            matrix[source_idx][target_idx] = weight;

            if !graph.directed {
                matrix[target_idx][source_idx] = weight;
            }
        }
    }

    (node_ids, matrix)
}

/// Convert adjacency matrix back to graph
pub fn adjacency_matrix_to_graph(
    node_ids: &[String],
    matrix: &[Vec<f64>],
    directed: bool,
    threshold: f64,
) -> Result<Graph> {
    let mut graph = Graph::new(directed);

    // Add nodes
    for id in node_ids {
        graph.add_node(GraphNode {
            id: id.clone(),
            label: None,
            attributes: IndexMap::new(),
            node_type: None,
        })?;
    }

    // Add edges
    for i in 0..matrix.len() {
        for j in 0..matrix[i].len() {
            let weight = matrix[i][j];
            if weight > threshold {
                // Skip symmetric edges in undirected graphs
                if !directed && i > j {
                    continue;
                }

                let mut attributes = IndexMap::new();
                attributes.insert("weight".to_string(), weight.to_string());

                graph.add_edge(GraphEdge {
                    source: node_ids[i].clone(),
                    target: node_ids[j].clone(),
                    label: None,
                    weight: Some(weight),
                    attributes,
                    edge_type: None,
                })?;
            }
        }
    }

    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new(false);

        let node1 = GraphNode {
            id: "1".to_string(),
            label: Some("Node 1".to_string()),
            attributes: IndexMap::new(),
            node_type: None,
        };

        let node2 = GraphNode {
            id: "2".to_string(),
            label: Some("Node 2".to_string()),
            attributes: IndexMap::new(),
            node_type: None,
        };

        graph.add_node(node1).unwrap();
        graph.add_node(node2).unwrap();

        let edge = GraphEdge {
            source: "1".to_string(),
            target: "2".to_string(),
            label: None,
            weight: Some(1.0),
            attributes: IndexMap::new(),
            edge_type: None,
        };

        graph.add_edge(edge).unwrap();

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);

        let stats = graph.calculate_stats();
        assert_eq!(stats.node_count, 2);
        assert_eq!(stats.edge_count, 1);
    }

    #[test]
    fn test_adjacency_matrix_conversion() {
        let mut graph = Graph::new(false);

        graph.add_node(GraphNode { id: "A".to_string(), label: None, attributes: IndexMap::new(), node_type: None }).unwrap();
        graph.add_node(GraphNode { id: "B".to_string(), label: None, attributes: IndexMap::new(), node_type: None }).unwrap();

        graph.add_edge(GraphEdge {
            source: "A".to_string(),
            target: "B".to_string(),
            label: None,
            weight: Some(2.0),
            attributes: IndexMap::new(),
            edge_type: None,
        }).unwrap();

        let (node_ids, matrix) = graph_to_adjacency_matrix(&graph);
        assert_eq!(node_ids.len(), 2);
        assert_eq!(matrix[0][1], 2.0);
        assert_eq!(matrix[1][0], 2.0); // Undirected graph

        let reconstructed = adjacency_matrix_to_graph(&node_ids, &matrix, false, 0.5).unwrap();
        assert_eq!(reconstructed.nodes.len(), 2);
        assert_eq!(reconstructed.edges.len(), 1);
    }

    #[test]
    fn test_cycle_detection() {
        let mut graph = Graph::new(true);

        // Create a cycle: A -> B -> C -> A
        graph.add_node(GraphNode { id: "A".to_string(), label: None, attributes: IndexMap::new(), node_type: None }).unwrap();
        graph.add_node(GraphNode { id: "B".to_string(), label: None, attributes: IndexMap::new(), node_type: None }).unwrap();
        graph.add_node(GraphNode { id: "C".to_string(), label: None, attributes: IndexMap::new(), node_type: None }).unwrap();

        graph.add_edge(GraphEdge { source: "A".to_string(), target: "B".to_string(), label: None, weight: None, attributes: IndexMap::new(), edge_type: None }).unwrap();
        graph.add_edge(GraphEdge { source: "B".to_string(), target: "C".to_string(), label: None, weight: None, attributes: IndexMap::new(), edge_type: None }).unwrap();
        graph.add_edge(GraphEdge { source: "C".to_string(), target: "A".to_string(), label: None, weight: None, attributes: IndexMap::new(), edge_type: None }).unwrap();

        let stats = graph.calculate_stats();
        assert!(stats.has_cycles);
    }
}