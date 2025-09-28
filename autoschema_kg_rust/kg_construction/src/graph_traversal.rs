//! Graph traversal utilities for context extraction

use crate::{
    error::{KgConstructionError, Result},
    types::GraphTraversal,
};
use std::collections::{HashMap, HashSet, VecDeque};

/// Graph node representation
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
    pub properties: HashMap<String, String>,
}

/// Graph edge representation
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
    pub properties: HashMap<String, String>,
}

/// In-memory graph structure for traversal operations
#[derive(Debug, Clone)]
pub struct Graph {
    nodes: HashMap<String, GraphNode>,
    outgoing_edges: HashMap<String, Vec<GraphEdge>>,
    incoming_edges: HashMap<String, Vec<GraphEdge>>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            outgoing_edges: HashMap::new(),
            incoming_edges: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: GraphNode) {
        let node_id = node.id.clone();
        self.nodes.insert(node_id.clone(), node);
        self.outgoing_edges.entry(node_id.clone()).or_default();
        self.incoming_edges.entry(node_id).or_default();
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: GraphEdge) -> Result<()> {
        // Ensure both nodes exist
        if !self.nodes.contains_key(&edge.source) {
            return Err(KgConstructionError::GraphError(format!(
                "Source node {} does not exist",
                edge.source
            )));
        }
        if !self.nodes.contains_key(&edge.target) {
            return Err(KgConstructionError::GraphError(format!(
                "Target node {} does not exist",
                edge.target
            )));
        }

        // Add to outgoing edges of source
        self.outgoing_edges
            .entry(edge.source.clone())
            .or_default()
            .push(edge.clone());

        // Add to incoming edges of target
        self.incoming_edges
            .entry(edge.target.clone())
            .or_default()
            .push(edge);

        Ok(())
    }

    /// Get successors (outgoing neighbors) of a node
    pub fn successors(&self, node_id: &str) -> Vec<String> {
        self.outgoing_edges
            .get(node_id)
            .map(|edges| edges.iter().map(|e| e.target.clone()).collect())
            .unwrap_or_default()
    }

    /// Get predecessors (incoming neighbors) of a node
    pub fn predecessors(&self, node_id: &str) -> Vec<String> {
        self.incoming_edges
            .get(node_id)
            .map(|edges| edges.iter().map(|e| e.source.clone()).collect())
            .unwrap_or_default()
    }

    /// Get node by ID
    pub fn get_node(&self, node_id: &str) -> Option<&GraphNode> {
        self.nodes.get(node_id)
    }

    /// Get outgoing edges from a node
    pub fn get_outgoing_edges(&self, node_id: &str) -> Vec<&GraphEdge> {
        self.outgoing_edges
            .get(node_id)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }

    /// Get incoming edges to a node
    pub fn get_incoming_edges(&self, node_id: &str) -> Vec<&GraphEdge> {
        self.incoming_edges
            .get(node_id)
            .map(|edges| edges.iter().collect())
            .unwrap_or_default()
    }

    /// Get all node IDs
    pub fn node_ids(&self) -> Vec<String> {
        self.nodes.keys().cloned().collect()
    }

    /// Check if node exists
    pub fn contains_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// Context extractor for generating rich context around nodes
pub struct ContextExtractor {
    graph: Graph,
    config: GraphTraversal,
}

impl ContextExtractor {
    pub fn new(graph: Graph, config: GraphTraversal) -> Self {
        Self { graph, config }
    }

    /// Extract context for a given node using graph traversal
    pub fn extract_context(&self, node_id: &str) -> Result<String> {
        if !self.graph.contains_node(node_id) {
            return Err(KgConstructionError::GraphError(format!(
                "Node {} not found in graph",
                node_id
            )));
        }

        let mut context_parts = Vec::new();

        // Get predecessors (incoming relationships)
        let predecessors = self.graph.predecessors(node_id);
        if !predecessors.is_empty() {
            let selected_predecessors = self.select_random_neighbors(&predecessors, 1);
            for pred_id in selected_predecessors {
                if let Some(pred_node) = self.graph.get_node(&pred_id) {
                    // Find the relation from predecessor to current node
                    let edges = self.graph.get_outgoing_edges(&pred_id);
                    for edge in edges {
                        if edge.target == node_id {
                            context_parts.push(format!("{} {}", pred_node.label, edge.relation));
                            break;
                        }
                    }
                }
            }
        }

        // Get successors (outgoing relationships)
        let successors = self.graph.successors(node_id);
        if !successors.is_empty() {
            let selected_successors = self.select_random_neighbors(&successors, 1);
            for succ_id in selected_successors {
                if let Some(succ_node) = self.graph.get_node(&succ_id) {
                    // Find the relation from current node to successor
                    let edges = self.graph.get_outgoing_edges(node_id);
                    for edge in edges {
                        if edge.target == succ_id {
                            context_parts.push(format!("{} {}", edge.relation, succ_node.label));
                            break;
                        }
                    }
                }
            }
        }

        Ok(context_parts.join(", "))
    }

    /// Extract multi-hop context using breadth-first search
    pub fn extract_multihop_context(&self, node_id: &str, max_hops: usize) -> Result<String> {
        if !self.graph.contains_node(node_id) {
            return Err(KgConstructionError::GraphError(format!(
                "Node {} not found in graph",
                node_id
            )));
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut context_parts = Vec::new();

        // Start BFS from the given node
        queue.push_back((node_id.to_string(), 0));
        visited.insert(node_id.to_string());

        while let Some((current_id, hop)) = queue.pop_front() {
            if hop >= max_hops {
                continue;
            }

            // Get neighbors
            let mut neighbors = self.graph.successors(&current_id);
            neighbors.extend(self.graph.predecessors(&current_id));

            for neighbor_id in neighbors {
                if !visited.contains(&neighbor_id) {
                    visited.insert(neighbor_id.clone());
                    queue.push_back((neighbor_id.clone(), hop + 1));

                    if let Some(neighbor_node) = self.graph.get_node(&neighbor_id) {
                        // Find the relation between current and neighbor
                        let relation = self.find_relation(&current_id, &neighbor_id);
                        context_parts.push(format!(
                            "{} -{}- {} (hop {})",
                            current_id, relation, neighbor_node.label, hop + 1
                        ));
                    }
                }
            }
        }

        Ok(context_parts.join(", "))
    }

    /// Find relation between two connected nodes
    fn find_relation(&self, from_id: &str, to_id: &str) -> String {
        // Check outgoing edges from 'from' to 'to'
        let outgoing = self.graph.get_outgoing_edges(from_id);
        for edge in outgoing {
            if edge.target == to_id {
                return edge.relation.clone();
            }
        }

        // Check incoming edges to 'from' from 'to'
        let incoming = self.graph.get_incoming_edges(from_id);
        for edge in incoming {
            if edge.source == to_id {
                return format!("{}^-1", edge.relation); // Inverse relation
            }
        }

        "unknown".to_string()
    }

    /// Select random neighbors (equivalent to Python's random.sample)
    fn select_random_neighbors(&self, neighbors: &[String], count: usize) -> Vec<String> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut rng = thread_rng();
        let actual_count = count.min(neighbors.len());

        if actual_count == 0 {
            return Vec::new();
        }

        neighbors.choose_multiple(&mut rng, actual_count).cloned().collect()
    }

    /// Extract context with type-aware sampling
    pub fn extract_typed_context(&self, node_id: &str, node_type: &str) -> Result<String> {
        let base_context = self.extract_context(node_id)?;

        // Add type-specific context enhancement
        match node_type.to_lowercase().as_str() {
            "entity" => {
                // For entities, focus on relationships and attributes
                let attrs = self.extract_node_attributes(node_id);
                if !attrs.is_empty() {
                    Ok(format!("{}, attributes: {}", base_context, attrs))
                } else {
                    Ok(base_context)
                }
            }
            "event" => {
                // For events, focus on temporal and causal relationships
                let temporal_context = self.extract_temporal_context(node_id);
                if !temporal_context.is_empty() {
                    Ok(format!("{}, temporal: {}", base_context, temporal_context))
                } else {
                    Ok(base_context)
                }
            }
            "relation" => {
                // For relations, focus on domain and range
                let domain_range = self.extract_domain_range_context(node_id);
                if !domain_range.is_empty() {
                    Ok(format!("{}, domain-range: {}", base_context, domain_range))
                } else {
                    Ok(base_context)
                }
            }
            _ => Ok(base_context),
        }
    }

    /// Extract node attributes as context
    fn extract_node_attributes(&self, node_id: &str) -> String {
        if let Some(node) = self.graph.get_node(node_id) {
            node.properties
                .iter()
                .map(|(k, v)| format!("{}:{}", k, v))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            String::new()
        }
    }

    /// Extract temporal context for events
    fn extract_temporal_context(&self, node_id: &str) -> String {
        let temporal_relations = ["before", "after", "during", "caused_by", "causes"];
        let mut temporal_parts = Vec::new();

        for edge in self.graph.get_outgoing_edges(node_id) {
            if temporal_relations.contains(&edge.relation.as_str()) {
                if let Some(target_node) = self.graph.get_node(&edge.target) {
                    temporal_parts.push(format!("{} {}", edge.relation, target_node.label));
                }
            }
        }

        for edge in self.graph.get_incoming_edges(node_id) {
            if temporal_relations.contains(&edge.relation.as_str()) {
                if let Some(source_node) = self.graph.get_node(&edge.source) {
                    temporal_parts.push(format!("{} {}", source_node.label, edge.relation));
                }
            }
        }

        temporal_parts.join(", ")
    }

    /// Extract domain and range context for relations
    fn extract_domain_range_context(&self, _node_id: &str) -> String {
        // This would be implemented based on specific domain/range modeling
        // For now, return empty string
        String::new()
    }
}

/// Compute hash ID from text (equivalent to Python version)
pub fn compute_hash_id(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Get node ID (equivalent to utils function)
pub fn get_node_id(node_name: &str) -> String {
    // Simple implementation - in practice this might involve more complex ID generation
    compute_hash_id(node_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();

        // Add nodes
        graph.add_node(GraphNode {
            id: "person1".to_string(),
            label: "John".to_string(),
            properties: HashMap::new(),
        });

        graph.add_node(GraphNode {
            id: "person2".to_string(),
            label: "Mary".to_string(),
            properties: HashMap::new(),
        });

        graph.add_node(GraphNode {
            id: "company1".to_string(),
            label: "TechCorp".to_string(),
            properties: HashMap::new(),
        });

        // Add edges
        graph.add_edge(GraphEdge {
            source: "person1".to_string(),
            target: "company1".to_string(),
            relation: "works_at".to_string(),
            properties: HashMap::new(),
        }).unwrap();

        graph.add_edge(GraphEdge {
            source: "person2".to_string(),
            target: "company1".to_string(),
            relation: "works_at".to_string(),
            properties: HashMap::new(),
        }).unwrap();

        graph
    }

    #[test]
    fn test_graph_creation() {
        let graph = create_test_graph();
        assert_eq!(graph.node_ids().len(), 3);
        assert!(graph.contains_node("person1"));
        assert!(graph.contains_node("company1"));
    }

    #[test]
    fn test_graph_traversal() {
        let graph = create_test_graph();
        let successors = graph.successors("person1");
        assert_eq!(successors, vec!["company1"]);

        let predecessors = graph.predecessors("company1");
        assert!(predecessors.contains(&"person1".to_string()));
        assert!(predecessors.contains(&"person2".to_string()));
    }

    #[test]
    fn test_context_extraction() {
        let graph = create_test_graph();
        let config = GraphTraversal::default();
        let extractor = ContextExtractor::new(graph, config);

        let context = extractor.extract_context("person1").unwrap();
        assert!(!context.is_empty());
    }

    #[test]
    fn test_compute_hash_id() {
        let hash1 = compute_hash_id("test");
        let hash2 = compute_hash_id("test");
        assert_eq!(hash1, hash2);

        let hash3 = compute_hash_id("different");
        assert_ne!(hash1, hash3);
    }
}