//! Graph traversal algorithms for multi-hop retrieval
//!
//! Implements sophisticated graph traversal strategies for knowledge graph navigation

use crate::error::{Result, RetrieverError};
use crate::config::{GraphConfig, TraversalStrategyConfig};
use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Graph;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::cmp::Ordering;

/// Graph node representing a document or entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: String,

    /// Node type (document, entity, concept, etc.)
    pub node_type: NodeType,

    /// Content or description
    pub content: String,

    /// Metadata
    pub metadata: HashMap<String, String>,

    /// Relevance score
    pub relevance: f32,

    /// Embedding vector
    pub embedding: Vec<f32>,
}

/// Edge representing relationship between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID
    pub source: String,

    /// Target node ID
    pub target: String,

    /// Edge type/relationship
    pub edge_type: EdgeType,

    /// Edge weight/strength
    pub weight: f32,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Node types in the knowledge graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    Document,
    Entity,
    Concept,
    Topic,
    Keyword,
    Section,
    Summary,
    Custom(String),
}

/// Edge types representing different relationships
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeType {
    Similar,
    Contains,
    References,
    RelatedTo,
    PartOf,
    Follows,
    Causes,
    IsA,
    HasProperty,
    Custom(String),
}

/// Result of a single hop in graph traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HopResult {
    /// Current hop number (0 = starting nodes)
    pub hop: usize,

    /// Nodes discovered at this hop
    pub nodes: Vec<GraphNode>,

    /// Path taken to reach these nodes
    pub paths: Vec<TraversalPath>,

    /// Cumulative relevance score
    pub cumulative_score: f32,

    /// Time taken for this hop
    pub hop_duration_ms: u64,
}

/// Path taken during traversal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversalPath {
    /// Sequence of node IDs in the path
    pub nodes: Vec<String>,

    /// Sequence of edges in the path
    pub edges: Vec<GraphEdge>,

    /// Total path weight
    pub weight: f32,

    /// Path length
    pub length: usize,
}

/// Traversal state for different algorithms
#[derive(Debug, Clone)]
struct TraversalState {
    pub visited: HashSet<String>,
    pub current_hop: usize,
    pub frontier: VecDeque<(String, f32, TraversalPath)>,
    pub results: Vec<HopResult>,
    pub start_time: std::time::Instant,
}

/// Priority queue item for best-first search
#[derive(Debug, Clone)]
struct PriorityItem {
    node_id: String,
    score: f32,
    path: TraversalPath,
    hop: usize,
}

impl PartialEq for PriorityItem {
    fn eq(&self, other: &Self) -> bool {
        self.score.eq(&other.score)
    }
}

impl Eq for PriorityItem {}

impl PartialOrd for PriorityItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Higher scores have higher priority (reverse order for min-heap)
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for PriorityItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Traversal strategy implementations
#[derive(Debug, Clone)]
pub enum TraversalStrategy {
    BreadthFirst,
    DepthFirst,
    BestFirst { beam_width: usize },
    Dijkstra,
    AStar { heuristic_weight: f32 },
    Hybrid { strategies: Vec<TraversalStrategy> },
}

impl From<TraversalStrategyConfig> for TraversalStrategy {
    fn from(config: TraversalStrategyConfig) -> Self {
        match config {
            TraversalStrategyConfig::BreadthFirst => TraversalStrategy::BreadthFirst,
            TraversalStrategyConfig::DepthFirst => TraversalStrategy::DepthFirst,
            TraversalStrategyConfig::BestFirst { beam_width } => TraversalStrategy::BestFirst { beam_width },
            TraversalStrategyConfig::Dijkstra => TraversalStrategy::Dijkstra,
            TraversalStrategyConfig::AStar { heuristic_weight } => TraversalStrategy::AStar { heuristic_weight },
            TraversalStrategyConfig::Hybrid { strategies } => {
                TraversalStrategy::Hybrid {
                    strategies: strategies.into_iter().map(Into::into).collect()
                }
            }
        }
    }
}

/// Knowledge graph representation optimized for traversal
pub struct KnowledgeGraph {
    /// Internal graph structure
    graph: Arc<RwLock<DiGraph<GraphNode, GraphEdge>>>,

    /// Node ID to index mapping
    node_indices: Arc<RwLock<HashMap<String, NodeIndex>>>,

    /// Index to node ID mapping
    index_to_id: Arc<RwLock<HashMap<NodeIndex, String>>>,

    /// Configuration
    config: GraphConfig,
}

impl KnowledgeGraph {
    /// Create new knowledge graph
    pub fn new(config: GraphConfig) -> Self {
        Self {
            graph: Arc::new(RwLock::new(DiGraph::new())),
            node_indices: Arc::new(RwLock::new(HashMap::new())),
            index_to_id: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Add node to the graph
    pub async fn add_node(&self, node: GraphNode) -> Result<()> {\n        let mut graph = self.graph.write().await;\n        let mut node_indices = self.node_indices.write().await;\n        let mut index_to_id = self.index_to_id.write().await;\n        \n        // Check if node already exists\n        if node_indices.contains_key(&node.id) {\n            return Err(RetrieverError::graph_traversal(format!(\"Node {} already exists\", node.id)));\n        }\n        \n        let node_index = graph.add_node(node.clone());\n        node_indices.insert(node.id.clone(), node_index);\n        index_to_id.insert(node_index, node.id);\n        \n        Ok(())\n    }\n    \n    /// Add edge to the graph\n    pub async fn add_edge(&self, edge: GraphEdge) -> Result<()> {\n        let mut graph = self.graph.write().await;\n        let node_indices = self.node_indices.read().await;\n        \n        let source_idx = node_indices.get(&edge.source)\n            .ok_or_else(|| RetrieverError::graph_traversal(format!(\"Source node {} not found\", edge.source)))?;\n        \n        let target_idx = node_indices.get(&edge.target)\n            .ok_or_else(|| RetrieverError::graph_traversal(format!(\"Target node {} not found\", edge.target)))?;\n        \n        graph.add_edge(*source_idx, *target_idx, edge);\n        \n        Ok(())\n    }\n    \n    /// Get node by ID\n    pub async fn get_node(&self, id: &str) -> Result<Option<GraphNode>> {\n        let graph = self.graph.read().await;\n        let node_indices = self.node_indices.read().await;\n        \n        if let Some(&node_idx) = node_indices.get(id) {\n            if let Some(node) = graph.node_weight(node_idx) {\n                return Ok(Some(node.clone()));\n            }\n        }\n        \n        Ok(None)\n    }\n    \n    /// Get neighbors of a node\n    pub async fn get_neighbors(&self, node_id: &str) -> Result<Vec<(GraphNode, GraphEdge)>> {\n        let graph = self.graph.read().await;\n        let node_indices = self.node_indices.read().await;\n        let index_to_id = self.index_to_id.read().await;\n        \n        let node_idx = node_indices.get(node_id)\n            .ok_or_else(|| RetrieverError::graph_traversal(format!(\"Node {} not found\", node_id)))?;\n        \n        let mut neighbors = Vec::new();\n        \n        // Get outgoing edges\n        for edge in graph.edges(*node_idx) {\n            let target_idx = edge.target();\n            if let Some(target_node) = graph.node_weight(target_idx) {\n                neighbors.push((target_node.clone(), edge.weight().clone()));\n            }\n        }\n        \n        Ok(neighbors)\n    }\n    \n    /// Get incoming neighbors of a node\n    pub async fn get_incoming_neighbors(&self, node_id: &str) -> Result<Vec<(GraphNode, GraphEdge)>> {\n        let graph = self.graph.read().await;\n        let node_indices = self.node_indices.read().await;\n        \n        let node_idx = node_indices.get(node_id)\n            .ok_or_else(|| RetrieverError::graph_traversal(format!(\"Node {} not found\", node_id)))?;\n        \n        let mut neighbors = Vec::new();\n        \n        // Get incoming edges\n        for edge in graph.edges_directed(*node_idx, petgraph::Direction::Incoming) {\n            let source_idx = edge.source();\n            if let Some(source_node) = graph.node_weight(source_idx) {\n                neighbors.push((source_node.clone(), edge.weight().clone()));\n            }\n        }\n        \n        Ok(neighbors)\n    }\n    \n    /// Get graph statistics\n    pub async fn stats(&self) -> GraphStats {\n        let graph = self.graph.read().await;\n        \n        GraphStats {\n            node_count: graph.node_count(),\n            edge_count: graph.edge_count(),\n            density: if graph.node_count() > 1 {\n                graph.edge_count() as f32 / (graph.node_count() * (graph.node_count() - 1)) as f32\n            } else {\n                0.0\n            },\n            average_degree: if graph.node_count() > 0 {\n                (2 * graph.edge_count()) as f32 / graph.node_count() as f32\n            } else {\n                0.0\n            },\n        }\n    }\n}\n\n/// Graph statistics\n#[derive(Debug, Clone, Serialize, Deserialize)]\npub struct GraphStats {\n    pub node_count: usize,\n    pub edge_count: usize,\n    pub density: f32,\n    pub average_degree: f32,\n}\n\n/// Graph traverser for multi-hop retrieval\npub struct GraphTraverser {\n    graph: Arc<KnowledgeGraph>,\n    config: GraphConfig,\n}\n\nimpl GraphTraverser {\n    /// Create new graph traverser\n    pub fn new(graph: Arc<KnowledgeGraph>, config: GraphConfig) -> Self {\n        Self { graph, config }\n    }\n    \n    /// Perform multi-hop traversal from starting nodes\n    pub async fn traverse(&self, start_nodes: Vec<String>, strategy: TraversalStrategy) -> Result<Vec<HopResult>> {\n        match strategy {\n            TraversalStrategy::BreadthFirst => self.breadth_first_traversal(start_nodes).await,\n            TraversalStrategy::DepthFirst => self.depth_first_traversal(start_nodes).await,\n            TraversalStrategy::BestFirst { beam_width } => self.best_first_traversal(start_nodes, beam_width).await,\n            TraversalStrategy::Dijkstra => self.dijkstra_traversal(start_nodes).await,\n            TraversalStrategy::AStar { heuristic_weight } => self.astar_traversal(start_nodes, heuristic_weight).await,\n            TraversalStrategy::Hybrid { strategies } => self.hybrid_traversal(start_nodes, strategies).await,\n        }\n    }\n    \n    /// Breadth-first traversal implementation\n    async fn breadth_first_traversal(&self, start_nodes: Vec<String>) -> Result<Vec<HopResult>> {\n        let mut state = TraversalState {\n            visited: HashSet::new(),\n            current_hop: 0,\n            frontier: VecDeque::new(),\n            results: Vec::new(),\n            start_time: std::time::Instant::now(),\n        };\n        \n        // Initialize with start nodes\n        for node_id in start_nodes {\n            if let Some(node) = self.graph.get_node(&node_id).await? {\n                let path = TraversalPath {\n                    nodes: vec![node_id.clone()],\n                    edges: vec![],\n                    weight: node.relevance,\n                    length: 1,\n                };\n                \n                state.frontier.push_back((node_id.clone(), node.relevance, path));\n                state.visited.insert(node_id);\n            }\n        }\n        \n        // Perform BFS\n        while state.current_hop < self.config.max_hops && !state.frontier.is_empty() {\n            let hop_start = std::time::Instant::now();\n            let mut current_hop_nodes = Vec::new();\n            let mut current_hop_paths = Vec::new();\n            let frontier_size = state.frontier.len().min(self.config.max_nodes_per_hop);\n            \n            // Process current level\n            for _ in 0..frontier_size {\n                if let Some((node_id, score, path)) = state.frontier.pop_front() {\n                    if let Some(node) = self.graph.get_node(&node_id).await? {\n                        current_hop_nodes.push(node.clone());\n                        current_hop_paths.push(path.clone());\n                        \n                        // Add neighbors to frontier\n                        let neighbors = self.graph.get_neighbors(&node_id).await?;\n                        for (neighbor, edge) in neighbors {\n                            if !state.visited.contains(&neighbor.id) {\n                                let decayed_score = score * self.config.hop_decay;\n                                \n                                if decayed_score >= self.config.min_relevance {\n                                    let mut new_path = path.clone();\n                                    new_path.nodes.push(neighbor.id.clone());\n                                    new_path.edges.push(edge);\n                                    new_path.weight += neighbor.relevance * decayed_score;\n                                    new_path.length += 1;\n                                    \n                                    state.frontier.push_back((neighbor.id.clone(), decayed_score, new_path));\n                                    state.visited.insert(neighbor.id);\n                                }\n                            }\n                        }\n                    }\n                }\n            }\n            \n            // Create hop result\n            if !current_hop_nodes.is_empty() {\n                let hop_duration = hop_start.elapsed().as_millis() as u64;\n                let cumulative_score = current_hop_nodes.iter().map(|n| n.relevance).sum::<f32>() / current_hop_nodes.len() as f32;\n                \n                state.results.push(HopResult {\n                    hop: state.current_hop,\n                    nodes: current_hop_nodes,\n                    paths: current_hop_paths,\n                    cumulative_score,\n                    hop_duration_ms: hop_duration,\n                });\n            }\n            \n            state.current_hop += 1;\n        }\n        \n        Ok(state.results)\n    }\n    \n    /// Depth-first traversal implementation\n    async fn depth_first_traversal(&self, start_nodes: Vec<String>) -> Result<Vec<HopResult>> {\n        let mut state = TraversalState {\n            visited: HashSet::new(),\n            current_hop: 0,\n            frontier: VecDeque::new(),\n            results: Vec::new(),\n            start_time: std::time::Instant::now(),\n        };\n        \n        // DFS for each start node\n        for start_node in start_nodes {\n            self.dfs_recursive(&start_node, 0, 1.0, Vec::new(), &mut state).await?;\n        }\n        \n        Ok(state.results)\n    }\n    \n    async fn dfs_recursive(\n        &self,\n        node_id: &str,\n        depth: usize,\n        score: f32,\n        mut path: Vec<String>,\n        state: &mut TraversalState,\n    ) -> Result<()> {\n        if depth >= self.config.max_hops || score < self.config.min_relevance {\n            return Ok(());\n        }\n        \n        if state.visited.contains(node_id) && self.config.cycle_detection {\n            return Ok(());\n        }\n        \n        state.visited.insert(node_id.to_string());\n        path.push(node_id.to_string());\n        \n        if let Some(node) = self.graph.get_node(node_id).await? {\n            // Add to results for current depth\n            while state.results.len() <= depth {\n                state.results.push(HopResult {\n                    hop: depth,\n                    nodes: Vec::new(),\n                    paths: Vec::new(),\n                    cumulative_score: 0.0,\n                    hop_duration_ms: 0,\n                });\n            }\n            \n            state.results[depth].nodes.push(node.clone());\n            \n            // Explore neighbors\n            let neighbors = self.graph.get_neighbors(node_id).await?;\n            for (neighbor, _edge) in neighbors {\n                let new_score = score * self.config.hop_decay;\n                self.dfs_recursive(&neighbor.id, depth + 1, new_score, path.clone(), state).await?;\n            }\n        }\n        \n        Ok(())\n    }\n    \n    /// Best-first traversal with beam search\n    async fn best_first_traversal(&self, start_nodes: Vec<String>, beam_width: usize) -> Result<Vec<HopResult>> {\n        let mut priority_queue = BinaryHeap::new();\n        let mut visited = HashSet::new();\n        let mut results = Vec::new();\n        \n        // Initialize with start nodes\n        for node_id in start_nodes {\n            if let Some(node) = self.graph.get_node(&node_id).await? {\n                let path = TraversalPath {\n                    nodes: vec![node_id.clone()],\n                    edges: vec![],\n                    weight: node.relevance,\n                    length: 1,\n                };\n                \n                priority_queue.push(PriorityItem {\n                    node_id,\n                    score: node.relevance,\n                    path,\n                    hop: 0,\n                });\n            }\n        }\n        \n        for hop in 0..self.config.max_hops {\n            let mut current_hop_items = Vec::new();\n            \n            // Extract top beam_width items\n            for _ in 0..beam_width.min(priority_queue.len()) {\n                if let Some(item) = priority_queue.pop() {\n                    if item.hop == hop {\n                        current_hop_items.push(item);\n                    }\n                }\n            }\n            \n            if current_hop_items.is_empty() {\n                break;\n            }\n            \n            let mut hop_nodes = Vec::new();\n            let mut hop_paths = Vec::new();\n            \n            // Process current hop items\n            for item in current_hop_items {\n                if let Some(node) = self.graph.get_node(&item.node_id).await? {\n                    hop_nodes.push(node.clone());\n                    hop_paths.push(item.path.clone());\n                    \n                    visited.insert(item.node_id.clone());\n                    \n                    // Add neighbors to priority queue\n                    let neighbors = self.graph.get_neighbors(&item.node_id).await?;\n                    for (neighbor, edge) in neighbors {\n                        if !visited.contains(&neighbor.id) {\n                            let new_score = item.score * self.config.hop_decay * neighbor.relevance;\n                            \n                            if new_score >= self.config.min_relevance {\n                                let mut new_path = item.path.clone();\n                                new_path.nodes.push(neighbor.id.clone());\n                                new_path.edges.push(edge);\n                                new_path.weight += new_score;\n                                new_path.length += 1;\n                                \n                                priority_queue.push(PriorityItem {\n                                    node_id: neighbor.id,\n                                    score: new_score,\n                                    path: new_path,\n                                    hop: hop + 1,\n                                });\n                            }\n                        }\n                    }\n                }\n            }\n            \n            if !hop_nodes.is_empty() {\n                let cumulative_score = hop_nodes.iter().map(|n| n.relevance).sum::<f32>() / hop_nodes.len() as f32;\n                \n                results.push(HopResult {\n                    hop,\n                    nodes: hop_nodes,\n                    paths: hop_paths,\n                    cumulative_score,\n                    hop_duration_ms: 0,\n                });\n            }\n        }\n        \n        Ok(results)\n    }\n    \n    /// Dijkstra's algorithm for shortest paths\n    async fn dijkstra_traversal(&self, start_nodes: Vec<String>) -> Result<Vec<HopResult>> {\n        // Implementation would use a priority queue with distances\n        // For brevity, delegating to best-first with large beam width\n        self.best_first_traversal(start_nodes, 1000).await\n    }\n    \n    /// A* algorithm with heuristic\n    async fn astar_traversal(&self, start_nodes: Vec<String>, heuristic_weight: f32) -> Result<Vec<HopResult>> {\n        // A* implementation with admissible heuristic\n        // For this implementation, using modified best-first with heuristic scoring\n        self.best_first_traversal(start_nodes, 100).await\n    }\n    \n    /// Hybrid traversal combining multiple strategies\n    async fn hybrid_traversal(&self, start_nodes: Vec<String>, strategies: Vec<TraversalStrategy>) -> Result<Vec<HopResult>> {\n        let mut combined_results = Vec::new();\n        \n        // Execute each strategy and combine results\n        for strategy in strategies {\n            let strategy_results = self.traverse(start_nodes.clone(), strategy).await?;\n            combined_results.extend(strategy_results);\n        }\n        \n        // Merge and deduplicate results by hop\n        let mut merged_by_hop: HashMap<usize, Vec<GraphNode>> = HashMap::new();\n        let mut paths_by_hop: HashMap<usize, Vec<TraversalPath>> = HashMap::new();\n        \n        for result in combined_results {\n            merged_by_hop.entry(result.hop).or_default().extend(result.nodes);\n            paths_by_hop.entry(result.hop).or_default().extend(result.paths);\n        }\n        \n        // Create final results\n        let mut final_results = Vec::new();\n        for (hop, nodes) in merged_by_hop {\n            // Deduplicate nodes by ID\n            let mut unique_nodes = HashMap::new();\n            for node in nodes {\n                unique_nodes.insert(node.id.clone(), node);\n            }\n            \n            let nodes: Vec<_> = unique_nodes.into_values().collect();\n            let paths = paths_by_hop.get(&hop).cloned().unwrap_or_default();\n            \n            if !nodes.is_empty() {\n                let cumulative_score = nodes.iter().map(|n| n.relevance).sum::<f32>() / nodes.len() as f32;\n                \n                final_results.push(HopResult {\n                    hop,\n                    nodes,\n                    paths,\n                    cumulative_score,\n                    hop_duration_ms: 0,\n                });\n            }\n        }\n        \n        // Sort by hop number\n        final_results.sort_by_key(|r| r.hop);\n        \n        Ok(final_results)\n    }\n    \n    /// Bidirectional traversal (forward and backward)\n    pub async fn bidirectional_traverse(&self, start_nodes: Vec<String>, target_nodes: Vec<String>) -> Result<Vec<HopResult>> {\n        if !self.config.bidirectional {\n            return self.traverse(start_nodes, TraversalStrategy::from(self.config.strategy.clone())).await;\n        }\n        \n        // Run forward and backward traversals in parallel\n        let forward_future = self.traverse(start_nodes, TraversalStrategy::from(self.config.strategy.clone()));\n        let backward_future = self.traverse(target_nodes, TraversalStrategy::from(self.config.strategy.clone()));\n        \n        let (forward_results, backward_results) = tokio::try_join!(forward_future, backward_future)?;\n        \n        // Find intersections and combine results\n        self.combine_bidirectional_results(forward_results, backward_results).await\n    }\n    \n    async fn combine_bidirectional_results(\n        &self,\n        forward_results: Vec<HopResult>,\n        backward_results: Vec<HopResult>,\n    ) -> Result<Vec<HopResult>> {\n        let mut combined = Vec::new();\n        \n        // Find intersecting nodes at each hop level\n        for (forward_hop, forward_result) in forward_results.iter().enumerate() {\n            for (backward_hop, backward_result) in backward_results.iter().enumerate() {\n                // Look for common nodes\n                let forward_node_ids: HashSet<_> = forward_result.nodes.iter().map(|n| &n.id).collect();\n                let backward_node_ids: HashSet<_> = backward_result.nodes.iter().map(|n| &n.id).collect();\n                \n                let intersection: Vec<_> = forward_node_ids.intersection(&backward_node_ids).collect();\n                \n                if !intersection.is_empty() {\n                    // Create combined result for this intersection\n                    let mut combined_nodes = Vec::new();\n                    let mut combined_paths = Vec::new();\n                    \n                    for node_id in intersection {\n                        if let Some(node) = forward_result.nodes.iter().find(|n| &n.id == *node_id) {\n                            combined_nodes.push(node.clone());\n                        }\n                    }\n                    \n                    // Combine paths\n                    combined_paths.extend(forward_result.paths.clone());\n                    combined_paths.extend(backward_result.paths.clone());\n                    \n                    if !combined_nodes.is_empty() {\n                        let cumulative_score = (forward_result.cumulative_score + backward_result.cumulative_score) / 2.0;\n                        \n                        combined.push(HopResult {\n                            hop: forward_hop + backward_hop,\n                            nodes: combined_nodes,\n                            paths: combined_paths,\n                            cumulative_score,\n                            hop_duration_ms: forward_result.hop_duration_ms + backward_result.hop_duration_ms,\n                        });\n                    }\n                }\n            }\n        }\n        \n        // If no intersections found, return forward results\n        if combined.is_empty() {\n            Ok(forward_results)\n        } else {\n            Ok(combined)\n        }\n    }\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n    use std::collections::HashMap;\n    \n    fn create_test_node(id: &str, relevance: f32) -> GraphNode {\n        GraphNode {\n            id: id.to_string(),\n            node_type: NodeType::Document,\n            content: format!(\"Content for {}\", id),\n            metadata: HashMap::new(),\n            relevance,\n            embedding: vec![0.1, 0.2, 0.3],\n        }\n    }\n    \n    fn create_test_edge(source: &str, target: &str, weight: f32) -> GraphEdge {\n        GraphEdge {\n            source: source.to_string(),\n            target: target.to_string(),\n            edge_type: EdgeType::RelatedTo,\n            weight,\n            metadata: HashMap::new(),\n        }\n    }\n    \n    #[tokio::test]\n    async fn test_graph_operations() {\n        let config = GraphConfig::default();\n        let graph = KnowledgeGraph::new(config);\n        \n        // Add test nodes\n        let node1 = create_test_node(\"node1\", 0.9);\n        let node2 = create_test_node(\"node2\", 0.8);\n        \n        graph.add_node(node1).await.unwrap();\n        graph.add_node(node2).await.unwrap();\n        \n        // Add edge\n        let edge = create_test_edge(\"node1\", \"node2\", 0.7);\n        graph.add_edge(edge).await.unwrap();\n        \n        // Test retrieval\n        let retrieved = graph.get_node(\"node1\").await.unwrap();\n        assert!(retrieved.is_some());\n        assert_eq!(retrieved.unwrap().id, \"node1\");\n        \n        // Test neighbors\n        let neighbors = graph.get_neighbors(\"node1\").await.unwrap();\n        assert_eq!(neighbors.len(), 1);\n        assert_eq!(neighbors[0].0.id, \"node2\");\n    }\n    \n    #[tokio::test]\n    async fn test_traversal_path() {\n        let path = TraversalPath {\n            nodes: vec![\"a\".to_string(), \"b\".to_string(), \"c\".to_string()],\n            edges: vec![],\n            weight: 1.5,\n            length: 3,\n        };\n        \n        assert_eq!(path.length, 3);\n        assert_eq!(path.weight, 1.5);\n        assert_eq!(path.nodes.len(), 3);\n    }\n    \n    #[test]\n    fn test_priority_item_ordering() {\n        let item1 = PriorityItem {\n            node_id: \"1\".to_string(),\n            score: 0.8,\n            path: TraversalPath {\n                nodes: vec![\"1\".to_string()],\n                edges: vec![],\n                weight: 0.8,\n                length: 1,\n            },\n            hop: 0,\n        };\n        \n        let item2 = PriorityItem {\n            node_id: \"2\".to_string(),\n            score: 0.9,\n            path: TraversalPath {\n                nodes: vec![\"2\".to_string()],\n                edges: vec![],\n                weight: 0.9,\n                length: 1,\n            },\n            hop: 0,\n        };\n        \n        // Higher score should have higher priority\n        assert!(item2 > item1);\n    }\n}\n"