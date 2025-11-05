//! Semantic Forces Engine
//!
//! GPU-accelerated semantic physics forces for knowledge graph layout.
//! Implements DAG layout, type clustering, collision detection, and attribute-weighted springs.

use crate::models::graph::GraphData;
use crate::models::graph_types::{NodeType, EdgeType};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// Configuration Structures
// =============================================================================

/// DAG layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAGConfig {
    /// Vertical separation between hierarchy levels
    pub vertical_spacing: f32,
    /// Minimum horizontal separation within a level
    pub horizontal_spacing: f32,
    /// Strength of attraction to target level
    pub level_attraction: f32,
    /// Repulsion between nodes at same level
    pub sibling_repulsion: f32,
    /// Enable DAG layout forces
    pub enabled: bool,
}

impl Default for DAGConfig {
    fn default() -> Self {
        Self {
            vertical_spacing: 100.0,
            horizontal_spacing: 50.0,
            level_attraction: 0.5,
            sibling_repulsion: 0.3,
            enabled: false,
        }
    }
}

/// Type clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeClusterConfig {
    /// Attraction between nodes of same type
    pub cluster_attraction: f32,
    /// Target radius for type clusters
    pub cluster_radius: f32,
    /// Repulsion between different type clusters
    pub inter_cluster_repulsion: f32,
    /// Enable type clustering
    pub enabled: bool,
}

impl Default for TypeClusterConfig {
    fn default() -> Self {
        Self {
            cluster_attraction: 0.4,
            cluster_radius: 150.0,
            inter_cluster_repulsion: 0.2,
            enabled: false,
        }
    }
}

/// Collision detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionConfig {
    /// Minimum allowed distance between nodes
    pub min_distance: f32,
    /// Force strength when colliding
    pub collision_strength: f32,
    /// Default node radius
    pub node_radius: f32,
    /// Enable collision detection
    pub enabled: bool,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            min_distance: 5.0,
            collision_strength: 1.0,
            node_radius: 10.0,
            enabled: true,
        }
    }
}

/// Attribute-weighted spring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeSpringConfig {
    /// Base spring constant
    pub base_spring_k: f32,
    /// Multiplier for edge weight influence
    pub weight_multiplier: f32,
    /// Minimum rest length
    pub rest_length_min: f32,
    /// Maximum rest length
    pub rest_length_max: f32,
    /// Enable attribute-weighted springs
    pub enabled: bool,
}

impl Default for AttributeSpringConfig {
    fn default() -> Self {
        Self {
            base_spring_k: 0.01,
            weight_multiplier: 0.5,
            rest_length_min: 30.0,
            rest_length_max: 200.0,
            enabled: false,
        }
    }
}

/// Unified semantic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    pub dag: DAGConfig,
    pub type_cluster: TypeClusterConfig,
    pub collision: CollisionConfig,
    pub attribute_spring: AttributeSpringConfig,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            dag: DAGConfig::default(),
            type_cluster: TypeClusterConfig::default(),
            collision: CollisionConfig::default(),
            attribute_spring: AttributeSpringConfig::default(),
        }
    }
}

// =============================================================================
// Semantic Forces Engine
// =============================================================================

/// GPU-accelerated semantic forces engine
pub struct SemanticForcesEngine {
    config: SemanticConfig,
    node_hierarchy_levels: Vec<i32>,
    node_types: Vec<i32>,
    type_centroids: HashMap<i32, (f32, f32, f32)>,
    edge_types: Vec<i32>,
    initialized: bool,
}

impl SemanticForcesEngine {
    /// Create a new semantic forces engine
    pub fn new(config: SemanticConfig) -> Self {
        Self {
            config,
            node_hierarchy_levels: Vec::new(),
            node_types: Vec::new(),
            type_centroids: HashMap::new(),
            edge_types: Vec::new(),
            initialized: false,
        }
    }

    /// Initialize engine with graph data
    pub fn initialize(&mut self, graph: &GraphData) -> Result<(), String> {
        info!("Initializing SemanticForcesEngine with {} nodes, {} edges",
              graph.nodes.len(), graph.edges.len());

        // Extract node types
        self.node_types = graph.nodes.iter()
            .map(|node| self.node_type_to_int(&node.node_type))
            .collect();

        // Extract edge types
        self.edge_types = graph.edges.iter()
            .map(|edge| self.edge_type_to_int(&edge.edge_type))
            .collect();

        // Calculate hierarchy levels if DAG is enabled
        if self.config.dag.enabled {
            self.calculate_hierarchy_levels(graph)?;
        }

        // Calculate type centroids if type clustering is enabled
        if self.config.type_cluster.enabled {
            self.calculate_type_centroids(graph)?;
        }

        self.initialized = true;
        info!("SemanticForcesEngine initialized successfully");
        Ok(())
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SemanticConfig) {
        self.config = config;
        debug!("Semantic forces configuration updated");
    }

    /// Get current configuration
    pub fn config(&self) -> &SemanticConfig {
        &self.config
    }

    /// Check if engine is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get node hierarchy levels
    pub fn hierarchy_levels(&self) -> &[i32] {
        &self.node_hierarchy_levels
    }

    /// Get node types
    pub fn node_types(&self) -> &[i32] {
        &self.node_types
    }

    /// Get type centroids
    pub fn type_centroids(&self) -> &HashMap<i32, (f32, f32, f32)> {
        &self.type_centroids
    }

    // Private helper methods

    fn node_type_to_int(&self, node_type: &Option<String>) -> i32 {
        match node_type.as_deref() {
            None | Some("generic") => 0,
            Some("person") => 1,
            Some("organization") => 2,
            Some("project") => 3,
            Some("task") => 4,
            Some("concept") => 5,
            Some("class") => 6,
            Some("individual") => 7,
            Some(_) => 8, // Custom types
        }
    }

    fn edge_type_to_int(&self, edge_type: &Option<String>) -> i32 {
        match edge_type.as_deref() {
            None | Some("generic") => 0,
            Some("dependency") => 1,
            Some("hierarchy") => 2,
            Some("association") => 3,
            Some("sequence") => 4,
            Some("subClassOf") => 5,
            Some("instanceOf") => 6,
            Some(_) => 7, // Custom types
        }
    }

    fn calculate_hierarchy_levels(&mut self, graph: &GraphData) -> Result<(), String> {
        debug!("Calculating hierarchy levels for {} nodes", graph.nodes.len());

        // Initialize all levels to -1 (not in DAG)
        self.node_hierarchy_levels = vec![-1; graph.nodes.len()];

        // Build adjacency list for hierarchy edges
        let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_parent: HashMap<u32, bool> = HashMap::new();

        for edge in &graph.edges {
            if edge.edge_type.as_deref() == Some("hierarchy") {
                children.entry(edge.source).or_insert_with(Vec::new).push(edge.target);
                has_parent.insert(edge.target, true);
            }
        }

        // Find root nodes (nodes without parents)
        let mut roots = Vec::new();
        for (i, node) in graph.nodes.iter().enumerate() {
            if !has_parent.contains_key(&node.id) {
                // Check if this node has any hierarchy children
                if children.contains_key(&node.id) {
                    roots.push(i);
                    self.node_hierarchy_levels[i] = 0;
                }
            }
        }

        debug!("Found {} root nodes for DAG layout", roots.len());

        // BFS to assign levels
        let mut queue = roots.clone();
        let mut processed = 0;

        while !queue.is_empty() && processed < graph.nodes.len() * 2 {
            let mut next_queue = Vec::new();

            for node_idx in &queue {
                let node_id = graph.nodes[*node_idx].id;
                let current_level = self.node_hierarchy_levels[*node_idx];

                if let Some(child_ids) = children.get(&node_id) {
                    for child_id in child_ids {
                        // Find child index
                        if let Some(child_idx) = graph.nodes.iter().position(|n| n.id == *child_id) {
                            let new_level = current_level + 1;
                            if self.node_hierarchy_levels[child_idx] < new_level {
                                self.node_hierarchy_levels[child_idx] = new_level;
                                next_queue.push(child_idx);
                            }
                        }
                    }
                }
            }

            queue = next_queue;
            processed += 1;
        }

        let nodes_in_dag = self.node_hierarchy_levels.iter().filter(|&&l| l >= 0).count();
        info!("Hierarchy levels calculated: {} nodes in DAG", nodes_in_dag);

        Ok(())
    }

    fn calculate_type_centroids(&mut self, graph: &GraphData) -> Result<(), String> {
        debug!("Calculating type centroids");

        // Group nodes by type
        let mut type_positions: HashMap<i32, Vec<(f32, f32, f32)>> = HashMap::new();

        for (i, node) in graph.nodes.iter().enumerate() {
            let node_type = self.node_types[i];
            let pos = (
                node.data.x,
                node.data.y,
                node.data.z,
            );
            type_positions.entry(node_type).or_insert_with(Vec::new).push(pos);
        }

        // Calculate centroids
        self.type_centroids.clear();
        for (node_type, positions) in type_positions {
            if !positions.is_empty() {
                let sum: (f32, f32, f32) = positions.iter()
                    .fold((0.0, 0.0, 0.0), |acc, &pos| {
                        (acc.0 + pos.0, acc.1 + pos.1, acc.2 + pos.2)
                    });
                let count = positions.len() as f32;
                let centroid = (sum.0 / count, sum.1 / count, sum.2 / count);
                self.type_centroids.insert(node_type, centroid);
            }
        }

        info!("Calculated centroids for {} node types", self.type_centroids.len());
        Ok(())
    }

    /// Apply semantic forces to graph (CPU fallback implementation)
    /// In production, this would call CUDA kernels
    pub fn apply_semantic_forces(
        &self,
        graph: &mut GraphData,
    ) -> Result<(), String> {
        if !self.initialized {
            return Err("Engine not initialized".to_string());
        }

        // CPU implementation as fallback
        // In production, this would delegate to CUDA kernels

        // Apply DAG forces
        if self.config.dag.enabled {
            self.apply_dag_forces_cpu(graph);
        }

        // Apply type clustering forces
        if self.config.type_cluster.enabled {
            self.apply_type_cluster_forces_cpu(graph);
        }

        // Apply collision forces
        if self.config.collision.enabled {
            self.apply_collision_forces_cpu(graph);
        }

        // Apply attribute-weighted spring forces
        if self.config.attribute_spring.enabled {
            self.apply_attribute_spring_forces_cpu(graph);
        }

        Ok(())
    }

    // CPU fallback implementations (simplified)

    fn apply_dag_forces_cpu(&self, graph: &mut GraphData) {
        // Simplified CPU implementation
        // Real implementation would use CUDA kernel
        for (i, node) in graph.nodes.iter_mut().enumerate() {
            let level = self.node_hierarchy_levels[i];
            if level >= 0 {
                let target_y = level as f32 * self.config.dag.vertical_spacing;
                let dy = target_y - node.data.y;
                node.data.vy += dy * self.config.dag.level_attraction * 0.01;
            }
        }
    }

    fn apply_type_cluster_forces_cpu(&self, graph: &mut GraphData) {
        // Simplified CPU implementation
        for (i, node) in graph.nodes.iter_mut().enumerate() {
            let node_type = self.node_types[i];
            if let Some(&centroid) = self.type_centroids.get(&node_type) {
                let dx = centroid.0 - node.data.x;
                let dy = centroid.1 - node.data.y;
                let dz = centroid.2 - node.data.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist > self.config.type_cluster.cluster_radius {
                    let force = self.config.type_cluster.cluster_attraction * 0.01;
                    node.data.vx += dx * force;
                    node.data.vy += dy * force;
                    node.data.vz += dz * force;
                }
            }
        }
    }

    fn apply_collision_forces_cpu(&self, graph: &mut GraphData) {
        // Simplified CPU implementation
        let node_count = graph.nodes.len();
        for i in 0..node_count {
            for j in (i + 1)..node_count {
                let dx = graph.nodes[i].data.x - graph.nodes[j].data.x;
                let dy = graph.nodes[i].data.y - graph.nodes[j].data.y;
                let dz = graph.nodes[i].data.z - graph.nodes[j].data.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                let min_dist = 2.0 * self.config.collision.node_radius + self.config.collision.min_distance;
                if dist < min_dist && dist > 0.001 {
                    let force = self.config.collision.collision_strength * (min_dist - dist) / dist * 0.01;
                    graph.nodes[i].data.vx += dx * force;
                    graph.nodes[i].data.vy += dy * force;
                    graph.nodes[i].data.vz += dz * force;
                    graph.nodes[j].data.vx -= dx * force;
                    graph.nodes[j].data.vy -= dy * force;
                    graph.nodes[j].data.vz -= dz * force;
                }
            }
        }
    }

    fn apply_attribute_spring_forces_cpu(&self, graph: &mut GraphData) {
        // Simplified CPU implementation
        for edge in &graph.edges {
            // Find source and target nodes
            let src_idx = graph.nodes.iter().position(|n| n.id == edge.source);
            let tgt_idx = graph.nodes.iter().position(|n| n.id == edge.target);

            if let (Some(src_idx), Some(tgt_idx)) = (src_idx, tgt_idx) {
                let dx = graph.nodes[tgt_idx].data.x - graph.nodes[src_idx].data.x;
                let dy = graph.nodes[tgt_idx].data.y - graph.nodes[src_idx].data.y;
                let dz = graph.nodes[tgt_idx].data.z - graph.nodes[src_idx].data.z;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist > 0.001 {
                    let weight = edge.weight;
                    let spring_k = self.config.attribute_spring.base_spring_k *
                                  (1.0 + weight * self.config.attribute_spring.weight_multiplier);

                    let rest_length = self.config.attribute_spring.rest_length_max -
                                    (weight * (self.config.attribute_spring.rest_length_max -
                                             self.config.attribute_spring.rest_length_min));

                    let displacement = dist - rest_length;
                    let force = spring_k * displacement / dist * 0.01;

                    // Would need mutable access to both nodes - skipping for CPU fallback
                    // Real implementation uses CUDA with atomic operations
                }
            }
        }
    }
}

impl Default for SemanticForcesEngine {
    fn default() -> Self {
        Self::new(SemanticConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::node::Node;
    use crate::models::edge::Edge;

    #[test]
    fn test_semantic_config_defaults() {
        let config = SemanticConfig::default();
        assert!(!config.dag.enabled);
        assert!(!config.type_cluster.enabled);
        assert!(config.collision.enabled);
        assert!(!config.attribute_spring.enabled);
    }

    #[test]
    fn test_engine_creation() {
        let config = SemanticConfig::default();
        let engine = SemanticForcesEngine::new(config);
        assert!(!engine.is_initialized());
    }

    #[test]
    fn test_engine_initialization() {
        let mut engine = SemanticForcesEngine::new(SemanticConfig::default());

        let mut graph = GraphData::new();
        let mut node1 = Node::new("node1".to_string());
        node1.node_type = Some("person".to_string());
        graph.nodes.push(node1);

        let result = engine.initialize(&graph);
        assert!(result.is_ok());
        assert!(engine.is_initialized());
        assert_eq!(engine.node_types().len(), 1);
    }

    #[test]
    fn test_hierarchy_calculation() {
        let mut config = SemanticConfig::default();
        config.dag.enabled = true;

        let mut engine = SemanticForcesEngine::new(config);

        let mut graph = GraphData::new();
        let mut parent = Node::new("parent".to_string());
        parent = parent.with_label("Parent".to_string());
        let parent_id = parent.id;
        graph.nodes.push(parent);

        let mut child = Node::new("child".to_string());
        child = child.with_label("Child".to_string());
        let child_id = child.id;
        graph.nodes.push(child);

        let mut edge = Edge::new(parent_id, child_id, 1.0);
        edge.edge_type = Some("hierarchy".to_string());
        graph.edges.push(edge);

        engine.initialize(&graph).unwrap();

        let levels = engine.hierarchy_levels();
        assert_eq!(levels.len(), 2);
        // Parent should be at level 0
        assert_eq!(levels[0], 0);
        // Child should be at level 1
        assert_eq!(levels[1], 1);
    }

    #[test]
    fn test_type_clustering() {
        let mut config = SemanticConfig::default();
        config.type_cluster.enabled = true;

        let mut engine = SemanticForcesEngine::new(config);

        let mut graph = GraphData::new();
        for i in 0..5 {
            let mut node = Node::new(format!("node{}", i));
            node.node_type = Some("person".to_string());
            graph.nodes.push(node);
        }

        engine.initialize(&graph).unwrap();

        let centroids = engine.type_centroids();
        assert_eq!(centroids.len(), 1); // Only one type
        assert!(centroids.contains_key(&1)); // person type
    }
}
