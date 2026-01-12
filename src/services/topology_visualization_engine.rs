//! Topology Visualization Engine
//!
//! This service provides advanced graph layout algorithms and topology visualization
//! for multi-agent systems and MCP server networks. It implements various layout
//! algorithms optimized for different network topologies and visualization requirements.

use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

use crate::services::agent_visualization_protocol::{Position, SwarmTopologyData};
use crate::types::Vec3Data;
use crate::utils::time;

#[derive(Debug, Clone)]
pub struct TopologyVisualizationEngine {
    
    pub layout_config: LayoutConfiguration,

    
    pub nodes: HashMap<String, TopologyNode>,

    
    pub edges: HashMap<String, TopologyEdge>,

    
    pub bounds: LayoutBounds,

    
    pub physics: PhysicsParameters,

    
    pub metrics: LayoutMetrics,

    
    pub layout_cache: HashMap<String, CachedLayout>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyNode {
    pub id: String,
    pub position: Position,
    pub velocity: Vec3Data,
    pub mass: f32,
    pub fixed: bool,
    pub group: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub connections: Vec<String>,
    pub importance: f32,
    pub last_updated: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyEdge {
    pub id: String,
    pub source: String,
    pub target: String,
    pub weight: f32,
    pub strength: f32,
    pub distance: f32,
    pub edge_type: EdgeType,
    pub metadata: HashMap<String, serde_json::Value>,
    pub bidirectional: bool,
    pub last_updated: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    
    Communication,
    
    Hierarchical,
    
    Collaboration,
    
    Dependency,
    
    DataFlow,
    
    ControlFlow,
    
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfiguration {
    pub algorithm: LayoutAlgorithm,
    pub iterations: u32,
    pub convergence_threshold: f32,
    pub adaptive_cooling: bool,
    pub multi_level: bool,
    pub quality_vs_speed: f32, 
    pub edge_bundling: bool,
    pub node_repulsion: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutAlgorithm {
    
    ForceDirected(ForceDirectedConfig),
    
    Hierarchical(HierarchicalConfig),
    
    Circular(CircularConfig),
    
    Grid(GridConfig),
    
    SpringElectrical(SpringElectricalConfig),
    
    StressMajorization(StressConfig),
    
    MultiLevel(MultiLevelConfig),
    
    Geographic(GeographicConfig),
    
    Custom(CustomLayoutConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceDirectedConfig {
    pub attraction_strength: f32,
    pub repulsion_strength: f32,
    pub optimal_distance: f32,
    pub damping: f32,
    pub max_velocity: f32,
    pub gravity: f32,
    pub center_gravity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalConfig {
    pub direction: HierarchicalDirection,
    pub layer_separation: f32,
    pub node_separation: f32,
    pub edge_separation: f32,
    pub rank_separation: f32,
    pub minimize_crossings: bool,
    pub align_nodes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchicalDirection {
    TopToBottom,
    BottomToTop,
    LeftToRight,
    RightToLeft,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircularConfig {
    pub radius: f32,
    pub start_angle: f32,
    pub clockwise: bool,
    pub group_separation: f32,
    pub concentric_layers: bool,
    pub layer_spacing: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub columns: Option<u32>,
    pub rows: Option<u32>,
    pub cell_width: f32,
    pub cell_height: f32,
    pub padding: f32,
    pub align_center: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpringElectricalConfig {
    pub spring_length: f32,
    pub spring_constant: f32,
    pub electrical_charge: f32,
    pub electrical_distance: f32,
    pub theta: f32, 
    pub use_barnes_hut: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressConfig {
    pub distance_matrix: bool,
    pub weighted_stress: bool,
    pub normalize_stress: bool,
    pub stress_majorization_iterations: u32,
    pub tolerance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLevelConfig {
    pub coarsening_ratio: f32,
    pub min_nodes_per_level: u32,
    pub max_levels: u32,
    pub base_algorithm: Box<LayoutAlgorithm>,
    pub refinement_iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicConfig {
    pub projection: MapProjection,
    pub bounds: GeographicBounds,
    pub scale_factor: f32,
    pub center_point: (f32, f32), 
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MapProjection {
    Mercator,
    Robinson,
    Orthographic,
    Stereographic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicBounds {
    pub min_lat: f32,
    pub max_lat: f32,
    pub min_lon: f32,
    pub max_lon: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomLayoutConfig {
    pub name: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub iterations: u32,
    pub use_physics: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub enforce_bounds: bool,
    pub aspect_ratio: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParameters {
    pub enabled: bool,
    pub time_step: f32,
    pub max_iterations: u32,
    pub convergence_threshold: f32,
    pub adaptive_time_step: bool,
    pub energy_threshold: f32,
    pub cooling_factor: f32,
    pub initial_temperature: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayoutMetrics {
    pub computation_time_ms: f64,
    pub iterations_performed: u32,
    pub convergence_achieved: bool,
    pub final_energy: f32,
    pub stress_value: f32,
    pub node_overlap_count: u32,
    pub edge_crossings: u32,
    pub layout_quality_score: f32,
    pub memory_usage_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedLayout {
    pub layout_hash: String,
    pub positions: HashMap<String, Position>,
    pub computation_time: f64,
    pub quality_score: f32,
    pub created_at: i64,
    pub access_count: u32,
}

impl Default for TopologyVisualizationEngine {
    fn default() -> Self {
        Self {
            layout_config: LayoutConfiguration::default(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
            bounds: LayoutBounds::default(),
            physics: PhysicsParameters::default(),
            metrics: LayoutMetrics::default(),
            layout_cache: HashMap::new(),
        }
    }
}

impl Default for LayoutConfiguration {
    fn default() -> Self {
        Self {
            algorithm: LayoutAlgorithm::ForceDirected(ForceDirectedConfig::default()),
            iterations: 1000,
            convergence_threshold: 0.01,
            adaptive_cooling: true,
            multi_level: false,
            quality_vs_speed: 0.5,
            edge_bundling: false,
            node_repulsion: true,
        }
    }
}

impl Default for ForceDirectedConfig {
    fn default() -> Self {
        Self {
            attraction_strength: 1.0,
            repulsion_strength: 1000.0,
            optimal_distance: 50.0,
            damping: 0.9,
            max_velocity: 10.0,
            gravity: 0.01,
            center_gravity: true,
        }
    }
}

impl Default for LayoutBounds {
    fn default() -> Self {
        Self {
            min_x: -1000.0,
            max_x: 1000.0,
            min_y: -1000.0,
            max_y: 1000.0,
            min_z: -100.0,
            max_z: 100.0,
            enforce_bounds: true,
            aspect_ratio: None,
        }
    }
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            enabled: true,
            time_step: 0.016, 
            max_iterations: 1000,
            convergence_threshold: 0.01,
            adaptive_time_step: true,
            energy_threshold: 0.001,
            cooling_factor: 0.95,
            initial_temperature: 100.0,
        }
    }
}

impl TopologyVisualizationEngine {
    
    pub fn new() -> Self {
        Self::default()
    }

    
    pub fn with_algorithm(algorithm: LayoutAlgorithm) -> Self {
        let mut engine = Self::new();
        engine.layout_config.algorithm = algorithm;
        engine
    }

    
    pub fn add_node(&mut self, node: TopologyNode) {
        debug!("Adding topology node: {}", node.id);
        self.nodes.insert(node.id.clone(), node);
        self.invalidate_cache();
    }

    
    pub fn add_edge(&mut self, edge: TopologyEdge) {
        debug!("Adding topology edge: {} -> {}", edge.source, edge.target);
        self.edges.insert(edge.id.clone(), edge);
        self.invalidate_cache();
    }

    
    pub fn remove_node(&mut self, node_id: &str) {
        if self.nodes.remove(node_id).is_some() {
            
            self.edges
                .retain(|_, edge| edge.source != node_id && edge.target != node_id);
            self.invalidate_cache();
            debug!("Removed topology node: {}", node_id);
        }
    }

    
    pub fn remove_edge(&mut self, edge_id: &str) {
        if self.edges.remove(edge_id).is_some() {
            self.invalidate_cache();
            debug!("Removed topology edge: {}", edge_id);
        }
    }

    
    pub fn update_node_position(&mut self, node_id: &str, position: Position) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.position = position;
            node.last_updated = time::timestamp_seconds();
        }
    }

    
    pub fn set_layout_algorithm(&mut self, algorithm: LayoutAlgorithm) {
        self.layout_config.algorithm = algorithm;
        self.invalidate_cache();
    }

    
    pub fn set_bounds(&mut self, bounds: LayoutBounds) {
        self.bounds = bounds;
        self.invalidate_cache();
    }

    
    pub fn compute_layout(&mut self) -> Result<HashMap<String, Position>, String> {
        let start_time = std::time::Instant::now();
        self.metrics = LayoutMetrics::default();

        
        let layout_hash = self.compute_layout_hash();
        if let Some(cached) = self.layout_cache.get_mut(&layout_hash) {
            cached.access_count += 1;
            info!("Using cached layout (hash: {})", layout_hash);
            return Ok(cached.positions.clone());
        }

        info!(
            "Computing layout with algorithm: {:?}",
            self.layout_config.algorithm
        );

        
        let algorithm = self.layout_config.algorithm.clone();
        let positions = match algorithm {
            LayoutAlgorithm::ForceDirected(config) => self.compute_force_directed_layout(&config),
            LayoutAlgorithm::Hierarchical(config) => self.compute_hierarchical_layout(&config),
            LayoutAlgorithm::Circular(config) => self.compute_circular_layout(&config),
            LayoutAlgorithm::Grid(config) => self.compute_grid_layout(&config),
            LayoutAlgorithm::SpringElectrical(config) => {
                self.compute_spring_electrical_layout(&config)
            }
            LayoutAlgorithm::StressMajorization(config) => {
                self.compute_stress_majorization_layout(&config)
            }
            LayoutAlgorithm::MultiLevel(config) => self.compute_multi_level_layout(&config),
            LayoutAlgorithm::Geographic(config) => self.compute_geographic_layout(&config),
            LayoutAlgorithm::Custom(config) => self.compute_custom_layout(&config),
        }?;

        
        self.metrics.computation_time_ms = start_time.elapsed().as_millis() as f64;
        self.metrics.layout_quality_score = self.compute_quality_score(&positions);

        
        let cached_layout = CachedLayout {
            layout_hash: layout_hash.clone(),
            positions: positions.clone(),
            computation_time: self.metrics.computation_time_ms,
            quality_score: self.metrics.layout_quality_score,
            created_at: time::timestamp_seconds(),
            access_count: 1,
        };
        self.layout_cache.insert(layout_hash, cached_layout);

        
        let final_positions = if self.bounds.enforce_bounds {
            self.apply_bounds(positions)
        } else {
            positions
        };

        
        for (node_id, position) in &final_positions {
            self.update_node_position(node_id, *position);
        }

        info!(
            "Layout computation completed in {:.2}ms with quality score: {:.3}",
            self.metrics.computation_time_ms, self.metrics.layout_quality_score
        );

        Ok(final_positions)
    }

    
    fn compute_force_directed_layout(
        &mut self,
        config: &ForceDirectedConfig,
    ) -> Result<HashMap<String, Position>, String> {
        let mut positions = HashMap::new();
        let mut velocities = HashMap::new();

        
        for (node_id, node) in &self.nodes {
            positions.insert(node_id.clone(), node.position);
            velocities.insert(node_id.clone(), node.velocity);
        }

        let k = config.optimal_distance;
        let area =
            (self.bounds.max_x - self.bounds.min_x) * (self.bounds.max_y - self.bounds.min_y);
        let temperature = config.optimal_distance * (area / self.nodes.len() as f32).sqrt();

        for iteration in 0..self.layout_config.iterations {
            let mut forces: HashMap<String, Vec3Data> = HashMap::new();
            let node_ids: Vec<String> = self.nodes.keys().cloned().collect();

            
            for i in 0..node_ids.len() {
                for j in (i + 1)..node_ids.len() {
                    let id1 = &node_ids[i];
                    let id2 = &node_ids[j];

                    if let (Some(pos1), Some(pos2)) = (positions.get(id1), positions.get(id2)) {
                        let dx = pos1.x - pos2.x;
                        let dy = pos1.y - pos2.y;
                        let distance = (dx * dx + dy * dy).sqrt().max(0.01);

                        let repulsive_force = config.repulsion_strength * k * k / distance;
                        let fx = (dx / distance) * repulsive_force;
                        let fy = (dy / distance) * repulsive_force;

                        forces.entry(id1.clone()).or_insert_with(Vec3Data::zero).x += fx;
                        forces.entry(id1.clone()).or_insert_with(Vec3Data::zero).y += fy;
                        forces.entry(id2.clone()).or_insert_with(Vec3Data::zero).x -= fx;
                        forces.entry(id2.clone()).or_insert_with(Vec3Data::zero).y -= fy;
                    }
                }
            }

            
            for edge in self.edges.values() {
                if let (Some(pos1), Some(pos2)) =
                    (positions.get(&edge.source), positions.get(&edge.target))
                {
                    let dx = pos2.x - pos1.x;
                    let dy = pos2.y - pos1.y;
                    let distance = (dx * dx + dy * dy).sqrt().max(0.01);

                    let attractive_force = config.attraction_strength * distance * distance / k;
                    let fx = (dx / distance) * attractive_force * edge.strength;
                    let fy = (dy / distance) * attractive_force * edge.strength;

                    forces
                        .entry(edge.source.clone())
                        .or_insert_with(Vec3Data::zero)
                        .x += fx;
                    forces
                        .entry(edge.source.clone())
                        .or_insert_with(Vec3Data::zero)
                        .y += fy;
                    forces
                        .entry(edge.target.clone())
                        .or_insert_with(Vec3Data::zero)
                        .x -= fx;
                    forces
                        .entry(edge.target.clone())
                        .or_insert_with(Vec3Data::zero)
                        .y -= fy;
                }
            }

            
            if config.center_gravity {
                let center_x = (self.bounds.min_x + self.bounds.max_x) / 2.0;
                let center_y = (self.bounds.min_y + self.bounds.max_y) / 2.0;

                for (node_id, position) in &positions {
                    let dx = center_x - position.x;
                    let dy = center_y - position.y;
                    let distance = (dx * dx + dy * dy).sqrt().max(0.01);

                    let gravity_force = config.gravity * distance;
                    let fx = (dx / distance) * gravity_force;
                    let fy = (dy / distance) * gravity_force;

                    forces
                        .entry(node_id.clone())
                        .or_insert_with(Vec3Data::zero)
                        .x += fx;
                    forces
                        .entry(node_id.clone())
                        .or_insert_with(Vec3Data::zero)
                        .y += fy;
                }
            }

            
            let current_temp = if self.layout_config.adaptive_cooling {
                temperature * (1.0 - iteration as f32 / self.layout_config.iterations as f32)
            } else {
                temperature
            };

            for (node_id, force) in forces {
                if let Some(node) = self.nodes.get(&node_id) {
                    if node.fixed {
                        continue;
                    }
                }

                let velocity = velocities.get_mut(&node_id).unwrap();
                velocity.x = (velocity.x + force.x) * config.damping;
                velocity.y = (velocity.y + force.y) * config.damping;

                
                let vel_mag = (velocity.x * velocity.x + velocity.y * velocity.y).sqrt();
                if vel_mag > config.max_velocity {
                    velocity.x = (velocity.x / vel_mag) * config.max_velocity;
                    velocity.y = (velocity.y / vel_mag) * config.max_velocity;
                }

                
                let disp_mag = vel_mag.min(current_temp);
                if vel_mag > 0.0 {
                    velocity.x = (velocity.x / vel_mag) * disp_mag;
                    velocity.y = (velocity.y / vel_mag) * disp_mag;
                }

                let position = positions.get_mut(&node_id).unwrap();
                position.x += velocity.x;
                position.y += velocity.y;
            }

            
            let total_energy: f32 = velocities.values().map(|v| v.x * v.x + v.y * v.y).sum();

            if total_energy < self.physics.energy_threshold {
                self.metrics.convergence_achieved = true;
                self.metrics.iterations_performed = iteration + 1;
                break;
            }
        }

        self.metrics.final_energy = velocities.values().map(|v| v.x * v.x + v.y * v.y).sum();

        Ok(positions)
    }

    
    fn compute_hierarchical_layout(
        &mut self,
        config: &HierarchicalConfig,
    ) -> Result<HashMap<String, Position>, String> {
        let mut positions = HashMap::new();

        
        let layers = self.assign_nodes_to_layers();

        
        let ordered_layers = if config.minimize_crossings {
            self.minimize_crossings(layers)
        } else {
            layers
        };

        
        let layer_height = config.layer_separation;
        let node_width = config.node_separation;

        for (layer_index, layer_nodes) in ordered_layers.iter().enumerate() {
            let y = match config.direction {
                HierarchicalDirection::TopToBottom => layer_index as f32 * layer_height,
                HierarchicalDirection::BottomToTop => {
                    (ordered_layers.len() - 1 - layer_index) as f32 * layer_height
                }
                HierarchicalDirection::LeftToRight | HierarchicalDirection::RightToLeft => 0.0,
            };

            for (node_index, node_id) in layer_nodes.iter().enumerate() {
                let x = match config.direction {
                    HierarchicalDirection::LeftToRight => layer_index as f32 * layer_height,
                    HierarchicalDirection::RightToLeft => {
                        (ordered_layers.len() - 1 - layer_index) as f32 * layer_height
                    }
                    HierarchicalDirection::TopToBottom | HierarchicalDirection::BottomToTop => {
                        (node_index as f32 - (layer_nodes.len() as f32 - 1.0) / 2.0) * node_width
                    }
                };

                let final_y = match config.direction {
                    HierarchicalDirection::LeftToRight | HierarchicalDirection::RightToLeft => {
                        (node_index as f32 - (layer_nodes.len() as f32 - 1.0) / 2.0) * node_width
                    }
                    _ => y,
                };

                positions.insert(
                    node_id.clone(),
                    Position {
                        x,
                        y: final_y,
                        z: 0.0,
                    },
                );
            }
        }

        Ok(positions)
    }

    
    fn compute_circular_layout(
        &mut self,
        config: &CircularConfig,
    ) -> Result<HashMap<String, Position>, String> {
        let mut positions = HashMap::new();
        let node_count = self.nodes.len();

        if node_count == 0 {
            return Ok(positions);
        }

        
        let groups = self.group_nodes_by_metadata();
        let mut current_angle = config.start_angle;

        if groups.len() > 1 && config.concentric_layers {
            
            for (layer_index, (group_name, group_nodes)) in groups.iter().enumerate() {
                let radius = config.radius + layer_index as f32 * config.layer_spacing;
                let angle_step = 2.0 * PI / group_nodes.len() as f32;

                for (i, node_id) in group_nodes.iter().enumerate() {
                    let angle = current_angle
                        + i as f32 * angle_step * if config.clockwise { 1.0 } else { -1.0 };
                    let x = radius * angle.cos();
                    let y = radius * angle.sin();

                    positions.insert(node_id.clone(), Position { x, y, z: 0.0 });
                }

                current_angle += config.group_separation;
            }
        } else {
            
            let angle_step = 2.0 * PI / node_count as f32;

            for (i, node_id) in self.nodes.keys().enumerate() {
                let angle = current_angle
                    + i as f32 * angle_step * if config.clockwise { 1.0 } else { -1.0 };
                let x = config.radius * angle.cos();
                let y = config.radius * angle.sin();

                positions.insert(node_id.clone(), Position { x, y, z: 0.0 });
            }
        }

        Ok(positions)
    }

    
    fn compute_grid_layout(
        &mut self,
        config: &GridConfig,
    ) -> Result<HashMap<String, Position>, String> {
        let mut positions = HashMap::new();
        let node_count = self.nodes.len();

        if node_count == 0 {
            return Ok(positions);
        }

        
        let (cols, rows) = match (config.columns, config.rows) {
            (Some(c), Some(r)) => (c, r),
            (Some(c), None) => (c, (node_count as f32 / c as f32).ceil() as u32),
            (None, Some(r)) => ((node_count as f32 / r as f32).ceil() as u32, r),
            (None, None) => {
                let side = (node_count as f32).sqrt().ceil() as u32;
                (side, side)
            }
        };

        let total_width = cols as f32 * config.cell_width;
        let total_height = rows as f32 * config.cell_height;

        let start_x = if config.align_center {
            -total_width / 2.0
        } else {
            0.0
        };
        let start_y = if config.align_center {
            -total_height / 2.0
        } else {
            0.0
        };

        for (i, node_id) in self.nodes.keys().enumerate() {
            let col = i as u32 % cols;
            let row = i as u32 / cols;

            let x = start_x + col as f32 * config.cell_width + config.padding;
            let y = start_y + row as f32 * config.cell_height + config.padding;

            positions.insert(node_id.clone(), Position { x, y, z: 0.0 });
        }

        Ok(positions)
    }

    
    fn compute_spring_electrical_layout(
        &mut self,
        config: &SpringElectricalConfig,
    ) -> Result<HashMap<String, Position>, String> {
        
        
        self.compute_force_directed_layout(&ForceDirectedConfig {
            attraction_strength: config.spring_constant,
            repulsion_strength: config.electrical_charge,
            optimal_distance: config.spring_length,
            damping: 0.9,
            max_velocity: 10.0,
            gravity: 0.01,
            center_gravity: true,
        })
    }

    
    fn compute_stress_majorization_layout(
        &mut self,
        config: &StressConfig,
    ) -> Result<HashMap<String, Position>, String> {
        
        
        let mut positions = HashMap::new();

        
        for node_id in self.nodes.keys() {
            positions.insert(
                node_id.clone(),
                Position {
                    x: (rand::random::<f32>() - 0.5) * 100.0,
                    y: (rand::random::<f32>() - 0.5) * 100.0,
                    z: 0.0,
                },
            );
        }

        
        for _iteration in 0..config.stress_majorization_iterations {
            
            let mut gradients: HashMap<String, Vec3Data> = HashMap::new();

            for node_id in self.nodes.keys() {
                gradients.insert(node_id.clone(), Vec3Data::zero());
            }

            
            for (node_id, gradient) in gradients {
                if let Some(position) = positions.get_mut(&node_id) {
                    position.x -= gradient.x * 0.01; 
                    position.y -= gradient.y * 0.01;
                }
            }
        }

        Ok(positions)
    }

    
    fn compute_multi_level_layout(
        &mut self,
        config: &MultiLevelConfig,
    ) -> Result<HashMap<String, Position>, String> {
        
        
        match config.base_algorithm.as_ref() {
            LayoutAlgorithm::ForceDirected(fd_config) => {
                self.compute_force_directed_layout(fd_config)
            }
            _ => self.compute_force_directed_layout(&ForceDirectedConfig::default()),
        }
    }

    
    fn compute_geographic_layout(
        &mut self,
        config: &GeographicConfig,
    ) -> Result<HashMap<String, Position>, String> {
        let mut positions = HashMap::new();

        
        for (node_id, node) in &self.nodes {
            let lat = node
                .metadata
                .get("latitude")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let lon = node
                .metadata
                .get("longitude")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;

            
            let (x, y) = match config.projection {
                MapProjection::Mercator => self.mercator_projection(lat, lon, config),
                MapProjection::Robinson => self.robinson_projection(lat, lon, config),
                MapProjection::Orthographic => self.orthographic_projection(lat, lon, config),
                MapProjection::Stereographic => self.stereographic_projection(lat, lon, config),
            };

            positions.insert(node_id.clone(), Position { x, y, z: 0.0 });
        }

        Ok(positions)
    }

    
    fn compute_custom_layout(
        &mut self,
        config: &CustomLayoutConfig,
    ) -> Result<HashMap<String, Position>, String> {
        
        warn!(
            "Custom layout '{}' not implemented, falling back to force-directed",
            config.name
        );
        self.compute_force_directed_layout(&ForceDirectedConfig::default())
    }

    

    fn assign_nodes_to_layers(&self) -> Vec<Vec<String>> {
        
        let mut layers = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut current_layer = Vec::new();

        
        for node_id in self.nodes.keys() {
            let has_incoming = self.edges.values().any(|edge| &edge.target == node_id);
            if !has_incoming {
                current_layer.push(node_id.clone());
                visited.insert(node_id.clone());
            }
        }

        if !current_layer.is_empty() {
            layers.push(current_layer);
        }

        
        while visited.len() < self.nodes.len() {
            let mut next_layer = Vec::new();

            for node_id in self.nodes.keys() {
                if visited.contains(node_id) {
                    continue;
                }

                
                let all_predecessors_visited = self
                    .edges
                    .values()
                    .filter(|edge| &edge.target == node_id)
                    .all(|edge| visited.contains(&edge.source));

                if all_predecessors_visited {
                    next_layer.push(node_id.clone());
                }
            }

            if next_layer.is_empty() {
                
                for node_id in self.nodes.keys() {
                    if !visited.contains(node_id) {
                        next_layer.push(node_id.clone());
                    }
                }
            }

            for node_id in &next_layer {
                visited.insert(node_id.clone());
            }

            if !next_layer.is_empty() {
                layers.push(next_layer);
            }
        }

        layers
    }

    fn minimize_crossings(&self, mut layers: Vec<Vec<String>>) -> Vec<Vec<String>> {
        
        for _iteration in 0..10 {
            
            for i in 1..layers.len() {
                let mut positions = Vec::new();

                for node_id in &layers[i] {
                    let mut sum = 0.0;
                    let mut count = 0;

                    
                    for edge in self.edges.values() {
                        if &edge.target == node_id {
                            if let Some(pos) =
                                layers[i - 1].iter().position(|id| id == &edge.source)
                            {
                                sum += pos as f32;
                                count += 1;
                            }
                        }
                    }

                    let barycenter = if count > 0 { sum / count as f32 } else { 0.0 };
                    positions.push((node_id.clone(), barycenter));
                }

                
                positions
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                layers[i] = positions.into_iter().map(|(id, _)| id).collect();
            }
        }

        layers
    }

    fn group_nodes_by_metadata(&self) -> HashMap<String, Vec<String>> {
        let mut groups = HashMap::new();

        for (node_id, node) in &self.nodes {
            let group_name = node
                .group
                .clone()
                .or_else(|| {
                    node.metadata
                        .get("group")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .unwrap_or_else(|| "default".to_string());

            groups
                .entry(group_name)
                .or_insert_with(Vec::new)
                .push(node_id.clone());
        }

        groups
    }

    fn mercator_projection(&self, lat: f32, lon: f32, config: &GeographicConfig) -> (f32, f32) {
        let lat_rad = lat.to_radians();
        let x = (lon - config.center_point.1) * config.scale_factor;
        let y = (lat_rad.tan() + (PI / 4.0 + lat_rad / 2.0).tan()).ln() * config.scale_factor;
        (x, y)
    }

    fn robinson_projection(&self, lat: f32, lon: f32, config: &GeographicConfig) -> (f32, f32) {
        
        let x = (lon - config.center_point.1) * config.scale_factor;
        let y = lat * config.scale_factor;
        (x, y)
    }

    fn orthographic_projection(&self, lat: f32, lon: f32, config: &GeographicConfig) -> (f32, f32) {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let center_lat_rad = config.center_point.0.to_radians();
        let center_lon_rad = config.center_point.1.to_radians();

        let x = lon_rad.cos() * lat_rad.sin() * config.scale_factor;
        let y = (center_lat_rad.cos() * lat_rad.sin()
            - center_lat_rad.sin() * lat_rad.cos() * (lon_rad - center_lon_rad).cos())
            * config.scale_factor;
        (x, y)
    }

    fn stereographic_projection(
        &self,
        lat: f32,
        lon: f32,
        config: &GeographicConfig,
    ) -> (f32, f32) {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let k = 2.0 / (1.0 + lat_rad.sin());

        let x = k * lat_rad.cos() * lon_rad.sin() * config.scale_factor;
        let y = k * lat_rad.cos() * lon_rad.cos() * config.scale_factor;
        (x, y)
    }

    fn apply_bounds(&self, mut positions: HashMap<String, Position>) -> HashMap<String, Position> {
        for position in positions.values_mut() {
            position.x = position.x.clamp(self.bounds.min_x, self.bounds.max_x);
            position.y = position.y.clamp(self.bounds.min_y, self.bounds.max_y);
            position.z = position.z.clamp(self.bounds.min_z, self.bounds.max_z);
        }

        positions
    }

    fn compute_layout_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        
        let mut node_ids: Vec<&String> = self.nodes.keys().collect();
        node_ids.sort();
        for id in node_ids {
            id.hash(&mut hasher);
        }

        let mut edge_ids: Vec<&String> = self.edges.keys().collect();
        edge_ids.sort();
        for id in edge_ids {
            id.hash(&mut hasher);
        }

        
        format!("layout_{:016x}", hasher.finish())
    }

    fn compute_quality_score(&self, positions: &HashMap<String, Position>) -> f32 {
        
        let mut total_score = 0.0;
        let mut score_count = 0;

        
        for edge in self.edges.values() {
            if let (Some(pos1), Some(pos2)) =
                (positions.get(&edge.source), positions.get(&edge.target))
            {
                let dx = pos2.x - pos1.x;
                let dy = pos2.y - pos1.y;
                let distance = (dx * dx + dy * dy).sqrt();

                
                let optimal_distance = 50.0; 
                let distance_score = 1.0 - (distance - optimal_distance).abs() / optimal_distance;
                total_score += distance_score.max(0.0);
                score_count += 1;
            }
        }

        if score_count > 0 {
            total_score / score_count as f32
        } else {
            0.5 
        }
    }

    fn invalidate_cache(&mut self) {
        
        if self.layout_cache.len() > 10 {
            let mut cache_entries: Vec<(String, i64)> = self
                .layout_cache
                .iter()
                .map(|(k, v)| (k.clone(), v.created_at))
                .collect();
            cache_entries.sort_by_key(|(_, created_at)| *created_at);

            
            for (key, _) in cache_entries.iter().take(self.layout_cache.len() - 10) {
                self.layout_cache.remove(key);
            }
        }
    }

    
    pub fn get_metrics(&self) -> &LayoutMetrics {
        &self.metrics
    }

    
    pub fn get_topology_stats(&self) -> SwarmTopologyData {
        SwarmTopologyData {
            topology_type: "mesh".to_string(), 
            total_agents: self.nodes.len() as u32,
            coordination_layers: self.group_nodes_by_metadata().len() as u32,
            efficiency_score: if self.nodes.len() > 0 {
                1.0 - (self.edges.len() as f32 / (self.nodes.len() * self.nodes.len()) as f32)
            } else {
                1.0
            },
            load_distribution: self.compute_layer_loads(),
            critical_paths: Vec::new(), 
            bottlenecks: Vec::new(),    
        }
    }

    fn compute_layer_loads(&self) -> Vec<crate::services::agent_visualization_protocol::LayerLoad> {
        let groups = self.group_nodes_by_metadata();
        groups
            .into_iter()
            .enumerate()
            .map(|(index, (_, nodes))| {
                crate::services::agent_visualization_protocol::LayerLoad {
                    layer_id: index as u32,
                    agent_count: nodes.len() as u32,
                    average_load: 0.5,                      
                    max_capacity: (nodes.len() * 2) as u32, 
                    utilization: if nodes.len() > 0 { 0.5 } else { 0.0 }, 
                }
            })
            .collect()
    }

    fn count_cross_server_connections(&self) -> u32 {
        let mut cross_server_count = 0;

        for edge in self.edges.values() {
            let default_group = "default".to_string();
            let source_group = self
                .nodes
                .get(&edge.source)
                .and_then(|n| n.group.as_ref())
                .unwrap_or(&default_group);
            let target_group = self
                .nodes
                .get(&edge.target)
                .and_then(|n| n.group.as_ref())
                .unwrap_or(&default_group);

            if source_group != target_group {
                cross_server_count += 1;
            }
        }

        cross_server_count
    }

    
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.layout_cache.clear();
        self.metrics = LayoutMetrics::default();
    }

    
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}
