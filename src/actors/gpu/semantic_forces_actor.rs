//! Semantic Forces Actor - Handles DAG layout, type clustering, and collision detection
//! Integrates with GPU kernels in semantic_forces.cu for advanced graph layout

use actix::prelude::*;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::shared::{GPUState, SharedGPUContext};
use crate::actors::messages::*;
use crate::telemetry::agent_telemetry::{
    get_telemetry_logger, CorrelationId, LogLevel, TelemetryEvent,
};

// Re-export message types for handlers
pub use crate::actors::messages::{
    ConfigureCollision, ConfigureDAG, ConfigureTypeClustering,
    GetHierarchyLevels, GetSemanticConfig, RecalculateHierarchy,
};

// =============================================================================
// GPU Kernel FFI Declarations
// =============================================================================
#[repr(C)]
struct DAGConfigGPU {
    vertical_spacing: f32,
    horizontal_spacing: f32,
    level_attraction: f32,
    sibling_repulsion: f32,
    enabled: bool,
}
#[repr(C)]
struct TypeClusterConfigGPU {
    cluster_attraction: f32,
    cluster_radius: f32,
    inter_cluster_repulsion: f32,
    enabled: bool,
}
#[repr(C)]
struct CollisionConfigGPU {
    min_distance: f32,
    collision_strength: f32,
    node_radius: f32,
    enabled: bool,
}
#[repr(C)]
struct AttributeSpringConfigGPU {
    base_spring_k: f32,
    weight_multiplier: f32,
    rest_length_min: f32,
    rest_length_max: f32,
    enabled: bool,
}
#[repr(C)]
struct SemanticConfigGPU {
    dag: DAGConfigGPU,
    type_cluster: TypeClusterConfigGPU,
    collision: CollisionConfigGPU,
    attribute_spring: AttributeSpringConfigGPU,
}

// =============================================================================
// Dynamic Force Configuration (Schema-Code Decoupling)
// =============================================================================

/// GPU-compatible dynamic force configuration for a relationship type
/// Matches the DynamicForceConfig struct in semantic_forces.cu
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DynamicForceConfigGPU {
    pub strength: f32,        // Spring strength (can be negative for repulsion)
    pub rest_length: f32,     // Rest length for spring calculations
    pub is_directional: i32,  // 1 = directional, 0 = bidirectional
    pub force_type: u32,      // Force behavior type (0=spring, 1=orbit, 2=cross-domain, 3=repulsion)
}

impl Default for DynamicForceConfigGPU {
    fn default() -> Self {
        Self {
            strength: 0.5,
            rest_length: 100.0,
            is_directional: 0,
            force_type: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Float3 {
    x: f32,
    y: f32,
    z: f32,
}
extern "C" {
    /// Upload semantic configuration to GPU constant memory
    fn set_semantic_config(config: *const SemanticConfigGPU);

    /// Apply DAG layout forces based on hierarchy levels
    fn apply_dag_force(
        node_hierarchy_levels: *const i32,
        node_types: *const i32,
        positions: *mut Float3,
        forces: *mut Float3,
        num_nodes: i32,
    );

    /// Apply type clustering forces
    fn apply_type_cluster_force(
        node_types: *const i32,
        type_centroids: *const Float3,
        positions: *mut Float3,
        forces: *mut Float3,
        num_nodes: i32,
        num_types: i32,
    );

    /// Apply collision detection and response forces
    fn apply_collision_force(
        node_radii: *const f32,
        positions: *mut Float3,
        forces: *mut Float3,
        num_nodes: i32,
    );

    /// Apply attribute-weighted spring forces
    fn apply_attribute_spring_force(
        edge_sources: *const i32,
        edge_targets: *const i32,
        edge_weights: *const f32,
        edge_types: *const i32,
        positions: *mut Float3,
        forces: *mut Float3,
        num_edges: i32,
    );

    /// Calculate hierarchy levels for DAG layout
    fn calculate_hierarchy_levels(
        edge_sources: *const i32,
        edge_targets: *const i32,
        edge_types: *const i32,
        node_levels: *mut i32,
        changed: *mut bool,
        num_edges: i32,
        num_nodes: i32,
    );

    /// Calculate centroid positions for each node type
    fn calculate_type_centroids(
        node_types: *const i32,
        positions: *const Float3,
        type_centroids: *mut Float3,
        type_counts: *mut i32,
        num_nodes: i32,
        num_types: i32,
    );

    /// Finalize centroids by dividing by count
    fn finalize_type_centroids(
        type_centroids: *mut Float3,
        type_counts: *const i32,
        num_types: i32,
    );

    // ==========================================================================
    // Dynamic Relationship Buffer Management (Hot-Reload)
    // ==========================================================================

    /// Upload dynamic relationship configurations to GPU
    /// Enables ontology changes without CUDA recompilation
    fn set_dynamic_relationship_buffer(
        configs: *const DynamicForceConfigGPU,
        num_types: i32,
        enabled: bool,
    ) -> i32;

    /// Update a single relationship type configuration (hot-reload)
    fn update_dynamic_relationship_config(
        type_id: i32,
        config: *const DynamicForceConfigGPU,
    ) -> i32;

    /// Enable or disable dynamic relationship forces
    fn set_dynamic_relationships_enabled(enabled: bool) -> i32;

    /// Get current buffer version for hot-reload detection
    fn get_dynamic_relationship_buffer_version() -> i32;

    /// Get maximum supported relationship types
    fn get_max_relationship_types() -> i32;

    /// Apply dynamic relationship forces (schema-code decoupled)
    fn apply_dynamic_relationship_force(
        edge_sources: *const i32,
        edge_targets: *const i32,
        edge_types: *const i32,
        node_cross_domain_count: *const i32,
        positions: *mut Float3,
        forces: *mut Float3,
        num_edges: i32,
    );
}

/// DAG layout configuration matching GPU kernel structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAGConfig {
    pub vertical_spacing: f32,      // Vertical separation between hierarchy levels
    pub horizontal_spacing: f32,    // Minimum horizontal separation within a level
    pub level_attraction: f32,      // Strength of attraction to target level
    pub sibling_repulsion: f32,     // Repulsion between nodes at same level
    pub enabled: bool,
    pub layout_mode: DAGLayoutMode,
}

/// DAG layout modes for different visual hierarchies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DAGLayoutMode {
    TopDown,      // Traditional top-down hierarchy
    Radial,       // Radial/circular hierarchy
    LeftRight,    // Left-to-right hierarchy
}

impl Default for DAGConfig {
    fn default() -> Self {
        Self {
            vertical_spacing: 100.0,
            horizontal_spacing: 50.0,
            level_attraction: 0.5,
            sibling_repulsion: 0.3,
            enabled: false,
            layout_mode: DAGLayoutMode::TopDown,
        }
    }
}

/// Type clustering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeClusterConfig {
    pub cluster_attraction: f32,    // Attraction between nodes of same type
    pub cluster_radius: f32,        // Target radius for type clusters
    pub inter_cluster_repulsion: f32, // Repulsion between different type clusters
    pub enabled: bool,
}

impl Default for TypeClusterConfig {
    fn default() -> Self {
        Self {
            cluster_attraction: 0.4,
            cluster_radius: 80.0,
            inter_cluster_repulsion: 0.2,
            enabled: false,
        }
    }
}

/// Collision detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionConfig {
    pub min_distance: f32,          // Minimum allowed distance between nodes
    pub collision_strength: f32,    // Force strength when colliding
    pub node_radius: f32,           // Default node radius
    pub enabled: bool,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            min_distance: 10.0,
            collision_strength: 0.8,
            node_radius: 15.0,
            enabled: true,
        }
    }
}

/// Attribute-weighted spring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeSpringConfig {
    pub base_spring_k: f32,         // Base spring constant
    pub weight_multiplier: f32,     // Multiplier for edge weight influence
    pub rest_length_min: f32,       // Minimum rest length
    pub rest_length_max: f32,       // Maximum rest length
    pub enabled: bool,
}

impl Default for AttributeSpringConfig {
    fn default() -> Self {
        Self {
            base_spring_k: 0.1,
            weight_multiplier: 1.5,
            rest_length_min: 50.0,
            rest_length_max: 200.0,
            enabled: false,
        }
    }
}

/// Combined semantic configuration
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

/// Node hierarchy level assignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyLevels {
    pub node_levels: Vec<i32>,      // Hierarchy level for each node (-1 = not in DAG)
    pub max_level: i32,             // Maximum hierarchy level
    pub level_counts: Vec<usize>,   // Number of nodes at each level
}

/// Type centroid positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCentroids {
    pub centroids: Vec<(f32, f32, f32)>,  // Centroid position for each type
    pub type_counts: Vec<usize>,          // Number of nodes of each type
}

/// Semantic Forces Actor - manages semantic layout forces
pub struct SemanticForcesActor {
    /// Shared GPU context for accessing GPU resources
    shared_context: Option<Arc<SharedGPUContext>>,

    /// Current semantic configuration
    config: SemanticConfig,

    /// GPU state tracking
    gpu_state: GPUState,

    /// Cached hierarchy levels (computed on demand)
    hierarchy_levels: Option<HierarchyLevels>,

    /// Cached type centroids (recomputed each frame)
    type_centroids: Option<TypeCentroids>,

    /// Number of node types in the graph
    num_types: usize,

    /// Cached node types array for GPU access
    node_types: Vec<i32>,

    /// Cached edge data for attribute springs
    edge_sources: Vec<i32>,
    edge_targets: Vec<i32>,
    edge_weights: Vec<f32>,
    edge_types: Vec<i32>,
}

impl SemanticForcesActor {
    pub fn new() -> Self {
        Self {
            shared_context: None,
            config: SemanticConfig::default(),
            gpu_state: GPUState::default(),
            hierarchy_levels: None,
            type_centroids: None,
            num_types: 0,
            node_types: Vec::new(),
            edge_sources: Vec::new(),
            edge_targets: Vec::new(),
            edge_weights: Vec::new(),
            edge_types: Vec::new(),
        }
    }

    /// Convert Rust config to GPU C-compatible struct
    fn config_to_gpu(&self) -> SemanticConfigGPU {
        SemanticConfigGPU {
            dag: DAGConfigGPU {
                vertical_spacing: self.config.dag.vertical_spacing,
                horizontal_spacing: self.config.dag.horizontal_spacing,
                level_attraction: self.config.dag.level_attraction,
                sibling_repulsion: self.config.dag.sibling_repulsion,
                enabled: self.config.dag.enabled,
            },
            type_cluster: TypeClusterConfigGPU {
                cluster_attraction: self.config.type_cluster.cluster_attraction,
                cluster_radius: self.config.type_cluster.cluster_radius,
                inter_cluster_repulsion: self.config.type_cluster.inter_cluster_repulsion,
                enabled: self.config.type_cluster.enabled,
            },
            collision: CollisionConfigGPU {
                min_distance: self.config.collision.min_distance,
                collision_strength: self.config.collision.collision_strength,
                node_radius: self.config.collision.node_radius,
                enabled: self.config.collision.enabled,
            },
            attribute_spring: AttributeSpringConfigGPU {
                base_spring_k: self.config.attribute_spring.base_spring_k,
                weight_multiplier: self.config.attribute_spring.weight_multiplier,
                rest_length_min: self.config.attribute_spring.rest_length_min,
                rest_length_max: self.config.attribute_spring.rest_length_max,
                enabled: self.config.attribute_spring.enabled,
            },
        }
    }

    /// Calculate hierarchy levels using topological sort (BFS-style on GPU)
    fn calculate_hierarchy_levels(
        &mut self,
        num_nodes: usize,
        num_edges: usize,
    ) -> Result<HierarchyLevels, String> {
        info!("SemanticForcesActor: Calculating hierarchy levels for {} nodes, {} edges",
              num_nodes, num_edges);

        let _shared_context = self.shared_context.as_ref()
            .ok_or("GPU context not initialized")?;

        // Initialize node levels to -1 (not in hierarchy)
        let mut node_levels = vec![-1i32; num_nodes];

        // Find root nodes (nodes with no incoming hierarchy edges)
        let mut has_incoming_hierarchy = vec![false; num_nodes];
        for i in 0..self.edge_sources.len() {
            if self.edge_types[i] == 2 { // Hierarchy edge type = 2
                let target = self.edge_targets[i] as usize;
                if target < num_nodes {
                    has_incoming_hierarchy[target] = true;
                }
            }
        }

        // Set root nodes to level 0
        for (i, &has_incoming) in has_incoming_hierarchy.iter().enumerate() {
            if !has_incoming {
                node_levels[i] = 0;
            }
        }
        {
            // GPU-accelerated hierarchy computation using parallel BFS
            if num_edges > 0 && !self.edge_sources.is_empty() {
                let mut changed = true;
                let mut iteration = 0;
                const MAX_ITERATIONS: usize = 100;

                while changed && iteration < MAX_ITERATIONS {
                    changed = false;
                    unsafe {
                        calculate_hierarchy_levels(
                            self.edge_sources.as_ptr(),
                            self.edge_targets.as_ptr(),
                            self.edge_types.as_ptr(),
                            node_levels.as_mut_ptr(),
                            &mut changed as *mut bool,
                            num_edges as i32,
                            num_nodes as i32,
                        );
                    }
                    iteration += 1;
                }

                if iteration >= MAX_ITERATIONS {
                    warn!("SemanticForcesActor: Hierarchy calculation reached max iterations");
                }
            }
        }

        // Calculate max_level and level_counts before moving node_levels
        let max_level = node_levels.iter().copied().max().unwrap_or(0);
        let mut level_counts = vec![0; (max_level + 1) as usize];
        for &level in &node_levels {
            if level >= 0 {
                level_counts[level as usize] += 1;
            }
        }

        // Return computed hierarchy levels
        Ok(HierarchyLevels {
            node_levels,
            max_level,
            level_counts,
        })
    }

    /// Calculate centroids for each node type
    fn calculate_type_centroids(
        &mut self,
        positions: &[(f32, f32, f32)],
        num_nodes: usize,
    ) -> Result<TypeCentroids, String> {
        if self.num_types == 0 {
            return Ok(TypeCentroids {
                centroids: Vec::new(),
                type_counts: Vec::new(),
            });
        }

        let mut centroids = vec![(0.0f32, 0.0f32, 0.0f32); self.num_types];
        let mut type_counts = vec![0usize; self.num_types];

        // Calculate type centroids using GPU
        if self.shared_context.is_some() && num_nodes > 0 && !self.node_types.is_empty() {
            let mut centroid_f3 = vec![Float3 { x: 0.0, y: 0.0, z: 0.0 }; self.num_types];
            let mut counts_i32 = vec![0i32; self.num_types];

            let positions_f3: Vec<Float3> = positions.iter()
                .map(|(x, y, z)| Float3 { x: *x, y: *y, z: *z })
                .collect();

            unsafe {
                calculate_type_centroids(
                    self.node_types.as_ptr(),
                    positions_f3.as_ptr(),
                    centroid_f3.as_mut_ptr(),
                    counts_i32.as_mut_ptr(),
                    num_nodes as i32,
                    self.num_types as i32,
                );

                finalize_type_centroids(
                    centroid_f3.as_mut_ptr(),
                    counts_i32.as_ptr(),
                    self.num_types as i32,
                );
            }

            centroids = centroid_f3.iter()
                .map(|f3| (f3.x, f3.y, f3.z))
                .collect();
            type_counts = counts_i32.iter()
                .map(|&c| c as usize)
                .collect();
        }

        Ok(TypeCentroids {
            centroids,
            type_counts,
        })
    }
}

// Actor implementation
impl Actor for SemanticForcesActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("SemanticForcesActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("SemanticForcesActor stopped");
    }
}