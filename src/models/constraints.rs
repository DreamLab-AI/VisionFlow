//! Constraint and physics parameter models for advanced force-directed layout
use serde::{Deserialize, Serialize};
// TODO: Re-enable after CUDA integration refactor
// use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

/// Type of constraint to apply to the graph layout
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum ConstraintKind {
    /// Fix nodes at specific positions
    FixedPosition = 0,
    /// Maintain minimum separation distance between nodes
    Separation = 1,
    /// Align nodes horizontally
    AlignmentHorizontal = 2,
    /// Align nodes vertically
    AlignmentVertical = 3,
    /// Align nodes along depth axis
    AlignmentDepth = 4,
    /// Group nodes in clusters based on similarity
    Clustering = 5,
    /// Keep nodes within boundary limits
    Boundary = 6,
    /// Enforce directional flow (e.g., hierarchical layout)
    DirectionalFlow = 7,
    /// Maintain radial distance from center
    RadialDistance = 8,
    /// Create layers of nodes at fixed depths
    LayerDepth = 9,
}

/// A constraint that affects graph layout computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Type of constraint
    pub kind: ConstraintKind,
    /// Indices of nodes affected by this constraint
    pub node_indices: Vec<u32>,
    /// Parameters specific to each constraint type:
    /// - FixedPosition: [x, y, z] coordinates
    /// - Separation: [min_distance]
    /// - AlignmentHorizontal: [y_coordinate]
    /// - AlignmentVertical: [x_coordinate]
    /// - AlignmentDepth: [z_coordinate]
    /// - Clustering: [cluster_id, strength]
    /// - Boundary: [min_x, max_x, min_y, max_y, min_z, max_z]
    /// - DirectionalFlow: [angle, strength]
    /// - RadialDistance: [center_x, center_y, center_z, radius]
    /// - LayerDepth: [layer_index, z_position]
    pub params: Vec<f32>,
    /// Weight/strength of this constraint (0.0 to 1.0)
    pub weight: f32,
    /// Whether this constraint is currently active
    pub active: bool,
}

impl Constraint {
    /// Create a new fixed position constraint
    pub fn fixed_position(node_idx: u32, x: f32, y: f32, z: f32) -> Self {
        Self {
            kind: ConstraintKind::FixedPosition,
            node_indices: vec![node_idx],
            params: vec![x, y, z],
            weight: 1.0,
            active: true,
        }
    }

    /// Create a new separation constraint between two nodes
    pub fn separation(node_a: u32, node_b: u32, min_distance: f32) -> Self {
        Self {
            kind: ConstraintKind::Separation,
            node_indices: vec![node_a, node_b],
            params: vec![min_distance],
            weight: 0.8,
            active: true,
        }
    }

    /// Create a horizontal alignment constraint for multiple nodes
    pub fn align_horizontal(node_indices: Vec<u32>, y_coord: f32) -> Self {
        Self {
            kind: ConstraintKind::AlignmentHorizontal,
            node_indices,
            params: vec![y_coord],
            weight: 0.6,
            active: true,
        }
    }

    /// Create a clustering constraint for a group of nodes
    pub fn cluster(node_indices: Vec<u32>, cluster_id: f32, strength: f32) -> Self {
        Self {
            kind: ConstraintKind::Clustering,
            node_indices,
            params: vec![cluster_id, strength],
            weight: 0.7,
            active: true,
        }
    }

    /// Create a boundary constraint for all specified nodes
    pub fn boundary(
        node_indices: Vec<u32>,
        min_x: f32, max_x: f32,
        min_y: f32, max_y: f32,
        min_z: f32, max_z: f32,
    ) -> Self {
        Self {
            kind: ConstraintKind::Boundary,
            node_indices,
            params: vec![min_x, max_x, min_y, max_y, min_z, max_z],
            weight: 0.9,
            active: true,
        }
    }

    /// Convert constraint to GPU-compatible format
    pub fn to_gpu_format(&self) -> ConstraintData {
        let mut gpu_constraint = ConstraintData {
            kind: self.kind as i32,
            count: self.node_indices.len().min(4) as i32,
            node_idx: [0, 0, 0, 0],
            params: [0.0; 8],
            weight: self.weight,
            activation_frame: 0, // Will be set when constraint is first applied
        };

        // Copy node indices (max 4)
        for (i, &node_idx) in self.node_indices.iter().take(4).enumerate() {
            gpu_constraint.node_idx[i] = node_idx as i32;
        }

        // Copy parameters (max 8)
        for (i, &param) in self.params.iter().take(8).enumerate() {
            gpu_constraint.params[i] = param;
        }

        gpu_constraint
    }
}

/// Advanced physics parameters for enhanced force simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedParams {
    /// Weight for semantic similarity forces (0.0 to 1.0)
    pub semantic_force_weight: f32,
    /// Weight for temporal co-evolution forces (0.0 to 1.0)
    pub temporal_force_weight: f32,
    /// Weight for structural dependency forces (0.0 to 1.0)
    pub structural_force_weight: f32,
    /// Weight for constraint satisfaction forces (0.0 to 1.0)
    pub constraint_force_weight: f32,
    /// Number of frames between stress-majorization steps
    pub stress_step_interval_frames: u32,
    /// Factor for node separation constraints (multiplier)
    pub separation_factor: f32,
    /// Weight for boundary containment forces (0.0 to 1.0)
    pub boundary_force_weight: f32,
    /// Weight for knowledge domain forces (0.0 to 1.0)
    pub knowledge_force_weight: f32,
    /// Weight for agent communication pattern forces (0.0 to 1.0)
    pub agent_communication_weight: f32,
    /// Enable adaptive force scaling based on graph density
    pub adaptive_force_scaling: bool,
    /// Target average edge length for layout optimization
    pub target_edge_length: f32,
    /// Maximum velocity allowed for any node
    pub max_velocity: f32,
    /// Minimum distance threshold for collision detection
    pub collision_threshold: f32,
    /// Enable hierarchical layout mode
    pub hierarchical_mode: bool,
    /// Depth separation for hierarchical layers
    pub layer_separation: f32,
}

impl Default for AdvancedParams {
    fn default() -> Self {
        Self {
            semantic_force_weight: 0.6,
            temporal_force_weight: 0.3,
            structural_force_weight: 0.5,
            constraint_force_weight: 0.8,
            stress_step_interval_frames: 600, // Re-enabled with safe cadence
            separation_factor: 1.5,
            boundary_force_weight: 0.7,
            knowledge_force_weight: 0.4,
            agent_communication_weight: 0.5,
            adaptive_force_scaling: true,
            target_edge_length: 150.0,
            max_velocity: 50.0,
            collision_threshold: 30.0,
            hierarchical_mode: false,
            layer_separation: 200.0,
        }
    }
}

impl AdvancedParams {
    /// Create parameters optimized for semantic clustering
    pub fn semantic_optimized() -> Self {
        Self {
            semantic_force_weight: 0.9,
            knowledge_force_weight: 0.8,
            temporal_force_weight: 0.4,
            ..Default::default()
        }
    }

    /// Create parameters optimized for agent swarm visualization
    pub fn agent_swarm_optimized() -> Self {
        Self {
            agent_communication_weight: 0.9,
            temporal_force_weight: 0.7,
            separation_factor: 2.0,
            collision_threshold: 50.0,
            ..Default::default()
        }
    }

    /// Create parameters optimized for hierarchical file structure
    pub fn hierarchical_optimized() -> Self {
        Self {
            hierarchical_mode: true,
            structural_force_weight: 0.9,
            layer_separation: 250.0,
            constraint_force_weight: 0.95,
            ..Default::default()
        }
    }
}

/// GPU-compatible constraint data for CUDA kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConstraintData {
    /// Discriminant matching ConstraintKind
    pub kind: i32,
    /// Number of node indices used
    pub count: i32,
    /// Node indices (max 4 for GPU efficiency)
    pub node_idx: [i32; 4],
    /// Parameters (max 8 for various constraint types)
    pub params: [f32; 8],
    /// Weight of this constraint
    pub weight: f32,
    /// Frame when this constraint was activated (for progressive activation)
    pub activation_frame: i32,
}

impl Default for ConstraintData {
    fn default() -> Self {
        Self {
            kind: 0,
            count: 0,
            node_idx: [0; 4],
            params: [0.0; 8],
            weight: 0.0,
            activation_frame: 0,
        }
    }
}

// Manual implementation of DeviceCopy for ConstraintData
unsafe impl cust::memory::DeviceCopy for ConstraintData {}

impl ConstraintData {
    /// Convert a Constraint to GPU-compatible format
    pub fn from_constraint(constraint: &Constraint) -> Self {
        let mut node_idx = [-1i32; 4];
        for (i, &idx) in constraint.node_indices.iter().take(4).enumerate() {
            node_idx[i] = idx as i32;
        }
        
        let mut params = [0.0f32; 8];
        for (i, &param) in constraint.params.iter().take(8).enumerate() {
            params[i] = param;
        }
        
        Self {
            kind: constraint.kind as i32,
            count: constraint.node_indices.len().min(4) as i32,
            node_idx,
            params,
            weight: constraint.weight,
            activation_frame: 0, // Will be set when constraint is first applied
        }
    }
}

/// Constraint set manager for organizing and applying constraints
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSet {
    /// All constraints in the system
    pub constraints: Vec<Constraint>,
    /// Named groups of constraints
    pub groups: std::collections::HashMap<String, Vec<usize>>,
}

impl ConstraintSet {
    /// Add a new constraint to the set
    pub fn add(&mut self, constraint: Constraint) -> usize {
        let idx = self.constraints.len();
        self.constraints.push(constraint);
        idx
    }
    
    /// Add a constraint to a named group
    pub fn add_to_group(&mut self, group_name: &str, constraint: Constraint) {
        let idx = self.add(constraint);
        self.groups.entry(group_name.to_string())
            .or_insert_with(Vec::new)
            .push(idx);
    }
    
    /// Enable/disable all constraints in a group
    pub fn set_group_active(&mut self, group_name: &str, active: bool) {
        if let Some(indices) = self.groups.get(group_name) {
            for &idx in indices {
                if let Some(constraint) = self.constraints.get_mut(idx) {
                    constraint.active = active;
                }
            }
        }
    }
    
    /// Get all active constraints
    pub fn active_constraints(&self) -> Vec<&Constraint> {
        self.constraints.iter()
            .filter(|c| c.active)
            .collect()
    }
    
    /// Convert to GPU-compatible format
    pub fn to_gpu_data(&self) -> Vec<ConstraintData> {
        self.active_constraints()
            .into_iter()
            .map(ConstraintData::from_constraint)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_constraint_creation() {
        let fixed = Constraint::fixed_position(0, 100.0, 200.0, 300.0);
        assert_eq!(fixed.kind, ConstraintKind::FixedPosition);
        assert_eq!(fixed.node_indices, vec![0]);
        assert_eq!(fixed.params, vec![100.0, 200.0, 300.0]);
        
        let sep = Constraint::separation(1, 2, 50.0);
        assert_eq!(sep.kind, ConstraintKind::Separation);
        assert_eq!(sep.node_indices, vec![1, 2]);
        assert_eq!(sep.params, vec![50.0]);
    }
    
    #[test]
    fn test_constraint_to_gpu_data() {
        let constraint = Constraint::cluster(vec![1, 2, 3], 1.0, 0.8);
        let gpu_data = ConstraintData::from_constraint(&constraint);
        
        assert_eq!(gpu_data.kind, ConstraintKind::Clustering as i32);
        assert_eq!(gpu_data.count, 3);
        assert_eq!(gpu_data.node_idx[0], 1);
        assert_eq!(gpu_data.node_idx[1], 2);
        assert_eq!(gpu_data.node_idx[2], 3);
        assert_eq!(gpu_data.params[0], 1.0);
        assert_eq!(gpu_data.params[1], 0.8);
    }
    
    #[test]
    fn test_constraint_set() {
        let mut set = ConstraintSet::default();
        
        set.add_to_group("fixed", Constraint::fixed_position(0, 0.0, 0.0, 0.0));
        set.add_to_group("fixed", Constraint::fixed_position(1, 100.0, 0.0, 0.0));
        set.add_to_group("separation", Constraint::separation(2, 3, 75.0));
        
        assert_eq!(set.constraints.len(), 3);
        assert_eq!(set.groups.get("fixed").unwrap().len(), 2);
        
        set.set_group_active("fixed", false);
        assert_eq!(set.active_constraints().len(), 1);
    }
}