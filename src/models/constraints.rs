//! Constraint and physics parameter models for advanced force-directed layout
use serde::{Deserialize, Serialize};
// TODO: Re-enable after CUDA integration refactor
// use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

///
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub enum ConstraintKind {
    
    FixedPosition = 0,
    
    Separation = 1,
    
    AlignmentHorizontal = 2,
    
    AlignmentVertical = 3,
    
    AlignmentDepth = 4,
    
    Clustering = 5,
    
    Boundary = 6,
    
    DirectionalFlow = 7,
    
    RadialDistance = 8,
    
    LayerDepth = 9,
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    
    pub kind: ConstraintKind,
    
    pub node_indices: Vec<u32>,
    
    
    
    
    
    
    
    
    
    
    
    pub params: Vec<f32>,
    
    pub weight: f32,
    
    pub active: bool,
}

impl Constraint {
    
    pub fn fixed_position(node_idx: u32, x: f32, y: f32, z: f32) -> Self {
        Self {
            kind: ConstraintKind::FixedPosition,
            node_indices: vec![node_idx],
            params: vec![x, y, z],
            weight: 1.0,
            active: true,
        }
    }

    
    pub fn separation(node_a: u32, node_b: u32, min_distance: f32) -> Self {
        Self {
            kind: ConstraintKind::Separation,
            node_indices: vec![node_a, node_b],
            params: vec![min_distance],
            weight: 0.8,
            active: true,
        }
    }

    
    pub fn align_horizontal(node_indices: Vec<u32>, y_coord: f32) -> Self {
        Self {
            kind: ConstraintKind::AlignmentHorizontal,
            node_indices,
            params: vec![y_coord],
            weight: 0.6,
            active: true,
        }
    }

    
    pub fn cluster(node_indices: Vec<u32>, cluster_id: f32, strength: f32) -> Self {
        Self {
            kind: ConstraintKind::Clustering,
            node_indices,
            params: vec![cluster_id, strength],
            weight: 0.7,
            active: true,
        }
    }

    
    pub fn boundary(
        node_indices: Vec<u32>,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
        min_z: f32,
        max_z: f32,
    ) -> Self {
        Self {
            kind: ConstraintKind::Boundary,
            node_indices,
            params: vec![min_x, max_x, min_y, max_y, min_z, max_z],
            weight: 0.9,
            active: true,
        }
    }

    
    pub fn to_gpu_format(&self) -> ConstraintData {
        let mut gpu_constraint = ConstraintData {
            kind: self.kind as i32,
            count: self.node_indices.len().min(4) as i32,
            node_idx: [0, 0, 0, 0],
            params: [0.0; 8],
            weight: self.weight,
            activation_frame: 0, 
        };

        
        for (i, &node_idx) in self.node_indices.iter().take(4).enumerate() {
            gpu_constraint.node_idx[i] = node_idx as i32;
        }

        
        for (i, &param) in self.params.iter().take(8).enumerate() {
            gpu_constraint.params[i] = param;
        }

        gpu_constraint
    }
}

///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedParams {
    
    pub semantic_force_weight: f32,
    
    pub temporal_force_weight: f32,
    
    pub structural_force_weight: f32,
    
    pub constraint_force_weight: f32,
    
    pub stress_step_interval_frames: u32,
    
    pub separation_factor: f32,
    
    pub boundary_force_weight: f32,
    
    pub knowledge_force_weight: f32,
    
    pub agent_communication_weight: f32,
    
    pub adaptive_force_scaling: bool,
    
    pub target_edge_length: f32,
    
    pub max_velocity: f32,
    
    pub collision_threshold: f32,
    
    pub hierarchical_mode: bool,
    
    pub layer_separation: f32,
}

impl Default for AdvancedParams {
    fn default() -> Self {
        Self {
            semantic_force_weight: 0.6,
            temporal_force_weight: 0.3,
            structural_force_weight: 0.5,
            constraint_force_weight: 0.8,
            stress_step_interval_frames: 600, 
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
    
    pub fn semantic_optimized() -> Self {
        Self {
            semantic_force_weight: 0.9,
            knowledge_force_weight: 0.8,
            temporal_force_weight: 0.4,
            ..Default::default()
        }
    }

    
    pub fn agent_swarm_optimized() -> Self {
        Self {
            agent_communication_weight: 0.9,
            temporal_force_weight: 0.7,
            separation_factor: 2.0,
            collision_threshold: 50.0,
            ..Default::default()
        }
    }

    
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

///
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConstraintData {
    
    pub kind: i32,
    
    pub count: i32,
    
    pub node_idx: [i32; 4],
    
    pub params: [f32; 8],
    
    pub weight: f32,
    
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

// Manual implementation of DeviceCopy for ConstraintData (only when gpu feature is enabled)
#[cfg(feature = "gpu")]
unsafe impl cust::memory::DeviceCopy for ConstraintData {}

impl ConstraintData {
    
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
            activation_frame: 0, 
        }
    }
}

///
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintSet {
    
    pub constraints: Vec<Constraint>,
    
    pub groups: std::collections::HashMap<String, Vec<usize>>,
}

impl ConstraintSet {
    
    pub fn add(&mut self, constraint: Constraint) -> usize {
        let idx = self.constraints.len();
        self.constraints.push(constraint);
        idx
    }

    
    pub fn add_to_group(&mut self, group_name: &str, constraint: Constraint) {
        let idx = self.add(constraint);
        self.groups
            .entry(group_name.to_string())
            .or_insert_with(Vec::new)
            .push(idx);
    }

    
    pub fn set_group_active(&mut self, group_name: &str, active: bool) {
        if let Some(indices) = self.groups.get(group_name) {
            for &idx in indices {
                if let Some(constraint) = self.constraints.get_mut(idx) {
                    constraint.active = active;
                }
            }
        }
    }

    
    pub fn active_constraints(&self) -> Vec<&Constraint> {
        self.constraints.iter().filter(|c| c.active).collect()
    }

    
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
