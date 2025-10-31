// GPU Converter - Convert Physics Constraints to CUDA Format
// Week 3 Deliverable: CUDA-Compatible Data Structures

use super::physics_constraint::*;

/// GPU-compatible constraint data structure
/// Matches CUDA struct layout for direct memory copy
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConstraintData {
    /// Constraint kind (enum as i32 for GPU compatibility)
    pub kind: i32,

    /// Number of nodes affected
    pub count: i32,

    /// Node indices (max 4 nodes per constraint)
    /// For pairwise: [node1, node2, -1, -1]
    /// For multi-node: [node1, node2, node3, node4]
    pub node_idx: [i32; 4],

    /// Constraint parameters
    /// Separation: [min_distance, strength, 0, 0]
    /// Clustering: [ideal_distance, stiffness, 0, 0]
    /// Colocation: [target_distance, strength, 0, 0]
    /// Boundary: [min_x, max_x, min_y, max_y] (uses params2 for z)
    /// HierarchicalLayer: [z_level, strength, 0, 0]
    /// Containment: [parent_node_id, radius, strength, 0]
    pub params: [f32; 4],

    /// Additional parameters for boundary constraints
    /// Boundary: [min_z, max_z, strength, 0]
    pub params2: [f32; 4],

    /// Priority weight (10^(-(priority-1)/9))
    pub weight: f32,

    /// Activation frame for progressive constraints
    /// -1 = activate immediately
    pub activation_frame: i32,

    /// Padding to align to 16-byte boundary (GPU optimization)
    _padding: [f32; 2],
}

/// GPU constraint kind enumeration
/// Must match CUDA enum values
pub mod gpu_constraint_kind {
    pub const NONE: i32 = 0;
    pub const SEPARATION: i32 = 1;
    pub const CLUSTERING: i32 = 2;
    pub const COLOCATION: i32 = 3;
    pub const BOUNDARY: i32 = 4;
    pub const HIERARCHICAL_LAYER: i32 = 5;
    pub const CONTAINMENT: i32 = 6;
}

impl Default for ConstraintData {
    fn default() -> Self {
        Self {
            kind: gpu_constraint_kind::NONE,
            count: 0,
            node_idx: [-1; 4],
            params: [0.0; 4],
            params2: [0.0; 4],
            weight: 1.0,
            activation_frame: -1,
            _padding: [0.0; 2],
        }
    }
}

/// Convert physics constraint to GPU format
pub fn to_gpu_constraint_data(constraint: &PhysicsConstraint) -> ConstraintData {
    let mut data = ConstraintData::default();

    // Set node indices (max 4 nodes)
    data.count = constraint.nodes.len().min(4) as i32;
    for (i, &node_id) in constraint.nodes.iter().take(4).enumerate() {
        data.node_idx[i] = node_id as i32;
    }

    // Set priority weight
    data.weight = constraint.priority_weight();

    // Set activation frame
    data.activation_frame = constraint.activation_frame.unwrap_or(-1);

    // Set kind and parameters based on constraint type
    match &constraint.constraint_type {
        PhysicsConstraintType::Separation { min_distance, strength } => {
            data.kind = gpu_constraint_kind::SEPARATION;
            data.params[0] = *min_distance;
            data.params[1] = *strength;
        }

        PhysicsConstraintType::Clustering { ideal_distance, stiffness } => {
            data.kind = gpu_constraint_kind::CLUSTERING;
            data.params[0] = *ideal_distance;
            data.params[1] = *stiffness;
        }

        PhysicsConstraintType::Colocation { target_distance, strength } => {
            data.kind = gpu_constraint_kind::COLOCATION;
            data.params[0] = *target_distance;
            data.params[1] = *strength;
        }

        PhysicsConstraintType::Boundary { bounds, strength } => {
            data.kind = gpu_constraint_kind::BOUNDARY;
            data.params[0] = bounds[0]; // min_x
            data.params[1] = bounds[1]; // max_x
            data.params[2] = bounds[2]; // min_y
            data.params[3] = bounds[3]; // max_y
            data.params2[0] = bounds[4]; // min_z
            data.params2[1] = bounds[5]; // max_z
            data.params2[2] = *strength;
        }

        PhysicsConstraintType::HierarchicalLayer { z_level, strength } => {
            data.kind = gpu_constraint_kind::HIERARCHICAL_LAYER;
            data.params[0] = *z_level;
            data.params[1] = *strength;
        }

        PhysicsConstraintType::Containment { parent_node, radius, strength } => {
            data.kind = gpu_constraint_kind::CONTAINMENT;
            data.params[0] = *parent_node as f32;
            data.params[1] = *radius;
            data.params[2] = *strength;
        }
    }

    data
}

/// Convert multiple constraints to GPU format
pub fn to_gpu_constraint_batch(constraints: &[PhysicsConstraint]) -> Vec<ConstraintData> {
    constraints
        .iter()
        .map(to_gpu_constraint_data)
        .collect()
}

/// GPU constraint buffer for CUDA kernel launch
pub struct GPUConstraintBuffer {
    /// Raw constraint data
    pub data: Vec<ConstraintData>,

    /// Total number of constraints
    pub count: usize,

    /// Maximum constraints capacity
    pub capacity: usize,
}

impl GPUConstraintBuffer {
    /// Create a new GPU constraint buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            count: 0,
            capacity,
        }
    }

    /// Add constraints to buffer
    pub fn add_constraints(&mut self, constraints: &[PhysicsConstraint]) -> Result<(), String> {
        if self.count + constraints.len() > self.capacity {
            return Err(format!(
                "Buffer overflow: {} + {} > {}",
                self.count,
                constraints.len(),
                self.capacity
            ));
        }

        let gpu_data = to_gpu_constraint_batch(constraints);
        self.data.extend(gpu_data);
        self.count += constraints.len();

        Ok(())
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.data.clear();
        self.count = 0;
    }

    /// Get raw data pointer for CUDA
    pub fn as_ptr(&self) -> *const ConstraintData {
        self.data.as_ptr()
    }

    /// Get data size in bytes
    pub fn size_bytes(&self) -> usize {
        self.count * std::mem::size_of::<ConstraintData>()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get number of constraints
    pub fn len(&self) -> usize {
        self.count
    }
}

/// Statistics about GPU constraint buffer
#[derive(Debug, Clone)]
pub struct ConstraintStats {
    pub total_constraints: usize,
    pub separation_count: usize,
    pub clustering_count: usize,
    pub colocation_count: usize,
    pub boundary_count: usize,
    pub hierarchical_count: usize,
    pub containment_count: usize,
    pub user_defined_count: usize,
    pub progressive_count: usize,
    pub total_weight: f32,
}

impl ConstraintStats {
    /// Calculate statistics from constraint buffer
    pub fn from_buffer(buffer: &GPUConstraintBuffer) -> Self {
        let mut stats = Self {
            total_constraints: buffer.count,
            separation_count: 0,
            clustering_count: 0,
            colocation_count: 0,
            boundary_count: 0,
            hierarchical_count: 0,
            containment_count: 0,
            user_defined_count: 0,
            progressive_count: 0,
            total_weight: 0.0,
        };

        for constraint_data in &buffer.data {
            match constraint_data.kind {
                gpu_constraint_kind::SEPARATION => stats.separation_count += 1,
                gpu_constraint_kind::CLUSTERING => stats.clustering_count += 1,
                gpu_constraint_kind::COLOCATION => stats.colocation_count += 1,
                gpu_constraint_kind::BOUNDARY => stats.boundary_count += 1,
                gpu_constraint_kind::HIERARCHICAL_LAYER => stats.hierarchical_count += 1,
                gpu_constraint_kind::CONTAINMENT => stats.containment_count += 1,
                _ => {}
            }

            stats.total_weight += constraint_data.weight;

            if constraint_data.activation_frame >= 0 {
                stats.progressive_count += 1;
            }

            // User-defined have weight = 1.0 (priority 1)
            if (constraint_data.weight - 1.0).abs() < 0.001 {
                stats.user_defined_count += 1;
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separation_constraint_conversion() {
        let constraint = PhysicsConstraint::separation(vec![1, 2], 35.0, 0.8, 5);
        let gpu_data = to_gpu_constraint_data(&constraint);

        assert_eq!(gpu_data.kind, gpu_constraint_kind::SEPARATION);
        assert_eq!(gpu_data.count, 2);
        assert_eq!(gpu_data.node_idx[0], 1);
        assert_eq!(gpu_data.node_idx[1], 2);
        assert_eq!(gpu_data.node_idx[2], -1);
        assert_eq!(gpu_data.params[0], 35.0);
        assert_eq!(gpu_data.params[1], 0.8);
        assert!(gpu_data.weight > 0.0);
    }

    #[test]
    fn test_clustering_constraint_conversion() {
        let constraint = PhysicsConstraint::clustering(vec![10, 20], 20.0, 0.6, 3);
        let gpu_data = to_gpu_constraint_data(&constraint);

        assert_eq!(gpu_data.kind, gpu_constraint_kind::CLUSTERING);
        assert_eq!(gpu_data.count, 2);
        assert_eq!(gpu_data.params[0], 20.0);
        assert_eq!(gpu_data.params[1], 0.6);
    }

    #[test]
    fn test_boundary_constraint_conversion() {
        let bounds = [-20.0, 20.0, -20.0, 20.0, -20.0, 20.0];
        let constraint = PhysicsConstraint::boundary(vec![1], bounds, 0.7, 5);
        let gpu_data = to_gpu_constraint_data(&constraint);

        assert_eq!(gpu_data.kind, gpu_constraint_kind::BOUNDARY);
        assert_eq!(gpu_data.params[0], -20.0); // min_x
        assert_eq!(gpu_data.params[1], 20.0);  // max_x
        assert_eq!(gpu_data.params[2], -20.0); // min_y
        assert_eq!(gpu_data.params[3], 20.0);  // max_y
        assert_eq!(gpu_data.params2[0], -20.0); // min_z
        assert_eq!(gpu_data.params2[1], 20.0);  // max_z
        assert_eq!(gpu_data.params2[2], 0.7);   // strength
    }

    #[test]
    fn test_hierarchical_layer_conversion() {
        let constraint = PhysicsConstraint::hierarchical_layer(vec![1, 2, 3], 100.0, 0.7, 5);
        let gpu_data = to_gpu_constraint_data(&constraint);

        assert_eq!(gpu_data.kind, gpu_constraint_kind::HIERARCHICAL_LAYER);
        assert_eq!(gpu_data.count, 3);
        assert_eq!(gpu_data.params[0], 100.0);
        assert_eq!(gpu_data.params[1], 0.7);
    }

    #[test]
    fn test_containment_conversion() {
        let constraint = PhysicsConstraint::containment(vec![1, 2], 100, 50.0, 0.8, 5);
        let gpu_data = to_gpu_constraint_data(&constraint);

        assert_eq!(gpu_data.kind, gpu_constraint_kind::CONTAINMENT);
        assert_eq!(gpu_data.params[0], 100.0); // parent_node
        assert_eq!(gpu_data.params[1], 50.0);  // radius
        assert_eq!(gpu_data.params[2], 0.8);   // strength
    }

    #[test]
    fn test_activation_frame() {
        let constraint = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5)
            .with_activation_frame(60);

        let gpu_data = to_gpu_constraint_data(&constraint);
        assert_eq!(gpu_data.activation_frame, 60);
    }

    #[test]
    fn test_priority_weight() {
        let c1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 1);
        let c2 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 10);

        let gpu1 = to_gpu_constraint_data(&c1);
        let gpu2 = to_gpu_constraint_data(&c2);

        assert!(gpu1.weight > gpu2.weight);
        assert!((gpu1.weight - 1.0).abs() < 0.001);
        assert!((gpu2.weight - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_batch_conversion() {
        let constraints = vec![
            PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5),
            PhysicsConstraint::clustering(vec![2, 3], 20.0, 0.6, 3),
            PhysicsConstraint::colocation(vec![3, 4], 2.0, 0.9, 1),
        ];

        let gpu_batch = to_gpu_constraint_batch(&constraints);
        assert_eq!(gpu_batch.len(), 3);
        assert_eq!(gpu_batch[0].kind, gpu_constraint_kind::SEPARATION);
        assert_eq!(gpu_batch[1].kind, gpu_constraint_kind::CLUSTERING);
        assert_eq!(gpu_batch[2].kind, gpu_constraint_kind::COLOCATION);
    }

    #[test]
    fn test_gpu_buffer() {
        let mut buffer = GPUConstraintBuffer::new(100);

        let constraints = vec![
            PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5),
            PhysicsConstraint::clustering(vec![2, 3], 20.0, 0.6, 3),
        ];

        assert!(buffer.add_constraints(&constraints).is_ok());
        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_empty());

        let size = buffer.size_bytes();
        assert_eq!(size, 2 * std::mem::size_of::<ConstraintData>());

        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buffer = GPUConstraintBuffer::new(2);

        let constraints = vec![
            PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5),
            PhysicsConstraint::clustering(vec![2, 3], 20.0, 0.6, 3),
            PhysicsConstraint::colocation(vec![3, 4], 2.0, 0.9, 1),
        ];

        assert!(buffer.add_constraints(&constraints).is_err());
    }

    #[test]
    fn test_constraint_stats() {
        let mut buffer = GPUConstraintBuffer::new(100);

        let constraints = vec![
            PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5),
            PhysicsConstraint::separation(vec![2, 3], 15.0, 0.6, 5),
            PhysicsConstraint::clustering(vec![3, 4], 20.0, 0.6, 3),
            PhysicsConstraint::colocation(vec![4, 5], 2.0, 0.9, 1).mark_user_defined(),
        ];

        buffer.add_constraints(&constraints).unwrap();

        let stats = ConstraintStats::from_buffer(&buffer);
        assert_eq!(stats.total_constraints, 4);
        assert_eq!(stats.separation_count, 2);
        assert_eq!(stats.clustering_count, 1);
        assert_eq!(stats.colocation_count, 1);
        assert_eq!(stats.user_defined_count, 1);
    }

    #[test]
    fn test_constraint_data_size() {
        // Verify struct size for GPU memory alignment
        let size = std::mem::size_of::<ConstraintData>();

        // Should be multiple of 16 bytes for optimal GPU memory access
        assert_eq!(size % 16, 0);
    }
}
