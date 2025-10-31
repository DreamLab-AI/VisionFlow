// Priority Resolver - Conflict Resolution for Physics Constraints
// Week 3 Deliverable: Weighted Blending of Conflicting Constraints

use super::physics_constraint::*;
use std::collections::HashMap;

/// Node pair identifier for grouping conflicting constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodePair {
    pub node1: NodeId,
    pub node2: NodeId,
}

impl NodePair {
    /// Create a new node pair (order-independent)
    pub fn new(node1: NodeId, node2: NodeId) -> Self {
        if node1 <= node2 {
            Self { node1, node2 }
        } else {
            Self { node1: node2, node2: node1 }
        }
    }

    /// Check if this pair contains the given node
    pub fn contains(&self, node: NodeId) -> bool {
        self.node1 == node || self.node2 == node
    }
}

/// Grouped constraints for the same node pair
#[derive(Debug, Clone)]
pub struct ConstraintGroup {
    pub node_pair: NodePair,
    pub constraints: Vec<PhysicsConstraint>,
}

impl ConstraintGroup {
    /// Create a new constraint group
    pub fn new(node_pair: NodePair) -> Self {
        Self {
            node_pair,
            constraints: Vec::new(),
        }
    }

    /// Add a constraint to this group
    pub fn add_constraint(&mut self, constraint: PhysicsConstraint) {
        self.constraints.push(constraint);
    }

    /// Check if there are conflicts (multiple constraints for same pair)
    pub fn has_conflicts(&self) -> bool {
        self.constraints.len() > 1
    }

    /// Get the highest priority constraint (user overrides)
    pub fn get_highest_priority(&self) -> Option<&PhysicsConstraint> {
        self.constraints
            .iter()
            .min_by_key(|c| c.priority) // Lower priority number = higher importance
    }

    /// Check if any constraint is user-defined
    pub fn has_user_defined(&self) -> bool {
        self.constraints.iter().any(|c| c.user_defined)
    }

    /// Calculate total weight for all constraints
    pub fn total_weight(&self) -> f32 {
        self.constraints
            .iter()
            .map(|c| c.priority_weight())
            .sum()
    }
}

/// Priority resolver for constraint conflicts
pub struct PriorityResolver {
    /// Group constraints by node pair
    constraint_groups: HashMap<NodePair, ConstraintGroup>,
}

impl PriorityResolver {
    /// Create a new priority resolver
    pub fn new() -> Self {
        Self {
            constraint_groups: HashMap::new(),
        }
    }

    /// Add constraints to the resolver
    pub fn add_constraints(&mut self, constraints: Vec<PhysicsConstraint>) {
        for constraint in constraints {
            self.add_constraint(constraint);
        }
    }

    /// Add a single constraint to the resolver
    pub fn add_constraint(&mut self, constraint: PhysicsConstraint) {
        // For pairwise constraints (Separation, Clustering, Colocation)
        if constraint.nodes.len() == 2 {
            let pair = NodePair::new(constraint.nodes[0], constraint.nodes[1]);
            self.constraint_groups
                .entry(pair)
                .or_insert_with(|| ConstraintGroup::new(pair))
                .add_constraint(constraint);
        } else {
            // For multi-node constraints (Boundary, HierarchicalLayer, Containment)
            // Create individual entries for each node
            for &node in &constraint.nodes {
                // Use a synthetic pair (node, node) for single-node constraints
                let pair = NodePair::new(node, node);
                self.constraint_groups
                    .entry(pair)
                    .or_insert_with(|| ConstraintGroup::new(pair))
                    .add_constraint(constraint.clone());
            }
        }
    }

    /// Resolve all conflicts and return final constraints
    /// Strategy:
    /// 1. User-defined constraints always win (priority 1)
    /// 2. Otherwise, use weighted blending based on priority weights
    pub fn resolve(&self) -> Vec<PhysicsConstraint> {
        self.constraint_groups
            .values()
            .filter_map(|group| self.resolve_group(group))
            .collect()
    }

    /// Resolve a single constraint group
    fn resolve_group(&self, group: &ConstraintGroup) -> Option<PhysicsConstraint> {
        if group.constraints.is_empty() {
            return None;
        }

        // If only one constraint, return it
        if group.constraints.len() == 1 {
            return Some(group.constraints[0].clone());
        }

        // If any constraint is user-defined, use highest priority (lowest number)
        if group.has_user_defined() {
            return group.get_highest_priority().cloned();
        }

        // Otherwise, use weighted blending
        self.blend_constraints(&group.constraints)
    }

    /// Blend multiple constraints using priority-weighted averaging
    fn blend_constraints(&self, constraints: &[PhysicsConstraint]) -> Option<PhysicsConstraint> {
        if constraints.is_empty() {
            return None;
        }

        // Group by constraint type (can only blend same types)
        let mut separation_constraints = Vec::new();
        let mut clustering_constraints = Vec::new();
        let mut colocation_constraints = Vec::new();
        let mut boundary_constraints = Vec::new();
        let mut hierarchical_constraints = Vec::new();
        let mut containment_constraints = Vec::new();

        for constraint in constraints {
            match &constraint.constraint_type {
                PhysicsConstraintType::Separation { .. } => {
                    separation_constraints.push(constraint);
                }
                PhysicsConstraintType::Clustering { .. } => {
                    clustering_constraints.push(constraint);
                }
                PhysicsConstraintType::Colocation { .. } => {
                    colocation_constraints.push(constraint);
                }
                PhysicsConstraintType::Boundary { .. } => {
                    boundary_constraints.push(constraint);
                }
                PhysicsConstraintType::HierarchicalLayer { .. } => {
                    hierarchical_constraints.push(constraint);
                }
                PhysicsConstraintType::Containment { .. } => {
                    containment_constraints.push(constraint);
                }
            }
        }

        // Blend the largest group (most constraints of same type)
        let groups = [
            (separation_constraints.len(), separation_constraints),
            (clustering_constraints.len(), clustering_constraints),
            (colocation_constraints.len(), colocation_constraints),
            (boundary_constraints.len(), boundary_constraints),
            (hierarchical_constraints.len(), hierarchical_constraints),
            (containment_constraints.len(), containment_constraints),
        ];

        let largest_group = groups
            .iter()
            .max_by_key(|(count, _)| *count)
            .and_then(|(count, group)| if *count > 0 { Some(group) } else { None })?;

        self.blend_same_type_constraints(largest_group)
    }

    /// Blend constraints of the same type using weighted averaging
    fn blend_same_type_constraints(
        &self,
        constraints: &[&PhysicsConstraint],
    ) -> Option<PhysicsConstraint> {
        if constraints.is_empty() {
            return None;
        }

        let total_weight: f32 = constraints.iter().map(|c| c.priority_weight()).sum();

        if total_weight == 0.0 {
            return Some(constraints[0].clone());
        }

        let first = constraints[0];
        let nodes = first.nodes.clone();

        match &first.constraint_type {
            PhysicsConstraintType::Separation { .. } => {
                let (blended_distance, blended_strength) =
                    self.blend_distance_strength(constraints, total_weight);

                Some(PhysicsConstraint::separation(
                    nodes,
                    blended_distance,
                    blended_strength,
                    self.calculate_average_priority(constraints),
                ))
            }

            PhysicsConstraintType::Clustering { .. } => {
                let (blended_distance, blended_strength) =
                    self.blend_distance_strength(constraints, total_weight);

                Some(PhysicsConstraint::clustering(
                    nodes,
                    blended_distance,
                    blended_strength,
                    self.calculate_average_priority(constraints),
                ))
            }

            PhysicsConstraintType::Colocation { .. } => {
                let (blended_distance, blended_strength) =
                    self.blend_distance_strength(constraints, total_weight);

                Some(PhysicsConstraint::colocation(
                    nodes,
                    blended_distance,
                    blended_strength,
                    self.calculate_average_priority(constraints),
                ))
            }

            PhysicsConstraintType::Boundary { .. } => {
                let (blended_bounds, blended_strength) =
                    self.blend_boundary(constraints, total_weight);

                Some(PhysicsConstraint::boundary(
                    nodes,
                    blended_bounds,
                    blended_strength,
                    self.calculate_average_priority(constraints),
                ))
            }

            PhysicsConstraintType::HierarchicalLayer { .. } => {
                let (blended_z, blended_strength) =
                    self.blend_hierarchical(constraints, total_weight);

                Some(PhysicsConstraint::hierarchical_layer(
                    nodes,
                    blended_z,
                    blended_strength,
                    self.calculate_average_priority(constraints),
                ))
            }

            PhysicsConstraintType::Containment { .. } => {
                // For containment, use highest priority (can't blend parent node)
                Some(
                    constraints
                        .iter()
                        .min_by_key(|c| c.priority)
                        .unwrap()
                        .clone(),
                )
            }
        }
    }

    /// Blend distance and strength parameters
    fn blend_distance_strength(
        &self,
        constraints: &[&PhysicsConstraint],
        total_weight: f32,
    ) -> (f32, f32) {
        let mut blended_distance = 0.0;
        let mut blended_strength = 0.0;

        for constraint in constraints {
            let weight = constraint.priority_weight();
            let (distance, strength) = match &constraint.constraint_type {
                PhysicsConstraintType::Separation { min_distance, strength } => {
                    (*min_distance, *strength)
                }
                PhysicsConstraintType::Clustering { ideal_distance, stiffness } => {
                    (*ideal_distance, *stiffness)
                }
                PhysicsConstraintType::Colocation { target_distance, strength } => {
                    (*target_distance, *strength)
                }
                _ => continue,
            };

            blended_distance += weight * distance;
            blended_strength += weight * strength;
        }

        (
            blended_distance / total_weight,
            blended_strength / total_weight,
        )
    }

    /// Blend boundary parameters
    fn blend_boundary(
        &self,
        constraints: &[&PhysicsConstraint],
        total_weight: f32,
    ) -> ([f32; 6], f32) {
        let mut blended_bounds = [0.0; 6];
        let mut blended_strength = 0.0;

        for constraint in constraints {
            let weight = constraint.priority_weight();
            if let PhysicsConstraintType::Boundary { bounds, strength } = &constraint.constraint_type
            {
                for i in 0..6 {
                    blended_bounds[i] += weight * bounds[i];
                }
                blended_strength += weight * strength;
            }
        }

        for i in 0..6 {
            blended_bounds[i] /= total_weight;
        }
        blended_strength /= total_weight;

        (blended_bounds, blended_strength)
    }

    /// Blend hierarchical layer parameters
    fn blend_hierarchical(
        &self,
        constraints: &[&PhysicsConstraint],
        total_weight: f32,
    ) -> (f32, f32) {
        let mut blended_z = 0.0;
        let mut blended_strength = 0.0;

        for constraint in constraints {
            let weight = constraint.priority_weight();
            if let PhysicsConstraintType::HierarchicalLayer { z_level, strength } =
                &constraint.constraint_type
            {
                blended_z += weight * z_level;
                blended_strength += weight * strength;
            }
        }

        (blended_z / total_weight, blended_strength / total_weight)
    }

    /// Calculate average priority (weighted by priority weights)
    fn calculate_average_priority(&self, constraints: &[&PhysicsConstraint]) -> u8 {
        let total_weight: f32 = constraints.iter().map(|c| c.priority_weight()).sum();

        if total_weight == 0.0 {
            return constraints[0].priority;
        }

        let weighted_priority: f32 = constraints
            .iter()
            .map(|c| c.priority_weight() * c.priority as f32)
            .sum();

        (weighted_priority / total_weight).round() as u8
    }

    /// Get all constraint groups
    pub fn get_groups(&self) -> Vec<&ConstraintGroup> {
        self.constraint_groups.values().collect()
    }

    /// Get conflicts (groups with multiple constraints)
    pub fn get_conflicts(&self) -> Vec<&ConstraintGroup> {
        self.constraint_groups
            .values()
            .filter(|g| g.has_conflicts())
            .collect()
    }

    /// Clear all constraints
    pub fn clear(&mut self) {
        self.constraint_groups.clear();
    }
}

impl Default for PriorityResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_pair_creation() {
        let pair1 = NodePair::new(1, 2);
        let pair2 = NodePair::new(2, 1);

        assert_eq!(pair1, pair2); // Order-independent
        assert!(pair1.contains(1));
        assert!(pair1.contains(2));
        assert!(!pair1.contains(3));
    }

    #[test]
    fn test_no_conflict_single_constraint() {
        let mut resolver = PriorityResolver::new();
        let constraint = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);

        resolver.add_constraint(constraint.clone());

        let resolved = resolver.resolve();
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].nodes, vec![1, 2]);
    }

    #[test]
    fn test_user_defined_override() {
        let mut resolver = PriorityResolver::new();

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 20.0, 0.8, 1)
            .mark_user_defined();

        resolver.add_constraints(vec![constraint1, constraint2]);

        let resolved = resolver.resolve();
        assert_eq!(resolved.len(), 1);

        // User-defined constraint should win
        match &resolved[0].constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                assert_eq!(*min_distance, 20.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_weighted_blending() {
        let mut resolver = PriorityResolver::new();

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 1); // weight = 1.0
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 20.0, 0.7, 5); // weight ≈ 0.5

        resolver.add_constraints(vec![constraint1, constraint2]);

        let resolved = resolver.resolve();
        assert_eq!(resolved.len(), 1);

        match &resolved[0].constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                // Blended: (1.0*10.0 + 0.5*20.0) / (1.0 + 0.5) ≈ 13.33
                assert!(*min_distance > 10.0 && *min_distance < 20.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_different_constraint_types() {
        let mut resolver = PriorityResolver::new();

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let constraint2 = PhysicsConstraint::clustering(vec![1, 2], 20.0, 0.6, 5);

        resolver.add_constraints(vec![constraint1, constraint2]);

        let groups = resolver.get_conflicts();
        assert_eq!(groups.len(), 1);
        assert!(groups[0].has_conflicts());
    }

    #[test]
    fn test_priority_weight_calculation() {
        let c1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 1);
        let c2 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let c10 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 10);

        let w1 = c1.priority_weight();
        let w5 = c2.priority_weight();
        let w10 = c10.priority_weight();

        assert!((w1 - 1.0).abs() < 0.001);
        assert!((w10 - 0.1).abs() < 0.001);
        assert!(w1 > w5);
        assert!(w5 > w10);
    }

    #[test]
    fn test_boundary_blending() {
        let mut resolver = PriorityResolver::new();

        let bounds1 = [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0];
        let bounds2 = [-20.0, 20.0, -20.0, 20.0, -20.0, 20.0];

        let constraint1 = PhysicsConstraint::boundary(vec![1], bounds1, 0.5, 1);
        let constraint2 = PhysicsConstraint::boundary(vec![1], bounds2, 0.7, 5);

        resolver.add_constraints(vec![constraint1, constraint2]);

        let resolved = resolver.resolve();
        assert_eq!(resolved.len(), 1);

        match &resolved[0].constraint_type {
            PhysicsConstraintType::Boundary { bounds, .. } => {
                // Blended bounds should be between bounds1 and bounds2
                assert!(bounds[0] > -20.0 && bounds[0] < -10.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_clear_constraints() {
        let mut resolver = PriorityResolver::new();
        resolver.add_constraint(PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5));

        assert_eq!(resolver.constraint_groups.len(), 1);

        resolver.clear();
        assert_eq!(resolver.constraint_groups.len(), 0);
    }

    #[test]
    fn test_get_conflicts() {
        let mut resolver = PriorityResolver::new();

        resolver.add_constraint(PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5));
        resolver.add_constraint(PhysicsConstraint::separation(vec![1, 2], 20.0, 0.7, 3));
        resolver.add_constraint(PhysicsConstraint::separation(vec![3, 4], 15.0, 0.6, 5));

        let conflicts = resolver.get_conflicts();
        assert_eq!(conflicts.len(), 1); // Only (1,2) has conflict
    }
}
