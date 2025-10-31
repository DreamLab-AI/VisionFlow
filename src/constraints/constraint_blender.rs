// Constraint Blender - Merge Conflicting Constraints
// Week 3 Deliverable: Advanced Blending Algorithms

use super::physics_constraint::*;
use super::priority_resolver::*;

/// Blending strategy for constraint resolution
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendingStrategy {
    /// Weighted average based on priority weights
    WeightedAverage,

    /// Take maximum value (strongest constraint wins)
    Maximum,

    /// Take minimum value (weakest constraint wins)
    Minimum,

    /// Highest priority only (no blending)
    HighestPriority,

    /// Median value (robust to outliers)
    Median,
}

impl Default for BlendingStrategy {
    fn default() -> Self {
        Self::WeightedAverage
    }
}

/// Configuration for constraint blending
#[derive(Debug, Clone)]
pub struct BlenderConfig {
    pub strategy: BlendingStrategy,

    /// Threshold for considering constraints as conflicting
    /// If distance difference < threshold, don't blend
    pub conflict_threshold: f32,

    /// Whether to preserve user-defined constraints unchanged
    pub preserve_user_defined: bool,

    /// Whether to normalize weights before blending
    pub normalize_weights: bool,
}

impl Default for BlenderConfig {
    fn default() -> Self {
        Self {
            strategy: BlendingStrategy::WeightedAverage,
            conflict_threshold: 5.0,
            preserve_user_defined: true,
            normalize_weights: true,
        }
    }
}

/// Advanced constraint blender with multiple strategies
pub struct ConstraintBlender {
    config: BlenderConfig,
}

impl ConstraintBlender {
    /// Create a new constraint blender with default configuration
    pub fn new() -> Self {
        Self {
            config: BlenderConfig::default(),
        }
    }

    /// Create a new constraint blender with custom configuration
    pub fn with_config(config: BlenderConfig) -> Self {
        Self { config }
    }

    /// Blend multiple constraints for the same node pair
    pub fn blend_constraints(
        &self,
        constraints: &[PhysicsConstraint],
    ) -> Option<PhysicsConstraint> {
        if constraints.is_empty() {
            return None;
        }

        // Single constraint - return as-is
        if constraints.len() == 1 {
            return Some(constraints[0].clone());
        }

        // User-defined constraint present - use it if config says so
        if self.config.preserve_user_defined {
            if let Some(user_constraint) = constraints.iter().find(|c| c.user_defined) {
                return Some(user_constraint.clone());
            }
        }

        // Check if constraints are similar enough to not need blending
        if !self.has_significant_conflict(constraints) {
            return Some(
                constraints
                    .iter()
                    .min_by_key(|c| c.priority)
                    .unwrap()
                    .clone(),
            );
        }

        // Group by constraint type
        let separation_constraints: Vec<_> = constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, PhysicsConstraintType::Separation { .. }))
            .collect();

        let clustering_constraints: Vec<_> = constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, PhysicsConstraintType::Clustering { .. }))
            .collect();

        let colocation_constraints: Vec<_> = constraints
            .iter()
            .filter(|c| matches!(c.constraint_type, PhysicsConstraintType::Colocation { .. }))
            .collect();

        // Blend the largest group
        if !separation_constraints.is_empty() {
            self.blend_separation_constraints(&separation_constraints)
        } else if !clustering_constraints.is_empty() {
            self.blend_clustering_constraints(&clustering_constraints)
        } else if !colocation_constraints.is_empty() {
            self.blend_colocation_constraints(&colocation_constraints)
        } else {
            // For other types, use highest priority
            Some(
                constraints
                    .iter()
                    .min_by_key(|c| c.priority)
                    .unwrap()
                    .clone(),
            )
        }
    }

    /// Check if constraints have significant conflict
    fn has_significant_conflict(&self, constraints: &[PhysicsConstraint]) -> bool {
        let distances: Vec<f32> = constraints
            .iter()
            .filter_map(|c| match &c.constraint_type {
                PhysicsConstraintType::Separation { min_distance, .. } => Some(*min_distance),
                PhysicsConstraintType::Clustering { ideal_distance, .. } => Some(*ideal_distance),
                PhysicsConstraintType::Colocation { target_distance, .. } => Some(*target_distance),
                _ => None,
            })
            .collect();

        if distances.len() < 2 {
            return false;
        }

        let max_distance = distances.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_distance = distances.iter().cloned().fold(f32::INFINITY, f32::min);

        (max_distance - min_distance) > self.config.conflict_threshold
    }

    /// Blend separation constraints
    fn blend_separation_constraints(
        &self,
        constraints: &[&PhysicsConstraint],
    ) -> Option<PhysicsConstraint> {
        if constraints.is_empty() {
            return None;
        }

        let nodes = constraints[0].nodes.clone();
        let (blended_distance, blended_strength) = self.blend_parameters(
            constraints,
            |c| match &c.constraint_type {
                PhysicsConstraintType::Separation { min_distance, .. } => Some(*min_distance),
                _ => None,
            },
            |c| match &c.constraint_type {
                PhysicsConstraintType::Separation { strength, .. } => Some(*strength),
                _ => None,
            },
        );

        let avg_priority = self.calculate_average_priority(constraints);

        Some(PhysicsConstraint::separation(
            nodes,
            blended_distance,
            blended_strength,
            avg_priority,
        ))
    }

    /// Blend clustering constraints
    fn blend_clustering_constraints(
        &self,
        constraints: &[&PhysicsConstraint],
    ) -> Option<PhysicsConstraint> {
        if constraints.is_empty() {
            return None;
        }

        let nodes = constraints[0].nodes.clone();
        let (blended_distance, blended_strength) = self.blend_parameters(
            constraints,
            |c| match &c.constraint_type {
                PhysicsConstraintType::Clustering { ideal_distance, .. } => Some(*ideal_distance),
                _ => None,
            },
            |c| match &c.constraint_type {
                PhysicsConstraintType::Clustering { stiffness, .. } => Some(*stiffness),
                _ => None,
            },
        );

        let avg_priority = self.calculate_average_priority(constraints);

        Some(PhysicsConstraint::clustering(
            nodes,
            blended_distance,
            blended_strength,
            avg_priority,
        ))
    }

    /// Blend colocation constraints
    fn blend_colocation_constraints(
        &self,
        constraints: &[&PhysicsConstraint],
    ) -> Option<PhysicsConstraint> {
        if constraints.is_empty() {
            return None;
        }

        let nodes = constraints[0].nodes.clone();
        let (blended_distance, blended_strength) = self.blend_parameters(
            constraints,
            |c| match &c.constraint_type {
                PhysicsConstraintType::Colocation { target_distance, .. } => Some(*target_distance),
                _ => None,
            },
            |c| match &c.constraint_type {
                PhysicsConstraintType::Colocation { strength, .. } => Some(*strength),
                _ => None,
            },
        );

        let avg_priority = self.calculate_average_priority(constraints);

        Some(PhysicsConstraint::colocation(
            nodes,
            blended_distance,
            blended_strength,
            avg_priority,
        ))
    }

    /// Generic parameter blending using configured strategy
    fn blend_parameters<F1, F2>(
        &self,
        constraints: &[&PhysicsConstraint],
        distance_extractor: F1,
        strength_extractor: F2,
    ) -> (f32, f32)
    where
        F1: Fn(&PhysicsConstraint) -> Option<f32>,
        F2: Fn(&PhysicsConstraint) -> Option<f32>,
    {
        let distances: Vec<f32> = constraints
            .iter()
            .filter_map(|c| distance_extractor(c))
            .collect();

        let strengths: Vec<f32> = constraints
            .iter()
            .filter_map(|c| strength_extractor(c))
            .collect();

        let weights: Vec<f32> = constraints.iter().map(|c| c.priority_weight()).collect();

        let blended_distance = self.blend_values(&distances, &weights);
        let blended_strength = self.blend_values(&strengths, &weights);

        (blended_distance, blended_strength)
    }

    /// Blend values using configured strategy
    fn blend_values(&self, values: &[f32], weights: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        match self.config.strategy {
            BlendingStrategy::WeightedAverage => self.weighted_average(values, weights),
            BlendingStrategy::Maximum => self.maximum(values),
            BlendingStrategy::Minimum => self.minimum(values),
            BlendingStrategy::HighestPriority => values[0], // Assumes sorted by priority
            BlendingStrategy::Median => self.median(values),
        }
    }

    /// Weighted average: Σ(weight_i × value_i) / Σ(weight_i)
    fn weighted_average(&self, values: &[f32], weights: &[f32]) -> f32 {
        if values.is_empty() || weights.is_empty() {
            return 0.0;
        }

        let total_weight: f32 = weights.iter().sum();
        if total_weight == 0.0 {
            return values.iter().sum::<f32>() / values.len() as f32;
        }

        let weighted_sum: f32 = values
            .iter()
            .zip(weights.iter())
            .map(|(v, w)| v * w)
            .sum();

        if self.config.normalize_weights {
            weighted_sum / total_weight
        } else {
            weighted_sum
        }
    }

    /// Maximum value
    fn maximum(&self, values: &[f32]) -> f32 {
        values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum value
    fn minimum(&self, values: &[f32]) -> f32 {
        values.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Median value (robust to outliers)
    fn median(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Calculate average priority (weighted)
    fn calculate_average_priority(&self, constraints: &[&PhysicsConstraint]) -> u8 {
        let weights: Vec<f32> = constraints.iter().map(|c| c.priority_weight()).collect();
        let priorities: Vec<f32> = constraints.iter().map(|c| c.priority as f32).collect();

        let avg = self.weighted_average(&priorities, &weights);
        avg.round() as u8
    }
}

impl Default for ConstraintBlender {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_constraint_no_blending() {
        let blender = ConstraintBlender::new();
        let constraint = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);

        let result = blender.blend_constraints(&[constraint.clone()]);
        assert!(result.is_some());
        assert_eq!(result.unwrap().nodes, vec![1, 2]);
    }

    #[test]
    fn test_user_defined_preserved() {
        let blender = ConstraintBlender::new();

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 20.0, 0.8, 1)
            .mark_user_defined();

        let result = blender.blend_constraints(&[constraint1, constraint2.clone()]);
        assert!(result.is_some());

        match result.unwrap().constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                assert_eq!(min_distance, 20.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_weighted_average_blending() {
        let config = BlenderConfig {
            strategy: BlendingStrategy::WeightedAverage,
            preserve_user_defined: false,
            ..Default::default()
        };

        let blender = ConstraintBlender::with_config(config);

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 1); // weight = 1.0
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 20.0, 0.7, 10); // weight = 0.1

        let result = blender.blend_constraints(&[constraint1, constraint2]);
        assert!(result.is_some());

        match result.unwrap().constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                // (1.0*10 + 0.1*20) / (1.0 + 0.1) ≈ 10.91
                assert!(min_distance > 10.0 && min_distance < 12.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_maximum_strategy() {
        let config = BlenderConfig {
            strategy: BlendingStrategy::Maximum,
            preserve_user_defined: false,
            ..Default::default()
        };

        let blender = ConstraintBlender::with_config(config);

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 20.0, 0.7, 5);

        let result = blender.blend_constraints(&[constraint1, constraint2]);
        assert!(result.is_some());

        match result.unwrap().constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                assert_eq!(min_distance, 20.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_minimum_strategy() {
        let config = BlenderConfig {
            strategy: BlendingStrategy::Minimum,
            preserve_user_defined: false,
            ..Default::default()
        };

        let blender = ConstraintBlender::with_config(config);

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 20.0, 0.7, 5);

        let result = blender.blend_constraints(&[constraint1, constraint2]);
        assert!(result.is_some());

        match result.unwrap().constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                assert_eq!(min_distance, 10.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_median_strategy() {
        let config = BlenderConfig {
            strategy: BlendingStrategy::Median,
            preserve_user_defined: false,
            ..Default::default()
        };

        let blender = ConstraintBlender::with_config(config);

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 15.0, 0.6, 5);
        let constraint3 = PhysicsConstraint::separation(vec![1, 2], 20.0, 0.7, 5);

        let result = blender.blend_constraints(&[constraint1, constraint2, constraint3]);
        assert!(result.is_some());

        match result.unwrap().constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                assert_eq!(min_distance, 15.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_no_significant_conflict() {
        let blender = ConstraintBlender::new();

        let constraint1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let constraint2 = PhysicsConstraint::separation(vec![1, 2], 12.0, 0.6, 3);

        // Difference is 2.0, less than threshold (5.0)
        assert!(!blender.has_significant_conflict(&[constraint1.clone(), constraint2.clone()]));

        let result = blender.blend_constraints(&[constraint1, constraint2]);
        assert!(result.is_some());
    }

    #[test]
    fn test_clustering_constraint_blending() {
        let blender = ConstraintBlender::new();

        let constraint1 = PhysicsConstraint::clustering(vec![1, 2], 20.0, 0.6, 3);
        let constraint2 = PhysicsConstraint::clustering(vec![1, 2], 30.0, 0.8, 5);

        let result = blender.blend_constraints(&[constraint1, constraint2]);
        assert!(result.is_some());

        match result.unwrap().constraint_type {
            PhysicsConstraintType::Clustering { ideal_distance, .. } => {
                assert!(ideal_distance > 20.0 && ideal_distance < 30.0);
            }
            _ => panic!("Wrong type"),
        }
    }

    #[test]
    fn test_empty_constraints() {
        let blender = ConstraintBlender::new();
        let result = blender.blend_constraints(&[]);
        assert!(result.is_none());
    }
}
