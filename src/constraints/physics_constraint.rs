// Physics Constraint System - Core Types
// Week 3 Deliverable: OWL Axiom → Physics Constraint Translation

use serde::{Deserialize, Serialize};
use std::fmt;

/// Node identifier (maps to graph_nodes.id)
pub type NodeId = i64;

/// Physics constraint types with configurable parameters
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PhysicsConstraintType {
    /// Separation force: Push nodes apart
    /// Used for: DisjointClasses, DisjointUnion
    Separation {
        min_distance: f32,
        strength: f32,
    },

    /// Clustering force: Pull nodes toward a target centroid
    /// Used for: SubClassOf (pull toward parent)
    Clustering {
        ideal_distance: f32,
        stiffness: f32,
    },

    /// Co-location force: Merge nodes to same location
    /// Used for: SameAs, EquivalentClasses
    Colocation {
        target_distance: f32,
        strength: f32,
    },

    /// Boundary constraint: Keep nodes within spatial bounds
    /// Used for: FunctionalProperty, domain/range restrictions
    Boundary {
        bounds: [f32; 6], // [min_x, max_x, min_y, max_y, min_z, max_z]
        strength: f32,
    },

    /// Hierarchical layering: Z-axis stratification
    /// Used for: SubClassOf hierarchy levels
    HierarchicalLayer {
        z_level: f32,
        strength: f32,
    },

    /// Containment: Keep children inside parent hull
    /// Used for: partOf, hasPart relations
    Containment {
        parent_node: NodeId,
        radius: f32,
        strength: f32,
    },
}

/// Priority levels for constraint resolution
/// Lower priority number = higher importance
pub const PRIORITY_USER_DEFINED: u8 = 1; // User overrides always win
pub const PRIORITY_INFERRED: u8 = 3;     // Reasoner-derived
pub const PRIORITY_ASSERTED: u8 = 5;     // Explicitly stated in ontology
pub const PRIORITY_DEFAULT: u8 = 8;      // Default physics behavior

/// Complete physics constraint with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConstraint {
    /// Type of constraint and its parameters
    pub constraint_type: PhysicsConstraintType,

    /// Nodes affected by this constraint
    pub nodes: Vec<NodeId>,

    /// Priority for conflict resolution (1=highest, 10=lowest)
    /// Priority 1: User-defined (always wins)
    /// Priority 2-4: Inferred (from reasoner)
    /// Priority 5: Asserted (from ontology)
    /// Priority 6-10: Default physics
    pub priority: u8,

    /// Whether this constraint was user-defined (via UI)
    pub user_defined: bool,

    /// Optional: Frame at which to activate (progressive activation)
    pub activation_frame: Option<i32>,

    /// Optional: Source axiom ID for traceability
    pub axiom_id: Option<i64>,
}

impl PhysicsConstraint {
    /// Create a new separation constraint
    pub fn separation(
        nodes: Vec<NodeId>,
        min_distance: f32,
        strength: f32,
        priority: u8,
    ) -> Self {
        Self {
            constraint_type: PhysicsConstraintType::Separation {
                min_distance,
                strength,
            },
            nodes,
            priority,
            user_defined: false,
            activation_frame: None,
            axiom_id: None,
        }
    }

    /// Create a new clustering constraint
    pub fn clustering(
        nodes: Vec<NodeId>,
        ideal_distance: f32,
        stiffness: f32,
        priority: u8,
    ) -> Self {
        Self {
            constraint_type: PhysicsConstraintType::Clustering {
                ideal_distance,
                stiffness,
            },
            nodes,
            priority,
            user_defined: false,
            activation_frame: None,
            axiom_id: None,
        }
    }

    /// Create a new colocation constraint
    pub fn colocation(
        nodes: Vec<NodeId>,
        target_distance: f32,
        strength: f32,
        priority: u8,
    ) -> Self {
        Self {
            constraint_type: PhysicsConstraintType::Colocation {
                target_distance,
                strength,
            },
            nodes,
            priority,
            user_defined: false,
            activation_frame: None,
            axiom_id: None,
        }
    }

    /// Create a new boundary constraint
    pub fn boundary(
        nodes: Vec<NodeId>,
        bounds: [f32; 6],
        strength: f32,
        priority: u8,
    ) -> Self {
        Self {
            constraint_type: PhysicsConstraintType::Boundary { bounds, strength },
            nodes,
            priority,
            user_defined: false,
            activation_frame: None,
            axiom_id: None,
        }
    }

    /// Create a new hierarchical layer constraint
    pub fn hierarchical_layer(
        nodes: Vec<NodeId>,
        z_level: f32,
        strength: f32,
        priority: u8,
    ) -> Self {
        Self {
            constraint_type: PhysicsConstraintType::HierarchicalLayer { z_level, strength },
            nodes,
            priority,
            user_defined: false,
            activation_frame: None,
            axiom_id: None,
        }
    }

    /// Create a new containment constraint
    pub fn containment(
        nodes: Vec<NodeId>,
        parent_node: NodeId,
        radius: f32,
        strength: f32,
        priority: u8,
    ) -> Self {
        Self {
            constraint_type: PhysicsConstraintType::Containment {
                parent_node,
                radius,
                strength,
            },
            nodes,
            priority,
            user_defined: false,
            activation_frame: None,
            axiom_id: None,
        }
    }

    /// Mark this constraint as user-defined (highest priority)
    pub fn mark_user_defined(mut self) -> Self {
        self.user_defined = true;
        self.priority = PRIORITY_USER_DEFINED;
        self
    }

    /// Link this constraint to a source axiom
    pub fn with_axiom_id(mut self, axiom_id: i64) -> Self {
        self.axiom_id = Some(axiom_id);
        self
    }

    /// Set activation frame for progressive constraint activation
    pub fn with_activation_frame(mut self, frame: i32) -> Self {
        self.activation_frame = Some(frame);
        self
    }

    /// Calculate priority weight for conflict resolution
    /// Weight = 10^(-(priority-1)/9)
    /// Priority 1 → weight = 1.0
    /// Priority 10 → weight = 0.1
    pub fn priority_weight(&self) -> f32 {
        10.0_f32.powf(-(self.priority as f32 - 1.0) / 9.0)
    }

    /// Get the number of nodes affected by this constraint
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if this constraint affects the given node
    pub fn affects_node(&self, node_id: NodeId) -> bool {
        self.nodes.contains(&node_id)
    }

    /// Get the constraint strength (varies by type)
    pub fn strength(&self) -> f32 {
        match &self.constraint_type {
            PhysicsConstraintType::Separation { strength, .. } => *strength,
            PhysicsConstraintType::Clustering { stiffness, .. } => *stiffness,
            PhysicsConstraintType::Colocation { strength, .. } => *strength,
            PhysicsConstraintType::Boundary { strength, .. } => *strength,
            PhysicsConstraintType::HierarchicalLayer { strength, .. } => *strength,
            PhysicsConstraintType::Containment { strength, .. } => *strength,
        }
    }
}

impl fmt::Display for PhysicsConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let type_str = match &self.constraint_type {
            PhysicsConstraintType::Separation { min_distance, .. } => {
                format!("Separation(min_dist={})", min_distance)
            }
            PhysicsConstraintType::Clustering { ideal_distance, .. } => {
                format!("Clustering(ideal_dist={})", ideal_distance)
            }
            PhysicsConstraintType::Colocation { target_distance, .. } => {
                format!("Colocation(target_dist={})", target_distance)
            }
            PhysicsConstraintType::Boundary { bounds, .. } => {
                format!("Boundary({:?})", bounds)
            }
            PhysicsConstraintType::HierarchicalLayer { z_level, .. } => {
                format!("HierarchicalLayer(z={})", z_level)
            }
            PhysicsConstraintType::Containment { parent_node, radius, .. } => {
                format!("Containment(parent={}, r={})", parent_node, radius)
            }
        };

        write!(
            f,
            "{} [nodes={}, priority={}, user={}]",
            type_str,
            self.nodes.len(),
            self.priority,
            self.user_defined
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separation_constraint_creation() {
        let constraint = PhysicsConstraint::separation(
            vec![1, 2],
            35.0,
            0.8,
            PRIORITY_ASSERTED,
        );

        assert_eq!(constraint.nodes.len(), 2);
        assert_eq!(constraint.priority, PRIORITY_ASSERTED);
        assert!(!constraint.user_defined);

        match constraint.constraint_type {
            PhysicsConstraintType::Separation { min_distance, strength } => {
                assert_eq!(min_distance, 35.0);
                assert_eq!(strength, 0.8);
            }
            _ => panic!("Wrong constraint type"),
        }
    }

    #[test]
    fn test_priority_weight_calculation() {
        let c1 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 1);
        let c2 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5);
        let c3 = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 10);

        assert!((c1.priority_weight() - 1.0).abs() < 0.001);
        assert!(c1.priority_weight() > c2.priority_weight());
        assert!(c2.priority_weight() > c3.priority_weight());
        assert!((c3.priority_weight() - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_user_defined_override() {
        let constraint = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5)
            .mark_user_defined();

        assert!(constraint.user_defined);
        assert_eq!(constraint.priority, PRIORITY_USER_DEFINED);
    }

    #[test]
    fn test_clustering_constraint() {
        let constraint = PhysicsConstraint::clustering(
            vec![10, 20, 30],
            20.0,
            0.6,
            PRIORITY_INFERRED,
        );

        assert_eq!(constraint.nodes.len(), 3);
        assert_eq!(constraint.priority, PRIORITY_INFERRED);
        assert_eq!(constraint.strength(), 0.6);
    }

    #[test]
    fn test_constraint_affects_node() {
        let constraint = PhysicsConstraint::separation(vec![1, 2, 3], 10.0, 0.5, 5);

        assert!(constraint.affects_node(1));
        assert!(constraint.affects_node(2));
        assert!(constraint.affects_node(3));
        assert!(!constraint.affects_node(4));
    }

    #[test]
    fn test_with_axiom_id() {
        let constraint = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5)
            .with_axiom_id(123);

        assert_eq!(constraint.axiom_id, Some(123));
    }

    #[test]
    fn test_with_activation_frame() {
        let constraint = PhysicsConstraint::separation(vec![1, 2], 10.0, 0.5, 5)
            .with_activation_frame(60);

        assert_eq!(constraint.activation_frame, Some(60));
    }

    #[test]
    fn test_hierarchical_layer_constraint() {
        let constraint = PhysicsConstraint::hierarchical_layer(
            vec![1, 2, 3],
            100.0,
            0.7,
            PRIORITY_ASSERTED,
        );

        match constraint.constraint_type {
            PhysicsConstraintType::HierarchicalLayer { z_level, strength } => {
                assert_eq!(z_level, 100.0);
                assert_eq!(strength, 0.7);
            }
            _ => panic!("Wrong constraint type"),
        }
    }

    #[test]
    fn test_containment_constraint() {
        let constraint = PhysicsConstraint::containment(
            vec![1, 2, 3],
            100,
            50.0,
            0.8,
            PRIORITY_ASSERTED,
        );

        match constraint.constraint_type {
            PhysicsConstraintType::Containment { parent_node, radius, strength } => {
                assert_eq!(parent_node, 100);
                assert_eq!(radius, 50.0);
                assert_eq!(strength, 0.8);
            }
            _ => panic!("Wrong constraint type"),
        }
    }
}
