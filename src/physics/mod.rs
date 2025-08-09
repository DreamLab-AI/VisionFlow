//! Physics engine modules for advanced knowledge graph layout optimization
//!
//! This module provides sophisticated physics-based algorithms for knowledge graph
//! layout optimization, including stress majorization and semantic constraint generation.
//! The physics engine integrates with the GPU compute pipeline for high-performance
//! real-time graph visualization and layout optimization.
//!
//! ## Architecture
//!
//! The physics module is organized into specialized components:
//! 
//! - **Stress Majorization**: Implements stress majorization algorithms for global
//!   layout optimization, minimizing the stress function to achieve visually pleasing
//!   node positions that satisfy multiple constraint types.
//!
//! - **Semantic Constraints**: Generates constraints based on semantic relationships,
//!   topic similarity, hierarchical structures, and domain knowledge to create
//!   meaningful spatial arrangements.
//!
//! ## Integration
//!
//! This module integrates with:
//! - GPU compute kernels for high-performance matrix operations
//! - Constraint system defined in `models::constraints`
//! - Graph data structures and node/edge representations
//! - Real-time visualization pipeline
//!
//! ## Usage
//!
//! ```rust
//! use crate::physics::{StressMajorizationSolver, SemanticConstraintGenerator};
//! use crate::models::constraints::ConstraintSet;
//! use crate::models::graph::GraphData;
//! 
//! // Create a stress majorization solver
//! let mut solver = StressMajorizationSolver::new(params);
//! 
//! // Generate semantic constraints
//! let constraint_generator = SemanticConstraintGenerator::new();
//! let constraints = constraint_generator.generate_constraints(&graph_data)?;
//! 
//! // Optimize layout
//! solver.optimize(&mut graph_data, &constraints)?;
//! ```

pub mod stress_majorization;
pub mod semantic_constraints;

pub use stress_majorization::StressMajorizationSolver;
pub use semantic_constraints::SemanticConstraintGenerator;

/// Re-export core types for convenience
pub use crate::models::constraints::{Constraint, ConstraintSet, ConstraintKind, AdvancedParams};
pub use crate::models::graph::GraphData;
pub use crate::models::node::Node;
pub use crate::models::metadata::Metadata;