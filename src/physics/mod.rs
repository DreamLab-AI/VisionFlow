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
//! - **Ontology Constraints**: Translates OWL axioms and logical inferences into
//!   physics constraints, bridging semantic reasoning with physical simulation to
//!   enforce ontological relationships in graph layout.
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
//! use crate::physics::{StressMajorizationSolver, SemanticConstraintGenerator, OntologyConstraintTranslator};
//! use crate::models::constraints::ConstraintSet;
//! use crate::models::graph::GraphData;
//!
//! // Create a stress majorization solver
//! let mut solver = StressMajorizationSolver::new(params);
//!
//! // Generate semantic constraints
//! let constraint_generator = SemanticConstraintGenerator::new();
//! let semantic_constraints = constraint_generator.generate_constraints(&graph_data)?;
//!
//! // Generate ontology constraints from OWL axioms
//! let mut ontology_translator = OntologyConstraintTranslator::new();
//! let ontology_constraints = ontology_translator.apply_ontology_constraints(&graph_data, &reasoning_report)?;
//!
//! // Combine and optimize layout
//! let mut combined_constraints = semantic_constraints.constraints;
//! combined_constraints.extend(ontology_constraints.constraints);
//!
//! let final_constraint_set = ConstraintSet {
//!     constraints: combined_constraints,
//!     advanced_params: ontology_constraints.advanced_params
//! };
//!
//! solver.optimize(&mut graph_data, &final_constraint_set)?;
//! ```

pub mod ontology_constraints;
pub mod semantic_constraints;
pub mod stress_majorization;

#[cfg(test)]
mod integration_tests;

pub use ontology_constraints::{
    OWLAxiom, OWLAxiomType, OntologyConstraintTranslator, OntologyInference,
    OntologyReasoningReport,
};
pub use semantic_constraints::SemanticConstraintGenerator;
pub use stress_majorization::StressMajorizationSolver;

/// Re-export core types for convenience
pub use crate::models::constraints::{AdvancedParams, Constraint, ConstraintKind, ConstraintSet};
pub use crate::models::graph::GraphData;
pub use crate::models::metadata::Metadata;
pub use crate::models::node::Node;
