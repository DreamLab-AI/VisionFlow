// src/cqrs/queries/mod.rs
//! Query definitions for CQRS pattern
//!
//! Queries represent read operations that do not modify system state.

pub mod graph_queries;
pub mod ontology_queries;
pub mod physics_queries;
pub mod settings_queries;

// Re-export commonly used queries
pub use graph_queries::*;
pub use ontology_queries::*;
pub use physics_queries::*;
pub use settings_queries::*;
