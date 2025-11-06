// src/repositories/mod.rs
//! Repository Utilities
//!
//! This module contains utility modules for repository implementations.
//!
//! NOTE: UnifiedOntologyRepository has been deprecated and replaced with
//! Neo4jOntologyRepository as part of the SQL deprecation effort.
//! See ADR-001 for details.

pub mod query_builder;

pub use query_builder::{BatchQueryBuilder, QueryBuilder, SqlValue};
