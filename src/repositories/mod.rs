// src/repositories/mod.rs
//! Unified Repository Adapters
//!
//! This module contains repository adapters that implement the
//! KnowledgeGraphRepository and OntologyRepository ports using
//! the unified.db schema. These adapters provide 100% API compatibility
//! with the legacy SQLite adapters while using a single unified database.

pub mod query_builder;
pub mod unified_ontology_repository;

pub use query_builder::{BatchQueryBuilder, QueryBuilder, SqlValue};
pub use unified_ontology_repository::UnifiedOntologyRepository;
