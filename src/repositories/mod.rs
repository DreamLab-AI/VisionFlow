// src/repositories/mod.rs
//! Unified Repository Adapters
//!
//! This module contains repository adapters that implement the
//! KnowledgeGraphRepository and OntologyRepository ports using
//! the unified.db schema. These adapters provide 100% API compatibility
//! with the legacy SQLite adapters while using a single unified database.

pub mod unified_graph_repository;
pub mod unified_ontology_repository;

pub use unified_graph_repository::UnifiedGraphRepository;
pub use unified_ontology_repository::UnifiedOntologyRepository;
