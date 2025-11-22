// src/repositories/mod.rs
//! Repository Utilities
//!
//! This module contains utility modules for repository implementations.
//!
//! NOTE: UnifiedOntologyRepository and SQL-based repositories have been deprecated
//! and replaced with Neo4j-based repositories as part of the SQL deprecation effort.
//! See ADR-001 for details.
//!
//! The generic_repository module is restored for use by SQLite-based adapters.

pub mod generic_repository;

pub use generic_repository::{
    convert_rusqlite_error, GenericRepository, RepositoryError, Result as RepositoryResult,
    SqliteRepository,
};
