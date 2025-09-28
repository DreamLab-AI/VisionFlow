//! Neo4j integration module for knowledge graph construction
//!
//! This module provides comprehensive Neo4j database operations for knowledge graph
//! storage and retrieval, including connection management, query building, and
//! batch operations with proper async handling.

pub mod connection;
pub mod query_builder;
pub mod operations;
pub mod batch;
pub mod models;
pub mod error;
pub mod index;
pub mod hooks;

pub use connection::Neo4jConnectionManager;
pub use query_builder::{CypherQueryBuilder, QueryBuilder};
pub use operations::{NodeOperations, RelationshipOperations, GraphOperations};
pub use batch::BatchProcessor;
pub use models::{Triple, Node, Relationship, GraphData};
pub use error::{Neo4jError, Result};
pub use index::IndexManager;
pub use hooks::{Neo4jHooks, HookEvent, HookHelpers};

/// Re-export commonly used types
pub use neo4rs::{Graph, Row, Node as Neo4jNode, Relation as Neo4jRelation};
