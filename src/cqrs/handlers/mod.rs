// src/cqrs/handlers/mod.rs
//! Command and Query Handlers
//!
//! Handlers implement the business logic for executing commands and queries.
//! They delegate to repositories and adapters.

pub mod graph_handlers;
pub mod ontology_handlers;
pub mod physics_handlers;
pub mod settings_handlers;

// Re-export handlers
pub use graph_handlers::{GraphCommandHandler, GraphQueryHandler};
pub use ontology_handlers::{OntologyCommandHandler, OntologyQueryHandler};
pub use physics_handlers::{PhysicsCommandHandler, PhysicsQueryHandler};
pub use settings_handlers::{SettingsCommandHandler, SettingsQueryHandler};
