// src/cqrs/commands/mod.rs
//! Command definitions for CQRS pattern
//!
//! Commands represent write operations that modify system state.
//! All commands are validated before execution.

pub mod graph_commands;
pub mod ontology_commands;
pub mod physics_commands;
pub mod settings_commands;

// Re-export commonly used commands
pub use graph_commands::*;
pub use ontology_commands::*;
pub use physics_commands::*;
pub use settings_commands::*;
