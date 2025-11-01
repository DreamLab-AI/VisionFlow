// src/cqrs/mod.rs
//! CQRS (Command Query Responsibility Segregation) Application Layer
//!
//! This module implements the CQRS pattern to separate read and write operations.
//! It provides a clean application layer between API handlers and repositories/adapters.
//!
//! # Architecture
//!
//! ```text
//! API Handlers
//!     ↓
//! Command/Query Bus
//!     ↓
//! Command/Query Handlers
//!     ↓
//! Repositories/Adapters
//! ```
//!
//! # Usage
//!
//! ```rust
//! use crate::cqrs::{CommandBus, QueryBus};
//! use crate::cqrs::commands::AddNodeCommand;
//! use crate::cqrs::queries::GetNodeQuery;
//!
//! 
//! let command = AddNodeCommand { node };
//! let node_id = command_bus.execute(command).await?;
//!
//! 
//! let query = GetNodeQuery { node_id };
//! let node = query_bus.execute(query).await?;
//! ```

pub mod bus;
pub mod commands;
pub mod handlers;
pub mod queries;
pub mod types;

// Re-export main types
pub use bus::{CommandBus, QueryBus};
pub use types::{Command, CommandHandler, Query, QueryHandler, Result};
