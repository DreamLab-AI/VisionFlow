//! Database Migration System
//!
//! Provides version-controlled database migration infrastructure with:
//! - Transaction-safe migration execution
//! - Automatic checksum verification
//! - Rollback support with zero data loss
//! - Detailed error messages and logging
//! - Dry-run mode for testing

pub mod rollback;
pub mod runner;
pub mod version;

pub use rollback::RollbackManager;
pub use runner::MigrationRunner;
pub use version::{MigrationVersion, VersionTracker};

use thiserror::Error;

/
#[derive(Error, Debug)]
pub enum MigrationError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Migration file not found: {0}")]
    FileNotFound(String),

    #[error("Invalid migration format: {0}")]
    InvalidFormat(String),

    #[error("Checksum mismatch for migration {0}: expected {1}, got {2}")]
    ChecksumMismatch(i32, String, String),

    #[error("Migration {0} already applied")]
    AlreadyApplied(i32),

    #[error("Migration {0} not applied (cannot rollback)")]
    NotApplied(i32),

    #[error("No migrations to apply")]
    NoMigrations,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Migration parsing error: {0}")]
    Parse(String),
}

pub type Result<T> = std::result::Result<T, MigrationError>;
