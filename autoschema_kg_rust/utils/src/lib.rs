//! Comprehensive data processing utilities for AutoSchema Knowledge Graph
//!
//! This crate provides efficient, memory-optimized utilities for processing
//! CSV, JSON, GraphML files and performing various data transformations.

use thiserror::Error;

/// Common error types for all utility modules
#[derive(Error, Debug)]
pub enum UtilsError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV processing error: {0}")]
    Csv(#[from] csv::Error),
    #[error("JSON processing error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("XML processing error: {0}")]
    Xml(#[from] quick_xml::Error),
    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    #[error("Custom error: {0}")]
    Custom(String),
}

pub type Result<T> = std::result::Result<T, UtilsError>;

// Export all modules
pub mod csv_processing;
pub mod json_processing;
pub mod markdown_processing;
pub mod graph_conversion;
pub mod hash_utils;
pub mod file_io;
pub mod text_cleaning;

// Re-export commonly used types
pub use csv_processing::*;
pub use json_processing::*;
pub use markdown_processing::*;
pub use graph_conversion::*;
pub use hash_utils::*;
pub use file_io::*;
pub use text_cleaning::*;
