//! Stub types for owl_validator when ontology feature is disabled
//!
//! This module provides minimal stub implementations to satisfy type requirements
//! when building without the ontology feature.

use serde::{Deserialize, Serialize};

/// Stub ValidationConfig for CPU-only builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub enabled: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self { enabled: false }
    }
}

/// Stub PropertyGraph for CPU-only builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<String>,
}

impl Default for PropertyGraph {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

/// Stub RdfTriple for CPU-only builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Stub ConstraintSummary for CPU-only builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSummary {
    pub total_constraints: usize,
    pub semantic_constraints: usize,
    pub structural_constraints: usize,
}

impl Default for ConstraintSummary {
    fn default() -> Self {
        Self {
            total_constraints: 0,
            semantic_constraints: 0,
            structural_constraints: 0,
        }
    }
}

/// Stub ValidationReport for CPU-only builds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub constraint_summary: ConstraintSummary,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: vec!["Ontology feature not enabled".to_string()],
            warnings: Vec::new(),
            constraint_summary: ConstraintSummary::default(),
        }
    }
}
