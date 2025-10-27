// src/inference/mod.rs
//! Inference Module
//!
//! Provides OWL 2 DL ontology reasoning and inference capabilities using whelk-rs.
//! This module includes OWL parsers, inference types, caching, and optimization.

pub mod owl_parser;
pub mod types;
pub mod cache;
pub mod optimization;

pub use owl_parser::{OWLParser, OWLFormat, ParseResult, ParseError};
pub use types::{
    Inference, InferenceType, InferenceExplanation, ValidationResult,
    ClassificationResult, ConsistencyReport, UnsatisfiableClass
};
pub use cache::{InferenceCache, CacheConfig, CacheEntry, CacheStatistics};
pub use optimization::{
    InferenceOptimizer, BatchInferenceRequest, IncrementalInference,
    ParallelClassification, OptimizationMetrics
};
