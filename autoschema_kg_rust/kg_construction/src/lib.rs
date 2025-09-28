//! Knowledge Graph Construction Library
//!
//! This library provides a complete pipeline for extracting knowledge graphs from text data
//! using transformer models and structured triple extraction.

// Core triple extraction pipeline modules
pub mod config;
pub mod chunker;
pub mod dataset;
pub mod loader;
pub mod parser;
pub mod extractor;
pub mod prompts;
pub mod ml_inference;

// Neo4j integration module
pub mod neo4j;
pub mod neo4j_integration;

// Existing modules for concept generation and processing
pub mod concept_generation;
pub mod concept_to_csv;
pub mod batch_processing;
pub mod data_loading;
pub mod llm_integration;
pub mod graph_traversal;
pub mod csv_utils;
pub mod statistics;
pub mod error;
pub mod types;

// Triple extraction exports
pub use config::ProcessingConfig;
pub use chunker::TextChunker;
pub use dataset::DatasetProcessor;
pub use loader::CustomDataLoader;
pub use parser::OutputParser;
pub use extractor::KnowledgeGraphExtractor;
pub use prompts::TripleInstructions;
pub use ml_inference::{ModelInference, InferenceConfig};

// Neo4j exports
pub use neo4j_integration::{Neo4jService, HealthStatus, ServiceStatistics};

// Existing exports
pub use concept_generation::ConceptGenerator;
pub use concept_to_csv::ConceptToCsv;
pub use error::{KgConstructionError, Result};
pub use types::*;

// Re-export common types
pub use serde_json::Value as JsonValue;
pub use std::collections::HashMap;
pub use futures::stream::StreamExt;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        // Triple extraction
        ProcessingConfig,
        TextChunker,
        DatasetProcessor,
        CustomDataLoader,
        OutputParser,
        KnowledgeGraphExtractor,
        TripleInstructions,
        ModelInference,
        InferenceConfig,

        // Neo4j integration
        Neo4jService,
        HealthStatus,
        ServiceStatistics,

        // Existing
        ConceptGenerator,
        ConceptToCsv,
        KgConstructionError,
        Result,
        NodeType,
        ConceptNode,
        BatchedData,
        GraphTraversal,
        Statistics,

        // Common types
        JsonValue,
        HashMap,
        StreamExt,
    };
}