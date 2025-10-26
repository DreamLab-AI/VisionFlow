// src/adapters/mod.rs
//! Hexagonal Architecture Adapters
//!
//! This module contains adapters that implement the port interfaces
//! using concrete technologies (actors, GPU compute, SQLite, etc.)

// Legacy adapters
// pub mod actor_graph_repository;  // REMOVED: Incomplete stub adapter
// pub mod gpu_physics_adapter;  // REMOVED: Incomplete stub adapter
pub mod gpu_semantic_analyzer;

// New hexser-based adapters
pub mod sqlite_knowledge_graph_repository;
pub mod sqlite_ontology_repository;
pub mod sqlite_settings_repository;
pub mod whelk_inference_engine;

// Legacy exports
// pub use actor_graph_repository::ActorGraphRepository;  // REMOVED: Incomplete stub

// GPU adapter implementation exports (these implement the traits from crate::ports)
// pub use gpu_physics_adapter::GpuPhysicsAdapter as GpuPhysicsAdapterImpl;  // REMOVED: Incomplete stub
pub use gpu_semantic_analyzer::GpuSemanticAnalyzerAdapter;

// New hexser-based adapter exports
pub use sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
pub use sqlite_ontology_repository::SqliteOntologyRepository;
pub use sqlite_settings_repository::SqliteSettingsRepository;
pub use whelk_inference_engine::WhelkInferenceEngine;
