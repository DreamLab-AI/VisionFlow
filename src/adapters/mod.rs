// src/adapters/mod.rs
//! Hexagonal Architecture Adapters
//!
//! This module contains adapters that implement the port interfaces
//! using concrete technologies (actors, GPU compute, SQLite, etc.)

// CQRS Phase 1: Actor-based adapter for gradual migration
pub mod actor_graph_repository;

// Legacy adapters
// pub mod gpu_physics_adapter;  // REMOVED: Incomplete stub adapter
pub mod gpu_semantic_analyzer;

// New hexser-based adapters
pub mod sqlite_knowledge_graph_repository;
pub mod sqlite_ontology_repository;
pub mod sqlite_settings_repository;
pub mod whelk_inference_engine;

// Phase 2.2: Actor system adapter wrappers
pub mod actix_physics_adapter;
pub mod actix_semantic_adapter;
pub mod messages;
pub mod whelk_inference_stub;

// Compatibility alias for physics orchestrator adapter
pub mod physics_orchestrator_adapter;

// CQRS Phase 1: Actor-based adapter exports
pub use actor_graph_repository::ActorGraphRepository;

// GPU adapter implementation exports (these implement the traits from crate::ports)
// pub use gpu_physics_adapter::GpuPhysicsAdapter as GpuPhysicsAdapterImpl;  // REMOVED: Incomplete stub
pub use gpu_semantic_analyzer::GpuSemanticAnalyzerAdapter;

// New hexser-based adapter exports
pub use sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
pub use sqlite_ontology_repository::SqliteOntologyRepository;
pub use sqlite_settings_repository::SqliteSettingsRepository;
pub use whelk_inference_engine::WhelkInferenceEngine;

// Phase 2.2: Actor wrapper adapter exports
pub use actix_physics_adapter::ActixPhysicsAdapter;
pub use actix_semantic_adapter::ActixSemanticAdapter;
pub use whelk_inference_stub::WhelkInferenceEngineStub;
