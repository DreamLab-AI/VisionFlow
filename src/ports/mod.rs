// src/ports/mod.rs
//! Hexagonal Architecture Ports
//!
//! This module defines the port interfaces (traits) that represent
//! the core application boundaries. These are technology-agnostic
//! interfaces that the domain logic depends on.

// Legacy ports (to be refactored)
pub mod graph_repository;
pub mod physics_simulator;
pub mod semantic_analyzer;

// New hexser-based ports
pub mod inference_engine;
pub mod knowledge_graph_repository;
pub mod ontology_repository;
pub mod settings_repository;

// GPU port trait definitions
pub mod gpu_physics_adapter;
pub mod gpu_semantic_analyzer;

// Legacy exports
pub use graph_repository::GraphRepository;
pub use physics_simulator::PhysicsSimulator;
pub use semantic_analyzer::SemanticAnalyzer;

// New hexser-based exports
pub use inference_engine::InferenceEngine;
pub use knowledge_graph_repository::KnowledgeGraphRepository;
pub use ontology_repository::OntologyRepository;
pub use settings_repository::SettingsRepository;

// GPU port trait exports (these are the TRAITS, not the implementations)
pub use gpu_physics_adapter::{
    GpuDeviceInfo, GpuPhysicsAdapter, GpuPhysicsAdapterError, NodeForce, PhysicsParameters,
    PhysicsStatistics, PhysicsStepResult,
};
pub use gpu_semantic_analyzer::{
    ClusteringAlgorithm, CommunityDetectionResult, GpuSemanticAnalyzer, GpuSemanticAnalyzerError,
    ImportanceAlgorithm, OptimizationResult, PathfindingResult, SemanticConstraintConfig,
    SemanticStatistics,
};
