// src/adapters/mod.rs
//! Hexagonal Architecture Adapters
//!
//! This module contains adapters that implement the port interfaces
//! using concrete technologies (actors, GPU compute, etc.)

pub mod actor_graph_repository;
pub mod gpu_physics_adapter;
pub mod gpu_semantic_analyzer;

pub use actor_graph_repository::ActorGraphRepository;
pub use gpu_physics_adapter::GpuPhysicsAdapter;
pub use gpu_semantic_analyzer::GpuSemanticAnalyzer;
