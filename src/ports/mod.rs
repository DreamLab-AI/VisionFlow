// src/ports/mod.rs
//! Hexagonal Architecture Ports
//!
//! This module defines the port interfaces (traits) that represent
//! the core application boundaries. These are technology-agnostic
//! interfaces that the domain logic depends on.

pub mod graph_repository;
pub mod physics_simulator;
pub mod semantic_analyzer;

pub use graph_repository::GraphRepository;
pub use physics_simulator::PhysicsSimulator;
pub use semantic_analyzer::SemanticAnalyzer;
