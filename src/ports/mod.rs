// Ports (Interfaces) - Hexagonal Architecture
// These traits define the boundaries between application and infrastructure

pub mod graph_repository;
pub mod physics_simulator;
pub mod semantic_analyzer;

pub use graph_repository::GraphRepository;
pub use physics_simulator::PhysicsSimulator;
pub use semantic_analyzer::{SemanticAnalyzer, SSSPResult, ClusteringResult, CommunityResult, ClusterAlgorithm};
