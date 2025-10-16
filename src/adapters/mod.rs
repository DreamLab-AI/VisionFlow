// Adapters module - implements hexagonal architecture adapters
// Adapters connect domain ports to actor infrastructure
// Future: Add #[derive(HexAdapter)] when Hexser available

pub mod actor_graph_repository;
pub mod gpu_physics_adapter;
pub mod gpu_semantic_analyzer;

pub use actor_graph_repository::ActorGraphRepository;
pub use gpu_physics_adapter::GpuPhysicsAdapter;
pub use gpu_semantic_analyzer::GpuSemanticAnalyzer;
