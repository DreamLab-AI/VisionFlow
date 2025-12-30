//! GPU actor module for specialized GPU computation actors
//!
//! ## Architecture (Phase 7: God Actor Decomposition)
//!
//! The GPU subsystem is organized into independent subsystems that receive
//! GPU context via event bus rather than through a centralized manager:
//!
//! - **PhysicsSubsystem**: Force computation, stress majorization, constraints
//! - **AnalyticsSubsystem**: Clustering, anomaly detection, PageRank
//! - **GraphSubsystem**: Shortest path, connected components
//!
//! Each subsystem receives `SharedGPUContext` via `GPUContextBus` broadcast.

pub mod anomaly_detection_actor;
pub mod clustering_actor;
pub mod constraint_actor;
pub mod context_bus;
pub mod cuda_stream_wrapper;
pub mod force_compute_actor;
pub mod gpu_manager_actor;
pub mod gpu_resource_actor;
pub mod ontology_constraint_actor;
pub mod pagerank_actor;
pub mod shortest_path_actor;
pub mod connected_components_actor;
pub mod shared;
pub mod stress_majorization_actor;
pub mod semantic_forces_actor;

pub use anomaly_detection_actor::AnomalyDetectionActor;
pub use clustering_actor::ClusteringActor;
pub use constraint_actor::ConstraintActor;
pub use context_bus::{GPUContextBus, GPUContextReady, GPUContextSubscriber, GPUContextSubscription};
pub use force_compute_actor::ForceComputeActor;
pub use gpu_manager_actor::GPUManagerActor;
pub use gpu_resource_actor::GPUResourceActor;
pub use ontology_constraint_actor::OntologyConstraintActor;
pub use pagerank_actor::PageRankActor;
pub use shortest_path_actor::ShortestPathActor;
pub use connected_components_actor::ConnectedComponentsActor;
pub use shared::{GPUContext, SharedGPUContext, UnifiedGPUCompute};
pub use stress_majorization_actor::StressMajorizationActor;
pub use semantic_forces_actor::SemanticForcesActor;
