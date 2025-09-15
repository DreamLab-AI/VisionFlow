//! GPU actor module for specialized GPU computation actors

pub mod shared;
pub mod gpu_manager_actor;
pub mod gpu_resource_actor;
pub mod force_compute_actor;
pub mod clustering_actor;
pub mod anomaly_detection_actor;
pub mod stress_majorization_actor;
pub mod constraint_actor;

pub use gpu_manager_actor::GPUManagerActor;
pub use gpu_resource_actor::GPUResourceActor;
pub use force_compute_actor::ForceComputeActor;
pub use clustering_actor::ClusteringActor;
pub use anomaly_detection_actor::AnomalyDetectionActor;
pub use stress_majorization_actor::StressMajorizationActor;
pub use constraint_actor::ConstraintActor;

// Re-export shared types for convenience
pub use shared::{SharedGPUContext, GPUState, StressMajorizationSafety, ChildActorAddresses};