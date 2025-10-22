//! GPU actor module for specialized GPU computation actors
//! Only available when the "gpu" feature is enabled

#[cfg(feature = "gpu")]
pub mod anomaly_detection_actor;
#[cfg(feature = "gpu")]
pub mod clustering_actor;
#[cfg(feature = "gpu")]
pub mod constraint_actor;
#[cfg(feature = "gpu")]
pub mod cuda_stream_wrapper;
#[cfg(feature = "gpu")]
pub mod force_compute_actor;
#[cfg(feature = "gpu")]
pub mod gpu_manager_actor;
#[cfg(feature = "gpu")]
pub mod gpu_resource_actor;
#[cfg(feature = "gpu")]
pub mod ontology_constraint_actor;
#[cfg(feature = "gpu")]
pub mod shared;
#[cfg(feature = "gpu")]
pub mod stress_majorization_actor;

#[cfg(feature = "gpu")]
pub use anomaly_detection_actor::AnomalyDetectionActor;
#[cfg(feature = "gpu")]
pub use clustering_actor::ClusteringActor;
#[cfg(feature = "gpu")]
pub use constraint_actor::ConstraintActor;
#[cfg(feature = "gpu")]
pub use force_compute_actor::ForceComputeActor;
#[cfg(feature = "gpu")]
pub use gpu_manager_actor::GPUManagerActor;
#[cfg(feature = "gpu")]
pub use gpu_resource_actor::GPUResourceActor;
#[cfg(feature = "gpu")]
pub use ontology_constraint_actor::OntologyConstraintActor;
#[cfg(feature = "gpu")]
pub use stress_majorization_actor::StressMajorizationActor;

// Re-export shared types for convenience
#[cfg(feature = "gpu")]
pub use shared::{ChildActorAddresses, GPUState, SharedGPUContext, StressMajorizationSafety};
