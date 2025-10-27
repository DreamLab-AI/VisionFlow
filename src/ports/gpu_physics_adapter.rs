// src/ports/gpu_physics_adapter.rs
//! GPU Physics Adapter Port
//!
//! Provides GPU-accelerated physics simulation for knowledge graph layout.
//! This port abstracts CUDA/OpenCL implementations for physics computations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::graph::GraphData;
use crate::models::node::Node;

pub type Result<T> = std::result::Result<T, GpuPhysicsAdapterError>;

#[derive(Debug, thiserror::Error)]
pub enum GpuPhysicsAdapterError {
    #[error("GPU not available")]
    GpuNotAvailable,

    #[error("Physics computation error: {0}")]
    ComputationError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Graph not loaded")]
    GraphNotLoaded,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub device_id: u32,
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub total_memory_mb: usize,
    pub free_memory_mb: usize,
    pub multiprocessor_count: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
}

/// Force calculation for a single node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeForce {
    pub node_id: u32,
    pub force_x: f32,
    pub force_y: f32,
    pub force_z: f32,
}

/// Physics simulation result for a single step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsStepResult {
    pub nodes_updated: usize,
    pub total_energy: f32,
    pub max_displacement: f32,
    pub converged: bool,
    pub computation_time_ms: f32,
}

/// Physics statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsStatistics {
    pub total_steps: u64,
    pub average_step_time_ms: f32,
    pub average_energy: f32,
    pub gpu_memory_used_mb: f32,
    pub cache_hit_rate: f32,
    pub last_convergence_iterations: u32,
}

/// Physics simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParameters {
    pub time_step: f32,
    pub damping: f32,
    pub spring_constant: f32,
    pub repulsion_strength: f32,
    pub attraction_strength: f32,
    pub max_velocity: f32,
    pub convergence_threshold: f32,
    pub max_iterations: u32,
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            time_step: 0.016, // 60 FPS
            damping: 0.8,
            spring_constant: 0.01,
            repulsion_strength: 100.0,
            attraction_strength: 0.1,
            max_velocity: 10.0,
            convergence_threshold: 0.01,
            max_iterations: 1000,
        }
    }
}

/// Port for GPU physics adapter operations
#[async_trait]
pub trait GpuPhysicsAdapter: Send + Sync {
    /// Initialize physics simulation with graph data
    async fn initialize(&mut self, graph: Arc<GraphData>, params: PhysicsParameters) -> Result<()>;

    /// Compute forces for all nodes
    /// Returns forces for each node (repulsion, attraction, spring forces)
    async fn compute_forces(&mut self) -> Result<Vec<NodeForce>>;

    /// Update node positions based on computed forces
    /// Returns new positions as Vec<(node_id, x, y, z)>
    async fn update_positions(&mut self, forces: &[NodeForce])
        -> Result<Vec<(u32, f32, f32, f32)>>;

    /// Perform a complete physics simulation step (compute forces + update positions)
    /// This is an optimized version that combines both operations
    async fn step(&mut self) -> Result<PhysicsStepResult>;

    /// Run simulation until convergence or max iterations
    /// Returns the final step result
    async fn simulate_until_convergence(&mut self) -> Result<PhysicsStepResult>;

    /// Apply custom forces to specific nodes (e.g., user dragging, constraints)
    /// Format: Vec<(node_id, force_x, force_y, force_z)>
    async fn apply_external_forces(&mut self, forces: Vec<(u32, f32, f32, f32)>) -> Result<()>;

    /// Pin nodes at specific positions (prevent movement)
    /// Format: Vec<(node_id, x, y, z)>
    async fn pin_nodes(&mut self, nodes: Vec<(u32, f32, f32, f32)>) -> Result<()>;

    /// Unpin nodes to allow free movement
    async fn unpin_nodes(&mut self, node_ids: Vec<u32>) -> Result<()>;

    /// Update physics parameters without reinitializing
    async fn update_parameters(&mut self, params: PhysicsParameters) -> Result<()>;

    /// Update graph data (e.g., after nodes/edges added/removed)
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<()>;

    /// Get GPU device information
    async fn get_gpu_status(&self) -> Result<GpuDeviceInfo>;

    /// Get physics simulation statistics
    async fn get_statistics(&self) -> Result<PhysicsStatistics>;

    /// Reset simulation state (clear velocities, forces)
    async fn reset(&mut self) -> Result<()>;

    /// Free GPU resources
    async fn cleanup(&mut self) -> Result<()>;
}
