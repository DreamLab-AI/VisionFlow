// src/ports/gpu_physics_adapter.rs
//! GPU Physics Adapter Port
//!
//! Provides GPU-accelerated physics simulation capabilities for graph layouts.
//! This port abstracts CUDA/OpenCL implementations and CPU fallbacks.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::models::constraints::ConstraintSet;
use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;

pub type Result<T> = std::result::Result<T, GpuPhysicsAdapterError>;

#[derive(Debug, thiserror::Error)]
pub enum GpuPhysicsAdapterError {
    #[error("GPU not available")]
    GpuNotAvailable,

    #[error("Simulation error: {0}")]
    SimulationError(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    #[error("Memory allocation failed: {0}")]
    MemoryAllocationFailed(String),
}

/// Result of a single physics simulation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsStepResult {
    pub iteration: u64,
    pub kinetic_energy: f32,
    pub potential_energy: f32,
    pub total_energy: f32,
    pub convergence_delta: f32,
    pub execution_time_ms: f32,
}

/// Physics simulation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsStatistics {
    pub total_steps: u64,
    pub average_step_time_ms: f32,
    pub current_fps: f32,
    pub gpu_memory_used_mb: f32,
    pub gpu_utilization_percent: f32,
    pub convergence_rate: f32,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub compute_capability: String,
    pub total_memory_mb: usize,
    pub available_memory_mb: usize,
    pub cuda_cores: Option<usize>,
}

/// Port for GPU physics simulation operations
#[async_trait]
pub trait GpuPhysicsAdapter: Send + Sync {
    /// Initialize GPU with graph data
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<()>;

    /// Execute a single physics simulation step
    async fn simulate_step(&mut self, params: &SimulationParams) -> Result<PhysicsStepResult>;

    /// Update graph data on GPU
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<()>;

    /// Upload constraints to GPU
    async fn upload_constraints(&mut self, constraints: &ConstraintSet) -> Result<()>;

    /// Clear all constraints
    async fn clear_constraints(&mut self) -> Result<()>;

    /// Update simulation parameters
    async fn update_parameters(&mut self, params: &SimulationParams) -> Result<()>;

    /// Get current positions from GPU
    /// Returns Vec<(node_id, x, y, z)>
    async fn get_positions(&self) -> Result<Vec<(u32, f32, f32, f32)>>;

    /// Set specific node position (for user interaction)
    async fn set_node_position(&mut self, node_id: u32, x: f32, y: f32, z: f32) -> Result<()>;

    /// Get physics statistics
    async fn get_statistics(&self) -> Result<PhysicsStatistics>;

    /// Check if GPU is available and initialized
    fn is_available(&self) -> bool;

    /// Get GPU device information
    fn get_device_info(&self) -> GpuDeviceInfo;
}
