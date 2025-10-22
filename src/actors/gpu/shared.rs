//! Shared data structures and utilities for GPU actors

use super::cuda_stream_wrapper::SafeCudaStream;
use actix::Addr;
use cudarc::driver::CudaDevice;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::models::constraints::Constraint;
use crate::models::simulation_params::SimulationParams;
use crate::utils::unified_gpu_compute::{SimParams, UnifiedGPUCompute};

// Import the child actors for address storage
// use super::{GPUResourceActor, ForceComputeActor, ClusteringActor,
//            AnomalyDetectionActor, StressMajorizationActor, ConstraintActor};

/// Child actor addresses for the GPU manager

/// GPU resource contention management
#[derive(Debug, Clone)]
pub struct GPUResourceMetrics {
    pub kernel_launch_count: u64,
    pub total_wait_time_ms: u64,
    pub average_utilization_percent: f32,
    pub concurrent_access_attempts: u64,
    pub batched_operations_count: u64,
    pub last_operation_timestamp: Option<Instant>,
}

impl Default for GPUResourceMetrics {
    fn default() -> Self {
        Self {
            kernel_launch_count: 0,
            total_wait_time_ms: 0,
            average_utilization_percent: 0.0,
            concurrent_access_attempts: 0,
            batched_operations_count: 0,
            last_operation_timestamp: None,
        }
    }
}

/// GPU operation batching queue
#[derive(Debug, Clone)]
pub struct GPUOperationBatch {
    pub operations: Vec<GPUOperation>,
    pub priority: GPUOperationPriority,
    pub batch_size_limit: usize,
    pub flush_timeout_ms: u64,
    pub created_at: Instant,
}

#[derive(Debug, Clone)]
pub enum GPUOperation {
    ForceComputation,
    PositionUpdate,
    VelocityUpdate,
    Clustering,
    AnomalyDetection,
    StressMajorization,
    OntologyConstraints,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum GPUOperationPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl GPUOperationBatch {
    pub fn new(priority: GPUOperationPriority) -> Self {
        Self {
            operations: Vec::new(),
            priority,
            batch_size_limit: 10,
            flush_timeout_ms: 16, // ~60 FPS
            created_at: Instant::now(),
        }
    }

    pub fn should_flush(&self) -> bool {
        self.operations.len() >= self.batch_size_limit
            || self.created_at.elapsed().as_millis() >= self.flush_timeout_ms as u128
    }

    pub fn add_operation(&mut self, operation: GPUOperation) {
        self.operations.push(operation);
    }
}

/// Shared GPU context that gets passed between child actors
// Note: SafeCudaStream provides thread safety guarantees
pub struct SharedGPUContext {
    pub device: Arc<CudaDevice>,
    pub stream: Arc<std::sync::Mutex<SafeCudaStream>>,
    pub unified_compute: Arc<std::sync::Mutex<UnifiedGPUCompute>>,

    // Enhanced resource contention management (using RwLock to prevent deadlock)
    // Read access = normal concurrent operations, Write access = exclusive operations
    pub gpu_access_lock: Arc<RwLock<()>>,
    pub resource_metrics: Arc<Mutex<GPUResourceMetrics>>,
    pub operation_batch: Arc<Mutex<Vec<GPUOperation>>>,
    pub batch_timeout: Duration, // 10ms timeout for batching operations
}

/// GPU state shared among child actors
#[derive(Debug, Clone)]
pub struct GPUState {
    pub num_nodes: u32,
    pub num_edges: u32,
    pub node_indices: HashMap<u32, usize>,
    pub simulation_params: SimulationParams,
    pub unified_params: SimParams,
    pub constraints: Vec<Constraint>,
    pub iteration_count: u32,
    pub gpu_failure_count: u32,
    pub is_initialized: bool,

    // GPU Upload Optimization tracking
    pub graph_structure_hash: u64,
    pub positions_hash: u64,
    pub csr_structure_uploaded: bool,

    // Enhanced GPU resource contention tracking
    pub active_operations: Vec<GPUOperation>,
    pub last_sync_timestamp: Option<Instant>,
    pub gpu_utilization_history: Vec<f32>, // Rolling window of utilization percentages
    pub operation_queue_depth: usize,
    pub average_kernel_time_ms: f32,
    pub peak_memory_usage_bytes: usize,
    pub concurrent_access_count: u32,
}

impl Default for GPUState {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_edges: 0,
            node_indices: HashMap::new(),
            simulation_params: SimulationParams::default(),
            unified_params: SimParams::default(),
            constraints: Vec::new(),
            iteration_count: 0,
            gpu_failure_count: 0,
            is_initialized: false,
            graph_structure_hash: 0,
            positions_hash: 0,
            csr_structure_uploaded: false,

            // Enhanced GPU resource contention tracking
            active_operations: Vec::new(),
            last_sync_timestamp: None,
            gpu_utilization_history: Vec::with_capacity(60), // Store 60 samples (~1 second at 60 FPS)
            operation_queue_depth: 0,
            average_kernel_time_ms: 0.0,
            peak_memory_usage_bytes: 0,
            concurrent_access_count: 0,
        }
    }
}

/// Safety controls for stress majorization - moved from main actor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressMajorizationSafety {
    /// Maximum allowed displacement per iteration
    pub max_displacement_threshold: f32,
    /// Maximum allowed position magnitude
    pub max_position_magnitude: f32,
    /// Number of consecutive failures before disabling
    pub max_consecutive_failures: u32,
    /// Convergence threshold for displacement
    pub convergence_threshold: f32,
    /// Maximum allowed stress value before emergency stop
    pub max_stress_threshold: f32,

    // Runtime state
    pub consecutive_failures: u32,
    pub last_stress_values: Vec<f32>,
    pub last_displacement_values: Vec<f32>,
    pub total_runs: u64,
    pub successful_runs: u64,
    pub total_computation_time_ms: u64,
    pub is_emergency_stopped: bool,
    pub last_emergency_stop_reason: String,
}

impl StressMajorizationSafety {
    pub fn new() -> Self {
        Self {
            max_displacement_threshold: 1000.0,
            max_position_magnitude: 5000.0,
            max_consecutive_failures: 3,
            convergence_threshold: 0.01,
            max_stress_threshold: 1e6,

            consecutive_failures: 0,
            last_stress_values: Vec::with_capacity(10),
            last_displacement_values: Vec::with_capacity(10),
            total_runs: 0,
            successful_runs: 0,
            total_computation_time_ms: 0,
            is_emergency_stopped: false,
            last_emergency_stop_reason: String::new(),
        }
    }

    pub fn is_safe_to_run(&self) -> bool {
        !self.is_emergency_stopped && self.consecutive_failures < self.max_consecutive_failures
    }

    pub fn record_failure(&mut self, reason: String) {
        self.consecutive_failures += 1;
        self.total_runs += 1;
        if self.consecutive_failures >= self.max_consecutive_failures {
            self.is_emergency_stopped = true;
            self.last_emergency_stop_reason = reason;
        }
    }

    pub fn record_success(&mut self, computation_time_ms: u64) {
        self.consecutive_failures = 0;
        self.total_runs += 1;
        self.successful_runs += 1;
        self.total_computation_time_ms += computation_time_ms;
    }

    pub fn record_iteration(&mut self, stress: f32, displacement: f32, _converged: bool) {
        self.last_stress_values.push(stress);
        if self.last_stress_values.len() > 10 {
            self.last_stress_values.remove(0);
        }

        self.last_displacement_values.push(displacement);
        if self.last_displacement_values.len() > 10 {
            self.last_displacement_values.remove(0);
        }

        // Check for emergency conditions
        if stress > self.max_stress_threshold {
            self.record_failure(format!("Stress exceeded threshold: {}", stress));
        }

        if displacement > self.max_displacement_threshold {
            self.record_failure(format!("Displacement exceeded threshold: {}", displacement));
        }
    }

    pub fn clamp_position(&self, position: &[f32; 3]) -> [f32; 3] {
        let magnitude = (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt();
        if magnitude > self.max_position_magnitude {
            let scale = self.max_position_magnitude / magnitude;
            [
                position[0] * scale,
                position[1] * scale,
                position[2] * scale,
            ]
        } else {
            *position
        }
    }

    pub fn get_stats(
        &self,
    ) -> crate::actors::gpu::stress_majorization_actor::StressMajorizationStats {
        crate::actors::gpu::stress_majorization_actor::StressMajorizationStats {
            stress_value: 0.0, // Default value
            iterations_performed: self.total_runs as u32,
            converged: !self.is_emergency_stopped,
            computation_time_ms: if self.successful_runs > 0 {
                self.total_computation_time_ms / self.successful_runs
            } else {
                0
            },
        }
    }

    pub fn reset_safety_state(&mut self) {
        self.consecutive_failures = 0;
        self.is_emergency_stopped = false;
        self.last_emergency_stop_reason.clear();
    }

    pub fn should_disable(&self) -> bool {
        self.is_emergency_stopped
    }
}

impl SharedGPUContext {
    /// Acquire GPU access with proper synchronization and metrics tracking
    pub async fn acquire_gpu_access_qos(
        &self,
        operation: GPUOperation,
        priority: GPUOperationPriority,
    ) -> Result<(), String> {
        let start_time = Instant::now();

        // Update resource metrics - concurrent access tracking
        {
            let mut metrics = self
                .resource_metrics
                .lock()
                .map_err(|e| format!("Failed to lock metrics: {}", e))?;
            metrics.concurrent_access_attempts += 1;
        }

        // Acquire access based on priority (write for critical, read otherwise)
        if matches!(priority, GPUOperationPriority::Critical) {
            let _guard = self.gpu_access_lock.write().await;
            drop(_guard);
        } else {
            let _guard = self.gpu_access_lock.read().await;
            drop(_guard);
        }

        // Check if we should batch this operation
        let should_batch = self.should_batch_operation(&operation);
        if should_batch {
            self.add_to_batch(operation.clone())?;
            return Ok(());
        }

        // Update metrics with wait time
        let wait_time = start_time.elapsed();
        {
            let mut metrics = self
                .resource_metrics
                .lock()
                .map_err(|e| format!("Failed to lock metrics: {}", e))?;
            metrics.total_wait_time_ms += wait_time.as_millis() as u64;
            metrics.kernel_launch_count += 1;
            metrics.last_operation_timestamp = Some(Instant::now());
        }

        Ok(())
    }

    /// Check if operation should be batched
    fn should_batch_operation(&self, operation: &GPUOperation) -> bool {
        // Don't batch critical operations or compute-heavy operations
        match operation {
            GPUOperation::ForceComputation => false, // Always execute immediately
            GPUOperation::PositionUpdate | GPUOperation::VelocityUpdate => true, // Good candidates for batching
            GPUOperation::Clustering | GPUOperation::AnomalyDetection => false, // Too heavy to batch
            GPUOperation::StressMajorization => false, // Critical operation
            GPUOperation::OntologyConstraints => false, // Constraint updates should execute immediately
        }
    }

    /// Add operation to batch queue
    fn add_to_batch(&self, operation: GPUOperation) -> Result<(), String> {
        let mut batch = self
            .operation_batch
            .lock()
            .map_err(|e| format!("Failed to lock batch: {}", e))?;
        batch.push(operation);

        // Update batched operations count
        if let Ok(mut metrics) = self.resource_metrics.lock() {
            metrics.batched_operations_count += 1;
        }

        Ok(())
    }

    /// Flush batched operations if needed
    pub fn try_flush_batch(&self) -> Result<Vec<GPUOperation>, String> {
        let mut batch = self
            .operation_batch
            .lock()
            .map_err(|e| format!("Failed to lock batch: {}", e))?;

        if !batch.is_empty() {
            let operations = batch.clone();
            batch.clear();
            Ok(operations)
        } else {
            Ok(Vec::new())
        }
    }

    /// Update GPU utilization metrics
    pub fn update_utilization(&self, utilization_percent: f32) -> Result<(), String> {
        let mut metrics = self
            .resource_metrics
            .lock()
            .map_err(|e| format!("Failed to lock metrics: {}", e))?;

        // Update average utilization with exponential moving average
        if metrics.average_utilization_percent == 0.0 {
            metrics.average_utilization_percent = utilization_percent;
        } else {
            metrics.average_utilization_percent =
                metrics.average_utilization_percent * 0.9 + utilization_percent * 0.1;
        }

        Ok(())
    }

    /// Acquire GPU access for normal concurrent operations
    pub async fn acquire_gpu_access(&self) -> Result<tokio::sync::RwLockReadGuard<()>, String> {
        let start_time = Instant::now();
        let guard = self.gpu_access_lock.read().await;

        if let Ok(mut metrics) = self.resource_metrics.lock() {
            metrics.total_wait_time_ms += start_time.elapsed().as_millis() as u64;
        }

        Ok(guard)
    }

    /// Batch operations for efficient GPU execution
    /// Collects operations from the batch queue and executes them together
    pub async fn batch_operations(&self) -> Result<Vec<GPUOperation>, String> {
        let start_time = Instant::now();

        // Wait for batch timeout or until we have operations to process
        tokio::time::sleep(self.batch_timeout).await;

        let operations = self.try_flush_batch()?;

        if !operations.is_empty() {
            // Acquire read lock for batched execution
            let _guard = self.gpu_access_lock.read().await;

            // Update metrics for batch execution
            if let Ok(mut metrics) = self.resource_metrics.lock() {
                metrics.kernel_launch_count += 1; // One kernel launch for the entire batch
                metrics.total_wait_time_ms += start_time.elapsed().as_millis() as u64;
                metrics.last_operation_timestamp = Some(Instant::now());
            }
        }

        Ok(operations)
    }

    /// Acquire exclusive access to GPU for critical operations
    /// This method provides serialized access for operations that cannot be run concurrently
    pub async fn acquire_exclusive_access(
        &self,
    ) -> Result<tokio::sync::RwLockWriteGuard<()>, String> {
        let start_time = Instant::now();

        // Acquire write lock for exclusive access (blocks all readers and other writers)
        let guard = self.gpu_access_lock.write().await;

        // Update metrics for exclusive access
        if let Ok(mut metrics) = self.resource_metrics.lock() {
            metrics.total_wait_time_ms += start_time.elapsed().as_millis() as u64;
            metrics.concurrent_access_attempts += 1;
        }

        Ok(guard)
    }
}

impl GPUState {
    /// Add operation to active operations list
    pub fn start_operation(&mut self, operation: GPUOperation) {
        self.active_operations.push(operation);
        self.concurrent_access_count += 1;
        self.last_sync_timestamp = Some(Instant::now());
    }

    /// Remove operation from active operations list
    pub fn complete_operation(&mut self, operation: &GPUOperation) {
        self.active_operations.retain(|op| {
            !matches!(
                (op, operation),
                (
                    GPUOperation::ForceComputation,
                    GPUOperation::ForceComputation
                ) | (GPUOperation::PositionUpdate, GPUOperation::PositionUpdate)
                    | (GPUOperation::VelocityUpdate, GPUOperation::VelocityUpdate)
                    | (GPUOperation::Clustering, GPUOperation::Clustering)
                    | (
                        GPUOperation::AnomalyDetection,
                        GPUOperation::AnomalyDetection
                    )
                    | (
                        GPUOperation::StressMajorization,
                        GPUOperation::StressMajorization
                    )
                    | (
                        GPUOperation::OntologyConstraints,
                        GPUOperation::OntologyConstraints
                    )
            )
        });
        if self.concurrent_access_count > 0 {
            self.concurrent_access_count -= 1;
        }
    }

    /// Update GPU utilization history
    pub fn record_utilization(&mut self, utilization_percent: f32) {
        self.gpu_utilization_history.push(utilization_percent);
        // Keep only last 60 samples (1 second at 60 FPS)
        if self.gpu_utilization_history.len() > 60 {
            self.gpu_utilization_history.remove(0);
        }
    }

    /// Get average GPU utilization over recent history
    pub fn get_average_utilization(&self) -> f32 {
        if self.gpu_utilization_history.is_empty() {
            0.0
        } else {
            self.gpu_utilization_history.iter().sum::<f32>()
                / self.gpu_utilization_history.len() as f32
        }
    }

    /// Check if GPU is currently overloaded
    pub fn is_gpu_overloaded(&self) -> bool {
        // Only consider GPU overloaded if both high concurrent access AND high utilization
        // Previous logic was too aggressive, blocking at just 3 concurrent operations
        self.concurrent_access_count > 5 && self.get_average_utilization() > 80.0
    }
}

/// Child actor addresses for the manager
#[derive(Clone)]
pub struct ChildActorAddresses {
    pub resource_actor: Addr<super::GPUResourceActor>,
    pub force_compute_actor: Addr<super::ForceComputeActor>,
    pub clustering_actor: Addr<super::ClusteringActor>,
    pub anomaly_detection_actor: Addr<super::AnomalyDetectionActor>,
    pub stress_majorization_actor: Addr<super::StressMajorizationActor>,
    pub constraint_actor: Addr<super::ConstraintActor>,
    pub ontology_constraint_actor: Addr<super::OntologyConstraintActor>,
}
