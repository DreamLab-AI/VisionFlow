//! GPU Safety Validation Module
//! 
//! Provides comprehensive bounds checking, memory validation, and safety measures
//! for all GPU operations in the VisionFlow system.

use std::collections::HashMap;
use log::{error, warn, info, debug};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// GPU safety configuration limits
#[derive(Debug, Clone)]
pub struct GPUSafetyConfig {
    /// Maximum number of nodes allowed
    pub max_nodes: usize,
    /// Maximum number of edges allowed
    pub max_edges: usize,
    /// Maximum GPU memory usage in bytes
    pub max_memory_bytes: usize,
    /// Maximum kernel execution time in milliseconds
    pub max_kernel_time_ms: u64,
    /// Enable strict bounds checking (may impact performance)
    pub strict_bounds_checking: bool,
    /// Enable memory usage tracking
    pub memory_tracking: bool,
    /// CPU fallback threshold (failure count before fallback)
    pub cpu_fallback_threshold: u32,
}

impl Default for GPUSafetyConfig {
    fn default() -> Self {
        Self {
            max_nodes: 1_000_000,           // 1M nodes max
            max_edges: 5_000_000,           // 5M edges max
            max_memory_bytes: 8_589_934_592, // 8GB max
            max_kernel_time_ms: 5000,       // 5 second timeout
            strict_bounds_checking: true,
            memory_tracking: true,
            cpu_fallback_threshold: 3,
        }
    }
}

/// Memory allocation tracker
#[derive(Debug)]
pub struct GPUMemoryTracker {
    allocations: HashMap<String, usize>,
    total_allocated: usize,
    max_allocated: usize,
    allocation_count: u64,
}

impl GPUMemoryTracker {
    pub fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            total_allocated: 0,
            max_allocated: 0,
            allocation_count: 0,
        }
    }

    pub fn track_allocation(&mut self, name: String, size: usize) {
        self.allocations.insert(name, size);
        self.total_allocated += size;
        self.allocation_count += 1;
        
        if self.total_allocated > self.max_allocated {
            self.max_allocated = self.total_allocated;
        }
    }

    pub fn track_deallocation(&mut self, name: &str) {
        if let Some(size) = self.allocations.remove(name) {
            self.total_allocated = self.total_allocated.saturating_sub(size);
        }
    }

    pub fn get_total_allocated(&self) -> usize {
        self.total_allocated
    }

    pub fn get_max_allocated(&self) -> usize {
        self.max_allocated
    }

    pub fn get_allocation_count(&self) -> u64 {
        self.allocation_count
    }
}

/// Kernel execution tracker
#[derive(Debug)]
pub struct KernelTracker {
    executions: HashMap<String, KernelStats>,
    total_executions: u64,
    total_failures: u64,
}

#[derive(Debug)]
pub struct KernelStats {
    pub name: String,
    pub executions: u64,
    pub failures: u64,
    pub total_time_ms: u64,
    pub average_time_ms: f64,
    pub last_execution: Option<Instant>,
}

impl KernelTracker {
    pub fn new() -> Self {
        Self {
            executions: HashMap::new(),
            total_executions: 0,
            total_failures: 0,
        }
    }

    pub fn track_execution(&mut self, kernel_name: String, duration_ms: u64, success: bool) {
        let stats = self.executions.entry(kernel_name.clone()).or_insert_with(|| {
            KernelStats {
                name: kernel_name,
                executions: 0,
                failures: 0,
                total_time_ms: 0,
                average_time_ms: 0.0,
                last_execution: None,
            }
        });

        stats.executions += 1;
        self.total_executions += 1;
        
        if success {
            stats.total_time_ms += duration_ms;
            stats.average_time_ms = stats.total_time_ms as f64 / stats.executions as f64;
        } else {
            stats.failures += 1;
            self.total_failures += 1;
        }
        
        stats.last_execution = Some(Instant::now());
    }

    pub fn get_failure_rate(&self, kernel_name: &str) -> f64 {
        if let Some(stats) = self.executions.get(kernel_name) {
            if stats.executions > 0 {
                stats.failures as f64 / stats.executions as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    pub fn get_total_failure_rate(&self) -> f64 {
        if self.total_executions > 0 {
            self.total_failures as f64 / self.total_executions as f64
        } else {
            0.0
        }
    }
}

/// GPU safety error types
#[derive(Debug, thiserror::Error)]
pub enum GPUSafetyError {
    #[error("Buffer bounds exceeded: index {index} >= size {size}")]
    BufferBoundsExceeded { index: usize, size: usize },

    #[error("Invalid buffer size: requested {requested}, max allowed {max_allowed}")]
    InvalidBufferSize { requested: usize, max_allowed: usize },

    #[error("Invalid kernel parameters: {reason}")]
    InvalidKernelParams { reason: String },

    #[error("Kernel execution timeout: {kernel_name} exceeded {timeout_ms}ms")]
    KernelTimeout { kernel_name: String, timeout_ms: u64 },

    #[error("GPU device error: {message}")]
    DeviceError { message: String },

    #[error("Out of GPU memory: requested {requested} bytes, available {available} bytes")]
    OutOfMemory { available: usize, requested: usize },

    #[error("Memory bounds error: {0}")]
    MemoryBounds(#[from] crate::utils::memory_bounds::MemoryBoundsError),

    #[error("Null pointer dereference detected")]
    NullPointer,

    #[error("Data race detected: {details}")]
    DataRace { details: String },

    #[error("CPU fallback required: GPU failure count exceeded threshold")]
    CPUFallbackRequired,

    #[error("Validation failed: {message}")]
    ValidationFailed { message: String },

    #[error("Resource exhaustion: {resource} count {current} exceeds limit {limit}")]
    ResourceExhaustion { resource: String, current: usize, limit: usize },
}

/// Main GPU safety validator
pub struct GPUSafetyValidator {
    config: GPUSafetyConfig,
    memory_tracker: Arc<Mutex<GPUMemoryTracker>>,
    kernel_tracker: Arc<Mutex<KernelTracker>>,
    failure_count: Arc<Mutex<u32>>,
    last_validation: Arc<Mutex<Option<Instant>>>,
}

impl GPUSafetyValidator {
    pub fn new(config: GPUSafetyConfig) -> Self {
        Self {
            config,
            memory_tracker: Arc::new(Mutex::new(GPUMemoryTracker::new())),
            kernel_tracker: Arc::new(Mutex::new(KernelTracker::new())),
            failure_count: Arc::new(Mutex::new(0)),
            last_validation: Arc::new(Mutex::new(None)),
        }
    }

    /// Validate buffer bounds and size
    pub fn validate_buffer_bounds(
        &self,
        buffer_name: &str,
        requested_size: usize,
        element_size: usize,
    ) -> Result<(), GPUSafetyError> {
        // Check for zero size
        if requested_size == 0 {
            return Err(GPUSafetyError::InvalidBufferSize {
                requested: 0,
                max_allowed: self.config.max_nodes,
            });
        }

        // Check maximum bounds
        if requested_size > self.config.max_nodes && buffer_name.contains("node") {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: requested_size,
                size: self.config.max_nodes,
            });
        }

        if requested_size > self.config.max_edges && buffer_name.contains("edge") {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index: requested_size,
                size: self.config.max_edges,
            });
        }

        // Check memory allocation size
        let total_bytes = requested_size.saturating_mul(element_size);
        if total_bytes > self.config.max_memory_bytes {
            return Err(GPUSafetyError::InvalidBufferSize {
                requested: total_bytes,
                max_allowed: self.config.max_memory_bytes,
            });
        }

        // Check for integer overflow
        if requested_size > 0 && total_bytes / requested_size != element_size {
            return Err(GPUSafetyError::InvalidBufferSize {
                requested: requested_size,
                max_allowed: usize::MAX / element_size,
            });
        }

        debug!("Buffer bounds validated: {} ({} elements, {} bytes)", 
               buffer_name, requested_size, total_bytes);
        Ok(())
    }

    /// Validate kernel execution parameters
    pub fn validate_kernel_params(
        &self,
        num_nodes: i32,
        num_edges: i32,
        num_constraints: i32,
        grid_size: u32,
        block_size: u32,
    ) -> Result<(), GPUSafetyError> {
        // Check for negative values
        if num_nodes < 0 || num_edges < 0 || num_constraints < 0 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Negative values detected: nodes={}, edges={}, constraints={}", 
                               num_nodes, num_edges, num_constraints),
            });
        }

        // Check reasonable bounds
        if num_nodes as usize > self.config.max_nodes {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Node count {} exceeds maximum {}", num_nodes, self.config.max_nodes),
            });
        }

        if num_edges as usize > self.config.max_edges {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Edge count {} exceeds maximum {}", num_edges, self.config.max_edges),
            });
        }

        // Validate grid/block dimensions
        if grid_size == 0 || block_size == 0 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: "Grid size and block size must be greater than 0".to_string(),
            });
        }

        // Check for reasonable grid/block sizes (prevent resource exhaustion)
        let total_threads = grid_size as u64 * block_size as u64;
        if total_threads > 1_000_000_000 {  // 1B thread limit
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Total thread count {} exceeds 1B limit", total_threads),
            });
        }

        debug!("Kernel parameters validated: nodes={}, edges={}, constraints={}, grid={}, block={}",
               num_nodes, num_edges, num_constraints, grid_size, block_size);
        Ok(())
    }

    /// Track memory allocation
    pub fn track_allocation(&self, name: String, size: usize) -> Result<(), GPUSafetyError> {
        if let Ok(mut tracker) = self.memory_tracker.lock() {
            // Check if this allocation would exceed memory limits
            let new_total = tracker.get_total_allocated() + size;
            if new_total > self.config.max_memory_bytes {
                return Err(GPUSafetyError::OutOfMemory {
                    available: self.config.max_memory_bytes - tracker.get_total_allocated(),
                    requested: size,
                });
            }

            tracker.track_allocation(name, size);
            debug!("Memory allocation tracked: {} bytes (total: {} bytes)", 
                   size, tracker.get_total_allocated());
        }
        Ok(())
    }

    /// Track memory deallocation
    pub fn track_deallocation(&self, name: &str) {
        if let Ok(mut tracker) = self.memory_tracker.lock() {
            tracker.track_deallocation(name);
            debug!("Memory deallocation tracked: {} (total: {} bytes)", 
                   name, tracker.get_total_allocated());
        }
    }

    /// Track kernel execution
    pub fn track_kernel_execution(&self, kernel_name: String, duration_ms: u64, success: bool) {
        if let Ok(mut tracker) = self.kernel_tracker.lock() {
            tracker.track_execution(kernel_name.clone(), duration_ms, success);
            
            if !success {
                if let Ok(mut count) = self.failure_count.lock() {
                    *count += 1;
                    warn!("Kernel execution failed: {} (failure count: {})", kernel_name, *count);
                }
            }
        }
    }

    /// Record general failure for CPU fallback decision
    pub fn record_failure(&self) {
        if let Ok(mut count) = self.failure_count.lock() {
            *count += 1;
            warn!("GPU operation failed (failure count: {})", *count);
        }
    }

    /// Check if CPU fallback should be triggered
    pub fn should_fallback_to_cpu(&self) -> bool {
        if let Ok(count) = self.failure_count.lock() {
            *count >= self.config.cpu_fallback_threshold
        } else {
            false
        }
    }

    /// Check if we should use CPU fallback - alias for compatibility
    pub fn should_use_cpu_fallback(&self) -> bool {
        self.should_fallback_to_cpu()
    }

    /// Reset failure count (call after successful CPU fallback recovery)
    pub fn reset_failure_count(&self) {
        if let Ok(mut count) = self.failure_count.lock() {
            *count = 0;
            info!("GPU failure count reset - returning to normal operation");
        }
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> Option<(usize, usize, u64)> {
        if let Ok(tracker) = self.memory_tracker.lock() {
            Some((
                tracker.get_total_allocated(),
                tracker.get_max_allocated(),
                tracker.get_allocation_count(),
            ))
        } else {
            None
        }
    }

    /// Get kernel execution statistics
    pub fn get_kernel_stats(&self, kernel_name: &str) -> Option<f64> {
        if let Ok(tracker) = self.kernel_tracker.lock() {
            Some(tracker.get_failure_rate(kernel_name))
        } else {
            None
        }
    }

    /// Comprehensive validation for a complete GPU operation
    pub async fn validate_operation(
        &self,
        operation_name: &str,
        node_count: usize,
        edge_count: usize,
        memory_required: usize,
    ) -> Result<(), GPUSafetyError> {
        // Update last validation timestamp
        if let Ok(mut last) = self.last_validation.lock() {
            *last = Some(Instant::now());
        }

        // Check if we should fallback to CPU
        if self.should_fallback_to_cpu() {
            return Err(GPUSafetyError::CPUFallbackRequired);
        }

        // Validate buffer bounds
        self.validate_buffer_bounds("nodes", node_count, std::mem::size_of::<f32>())?;
        self.validate_buffer_bounds("edges", edge_count, std::mem::size_of::<f32>())?;

        // Check memory requirements
        if memory_required > 0 {
            self.track_allocation(format!("{}_operation", operation_name), memory_required)?;
        }

        // Validate kernel parameters (using reasonable defaults)
        let grid_size = ((node_count + 255) / 256) as u32;
        let block_size = 256u32;
        
        self.validate_kernel_params(
            node_count as i32,
            edge_count as i32,
            0,
            grid_size,
            block_size,
        )?;

        info!("GPU operation validated: {} (nodes: {}, edges: {}, memory: {} bytes)",
              operation_name, node_count, edge_count, memory_required);
        
        Ok(())
    }
}

impl Default for GPUSafetyValidator {
    fn default() -> Self {
        Self::new(GPUSafetyConfig::default())
    }
}

/// Kernel execution wrapper with timeout and safety checks
pub struct SafeKernelExecutor {
    validator: Arc<GPUSafetyValidator>,
}

impl SafeKernelExecutor {
    pub fn new(validator: Arc<GPUSafetyValidator>) -> Self {
        Self { validator }
    }

    /// Execute a kernel with comprehensive safety checks and timeout
    pub async fn execute_with_timeout<F, R>(&self, operation: F) -> Result<R, GPUSafetyError>
    where
        F: std::future::Future<Output = Result<R, GPUSafetyError>>,
    {
        let start_time = Instant::now();
        let timeout_duration = Duration::from_millis(self.validator.config.max_kernel_time_ms);

        // Execute with timeout
        let result = tokio::time::timeout(timeout_duration, operation).await;

        let execution_time = start_time.elapsed();
        let execution_time_ms = execution_time.as_millis() as u64;

        match result {
            Ok(Ok(value)) => {
                // Success
                self.validator.track_kernel_execution(
                    "safe_kernel_execution".to_string(),
                    execution_time_ms,
                    true,
                );
                debug!("Kernel executed successfully in {}ms", execution_time_ms);
                Ok(value)
            }
            Ok(Err(e)) => {
                // Kernel failed
                self.validator.track_kernel_execution(
                    "safe_kernel_execution".to_string(),
                    execution_time_ms,
                    false,
                );
                error!("Kernel execution failed: {}", e);
                Err(e)
            }
            Err(_) => {
                // Timeout
                self.validator.track_kernel_execution(
                    "safe_kernel_execution".to_string(),
                    execution_time_ms,
                    false,
                );
                error!("Kernel execution timed out after {}ms", execution_time_ms);
                Err(GPUSafetyError::KernelTimeout {
                    kernel_name: "safe_kernel_execution".to_string(),
                    timeout_ms: self.validator.config.max_kernel_time_ms,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_bounds_validation() {
        let validator = GPUSafetyValidator::default();
        
        // Valid buffer
        assert!(validator.validate_buffer_bounds("test_nodes", 1000, 4).is_ok());
        
        // Zero size
        assert!(validator.validate_buffer_bounds("test_nodes", 0, 4).is_err());
        
        // Too large
        assert!(validator.validate_buffer_bounds("test_nodes", 2_000_000, 4).is_err());
    }

    #[test]
    fn test_kernel_params_validation() {
        let validator = GPUSafetyValidator::default();
        
        // Valid parameters
        assert!(validator.validate_kernel_params(1000, 5000, 100, 4, 256).is_ok());
        
        // Negative values
        assert!(validator.validate_kernel_params(-1, 5000, 100, 4, 256).is_err());
        
        // Zero grid/block size
        assert!(validator.validate_kernel_params(1000, 5000, 100, 0, 256).is_err());
        
        // Too many nodes
        assert!(validator.validate_kernel_params(2_000_000, 5000, 100, 4, 256).is_err());
    }

    #[test]
    fn test_memory_tracking() {
        let validator = GPUSafetyValidator::default();
        
        // Track allocation
        assert!(validator.track_allocation("test_buffer".to_string(), 1024).is_ok());
        
        let (total, _, count) = validator.get_memory_stats().unwrap();
        assert_eq!(total, 1024);
        assert_eq!(count, 1);
        
        // Track deallocation
        validator.track_deallocation("test_buffer");
        
        let (total, _, _) = validator.get_memory_stats().unwrap();
        assert_eq!(total, 0);
    }

    #[test]
    fn test_cpu_fallback() {
        let mut config = GPUSafetyConfig::default();
        config.cpu_fallback_threshold = 2;
        let validator = GPUSafetyValidator::new(config);
        
        // Should not fallback initially
        assert!(!validator.should_fallback_to_cpu());
        
        // Record failures
        validator.record_failure();
        assert!(!validator.should_fallback_to_cpu());
        
        validator.record_failure();
        assert!(validator.should_fallback_to_cpu());
        
        // Reset
        validator.reset_failure_count();
        assert!(!validator.should_fallback_to_cpu());
    }
}