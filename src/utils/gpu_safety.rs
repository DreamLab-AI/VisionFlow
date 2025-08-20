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

    pub fn track_allocation(&mut self, name: String, size: usize) -> Result<(), GPUSafetyError> {
        if self.total_allocated + size > 8_589_934_592 { // 8GB limit
            return Err(GPUSafetyError::MemoryLimitExceeded {
                requested: size,
                current: self.total_allocated,
                limit: 8_589_934_592,
            });
        }

        self.allocations.insert(name, size);
        self.total_allocated += size;
        self.allocation_count += 1;
        
        if self.total_allocated > self.max_allocated {
            self.max_allocated = self.total_allocated;
        }

        info!("GPU memory allocated: {} bytes (total: {} bytes)", size, self.total_allocated);
        Ok(())
    }

    pub fn track_deallocation(&mut self, name: &str) {
        if let Some(size) = self.allocations.remove(name) {
            self.total_allocated = self.total_allocated.saturating_sub(size);
            debug!("GPU memory deallocated: {} bytes (total: {} bytes)", size, self.total_allocated);
        }
    }

    pub fn get_usage_stats(&self) -> MemoryUsageStats {
        MemoryUsageStats {
            current_allocated: self.total_allocated,
            max_allocated: self.max_allocated,
            allocation_count: self.allocation_count,
            active_allocations: self.allocations.len(),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub current_allocated: usize,
    pub max_allocated: usize,
    pub allocation_count: u64,
    pub active_allocations: usize,
}

/// GPU safety error types
#[derive(Debug, thiserror::Error)]
pub enum GPUSafetyError {
    #[error("Buffer bounds exceeded: index {index} >= size {size}")]
    BufferBoundsExceeded { index: usize, size: usize },
    
    #[error("Invalid buffer size: requested {requested}, maximum allowed {max_allowed}")]
    InvalidBufferSize { requested: usize, max_allowed: usize },
    
    #[error("Memory limit exceeded: requested {requested} bytes, current {current} bytes, limit {limit} bytes")]
    MemoryLimitExceeded { requested: usize, current: usize, limit: usize },
    
    #[error("Kernel execution timeout: {duration_ms}ms > {limit_ms}ms")]
    KernelTimeout { duration_ms: u64, limit_ms: u64 },
    
    #[error("Invalid kernel parameters: {reason}")]
    InvalidKernelParams { reason: String },
    
    #[error("GPU device error: {message}")]
    DeviceError { message: String },
    
    #[error("Null pointer detected in GPU operation: {operation}")]
    NullPointer { operation: String },
    
    #[error("Misaligned memory access: address {address:#x}, required alignment {alignment}")]
    MisalignedAccess { address: usize, alignment: usize },
    
    #[error("Resource exhaustion: {resource} count {current} exceeds limit {limit}")]
    ResourceExhaustion { resource: String, current: usize, limit: usize },
}

/// GPU safety validator with comprehensive checking
pub struct GPUSafetyValidator {
    config: GPUSafetyConfig,
    memory_tracker: Arc<Mutex<GPUMemoryTracker>>,
    failure_count: Arc<Mutex<u32>>,
    last_kernel_time: Arc<Mutex<Option<Duration>>>,
}

impl GPUSafetyValidator {
    pub fn new(config: GPUSafetyConfig) -> Self {
        Self {
            config,
            memory_tracker: Arc::new(Mutex::new(GPUMemoryTracker::new())),
            failure_count: Arc::new(Mutex::new(0)),
            last_kernel_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Validate buffer dimensions and bounds
    pub fn validate_buffer_bounds(
        &self, 
        buffer_name: &str,
        requested_size: usize,
        element_size: usize,
    ) -> Result<(), GPUSafetyError> {
        let total_bytes = requested_size * element_size;
        
        // Check against configuration limits
        match buffer_name {
            name if name.contains("node") => {
                if requested_size > self.config.max_nodes {
                    return Err(GPUSafetyError::InvalidBufferSize {
                        requested: requested_size,
                        max_allowed: self.config.max_nodes,
                    });
                }
            },
            name if name.contains("edge") => {
                if requested_size > self.config.max_edges {
                    return Err(GPUSafetyError::InvalidBufferSize {
                        requested: requested_size,
                        max_allowed: self.config.max_edges,
                    });
                }
            },
            _ => {} // Other buffers use general memory limits
        }

        // Check memory limits
        if let Ok(tracker) = self.memory_tracker.lock() {
            if tracker.total_allocated + total_bytes > self.config.max_memory_bytes {
                return Err(GPUSafetyError::MemoryLimitExceeded {
                    requested: total_bytes,
                    current: tracker.total_allocated,
                    limit: self.config.max_memory_bytes,
                });
            }
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

        if block_size > 1024 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Block size {} exceeds CUDA maximum of 1024", block_size),
            });
        }

        if grid_size > 65535 {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Grid size {} exceeds CUDA maximum of 65535", grid_size),
            });
        }

        // Check total thread count doesn't overflow
        let total_threads = grid_size as u64 * block_size as u64;
        if total_threads > (i32::MAX as u64) {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Total thread count {} exceeds maximum", total_threads),
            });
        }

        debug!("Kernel parameters validated: nodes={}, edges={}, constraints={}, grid={}, block={}", 
               num_nodes, num_edges, num_constraints, grid_size, block_size);
        Ok(())
    }

    /// Validate memory alignment
    pub fn validate_memory_alignment(&self, ptr: *const u8, alignment: usize) -> Result<(), GPUSafetyError> {
        if ptr.is_null() {
            return Err(GPUSafetyError::NullPointer {
                operation: "memory alignment check".to_string(),
            });
        }

        let address = ptr as usize;
        if address % alignment != 0 {
            return Err(GPUSafetyError::MisalignedAccess { address, alignment });
        }

        Ok(())
    }

    /// Track memory allocation with safety checks
    pub fn track_allocation(&self, name: String, size: usize) -> Result<(), GPUSafetyError> {
        let mut tracker = self.memory_tracker.lock()
            .map_err(|_| GPUSafetyError::DeviceError {
                message: "Failed to acquire memory tracker lock".to_string(),
            })?;

        tracker.track_allocation(name, size)
    }

    /// Track memory deallocation
    pub fn track_deallocation(&self, name: &str) {
        if let Ok(mut tracker) = self.memory_tracker.lock() {
            tracker.track_deallocation(name);
        }
    }

    /// Record kernel execution time and check for timeouts
    pub fn record_kernel_execution(&self, duration: Duration) -> Result<(), GPUSafetyError> {
        let duration_ms = duration.as_millis() as u64;
        
        if duration_ms > self.config.max_kernel_time_ms {
            return Err(GPUSafetyError::KernelTimeout {
                duration_ms,
                limit_ms: self.config.max_kernel_time_ms,
            });
        }

        if let Ok(mut last_time) = self.last_kernel_time.lock() {
            *last_time = Some(duration);
        }

        debug!("Kernel execution time: {}ms", duration_ms);
        Ok(())
    }

    /// Check if CPU fallback should be triggered
    pub fn should_use_cpu_fallback(&self) -> bool {
        if let Ok(failure_count) = self.failure_count.lock() {
            *failure_count >= self.config.cpu_fallback_threshold
        } else {
            false
        }
    }

    /// Record a GPU operation failure
    pub fn record_failure(&self) {
        if let Ok(mut failure_count) = self.failure_count.lock() {
            *failure_count += 1;
            warn!("GPU failure recorded (count: {})", *failure_count);
            
            if *failure_count >= self.config.cpu_fallback_threshold {
                error!("GPU failure threshold reached, CPU fallback recommended");
            }
        }
    }

    /// Reset failure counter (call after successful operation)
    pub fn reset_failures(&self) {
        if let Ok(mut failure_count) = self.failure_count.lock() {
            if *failure_count > 0 {
                debug!("Resetting GPU failure count from {}", *failure_count);
                *failure_count = 0;
            }
        }
    }

    /// Get current memory usage statistics
    pub fn get_memory_stats(&self) -> Option<MemoryUsageStats> {
        self.memory_tracker.lock().ok().map(|tracker| tracker.get_usage_stats())
    }

    /// Validate array indices for bounds checking
    pub fn validate_array_access(&self, index: usize, array_size: usize, array_name: &str) -> Result<(), GPUSafetyError> {
        if self.config.strict_bounds_checking && index >= array_size {
            return Err(GPUSafetyError::BufferBoundsExceeded {
                index,
                size: array_size,
            });
        }
        Ok(())
    }

    /// Comprehensive pre-kernel validation
    pub fn pre_kernel_validation(
        &self,
        nodes: &[(f32, f32, f32)],
        edges: &[(i32, i32, f32)],
        grid_size: u32,
        block_size: u32,
    ) -> Result<(), GPUSafetyError> {
        // Validate basic parameters
        self.validate_kernel_params(
            nodes.len() as i32,
            edges.len() as i32,
            0, // constraints
            grid_size,
            block_size,
        )?;

        // Validate edge references
        for (i, &(src, dst, weight)) in edges.iter().enumerate() {
            if src < 0 || dst < 0 {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Edge {} has negative node indices: src={}, dst={}", i, src, dst),
                });
            }

            let src_idx = src as usize;
            let dst_idx = dst as usize;

            if src_idx >= nodes.len() {
                return Err(GPUSafetyError::BufferBoundsExceeded {
                    index: src_idx,
                    size: nodes.len(),
                });
            }

            if dst_idx >= nodes.len() {
                return Err(GPUSafetyError::BufferBoundsExceeded {
                    index: dst_idx,
                    size: nodes.len(),
                });
            }

            // Check for invalid weights
            if !weight.is_finite() {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Edge {} has invalid weight: {}", i, weight),
                });
            }
        }

        // Validate node positions
        for (i, &(x, y, z)) in nodes.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return Err(GPUSafetyError::InvalidKernelParams {
                    reason: format!("Node {} has invalid position: ({}, {}, {})", i, x, y, z),
                });
            }
        }

        debug!("Pre-kernel validation passed: {} nodes, {} edges", nodes.len(), edges.len());
        Ok(())
    }
}

/// Safe kernel execution wrapper with timeout and error recovery
pub struct SafeKernelExecutor {
    validator: Arc<GPUSafetyValidator>,
}

impl SafeKernelExecutor {
    pub fn new(validator: Arc<GPUSafetyValidator>) -> Self {
        Self { validator }
    }

    /// Execute kernel with comprehensive safety checks and timeout
    pub async fn execute_with_timeout<F, R>(&self, operation: F) -> Result<R, GPUSafetyError>
    where
        F: FnOnce() -> Result<R, GPUSafetyError> + Send + 'static,
        R: Send + 'static,
    {
        let start_time = Instant::now();
        
        // Use tokio timeout for async execution
        let timeout_duration = Duration::from_millis(self.validator.config.max_kernel_time_ms);
        
        let result = tokio::time::timeout(timeout_duration, async move {
            tokio::task::spawn_blocking(operation).await
                .map_err(|e| GPUSafetyError::DeviceError {
                    message: format!("Task execution failed: {}", e),
                })?
        }).await;

        let execution_time = start_time.elapsed();
        
        match result {
            Ok(Ok(value)) => {
                self.validator.record_kernel_execution(execution_time)?;
                self.validator.reset_failures();
                Ok(value)
            },
            Ok(Err(e)) => {
                self.validator.record_failure();
                Err(e)
            },
            Err(_) => {
                self.validator.record_failure();
                Err(GPUSafetyError::KernelTimeout {
                    duration_ms: execution_time.as_millis() as u64,
                    limit_ms: self.validator.config.max_kernel_time_ms,
                })
            }
        }
    }
}

/// CPU fallback implementation for critical operations
pub mod cpu_fallback {
    use super::*;

    /// Basic force-directed layout computation on CPU
    pub fn compute_forces_cpu(
        positions: &mut [(f32, f32, f32)],
        velocities: &mut [(f32, f32, f32)],
        edges: &[(i32, i32, f32)],
        spring_k: f32,
        repel_k: f32,
        damping: f32,
        dt: f32,
    ) -> Result<(), GPUSafetyError> {
        let num_nodes = positions.len();
        
        if velocities.len() != num_nodes {
            return Err(GPUSafetyError::InvalidKernelParams {
                reason: format!("Velocity array size {} doesn't match position array size {}", 
                               velocities.len(), num_nodes),
            });
        }

        // Calculate forces for each node
        let mut forces = vec![(0.0f32, 0.0f32, 0.0f32); num_nodes];

        // Repulsive forces (N^2 computation)
        for i in 0..num_nodes {
            for j in (i+1)..num_nodes {
                let dx = positions[i].0 - positions[j].0;
                let dy = positions[i].1 - positions[j].1;
                let dz = positions[i].2 - positions[j].2;
                
                let dist_sq = dx*dx + dy*dy + dz*dz + 0.01; // Avoid division by zero
                let dist = dist_sq.sqrt();
                
                let force_magnitude = repel_k / dist_sq;
                let fx = (dx / dist) * force_magnitude;
                let fy = (dy / dist) * force_magnitude;
                let fz = (dz / dist) * force_magnitude;
                
                forces[i].0 += fx;
                forces[i].1 += fy;
                forces[i].2 += fz;
                
                forces[j].0 -= fx;
                forces[j].1 -= fy;
                forces[j].2 -= fz;
            }
        }

        // Attractive forces from edges
        for &(src, dst, weight) in edges {
            let src_idx = src as usize;
            let dst_idx = dst as usize;
            
            if src_idx >= num_nodes || dst_idx >= num_nodes {
                continue; // Skip invalid edges
            }

            let dx = positions[dst_idx].0 - positions[src_idx].0;
            let dy = positions[dst_idx].1 - positions[src_idx].1;
            let dz = positions[dst_idx].2 - positions[src_idx].2;
            
            let dist = (dx*dx + dy*dy + dz*dz).sqrt();
            if dist > 0.01 {
                let force_magnitude = spring_k * weight;
                let fx = (dx / dist) * force_magnitude;
                let fy = (dy / dist) * force_magnitude;
                let fz = (dz / dist) * force_magnitude;
                
                forces[src_idx].0 += fx;
                forces[src_idx].1 += fy;
                forces[src_idx].2 += fz;
                
                forces[dst_idx].0 -= fx;
                forces[dst_idx].1 -= fy;
                forces[dst_idx].2 -= fz;
            }
        }

        // Update velocities and positions
        for i in 0..num_nodes {
            // Update velocity with damping
            velocities[i].0 = (velocities[i].0 + forces[i].0 * dt) * damping;
            velocities[i].1 = (velocities[i].1 + forces[i].1 * dt) * damping;
            velocities[i].2 = (velocities[i].2 + forces[i].2 * dt) * damping;
            
            // Clamp velocity
            let vel_mag = (velocities[i].0*velocities[i].0 + 
                          velocities[i].1*velocities[i].1 + 
                          velocities[i].2*velocities[i].2).sqrt();
            if vel_mag > 10.0 {
                let scale = 10.0 / vel_mag;
                velocities[i].0 *= scale;
                velocities[i].1 *= scale;
                velocities[i].2 *= scale;
            }
            
            // Update position
            positions[i].0 += velocities[i].0 * dt;
            positions[i].1 += velocities[i].1 * dt;
            positions[i].2 += velocities[i].2 * dt;
        }

        info!("CPU fallback computation completed for {} nodes, {} edges", num_nodes, edges.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_bounds_validation() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Test valid bounds
        assert!(validator.validate_buffer_bounds("node_positions", 1000, 12).is_ok());

        // Test exceeding node limit
        assert!(validator.validate_buffer_bounds("node_positions", 2_000_000, 12).is_err());

        // Test exceeding edge limit
        assert!(validator.validate_buffer_bounds("edge_data", 10_000_000, 12).is_err());
    }

    #[test]
    fn test_kernel_params_validation() {
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);

        // Test valid parameters
        assert!(validator.validate_kernel_params(1000, 2000, 0, 4, 256).is_ok());

        // Test negative values
        assert!(validator.validate_kernel_params(-1, 2000, 0, 4, 256).is_err());

        // Test exceeding limits
        assert!(validator.validate_kernel_params(2_000_000, 2000, 0, 4, 256).is_err());

        // Test invalid grid/block sizes
        assert!(validator.validate_kernel_params(1000, 2000, 0, 0, 256).is_err());
        assert!(validator.validate_kernel_params(1000, 2000, 0, 4, 2048).is_err());
    }

    #[test]
    fn test_cpu_fallback() {
        let mut positions = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let mut velocities = vec![(0.0, 0.0, 0.0); 3];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];

        let result = cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1, 0.1, 0.9, 0.01
        );

        assert!(result.is_ok());
        
        // Positions should have changed
        assert!(positions[0] != (0.0, 0.0, 0.0) || 
                positions[1] != (1.0, 0.0, 0.0) || 
                positions[2] != (0.0, 1.0, 0.0));
    }
}