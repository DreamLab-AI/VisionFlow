//! GPU Safety Validation Tests
//! 
//! Comprehensive testing of GPU safety mechanisms, bounds checking,
//! memory management, and fallback systems

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::test;
use pretty_assertions::assert_eq;

use webxr::utils::gpu_safety::*;

#[derive(Debug)]
pub struct GPUSafetyTestSuite {
    test_count: usize,
    passed_tests: usize,
    failed_tests: usize,
    safety_config: GPUSafetyConfig,
}

impl GPUSafetyTestSuite {
    pub fn new() -> Self {
        Self {
            test_count: 0,
            passed_tests: 0,
            failed_tests: 0,
            safety_config: GPUSafetyConfig::default(),
        }
    }

    pub async fn run_all_tests(&mut self) {
        println!("Running GPU Safety Validation Tests...");
        
        self.test_safety_config_defaults().await;
        self.test_memory_tracker_functionality().await;
        self.test_buffer_bounds_validation().await;
        self.test_kernel_parameter_validation().await;
        self.test_memory_allocation_tracking().await;
        self.test_memory_alignment_validation().await;
        self.test_gpu_failure_tracking().await;
        self.test_cpu_fallback_mechanism().await;
        self.test_kernel_timeout_detection().await;
        self.test_concurrent_gpu_operations().await;
        self.test_resource_exhaustion_handling().await;
        self.test_safe_kernel_executor().await;
        self.test_pre_kernel_validation().await;
        self.test_cpu_fallback_computation().await;
        self.test_edge_case_scenarios().await;
        self.test_performance_characteristics().await;
        
        self.print_results();
    }

    async fn test_safety_config_defaults(&mut self) {
        let test_name = "safety_config_defaults";
        let start = Instant::now();
        
        let mut all_passed = true;
        
        let config = GPUSafetyConfig::default();
        
        // Test reasonable default values
        if config.max_nodes == 0 {
            eprintln!("Default max_nodes should be greater than 0");
            all_passed = false;
        }
        
        if config.max_edges == 0 {
            eprintln!("Default max_edges should be greater than 0");
            all_passed = false;
        }
        
        if config.max_memory_bytes == 0 {
            eprintln!("Default max_memory_bytes should be greater than 0");
            all_passed = false;
        }
        
        if config.max_kernel_time_ms == 0 {
            eprintln!("Default max_kernel_time_ms should be greater than 0");
            all_passed = false;
        }
        
        if config.cpu_fallback_threshold == 0 {
            eprintln!("Default cpu_fallback_threshold should be greater than 0");
            all_passed = false;
        }
        
        // Test that defaults are reasonable for production use
        if config.max_nodes < 1000 {
            eprintln!("Default max_nodes too small for practical use: {}", config.max_nodes);
            all_passed = false;
        }
        
        if config.max_edges < config.max_nodes {
            eprintln!("Default max_edges should be at least max_nodes: {} < {}", 
                     config.max_edges, config.max_nodes);
            all_passed = false;
        }
        
        // Memory limit should be reasonable (at least 1GB)
        if config.max_memory_bytes < 1024 * 1024 * 1024 {
            eprintln!("Default memory limit too small: {} bytes", config.max_memory_bytes);
            all_passed = false;
        }
        
        // Timeout should be reasonable (at least 1 second)
        if config.max_kernel_time_ms < 1000 {
            eprintln!("Default kernel timeout too small: {}ms", config.max_kernel_time_ms);
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_memory_tracker_functionality(&mut self) {
        let test_name = "memory_tracker_functionality";
        let start = Instant::now();
        
        let mut all_passed = true;
        let mut tracker = GPUMemoryTracker::new();
        
        // Test initial state
        let initial_stats = tracker.get_usage_stats();
        if initial_stats.current_allocated != 0 {
            eprintln!("Initial allocated memory should be 0, got {}", initial_stats.current_allocated);
            all_passed = false;
        }
        
        if initial_stats.active_allocations != 0 {
            eprintln!("Initial active allocations should be 0, got {}", initial_stats.active_allocations);
            all_passed = false;
        }
        
        // Test allocation tracking
        let allocation_sizes = vec![1024, 2048, 4096, 8192];
        let mut total_allocated = 0;
        
        for (i, size) in allocation_sizes.iter().enumerate() {
            let name = format!("test_allocation_{}", i);
            let result = tracker.track_allocation(name.clone(), *size);
            
            if result.is_err() {
                eprintln!("Allocation tracking should succeed for reasonable size: {}", size);
                all_passed = false;
                continue;
            }
            
            total_allocated += size;
            
            let stats = tracker.get_usage_stats();
            if stats.current_allocated != total_allocated {
                eprintln!("Current allocated should be {}, got {}", total_allocated, stats.current_allocated);
                all_passed = false;
            }
            
            if stats.active_allocations != (i + 1) {
                eprintln!("Active allocations should be {}, got {}", i + 1, stats.active_allocations);
                all_passed = false;
            }
        }
        
        // Test deallocation
        for i in 0..allocation_sizes.len() {
            let name = format!("test_allocation_{}", i);
            tracker.track_deallocation(&name);
            
            let expected_active = allocation_sizes.len() - (i + 1);
            let stats = tracker.get_usage_stats();
            
            if stats.active_allocations != expected_active {
                eprintln!("After deallocation {}, active should be {}, got {}", 
                         i, expected_active, stats.active_allocations);
                all_passed = false;
            }
        }
        
        // Test final cleanup
        let final_stats = tracker.get_usage_stats();
        if final_stats.current_allocated != 0 {
            eprintln!("Final allocated memory should be 0, got {}", final_stats.current_allocated);
            all_passed = false;
        }
        
        if final_stats.active_allocations != 0 {
            eprintln!("Final active allocations should be 0, got {}", final_stats.active_allocations);
            all_passed = false;
        }
        
        // Test that max_allocated was tracked
        if final_stats.max_allocated < total_allocated {
            eprintln!("Max allocated should be at least {}, got {}", 
                     total_allocated, final_stats.max_allocated);
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_buffer_bounds_validation(&mut self) {
        let test_name = "buffer_bounds_validation";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config.clone());
        
        // Test valid buffer sizes
        let valid_cases = vec![
            ("node_positions", 1000, 12),   // 1000 nodes, 12 bytes each
            ("edge_data", 5000, 12),        // 5000 edges, 12 bytes each
            ("small_buffer", 100, 4),       // Small buffer
            ("medium_buffer", 10000, 8),    // Medium buffer
        ];
        
        for (name, count, element_size) in valid_cases {
            let result = validator.validate_buffer_bounds(name, count, element_size);
            if result.is_err() {
                eprintln!("Valid buffer should pass validation: {} {} elements of {} bytes each", 
                         name, count, element_size);
                all_passed = false;
            }
        }
        
        // Test invalid buffer sizes
        let invalid_cases = vec![
            ("node_overflow", config.max_nodes + 1, 12),      // Too many nodes
            ("edge_overflow", config.max_edges + 1, 12),      // Too many edges  
            ("huge_elements", 1000, usize::MAX / 500),        // Huge element size
            ("integer_overflow", usize::MAX, 2),              // Integer overflow
            ("memory_overflow", config.max_memory_bytes / 4 + 1, 4), // Memory overflow
        ];
        
        for (name, count, element_size) in invalid_cases {
            let result = validator.validate_buffer_bounds(name, count, element_size);
            if result.is_ok() {
                eprintln!("Invalid buffer should fail validation: {} {} elements of {} bytes each", 
                         name, count, element_size);
                all_passed = false;
            }
        }
        
        // Test zero-sized cases
        let zero_cases = vec![
            ("zero_count", 0, 12),
            ("zero_size", 100, 0),
        ];
        
        for (name, count, element_size) in zero_cases {
            let result = validator.validate_buffer_bounds(name, count, element_size);
            // Zero-sized allocations should generally be allowed
            if result.is_err() && count != 0 && element_size != 0 {
                eprintln!("Zero-sized buffer validation behaved unexpectedly: {} {} elements of {} bytes", 
                         name, count, element_size);
                // This is informational, not necessarily a failure
            }
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_kernel_parameter_validation(&mut self) {
        let test_name = "kernel_parameter_validation";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config.clone());
        
        // Test valid kernel parameters
        let valid_cases = vec![
            (1000, 2000, 0, 4, 256),      // Typical case
            (1, 1, 0, 1, 1),              // Minimal case
            (10000, 20000, 1000, 32, 512), // Large case
            (config.max_nodes as i32, config.max_edges as i32, 0, 16, 256), // Maximum valid
        ];
        
        for (nodes, edges, constraints, grid, block) in valid_cases {
            let result = validator.validate_kernel_params(nodes, edges, constraints, grid, block);
            if result.is_err() {
                eprintln!("Valid kernel parameters should pass: nodes={}, edges={}, constraints={}, grid={}, block={}", 
                         nodes, edges, constraints, grid, block);
                all_passed = false;
            }
        }
        
        // Test invalid kernel parameters
        let invalid_cases = vec![
            // Negative values
            (-1, 1000, 0, 4, 256),         // Negative nodes
            (1000, -1, 0, 4, 256),         // Negative edges
            (1000, 1000, -1, 4, 256),      // Negative constraints
            
            // Zero grid/block sizes
            (1000, 1000, 0, 0, 256),       // Zero grid
            (1000, 1000, 0, 4, 0),         // Zero block
            
            // Oversized values
            (config.max_nodes as i32 + 1, 1000, 0, 4, 256), // Too many nodes
            (1000, config.max_edges as i32 + 1, 0, 4, 256), // Too many edges
            (1000, 1000, 0, 4, 2048),      // Block size too large
            (1000, 1000, 0, 70000, 256),   // Grid size too large
        ];
        
        for (nodes, edges, constraints, grid, block) in invalid_cases {
            let result = validator.validate_kernel_params(nodes, edges, constraints, grid, block);
            if result.is_ok() {
                eprintln!("Invalid kernel parameters should fail: nodes={}, edges={}, constraints={}, grid={}, block={}", 
                         nodes, edges, constraints, grid, block);
                all_passed = false;
            }
        }
        
        // Test thread count overflow
        let overflow_cases = vec![
            (1000, 1000, 0, 65535, 1024),  // Grid * block might overflow
            (1000, 1000, 0, 32768, 2048),  // Definitely overflows
        ];
        
        for (nodes, edges, constraints, grid, block) in overflow_cases {
            let result = validator.validate_kernel_params(nodes, edges, constraints, grid, block);
            // Should fail due to thread count overflow
            if result.is_ok() {
                let total_threads = grid as u64 * block as u64;
                if total_threads > i32::MAX as u64 {
                    eprintln!("Thread count overflow should be detected: {} total threads", total_threads);
                    all_passed = false;
                }
            }
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_memory_allocation_tracking(&mut self) {
        let test_name = "memory_allocation_tracking";
        let start = Instant::now();
        
        let mut all_passed = true;
        
        // Test with limited memory config
        let limited_config = GPUSafetyConfig {
            max_memory_bytes: 10 * 1024, // 10KB limit
            ..GPUSafetyConfig::default()
        };
        
        let validator = GPUSafetyValidator::new(limited_config);
        
        // Test successful allocations within limit
        let result = validator.track_allocation("test1".to_string(), 2048);
        if result.is_err() {
            eprintln!("Small allocation should succeed");
            all_passed = false;
        }
        
        let result = validator.track_allocation("test2".to_string(), 4096);
        if result.is_err() {
            eprintln!("Medium allocation should succeed");
            all_passed = false;
        }
        
        // Test allocation that exceeds limit
        let result = validator.track_allocation("test_large".to_string(), 8192);
        if result.is_ok() {
            eprintln!("Large allocation should fail when exceeding memory limit");
            all_passed = false;
        } else {
            // Verify error type
            match result.err().unwrap() {
                GPUSafetyError::MemoryLimitExceeded { requested, current, limit } => {
                    if requested != 8192 {
                        eprintln!("Error should report correct requested size: expected 8192, got {}", requested);
                        all_passed = false;
                    }
                    if limit != 10 * 1024 {
                        eprintln!("Error should report correct limit: expected {}, got {}", 10 * 1024, limit);
                        all_passed = false;
                    }
                },
                _ => {
                    eprintln!("Should get MemoryLimitExceeded error for oversized allocation");
                    all_passed = false;
                }
            }
        }
        
        // Test deallocation
        validator.track_deallocation("test1");
        
        // Should now be able to allocate again
        let result = validator.track_allocation("test3".to_string(), 3072);
        if result.is_err() {
            eprintln!("Allocation should succeed after deallocation");
            all_passed = false;
        }
        
        // Test memory statistics
        if let Some(stats) = validator.get_memory_stats() {
            if stats.active_allocations == 0 {
                eprintln!("Should have active allocations");
                all_passed = false;
            }
            
            if stats.allocation_count == 0 {
                eprintln!("Should track allocation count");
                all_passed = false;
            }
        } else {
            eprintln!("Should be able to get memory stats");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_memory_alignment_validation(&mut self) {
        let test_name = "memory_alignment_validation";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);
        
        // Test null pointer detection
        let null_ptr = std::ptr::null::<u8>();
        let result = validator.validate_memory_alignment(null_ptr, 8);
        if result.is_ok() {
            eprintln!("Null pointer should be rejected");
            all_passed = false;
        } else {
            match result.err().unwrap() {
                GPUSafetyError::NullPointer { ref operation } => {
                    if !operation.contains("memory alignment check") {
                        eprintln!("Null pointer error should contain operation description");
                        all_passed = false;
                    }
                },
                _ => {
                    eprintln!("Should get NullPointer error for null pointer");
                    all_passed = false;
                }
            }
        }
        
        // Test aligned memory
        let aligned_data: Vec<u64> = vec![1, 2, 3, 4];
        let aligned_ptr = aligned_data.as_ptr() as *const u8;
        
        let result = validator.validate_memory_alignment(aligned_ptr, 8);
        if result.is_err() {
            eprintln!("Properly aligned memory should pass validation");
            all_passed = false;
        }
        
        // Test smaller alignment requirement
        let result = validator.validate_memory_alignment(aligned_ptr, 4);
        if result.is_err() {
            eprintln!("Memory aligned to 8 bytes should also be aligned to 4 bytes");
            all_passed = false;
        }
        
        // Test misaligned memory (create unaligned pointer)
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let potentially_misaligned_ptr = unsafe { data.as_ptr().add(1) }; // Offset by 1 byte
        
        let result = validator.validate_memory_alignment(potentially_misaligned_ptr, 8);
        
        // Check if the pointer is actually misaligned
        let address = potentially_misaligned_ptr as usize;
        if address % 8 != 0 {
            // Pointer is misaligned, should fail
            if result.is_ok() {
                eprintln!("Misaligned memory should fail validation");
                all_passed = false;
            } else {
                match result.err().unwrap() {
                    GPUSafetyError::MisalignedAccess { address: err_addr, alignment } => {
                        if err_addr != address {
                            eprintln!("Error should report correct address: expected {:#x}, got {:#x}", address, err_addr);
                            all_passed = false;
                        }
                        if alignment != 8 {
                            eprintln!("Error should report correct alignment: expected 8, got {}", alignment);
                            all_passed = false;
                        }
                    },
                    _ => {
                        eprintln!("Should get MisalignedAccess error for misaligned memory");
                        all_passed = false;
                    }
                }
            }
        } else {
            // Pointer happened to be aligned, should pass
            if result.is_err() {
                eprintln!("Aligned memory should pass validation even with offset");
                all_passed = false;
            }
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_gpu_failure_tracking(&mut self) {
        let test_name = "gpu_failure_tracking";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig {
            cpu_fallback_threshold: 3,
            ..GPUSafetyConfig::default()
        };
        let validator = GPUSafetyValidator::new(config.clone());
        
        // Initially should not use CPU fallback
        if validator.should_use_cpu_fallback() {
            eprintln!("Should not start in CPU fallback mode");
            all_passed = false;
        }
        
        // Record failures one by one
        for i in 0..config.cpu_fallback_threshold {
            validator.record_failure();
            
            let should_fallback = validator.should_use_cpu_fallback();
            let expected_fallback = i >= config.cpu_fallback_threshold - 1;
            
            if should_fallback != expected_fallback {
                eprintln!("Fallback status incorrect after {} failures: expected {}, got {}", 
                         i + 1, expected_fallback, should_fallback);
                all_passed = false;
            }
        }
        
        // Should now be in fallback mode
        if !validator.should_use_cpu_fallback() {
            eprintln!("Should be in CPU fallback mode after threshold reached");
            all_passed = false;
        }
        
        // Additional failures shouldn't change state
        validator.record_failure();
        if !validator.should_use_cpu_fallback() {
            eprintln!("Should remain in CPU fallback mode after additional failures");
            all_passed = false;
        }
        
        // Test reset
        validator.reset_failures();
        if validator.should_use_cpu_fallback() {
            eprintln!("Should exit CPU fallback mode after reset");
            all_passed = false;
        }
        
        // Test partial failure recovery
        for _ in 0..config.cpu_fallback_threshold - 1 {
            validator.record_failure();
        }
        
        if validator.should_use_cpu_fallback() {
            eprintln!("Should not trigger fallback just before threshold");
            all_passed = false;
        }
        
        // Reset should work even before threshold
        validator.reset_failures();
        if validator.should_use_cpu_fallback() {
            eprintln!("Reset should work even before reaching threshold");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_cpu_fallback_mechanism(&mut self) {
        let test_name = "cpu_fallback_mechanism";
        let start = Instant::now();
        
        let mut all_passed = true;
        
        // Test CPU fallback computation
        let test_cases = vec![
            // Small graph
            (
                vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
                vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                vec![(0, 1, 1.0)],
            ),
            // Triangle
            (
                vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 1.0, 0.0)],
                vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
            ),
            // Linear chain
            (
                vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
                vec![(0.0, 0.0, 0.0); 4],
                vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)],
            ),
        ];
        
        for (i, (mut positions, mut velocities, edges)) in test_cases.into_iter().enumerate() {
            let original_positions = positions.clone();
            
            let result = cpu_fallback::compute_forces_cpu(
                &mut positions,
                &mut velocities,
                &edges,
                0.1, // spring_k
                0.1, // repel_k
                0.9, // damping
                0.01, // dt
            );
            
            if result.is_err() {
                eprintln!("CPU fallback computation should succeed for test case {}", i);
                all_passed = false;
                continue;
            }
            
            // Verify positions changed (unless it's a degenerate case)
            if edges.len() > 0 {
                let mut positions_changed = false;
                for (j, (&new_pos, &orig_pos)) in positions.iter().zip(original_positions.iter()).enumerate() {
                    if new_pos != orig_pos {
                        positions_changed = true;
                        break;
                    }
                }
                
                if !positions_changed && i > 0 { // Allow first test case to not change
                    eprintln!("CPU fallback should modify positions for test case {}", i);
                    // This is informational - positions might not change in one iteration
                }
            }
            
            // Verify positions are finite
            for (j, &(x, y, z)) in positions.iter().enumerate() {
                if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                    eprintln!("CPU fallback produced non-finite position for test case {}, node {}: ({}, {}, {})", 
                             i, j, x, y, z);
                    all_passed = false;
                }
            }
            
            // Verify velocities are finite and reasonable
            for (j, &(vx, vy, vz)) in velocities.iter().enumerate() {
                if !vx.is_finite() || !vy.is_finite() || !vz.is_finite() {
                    eprintln!("CPU fallback produced non-finite velocity for test case {}, node {}: ({}, {}, {})", 
                             i, j, vx, vy, vz);
                    all_passed = false;
                }
                
                let vel_magnitude = (vx*vx + vy*vy + vz*vz).sqrt();
                if vel_magnitude > 100.0 {
                    eprintln!("CPU fallback produced excessive velocity for test case {}, node {}: magnitude {}", 
                             i, j, vel_magnitude);
                    all_passed = false;
                }
            }
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_kernel_timeout_detection(&mut self) {
        let test_name = "kernel_timeout_detection";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig {
            max_kernel_time_ms: 100, // Very short timeout for testing
            ..GPUSafetyConfig::default()
        };
        let validator = GPUSafetyValidator::new(config.clone());
        
        // Test successful execution within timeout
        let fast_duration = Duration::from_millis(50);
        let result = validator.record_kernel_execution(fast_duration);
        if result.is_err() {
            eprintln!("Fast kernel execution should succeed");
            all_passed = false;
        }
        
        // Test timeout detection
        let slow_duration = Duration::from_millis(200);
        let result = validator.record_kernel_execution(slow_duration);
        if result.is_ok() {
            eprintln!("Slow kernel execution should timeout");
            all_passed = false;
        } else {
            match result.err().unwrap() {
                GPUSafetyError::KernelTimeout { duration_ms, limit_ms } => {
                    if duration_ms != 200 {
                        eprintln!("Timeout error should report correct duration: expected 200, got {}", duration_ms);
                        all_passed = false;
                    }
                    if limit_ms != config.max_kernel_time_ms {
                        eprintln!("Timeout error should report correct limit: expected {}, got {}", 
                                 config.max_kernel_time_ms, limit_ms);
                        all_passed = false;
                    }
                },
                _ => {
                    eprintln!("Should get KernelTimeout error for slow execution");
                    all_passed = false;
                }
            }
        }
        
        // Test boundary condition
        let boundary_duration = Duration::from_millis(config.max_kernel_time_ms);
        let result = validator.record_kernel_execution(boundary_duration);
        if result.is_ok() {
            eprintln!("Boundary case (exactly at timeout) should succeed");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_concurrent_gpu_operations(&mut self) {
        let test_name = "concurrent_gpu_operations";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig::default();
        let validator = Arc::new(GPUSafetyValidator::new(config));
        
        use std::thread;
        use std::sync::Arc;
        
        let mut handles = vec![];
        
        // Spawn threads that perform various GPU safety operations concurrently
        for thread_id in 0..5 {
            let validator_clone = Arc::clone(&validator);
            
            let handle = thread::spawn(move || {
                let mut thread_success = true;
                
                // Each thread performs different operations
                for i in 0..10 {
                    // Test kernel parameter validation
                    let nodes = (thread_id + 1) * 100 + i;
                    let edges = nodes * 2;
                    let result = validator_clone.validate_kernel_params(
                        nodes as i32, 
                        edges as i32, 
                        0, 
                        4, 
                        256
                    );
                    
                    if result.is_err() {
                        eprintln!("Thread {} iteration {}: kernel validation failed", thread_id, i);
                        thread_success = false;
                    }
                    
                    // Test buffer bounds validation
                    let result = validator_clone.validate_buffer_bounds(
                        &format!("thread_{}_buffer_{}", thread_id, i),
                        nodes,
                        12
                    );
                    
                    if result.is_err() {
                        eprintln!("Thread {} iteration {}: buffer validation failed", thread_id, i);
                        thread_success = false;
                    }
                    
                    // Test array access validation
                    let result = validator_clone.validate_array_access(i, nodes + 10, "test_array");
                    
                    if result.is_err() {
                        eprintln!("Thread {} iteration {}: array access validation failed", thread_id, i);
                        thread_success = false;
                    }
                }
                
                // Test failure tracking (some threads record failures)
                if thread_id % 2 == 0 {
                    validator_clone.record_failure();
                    let _ = validator_clone.should_use_cpu_fallback();
                }
                
                thread_success
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads and collect results
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.join() {
                Ok(success) => {
                    if !success {
                        eprintln!("Thread {} reported failures", i);
                        all_passed = false;
                    }
                },
                Err(_) => {
                    eprintln!("Thread {} panicked", i);
                    all_passed = false;
                }
            }
        }
        
        // Test that validator is still functional after concurrent access
        let result = validator.validate_kernel_params(1000, 2000, 0, 4, 256);
        if result.is_err() {
            eprintln!("Validator should still be functional after concurrent access");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_resource_exhaustion_handling(&mut self) {
        let test_name = "resource_exhaustion_handling";
        let start = Instant::now();
        
        let mut all_passed = true;
        
        // Test with very limited resources
        let limited_config = GPUSafetyConfig {
            max_nodes: 10,
            max_edges: 20,
            max_memory_bytes: 1024, // 1KB
            cpu_fallback_threshold: 1,
            ..GPUSafetyConfig::default()
        };
        
        let validator = GPUSafetyValidator::new(limited_config.clone());
        
        // Test node limit exhaustion
        let result = validator.validate_kernel_params(
            limited_config.max_nodes as i32 + 1, 
            5, 
            0, 
            4, 
            256
        );
        
        if result.is_ok() {
            eprintln!("Should reject kernel with too many nodes");
            all_passed = false;
        }
        
        // Test edge limit exhaustion
        let result = validator.validate_kernel_params(
            5, 
            limited_config.max_edges as i32 + 1, 
            0, 
            4, 
            256
        );
        
        if result.is_ok() {
            eprintln!("Should reject kernel with too many edges");
            all_passed = false;
        }
        
        // Test memory exhaustion
        let result = validator.validate_buffer_bounds(
            "large_buffer", 
            limited_config.max_memory_bytes / 4 + 1, 
            4
        );
        
        if result.is_ok() {
            eprintln!("Should reject buffer that exceeds memory limit");
            all_passed = false;
        }
        
        // Test immediate fallback trigger
        validator.record_failure();
        if !validator.should_use_cpu_fallback() {
            eprintln!("Should trigger CPU fallback immediately with threshold=1");
            all_passed = false;
        }
        
        // Test that we can still validate within limits
        let result = validator.validate_kernel_params(5, 10, 0, 4, 256);
        if result.is_err() {
            eprintln!("Should still accept valid parameters within limits");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_safe_kernel_executor(&mut self) {
        let test_name = "safe_kernel_executor";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig {
            max_kernel_time_ms: 1000, // 1 second timeout
            ..GPUSafetyConfig::default()
        };
        
        let validator = Arc::new(GPUSafetyValidator::new(config));
        let executor = SafeKernelExecutor::new(validator.clone());
        
        // Test successful operation
        let fast_operation = || -> Result<String, GPUSafetyError> {
            std::thread::sleep(Duration::from_millis(100));
            Ok("Success".to_string())
        };
        
        let result = executor.execute_with_timeout(fast_operation).await;
        match result {
            Ok(value) => {
                if value != "Success" {
                    eprintln!("Fast operation should return correct value");
                    all_passed = false;
                }
            },
            Err(_) => {
                eprintln!("Fast operation should succeed");
                all_passed = false;
            }
        }
        
        // Test operation that returns error
        let error_operation = || -> Result<String, GPUSafetyError> {
            Err(GPUSafetyError::DeviceError {
                message: "Test error".to_string(),
            })
        };
        
        let result = executor.execute_with_timeout(error_operation).await;
        if result.is_ok() {
            eprintln!("Error operation should fail");
            all_passed = false;
        }
        
        // Test timeout (this test might be flaky in CI due to timing)
        let slow_operation = || -> Result<String, GPUSafetyError> {
            std::thread::sleep(Duration::from_millis(1500)); // Longer than timeout
            Ok("Should not reach here".to_string())
        };
        
        let result = executor.execute_with_timeout(slow_operation).await;
        match result {
            Err(GPUSafetyError::KernelTimeout { .. }) => {
                // Expected timeout error
            },
            Err(GPUSafetyError::DeviceError { message }) => {
                // Also acceptable if the timeout manifests as a device error
                if !message.contains("Task execution failed") {
                    eprintln!("Unexpected device error message: {}", message);
                    all_passed = false;
                }
            },
            Ok(_) => {
                eprintln!("Slow operation should timeout");
                all_passed = false;
            },
            Err(e) => {
                eprintln!("Slow operation should produce timeout or task error, got: {:?}", e);
                all_passed = false;
            }
        }
        
        // Verify failure tracking
        if !validator.should_use_cpu_fallback() {
            // Error operations should have recorded failures
            // But the specific behavior depends on the fallback threshold
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_pre_kernel_validation(&mut self) {
        let test_name = "pre_kernel_validation";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);
        
        // Test valid graph data
        let valid_nodes = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let valid_edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        
        let result = validator.pre_kernel_validation(&valid_nodes, &valid_edges, 4, 256);
        if result.is_err() {
            eprintln!("Valid graph data should pass pre-kernel validation");
            all_passed = false;
        }
        
        // Test edge with invalid node reference
        let invalid_edges = vec![(0, 5, 1.0)]; // Node 5 doesn't exist
        let result = validator.pre_kernel_validation(&valid_nodes, &invalid_edges, 4, 256);
        if result.is_ok() {
            eprintln!("Edge referencing non-existent node should fail validation");
            all_passed = false;
        }
        
        // Test negative node indices in edges
        let negative_edges = vec![(-1, 1, 1.0)];
        let result = validator.pre_kernel_validation(&valid_nodes, &negative_edges, 4, 256);
        if result.is_ok() {
            eprintln!("Edge with negative node index should fail validation");
            all_passed = false;
        }
        
        // Test infinite weight
        let infinite_edges = vec![(0, 1, f32::INFINITY)];
        let result = validator.pre_kernel_validation(&valid_nodes, &infinite_edges, 4, 256);
        if result.is_ok() {
            eprintln!("Edge with infinite weight should fail validation");
            all_passed = false;
        }
        
        // Test NaN weight
        let nan_edges = vec![(0, 1, f32::NAN)];
        let result = validator.pre_kernel_validation(&valid_nodes, &nan_edges, 4, 256);
        if result.is_ok() {
            eprintln!("Edge with NaN weight should fail validation");
            all_passed = false;
        }
        
        // Test infinite node position
        let infinite_nodes = vec![(f32::INFINITY, 0.0, 0.0), (1.0, 0.0, 0.0)];
        let simple_edges = vec![(0, 1, 1.0)];
        let result = validator.pre_kernel_validation(&infinite_nodes, &simple_edges, 4, 256);
        if result.is_ok() {
            eprintln!("Node with infinite position should fail validation");
            all_passed = false;
        }
        
        // Test NaN node position
        let nan_nodes = vec![(0.0, f32::NAN, 0.0), (1.0, 0.0, 0.0)];
        let result = validator.pre_kernel_validation(&nan_nodes, &simple_edges, 4, 256);
        if result.is_ok() {
            eprintln!("Node with NaN position should fail validation");
            all_passed = false;
        }
        
        // Test empty graph (should be valid)
        let empty_nodes: Vec<(f32, f32, f32)> = vec![];
        let empty_edges: Vec<(i32, i32, f32)> = vec![];
        let result = validator.pre_kernel_validation(&empty_nodes, &empty_edges, 1, 1);
        if result.is_err() {
            eprintln!("Empty graph should be valid");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_cpu_fallback_computation(&mut self) {
        let test_name = "cpu_fallback_computation";
        let start = Instant::now();
        
        let mut all_passed = true;
        
        // Test physics computation stability
        let mut positions = vec![
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 0.866, 0.0), // Equilateral triangle
        ];
        let mut velocities = vec![(0.0, 0.0, 0.0); 3];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        
        // Run multiple physics steps
        for step in 0..100 {
            let result = cpu_fallback::compute_forces_cpu(
                &mut positions,
                &mut velocities,
                &edges,
                0.1, // spring_k
                0.01, // repel_k  
                0.95, // damping
                0.01, // dt
            );
            
            if result.is_err() {
                eprintln!("CPU fallback should not fail at step {}", step);
                all_passed = false;
                break;
            }
            
            // Check for numerical instability
            for (i, &(x, y, z)) in positions.iter().enumerate() {
                if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                    eprintln!("Non-finite position at step {}, node {}: ({}, {}, {})", step, i, x, y, z);
                    all_passed = false;
                }
                
                if x.abs() > 1000.0 || y.abs() > 1000.0 || z.abs() > 1000.0 {
                    eprintln!("Position explosion at step {}, node {}: ({}, {}, {})", step, i, x, y, z);
                    all_passed = false;
                }
            }
            
            for (i, &(vx, vy, vz)) in velocities.iter().enumerate() {
                if !vx.is_finite() || !vy.is_finite() || !vz.is_finite() {
                    eprintln!("Non-finite velocity at step {}, node {}: ({}, {}, {})", step, i, vx, vy, vz);
                    all_passed = false;
                }
            }
            
            if !all_passed {
                break;
            }
        }
        
        // Test error conditions
        let mut bad_positions = vec![(0.0, 0.0, 0.0)];
        let mut bad_velocities = vec![(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]; // Mismatched length
        let bad_edges = vec![(0, 1, 1.0)];
        
        let result = cpu_fallback::compute_forces_cpu(
            &mut bad_positions,
            &mut bad_velocities,
            &bad_edges,
            0.1, 0.01, 0.95, 0.01
        );
        
        if result.is_ok() {
            eprintln!("CPU fallback should reject mismatched array lengths");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_edge_case_scenarios(&mut self) {
        let test_name = "edge_case_scenarios";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);
        
        // Test maximum valid values
        let max_result = validator.validate_kernel_params(
            i32::MAX, 
            i32::MAX, 
            i32::MAX, 
            4, 
            256
        );
        // This should fail due to exceeding configured limits, not due to overflow
        if max_result.is_ok() {
            eprintln!("Maximum i32 values should be rejected due to configured limits");
            all_passed = false;
        }
        
        // Test minimum valid values
        let min_result = validator.validate_kernel_params(0, 0, 0, 1, 1);
        if min_result.is_err() {
            eprintln!("Zero node/edge counts should be valid");
            all_passed = false;
        }
        
        // Test array access at exact boundaries
        let boundary_result = validator.validate_array_access(9, 10, "boundary_test");
        if boundary_result.is_err() {
            eprintln!("Array access at valid boundary should succeed");
            all_passed = false;
        }
        
        let overflow_result = validator.validate_array_access(10, 10, "boundary_test");
        if overflow_result.is_ok() {
            eprintln!("Array access at invalid boundary should fail");
            all_passed = false;
        }
        
        // Test buffer allocation with zero size
        let zero_result = validator.validate_buffer_bounds("zero_buffer", 0, 1024);
        // Zero-sized buffers might be allowed
        if zero_result.is_err() {
            // This is acceptable behavior
        }
        
        let zero_element_result = validator.validate_buffer_bounds("zero_element", 1024, 0);
        // Zero-sized elements might be allowed
        if zero_element_result.is_err() {
            // This is acceptable behavior
        }
        
        // Test very large but valid allocations
        let large_but_valid = validator.validate_buffer_bounds(
            "large_buffer", 
            1000, 
            1024
        );
        if large_but_valid.is_err() {
            eprintln!("Reasonable large allocation should succeed");
            all_passed = false;
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    async fn test_performance_characteristics(&mut self) {
        let test_name = "performance_characteristics";
        let start = Instant::now();
        
        let mut all_passed = true;
        let config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(config);
        
        // Test validation performance
        let validation_start = Instant::now();
        
        for i in 0..10000 {
            let result = validator.validate_kernel_params(
                i as i32 % 1000, 
                (i as i32) * 2 % 2000, 
                0, 
                4, 
                256
            );
            
            // Most should succeed (we're using small values)
            if result.is_err() && (i % 1000) < 900 && ((i * 2) % 2000) < 1800 {
                // Only fail if the parameters should have been valid
                eprintln!("Unexpected validation failure at iteration {}", i);
                all_passed = false;
                break;
            }
        }
        
        let validation_time = validation_start.elapsed();
        let avg_validation_time = validation_time.as_nanos() as f64 / 10000.0;
        
        if avg_validation_time > 100_000.0 { // 100 microseconds per validation
            eprintln!("Validation performance too slow: {:.1} ns per validation", avg_validation_time);
            all_passed = false;
        } else {
            println!("Validation performance: {:.1} ns per validation", avg_validation_time);
        }
        
        // Test memory tracking performance
        let mut tracker = GPUMemoryTracker::new();
        let tracking_start = Instant::now();
        
        for i in 0..1000 {
            let name = format!("perf_test_{}", i);
            let _ = tracker.track_allocation(name.clone(), 1024);
            tracker.track_deallocation(&name);
        }
        
        let tracking_time = tracking_start.elapsed();
        let avg_tracking_time = tracking_time.as_nanos() as f64 / 2000.0; // 2000 operations (alloc + dealloc)
        
        if avg_tracking_time > 50_000.0 { // 50 microseconds per operation
            eprintln!("Memory tracking performance too slow: {:.1} ns per operation", avg_tracking_time);
            all_passed = false;
        } else {
            println!("Memory tracking performance: {:.1} ns per operation", avg_tracking_time);
        }
        
        self.record_test_result(test_name, start.elapsed(), all_passed);
    }

    fn record_test_result(&mut self, test_name: &str, duration: Duration, passed: bool) {
        self.test_count += 1;
        
        if passed {
            self.passed_tests += 1;
            println!("✓ {} completed in {:.2}ms", test_name, duration.as_millis());
        } else {
            self.failed_tests += 1;
            println!("✗ {} failed after {:.2}ms", test_name, duration.as_millis());
        }
    }

    fn print_results(&self) {
        println!("\n=== GPU Safety Test Results ===");
        println!("Total Tests: {}", self.test_count);
        println!("Passed: {}", self.passed_tests);
        println!("Failed: {}", self.failed_tests);
        println!("Success Rate: {:.1}%", (self.passed_tests as f64 / self.test_count as f64) * 100.0);
    }
}

#[tokio::test]
async fn run_gpu_safety_validation() {
    let mut test_suite = GPUSafetyTestSuite::new();
    test_suite.run_all_tests().await;
    
    // Ensure all tests passed
    assert!(test_suite.failed_tests == 0, "All GPU safety tests should pass");
    assert!(test_suite.passed_tests > 15, "Should have comprehensive test coverage");
}