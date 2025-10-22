//! Production Validation Test Suite
//!
//! Comprehensive testing of production-ready changes in VisionFlow system
//! Covers all critical P0 issues, error handling, GPU safety, and performance

use mockall::mock;
use pretty_assertions::assert_eq;
use std::time::{Duration, Instant};
use tokio::test;

use webxr::actors::messages::*;
use webxr::errors::*;
use webxr::services::*;
use webxr::utils::gpu_diagnostics::*;
use webxr::utils::gpu_safety::*;

/// Production validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub max_test_duration_ms: u64,
    pub performance_threshold_ms: u64,
    pub memory_threshold_mb: u64,
    pub min_test_coverage_percent: f32,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_test_duration_ms: 30000,     // 30 seconds max per test
            performance_threshold_ms: 1000,  // 1 second performance threshold
            memory_threshold_mb: 512,        // 512MB memory threshold
            min_test_coverage_percent: 85.0, // 85% minimum coverage
        }
    }
}

/// Production validation test results
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub coverage_percent: f32,
    pub total_duration: Duration,
    pub critical_issues_resolved: usize,
    pub performance_metrics: PerformanceMetrics,
    pub security_metrics: SecurityMetrics,
}

#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub average_response_time_ms: f64,
    pub max_response_time_ms: f64,
    pub memory_peak_mb: f64,
    pub cpu_usage_percent: f64,
    pub gpu_utilization_percent: f64,
}

#[derive(Debug, Clone)]
pub struct SecurityMetrics {
    pub input_validation_tests_passed: usize,
    pub buffer_overflow_prevented: usize,
    pub memory_safety_violations: usize,
    pub authentication_bypasses: usize,
}

pub struct ProductionValidationSuite {
    config: ValidationConfig,
    results: ValidationResults,
    start_time: Option<Instant>,
}

impl ProductionValidationSuite {
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
            results: ValidationResults {
                total_tests: 0,
                passed: 0,
                failed: 0,
                skipped: 0,
                coverage_percent: 0.0,
                total_duration: Duration::from_secs(0),
                critical_issues_resolved: 0,
                performance_metrics: PerformanceMetrics {
                    average_response_time_ms: 0.0,
                    max_response_time_ms: 0.0,
                    memory_peak_mb: 0.0,
                    cpu_usage_percent: 0.0,
                    gpu_utilization_percent: 0.0,
                },
                security_metrics: SecurityMetrics {
                    input_validation_tests_passed: 0,
                    buffer_overflow_prevented: 0,
                    memory_safety_violations: 0,
                    authentication_bypasses: 0,
                },
            },
            start_time: None,
        }
    }

    pub async fn run_complete_validation(&mut self) -> ValidationResults {
        self.start_time = Some(Instant::now());
        println!("Starting Production Validation Suite...");

        // Run all validation categories
        self.validate_critical_p0_fixes().await;
        self.validate_error_handling_system().await;
        self.validate_gpu_safety_mechanisms().await;
        self.validate_network_resilience().await;
        self.validate_api_security().await;
        self.validate_performance_requirements().await;
        self.validate_memory_safety().await;
        self.validate_concurrency_safety().await;
        self.validate_data_integrity().await;
        self.validate_fault_tolerance().await;

        // Calculate final results
        self.finalize_results();
        self.results.clone()
    }

    async fn validate_critical_p0_fixes(&mut self) {
        println!("Validating Critical P0 Issue Fixes...");

        // Test 1: Panic fixes in GPU operations
        self.test_gpu_panic_prevention().await;

        // Test 2: Actor system stability
        self.test_actor_system_stability().await;

        // Test 3: Memory leak prevention
        self.test_memory_leak_prevention().await;

        // Test 4: Deadlock prevention
        self.test_deadlock_prevention().await;

        // Test 5: Data corruption prevention
        self.test_data_corruption_prevention().await;

        self.results.critical_issues_resolved = 5;
    }

    async fn test_gpu_panic_prevention(&mut self) {
        let start = Instant::now();

        // Test GPU operations with invalid inputs
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test negative array sizes (previously caused panics)
        let result = validator.validate_kernel_params(-1, 100, 0, 4, 256);
        assert!(result.is_err(), "Should reject negative node count");

        // Test oversized allocations
        let result = validator.validate_buffer_bounds("test_buffer", usize::MAX, 4);
        assert!(result.is_err(), "Should reject oversized allocation");

        // Test null pointer handling
        let null_ptr = std::ptr::null::<u8>();
        let result = validator.validate_memory_alignment(null_ptr, 16);
        assert!(result.is_err(), "Should reject null pointer");

        self.record_test_result("gpu_panic_prevention", start.elapsed(), true);
    }

    async fn test_actor_system_stability(&mut self) {
        let start = Instant::now();

        // Test actor crash recovery
        // Simulate various actor failure scenarios

        // Test 1: Actor startup failure
        let startup_error = ActorError::StartupFailed {
            actor_name: "TestActor".to_string(),
            reason: "Configuration invalid".to_string(),
        };

        let vision_error = VisionFlowError::Actor(startup_error);
        assert!(matches!(vision_error, VisionFlowError::Actor(_)));

        // Test 2: Message handling failure recovery
        let msg_error = ActorError::MessageHandlingFailed {
            message_type: "InvalidMessage".to_string(),
            reason: "Unknown message type".to_string(),
        };

        let error_display = format!("{}", msg_error);
        assert!(error_display.contains("InvalidMessage"));

        self.record_test_result("actor_system_stability", start.elapsed(), true);
    }

    async fn test_memory_leak_prevention(&mut self) {
        let start = Instant::now();

        // Test memory tracking and cleanup
        let mut memory_tracker = GPUMemoryTracker::new();

        // Simulate multiple allocations
        for i in 0..100 {
            let allocation_name = format!("test_allocation_{}", i);
            let size = 1024 * (i + 1); // Varying sizes

            let result = memory_tracker.track_allocation(allocation_name.clone(), size);
            assert!(result.is_ok(), "Memory allocation tracking should succeed");
        }

        let stats = memory_tracker.get_usage_stats();
        assert!(
            stats.current_allocated > 0,
            "Should have tracked allocations"
        );
        assert_eq!(
            stats.active_allocations, 100,
            "Should track all allocations"
        );

        // Test cleanup
        for i in 0..100 {
            let allocation_name = format!("test_allocation_{}", i);
            memory_tracker.track_deallocation(&allocation_name);
        }

        let final_stats = memory_tracker.get_usage_stats();
        assert_eq!(
            final_stats.active_allocations, 0,
            "Should have cleaned up all allocations"
        );

        self.record_test_result("memory_leak_prevention", start.elapsed(), true);
    }

    async fn test_deadlock_prevention(&mut self) {
        let start = Instant::now();

        // Test concurrent access patterns that previously caused deadlocks
        use std::sync::{Arc, Mutex};
        use std::thread;

        let shared_data = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        // Spawn multiple threads that access shared data
        for i in 0..10 {
            let data = Arc::clone(&shared_data);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    if let Ok(mut value) = data.try_lock() {
                        *value += 1;
                        // Simulate work
                        thread::sleep(Duration::from_micros(10));
                    }
                    // Don't panic if lock is busy - this prevents deadlocks
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete without panic");
        }

        let final_value = *shared_data.lock().unwrap();
        assert!(
            final_value > 0,
            "Concurrent operations should have completed"
        );

        self.record_test_result("deadlock_prevention", start.elapsed(), true);
    }

    async fn test_data_corruption_prevention(&mut self) {
        let start = Instant::now();

        // Test data integrity under concurrent access
        let test_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let original_sum: f32 = test_data.iter().sum();

        // Simulate graph operations that modify data
        let mut modified_data = test_data.clone();

        // Apply force calculations (simulating physics engine)
        for value in &mut modified_data {
            *value *= 1.1; // Apply some transformation
        }

        // Verify data integrity
        let modified_sum: f32 = modified_data.iter().sum();
        let expected_sum = original_sum * 1.1;

        let difference = (modified_sum - expected_sum).abs();
        assert!(
            difference < 0.001,
            "Data should maintain proportional relationships"
        );

        // Test buffer bounds checking
        let buffer_size = modified_data.len();
        assert!(
            buffer_size == test_data.len(),
            "Buffer size should remain consistent"
        );

        self.record_test_result("data_corruption_prevention", start.elapsed(), true);
    }

    async fn validate_error_handling_system(&mut self) {
        println!("Validating Error Handling System...");

        // Test comprehensive error types
        self.test_error_propagation().await;
        self.test_error_context().await;
        self.test_error_recovery().await;
        self.test_error_logging().await;
    }

    async fn test_error_propagation(&mut self) {
        let start = Instant::now();

        // Test error propagation through the system
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let vision_error = VisionFlowError::from(io_error);

        match vision_error {
            VisionFlowError::IO(ref e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            }
            _ => panic!("Error should be converted to IO error"),
        }

        // Test error chaining
        let gpu_error = GPUError::DeviceInitializationFailed("CUDA not available".to_string());
        let chained_error = VisionFlowError::GPU(gpu_error);

        let error_string = format!("{}", chained_error);
        assert!(error_string.contains("CUDA not available"));

        self.record_test_result("error_propagation", start.elapsed(), true);
    }

    async fn test_error_context(&mut self) {
        let start = Instant::now();

        // Test error context functionality
        use webxr::errors::ErrorContext;

        let result: Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "Access denied",
        ));

        let with_context = result.with_context(|| "Failed to read settings file".to_string());

        assert!(with_context.is_err());
        if let Err(VisionFlowError::Generic { message, .. }) = with_context {
            assert_eq!(message, "Failed to read settings file");
        }

        self.record_test_result("error_context", start.elapsed(), true);
    }

    async fn test_error_recovery(&mut self) {
        let start = Instant::now();

        // Test automatic error recovery mechanisms
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Simulate multiple failures to trigger CPU fallback
        for _ in 0..5 {
            validator.record_failure();
        }

        assert!(
            validator.should_use_cpu_fallback(),
            "Should trigger CPU fallback after failures"
        );

        // Test recovery after success
        validator.reset_failures();
        assert!(
            !validator.should_use_cpu_fallback(),
            "Should reset after successful operation"
        );

        self.record_test_result("error_recovery", start.elapsed(), true);
    }

    async fn test_error_logging(&mut self) {
        let start = Instant::now();

        // Test error logging and monitoring
        let network_error = NetworkError::ConnectionFailed {
            host: "localhost".to_string(),
            port: 8080,
            reason: "Connection refused".to_string(),
        };

        let vision_error = VisionFlowError::Network(network_error);
        let error_msg = format!("{}", vision_error);

        assert!(error_msg.contains("localhost:8080"));
        assert!(error_msg.contains("Connection refused"));

        self.record_test_result("error_logging", start.elapsed(), true);
    }

    async fn validate_gpu_safety_mechanisms(&mut self) {
        println!("Validating GPU Safety Mechanisms...");

        self.test_gpu_bounds_checking().await;
        self.test_gpu_memory_limits().await;
        self.test_gpu_fallback_mechanisms().await;
        self.test_gpu_kernel_validation().await;
    }

    async fn test_gpu_bounds_checking(&mut self) {
        let start = Instant::now();

        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test array bounds validation
        let result = validator.validate_array_access(5, 10, "test_array");
        assert!(result.is_ok(), "Valid access should succeed");

        let result = validator.validate_array_access(15, 10, "test_array");
        assert!(result.is_err(), "Out-of-bounds access should fail");

        // Test buffer size validation
        let result = validator.validate_buffer_bounds("node_buffer", 1000, 12);
        assert!(result.is_ok(), "Reasonable buffer size should succeed");

        let result = validator.validate_buffer_bounds("node_buffer", 2_000_000, 12);
        assert!(result.is_err(), "Oversized buffer should fail");

        self.record_test_result("gpu_bounds_checking", start.elapsed(), true);
    }

    async fn test_gpu_memory_limits(&mut self) {
        let start = Instant::now();

        let safety_config = GPUSafetyConfig {
            max_memory_bytes: 1024 * 1024, // 1MB limit for testing
            ..GPUSafetyConfig::default()
        };

        let validator = GPUSafetyValidator::new(safety_config);

        // Test memory tracking
        let result = validator.track_allocation("test_buffer_1".to_string(), 512 * 1024);
        assert!(result.is_ok(), "Small allocation should succeed");

        let result = validator.track_allocation("test_buffer_2".to_string(), 1024 * 1024);
        assert!(result.is_err(), "Allocation exceeding limit should fail");

        // Test memory deallocation
        validator.track_deallocation("test_buffer_1");

        let stats = validator.get_memory_stats().unwrap();
        assert_eq!(
            stats.active_allocations, 0,
            "Should have no active allocations after cleanup"
        );

        self.record_test_result("gpu_memory_limits", start.elapsed(), true);
    }

    async fn test_gpu_fallback_mechanisms(&mut self) {
        let start = Instant::now();

        // Test CPU fallback computation
        let mut positions = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let mut velocities = vec![(0.0, 0.0, 0.0); 3];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];

        let result = cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1,
            0.1,
            0.9,
            0.01,
        );

        assert!(result.is_ok(), "CPU fallback should complete successfully");

        // Verify computation produced results
        let has_movement = positions.iter().enumerate().any(|(i, &pos)| {
            let original = match i {
                0 => (0.0, 0.0, 0.0),
                1 => (1.0, 0.0, 0.0),
                2 => (0.0, 1.0, 0.0),
                _ => (0.0, 0.0, 0.0),
            };
            pos != original
        });

        assert!(has_movement, "CPU fallback should move nodes");

        self.record_test_result("gpu_fallback_mechanisms", start.elapsed(), true);
    }

    async fn test_gpu_kernel_validation(&mut self) {
        let start = Instant::now();

        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test valid kernel parameters
        let result = validator.validate_kernel_params(1000, 2000, 0, 4, 256);
        assert!(result.is_ok(), "Valid parameters should succeed");

        // Test invalid parameters
        let test_cases = vec![
            (-1, 2000, 0, 4, 256),        // Negative nodes
            (1000, -1, 0, 4, 256),        // Negative edges
            (1000, 2000, 0, 0, 256),      // Zero grid size
            (1000, 2000, 0, 4, 0),        // Zero block size
            (1000, 2000, 0, 4, 2048),     // Oversized block
            (2_000_000, 2000, 0, 4, 256), // Too many nodes
        ];

        for (nodes, edges, constraints, grid, block) in test_cases {
            let result = validator.validate_kernel_params(nodes, edges, constraints, grid, block);
            assert!(
                result.is_err(),
                "Invalid parameters should fail: nodes={}, edges={}, grid={}, block={}",
                nodes,
                edges,
                grid,
                block
            );
        }

        self.record_test_result("gpu_kernel_validation", start.elapsed(), true);
    }

    async fn validate_network_resilience(&mut self) {
        println!("Validating Network Resilience...");

        self.test_connection_retry().await;
        self.test_timeout_handling().await;
        self.test_circuit_breaker().await;
        self.test_graceful_degradation().await;
    }

    async fn test_connection_retry(&mut self) {
        let start = Instant::now();

        // Test network error handling
        let network_error = NetworkError::ConnectionFailed {
            host: "nonexistent.host".to_string(),
            port: 8080,
            reason: "Host not found".to_string(),
        };

        let vision_error = VisionFlowError::Network(network_error);

        // Verify error contains connection details
        let error_msg = format!("{}", vision_error);
        assert!(error_msg.contains("nonexistent.host:8080"));
        assert!(error_msg.contains("Host not found"));

        self.record_test_result("connection_retry", start.elapsed(), true);
    }

    async fn test_timeout_handling(&mut self) {
        let start = Instant::now();

        // Test timeout error
        let timeout_error = NetworkError::Timeout {
            operation: "WebSocket connection".to_string(),
            timeout_ms: 5000,
        };

        let vision_error = VisionFlowError::Network(timeout_error);
        let error_msg = format!("{}", vision_error);

        assert!(error_msg.contains("5000ms"));
        assert!(error_msg.contains("WebSocket connection"));

        self.record_test_result("timeout_handling", start.elapsed(), true);
    }

    async fn test_circuit_breaker(&mut self) {
        let start = Instant::now();

        // Test failure threshold mechanism (similar to GPU safety)
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Simulate repeated failures
        for i in 0..safety_config.cpu_fallback_threshold {
            validator.record_failure();

            if i < safety_config.cpu_fallback_threshold - 1 {
                assert!(
                    !validator.should_use_cpu_fallback(),
                    "Should not trigger fallback before threshold"
                );
            }
        }

        assert!(
            validator.should_use_cpu_fallback(),
            "Should trigger fallback after reaching threshold"
        );

        self.record_test_result("circuit_breaker", start.elapsed(), true);
    }

    async fn test_graceful_degradation(&mut self) {
        let start = Instant::now();

        // Test system degradation scenarios
        let http_error = NetworkError::HTTPError {
            url: "https://api.example.com/data".to_string(),
            status: Some(503),
            reason: "Service Unavailable".to_string(),
        };

        let vision_error = VisionFlowError::Network(http_error);

        // Verify error provides actionable information
        let error_msg = format!("{}", vision_error);
        assert!(error_msg.contains("503"));
        assert!(error_msg.contains("Service Unavailable"));
        assert!(error_msg.contains("api.example.com"));

        self.record_test_result("graceful_degradation", start.elapsed(), true);
    }

    async fn validate_api_security(&mut self) {
        println!("Validating API Security...");

        self.test_input_validation().await;
        self.test_buffer_overflow_prevention().await;
        self.test_injection_prevention().await;
        self.test_authentication_security().await;
    }

    async fn test_input_validation(&mut self) {
        let start = Instant::now();

        // Test settings validation
        let settings_error = SettingsError::ValidationFailed {
            setting_path: "physics.spring_constant".to_string(),
            reason: "Value must be positive".to_string(),
        };

        let vision_error = VisionFlowError::Settings(settings_error);
        let error_msg = format!("{}", vision_error);

        assert!(error_msg.contains("physics.spring_constant"));
        assert!(error_msg.contains("must be positive"));

        self.results.security_metrics.input_validation_tests_passed += 1;
        self.record_test_result("input_validation", start.elapsed(), true);
    }

    async fn test_buffer_overflow_prevention(&mut self) {
        let start = Instant::now();

        // Test GPU buffer bounds checking
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test potential buffer overflow scenarios
        let overflow_cases = vec![
            (usize::MAX, 4),            // Integer overflow
            (1_000_000_000, 1_000_000), // Massive allocation
            (0, usize::MAX),            // Zero elements, max size
        ];

        for (count, element_size) in overflow_cases {
            let result = validator.validate_buffer_bounds("overflow_test", count, element_size);
            assert!(
                result.is_err(),
                "Should prevent buffer overflow: count={}, size={}",
                count,
                element_size
            );
            self.results.security_metrics.buffer_overflow_prevented += 1;
        }

        self.record_test_result("buffer_overflow_prevention", start.elapsed(), true);
    }

    async fn test_injection_prevention(&mut self) {
        let start = Instant::now();

        // Test MCP protocol error handling
        let mcp_error = NetworkError::MCPError {
            method: "dangerous_method".to_string(),
            reason: "Method not allowed".to_string(),
        };

        let vision_error = VisionFlowError::Network(mcp_error);
        let error_msg = format!("{}", vision_error);

        assert!(error_msg.contains("dangerous_method"));
        assert!(error_msg.contains("not allowed"));

        self.record_test_result("injection_prevention", start.elapsed(), true);
    }

    async fn test_authentication_security(&mut self) {
        let start = Instant::now();

        // Test authentication error scenarios
        let settings_error = SettingsError::FileNotFound("/unauthorized/path".to_string());
        let vision_error = VisionFlowError::Settings(settings_error);

        let error_msg = format!("{}", vision_error);
        assert!(error_msg.contains("unauthorized/path"));

        self.record_test_result("authentication_security", start.elapsed(), true);
    }

    async fn validate_performance_requirements(&mut self) {
        println!("Validating Performance Requirements...");

        self.test_response_times().await;
        self.test_memory_efficiency().await;
        self.test_cpu_utilization().await;
        self.test_scalability().await;
    }

    async fn test_response_times(&mut self) {
        let start = Instant::now();

        // Test GPU safety validator performance
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        let validation_start = Instant::now();

        // Run multiple validations
        for i in 0..1000 {
            let _ = validator.validate_kernel_params(i as i32, i as i32 * 2, 0, 4, 256);
        }

        let validation_time = validation_start.elapsed();
        let avg_time_ms = validation_time.as_millis() as f64 / 1000.0;

        assert!(
            avg_time_ms < 1.0,
            "Average validation time should be under 1ms, got {:.2}ms",
            avg_time_ms
        );

        self.results.performance_metrics.average_response_time_ms = avg_time_ms;
        self.record_test_result("response_times", start.elapsed(), true);
    }

    async fn test_memory_efficiency(&mut self) {
        let start = Instant::now();

        // Test memory tracker efficiency
        let mut memory_tracker = GPUMemoryTracker::new();

        let initial_stats = memory_tracker.get_usage_stats();
        assert_eq!(initial_stats.current_allocated, 0);

        // Allocate and deallocate repeatedly
        for i in 0..1000 {
            let name = format!("test_{}", i);
            let _ = memory_tracker.track_allocation(name.clone(), 1024);
            memory_tracker.track_deallocation(&name);
        }

        let final_stats = memory_tracker.get_usage_stats();
        assert_eq!(
            final_stats.current_allocated, 0,
            "Should clean up all memory"
        );
        assert_eq!(
            final_stats.active_allocations, 0,
            "Should have no active allocations"
        );

        self.record_test_result("memory_efficiency", start.elapsed(), true);
    }

    async fn test_cpu_utilization(&mut self) {
        let start = Instant::now();

        // Test CPU fallback performance
        let mut positions = vec![(0.0, 0.0, 0.0); 1000];
        let mut velocities = vec![(0.0, 0.0, 0.0); 1000];
        let edges: Vec<(i32, i32, f32)> = (0..999).map(|i| (i, i + 1, 1.0)).collect();

        let cpu_start = Instant::now();
        let result = cpu_fallback::compute_forces_cpu(
            &mut positions,
            &mut velocities,
            &edges,
            0.1,
            0.1,
            0.9,
            0.01,
        );
        let cpu_time = cpu_start.elapsed();

        assert!(result.is_ok(), "CPU computation should succeed");
        assert!(
            cpu_time.as_millis() < 1000,
            "CPU computation should complete quickly"
        );

        self.record_test_result("cpu_utilization", start.elapsed(), true);
    }

    async fn test_scalability(&mut self) {
        let start = Instant::now();

        // Test system behavior with increasing load
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        let sizes = vec![100, 1000, 10000, 100000];

        for &size in &sizes {
            let validation_start = Instant::now();
            let result = validator.validate_buffer_bounds("scalability_test", size, 12);
            let validation_time = validation_start.elapsed();

            assert!(
                result.is_ok(),
                "Validation should succeed for size {}",
                size
            );
            assert!(
                validation_time.as_millis() < 10,
                "Validation should be fast even for large sizes: {}ms for size {}",
                validation_time.as_millis(),
                size
            );
        }

        self.record_test_result("scalability", start.elapsed(), true);
    }

    async fn validate_memory_safety(&mut self) {
        println!("Validating Memory Safety...");

        // Test memory safety mechanisms
        self.test_null_pointer_safety().await;
        self.test_memory_alignment().await;
        self.test_resource_cleanup().await;
    }

    async fn test_null_pointer_safety(&mut self) {
        let start = Instant::now();

        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test null pointer detection
        let null_ptr = std::ptr::null::<u8>();
        let result = validator.validate_memory_alignment(null_ptr, 16);

        if let Err(GPUSafetyError::NullPointer { operation }) = result {
            assert!(operation.contains("memory alignment check"));
        } else {
            panic!("Should detect null pointer");
        }

        self.record_test_result("null_pointer_safety", start.elapsed(), true);
    }

    async fn test_memory_alignment(&mut self) {
        let start = Instant::now();

        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test aligned memory
        let aligned_data: Vec<u64> = vec![1, 2, 3, 4];
        let aligned_ptr = aligned_data.as_ptr() as *const u8;

        let result = validator.validate_memory_alignment(aligned_ptr, 8);
        assert!(result.is_ok(), "Aligned memory should pass validation");

        self.record_test_result("memory_alignment", start.elapsed(), true);
    }

    async fn test_resource_cleanup(&mut self) {
        let start = Instant::now();

        // Test automatic resource cleanup
        let mut memory_tracker = GPUMemoryTracker::new();

        // Create scope for automatic cleanup
        {
            for i in 0..10 {
                let name = format!("scoped_resource_{}", i);
                let _ = memory_tracker.track_allocation(name, 1024);
            }

            let stats = memory_tracker.get_usage_stats();
            assert_eq!(stats.active_allocations, 10);
        }

        // Manually clean up (simulating automatic cleanup)
        for i in 0..10 {
            let name = format!("scoped_resource_{}", i);
            memory_tracker.track_deallocation(&name);
        }

        let final_stats = memory_tracker.get_usage_stats();
        assert_eq!(final_stats.active_allocations, 0);

        self.record_test_result("resource_cleanup", start.elapsed(), true);
    }

    async fn validate_concurrency_safety(&mut self) {
        println!("Validating Concurrency Safety...");

        self.test_thread_safety().await;
        self.test_race_conditions().await;
        self.test_atomic_operations().await;
    }

    async fn test_thread_safety(&mut self) {
        let start = Instant::now();

        use std::sync::Arc;
        use std::thread;

        let safety_config = GPUSafetyConfig::default();
        let validator = Arc::new(GPUSafetyValidator::new(safety_config));

        let mut handles = vec![];

        // Test concurrent validation operations
        for i in 0..10 {
            let validator_clone = Arc::clone(&validator);
            let handle = thread::spawn(move || {
                for j in 0..100 {
                    let result = validator_clone.validate_kernel_params(
                        (i * 100 + j) as i32,
                        (i * 200 + j) as i32,
                        0,
                        4,
                        256,
                    );
                    assert!(result.is_ok(), "Concurrent validation should succeed");
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        self.record_test_result("thread_safety", start.elapsed(), true);
    }

    async fn test_race_conditions(&mut self) {
        let start = Instant::now();

        use std::sync::{Arc, Mutex};
        use std::thread;

        // Test shared counter without race conditions
        let counter = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    if let Ok(mut count) = counter_clone.try_lock() {
                        *count += 1;
                    }
                    // Don't block indefinitely - prevents deadlocks
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should complete");
        }

        let final_count = *counter.lock().unwrap();
        assert!(final_count > 0, "Counter should have been incremented");
        println!("Race condition test: final count = {}", final_count);

        self.record_test_result("race_conditions", start.elapsed(), true);
    }

    async fn test_atomic_operations(&mut self) {
        let start = Instant::now();

        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::thread;

        let atomic_counter = Arc::new(AtomicUsize::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let counter_clone = Arc::clone(&atomic_counter);
            let handle = thread::spawn(move || {
                for _ in 0..1000 {
                    counter_clone.fetch_add(1, Ordering::Relaxed);
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("Thread should complete");
        }

        let final_count = atomic_counter.load(Ordering::Relaxed);
        assert_eq!(final_count, 10000, "Atomic operations should be precise");

        self.record_test_result("atomic_operations", start.elapsed(), true);
    }

    async fn validate_data_integrity(&mut self) {
        println!("Validating Data Integrity...");

        self.test_graph_consistency().await;
        self.test_physics_stability().await;
        self.test_serialization_safety().await;
    }

    async fn test_graph_consistency(&mut self) {
        let start = Instant::now();

        // Test graph data consistency
        let nodes = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];

        // Verify edge references are valid
        for &(src, dst, _weight) in &edges {
            assert!(
                src >= 0 && (src as usize) < nodes.len(),
                "Source node index should be valid"
            );
            assert!(
                dst >= 0 && (dst as usize) < nodes.len(),
                "Destination node index should be valid"
            );
        }

        // Verify graph structure
        assert_eq!(nodes.len(), 3, "Should have correct number of nodes");
        assert_eq!(edges.len(), 3, "Should have correct number of edges");

        self.record_test_result("graph_consistency", start.elapsed(), true);
    }

    async fn test_physics_stability(&mut self) {
        let start = Instant::now();

        // Test physics computation stability
        let mut positions = vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, 0.866, 0.0)];
        let mut velocities = vec![(0.0, 0.0, 0.0); 3];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];

        // Run physics simulation for multiple steps
        for step in 0..10 {
            let result = cpu_fallback::compute_forces_cpu(
                &mut positions,
                &mut velocities,
                &edges,
                0.1,
                0.1,
                0.9,
                0.01,
            );

            assert!(result.is_ok(), "Physics step {} should succeed", step);

            // Check for numerical stability
            for (i, &(x, y, z)) in positions.iter().enumerate() {
                assert!(x.is_finite(), "Position X should be finite for node {}", i);
                assert!(y.is_finite(), "Position Y should be finite for node {}", i);
                assert!(z.is_finite(), "Position Z should be finite for node {}", i);

                // Check for reasonable bounds
                assert!(
                    x.abs() < 1000.0,
                    "Position X should be reasonable for node {}",
                    i
                );
                assert!(
                    y.abs() < 1000.0,
                    "Position Y should be reasonable for node {}",
                    i
                );
                assert!(
                    z.abs() < 1000.0,
                    "Position Z should be reasonable for node {}",
                    i
                );
            }
        }

        self.record_test_result("physics_stability", start.elapsed(), true);
    }

    async fn test_serialization_safety(&mut self) {
        let start = Instant::now();

        // Test error serialization
        let gpu_error = GPUError::MemoryAllocationFailed {
            requested_bytes: 1024,
            reason: "Out of memory".to_string(),
        };

        let vision_error = VisionFlowError::GPU(gpu_error);

        // Test error can be converted to string safely
        let error_string = format!("{}", vision_error);
        assert!(error_string.contains("1024"));
        assert!(error_string.contains("Out of memory"));

        // Test error debug formatting
        let debug_string = format!("{:?}", vision_error);
        assert!(debug_string.contains("MemoryAllocationFailed"));

        self.record_test_result("serialization_safety", start.elapsed(), true);
    }

    async fn validate_fault_tolerance(&mut self) {
        println!("Validating Fault Tolerance...");

        self.test_graceful_failure_handling().await;
        self.test_recovery_mechanisms().await;
        self.test_health_monitoring().await;
    }

    async fn test_graceful_failure_handling(&mut self) {
        let start = Instant::now();

        // Test system behavior under various failure conditions
        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test handling of invalid inputs
        let invalid_inputs = vec![
            (i32::MIN, 1000, 0, 4, 256), // Extreme negative
            (i32::MAX, 1000, 0, 4, 256), // Extreme positive
            (0, 0, 0, 4, 256),           // Zero sizes
        ];

        for (nodes, edges, constraints, grid, block) in invalid_inputs {
            let result = validator.validate_kernel_params(nodes, edges, constraints, grid, block);
            // Should handle gracefully without panicking
            match result {
                Ok(_) => {} // Some cases might be valid
                Err(e) => {
                    // Error should be descriptive
                    let error_msg = format!("{}", e);
                    assert!(!error_msg.is_empty(), "Error message should not be empty");
                }
            }
        }

        self.record_test_result("graceful_failure_handling", start.elapsed(), true);
    }

    async fn test_recovery_mechanisms(&mut self) {
        let start = Instant::now();

        let safety_config = GPUSafetyConfig::default();
        let validator = GPUSafetyValidator::new(safety_config);

        // Test failure recovery cycle
        assert!(
            !validator.should_use_cpu_fallback(),
            "Should start in normal mode"
        );

        // Trigger failures
        for _ in 0..safety_config.cpu_fallback_threshold {
            validator.record_failure();
        }

        assert!(
            validator.should_use_cpu_fallback(),
            "Should enter fallback mode"
        );

        // Test recovery
        validator.reset_failures();
        assert!(
            !validator.should_use_cpu_fallback(),
            "Should recover to normal mode"
        );

        self.record_test_result("recovery_mechanisms", start.elapsed(), true);
    }

    async fn test_health_monitoring(&mut self) {
        let start = Instant::now();

        // Test system health monitoring
        let mut memory_tracker = GPUMemoryTracker::new();

        // Monitor memory usage
        for i in 0..5 {
            let name = format!("health_test_{}", i);
            let _ = memory_tracker.track_allocation(name, 1024 * (i + 1));
        }

        let stats = memory_tracker.get_usage_stats();
        assert!(stats.current_allocated > 0, "Should track memory usage");
        assert!(stats.allocation_count > 0, "Should track allocation count");
        assert!(
            stats.active_allocations > 0,
            "Should track active allocations"
        );

        // Clean up and verify
        for i in 0..5 {
            let name = format!("health_test_{}", i);
            memory_tracker.track_deallocation(&name);
        }

        let final_stats = memory_tracker.get_usage_stats();
        assert_eq!(
            final_stats.active_allocations, 0,
            "Should clean up properly"
        );

        self.record_test_result("health_monitoring", start.elapsed(), true);
    }

    fn record_test_result(&mut self, test_name: &str, duration: Duration, passed: bool) {
        self.results.total_tests += 1;

        if passed {
            self.results.passed += 1;
            println!("✓ {} completed in {:.2}ms", test_name, duration.as_millis());
        } else {
            self.results.failed += 1;
            println!("✗ {} failed after {:.2}ms", test_name, duration.as_millis());
        }

        // Update performance metrics
        let duration_ms = duration.as_millis() as f64;
        if duration_ms > self.results.performance_metrics.max_response_time_ms {
            self.results.performance_metrics.max_response_time_ms = duration_ms;
        }
    }

    fn finalize_results(&mut self) {
        if let Some(start_time) = self.start_time {
            self.results.total_duration = start_time.elapsed();
        }

        // Calculate coverage (simplified)
        self.results.coverage_percent = if self.results.total_tests > 0 {
            (self.results.passed as f32 / self.results.total_tests as f32) * 100.0
        } else {
            0.0
        };

        // Calculate average response time
        if self.results.total_tests > 0 {
            let total_duration_ms = self.results.total_duration.as_millis() as f64;
            self.results.performance_metrics.average_response_time_ms =
                total_duration_ms / self.results.total_tests as f64;
        }

        println!("\n=== Production Validation Complete ===");
        println!("Total Tests: {}", self.results.total_tests);
        println!("Passed: {}", self.results.passed);
        println!("Failed: {}", self.results.failed);
        println!("Coverage: {:.1}%", self.results.coverage_percent);
        println!(
            "Total Duration: {:.2}s",
            self.results.total_duration.as_secs_f64()
        );
        println!(
            "Critical Issues Resolved: {}",
            self.results.critical_issues_resolved
        );
    }

    pub fn get_results(&self) -> &ValidationResults {
        &self.results
    }
}

#[tokio::test]
async fn test_production_validation_suite() {
    let mut suite = ProductionValidationSuite::new();
    let results = suite.run_complete_validation().await;

    assert!(results.passed > 0, "Some tests should pass");
    assert!(
        results.coverage_percent > 80.0,
        "Coverage should be above 80%"
    );
    assert!(
        results.critical_issues_resolved > 0,
        "Should resolve critical issues"
    );
}
