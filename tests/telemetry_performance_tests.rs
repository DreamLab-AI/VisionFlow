use log::Level;
use serde_json::json;
use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};
use tempfile::tempdir;

use super::super::src::utils::advanced_logging::{
    get_performance_summary, log_gpu_error, log_gpu_kernel, log_memory_event, log_performance,
    log_structured, AdvancedLogger, LogComponent,
};

/// Performance validation tests for telemetry system
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_logging_latency_benchmarks() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = AdvancedLogger::new(&log_dir).expect("Failed to create logger");

        const BENCHMARK_ITERATIONS: usize = 10000;

        // Benchmark simple structured logging
        let start_time = Instant::now();
        for i in 0..BENCHMARK_ITERATIONS {
            logger.log_structured(
                LogComponent::Performance,
                Level::Info,
                &format!("Benchmark log entry {}", i),
                None,
            );
        }
        let simple_log_duration = start_time.elapsed();

        // Benchmark complex structured logging with metadata
        let complex_metadata = HashMap::from([
            ("benchmark_id".to_string(), json!("complex_logging")),
            ("iteration".to_string(), json!(0)),
            ("timestamp".to_string(), json!(0u64)),
            ("data_size_mb".to_string(), json!(1.5)),
            ("processing_time_ms".to_string(), json!(42.7)),
            ("status".to_string(), json!("processing")),
            (
                "tags".to_string(),
                json!(["performance", "benchmark", "complex"]),
            ),
            (
                "nested_data".to_string(),
                json!({
                    "sub_field_1": "value1",
                    "sub_field_2": 123,
                    "sub_field_3": true,
                    "sub_array": [1, 2, 3, 4, 5]
                }),
            ),
        ]);

        let start_time = Instant::now();
        for i in 0..BENCHMARK_ITERATIONS {
            let mut metadata = complex_metadata.clone();
            metadata.insert("iteration".to_string(), json!(i));
            metadata.insert(
                "timestamp".to_string(),
                json!(start_time.elapsed().as_millis()),
            );

            logger.log_structured(
                LogComponent::Performance,
                Level::Info,
                &format!("Complex benchmark log entry {}", i),
                Some(metadata),
            );
        }
        let complex_log_duration = start_time.elapsed();

        // Benchmark GPU kernel logging
        let start_time = Instant::now();
        for i in 0..BENCHMARK_ITERATIONS {
            logger.log_gpu_kernel(
                &format!("benchmark_kernel_{}", i % 10),
                (i as f64) * 1.5,
                128.0 + (i as f64) * 0.1,
                256.0 + (i as f64) * 0.2,
            );
        }
        let gpu_log_duration = start_time.elapsed();

        // Calculate and verify performance metrics
        let simple_latency_ns = simple_log_duration.as_nanos() / BENCHMARK_ITERATIONS as u128;
        let complex_latency_ns = complex_log_duration.as_nanos() / BENCHMARK_ITERATIONS as u128;
        let gpu_latency_ns = gpu_log_duration.as_nanos() / BENCHMARK_ITERATIONS as u128;

        println!("Logging Performance Benchmarks:");
        println!("Simple logging: {} ns per log", simple_latency_ns);
        println!("Complex logging: {} ns per log", complex_latency_ns);
        println!("GPU kernel logging: {} ns per log", gpu_latency_ns);

        // Performance assertions (adjust thresholds based on requirements)
        const MAX_SIMPLE_LATENCY_NS: u128 = 50_000; // 50 microseconds
        const MAX_COMPLEX_LATENCY_NS: u128 = 100_000; // 100 microseconds
        const MAX_GPU_LATENCY_NS: u128 = 75_000; // 75 microseconds

        assert!(
            simple_latency_ns < MAX_SIMPLE_LATENCY_NS,
            "Simple logging latency {} ns exceeds maximum {} ns",
            simple_latency_ns,
            MAX_SIMPLE_LATENCY_NS
        );

        assert!(
            complex_latency_ns < MAX_COMPLEX_LATENCY_NS,
            "Complex logging latency {} ns exceeds maximum {} ns",
            complex_latency_ns,
            MAX_COMPLEX_LATENCY_NS
        );

        assert!(
            gpu_latency_ns < MAX_GPU_LATENCY_NS,
            "GPU logging latency {} ns exceeds maximum {} ns",
            gpu_latency_ns,
            MAX_GPU_LATENCY_NS
        );
    }

    #[test]
    fn test_concurrent_logging_scalability() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = Arc::new(AdvancedLogger::new(&log_dir).expect("Failed to create logger"));

        // Test different levels of concurrency
        let thread_counts = vec![1, 2, 4, 8, 16];
        const LOGS_PER_THREAD: usize = 1000;

        for &num_threads in &thread_counts {
            let barrier = Arc::new(Barrier::new(num_threads + 1));
            let logger_clone = Arc::clone(&logger);

            let mut handles = vec![];
            let start_time = Instant::now();

            // Start all threads
            for thread_id in 0..num_threads {
                let logger_thread = Arc::clone(&logger_clone);
                let barrier_thread = Arc::clone(&barrier);

                let handle = thread::spawn(move || {
                    // Wait for all threads to be ready
                    barrier_thread.wait();

                    let thread_start = Instant::now();

                    for i in 0..LOGS_PER_THREAD {
                        let component = match i % 4 {
                            0 => LogComponent::Server,
                            1 => LogComponent::GPU,
                            2 => LogComponent::Memory,
                            _ => LogComponent::Performance,
                        };

                        logger_thread.log_structured(
                            component,
                            Level::Info,
                            &format!("Scalability test - thread {} log {}", thread_id, i),
                            Some(HashMap::from([
                                ("thread_id".to_string(), json!(thread_id)),
                                ("log_index".to_string(), json!(i)),
                                ("scalability_test".to_string(), json!(true)),
                            ])),
                        );

                        // Add some GPU logs for realistic workload
                        if i % 100 == 0 {
                            logger_thread.log_gpu_kernel(
                                &format!("scalability_kernel_{}", thread_id),
                                (i as f64) * 1.2,
                                64.0,
                                128.0,
                            );
                        }
                    }

                    thread_start.elapsed()
                });

                handles.push(handle);
            }

            // Start the race
            barrier.wait();
            let race_start = Instant::now();

            // Collect results
            let mut thread_durations = vec![];
            for handle in handles {
                let duration = handle.join().expect("Thread should complete successfully");
                thread_durations.push(duration);
            }

            let total_duration = race_start.elapsed();
            let max_thread_duration = thread_durations.iter().max().unwrap();
            let min_thread_duration = thread_durations.iter().min().unwrap();
            let avg_thread_duration = Duration::from_nanos(
                thread_durations
                    .iter()
                    .map(|d| d.as_nanos() as u64)
                    .sum::<u64>()
                    / num_threads as u64,
            );

            println!("Concurrency Test - {} threads:", num_threads);
            println!("  Total duration: {:?}", total_duration);
            println!("  Average thread duration: {:?}", avg_thread_duration);
            println!("  Min thread duration: {:?}", min_thread_duration);
            println!("  Max thread duration: {:?}", max_thread_duration);
            println!(
                "  Throughput: {:.2} logs/sec",
                (num_threads * LOGS_PER_THREAD) as f64 / total_duration.as_secs_f64()
            );

            // Verify reasonable scalability (shouldn't get dramatically worse with more threads)
            if num_threads > 1 {
                const MAX_ACCEPTABLE_DEGRADATION: f64 = 3.0; // 3x slowdown max
                let slowdown_factor =
                    max_thread_duration.as_secs_f64() / min_thread_duration.as_secs_f64();

                assert!(
                    slowdown_factor < MAX_ACCEPTABLE_DEGRADATION,
                    "Thread performance degradation too high: {:.2}x with {} threads",
                    slowdown_factor,
                    num_threads
                );
            }
        }
    }

    #[test]
    fn test_memory_usage_under_load() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = Arc::new(AdvancedLogger::new(&log_dir).expect("Failed to create logger"));

        // Get baseline memory usage
        let initial_summary = get_performance_summary();

        const MEMORY_TEST_ITERATIONS: usize = 50000;

        // Generate many different kernel names to test memory management
        for i in 0..MEMORY_TEST_ITERATIONS {
            // Use many different kernel names to test internal memory management
            let kernel_name = format!("memory_test_kernel_{}", i % 1000);
            logger.log_gpu_kernel(&kernel_name, (i as f64) * 0.5, 64.0, 128.0);

            // Add memory events
            if i % 50 == 0 {
                logger.log_memory_event("allocation", (i as f64) * 0.1, (i as f64) * 0.15);
            }

            // Add performance logs
            if i % 100 == 0 {
                logger.log_performance(
                    &format!("memory_op_{}", i % 20),
                    (i as f64) * 0.8,
                    Some(50.0),
                );
            }

            // Periodically check memory doesn't grow unbounded
            if i % 10000 == 0 && i > 0 {
                let current_summary = get_performance_summary();
                let tracked_items = current_summary.len();

                println!(
                    "After {} iterations: {} tracked performance items",
                    i, tracked_items
                );

                // Memory usage should be bounded (not grow linearly with iterations)
                assert!(
                    tracked_items < 2000,
                    "Too many items tracked in memory: {}. Possible memory leak at iteration {}",
                    tracked_items,
                    i
                );
            }
        }

        let final_summary = get_performance_summary();
        println!("Final performance summary items: {}", final_summary.len());

        // Verify memory usage is reasonable
        assert!(
            final_summary.len() < 1500,
            "Final tracked items {} indicates possible memory leak",
            final_summary.len()
        );
    }

    #[test]
    fn test_file_io_performance_under_pressure() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = AdvancedLogger::new(&log_dir).expect("Failed to create logger");

        const IO_PRESSURE_ITERATIONS: usize = 20000;
        const LARGE_DATA_SIZE: usize = 1024; // 1KB per log

        let large_data = "X".repeat(LARGE_DATA_SIZE);

        let start_time = Instant::now();

        // Apply I/O pressure with large log entries
        for i in 0..IO_PRESSURE_ITERATIONS {
            let metadata = HashMap::from([
                ("large_data".to_string(), json!(large_data)),
                ("iteration".to_string(), json!(i)),
                ("io_pressure_test".to_string(), json!(true)),
                ("data_size_kb".to_string(), json!(LARGE_DATA_SIZE / 1024)),
            ]);

            logger.log_structured(
                LogComponent::Performance,
                Level::Info,
                &format!("I/O pressure test iteration {}", i),
                Some(metadata),
            );

            // Mix in other log types
            if i % 50 == 0 {
                logger.log_gpu_kernel("io_pressure_kernel", (i as f64) * 0.3, 128.0, 256.0);
            }

            if i % 100 == 0 {
                logger.log_memory_event("io_pressure_memory", (i as f64) * 0.2, (i as f64) * 0.4);
            }
        }

        let total_duration = start_time.elapsed();
        let total_data_mb = (IO_PRESSURE_ITERATIONS * LARGE_DATA_SIZE) as f64 / (1024.0 * 1024.0);

        println!("I/O Pressure Test Results:");
        println!("Total data written: {:.2} MB", total_data_mb);
        println!("Total duration: {:?}", total_duration);
        println!(
            "Write throughput: {:.2} MB/s",
            total_data_mb / total_duration.as_secs_f64()
        );

        // Verify reasonable I/O performance
        const MIN_THROUGHPUT_MBS: f64 = 5.0; // 5 MB/s minimum
        let throughput = total_data_mb / total_duration.as_secs_f64();

        assert!(
            throughput > MIN_THROUGHPUT_MBS,
            "I/O throughput too low: {:.2} MB/s < {} MB/s",
            throughput,
            MIN_THROUGHPUT_MBS
        );

        // Verify all data was written correctly
        let perf_log = log_dir.join("performance.log");
        assert!(perf_log.exists(), "Performance log should exist");

        let log_size = fs::metadata(&perf_log)
            .expect("Should get file metadata")
            .len() as f64
            / (1024.0 * 1024.0);

        println!("Final log file size: {:.2} MB", log_size);

        // Should have written significant amount of data
        assert!(
            log_size > total_data_mb * 0.8,
            "Log file size {:.2} MB is too small compared to expected {:.2} MB",
            log_size,
            total_data_mb
        );
    }

    #[test]
    fn test_log_rotation_performance_impact() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = AdvancedLogger::new(&log_dir).expect("Failed to create logger");

        const ROTATION_TEST_ITERATIONS: usize = 15000;
        const LARGE_LOG_SIZE: usize = 4096; // 4KB per log to trigger rotation

        let large_log_data = "R".repeat(LARGE_LOG_SIZE);
        let mut rotation_times = vec![];

        for i in 0..ROTATION_TEST_ITERATIONS {
            let iteration_start = Instant::now();

            logger.log_structured(
                LogComponent::Server,
                Level::Info,
                &format!("Rotation test iteration {}", i),
                Some(HashMap::from([
                    ("large_data".to_string(), json!(large_log_data)),
                    ("rotation_test".to_string(), json!(true)),
                    ("iteration".to_string(), json!(i)),
                ])),
            );

            let iteration_duration = iteration_start.elapsed();

            // Track operations that take unusually long (might indicate rotation)
            if iteration_duration.as_millis() > 10 {
                rotation_times.push((i, iteration_duration));
                println!(
                    "Potential rotation at iteration {}: {:?}",
                    i, iteration_duration
                );
            }

            // Check if rotation occurred
            if i % 1000 == 0 {
                let archived_dir = log_dir.join("archived");
                if archived_dir.exists() {
                    let archived_count = fs::read_dir(&archived_dir)
                        .map(|entries| entries.count())
                        .unwrap_or(0);

                    if archived_count > 0 {
                        println!("Found {} archived files at iteration {}", archived_count, i);
                    }
                }
            }
        }

        println!(
            "Detected {} potential rotation events",
            rotation_times.len()
        );

        // Verify that rotation doesn't cause excessive performance degradation
        if !rotation_times.is_empty() {
            let max_rotation_time = rotation_times
                .iter()
                .map(|(_, duration)| duration.as_millis())
                .max()
                .unwrap();

            const MAX_ACCEPTABLE_ROTATION_TIME_MS: u128 = 1000; // 1 second

            assert!(
                max_rotation_time < MAX_ACCEPTABLE_ROTATION_TIME_MS,
                "Log rotation took too long: {} ms > {} ms",
                max_rotation_time,
                MAX_ACCEPTABLE_ROTATION_TIME_MS
            );
        }

        // Verify that archived files were created
        let archived_dir = log_dir.join("archived");
        if archived_dir.exists() {
            let archived_files: Vec<_> = fs::read_dir(&archived_dir)
                .expect("Should read archived dir")
                .collect();

            println!("Final archived files count: {}", archived_files.len());
            assert!(
                !archived_files.is_empty(),
                "Should have created archived files"
            );
        }

        // Verify current log file exists and is reasonable size
        let server_log = log_dir.join("server.log");
        assert!(server_log.exists(), "Current server log should exist");

        let current_size_mb = fs::metadata(&server_log)
            .expect("Should get metadata")
            .len() as f64
            / (1024.0 * 1024.0);

        println!("Current log file size: {:.2} MB", current_size_mb);

        // Should be under rotation limit (50MB by default)
        assert!(
            current_size_mb < 50.0,
            "Current log file too large: {:.2} MB",
            current_size_mb
        );
    }

    #[test]
    fn test_performance_summary_generation_efficiency() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = Arc::new(AdvancedLogger::new(&log_dir).expect("Failed to create logger"));

        // Generate significant amount of performance data
        const KERNEL_TYPES: usize = 100;
        const ITERATIONS_PER_KERNEL: usize = 500;

        for kernel_type in 0..KERNEL_TYPES {
            for iteration in 0..ITERATIONS_PER_KERNEL {
                logger.log_gpu_kernel(
                    &format!("perf_summary_kernel_{}", kernel_type),
                    (iteration as f64) * 1.5 + (kernel_type as f64) * 10.0,
                    128.0 + (kernel_type as f64),
                    256.0 + (kernel_type as f64) * 2.0,
                );
            }
        }

        // Benchmark performance summary generation
        const SUMMARY_BENCHMARK_ITERATIONS: usize = 100;

        let start_time = Instant::now();
        for _ in 0..SUMMARY_BENCHMARK_ITERATIONS {
            let _summary = get_performance_summary();
        }
        let summary_generation_duration = start_time.elapsed();

        let avg_summary_time =
            summary_generation_duration.as_micros() / SUMMARY_BENCHMARK_ITERATIONS as u128;

        println!("Performance Summary Generation Benchmark:");
        println!(
            "Total time for {} summaries: {:?}",
            SUMMARY_BENCHMARK_ITERATIONS, summary_generation_duration
        );
        println!("Average time per summary: {} μs", avg_summary_time);

        // Verify performance summary generation is efficient
        const MAX_SUMMARY_TIME_US: u128 = 10000; // 10 milliseconds

        assert!(
            avg_summary_time < MAX_SUMMARY_TIME_US,
            "Performance summary generation too slow: {} μs > {} μs",
            avg_summary_time,
            MAX_SUMMARY_TIME_US
        );

        // Verify the summary contains expected data
        let final_summary = get_performance_summary();
        assert!(
            !final_summary.is_empty(),
            "Performance summary should not be empty"
        );
        assert_eq!(
            final_summary.len(),
            KERNEL_TYPES + 2, // +2 for gpu_errors and recovery_attempts
            "Summary should contain all kernel types"
        );

        // Verify summary data quality
        for (kernel_name, data) in &final_summary {
            if kernel_name.starts_with("perf_summary_kernel_") {
                let sample_count = data
                    .get("sample_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                assert_eq!(
                    sample_count, ITERATIONS_PER_KERNEL as u64,
                    "Kernel {} should have {} samples",
                    kernel_name, ITERATIONS_PER_KERNEL
                );
            }
        }
    }
}
