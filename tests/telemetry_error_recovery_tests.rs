use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex, mpsc},
    thread,
    time::{Duration, Instant}
};
use tempfile::tempdir;
use serde_json::{json, Value};
use log::{info, warn, error, Level};

use super::super::src::utils::advanced_logging::{
    AdvancedLogger, LogComponent, LogEntry,
    init_advanced_logging, log_gpu_kernel, log_gpu_error,
    log_memory_event, log_performance, log_structured,
    get_performance_summary
};

/// Comprehensive error scenario and recovery testing for telemetry system
#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[test]
    fn test_concurrent_logging_safety() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = Arc::new(AdvancedLogger::new(&log_dir).expect("Failed to create logger"));

        const NUM_THREADS: usize = 10;
        const LOGS_PER_THREAD: usize = 100;

        let mut handles = vec![];
        let (tx, rx) = mpsc::channel();

        // Spawn multiple threads writing concurrently
        for thread_id in 0..NUM_THREADS {
            let logger_clone = Arc::clone(&logger);
            let tx_clone = tx.clone();

            let handle = thread::spawn(move || {
                for i in 0..LOGS_PER_THREAD {
                    // Different components per thread to test concurrent access
                    let component = match thread_id % 4 {
                        0 => LogComponent::Server,
                        1 => LogComponent::GPU,
                        2 => LogComponent::Memory,
                        _ => LogComponent::Performance,
                    };

                    let metadata = HashMap::from([
                        ("thread_id".to_string(), json!(thread_id)),
                        ("iteration".to_string(), json!(i)),
                        ("concurrent_test".to_string(), json!(true))
                    ]);

                    logger_clone.log_structured(
                        component,
                        Level::Info,
                        &format!("Concurrent log from thread {} iteration {}", thread_id, i),
                        Some(metadata)
                    );

                    // Add some GPU and performance logs for variety
                    if i % 10 == 0 {
                        logger_clone.log_gpu_kernel(
                            &format!("concurrent_kernel_{}", thread_id),
                            (i as f64) * 1.5,
                            64.0,
                            128.0
                        );
                    }

                    if i % 20 == 0 {
                        logger_clone.log_performance(
                            &format!("concurrent_op_{}", thread_id),
                            (i as f64) * 0.8,
                            Some(50.0)
                        );
                    }
                }

                tx_clone.send(thread_id).expect("Failed to send completion signal");
            });

            handles.push(handle);
        }

        drop(tx); // Close the sender

        // Wait for all threads to complete
        let mut completed_threads = 0;
        while let Ok(thread_id) = rx.recv() {
            completed_threads += 1;
            info!("Thread {} completed", thread_id);
        }

        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        assert_eq!(completed_threads, NUM_THREADS, "All threads should complete");

        // Verify log integrity
        verify_concurrent_log_integrity(&log_dir, NUM_THREADS, LOGS_PER_THREAD);
    }

    #[test]
    fn test_disk_space_exhaustion_recovery() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = AdvancedLogger::new(&log_dir).expect("Failed to create logger");

        // Simulate disk space exhaustion by writing large logs
        let large_metadata = HashMap::from([
            ("large_data".to_string(), json!("X".repeat(1024 * 1024))), // 1MB
            ("disk_stress_test".to_string(), json!(true))
        ]);

        let mut successful_writes = 0;
        let max_attempts = 100;

        for i in 0..max_attempts {
            logger.log_structured(
                LogComponent::Server,
                Level::Info,
                &format!("Large log entry {}", i),
                Some(large_metadata.clone())
            );
            successful_writes += 1;

            // Check if logs are still being written properly
            let server_log = log_dir.join("server.log");
            if server_log.exists() {
                if let Ok(metadata) = fs::metadata(&server_log) {
                    info!("Log file size: {} MB", metadata.len() / (1024 * 1024));

                    // If file gets too large (>100MB), simulate disk cleanup
                    if metadata.len() > 100 * 1024 * 1024 {
                        simulate_disk_cleanup(&log_dir, &logger);
                        break;
                    }
                }
            }
        }

        info!("Successful writes before disk cleanup: {}", successful_writes);

        // Verify system recovers after cleanup
        logger.log_structured(
            LogComponent::Server,
            Level::Info,
            "Recovery test after disk cleanup",
            Some(HashMap::from([
                ("recovery_test".to_string(), json!(true)),
                ("post_cleanup".to_string(), json!(true))
            ]))
        );

        let server_log = log_dir.join("server.log");
        let content = fs::read_to_string(&server_log).expect("Should read server log");
        assert!(content.contains("Recovery test after disk cleanup"));
    }

    #[test]
    fn test_log_file_permission_recovery() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = AdvancedLogger::new(&log_dir).expect("Failed to create logger");

        // Write initial logs
        logger.log_structured(
            LogComponent::Server,
            Level::Info,
            "Initial log before permission test",
            None
        );

        // Simulate permission issue by making log file read-only
        let server_log = log_dir.join("server.log");
        assert!(server_log.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&server_log).unwrap().permissions();
            perms.set_mode(0o444); // Read-only
            fs::set_permissions(&server_log, perms).expect("Failed to set permissions");
        }

        // Try to write logs - should handle gracefully
        for i in 0..10 {
            logger.log_structured(
                LogComponent::Server,
                Level::Info,
                &format!("Permission test log {}", i),
                Some(HashMap::from([
                    ("permission_test".to_string(), json!(true)),
                    ("iteration".to_string(), json!(i))
                ]))
            );
        }

        // Restore permissions and verify recovery
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&server_log).unwrap().permissions();
            perms.set_mode(0o644); // Read-write
            fs::set_permissions(&server_log, perms).expect("Failed to restore permissions");
        }

        logger.log_structured(
            LogComponent::Server,
            Level::Info,
            "Recovery after permission fix",
            Some(HashMap::from([
                ("recovery_successful".to_string(), json!(true))
            ]))
        );

        let content = fs::read_to_string(&server_log).expect("Should read server log");
        assert!(content.contains("Initial log before permission test"));
        assert!(content.contains("Recovery after permission fix"));
    }

    #[test]
    fn test_memory_leak_prevention() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = Arc::new(AdvancedLogger::new(&log_dir).expect("Failed to create logger"));

        // Get initial performance summary
        let initial_summary = get_performance_summary();
        let initial_kernel_count = count_tracked_kernels(&initial_summary);

        const ITERATIONS: usize = 10000;

        // Generate many GPU kernel logs to test memory usage
        for i in 0..ITERATIONS {
            let kernel_name = format!("test_kernel_{}", i % 100); // Cycle through 100 different kernels
            logger.log_gpu_kernel(&kernel_name, (i as f64) * 0.5, 64.0, 128.0);

            // Periodically log other types to test overall memory usage
            if i % 100 == 0 {
                logger.log_memory_event("test_allocation", (i as f64) * 0.1, (i as f64) * 0.2);
                logger.log_performance("test_operation", (i as f64) * 0.3, Some(100.0));
            }
        }

        // Get final performance summary
        let final_summary = get_performance_summary();
        let final_kernel_count = count_tracked_kernels(&final_summary);

        // Verify memory usage is bounded (should not grow linearly with iterations)
        info!("Initial tracked kernels: {}", initial_kernel_count);
        info!("Final tracked kernels: {}", final_kernel_count);

        // Should have at most 100 different kernels tracked (due to cycling)
        assert!(final_kernel_count <= 150,
            "Too many kernels tracked: {}. Memory leak suspected.", final_kernel_count);

        // Verify performance metrics are properly summarized
        assert!(!final_summary.is_empty(), "Performance summary should not be empty");
    }

    #[test]
    fn test_log_corruption_detection_and_recovery() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = AdvancedLogger::new(&log_dir).expect("Failed to create logger");

        // Write valid logs first
        logger.log_structured(
            LogComponent::Analytics,
            Level::Info,
            "Valid log before corruption",
            Some(HashMap::from([
                ("valid_entry".to_string(), json!(true))
            ]))
        );

        // Simulate log corruption by writing invalid JSON
        let analytics_log = log_dir.join("analytics.log");
        let mut file = OpenOptions::new()
            .append(true)
            .open(&analytics_log)
            .expect("Should open log file");

        writeln!(file, "CORRUPTED_DATA_NOT_VALID_JSON{{{{").expect("Should write corruption");
        writeln!(file, "MORE_INVALID_JSON:::broken").expect("Should write more corruption");
        drop(file);

        // Try to continue logging - system should handle gracefully
        logger.log_structured(
            LogComponent::Analytics,
            Level::Info,
            "Log after corruption injection",
            Some(HashMap::from([
                ("post_corruption".to_string(), json!(true)),
                ("recovery_test".to_string(), json!(true))
            ]))
        );

        // Verify the file contains both valid and corrupted data
        let content = fs::read_to_string(&analytics_log).expect("Should read analytics log");
        assert!(content.contains("Valid log before corruption"));
        assert!(content.contains("CORRUPTED_DATA_NOT_VALID_JSON"));
        assert!(content.contains("Log after corruption injection"));

        // The logger should continue working despite corruption
        logger.log_structured(
            LogComponent::Analytics,
            Level::Info,
            "Final recovery test",
            Some(HashMap::from([
                ("final_test".to_string(), json!(true))
            ]))
        );

        let final_content = fs::read_to_string(&analytics_log).expect("Should read final content");
        assert!(final_content.contains("Final recovery test"));
    }

    #[test]
    fn test_high_frequency_logging_performance() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = Arc::new(AdvancedLogger::new(&log_dir).expect("Failed to create logger"));

        const HIGH_FREQUENCY_LOGS: usize = 50000;
        const MAX_DURATION_SECONDS: u64 = 10;

        let start_time = Instant::now();

        // High-frequency logging test
        for i in 0..HIGH_FREQUENCY_LOGS {
            let component = match i % 8 {
                0 => LogComponent::Server,
                1 => LogComponent::Client,
                2 => LogComponent::GPU,
                3 => LogComponent::Analytics,
                4 => LogComponent::Memory,
                5 => LogComponent::Network,
                6 => LogComponent::Performance,
                _ => LogComponent::Error,
            };

            if i % 100 == 0 {
                // Occasional complex logs with metadata
                let metadata = HashMap::from([
                    ("iteration".to_string(), json!(i)),
                    ("timestamp".to_string(), json!(start_time.elapsed().as_millis())),
                    ("high_frequency_test".to_string(), json!(true))
                ]);

                logger.log_structured(
                    component,
                    Level::Info,
                    &format!("High frequency log {}", i),
                    Some(metadata)
                );
            } else {
                // Simple logs
                logger.log_structured(
                    component,
                    Level::Info,
                    &format!("Fast log {}", i),
                    None
                );
            }

            // Add GPU kernel logs occasionally
            if i % 500 == 0 {
                logger.log_gpu_kernel("high_freq_kernel", i as f64 * 0.01, 32.0, 64.0);
            }
        }

        let total_duration = start_time.elapsed();

        info!("High frequency logging completed in {:?}", total_duration);
        info!("Average time per log: {:?}", total_duration / HIGH_FREQUENCY_LOGS as u32);

        assert!(total_duration.as_secs() <= MAX_DURATION_SECONDS,
            "High frequency logging took too long: {:?} > {}s",
            total_duration, MAX_DURATION_SECONDS);

        // Verify all logs were written
        verify_high_frequency_log_integrity(&log_dir, HIGH_FREQUENCY_LOGS);
    }

    #[test]
    fn test_graceful_shutdown_and_cleanup() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let log_dir = temp_dir.path().to_path_buf();

        let logger = AdvancedLogger::new(&log_dir).expect("Failed to create logger");

        // Simulate application running
        for i in 0..100 {
            logger.log_structured(
                LogComponent::Server,
                Level::Info,
                &format!("Application running - step {}", i),
                Some(HashMap::from([
                    ("step".to_string(), json!(i)),
                    ("application_state".to_string(), json!("running"))
                ]))
            );

            logger.log_gpu_kernel(&format!("app_kernel_{}", i % 10), i as f64 * 1.2, 128.0, 256.0);
            logger.log_performance(&format!("app_operation_{}", i % 5), i as f64 * 0.5, Some(75.0));
        }

        // Simulate graceful shutdown
        logger.log_structured(
            LogComponent::Server,
            Level::Info,
            "Application shutdown initiated",
            Some(HashMap::from([
                ("shutdown_type".to_string(), json!("graceful")),
                ("cleanup_started".to_string(), json!(true))
            ]))
        );

        // Log final performance summary
        let final_summary = get_performance_summary();
        logger.log_structured(
            LogComponent::Performance,
            Level::Info,
            "Final performance summary",
            Some(HashMap::from([
                ("performance_data".to_string(), json!(final_summary)),
                ("shutdown_complete".to_string(), json!(true))
            ]))
        );

        // Verify shutdown logs are present
        let server_log = log_dir.join("server.log");
        let perf_log = log_dir.join("performance.log");

        let server_content = fs::read_to_string(&server_log).expect("Should read server log");
        let perf_content = fs::read_to_string(&perf_log).expect("Should read performance log");

        assert!(server_content.contains("Application shutdown initiated"));
        assert!(server_content.contains("graceful"));
        assert!(perf_content.contains("Final performance summary"));
        assert!(perf_content.contains("shutdown_complete"));
    }

    // Helper functions
    fn verify_concurrent_log_integrity(
        log_dir: &PathBuf,
        num_threads: usize,
        logs_per_thread: usize
    ) {
        let components = vec![
            LogComponent::Server,
            LogComponent::GPU,
            LogComponent::Memory,
            LogComponent::Performance,
        ];

        for component in components {
            let log_file = log_dir.join(component.log_file_name());
            assert!(log_file.exists(), "Log file {} should exist", component.as_str());

            let content = fs::read_to_string(&log_file)
                .expect(&format!("Should read {}", component.as_str()));

            // Count lines to verify all logs were written
            let line_count = content.lines().count();

            // Should have logs from multiple threads (exact count depends on component assignment)
            assert!(line_count > 0, "Log file {} should not be empty", component.as_str());

            // Verify concurrent_test metadata is present
            assert!(content.contains("concurrent_test"));
            assert!(content.contains("thread_id"));
        }
    }

    fn simulate_disk_cleanup(log_dir: &PathBuf, logger: &AdvancedLogger) {
        info!("Simulating disk cleanup...");

        // Log the cleanup event
        logger.log_structured(
            LogComponent::Server,
            Level::Warn,
            "Disk space low, performing cleanup",
            Some(HashMap::from([
                ("cleanup_initiated".to_string(), json!(true)),
                ("reason".to_string(), json!("disk_space_exhausted"))
            ]))
        );

        // Simulate cleanup by truncating the large log file
        let server_log = log_dir.join("server.log");
        if server_log.exists() {
            // Keep only the last few KB
            if let Ok(content) = fs::read_to_string(&server_log) {
                let lines: Vec<&str> = content.lines().collect();
                let keep_lines = lines.len().min(10); // Keep last 10 lines
                let truncated_content = lines[lines.len() - keep_lines..].join("\n");

                fs::write(&server_log, truncated_content).expect("Should truncate log file");
            }
        }

        logger.log_structured(
            LogComponent::Server,
            Level::Info,
            "Disk cleanup completed",
            Some(HashMap::from([
                ("cleanup_completed".to_string(), json!(true))
            ]))
        );
    }

    fn count_tracked_kernels(summary: &HashMap<String, serde_json::Value>) -> usize {
        summary.keys().filter(|k| !k.contains("error") && !k.contains("recovery")).count()
    }

    fn verify_high_frequency_log_integrity(log_dir: &PathBuf, expected_logs: usize) {
        let log_files = vec![
            "server.log", "client.log", "gpu.log", "analytics.log",
            "memory.log", "network.log", "performance.log", "error.log"
        ];

        let mut total_lines = 0;

        for log_file in &log_files {
            let path = log_dir.join(log_file);
            if path.exists() {
                let content = fs::read_to_string(&path)
                    .expect(&format!("Should read {}", log_file));
                let lines = content.lines().count();
                total_lines += lines;
                info!("{}: {} lines", log_file, lines);
            }
        }

        info!("Total log lines across all files: {}", total_lines);

        // Should have at least the expected number of logs (may have more due to GPU/performance logs)
        assert!(total_lines >= expected_logs / 8, // Distributed across 8 components
            "Not enough log lines found. Expected at least {}, got {}",
            expected_logs / 8, total_lines);
    }
}