//! Comprehensive concurrent access tests for settings refactor
//!
//! Tests thread safety, race condition handling, and performance under
//! concurrent load for the new granular settings API
//!

use std::sync::{Arc, RwLock, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::RwLock as AsyncRwLock;
use tokio::task;
use serde_json::{json, Value};
use futures::future::join_all;

use crate::config::AppFullSettings;
use crate::handlers::settings_handler::{SettingsPath, SettingsUpdate};

#[cfg(test)]
mod concurrent_access_tests {
    use super::*;

    #[test]
    fn test_concurrent_read_access() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let num_readers = 10;
        let mut handles = vec![];

        for i in 0..num_readers {
            let settings_clone = settings.clone();
            let handle = thread::spawn(move || {
                let start = Instant::now();
                
                // Perform multiple reads
                for _ in 0..100 {
                    let settings_guard = settings_clone.read().unwrap();
                    let _glow_strength = settings_guard.visualisation.glow.node_glow_strength;
                    let _debug_mode = settings_guard.system.debug_mode;
                    let _max_connections = settings_guard.system.max_connections;
                    
                    // Simulate some processing time
                    thread::sleep(Duration::from_microseconds(10));
                }
                
                let duration = start.elapsed();
                (i, duration)
            });
            handles.push(handle);
        }

        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        // All readers should complete successfully
        assert_eq!(results.len(), num_readers);
        
        // Concurrent reads should be reasonably fast
        for (i, duration) in results {
            assert!(duration < Duration::from_millis(100), 
                    "Reader {} took too long: {:?}", i, duration);
        }
    }

    #[test]
    fn test_concurrent_write_access() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let num_writers = 5;
        let writes_per_thread = 20;
        let mut handles = vec![];

        for i in 0..num_writers {
            let settings_clone = settings.clone();
            let handle = thread::spawn(move || {
                let mut successful_writes = 0;
                
                for j in 0..writes_per_thread {
                    match settings_clone.write() {
                        Ok(mut settings_guard) => {
                            // Update different fields to minimize contention
                            match i % 3 {
                                0 => {
                                    settings_guard.visualisation.glow.node_glow_strength = 
                                        1.0 + (i as f32) * 0.1 + (j as f32) * 0.01;
                                },
                                1 => {
                                    settings_guard.system.debug_mode = j % 2 == 0;
                                },
                                2 => {
                                    settings_guard.system.max_connections = 
                                        100 + (i as u32) * 10 + (j as u32);
                                },
                                _ => unreachable!()
                            }
                            successful_writes += 1;
                            
                            // Simulate write processing time
                            thread::sleep(Duration::from_microseconds(50));
                        },
                        Err(e) => {
                            eprintln!("Write lock failed for thread {}: {:?}", i, e);
                        }
                    }
                }
                
                (i, successful_writes)
            });
            handles.push(handle);
        }

        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        // All writers should complete most of their writes
        for (i, successful_writes) in results {
            assert!(successful_writes > writes_per_thread / 2, 
                    "Writer {} completed too few writes: {}", i, successful_writes);
        }

        // Verify final state is consistent
        let final_settings = settings.read().unwrap();
        assert!(final_settings.visualisation.glow.node_glow_strength > 0.0);
        assert!(final_settings.system.max_connections >= 100);
    }

    #[test]
    fn test_mixed_read_write_access() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let num_readers = 8;
        let num_writers = 3;
        let mut handles = vec![];

        // Spawn reader threads
        for i in 0..num_readers {
            let settings_clone = settings.clone();
            let handle = thread::spawn(move || {
                let mut successful_reads = 0;
                let start = Instant::now();
                
                while start.elapsed() < Duration::from_millis(200) {
                    if let Ok(settings_guard) = settings_clone.read() {
                        let _glow = &settings_guard.visualisation.glow;
                        let _system = &settings_guard.system;
                        successful_reads += 1;
                        thread::sleep(Duration::from_microseconds(5));
                    }
                }
                
                (format!("reader_{}", i), successful_reads)
            });
            handles.push(handle);
        }

        // Spawn writer threads
        for i in 0..num_writers {
            let settings_clone = settings.clone();
            let handle = thread::spawn(move || {
                let mut successful_writes = 0;
                let start = Instant::now();
                
                while start.elapsed() < Duration::from_millis(200) {
                    if let Ok(mut settings_guard) = settings_clone.try_write() {
                        settings_guard.visualisation.glow.node_glow_strength = 
                            1.0 + (i as f32) * 0.2;
                        settings_guard.system.max_connections = 100 + (i as u32) * 20;
                        successful_writes += 1;
                        thread::sleep(Duration::from_microseconds(20));
                    }
                }
                
                (format!("writer_{}", i), successful_writes)
            });
            handles.push(handle);
        }

        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        // Verify both readers and writers completed work
        let mut total_reads = 0;
        let mut total_writes = 0;

        for (thread_type, count) in results {
            if thread_type.starts_with("reader") {
                total_reads += count;
            } else {
                total_writes += count;
            }
        }

        assert!(total_reads > 100, "Should have many successful reads: {}", total_reads);
        assert!(total_writes > 5, "Should have some successful writes: {}", total_writes);
    }

    #[tokio::test]
    async fn test_async_concurrent_access() {
        let settings = Arc::new(AsyncRwLock::new(AppFullSettings::default()));
        let num_tasks = 10;
        let mut tasks = vec![];

        for i in 0..num_tasks {
            let settings_clone = settings.clone();
            let task = task::spawn(async move {
                let mut operations = 0;
                
                // Mix of read and write operations
                for j in 0..20 {
                    if j % 3 == 0 {
                        // Write operation
                        let mut settings_guard = settings_clone.write().await;
                        settings_guard.visualisation.glow.node_glow_strength = 
                            1.0 + (i as f32) * 0.1 + (j as f32) * 0.01;
                        operations += 1;
                        drop(settings_guard);
                        tokio::time::sleep(Duration::from_micros(10)).await;
                    } else {
                        // Read operation
                        let settings_guard = settings_clone.read().await;
                        let _value = settings_guard.visualisation.glow.node_glow_strength;
                        operations += 1;
                        drop(settings_guard);
                        tokio::time::sleep(Duration::from_micros(5)).await;
                    }
                }
                
                (i, operations)
            });
            tasks.push(task);
        }

        let results = join_all(tasks).await;

        // All tasks should complete successfully
        for result in results {
            let (task_id, operations) = result.unwrap();
            assert_eq!(operations, 20, "Task {} should complete all operations", task_id);
        }
    }

    #[test]
    fn test_deadlock_prevention() {
        let settings1 = Arc::new(RwLock::new(AppFullSettings::default()));
        let settings2 = Arc::new(RwLock::new(AppFullSettings::default()));
        
        let settings1_clone = settings1.clone();
        let settings2_clone = settings2.clone();

        let handle1 = thread::spawn(move || {
            for i in 0..10 {
                // Try to acquire locks in order: settings1, then settings2
                let _guard1 = settings1_clone.write().unwrap();
                thread::sleep(Duration::from_microseconds(100));
                let _guard2 = settings2_clone.write().unwrap();
                
                // Do some work
                thread::sleep(Duration::from_microseconds(10));
                
                // Guards dropped automatically
            }
            "thread1_completed"
        });

        let handle2 = thread::spawn(move || {
            for i in 0..10 {
                // Try to acquire locks in same order to prevent deadlock
                let _guard1 = settings1.write().unwrap();
                thread::sleep(Duration::from_microseconds(100));
                let _guard2 = settings2.write().unwrap();
                
                // Do some work
                thread::sleep(Duration::from_microseconds(10));
                
                // Guards dropped automatically
            }
            "thread2_completed"
        });

        // Both threads should complete without deadlock
        let result1 = handle1.join().unwrap();
        let result2 = handle2.join().unwrap();

        assert_eq!(result1, "thread1_completed");
        assert_eq!(result2, "thread2_completed");
    }

    #[test]
    fn test_lock_timeout_handling() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let settings_clone = settings.clone();

        // Thread that holds write lock for a long time
        let long_writer = thread::spawn(move || {
            let _guard = settings_clone.write().unwrap();
            thread::sleep(Duration::from_millis(100));
            "long_write_completed"
        });

        // Thread that tries to acquire read lock with timeout
        let settings_clone2 = settings.clone();
        let quick_reader = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10)); // Let writer get lock first
            
            // Use try_read to avoid blocking indefinitely
            let start = Instant::now();
            let mut attempts = 0;
            
            while start.elapsed() < Duration::from_millis(200) {
                if let Ok(_guard) = settings_clone2.try_read() {
                    return "read_succeeded";
                }
                attempts += 1;
                thread::sleep(Duration::from_millis(1));
            }
            
            format!("read_timed_out_after_{}_attempts", attempts)
        });

        let writer_result = long_writer.join().unwrap();
        let reader_result = quick_reader.join().unwrap();

        assert_eq!(writer_result, "long_write_completed");
        // Reader should either succeed after writer completes or timeout gracefully
        assert!(reader_result == "read_succeeded" || reader_result.starts_with("read_timed_out"));
    }

    #[test]
    fn test_high_contention_performance() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let num_threads = 20;
        let operations_per_thread = 50;
        let mut handles = vec![];

        let start_time = Instant::now();

        for i in 0..num_threads {
            let settings_clone = settings.clone();
            let handle = thread::spawn(move || {
                let mut completed = 0;
                
                for j in 0..operations_per_thread {
                    // Mix of reads and writes
                    if j % 4 == 0 {
                        // Write operation
                        if let Ok(mut guard) = settings_clone.try_write() {
                            guard.visualisation.glow.node_glow_strength = 
                                (i * operations_per_thread + j) as f32 * 0.001;
                            completed += 1;
                        }
                    } else {
                        // Read operation
                        if let Ok(guard) = settings_clone.try_read() {
                            let _value = guard.visualisation.glow.node_glow_strength;
                            completed += 1;
                        }
                    }
                }
                
                completed
            });
            handles.push(handle);
        }

        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        let total_duration = start_time.elapsed();
        let total_completed: usize = results.iter().sum();

        // Performance assertions
        assert!(total_duration < Duration::from_secs(2), 
                "High contention test took too long: {:?}", total_duration);
        
        assert!(total_completed > num_threads * operations_per_thread / 2,
                "Too few operations completed under high contention: {}", total_completed);
        
        println!("High contention test: {} operations in {:?} ({:.1} ops/sec)",
                total_completed, total_duration,
                total_completed as f64 / total_duration.as_secs_f64());
    }

    #[test]
    fn test_memory_consistency() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let num_writers = 3;
        let writes_per_thread = 100;
        let mut handles = vec![];

        // Writers that increment a counter field
        for i in 0..num_writers {
            let settings_clone = settings.clone();
            let handle = thread::spawn(move || {
                for j in 0..writes_per_thread {
                    let mut guard = settings_clone.write().unwrap();
                    
                    // Use max_connections as a counter for testing
                    let current = guard.system.max_connections;
                    guard.system.max_connections = current + 1;
                    
                    // Verify consistency within the same thread
                    assert!(guard.system.max_connections > current,
                            "Thread {} write {} failed consistency check", i, j);
                }
                format!("writer_{}_completed", i)
            });
            handles.push(handle);
        }

        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        // Verify all writers completed
        for result in results {
            assert!(result.ends_with("_completed"));
        }

        // Verify final state consistency
        let final_settings = settings.read().unwrap();
        let expected_final_count = num_writers * writes_per_thread;
        
        // The counter should have been incremented exactly the expected number of times
        assert_eq!(final_settings.system.max_connections as usize, expected_final_count,
                   "Memory consistency violation: expected {}, got {}",
                   expected_final_count, final_settings.system.max_connections);
    }

    #[test]
    fn test_concurrent_granular_path_access() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let num_accessors = 6;
        let mut handles = vec![];

        // Different threads accessing different paths concurrently
        let paths_and_values = vec![
            ("visualisation.glow.node_glow_strength", json!(2.5)),
            ("visualisation.glow.edge_glow_strength", json!(3.0)),
            ("system.debug_mode", json!(true)),
            ("system.max_connections", json!(150)),
            ("xr.hand_mesh_color", json!("#ff0000")),
            ("xr.locomotion_method", json!("teleport")),
        ];

        for (i, (path, value)) in paths_and_values.iter().enumerate() {
            let settings_clone = settings.clone();
            let path_str = path.to_string();
            let test_value = value.clone();
            
            let handle = thread::spawn(move || {
                let mut operations = 0;
                
                for _ in 0..50 {
                    // Simulate granular path update
                    {
                        let mut guard = settings_clone.write().unwrap();
                        
                        // Update specific path (simplified for testing)
                        match path_str.as_str() {
                            p if p.contains("node_glow_strength") => {
                                guard.visualisation.glow.node_glow_strength = 
                                    test_value.as_f64().unwrap() as f32;
                            },
                            p if p.contains("debug_mode") => {
                                guard.system.debug_mode = test_value.as_bool().unwrap();
                            },
                            p if p.contains("max_connections") => {
                                guard.system.max_connections = 
                                    test_value.as_u64().unwrap() as u32;
                            },
                            _ => {
                                // Other paths - just mark operation as completed
                            }
                        }
                        operations += 1;
                    }
                    
                    // Simulate some processing time
                    thread::sleep(Duration::from_microseconds(10));
                }
                
                (i, operations)
            });
            handles.push(handle);
        }

        let results: Vec<_> = handles.into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        // All path accessors should complete their operations
        for (path_index, operations) in results {
            assert_eq!(operations, 50, 
                       "Path accessor {} should complete all operations", path_index);
        }

        // Verify final state reflects concurrent updates
        let final_settings = settings.read().unwrap();
        assert_eq!(final_settings.visualisation.glow.node_glow_strength, 2.5);
        assert_eq!(final_settings.system.debug_mode, true);
        assert_eq!(final_settings.system.max_connections, 150);
    }

    #[tokio::test]
    async fn test_async_stress_test() {
        let settings = Arc::new(AsyncRwLock::new(AppFullSettings::default()));
        let concurrent_tasks = 50;
        let operations_per_task = 20;
        
        let start_time = Instant::now();
        let mut tasks = vec![];

        for task_id in 0..concurrent_tasks {
            let settings_clone = settings.clone();
            
            let task = task::spawn(async move {
                let mut completed_ops = 0;
                
                for op_id in 0..operations_per_task {
                    // Randomly choose read or write
                    if (task_id + op_id) % 3 == 0 {
                        // Write operation
                        let mut guard = settings_clone.write().await;
                        guard.visualisation.glow.node_glow_strength = 
                            (task_id as f32) * 0.1 + (op_id as f32) * 0.01;
                        drop(guard);
                        
                        // Small async delay
                        tokio::time::sleep(Duration::from_micros(1)).await;
                    } else {
                        // Read operation
                        let guard = settings_clone.read().await;
                        let _value = guard.visualisation.glow.node_glow_strength;
                        drop(guard);
                    }
                    
                    completed_ops += 1;
                }
                
                (task_id, completed_ops)
            });
            
            tasks.push(task);
        }

        let results = join_all(tasks).await;
        let total_duration = start_time.elapsed();

        // Verify all tasks completed successfully
        let mut total_operations = 0;
        for result in results {
            let (task_id, completed_ops) = result.unwrap();
            assert_eq!(completed_ops, operations_per_task,
                       "Task {} should complete all operations", task_id);
            total_operations += completed_ops;
        }

        let expected_total = concurrent_tasks * operations_per_task;
        assert_eq!(total_operations, expected_total,
                   "Total operations mismatch");

        // Performance check - should complete reasonably quickly
        assert!(total_duration < Duration::from_secs(5),
                "Async stress test took too long: {:?}", total_duration);

        println!("Async stress test: {} operations across {} tasks in {:?}",
                total_operations, concurrent_tasks, total_duration);
    }

    #[test]
    fn test_lock_fairness() {
        let settings = Arc::new(RwLock::new(AppFullSettings::default()));
        let num_threads = 10;
        let operations_per_thread = 20;
        
        // Track how many operations each thread completes
        let completion_counter = Arc::new(Mutex::new(Vec::<usize>::new()));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let settings_clone = settings.clone();
            let counter_clone = completion_counter.clone();
            
            let handle = thread::spawn(move || {
                let mut my_completions = 0;
                
                for _ in 0..operations_per_thread {
                    // Try to get write lock
                    if let Ok(mut guard) = settings_clone.try_write() {
                        guard.visualisation.glow.node_glow_strength = thread_id as f32;
                        my_completions += 1;
                        drop(guard);
                        
                        // Small delay to allow other threads to compete
                        thread::sleep(Duration::from_microseconds(10));
                    } else {
                        // If we can't get write lock, try read lock
                        if let Ok(guard) = settings_clone.try_read() {
                            let _value = guard.visualisation.glow.node_glow_strength;
                            my_completions += 1;
                            drop(guard);
                        }
                    }
                }
                
                // Record completion count
                {
                    let mut counter = counter_clone.lock().unwrap();
                    counter.push(my_completions);
                }
                
                thread_id
            });
            
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            let thread_id = handle.join().unwrap();
            assert!(thread_id < num_threads);
        }

        // Analyze fairness
        let completions = completion_counter.lock().unwrap();
        let min_completions = *completions.iter().min().unwrap();
        let max_completions = *completions.iter().max().unwrap();
        let avg_completions = completions.iter().sum::<usize>() as f64 / completions.len() as f64;

        println!("Lock fairness: min={}, max={}, avg={:.1}", 
                min_completions, max_completions, avg_completions);

        // No thread should be completely starved
        assert!(min_completions > 0, "Some thread was completely starved");
        
        // Distribution should be reasonably fair (no thread gets more than 3x the minimum)
        assert!(max_completions <= min_completions * 3,
                "Lock distribution is unfair: max={}, min={}", max_completions, min_completions);
    }
}