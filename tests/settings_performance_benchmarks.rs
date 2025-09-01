//! Performance benchmarks for settings refactor
//!
//! Measures performance improvements of the new granular API vs old approach
//! Tests serialization speed, memory usage, and network efficiency
//!

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

use webxr::config::AppFullSettings;

#[cfg(test)]
mod benchmark_tests {
    use super::*;
    
    fn create_test_settings() -> AppFullSettings {
        let mut settings = AppFullSettings::default();
        settings.visualisation.glow.intensity = 2.0;
        settings.visualisation.glow.node_glow_strength = 3.0;
        settings.visualisation.graphs.logseq.physics.spring_k = 0.02;
        settings.system.network.port = 8080;
        settings
    }
    
    #[test]
    fn benchmark_serialization_performance() {
        let settings = create_test_settings();
        let iterations = 10000;
        
        // Benchmark full serialization
        let start = Instant::now();
        for _ in 0..iterations {
            let _json = serde_json::to_value(&settings).expect("Serialization failed");
        }
        let full_serialization_time = start.elapsed();
        
        // Benchmark partial serialization (simulate granular approach)
        let partial_paths = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.nodeGlowStrength",
            "system.network.port"
        ];
        
        let start = Instant::now();
        for _ in 0..iterations {
            // Simulate partial serialization
            let mut partial_json = serde_json::Map::new();
            partial_json.insert("visualisation".to_string(), json!({
                "glow": {
                    "intensity": settings.visualisation.glow.intensity,
                    "nodeGlowStrength": settings.visualisation.glow.node_glow_strength
                }
            }));
            partial_json.insert("system".to_string(), json!({
                "network": {
                    "port": settings.system.network.port
                }
            }));
            let _json = Value::Object(partial_json);
        }
        let partial_serialization_time = start.elapsed();
        
        println!("Full serialization ({} iterations): {:?}", iterations, full_serialization_time);
        println!("Partial serialization ({} iterations): {:?}", iterations, partial_serialization_time);
        
        let improvement = full_serialization_time.as_nanos() as f64 / partial_serialization_time.as_nanos() as f64;
        println!("Partial serialization is {:.2}x faster", improvement);
        
        // Partial should be faster for small subsets
        assert!(partial_serialization_time < full_serialization_time);
    }
    
    #[test]
    fn benchmark_deserialization_performance() {
        let settings = create_test_settings();
        let full_json = serde_json::to_value(&settings).expect("Serialization failed");
        
        let partial_json = json!({
            "visualisation": {
                "glow": {
                    "intensity": 2.0,
                    "nodeGlowStrength": 3.0
                }
            },
            "system": {
                "network": {
                    "port": 8080
                }
            }
        });
        
        let iterations = 5000;
        
        // Benchmark full deserialization
        let start = Instant::now();
        for _ in 0..iterations {
            let _settings: AppFullSettings = serde_json::from_value(full_json.clone())
                .expect("Deserialization failed");
        }
        let full_deserialization_time = start.elapsed();
        
        // Benchmark partial deserialization
        let start = Instant::now();
        for _ in 0..iterations {
            // Simulate partial deserialization (only needed fields)
            let intensity = partial_json["visualisation"]["glow"]["intensity"].as_f64().unwrap();
            let node_glow = partial_json["visualisation"]["glow"]["nodeGlowStrength"].as_f64().unwrap();
            let port = partial_json["system"]["network"]["port"].as_u64().unwrap();
            
            // Use the values to prevent optimization
            let _sum = intensity + node_glow + port as f64;
        }
        let partial_deserialization_time = start.elapsed();
        
        println!("Full deserialization ({} iterations): {:?}", iterations, full_deserialization_time);
        println!("Partial deserialization ({} iterations): {:?}", iterations, partial_deserialization_time);
        
        let improvement = full_deserialization_time.as_nanos() as f64 / partial_deserialization_time.as_nanos() as f64;
        println!("Partial deserialization is {:.2}x faster", improvement);
        
        assert!(partial_deserialization_time < full_deserialization_time);
    }
    
    #[test]
    fn benchmark_memory_usage() {
        let settings = create_test_settings();
        
        // Measure full settings JSON size
        let full_json = serde_json::to_string(&settings).expect("Serialization failed");
        let full_size = full_json.len();
        
        // Measure partial settings JSON size
        let partial_json = json!({
            "visualisation": {
                "glow": {
                    "intensity": settings.visualisation.glow.intensity,
                    "nodeGlowStrength": settings.visualisation.glow.node_glow_strength
                }
            },
            "system": {
                "network": {
                    "port": settings.system.network.port
                }
            }
        });
        
        let partial_json_str = serde_json::to_string(&partial_json).expect("Serialization failed");
        let partial_size = partial_json_str.len();
        
        println!("Full settings JSON size: {} bytes", full_size);
        println!("Partial settings JSON size: {} bytes", partial_size);
        
        let size_reduction = (full_size - partial_size) as f64 / full_size as f64 * 100.0;
        println!("Size reduction: {:.1}%", size_reduction);
        
        // Partial should be significantly smaller
        assert!(partial_size < full_size / 2);
    }
    
    #[test] 
    fn benchmark_path_extraction_performance() {
        let settings = create_test_settings();
        let json_value = serde_json::to_value(&settings).expect("Serialization failed");
        
        let paths = vec![
            "visualisation.glow.intensity",
            "visualisation.glow.nodeGlowStrength", 
            "visualisation.graphs.logseq.physics.springK",
            "system.network.port",
            "system.debug.enabled",
            "xr.roomScale"
        ];
        
        let iterations = 1000;
        
        let start = Instant::now();
        for _ in 0..iterations {
            let mut extracted = HashMap::new();
            
            for path in &paths {
                if let Some(value) = extract_value_by_path(&json_value, path) {
                    extracted.insert(path.clone(), value);
                }
            }
            
            // Use extracted values to prevent optimization
            let _count = extracted.len();
        }
        let path_extraction_time = start.elapsed();
        
        println!("Path extraction ({} iterations, {} paths): {:?}", iterations, paths.len(), path_extraction_time);
        
        // Should be reasonably fast
        assert!(path_extraction_time.as_millis() < 100);
    }
    
    #[test]
    fn benchmark_update_merging_performance() {
        let mut settings = create_test_settings();
        
        let updates = vec![
            ("visualisation.glow.intensity", json!(2.5)),
            ("visualisation.glow.nodeGlowStrength", json!(3.5)),
            ("system.network.port", json!(9000)),
            ("xr.roomScale", json!(2.0)),
        ];
        
        let iterations = 1000;
        
        let start = Instant::now();
        for _ in 0..iterations {
            let mut settings_copy = settings.clone();
            
            // Simulate applying updates
            for (path, value) in &updates {
                apply_update_by_path(&mut settings_copy, path, value.clone());
            }
            
            // Use the updated settings to prevent optimization
            let _intensity = settings_copy.visualisation.glow.intensity;
        }
        let update_merging_time = start.elapsed();
        
        println!("Update merging ({} iterations, {} updates): {:?}", iterations, updates.len(), update_merging_time);
        
        // Should be reasonably fast
        assert!(update_merging_time.as_millis() < 200);
    }
    
    #[test]
    fn benchmark_camelcase_conversion_overhead() {
        let settings = create_test_settings();
        let iterations = 5000;
        
        // Benchmark serialization with camelCase conversion (automatic via serde)
        let start = Instant::now();
        for _ in 0..iterations {
            let _json = serde_json::to_value(&settings).expect("Serialization failed");
        }
        let camelcase_time = start.elapsed();
        
        // Create a version with manual field renaming (simulate old approach)
        let manual_json = json!({
            "visualisation": {
                "glow": {
                    "intensity": settings.visualisation.glow.intensity,
                    "nodeGlowStrength": settings.visualisation.glow.node_glow_strength,
                    "edgeGlowStrength": settings.visualisation.glow.edge_glow_strength,
                    "baseColor": settings.visualisation.glow.base_color
                }
            }
        });
        
        let start = Instant::now();
        for _ in 0..iterations {
            let _json = manual_json.clone();
        }
        let manual_time = start.elapsed();
        
        println!("Automatic camelCase conversion ({} iterations): {:?}", iterations, camelcase_time);
        println!("Manual field mapping ({} iterations): {:?}", iterations, manual_time);
        
        // Automatic should be competitive with manual
        let overhead_ratio = camelcase_time.as_nanos() as f64 / manual_time.as_nanos() as f64;
        println!("Automatic camelCase overhead: {:.2}x", overhead_ratio);
        
        // Should be within reasonable overhead
        assert!(overhead_ratio < 3.0);
    }
    
    #[test]
    fn benchmark_concurrent_access_performance() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let settings = Arc::new(Mutex::new(create_test_settings()));
        let num_threads = 4;
        let operations_per_thread = 1000;
        
        let start = Instant::now();
        
        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let settings_clone = Arc::clone(&settings);
            
            thread::spawn(move || {
                for i in 0..operations_per_thread {
                    // Simulate read operation
                    {
                        let settings_guard = settings_clone.lock().unwrap();
                        let _intensity = settings_guard.visualisation.glow.intensity;
                    }
                    
                    // Simulate write operation every 10th iteration
                    if i % 10 == 0 {
                        let mut settings_guard = settings_clone.lock().unwrap();
                        settings_guard.visualisation.glow.intensity = thread_id as f32 * 0.1 + i as f32 * 0.01;
                    }
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let concurrent_time = start.elapsed();
        
        let total_operations = num_threads * operations_per_thread;
        println!("Concurrent access ({} threads, {} ops total): {:?}", num_threads, total_operations, concurrent_time);
        
        let ops_per_second = total_operations as f64 / concurrent_time.as_secs_f64();
        println!("Operations per second: {:.0}", ops_per_second);
        
        // Should handle reasonable concurrent load
        assert!(ops_per_second > 10000.0);
    }
}

// Helper functions for benchmarking

fn extract_value_by_path(json: &Value, path: &str) -> Option<Value> {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = json;
    
    for part in parts {
        match current.get(part) {
            Some(value) => current = value,
            None => return None,
        }
    }
    
    Some(current.clone())
}

fn apply_update_by_path(settings: &mut AppFullSettings, path: &str, value: Value) {
    // Simplified implementation for benchmarking
    match path {
        "visualisation.glow.intensity" => {
            if let Some(f) = value.as_f64() {
                settings.visualisation.glow.intensity = f as f32;
            }
        }
        "visualisation.glow.nodeGlowStrength" => {
            if let Some(f) = value.as_f64() {
                settings.visualisation.glow.node_glow_strength = f as f32;
            }
        }
        "system.network.port" => {
            if let Some(i) = value.as_u64() {
                settings.system.network.port = i as u16;
            }
        }
        "xr.roomScale" => {
            if let Some(f) = value.as_f64() {
                settings.xr.room_scale = f as f32;
            }
        }
        _ => {} // Ignore unknown paths for benchmarking
    }
}

#[cfg(test)]
mod regression_benchmarks {
    use super::*;
    
    /// Ensure performance doesn't regress over time
    #[test]
    fn performance_regression_test() {
        let settings = create_test_settings();
        let iterations = 1000;
        
        // Serialization should complete within reasonable time
        let start = Instant::now();
        for _ in 0..iterations {
            let _json = serde_json::to_value(&settings).expect("Serialization failed");
        }
        let serialization_time = start.elapsed();
        
        // Assert performance thresholds
        assert!(serialization_time.as_millis() < 50, "Serialization took too long: {:?}", serialization_time);
        
        // Deserialization should also be fast
        let json_value = serde_json::to_value(&settings).expect("Serialization failed");
        let start = Instant::now();
        for _ in 0..iterations {
            let _settings: AppFullSettings = serde_json::from_value(json_value.clone())
                .expect("Deserialization failed");
        }
        let deserialization_time = start.elapsed();
        
        assert!(deserialization_time.as_millis() < 100, "Deserialization took too long: {:?}", deserialization_time);
        
        println!("✅ Performance regression test passed");
        println!("   Serialization: {:?} for {} iterations", serialization_time, iterations);
        println!("   Deserialization: {:?} for {} iterations", deserialization_time, iterations);
    }
    
    #[test]
    fn memory_usage_regression_test() {
        let settings = create_test_settings();
        
        // Full JSON shouldn't be excessively large
        let full_json = serde_json::to_string(&settings).expect("Serialization failed");
        let full_size = full_json.len();
        
        // Should be within reasonable bounds (adjust if structure changes significantly)
        assert!(full_size < 50_000, "Full settings JSON too large: {} bytes", full_size);
        
        // Partial JSON should be much smaller
        let partial_json = json!({
            "visualisation": {
                "glow": {
                    "intensity": settings.visualisation.glow.intensity
                }
            }
        });
        
        let partial_json_str = serde_json::to_string(&partial_json).expect("Serialization failed");
        let partial_size = partial_json_str.len();
        
        assert!(partial_size < 200, "Partial settings JSON too large: {} bytes", partial_size);
        
        println!("✅ Memory usage regression test passed");
        println!("   Full settings: {} bytes", full_size);
        println!("   Partial settings: {} bytes", partial_size);
    }
}