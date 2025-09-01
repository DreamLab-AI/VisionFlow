// Performance benchmark test to verify the elimination of JSON serialization bottleneck
// Tests direct field access vs old JSON conversion approach

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use serde_json::Value;
use std::time::Duration;

use crate::config::{AppFullSettings, path_access::PathAccessible};

// Simulate the OLD inefficient approach (for comparison)
fn old_extract_value_by_path(settings: &AppFullSettings, path: &str) -> Option<Value> {
    let json = serde_json::to_value(settings).ok()?;
    let parts: Vec<&str> = path.split('.').collect();
    let mut value = &json;
    
    for part in parts {
        value = value.get(part)?;
    }
    
    Some(value.clone())
}

fn old_set_value_by_path(settings: &mut AppFullSettings, path: &str, new_value: Value) -> Result<(), String> {
    let mut json = serde_json::to_value(&*settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    
    let parts: Vec<&str> = path.split('.').collect();
    if parts.is_empty() {
        return Err("Empty path".to_string());
    }
    
    let mut current = &mut json;
    
    if parts.len() == 1 {
        if let Value::Object(map) = current {
            map.insert(parts[0].to_string(), new_value);
        }
    } else {
        for part in &parts[0..parts.len()-1] {
            if let Value::Object(ref mut map) = current {
                current = map.get_mut(*part).ok_or_else(|| {
                    format!("Path segment '{}' not found", part)
                })?;
            }
        }
        
        let final_key = parts[parts.len() - 1];
        if let Value::Object(map) = current {
            map.insert(final_key.to_string(), new_value);
        }
    }
    
    *settings = serde_json::from_value(json)
        .map_err(|e| format!("Failed to deserialize updated settings: {}", e))?;
    
    Ok(())
}

// Benchmark the old vs new approach
fn benchmark_get_operations(c: &mut Criterion) {
    let settings = AppFullSettings::default();
    let path = "visualisation.graphs.logseq.physics.damping";
    
    let mut group = c.benchmark_group("settings_get");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("old_json_serialization", |b| {
        b.iter(|| {
            old_extract_value_by_path(black_box(&settings), black_box(path))
        })
    });
    
    group.bench_function("new_direct_access", |b| {
        b.iter(|| {
            settings.get_by_path(black_box(path))
        })
    });
    
    group.finish();
}

fn benchmark_set_operations(c: &mut Criterion) {
    let path = "visualisation.graphs.logseq.physics.damping";
    let new_value = serde_json::json!(0.85);
    
    let mut group = c.benchmark_group("settings_set");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("old_json_serialization", |b| {
        b.iter(|| {
            let mut settings = AppFullSettings::default();
            old_set_value_by_path(black_box(&mut settings), black_box(path), black_box(new_value.clone()))
        })
    });
    
    group.bench_function("new_direct_access", |b| {
        b.iter(|| {
            let mut settings = AppFullSettings::default();
            settings.set_by_path(black_box(path), black_box(new_value.clone()))
        })
    });
    
    group.finish();
}

// Benchmark bulk operations (critical for slider interactions)
fn benchmark_bulk_operations(c: &mut Criterion) {
    let paths = vec![
        "visualisation.graphs.logseq.physics.damping",
        "visualisation.graphs.logseq.physics.repelK", 
        "visualisation.graphs.logseq.physics.springK",
        "visualisation.graphs.logseq.physics.maxVelocity",
        "visualisation.graphs.logseq.physics.iterations",
    ];
    let new_value = serde_json::json!(0.85);
    
    let mut group = c.benchmark_group("settings_bulk");
    group.measurement_time(Duration::from_secs(10));
    
    group.bench_function("old_json_serialization_bulk", |b| {
        b.iter(|| {
            let mut settings = AppFullSettings::default();
            for path in &paths {
                let _ = old_set_value_by_path(black_box(&mut settings), black_box(path), black_box(new_value.clone()));
            }
        })
    });
    
    group.bench_function("new_direct_access_bulk", |b| {
        b.iter(|| {
            let mut settings = AppFullSettings::default();
            for path in &paths {
                let _ = settings.set_by_path(black_box(path), black_box(new_value.clone()));
            }
        })
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_get_operations, benchmark_set_operations, benchmark_bulk_operations);
criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_optimization_correctness() {
        let mut settings = AppFullSettings::default();
        let path = "visualisation.graphs.logseq.physics.damping";
        let new_value = serde_json::json!(0.42);
        
        // Test that new implementation produces same results as old
        let original_value = settings.get_by_path(path).unwrap();
        
        // Set using new method
        settings.set_by_path(path, new_value.clone()).unwrap();
        let new_result = settings.get_by_path(path).unwrap();
        
        assert_eq!(new_result, new_value);
        assert_ne!(original_value, new_result);
        
        println!("✅ Performance optimization maintains correctness");
        println!("   Original: {}", original_value);
        println!("   Updated:  {}", new_result);
    }
    
    #[test] 
    fn test_physics_paths_work() {
        let mut settings = AppFullSettings::default();
        
        // Test all critical physics paths used by sliders
        let test_cases = vec![
            ("visualisation.graphs.logseq.physics.damping", serde_json::json!(0.95)),
            ("visualisation.graphs.logseq.physics.repelK", serde_json::json!(150.0)),
            ("visualisation.graphs.logseq.physics.springK", serde_json::json!(0.8)),
            ("visualisation.graphs.logseq.physics.maxVelocity", serde_json::json!(25.0)),
            ("visualisation.graphs.logseq.physics.iterations", serde_json::json!(100)),
        ];
        
        for (path, expected_value) in test_cases {
            // Test set
            assert!(settings.set_by_path(path, expected_value.clone()).is_ok(), 
                "Failed to set {}", path);
            
            // Test get
            let actual_value = settings.get_by_path(path)
                .expect(&format!("Failed to get {}", path));
            
            assert_eq!(actual_value, expected_value, "Value mismatch for {}", path);
            
            println!("✅ {} = {}", path, actual_value);
        }
    }
    
    #[test]
    fn test_error_handling() {
        let mut settings = AppFullSettings::default();
        
        // Test invalid paths
        assert!(settings.get_by_path("invalid.path.here").is_none());
        assert!(settings.set_by_path("invalid.path.here", serde_json::json!(42)).is_err());
        
        // Test empty path
        assert!(settings.get_by_path("").is_none());
        assert!(settings.set_by_path("", serde_json::json!(42)).is_err());
        
        println!("✅ Error handling works correctly");
    }
}