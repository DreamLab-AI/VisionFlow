use serde_json::json;
use std::path::PathBuf;

// Import the config module to test deserialization
mod config;
use config::{AppFullSettings, PhysicsSettings, JsonPathAccessible};

fn main() {
    println!("Testing serialization behavior...");
    
    // Test 1: Create a physics settings and serialize
    let mut physics = PhysicsSettings::default();
    physics.spring_k = 0.005;
    physics.repel_k = 50.0;
    
    let serialized = serde_json::to_string_pretty(&physics).unwrap();
    println!("Physics serialized:\n{}\n", serialized);
    
    // Test 2: Test camelCase JSON input
    let camel_case_json = json!({
        "springK": 0.1,
        "repelK": 100.0,
        "damping": 0.95
    });
    
    println!("Attempting to deserialize camelCase JSON:");
    println!("{}\n", serde_json::to_string_pretty(&camel_case_json).unwrap());
    
    match serde_json::from_value::<PhysicsSettings>(camel_case_json.clone()) {
        Ok(physics) => {
            println!("✅ SUCCESS: Deserialized camelCase JSON");
            println!("  springK: {} (expected: 0.1)", physics.spring_k);
            println!("  repelK: {} (expected: 100.0)", physics.repel_k);
            println!("  damping: {} (expected: 0.95)", physics.damping);
        }
        Err(e) => {
            println!("❌ FAILED: Could not deserialize camelCase JSON: {}", e);
        }
    }
    
    // Test 3: Test snake_case JSON input
    let snake_case_json = json!({
        "spring_k": 0.2,
        "repel_k": 200.0,
        "damping": 0.85
    });
    
    println!("\nAttempting to deserialize snake_case JSON:");
    println!("{}\n", serde_json::to_string_pretty(&snake_case_json).unwrap());
    
    match serde_json::from_value::<PhysicsSettings>(snake_case_json) {
        Ok(physics) => {
            println!("✅ SUCCESS: Deserialized snake_case JSON");
            println!("  spring_k: {} (expected: 0.2)", physics.spring_k);
            println!("  repel_k: {} (expected: 200.0)", physics.repel_k);
            println!("  damping: {} (expected: 0.85)", physics.damping);
        }
        Err(e) => {
            println!("❌ FAILED: Could not deserialize snake_case JSON: {}", e);
        }
    }
    
    // Test 4: Test JsonPathAccessible with camelCase paths
    println!("\nTesting JsonPathAccessible with camelCase paths:");
    let mut settings = PhysicsSettings::default();
    
    // Test get
    match settings.get_json_by_path("springK") {
        Ok(value) => {
            println!("✅ SUCCESS: Got springK value: {}", value);
        }
        Err(e) => {
            println!("❌ FAILED: Could not get springK: {}", e);
        }
    }
    
    // Test set
    match settings.set_json_by_path("springK", json!(0.123)) {
        Ok(()) => {
            println!("✅ SUCCESS: Set springK to 0.123, actual value: {}", settings.spring_k);
        }
        Err(e) => {
            println!("❌ FAILED: Could not set springK: {}", e);
        }
    }
}