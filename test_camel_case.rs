use ext::config::{PhysicsSettings, JsonPathAccessible};
use serde_json::json;

#[test]
fn test_camel_case_serialization() {
    // Create a PhysicsSettings with known values
    let mut physics = PhysicsSettings::default();
    physics.spring_k = 0.005;
    physics.repel_k = 50.0;
    
    // Serialize to JSON
    let serialized = serde_json::to_string_pretty(&physics).unwrap();
    println!("Serialized PhysicsSettings:\n{}", serialized);
    
    // Check if it uses camelCase
    assert!(serialized.contains("\"springK\""), "Should serialize as springK (camelCase)");
    assert!(serialized.contains("\"repelK\""), "Should serialize as repelK (camelCase)");
    
    println!("✅ Serialization uses camelCase");
}

#[test]
fn test_camel_case_deserialization() {
    // Test camelCase JSON input
    let camel_case_json = json!({
        "springK": 0.1,
        "repelK": 100.0,
        "damping": 0.95
    });
    
    println!("Attempting to deserialize camelCase JSON:");
    println!("{}", serde_json::to_string_pretty(&camel_case_json).unwrap());
    
    let result = serde_json::from_value::<PhysicsSettings>(camel_case_json);
    match result {
        Ok(physics) => {
            println!("✅ SUCCESS: Deserialized camelCase JSON");
            assert_eq!(physics.spring_k, 0.1, "springK should be 0.1");
            assert_eq!(physics.repel_k, 100.0, "repelK should be 100.0");
            assert_eq!(physics.damping, 0.95, "damping should be 0.95");
        }
        Err(e) => {
            println!("❌ FAILED: Could not deserialize camelCase JSON: {}", e);
            panic!("Deserialization should work with camelCase");
        }
    }
}

#[test]
fn test_json_path_accessible() {
    let mut physics = PhysicsSettings::default();
    physics.spring_k = 0.005;
    
    // Test getting a value with camelCase path
    let value = physics.get_json_by_path("springK").unwrap();
    assert_eq!(value, json!(0.005));
    println!("✅ JsonPathAccessible can get springK value");
    
    // Test setting a value with camelCase path
    physics.set_json_by_path("springK", json!(0.123)).unwrap();
    assert_eq!(physics.spring_k, 0.123);
    println!("✅ JsonPathAccessible can set springK value");
}

fn main() {
    test_camel_case_serialization();
    test_camel_case_deserialization();
    test_json_path_accessible();
    println!("All tests passed! The serialization fix is working correctly.");
}