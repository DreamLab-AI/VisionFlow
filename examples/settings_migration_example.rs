//! Settings Migration Example
//!
//! Demonstrates settings migration from YAML to SQLite with dual key format support.
//! Shows how both camelCase (client) and snake_case (server) keys work simultaneously.

use std::sync::Arc;

#[cfg(feature = "ontology")]
use webxr::services::database_service::{DatabaseService, SettingValue};
#[cfg(feature = "ontology")]
use webxr::services::settings_migration::{SettingsMigration, KeyFormatConverter};

#[cfg(feature = "ontology")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    println!("=== Settings Migration Example ===\n");

    // 1. Create in-memory database for demonstration
    let db_service = Arc::new(DatabaseService::new(":memory:")?);

    // 2. Initialize schema
    println!("📊 Initializing database schema...");
    db_service.initialize_schema()?;
    println!("✅ Schema initialized\n");

    // 3. Create migration service
    let migration = SettingsMigration::new(Arc::clone(&db_service));

    // 4. Run migration (in production this would read from actual YAML files)
    println!("⚙️  Running settings migration...");

    // For this example, we'll manually insert some settings to demonstrate dual key format
    println!("\n--- Demonstrating Dual Key Format ---\n");

    // Insert a setting using camelCase (client format)
    let camel_key = "visualisation.graphs.logseq.nodes.baseColor";
    db_service.set_setting(camel_key, SettingValue::String("#202724".to_string()), Some("Node base color"))?;
    println!("✅ Stored camelCase key: {}", camel_key);

    // Insert the same setting using snake_case (server format)
    let snake_key = KeyFormatConverter::to_snake_case(camel_key);
    db_service.set_setting(&snake_key, SettingValue::String("#202724".to_string()), Some("Node base color"))?;
    println!("✅ Stored snake_case key: {}", snake_key);

    // Retrieve using camelCase
    println!("\n--- Retrieving Settings ---\n");
    if let Some(value) = db_service.get_setting(camel_key)? {
        println!("Retrieved via camelCase '{}': {:?}", camel_key, value);
    }

    // Retrieve using snake_case
    if let Some(value) = db_service.get_setting(&snake_key)? {
        println!("Retrieved via snake_case '{}': {:?}", snake_key, value);
    }

    // Demonstrate key conversion utilities
    println!("\n--- Key Format Conversion ---\n");

    let test_keys = vec![
        "visualisation.rendering.enableAntialiasing",
        "visualisation.graphs.logseq.physics.springK",
        "system.network.bindAddress",
        "xr.clientSideEnableXr",
    ];

    for key in test_keys {
        let (camel, snake) = KeyFormatConverter::both_formats(key);
        println!("Original: {}", key);
        println!("  camelCase: {}", camel);
        println!("  snake_case: {}", snake);
        println!();
    }

    // Demonstrate physics settings migration
    println!("--- Physics Settings ---\n");

    use webxr::config::PhysicsSettings;

    let logseq_physics = PhysicsSettings {
        damping: 0.6,
        dt: 0.02269863,
        iterations: 50,
        max_velocity: 100.0,
        max_force: 1000.0,
        repel_k: 13.28022,
        spring_k: 4.6001,
        mass_scale: 0.39917582,
        boundary_damping: 0.95,
        temperature: 2.0,
        gravity: 0.0001,
        bounds_size: 1000.0,
        enable_bounds: false,
        rest_length: 50.0,
        repulsion_cutoff: 50.0,
        repulsion_softening_epsilon: 0.0001,
        center_gravity_k: 0.005,
        grid_cell_size: 28.543957,
        warmup_iterations: 100,
        cooling_rate: 0.001,
        constraint_ramp_frames: 60,
        constraint_max_force_per_node: 50.0,
        ..PhysicsSettings::default()
    };

    db_service.save_physics_settings("logseq", &logseq_physics)?;
    println!("✅ Saved physics profile: logseq");

    let retrieved = db_service.get_physics_settings("logseq")?;
    println!("✅ Retrieved physics profile: logseq");
    println!("   Spring constant: {}", retrieved.spring_k);
    println!("   Repulsion constant: {}", retrieved.repel_k);
    println!("   Damping: {}", retrieved.damping);

    // Demonstrate different value types
    println!("\n--- Different Value Types ---\n");

    db_service.set_setting("test.string", SettingValue::String("hello world".to_string()), None)?;
    db_service.set_setting("test.integer", SettingValue::Integer(42), None)?;
    db_service.set_setting("test.float", SettingValue::Float(3.14159), None)?;
    db_service.set_setting("test.boolean", SettingValue::Boolean(true), None)?;

    let json_value = serde_json::json!({
        "array": [1, 2, 3],
        "nested": {
            "key": "value"
        }
    });
    db_service.set_setting("test.json", SettingValue::Json(json_value), None)?;

    if let Some(SettingValue::String(s)) = db_service.get_setting("test.string")? {
        println!("String: {}", s);
    }
    if let Some(SettingValue::Integer(i)) = db_service.get_setting("test.integer")? {
        println!("Integer: {}", i);
    }
    if let Some(SettingValue::Float(f)) = db_service.get_setting("test.float")? {
        println!("Float: {}", f);
    }
    if let Some(SettingValue::Boolean(b)) = db_service.get_setting("test.boolean")? {
        println!("Boolean: {}", b);
    }
    if let Some(SettingValue::Json(j)) = db_service.get_setting("test.json")? {
        println!("JSON: {}", serde_json::to_string_pretty(&j)?);
    }

    println!("\n=== Example Complete ===");

    Ok(())
}

#[cfg(not(feature = "ontology"))]
fn main() {
    println!("This example requires the 'ontology' feature to be enabled.");
    println!("Run with: cargo run --example settings_migration_example --features ontology");
}
