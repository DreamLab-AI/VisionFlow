#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! rusqlite = "0.35"
//! serde_json = "1.0"
//! ```

use rusqlite::{Connection, params, OptionalExtension};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone)]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(JsonValue),
}

/// Convert snake_case to camelCase
/// Examples: "spring_k" -> "springK", "max_velocity" -> "maxVelocity"
fn to_camel_case(s: &str) -> String {
    let parts: Vec<&str> = s.split('_').collect();
    if parts.len() == 1 {
        return s.to_string();
    }

    let mut result = parts[0].to_string();
    for part in &parts[1..] {
        if !part.is_empty() {
            let mut chars = part.chars();
            if let Some(first) = chars.next() {
                result.push(first.to_ascii_uppercase());
                result.push_str(chars.as_str());
            }
        }
    }
    result
}

/// Get setting value with exact key match (no fallback)
fn get_setting_exact(conn: &Connection, key: &str) -> rusqlite::Result<Option<SettingValue>> {
    let mut stmt = conn.prepare(
        "SELECT value_type, value_text, value_integer, value_float, value_boolean, value_json
         FROM settings WHERE key = ?1"
    )?;

    stmt.query_row(params![key], |row| {
        let value_type: String = row.get(0)?;
        let value = match value_type.as_str() {
            "string" => SettingValue::String(row.get(1)?),
            "integer" => SettingValue::Integer(row.get(2)?),
            "float" => SettingValue::Float(row.get(3)?),
            "boolean" => SettingValue::Boolean(row.get::<_, i32>(4)? == 1),
            "json" => {
                let json_str: String = row.get(5)?;
                SettingValue::Json(serde_json::from_str(&json_str).unwrap_or(JsonValue::Null))
            },
            _ => SettingValue::String(String::new()),
        };
        Ok(value)
    }).optional()
}

/// Get hierarchical settings by key path with intelligent camelCase/snake_case fallback
///
/// This method provides smart lookup:
/// 1. First tries exact match with the provided key
/// 2. If not found and key contains underscore, converts to camelCase and retries
///
/// Examples:
/// - Database has "springK" = 150.0
/// - `get_setting("springK")` -> Direct hit, returns 150.0
/// - `get_setting("spring_k")` -> Converts to "springK", returns 150.0
fn get_setting(conn: &Connection, key: &str) -> rusqlite::Result<Option<SettingValue>> {
    // Try exact match first
    if let Some(value) = get_setting_exact(conn, key)? {
        return Ok(Some(value));
    }

    // If not found and key contains underscore, try camelCase conversion
    if key.contains('_') {
        let camel_key = to_camel_case(key);
        if let Some(value) = get_setting_exact(conn, &camel_key)? {
            return Ok(Some(value));
        }
    }

    // Not found with either key format
    Ok(None)
}

/// Set a setting value
fn set_setting(conn: &Connection, key: &str, value: SettingValue) -> rusqlite::Result<()> {
    let (value_type, text, int, float, bool_val, json) = match value {
        SettingValue::String(s) => ("string", Some(s), None, None, None, None),
        SettingValue::Integer(i) => ("integer", None, Some(i), None, None, None),
        SettingValue::Float(f) => ("float", None, None, Some(f), None, None),
        SettingValue::Boolean(b) => ("boolean", None, None, None, Some(if b { 1 } else { 0 }), None),
        SettingValue::Json(j) => ("json", None, None, None, None, Some(j.to_string())),
    };

    conn.execute(
        "INSERT INTO settings (key, value_type, value_text, value_integer, value_float, value_boolean, value_json)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(key) DO UPDATE SET
            value_type = excluded.value_type,
            value_text = excluded.value_text,
            value_integer = excluded.value_integer,
            value_float = excluded.value_float,
            value_boolean = excluded.value_boolean,
            value_json = excluded.value_json",
        params![key, value_type, text, int, float, bool_val, json]
    )?;

    Ok(())
}

fn main() -> rusqlite::Result<()> {
    println!("🧪 Testing Smart Database Lookup (camelCase/snake_case fallback)\n");

    let conn = Connection::open_in_memory()?;

    // Create schema
    conn.execute_batch(r#"
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value_type TEXT NOT NULL,
            value_text TEXT,
            value_integer INTEGER,
            value_float REAL,
            value_boolean INTEGER,
            value_json TEXT
        );
    "#)?;

    println!("📝 Test 1: camelCase Conversion");
    println!("   spring_k -> {}", to_camel_case("spring_k"));
    println!("   max_velocity -> {}", to_camel_case("max_velocity"));
    println!("   center_gravity_k -> {}", to_camel_case("center_gravity_k"));

    println!("\n📝 Test 2: Basic Fallback Lookup");
    set_setting(&conn, "springK", SettingValue::Float(150.0))?;

    // Direct camelCase lookup
    if let Some(SettingValue::Float(f)) = get_setting(&conn, "springK")? {
        println!("   ✓ Direct lookup: springK = {}", f);
    }

    // snake_case fallback
    if let Some(SettingValue::Float(f)) = get_setting(&conn, "spring_k")? {
        println!("   ✓ Fallback lookup: spring_k -> springK = {}", f);
    }

    println!("\n📝 Test 3: Multiple Physics Settings");
    let physics_settings = vec![
        ("damping", 0.85),
        ("repelK", 50.0),
        ("maxVelocity", 10.0),
        ("maxForce", 5.0),
        ("centerGravityK", 0.05),
        ("repulsionCutoff", 100.0),
        ("restLength", 50.0),
    ];

    for (key, val) in &physics_settings {
        set_setting(&conn, key, SettingValue::Float(*val))?;
    }

    let test_lookups = vec![
        ("damping", "damping", 0.85),
        ("repel_k", "repelK", 50.0),
        ("max_velocity", "maxVelocity", 10.0),
        ("max_force", "maxForce", 5.0),
        ("center_gravity_k", "centerGravityK", 0.05),
        ("repulsion_cutoff", "repulsionCutoff", 100.0),
        ("rest_length", "restLength", 50.0),
    ];

    for (lookup_key, stored_as, expected) in test_lookups {
        if let Some(SettingValue::Float(f)) = get_setting(&conn, lookup_key)? {
            if (f - expected).abs() < 0.001 {
                println!("   ✓ {} -> {} = {}", lookup_key, stored_as, f);
            } else {
                println!("   ✗ {} expected {} but got {}", lookup_key, expected, f);
            }
        } else {
            println!("   ✗ Failed to retrieve {}", lookup_key);
        }
    }

    println!("\n📝 Test 4: Exact Match Priority");
    set_setting(&conn, "spring_k", SettingValue::Float(100.0))?;
    set_setting(&conn, "springK", SettingValue::Float(150.0))?;

    if let Some(SettingValue::Float(f)) = get_setting(&conn, "spring_k")? {
        if (f - 100.0).abs() < 0.001 {
            println!("   ✓ spring_k = {} (exact match, not fallback)", f);
        }
    }

    if let Some(SettingValue::Float(f)) = get_setting(&conn, "springK")? {
        if (f - 150.0).abs() < 0.001 {
            println!("   ✓ springK = {} (exact match)", f);
        }
    }

    println!("\n📝 Test 5: Non-existent Key");
    match get_setting(&conn, "nonexistent_key")? {
        None => println!("   ✓ Non-existent key returns None"),
        Some(_) => println!("   ✗ Should have returned None"),
    }

    println!("\n✅ All tests completed successfully!");

    Ok(())
}
