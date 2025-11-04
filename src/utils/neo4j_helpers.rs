// src/utils/neo4j_helpers.rs
//! Neo4j BoltType Conversion Utilities
//!
//! Provides helper functions for converting Rust types to Neo4j BoltType
//! for database operations with the neo4rs library.

use neo4rs::BoltType;
use serde_json::Value as JsonValue;
use std::collections::HashMap;

/// Convert serde_json::Value to Neo4j BoltType
///
/// Recursively converts JSON values to their corresponding BoltType representations.
/// This handles all JSON types including nested objects and arrays.
///
/// Note: neo4rs provides automatic From implementations for Vec<T> and HashMap<String, T>
/// where T: Into<BoltType>, so we use those instead of manually constructing BoltList/BoltMap.
pub fn json_to_bolt(value: JsonValue) -> BoltType {
    match value {
        JsonValue::Null => BoltType::Null(neo4rs::BoltNull),
        JsonValue::Bool(b) => BoltType::from(b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                BoltType::from(i)
            } else if let Some(f) = n.as_f64() {
                BoltType::from(f)
            } else {
                // Fallback to string representation for edge cases
                BoltType::from(n.to_string())
            }
        }
        JsonValue::String(s) => BoltType::from(s),
        JsonValue::Array(arr) => {
            // Vec<BoltType> automatically converts to BoltType::List via From trait
            let list: Vec<BoltType> = arr.into_iter().map(json_to_bolt).collect();
            BoltType::from(list)
        }
        JsonValue::Object(obj) => {
            // HashMap<String, BoltType> automatically converts to BoltType::Map via From trait
            let map: HashMap<String, BoltType> = obj
                .into_iter()
                .map(|(k, v)| (k, json_to_bolt(v)))
                .collect();
            BoltType::from(map)
        }
    }
}

/// Convert a string reference to Neo4j BoltType
///
/// Creates a BoltString from a string reference using the From trait.
pub fn string_ref_to_bolt(s: &str) -> BoltType {
    BoltType::from(s.to_string())
}

/// Convert owned String to Neo4j BoltType
pub fn string_to_bolt(s: String) -> BoltType {
    BoltType::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_to_bolt_null() {
        let value = json!(null);
        let bolt = json_to_bolt(value);
        assert!(matches!(bolt, BoltType::Null(_)));
    }

    #[test]
    fn test_json_to_bolt_bool() {
        let value = json!(true);
        let bolt = json_to_bolt(value);
        assert!(matches!(bolt, BoltType::Boolean(_)));
    }

    #[test]
    fn test_json_to_bolt_number() {
        let value = json!(42);
        let bolt = json_to_bolt(value);
        assert!(matches!(bolt, BoltType::Integer(_)));

        let value = json!(3.14);
        let bolt = json_to_bolt(value);
        assert!(matches!(bolt, BoltType::Float(_)));
    }

    #[test]
    fn test_json_to_bolt_string() {
        let value = json!("hello");
        let bolt = json_to_bolt(value);
        assert!(matches!(bolt, BoltType::String(_)));
    }

    #[test]
    fn test_json_to_bolt_array() {
        let value = json!([1, 2, 3]);
        let bolt = json_to_bolt(value);
        assert!(matches!(bolt, BoltType::List(_)));
    }

    #[test]
    fn test_json_to_bolt_object() {
        let value = json!({"key": "value"});
        let bolt = json_to_bolt(value);
        assert!(matches!(bolt, BoltType::Map(_)));
    }

    #[test]
    fn test_string_ref_to_bolt() {
        let s = "test string";
        let bolt = string_ref_to_bolt(s);
        assert!(matches!(bolt, BoltType::String(_)));
    }

    #[test]
    fn test_string_to_bolt() {
        let s = String::from("test string");
        let bolt = string_to_bolt(s);
        assert!(matches!(bolt, BoltType::String(_)));
    }
}
