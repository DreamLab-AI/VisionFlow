// API wrapper for automatic case conversion between snake_case (Rust) and camelCase (API)
// This provides a clean separation between internal representation and API format

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Wrapper that automatically converts between snake_case and camelCase at serialization boundaries
#[derive(Debug, Clone)]
pub struct ApiWrapper<T> {
    inner: T,
}

impl<T> ApiWrapper<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
    
    pub fn into_inner(self) -> T {
        self.inner
    }
    
    pub fn inner(&self) -> &T {
        &self.inner
    }
    
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T: Serialize> Serialize for ApiWrapper<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // First serialize to Value
        let value = serde_json::to_value(&self.inner)
            .map_err(serde::ser::Error::custom)?;
        
        // Convert keys to camelCase
        let camel_value = convert_keys_to_camel(value);
        
        // Serialize the converted value
        camel_value.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ApiWrapper<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // First deserialize to Value
        let value = Value::deserialize(deserializer)?;
        
        // Convert keys to snake_case
        let snake_value = convert_keys_to_snake(value);
        
        // Deserialize to inner type
        let inner = T::deserialize(snake_value)
            .map_err(serde::de::Error::custom)?;
        
        Ok(ApiWrapper::new(inner))
    }
}

/// Convert all keys in a JSON value from snake_case to camelCase recursively
fn convert_keys_to_camel(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (key, val) in map {
                let camel_key = snake_to_camel(&key);
                new_map.insert(camel_key, convert_keys_to_camel(val));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(convert_keys_to_camel).collect())
        }
        other => other,
    }
}

/// Convert all keys in a JSON value from camelCase to snake_case recursively
fn convert_keys_to_snake(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = serde_json::Map::new();
            for (key, val) in map {
                let snake_key = camel_to_snake(&key);
                new_map.insert(snake_key, convert_keys_to_snake(val));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(convert_keys_to_snake).collect())
        }
        other => other,
    }
}

/// Convert snake_case to camelCase
fn snake_to_camel(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;
    
    for ch in s.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    
    result
}

/// Convert camelCase to snake_case
fn camel_to_snake(s: &str) -> String {
    let mut result = String::new();
    
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
    }
    
    result
}