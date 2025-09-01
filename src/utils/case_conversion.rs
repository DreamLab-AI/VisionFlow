use serde_json::{Map, Value};

/// Convert all keys in a JSON value from snake_case to camelCase recursively
pub fn keys_to_camel_case(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = Map::new();
            for (key, val) in map {
                let camel_key = snake_to_camel(&key);
                new_map.insert(camel_key, keys_to_camel_case(val));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(keys_to_camel_case).collect())
        }
        other => other,
    }
}

/// Convert all keys in a JSON value from camelCase to snake_case recursively
pub fn keys_to_snake_case(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut new_map = Map::new();
            for (key, val) in map {
                let snake_key = camel_to_snake(&key);
                new_map.insert(snake_key, keys_to_snake_case(val));
            }
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(keys_to_snake_case).collect())
        }
        other => other,
    }
}

/// Convert a snake_case string to camelCase
pub fn snake_to_camel(s: &str) -> String {
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

/// Convert a camelCase string to snake_case
pub fn camel_to_snake(s: &str) -> String {
    let mut result = String::new();
    let mut prev_was_uppercase = false;
    
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() {
            // Don't add underscore at the beginning or after another uppercase
            if i > 0 && !prev_was_uppercase {
                result.push('_');
            }
            result.push(ch.to_ascii_lowercase());
            prev_was_uppercase = true;
        } else {
            result.push(ch);
            prev_was_uppercase = false;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_snake_to_camel() {
        assert_eq!(snake_to_camel("snake_case"), "snakeCase");
        assert_eq!(snake_to_camel("auto_balance"), "autoBalance");
        assert_eq!(snake_to_camel("auto_balance_interval_ms"), "autoBalanceIntervalMs");
        assert_eq!(snake_to_camel("spring_k"), "springK");
        assert_eq!(snake_to_camel("simple"), "simple");
    }

    #[test]
    fn test_camel_to_snake() {
        assert_eq!(camel_to_snake("camelCase"), "camel_case");
        assert_eq!(camel_to_snake("autoBalance"), "auto_balance");
        assert_eq!(camel_to_snake("autoBalanceIntervalMs"), "auto_balance_interval_ms");
        assert_eq!(camel_to_snake("springK"), "spring_k");
        assert_eq!(camel_to_snake("simple"), "simple");
    }

    #[test]
    fn test_keys_to_camel_case() {
        let input = json!({
            "auto_balance": true,
            "spring_k": 0.5,
            "nested_object": {
                "repel_k": 0.1,
                "max_velocity": 1.0
            },
            "simple_array": [1, 2, 3]
        });

        let expected = json!({
            "autoBalance": true,
            "springK": 0.5,
            "nestedObject": {
                "repelK": 0.1,
                "maxVelocity": 1.0
            },
            "simpleArray": [1, 2, 3]
        });

        assert_eq!(keys_to_camel_case(input), expected);
    }

    #[test]
    fn test_keys_to_snake_case() {
        let input = json!({
            "autoBalance": true,
            "springK": 0.5,
            "nestedObject": {
                "repelK": 0.1,
                "maxVelocity": 1.0
            },
            "simpleArray": [1, 2, 3]
        });

        let expected = json!({
            "auto_balance": true,
            "spring_k": 0.5,
            "nested_object": {
                "repel_k": 0.1,
                "max_velocity": 1.0
            },
            "simple_array": [1, 2, 3]
        });

        assert_eq!(keys_to_snake_case(input), expected);
    }

    #[test]
    fn test_round_trip_conversion() {
        let original = json!({
            "auto_balance": true,
            "spring_k": 0.5,
            "physics_settings": {
                "repel_k": 0.1,
                "max_velocity": 1.0,
                "enable_bounds": false
            }
        });

        let camel = keys_to_camel_case(original.clone());
        let back_to_snake = keys_to_snake_case(camel);
        
        assert_eq!(original, back_to_snake);
    }
}