#[test]
fn test_string_to_number_conversion() {
    use serde_json::{json, Value};

    // Test that strings can be parsed as numbers
    let test_value = json!("1.35");
    if let Value::String(s) = &test_value {
        let parsed: Result<f64, _> = s.parse();
        assert!(parsed.is_ok());
        assert_eq!(parsed.unwrap(), 1.35);
    }

    // Test various numeric strings
    let numeric_strings = vec!["0.65", "0.55", "-1.5", "100", "0"];
    for s in numeric_strings {
        let parsed: Result<f64, _> = s.parse();
        assert!(parsed.is_ok(), "Failed to parse '{}'", s);
    }
}

#[test]
fn test_values_compatible_with_conversion() {
    use serde_json::{json, Value};

    // Simulate the improved values_have_compatible_types logic
    fn values_compatible(existing: &Value, new_value: &Value) -> bool {
        match (existing, new_value) {
            (Value::Number(_), Value::String(s)) => s.parse::<f64>().is_ok(),
            (Value::Number(_), Value::Number(_)) => true,
            (Value::String(_), Value::String(_)) => true,
            _ => false,
        }
    }

    let existing = json!(1.0);
    let new_string = json!("1.35");
    let new_number = json!(1.35);

    assert!(
        values_compatible(&existing, &new_string),
        "String should be compatible with number"
    );
    assert!(
        values_compatible(&existing, &new_number),
        "Number should be compatible with number"
    );
}
