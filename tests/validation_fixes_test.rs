#[cfg(test)]
mod validation_fixes_tests {
    use serde_json::{json, Value};

    // Test that numeric strings can be parsed
    #[test]
    fn test_parse_numeric_strings() {
        let test_cases = vec![
            ("1.35", 1.35),
            ("0.65", 0.65),
            ("0.55", 0.55),
            ("-1.5", -1.5),
            ("100", 100.0),
            ("0", 0.0),
        ];

        for (input, expected) in test_cases {
            let parsed: f64 = input.parse().unwrap();
            assert_eq!(parsed, expected, "Failed to parse '{}'", input);
        }
    }

    // Test that Value::String can be converted to Value::Number
    #[test]
    fn test_string_to_number_conversion() {
        let string_val = Value::String("1.35".to_string());

        // Extract string and parse
        if let Value::String(s) = &string_val {
            let parsed: f64 = s.parse().unwrap();
            let number_val = Value::Number(serde_json::Number::from_f64(parsed).unwrap());

            assert!(number_val.is_number());
            assert_eq!(number_val.as_f64().unwrap(), 1.35);
        }
    }

    // Test compatible types with string-to-number conversion
    #[test]
    fn test_compatible_types_with_conversion() {
        let existing = json!(1.0);
        let new_string = json!("1.35");

        // Should be compatible if we allow string-to-number conversion
        let compatible = match (&existing, &new_string) {
            (Value::Number(_), Value::String(s)) => s.parse::<f64>().is_ok(),
            _ => false,
        };

        assert!(
            compatible,
            "String '1.35' should be compatible with number type"
        );
    }
}
