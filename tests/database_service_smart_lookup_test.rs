// Standalone test for database_service smart camelCase/snake_case lookup
// This test can be run independently to verify the implementation

#[cfg(test)]
mod smart_lookup_tests {
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

    /// Get setting with exact key match
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

    /// Get setting with smart camelCase/snake_case fallback
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

    fn setup_test_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();

        // Create minimal schema
        conn.execute_batch(r#"
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value_type TEXT NOT NULL,
                value_text TEXT,
                value_integer INTEGER,
                value_float REAL,
                value_boolean INTEGER,
                value_json TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        "#).unwrap();

        conn
    }

    #[test]
    fn test_to_camel_case() {
        assert_eq!(to_camel_case("spring_k"), "springK");
        assert_eq!(to_camel_case("max_velocity"), "maxVelocity");
        assert_eq!(to_camel_case("repulsion_cutoff"), "repulsionCutoff");
        assert_eq!(to_camel_case("center_gravity_k"), "centerGravityK");
        assert_eq!(to_camel_case("springK"), "springK");
        assert_eq!(to_camel_case("damping"), "damping");
        println!("✓ camelCase conversion tests passed");
    }

    #[test]
    fn test_camel_snake_fallback_lookup() {
        let conn = setup_test_db();

        // Store setting in camelCase format (preferred format)
        set_setting(&conn, "springK", SettingValue::Float(150.0)).unwrap();

        // Test direct camelCase lookup (exact match)
        let value = get_setting(&conn, "springK").unwrap();
        assert!(value.is_some());
        if let Some(SettingValue::Float(f)) = value {
            assert_eq!(f, 150.0);
            println!("✓ Direct camelCase lookup: springK = {}", f);
        }

        // Test snake_case lookup with camelCase fallback
        let value = get_setting(&conn, "spring_k").unwrap();
        assert!(value.is_some());
        if let Some(SettingValue::Float(f)) = value {
            assert_eq!(f, 150.0);
            println!("✓ snake_case fallback lookup: spring_k -> springK = {}", f);
        }

        // Test non-existent key returns None
        let value = get_setting(&conn, "nonexistent_key").unwrap();
        assert!(value.is_none());
        println!("✓ Non-existent key returns None");
    }

    #[test]
    fn test_multiple_physics_settings_fallback() {
        let conn = setup_test_db();

        // Store multiple physics settings in camelCase
        set_setting(&conn, "repelK", SettingValue::Float(50.0)).unwrap();
        set_setting(&conn, "maxVelocity", SettingValue::Float(10.0)).unwrap();
        set_setting(&conn, "centerGravityK", SettingValue::Float(0.1)).unwrap();

        // Test snake_case lookups
        assert_eq!(
            match get_setting(&conn, "repel_k").unwrap() {
                Some(SettingValue::Float(f)) => f,
                _ => panic!("Expected float"),
            },
            50.0
        );
        println!("✓ repel_k -> repelK = 50.0");

        assert_eq!(
            match get_setting(&conn, "max_velocity").unwrap() {
                Some(SettingValue::Float(f)) => f,
                _ => panic!("Expected float"),
            },
            10.0
        );
        println!("✓ max_velocity -> maxVelocity = 10.0");

        assert_eq!(
            match get_setting(&conn, "center_gravity_k").unwrap() {
                Some(SettingValue::Float(f)) => f,
                _ => panic!("Expected float"),
            },
            0.1
        );
        println!("✓ center_gravity_k -> centerGravityK = 0.1");
    }

    #[test]
    fn test_exact_match_priority() {
        let conn = setup_test_db();

        // Store both snake_case and camelCase versions
        set_setting(&conn, "spring_k", SettingValue::Float(100.0)).unwrap();
        set_setting(&conn, "springK", SettingValue::Float(150.0)).unwrap();

        // Exact match should always take priority
        let value1 = get_setting(&conn, "spring_k").unwrap();
        if let Some(SettingValue::Float(f)) = value1 {
            assert_eq!(f, 100.0);
            println!("✓ Exact match priority: spring_k = 100.0 (not fallback)");
        }

        let value2 = get_setting(&conn, "springK").unwrap();
        if let Some(SettingValue::Float(f)) = value2 {
            assert_eq!(f, 150.0);
            println!("✓ Exact match priority: springK = 150.0");
        }
    }

    #[test]
    fn test_comprehensive_physics_settings() {
        let conn = setup_test_db();

        // Simulate storing all physics settings in camelCase (as they come from frontend)
        let physics_settings = vec![
            ("damping", 0.85),
            ("springK", 150.0),
            ("repelK", 50.0),
            ("maxVelocity", 10.0),
            ("maxForce", 5.0),
            ("centerGravityK", 0.05),
            ("repulsionCutoff", 100.0),
            ("restLength", 50.0),
        ];

        for (key, val) in &physics_settings {
            set_setting(&conn, key, SettingValue::Float(*val)).unwrap();
        }

        // Test retrieval with snake_case (as used in Rust code)
        let test_cases = vec![
            ("damping", 0.85),
            ("spring_k", 150.0),
            ("repel_k", 50.0),
            ("max_velocity", 10.0),
            ("max_force", 5.0),
            ("center_gravity_k", 0.05),
            ("repulsion_cutoff", 100.0),
            ("rest_length", 50.0),
        ];

        println!("\n=== Comprehensive Physics Settings Test ===");
        for (key, expected) in test_cases {
            let value = get_setting(&conn, key).unwrap();
            if let Some(SettingValue::Float(f)) = value {
                assert_eq!(f, expected);
                println!("✓ {} = {}", key, f);
            } else {
                panic!("Failed to retrieve {}", key);
            }
        }
    }
}
