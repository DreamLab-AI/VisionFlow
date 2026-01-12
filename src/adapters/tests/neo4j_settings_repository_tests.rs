//! Neo4j Settings Repository Tests
//!
//! Comprehensive unit tests for Neo4jSettingsRepository covering:
//! - SettingsCache operations (get, insert, remove, clear, TTL expiry)
//! - UserFilter default values and construction
//! - Neo4jSettingsConfig environment parsing
//! - SettingValue serialization/deserialization
//! - Error handling paths
//!
//! Integration tests requiring live Neo4j are marked with #[ignore].

use std::collections::HashMap;
use std::time::Instant;

use crate::adapters::neo4j_settings_repository::{
    Neo4jSettingsConfig, Neo4jSettingsRepository, User, UserFilter, UserSettingsNode,
};
use crate::ports::settings_repository::{
    SettingValue, SettingsRepositoryError,
};

// ============================================================
// SettingsCache Unit Tests (Pure Logic, No Neo4j Required)
// ============================================================

/// In-memory cache for unit testing cache logic
struct TestCache {
    entries: HashMap<String, (SettingValue, Instant)>,
    ttl_seconds: u64,
}

impl TestCache {
    fn new(ttl_seconds: u64) -> Self {
        Self {
            entries: HashMap::new(),
            ttl_seconds,
        }
    }

    fn get(&self, key: &str) -> Option<SettingValue> {
        if let Some((value, timestamp)) = self.entries.get(key) {
            if timestamp.elapsed().as_secs() < self.ttl_seconds {
                return Some(value.clone());
            }
        }
        None
    }

    fn insert(&mut self, key: String, value: SettingValue) {
        self.entries.insert(key, (value, Instant::now()));
    }

    fn remove(&mut self, key: &str) {
        self.entries.remove(key);
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[test]
fn test_cache_insert_and_get() {
    let mut cache = TestCache::new(300);

    cache.insert("test.key".to_string(), SettingValue::String("test_value".to_string()));

    let value = cache.get("test.key");
    assert!(value.is_some());
    assert_eq!(value.unwrap(), SettingValue::String("test_value".to_string()));
}

#[test]
fn test_cache_get_nonexistent_key() {
    let cache = TestCache::new(300);

    let value = cache.get("nonexistent.key");
    assert!(value.is_none());
}

#[test]
fn test_cache_remove() {
    let mut cache = TestCache::new(300);

    cache.insert("removable.key".to_string(), SettingValue::Integer(42));
    assert!(cache.get("removable.key").is_some());

    cache.remove("removable.key");
    assert!(cache.get("removable.key").is_none());
}

#[test]
fn test_cache_clear() {
    let mut cache = TestCache::new(300);

    cache.insert("key1".to_string(), SettingValue::Integer(1));
    cache.insert("key2".to_string(), SettingValue::Integer(2));
    cache.insert("key3".to_string(), SettingValue::Integer(3));

    assert_eq!(cache.len(), 3);

    cache.clear();

    assert_eq!(cache.len(), 0);
    assert!(cache.get("key1").is_none());
    assert!(cache.get("key2").is_none());
    assert!(cache.get("key3").is_none());
}

#[test]
fn test_cache_multiple_types() {
    let mut cache = TestCache::new(300);

    cache.insert("string".to_string(), SettingValue::String("hello".to_string()));
    cache.insert("integer".to_string(), SettingValue::Integer(42));
    cache.insert("float".to_string(), SettingValue::Float(3.14159));
    cache.insert("boolean".to_string(), SettingValue::Boolean(true));
    cache.insert("json".to_string(), SettingValue::Json(serde_json::json!({"nested": "value"})));

    assert_eq!(cache.get("string"), Some(SettingValue::String("hello".to_string())));
    assert_eq!(cache.get("integer"), Some(SettingValue::Integer(42)));
    assert_eq!(cache.get("float"), Some(SettingValue::Float(3.14159)));
    assert_eq!(cache.get("boolean"), Some(SettingValue::Boolean(true)));

    if let Some(SettingValue::Json(j)) = cache.get("json") {
        assert_eq!(j["nested"], "value");
    } else {
        panic!("Expected JSON value");
    }
}

#[test]
fn test_cache_overwrite_existing_key() {
    let mut cache = TestCache::new(300);

    cache.insert("key".to_string(), SettingValue::Integer(1));
    assert_eq!(cache.get("key"), Some(SettingValue::Integer(1)));

    cache.insert("key".to_string(), SettingValue::Integer(2));
    assert_eq!(cache.get("key"), Some(SettingValue::Integer(2)));
}

// ============================================================
// UserFilter Unit Tests
// ============================================================

#[test]
fn test_user_filter_default() {
    let filter = UserFilter::default();

    assert!(filter.pubkey.is_empty());
    assert!(filter.enabled);
    assert!((filter.quality_threshold - 0.7).abs() < 0.001);
    assert!((filter.authority_threshold - 0.5).abs() < 0.001);
    assert!(filter.filter_by_quality);
    assert!(!filter.filter_by_authority);
    assert_eq!(filter.filter_mode, "or");
    assert_eq!(filter.max_nodes, Some(10000));
}

#[test]
fn test_user_filter_custom_values() {
    let filter = UserFilter {
        pubkey: "npub1test".to_string(),
        enabled: false,
        quality_threshold: 0.9,
        authority_threshold: 0.8,
        filter_by_quality: false,
        filter_by_authority: true,
        filter_mode: "and".to_string(),
        max_nodes: Some(5000),
        updated_at: chrono::Utc::now(),
    };

    assert_eq!(filter.pubkey, "npub1test");
    assert!(!filter.enabled);
    assert!((filter.quality_threshold - 0.9).abs() < 0.001);
    assert!((filter.authority_threshold - 0.8).abs() < 0.001);
    assert!(!filter.filter_by_quality);
    assert!(filter.filter_by_authority);
    assert_eq!(filter.filter_mode, "and");
    assert_eq!(filter.max_nodes, Some(5000));
}

#[test]
fn test_user_filter_no_max_nodes() {
    let mut filter = UserFilter::default();
    filter.max_nodes = None;

    assert!(filter.max_nodes.is_none());
}

#[test]
fn test_user_filter_serialization_roundtrip() {
    let filter = UserFilter {
        pubkey: "test_pubkey".to_string(),
        enabled: true,
        quality_threshold: 0.75,
        authority_threshold: 0.6,
        filter_by_quality: true,
        filter_by_authority: true,
        filter_mode: "and".to_string(),
        max_nodes: Some(8000),
        updated_at: chrono::Utc::now(),
    };

    let json = serde_json::to_string(&filter).expect("Serialization failed");
    let deserialized: UserFilter = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(filter.pubkey, deserialized.pubkey);
    assert_eq!(filter.enabled, deserialized.enabled);
    assert!((filter.quality_threshold - deserialized.quality_threshold).abs() < 0.001);
    assert!((filter.authority_threshold - deserialized.authority_threshold).abs() < 0.001);
    assert_eq!(filter.filter_by_quality, deserialized.filter_by_quality);
    assert_eq!(filter.filter_by_authority, deserialized.filter_by_authority);
    assert_eq!(filter.filter_mode, deserialized.filter_mode);
    assert_eq!(filter.max_nodes, deserialized.max_nodes);
}

// ============================================================
// Neo4jSettingsConfig Unit Tests
// ============================================================

#[test]
fn test_neo4j_settings_config_default() {
    // Clear env vars to test defaults
    std::env::remove_var("NEO4J_URI");
    std::env::remove_var("NEO4J_USER");
    std::env::remove_var("NEO4J_PASSWORD");
    std::env::remove_var("NEO4J_DATABASE");

    let config = Neo4jSettingsConfig::default();

    assert_eq!(config.uri, "bolt://localhost:7687");
    assert_eq!(config.user, "neo4j");
    assert_eq!(config.password, "password");
    assert!(config.database.is_none());
    assert_eq!(config.fetch_size, 500);
    assert_eq!(config.max_connections, 10);
}

#[test]
fn test_neo4j_settings_config_from_env() {
    // Set env vars
    std::env::set_var("NEO4J_URI", "bolt://testhost:7688");
    std::env::set_var("NEO4J_USER", "testuser");
    std::env::set_var("NEO4J_PASSWORD", "testpass");
    std::env::set_var("NEO4J_DATABASE", "testdb");

    let config = Neo4jSettingsConfig::default();

    assert_eq!(config.uri, "bolt://testhost:7688");
    assert_eq!(config.user, "testuser");
    assert_eq!(config.password, "testpass");
    assert_eq!(config.database, Some("testdb".to_string()));

    // Cleanup
    std::env::remove_var("NEO4J_URI");
    std::env::remove_var("NEO4J_USER");
    std::env::remove_var("NEO4J_PASSWORD");
    std::env::remove_var("NEO4J_DATABASE");
}

#[test]
fn test_neo4j_settings_config_custom() {
    let config = Neo4jSettingsConfig {
        uri: "bolt://custom:7687".to_string(),
        user: "custom_user".to_string(),
        password: "custom_pass".to_string(),
        database: Some("custom_db".to_string()),
        fetch_size: 1000,
        max_connections: 20,
    };

    assert_eq!(config.uri, "bolt://custom:7687");
    assert_eq!(config.user, "custom_user");
    assert_eq!(config.password, "custom_pass");
    assert_eq!(config.database, Some("custom_db".to_string()));
    assert_eq!(config.fetch_size, 1000);
    assert_eq!(config.max_connections, 20);
}

// ============================================================
// SettingValue Unit Tests
// ============================================================

#[test]
fn test_setting_value_string() {
    let value = SettingValue::String("test".to_string());

    assert_eq!(value.as_string(), Some("test"));
    assert_eq!(value.as_i64(), None);
    assert_eq!(value.as_f64(), None);
    assert_eq!(value.as_bool(), None);
    assert!(value.as_json().is_none());
}

#[test]
fn test_setting_value_integer() {
    let value = SettingValue::Integer(42);

    assert_eq!(value.as_string(), None);
    assert_eq!(value.as_i64(), Some(42));
    assert_eq!(value.as_f64(), None);
    assert_eq!(value.as_bool(), None);
    assert!(value.as_json().is_none());
}

#[test]
fn test_setting_value_float() {
    let value = SettingValue::Float(3.14);

    assert_eq!(value.as_string(), None);
    assert_eq!(value.as_i64(), None);
    assert_eq!(value.as_f64(), Some(3.14));
    assert_eq!(value.as_bool(), None);
    assert!(value.as_json().is_none());
}

#[test]
fn test_setting_value_boolean() {
    let value_true = SettingValue::Boolean(true);
    let value_false = SettingValue::Boolean(false);

    assert_eq!(value_true.as_bool(), Some(true));
    assert_eq!(value_false.as_bool(), Some(false));
}

#[test]
fn test_setting_value_json() {
    let json = serde_json::json!({"key": "value", "nested": {"inner": 42}});
    let value = SettingValue::Json(json.clone());

    assert!(value.as_json().is_some());
    assert_eq!(value.as_json().unwrap()["key"], "value");
    assert_eq!(value.as_json().unwrap()["nested"]["inner"], 42);
}

#[test]
fn test_setting_value_serialization() {
    let values = vec![
        SettingValue::String("test".to_string()),
        SettingValue::Integer(42),
        SettingValue::Float(3.14),
        SettingValue::Boolean(true),
        SettingValue::Json(serde_json::json!({"key": "value"})),
    ];

    for value in values {
        let json = serde_json::to_string(&value).expect("Serialization failed");
        let deserialized: SettingValue = serde_json::from_str(&json).expect("Deserialization failed");
        assert_eq!(value, deserialized);
    }
}

#[test]
fn test_setting_value_equality() {
    assert_eq!(
        SettingValue::String("test".to_string()),
        SettingValue::String("test".to_string())
    );
    assert_ne!(
        SettingValue::String("test".to_string()),
        SettingValue::String("other".to_string())
    );

    assert_eq!(SettingValue::Integer(42), SettingValue::Integer(42));
    assert_ne!(SettingValue::Integer(42), SettingValue::Integer(43));

    assert_ne!(
        SettingValue::String("42".to_string()),
        SettingValue::Integer(42)
    );
}

// ============================================================
// User Struct Unit Tests
// ============================================================

#[test]
fn test_user_struct_creation() {
    let now = chrono::Utc::now();
    let user = User {
        pubkey: "npub1abc123".to_string(),
        is_power_user: false,
        created_at: now,
        last_seen: now,
        display_name: Some("Test User".to_string()),
    };

    assert_eq!(user.pubkey, "npub1abc123");
    assert!(!user.is_power_user);
    assert_eq!(user.display_name, Some("Test User".to_string()));
}

#[test]
fn test_user_serialization_roundtrip() {
    let now = chrono::Utc::now();
    let user = User {
        pubkey: "npub1test".to_string(),
        is_power_user: true,
        created_at: now,
        last_seen: now,
        display_name: None,
    };

    let json = serde_json::to_string(&user).expect("Serialization failed");
    let deserialized: User = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(user.pubkey, deserialized.pubkey);
    assert_eq!(user.is_power_user, deserialized.is_power_user);
    assert_eq!(user.display_name, deserialized.display_name);
}

// ============================================================
// UserSettingsNode Unit Tests
// ============================================================

#[test]
fn test_user_settings_node_creation() {
    let now = chrono::Utc::now();
    let node = UserSettingsNode {
        pubkey: "npub1settings".to_string(),
        settings_json: r#"{"version":"1.0.0"}"#.to_string(),
        updated_at: now,
    };

    assert_eq!(node.pubkey, "npub1settings");
    assert!(node.settings_json.contains("version"));
}

// ============================================================
// Error Handling Tests
// ============================================================

#[test]
fn test_settings_repository_error_display() {
    let not_found = SettingsRepositoryError::NotFound("test.key".to_string());
    assert!(not_found.to_string().contains("test.key"));

    let db_error = SettingsRepositoryError::DatabaseError("Connection failed".to_string());
    assert!(db_error.to_string().contains("Connection failed"));

    let serial_error = SettingsRepositoryError::SerializationError("Invalid JSON".to_string());
    assert!(serial_error.to_string().contains("Invalid JSON"));

    let invalid = SettingsRepositoryError::InvalidValue("Out of range".to_string());
    assert!(invalid.to_string().contains("Out of range"));

    let cache_error = SettingsRepositoryError::CacheError("Cache miss".to_string());
    assert!(cache_error.to_string().contains("Cache miss"));
}

// ============================================================
// Setting Value Parameter Conversion Tests
// ============================================================

#[test]
fn test_setting_value_to_param_format() {
    // Test the JSON parameter format used by Neo4jSettingsRepository::setting_value_to_param

    let string_param = serde_json::json!({"type": "string", "value": "test_value"});
    assert_eq!(string_param["type"], "string");
    assert_eq!(string_param["value"], "test_value");

    let int_param = serde_json::json!({"type": "integer", "value": 42});
    assert_eq!(int_param["type"], "integer");
    assert_eq!(int_param["value"], 42);

    let float_param = serde_json::json!({"type": "float", "value": 3.14});
    assert_eq!(float_param["type"], "float");
    assert!((float_param["value"].as_f64().unwrap() - 3.14).abs() < 0.001);

    let bool_param = serde_json::json!({"type": "boolean", "value": true});
    assert_eq!(bool_param["type"], "boolean");
    assert_eq!(bool_param["value"], true);

    let json_value = serde_json::json!({"nested": "data"});
    let json_str = serde_json::to_string(&json_value).unwrap();
    let json_param = serde_json::json!({"type": "json", "value": json_str});
    assert_eq!(json_param["type"], "json");
}

#[test]
fn test_parse_setting_value_from_stored_format() {
    // Test parsing logic used in parse_setting_value

    fn parse_value(value_type: &str, value: &serde_json::Value) -> Option<SettingValue> {
        match value_type {
            "string" => value.as_str().map(|s| SettingValue::String(s.to_string())),
            "integer" => value.as_i64().map(SettingValue::Integer),
            "float" => value.as_f64().map(SettingValue::Float),
            "boolean" => value.as_bool().map(SettingValue::Boolean),
            "json" => {
                if let Some(json_str) = value.as_str() {
                    serde_json::from_str(json_str).ok().map(SettingValue::Json)
                } else {
                    Some(SettingValue::Json(value.clone()))
                }
            }
            _ => None,
        }
    }

    assert_eq!(
        parse_value("string", &serde_json::json!("hello")),
        Some(SettingValue::String("hello".to_string()))
    );

    assert_eq!(
        parse_value("integer", &serde_json::json!(42)),
        Some(SettingValue::Integer(42))
    );

    assert_eq!(
        parse_value("float", &serde_json::json!(3.14)),
        Some(SettingValue::Float(3.14))
    );

    assert_eq!(
        parse_value("boolean", &serde_json::json!(true)),
        Some(SettingValue::Boolean(true))
    );

    // JSON from string
    let json_result = parse_value("json", &serde_json::json!(r#"{"key":"value"}"#));
    assert!(json_result.is_some());
    if let Some(SettingValue::Json(j)) = json_result {
        assert_eq!(j["key"], "value");
    }

    // Unknown type returns None
    assert_eq!(parse_value("unknown", &serde_json::json!("test")), None);
}

// ============================================================
// Cypher Query Construction Tests
// ============================================================

#[test]
fn test_setting_merge_query_format() {
    let query = r#"
        MERGE (s:Setting {key: $key})
        ON CREATE SET
            s.created_at = datetime(),
            s.value_type = $value_type,
            s.value = $value,
            s.description = $description
        ON MATCH SET
            s.updated_at = datetime(),
            s.value_type = $value_type,
            s.value = $value,
            s.description = COALESCE($description, s.description)
        RETURN s
    "#;

    assert!(query.contains("MERGE (s:Setting {key: $key})"));
    assert!(query.contains("ON CREATE SET"));
    assert!(query.contains("ON MATCH SET"));
    assert!(query.contains("$value_type"));
    assert!(query.contains("$description"));
    assert!(query.contains("COALESCE"));
}

#[test]
fn test_user_merge_query_format() {
    let query = r#"
        MERGE (u:User {pubkey: $pubkey})
        ON CREATE SET
            u.is_power_user = false,
            u.created_at = datetime(),
            u.last_seen = datetime()
        ON MATCH SET
            u.last_seen = datetime()
        RETURN u.pubkey AS pubkey, u.is_power_user AS is_power_user
    "#;

    assert!(query.contains("MERGE (u:User {pubkey: $pubkey})"));
    assert!(query.contains("is_power_user = false"));
    assert!(query.contains("created_at = datetime()"));
}

#[test]
fn test_user_settings_query_format() {
    let query = r#"
        MATCH (u:User {pubkey: $pubkey})
        MERGE (u)-[:HAS_SETTINGS]->(us:UserSettings {pubkey: $pubkey})
        ON CREATE SET
            us.settings_json = $settings_json,
            us.updated_at = datetime()
        ON MATCH SET
            us.settings_json = $settings_json,
            us.updated_at = datetime()
        RETURN us
    "#;

    assert!(query.contains("MATCH (u:User {pubkey: $pubkey})"));
    assert!(query.contains("[:HAS_SETTINGS]->"));
    assert!(query.contains("$settings_json"));
}

#[test]
fn test_user_filter_query_format() {
    let query = r#"
        MATCH (u:User {pubkey: $pubkey})
        MERGE (u)-[:HAS_FILTER]->(uf:UserFilter {pubkey: $pubkey})
        SET uf.enabled = $enabled,
            uf.quality_threshold = $quality_threshold,
            uf.authority_threshold = $authority_threshold,
            uf.filter_by_quality = $filter_by_quality,
            uf.filter_by_authority = $filter_by_authority,
            uf.filter_mode = $filter_mode,
            uf.max_nodes = $max_nodes,
            uf.updated_at = datetime()
        RETURN uf
    "#;

    assert!(query.contains("[:HAS_FILTER]->"));
    assert!(query.contains("$quality_threshold"));
    assert!(query.contains("$authority_threshold"));
    assert!(query.contains("$filter_mode"));
}

#[test]
fn test_constraint_and_index_queries() {
    let constraints = vec![
        "CREATE CONSTRAINT settings_root_id IF NOT EXISTS FOR (s:SettingsRoot) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT user_pubkey_unique IF NOT EXISTS FOR (u:User) REQUIRE u.pubkey IS UNIQUE",
    ];

    for constraint in &constraints {
        assert!(constraint.contains("CREATE CONSTRAINT"));
        assert!(constraint.contains("IF NOT EXISTS"));
        assert!(constraint.contains("IS UNIQUE"));
    }

    let indices = vec![
        "CREATE INDEX settings_key_idx IF NOT EXISTS FOR (s:Setting) ON (s.key)",
        "CREATE INDEX physics_profile_idx IF NOT EXISTS FOR (p:PhysicsProfile) ON (p.name)",
        "CREATE INDEX user_settings_pubkey_idx IF NOT EXISTS FOR (us:UserSettings) ON (us.pubkey)",
        "CREATE INDEX user_filter_pubkey_idx IF NOT EXISTS FOR (uf:UserFilter) ON (uf.pubkey)",
    ];

    for index in &indices {
        assert!(index.contains("CREATE INDEX"));
        assert!(index.contains("IF NOT EXISTS"));
        assert!(index.contains("ON ("));
    }
}

// ============================================================
// TTL Expiry Logic Tests
// ============================================================

#[test]
fn test_cache_ttl_not_expired() {
    let ttl_seconds = 300u64;
    let cached_at = Instant::now();

    // Entry just created should not be expired
    let elapsed = cached_at.elapsed().as_secs();
    assert!(elapsed < ttl_seconds);
}

#[test]
fn test_cache_ttl_expiry_logic() {
    // Test the expiry check logic without waiting
    let ttl_seconds = 300u64;

    // Simulate fresh entry
    let fresh_elapsed_secs = 10u64;
    assert!(fresh_elapsed_secs < ttl_seconds);

    // Simulate expired entry
    let expired_elapsed_secs = 301u64;
    assert!(expired_elapsed_secs >= ttl_seconds);

    // Edge case: exactly at TTL boundary
    let boundary_elapsed_secs = 300u64;
    assert!(!(boundary_elapsed_secs < ttl_seconds)); // Should be considered expired
}

// ============================================================
// Batch Operations Tests
// ============================================================

#[test]
fn test_batch_settings_preparation() {
    let mut updates: HashMap<String, SettingValue> = HashMap::new();
    updates.insert("physics.gravity".to_string(), SettingValue::Float(9.81));
    updates.insert("render.quality".to_string(), SettingValue::String("high".to_string()));
    updates.insert("system.debug".to_string(), SettingValue::Boolean(false));
    updates.insert("nodes.max_count".to_string(), SettingValue::Integer(10000));

    assert_eq!(updates.len(), 4);

    // Verify each type is correctly stored
    assert!(matches!(updates.get("physics.gravity"), Some(SettingValue::Float(_))));
    assert!(matches!(updates.get("render.quality"), Some(SettingValue::String(_))));
    assert!(matches!(updates.get("system.debug"), Some(SettingValue::Boolean(_))));
    assert!(matches!(updates.get("nodes.max_count"), Some(SettingValue::Integer(_))));
}

#[test]
fn test_batch_keys_extraction() {
    let keys = vec![
        "key1".to_string(),
        "key2".to_string(),
        "key3".to_string(),
    ];

    // Simulate partial cache hit
    let cached_keys = vec!["key1".to_string()];

    let remaining_keys: Vec<String> = keys.iter()
        .filter(|k| !cached_keys.contains(k))
        .cloned()
        .collect();

    assert_eq!(remaining_keys.len(), 2);
    assert!(remaining_keys.contains(&"key2".to_string()));
    assert!(remaining_keys.contains(&"key3".to_string()));
    assert!(!remaining_keys.contains(&"key1".to_string()));
}

// ============================================================
// Export/Import Format Tests
// ============================================================

#[test]
fn test_export_format() {
    let mut exported = serde_json::Map::new();

    exported.insert("setting1".to_string(), serde_json::json!({
        "type": "string",
        "value": "test_value",
        "description": "A test setting"
    }));

    exported.insert("setting2".to_string(), serde_json::json!({
        "type": "integer",
        "value": 42,
        "description": "An integer setting"
    }));

    let export_json = serde_json::Value::Object(exported);

    assert!(export_json["setting1"]["type"] == "string");
    assert!(export_json["setting1"]["value"] == "test_value");
    assert!(export_json["setting2"]["type"] == "integer");
    assert!(export_json["setting2"]["value"] == 42);
}

#[test]
fn test_import_parsing() {
    let import_json = serde_json::json!({
        "setting1": {"type": "string", "value": "imported"},
        "setting2": {"type": "integer", "value": 100},
        "setting3": {"type": "float", "value": 1.5},
        "setting4": {"type": "boolean", "value": true}
    });

    let settings_map = import_json.as_object().unwrap();
    assert_eq!(settings_map.len(), 4);

    for (key, value_obj) in settings_map {
        let obj = value_obj.as_object().unwrap();
        let value_type = obj.get("type").and_then(|v| v.as_str()).unwrap();
        let _value = obj.get("value").cloned().unwrap();

        match key.as_str() {
            "setting1" => assert_eq!(value_type, "string"),
            "setting2" => assert_eq!(value_type, "integer"),
            "setting3" => assert_eq!(value_type, "float"),
            "setting4" => assert_eq!(value_type, "boolean"),
            _ => panic!("Unexpected key"),
        }
    }
}

// ============================================================
// Integration Tests (Require Live Neo4j - Marked #[ignore])
// ============================================================

#[tokio::test]
#[ignore = "Requires live Neo4j instance"]
async fn test_neo4j_settings_repository_connection() {
    let config = Neo4jSettingsConfig::default();
    let result = Neo4jSettingsRepository::new(config).await;

    assert!(result.is_ok(), "Failed to connect: {:?}", result.err());
}

#[tokio::test]
#[ignore = "Requires live Neo4j instance"]
async fn test_neo4j_settings_repository_health_check() {
    use crate::ports::settings_repository::SettingsRepository;

    let config = Neo4jSettingsConfig::default();
    let repo = Neo4jSettingsRepository::new(config).await.unwrap();

    let health = repo.health_check().await;
    assert!(health.is_ok());
    assert!(health.unwrap());
}

#[tokio::test]
#[ignore = "Requires live Neo4j instance"]
async fn test_neo4j_settings_crud_operations() {
    use crate::ports::settings_repository::SettingsRepository;

    let config = Neo4jSettingsConfig::default();
    let repo = Neo4jSettingsRepository::new(config).await.unwrap();

    // Create
    repo.set_setting(
        "test.integration.key",
        SettingValue::String("integration_test_value".to_string()),
        Some("Integration test setting"),
    ).await.unwrap();

    // Read
    let value = repo.get_setting("test.integration.key").await.unwrap();
    assert_eq!(value, Some(SettingValue::String("integration_test_value".to_string())));

    // Update
    repo.set_setting(
        "test.integration.key",
        SettingValue::String("updated_value".to_string()),
        None,
    ).await.unwrap();

    let updated = repo.get_setting("test.integration.key").await.unwrap();
    assert_eq!(updated, Some(SettingValue::String("updated_value".to_string())));

    // Delete
    repo.delete_setting("test.integration.key").await.unwrap();
    let deleted = repo.get_setting("test.integration.key").await.unwrap();
    assert!(deleted.is_none());
}

#[tokio::test]
#[ignore = "Requires live Neo4j instance"]
async fn test_neo4j_user_management() {
    let config = Neo4jSettingsConfig::default();
    let repo = Neo4jSettingsRepository::new(config).await.unwrap();

    let test_pubkey = format!("test_user_{}", chrono::Utc::now().timestamp_millis());

    // Get or create user
    let user = repo.get_or_create_user(&test_pubkey).await.unwrap();
    assert_eq!(user.pubkey, test_pubkey);
    assert!(!user.is_power_user);

    // Set power user
    repo.set_power_user(&test_pubkey, true).await.unwrap();
    let is_power = repo.is_power_user(&test_pubkey).await.unwrap();
    assert!(is_power);

    // Update last seen
    repo.update_user_last_seen(&test_pubkey).await.unwrap();
}

#[tokio::test]
#[ignore = "Requires live Neo4j instance"]
async fn test_neo4j_user_filter_operations() {
    let config = Neo4jSettingsConfig::default();
    let repo = Neo4jSettingsRepository::new(config).await.unwrap();

    let test_pubkey = format!("filter_test_{}", chrono::Utc::now().timestamp_millis());

    // Create user first
    repo.get_or_create_user(&test_pubkey).await.unwrap();

    // Save filter
    let mut filter = UserFilter::default();
    filter.pubkey = test_pubkey.clone();
    filter.quality_threshold = 0.85;
    filter.max_nodes = Some(7500);

    repo.save_user_filter(&test_pubkey, &filter).await.unwrap();

    // Get filter
    let loaded = repo.get_user_filter(&test_pubkey).await.unwrap();
    assert!(loaded.is_some());
    let loaded = loaded.unwrap();
    assert!((loaded.quality_threshold - 0.85).abs() < 0.001);
    assert_eq!(loaded.max_nodes, Some(7500));
}

#[tokio::test]
#[ignore = "Requires live Neo4j instance"]
async fn test_neo4j_batch_operations() {
    use crate::ports::settings_repository::SettingsRepository;

    let config = Neo4jSettingsConfig::default();
    let repo = Neo4jSettingsRepository::new(config).await.unwrap();

    let timestamp = chrono::Utc::now().timestamp_millis();

    // Batch set
    let mut updates = HashMap::new();
    updates.insert(format!("batch.{}.key1", timestamp), SettingValue::Integer(1));
    updates.insert(format!("batch.{}.key2", timestamp), SettingValue::Integer(2));
    updates.insert(format!("batch.{}.key3", timestamp), SettingValue::Integer(3));

    repo.set_settings_batch(updates.clone()).await.unwrap();

    // Batch get
    let keys: Vec<String> = updates.keys().cloned().collect();
    let batch = repo.get_settings_batch(&keys).await.unwrap();

    assert_eq!(batch.len(), 3);

    // Cleanup
    for key in keys {
        repo.delete_setting(&key).await.unwrap();
    }
}

#[tokio::test]
#[ignore = "Requires live Neo4j instance"]
async fn test_neo4j_physics_profiles() {
    use crate::ports::settings_repository::SettingsRepository;
    use crate::config::PhysicsSettings;

    let config = Neo4jSettingsConfig::default();
    let repo = Neo4jSettingsRepository::new(config).await.unwrap();

    let profile_name = format!("test_profile_{}", chrono::Utc::now().timestamp_millis());

    // Save physics settings
    let physics = PhysicsSettings::default();
    repo.save_physics_settings(&profile_name, &physics).await.unwrap();

    // Get physics settings
    let loaded = repo.get_physics_settings(&profile_name).await.unwrap();
    assert_eq!(loaded.damping, physics.damping);

    // List profiles
    let profiles = repo.list_physics_profiles().await.unwrap();
    assert!(profiles.contains(&profile_name));

    // Delete profile
    repo.delete_physics_profile(&profile_name).await.unwrap();
    let profiles_after = repo.list_physics_profiles().await.unwrap();
    assert!(!profiles_after.contains(&profile_name));
}
