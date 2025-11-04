# SettingsRepository Port

> ⚠️ **MIGRATION NOTICE (November 2025)**
> This document has been updated to reflect the completed migration from SQLite to Neo4j for settings storage.
> Production code now uses `Neo4jSettingsRepository`. See `/docs/guides/neo4j-migration.md` for migration details.

## Purpose

The **SettingsRepository** port provides a unified interface for managing all application, user, and developer configuration settings. It abstracts persistence operations for settings stored in a graph database.

## Location

- **Trait Definition**: `src/ports/settings_repository.rs`
- **Current Adapter**: `src/adapters/neo4j_settings_repository.rs` ✅ **ACTIVE**
- **Legacy Adapter**: `src/adapters/sqlite_settings_repository.rs` ❌ **DEPRECATED**

## Interface

```rust
#[async_trait]
pub trait SettingsRepository: Send + Sync {
    // Single setting operations
    async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>>;
    async fn set_setting(&self, key: &str, value: SettingValue, description: Option<&str>) -> Result<()>;
    async fn delete_setting(&self, key: &str) -> Result<()>;
    async fn has_setting(&self, key: &str) -> Result<bool>;

    // Batch operations
    async fn get_settings_batch(&self, keys: &[String]) -> Result<HashMap<String, SettingValue>>;
    async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> Result<()>;
    async fn list_settings(&self, prefix: Option<&str>) -> Result<Vec<String>>;

    // Application settings
    async fn load_all_settings(&self) -> Result<Option<AppFullSettings>>;
    async fn save_all_settings(&self, settings: &AppFullSettings) -> Result<()>;

    // Physics profiles
    async fn get_physics_settings(&self, profile_name: &str) -> Result<PhysicsSettings>;
    async fn save_physics_settings(&self, profile_name: &str, settings: &PhysicsSettings) -> Result<()>;
    async fn list_physics_profiles(&self) -> Result<Vec<String>>;
    async fn delete_physics_profile(&self, profile_name: &str) -> Result<()>;

    // Import/Export
    async fn export_settings(&self) -> Result<serde_json::Value>;
    async fn import_settings(&self, settings_json: &serde_json::Value) -> Result<()>;

    // Maintenance
    async fn clear_cache(&self) -> Result<()>;
    async fn health_check(&self) -> Result<bool>;
}
```

## Types

### SettingValue

A polymorphic value type supporting multiple data types:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(serde_json::Value),
}
```

**Helper Methods**:
- `as_string() -> Option<&str>`
- `as_i64() -> Option<i64>`
- `as_f64() -> Option<f64>`
- `as_bool() -> Option<bool>`
- `as_json() -> Option<&serde_json::Value>`

### AppFullSettings

Complete application settings structure:

```rust
pub struct AppFullSettings {
    pub log_level: String,
    pub graph_directory: String,
    pub ontology_directory: String,
    pub physics_settings: PhysicsSettings,
    // ... additional fields
}
```

### PhysicsSettings

Physics simulation configuration:

```rust
pub struct PhysicsSettings {
    pub time_step: f32,
    pub damping: f32,
    pub spring_strength: f32,
    pub repulsion_strength: f32,
    pub max_velocity: f32,
    // ... additional physics parameters
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum SettingsRepositoryError {
    #[error("Setting not found: {0}")]
    NotFound(String),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Cache error: {0}")]
    CacheError(String),
}
```

## Usage Examples

### Basic Setting Operations

```rust
// Initialize Neo4j settings repository
let settings_config = Neo4jSettingsConfig {
    uri: std::env::var("NEO4J_URI").unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
    user: std::env::var("NEO4J_USER").unwrap_or_else(|_| "neo4j".to_string()),
    password: std::env::var("NEO4J_PASSWORD").unwrap_or_else(|_| "password".to_string()),
    database: std::env::var("NEO4J_DATABASE").ok(),
    fetch_size: 500,
    max_connections: 10,
};

let repo: Arc<dyn SettingsRepository> = Arc::new(
    Neo4jSettingsRepository::new(settings_config).await?
);

// Set a setting
repo.set_setting(
    "log_level",
    SettingValue::String("debug".into()),
    Some("Application log level")
).await?;

// Get a setting
if let Some(SettingValue::String(level)) = repo.get_setting("log_level").await? {
    println!("Log level: {}", level);
}

// Check existence
if repo.has_setting("log_level").await? {
    println!("Setting exists!");
}
```

### Batch Operations

```rust
// Get multiple settings at once
let keys = vec!["log_level".to_string(), "graph_directory".to_string()];
let settings = repo.get_settings_batch(&keys).await?;

// Set multiple settings atomically
let mut updates = HashMap::new();
updates.insert("log_level".to_string(), SettingValue::String("info".into()));
updates.insert("max_nodes".to_string(), SettingValue::Integer(1000));
repo.set_settings_batch(updates).await?;
```

### Physics Profiles

```rust
// Save a physics profile
let physics_settings = PhysicsSettings {
    time_step: 0.016,
    damping: 0.8,
    spring_strength: 0.01,
    repulsion_strength: 100.0,
    max_velocity: 10.0,
};

repo.save_physics_settings("logseq_layout", &physics_settings).await?;

// Load a physics profile
let settings = repo.get_physics_settings("logseq_layout").await?;

// List all profiles
let profiles = repo.list_physics_profiles().await?;
for profile in profiles {
    println!("Profile: {}", profile);
}

// Delete a profile
repo.delete_physics_profile("old_profile").await?;
```

### Application Settings

```rust
// Load complete application settings
if let Some(app_settings) = repo.load_all_settings().await? {
    println!("Log level: {}", app_settings.log_level);
    println!("Graph dir: {}", app_settings.graph_directory);
}

// Save complete application settings
let app_settings = AppFullSettings {
    log_level: "info".to_string(),
    graph_directory: "./data/graphs".to_string(),
    ontology_directory: "./data/ontology".to_string(),
    physics_settings: PhysicsSettings::default(),
};

repo.save_all_settings(&app_settings).await?;
```

### Import/Export

```rust
// Export all settings to JSON
let settings_json = repo.export_settings().await?;
std::fs::write("settings_backup.json", serde_json::to_string_pretty(&settings_json)?)?;

// Import settings from JSON
let json_str = std::fs::read_to_string("settings_backup.json")?;
let settings_json: serde_json::Value = serde_json::from_str(&json_str)?;
repo.import_settings(&settings_json).await?;
```

## Implementation Notes

### Neo4j Schema Design

Settings are stored as `:Setting` nodes in Neo4j with the following structure:

```cypher
// Settings Root Node (singleton)
CREATE (r:SettingsRoot {id: 'default', version: '1.0.0', created_at: datetime()})

// Individual Setting Nodes
CREATE (s:Setting {
  key: 'visualisation.theme',
  value_type: 'string',
  value: 'dark',
  description: 'UI theme setting',
  created_at: datetime(),
  updated_at: datetime()
})

// Indices for performance
CREATE INDEX settings_key_idx IF NOT EXISTS FOR (s:Setting) ON (s.key)
CREATE CONSTRAINT settings_root_id IF NOT EXISTS FOR (s:SettingsRoot) REQUIRE s.id IS UNIQUE
```

### Caching Strategy

The Neo4j adapter implements LRU caching with TTL for frequently accessed settings:

```rust
pub struct Neo4jSettingsRepository {
    graph: Arc<Graph>,
    cache: Arc<RwLock<SettingsCache>>,
    config: Neo4jSettingsConfig,
}

struct SettingsCache {
    settings: HashMap<String, CachedSetting>,
    last_updated: Instant,
    ttl_seconds: u64,  // Default: 300 seconds (5 minutes)
}

struct CachedSetting {
    value: SettingValue,
    timestamp: Instant,
}
```

**Cache Invalidation**:
- Call `clear_cache()` after batch updates
- TTL-based expiration (default: 5 minutes)
- Write-through caching for `set_setting`
- Cache hit provides ~90x speedup for repeated reads

### Transaction Support

Batch operations use Neo4j transactions for atomicity:

```rust
async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> Result<()> {
    // Start Neo4j transaction
    let mut txn = self.graph.start_txn().await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Failed to start transaction: {}", e)))?;

    for (key, value) in &updates {
        let value_param = self.setting_value_to_param(value);
        let query_str = "MERGE (s:Setting {key: $key})
                         ON CREATE SET s.created_at = datetime(), s.value = $value, s.value_type = $value_type
                         ON MATCH SET s.updated_at = datetime(), s.value = $value, s.value_type = $value_type";

        txn.run_queries(vec![
            query(query_str)
                .param("key", key.as_str())
                .param("value", json_to_bolt(value_param["value"].clone()))
                .param("value_type", value_param["type"].as_str().unwrap())
        ]).await.map_err(|e| SettingsRepositoryError::DatabaseError(format!("Failed to execute batch update: {}", e)))?;
    }

    txn.commit().await
        .map_err(|e| SettingsRepositoryError::DatabaseError(format!("Failed to commit transaction: {}", e)))?;

    // Clear cache after batch update
    self.clear_cache_internal().await?;
    Ok(())
}
```

### Key Normalization

Support both camelCase and snake_case keys:

```rust
fn normalize_key(key: &str) -> String {
    // Convert camelCase to snake_case
    // "logLevel" -> "log_level"
}
```

## Neo4j Graph Schema

Settings are stored as nodes in Neo4j with the following structure:

```cypher
// Schema initialization (automatic on repository creation)
// Constraints
CREATE CONSTRAINT settings_root_id IF NOT EXISTS
  FOR (s:SettingsRoot) REQUIRE s.id IS UNIQUE;

// Indices for performance
CREATE INDEX settings_key_idx IF NOT EXISTS
  FOR (s:Setting) ON (s.key);

CREATE INDEX physics_profile_idx IF NOT EXISTS
  FOR (p:PhysicsProfile) ON (p.name);

// Node structure examples
(:SettingsRoot {
  id: 'default',
  version: '1.0.0',
  created_at: datetime(),
  updated_at: datetime()
})

(:Setting {
  key: 'visualisation.theme',
  value_type: 'string',  // 'string', 'integer', 'float', 'boolean', 'json'
  value: 'dark',
  description: 'UI theme setting',
  created_at: datetime(),
  updated_at: datetime()
})

(:PhysicsProfile {
  name: 'logseq_layout',
  settings: '{"time_step": 0.016, "damping": 0.8, ...}',
  created_at: datetime(),
  updated_at: datetime()
})
```

### Legacy SQLite Schema (DEPRECATED)

The previous SQLite implementation used this schema (no longer active):

```sql
-- DEPRECATED: This schema is no longer used
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## Testing

### Mock Implementation

```rust
pub struct MockSettingsRepository {
    data: Arc<RwLock<HashMap<String, SettingValue>>>,
}

#[async_trait]
impl SettingsRepository for MockSettingsRepository {
    async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>> {
        Ok(self.data.read().await.get(key).cloned())
    }

    async fn set_setting(&self, key: &str, value: SettingValue, _: Option<&str>) -> Result<()> {
        self.data.write().await.insert(key.to_string(), value);
        Ok(())
    }

    // ... implement remaining methods
}
```

### Contract Tests

```rust
#[tokio::test]
async fn test_settings_repository_contract() {
    let repo = MockSettingsRepository::new();

    // Test set/get
    repo.set_setting("key", SettingValue::String("value".into()), None).await.unwrap();
    assert_eq!(
        repo.get_setting("key").await.unwrap(),
        Some(SettingValue::String("value".into()))
    );

    // Test batch operations
    let mut updates = HashMap::new();
    updates.insert("k1".to_string(), SettingValue::Integer(42));
    updates.insert("k2".to_string(), SettingValue::Boolean(true));
    repo.set_settings_batch(updates).await.unwrap();

    let keys = vec!["k1".to_string(), "k2".to_string()];
    let batch = repo.get_settings_batch(&keys).await.unwrap();
    assert_eq!(batch.len(), 2);
}
```

## Performance Considerations

### Optimization Strategies

1. **Batch Operations**: Use `get_settings_batch` and `set_settings_batch` for multiple settings
2. **Caching**: Implement LRU cache with TTL for frequently accessed settings
3. **Connection Pooling**: Use r2d2 connection pool for SQLite
4. **Prepared Statements**: Reuse compiled SQL statements

### Benchmarks

Target performance (Neo4j adapter):
- Single get: < 0.1ms (cached), < 3ms (uncached with network latency)
- Batch get (10 items): < 8ms
- Single set: < 4ms
- Batch set (10 items): < 15ms (within transaction)
- Cache hit rate: > 85% for frequently accessed settings
- Network latency overhead: ~1-2ms for local Neo4j instance

**Performance Notes**:
- Cache provides ~90x speedup for repeated reads
- Connection pooling (default: 10 connections) optimizes concurrent access
- Batch operations use transactions for atomicity without sacrificing speed

## Security Considerations

1. **Secrets Management**: Never store sensitive data (API keys, passwords) as plain SettingValue
2. **Encryption**: Use `SettingValue::Json` with encrypted payloads for sensitive settings
3. **Validation**: Validate setting values before storage
4. **Access Control**: Implement role-based access if exposing settings via API

## Migration Guide

### From SQLite to Neo4j

**Before (SQLite - DEPRECATED)**:
```rust
let conn = pool.get()?;
let value: String = conn.query_row(
    "SELECT value FROM settings WHERE key = ?",
    params![key],
    |row| row.get(0)
)?;
```

**After (Neo4j - CURRENT)**:
```rust
// Configure Neo4j connection
let settings_config = Neo4jSettingsConfig::default();
let repo: Arc<dyn SettingsRepository> = Arc::new(
    Neo4jSettingsRepository::new(settings_config).await?
);

// Query settings using port interface
let value = repo.get_setting(key).await?
    .and_then(|v| v.as_string().map(|s| s.to_string()));
```

### Migration Path

For projects migrating from SQLite to Neo4j:

1. **Install Neo4j**: Docker or native installation
2. **Configure environment variables**:
   ```bash
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-secure-password
   ```
3. **Run migration tool**:
   ```bash
   cargo run --features neo4j --bin migrate_settings_to_neo4j
   ```
4. **Update application code**: Replace `SqliteSettingsRepository` with `Neo4jSettingsRepository`
5. **Verify migration**: Check Neo4j Browser for migrated settings

See `/docs/guides/neo4j-migration.md` for detailed migration instructions.

## References

- **Repository Pattern**: https://martinfowler.com/eaaCatalog/repository.html
- **Settings Management**: https://12factor.net/config
- **Caching Strategies**: https://aws.amazon.com/caching/best-practices/

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
