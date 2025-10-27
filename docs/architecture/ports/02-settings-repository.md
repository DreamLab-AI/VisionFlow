# SettingsRepository Port

## Purpose

The **SettingsRepository** port provides a unified interface for managing all application, user, and developer configuration settings. It abstracts persistence operations for settings stored in the `settings` table.

## Location

- **Trait Definition**: `src/ports/settings_repository.rs`
- **Adapter Implementation**: `src/adapters/sqlite_settings_repository.rs`

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
let repo: Arc<dyn SettingsRepository> = Arc::new(SqliteSettingsAdapter::new(pool));

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

### Caching Strategy

The adapter should implement caching for frequently accessed settings:

```rust
pub struct SqliteSettingsAdapter {
    pool: Arc<SqlitePool>,
    cache: Arc<RwLock<HashMap<String, CachedSetting>>>,
}

struct CachedSetting {
    value: SettingValue,
    expires_at: Instant,
}
```

**Cache Invalidation**:
- Call `clear_cache()` after batch updates
- TTL-based expiration (e.g., 5 minutes)
- Write-through caching for `set_setting`

### Transaction Support

Batch operations should use database transactions:

```rust
async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> Result<()> {
    let mut conn = self.pool.get().await?;
    let tx = conn.transaction()?;

    for (key, value) in updates {
        // Insert/update within transaction
    }

    tx.commit()?;
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

## Database Schema

```sql
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    value_type TEXT NOT NULL, -- 'string', 'integer', 'float', 'boolean', 'json'
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_settings_type ON settings(value_type);
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

Target performance (SQLite adapter):
- Single get: < 1ms (cached), < 5ms (uncached)
- Batch get (10 items): < 10ms
- Single set: < 5ms
- Batch set (10 items): < 20ms

## Security Considerations

1. **Secrets Management**: Never store sensitive data (API keys, passwords) as plain SettingValue
2. **Encryption**: Use `SettingValue::Json` with encrypted payloads for sensitive settings
3. **Validation**: Validate setting values before storage
4. **Access Control**: Implement role-based access if exposing settings via API

## Migration Guide

### From Direct Database Access

**Before**:
```rust
let conn = pool.get()?;
let value: String = conn.query_row(
    "SELECT value FROM settings WHERE key = ?",
    params![key],
    |row| row.get(0)
)?;
```

**After**:
```rust
let repo: Arc<dyn SettingsRepository> = Arc::new(SqliteSettingsAdapter::new(pool));
let value = repo.get_setting(key).await?
    .and_then(|v| v.as_string().map(|s| s.to_string()));
```

## References

- **Repository Pattern**: https://martinfowler.com/eaaCatalog/repository.html
- **Settings Management**: https://12factor.net/config
- **Caching Strategies**: https://aws.amazon.com/caching/best-practices/

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
