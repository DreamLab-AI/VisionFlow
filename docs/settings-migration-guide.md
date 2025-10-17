# Settings Migration Guide for Developers

## Overview

This guide explains how to work with the new SQLite-backed settings system, how to query settings in your code, add new settings, and understand the dual key format migration.

## For Application Developers

### Querying Settings in Rust Code

#### 1. Get Global Settings

```rust
use crate::app_state::AppState;
use actix_web::web;

async fn my_handler(app_state: web::Data<AppState>) -> Result<HttpResponse> {
    // Get current global settings
    let settings = app_state.settings.lock().await.clone();

    // Access nested settings
    let ambient_light = settings.visualisation.rendering.ambient_light_intensity;
    let spring_k = settings.visualisation.graphs.logseq.physics.spring_k;

    // Use settings...
    Ok(HttpResponse::Ok().json(json!({
        "ambientLight": ambient_light,
        "springK": spring_k
    })))
}
```

#### 2. Get User-Specific Settings

```rust
use crate::models::user_settings::UserSettings;

async fn get_user_settings(pubkey: &str) -> Result<AppFullSettings> {
    // Load user settings (automatically merges with global defaults)
    let user_settings = UserSettings::load(pubkey)
        .ok_or_else(|| Error::UserNotFound)?;

    Ok(user_settings.settings)
}
```

#### 3. Update Settings via Actor

```rust
use crate::actors::messages::{UpdateSettings, GetSettings};
use actix::Addr;

async fn update_physics(
    settings_actor: Addr<OptimizedSettingsActor>,
    new_spring_k: f32,
) -> Result<()> {
    // Get current settings
    let current = settings_actor.send(GetSettings).await??;

    // Modify settings
    let mut updated = current.clone();
    updated.visualisation.graphs.logseq.physics.spring_k = new_spring_k;

    // Send update
    settings_actor.send(UpdateSettings {
        settings: updated,
        user_id: None  // Global update
    }).await??;

    Ok(())
}
```

### Working with Dual Key Format

The system supports both `camelCase` (frontend) and `snake_case` (backend):

#### Example: Field Name Mapping

| Frontend (camelCase)          | Backend (snake_case)           |
|-------------------------------|--------------------------------|
| `ambientLightIntensity`       | `ambient_light_intensity`      |
| `enableShadows`               | `enable_shadows`               |
| `nodeSize`                    | `node_size`                    |
| `springK`                     | `spring_k`                     |
| `repelK`                      | `repel_k`                      |
| `equilibriumCheckFrames`      | `equilibrium_check_frames`     |

#### Automatic Conversion

The API layer handles conversion automatically:

```rust
// Frontend sends:
{
  "ambientLightIntensity": 0.7
}

// Backend receives (after conversion):
AppFullSettings {
    visualisation: VisualisationSettings {
        rendering: RenderingSettings {
            ambient_light_intensity: 0.7,
            ...
        }
    }
}

// Backend sends:
AppFullSettings { ambient_light_intensity: 0.7 }

// Frontend receives (after conversion):
{
  "ambientLightIntensity": 0.7
}
```

#### Manual Conversion (if needed)

```rust
use crate::config::normalize_field_names_to_camel_case;
use serde_json::Value;

// Convert snake_case to camelCase
let snake_case_json = json!({
    "ambient_light_intensity": 0.5,
    "enable_shadows": true
});

let camel_case_json = normalize_field_names_to_camel_case(snake_case_json)?;
// Result: { "ambientLightIntensity": 0.5, "enableShadows": true }
```

## Adding New Settings

### Step 1: Add Field to Config Struct

**File:** `src/config/mod.rs`

```rust
#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct MyNewSettings {
    #[validate(range(min = 0.0, max = 10.0))]
    #[serde(alias = "my_new_parameter")]
    pub my_new_parameter: f32,

    #[validate(custom(function = "validate_hex_color"))]
    #[serde(alias = "my_color")]
    pub my_color: String,
}
```

**Important:**
- Use `#[serde(rename_all = "camelCase")]` on the struct
- Add `#[serde(alias = "snake_case_name")]` for each field
- Add validation attributes
- Implement `Type` for TypeScript generation

### Step 2: Add Field to Parent Settings

```rust
#[derive(Debug, Serialize, Deserialize, Clone, Type)]
#[serde(rename_all = "camelCase")]
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings,
    pub system: SystemSettings,
    pub my_new_settings: MyNewSettings,  // Add here
    // ...
}
```

### Step 3: Add Default Implementation

```rust
impl Default for MyNewSettings {
    fn default() -> Self {
        Self {
            my_new_parameter: 5.0,
            my_color: "#ff0000".to_string(),
        }
    }
}
```

### Step 4: Add Validation Rules

**File:** `src/handlers/settings_validation_fix.rs`

```rust
pub fn validate_my_new_settings(settings: &MyNewSettings) -> Result<(), ValidationError> {
    if settings.my_new_parameter < 0.0 || settings.my_new_parameter > 10.0 {
        return Err(ValidationError::new("my_new_parameter_out_of_range"));
    }

    validate_hex_color(&settings.my_color)?;

    Ok(())
}
```

### Step 5: Add Field Mapping

**File:** `src/config/mod.rs` - Update `FIELD_MAPPINGS`

```rust
static FIELD_MAPPINGS: std::sync::LazyLock<std::collections::HashMap<&'static str, &'static str>> =
    std::sync::LazyLock::new(|| {
        let mut field_mappings = std::collections::HashMap::new();

        // Add your mappings
        field_mappings.insert("my_new_parameter", "myNewParameter");
        field_mappings.insert("my_color", "myColor");

        // ... existing mappings

        field_mappings
    });
```

### Step 6: Update TypeScript Types

```bash
cargo run --bin generate_types
```

This generates updated TypeScript types in `client/src/types/settings.ts`.

### Step 7: Add Migration (if needed)

If adding settings that need initialization from existing data:

**File:** `src/db/migrations/add_my_settings.sql`

```sql
-- Add new column to settings table
ALTER TABLE settings ADD COLUMN my_new_parameter REAL DEFAULT 5.0;
ALTER TABLE settings ADD COLUMN my_color TEXT DEFAULT '#ff0000';

-- Add to user_settings table
ALTER TABLE user_settings ADD COLUMN my_new_parameter REAL;
ALTER TABLE user_settings ADD COLUMN my_color TEXT;
```

### Step 8: Document Your Settings

Add documentation to:
- `docs/settings-validation.md` - Validation rules
- `docs/settings-api.md` - API examples
- `docs/settings-system.md` - Architecture notes

## Database Queries for Settings

### Direct SQL Queries (Advanced)

If you need to query settings directly from SQLite:

```rust
use rusqlite::{Connection, params};

fn get_setting_value(conn: &Connection, key: &str) -> Result<String> {
    let value: String = conn.query_row(
        "SELECT value FROM settings WHERE key = ?1",
        params![key],
        |row| row.get(0)
    )?;
    Ok(value)
}

fn get_user_setting_override(
    conn: &Connection,
    user_id: &str,
    key: &str
) -> Result<Option<String>> {
    let value: Option<String> = conn.query_row(
        "SELECT value FROM user_settings WHERE user_id = ?1 AND key = ?2",
        params![user_id, key],
        |row| row.get(0)
    ).optional()?;
    Ok(value)
}
```

### Using Connection Pool

```rust
use crate::db::get_connection_pool;

async fn query_with_pool() -> Result<()> {
    let pool = get_connection_pool().await?;
    let conn = pool.get().await?;

    let settings: Vec<(String, String)> = conn.prepare(
        "SELECT key, value FROM settings WHERE category = ?1"
    )?
    .query_map(params!["visualisation"], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?
    .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}
```

## User-Specific Override Behavior

### How Overrides Work

1. **Global Default:** Set in `settings` table
2. **User Override:** Set in `user_settings` table for specific user
3. **Resolution:** User override takes precedence if exists, else global default

### Setting User Override

```rust
use crate::actors::messages::UpdateSettings;

async fn set_user_preference(
    settings_actor: Addr<OptimizedSettingsActor>,
    user_pubkey: String,
    spring_k: f32,
) -> Result<()> {
    // Load user's current settings
    let mut user_settings = UserSettings::load(&user_pubkey)
        .unwrap_or_else(|| {
            // Create new user settings from global defaults
            let global = get_global_settings().await?;
            UserSettings::new(&user_pubkey, global)
        });

    // Modify specific field
    user_settings.settings.visualisation.graphs.logseq.physics.spring_k = spring_k;

    // Save user settings
    user_settings.save()?;

    // Notify actor
    settings_actor.send(UpdateSettings {
        settings: user_settings.settings,
        user_id: Some(user_pubkey),
    }).await??;

    Ok(())
}
```

### Removing User Override

```rust
async fn remove_user_override(
    conn: &Connection,
    user_id: &str,
    key: &str
) -> Result<()> {
    conn.execute(
        "DELETE FROM user_settings WHERE user_id = ?1 AND key = ?2",
        params![user_id, key]
    )?;
    Ok(())
}
```

## Migration from YAML to SQLite

### Automatic Migration

The system performs automatic migration on first startup:

```rust
// In main.rs or initialization code
pub async fn initialize_settings(db_path: &Path) -> Result<AppState> {
    // Check if database exists
    let db_exists = db_path.exists();

    if !db_exists {
        // Initialize database schema
        initialize_database(db_path).await?;

        // Check for legacy YAML settings
        if let Ok(yaml_content) = fs::read_to_string("/app/settings.yaml") {
            info!("Found legacy settings.yaml, migrating to SQLite");

            // Parse YAML
            let yaml_settings: AppFullSettings = serde_yaml::from_str(&yaml_content)?;

            // Import to SQLite
            import_settings_to_db(&yaml_settings, db_path).await?;

            info!("Migration complete");
        }
    }

    Ok(app_state)
}
```

### Manual Migration

If you need to manually migrate settings:

```bash
# Export current SQLite settings to JSON
sqlite3 /app/data/settings.db \
  "SELECT json_group_object(key, value) FROM settings;" \
  > settings_export.json

# Import JSON to new database
cargo run --bin import_settings -- \
  --input settings_export.json \
  --database /app/data/new_settings.db
```

## Permission Checks

### Checking Power User Permission

```rust
use crate::models::user::User;

fn require_power_user(user: &User) -> Result<()> {
    if !user.is_power_user {
        return Err(Error::PermissionDenied(
            "Power user permission required".to_string()
        ));
    }
    Ok(())
}
```

### Granting Power User Permission

```rust
async fn grant_power_user(
    conn: &Connection,
    user_pubkey: &str,
    granted_by: &str,
) -> Result<()> {
    conn.execute(
        "UPDATE users SET is_power_user = TRUE WHERE pubkey = ?1",
        params![user_pubkey]
    )?;

    // Log to audit
    conn.execute(
        "INSERT INTO settings_audit_log (user_id, action, details, timestamp)
         VALUES (?1, 'permission_granted', ?2, CURRENT_TIMESTAMP)",
        params![
            user_pubkey,
            format!("Power user granted by {}", granted_by)
        ]
    )?;

    Ok(())
}
```

## Best Practices

### 1. Always Validate Before Persisting

```rust
use crate::handlers::validation_handler::ValidationService;

async fn update_settings_safely(new_settings: AppFullSettings) -> Result<()> {
    // Validate first
    ValidationService::validate_settings(&new_settings)?;

    // Then persist
    save_settings(&new_settings).await?;

    Ok(())
}
```

### 2. Use Transactions for Related Updates

```rust
async fn update_multiple_settings(conn: &Connection) -> Result<()> {
    let tx = conn.transaction()?;

    tx.execute("UPDATE settings SET value = ?1 WHERE key = 'setting1'", params![val1])?;
    tx.execute("UPDATE settings SET value = ?1 WHERE key = 'setting2'", params![val2])?;
    tx.execute("UPDATE settings SET value = ?1 WHERE key = 'setting3'", params![val3])?;

    tx.commit()?;
    Ok(())
}
```

### 3. Cache Frequently Accessed Settings

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

struct SettingsCache {
    cached_settings: Arc<RwLock<AppFullSettings>>,
    last_updated: Arc<RwLock<Instant>>,
}

impl SettingsCache {
    async fn get(&self) -> AppFullSettings {
        let elapsed = self.last_updated.read().await.elapsed();

        if elapsed > Duration::from_secs(60) {
            // Reload from database
            let fresh = load_settings_from_db().await.unwrap();
            *self.cached_settings.write().await = fresh.clone();
            *self.last_updated.write().await = Instant::now();
            fresh
        } else {
            self.cached_settings.read().await.clone()
        }
    }
}
```

### 4. Audit Important Changes

```rust
async fn audit_setting_change(
    conn: &Connection,
    user_id: &str,
    key: &str,
    old_value: &str,
    new_value: &str,
) -> Result<()> {
    conn.execute(
        "INSERT INTO settings_audit_log
         (user_id, setting_key, old_value, new_value, timestamp)
         VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)",
        params![user_id, key, old_value, new_value]
    )?;
    Ok(())
}
```

## Troubleshooting

### Settings Not Updating

**Problem:** Settings changes don't reflect in application.

**Solutions:**
1. Check WebSocket connection for live updates
2. Verify settings actor is receiving messages
3. Clear settings cache: `SettingsCache::invalidate()`
4. Check database locks: `PRAGMA busy_timeout = 5000;`

### Validation Errors

**Problem:** Settings update fails with validation error.

**Solutions:**
1. Use `/api/settings/validate` endpoint to test first
2. Check validation rules in `settings-validation.md`
3. Verify field types match expected types
4. Check for NaN/Infinity in numeric fields

### Database Locked

**Problem:** `database is locked` error.

**Solutions:**
1. Use connection pool with proper timeout
2. Reduce transaction duration
3. Retry with exponential backoff:

```rust
async fn retry_with_backoff<F, T>(mut f: F, max_retries: u32) -> Result<T>
where
    F: FnMut() -> Result<T>,
{
    let mut retries = 0;
    loop {
        match f() {
            Ok(result) => return Ok(result),
            Err(e) if retries < max_retries => {
                retries += 1;
                let delay = Duration::from_millis(100 * 2u64.pow(retries));
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

### Type Mismatches

**Problem:** `expected f32, found String` deserialization errors.

**Solutions:**
1. Check JSON payload field types
2. Verify camelCase/snake_case conversion
3. Use validation endpoint to test payload
4. Check TypeScript type definitions match Rust types

## Testing Settings Changes

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_settings_validation() {
        let mut settings = AppFullSettings::default();
        settings.visualisation.rendering.ambient_light_intensity = 15.0;

        let result = ValidationService::validate_settings(&settings);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_user_override() {
        let global = AppFullSettings::default();
        let mut user = UserSettings::new("test_user", global.clone());

        user.settings.visualisation.rendering.ambient_light_intensity = 0.8;
        assert_eq!(user.settings.visualisation.rendering.ambient_light_intensity, 0.8);
        assert_eq!(global.visualisation.rendering.ambient_light_intensity, 0.5);
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_settings_api() {
    let app = test::init_service(App::new().configure(settings_routes)).await;

    let req = test::TestRequest::put()
        .uri("/api/settings")
        .set_json(&json!({
            "visualisation": {
                "rendering": {
                    "ambientLightIntensity": 0.7
                }
            }
        }))
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), StatusCode::OK);
}
```

## Related Documentation

- [Settings System Architecture](./settings-system.md)
- [Settings API Reference](./settings-api.md)
- [Validation Rules](./settings-validation.md)
- [Database Schema](./settings-schema.md)
- [User Permissions](./user-permissions.md)
- [Migration Checklist](./MIGRATION_CHECKLIST.md)
