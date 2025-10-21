# SQLite Database Settings Storage Analysis

## Executive Summary

The application uses a **dual storage approach** for settings:
1. **Individual settings** stored as rows with dot-notation paths
2. **Complete settings blob** stored as a single JSON row

## Database File Location

**Path:** `${DATA_ROOT}/ontology_db.sqlite3`
**Default:** `/app/data/ontology_db.sqlite3`
**Initialization:** `src/main.rs` line 254-267

```rust
let db_path = std::env::var("DATA_ROOT")
    .unwrap_or_else(|_| "/app/data".to_string());
let db_file = std::path::PathBuf::from(&db_path).join("ontology_db.sqlite3");
let db_service = DatabaseService::new(&db_file)?;
db_service.initialize_schema()?;
```

## Table Schemas

### 1. `settings` Table (Individual Settings Storage)

```sql
CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,              -- Hierarchical path: "visualisation.graphs.logseq.nodes.baseColor"
    parent_key TEXT,                        -- Optional hierarchy (not currently used)
    value_type TEXT NOT NULL,               -- 'string', 'integer', 'float', 'boolean', 'json'
    value_text TEXT,                        -- For string values
    value_integer INTEGER,                  -- For integer values
    value_float REAL,                       -- For float values
    value_boolean INTEGER,                  -- For boolean values (0/1)
    value_json TEXT,                        -- For complex objects/arrays
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (parent_key) REFERENCES settings(key) ON DELETE CASCADE
);
```

**Storage Format:**
- **One row per setting** with hierarchical dot-notation keys
- **Sparse columns:** Only the column matching `value_type` contains data
- **Example rows:**

| key | value_type | value_text | value_float | value_boolean |
|-----|------------|------------|-------------|---------------|
| `visualisation.graphs.logseq.nodes.baseColor` | string | #FF5733 | NULL | NULL |
| `visualisation.graphs.logseq.physics.damping` | float | NULL | 0.95 | NULL |
| `visualisation.graphs.logseq.physics.enableBounds` | boolean | NULL | NULL | 1 |

### 2. `physics_settings` Table (Dedicated Physics Storage)

```sql
CREATE TABLE IF NOT EXISTS physics_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_name TEXT NOT NULL UNIQUE DEFAULT 'default',
    -- 22 physics parameters as dedicated columns
    damping REAL NOT NULL DEFAULT 0.95,
    dt REAL NOT NULL DEFAULT 0.016,
    iterations INTEGER NOT NULL DEFAULT 100,
    max_velocity REAL NOT NULL DEFAULT 1.0,
    max_force REAL NOT NULL DEFAULT 100.0,
    repel_k REAL NOT NULL DEFAULT 50.0,
    spring_k REAL NOT NULL DEFAULT 0.005,
    mass_scale REAL NOT NULL DEFAULT 1.0,
    boundary_damping REAL NOT NULL DEFAULT 0.95,
    temperature REAL NOT NULL DEFAULT 0.01,
    gravity REAL NOT NULL DEFAULT 0.0001,
    bounds_size REAL NOT NULL DEFAULT 500.0,
    enable_bounds INTEGER NOT NULL DEFAULT 1,
    rest_length REAL NOT NULL DEFAULT 50.0,
    repulsion_cutoff REAL NOT NULL DEFAULT 50.0,
    repulsion_softening_epsilon REAL NOT NULL DEFAULT 0.0001,
    center_gravity_k REAL NOT NULL DEFAULT 0.0,
    grid_cell_size REAL NOT NULL DEFAULT 50.0,
    warmup_iterations INTEGER NOT NULL DEFAULT 100,
    cooling_rate REAL NOT NULL DEFAULT 0.001,
    constraint_ramp_frames INTEGER NOT NULL DEFAULT 60,
    constraint_max_force_per_node REAL NOT NULL DEFAULT 50.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Purpose:** Fast access to physics parameters without JSON parsing

### 3. `user_settings` Table (Per-User Overrides)

```sql
CREATE TABLE IF NOT EXISTS user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key TEXT NOT NULL,                      -- Same hierarchical path format
    value_type TEXT NOT NULL,               -- Same as settings table
    value_text TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER,
    value_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE(user_id, key)
);
```

**Purpose:** User-specific overrides that take precedence over global settings

## Dual Storage Approach

### Approach 1: Individual Settings Rows (Granular)

**Implementation:** `src/services/database_service.rs` lines 74-127

```rust
pub fn get_setting(&self, key: &str) -> SqliteResult<Option<SettingValue>> {
    let conn = self.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT value_type, value_text, value_integer, value_float, value_boolean, value_json
         FROM settings WHERE key = ?1"
    )?;

    stmt.query_row(params![key], |row| {
        let value_type: String = row.get(0)?;
        match value_type.as_str() {
            "string" => SettingValue::String(row.get(1)?),
            "integer" => SettingValue::Integer(row.get(2)?),
            "float" => SettingValue::Float(row.get(3)?),
            "boolean" => SettingValue::Boolean(row.get::<_, i32>(4)? == 1),
            "json" => {
                let json_str: String = row.get(5)?;
                SettingValue::Json(serde_json::from_str(&json_str)?)
            },
            _ => SettingValue::String(String::new()),
        }
    }).optional()
}

pub fn set_setting(&self, key: &str, value: SettingValue, description: Option<&str>) -> SqliteResult<()> {
    // Upsert individual setting row
}
```

**Use Case:** Real-time setting updates via WebSocket, granular permission checks

### Approach 2: Complete Settings Blob (Atomic)

**Implementation:** `src/services/database_service.rs` lines 206-253

```rust
pub fn save_all_settings(&self, settings: &AppFullSettings) -> SqliteResult<()> {
    let conn = self.conn.lock().unwrap();

    // Serialize entire AppFullSettings as JSON
    let settings_json = serde_json::to_string(settings)?;

    // Store as single row with key 'app_full_settings'
    conn.execute(
        "INSERT INTO settings (key, value_type, value_json, description)
         VALUES ('app_full_settings', 'json', ?1, 'Complete application settings')
         ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at = CURRENT_TIMESTAMP",
        params![settings_json]
    )?;

    // Also save physics to dedicated table for fast access
    self.save_physics_settings("default", &settings.visualisation.graphs.logseq.physics)?;
    Ok(())
}

pub fn load_all_settings(&self) -> SqliteResult<Option<AppFullSettings>> {
    let conn = self.conn.lock().unwrap();

    let settings_json: Option<String> = conn.query_row(
        "SELECT value_json FROM settings WHERE key = 'app_full_settings'",
        [],
        |row| row.get(0)
    ).optional()?;

    if let Some(json_str) = settings_json {
        let settings: AppFullSettings = serde_json::from_str(&json_str)?;
        Ok(Some(settings))
    } else {
        Ok(None)
    }
}
```

**Use Case:** Application startup, bulk settings export, backup/restore

## AppFullSettings Rust Structure

**Location:** `src/config/mod.rs`

```rust
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings,  // Nested 5+ levels deep
    pub system: SystemSettings,
    pub xr: XRSettings,
    pub auth: AuthSettings,
    pub ragflow: Option<RagFlowSettings>,
    pub perplexity: Option<PerplexitySettings>,
    pub openai: Option<OpenAISettings>,
    pub kokoro: Option<KokoroSettings>,
    pub whisper: Option<WhisperSettings>,
    pub version: String,
}

pub struct VisualisationSettings {
    pub graphs: GraphsSettings,               // Contains logseq, visionflow
    pub rendering: RenderingSettings,
    pub glow: GlowSettings,
    pub bloom: BloomSettings,
    // ... many more nested structs
}
```

**Nesting Depth:** Up to 7 levels deep (e.g., `visualisation.graphs.logseq.nodes.glow.baseColor`)

## Database vs. Struct Mismatch

### Critical Mismatch: Storage Formats

#### In Database (Individual Settings)
```sql
-- Flat rows with dot-notation keys
INSERT INTO settings (key, value_type, value_text)
VALUES ('visualisation.graphs.logseq.nodes.baseColor', 'string', '#FF5733');
```

#### In AppFullSettings (Nested Structs)
```rust
AppFullSettings {
    visualisation: VisualisationSettings {
        graphs: GraphsSettings {
            logseq: LogseqGraphSettings {
                nodes: NodeSettings {
                    baseColor: "#FF5733"
                }
            }
        }
    }
}
```

### How They Reconcile

1. **Complete Settings Blob:** Entire `AppFullSettings` serialized as JSON in one row
   - **Database row:** `key='app_full_settings', value_json='{...entire JSON...}'`
   - **Matches struct:** Perfect 1:1 match (JSON serialization preserves structure)

2. **Individual Settings:** Flattened paths for granular access
   - **Migration:** `src/services/settings_migration.rs` flattens nested YAML/TOML
   - **Runtime access:** `SettingsService` converts between flat paths and nested structs

## Key Format Conversion

**Dual Key Format Support:** Both camelCase (client) and snake_case (server)

### Implementation: `src/services/settings_migration.rs` lines 187-248

```rust
fn migrate_setting(&self, key: &str, value: &YamlValue) -> Result<(), String> {
    let setting_value = self.yaml_to_setting_value(value)?;

    // Store with camelCase key (original format)
    self.db_service.set_setting(key, setting_value.clone(), None)?;

    // Generate snake_case equivalent
    let snake_key = self.to_snake_case_key(key);

    // Store with snake_case key if different
    if snake_key != key {
        self.db_service.set_setting(&snake_key, setting_value, None)?;
    }

    Ok(())
}

fn to_snake_case_key(&self, key: &str) -> String {
    key.split('.')
        .map(|part| Self::to_snake_case_part(part))
        .collect::<Vec<_>>()
        .join(".")
}
```

**Example Dual Keys:**
- camelCase: `visualisation.graphs.logseq.nodes.baseColor`
- snake_case: `visualisation.graphs.logseq.nodes.base_color`

**Both stored as separate rows** for client/server compatibility

## Settings Migration Process

**Location:** `src/services/settings_migration.rs`

### Migration Steps

1. **Load YAML files** (legacy format)
   - `data/settings.yaml`
   - `data/settings_ontology_extension.yaml`

2. **Flatten nested structure** (lines 144-184)
   ```rust
   fn flatten_yaml(&self, value: &YamlValue, prefix: &str) -> HashMap<String, YamlValue> {
       // Recursively converts:
       // visualisation:
       //   graphs:
       //     logseq:
       //       nodes:
       //         baseColor: "#FF5733"
       //
       // Into:
       // "visualisation.graphs.logseq.nodes.baseColor" => "#FF5733"
   }
   ```

3. **Store both key formats** (camelCase + snake_case)

4. **Extract physics profiles** (lines 296-360)
   - Parse `visualisation.graphs.{profile}.physics`
   - Insert into dedicated `physics_settings` table

5. **Migrate dev_config.toml** (lines 398-506)
   - Load TOML sections (physics, cuda, network, rendering, performance, debug)
   - Store with prefix `dev.{section}.{key}`

### Migration Result Statistics

```rust
pub struct MigrationResult {
    pub settings_migrated: usize,           // Individual setting rows
    pub physics_profiles_migrated: usize,   // physics_settings table rows
    pub dev_config_params_migrated: usize,  // dev_config parameters
    pub errors: Vec<String>,
    pub duration: std::time::Duration,
}
```

## Sample Database Queries

### Get Individual Setting
```sql
SELECT value_type, value_text, value_float, value_boolean, value_json
FROM settings
WHERE key = 'visualisation.graphs.logseq.nodes.baseColor';
```

### Get Complete Settings Blob
```sql
SELECT value_json
FROM settings
WHERE key = 'app_full_settings';
```

### Get Physics Profile
```sql
SELECT *
FROM physics_settings
WHERE profile_name = 'default';
```

### Get User Override
```sql
SELECT s.*, us.value_text as user_override
FROM settings s
LEFT JOIN user_settings us ON s.key = us.key AND us.user_id = ?
WHERE s.key = 'visualisation.graphs.logseq.nodes.baseColor';
```

### Search Settings by Pattern
```sql
SELECT key, value_type, value_text, value_float, value_boolean
FROM settings
WHERE key LIKE 'visualisation.graphs.%'
ORDER BY key;
```

## Mismatches and Gotchas

### 1. parent_key Column (Unused)
- **Schema defines:** `parent_key TEXT` for hierarchy
- **Migration ignores:** Never populates this column
- **Impact:** None (flat dot-notation keys work fine)

### 2. Dual Storage Redundancy
- **Complete blob:** `key='app_full_settings'` contains ALL settings as JSON
- **Individual rows:** Same data duplicated as separate rows
- **Reason:** Performance tradeoff (fast granular access vs. atomic updates)

### 3. AppFullSettings != Individual Rows
- **Struct fields:** 10 top-level fields (visualisation, system, xr, auth, etc.)
- **Database rows:** 100+ individual setting rows
- **Resolution:** JSON blob matches struct 1:1, individual rows are flattened view

### 4. No Schema Version Migration
- **Current approach:** Full schema recreation on startup
- **Risk:** Schema changes require manual migration logic
- **Mitigation:** `schema_version` table exists but not actively used for migrations

## Recommendations

### For Database Schema Changes

1. **Use schema_version table** for tracking migrations
2. **Write migration scripts** for schema evolution
3. **Test migration path** from current production DB

### For Adding New Settings

1. **Update AppFullSettings struct** in `src/config/mod.rs`
2. **Add validation** in `validate_*` functions
3. **Update migration** in `settings_migration.rs` if needed
4. **Add to both formats:** camelCase + snake_case

### For Performance Optimization

1. **Index frequently accessed keys:**
   ```sql
   CREATE INDEX idx_settings_key_prefix ON settings(key) WHERE key LIKE 'visualisation.%';
   ```

2. **Use physics_settings table** for physics parameters (already optimized)

3. **Cache complete settings blob** in application memory (already done via `SettingsService`)

### For Debugging

1. **Check both storage formats:**
   ```bash
   sqlite3 /app/data/ontology_db.sqlite3
   .mode column
   .headers on

   -- Individual settings
   SELECT * FROM settings WHERE key LIKE 'visualisation.%' LIMIT 10;

   -- Complete blob
   SELECT key, length(value_json) as json_size FROM settings WHERE key = 'app_full_settings';

   -- Physics table
   SELECT * FROM physics_settings WHERE profile_name = 'default';
   ```

2. **Compare struct vs. database:**
   ```rust
   // Load from database
   let db_settings = db_service.load_all_settings()?;

   // Compare with default struct
   let default_settings = AppFullSettings::default();

   // Serialize both to JSON and diff
   let db_json = serde_json::to_string_pretty(&db_settings)?;
   let default_json = serde_json::to_string_pretty(&default_settings)?;
   ```

## Conclusion

The SQLite settings storage uses a **hybrid approach**:

1. **Atomic blob storage** (`app_full_settings` JSON row)
   - Matches `AppFullSettings` struct perfectly (1:1 serialization)
   - Used for application startup and bulk operations

2. **Granular row storage** (individual setting rows with dot-notation keys)
   - Flattened view of nested structure
   - Used for real-time updates, WebSocket synchronization, permission checks
   - Supports both camelCase (client) and snake_case (server) key formats

3. **Dedicated physics table** (`physics_settings`)
   - Optimized for fast access without JSON parsing
   - Stores 22 physics parameters as dedicated columns

**No fundamental mismatch** exists—both storage formats are intentionally maintained:
- Blob for complete struct operations
- Rows for granular path-based access
- Both synchronized during migration and runtime updates
