# Settings Migration Implementation Plan

## Executive Summary

This document outlines the strategy for migrating VisionFlow's settings from YAML/TOML files to a SQLite-based database system. The migration preserves all existing functionality while adding hot-reload capabilities, improved validation, and a developer-friendly control panel.

**Current State:**
- **settings.yaml** (498 lines): Complete application settings including visualization, physics, XR, auth, and API configurations
- **dev_config.toml** (169 lines): Developer-specific settings for physics tuning, CUDA parameters, network pooling, and debugging
- **Existing infrastructure**: SQLite database with settings schema already deployed (`settings_db.sql`)

**Target State:**
- Unified SQLite database (`settings.db`) with all settings
- Hot-reload without server restart
- Real-time WebSocket notifications for setting changes
- Developer CLI for advanced settings management
- Backward compatibility during transition

---

## Architecture Overview

### Current System
```
┌─────────────────────┐
│  settings.yaml      │──┐
│  (498 lines)        │  │
└─────────────────────┘  │    ┌──────────────────┐
                         ├───▶│  SettingsService │
┌─────────────────────┐  │    │  (in-memory)     │
│  dev_config.toml    │──┘    └──────────────────┘
│  (169 lines)        │                │
└─────────────────────┘                │
                                       ▼
                            ┌────────────────────┐
                            │  Application       │
                            │  Components        │
                            └────────────────────┘
```

### Target System
```
┌──────────────────────────────────────────────────────────┐
│                    settings.db                            │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   settings   │  │   physics   │  │  audit_log      │  │
│  │   (k/v)      │  │  _settings  │  │  (changes)      │  │
│  └──────────────┘  └─────────────┘  └─────────────────┘  │
└──────────────────────────────────────────────────────────┘
         │                      │                  │
         │   ┌──────────────────┴──────────────────┘
         │   │
         ▼   ▼
┌─────────────────────────┐         ┌──────────────────┐
│  DatabaseService        │◀───────▶│  SettingsService │
│  (connection pool)      │         │  (with cache)    │
└─────────────────────────┘         └──────────────────┘
                                             │
              ┌──────────────────────────────┼─────────────────┐
              │                              │                 │
              ▼                              ▼                 ▼
    ┌─────────────────┐          ┌─────────────────┐  ┌────────────┐
    │  WebSocket      │          │  REST API       │  │  Dev CLI   │
    │  Notifications  │          │  Endpoints      │  │  Tool      │
    └─────────────────┘          └─────────────────┘  └────────────┘
```

---

## Phase 1: Database Schema Validation (Week 1, Days 1-2)

### Current Status ✅
The database schema is **already implemented** in `/home/devuser/workspace/project/schema/settings_db.sql`:
- ✅ `settings` table with typed columns (string, integer, float, boolean, json)
- ✅ `physics_settings` table with 22+ physics parameters
- ✅ `users`, `api_keys`, `sessions`, `rate_limits` tables
- ✅ `settings_audit_log` for change tracking
- ✅ `feature_flags` for A/B testing
- ✅ Proper indexes and triggers for automatic timestamps
- ✅ Connection pooling via `r2d2_sqlite`

### Tasks
1. **Test schema initialization** ✅ (Already working)
   - Verify schema creates without errors
   - Test foreign key constraints
   - Validate default data insertion

2. **Add migration support**
   ```rust
   // src/services/database_service.rs additions
   impl DatabaseService {
       /// Check if settings have been migrated
       pub fn is_migrated(&self) -> SqliteResult<bool> {
           let conn = self.get_settings_connection()?;
           let exists: bool = conn.query_row(
               "SELECT EXISTS(SELECT 1 FROM settings WHERE key = 'migration_completed')",
               [],
               |row| row.get(0)
           )?;
           Ok(exists)
       }

       /// Mark migration as complete
       pub fn mark_migrated(&self) -> SqliteResult<()> {
           self.set_setting(
               "migration_completed",
               SettingValue::Boolean(true),
               Some("Settings migrated from YAML/TOML")
           )
       }
   }
   ```

3. **Create validation suite**
   ```rust
   // tests/settings_migration_tests.rs
   #[tokio::test]
   async fn test_schema_integrity() {
       let db = DatabaseService::new("test_settings.db").unwrap();
       db.initialize_schema().unwrap();

       // Test all tables exist
       let tables = vec!["settings", "physics_settings", "settings_audit_log"];
       for table in tables {
           assert!(db.table_exists(table).unwrap());
       }
   }
   ```

**Deliverables:**
- ✅ Schema validated (already done)
- Migration status tracking functions
- Test suite for schema integrity
- Documentation of schema structure

---

## Phase 2: Migration Script Implementation (Week 1, Days 3-5)

### Overview
Create a robust migration script that parses YAML/TOML files and populates the database while preserving data integrity.

### 2.1 YAML Parser Module

```rust
// src/migration/yaml_parser.rs
use serde_yaml::Value as YamlValue;
use std::fs;
use crate::config::AppFullSettings;

pub struct YamlMigrator {
    settings: AppFullSettings,
}

impl YamlMigrator {
    /// Load settings from YAML file
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read YAML: {}", e))?;

        let settings: AppFullSettings = serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse YAML: {}", e))?;

        Ok(Self { settings })
    }

    /// Flatten settings into key-value pairs
    pub fn flatten(&self) -> Vec<(String, SettingValue)> {
        let mut result = Vec::new();
        self.flatten_recursive("", &serde_yaml::to_value(&self.settings).unwrap(), &mut result);
        result
    }

    fn flatten_recursive(&self, prefix: &str, value: &YamlValue, result: &mut Vec<(String, SettingValue)>) {
        match value {
            YamlValue::Mapping(map) => {
                for (k, v) in map {
                    let key = if prefix.is_empty() {
                        k.as_str().unwrap().to_string()
                    } else {
                        format!("{}.{}", prefix, k.as_str().unwrap())
                    };
                    self.flatten_recursive(&key, v, result);
                }
            },
            YamlValue::String(s) => result.push((prefix.to_string(), SettingValue::String(s.clone()))),
            YamlValue::Number(n) => {
                if n.is_f64() {
                    result.push((prefix.to_string(), SettingValue::Float(n.as_f64().unwrap())));
                } else {
                    result.push((prefix.to_string(), SettingValue::Integer(n.as_i64().unwrap())));
                }
            },
            YamlValue::Bool(b) => result.push((prefix.to_string(), SettingValue::Boolean(*b))),
            YamlValue::Sequence(_) | YamlValue::Tagged(_) => {
                result.push((prefix.to_string(), SettingValue::Json(serde_json::to_value(value).unwrap())));
            },
            _ => {},
        }
    }
}
```

### 2.2 TOML Parser Module

```rust
// src/migration/toml_parser.rs
use toml::Value as TomlValue;
use std::fs;

pub struct TomlMigrator {
    content: TomlValue,
}

impl TomlMigrator {
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read TOML: {}", e))?;

        let parsed: TomlValue = toml::from_str(&content)
            .map_err(|e| format!("Failed to parse TOML: {}", e))?;

        Ok(Self { content: parsed })
    }

    pub fn flatten(&self) -> Vec<(String, SettingValue)> {
        let mut result = Vec::new();
        self.flatten_recursive("dev", &self.content, &mut result);
        result
    }

    fn flatten_recursive(&self, prefix: &str, value: &TomlValue, result: &mut Vec<(String, SettingValue)>) {
        match value {
            TomlValue::Table(table) => {
                for (k, v) in table {
                    let key = format!("{}.{}", prefix, k);
                    self.flatten_recursive(&key, v, result);
                }
            },
            TomlValue::String(s) => result.push((prefix.to_string(), SettingValue::String(s.clone()))),
            TomlValue::Integer(i) => result.push((prefix.to_string(), SettingValue::Integer(*i))),
            TomlValue::Float(f) => result.push((prefix.to_string(), SettingValue::Float(*f))),
            TomlValue::Boolean(b) => result.push((prefix.to_string(), SettingValue::Boolean(*b))),
            TomlValue::Array(_) | TomlValue::Datetime(_) => {
                result.push((prefix.to_string(), SettingValue::Json(serde_json::to_value(value).unwrap())));
            },
        }
    }
}
```

### 2.3 Migration Orchestrator

```rust
// src/migration/mod.rs
mod yaml_parser;
mod toml_parser;

use crate::services::database_service::DatabaseService;
use log::{info, warn};

pub struct SettingsMigration {
    db: DatabaseService,
}

impl SettingsMigration {
    pub fn new(db: DatabaseService) -> Self {
        Self { db }
    }

    /// Execute complete migration
    pub async fn migrate(&self) -> Result<MigrationReport, String> {
        info!("Starting settings migration...");

        let mut report = MigrationReport::default();

        // Check if already migrated
        if self.db.is_migrated().unwrap_or(false) {
            warn!("Settings already migrated. Use --force to re-migrate.");
            return Ok(report);
        }

        // Phase 1: Migrate YAML settings
        info!("Migrating settings.yaml...");
        match self.migrate_yaml("data/settings.yaml") {
            Ok(count) => {
                report.yaml_keys_migrated = count;
                info!("Migrated {} keys from YAML", count);
            },
            Err(e) => {
                report.errors.push(format!("YAML migration failed: {}", e));
                return Err(format!("YAML migration failed: {}", e));
            }
        }

        // Phase 2: Migrate TOML settings
        info!("Migrating dev_config.toml...");
        match self.migrate_toml("data/dev_config.toml") {
            Ok(count) => {
                report.toml_keys_migrated = count;
                info!("Migrated {} keys from TOML", count);
            },
            Err(e) => {
                report.errors.push(format!("TOML migration failed: {}", e));
                return Err(format!("TOML migration failed: {}", e));
            }
        }

        // Phase 3: Validate migration
        info!("Validating migrated data...");
        self.validate_migration(&mut report)?;

        // Phase 4: Mark as complete
        self.db.mark_migrated()
            .map_err(|e| format!("Failed to mark migration complete: {}", e))?;

        report.success = true;
        info!("Migration completed successfully!");
        info!("Total: {} YAML keys, {} TOML keys",
              report.yaml_keys_migrated, report.toml_keys_migrated);

        Ok(report)
    }

    fn migrate_yaml(&self, path: &str) -> Result<usize, String> {
        let migrator = yaml_parser::YamlMigrator::load_from_file(path)?;
        let settings = migrator.flatten();

        let mut count = 0;
        for (key, value) in settings {
            self.db.set_setting(&key, value, None)
                .map_err(|e| format!("Failed to set {}: {}", key, e))?;
            count += 1;
        }

        Ok(count)
    }

    fn migrate_toml(&self, path: &str) -> Result<usize, String> {
        let migrator = toml_parser::TomlMigrator::load_from_file(path)?;
        let settings = migrator.flatten();

        let mut count = 0;
        for (key, value) in settings {
            self.db.set_setting(&key, value, Some("Developer setting"))
                .map_err(|e| format!("Failed to set {}: {}", key, e))?;
            count += 1;
        }

        Ok(count)
    }

    fn validate_migration(&self, report: &mut MigrationReport) -> Result<(), String> {
        // Validate critical settings exist
        let critical_keys = vec![
            "system.network.port",
            "system.websocket.updateRate",
            "visualisation.rendering.enableAntialiasing",
            "dev.physics.rest_length",
        ];

        for key in critical_keys {
            match self.db.get_setting(key) {
                Ok(Some(_)) => report.validated_keys += 1,
                Ok(None) => {
                    let msg = format!("Critical setting missing: {}", key);
                    report.warnings.push(msg);
                },
                Err(e) => {
                    let msg = format!("Validation error for {}: {}", key, e);
                    report.errors.push(msg);
                }
            }
        }

        Ok(())
    }
}

#[derive(Default, Debug)]
pub struct MigrationReport {
    pub success: bool,
    pub yaml_keys_migrated: usize,
    pub toml_keys_migrated: usize,
    pub validated_keys: usize,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}
```

### 2.4 CLI Migration Tool

```rust
// src/bin/migrate_settings.rs
use webxr::migration::SettingsMigration;
use webxr::services::database_service::DatabaseService;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    let db_path = args.get(1)
        .map(|s| s.as_str())
        .unwrap_or("data/settings.db");

    println!("VisionFlow Settings Migration Tool");
    println!("Database: {}", db_path);
    println!("-----------------------------------\n");

    // Initialize database
    let db = DatabaseService::new(db_path)?;
    db.initialize_schema()?;

    // Run migration
    let migration = SettingsMigration::new(db);
    let report = migration.migrate().await?;

    // Print report
    println!("\n=== Migration Report ===");
    println!("Success: {}", report.success);
    println!("YAML keys migrated: {}", report.yaml_keys_migrated);
    println!("TOML keys migrated: {}", report.toml_keys_migrated);
    println!("Validated keys: {}", report.validated_keys);

    if !report.warnings.is_empty() {
        println!("\nWarnings:");
        for warning in &report.warnings {
            println!("  ⚠️  {}", warning);
        }
    }

    if !report.errors.is_empty() {
        println!("\nErrors:");
        for error in &report.errors {
            println!("  ❌ {}", error);
        }
    }

    Ok(())
}
```

**Usage:**
```bash
# Run migration
cargo run --bin migrate_settings data/settings.db

# Force re-migration
cargo run --bin migrate_settings data/settings.db --force
```

**Deliverables:**
- YAML parser with flattening logic
- TOML parser with flattening logic
- Migration orchestrator with validation
- CLI tool for running migrations
- Test suite for parsers and migration logic

---

## Phase 3: Hot-Reload Implementation (Week 2, Days 1-3)

### Overview
Implement hot-reload mechanism allowing settings changes without server restart.

### 3.1 File Watcher Integration

```rust
// Cargo.toml addition
[dependencies]
notify = "6.1"
```

```rust
// src/services/settings_watcher.rs
use notify::{Watcher, RecursiveMode, Result as NotifyResult};
use std::sync::Arc;
use std::path::Path;
use tokio::sync::mpsc;
use log::{info, error};

pub struct SettingsWatcher {
    db: Arc<DatabaseService>,
    tx: mpsc::Sender<SettingChangeEvent>,
}

#[derive(Debug, Clone)]
pub struct SettingChangeEvent {
    pub key: String,
    pub old_value: Option<SettingValue>,
    pub new_value: SettingValue,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl SettingsWatcher {
    pub fn new(db: Arc<DatabaseService>) -> (Self, mpsc::Receiver<SettingChangeEvent>) {
        let (tx, rx) = mpsc::channel(100);
        (Self { db, tx }, rx)
    }

    /// Start watching for database changes
    pub async fn start(&self) -> NotifyResult<()> {
        info!("Starting settings watcher...");

        let db = self.db.clone();
        let tx = self.tx.clone();

        tokio::spawn(async move {
            // Poll database for changes every 1 second
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));
            let mut last_check = chrono::Utc::now();

            loop {
                interval.tick().await;

                // Query for recent changes
                match Self::check_changes(&db, last_check).await {
                    Ok(changes) => {
                        for change in changes {
                            if let Err(e) = tx.send(change).await {
                                error!("Failed to send change event: {}", e);
                            }
                        }
                    },
                    Err(e) => error!("Failed to check for changes: {}", e),
                }

                last_check = chrono::Utc::now();
            }
        });

        Ok(())
    }

    async fn check_changes(
        db: &DatabaseService,
        since: chrono::DateTime<chrono::Utc>
    ) -> Result<Vec<SettingChangeEvent>, String> {
        // Query settings_audit_log for recent changes
        let conn = db.get_settings_connection()?;

        let mut stmt = conn.prepare(
            "SELECT setting_key, old_value, new_value, changed_at
             FROM settings_audit_log
             WHERE changed_at > ?1
             ORDER BY changed_at ASC"
        ).map_err(|e| format!("SQL error: {}", e))?;

        let changes = stmt.query_map(
            [since.to_rfc3339()],
            |row| {
                Ok(SettingChangeEvent {
                    key: row.get(0)?,
                    old_value: None, // Parse from row.get(1)?
                    new_value: SettingValue::String(row.get(2)?), // Parse from row.get(2)?
                    timestamp: chrono::Utc::now(),
                })
            }
        ).map_err(|e| format!("Query error: {}", e))?;

        Ok(changes.filter_map(Result::ok).collect())
    }
}
```

### 3.2 WebSocket Notifications

```rust
// src/handlers/settings_ws.rs
use actix::{Actor, StreamHandler, Handler, Message as ActixMessage};
use actix_web::{web, HttpRequest, HttpResponse, Error};
use actix_web_actors::ws;

pub struct SettingsWebSocket {
    user_id: Option<String>,
}

impl Actor for SettingsWebSocket {
    type Context = ws::WebsocketContext<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        log::info!("Settings WebSocket connected");
    }
}

impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for SettingsWebSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Ping(msg)) => ctx.pong(&msg),
            Ok(ws::Message::Text(text)) => {
                // Handle subscription requests
                if text.starts_with("subscribe:") {
                    let key = text.strip_prefix("subscribe:").unwrap();
                    log::info!("Client subscribed to: {}", key);
                }
            },
            _ => {}
        }
    }
}

// Broadcast setting change to all connected clients
#[derive(ActixMessage)]
#[rtype(result = "()")]
pub struct SettingChangedBroadcast {
    pub key: String,
    pub value: SettingValue,
}

impl Handler<SettingChangedBroadcast> for SettingsWebSocket {
    type Result = ();

    fn handle(&mut self, msg: SettingChangedBroadcast, ctx: &mut Self::Context) {
        let notification = serde_json::json!({
            "event": "setting_changed",
            "key": msg.key,
            "value": msg.value,
            "timestamp": chrono::Utc::now().to_rfc3339()
        });

        ctx.text(notification.to_string());
    }
}

pub async fn settings_websocket(
    req: HttpRequest,
    stream: web::Payload,
) -> Result<HttpResponse, Error> {
    ws::start(SettingsWebSocket { user_id: None }, &req, stream)
}
```

### 3.3 Settings Service Integration

```rust
// src/services/settings_service.rs additions
impl SettingsService {
    /// Enable hot-reload with WebSocket broadcasting
    pub async fn enable_hot_reload(&self) -> Result<(), String> {
        let (watcher, mut rx) = SettingsWatcher::new(self.db.clone());
        watcher.start().await.map_err(|e| format!("Failed to start watcher: {}", e))?;

        // Spawn task to handle change events
        let listeners = self.change_listeners.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                let listeners = listeners.read().await;
                for listener in listeners.iter() {
                    listener(&event.key, &event.new_value);
                }
            }
        });

        Ok(())
    }
}
```

**Deliverables:**
- File/database watcher implementation
- WebSocket broadcasting for changes
- Integration with SettingsService
- Frontend listener example
- Test suite for hot-reload

---

## Phase 4: Backward Compatibility Layer (Week 2, Days 4-5)

### Overview
Ensure smooth transition by supporting both database and file-based settings during migration period.

### 4.1 Hybrid Settings Loader

```rust
// src/services/hybrid_settings_loader.rs
pub struct HybridSettingsLoader {
    db: Arc<DatabaseService>,
    yaml_path: Option<String>,
    toml_path: Option<String>,
}

impl HybridSettingsLoader {
    pub fn new(db: Arc<DatabaseService>) -> Self {
        Self {
            db,
            yaml_path: Some("data/settings.yaml".to_string()),
            toml_path: Some("data/dev_config.toml".to_string()),
        }
    }

    /// Load setting with fallback priority: DB → YAML → TOML → Default
    pub async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>, String> {
        // Priority 1: Database
        if let Ok(Some(value)) = self.db.get_setting(key) {
            log::debug!("Setting '{}' loaded from database", key);
            return Ok(Some(value));
        }

        // Priority 2: YAML file
        if let Some(yaml_path) = &self.yaml_path {
            if let Ok(value) = self.load_from_yaml(yaml_path, key) {
                log::warn!("Setting '{}' loaded from YAML (fallback). Consider migrating.", key);
                return Ok(Some(value));
            }
        }

        // Priority 3: TOML file
        if let Some(toml_path) = &self.toml_path {
            if let Ok(value) = self.load_from_toml(toml_path, key) {
                log::warn!("Setting '{}' loaded from TOML (fallback). Consider migrating.", key);
                return Ok(Some(value));
            }
        }

        Ok(None)
    }

    fn load_from_yaml(&self, path: &str, key: &str) -> Result<SettingValue, String> {
        // Implementation using serde_yaml
        todo!()
    }

    fn load_from_toml(&self, path: &str, key: &str) -> Result<SettingValue, String> {
        // Implementation using toml crate
        todo!()
    }
}
```

### 4.2 Deprecation Warnings

```rust
// Add to main.rs startup
async fn check_legacy_settings() {
    if Path::new("data/settings.yaml").exists() {
        log::warn!("⚠️  settings.yaml detected. This file is deprecated.");
        log::warn!("   Run 'cargo run --bin migrate_settings' to migrate to database.");
    }

    if Path::new("data/dev_config.toml").exists() {
        log::warn!("⚠️  dev_config.toml detected. This file is deprecated.");
        log::warn!("   Developer settings should be managed via the database.");
    }
}
```

**Deliverables:**
- Hybrid loader with fallback logic
- Deprecation warnings in logs
- Migration status dashboard
- Documentation for transition period

---

## Phase 5: Frontend Control Panel (Week 3, Days 1-4)

### Overview
Build intuitive control panel for managing settings in real-time.

### 5.1 Settings Schema API

```rust
// src/handlers/settings_schema_handler.rs
#[derive(Serialize)]
pub struct SettingSchema {
    pub key: String,
    pub value_type: String,
    pub category: String,
    pub description: String,
    pub default_value: Option<SettingValue>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub options: Option<Vec<String>>,
}

pub async fn get_settings_schema() -> HttpResponse {
    let schema = vec![
        SettingSchema {
            key: "system.network.port".to_string(),
            value_type: "integer".to_string(),
            category: "Network".to_string(),
            description: "Server port (1024-65535)".to_string(),
            default_value: Some(SettingValue::Integer(4000)),
            min: Some(1024.0),
            max: Some(65535.0),
            options: None,
        },
        // ... more settings
    ];

    HttpResponse::Ok().json(schema)
}
```

### 5.2 Frontend Component Structure

```typescript
// frontend/src/components/SettingsPanel.tsx
interface SettingConfig {
  key: string;
  type: 'string' | 'integer' | 'float' | 'boolean' | 'json';
  category: string;
  description: string;
  value: any;
  min?: number;
  max?: number;
  options?: string[];
}

const SettingsPanel: React.FC = () => {
  const [settings, setSettings] = useState<SettingConfig[]>([]);
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('All');

  // Load settings schema
  useEffect(() => {
    fetch('/api/settings/schema')
      .then(res => res.json())
      .then(data => setSettings(data));
  }, []);

  // Real-time updates via WebSocket
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:4000/ws/settings');

    ws.onmessage = (event) => {
      const change = JSON.parse(event.data);
      if (change.event === 'setting_changed') {
        updateSetting(change.key, change.value);
      }
    };

    return () => ws.close();
  }, []);

  const updateSetting = async (key: string, value: any) => {
    await fetch(`/api/settings/${key}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ value })
    });
  };

  return (
    <div className="settings-panel">
      <SearchBar value={search} onChange={setSearch} />
      <CategoryFilter value={category} onChange={setCategory} />

      <SettingsGrid>
        {filteredSettings.map(setting => (
          <SettingControl
            key={setting.key}
            config={setting}
            onChange={(value) => updateSetting(setting.key, value)}
          />
        ))}
      </SettingsGrid>
    </div>
  );
};
```

### 5.3 Dynamic Form Generation

```typescript
// frontend/src/components/SettingControl.tsx
const SettingControl: React.FC<{ config: SettingConfig }> = ({ config }) => {
  switch (config.type) {
    case 'boolean':
      return <Toggle value={config.value} onChange={...} />;

    case 'integer':
    case 'float':
      return (
        <Slider
          value={config.value}
          min={config.min}
          max={config.max}
          step={config.type === 'integer' ? 1 : 0.01}
          onChange={...}
        />
      );

    case 'string':
      if (config.options) {
        return <Select options={config.options} value={config.value} />;
      }
      return <Input value={config.value} onChange={...} />;

    case 'json':
      return <JsonEditor value={config.value} onChange={...} />;
  }
};
```

**Deliverables:**
- Settings schema API endpoint
- React control panel components
- Real-time WebSocket integration
- Search and filter functionality
- Category-based organization
- Validation feedback

---

## Phase 6: Developer CLI Tool (Week 3, Days 5-7 & Week 4, Days 1-2)

### Overview
Command-line tool for advanced settings management, scripting, and automation.

### 6.1 CLI Architecture

```rust
// src/bin/settings_cli.rs
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "settings-cli")]
#[command(about = "VisionFlow Settings Management CLI", long_about = None)]
struct Cli {
    /// Database path
    #[arg(short, long, default_value = "data/settings.db")]
    database: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Get a setting value
    Get {
        /// Setting key (supports dot notation)
        key: String,

        /// Output format (json, yaml, toml)
        #[arg(short, long, default_value = "json")]
        format: String,
    },

    /// Set a setting value
    Set {
        /// Setting key
        key: String,

        /// New value
        value: String,

        /// Value type (auto, string, integer, float, boolean, json)
        #[arg(short, long, default_value = "auto")]
        r#type: String,
    },

    /// List all settings
    List {
        /// Filter by category
        #[arg(short, long)]
        category: Option<String>,

        /// Search pattern
        #[arg(short, long)]
        search: Option<String>,

        /// Output format
        #[arg(short, long, default_value = "table")]
        format: String,
    },

    /// Bulk operations
    Bulk {
        #[command(subcommand)]
        operation: BulkOperations,
    },

    /// Export settings
    Export {
        /// Output file
        output: String,

        /// Format (json, yaml, toml)
        #[arg(short, long, default_value = "yaml")]
        format: String,
    },

    /// Import settings
    Import {
        /// Input file
        input: String,

        /// Merge strategy (replace, merge, skip)
        #[arg(short, long, default_value = "merge")]
        strategy: String,
    },

    /// Show diff between two states
    Diff {
        /// Compare with file
        #[arg(short, long)]
        file: Option<String>,

        /// Compare with backup
        #[arg(short, long)]
        backup: Option<String>,
    },

    /// Watch for changes
    Watch {
        /// Filter by key pattern
        #[arg(short, long)]
        pattern: Option<String>,
    },

    /// Backup/restore operations
    Backup {
        #[command(subcommand)]
        operation: BackupOperations,
    },
}

#[derive(Subcommand)]
enum BulkOperations {
    /// Edit multiple settings from file
    Edit { file: String },

    /// Reset category to defaults
    Reset { category: String },

    /// Validate all settings
    Validate,
}

#[derive(Subcommand)]
enum BackupOperations {
    /// Create backup
    Create { name: String },

    /// List backups
    List,

    /// Restore from backup
    Restore { name: String },
}
```

### 6.2 CLI Implementation Examples

```rust
// Get setting
async fn handle_get(db: &DatabaseService, key: &str, format: &str) -> Result<(), String> {
    match db.get_setting(key) {
        Ok(Some(value)) => {
            let output = match format {
                "json" => serde_json::to_string_pretty(&value).unwrap(),
                "yaml" => serde_yaml::to_string(&value).unwrap(),
                _ => format!("{:?}", value),
            };
            println!("{}", output);
            Ok(())
        },
        Ok(None) => {
            eprintln!("Setting '{}' not found", key);
            Err("Not found".to_string())
        },
        Err(e) => Err(format!("Database error: {}", e)),
    }
}

// List settings with table output
async fn handle_list(db: &DatabaseService, category: Option<String>, format: &str) -> Result<(), String> {
    // Query all settings
    let settings = db.get_all_settings()?;

    match format {
        "table" => {
            println!("{:<40} {:<15} {:<30}", "Key", "Type", "Value");
            println!("{}", "-".repeat(85));

            for (key, value) in settings {
                let value_str = format!("{:?}", value);
                let truncated = if value_str.len() > 27 {
                    format!("{}...", &value_str[..27])
                } else {
                    value_str
                };
                println!("{:<40} {:<15} {:<30}", key, "string", truncated);
            }
        },
        "json" => {
            println!("{}", serde_json::to_string_pretty(&settings).unwrap());
        },
        _ => {}
    }

    Ok(())
}

// Watch for changes
async fn handle_watch(db: &DatabaseService, pattern: Option<String>) -> Result<(), String> {
    println!("Watching for setting changes... (Press Ctrl+C to stop)");

    let (watcher, mut rx) = SettingsWatcher::new(Arc::new(db.clone()));
    watcher.start().await?;

    while let Some(event) = rx.recv().await {
        if let Some(ref pat) = pattern {
            if !event.key.contains(pat) {
                continue;
            }
        }

        println!(
            "[{}] {} changed: {:?} → {:?}",
            event.timestamp.format("%H:%M:%S"),
            event.key,
            event.old_value,
            event.new_value
        );
    }

    Ok(())
}
```

### 6.3 Usage Examples

```bash
# Get single setting
settings-cli get system.network.port

# Set setting with type inference
settings-cli set system.network.port 8080

# Set boolean
settings-cli set debug_mode true --type boolean

# List all settings in category
settings-cli list --category network --format table

# Search settings
settings-cli list --search "physics" --format json

# Export all settings
settings-cli export backup.yaml --format yaml

# Import settings
settings-cli import new_config.toml --strategy merge

# Show diff
settings-cli diff --file production.yaml

# Watch for changes matching pattern
settings-cli watch --pattern "physics"

# Bulk edit from JSON file
settings-cli bulk edit updates.json

# Reset category to defaults
settings-cli bulk reset visualization

# Validate all settings
settings-cli bulk validate

# Create backup
settings-cli backup create pre-migration-2025-10-22

# List backups
settings-cli backup list

# Restore backup
settings-cli backup restore pre-migration-2025-10-22
```

**Deliverables:**
- Complete CLI tool with clap
- All subcommands implemented
- Table, JSON, YAML output formats
- Watch mode for real-time monitoring
- Backup/restore functionality
- Bulk operations support
- Comprehensive help documentation

---

## Phase 7: Testing & Rollout (Week 4, Days 3-5)

### 7.1 Test Plan

#### Unit Tests
```rust
// tests/settings_migration_tests.rs
#[tokio::test]
async fn test_yaml_migration() {
    let db = DatabaseService::new("test_migration.db").unwrap();
    db.initialize_schema().unwrap();

    let migration = SettingsMigration::new(db);
    let report = migration.migrate_yaml("tests/fixtures/test_settings.yaml").unwrap();

    assert!(report > 0);
}

#[tokio::test]
async fn test_hot_reload() {
    let db = Arc::new(DatabaseService::new("test_hot_reload.db").unwrap());
    let service = SettingsService::new(db.clone()).unwrap();

    service.enable_hot_reload().await.unwrap();

    // Change setting
    service.set_setting("test_key", SettingValue::String("new_value".into())).await.unwrap();

    // Verify notification received
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
}
```

#### Integration Tests
```rust
// tests/integration/settings_api_tests.rs
#[actix_web::test]
async fn test_settings_api_flow() {
    let app = test::init_service(App::new()
        .configure(configure_settings_routes)).await;

    // GET setting
    let req = test::TestRequest::get()
        .uri("/api/settings/system.network.port")
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    // SET setting
    let req = test::TestRequest::put()
        .uri("/api/settings/system.network.port")
        .set_json(&json!({"value": 8080}))
        .to_request();
    let resp = test::call_service(&app, req).await;
    assert_eq!(resp.status(), 200);

    // Verify change
    let req = test::TestRequest::get()
        .uri("/api/settings/system.network.port")
        .to_request();
    let resp = test::call_service(&app, req).await;
    let body: SettingValue = test::read_body_json(resp).await;
    assert_eq!(body, SettingValue::Integer(8080));
}
```

### 7.2 Rollout Strategy

#### Phase 1: Internal Testing (Days 1-2)
1. Deploy to development environment
2. Run migration on test data
3. Verify all settings migrated correctly
4. Test hot-reload functionality
5. Validate WebSocket notifications

#### Phase 2: Gradual Rollout (Days 3-4)
1. Enable database settings for non-critical features
2. Keep YAML/TOML as fallback
3. Monitor for errors and performance issues
4. Collect user feedback

#### Phase 3: Full Migration (Day 5)
1. Run migration on production database
2. Remove YAML/TOML loading (keep files as backup)
3. Enable deprecation warnings
4. Update documentation

### 7.3 Rollback Plan

If issues occur:
```rust
// Emergency rollback procedure
async fn emergency_rollback() -> Result<(), String> {
    // 1. Disable database loading
    std::env::set_var("SETTINGS_USE_FILES", "true");

    // 2. Restart server (will load from YAML/TOML)
    // No code changes required - hybrid loader handles this

    // 3. Export database settings for analysis
    let settings = db.load_all_settings()?;
    let yaml = serde_yaml::to_string(&settings)?;
    std::fs::write("rollback_dump.yaml", yaml)?;

    Ok(())
}
```

**Rollback triggers:**
- Database corruption detected
- Critical settings not accessible
- Performance degradation >20%
- More than 5 user-reported issues

---

## Code Structure Summary

### New Files to Create

```
src/
├── migration/
│   ├── mod.rs                    # Migration orchestrator
│   ├── yaml_parser.rs            # YAML file parser
│   └── toml_parser.rs            # TOML file parser
├── services/
│   ├── settings_watcher.rs       # File/DB change watcher
│   └── hybrid_settings_loader.rs # Fallback loader
├── handlers/
│   ├── settings_schema_handler.rs  # Schema API
│   └── settings_ws.rs            # WebSocket handler
└── bin/
    ├── migrate_settings.rs       # Migration CLI
    └── settings_cli.rs           # Settings management CLI

tests/
├── settings_migration_tests.rs
├── settings_hot_reload_tests.rs
└── integration/
    └── settings_api_tests.rs

docs/
└── settings-migration-plan.md    # This document
```

### Updated Files

```
src/
├── services/
│   ├── database_service.rs       # Add migration helpers
│   └── settings_service.rs       # Add hot-reload support
└── main.rs                       # Add deprecation warnings

Cargo.toml                        # Add dependencies: notify, clap
```

---

## Dependencies Required

Add to `Cargo.toml`:

```toml
[dependencies]
# Existing dependencies already present ✅
rusqlite = { version = "0.37", features = ["bundled"] }
r2d2 = "0.8"
r2d2_sqlite = "0.31"
serde_yaml = "0.9"
toml = "0.9.5"

# New dependencies for migration
notify = "6.1"                # File system watcher
clap = { version = "4.5", features = ["derive"] }  # CLI framework

[dev-dependencies]
tempfile = "3.14"
```

---

## Timeline & Milestones

### Week 1: Foundation
- **Days 1-2**: Schema validation, migration tracking
- **Days 3-5**: Migration script implementation, parsers, CLI tool

**Milestone 1**: Successfully migrate test settings.yaml and dev_config.toml to database

### Week 2: Hot-Reload & Compatibility
- **Days 1-3**: File watcher, WebSocket notifications, hot-reload
- **Days 4-5**: Backward compatibility layer, deprecation warnings

**Milestone 2**: Settings can be changed without restart, fallback works

### Week 3: Frontend & CLI
- **Days 1-4**: Settings control panel, dynamic forms, real-time updates
- **Days 5-7**: Developer CLI tool, bulk operations, watch mode

**Milestone 3**: Full-featured control panel and CLI operational

### Week 4: Testing & Launch
- **Days 1-2**: CLI completion, backup/restore, validation
- **Days 3-5**: Comprehensive testing, gradual rollout, monitoring

**Milestone 4**: Production-ready migration with monitoring

---

## Risk Mitigation

### Risk: Data Loss During Migration
**Mitigation:**
- Automatic backups before migration
- Transaction-based migration (atomic)
- Validation after migration
- Keep original YAML/TOML files

### Risk: Performance Degradation
**Mitigation:**
- Connection pooling (already implemented ✅)
- In-memory caching in SettingsService (already implemented ✅)
- Indexed database queries (already implemented ✅)
- Benchmark tests to compare performance

### Risk: Breaking Changes
**Mitigation:**
- Hybrid loader during transition
- Extensive testing with production-like data
- Gradual rollout strategy
- Rollback plan ready

### Risk: WebSocket Scalability
**Mitigation:**
- Rate limiting on notifications
- Client-side debouncing
- Optional polling fallback
- Load testing before production

---

## Success Metrics

### Migration Success
- ✅ 100% of settings migrated without data loss
- ✅ All validation checks pass
- ✅ Zero downtime during migration

### Performance
- Settings load time: <50ms (from cache)
- Database query time: <10ms (indexed)
- Hot-reload latency: <500ms
- WebSocket notification delay: <100ms

### Usability
- Control panel loads in <2s
- Search/filter results: <200ms
- CLI commands execute in <1s
- Developer satisfaction: >90% positive feedback

---

## Future Enhancements (Post-Launch)

### Version 2.1 (Q1 2026)
- **Settings Templates**: Pre-configured profiles (development, staging, production)
- **A/B Testing**: Enable/disable features for specific user segments
- **Settings History**: Time-travel debugging for settings changes
- **API Rate Limits**: Per-user, per-endpoint limits stored in DB

### Version 2.2 (Q2 2026)
- **Multi-Tenant Support**: Isolated settings per organization
- **Settings Inheritance**: Override hierarchy (global → org → user)
- **GraphQL API**: Alternative to REST for complex queries
- **Audit Compliance**: GDPR/SOC2 audit trail export

### Version 2.3 (Q3 2026)
- **Distributed Settings**: Redis-backed caching for multi-instance deployments
- **Settings Versioning**: Semantic versioning for schema changes
- **Canary Deployments**: Gradual rollout of setting changes
- **Machine Learning**: Anomaly detection for invalid settings

---

## Conclusion

This migration plan provides a comprehensive, phased approach to transitioning VisionFlow's settings from static YAML/TOML files to a dynamic SQLite database system. The strategy emphasizes:

1. **Safety**: Multiple validation layers, backups, and rollback capability
2. **Performance**: Caching, indexing, and connection pooling
3. **Developer Experience**: Powerful CLI tools and intuitive control panel
4. **Flexibility**: Hot-reload, real-time updates, and WebSocket notifications
5. **Backward Compatibility**: Hybrid loading during transition period

The existing infrastructure (DatabaseService, schema, SettingsService) provides a solid foundation. The migration focuses on building parsers, hot-reload, and developer tools around this core.

**Estimated Total Effort**: 4 weeks (1 senior developer)

**Risk Level**: Low to Medium (with mitigations in place)

**Expected Outcome**: Production-ready settings management system with zero-downtime migration

---

## Appendix A: Migration Command Reference

```bash
# Initial setup
cargo build --release --bin migrate_settings

# Run migration
./target/release/migrate_settings data/settings.db

# Verify migration
settings-cli list --format table

# Export for backup
settings-cli export pre-migration-backup.yaml

# Test hot-reload
settings-cli watch --pattern "system"

# In another terminal:
settings-cli set system.network.port 8080

# Rollback if needed (emergency)
SETTINGS_USE_FILES=true cargo run
```

## Appendix B: Schema Categories

Settings organized by category for control panel:

| Category | Example Keys | Count |
|----------|-------------|-------|
| **Visualization** | `visualisation.rendering.*` | ~50 |
| **Physics** | `visualisation.graphs.*.physics.*` | ~180 |
| **Network** | `system.network.*` | ~15 |
| **WebSocket** | `system.websocket.*` | ~12 |
| **Security** | `system.security.*` | ~10 |
| **Auth** | `auth.*` | ~3 |
| **XR** | `xr.*` | ~40 |
| **APIs** | `ragflow.*`, `perplexity.*`, etc. | ~25 |
| **Developer** | `dev.physics.*`, `dev.cuda.*` | ~50 |
| **Debug** | `debug.*` | ~10 |

**Total Settings**: ~395 keys

## Appendix C: Performance Benchmarks

Expected performance (based on similar implementations):

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Initial load (cold) | <200ms | Time to load all settings from DB |
| Subsequent load (cache) | <10ms | Time to retrieve cached setting |
| Single setting update | <50ms | Time to write + notify |
| Bulk update (100 keys) | <2s | Batched transaction |
| WebSocket broadcast | <100ms | Notification to all clients |
| CLI command | <500ms | Average execution time |
| Migration (full) | <10s | Complete YAML/TOML → DB |

---

**Document Version**: 1.0
**Created**: 2025-10-22
**Author**: Claude Code (Coder Agent)
**Status**: Ready for Implementation
