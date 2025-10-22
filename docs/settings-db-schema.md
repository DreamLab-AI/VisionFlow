# Unified Settings Database Architecture

**Version:** 1.0.0
**Date:** 2025-10-22
**Status:** Architecture Design Document

## Executive Summary

This document defines a comprehensive SQLite database schema to unify ALL application settings (visualization, physics, network, security, XR, etc.) into a single, type-safe, validated data store with automatic migration from YAML/TOML formats.

### Design Goals

1. **Single Source of Truth**: Replace scattered YAML/TOML files with one database
2. **Type Safety**: Enforce value types (string, number, boolean, JSON) with validation
3. **Validation Rules**: Min/max bounds, regex patterns, enum constraints
4. **Category Hierarchy**: Organized by domain → group → setting (e.g., `physics.forces.springK`)
5. **Version Migration**: Automatic schema evolution and data migration
6. **Performance**: Caching layer with 5-minute TTL (already implemented)
7. **Audit Trail**: Track all changes with timestamps and user context

---

## 1. Core Database Schema

### 1.1 Settings Table (Primary)

```sql
-- Core settings table with comprehensive validation
CREATE TABLE settings (
  -- Identity
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT UNIQUE NOT NULL,           -- Dot notation: "physics.forces.springK"

  -- Value Storage (JSON-encoded for flexibility)
  value TEXT NOT NULL,                 -- JSON-encoded value
  value_type TEXT NOT NULL             -- "number", "string", "boolean", "json"
    CHECK(value_type IN ('number', 'string', 'boolean', 'json')),
  default_value TEXT NOT NULL,         -- JSON-encoded default

  -- Hierarchy & Organization
  category TEXT NOT NULL,              -- Top-level: "visualization", "physics", "system", "xr"
  subcategory TEXT,                    -- Second-level: "nodes", "edges", "forces", "network"
  group_name TEXT,                     -- Third-level: "rendering", "simulation", "security"

  -- Priority & Behavior
  priority TEXT NOT NULL DEFAULT 'medium'
    CHECK(priority IN ('critical', 'high', 'medium', 'low')),
  requires_restart BOOLEAN DEFAULT 0,   -- Requires app restart to apply
  requires_gpu_reinit BOOLEAN DEFAULT 0, -- Requires GPU reinitialization
  user_visible BOOLEAN DEFAULT 1,       -- Show in UI
  developer_only BOOLEAN DEFAULT 0,     -- Dev tools only
  protected BOOLEAN DEFAULT 0,          -- Encrypted/sensitive (API keys)

  -- Validation Rules (JSON-encoded for complex rules)
  min_value REAL,                       -- Numeric minimum
  max_value REAL,                       -- Numeric maximum
  allowed_values TEXT,                  -- JSON array for enums: ["low", "medium", "high"]
  validation_regex TEXT,                -- Regex pattern for strings
  validation_rules TEXT,                -- JSON object for complex validation

  -- Metadata
  display_name TEXT,                    -- UI-friendly name: "Spring Force Constant"
  description TEXT,                     -- Help text for UI
  unit TEXT,                            -- Display unit: "ms", "px", "N/m"
  tooltip TEXT,                         -- Extended help for tooltips

  -- Dependencies
  depends_on TEXT,                      -- JSON array of setting keys this depends on
  affects TEXT,                         -- JSON array of settings affected by this

  -- Audit Trail
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_by TEXT,                      -- User pubkey or "system"
  version INTEGER DEFAULT 1             -- Schema version for migration
);

-- Indexes for performance
CREATE INDEX idx_settings_category ON settings(category);
CREATE INDEX idx_settings_subcategory ON settings(category, subcategory);
CREATE INDEX idx_settings_priority ON settings(priority);
CREATE INDEX idx_settings_user_visible ON settings(user_visible);
CREATE INDEX idx_settings_developer_only ON settings(developer_only);
CREATE INDEX idx_settings_key_lookup ON settings(key);

-- Trigger to update timestamp on modification
CREATE TRIGGER settings_updated_at
AFTER UPDATE ON settings
FOR EACH ROW
BEGIN
  UPDATE settings SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
```

---

### 1.2 Settings Groups Table (UI Organization)

```sql
-- Defines UI panel groups and their hierarchy
CREATE TABLE settings_groups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  group_key TEXT UNIQUE NOT NULL,      -- "physics.forces", "visualization.rendering"
  parent_group TEXT,                   -- Hierarchical: "physics" → "physics.forces"

  -- UI Display
  display_name TEXT NOT NULL,          -- "Force Configuration"
  icon TEXT,                           -- Icon name/path for UI
  order_index INTEGER DEFAULT 0,       -- Sort order in UI
  collapsible BOOLEAN DEFAULT 1,       -- Can collapse in UI
  expanded_default BOOLEAN DEFAULT 0,  -- Start expanded

  -- Access Control
  requires_permission TEXT,            -- "developer", "power_user", "admin"
  visible_in_context TEXT,             -- JSON array: ["desktop", "xr", "mobile"]

  -- Description
  description TEXT,
  help_url TEXT,                       -- Link to documentation

  FOREIGN KEY (parent_group) REFERENCES settings_groups(group_key)
);

CREATE INDEX idx_groups_parent ON settings_groups(parent_group);
CREATE INDEX idx_groups_order ON settings_groups(order_index);
```

---

### 1.3 Settings History Table (Audit Trail)

```sql
-- Complete audit trail of all setting changes
CREATE TABLE settings_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  setting_key TEXT NOT NULL,

  -- Change Data
  old_value TEXT,                      -- Previous JSON-encoded value
  new_value TEXT NOT NULL,             -- New JSON-encoded value
  change_type TEXT NOT NULL            -- "create", "update", "delete", "reset"
    CHECK(change_type IN ('create', 'update', 'delete', 'reset')),

  -- Context
  changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  changed_by TEXT,                     -- User pubkey, IP, or "system"
  change_reason TEXT,                  -- Optional: "user_edit", "migration", "auto_tune"
  session_id TEXT,                     -- Track session for batch changes

  -- Metadata
  client_info TEXT,                    -- JSON: {"ip": "...", "user_agent": "..."}
  rollback_id INTEGER,                 -- Link to rollback entry if reverted

  FOREIGN KEY (setting_key) REFERENCES settings(key)
);

CREATE INDEX idx_history_setting ON settings_history(setting_key);
CREATE INDEX idx_history_timestamp ON settings_history(changed_at);
CREATE INDEX idx_history_user ON settings_history(changed_by);
```

---

### 1.4 Settings Profiles Table (Named Configurations)

```sql
-- Save/load named configuration profiles
CREATE TABLE settings_profiles (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_name TEXT UNIQUE NOT NULL,

  -- Profile Data
  settings_snapshot TEXT NOT NULL,     -- JSON snapshot of all settings
  description TEXT,
  tags TEXT,                           -- JSON array: ["physics", "high-performance"]

  -- Metadata
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  created_by TEXT,
  is_system_profile BOOLEAN DEFAULT 0, -- Built-in vs user-created
  is_active BOOLEAN DEFAULT 0,         -- Currently active profile

  -- Stats
  load_count INTEGER DEFAULT 0,
  last_loaded_at TIMESTAMP
);

CREATE INDEX idx_profiles_active ON settings_profiles(is_active);
CREATE INDEX idx_profiles_system ON settings_profiles(is_system_profile);
```

---

### 1.5 Settings Validation Rules Table (Complex Validation)

```sql
-- Define complex validation rules for settings
CREATE TABLE validation_rules (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  rule_name TEXT UNIQUE NOT NULL,

  -- Rule Definition
  rule_type TEXT NOT NULL              -- "range", "regex", "enum", "custom", "cross_field"
    CHECK(rule_type IN ('range', 'regex', 'enum', 'custom', 'cross_field')),
  rule_config TEXT NOT NULL,           -- JSON configuration for the rule

  -- Application
  applies_to TEXT NOT NULL,            -- JSON array of setting keys or patterns
  error_message TEXT NOT NULL,         -- User-friendly error message
  severity TEXT DEFAULT 'error'        -- "error", "warning", "info"
    CHECK(severity IN ('error', 'warning', 'info')),

  -- Metadata
  description TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  enabled BOOLEAN DEFAULT 1
);

CREATE INDEX idx_validation_type ON validation_rules(rule_type);
CREATE INDEX idx_validation_enabled ON validation_rules(enabled);
```

---

### 1.6 Settings Schema Migrations Table

```sql
-- Track database schema versions and migrations
CREATE TABLE schema_migrations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  version INTEGER UNIQUE NOT NULL,

  -- Migration Details
  migration_name TEXT NOT NULL,
  migration_sql TEXT NOT NULL,         -- SQL statements executed

  -- Status
  applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  execution_time_ms INTEGER,
  status TEXT DEFAULT 'success'        -- "success", "failed", "rolled_back"
    CHECK(status IN ('success', 'failed', 'rolled_back')),
  error_message TEXT,

  -- Rollback
  rollback_sql TEXT,                   -- SQL to undo this migration
  rolled_back_at TIMESTAMP
);

CREATE INDEX idx_migrations_version ON schema_migrations(version);
CREATE INDEX idx_migrations_status ON schema_migrations(status);
```

---

## 2. Category Hierarchy Design

### 2.1 Top-Level Categories

| Category         | Subcategories                               | Purpose                                    |
|------------------|---------------------------------------------|--------------------------------------------|
| `visualization`  | rendering, animations, glow, hologram       | Visual rendering and effects               |
| `physics`        | forces, dynamics, boundaries, constraints   | Physics simulation parameters              |
| `graphs`         | logseq, visionflow                          | Graph-specific settings (multi-graph)      |
| `nodes`          | appearance, behavior, interaction           | Node visualization and behavior            |
| `edges`          | appearance, behavior, flow                  | Edge visualization and behavior            |
| `labels`         | appearance, behavior, positioning           | Label rendering                            |
| `system`         | websocket, debug, performance               | System-level configuration                 |
| `network`        | connection, security, rate_limit            | Network and security settings              |
| `xr`             | hand_tracking, locomotion, environment      | XR/AR/VR settings                          |
| `auth`           | nostr, providers, sessions                  | Authentication and authorization           |
| `services`       | ragflow, perplexity, openai, kokoro         | External service integrations              |
| `developer`      | gpu, constraints, debugging                 | Developer tools and diagnostics            |
| `analytics`      | clustering, metrics, performance            | Analytics and monitoring                   |

---

### 2.2 Example Hierarchical Keys

```
physics.forces.springK              → Spring force constant (0.1-10.0)
physics.forces.repelK               → Repulsion force constant (1-1000)
physics.dynamics.dt                 → Time step for integration (0.01-1.0)
physics.boundaries.size             → Viewport boundary size (100-5000)

visualization.rendering.backgroundColor  → Background color (#RRGGBB)
visualization.animations.pulseSpeed      → Animation pulse speed (0-10)
visualization.glow.intensity             → Glow effect intensity (0-10)

graphs.logseq.nodes.baseColor       → Node color for Logseq graph
graphs.visionflow.physics.enabled   → Enable physics for Visionflow

system.websocket.reconnectAttempts  → WebSocket reconnection attempts (1-10)
system.debug.enabled                → Enable debug mode (boolean)

xr.handTracking.enabled             → Enable hand tracking (boolean)
xr.locomotion.method                → Locomotion method (enum: teleport/continuous)

services.ragflow.apiKey             → RAGFlow API key (protected, encrypted)
services.perplexity.model           → Perplexity model name (string)
```

---

## 3. Validation Rule Engine

### 3.1 Validation Rule Types

#### A. Range Validation (Numeric)

```json
{
  "type": "range",
  "min": 0.1,
  "max": 10.0,
  "step": 0.1,
  "default": 0.5,
  "unit": "N/m"
}
```

#### B. Regex Validation (String)

```json
{
  "type": "regex",
  "pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$",
  "error": "Must be valid hex color (#RRGGBB or #RRGGBBAA)"
}
```

#### C. Enum Validation (Allowed Values)

```json
{
  "type": "enum",
  "allowed_values": ["low", "medium", "high"],
  "default": "medium",
  "case_sensitive": false
}
```

#### D. Cross-Field Validation

```json
{
  "type": "cross_field",
  "rule": "physics.forces.springK < physics.forces.repelK",
  "error": "Spring force must be less than repulsion force"
}
```

#### E. Custom Validation (Function)

```json
{
  "type": "custom",
  "function": "validate_bloom_glow_settings",
  "params": ["glow.intensity", "bloom.threshold"],
  "error": "Bloom/Glow settings conflict detected"
}
```

---

### 3.2 Validation Rule Examples

```sql
-- Example 1: Physics spring force range
INSERT INTO validation_rules (rule_name, rule_type, rule_config, applies_to, error_message) VALUES
(
  'physics_spring_force_range',
  'range',
  '{"min": 0.1, "max": 10.0, "step": 0.1}',
  '["physics.forces.springK"]',
  'Spring force must be between 0.1 and 10.0'
);

-- Example 2: Color hex validation
INSERT INTO validation_rules (rule_name, rule_type, rule_config, applies_to, error_message) VALUES
(
  'color_hex_format',
  'regex',
  '{"pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$"}',
  '["*.baseColor", "*.backgroundColor", "*.color"]',
  'Color must be valid hex format (#RRGGBB or #RRGGBBAA)'
);

-- Example 3: Quality enum
INSERT INTO validation_rules (rule_name, rule_type, rule_config, applies_to, error_message) VALUES
(
  'quality_enum',
  'enum',
  '{"allowed_values": ["low", "medium", "high"]}',
  '["*.quality"]',
  'Quality must be one of: low, medium, high'
);

-- Example 4: Cross-field GPU stability
INSERT INTO validation_rules (rule_name, rule_type, rule_config, applies_to, error_message) VALUES
(
  'gpu_stability_check',
  'custom',
  '{"function": "validate_bloom_glow_settings", "params": ["glow", "bloom"]}',
  '["visualization.glow.*", "visualization.bloom.*"]',
  'Bloom/Glow settings must not exceed GPU kernel safe ranges'
);
```

---

## 4. Migration Strategy from YAML/TOML

### 4.1 Migration Architecture

```
Current State:                     Target State:
┌─────────────────┐                ┌─────────────────┐
│ settings.yaml   │                │                 │
│ protected.toml  │   ──────────►  │  settings.db    │
│ user_xyz.yaml   │                │  (SQLite)       │
│ dev_config.rs   │                │                 │
└─────────────────┘                └─────────────────┘
```

### 4.2 Migration SQL Script

```sql
-- Migration script to populate settings table from existing configs

-- Step 1: Create temporary staging table
CREATE TEMP TABLE staging_settings (
  key TEXT PRIMARY KEY,
  value TEXT,
  value_type TEXT,
  category TEXT,
  subcategory TEXT
);

-- Step 2: Insert physics settings (example)
INSERT INTO staging_settings (key, value, value_type, category, subcategory) VALUES
  ('physics.enabled', 'true', 'boolean', 'physics', 'core'),
  ('physics.forces.springK', '0.5', 'number', 'physics', 'forces'),
  ('physics.forces.repelK', '100', 'number', 'physics', 'forces'),
  ('physics.dynamics.dt', '0.2', 'number', 'physics', 'dynamics'),
  ('physics.dynamics.damping', '0.5', 'number', 'physics', 'dynamics'),
  ('physics.boundaries.size', '1000', 'number', 'physics', 'boundaries'),
  ('physics.boundaries.enabled', 'true', 'boolean', 'physics', 'boundaries');

-- Step 3: Insert visualization settings
INSERT INTO staging_settings (key, value, value_type, category, subcategory) VALUES
  ('visualization.rendering.backgroundColor', '"#000000"', 'string', 'visualization', 'rendering'),
  ('visualization.animations.pulseSpeed', '1.0', 'number', 'visualization', 'animations'),
  ('visualization.glow.enabled', 'true', 'boolean', 'visualization', 'glow'),
  ('visualization.glow.intensity', '2.0', 'number', 'visualization', 'glow');

-- Step 4: Transfer to main settings table with full metadata
INSERT INTO settings (
  key, value, value_type, default_value, category, subcategory,
  priority, user_visible, display_name, description, min_value, max_value
)
SELECT
  key,
  value,
  value_type,
  value as default_value,
  category,
  subcategory,
  'medium' as priority,
  1 as user_visible,
  replace(replace(key, '_', ' '), '.', ' → ') as display_name,
  'Migrated from YAML/TOML configuration' as description,
  CASE value_type
    WHEN 'number' THEN 0.0
    ELSE NULL
  END as min_value,
  CASE value_type
    WHEN 'number' THEN 10000.0
    ELSE NULL
  END as max_value
FROM staging_settings;

-- Step 5: Drop staging table
DROP TABLE staging_settings;

-- Step 6: Record migration
INSERT INTO schema_migrations (version, migration_name, migration_sql, status)
VALUES (1, 'initial_settings_migration', '-- SQL above --', 'success');
```

---

### 4.3 Rust Migration Tool

```rust
// scripts/migrate_legacy_configs.rs
use rusqlite::{Connection, Result};
use serde_json::Value;
use std::fs;

pub fn migrate_yaml_to_sqlite(
    yaml_path: &str,
    db_path: &str,
) -> Result<()> {
    // Load YAML
    let yaml_content = fs::read_to_string(yaml_path)?;
    let yaml_value: Value = serde_yaml::from_str(&yaml_content)
        .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

    // Open database
    let conn = Connection::open(db_path)?;

    // Recursive flatten and insert
    fn insert_recursive(
        conn: &Connection,
        prefix: &str,
        value: &Value,
        category: &str,
    ) -> Result<()> {
        match value {
            Value::Object(map) => {
                for (key, val) in map {
                    let full_key = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", prefix, key)
                    };
                    insert_recursive(conn, &full_key, val, category)?;
                }
            }
            Value::String(s) => {
                conn.execute(
                    "INSERT INTO settings (key, value, value_type, default_value, category)
                     VALUES (?1, ?2, 'string', ?2, ?3)",
                    [prefix, &format!("\"{}\"", s), category],
                )?;
            }
            Value::Number(n) => {
                conn.execute(
                    "INSERT INTO settings (key, value, value_type, default_value, category)
                     VALUES (?1, ?2, 'number', ?2, ?3)",
                    [prefix, &n.to_string(), category],
                )?;
            }
            Value::Bool(b) => {
                conn.execute(
                    "INSERT INTO settings (key, value, value_type, default_value, category)
                     VALUES (?1, ?2, 'boolean', ?2, ?3)",
                    [prefix, &b.to_string(), category],
                )?;
            }
            _ => {}
        }
        Ok(())
    }

    insert_recursive(&conn, "", &yaml_value, "general")?;
    Ok(())
}
```

---

## 5. CLI Interface Specification

### 5.1 Command Structure

```bash
# Get a setting
settings get <key>
settings get physics.forces.springK

# Set a setting
settings set <key> <value>
settings set physics.forces.springK 0.8

# List settings by category
settings list [--category <cat>] [--format json|table]
settings list --category physics
settings list --category visualization.rendering --format json

# Reset to default
settings reset <key>
settings reset physics.forces.springK
settings reset --all-physics  # Reset entire category

# Validate settings
settings validate [--key <key>] [--fix]
settings validate --key physics.forces.springK
settings validate --all --fix  # Validate all and auto-fix

# Import/Export
settings export [--output <file>] [--category <cat>]
settings export --output physics_settings.json --category physics
settings import --input settings.json [--merge]

# Profiles
settings profile save <name>
settings profile load <name>
settings profile list
settings profile delete <name>

# History
settings history [--key <key>] [--limit <n>]
settings history --key physics.forces.springK --limit 10
settings rollback <history_id>

# Migration
settings migrate --from-yaml settings.yaml
settings migrate --from-toml protected.toml

# Cache management
settings cache clear
settings cache stats
```

---

### 5.2 CLI Output Examples

```bash
$ settings get physics.forces.springK

Key:           physics.forces.springK
Value:         0.5
Type:          number
Default:       0.5
Range:         0.1 - 10.0
Category:      physics → forces
Description:   Spring force constant for connected nodes
Requires:      GPU reinitialization
Last Updated:  2025-10-22 14:30:15
Updated By:    user_pubkey_abc123

$ settings list --category physics --format table

┌─────────────────────────────┬────────┬───────┬─────────────────┐
│ Key                         │ Value  │ Type  │ Description     │
├─────────────────────────────┼────────┼───────┼─────────────────┤
│ physics.enabled             │ true   │ bool  │ Enable physics  │
│ physics.forces.springK      │ 0.5    │ num   │ Spring constant │
│ physics.forces.repelK       │ 100    │ num   │ Repulsion force │
│ physics.dynamics.dt         │ 0.2    │ num   │ Time step       │
│ physics.boundaries.size     │ 1000   │ num   │ Boundary size   │
└─────────────────────────────┴────────┴───────┴─────────────────┘

$ settings validate --all

✓ physics.forces.springK: VALID (0.5 in range 0.1-10.0)
✗ visualization.glow.intensity: INVALID (15.0 exceeds max 10.0)
✗ edges.color: INVALID ("#gggggg" is not valid hex color)
✓ system.websocket.reconnectAttempts: VALID (3)

Summary: 2 errors, 0 warnings, 147 valid settings

$ settings history --key physics.forces.springK --limit 5

ID      Date                   User            Old Value  New Value
─────────────────────────────────────────────────────────────────
1234    2025-10-22 14:30:15    user_abc123     0.5        0.8
1198    2025-10-22 12:15:32    system          0.3        0.5
1056    2025-10-21 18:45:00    user_def456     0.5        0.3
0987    2025-10-21 10:22:11    system          0.5        0.5
0876    2025-10-20 09:00:00    migration       NULL       0.5
```

---

## 6. UI Panel Integration

### 6.1 Settings Panel Hierarchy

```
Settings Panel (Sidebar)
├─ 🎨 Visualization
│  ├─ Rendering
│  │  ├─ Background Color
│  │  ├─ Ambient Light
│  │  └─ Shadows
│  ├─ Animations
│  │  ├─ Pulse Speed
│  │  └─ Motion Blur
│  └─ Effects
│     ├─ Glow
│     └─ Hologram
│
├─ ⚡ Physics
│  ├─ Forces
│  │  ├─ Spring Constant (slider: 0.1-10.0)
│  │  └─ Repulsion Force (slider: 1-1000)
│  ├─ Dynamics
│  │  ├─ Time Step
│  │  └─ Damping
│  └─ Boundaries
│     ├─ Enabled (toggle)
│     └─ Size (slider)
│
├─ 🕸️ Graphs
│  ├─ Logseq
│  │  ├─ Nodes
│  │  └─ Physics
│  └─ VisionFlow
│     ├─ Nodes
│     └─ Physics
│
├─ 🔧 System
│  ├─ WebSocket
│  ├─ Performance
│  └─ Debug
│
└─ 🥽 XR Settings
   ├─ Hand Tracking
   └─ Locomotion
```

---

### 6.2 UI Component Mapping

```typescript
// Type definition for UI components
interface SettingUIComponent {
  key: string;
  displayName: string;
  componentType: 'slider' | 'toggle' | 'input' | 'select' | 'color';
  props: {
    min?: number;
    max?: number;
    step?: number;
    options?: string[];
    placeholder?: string;
  };
  validation: ValidationRule[];
  tooltip: string;
  dependsOn?: string[];  // Other settings that affect this one
}

// Example: Physics spring force slider
const springKSetting: SettingUIComponent = {
  key: 'physics.forces.springK',
  displayName: 'Spring Force Constant',
  componentType: 'slider',
  props: {
    min: 0.1,
    max: 10.0,
    step: 0.1
  },
  validation: [
    { type: 'range', min: 0.1, max: 10.0 }
  ],
  tooltip: 'Controls the attractive force between connected nodes. Higher values create tighter clusters.',
  dependsOn: ['physics.enabled']
};
```

---

### 6.3 React Component Example

```typescript
// Auto-generated settings panel component
import { useSettings } from '@/hooks/useSettings';

export function PhysicsSettingsPanel() {
  const { getSetting, setSetting, validateSetting } = useSettings();

  const springK = getSetting<number>('physics.forces.springK');
  const repelK = getSetting<number>('physics.forces.repelK');

  const handleSpringKChange = (value: number) => {
    const validation = validateSetting('physics.forces.springK', value);
    if (validation.valid) {
      setSetting('physics.forces.springK', value);
    } else {
      toast.error(validation.error);
    }
  };

  return (
    <SettingsGroup title="Forces" icon={ForceIcon}>
      <SettingSlider
        label="Spring Constant"
        value={springK}
        min={0.1}
        max={10.0}
        step={0.1}
        onChange={handleSpringKChange}
        tooltip="Attractive force between connected nodes"
        unit="N/m"
      />
      <SettingSlider
        label="Repulsion Force"
        value={repelK}
        min={1}
        max={1000}
        step={1}
        onChange={(v) => setSetting('physics.forces.repelK', v)}
        tooltip="Repulsive force between all nodes"
      />
    </SettingsGroup>
  );
}
```

---

## 7. API Surface Design

### 7.1 Rust API (Backend)

```rust
// src/ports/settings_repository.rs
#[async_trait]
pub trait SettingsRepository: Send + Sync {
    // Core CRUD
    async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>>;
    async fn set_setting(&self, key: &str, value: SettingValue, description: Option<&str>) -> Result<()>;
    async fn delete_setting(&self, key: &str) -> Result<()>;

    // Batch operations
    async fn get_settings_batch(&self, keys: &[String]) -> Result<HashMap<String, SettingValue>>;
    async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> Result<()>;

    // Category operations
    async fn get_settings_by_category(&self, category: &str) -> Result<Vec<Setting>>;
    async fn reset_category(&self, category: &str) -> Result<()>;

    // Validation
    async fn validate_setting(&self, key: &str, value: &SettingValue) -> Result<ValidationResult>;
    async fn validate_all(&self) -> Result<Vec<ValidationResult>>;

    // Profiles
    async fn save_profile(&self, name: &str, settings: &HashMap<String, SettingValue>) -> Result<()>;
    async fn load_profile(&self, name: &str) -> Result<HashMap<String, SettingValue>>;
    async fn list_profiles(&self) -> Result<Vec<String>>;

    // History
    async fn get_history(&self, key: Option<&str>, limit: usize) -> Result<Vec<SettingHistoryEntry>>;
    async fn rollback(&self, history_id: i64) -> Result<()>;

    // Cache
    async fn clear_cache(&self) -> Result<()>;
}

// Value enum with type safety
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(serde_json::Value),
}
```

---

### 7.2 HTTP API (REST Endpoints)

```
# Core Settings API
GET    /api/settings/:key                   # Get single setting
PUT    /api/settings/:key                   # Update single setting
DELETE /api/settings/:key                   # Delete setting
POST   /api/settings/batch                  # Batch get/set

# Category Operations
GET    /api/settings/category/:category     # Get all in category
POST   /api/settings/category/:category/reset  # Reset category

# Validation
POST   /api/settings/validate                # Validate settings
GET    /api/settings/validate/:key          # Validate single setting

# Profiles
GET    /api/settings/profiles               # List profiles
POST   /api/settings/profiles/:name         # Save profile
GET    /api/settings/profiles/:name         # Load profile
DELETE /api/settings/profiles/:name         # Delete profile

# History & Audit
GET    /api/settings/history                # Get change history
POST   /api/settings/rollback/:id           # Rollback to version

# Export/Import
GET    /api/settings/export                 # Export as JSON
POST   /api/settings/import                 # Import from JSON
```

---

### 7.3 WebSocket API (Real-time Updates)

```typescript
// Subscribe to setting changes
ws.send({
  type: 'settings:subscribe',
  keys: ['physics.forces.*', 'visualization.glow.*']
});

// Receive updates
ws.on('settings:changed', (event) => {
  console.log(`Setting ${event.key} changed: ${event.oldValue} → ${event.newValue}`);
});

// Batch update with validation
ws.send({
  type: 'settings:update_batch',
  settings: {
    'physics.forces.springK': 0.8,
    'physics.dynamics.dt': 0.15
  },
  validate: true
});
```

---

## 8. Example Seed Data

```sql
-- Insert core physics settings
INSERT INTO settings (key, value, value_type, default_value, category, subcategory, priority, display_name, description, min_value, max_value, unit, user_visible, requires_gpu_reinit) VALUES

-- Physics Forces
('physics.enabled', 'true', 'boolean', 'true', 'physics', 'core', 'critical', 'Enable Physics', 'Master toggle for physics simulation', NULL, NULL, NULL, 1, 1),
('physics.forces.springK', '0.5', 'number', '0.5', 'physics', 'forces', 'high', 'Spring Constant', 'Attractive force between connected nodes', 0.1, 10.0, 'N/m', 1, 1),
('physics.forces.repelK', '100', 'number', '100', 'physics', 'forces', 'high', 'Repulsion Force', 'Repulsive force between all nodes', 1.0, 1000.0, 'N', 1, 1),
('physics.forces.centerGravityK', '0.05', 'number', '0.05', 'physics', 'forces', 'medium', 'Center Gravity', 'Attractive force toward center', 0.0, 1.0, NULL, 1, 1),

-- Physics Dynamics
('physics.dynamics.dt', '0.2', 'number', '0.2', 'physics', 'dynamics', 'high', 'Time Step', 'Integration time step (lower = more stable)', 0.01, 1.0, 'ms', 1, 1),
('physics.dynamics.damping', '0.5', 'number', '0.5', 'physics', 'dynamics', 'high', 'Damping', 'Velocity damping factor', 0.0, 1.0, NULL, 1, 1),
('physics.dynamics.maxVelocity', '500', 'number', '500', 'physics', 'dynamics', 'medium', 'Max Velocity', 'Maximum node velocity', 1.0, 10000.0, 'px/s', 1, 0),

-- Physics Boundaries
('physics.boundaries.enabled', 'true', 'boolean', 'true', 'physics', 'boundaries', 'high', 'Enable Boundaries', 'Confine nodes within viewport', NULL, NULL, NULL, 1, 0),
('physics.boundaries.size', '1000', 'number', '1000', 'physics', 'boundaries', 'medium', 'Boundary Size', 'Viewport boundary radius', 100.0, 5000.0, 'px', 1, 0),
('physics.boundaries.damping', '0.9', 'number', '0.9', 'physics', 'boundaries', 'low', 'Boundary Damping', 'Velocity reduction at boundaries', 0.5, 1.0, NULL, 1, 0),

-- Visualization Rendering
('visualization.rendering.backgroundColor', '"#000000"', 'string', '"#000000"', 'visualization', 'rendering', 'low', 'Background Color', 'Scene background color', NULL, NULL, NULL, 1, 0),
('visualization.rendering.ambientLightIntensity', '0.5', 'number', '0.5', 'visualization', 'rendering', 'low', 'Ambient Light', 'Ambient light intensity', 0.0, 2.0, NULL, 1, 0),
('visualization.rendering.enableShadows', 'false', 'boolean', 'false', 'visualization', 'rendering', 'low', 'Enable Shadows', 'Render shadows (performance cost)', NULL, NULL, NULL, 1, 0),

-- Visualization Glow
('visualization.glow.enabled', 'true', 'boolean', 'true', 'visualization', 'glow', 'medium', 'Enable Glow', 'Enable glow post-processing effect', NULL, NULL, NULL, 1, 0),
('visualization.glow.intensity', '2.0', 'number', '2.0', 'visualization', 'glow', 'medium', 'Glow Intensity', 'Glow effect intensity', 0.0, 10.0, NULL, 1, 0),
('visualization.glow.threshold', '0.8', 'number', '0.8', 'visualization', 'glow', 'low', 'Glow Threshold', 'Brightness threshold for glow', 0.0, 1.0, NULL, 1, 0);

-- Insert validation rules
INSERT INTO validation_rules (rule_name, rule_type, rule_config, applies_to, error_message, severity) VALUES
('hex_color_format', 'regex', '{"pattern": "^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$"}', '["*.backgroundColor", "*.baseColor", "*.color"]', 'Must be valid hex color (#RRGGBB or #RRGGBBAA)', 'error'),
('positive_number', 'range', '{"min": 0.0}', '["physics.forces.*", "visualization.glow.intensity"]', 'Value must be positive', 'error'),
('percentage_range', 'range', '{"min": 0.0, "max": 1.0}', '["*.opacity", "*.damping", "*.threshold"]', 'Value must be between 0.0 and 1.0', 'error');

-- Insert settings groups for UI
INSERT INTO settings_groups (group_key, parent_group, display_name, icon, order_index, description) VALUES
('physics', NULL, 'Physics Simulation', 'atom', 1, 'Physics simulation parameters'),
('physics.forces', 'physics', 'Forces', 'magnet', 1, 'Force configuration'),
('physics.dynamics', 'physics', 'Dynamics', 'activity', 2, 'Dynamics and integration'),
('physics.boundaries', 'physics', 'Boundaries', 'box', 3, 'Boundary constraints'),
('visualization', NULL, 'Visualization', 'eye', 2, 'Visual rendering settings'),
('visualization.rendering', 'visualization', 'Rendering', 'monitor', 1, 'Core rendering'),
('visualization.glow', 'visualization', 'Glow Effects', 'zap', 2, 'Glow post-processing');
```

---

## 9. Migration Checklist

### Phase 1: Database Setup (Week 1)
- [ ] Create SQLite database schema
- [ ] Implement Rust SQLite adapter
- [ ] Add migration from existing YAML/TOML
- [ ] Write comprehensive tests
- [ ] Document schema

### Phase 2: API Integration (Week 2)
- [ ] Implement REST endpoints
- [ ] Add WebSocket real-time updates
- [ ] Create validation engine
- [ ] Add caching layer
- [ ] Write API tests

### Phase 3: CLI Tool (Week 3)
- [ ] Implement CLI commands
- [ ] Add import/export functionality
- [ ] Create profile management
- [ ] Add history/rollback
- [ ] Write CLI tests

### Phase 4: UI Integration (Week 4)
- [ ] Create React settings components
- [ ] Implement auto-generated UI panels
- [ ] Add real-time WebSocket sync
- [ ] Create settings search/filter
- [ ] Add validation feedback

### Phase 5: Migration & Deployment (Week 5)
- [ ] Run migration on production data
- [ ] Monitor performance
- [ ] Fix bugs and edge cases
- [ ] Update documentation
- [ ] Train users

---

## 10. Performance Considerations

### 10.1 Caching Strategy

```rust
// Existing cache implementation (5-minute TTL)
// src/adapters/sqlite_settings_repository.rs

struct SettingsCache {
    settings: HashMap<String, CachedSetting>,
    last_updated: std::time::Instant,
    ttl_seconds: u64,  // 300 seconds (5 minutes)
}

// Cache hit ratio: Target 95%+
// Cache miss penalty: ~5ms (SQLite query)
// Memory overhead: ~50KB for 1000 settings
```

### 10.2 Query Optimization

```sql
-- Use indexes for common queries
CREATE INDEX idx_settings_category_subcategory
  ON settings(category, subcategory);
CREATE INDEX idx_settings_user_visible
  ON settings(user_visible) WHERE user_visible = 1;

-- Compound index for UI queries
CREATE INDEX idx_settings_ui_lookup
  ON settings(category, user_visible, priority);
```

### 10.3 Batch Operations

```rust
// Batch updates reduce round-trips
async fn set_settings_batch(
    &self,
    updates: HashMap<String, SettingValue>
) -> Result<()> {
    let mut tx = self.db.transaction()?;
    for (key, value) in updates {
        tx.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
            [&key, &value.to_json()]
        )?;
    }
    tx.commit()?;
    self.invalidate_cache_batch(updates.keys()).await;
    Ok(())
}
```

---

## 11. Security Considerations

### 11.1 Protected Settings

```sql
-- Encrypted storage for sensitive settings
CREATE TABLE protected_settings (
  id INTEGER PRIMARY KEY,
  key TEXT UNIQUE NOT NULL,
  encrypted_value BLOB NOT NULL,  -- AES-256-GCM encrypted
  encryption_key_id TEXT NOT NULL,
  nonce BLOB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  accessed_by TEXT,
  last_accessed TIMESTAMP
);

-- Examples: API keys, passwords, tokens
INSERT INTO settings (key, value_type, protected, description) VALUES
  ('services.ragflow.apiKey', 'string', 1, 'RAGFlow API key (encrypted)'),
  ('services.openai.apiKey', 'string', 1, 'OpenAI API key (encrypted)');
```

### 11.2 Access Control

```rust
// Permission-based access
pub enum SettingPermission {
    Read,
    Write,
    Delete,
    Admin,  // Can access developer_only and protected settings
}

pub trait SettingsRepository {
    async fn check_permission(
        &self,
        user_pubkey: &str,
        key: &str,
        permission: SettingPermission
    ) -> Result<bool>;
}
```

---

## 12. Future Enhancements

### 12.1 Machine Learning Integration

```sql
-- Auto-tuning history for ML training
CREATE TABLE settings_autotuning (
  id INTEGER PRIMARY KEY,
  setting_key TEXT NOT NULL,
  graph_size INTEGER,
  graph_density REAL,
  suggested_value TEXT,
  performance_score REAL,
  applied BOOLEAN DEFAULT 0,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Use for reinforcement learning to optimize physics parameters
```

### 12.2 Multi-Tenancy Support

```sql
-- Per-user or per-workspace settings
ALTER TABLE settings ADD COLUMN workspace_id TEXT;
ALTER TABLE settings ADD COLUMN user_pubkey TEXT;
CREATE INDEX idx_settings_workspace ON settings(workspace_id);
```

### 12.3 Settings Diffing

```sql
-- Compare two profiles or versions
SELECT
  a.key,
  a.value as profile_a_value,
  b.value as profile_b_value
FROM settings a
LEFT JOIN settings b ON a.key = b.key
WHERE a.profile_id = 1 AND b.profile_id = 2
  AND a.value != b.value;
```

---

## Conclusion

This unified settings database architecture provides:

1. **Single Source of Truth**: All settings in one SQLite database
2. **Type Safety**: Strongly typed values with validation
3. **Hierarchical Organization**: Clear category structure
4. **Validation Engine**: Comprehensive rule system
5. **Migration Path**: Smooth transition from YAML/TOML
6. **Performance**: Caching with 5-minute TTL
7. **Audit Trail**: Complete change history
8. **API Surface**: REST, WebSocket, and Rust interfaces
9. **UI Integration**: Auto-generated settings panels
10. **CLI Tools**: Comprehensive command-line interface

**Next Steps:**
1. Review and approve architecture
2. Implement Phase 1 (database schema)
3. Begin migration from existing YAML/TOML files
4. Deploy incrementally with feature flags
5. Monitor performance and gather feedback

---

**Document Status:** ✅ Complete
**Approval Required:** System Architect, Backend Lead, Frontend Lead
**Implementation Timeline:** 5 weeks
**Risk Level:** Medium (requires careful migration)
