# Settings Database Schema

## Overview

The settings system uses SQLite as the persistence layer, with a normalized schema designed for performance, scalability, and data integrity.

## Database File Location

```
/app/data/settings.db
```

## Schema Version

**Current Version:** 2.0
**Last Updated:** 2025-10-17

## Tables

### 1. `settings` - Global Default Settings

Stores the default configuration values that apply to all users unless overridden.

```sql
CREATE TABLE settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,           -- e.g., 'visualisation', 'system', 'physics'
    key TEXT NOT NULL UNIQUE,         -- Dot-separated path: 'visualisation.rendering.ambient_light_intensity'
    value TEXT NOT NULL,              -- JSON-encoded value
    value_type TEXT NOT NULL,         -- 'string', 'number', 'boolean', 'object', 'array'
    description TEXT,                 -- Human-readable description
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_settings_category ON settings(category);
CREATE INDEX idx_settings_key ON settings(key);
CREATE INDEX idx_settings_updated ON settings(updated_at DESC);
```

**Example Rows:**

| id  | category       | key                                                | value | value_type | description                |
|-----|----------------|----------------------------------------------------|-------|------------|----------------------------|
| 1   | visualisation  | visualisation.rendering.ambient_light_intensity    | 0.5   | number     | Ambient light intensity    |
| 2   | visualisation  | visualisation.rendering.background_color           | "#000000" | string | Background color hex       |
| 3   | system         | system.network.port                                | 4000  | number     | Server port                |

---

### 2. `user_settings` - User-Specific Overrides

Stores per-user configuration overrides. When a user sets a preference, it's stored here and takes precedence over global defaults.

```sql
CREATE TABLE user_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,            -- Nostr public key (hex format)
    key TEXT NOT NULL,                -- Same format as settings.key
    value TEXT NOT NULL,              -- JSON-encoded value
    value_type TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, key)              -- One override per user per key
);

-- Indexes
CREATE INDEX idx_user_settings_user ON user_settings(user_id);
CREATE INDEX idx_user_settings_user_key ON user_settings(user_id, key);
CREATE INDEX idx_user_settings_key ON user_settings(key);
```

**Example Rows:**

| id  | user_id     | key                                             | value | value_type |
|-----|-------------|-------------------------------------------------|-------|------------|
| 1   | abc123...   | visualisation.rendering.ambient_light_intensity | 0.8   | number     |
| 2   | abc123...   | visualisation.graphs.logseq.physics.spring_k    | 5.0   | number     |
| 3   | def456...   | system.websocket.update_rate                    | 30    | number     |

---

### 3. `users` - User Accounts and Permissions

Stores user account information, authentication details, and permission grants.

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pubkey TEXT NOT NULL UNIQUE,      -- Nostr public key (primary identifier)
    display_name TEXT,                -- Optional display name
    is_power_user BOOLEAN DEFAULT FALSE,  -- Permission to modify settings
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    metadata TEXT                     -- JSON blob for additional user data
);

-- Indexes
CREATE INDEX idx_users_pubkey ON users(pubkey);
CREATE INDEX idx_users_power ON users(is_power_user);
```

**Example Rows:**

| id  | pubkey      | display_name | is_power_user | last_login          |
|-----|-------------|--------------|---------------|---------------------|
| 1   | abc123...   | Alice        | TRUE          | 2025-10-17 10:30:00 |
| 2   | def456...   | Bob          | FALSE         | 2025-10-17 09:15:00 |
| 3   | ghi789...   | Charlie      | TRUE          | 2025-10-17 11:00:00 |

---

### 4. `physics_settings` - Physics Configuration by Graph Type

Stores physics engine parameters separately for performance optimization and easier querying.

```sql
CREATE TABLE physics_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_type TEXT NOT NULL,         -- 'logseq', 'visionflow', 'ontology'
    user_id TEXT,                     -- NULL for global, pubkey for user override
    spring_k REAL NOT NULL DEFAULT 10.0,
    repel_k REAL NOT NULL DEFAULT 100.0,
    damping REAL NOT NULL DEFAULT 0.1,
    dt REAL NOT NULL DEFAULT 0.016,
    max_velocity REAL NOT NULL DEFAULT 50.0,
    max_force REAL NOT NULL DEFAULT 500.0,
    iterations INTEGER NOT NULL DEFAULT 50,
    mass_scale REAL NOT NULL DEFAULT 1.0,
    boundary_damping REAL NOT NULL DEFAULT 0.95,
    bounds_size REAL NOT NULL DEFAULT 1000.0,
    separation_radius REAL NOT NULL DEFAULT 2.0,
    enable_bounds BOOLEAN DEFAULT FALSE,
    enabled BOOLEAN DEFAULT TRUE,
    compute_mode INTEGER DEFAULT 1,   -- 0=CPU, 1=GPU, 2=Hybrid
    rest_length REAL NOT NULL DEFAULT 50.0,
    repulsion_cutoff REAL NOT NULL DEFAULT 50.0,
    repulsion_softening_epsilon REAL NOT NULL DEFAULT 0.0001,
    center_gravity_k REAL NOT NULL DEFAULT 0.0,
    grid_cell_size REAL NOT NULL DEFAULT 50.0,
    warmup_iterations INTEGER NOT NULL DEFAULT 100,
    cooling_rate REAL NOT NULL DEFAULT 0.001,
    warmup_curve TEXT NOT NULL DEFAULT 'exponential',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(graph_type, user_id)
);

-- Indexes
CREATE INDEX idx_physics_graph ON physics_settings(graph_type);
CREATE INDEX idx_physics_user ON physics_settings(user_id);
CREATE INDEX idx_physics_graph_user ON physics_settings(graph_type, user_id);
```

**Example Rows:**

| id  | graph_type | user_id  | spring_k | repel_k | damping |
|-----|------------|----------|----------|---------|---------|
| 1   | logseq     | NULL     | 10.0     | 100.0   | 0.1     |
| 2   | visionflow | NULL     | 8.0      | 80.0    | 0.15    |
| 3   | ontology   | NULL     | 12.0     | 120.0   | 0.08    |
| 4   | logseq     | abc123...| 5.0      | 50.0    | 0.2     |

---

### 5. `settings_audit_log` - Change Tracking

Comprehensive audit log for all settings modifications, including who changed what and when.

```sql
CREATE TABLE settings_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,            -- User who made the change
    action TEXT NOT NULL,             -- 'create', 'update', 'delete', 'permission_grant', etc.
    setting_key TEXT,                 -- Which setting was changed
    old_value TEXT,                   -- Previous value (JSON)
    new_value TEXT,                   -- New value (JSON)
    reason TEXT,                      -- Optional: why the change was made
    ip_address TEXT,                  -- Client IP address
    user_agent TEXT,                  -- Client user agent
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for audit queries
CREATE INDEX idx_audit_user ON settings_audit_log(user_id);
CREATE INDEX idx_audit_timestamp ON settings_audit_log(timestamp DESC);
CREATE INDEX idx_audit_key ON settings_audit_log(setting_key);
CREATE INDEX idx_audit_action ON settings_audit_log(action);
```

**Example Rows:**

| id  | user_id   | action | setting_key                              | old_value | new_value | timestamp           |
|-----|-----------|--------|------------------------------------------|-----------|-----------|---------------------|
| 1   | abc123... | update | visualisation.rendering.ambient_light... | 0.5       | 0.7       | 2025-10-17 10:30:00 |
| 2   | def456... | create | visualisation.graphs.custom.physics...   | NULL      | 15.0      | 2025-10-17 10:35:00 |
| 3   | admin...  | permission_grant | N/A                           | NULL      | abc123... | 2025-10-17 09:00:00 |

---

### 6. `auto_balance_config` - Auto-Balance Parameters

Stores adaptive physics tuning parameters for the auto-balance system.

```sql
CREATE TABLE auto_balance_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_type TEXT NOT NULL,
    user_id TEXT,
    stability_variance_threshold REAL DEFAULT 100.0,
    stability_frame_count INTEGER DEFAULT 180,
    clustering_distance_threshold REAL DEFAULT 20.0,
    clustering_hysteresis_buffer REAL DEFAULT 5.0,
    bouncing_node_percentage REAL DEFAULT 0.33,
    boundary_min_distance REAL DEFAULT 90.0,
    boundary_max_distance REAL DEFAULT 100.0,
    extreme_distance_threshold REAL DEFAULT 1000.0,
    explosion_distance_threshold REAL DEFAULT 10000.0,
    parameter_adjustment_rate REAL DEFAULT 0.1,
    max_adjustment_factor REAL DEFAULT 0.2,
    min_adjustment_factor REAL DEFAULT -0.2,
    adjustment_cooldown_ms INTEGER DEFAULT 2000,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(graph_type, user_id)
);

CREATE INDEX idx_autobalance_graph ON auto_balance_config(graph_type);
```

---

### 7. `auto_pause_config` - Auto-Pause Parameters

Stores energy-based simulation pause/resume parameters.

```sql
CREATE TABLE auto_pause_config (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    graph_type TEXT NOT NULL,
    user_id TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    equilibrium_velocity_threshold REAL DEFAULT 0.1,
    equilibrium_check_frames INTEGER DEFAULT 30,
    equilibrium_energy_threshold REAL DEFAULT 0.01,
    pause_on_equilibrium BOOLEAN DEFAULT TRUE,
    resume_on_interaction BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(graph_type, user_id)
);

CREATE INDEX idx_autopause_graph ON auto_pause_config(graph_type);
```

---

## Relationships

```
┌──────────────┐
│    users     │
└──────┬───────┘
       │ 1
       │
       │ N
       ├────────────────────────────────┐
       │                                │
       ↓                                ↓
┌──────────────┐                ┌──────────────┐
│user_settings │                │settings_audit│
│              │                │    _log      │
└──────┬───────┘                └──────────────┘
       │
       │ references key
       │
       ↓
┌──────────────┐
│   settings   │
└──────────────┘

┌──────────────┐
│    users     │
└──────┬───────┘
       │
       │ 1:N
       │
       ↓
┌──────────────┐
│physics_      │
│  settings    │
└──────────────┘
```

## Query Patterns

### Get Effective Setting for User

Merges user override with global default:

```sql
SELECT COALESCE(
    (SELECT value FROM user_settings WHERE user_id = ? AND key = ?),
    (SELECT value FROM settings WHERE key = ?)
) AS effective_value;
```

### Get All User Overrides

```sql
SELECT
    us.key,
    us.value AS user_value,
    s.value AS global_value,
    us.updated_at
FROM user_settings us
LEFT JOIN settings s ON us.key = s.key
WHERE us.user_id = ?
ORDER BY us.updated_at DESC;
```

### Get Physics Settings with Inheritance

```sql
SELECT
    COALESCE(user_ps.spring_k, global_ps.spring_k) AS spring_k,
    COALESCE(user_ps.repel_k, global_ps.repel_k) AS repel_k,
    COALESCE(user_ps.damping, global_ps.damping) AS damping
FROM physics_settings global_ps
LEFT JOIN physics_settings user_ps
    ON global_ps.graph_type = user_ps.graph_type
    AND user_ps.user_id = ?
WHERE global_ps.graph_type = ?
    AND global_ps.user_id IS NULL;
```

### Audit Log Query (Recent Changes)

```sql
SELECT
    u.display_name,
    a.action,
    a.setting_key,
    a.old_value,
    a.new_value,
    a.timestamp
FROM settings_audit_log a
JOIN users u ON a.user_id = u.pubkey
WHERE a.timestamp > datetime('now', '-24 hours')
ORDER BY a.timestamp DESC
LIMIT 100;
```

### Find Settings by Category

```sql
SELECT key, value, description
FROM settings
WHERE category = 'visualisation'
ORDER BY key;
```

## Indexes and Performance

### Index Strategy

1. **Primary Keys:** Auto-increment integers for fast lookups
2. **User Lookups:** Indexed on `pubkey` and `user_id` columns
3. **Setting Paths:** Indexed on `key` for fast resolution
4. **Audit Queries:** Indexed on `timestamp DESC` and `user_id`
5. **Composite Indexes:** For common join patterns (e.g., `user_id, key`)

### Query Performance Tips

1. **Use EXPLAIN QUERY PLAN** to verify index usage:
   ```sql
   EXPLAIN QUERY PLAN
   SELECT * FROM user_settings WHERE user_id = ? AND key = ?;
   ```

2. **Batch updates** in transactions:
   ```sql
   BEGIN TRANSACTION;
   UPDATE settings SET value = ? WHERE key = ?;
   UPDATE settings SET value = ? WHERE key = ?;
   COMMIT;
   ```

3. **Vacuum regularly** to reclaim space:
   ```sql
   PRAGMA vacuum;
   ```

4. **Analyze for statistics**:
   ```sql
   ANALYZE;
   ```

## Database Configuration

### SQLite Pragmas

```sql
-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Set busy timeout (5 seconds)
PRAGMA busy_timeout = 5000;

-- Use WAL mode for better concurrency
PRAGMA journal_mode = WAL;

-- Optimize page size
PRAGMA page_size = 4096;

-- Cache size (10MB)
PRAGMA cache_size = -10000;

-- Synchronous mode (balanced safety/performance)
PRAGMA synchronous = NORMAL;

-- Memory-mapped I/O (64MB)
PRAGMA mmap_size = 67108864;
```

### Connection Pool Settings

```rust
use rusqlite::Connection;
use r2d2_sqlite::SqliteConnectionManager;
use r2d2::Pool;

pub fn create_pool(db_path: &str) -> Result<Pool<SqliteConnectionManager>> {
    let manager = SqliteConnectionManager::file(db_path)
        .with_init(|conn| {
            conn.execute_batch(
                "PRAGMA foreign_keys = ON;
                 PRAGMA busy_timeout = 5000;
                 PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;"
            )?;
            Ok(())
        });

    let pool = Pool::builder()
        .max_size(10)                    // Max 10 connections
        .min_idle(Some(2))               // Keep 2 idle connections
        .connection_timeout(Duration::from_secs(5))
        .build(manager)?;

    Ok(pool)
}
```

## Migrations

### Schema Versioning

Track schema version in dedicated table:

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT INTO schema_version (version, description)
VALUES (2, 'Added auto_balance_config and auto_pause_config tables');
```

### Migration Example

```sql
-- Migration: 001_add_physics_constraints.sql
BEGIN TRANSACTION;

-- Add new column
ALTER TABLE physics_settings ADD COLUMN constraint_force_weight REAL DEFAULT 1.0;

-- Update version
INSERT INTO schema_version (version, description)
VALUES (3, 'Added constraint_force_weight to physics_settings');

COMMIT;
```

## Backup and Restore

### Backup

```bash
# SQLite backup
sqlite3 /app/data/settings.db ".backup /backup/settings-$(date +%Y%m%d-%H%M%S).db"

# Or using sqlite3 command-line tool
sqlite3 /app/data/settings.db ".dump" > /backup/settings-dump.sql
```

### Restore

```bash
# Restore from backup file
sqlite3 /app/data/settings.db < /backup/settings-dump.sql

# Or copy backup file
cp /backup/settings-20251017.db /app/data/settings.db
```

### Automatic Backups

```rust
use std::process::Command;

async fn backup_database() -> Result<()> {
    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let backup_path = format!("/backup/settings-{}.db", timestamp);

    let output = Command::new("sqlite3")
        .args(&[
            "/app/data/settings.db",
            &format!(".backup {}", backup_path)
        ])
        .output()?;

    if !output.status.success() {
        return Err(Error::BackupFailed);
    }

    info!("Database backed up to {}", backup_path);
    Ok(())
}
```

## Database Integrity

### Integrity Check

```sql
PRAGMA integrity_check;
```

Expected output: `ok`

### Foreign Key Check

```sql
PRAGMA foreign_key_check;
```

Expected output: Empty (no violations)

### Quick Check (Fast)

```sql
PRAGMA quick_check;
```

## Related Documentation

- [Settings System Architecture](./settings-system.md)
- [Settings API Reference](./settings-api.md)
- [Validation Rules](./settings-validation.md)
- [Migration Guide](./settings-migration-guide.md)
- [User Permissions](./user-permissions.md)
