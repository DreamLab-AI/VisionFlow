# Settings System Architecture

## System Overview

This document describes the technical architecture of VisionFlow's settings management system after database migration.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  React Control   │  │  CLI Tool        │  │  External API   │  │
│  │  Panel           │  │  (settings-cli)  │  │  Clients        │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘  │
│           │                     │                      │            │
│           │                     │                      │            │
└───────────┼─────────────────────┼──────────────────────┼────────────┘
            │                     │                      │
            │ WebSocket           │ Direct               │ REST API
            │                     │                      │
┌───────────▼─────────────────────▼──────────────────────▼────────────┐
│                         API LAYER                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │  WebSocket       │  │  REST Endpoints  │  │  Schema API     │  │
│  │  Handler         │  │  (/api/settings) │  │  (metadata)     │  │
│  │  (settings_ws)   │  │                  │  │                 │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘  │
│           │                     │                      │            │
│           └─────────────────────┴──────────────────────┘            │
│                                 │                                   │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│                         SERVICE LAYER                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    SettingsService                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │  Cache       │  │  Validation  │  │  Change          │  │  │
│  │  │  (5 min TTL) │  │  Engine      │  │  Notification    │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │  │
│  └─────────────────────────────┬────────────────────────────────┘  │
│                                │                                   │
│  ┌─────────────────────────────▼────────────────────────────────┐  │
│  │                    SettingsWatcher                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │  File Watch  │  │  DB Monitor  │  │  Event Queue     │  │  │
│  │  │  (notify)    │  │  (polling)   │  │  (mpsc)          │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                         DATA ACCESS LAYER                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    DatabaseService                           │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │  │
│  │  │  Connection  │  │  Query       │  │  Transaction     │  │  │
│  │  │  Pool        │  │  Builder     │  │  Manager         │  │  │
│  │  │  (r2d2)      │  │              │  │                  │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │  │
│  └─────────────────────────────┬────────────────────────────────┘  │
│                                │                                   │
└────────────────────────────────┼───────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────┐
│                         DATABASE LAYER                             │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌────────────────────┐  ┌────────────────────┐  ┌─────────────┐ │
│  │  settings.db       │  │  Indices           │  │  Triggers   │ │
│  │                    │  │                    │  │             │ │
│  │  ┌──────────────┐  │  │  idx_settings_key  │  │  Auto       │ │
│  │  │  settings    │  │  │  idx_physics_prof  │  │  timestamp  │ │
│  │  │  (k/v)       │  │  │  idx_audit_date    │  │  update     │ │
│  │  └──────────────┘  │  └────────────────────┘  └─────────────┘ │
│  │                    │                                           │
│  │  ┌──────────────┐  │                                           │
│  │  │  physics_    │  │                                           │
│  │  │  settings    │  │                                           │
│  │  └──────────────┘  │                                           │
│  │                    │                                           │
│  │  ┌──────────────┐  │                                           │
│  │  │  audit_log   │  │                                           │
│  │  └──────────────┘  │                                           │
│  └────────────────────┘                                           │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### 1. Read Setting (Cache Hit)

```
Client Request
    │
    ▼
SettingsService.get_setting("key")
    │
    ├─▶ Cache.get("key")
    │       │
    │       ▼
    │   [Found in cache] ─────▶ Return value
    │
    └─▶ [Not in cache] ─────▶ DatabaseService.get_setting("key")
                                    │
                                    ▼
                                Query database
                                    │
                                    ▼
                                Cache.set("key", value)
                                    │
                                    ▼
                                Return value
```

### 2. Update Setting (with Hot-Reload)

```
Client Update Request
    │
    ▼
SettingsService.set_setting("key", value)
    │
    ├─▶ Validate value
    │       │
    │       ▼
    │   [Valid] ────────────────────────┐
    │                                   │
    │   [Invalid] ──▶ Return error     │
    │                                   │
    ├─▶ DatabaseService.set_setting() ◀┘
    │       │
    │       ▼
    │   INSERT/UPDATE in database
    │       │
    │       ▼
    │   Audit log entry
    │       │
    │       ▼
    ├─▶ Cache.invalidate("key")
    │
    ├─▶ Notify change listeners
    │       │
    │       ▼
    │   SettingsWatcher detects change
    │       │
    │       ▼
    │   Broadcast via WebSocket
    │       │
    │       ▼
    └─▶ All connected clients receive update
```

### 3. Migration Flow

```
Start Migration
    │
    ▼
Check if already migrated
    │
    ├─▶ [Already migrated] ──▶ Exit
    │
    └─▶ [Not migrated]
            │
            ▼
        Load YAML file
            │
            ▼
        Parse & flatten
            │
            ▼
        Load TOML file
            │
            ▼
        Parse & flatten
            │
            ▼
        Begin transaction
            │
            ├─▶ Insert YAML settings (batched)
            │
            ├─▶ Insert TOML settings (batched)
            │
            ├─▶ Validate critical settings
            │       │
            │       ├─▶ [Valid] ────────────────┐
            │       │                           │
            │       └─▶ [Invalid] ──▶ Rollback ─┘
            │                           │
            │                           ▼
            ├─▶ Mark migration complete    Exit with error
            │
            ▼
        Commit transaction
            │
            ▼
        Generate report
            │
            ▼
        Exit success
```

---

## Component Details

### SettingsService

**Responsibilities:**
- Primary interface for settings access
- Caching with TTL (5 minutes)
- Validation before writes
- Change notification management

**Key Methods:**
```rust
async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>>
async fn set_setting(&self, key: &str, value: SettingValue) -> Result<()>
async fn get_settings_batch(&self, keys: &[String]) -> Result<HashMap<String, SettingValue>>
async fn enable_hot_reload(&self) -> Result<()>
async fn add_change_listener<F>(&self, listener: F)
```

**Dependencies:**
- `DatabaseService` (data access)
- `SettingsWatcher` (change detection)
- `Validation` (value validation)

---

### DatabaseService

**Responsibilities:**
- Connection pooling (r2d2)
- Raw database operations
- Schema initialization
- Transaction management

**Key Methods:**
```rust
fn get_setting(&self, key: &str) -> SqliteResult<Option<SettingValue>>
fn set_setting(&self, key: &str, value: SettingValue, description: Option<&str>) -> SqliteResult<()>
fn get_physics_settings(&self, profile: &str) -> SqliteResult<PhysicsSettings>
fn save_all_settings(&self, settings: &AppFullSettings) -> SqliteResult<()>
fn initialize_schema(&self) -> SqliteResult<()>
```

**Configuration:**
- Pool size: 10 connections
- Idle timeout: 90 seconds
- WAL mode enabled
- Foreign keys enforced

---

### SettingsWatcher

**Responsibilities:**
- Detect database changes
- Poll audit log for new entries
- Emit change events
- Manage event queue

**Key Methods:**
```rust
async fn start(&self) -> Result<()>
async fn check_changes(since: DateTime) -> Result<Vec<SettingChangeEvent>>
```

**Polling:**
- Interval: 1 second
- Source: `settings_audit_log` table
- Event queue: 100 capacity

---

### Migration Module

**Responsibilities:**
- Parse YAML/TOML files
- Flatten nested structures
- Batch insert to database
- Validation and reporting

**Components:**
1. **YamlMigrator**: Parse settings.yaml
2. **TomlMigrator**: Parse dev_config.toml
3. **SettingsMigration**: Orchestrate migration
4. **MigrationReport**: Track progress and errors

**Key Methods:**
```rust
async fn migrate(&self) -> Result<MigrationReport>
fn migrate_yaml(&self, path: &str) -> Result<usize>
fn migrate_toml(&self, path: &str) -> Result<usize>
fn validate_migration(&self, report: &mut MigrationReport) -> Result<()>
```

---

## Database Schema

### settings Table

| Column | Type | Description |
|--------|------|-------------|
| key | TEXT (PK) | Unique setting identifier |
| value_type | TEXT | Type: string, integer, float, boolean, json |
| value_text | TEXT | String values |
| value_integer | INTEGER | Integer values |
| value_float | REAL | Float values |
| value_boolean | INTEGER | Boolean values (0/1) |
| value_json | TEXT | JSON values |
| description | TEXT | Human-readable description |
| created_at | DATETIME | Creation timestamp |
| updated_at | DATETIME | Last update timestamp |

**Indices:**
- `idx_settings_key` on `key`
- `idx_settings_updated_at` on `updated_at`
- `idx_settings_type` on `value_type`

---

### physics_settings Table

| Column | Type | Description |
|--------|------|-------------|
| profile_name | TEXT (PK) | Profile identifier |
| damping | REAL | Force damping |
| dt | REAL | Time step |
| iterations | INTEGER | Simulation iterations |
| max_velocity | REAL | Velocity clamp |
| max_force | REAL | Force clamp |
| repel_k | REAL | Repulsion strength |
| spring_k | REAL | Spring strength |
| ... | ... | +15 more parameters |
| created_at | DATETIME | Creation timestamp |
| updated_at | DATETIME | Last update timestamp |

**Indices:**
- `idx_physics_profile` on `profile_name`
- `idx_physics_default` on `is_default`

---

### settings_audit_log Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Auto-increment ID |
| setting_key | TEXT | Setting identifier |
| old_value | TEXT | Previous value |
| new_value | TEXT | New value |
| changed_by | TEXT | User ID or 'system' |
| change_reason | TEXT | Optional reason |
| change_type | TEXT | create, update, delete |
| ip_address | TEXT | Client IP |
| user_agent | TEXT | Client user agent |
| changed_at | DATETIME | Change timestamp |

**Indices:**
- `idx_audit_key` on `setting_key`
- `idx_audit_date` on `changed_at`
- `idx_audit_changed_by` on `changed_by`
- `idx_audit_type` on `change_type`

---

## WebSocket Protocol

### Connection
```
Client connects to: ws://localhost:4000/ws/settings
```

### Message Format

#### Client → Server (Subscribe)
```json
{
  "action": "subscribe",
  "pattern": "system.network.*"
}
```

#### Server → Client (Setting Changed)
```json
{
  "event": "setting_changed",
  "key": "system.network.port",
  "value": 8080,
  "timestamp": "2025-10-22T15:30:45Z"
}
```

#### Server → Client (Bulk Update)
```json
{
  "event": "settings_updated",
  "keys": ["system.network.port", "system.network.domain"],
  "timestamp": "2025-10-22T15:30:45Z"
}
```

---

## REST API Endpoints

### GET /api/settings/:key
Get single setting value

**Response:**
```json
{
  "key": "system.network.port",
  "value": 4000,
  "type": "integer",
  "updated_at": "2025-10-22T15:30:45Z"
}
```

### PUT /api/settings/:key
Update single setting

**Request:**
```json
{
  "value": 8080
}
```

**Response:**
```json
{
  "success": true,
  "key": "system.network.port",
  "old_value": 4000,
  "new_value": 8080
}
```

### GET /api/settings
List all settings (with pagination)

**Query params:**
- `category`: Filter by category
- `search`: Search pattern
- `page`: Page number
- `limit`: Results per page

### GET /api/settings/schema
Get settings metadata

**Response:**
```json
[
  {
    "key": "system.network.port",
    "type": "integer",
    "category": "Network",
    "description": "Server port",
    "min": 1024,
    "max": 65535,
    "default": 4000
  }
]
```

### POST /api/settings/batch
Update multiple settings atomically

**Request:**
```json
{
  "updates": {
    "system.network.port": 8080,
    "system.network.domain": "example.com"
  }
}
```

---

## Performance Characteristics

### Cache Performance
- **Hit rate**: >95% (typical workload)
- **TTL**: 5 minutes
- **Eviction**: LRU when memory limit reached
- **Size**: ~1000 entries (typical)

### Database Performance
- **Connection pool**: 10 connections
- **Indexed queries**: <5ms (avg)
- **Unindexed queries**: <20ms (avg)
- **Batch insert**: ~100 settings/sec

### WebSocket Performance
- **Connections**: 100 concurrent (default)
- **Broadcast latency**: <100ms
- **Message rate**: 1000 msg/sec (max)

---

## Security Considerations

### Authentication
- Settings API requires authentication
- WebSocket connections validated
- Rate limiting per user/IP

### Audit Trail
- All changes logged with user ID
- IP address and user agent captured
- Immutable audit log

### Validation
- Type checking on all updates
- Range validation for numeric values
- Schema validation for JSON values

### Encryption
- API keys encrypted at rest (AES-256)
- TLS for WebSocket connections
- Database file permissions: 0600

---

## Monitoring & Observability

### Metrics Exposed

```rust
// Prometheus-style metrics
settings_cache_hit_rate
settings_cache_miss_rate
settings_db_query_duration_seconds
settings_update_total
settings_error_total
settings_websocket_connections
```

### Logging

```rust
// Log levels
DEBUG: Cache hits/misses, query plans
INFO: Setting updates, migrations
WARN: Validation failures, deprecated usage
ERROR: Database errors, connection failures
```

### Health Check

```bash
GET /api/settings/health
```

**Response:**
```json
{
  "status": "healthy",
  "database": {
    "connected": true,
    "pool_size": 10,
    "idle_connections": 7
  },
  "cache": {
    "entries": 245,
    "hit_rate": 0.96
  },
  "websocket": {
    "connections": 12
  }
}
```

---

## Deployment Architecture

### Single Server
```
┌─────────────────────────────┐
│  VisionFlow Server          │
│  ┌───────────────────────┐  │
│  │  SettingsService      │  │
│  │  + Local Cache        │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  DatabaseService      │  │
│  │  + Connection Pool    │  │
│  └───────────────────────┘  │
│  ┌───────────────────────┐  │
│  │  settings.db (SQLite) │  │
│  └───────────────────────┘  │
└─────────────────────────────┘
```

### Multi-Server (Future)
```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Server 1   │   │  Server 2   │   │  Server 3   │
│  + Cache    │   │  + Cache    │   │  + Cache    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
                    ┌────▼────┐
                    │  Redis  │
                    │  (sync) │
                    └────┬────┘
                         │
                  ┌──────▼──────┐
                  │  PostgreSQL │
                  │  (shared)   │
                  └─────────────┘
```

---

## References

- **Full Implementation Plan**: `settings-migration-plan.md`
- **Quick Start Guide**: `settings-migration-quickstart.md`
- **Summary**: `settings-migration-summary.md`
- **Database Schema**: `../schema/settings_db.sql`
- **Current Code**: `../src/services/{database_service,settings_service}.rs`

---

**Last Updated**: 2025-10-22
**Version**: 1.0
**Status**: Approved for implementation
