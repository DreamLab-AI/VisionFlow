# Settings Migration Summary

## Quick Overview

**Goal**: Migrate VisionFlow settings from YAML/TOML files to SQLite database with hot-reload support.

**Timeline**: 4 weeks | **Risk**: Low-Medium | **Status**: Ready to implement

---

## Current State

```
settings.yaml (498 lines)     →  Visualization, physics, XR, auth configs
dev_config.toml (169 lines)   →  Developer settings, CUDA params
```

**Issues**:
- Requires server restart for changes
- No validation or audit trail
- Hard to manage 395+ settings
- No real-time updates

---

## Target State

```
settings.db (SQLite)
├── settings table (k/v with types)
├── physics_settings (22+ params)
├── audit_log (change tracking)
└── feature_flags (A/B testing)
```

**Benefits**:
- ✅ Hot-reload without restart
- ✅ WebSocket real-time updates
- ✅ Full audit trail
- ✅ Validation engine
- ✅ Developer CLI tool
- ✅ Control panel UI

---

## Architecture

```
┌─────────────┐
│ YAML/TOML   │  ──Migration──▶  ┌──────────────┐
│ (Legacy)    │                  │ settings.db   │
└─────────────┘                  └──────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
            │  WebSocket   │    │  REST API   │    │  CLI Tool   │
            │  (real-time) │    │  (CRUD)     │    │  (dev ops)  │
            └──────────────┘    └─────────────┘    └─────────────┘
```

---

## Implementation Phases

### Week 1: Foundation & Migration
- **Phase 1**: Validate existing schema (already done ✅)
- **Phase 2**: Build parsers (YAML, TOML → Database)
  - `src/migration/yaml_parser.rs`
  - `src/migration/toml_parser.rs`
  - `src/migration/mod.rs` (orchestrator)
  - `src/bin/migrate_settings.rs` (CLI)

**Milestone 1**: Successfully migrate all 395 settings to database

### Week 2: Hot-Reload & Compatibility
- **Phase 3**: Hot-reload mechanism
  - `src/services/settings_watcher.rs`
  - `src/handlers/settings_ws.rs`
  - WebSocket broadcasting
- **Phase 4**: Backward compatibility
  - `src/services/hybrid_settings_loader.rs`
  - Fallback to YAML/TOML during transition

**Milestone 2**: Settings change without restart, notifications work

### Week 3: Developer Tools
- **Phase 5**: Frontend control panel
  - Settings schema API
  - React components
  - Real-time validation
- **Phase 6**: Developer CLI (Days 1-2)
  - `src/bin/settings_cli.rs`
  - Full CRUD operations
  - Watch mode, diff, export/import

**Milestone 3**: Control panel and CLI fully functional

### Week 4: Polish & Launch
- **Phase 6**: CLI completion (Days 3-4)
  - Backup/restore
  - Bulk operations
  - Validation
- **Phase 7**: Testing & rollout (Days 5-7)
  - Unit tests, integration tests
  - Gradual rollout
  - Production migration

**Milestone 4**: Production-ready with monitoring

---

## Key Features

### Developer CLI
```bash
# Get setting
settings-cli get system.network.port

# Set setting
settings-cli set system.network.port 8080

# List with filter
settings-cli list --category physics --format table

# Watch changes
settings-cli watch --pattern "physics"

# Export/import
settings-cli export backup.yaml
settings-cli import new_config.toml --strategy merge

# Backup
settings-cli backup create pre-migration
settings-cli backup restore pre-migration
```

### Control Panel Features
- 🔍 Search and filter by category
- 📊 Real-time value updates (WebSocket)
- ✅ Inline validation
- 📝 Change history
- 🎚️ Dynamic controls (sliders, toggles, selects)
- 📁 Grouped by category

### Hot-Reload
- No server restart required
- <500ms propagation delay
- WebSocket notifications to all clients
- Audit log of all changes

---

## Migration Safety

### Safeguards
1. **Atomic transactions**: All-or-nothing migration
2. **Validation**: Verify all critical settings after migration
3. **Backup**: Original YAML/TOML files preserved
4. **Rollback**: Emergency fallback to file-based loading
5. **Testing**: Comprehensive test suite before production

### Rollback Plan
```bash
# If issues occur, revert to file-based loading
SETTINGS_USE_FILES=true cargo run

# Export database for analysis
settings-cli export rollback_dump.yaml
```

---

## Performance Targets

| Operation | Target | Status |
|-----------|--------|--------|
| Settings load (cache) | <10ms | ✅ Already achieved |
| Settings load (DB) | <50ms | ✅ Indexed queries |
| Update + notify | <100ms | 🎯 Target |
| WebSocket broadcast | <100ms | 🎯 Target |
| Full migration | <10s | 🎯 Target |

---

## Database Schema (Already Implemented ✅)

```sql
-- General settings (k/v with types)
CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value_type TEXT CHECK (value_type IN ('string', 'integer', 'float', 'boolean', 'json')),
    value_text TEXT,
    value_integer INTEGER,
    value_float REAL,
    value_boolean INTEGER,
    value_json TEXT,
    description TEXT,
    created_at DATETIME,
    updated_at DATETIME
);

-- Physics profiles
CREATE TABLE physics_settings (
    profile_name TEXT PRIMARY KEY,
    damping REAL, dt REAL, iterations INTEGER,
    max_velocity REAL, max_force REAL,
    -- ... 22+ parameters
);

-- Audit trail
CREATE TABLE settings_audit_log (
    id INTEGER PRIMARY KEY,
    setting_key TEXT,
    old_value TEXT,
    new_value TEXT,
    changed_by TEXT,
    changed_at DATETIME
);
```

---

## Code Structure

### New Files
```
src/
├── migration/
│   ├── mod.rs              # Orchestrator
│   ├── yaml_parser.rs      # YAML → DB
│   └── toml_parser.rs      # TOML → DB
├── services/
│   ├── settings_watcher.rs       # Change detection
│   └── hybrid_settings_loader.rs # Fallback loader
├── handlers/
│   ├── settings_schema_handler.rs  # Schema API
│   └── settings_ws.rs              # WebSocket
└── bin/
    ├── migrate_settings.rs   # Migration CLI
    └── settings_cli.rs       # Management CLI (700+ lines)
```

### Dependencies
```toml
notify = "6.1"              # File watcher
clap = { version = "4.5", features = ["derive"] }  # CLI
```

---

## Risk Assessment

| Risk | Severity | Mitigation | Status |
|------|----------|------------|--------|
| Data loss | High | Backups, validation, transactions | ✅ |
| Performance | Medium | Caching, indexing, pooling | ✅ |
| Breaking changes | Medium | Hybrid loader, gradual rollout | ✅ |
| WebSocket scale | Low | Rate limiting, debouncing | 🎯 |

**Overall Risk**: Low-Medium with mitigations

---

## Success Criteria

- ✅ 100% settings migrated without data loss
- ✅ Zero downtime during migration
- ✅ Hot-reload working (<500ms latency)
- ✅ Control panel functional with real-time updates
- ✅ CLI tool feature-complete
- ✅ Backward compatibility maintained
- ✅ Performance meets targets
- ✅ Developer satisfaction >90%

---

## Next Steps

1. **Immediate**: Review and approve migration plan
2. **Week 1**: Implement migration scripts and parsers
3. **Week 2**: Build hot-reload and WebSocket system
4. **Week 3**: Develop control panel and CLI
5. **Week 4**: Test, rollout, monitor

---

## Resources

- **Full Plan**: `/home/devuser/workspace/project/docs/settings-migration-plan.md` (19,000+ words)
- **Schema**: `/home/devuser/workspace/project/schema/settings_db.sql`
- **Current Settings**: `data/settings.yaml`, `data/dev_config.toml`
- **Database Service**: `src/services/database_service.rs`
- **Settings Service**: `src/services/settings_service.rs`

---

**Status**: Ready for implementation
**Estimated Effort**: 4 weeks (1 senior developer)
**Created**: 2025-10-22
