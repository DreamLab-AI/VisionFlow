# Settings Migration Quick Start Guide

## For Developers Implementing This Migration

### Prerequisites
- Rust 1.75+
- SQLite 3.x
- Familiarity with VisionFlow codebase

---

## Step-by-Step Implementation

### Day 1: Setup & Validation

1. **Review existing infrastructure** ✅
   ```bash
   # These already exist and are working:
   cat schema/settings_db.sql
   cat src/services/database_service.rs
   cat src/services/settings_service.rs
   ```

2. **Run schema test**
   ```rust
   #[test]
   fn test_schema_initialization() {
       let db = DatabaseService::new("test.db").unwrap();
       db.initialize_schema().unwrap();
       assert!(db.health_check().unwrap().all_healthy);
   }
   ```

3. **Add new dependencies to Cargo.toml**
   ```toml
   [dependencies]
   notify = "6.1"
   clap = { version = "4.5", features = ["derive"] }
   ```

### Day 2-3: Migration Scripts

1. **Create migration module**
   ```bash
   mkdir src/migration
   touch src/migration/mod.rs
   touch src/migration/yaml_parser.rs
   touch src/migration/toml_parser.rs
   ```

2. **Implement YAML parser** (see full plan Section 2.1)
   ```rust
   // src/migration/yaml_parser.rs
   pub struct YamlMigrator {
       settings: AppFullSettings,
   }

   impl YamlMigrator {
       pub fn load_from_file(path: &str) -> Result<Self, String> { ... }
       pub fn flatten(&self) -> Vec<(String, SettingValue)> { ... }
   }
   ```

3. **Implement TOML parser** (see full plan Section 2.2)
   ```rust
   // src/migration/toml_parser.rs
   pub struct TomlMigrator {
       content: TomlValue,
   }
   ```

4. **Create migration orchestrator** (see full plan Section 2.3)
   ```rust
   // src/migration/mod.rs
   pub struct SettingsMigration {
       db: DatabaseService,
   }

   impl SettingsMigration {
       pub async fn migrate(&self) -> Result<MigrationReport, String> { ... }
   }
   ```

5. **Build migration CLI** (see full plan Section 2.4)
   ```bash
   mkdir -p src/bin
   # Create src/bin/migrate_settings.rs
   ```

6. **Test migration**
   ```bash
   cargo run --bin migrate_settings data/settings.db
   ```

### Day 4-5: Hot-Reload System

1. **Create settings watcher** (see full plan Section 3.1)
   ```bash
   touch src/services/settings_watcher.rs
   ```

2. **Implement WebSocket handler** (see full plan Section 3.2)
   ```bash
   touch src/handlers/settings_ws.rs
   ```

3. **Integrate with SettingsService** (see full plan Section 3.3)
   ```rust
   // Add to src/services/settings_service.rs
   impl SettingsService {
       pub async fn enable_hot_reload(&self) -> Result<(), String> { ... }
   }
   ```

4. **Test hot-reload**
   ```bash
   # Terminal 1: Start server
   cargo run

   # Terminal 2: Watch for changes
   cargo run --bin settings_cli watch

   # Terminal 3: Update setting
   cargo run --bin settings_cli set test_key new_value
   ```

### Day 6-8: Developer CLI

1. **Create CLI skeleton** (see full plan Section 6.1)
   ```bash
   touch src/bin/settings_cli.rs
   ```

2. **Implement core commands** (see full plan Section 6.2)
   - `get`: Retrieve single setting
   - `set`: Update setting
   - `list`: Display all settings
   - `export`: Export to file
   - `import`: Import from file

3. **Add advanced features**
   - `watch`: Monitor changes
   - `diff`: Compare states
   - `backup`: Create/restore backups

4. **Test CLI**
   ```bash
   cargo build --release --bin settings_cli
   ./target/release/settings_cli --help
   ./target/release/settings_cli get system.network.port
   ```

### Day 9-12: Frontend Control Panel

1. **Create schema API endpoint** (see full plan Section 5.1)
   ```rust
   // src/handlers/settings_schema_handler.rs
   pub async fn get_settings_schema() -> HttpResponse { ... }
   ```

2. **Build React components** (see full plan Section 5.2)
   ```typescript
   // frontend/src/components/SettingsPanel.tsx
   const SettingsPanel: React.FC = () => { ... }
   ```

3. **Add dynamic form controls** (see full plan Section 5.3)
   ```typescript
   // frontend/src/components/SettingControl.tsx
   const SettingControl: React.FC = ({ config }) => { ... }
   ```

### Day 13-15: Testing

1. **Unit tests**
   ```rust
   // tests/settings_migration_tests.rs
   #[tokio::test]
   async fn test_yaml_migration() { ... }

   #[tokio::test]
   async fn test_hot_reload() { ... }
   ```

2. **Integration tests**
   ```rust
   // tests/integration/settings_api_tests.rs
   #[actix_web::test]
   async fn test_settings_api_flow() { ... }
   ```

3. **Load testing**
   ```bash
   # Use wrk or similar
   wrk -t4 -c100 -d30s http://localhost:4000/api/settings/test_key
   ```

### Day 16-20: Rollout

1. **Development environment**
   ```bash
   # Backup current settings
   cp data/settings.yaml data/settings.yaml.backup

   # Run migration
   cargo run --bin migrate_settings data/settings.db

   # Verify
   cargo run --bin settings_cli list --format table
   ```

2. **Staging environment**
   - Deploy with hybrid loader enabled
   - Monitor for 24 hours
   - Collect metrics

3. **Production rollout**
   - Create backup: `settings-cli backup create pre-migration`
   - Run migration during low-traffic period
   - Monitor error rates
   - Keep rollback ready

---

## Critical Files to Create

### Priority 1 (Week 1)
- [ ] `src/migration/mod.rs`
- [ ] `src/migration/yaml_parser.rs`
- [ ] `src/migration/toml_parser.rs`
- [ ] `src/bin/migrate_settings.rs`
- [ ] `tests/settings_migration_tests.rs`

### Priority 2 (Week 2)
- [ ] `src/services/settings_watcher.rs`
- [ ] `src/handlers/settings_ws.rs`
- [ ] `src/services/hybrid_settings_loader.rs`
- [ ] `tests/settings_hot_reload_tests.rs`

### Priority 3 (Week 3)
- [ ] `src/bin/settings_cli.rs`
- [ ] `src/handlers/settings_schema_handler.rs`
- [ ] `frontend/src/components/SettingsPanel.tsx`
- [ ] `frontend/src/components/SettingControl.tsx`

### Priority 4 (Week 4)
- [ ] `tests/integration/settings_api_tests.rs`
- [ ] `docs/settings-migration-runbook.md`
- [ ] Performance benchmarks
- [ ] Production deployment checklist

---

## Quick Commands Reference

### Migration
```bash
# Run migration
cargo run --bin migrate_settings data/settings.db

# Force re-migration
cargo run --bin migrate_settings data/settings.db --force

# Dry run (validation only)
cargo run --bin migrate_settings data/settings.db --dry-run
```

### CLI Operations
```bash
# Get setting
settings-cli get system.network.port

# Set setting
settings-cli set system.network.port 8080

# List all
settings-cli list --format table

# Export
settings-cli export backup.yaml --format yaml

# Import
settings-cli import new_config.toml --strategy merge

# Watch changes
settings-cli watch --pattern "physics"

# Backup
settings-cli backup create pre-migration-$(date +%Y%m%d)
settings-cli backup restore pre-migration-20251022
```

### Development
```bash
# Run tests
cargo test settings_migration
cargo test settings_hot_reload
cargo test --test settings_api_tests

# Build release
cargo build --release --bin migrate_settings
cargo build --release --bin settings_cli

# Run with debug logging
RUST_LOG=debug cargo run --bin migrate_settings
```

---

## Troubleshooting

### Migration fails with "Setting not found"
```bash
# Check if YAML/TOML files exist
ls -la data/settings.yaml data/dev_config.toml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('data/settings.yaml'))"

# Check database schema
sqlite3 data/settings.db ".schema settings"
```

### Hot-reload not working
```bash
# Verify WebSocket connection
wscat -c ws://localhost:4000/ws/settings

# Check watcher is running
ps aux | grep settings_watcher

# Test manual update
settings-cli set test_key test_value
```

### Performance issues
```bash
# Check database indices
sqlite3 data/settings.db ".indexes"

# Analyze query plan
sqlite3 data/settings.db "EXPLAIN QUERY PLAN SELECT * FROM settings WHERE key = 'test';"

# Clear cache
settings-cli cache clear
```

---

## Rollback Procedure

If critical issues occur:

1. **Stop server**
   ```bash
   systemctl stop visionflow
   ```

2. **Enable file-based loading**
   ```bash
   export SETTINGS_USE_FILES=true
   ```

3. **Restart server**
   ```bash
   systemctl start visionflow
   ```

4. **Export database for analysis**
   ```bash
   settings-cli export rollback_analysis.yaml
   ```

5. **Restore from backup**
   ```bash
   cp data/settings.yaml.backup data/settings.yaml
   ```

---

## Success Checklist

- [ ] Schema initialized without errors
- [ ] Migration completes in <10 seconds
- [ ] All 395 settings migrated successfully
- [ ] Validation passes for all critical settings
- [ ] Hot-reload latency <500ms
- [ ] WebSocket notifications working
- [ ] CLI tool functional (all commands)
- [ ] Control panel loads and updates in real-time
- [ ] Performance targets met
- [ ] Rollback tested and documented

---

## Getting Help

- **Full Plan**: `docs/settings-migration-plan.md`
- **Summary**: `docs/settings-migration-summary.md`
- **Schema**: `schema/settings_db.sql`
- **Code Examples**: See full plan sections 2-6

**Questions?** Review the comprehensive plan in `docs/settings-migration-plan.md`

---

**Last Updated**: 2025-10-22
**Status**: Ready for implementation
