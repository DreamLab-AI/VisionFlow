# Legacy Configuration Files - Deletion Checklist

**⚠️ IMPORTANT: DO NOT DELETE UNTIL MIGRATION VERIFIED**

## Files Marked for Deletion

After running the migration script and verifying the database contains all settings, these files can be safely deleted:

### Configuration Files

1. **data/settings.yaml** ✗
   - Size: 498 lines
   - Contains: Main visualization and system settings
   - Migrates to: `settings.db` table
   - Backup location: `data/legacy_backup/settings.yaml`

2. **data/settings_ontology_extension.yaml** ✗
   - Size: 142 lines
   - Contains: Ontology-specific configuration
   - Migrates to: `settings.db` table (ontology.* keys)
   - Backup location: `data/legacy_backup/settings_ontology_extension.yaml`

3. **data/dev_config.toml** ✗
   - Size: 169 lines
   - Contains: Developer/internal configuration
   - Migrates to: `settings.db` table (developer tier)
   - Backup location: `data/legacy_backup/dev_config.toml`

## Pre-Deletion Verification Steps

### Step 1: Create Backup
```bash
mkdir -p data/legacy_backup
cp data/settings.yaml data/legacy_backup/
cp data/settings_ontology_extension.yaml data/legacy_backup/
cp data/dev_config.toml data/legacy_backup/
```

### Step 2: Run Migration Script
```bash
cd /home/devuser/workspace/project
cargo run --bin migrate_legacy_configs
```

Expected output:
```
🔄 Starting legacy config migration...
📄 Processing: data/settings.yaml
✓ Successfully migrated: data/settings.yaml
📄 Processing: data/settings_ontology_extension.yaml
✓ Successfully migrated: data/settings_ontology_extension.yaml
📄 Processing: data/dev_config.toml
✓ Successfully migrated: data/dev_config.toml

=== MIGRATION REPORT ===
Files Processed: 3
Total Records Migrated: [NUMBER]
Errors: 0
✓ Migration completed successfully
```

### Step 3: Verify Database

```bash
# Check total settings count
sqlite3 data/visionflow.db "SELECT COUNT(*) FROM settings;"

# Verify migration sources
sqlite3 data/visionflow.db "
SELECT source, COUNT(*) as count
FROM settings
GROUP BY source;
"

# Sample settings from each source
sqlite3 data/visionflow.db "
SELECT key, tier, source
FROM settings
WHERE source LIKE '%yaml'
LIMIT 10;
"

# Check critical settings
sqlite3 data/visionflow.db "
SELECT key, value, tier
FROM settings
WHERE key IN (
    'visualisation.rendering.ambientLightIntensity',
    'system.network.port',
    'physics.rest_length',
    'ontology.enabled'
);
"
```

### Step 4: Test Application

```bash
# Start application
cargo run

# Check logs for:
# ✓ "Database and settings service initialized successfully"
# ✓ "Settings will be loaded from database"
# ✗ NO "Failed to read YAML file" errors
# ✗ NO "settings.yaml not found" warnings
```

### Step 5: Verify Settings Load

Check application logs for these messages:
```
[AppState::new] Initializing SQLite database (NEW architecture)
[AppState::new] Initializing database schema
[AppState::new] Migrating settings to database
[AppState::new] Creating SettingsService (UI → Database direct connection)
[AppState::new] Database and settings service initialized successfully
```

### Step 6: Review Migration Report

```bash
cat data/migration_report.json
```

Expected structure:
```json
{
  "timestamp": "2025-10-22T...",
  "files_processed": [
    "data/settings.yaml",
    "data/settings_ontology_extension.yaml",
    "data/dev_config.toml"
  ],
  "records_migrated": [NUMBER],
  "errors": []
}
```

## Deletion Commands

**⚠️ ONLY RUN AFTER ALL VERIFICATION STEPS PASS**

```bash
# Git remove (preferred - keeps in history)
git rm data/settings.yaml
git rm data/settings_ontology_extension.yaml
git rm data/dev_config.toml

# Commit the deletion
git commit -m "Remove legacy YAML/TOML config files

All configuration now stored in SQLite database:
- settings.yaml → settings.db (system tier)
- settings_ontology_extension.yaml → settings.db (ontology.* keys)
- dev_config.toml → settings.db (developer tier)

Migration completed successfully with [NUMBER] records transferred.
Backup preserved in data/legacy_backup/
"
```

## Rollback Procedure

If issues are discovered after deletion:

```bash
# Restore files from backup
cp data/legacy_backup/settings.yaml data/
cp data/legacy_backup/settings_ontology_extension.yaml data/
cp data/legacy_backup/dev_config.toml data/

# Or restore from git
git checkout HEAD~1 -- data/settings.yaml data/settings_ontology_extension.yaml data/dev_config.toml
```

## Post-Deletion Cleanup

After successful deletion and testing:

1. **Update Documentation**
   - Remove references to config files in README
   - Update deployment docs
   - Update environment setup guides

2. **Update CI/CD**
   - Remove config file expectations from deployment scripts
   - Update Docker configs if needed
   - Remove file copy commands

3. **Update .gitignore** (optional)
   ```bash
   echo "# Legacy config files removed - now using database" >> .gitignore
   echo "data/*.yaml" >> .gitignore
   echo "data/*.toml" >> .gitignore
   ```

4. **Clean Migration Artifacts** (optional, after 30 days)
   ```bash
   rm -rf data/legacy_backup/
   rm data/migration_report.json
   ```

## What NOT to Delete

**KEEP THESE FILES:**
- `src/config/mod.rs` - Structure definitions still needed
- `src/actors/graph_service_supervisor.rs` - Current architecture (not legacy)
- `Cargo.toml` - serde_yaml still needed for OWL ontology files
- `data/visionflow.db` - The NEW source of truth

## Deletion Checklist

- [ ] Backup created in `data/legacy_backup/`
- [ ] Migration script ran successfully
- [ ] Migration report reviewed (no errors)
- [ ] Database contains all expected settings
- [ ] Application starts without YAML errors
- [ ] Settings load from database confirmed
- [ ] Critical settings verified in database
- [ ] Files deleted via `git rm`
- [ ] Commit created with migration details
- [ ] Application tested after deletion
- [ ] Team notified of architecture change
- [ ] Documentation updated

## Status

**Current Status:** ⏸️ Files identified, ready for deletion after migration
**Next Action:** Run migration script and verify database
**Assigned To:** DevOps / System Administrator
**Due Date:** After successful migration verification

---

**Generated:** 2025-10-22
**Report Location:** `/home/devuser/workspace/project/docs/LEGACY_FILES_FOR_DELETION.md`
**Migration Script:** `/home/devuser/workspace/project/scripts/migrate_legacy_configs.rs`
**Completion Report:** `/home/devuser/workspace/project/docs/legacy-config-removal-report.md`
