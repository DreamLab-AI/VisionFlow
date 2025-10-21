# Settings Migration Quick Fix

**Problem**: "No settings available" error in frontend

**Root Cause**: Migration code expects YAML files that don't exist

---

## Quick Fix (5 minutes)

### Option A: Seed Database Directly (Fastest)

Create default settings in database without YAML migration:

```bash
# Create seeding script
cat > /tmp/seed_settings.sql << 'EOF'
-- Seed with minimal working settings
INSERT INTO settings (key, value_type, value_json, description)
VALUES (
    'app_full_settings',
    'json',
    '{
        "visualisation": {
            "rendering": {
                "ambient_light_intensity": 0.5,
                "background_color": "#1a1a1a",
                "enable_ambient_occlusion": true
            },
            "graphs": {
                "logseq": {
                    "physics": {
                        "damping": 0.95,
                        "spring_k": 0.005,
                        "repel_k": 50.0,
                        "dt": 0.016,
                        "max_velocity": 1.0,
                        "max_force": 100.0
                    }
                }
            }
        },
        "system": {
            "network": {
                "port": 4000,
                "bind_address": "0.0.0.0"
            }
        }
    }',
    'Complete application settings'
)
ON CONFLICT(key) DO UPDATE SET
    value_json = excluded.value_json,
    updated_at = CURRENT_TIMESTAMP;

INSERT INTO physics_settings (
    profile_name, damping, dt, iterations, max_velocity, max_force,
    repel_k, spring_k, mass_scale, boundary_damping
) VALUES (
    'default', 0.95, 0.016, 100, 1.0, 100.0, 50.0, 0.005, 1.0, 0.95
)
ON CONFLICT(profile_name) DO UPDATE SET
    damping = excluded.damping,
    dt = excluded.dt,
    updated_at = CURRENT_TIMESTAMP;

-- Set version key so migration doesn't run again
INSERT INTO settings (key, value_type, value_text, description)
VALUES ('version', 'string', '2.0', 'Settings schema version')
ON CONFLICT(key) DO NOTHING;
EOF

# Apply seeding script
sqlite3 /app/data/ontology_db.sqlite3 < /tmp/seed_settings.sql

# Verify
sqlite3 /app/data/ontology_db.sqlite3 "SELECT key FROM settings WHERE key = 'app_full_settings';"
```

### Option B: Create YAML Files (Proper Migration)

Create source files for migration to process:

```bash
# Create settings.yaml
cat > /app/data/settings.yaml << 'EOF'
visualisation:
  rendering:
    ambient_light_intensity: 0.5
    background_color: "#1a1a1a"
    enable_ambient_occlusion: true
  graphs:
    logseq:
      physics:
        damping: 0.95
        dt: 0.016
        iterations: 100
        max_velocity: 1.0
        max_force: 100.0
        repel_k: 50.0
        spring_k: 0.005
        mass_scale: 1.0
        boundary_damping: 0.95

system:
  network:
    port: 4000
    bind_address: "0.0.0.0"
EOF

# Restart to trigger migration
docker restart <container-name>

# Check logs
docker logs -f <container-name> | grep -i migration
```

---

## Verify Fix

### Check Database:
```bash
sqlite3 /app/data/ontology_db.sqlite3 << EOF
.mode column
.headers on
SELECT key, substr(value_json, 1, 50) as value_preview FROM settings WHERE key = 'app_full_settings';
SELECT profile_name, damping, spring_k FROM physics_settings;
EOF
```

Expected output:
```
key                value_preview
-----------------  --------------------------------------------------
app_full_settings  {"visualisation":{"rendering":{"ambient_light_...

profile_name  damping  spring_k
------------  -------  --------
default       0.95     0.005
```

### Check API:
```bash
curl http://localhost:4000/api/settings | jq .
```

Expected: Full settings object, NOT empty/null

---

## Troubleshooting

### If database is locked:
```bash
# Stop application
docker stop <container>

# Seed database
sqlite3 /app/data/ontology_db.sqlite3 < /tmp/seed_settings.sql

# Restart
docker start <container>
```

### If migration still fails:
Check migration detection logic in `src/services/settings_migration.rs:378`:
```rust
pub fn is_migrated(&self) -> bool {
    match self.db_service.get_setting("version") {  // Should check "app_full_settings"
        Ok(Some(_)) => true,
        _ => false,
    }
}
```

**Fix**: Change `"version"` to `"app_full_settings"` or ensure `version` key exists.

### If settings still empty:
Check actor initialization in `src/actors/optimized_settings_actor.rs:207`:
```rust
fn load_settings_from_database() {
    match db.load_all_settings() {
        Ok(Some(settings)) => settings,  // ← Should load from DB
        _ => AppFullSettings::default()  // ← Falls back to empty
    }
}
```

Verify database actually has data before restarting.

---

## Production Deployment

### Before Deployment:
1. ✅ Create `settings.yaml` in deployment package
2. ✅ Run migration in staging first
3. ✅ Backup existing database
4. ✅ Test settings API endpoint

### Deployment Steps:
```bash
# 1. Backup
cp /app/data/ontology_db.sqlite3 /backup/ontology_db_$(date +%Y%m%d).sqlite3

# 2. Apply seed/migration
sqlite3 /app/data/ontology_db.sqlite3 < seed_settings.sql

# 3. Verify
sqlite3 /app/data/ontology_db.sqlite3 "SELECT COUNT(*) FROM settings;"

# 4. Restart
systemctl restart turbo-flow-backend
```

---

## Related Files

- **Root Cause Analysis**: `docs/SETTINGS_MIGRATION_ROOT_CAUSE_ANALYSIS.md`
- **Migration Code**: `src/services/settings_migration.rs`
- **Database Service**: `src/services/database_service.rs`
- **Schema**: `schema/ontology_db.sql`
- **Settings Actor**: `src/actors/optimized_settings_actor.rs`

---

**Last Updated**: 2025-10-21
**Status**: ACTIONABLE FIX AVAILABLE
