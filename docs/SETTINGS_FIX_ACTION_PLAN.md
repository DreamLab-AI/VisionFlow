# Settings Migration Fix - Action Plan
## Quick Reference for "No Settings Available" Issue

**Status:** 🔴 CRITICAL - Database Empty, 50% Schema Coverage Gap
**Created:** 2025-10-21
**ETA to Fix:** 1-2 hours (Phase 1), 1-2 days (Complete)

---

## 🎯 Quick Summary

**Problem:** Components request settings paths that don't exist in schema → Database returns `null` → UI shows "No settings available"

**Root Cause:**
1. Database is **empty** (0 bytes) - migration never ran
2. Migration script only covers **50%** of schema (physics + dev config)
3. Missing 6 critical namespaces (181 paths total)

**Impact:** All graph visualization tabs broken (sync, effects, performance, interaction, export)

---

## ⚡ Immediate Fix (1-2 hours)

### Step 1: Add Missing Structs

**File:** `/home/devuser/workspace/project/src/config/mod.rs`

**Location:** After line 1133 (after `NetworkSettings`)

```rust
// ===== ADD THESE 6 STRUCTS =====

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
#[serde(rename_all = "camelCase")]
pub struct SyncSettings {
    #[serde(default = "default_true", alias = "enabled")]
    pub enabled: bool,
    #[serde(default = "default_true", alias = "camera")]
    pub camera: bool,
    #[serde(default = "default_true", alias = "selection")]
    pub selection: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
#[serde(rename_all = "camelCase")]
pub struct EffectsSettings {
    #[serde(default = "default_false", alias = "bloom")]
    pub bloom: bool,
    #[serde(default = "default_false", alias = "glow")]
    pub glow: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Type)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceSettings {
    #[serde(default = "default_true", alias = "auto_optimize")]
    pub auto_optimize: bool,
    #[serde(default = "default_false", alias = "simplify_edges")]
    pub simplify_edges: bool,
    #[serde(default = "default_cull_distance", alias = "cull_distance")]
    pub cull_distance: f32,
}

impl Default for PerformanceSettings {
    fn default() -> Self {
        Self {
            auto_optimize: true,
            simplify_edges: false,
            cull_distance: 1000.0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Type)]
#[serde(rename_all = "camelCase")]
pub struct InteractionSettings {
    #[serde(default = "default_true", alias = "enable_hover")]
    pub enable_hover: bool,
    #[serde(default = "default_true", alias = "enable_click")]
    pub enable_click: bool,
    #[serde(default = "default_true", alias = "enable_drag")]
    pub enable_drag: bool,
    #[serde(default = "default_hover_delay", alias = "hover_delay")]
    pub hover_delay: u32,
}

impl Default for InteractionSettings {
    fn default() -> Self {
        Self {
            enable_hover: true,
            enable_click: true,
            enable_drag: true,
            hover_delay: 300,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, Type)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
    #[serde(default = "default_export_format", alias = "format")]
    pub format: String,
    #[serde(default = "default_true", alias = "include_metadata")]
    pub include_metadata: bool,
}

// ===== HELPER FUNCTIONS (add near other default_* functions) =====

fn default_true() -> bool { true }
fn default_false() -> bool { false }
fn default_cull_distance() -> f32 { 1000.0 }
fn default_hover_delay() -> u32 { 300 }
fn default_export_format() -> String { "png".to_string() }
```

### Step 2: Update VisualisationSettings

**Find this struct (around line 1114):**

```rust
#[derive(Debug, Serialize, Deserialize, Clone, Default, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct VisualisationSettings {
    #[validate(nested)]
    pub rendering: RenderingSettings,
    #[validate(nested)]
    pub animations: AnimationSettings,
    #[validate(nested)]
    pub glow: GlowSettings,
    #[validate(nested)]
    pub bloom: BloomSettings,
    #[validate(nested)]
    pub hologram: HologramSettings,
    #[validate(nested)]
    pub graphs: GraphsSettings,

    // ===== ADD THESE TWO LINES =====
    #[validate(nested)]
    pub sync: SyncSettings,
    #[validate(nested)]
    pub effects: EffectsSettings,
    // ==============================

    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraSettings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub space_pilot: Option<SpacePilotSettings>,
}
```

### Step 3: Update AppFullSettings

**Find this struct (search for "pub struct AppFullSettings"):**

```rust
#[derive(Debug, Serialize, Deserialize, Clone, Type, Validate)]
#[serde(rename_all = "camelCase")]
pub struct AppFullSettings {
    pub visualisation: VisualisationSettings,
    pub system: SystemSettings,

    // ===== ADD THESE THREE LINES =====
    pub performance: PerformanceSettings,
    pub interaction: InteractionSettings,
    pub export: ExportSettings,
    // =================================

    // ... rest of existing fields
}
```

### Step 4: Rebuild & Test

```bash
# Rebuild server
cd /home/devuser/workspace/project
cargo build --release

# Test settings API (after starting server)
curl http://localhost:4000/api/settings/path?path=visualisation.sync.enabled
curl http://localhost:4000/api/settings/path?path=performance.autoOptimize
curl http://localhost:4000/api/settings/path?path=interaction.enableHover

# Expected: All return actual values (true/false), NOT null
```

---

## 🗄️ Database Seeding (Phase 2)

### Create Seed Script

**File:** `scripts/seed_missing_settings.sql`

```sql
-- Visualisation Sync Settings (3 paths)
INSERT OR IGNORE INTO settings (category, key, value, value_type, description) VALUES
  ('visualisation', 'visualisation.sync.enabled', 'true', 'boolean', 'Enable multi-user synchronization'),
  ('visualisation', 'visualisation.sync.camera', 'true', 'boolean', 'Sync camera position across users'),
  ('visualisation', 'visualisation.sync.selection', 'true', 'boolean', 'Sync node selection across users');

-- Visualisation Effects Settings (2 paths)
INSERT OR IGNORE INTO settings (category, key, value, value_type, description) VALUES
  ('visualisation', 'visualisation.effects.bloom', 'false', 'boolean', 'Enable bloom post-processing'),
  ('visualisation', 'visualisation.effects.glow', 'false', 'boolean', 'Enable glow effect on nodes');

-- Performance Settings (3 paths)
INSERT OR IGNORE INTO settings (category, key, value, value_type, description) VALUES
  ('performance', 'performance.autoOptimize', 'true', 'boolean', 'Auto-optimize rendering'),
  ('performance', 'performance.simplifyEdges', 'false', 'boolean', 'Simplify edge rendering when zoomed'),
  ('performance', 'performance.cullDistance', '1000.0', 'number', 'Node culling distance (meters)');

-- Interaction Settings (4 paths)
INSERT OR IGNORE INTO settings (category, key, value, value_type, description) VALUES
  ('interaction', 'interaction.enableHover', 'true', 'boolean', 'Enable hover interactions'),
  ('interaction', 'interaction.enableClick', 'true', 'boolean', 'Enable click interactions'),
  ('interaction', 'interaction.enableDrag', 'true', 'boolean', 'Enable drag interactions'),
  ('interaction', 'interaction.hoverDelay', '300', 'number', 'Hover tooltip delay (ms)');

-- Export Settings (2 paths)
INSERT OR IGNORE INTO settings (category, key, value, value_type, description) VALUES
  ('export', 'export.format', '"png"', 'string', 'Default export format'),
  ('export', 'export.includeMetadata', 'true', 'boolean', 'Include metadata in exports');

-- Snake_case aliases (for dual key format support)
INSERT OR IGNORE INTO settings (category, key, value, value_type, description) VALUES
  ('visualisation', 'visualisation.sync.enabled', 'true', 'boolean', 'Enable multi-user synchronization'),
  ('interaction', 'interaction.enable_hover', 'true', 'boolean', 'Enable hover interactions'),
  ('interaction', 'interaction.enable_click', 'true', 'boolean', 'Enable click interactions'),
  ('interaction', 'interaction.enable_drag', 'true', 'boolean', 'Enable drag interactions'),
  ('interaction', 'interaction.hover_delay', '300', 'number', 'Hover tooltip delay (ms)');
```

### Apply Seeding

```bash
# Initialize database schema if needed
cd /home/devuser/workspace/project
sqlite3 data/settings.db < schema/settings_schema.sql  # If schema.sql exists

# Seed missing settings
sqlite3 data/settings.db < scripts/seed_missing_settings.sql

# Verify seeding
sqlite3 data/settings.db "SELECT COUNT(*) FROM settings;"
# Should return > 0 (not empty)

sqlite3 data/settings.db "SELECT key FROM settings WHERE key LIKE 'visualisation.sync%';"
# Should return 3 rows
```

---

## 📊 Testing Checklist

### Manual Testing

- [ ] Server starts without errors
- [ ] API returns settings for all new paths:
  - [ ] `visualisation.sync.enabled` → `true`
  - [ ] `visualisation.effects.bloom` → `false`
  - [ ] `performance.autoOptimize` → `true`
  - [ ] `interaction.enableHover` → `true`
  - [ ] `export.format` → `"png"`
- [ ] UI graph tabs load without "No settings available"
- [ ] Settings persist after server restart
- [ ] Settings can be updated via API

### Automated Testing

```bash
# Run unit tests
cargo test settings -- --nocapture

# Run integration tests
cargo test --test settings_integration

# Check code coverage
cargo tarpaulin --out Stdout --exclude-files tests/ -- settings
```

---

## 🔍 Verification

### Database Status

```bash
# Check database size (should be > 0 bytes)
ls -lh /home/devuser/workspace/project/data/settings.db

# Count total settings
sqlite3 data/settings.db "SELECT COUNT(*) FROM settings;"

# List all categories
sqlite3 data/settings.db "SELECT DISTINCT category FROM settings ORDER BY category;"

# Show sample of new settings
sqlite3 data/settings.db "SELECT key, value FROM settings WHERE category IN ('performance', 'interaction', 'export');"
```

### API Testing

```bash
# Test each critical path
for path in \
  "visualisation.sync.enabled" \
  "visualisation.sync.camera" \
  "visualisation.effects.bloom" \
  "performance.autoOptimize" \
  "interaction.enableHover" \
  "export.format"
do
  echo "Testing: $path"
  curl -s "http://localhost:4000/api/settings/path?path=$path" | jq .
done
```

Expected output: All return actual values (not `null`)

---

## 📈 Migration Coverage Report

### Current Status

| Category | Paths | Migrated | Coverage |
|----------|-------|----------|----------|
| Physics Profiles | 96 | 96 | ✅ 100% |
| Dev Config | 88 | 88 | ✅ 100% |
| **Visualisation (new)** | **5** | **0** | ❌ 0% |
| **Performance** | **3** | **0** | ❌ 0% |
| **Interaction** | **4** | **0** | ❌ 0% |
| **Export** | **2** | **0** | ❌ 0% |
| System Settings | 35 | 0 | ⚠️ 0% |
| **TOTAL** | **233** | **184** | **79%** |

### After Phase 1 Fix

| Category | Paths | Migrated | Coverage |
|----------|-------|----------|----------|
| Physics Profiles | 96 | 96 | ✅ 100% |
| Dev Config | 88 | 88 | ✅ 100% |
| **Visualisation (new)** | **5** | **5** | ✅ **100%** |
| **Performance** | **3** | **3** | ✅ **100%** |
| **Interaction** | **4** | **4** | ✅ **100%** |
| **Export** | **2** | **2** | ✅ **100%** |
| System Settings | 35 | 0 | ⚠️ 0% |
| **TOTAL** | **233** | **198** | **85%** |

---

## 🚨 Rollback Plan

If deployment fails:

```bash
# 1. Stop server
sudo systemctl stop your-app-service

# 2. Restore previous binary
cp /backup/old_binary /path/to/production/binary

# 3. Restore database (if seeded)
cp /backup/settings.db.backup data/settings.db

# 4. Restart server
sudo systemctl start your-app-service

# 5. Verify rollback
curl http://localhost:4000/health
```

---

## 📚 Related Documentation

- **Full Analysis:** `/docs/architecture/settings-migration-impact-synthesis.md`
- **Migration Guide:** `/docs/settings-migration-guide.md`
- **Schema Reference:** `/docs/settings-schema.md`
- **Client Analysis:** `/client/docs/SETTINGS_NO_AVAILABLE_ANALYSIS.md`

---

## 📞 Support

**Issue Tracker:** [GitHub Issues](https://github.com/your-repo/issues)
**Slack Channel:** #settings-migration
**On-Call:** System Architect Team

---

**Last Updated:** 2025-10-21
**Status:** ✅ Ready for Implementation
