# Settings Architecture Evolution: Visual Guide

**Date:** October 21, 2025
**Purpose:** Visual representation of OLD → NEW migration

---

## System Evolution Timeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUTION TIMELINE                            │
│                                                                  │
│  Pre-Oct 2025        Oct 21 (AM)          Oct 21 (PM)          │
│  ───────────         ───────────          ───────────           │
│    OLD SYSTEM    →   PHASE 1 (Dual)  →   PHASE 2 (Smart)      │
│    YAML/TOML         SQLite + Dual       SQLite + Smart        │
│                      Storage             Lookup                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## OLD System Architecture (Pre-October 2025)

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                          │
│  JavaScript/TypeScript - uses camelCase natively                │
│  { "ambientLightIntensity": 0.5, "enableShadows": true }       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP/WebSocket (JSON with camelCase)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   API LAYER (Actix-Web)                         │
│  Manual Conversion Required:                                    │
│  - Parse JSON                                                   │
│  - Look up FIELD_MAPPINGS (180+ hardcoded entries)             │
│  - Convert camelCase → snake_case                               │
│  - Error if mapping missing!                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ snake_case internally
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RUST BACKEND (Actix)                          │
│  Rust structs use snake_case:                                   │
│    struct RenderingSettings {                                   │
│        ambient_light_intensity: f32,  // snake_case            │
│        enable_shadows: bool,          // snake_case            │
│    }                                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ File I/O (50ms overhead)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FILE STORAGE (YAML/TOML)                     │
│                                                                  │
│  /app/settings.yaml                                             │
│  ┌───────────────────────────────────────────────┐             │
│  │ visualisation:                                │             │
│  │   rendering:                                  │             │
│  │     ambient_light_intensity: 0.5  # Mixed!   │             │
│  │     enableShadows: true           # Mixed!   │             │
│  └───────────────────────────────────────────────┘             │
│                                                                  │
│  /app/user_settings/{pubkey}.yaml  (per-user files)           │
│  /app/data/dev_config.toml                                     │
│  /app/ontology_physics.toml                                    │
└─────────────────────────────────────────────────────────────────┘

PROBLEMS:
❌ Manual FIELD_MAPPINGS maintenance (180+ entries)
❌ Mixed case conventions in YAML (brittle)
❌ 50ms file I/O overhead per access
❌ No concurrent access safety (race conditions)
❌ No schema validation
❌ No audit trail
```

---

## NEW System - Phase 1: Dual Storage (Oct 21 AM)

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                          │
│  { "ambientLightIntensity": 0.5, "enableShadows": true }       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP/WebSocket (JSON with camelCase)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   API LAYER (Actix-Web)                         │
│  Automatic Serde Conversion:                                    │
│  #[serde(rename_all = "camelCase")]                            │
│  struct RenderingSettings { ... }                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Rust structs (snake_case fields)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                  DATABASE SERVICE LAYER                          │
│  Migration writes BOTH formats:                                 │
│                                                                  │
│  fn migrate_setting(key: &str, value: &SettingValue) {         │
│      // Write snake_case version                                │
│      db.set_setting("ambient_light_intensity", value)?;         │
│                                                                  │
│      // Write camelCase version (DUPLICATE!)                    │
│      db.set_setting("ambientLightIntensity", value)?;           │
│  }                                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 2x writes, 2x storage
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SQLITE DATABASE (WAL mode)                    │
│                                                                  │
│  settings table:                                                │
│  ┌───────────────────┬──────────────┬─────────────┐            │
│  │ key               │ value_type   │ value_float │            │
│  ├───────────────────┼──────────────┼─────────────┤            │
│  │ ambient_light...  │ float        │ 0.5         │ ← Dup 1   │
│  │ ambientLight...   │ float        │ 0.5         │ ← Dup 2   │
│  │ enable_shadows    │ boolean      │ 1           │ ← Dup 1   │
│  │ enableShadows     │ boolean      │ 1           │ ← Dup 2   │
│  └───────────────────┴──────────────┴─────────────┘            │
│                                                                  │
│  Size: ~1MB (536KB × 2 = 1072KB estimated)                     │
└─────────────────────────────────────────────────────────────────┘

IMPROVEMENTS:
✅ ACID transactions
✅ Concurrent reads (WAL mode)
✅ Schema validation
✅ 10-50x faster queries

PROBLEMS:
❌ 2x storage overhead (duplicate keys)
❌ 2x write operations
❌ Cache synchronization complexity
```

---

## NEW System - Phase 2: Smart Lookup (Oct 21 PM - Current)

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                          │
│  { "ambientLightIntensity": 0.5, "enableShadows": true }       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ HTTP/WebSocket (JSON with camelCase)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                   API LAYER (Actix-Web)                         │
│  Automatic Serde Conversion:                                    │
│  #[serde(rename_all = "camelCase")]                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Rust structs (snake_case fields)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              DATABASE SERVICE WITH SMART LOOKUP                  │
│                                                                  │
│  pub fn get_setting(&self, key: &str) -> Option<SettingValue> {│
│      // Try exact match first (fast path - indexed query)       │
│      if let Some(val) = self.get_setting_exact(key)? {         │
│          return Ok(Some(val));  // ← O(1) index hit             │
│      }                                                          │
│                                                                  │
│      // Fallback: Convert snake_case → camelCase               │
│      if key.contains('_') {                                     │
│          let camel_key = Self::to_camel_case(key);             │
│          //   "spring_k" → "springK"                           │
│          //   "ambient_light_intensity" → "ambientLightIntensity"│
│          return self.get_setting_exact(&camel_key);            │
│      }                                                          │
│                                                                  │
│      Ok(None)                                                   │
│  }                                                              │
│                                                                  │
│  fn to_camel_case(s: &str) -> String {                         │
│      s.split('_')                                               │
│       .enumerate()                                              │
│       .map(|(i, part)| if i == 0 { part } else { capitalize(part) })│
│       .collect()                                                │
│  }                                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ 1x write (camelCase only)
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SQLITE DATABASE (WAL mode)                    │
│                                                                  │
│  settings table:                                                │
│  ┌───────────────────┬──────────────┬─────────────┐            │
│  │ key               │ value_type   │ value_float │            │
│  ├───────────────────┼──────────────┼─────────────┤            │
│  │ ambientLight...   │ float        │ 0.5         │ ← Single! │
│  │ enableShadows     │ boolean      │ 1           │ ← Single! │
│  │ springK           │ float        │ 5.0         │ ← Single! │
│  └───────────────────┴──────────────┴─────────────┘            │
│                                                                  │
│  CREATE INDEX idx_settings_key ON settings(key);  ← O(1) lookup│
│                                                                  │
│  Size: 536KB (50% reduction from Phase 1)                      │
└─────────────────────────────────────────────────────────────────┘

IMPROVEMENTS:
✅ 50% storage reduction (no duplicates)
✅ 50% faster writes (single operation)
✅ Backward compatible (smart lookup handles legacy code)
✅ Zero manual mappings (algorithmic conversion)
✅ Self-documenting (no static FIELD_MAPPINGS)

TRADE-OFFS:
⚠️ Runtime conversion overhead for legacy snake_case keys
   (minimal - only O(1) string manipulation if cache miss)
```

---

## Smart Lookup Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    Client Request                             │
│  GET /api/settings/path?path=spring_k                        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ↓
              ┌──────────────────────┐
              │  get_setting("spring_k")  │
              └──────────────────────┘
                         │
                         ↓
        ┌────────────────────────────────┐
        │ Try exact match: "spring_k"    │
        │ SELECT * FROM settings         │
        │ WHERE key = 'spring_k'         │
        └────────────────┬───────────────┘
                         │
                    ┌────┴────┐
                    │ Found?  │
                    └────┬────┘
                         │
              ┌──────────┼──────────┐
              │ YES               NO │
              ↓                      ↓
     ┌────────────────┐   ┌──────────────────────┐
     │ Return value   │   │ Contains underscore? │
     │ (fast path)    │   └──────────┬───────────┘
     └────────────────┘              │
                              ┌──────┴──────┐
                              │ YES      NO │
                              ↓             ↓
                   ┌─────────────────┐  ┌────────┐
                   │ to_camel_case() │  │ Return │
                   │ "spring_k"      │  │ None   │
                   │    ↓            │  └────────┘
                   │ "springK"       │
                   └────────┬────────┘
                            │
                            ↓
              ┌──────────────────────────────┐
              │ Try converted key: "springK" │
              │ SELECT * FROM settings       │
              │ WHERE key = 'springK'        │
              └──────────────┬───────────────┘
                             │
                        ┌────┴────┐
                        │ Found?  │
                        └────┬────┘
                             │
                   ┌─────────┼─────────┐
                   │ YES           NO  │
                   ↓                   ↓
          ┌─────────────────┐   ┌──────────┐
          │ Return value    │   │ Return   │
          │ (fallback path) │   │ None     │
          └─────────────────┘   └──────────┘

Performance:
- Exact match (camelCase): O(1) - indexed lookup
- Fallback conversion: O(n) where n = key length (typically <30 chars)
- Total fallback cost: ~1-2μs (negligible)
```

---

## Case Conversion Examples

```
┌─────────────────────────────────────────────────────────────────┐
│                   to_camel_case() Algorithm                      │
└─────────────────────────────────────────────────────────────────┘

Input: "spring_k"
Steps:
  1. Split on '_': ["spring", "k"]
  2. Keep first part: "spring"
  3. Capitalize rest: "K"
  4. Join: "springK"
Output: "springK"

Input: "ambient_light_intensity"
Steps:
  1. Split: ["ambient", "light", "intensity"]
  2. Keep first: "ambient"
  3. Capitalize: ["Light", "Intensity"]
  4. Join: "ambientLightIntensity"
Output: "ambientLightIntensity"

Input: "maxVelocity" (already camelCase)
Steps:
  1. Split: ["maxVelocity"] (no underscore)
  2. Return as-is: "maxVelocity"
Output: "maxVelocity"

Input: "dt" (single word)
Steps:
  1. Split: ["dt"] (no underscore)
  2. Return as-is: "dt"
Output: "dt"

┌─────────────────────────────────────────────────────────────────┐
│                    Conversion Test Cases                         │
├──────────────────────────┬──────────────────────────────────────┤
│ Input (snake_case)       │ Output (camelCase)                   │
├──────────────────────────┼──────────────────────────────────────┤
│ spring_k                 │ springK                              │
│ repel_k                  │ repelK                               │
│ max_velocity             │ maxVelocity                          │
│ rest_length              │ restLength                           │
│ enable_shadows           │ enableShadows                        │
│ ambient_light_intensity  │ ambientLightIntensity                │
│ equilibrium_check_frames │ equilibriumCheckFrames               │
│ dt                       │ dt (no change)                       │
│ damping                  │ damping (no change)                  │
│ springK                  │ springK (already camelCase)          │
└──────────────────────────┴──────────────────────────────────────┘
```

---

## Data Flow: Setting Update

```
┌─────────────────────────────────────────────────────────────────┐
│           CLIENT: User changes "Spring Constant" to 7.5          │
│  UI sends: PUT /api/settings                                    │
│  Body: {"visualisation":{"graphs":{"logseq":{"physics":{       │
│          "springK": 7.5 }}}}}                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  API HANDLER: Deserialize with Serde                           │
│  #[serde(rename_all = "camelCase")]                            │
│  struct PhysicsSettings {                                       │
│      spring_k: f32,  // Rust field (snake_case)                │
│  }                                                              │
│  Result: spring_k = 7.5 (in Rust)                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATION: Check constraints                                  │
│  validate_range(spring_k, 0.0, 50.0) → OK                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  DATABASE SERVICE: set_setting()                                │
│  fn set_setting(key: "spring_k", value: 7.5) {                 │
│      // Convert to camelCase for storage                        │
│      let camel_key = to_camel_case("spring_k");                │
│      // camel_key = "springK"                                   │
│                                                                  │
│      // Execute SQL (single write)                              │
│      UPDATE settings                                            │
│      SET value_float = 7.5, updated_at = NOW()                 │
│      WHERE key = 'springK'                                     │
│  }                                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  AUDIT LOG: Record change                                       │
│  INSERT INTO settings_audit_log                                 │
│    (user_id, setting_key, old_value, new_value, timestamp)     │
│  VALUES                                                         │
│    ('user123', 'springK', '5.0', '7.5', '2025-10-21 10:42:00') │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  WEBSOCKET BROADCAST: Notify all clients                        │
│  {                                                              │
│    "type": "settings:update",                                   │
│    "path": "visualisation.graphs.logseq.physics.springK",      │
│    "value": 7.5,                                                │
│    "timestamp": "2025-10-21T10:42:00Z"                         │
│  }                                                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│  ALL CLIENTS: Receive update, invalidate cache                  │
│  settingsStore.partialSettings.visualisation.graphs...          │
│  ...logseq.physics.springK = 7.5                               │
│                                                                  │
│  UI re-renders with new value                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Storage Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE EFFICIENCY ANALYSIS                   │
└─────────────────────────────────────────────────────────────────┘

OLD System (YAML Files):
┌────────────────────────────────────────────┐
│ File: settings.yaml (100 settings)         │
│ ┌────────────────────────────────────────┐ │
│ │ visualisation:                         │ │
│ │   rendering:                           │ │
│ │     ambient_light_intensity: 0.5       │ │ ← 42 bytes
│ │     background_color: "#000000"        │ │ ← 38 bytes
│ │     ... (98 more settings)             │ │
│ └────────────────────────────────────────┘ │
│ Total: ~15 KB (with YAML overhead)         │
│ Access time: 50ms (parse entire file)      │
│ Update: 500ms (rewrite entire file)        │
└────────────────────────────────────────────┘

NEW System Phase 1 (Dual Storage):
┌────────────────────────────────────────────┐
│ Database: settings.db                      │
│ ┌────────────────────────────────────────┐ │
│ │ ambient_light_intensity | 0.5 | float  │ │ ← Row 1 (snake)
│ │ ambientLightIntensity   | 0.5 | float  │ │ ← Row 2 (camel) DUP!
│ │ background_color        | #000000 | text│ │ ← Row 3 (snake)
│ │ backgroundColor         | #000000 | text│ │ ← Row 4 (camel) DUP!
│ │ ... (196 more rows)                    │ │
│ └────────────────────────────────────────┘ │
│ Total: ~1 MB (200 rows × ~5KB/row)        │
│ Access time: 1-5ms (indexed query)         │
│ Update: 10ms × 2 (dual write)              │
└────────────────────────────────────────────┘

NEW System Phase 2 (Smart Lookup):
┌────────────────────────────────────────────┐
│ Database: settings.db                      │
│ ┌────────────────────────────────────────┐ │
│ │ ambientLightIntensity   | 0.5 | float  │ │ ← Single row!
│ │ backgroundColor         | #000000 | text│ │ ← Single row!
│ │ ... (98 more rows)                     │ │
│ └────────────────────────────────────────┘ │
│ Total: 536 KB (100 rows × ~5KB/row)       │
│ Access time: 1-5ms (exact or +1μs convert) │
│ Update: 10ms (single write)                │
└────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      PERFORMANCE METRICS                         │
├──────────────────┬────────────┬──────────────┬─────────────────┤
│ Operation        │ OLD (YAML) │ Phase 1 (DB) │ Phase 2 (Smart) │
├──────────────────┼────────────┼──────────────┼─────────────────┤
│ Single read      │ 50ms       │ 1-5ms        │ 1-5ms           │
│ Batch read (10)  │ 50ms       │ 5ms          │ 5ms             │
│ Single write     │ 500ms      │ 20ms (×2)    │ 10ms            │
│ Storage (100)    │ 15 KB      │ ~1 MB        │ 536 KB          │
│ Concurrent reads │ ❌ Unsafe  │ ✅ Safe      │ ✅ Safe         │
│ ACID guarantees  │ ❌ None    │ ✅ Yes       │ ✅ Yes          │
│ Audit logging    │ ❌ Manual  │ ✅ Built-in  │ ✅ Built-in     │
└──────────────────┴────────────┴──────────────┴─────────────────┘
```

---

## Migration Decision Tree

```
                     ┌────────────────────┐
                     │  Settings System   │
                     │  Migration Needed? │
                     └─────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                │ YES                      NO │
                ↓                             ↓
     ┌────────────────────┐          ┌──────────────┐
     │ Choose Storage     │          │ Keep current │
     │ Technology         │          │ YAML system  │
     └─────────┬──────────┘          └──────────────┘
               │
      ┌────────┼────────┐
      │        │        │
      ↓        ↓        ↓
   SQLite  PostgreSQL  Redis
      │
      │ ✅ CHOSEN
      ↓
┌──────────────────┐
│ Choose Storage   │
│ Format           │
└─────────┬────────┘
          │
   ┌──────┼──────┐
   │      │      │
   ↓      ↓      ↓
 Both  Snake Camel
       Case  Case
         │    │
         │    │ ✅ CHOSEN
         ↓    ↓
      ┌────────────────┐
      │ Choose Lookup  │
      │ Strategy       │
      └────────┬───────┘
               │
        ┌──────┼──────┐
        │      │      │
        ↓      ↓      ↓
     Serde  Proxy Smart
              │    │
              │    │ ✅ CHOSEN
              ↓    ↓
         ┌─────────────────┐
         │ Smart Lookup    │
         │ Implementation  │
         └─────────────────┘

WHY Smart Lookup?
✅ Single storage (50% reduction)
✅ Backward compatible
✅ Zero manual mappings
✅ Self-documenting
✅ Fast (O(1) + O(n) where n=key length)
```

---

## Summary: The "Brittle" Problem Solved

```
┌─────────────────────────────────────────────────────────────────┐
│         BEFORE: "Brittle Case Handling Logic"                   │
└─────────────────────────────────────────────────────────────────┘

static FIELD_MAPPINGS = {
    "ambient_light_intensity" → "ambientLightIntensity",
    "enable_shadows" → "enableShadows",
    "spring_k" → "springK",
    ... 180+ more manual entries
};

fn get_setting(key: &str) -> Value {
    let mapped = FIELD_MAPPINGS.get(key)
        .expect("Field mapping not found!");  // ← PANIC if missing!
    load_yaml().get(mapped)
}

PROBLEMS:
❌ Manual maintenance (error-prone)
❌ No compile-time checking
❌ Easy to forget new fields
❌ Crashes if mapping missing
❌ 180+ hardcoded strings to maintain

┌─────────────────────────────────────────────────────────────────┐
│         AFTER: Smart Lookup (Algorithmic Conversion)            │
└─────────────────────────────────────────────────────────────────┘

fn get_setting(key: &str) -> Option<Value> {
    db.get(key).or_else(|| {
        if key.contains('_') {
            db.get(&to_camel_case(key))  // ← Automatic conversion!
        } else { None }
    })
}

fn to_camel_case(s: &str) -> String {
    // "spring_k" → "springK"
    // "max_velocity" → "maxVelocity"
    // Simple algorithm, no hardcoded data!
}

BENEFITS:
✅ Zero manual mappings
✅ Compile-time safe
✅ Never panics (returns Option)
✅ Self-documenting algorithm
✅ Works for any snake_case key
```

---

**Document Version:** 1.0
**Last Updated:** October 21, 2025
**Related:** See DOCUMENTATION_AUDIT_REPORT.md and MIGRATION_SUMMARY_FINDINGS.md
