# Backend Settings Implementation Audit Report
**Generated:** 2025-10-21
**Scope:** Rust server settings storage, API contracts, and case conversion analysis

---

## Executive Summary

**ACTUAL STATE:** The backend implements a **hybrid camelCase storage + smart fallback lookup** system, NOT the dual storage format described in documentation.

**Key Finding:** Database stores in **camelCase ONLY**, with intelligent snake_case→camelCase conversion during reads.

---

## 1. Storage Format Analysis

### Database Schema (`schema/ontology_db.sql`)
**Lines 42-73: `physics_settings` table**

```sql
CREATE TABLE IF NOT EXISTS physics_settings (
    damping REAL NOT NULL DEFAULT 0.95,
    dt REAL NOT NULL DEFAULT 0.016,
    max_velocity REAL NOT NULL DEFAULT 1.0,    -- SNAKE_CASE column names
    max_force REAL NOT NULL DEFAULT 100.0,
    repel_k REAL NOT NULL DEFAULT 50.0,        -- SNAKE_CASE column names
    spring_k REAL NOT NULL DEFAULT 0.005,
    ...
)
```

**Finding:** Table columns use **snake_case** (SQL convention).

### Generic Settings Table (`settings` table)
**Lines 21-35: Generic key-value storage**

```sql
CREATE TABLE IF NOT EXISTS settings (
    key TEXT NOT NULL UNIQUE,
    value_type TEXT NOT NULL,
    value_json TEXT,
    ...
)
```

**Finding:** Stores arbitrary keys in the `key` column. Migration stores **camelCase** keys.

---

## 2. Migration Implementation

### File: `src/services/settings_migration.rs`

#### **Lines 188-197: Single Format Storage (camelCase ONLY)**
```rust
/// Migrate a single setting to database
fn migrate_setting(&self, key: &str, value: &YamlValue) -> Result<(), String> {
    let setting_value = self.yaml_to_setting_value(value)?;

    // Store with camelCase key only
    self.db_service.set_setting(key, setting_value, None)
        .map_err(|e| format!("Failed to store setting: {}", e))?;

    debug!("Migrated: {}", key);
    Ok(())
}
```

**CRITICAL:** Comment states "Store with camelCase key only" — **NO dual storage**.

#### **Lines 318-350: Physics Profile Migration**
```rust
fn migrate_physics_profile(&self, profile_name: &str, physics: &YamlValue) -> Result<(), String> {
    let settings = PhysicsSettings {
        damping: self.get_f32(physics, "damping").unwrap_or(0.95),
        max_velocity: self.get_f32(physics, "maxVelocity").unwrap_or(1.0),  // Reads camelCase from YAML
        repel_k: self.get_f32(physics, "repelK").unwrap_or(50.0),          // Reads camelCase
        spring_k: self.get_f32(physics, "springK").unwrap_or(0.005),
        ...
    };
    self.db_service.save_physics_settings(profile_name, &settings)
}
```

**Finding:** Migration reads **camelCase** from YAML, stores into snake_case SQL columns via Rust struct mapping.

---

## 3. Database Service Smart Lookup

### File: `src/services/database_service.rs`

#### **Lines 74-93: snake_case → camelCase Converter**
```rust
/// Convert snake_case to camelCase
/// Examples: "spring_k" -> "springK", "max_velocity" -> "maxVelocity"
fn to_camel_case(s: &str) -> String {
    let parts: Vec<&str> = s.split('_').collect();
    if parts.len() == 1 {
        return s.to_string();
    }

    let mut result = parts[0].to_string();
    for part in &parts[1..] {
        if !part.is_empty() {
            let mut chars = part.chars();
            if let Some(first) = chars.next() {
                result.push(first.to_ascii_uppercase());
                result.push_str(chars.as_str());
            }
        }
    }
    result
}
```

#### **Lines 120-146: Smart Fallback Lookup**
```rust
/// Get hierarchical settings by key path with intelligent camelCase/snake_case fallback
///
/// This method provides smart lookup:
/// 1. First tries exact match with the provided key
/// 2. If not found and key contains underscore, converts to camelCase and retries
///
/// Examples:
/// - Database has "springK" = 150.0
/// - `get_setting("springK")` -> Direct hit, returns 150.0
/// - `get_setting("spring_k")` -> Converts to "springK", returns 150.0
pub fn get_setting(&self, key: &str) -> SqliteResult<Option<SettingValue>> {
    // Try exact match first
    if let Some(value) = self.get_setting_exact(key)? {
        return Ok(Some(value));
    }

    // If not found and key contains underscore, try camelCase conversion
    if key.contains('_') {
        let camel_key = Self::to_camel_case(key);
        if let Some(value) = self.get_setting_exact(&camel_key)? {
            return Ok(Some(value));
        }
    }

    // Not found with either key format
    Ok(None)
}
```

**CRITICAL FINDING:** This is the **"smart lookup" implementation** mentioned in documentation.

**Behavior:**
1. Database stores: `{ "springK": 150.0 }`
2. Client requests: `get_setting("spring_k")`
3. Conversion: `"spring_k"` → `"springK"`
4. Returns: `150.0`

**Storage Reality:** ONLY stores camelCase. Fallback is READ-ONLY conversion.

---

## 4. Settings Service Layer

### File: `src/services/settings_service.rs`

#### **Lines 59-92: Cache + Normalization**
```rust
/// Get setting by key (supports both camelCase and snake_case)
pub async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>, String> {
    // Normalize key to snake_case
    let normalized_key = self.normalize_key(key);

    // Check cache first
    {
        let cache = self.cache.read().await;
        if let Some(cached) = cache.settings.get(&normalized_key) {
            if cached.timestamp.elapsed().as_secs() < 300 { // 5 min TTL
                debug!("Cache hit for setting: {}", normalized_key);
                return Ok(Some(cached.value.clone()));
            }
        }
    }

    // Query database
    match self.db.get_setting(&normalized_key) {
        Ok(Some(value)) => {
            // Update cache...
        }
    }
}
```

#### **Lines 286-299: Normalization Logic**
```rust
/// Normalize key to snake_case (convert camelCase if needed)
fn normalize_key(&self, key: &str) -> String {
    // Simple camelCase to snake_case conversion
    let mut result = String::new();
    for (i, ch) in key.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
            result.push(ch.to_lowercase().next().unwrap());
        } else {
            result.push(ch);
        }
    }
    result
}
```

**CONTRADICTION FOUND:**
- Comment says "Normalize key to snake_case"
- But database expects **camelCase** keys!
- This layer converts `"springK"` → `"spring_k"` before querying DB
- DB's `get_setting()` then converts back: `"spring_k"` → `"springK"`

**Result:** Double conversion that cancels out!

---

## 5. API Handler Contract

### File: `src/handlers/settings_paths.rs`

#### **Lines 17-93: Path-Based GET**
```rust
pub async fn get_settings_by_path(
    req: HttpRequest,
    query: web::Query<PathQuery>,
    state: web::Data<AppState>,
) -> ActixResult<HttpResponse> {
    let path = &query.path;

    match state.settings_addr.send(GetSettings).await {
        Ok(Ok(settings)) => {
            // Use JsonPathAccessible trait to get the value
            match settings.get_json_by_path(path) {
                Ok(value) => {
                    Ok(HttpResponse::Ok().json(json!({
                        "value": value,
                        "path": path,
                        "success": true
                    })))
                }
            }
        }
    }
}
```

**API Contract:**
- **Input:** `GET /api/settings/path?path=visualisation.physics.damping`
- **Path Format:** Can be **camelCase or snake_case** (handled by `JsonPathAccessible`)
- **Output:** JSON with camelCase fields (due to `#[serde(rename_all = "camelCase")]`)

---

## 6. PathAccessible Implementation

### File: `src/config/path_access.rs`

#### **Lines 126-148: Flexible Path Navigation**
```rust
/// Navigate to a JSON value by dot-notation path
/// Supports both camelCase and snake_case field names automatically
fn navigate_json_path(root: &Value, path: &str) -> Option<Value> {
    let segments: Vec<&str> = path.split('.').filter(|s| !s.is_empty()).collect();
    let mut current = root;

    for segment in segments {
        match current {
            Value::Object(map) => {
                // Try the segment as-is first, then try snake_case conversion if camelCase fails
                current = map.get(segment)
                    .or_else(|| map.get(&camel_to_snake_case(segment)))
                    .or_else(|| map.get(&snake_to_camel_case(segment)))?;
            }
            _ => return None,
        }
    }

    Some(current.clone())
}
```

#### **Lines 258-289: Case Converters**
```rust
/// Convert snake_case to camelCase
fn snake_to_camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;

    for ch in s.chars() {
        if ch == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(ch.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(ch);
        }
    }
    result
}

/// Convert camelCase to snake_case
fn camel_to_snake_case(s: &str) -> String {
    let mut result = String::new();

    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_ascii_lowercase());
    }
    result
}
```

**Finding:** This is the **third independent case converter** in the codebase!

---

## 7. Case Conversion Functions Inventory

### **Location 1: `database_service.rs` (Lines 74-93)**
- Function: `to_camel_case()`
- Purpose: Convert snake_case → camelCase for DB key lookup
- Used in: Smart fallback for `get_setting()`

### **Location 2: `settings_service.rs` (Lines 286-299)**
- Function: `normalize_key()`
- Purpose: Convert camelCase → snake_case
- **Problem:** Works against database expectations!

### **Location 3: `path_access.rs` (Lines 258-289)**
- Functions: `snake_to_camel_case()`, `camel_to_snake_case()`
- Purpose: Flexible JSON path navigation
- Used in: API path resolution

### **Location 4: `settings_migration.rs` (Lines 241-284)**
- Functions: `to_snake_case_part()`, `to_camel_case_part()`
- Purpose: Migration-time key conversion
- **Status:** Used only during initial migration

### **Location 5: `settings_validation_fix.rs` (Lines 139-163)**
- Function: `camel_to_snake_case()`
- Purpose: GPU parameter validation
- **Special handling:** Maps `"ssspAlpha"` → `"sssp_alpha"`

---

## 8. Configuration Struct Annotations

### File: `src/config/mod.rs`

**ALL settings structs use:**
```rust
#[serde(rename_all = "camelCase")]
```

**Examples:**
- Line 429: `AppFullSettings`
- Line 468: `VisualisationSettings`
- Line 525: `GraphsSettings`
- Line 645: `PhysicsSettings`
- Line 815: `NodesSettings`
- ... (35+ structs total)

**Implication:**
- Rust fields: `enable_hologram` (snake_case)
- JSON serialization: `"enableHologram"` (camelCase)
- YAML deserialization: Accepts BOTH formats via serde aliases

---

## 9. Data Flow Diagram

```
CLIENT REQUEST
    ↓
[1] API Handler receives: path="visualisation.physics.springK"
    ↓
[2] settings_service.get_setting("springK")
    ↓
[3] normalize_key() converts: "springK" → "spring_k"  ❌ WRONG DIRECTION
    ↓
[4] database_service.get_setting("spring_k")
    ↓
[5] get_setting_exact("spring_k") → NOT FOUND (DB has "springK")
    ↓
[6] Fallback: to_camel_case("spring_k") → "springK"  ✅ CORRECTS THE MISTAKE
    ↓
[7] get_setting_exact("springK") → FOUND
    ↓
[8] Return value to client as camelCase JSON
```

**Critical Observation:** The system works by **accident**:
1. Settings service incorrectly converts camelCase → snake_case
2. Database service corrects it back with snake_case → camelCase fallback
3. Final lookup succeeds, but with unnecessary double conversion

---

## 10. Migration State

### File: `src/services/settings_migration.rs` (Lines 369-375)

```rust
/// Check if migration has been run
pub fn is_migrated(&self) -> bool {
    // Check if migrated settings exist (use full settings key instead of version)
    match self.db_service.get_setting("app_full_settings") {
        Ok(Some(_)) => true,
        _ => false,
    }
}
```

**Migration Marker:** Checks for `"app_full_settings"` key existence.

**Migration Format:**
- Reads from YAML files (paths at lines 33-36)
- Stores physics as **camelCase** (line 193: comment confirms)
- Stores in `physics_settings` table with **snake_case columns**
- Generic settings stored with **camelCase keys** in `settings` table

---

## 11. Incomplete Implementations

### PathAccessible Trait (`path_access.rs`)

**Lines 7-13: Core trait defined but NOT implemented for PhysicsSettings**
```rust
pub trait PathAccessible {
    fn get_by_path(&self, path: &str) -> Result<Box<dyn Any>, String>;
    fn set_by_path(&mut self, path: &str, value: Box<dyn Any>) -> Result<(), String>;
}
```

**Lines 46-47: Auto-implementation for JSON-serializable types**
```rust
impl<T: serde::Serialize + serde::de::DeserializeOwned> JsonPathAccessible for T {}
```

**Status:** `JsonPathAccessible` is automatically implemented for ALL settings structs.
**Manual `PathAccessible` implementations:** NONE found in codebase.

**Conclusion:** The macro at lines 69-119 is **never used**. All path access goes through JSON conversion.

---

## 12. Performance Implications

### Current Overhead Per Request:

1. **API Handler:** Parse path string
2. **Settings Service:** `normalize_key()` conversion (unnecessary)
3. **Database Service:**
   - Exact lookup (fails if client sent camelCase)
   - Fallback conversion (corrects settings_service mistake)
   - Second exact lookup (succeeds)
4. **PathAccess:** Serialize entire struct to JSON to navigate path
5. **Response:** Serialize result back to JSON

**Bottleneck:** Full struct serialization for single field access (documented in `path_access.rs:5-6`).

---

## 13. Validation Implementation

### File: `src/handlers/settings_validation_fix.rs`

#### **Lines 6-74: GPU Parameter Bounds**
```rust
pub fn validate_physics_settings_complete(physics: &Value) -> Result<(), String> {
    // CRITICAL GPU PARAMETER BOUNDS - Prevents NaN and explosions

    // dt (time step) - CRITICAL for stability
    if let Some(dt) = physics.get("dt").or_else(|| physics.get("timeStep")) {
        let val = dt.as_f64().ok_or("dt must be a number")?;
        if val <= 0.0 || val > 0.1 {
            return Err("dt must be between 0.001 and 0.1 for GPU stability".to_string());
        }
    }

    // maxVelocity - Prevent position explosions
    if let Some(max_vel) = physics.get("maxVelocity") {
        let val = max_vel.as_f64().ok_or("maxVelocity must be a number")?;
        if val <= 0.0 || val > 100.0 {
            return Err("maxVelocity must be between 0.1 and 100.0".to_string());
        }
    }
    ...
}
```

**Finding:** Validation uses **camelCase** field names directly from JSON.

---

## 14. API Endpoint Summary

### Batch Read (`/api/settings/batch` POST)
**File:** `settings_paths.rs:241-300`

**Request:**
```json
{
  "paths": ["visualisation.physics.damping", "visualisation.physics.gravity"]
}
```

**Response:**
```json
{
  "visualisation.physics.damping": 0.95,
  "visualisation.physics.gravity": 0.0001
}
```

**Case Handling:** Accepts paths in **either format**, returns keys **as-provided**.

### Batch Update (`/api/settings/batch` PUT)
**File:** `settings_paths.rs:311-462`

**Request:**
```json
{
  "updates": [
    { "path": "visualisation.physics.damping", "value": 0.98 },
    { "path": "visualisation.physics.gravity", "value": 0.001 }
  ]
}
```

**Validation:** Line 392 calls `validate_config_camel_case()` after all updates.

---

## 15. Key Findings Summary

| Question | Answer | Evidence |
|----------|--------|----------|
| **Storage format?** | **camelCase ONLY** (generic `settings` table) + **snake_case columns** (`physics_settings` table) | `settings_migration.rs:193`, `ontology_db.sql:42-73` |
| **Smart lookup exists?** | **YES** | `database_service.rs:120-146` |
| **Dual storage?** | **NO** | Migration stores single format only |
| **SettingsService normalization?** | **BROKEN** (converts wrong direction) | `settings_service.rs:287-299` |
| **API contract?** | Accepts **both formats** in paths, returns **camelCase** JSON | `settings_paths.rs:17-93` |
| **PathAccessible complete?** | **NO** — only `JsonPathAccessible` is used | `path_access.rs:46-47` |
| **Migration status?** | Check via `"app_full_settings"` key | `settings_migration.rs:369-375` |
| **Case converters count?** | **5 independent implementations** | See Section 7 |
| **Performance issue?** | Full JSON serialization per field access | `path_access.rs:5-6` comment |

---

## 16. Recommendations

### **Critical Issues:**

1. **Fix `settings_service.rs` normalization**
   - Line 287: Remove or reverse `normalize_key()` logic
   - Currently converts camelCase → snake_case (wrong direction)
   - Should pass keys through unchanged or convert snake_case → camelCase

2. **Consolidate case converters**
   - 5 independent implementations of same logic
   - Create single utility module: `src/utils/case_conversion.rs`
   - Use consistent algorithm across codebase

3. **Document actual storage format**
   - Update documentation to reflect camelCase-only storage
   - Remove references to "dual format" storage
   - Clarify that snake_case support is read-only fallback

### **Performance Optimizations:**

4. **Implement direct field access**
   - Complete `PathAccessible` trait implementation
   - Avoid full struct serialization for single field reads
   - Use macro at `path_access.rs:69-119`

5. **Optimize batch operations**
   - Batch reads currently serialize entire struct 50 times
   - Implement single serialization with multiple path extractions

### **Code Quality:**

6. **Add integration tests**
   - Test camelCase client requests
   - Test snake_case client requests (fallback)
   - Test mixed-case batch operations
   - Verify double conversion doesn't break edge cases

7. **Schema consistency**
   - Choose single format for SQL column names
   - Either rename `physics_settings` columns to camelCase
   - Or store generic settings with snake_case keys

---

## 17. Migration Format Details

**Physics Settings Migration (Lines 318-350):**
```rust
// YAML input (camelCase)
{
  "damping": 0.95,
  "maxVelocity": 1.0,
  "springK": 0.005,
  "repelK": 50.0
}

// Stored in physics_settings table (snake_case columns via struct mapping)
INSERT INTO physics_settings (
  damping,        -- Maps from Rust field PhysicsSettings.damping
  max_velocity,   -- Maps from Rust field PhysicsSettings.max_velocity
  spring_k,       -- Maps from Rust field PhysicsSettings.spring_k
  repel_k         -- Maps from Rust field PhysicsSettings.repel_k
) VALUES (0.95, 1.0, 0.005, 50.0);
```

**Generic Settings Migration (Lines 188-197):**
```rust
// YAML input (camelCase)
"visualisation.nodes.baseColor": "#ff0000"

// Stored in settings table (camelCase key preserved)
INSERT INTO settings (key, value_type, value_text)
VALUES ("visualisation.nodes.baseColor", "string", "#ff0000");
```

---

## Conclusion

The backend implements a **functional but inefficient** system:

✅ **Works:** Smart fallback allows clients to use either case format
❌ **Inefficient:** Double conversion (`camelCase → snake_case → camelCase`)
❌ **Inconsistent:** 5 different case converter implementations
❌ **Performance:** Full JSON serialization for single field access
❌ **Documentation:** Claims dual storage, actually single format + fallback

**Priority Fix:** Repair `settings_service.rs` normalization logic to eliminate double conversion.
