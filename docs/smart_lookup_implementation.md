# Smart Database Lookup Implementation

## Overview

This document describes the intelligent camelCase/snake_case fallback lookup implementation for `/home/devuser/workspace/project/src/services/database_service.rs`.

## Problem Solved

Previously, the system required dual storage of settings in both snake_case and camelCase formats:
- Frontend sends: `springK` (camelCase)
- Rust code expects: `spring_k` (snake_case)
- Result: Database had both entries with duplicate data

## Solution

Implemented smart lookup that automatically converts snake_case to camelCase when exact match fails:

1. **Try exact match first** - Always prioritize the exact key provided
2. **Fallback to camelCase** - If key contains underscore and not found, convert to camelCase and retry
3. **Return None** - If both attempts fail, return None (no error)

## Implementation Details

### Helper Function: `to_camel_case`

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

**Conversion Examples:**
- `spring_k` → `springK`
- `max_velocity` → `maxVelocity`
- `repulsion_cutoff` → `repulsionCutoff`
- `center_gravity_k` → `centerGravityK`
- `springK` → `springK` (no change)
- `damping` → `damping` (no change)

### New Function: `get_setting_exact`

Extracted the exact match logic from original `get_setting`:

```rust
/// Get setting value with exact key match (no fallback)
fn get_setting_exact(&self, key: &str) -> SqliteResult<Option<SettingValue>> {
    let conn = self.conn.lock().unwrap();
    let mut stmt = conn.prepare(
        "SELECT value_type, value_text, value_integer, value_float, value_boolean, value_json
         FROM settings WHERE key = ?1"
    )?;

    stmt.query_row(params![key], |row| {
        let value_type: String = row.get(0)?;
        let value = match value_type.as_str() {
            "string" => SettingValue::String(row.get(1)?),
            "integer" => SettingValue::Integer(row.get(2)?),
            "float" => SettingValue::Float(row.get(3)?),
            "boolean" => SettingValue::Boolean(row.get::<_, i32>(4)? == 1),
            "json" => {
                let json_str: String = row.get(5)?;
                SettingValue::Json(serde_json::from_str(&json_str).unwrap_or(JsonValue::Null))
            },
            _ => SettingValue::String(String::new()),
        };
        Ok(value)
    }).optional()
}
```

### Modified Function: `get_setting`

Now implements the smart fallback logic:

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

## Usage Examples

### Example 1: Direct Match
```rust
// Database has: "springK" = 150.0
let value = db.get_setting("springK")?;  // ✓ Returns Some(150.0)
```

### Example 2: Fallback Conversion
```rust
// Database has: "springK" = 150.0
let value = db.get_setting("spring_k")?;  // ✓ Converts to "springK", returns Some(150.0)
```

### Example 3: Non-existent Key
```rust
let value = db.get_setting("nonexistent_key")?;  // ✓ Returns None
```

### Example 4: Exact Match Priority
```rust
// Database has both:
//   "spring_k" = 100.0
//   "springK" = 150.0

let val1 = db.get_setting("spring_k")?;  // ✓ Returns Some(100.0) - exact match
let val2 = db.get_setting("springK")?;   // ✓ Returns Some(150.0) - exact match
```

### Example 5: Physics Settings Migration
```rust
// Old code (requires dual storage):
db.set_setting("spring_k", SettingValue::Float(150.0), None)?;
db.set_setting("springK", SettingValue::Float(150.0), None)?;  // Duplicate!

// New code (single storage in camelCase):
db.set_setting("springK", SettingValue::Float(150.0), None)?;

// Both access patterns work:
let val1 = db.get_setting("springK")?;   // ✓ Direct hit
let val2 = db.get_setting("spring_k")?;  // ✓ Fallback conversion
```

## Benefits

1. **Eliminates Dual Storage**: Store settings once in preferred format (camelCase)
2. **Backward Compatible**: Existing code using snake_case continues to work
3. **No Breaking Changes**: All existing queries work exactly as before
4. **Performance**: Only one additional query if first attempt fails
5. **Simple Migration**: Just stop writing duplicate entries

## Test Coverage

Five comprehensive test cases added:

1. **`test_to_camel_case`** - Verifies conversion logic for various patterns
2. **`test_camel_snake_fallback_lookup`** - Tests basic fallback functionality
3. **`test_multiple_physics_settings_fallback`** - Tests multiple physics parameters
4. **`test_exact_match_priority`** - Ensures exact matches always win
5. **`test_comprehensive_physics_settings`** - Real-world physics settings scenario

## Migration Path

### Phase 1: Current (Dual Storage)
```rust
// Settings stored as both:
db.set_setting("spring_k", val, None)?;
db.set_setting("springK", val, None)?;
```

### Phase 2: Transition (Smart Lookup Enabled)
```rust
// Deploy smart lookup code
// Both formats still work, but only store once:
db.set_setting("springK", val, None)?;  // Preferred format

// Old code still works:
let val = db.get_setting("spring_k")?;  // ✓ Fallback conversion
```

### Phase 3: Cleanup (Optional)
```rust
// Remove duplicate snake_case entries from database
// (Optional - fallback ensures backward compatibility)
```

## Performance Considerations

- **Best Case**: Exact match on first try - single query
- **Fallback Case**: Two queries (exact + camelCase conversion)
- **Worst Case**: Still just two queries, returns None
- **No Performance Impact**: For exact matches (most common case)

## File Locations

- **Implementation**: `/home/devuser/workspace/project/src/services/database_service.rs`
- **Lines**: 74-146 (added helper and modified methods)
- **Tests**: Lines 736-848 (comprehensive test suite)

## Related Issues

This implementation addresses the need to eliminate duplicate storage of physics settings that come from the frontend in camelCase but are accessed in Rust code using snake_case conventions.
