# Case Conversion Fix Documentation

## Problem Summary

The VisionFlow server was failing to start due to a critical case conversion synchronization issue between the Rust backend and TypeScript client. The primary issue was a field name mismatch in the settings structure.

## Root Cause Analysis

### 1. Missing Field Error
The server was failing with the error:
```
Failed to deserialize AppFullSettings from "/app/settings.yaml": missing field `glow`
```

**Issue**: The Rust `VisualisationSettings` struct defined a field called `glow: GlowSettings`, but the settings.yaml file contained a section named `bloom` instead.

### 2. Case Conversion Logic
The server uses bidirectional case conversion:
- **Client → Server**: `camelCase` → `snake_case`
- **Server → Client**: `snake_case` → `camelCase`

This conversion happens in:
- **Rust backend**: `keys_to_snake_case()` and `keys_to_camel_case()` functions
- **TypeScript client**: `convertSnakeToCamelCase()` and `convertCamelToSnakeCase()` functions

## Implemented Fixes

### 1. Field Name Alignment
**File**: `/workspace/ext/src/config/mod.rs`

Fixed the struct field mapping:
```rust
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct VisualisationSettings {
    pub rendering: RenderingSettings,
    pub animations: AnimationSettings,
    #[serde(rename = "bloom")]  // ← Added this annotation
    pub glow: GlowSettings,     // ← Field name remains 'glow' internally
    pub hologram: HologramSettings,
    // ...
}
```

This allows the Rust struct to maintain the `glow` field name internally while correctly mapping to the `bloom` field in the YAML file.

### 2. Enhanced Case Conversion Functions

**Rust Backend** (`/workspace/ext/src/config/mod.rs`):

#### `keys_to_camel_case()` improvements:
```rust
fn keys_to_camel_case(value: Value) -> Value {
    // Improved handling of empty parts and proper capitalization
    let camel_key = k.split('_').enumerate().map(|(i, part)| {
        if i == 0 {
            part.to_string()
        } else {
            // Handle empty parts and properly capitalize
            if part.is_empty() {
                String::new()
            } else {
                let mut chars = part.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            }
        }
    }).collect::<String>();
}
```

#### `keys_to_snake_case()` improvements:
```rust
fn keys_to_snake_case(value: Value) -> Value {
    // Improved camelCase to snake_case conversion
    let snake_key = k.chars().fold(String::new(), |mut acc, c| {
        if c.is_ascii_uppercase() {
            // Add underscore before uppercase letters, but not at the start
            if !acc.is_empty() {
                acc.push('_');
            }
            acc.push(c.to_ascii_lowercase());
        } else {
            acc.push(c);
        }
        acc
    });
}
```

**TypeScript Client** (`/workspace/ext/client/src/utils/caseConversion.ts`):

#### `camelToSnake()` improvements:
```typescript
export function camelToSnake(str: string): string {
  return str.replace(/([A-Z])/g, (match, letter) => `_${letter.toLowerCase()}`)
    .replace(/^_/, ''); // Remove leading underscore if present
}
```

### 3. Comprehensive Testing

**File**: `/workspace/ext/src/config/mod.rs` (test module)

Added comprehensive unit tests to verify:
- Snake case to camel case conversion
- Camel case to snake case conversion
- Round-trip conversion (camelCase → snake_case → camelCase)
- Nested object handling
- Array handling
- Edge cases (empty keys, consecutive capitals, etc.)

## Workflow Overview

### Settings Loading (Server Startup)
1. Server loads `settings.yaml` with `snake_case` field names
2. Serde deserializes with proper field mapping (`bloom` → `glow`)
3. Settings are available internally with consistent Rust naming

### Settings API (Client ↔ Server)
1. **Client sends update** (camelCase):
   ```json
   {
     "visualisation": {
       "graphs": {
         "logseq": {
           "physics": {
             "autoBalance": true,
             "springK": 0.1,
             "repelK": 2.0
           }
         }
       }
     }
   }
   ```

2. **Server converts to snake_case**:
   ```json
   {
     "visualisation": {
       "graphs": {
         "logseq": {
           "physics": {
             "auto_balance": true,
             "spring_k": 0.1,
             "repel_k": 2.0
           }
         }
       }
     }
   }
   ```

3. **Server responds with camelCase**:
   - Calls `app_settings.to_camel_case_json()`
   - Returns properly converted camelCase JSON to client

### Error Handling
Enhanced error messages for debugging:
```rust
let camel_case_json = app_settings.to_camel_case_json()
    .map_err(|e| {
        error!("Failed to convert settings to camelCase JSON: {}", e);
        error!("This usually indicates a case conversion or serialization issue");
        actix_web::error::ErrorInternalServerError("Settings serialization error - check server logs")
    })?;
```

## Testing the Fix

### Unit Tests
Run the case conversion tests:
```bash
cargo test case_conversion
```

### Integration Testing
1. Start the server - should no longer fail with "missing field `glow`"
2. Check settings endpoint: `GET /api/settings`
3. Test settings update: `POST /api/settings` with camelCase payload
4. Verify round-trip conversion works properly

## Key Principles

1. **Server is source of truth**: Settings.yaml defines the canonical structure
2. **Bidirectional conversion**: Client ↔ Server communication uses automatic case conversion
3. **Internal consistency**: Rust structs use snake_case, TypeScript uses camelCase
4. **Graceful handling**: Edge cases in conversion don't crash the system
5. **Clear error messages**: Debugging information helps identify issues quickly

## Future Maintenance

When adding new settings fields:

1. **Add to YAML** with snake_case name
2. **Add to Rust struct** with snake_case name  
3. **Add to TypeScript interface** with camelCase name
4. **Test conversion** with unit tests
5. **Verify in browser** that settings sync properly

The case conversion system should handle new fields automatically without additional configuration.