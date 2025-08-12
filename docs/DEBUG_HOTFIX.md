# Debug Configuration Hotfix

## Issue
After removing the debug section from `settings.yaml`, the application failed to start with 502 errors because the Rust struct `SystemSettings` still requires a `debug` field.

## Solution
Restored a minimal debug section in `settings.yaml` with default values that serve as placeholders. The actual debug behavior is still controlled by environment variables:

- **`DEBUG_ENABLED`**: Controls actual debug features
- **`RUST_LOG`**: Controls actual logging levels

## Implementation

### 1. Settings.yaml
Added back minimal debug section with all values set to false/defaults:
```yaml
debug:
  enabled: false
  enable_data_debug: false
  # ... all other flags set to false
  log_level: info
  log_format: json
```

### 2. Config Struct
Added Default implementation for DebugSettings to provide safe defaults.

### 3. Runtime Behavior
The application now:
1. Loads the default debug config from settings.yaml (satisfies struct requirements)
2. Overrides with environment variables at runtime (actual behavior control)
3. Uses `DEBUG_ENABLED` and `RUST_LOG` as the source of truth

## Key Points
- The debug section in settings.yaml is now just for structural compatibility
- Actual debug control remains centralized in environment variables
- No functional regression - environment variables still control all debug behavior
- This is a compatibility layer to maintain struct expectations while using env vars

## Future Improvement
Consider making the debug field optional in SystemSettings:
```rust
pub struct SystemSettings {
    // ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<DebugSettings>,
}
```

This would allow complete removal of the debug section from settings.yaml.