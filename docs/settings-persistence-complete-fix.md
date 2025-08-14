# Complete Settings Persistence Fix

## Issues Found and Fixed

### 1. ❌ Missing persist_settings Check
**Problem**: The `save()` function in `/workspace/ext/src/config/mod.rs` was NOT checking the `persist_settings` flag at all.

**Original Code**:
```rust
pub fn save(&self) -> Result<(), String> {
    let settings_path = std::env::var("SETTINGS_FILE_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/app/settings.yaml"));
    // ... always saves regardless of flag
}
```

**Fixed Code**:
```rust
pub fn save(&self) -> Result<(), String> {
    // Check if persist_settings is enabled
    if !self.system.persist_settings {
        debug!("Settings persistence is disabled (persist_settings: false), skipping save");
        return Ok(());
    }
    // ... rest of save logic
}
```

### 2. ❌ Wrong Default File Path
**Problem**: The default path was `/app/settings.yaml` instead of `data/settings.yaml`

**Fixed**: Changed both `new()` and `save()` functions to use:
```rust
.unwrap_or_else(|_| PathBuf::from("data/settings.yaml"));
```

## Complete Fix Applied

### Files Modified:
1. `/workspace/ext/data/settings.yaml` - Set `persist_settings: true`
2. `/workspace/ext/src/config/mod.rs` - Added persist_settings check and fixed path

### The Complete Flow Now:

1. **User changes settings** → UI sends to `/api/settings`
2. **Power user check** → Verifies against `POWER_USER_PUBKEYS`
3. **SettingsActor receives** → `UpdateSettings` message
4. **save() function**:
   - ✅ Checks `persist_settings` flag
   - ✅ If true, saves to `data/settings.yaml`
   - ✅ If false, skips save (returns Ok)

## How to Apply:

1. **Rebuild the application**:
```bash
cargo build --release
```

2. **Restart the container**:
```bash
./scripts/launch.sh restart
```

3. **Verify settings.yaml has persist_settings enabled**:
```bash
grep persist_settings data/settings.yaml
# Should output: persist_settings: true
```

## Testing:

1. Make a settings change as a power user
2. Check the logs for: `"Saving AppFullSettings to YAML file: "data/settings.yaml"`
3. Verify the file is updated: `ls -la data/settings.yaml`
4. Refresh the page - settings should persist

## Summary

The issue was two-fold:
1. The `persist_settings` flag was being completely ignored in the save function
2. The default path was incorrect (`/app/settings.yaml` vs `data/settings.yaml`)

Both issues are now fixed. The system will properly check the flag and save to the correct location.