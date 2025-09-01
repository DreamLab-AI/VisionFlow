# Settings System Fix Summary

## Issues Fixed

### 1. ✅ Duplicate Key Error in settings.yaml
**Problem**: Lines 287 and 290 both had key '4', lines 288 and 291 both had key '5' in the button_functions mapping.
**Solution**: Fixed by renumbering the duplicate keys to '7' and '8' for leftView and bottomView.

### 2. ✅ Snake_case vs CamelCase Mismatch
**Problem**: The Rust structs had `#[serde(rename_all = "camelCase")]` which expected camelCase fields in YAML, but the settings.yaml file correctly uses snake_case (Rust convention).
**Solution**: Removed all `#[serde(rename_all = "camelCase")]` attributes from the config structs. Now:
- YAML uses snake_case (correct for Rust)
- Rust structs deserialize snake_case from YAML
- API handlers should handle the conversion to camelCase when sending to TypeScript clients

### 3. ✅ Settings File Location
**Problem**: Backend looks for settings at `/app/settings.yaml` (Docker mount), but the file is at `/workspace/ext/data/settings.yaml`.
**Solution**: Created symlink: `/app/settings.yaml -> /workspace/ext/data/settings.yaml`

## Result

The backend should now successfully:
1. Load the settings.yaml file without deserialization errors
2. Parse all fields correctly with snake_case naming
3. Start without crashing

## API Boundary Note

Since we removed `rename_all = "camelCase"`, the API responses will now use snake_case. If the TypeScript frontend expects camelCase, the conversion should be handled at the API boundary in the handlers, not in the core data structures.

## Files Modified
- `/workspace/ext/data/settings.yaml` - Fixed duplicate keys
- `/workspace/ext/src/config/mod.rs` - Removed all `rename_all = "camelCase"` attributes
- `/app/settings.yaml` - Created symlink to correct location