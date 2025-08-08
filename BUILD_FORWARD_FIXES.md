# Build Forward Fixes - Unified Settings Compilation

## Decision: Build Forward
Rather than restoring deleted files, we chose to fix the compilation errors by properly implementing the unified settings architecture.

## Fixes Applied

### 1. Graph Path Access
**Error**: `no field 'graphs' on type AppFullSettings`
**Fix**: Changed `settings.graphs.logseq` to `settings.visualisation.graphs.logseq`

### 2. Type Conversions
**Error**: Type mismatches between `UnifiedSettings` and `AppFullSettings`
**Fixes**:
- Added `.into()` conversions where needed
- Used existing conversion implementations from `config/migration.rs`
- Convert flow: `UnifiedSettings` ↔ `AppFullSettings` → `UISettings`

### 3. User Settings
**Error**: `UserSettings::new()` expected `UISettings` not `UnifiedSettings`
**Fix**: 
```rust
let app_settings: AppFullSettings = UnifiedSettings::default().into();
let ui_settings: UISettings = (&app_settings).into();
UserSettings::new(&pubkey, ui_settings)
```

### 4. Settings Actor
**Error**: Mismatched types in actor handler
**Fix**: Added `.into()` conversion in `UpdateSettings<AppFullSettings>` handler

## Architecture Benefits

1. **Clean Separation**: 
   - `UnifiedSettings` - Internal unified structure
   - `AppFullSettings` - Legacy compatibility layer
   - `UISettings` - Client-facing structure

2. **Type Safety**: All conversions are explicit and type-checked

3. **Migration Path**: Existing code continues to work through conversion layer

## Next Steps

1. Build should now succeed
2. Monitor for runtime issues
3. Eventually phase out `AppFullSettings` in favor of pure `UnifiedSettings`

## Files Modified
- `src/handlers/unified_settings_handler.rs` - Fixed all type conversions
- `src/actors/unified_settings_actor.rs` - Added conversion in handler
- Removed `src/config/minimal.rs` - No longer needed
- Updated module declarations to remove old references