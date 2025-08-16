# ✅ Fixed Validation Ranges for Unified Settings

## Issue Resolved
The 400 Bad Request error: **"boundsSize must be between 100.0 and 50000.0"** was caused by mismatched validation ranges between client UI and server validation.

## Key Fixes Applied

### 1. boundsSize Range Fixed
- **Old Server Range:** 100.0 - 50000.0 (too restrictive)
- **Client UI Range:** 1 - 50 (IntegratedControlPanel)
- **New Server Range:** 1.0 - 50000.0 (now accepts client values)

### 2. massScale Range Fixed  
- **Old Validation:** `val <= 0.0` (rejected 0.0 despite error saying "between 0.0 and 10.0")
- **New Validation:** `val < 0.1` (matches UI minimum of 0.1)

### 3. iterations Type Handling
- Now accepts floats and rounds them (JavaScript sends 100.0)

## Complete Validation Ranges (After Fix)

### Physics Settings

| Field | Client UI Range | Server Validation | Status |
|-------|----------------|-------------------|---------|
| `iterations` | 10-500 | 1-1000 (accepts float, rounds) | ✅ Compatible |
| `damping` | 0-1 | 0.0-1.0 | ✅ Compatible |
| `repulsionStrength` | 0-2 | 0.01-1000.0 | ✅ Compatible |
| `attractionStrength` | 0-1 | 0.0-10.0 | ✅ Compatible |
| `springStrength` | 0-1 | 0.0-10.0 | ✅ Compatible |
| `maxVelocity` | 0.001-0.5 | 0.0-1000.0 | ✅ Compatible |
| `timeStep` | N/A | 0.0-1.0 | ✅ Compatible |
| `temperature` | N/A | 0.0-10.0 | ✅ Compatible |
| `gravity` | N/A | -10.0-10.0 | ✅ Compatible |
| **`boundsSize`** | **1-50** | **1.0-50000.0** | **✅ FIXED** |
| `collisionRadius` | 0.1-5 | 0.0-100.0 | ✅ Compatible |
| `repulsionDistance` | 0.1-10 | N/A | ✅ No validation |
| **`massScale`** | **0.1-10** | **0.1-10.0** | **✅ FIXED** |
| `boundaryDamping` | 0-1 | 0.0-1.0 | ✅ Compatible |
| `updateThreshold` | 0-0.5 | 0.0-1.0 | ✅ Compatible |

### Node Settings

| Field | Client UI Range | Server Validation | Status |
|-------|----------------|-------------------|---------|
| `nodeSize` | 0.2-2 | 0.0-10.0 | ✅ Compatible |
| `opacity` | 0-1 | 0.0-1.0 | ✅ Compatible |
| `metalness` | 0-1 | 0.0-1.0 | ✅ Compatible |
| `roughness` | 0-1 | 0.0-1.0 | ✅ Compatible |

### XR Settings

| Field | Client UI Range | Server Validation | Status |
|-------|----------------|-------------------|---------|
| `renderScale` | 0.5-2 | 0.5-2.0 | ✅ Compatible |
| `roomScale` | N/A | 0.0-10.0 | ✅ Compatible |

### System Settings

| Field | Client Values | Server Validation | Status |
|-------|--------------|-------------------|---------|
| `debug.logLevel` | 0-3 | 0-3 (normalized by client) | ✅ Compatible |
| `debug.enabled` | true/false | boolean | ✅ Compatible |

## Testing After Deployment

1. **Test boundsSize slider:**
   - Move slider to minimum (1) - should save successfully
   - Move slider to maximum (50) - should save successfully

2. **Test massScale slider:**
   - Set to 0.1 - should save successfully
   - Set to 10 - should save successfully

3. **Test iterations slider:**
   - Any value should work (client normalizes float to int)

## Code Changes Summary

### `/workspace/ext/src/handlers/settings_handler.rs`
```rust
// boundsSize - now accepts smaller values from UI
if val < 1.0 || val > 50000.0 {  // Was: val < 100.0
    return Err("boundsSize must be between 1.0 and 50000.0".to_string());
}

// massScale - fixed minimum to match UI
if val < 0.1 || val > 10.0 {  // Was: val <= 0.0
    return Err("massScale must be between 0.1 and 10.0".to_string());
}

// iterations - accepts floats
let val = iterations.as_f64()
    .map(|f| f.round() as u64)  // Round float to integer
    .or_else(|| iterations.as_u64())
    .ok_or("iterations must be a positive number")?;
```

### `/workspace/ext/client/src/store/settingsStore.ts`
```javascript
// Normalizes iterations to integer before sending
if (physics && physics.iterations !== undefined) {
    physics.iterations = Math.round(physics.iterations);
}
```

## Deployment

After deploying these changes:
1. Rebuild server: `cargo build --release`
2. Restart server to load new validation
3. Clear browser cache if needed
4. Test all sliders in IntegratedControlPanel

## Expected Result

✅ No more 400 Bad Request errors for `boundsSize`
✅ All physics sliders save successfully
✅ Settings persist correctly with proper validation