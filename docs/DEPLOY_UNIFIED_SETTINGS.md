# 🚀 Deploying Unified Settings System

## Current Status
The unified settings validation has been implemented in the code but the **server needs to be restarted** to load the new validation logic.

## What Changed

### Server-Side Changes (`/workspace/ext/src/handlers/settings_handler.rs`)
1. **Removed multiple format acceptance** - Server now ONLY accepts the unified camelCase format
2. **Strict validation** - No more `.or_else()` chains accepting variations
3. **Better error logging** - Clear messages about what field failed validation

### Client-Side Changes  
1. **Added detailed logging** to show exactly what's being sent to the server
2. **Enhanced error reporting** to show server response details

## Required Actions

### 1. Rebuild and Restart the Server

```bash
# In the server Docker container or host:
cd /workspace/ext

# Rebuild the Rust server
cargo build --release

# Restart the server (method depends on your setup)
# Option A: If using systemd
sudo systemctl restart webxr

# Option B: If running directly
# Kill the current process and restart
pkill webxr
./target/release/webxr

# Option C: If using Docker Compose
docker-compose restart webxr-server
```

### 2. Verify the Changes

After restarting, the browser console should show:

```javascript
// New debug logging will show:
[SETTINGS DEBUG] Sending settings payload to server: {
  endpoint: '/api/settings',
  payloadKeys: [...],
  sampleFields: {
    'xr.enabled': true,  // Should have value
    'xr.enableXrMode': undefined,  // Should be undefined (old format)
    'system.debug.enabled': true,  // Should have value
    'system.debug.enableClientDebugMode': undefined  // Should be undefined
  }
}
```

### 3. Test the Unified Format

Try toggling settings in the IntegratedControlPanel. You should see:
- ✅ Successful saves (200 OK)
- ✅ No more 400 Bad Request errors
- ✅ Toast notifications: "Settings Saved"

## Field Mapping Reference

| Component | Old Format | Unified Format |
|-----------|------------|----------------|
| XR Mode | `xr.enableXrMode` | `xr.enabled` |
| Debug Mode | `system.debug.enableClientDebugMode` | `system.debug.enabled` |
| Physics Repulsion | `repulsion` | `repulsionStrength` |
| Physics Attraction | `attraction` | `attractionStrength` |
| Physics Spring | `spring` | `springStrength` |
| Node Color | `base_color` | `baseColor` |
| Node Size | `node_size` | `nodeSize` |
| Ambient Light | `ambient_light_intensity` | `ambientLightIntensity` |

## Validation Rules (Server Enforces)

The server now STRICTLY validates:
- ✅ Accepts: `xr.enabled` 
- ❌ Rejects: `xr.enableXrMode`
- ✅ Accepts: `system.debug.enabled`
- ❌ Rejects: `system.debug.enableClientDebugMode`
- ✅ Accepts: `repulsionStrength`, `attractionStrength`, `springStrength`
- ❌ Rejects: `repulsion`, `attraction`, `spring`

## Troubleshooting

### If 400 Errors Continue After Restart:

1. **Check server logs** for validation errors:
```bash
tail -f /workspace/ext/logs/server.log | grep -i validation
```

2. **Check browser console** for the debug payload:
- Look for `[SETTINGS DEBUG]` messages
- Verify `sampleFields` shows correct unified format
- Old format fields should be `undefined`

3. **Clear browser cache** and reload:
```javascript
// In browser console:
localStorage.clear();
location.reload();
```

4. **Verify server is running new code**:
```bash
# Check binary modification time
ls -la /workspace/ext/target/release/webxr

# Should be newer than your code changes
```

## Expected Result

Once the server is restarted with the new validation code:
1. Settings will save successfully using the unified format
2. No more 400 Bad Request errors
3. Physics controls will update the simulation properly
4. All UI components will work seamlessly

## Summary

The unified settings system is **fully implemented in code** but requires a **server restart** to activate the new strict validation that enforces "ONE format, ONE truth".