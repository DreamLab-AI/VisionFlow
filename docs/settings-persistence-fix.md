# Settings Persistence Fix for Power Users

## Problem
Power user settings changes weren't persisting to `settings.yaml` despite proper authentication. Changes would revert on page refresh.

## Root Cause
The `persist_settings` flag in `data/settings.yaml` was set to `false` (line 222), preventing any file writes regardless of user permissions.

## Solution Applied
Changed `persist_settings` from `false` to `true` in `/workspace/ext/data/settings.yaml`:

```yaml
system:
  # General system settings
  persist_settings: true  # Changed from false
```

## How It Works

### Settings Persistence Flow:
1. **UI Change**: User modifies settings in `PhysicsEngineControls.tsx`
2. **API Call**: Client sends POST to `/api/settings` with authentication headers
3. **Power User Check**: Server verifies user pubkey against `POWER_USER_PUBKEYS` env variable
4. **Settings Actor**: `UpdateSettings` message sent to `SettingsActor`
5. **File Write**: If `persist_settings: true`, actor writes to `data/settings.yaml`
6. **Propagation**: Physics settings propagated to GPU compute actor

### Key Components:
- **Flag Location**: `/workspace/ext/data/settings.yaml` line 222
- **Actor Logic**: `/workspace/ext/src/actors/settings_actor.rs`
- **Handler**: `/workspace/ext/src/handlers/settings_handler.rs`
- **Power User Check**: `/workspace/ext/src/config/feature_access.rs`

## Verification Steps

1. **Restart the container** for the change to take effect:
   ```bash
   ./scripts/launch.sh restart
   ```

2. **Verify flag is set**:
   ```bash
   grep persist_settings data/settings.yaml
   # Should output: persist_settings: true
   ```

3. **Check Docker logs** while making a settings change:
   ```bash
   docker logs visionflow_container | grep "Settings saved"
   ```

4. **Verify power user environment variable**:
   ```bash
   docker exec visionflow_container env | grep POWER_USER_PUBKEYS
   ```

## Additional Considerations

### File Permissions (if still not working):
If settings still don't persist after enabling the flag, check file permissions:
```bash
# Check current permissions
ls -l data/settings.yaml

# Grant write permissions (safe for local development)
chmod 666 data/settings.yaml
```

### Power User Authentication:
Ensure your Nostr public key is in the `.env` file:
```env
POWER_USER_PUBKEYS=your_nostr_pubkey_here
```

## Summary
The issue was simply that the `persist_settings` flag was disabled, which is a safe default for production environments but needs to be enabled for development with power user features.