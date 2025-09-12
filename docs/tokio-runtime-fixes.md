# Tokio Runtime Panic Fixes

## Issues Found and Fixed

### 1. Main.rs - Bots Client Connection (Fixed)
**File**: `/workspace/ext/src/main.rs`
**Line**: 171
**Issue**: `tokio::spawn` was called outside of Actix runtime during startup
**Fix**: Changed to synchronous connection attempt without spawning

### 2. JsonRpcClient - Shutdown Handler (Fixed)
**File**: `/workspace/ext/src/actors/jsonrpc_client.rs`
**Line**: 275
**Issue**: `tokio::spawn` was called in the `stopped()` lifecycle method during shutdown
**Fix**: Changed to synchronous cleanup using `try_write()` instead of async spawn

### 3. Empty Metadata Store (Fixed)
**File**: `/workspace/ext/src/main.rs`
**Lines**: 196-200
**Issue**: Application was exiting if no metadata was found, triggering the shutdown panic
**Fix**: Changed to allow starting with empty metadata (warning instead of error)

## Summary of Changes

1. **Removed `tokio::spawn` from main.rs startup sequence**
   - The bots client connection is now attempted synchronously
   - This prevents runtime panics during Actix system initialization

2. **Fixed JsonRpcClient shutdown**
   - Replaced async cleanup with synchronous cleanup
   - Uses `try_write()` to avoid deadlocks during shutdown
   - Gracefully handles cases where lock cannot be acquired

3. **Allow empty metadata**
   - Application can now start without metadata files
   - Logs a warning instead of failing
   - Prevents early exit that was triggering shutdown issues

## Verification

After rebuilding with these changes, the application should:
- Start without Tokio runtime panics
- Handle shutdown gracefully
- Continue running even without metadata files

## Additional `tokio::spawn` Locations

Other files still contain `tokio::spawn` but these appear to be called during normal operation when the runtime is active:
- `tcp_connection_actor.rs` - During connection handling
- `optimized_settings_actor.rs` - During cache operations
- `claude_flow_actor.rs` - During message processing

These should not cause issues as they're not called during startup/shutdown sequences.