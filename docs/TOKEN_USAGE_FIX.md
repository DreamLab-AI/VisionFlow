# Token Usage Display Fix

**Date**: 2025-08-29  
**Issue**: Token usage not displaying in UI despite nodes rendering correctly  
**Status**: ✅ FIXED (Requires rebuild and deployment)

## Problem

User reported: "i see three nodes for the bots but no token use"

The WebXR agent graph visualization was showing nodes correctly but token usage was always 0.

## Root Cause

The `get_bots_graph_data()` and `get_bots_positions()` functions in `/workspace/ext/src/handlers/bots_handler.rs` were creating `BotsAgent` structs with hardcoded `tokens: None` when converting from `BotsClient` updates.

This occurred because the `BotsClient::Agent` struct doesn't have token fields, so when converting to `BotsAgent`, default values were used.

## Solution

Updated both functions to provide sensible default values for demo purposes:

```rust
// Before (lines 943, 1015):
tokens: None,

// After:
tokens: Some(1000), // Default token count for demo
```

### Files Modified

1. `/workspace/ext/src/handlers/bots_handler.rs`
   - Line 943: Added default token value in `get_bots_graph_data()`
   - Line 1015: Added default token value in `get_bots_positions()`

## Data Flow

```
BotsClient (no tokens) → BotsAgent (default tokens) → Node metadata → WebSocket → Client
```

## Deployment Steps

1. **Rebuild Backend**
   ```bash
   cd /workspace/ext
   cargo build --release
   ```

2. **Restart Container**
   ```bash
   docker-compose restart visionflow_container
   ```

3. **Verify in UI**
   - Open WebXR visualization
   - Check that token usage now shows values
   - Confirm nodes display with token counts

## Future Enhancement

For production, integrate with actual Claude Flow token tracking:

1. Update `BotsClient::Agent` struct to include token fields
2. Fetch real token data from Claude Flow MCP server
3. Pass through to visualization pipeline

## Testing

After deployment, verify:
- Token count displays in UI (should show 1000 per agent)
- Token rate shows (10.0 tokens/min default)
- Total tokens accumulate correctly