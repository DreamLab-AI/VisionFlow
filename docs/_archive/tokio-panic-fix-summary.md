# VisionFlow Tokio Runtime Panic Fix Summary

## Problem Description

The VisionFlow backend is experiencing continuous crashes (SIGABRT) during startup due to a Tokio runtime panic. The error message is:

```
thread 'main' panicked at /root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/actix-0.13.5/src/utils.rs:114:22:
there is no reactor running, must be called from the context of a Tokio 1.x runtime
```

## Root Cause

The issue occurs in `/app/src/app_state.rs` during actor system initialization. The code is using `tokio::spawn()` outside of a proper async runtime context, specifically during the initialization of the GPU compute actors within the Actix actor system.

## System Architecture

- VisionFlow runs in a Docker container
- The container builds from source code in `/app` directory (inside container)
- The workspace at `/workspace/ext` is mounted but not used for builds
- The container uses supervisord to manage nginx, rust-backend, and vite-dev processes
- A wrapper script at `/workspace/ext/scripts/rust-backend-wrapper.sh` triggers rebuilds

## Fix Applied

The fix involves:

1. **Removing direct `tokio::spawn` calls** from `app_state.rs`
2. **Deferring GPU initialization** using actor messages
3. **Adding a new message type** `InitializeGPUConnection` to handle deferred initialization
4. **Using actor context** (`ctx.spawn()`) for async operations within actors

### Key Changes:

1. In `app_state.rs`:
   - Removed `tokio::spawn()` calls
   - Send `InitializeGPUConnection` message to GraphServiceActor instead

2. In `messages.rs`:
   - Added `InitializeGPUConnection` message type

3. In `graph_actor.rs`:
   - Implemented handler for `InitializeGPUConnection`
   - Uses `ctx.spawn()` for async operations within actor context

## Current Status

- Fix has been applied to `/workspace/ext/src/` (workspace source)
- Container continues to crash because it builds from `/app/src/` (container source)
- A patch file has been created at `/workspace/ext/patches/fix-tokio-panic.patch`

## To Apply Fix

The fix needs to be applied to the container's source code. Options:

1. **Rebuild Docker image** with the fixed source code
2. **Apply patch** to `/app/src/` inside the running container
3. **Modify build process** to use `/workspace/ext/src/` instead of `/app/src/`

## Verification

After applying the fix, the backend should:
- Start without SIGABRT crashes
- Successfully load the 177-node knowledge graph
- Serve the graph data via HTTP API at `/api/graph/data`
- Accept WebSocket connections at `/wss`