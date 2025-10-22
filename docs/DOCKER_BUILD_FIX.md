# Docker Build Fix - whelk-rs Dependency

## Problem

The Docker build was failing with:
```
error: failed to get `whelk` as a dependency of package `webxr v0.1.0 (/app)`
Caused by:
  failed to load source for dependency `whelk`
Caused by:
  Unable to update /app/whelk-rs
Caused by:
  failed to read `/app/whelk-rs/Cargo.toml`
```

## Root Cause

In `Cargo.toml`, whelk is configured as a local path dependency:
```toml
whelk = { path = "./whelk-rs", optional = true }
```

However, the `Dockerfile.dev` was running `cargo fetch` **before** copying the `whelk-rs` directory into the container, causing the fetch to fail.

## Solution

Modified `Dockerfile.dev` to copy `whelk-rs` before running `cargo fetch`:

```dockerfile
# Copy whelk-rs local dependency (required for cargo fetch)
COPY whelk-rs ./whelk-rs
```

This line was added at line 71, before the `cargo fetch` command at line 85.

## Files Modified

- `Dockerfile.dev` - Added `COPY whelk-rs ./whelk-rs` instruction

## Verification

Build with:
```bash
docker build -f Dockerfile.dev -t webxr:dev .
```

The build should now succeed through the `cargo fetch` stage.

---
**Date**: 2025-10-22
**Status**: ✅ FIXED
