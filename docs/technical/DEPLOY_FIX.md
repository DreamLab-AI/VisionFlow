# Settings System Fix - Deployment Instructions

*[Technical](../index.md)*

## Problem Identified
The Docker container was serving a **stale client bundle** that was built BEFORE the settings API fixes were applied. This caused 404 errors on `/api/settings/batch` and `/api/settings/path` endpoints.

## What Has Been Fixed

### ✅ Server-Side (Already Fixed)
- **SettingsActor** now uses `JsonPathAccessible` trait (respects serde rename_all)
- **Batch read endpoint** returns simple key-value map format
- **Routes properly registered** at `/api/settings/path` and `/api/settings/batch`

### ✅ Client-Side (Fixed and Rebuilt)
- **settingsApi.ts** uses correct endpoint paths
- **Client bundle rebuilt** at `/workspace/ext/client/dist`

## Deployment Steps

### Option 1: Rebuild Docker Image (Recommended)
```bash
cd /workspace/ext
docker build -f Dockerfile.production -t visionflow:latest .
docker run -p 4000:4000 visionflow:latest
```

### Option 2: Copy New Client Bundle to Running Container
```bash
# Find your container ID
docker ps

# Copy the rebuilt client bundle
docker cp /workspace/ext/client/dist CONTAINER_ID:/app/client/

# Restart nginx in the container (if needed)
docker exec CONTAINER_ID nginx -s reload
```

### Option 3: Mount Volume for Development
```bash
# Run container with client directory mounted
docker run -v /workspace/ext/client/dist:/app/client/dist -p 4000:4000 visionflow:latest
```

## Verification Steps

After deployment, verify the fixes work:

1. Open browser DevTools Network tab
2. Navigate to the application
3. Adjust physics settings sliders
4. Confirm NO 404 errors on:
   - `/api/settings/path` 
   - `/api/settings/batch`
5. Verify settings actually update (physics should respond)

## Technical Summary

The root cause was a **serialization mismatch**:
- Client sent `springK` (camelCase)
- SettingsActor incorrectly converted to `spring_k` (snake_case)
- PathAccessible expected `springK` (camelCase)
- Result: Silent failures, no updates applied

The fix ensures Serde's `#[serde(rename_all = "camelCase")]` is the single source of truth for all conversions.

## Files Modified
- `/workspace/ext/src/actors/settings_actor.rs`
- `/workspace/ext/src/handlers/settings_paths.rs`
- `/workspace/ext/client/src/api/settingsApi.ts`
- `/workspace/ext/client/dist/*` (rebuilt bundle)

## Status
✅ All fixes implemented and client bundle rebuilt
⏳ Awaiting Docker container restart/rebuild