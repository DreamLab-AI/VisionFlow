# Physics Settings Update Issue - Analysis and Fix

## Problem Summary
The physics controls in the UI were updating the settings.yaml file but not affecting the running force-directed graph simulation. The server had to be restarted for changes to take effect.

## Root Cause Analysis

### 1. Missing Physics Propagation in Settings Handlers
**File:** `/src/handlers/settings_handler.rs`

The `update_setting_by_path` and `batch_update_settings` functions were updating the settings.yaml file but **NOT** calling `propagate_physics_to_gpu` to notify the GraphServiceActor.

### 2. Target Parameters Not Updated
**File:** `/src/actors/graph_actor.rs`

The `UpdateSimulationParams` message handler was only updating `simulation_params` but not `target_params`. This caused the `smooth_transition_params` function to revert values back to the old target parameters.

## The Complete Flow (How It Should Work)

```
Client UI Control Change
    ↓
HTTP POST /api/settings/batch
    ↓
settings_handler::batch_update_settings()
    ↓
1. Update settings.yaml
2. Call propagate_physics_to_gpu() ← WAS MISSING
    ↓
Send UpdateSimulationParams to GraphServiceActor
    ↓
GraphServiceActor updates:
  - simulation_params
  - target_params ← WAS MISSING
    ↓
GPU physics simulation uses new parameters
```

## Applied Fixes

### Fix 1: Add Physics Propagation to Settings Handlers
```rust
// In update_setting_by_path (line 1677-1693)
if path.contains(".physics.") || path.contains(".graphs.logseq.") || path.contains(".graphs.visionflow.") {
    info!("Physics setting changed, propagating to GPU actors");
    let graph_name = if path.contains(".graphs.logseq.") {
        "logseq"
    } else if path.contains(".graphs.visionflow.") {
        "visionflow"
    } else {
        "logseq"
    };
    propagate_physics_to_gpu(&state, &app_settings, graph_name).await;
}

// Similar code added to batch_update_settings (line 1874-1890)
```

### Fix 2: Update Target Parameters
```rust
// In GraphServiceActor::handle UpdateSimulationParams (line 2013)
self.simulation_params = msg.params.clone();
// CRITICAL: Also update target_params so smooth transitions work correctly
self.target_params = msg.params.clone();
```

### Fix 3: Client Path Corrections
```typescript
// In settingsStore.ts - Add physics to ESSENTIAL_PATHS
'visualisation.graphs.logseq.physics',
'visualisation.graphs.visionflow.physics'

// In PhysicsEngineTab.tsx - Fix paths
// OLD: visualisation.physics.*
// NEW: visualisation.graphs.[graph].physics.*
```

## Verification Steps

1. **Check if settings are being saved:**
   ```bash
   grep "repelK\|springK" /data/settings.yaml
   ```

2. **Check if propagation is happening (after rebuild):**
   ```bash
   tail -f /logs/rust-error.log | grep "propagating to GPU"
   ```

3. **Check if UpdateSimulationParams is sent:**
   ```bash
   tail -f /logs/rust-error.log | grep "UpdateSimulationParams"
   ```

## Current Status
- ✅ Code fixes have been applied
- ⏳ Server needs to be rebuilt with the new code
- ⏳ Once rebuilt, physics changes should apply immediately without restart

## Testing After Rebuild

1. Change a physics parameter in the UI (e.g., repelK)
2. Observe the graph should immediately respond
3. Check logs for "Physics setting changed, propagating to GPU actors"
4. Verify no server restart is needed