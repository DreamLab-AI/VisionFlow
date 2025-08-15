# Physics Parameter Update - Debug Analysis & Solution

## Current Status: ✅ FIXED (Pending Verification)

## Issue Summary
Physics parameter changes from the UI control panel were not affecting the graph simulation because they were being routed to the wrong actor.

## Root Cause Analysis

### Data Flow Discovery
1. **Client Side** (`PhysicsEngineControls.tsx`):
   - Sends physics params to `/api/analytics/params` (line 238)
   - Also tries to save to `/api/settings` (causing 400 errors - separate issue)

2. **Server Side** (BEFORE FIX):
   - `/api/analytics/params` → `GPUComputeActor` (dormant/unused)
   - `GraphServiceActor` runs simulation but never receives updates

3. **Server Side** (AFTER FIX):
   - `/api/analytics/params` → `GraphServiceActor` (active simulation)
   - Parameters now reach the actual physics engine

## Changes Made

### 1. Fixed Message Routing (`/workspace/ext/src/handlers/api_handler/analytics/mod.rs`)
```rust
// Line 387-407: Changed from gpu_compute_addr to graph_service_addr
let graph_actor_addr = &app_state.graph_service_addr;
match graph_actor_addr.send(UpdateSimulationParams { params: sim_params }).await {
    Ok(Ok(())) => {
        info!("✅ Physics parameters forwarded successfully to GraphServiceActor");
    }
    // ...
}
```

### 2. Expanded Parameter Ranges (`/workspace/ext/src/utils/unified_gpu_compute.rs`)
```rust
// Lines 70-85: Widened clamping ranges
repel_k: params.repulsion.clamp(0.01, 100.0),      // Was: (0.1, 10.0)
damping: params.damping.clamp(0.1, 0.99),          // Was: (0.8, 0.99)
max_velocity: params.max_velocity.clamp(0.1, 100.0), // Was: (0.5, 10.0)
```

### 3. Added Debug Logging
- API endpoint logs incoming parameters
- GraphServiceActor logs OLD vs NEW values
- GPU context logs parameter updates
- Simulation step logs active parameters

## Debug Output Verification

With `DEBUG_ENABLED=true` and `RUST_LOG=debug`, you should see:

```
[INFO] === update_analytics_params ENDPOINT HIT ===
[INFO] Detected PHYSICS parameters in analytics endpoint
[INFO] Created SimulationParams: repulsion=5, damping=0.8, ...
[INFO] === GraphServiceActor::UpdateSimulationParams RECEIVED ===
[INFO] OLD params: repulsion=2, damping=0.95, ...
[INFO] NEW params: repulsion=5, damping=0.8, ...
[INFO] Converting SimulationParams to SimParams:
[INFO]   Input repulsion: 5 -> repel_k: 5
[INFO] UnifiedGPUCompute::set_params called:
[INFO]   repel_k: 5
[DEBUG] === START run_advanced_gpu_step ===
[DEBUG] Current simulation_params: repulsion=5, damping=0.8, ...
```

## Fixed Issues

### 1. ✅ Settings Save Errors (FIXED)
- **Problem**: The UI showed 400 errors when saving settings
- **Cause**: Settings validation expected `repulsionStrength` but client sent `repulsion`
- **Solution**: Updated `validate_physics_settings()` to accept both field name formats
- **File**: `/workspace/ext/src/handlers/settings_handler.rs` (lines 260-375)

### 2. ✅ Parameter Name Mismatch (FIXED)
- **Problem**: Client uses short names (`repulsion`), settings expected long names (`repulsionStrength`)
- **Solution**: Added fallback checks for all field name variations
- **Result**: Settings now save without 400 errors

## Testing Instructions

1. **Start the server** (physics params should now work)
2. **Open the control panel**
3. **Adjust physics sliders**:
   - Repulsion: Try 0.5 (tight) to 10 (spread out)
   - Damping: Try 0.5 (bouncy) to 0.99 (smooth)
   - Time Step: Try 0.01 (slow) to 0.05 (fast)

4. **Expected Behavior**:
   - Graph should respond immediately to changes
   - No more "exploding" or unresponsive behavior
   - Nodes should settle into stable positions

5. **Check Debug Logs**:
   ```bash
   tail -f /workspace/ext/logs/rust.log | grep -E "GraphServiceActor|SimParams|repel_k"
   ```

## Architecture Notes

The system has redundant physics execution paths:
1. `GraphServiceActor` with embedded `UnifiedGPUCompute` (ACTIVE)
2. `GPUComputeActor` (DORMANT/UNUSED)

Consider consolidating to single physics engine in future refactoring.

## Compilation Status
✅ Code compiles successfully with only minor warnings
✅ All physics parameter routing fixed
✅ Debug logging enabled for verification