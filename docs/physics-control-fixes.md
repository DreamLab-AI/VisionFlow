# Physics Control System Fixes

## Problem Summary

The physics controls in the UI were not affecting the simulation because of multiple issues:

1. **Wrong endpoint**: Client sends physics to `/api/analytics/params` (misnamed)
2. **Wrong message type**: The endpoint expected `VisualAnalyticsParams` which doesn't have physics fields
3. **Missing GPU update**: Even when params were received, they weren't pushed to GPU
4. **No enabled flag**: Physics couldn't be disabled
5. **Double-execute remained**: Legacy GPU path still had the bug

## Fixes Applied

### 1. Analytics Endpoint Workaround

**File**: `src/handlers/api_handler/analytics/mod.rs`

The `/api/analytics/params` endpoint now accepts raw JSON and detects if it contains physics parameters. When physics params are detected, it creates a proper `UpdateSimulationParams` message and forwards it to the GPU actor.

```rust
// Detect physics params in the JSON
if params.get("repulsion").is_some() || params.get("damping").is_some() {
    // Convert to SimulationParams and send UpdateSimulationParams
}
```

### 2. UpdateSimulationParams Handler Fix

**File**: `src/actors/gpu_compute_actor.rs`

The handler now:
- Logs the received parameters including enabled state
- Converts to GPU params with validation
- Immediately pushes to GPU via `set_params()`

### 3. Added Physics Enabled Flag

**File**: `src/models/simulation_params.rs`

Added `enabled: bool` field to `SimulationParams` struct to allow disabling physics entirely.

**File**: `src/actors/gpu_compute_actor.rs`

The `compute_forces_internal()` method now checks this flag before computing:

```rust
if !self.simulation_params.enabled {
    return Ok(());  // Skip physics computation
}
```

### 4. Communication Pathway Documentation

Added extensive inline documentation to clarify the data flow:

**Client** (`PhysicsEngineControls.tsx`):
```typescript
// IMPORTANT COMMUNICATION PATHWAY:
// Physics parameters are sent via REST API, NOT WebSocket!
// Data flow: UI -> REST /api/analytics/params -> UpdateSimulationParams -> GPU
```

**Server** (`socket_flow_handler.rs`):
```rust
// WARNING: This WebSocket path is DEPRECATED and should not be used!
// WebSocket is ONLY for:
//   - Binary position/velocity streaming (high frequency)
//   - Real-time graph updates
```

## Communication Architecture

### Correct Data Flow

```
Physics Controls (UI)
        ↓
REST: POST /api/analytics/params
        ↓
analytics::update_analytics_params() [detects physics JSON]
        ↓
UpdateSimulationParams message
        ↓
GPUComputeActor::handle<UpdateSimulationParams>
        ↓
SimParams::from() [validation & clamping]
        ↓
unified_compute.set_params()
        ↓
GPU Kernel
```

### WebSocket vs REST

| Channel | Purpose | Frequency | Data Format |
|---------|---------|-----------|-------------|
| **WebSocket** | Position/velocity streaming | 60 FPS | Binary protocol |
| **REST API** | Control parameters | On change | JSON |

### Message Types

- `UpdateSimulationParams`: Primary physics update (from REST)
- `UpdatePhysicsParams`: Legacy/deprecated (from WebSocket)
- `UpdateVisualAnalyticsParams`: Visual analytics only
- `ComputeForces`: Trigger physics step
- `GetNodeData`: Retrieve positions (fixed to not double-execute)

## Remaining Issues

1. **Endpoint Naming**: `/api/analytics/params` should be `/api/physics/params`
2. **Message Confusion**: Multiple overlapping message types for physics
3. **Legacy Paths**: Deprecated WebSocket physics path still exists

## Testing Checklist

- [ ] Move physics sliders - values should update in GPU logs
- [ ] Disable physics checkbox - simulation should stop
- [ ] Check enabled state persists across updates
- [ ] Verify no double-stepping (smooth motion)
- [ ] Confirm WebSocket only carries positions

## Future Improvements

1. Create dedicated `/api/physics/params` endpoint
2. Remove deprecated WebSocket physics handler
3. Consolidate physics message types
4. Add physics state query endpoint
5. Implement proper enable/disable throughout pipeline