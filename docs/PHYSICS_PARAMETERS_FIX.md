# Physics Parameters Fix - Settings.yaml as Source of Truth

## Issue
Physics parameters were being hardcoded in multiple places instead of using values from `settings.yaml`, causing inconsistencies and making it impossible to configure physics behavior through the settings file.

## Root Cause Analysis

### 1. Hardcoded Values in Multiple Places
- `SimParams::default()` had hardcoded values
- `SimulationParams::new()` had different hardcoded values
- Conversion between types was incomplete

### 2. Missing Parameter Flow
The parameter flow was broken:
```
settings.yaml → PhysicsSettings → SimulationParams → SimParams → GPU
                                            ↑
                                     [MISSING CONVERSION]
```

## Solution Implemented

### 1. Complete Parameter Flow
Created proper conversion chain:
- `PhysicsSettings` (from settings.yaml) → `SimulationParams` (via From trait)
- `SimulationParams` → `SimParams` (via From trait)
- All parameters now flow from settings.yaml to GPU

### 2. Updated Conversions

#### Added missing fields to SimulationParams:
```rust
pub max_velocity: f32,
pub attraction_strength: f32,
pub collision_radius: f32,
pub temperature: f32,
```

#### Created SimParams conversion:
```rust
impl From<&SimulationParams> for SimParams {
    fn from(params: &SimulationParams) -> Self {
        Self {
            spring_k: params.spring_strength,
            repel_k: params.repulsion,
            damping: params.damping,
            dt: params.time_step,
            max_velocity: params.max_velocity,
            // ... all fields mapped from settings
        }
    }
}
```

### 3. Updated Message Handlers
`UpdateSimulationParams` handler now:
1. Updates `simulation_params`
2. Converts to `unified_params` using From trait
3. Updates unified GPU compute with new params

### 4. Settings.yaml Values Used

From `/workspace/ext/data/settings.yaml`:
```yaml
physics:
  enabled: true
  iterations: 200
  damping: 0.9
  spring_strength: 0.005
  repulsion_strength: 50.0
  repulsion_distance: 50.0
  attraction_strength: 0.001
  max_velocity: 1.0
  collision_radius: 0.15
  bounds_size: 200.0
  enable_bounds: true
  mass_scale: 1.0
  boundary_damping: 0.95
  update_threshold: 0.05
  time_step: 0.01
  temperature: 0.5
  gravity: 0.0
```

## Parameter Flow Diagram

```
1. settings.yaml loaded at startup
   ↓
2. AppFullSettings::load()
   ↓
3. PhysicsSettings extracted from visualisation.graphs.logseq.physics
   ↓
4. SimulationParams::from(&PhysicsSettings)
   ↓
5. UpdateSimulationParams message sent to GPUComputeActor
   ↓
6. SimParams::from(&SimulationParams)
   ↓
7. unified_compute.set_params(sim_params)
   ↓
8. GPU kernel uses parameters
```

## Control Center Updates
When physics parameters are modified through the control center:
1. WebSocket receives parameter update
2. Creates new SimulationParams with updated values
3. Sends UpdateSimulationParams to GPUComputeActor
4. Conversion chain ensures GPU gets proper values

## Files Modified

1. `/workspace/ext/src/utils/unified_gpu_compute.rs`
   - Updated SimParams::default() to use settings.yaml values
   - Added From<&SimulationParams> for SimParams

2. `/workspace/ext/src/actors/gpu_compute_actor.rs`
   - Updated UpdateSimulationParams handler
   - Now converts and updates unified_params

3. `/workspace/ext/src/models/simulation_params.rs`
   - Added missing physics fields
   - Updated From<&PhysicsSettings> implementation

4. `/workspace/ext/src/app_state.rs`
   - Uses From trait for conversion instead of manual construction

## Verification

To verify the fix:
1. Modify physics values in settings.yaml
2. Restart the application
3. Physics should use the new values immediately
4. Control center modifications should work correctly

## Benefits

1. **Single Source of Truth**: All physics parameters originate from settings.yaml
2. **Consistency**: Same values used throughout the system
3. **Configurability**: Easy to tune physics without code changes
4. **Maintainability**: Clear parameter flow, easy to debug
5. **Type Safety**: Compile-time checked conversions

## Testing Recommendations

1. Test with extreme values to ensure proper bounds
2. Verify control center updates work correctly
3. Test with different graph types (logseq vs visionflow)
4. Ensure physics behaves consistently across restarts