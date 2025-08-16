# Physics Settings Updates - Conservative Presets

## Summary
Updated the physics settings across the entire stack to use more conservative presets that ensure immediate settling and stable graph visualization.

## Changes Made

### 1. Settings.yaml Updates
**File**: `/workspace/ext/data/settings.yaml`

Updated both `logseq` and `visionflow` graph physics with conservative presets:

```yaml
physics:
  # Core Forces (Tuned for a close, stable cluster)
  spring_strength: 0.02          # Was: 0.001 - Increased to pull connected nodes together
  repulsion_strength: 10.0       # Was: 50.0 - Drastically reduced to prevent explosion
  attraction_strength: 0.01      # Was: 0.0005 - Increased for gentle centre pull
  gravity: 0.0                   # Was: 0.0 - Disabled (no change)
  
  # Stability & Settling Controls
  damping: 0.98                  # Was: 0.98 - Very high friction (no change)
  max_velocity: 0.5              # Was: 0.5 - Very low to prevent fast movement (no change)
  temperature: 0.0               # Was: 0.5 - Disabled random energy
  time_step: 0.005               # Was: 0.01 - Smaller for accuracy
  
  # Boundary & Collision
  enable_bounds: true            # Was: true - Keep nodes in space (no change)
  bounds_size: 50.0              # Was: 200.0 - Smaller for tight cluster
  boundary_damping: 0.99         # Was: 0.95 - Higher to stick to boundary
  collision_radius: 0.5          # Was: 0.15 - Increased to prevent overlap
  
  # Advanced Parameters
  iterations: 150                # Was: 200 - Reduced as it settles faster
  repulsion_distance: 20.0       # Was: 50.0 - Smaller repulsion range
  mass_scale: 0.5                # Was: 1.0 - Lower inertia
  update_threshold: 0.01         # Was: 0.05 - Lower for processing small movements
```

### 2. Client-Side Updates

#### PhysicsEngineControls.tsx
**File**: `/workspace/ext/client/src/features/physics/components/PhysicsEngineControls.tsx`

- Updated default values to match conservative presets
- Adjusted slider ranges for appropriate control:
  - Repulsion: 0-100 (was 0-2000)
  - Attraction: 0-0.05 (was 0-0.1)
  - Damping: 0.8-1.0 (was 0-1)
  - Temperature: 0-2 (was 0-10)
  - Gravity: 0-0.1 (was 0-1)
  - Max Velocity: 0.1-5 (was 1-50)
  - Time Step: 0.001-0.05 (was 0.01-0.5)

- Fixed property name mapping to snake_case:
  - `repulsionStrength` → `repulsion_strength`
  - `attractionStrength` → `attraction_strength`
  - `springStrength` → `spring_strength`
  - `maxVelocity` → `max_velocity`
  - `timeStep` → `time_step`

#### PhysicsSettings Interface
**File**: `/workspace/ext/client/src/features/settings/config/settings.ts`

Updated interface to use snake_case convention matching settings.yaml:

```typescript
export interface PhysicsSettings {
  enabled: boolean;
  
  // Core Forces
  spring_strength: number;        // Was: springStrength
  repulsion_strength: number;     // Was: repulsionStrength
  attraction_strength: number;    // Was: attractionStrength
  gravity: number;
  
  // Stability & Settling Controls
  damping: number;
  max_velocity: number;           // Was: maxVelocity
  temperature: number;
  time_step: number;              // Was: timeStep
  
  // Boundary & Collision
  enable_bounds: boolean;         // Was: enableBounds
  bounds_size: number;            // Was: boundsSize
  boundary_damping: number;       // Was: boundaryDamping
  collision_radius: number;       // Was: collisionRadius
  
  // Advanced Parameters
  iterations: number;
  repulsion_distance: number;     // Was: repulsionDistance
  mass_scale: number;             // Was: massScale
  update_threshold: number;       // Was: updateThreshold
}
```

### 3. Key Benefits of Conservative Settings

1. **Immediate Settling**: High damping (0.98) and low time step (0.005) ensure the graph settles quickly
2. **No Explosion**: Reduced repulsion (10.0 vs 50.0) prevents nodes from flying apart
3. **Stable Clustering**: Increased spring strength (0.02 vs 0.001) keeps connected nodes together
4. **No Random Motion**: Temperature set to 0 eliminates random energy
5. **Clear Separation**: Increased collision radius (0.5 vs 0.15) prevents overlap
6. **Compact Layout**: Smaller bounds (50 vs 200) creates a tighter visualization

### 4. Validation Logic

All slider boundaries have been adjusted to:
- Prevent invalid values that could destabilize the simulation
- Provide appropriate precision for fine-tuning
- Match the scale of the conservative defaults
- Ensure minimum values that maintain stability (e.g., damping min 0.8)

### 5. Client-Server Connection

The physics parameters flow through the system as follows:

1. **Settings Store** → Updates local state and sends to server
2. **API Endpoint** → `/api/settings` receives full settings updates
3. **GPU Actor** → Receives `UpdateSimulationParams` message
4. **CUDA Kernels** → Apply physics in unified GPU compute

The snake_case naming convention is now consistent across:
- settings.yaml (source of truth)
- TypeScript interfaces
- API payloads
- Rust structs

## Testing Recommendations

1. Test graph initialization with new defaults
2. Verify sliders update physics in real-time
3. Confirm settings persist across sessions
4. Check GPU actor receives correct parameter names
5. Validate that graphs settle immediately without bouncing

## Migration Notes

Existing users with saved settings may need to:
1. Clear local storage to pick up new defaults
2. Adjust their custom settings to the new ranges
3. Re-save presets with the updated parameter names