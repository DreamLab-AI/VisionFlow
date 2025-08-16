# GPU Physics Parameter Alignment Plan

## GPU CUDA Kernel (Ground Truth)

Based on analysis of `/workspace/ext/src/utils/visionflow_unified.cu`, the GPU kernel uses the following `SimParams` structure:

```cuda
struct SimParams {
    // Force parameters
    float spring_k;              // Spring constant for edge attraction
    float repel_k;               // Repulsion constant between nodes
    float damping;               // Velocity damping (0-1, where 1.0 = frozen)
    float dt;                    // Time step for integration
    float max_velocity;          // Maximum node velocity
    float max_force;             // Maximum force magnitude
    
    // Stress majorization (advanced layout)
    float stress_weight;         // Weight for stress optimization
    float stress_alpha;          // Blending factor for stress updates
    
    // Constraints
    float separation_radius;     // Minimum separation between nodes
    float boundary_limit;        // Hard boundary limit
    float alignment_strength;    // Force for alignment constraints
    float cluster_strength;      // Force for cluster cohesion
    
    // Boundary control
    float boundary_damping;      // Damping when near boundaries
    
    // System
    float viewport_bounds;       // Soft boundary size
    float temperature;           // System temperature (simulated annealing)
    int iteration;               // Current iteration number
    int compute_mode;            // 0=basic, 1=dual, 2=constraints, 3=analytics
}
```

## Current Misalignments

### 1. Parameter Name Mismatches

| GPU Kernel | Rust SimulationParams | Client Settings | Issue |
|------------|----------------------|-----------------|-------|
| `spring_k` | `spring_strength` | `springStrength` | Name mismatch |
| `repel_k` | `repulsion` | `repulsionStrength` | Name mismatch |
| `dt` | `time_step` | `timeStep` | Name mismatch |
| `separation_radius` | `collision_radius` | `collisionRadius` | Name mismatch |
| `stress_weight` | ❌ Missing | ❌ Missing | Not exposed |
| `stress_alpha` | ❌ Missing | ❌ Missing | Not exposed |
| `boundary_limit` | ❌ Missing | ❌ Missing | Not exposed |
| `alignment_strength` | ❌ Missing | ❌ Missing | Not exposed |
| `cluster_strength` | ❌ Missing | ❌ Missing | Not exposed |
| `iteration` | ❌ Not passed | ❌ Not tracked | GPU tracks internally |
| `compute_mode` | ❌ Not configurable | ❌ Not exposed | Hardcoded |

### 2. Force Calculation Differences

#### GPU Spring Force (Line 153-159, 213-221):
```cuda
// Uses natural length concept
float natural_length = fminf(params.separation_radius * 5.0f, 10.0f);
float displacement = dist - natural_length;
float spring_force = params.spring_k * displacement * edge_weight[e];
spring_force = fmaxf(-params.max_force * 0.5f, fminf(params.max_force * 0.5f, spring_force));
```

**Key insights:**
- Spring force uses **displacement from natural length**, not raw distance
- Natural length is adaptive: `min(separation_radius * 5, 10)`
- Spring force is clamped to `±max_force * 0.5`

#### GPU Repulsion Force (Line 109-138):
```cuda
const float MIN_DISTANCE = 0.15f;
const float MAX_REPULSION_DIST = 50.0f;

if (dist < MIN_DISTANCE) {
    // Strong repulsion when too close
    float push_force = params.repel_k * (MIN_DISTANCE - dist + 1.0f) / (MIN_DISTANCE * MIN_DISTANCE);
} else if (dist < MAX_REPULSION_DIST) {
    // Normal repulsion with distance squared
    float dist_sq = fmaxf(dist * dist, MIN_DISTANCE * MIN_DISTANCE);
    float repulsion = params.repel_k / dist_sq;
}
```

**Key insights:**
- MIN_DISTANCE = 0.15 is hardcoded (should match `separation_radius`)
- MAX_REPULSION_DIST = 50.0 is hardcoded (should use `max_repulsion_distance`)
- Special handling for coincident nodes using golden ratio

### 3. Boundary Handling (Line 489-551)

GPU uses progressive boundary damping:
```cuda
float boundary_margin = p.params.viewport_bounds * 0.85f;
float boundary_force_strength = 2.0f;

if (fabsf(position.x) > boundary_margin) {
    float distance_ratio = (fabsf(position.x) - boundary_margin) / (p.params.viewport_bounds - boundary_margin);
    // Quadratic force increase
    float boundary_force = -distance_ratio * distance_ratio * boundary_force_strength;
    // Progressive damping
    float progressive_damping = p.params.boundary_damping * (1.0f - 0.5f * distance_ratio);
    velocity.x *= progressive_damping;
}
```

**Key insights:**
- Boundary forces start at 85% of viewport_bounds
- Uses quadratic force increase near boundaries
- Progressive damping increases as nodes approach boundary
- Soft clamp at 98% of viewport_bounds with 50% velocity reduction

### 4. Warmup & Stability (Line 461-484)

GPU has built-in warmup period:
```cuda
if (p.params.iteration < 200) {
    float warmup = p.params.iteration / 200.0f;
    force = vec3_scale(force, warmup * warmup); // Quadratic warmup
    float extra_damping = 0.98f - 0.13f * warmup; // From 0.98 to 0.85
}

if (p.params.iteration < 5) {
    velocity = make_vec3(0.0f, 0.0f, 0.0f); // Zero velocity initially
}
```

## Required Changes

### 1. Update Rust SimParams Structure

```rust
// Match GPU structure exactly
pub struct SimParams {
    pub spring_k: f32,              // Rename from spring_strength
    pub repel_k: f32,               // Rename from repulsion
    pub damping: f32,
    pub dt: f32,                    // Rename from time_step
    pub max_velocity: f32,
    pub max_force: f32,
    
    // Add missing parameters
    pub stress_weight: f32,         // NEW
    pub stress_alpha: f32,          // NEW (default: 0.1)
    
    pub separation_radius: f32,     // Rename from collision_radius
    pub boundary_limit: f32,        // NEW (default: viewport_bounds * 0.98)
    pub alignment_strength: f32,    // NEW (default: 0.0)
    pub cluster_strength: f32,      // NEW (default: 0.0)
    
    pub boundary_damping: f32,
    pub viewport_bounds: f32,
    pub temperature: f32,
    pub iteration: i32,             // NEW - track iteration count
    pub compute_mode: i32,          // NEW (default: 0 for basic)
}
```

### 2. Update Conversion Functions

The conversion from PhysicsSettings to SimParams needs updating:
- Map field names correctly
- Add default values for missing parameters
- Pass iteration count from actor

### 3. Update REST API

Add new physics parameters to validation:
- `stressWeight` (0.0-1.0)
- `stressAlpha` (0.0-1.0)
- `alignmentStrength` (0.0-10.0)
- `clusterStrength` (0.0-10.0)
- `computeMode` (0-3)

### 4. Update Control Centre UI

Add controls for new parameters in Developer panel:
- Stress optimization controls
- Constraint controls
- Compute mode selector
- Move debug settings from XR/AR panel

### 5. Fix Physics Constants

Update hardcoded values to match GPU:
- MIN_DISTANCE = 0.15
- MAX_REPULSION_DIST should use max_repulsion_distance parameter
- Boundary margin = 85% of viewport_bounds
- Boundary force strength = 2.0
- Warmup iterations = 200
- Zero velocity iterations = 5

## Implementation Priority

1. **Critical** - Fix SimParams structure alignment (prevents bouncing)
2. **Critical** - Fix force calculations to match GPU
3. **High** - Add iteration tracking and warmup logic
4. **Medium** - Add missing advanced parameters
5. **Low** - UI improvements and debug panel reorganization

## Testing Plan

1. Set damping = 0.95 (high damping for stability)
2. Set repulsion = 50.0 (much lower than current)
3. Set spring_strength = 0.005 (very gentle)
4. Set max_velocity = 1.0 (prevent explosions)
5. Set boundary_damping = 0.95 (strong boundary control)
6. Enable iteration tracking for warmup period