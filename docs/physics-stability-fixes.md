# Force-Directed Graph Physics Stability Fixes

## Executive Summary

This document details the comprehensive fixes applied to resolve critical instability issues in the VisionFlow force-directed graph physics system. The primary issue was nodes "exploding and bouncing" during simulation, caused by multiple compounding factors including a double-execute bug, parameter scaling issues, buffer management problems, and boundary handling defects.

## Critical Issues Resolved

### 1. Double-Execute Bug (HIGHEST PRIORITY - FIXED)

**Location**: `src/actors/gpu_compute_actor.rs:462-468`

**Problem**: The `get_node_data_internal()` method was calling `unified_compute.execute()` to retrieve positions, causing the physics simulation to advance twice per frame. This led to:
- Doubled velocity integration
- Accelerated warmup progression
- Unstable force accumulation
- Visible node explosion and bouncing

**Solution**: 
- Added new `get_positions()` method in `UnifiedGPUCompute` that retrieves positions WITHOUT executing physics
- Modified `get_node_data_internal()` to use `get_positions()` instead of `execute()`

```rust
// Before (BROKEN):
let positions = unified_compute.execute() // This advances physics!

// After (FIXED):
let positions = unified_compute.get_positions() // Just reads positions
```

### 2. Parameter Scale and Validation Issues (FIXED)

**Location**: `src/utils/unified_gpu_compute.rs:56-79`

**Problem**: Physics parameters from YAML configuration could have extreme values causing instability:
- Excessive repulsion forces (up to 600+)
- Large timesteps (up to 0.12)
- High temperature values
- Insufficient damping

**Solution**: Implemented parameter validation with safe clamping ranges:

```rust
impl From<&SimulationParams> for SimParams {
    fn from(params: &SimulationParams) -> Self {
        Self {
            spring_k: params.spring_strength.clamp(0.0001, 0.1),
            repel_k: params.repulsion.clamp(0.1, 10.0),
            damping: params.damping.clamp(0.8, 0.99),
            dt: params.time_step.clamp(0.001, 0.05),
            max_velocity: params.max_velocity.clamp(0.5, 10.0),
            max_force: params.max_force.clamp(1.0, 20.0),
            // ... additional parameters with bounds
        }
    }
}
```

### 3. Edge Buffer Size Drift (FIXED)

**Location**: `src/actors/gpu_compute_actor.rs:276-288`, `src/utils/unified_gpu_compute.rs:560-627`

**Problem**: When graph topology changed (nodes/edges added/removed), the `UnifiedGPUCompute` buffers weren't resized, causing:
- Edge upload failures
- Stale spring forces
- Unbalanced force calculations

**Solution**: 
- Implemented `resize_buffers()` method for dynamic buffer management
- Properly handles both node and edge count changes
- Preserves existing position data during resize

```rust
pub fn resize_buffers(&mut self, new_num_nodes: usize, new_num_edges: usize) {
    // Allocates new buffers
    // Copies existing data
    // Updates buffer references
}
```

### 4. Boundary Bounce Issues (FIXED)

**Location**: `src/utils/visionflow_unified.cu:486-548`

**Problem**: Hard boundary clamping with high damping values caused visible bouncing:
- Abrupt velocity reversals at boundaries
- Insufficient progressive damping
- Energy accumulation at edges

**Solution**: Implemented progressive boundary handling with:
- Gradual force application starting at 85% of boundary
- Quadratic force scaling for smooth deceleration
- Progressive damping based on boundary distance
- Soft clamping with velocity reduction

```cuda
// Progressive damping calculation
float distance_ratio = (fabsf(position.x) - boundary_margin) / 
                      (p.params.viewport_bounds - boundary_margin);
float boundary_force = -distance_ratio * distance_ratio * 
                      boundary_force_strength * copysignf(1.0f, position.x);
float progressive_damping = p.params.boundary_damping * 
                           (1.0f - 0.5f * distance_ratio);
```

## Stable Physics Configuration

### Recommended Settings (from `data/settings.yaml`)

```yaml
physics:
  # Core Forces (Balanced for stability)
  spring_strength: 0.005         # Moderate spring tension
  repulsion_strength: 2.0        # Low repulsion prevents explosion
  attraction_strength: 0.0001    # Very light centre gravity
  
  # Stability Controls
  damping: 0.95                  # High friction for quick settling
  max_velocity: 2.0              # Reasonable movement speed
  temperature: 0.01              # Minimal random energy
  time_step: 0.016               # Standard 60fps timestep
  
  # Boundaries
  bounds_size: 500.0             # Large space for nodes
  boundary_damping: 0.5          # Soft boundary response
  collision_radius: 2.0          # Personal space per node
```

## Performance Impact

The fixes have minimal performance impact whilst providing significant stability improvements:

- **Double-execute fix**: ~50% performance improvement (eliminates redundant computation)
- **Parameter validation**: Negligible overhead (one-time conversion)
- **Buffer resizing**: Only occurs on topology changes
- **Boundary handling**: Same computational complexity, better behaviour

## Testing Recommendations

1. **Warmup Testing**: Verify smooth force progression during first 200 iterations
2. **Boundary Testing**: Place nodes near viewport edges and observe settling
3. **Dynamic Graph Testing**: Add/remove nodes and edges during simulation
4. **Parameter Range Testing**: Test with extreme UI slider values
5. **Long-run Stability**: Run simulation for 10,000+ iterations

## Implementation Checklist

- [x] Fix double-execute bug in `get_node_data_internal()`
- [x] Add `get_positions()` method to `UnifiedGPUCompute`
- [x] Implement parameter validation and clamping
- [x] Add dynamic buffer resizing support
- [x] Fix boundary bounce with progressive damping
- [x] Update YAML configuration with stable defaults
- [x] Document all changes and constraints

## Migration Notes

For existing deployments:

1. **Rebuild PTX**: The CUDA kernel must be recompiled
2. **Update Configuration**: Review and update physics settings in YAML
3. **Clear Cache**: Remove any cached physics state
4. **Monitor Logs**: Watch for resize operations and parameter clamping

## Future Improvements

1. **Adaptive Timestep**: Dynamically adjust dt based on system energy
2. **Hierarchical Force Calculation**: Use Barnes-Hut for O(n log n) complexity
3. **GPU Memory Pooling**: Reuse allocations for buffer resizing
4. **Multi-resolution Simulation**: LOD for distant node clusters
5. **Constraint Solver**: Implement proper constraint satisfaction solver

## References

- Original issue analysis: `ext/task.md`
- GPU compute actor: `src/actors/gpu_compute_actor.rs`
- Unified compute module: `src/utils/unified_gpu_compute.rs`
- CUDA kernel: `src/utils/visionflow_unified.cu`
- Configuration: `data/settings.yaml`