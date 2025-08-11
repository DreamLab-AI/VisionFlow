# GPU Compute Improvements & Troubleshooting Guide

This document details recent improvements to the GPU compute system to address graph stability issues and provides troubleshooting guidance.

## Problem: Graph Bouncing

The graph visualization exhibited excessive oscillation ("bouncing") due to:
1. Aggressive default physics parameters
2. Inconsistent PTX file loading paths
3. Missing warmup period in advanced kernels
4. Excessive boundary forces

## Solutions Implemented

### 1. Optimized Physics Parameters

#### Default SimulationParams (Reduced Bouncing)
```rust
// src/models/simulation_params.rs
SimulationParams {
    time_step: 0.15,           // Reduced from 0.2
    spring_strength: 0.015,     // Reduced from 0.02
    repulsion: 1000.0,          // Reduced from 1500.0 (33% reduction)
    damping: 0.9,               // Increased from 0.85
    boundary_damping: 0.95,     // Increased from 0.9
    // ...
}
```

#### Phase-Specific Presets
- **Initial Phase**: Higher repulsion (1800.0) for better initial spread
- **Dynamic Phase**: Lower time_step (0.12) for optimal stability
- **Finalize Phase**: High damping (0.95) for settling

### 2. PTX Loading Path Standardization

#### Search Order (Priority)
1. `/app/src/utils/ptx/*.ptx` - Primary location
2. `/app/src/utils/*.ptx` - Legacy fallback
3. `src/utils/*.ptx` - Development fallback

#### Docker Configuration
```dockerfile
# Dockerfile.dev - Copy to both locations for compatibility
RUN mkdir -p /app/src/utils/ptx /app/src/utils
COPY src/utils/ptx/*.ptx /app/src/utils/ptx/
COPY src/utils/ptx/*.ptx /app/src/utils/
```

### 3. Advanced Kernel Improvements

#### Force Clamping & Warmup
```cuda
// src/utils/advanced_compute_forces.cu

// Warmup period reduces initial forces
const int WARMUP_ITERATIONS = 100;
if (params.iteration < WARMUP_ITERATIONS) {
    warmup_factor = 0.1f + 0.9f * (float(params.iteration) / float(WARMUP_ITERATIONS));
}

// Reduced force clamp for stability
total_force = clamp3(total_force, 500.0f);  // Reduced from 1000.0f

// Gentler boundary forces
if (abs_p > hard) {
    return -p * 5.0f;  // Reduced from 10.0f
}
```

#### Velocity Damping Near Boundaries
```cuda
// Extra damping when near boundaries prevents "bouncing off walls"
float dist_from_boundary = fminf(
    params.viewport_bounds - fabsf(pos.x),
    params.viewport_bounds - fabsf(pos.y)
);
if (dist_from_boundary < params.viewport_bounds * 0.1f) {
    float boundary_damp = 0.95f;
    vel *= boundary_damp;
}
```

### 4. Kernel Selection Logic

```rust
// src/actors/gpu_compute_actor.rs
fn determine_kernel_mode(num_nodes, has_isolation_layers, has_complex_analysis) -> KernelMode {
    // Priority 1: Visual Analytics for complex analysis
    if (has_isolation_layers || has_complex_analysis) && visual_analytics_kernel.is_some() {
        return KernelMode::VisualAnalytics;
    }
    
    // Priority 2: Advanced for medium graphs (1000-10000 nodes)
    if num_nodes >= 1000 && num_nodes <= 10000 && advanced_gpu_kernel.is_some() {
        return KernelMode::Advanced;
    }
    
    // Priority 3: Visual Analytics for very large graphs (>10000)
    if num_nodes > 10000 && visual_analytics_kernel.is_some() {
        return KernelMode::VisualAnalytics;
    }
    
    // Fallback: Legacy kernel
    return KernelMode::Legacy;
}
```

## Runtime Diagnostics

### Enable Debug Logging

The system now includes comprehensive logging for debugging physics issues:

```bash
# Set log level to see debug messages
RUST_LOG=info,webxr=debug cargo run
```

### Key Log Messages

#### PTX Loading
```
PTX_LOAD: Checking legacy kernel at /app/src/utils/ptx/compute_forces.ptx
PTX_LOAD: Legacy PTX file exists, loading...
PTX_LOAD: Legacy kernel function loaded successfully
```

#### Physics State (Every 60 frames)
```
PHYSICS_STATE: iteration=120, kernel_mode=Legacy, compute_mode=Legacy, nodes=500, edges=1000, constraints=0
PHYSICS_PARAMS: time_step=0.1500, spring=0.0150, repulsion=1000.0, damping=0.9000, boundary_damping=0.9500, max_repulsion_dist=2000.0
```

#### Advanced Parameters (When active)
```
ADVANCED_PARAMS: iteration=60, nodes=500
  Physics: spring=0.0150, damping=0.9000, repel=1000.0, dt=0.1500
  Force weights: semantic=0.70, temporal=0.80, structural=0.85, constraint=0.80, boundary=0.70
```

#### Message Handler State Changes
```
MSG_HANDLER: UpdateConstraints received
  Current state - compute_mode: Legacy, kernel_mode: Legacy, constraints: 0
  Parsed 5 new constraints (was 0)
  Switching compute mode from Legacy to Advanced due to constraints
```

#### Kernel Selection Decision
```
KERNEL_SELECT: Determining kernel mode
  Input factors - nodes: 1500, isolation_layers: false, complex_analysis: true
  Available kernels - VA: true, AdvGPU: true, Adv: true
  Selecting Advanced (medium-large graph)
```

## Troubleshooting Guide

### Issue: Graph Still Bouncing

1. **Check Active Kernel**
   - Look for `PHYSICS_STATE` logs to see which kernel is active
   - Legacy kernel should be active for graphs <1000 nodes
   - Verify PTX files are loaded correctly (check `PTX_LOAD` messages)

2. **Verify Physics Parameters**
   - Check `PHYSICS_PARAMS` logs for actual values
   - Ensure `time_step` is ≤0.15 and `damping` is ≥0.9
   - If values are wrong, check `data/settings.yaml` overrides

3. **Check for Constraints**
   - Constraints can trigger advanced mode with different physics
   - Look for `MSG_HANDLER: UpdateConstraints` messages
   - Verify constraint forces aren't too strong

### Issue: Advanced Kernel Not Loading

1. **Check PTX File Locations**
   ```bash
   # In container
   ls -la /app/src/utils/ptx/
   ls -la /app/src/utils/*.ptx
   ```

2. **Verify Compilation**
   ```bash
   # Recompile PTX files
   cd /app
   ./scripts/precompile-ptx.sh
   ```

3. **Check CUDA Compatibility**
   - Ensure PTX was compiled for correct compute capability
   - Default is SM_86 (RTX 30 series/A6000)

### Issue: Controls Not Affecting Physics

1. **Check Message Handlers**
   - Look for `MSG_HANDLER` logs when toggling controls
   - Verify state changes are logged

2. **Verify WebSocket Messages**
   - Check browser console for sent messages
   - Ensure control frames contain correct parameters

### Performance Monitoring

Monitor key metrics:
- **Iteration count**: Should increase steadily
- **Frame time**: Target <16.67ms for 60 FPS
- **Constraint count**: More constraints = higher computation
- **Kernel mode changes**: Frequent changes indicate instability

## Configuration Reference

### Settings Hierarchy
1. Default values in code
2. `data/settings.yaml` overrides
3. Runtime updates via API/WebSocket

### Key Configuration Files
- `src/models/simulation_params.rs` - Default physics parameters
- `src/models/constraints.rs` - Advanced physics parameters
- `data/settings.yaml` - Runtime configuration overrides
- `Dockerfile.dev` - PTX file deployment paths

## Testing Procedure

1. **Start with logging enabled**:
   ```bash
   RUST_LOG=debug docker-compose up dev
   ```

2. **Load a small graph** (100-500 nodes)
   - Should use Legacy kernel
   - Should settle within 5-10 seconds

3. **Test constraint toggle**:
   - Enable "Analytics" in UI
   - Check logs for mode switch
   - Verify stability is maintained

4. **Load larger graph** (1000+ nodes)
   - Should switch to Advanced kernel if available
   - Monitor warmup period (first 100 iterations)

5. **Collect metrics**:
   - Run for 60-120 seconds
   - Save logs for analysis
   - Compare before/after parameter changes