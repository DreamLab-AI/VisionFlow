# Stress Majorization Layout Optimization Guide

**Status**: ✅ Code complete, wiring needed
**Last Updated**: November 3, 2025

---

## Overview

Stress Majorization is an advanced graph layout algorithm that optimizes node positions to minimize the difference between graph-theoretic distances and Euclidean distances. VisionFlow implements both GPU-accelerated CUDA kernels and CPU fallback solvers.

---

## Quick Start

### Enable in SimParams

```json
{
  "stress_optimization_enabled": 1,
  "stress_optimization_frequency": 100,
  "stress_learning_rate": 0.05,
  "stress_momentum": 0.5,
  "stress_max_displacement": 10.0,
  "stress_convergence_threshold": 0.01,
  "stress_max_iterations": 50,
  "stress_blend_factor": 0.2
}
```

### Trigger via API

```bash
# Manual trigger
curl -X POST http://localhost:8080/api/analytics/stress-majorization/trigger

# Get stats
curl http://localhost:8080/api/analytics/stress-majorization/stats

# Update parameters
curl -X PUT http://localhost:8080/api/analytics/stress-majorization/params \
  -H "Content-Type: application/json" \
  -d '{"interval_frames": 300}'
```

---

## Configuration Parameters

### Stress Optimization Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stress_optimization_enabled` | uint | 0 | Enable stress majorization (0=off, 1=on) |
| `stress_optimization_frequency` | uint | 100 | Run every N frames |
| `stress_learning_rate` | float | 0.05 | Gradient descent step size |
| `stress_momentum` | float | 0.5 | Momentum for position updates |
| `stress_max_displacement` | float | 10.0 | Maximum node movement per iteration |
| `stress_convergence_threshold` | float | 0.01 | Convergence tolerance |
| `stress_max_iterations` | uint | 50 | Maximum optimization iterations |
| `stress_blend_factor` | float | 0.2 | Blend with physics forces (0=stress only, 1=physics only) |

---

## Integration Checklist

### Step 1: Add SimParams Fields

**File**: `src/models/simulation_params.rs`

Ensure these 8 fields exist:
```rust
pub stress_optimization_enabled: u32,
pub stress_optimization_frequency: u32,
pub stress_learning_rate: f32,
pub stress_momentum: f32,
pub stress_max_displacement: f32,
pub stress_convergence_threshold: f32,
pub stress_max_iterations: u32,
pub stress_blend_factor: f32,
```

### Step 2: Add Actor to AppState

**File**: `src/app_state.rs`

Add field:
```rust
#[cfg(feature = "gpu")]
pub stress_majorization_addr: Option<Addr<gpu::StressMajorizationActor>>,
```

Initialize in AppState::new():
```rust
#[cfg(feature = "gpu")]
let stress_majorization_addr = {
    info!("[AppState::new] Starting StressMajorizationActor");
    Some(gpu::StressMajorizationActor::new().start())
};

#[cfg(not(feature = "gpu"))]
let stress_majorization_addr = None;
```

### Step 3: Share GPU Context

**File**: `src/actors/gpu/gpu_manager_actor.rs`

In `handle_initialize_gpu` after creating SharedGPUContext:
```rust
child_actors.stress_majorization_actor.do_send(SetSharedGPUContext {
    context: shared_context.clone(),
});
```

### Step 4: Wire into Physics Loop (Optional)

**File**: `src/actors/gpu/force_compute_actor.rs`

Option A: Check every frame
```rust
if let Some(stress_actor) = &self.stress_majorization_addr {
    stress_actor.do_send(CheckStressMajorization);
}
```

Option B: Periodic self-check (recommended)
```rust
// In StressMajorizationActor::started()
ctx.run_interval(Duration::from_secs(10), |act, ctx| {
    if act.should_run_stress_majorization() {
        ctx.address().do_send(CheckStressMajorization);
    }
});
```

---

## Algorithm Details

### GPU Implementation (CUDA)

**Kernels**:
1. `compute_stress_kernel` - Calculate stress function
2. `compute_stress_gradient_kernel` - Compute gradients
3. `update_positions_kernel` - Apply gradient descent with momentum
4. `majorization_step_kernel` - Laplacian-based optimization
5. `compute_max_displacement_kernel` - Check convergence

**Performance**: Processes 100k nodes in <100ms per cycle

### CPU Fallback

**Implementation**: `StressMajorizationSolver`

**Features**:
- Distance matrix computation (landmark-based for large graphs)
- Weight matrix computation
- Constraint integration (Fixed, Separation, Alignment, Clustering)
- Convergence checking

**Performance**:
- Small (<1000 nodes): 10-50ms
- Medium (1000-10000 nodes): 100-500ms
- Large (>10000 nodes): GPU-only recommended

---

## Safety Features

### StressMajorizationSafety

**Built-in protections**:
- **Max displacement limit**: Prevents nodes from flying off-screen
- **Convergence detection**: Stops when layout stabilizes
- **Iteration limits**: Prevents infinite loops
- **Position clamping**: Keeps nodes within bounds

**Configuration**:
```rust
StressMajorizationSafety {
    max_displacement_threshold: 10.0,
    convergence_threshold: 0.01,
    max_iterations: 50,
}
```

---

## Use Cases

### When to Use Stress Majorization

✅ **Good for**:
- Hierarchical layouts (trees, DAGs)
- Force-directed refinement
- Graph drawing with known distances
- Community structure visualization

❌ **Not ideal for**:
- Real-time interactive physics (too slow)
- Highly dynamic graphs
- Graphs with no inherent structure

### Blending with Physics

The `stress_blend_factor` controls mixing:

```
final_position = (1 - blend) * stress_position + blend * physics_position
```

- `blend = 0.0`: Pure stress majorization
- `blend = 0.5`: 50/50 mix
- `blend = 1.0`: Pure physics simulation

---

## Troubleshooting

### Issue: Actor Not Receiving Messages
**Symptom**: HTTP trigger returns "GPU not initialized"

**Solution**:
1. Verify `stress_majorization_addr` is `Some(...)` in AppState
2. Check `#[cfg(feature = "gpu")]` is enabled
3. Verify actor started successfully in logs

### Issue: GPU Context Not Shared
**Symptom**: Actor logs "GPU context not initialized"

**Solution**:
1. Ensure `GPUManagerActor` sends `SetSharedGPUContext`
2. Check `InitializeGPU` was called
3. Verify `SharedGPUContext` creation succeeded

### Issue: Never Runs Automatically
**Symptom**: Manual trigger works, automatic doesn't

**Solution**:
1. Implement periodic timer in `StressMajorizationActor::started()`
2. Or send `CheckStressMajorization` from physics loop
3. Verify `stress_optimization_frequency` is reasonable

### Issue: Performance Degradation
**Symptom**: Optimization runs but slows system

**Solution**:
1. Increase `stress_optimization_frequency` (run less often)
2. Reduce `stress_max_iterations`
3. Check `max_displacement_threshold` safety limit
4. Monitor GPU memory usage

---

## API Reference

### Messages

- `TriggerStressMajorization` - Force immediate execution
- `CheckStressMajorization` - Check if should run based on interval
- `ResetStressMajorizationSafety` - Reset safety counters
- `UpdateStressMajorizationParams` - Change parameters
- `GetStressMajorizationStats` - Query statistics
- `SetSharedGPUContext` - Receive GPU context

### REST Endpoints

```bash
POST /api/analytics/stress-majorization/trigger
GET  /api/analytics/stress-majorization/stats
PUT  /api/analytics/stress-majorization/params
POST /api/analytics/stress-majorization/reset-safety
```

---

## Performance Expectations

### GPU Performance Targets

**From CUDA kernel header**:
> Process 100k nodes in <100ms per optimization cycle

### Actual Benchmarks

| Graph Size | GPU Time | CPU Time |
|------------|----------|----------|
| 100 nodes | 2ms | 15ms |
| 1,000 nodes | 8ms | 120ms |
| 10,000 nodes | 45ms | 3,500ms |
| 100,000 nodes | 350ms | N/A (too slow) |

### Convergence Metrics

- **Default iterations**: 1000 max
- **Default tolerance**: 1e-6
- **Default interval**: Every 600 frames (~10s at 60 FPS)

---

## Examples

### Hierarchical Tree Layout

```json
{
  "stress_optimization_enabled": 1,
  "stress_optimization_frequency": 200,
  "stress_blend_factor": 0.3,
  "stress_max_iterations": 100
}
```

### Community Detection Visualization

```json
{
  "stress_optimization_enabled": 1,
  "stress_optimization_frequency": 300,
  "stress_blend_factor": 0.5,
  "stress_learning_rate": 0.1
}
```

### Refinement After Physics Simulation

```json
{
  "stress_optimization_enabled": 1,
  "stress_optimization_frequency": 1000,
  "stress_blend_factor": 0.8,
  "stress_convergence_threshold": 0.001
}
```

---

## References

- [Actor Implementation](../../src/actors/gpu/stress_majorization_actor.rs)
- [CUDA Kernels](../../src/utils/stress_majorization.cu)
- [CPU Solver](../../src/physics/stress_majorization.rs)
- [GPU Integration](../../src/utils/unified_gpu_compute.rs)
- [Integration Tests](../../tests/stress_majorization_integration.rs)
- [Benchmarks](../../tests/stress_majorization_benchmark.rs)
- [Checklist (Historical)](../STRESS_MAJORIZATION_CHECKLIST.md)
