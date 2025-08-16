# 🔍 Physics Debug Logging Implementation

## Purpose
Added comprehensive logging to track physics parameter updates through the entire pipeline to diagnose explosion and bouncing issues in the knowledge graph.

## Changes Made

### 1. Settings Handler (`/workspace/ext/src/handlers/settings_handler.rs`)
Enhanced `propagate_physics_to_gpu()` with detailed logging:

```rust
[PHYSICS UPDATE] Propagating logseq physics to actors:
  - repulsion_strength: 2.000 (affects node spreading)
  - spring_strength: 0.005 (affects edge tension)
  - attraction_strength: 0.001 (affects clustering)
  - damping: 0.950 (affects settling, 1.0 = no movement)
  - time_step: 0.016 (simulation speed)
  - max_velocity: 2.000 (prevents explosions)
  - temperature: 0.010 (random motion)
  - gravity: 0.000 (directional force)
```

**Debug-gated additional parameters:**
- bounds_size, collision_radius, mass_scale, boundary_damping, update_threshold, iterations, enabled

### 2. Graph Service Actor (`/workspace/ext/src/actors/graph_actor.rs`)

#### UpdateSimulationParams Handler
Logs before/after parameter updates:
```rust
[GRAPH ACTOR] === UpdateSimulationParams RECEIVED ===
[GRAPH ACTOR] OLD physics values:
  - repulsion: 2 (was)
  - damping: 0.950 (was)
  - time_step: 0.016 (was)
  - spring_strength: 0.005 (was)
  - attraction_strength: 0.001 (was)
  - max_velocity: 2.000 (was)
  - enabled: true (was)

[GRAPH ACTOR] NEW physics values:
  - repulsion: 10 (new)
  - damping: 0.800 (new)
  ... (all parameters logged)
```

#### GPU Step Execution
Logs parameters used during actual simulation:
```rust
[GPU STEP] === Starting physics simulation step ===
[GPU STEP] Current physics parameters:
  - repulsion: 10 (node spreading force)
  - damping: 0.800 (velocity reduction, 1.0 = frozen)
  - time_step: 0.016 (simulation speed)
  - spring_strength: 0.010 (edge tension)
  - attraction_strength: 0.005 (clustering force)
  - max_velocity: 5.000 (explosion prevention)
  - enabled: true (is physics on?)
```

### 3. Debug Gating
All verbose logging is properly gated using `crate::utils::logging::is_debug_enabled()`:
- Checks `DEBUG_ENABLED` environment variable first
- Falls back to `settings.yaml` system.debug.enabled
- Critical physics values always logged at INFO level
- Detailed parameters only logged when debug is enabled

## Log Levels

| Level | Content | When |
|-------|---------|------|
| INFO | Critical physics parameters | Always |
| INFO | Update success/failure | Always |
| DEBUG | Additional parameters | When debug enabled |
| WARN | Missing actors/contexts | Always |
| ERROR | Update failures | Always |

## Diagnosing Physics Issues

### 1. Check Parameter Flow
```bash
tail -f /workspace/ext/logs/rust-error.log | grep "PHYSICS UPDATE"
```
Shows:
- What values are being sent from settings
- Conversion to SimulationParams
- Delivery to actors

### 2. Monitor GPU Steps
```bash
tail -f /workspace/ext/logs/rust-error.log | grep "GPU STEP"
```
Shows:
- Actual parameters used during simulation
- Whether physics is enabled
- Step-by-step execution

### 3. Track Actor Updates
```bash
tail -f /workspace/ext/logs/rust-error.log | grep "GRAPH ACTOR"
```
Shows:
- Old vs new parameter values
- Successful parameter updates

## Common Issues to Look For

### Explosion Issues
- **High repulsion** (> 100): Nodes fly apart
- **Low damping** (< 0.5): Velocities don't decay
- **High time_step** (> 0.1): Large position jumps
- **Low max_velocity** might not be enforced

### Bouncing Issues
- **High spring_strength** (> 0.1): Oscillations
- **Low damping** (< 0.8): Can't settle
- **High temperature** (> 0.1): Random jitter

### Frozen Graph
- **damping = 1.0**: No movement possible
- **enabled = false**: Physics turned off
- **time_step = 0**: No progression

## Enabling Debug Logging

### Method 1: Environment Variable
```bash
export DEBUG_ENABLED=true
export RUST_LOG=info,webxr=debug
```

### Method 2: Settings File
Edit `/workspace/ext/data/settings.yaml`:
```yaml
system:
  debug:
    enabled: true
```

## Deployment

1. **Build Server:**
```bash
cd /workspace/ext
cargo build --release
```

2. **Restart Docker Container:**
```bash
# Server needs restart to load new logging code
docker-compose restart webxr-server
```

3. **Monitor Logs:**
```bash
tail -f /workspace/ext/logs/rust-error.log | grep -E "PHYSICS|GPU|GRAPH"
```

## Expected Output After Deployment

When physics settings are changed in UI:
1. `[PHYSICS UPDATE]` logs show new values being propagated
2. `[GRAPH ACTOR]` logs confirm receipt and old→new transition
3. `[GPU STEP]` logs show values actually used in simulation

This comprehensive logging will help identify:
- Whether settings are reaching the simulation
- What values are actually being used
- Where the explosion/bouncing originates

## Status
✅ **Ready for deployment** - Code compiles successfully with enhanced logging