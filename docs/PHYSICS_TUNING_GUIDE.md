# Physics Tuning Guide for VisionFlow

## Problem Symptoms & Solutions

### 1. Nodes Collapsing to Origin
**Symptoms**: All nodes converge to (0,0,0)
**Causes**: 
- Repulsion too weak
- No minimum distance enforcement
- Uninitialized positions

**Solution**:
```cuda
const float MIN_DISTANCE = 0.15f;
if (dist < MIN_DISTANCE) {
    // Generate unique separation direction
    float angle = (float)(idx - j) * 0.618034f; // Golden ratio
    push_dir = make_vec3(cosf(angle), sinf(angle), 0.1f * (idx - j));
}
```

### 2. Nodes Exploding Outward
**Symptoms**: Nodes fly off to infinity
**Causes**:
- Forces too strong
- Timestep too large
- Insufficient damping

**Solution**:
```rust
SimParams {
    spring_k: 0.005,      // Very gentle (was 0.1)
    repel_k: 50.0,        // Moderate (was 1000.0)
    damping: 0.9,         // High damping
    dt: 0.01,             // Small timestep
    max_force: 2.0,       // Low cap (was 100.0)
}
```

### 3. Oscillation/Jittering
**Symptoms**: Nodes vibrate in place
**Causes**:
- Spring forces too strong
- Damping too low
- No progressive warmup

**Solution**:
```cuda
// Quadratic warmup over 200 iterations
if (params.iteration < 200) {
    float warmup = params.iteration / 200.0f;
    force = vec3_scale(force, warmup * warmup);
}
```

## Recommended Parameter Ranges

### For Small Graphs (< 100 nodes)
```rust
SimParams {
    spring_k: 0.01,
    repel_k: 30.0,
    damping: 0.85,
    dt: 0.02,
    max_velocity: 2.0,
    max_force: 5.0,
    viewport_bounds: 100.0,
}
```

### For Medium Graphs (100-1000 nodes)
```rust
SimParams {
    spring_k: 0.005,
    repel_k: 50.0,
    damping: 0.9,
    dt: 0.01,
    max_velocity: 1.0,
    max_force: 2.0,
    viewport_bounds: 200.0,
}
```

### For Large Graphs (1000+ nodes)
```rust
SimParams {
    spring_k: 0.002,
    repel_k: 100.0,
    damping: 0.95,
    dt: 0.005,
    max_velocity: 0.5,
    max_force: 1.0,
    viewport_bounds: 500.0,
}
```

## Key Relationships

### Force Balance
```
Equilibrium Distance = sqrt(repel_k / spring_k)
```
- If nodes too close: Increase repel_k or decrease spring_k
- If nodes too far: Decrease repel_k or increase spring_k

### Stability Formula
```
Stability = damping / (dt * max_force)
```
- Higher value = more stable but slower convergence
- Lower value = faster but may explode

### Natural Edge Length
```cuda
float natural_length = 10.0f; // Ideal spacing
```
- Too small: Nodes cluster
- Too large: Graph spreads out

## Debugging Tips

### 1. Check Initial Positions
```rust
// Log first few node positions
for i in 0..5.min(positions.len()) {
    println!("Node {}: ({:.2}, {:.2}, {:.2})", 
             i, positions[i].0, positions[i].1, positions[i].2);
}
```

### 2. Monitor Force Magnitudes
```cuda
// Add debug output in kernel
if (idx == 0 && params.iteration % 100 == 0) {
    printf("Iter %d: Force magnitude = %.4f\n", 
           params.iteration, vec3_length(force));
}
```

### 3. Track Energy
```rust
// Calculate total kinetic energy
let energy: f32 = velocities.iter()
    .map(|v| v.0*v.0 + v.1*v.1 + v.2*v.2)
    .sum();
println!("System energy: {:.4}", energy);
```

## Common Fixes

### "Nodes at origin" → Increase initial spread
```rust
let spread_radius = 30.0; // Was 20.0
```

### "Nodes exploding" → Reduce forces
```rust
max_force: 2.0,  // Was 10.0
repel_k: 50.0,   // Was 1000.0
```

### "Unstable simulation" → Increase damping
```rust
damping: 0.95,   // Was 0.85
dt: 0.005,       // Was 0.02
```

### "Nodes too clustered" → Adjust natural length
```cuda
float natural_length = 15.0f; // Was 5.0f
```

## Progressive Tuning Strategy

1. **Start Conservative**
   - Low forces, high damping
   - Small timestep
   - Let system stabilize

2. **Gradually Increase**
   - Slowly increase spring_k
   - Reduce damping slightly
   - Monitor for instability

3. **Fine-tune**
   - Adjust natural_length for spacing
   - Tweak repel_k for separation
   - Balance viewport_bounds

4. **Optimize**
   - Increase timestep if stable
   - Reduce iterations needed
   - Profile performance

## Testing Commands

```bash
# Test with different node counts
cargo run --example test_physics -- --nodes 50
cargo run --example test_physics -- --nodes 500
cargo run --example test_physics -- --nodes 5000

# Test different modes
cargo run --example test_physics -- --mode basic
cargo run --example test_physics -- --mode dual
cargo run --example test_physics -- --mode constraints
```

## Expected Behavior

### Good Simulation
- Nodes spread evenly
- Edges maintain consistent length
- System reaches equilibrium in ~500 iterations
- No oscillation after settling

### Warning Signs
- Nodes at (0,0,0): Check initialization
- Nodes at viewport edge: Forces too strong
- Never settling: Damping too low
- Collapse after initial spread: Spring forces dominating