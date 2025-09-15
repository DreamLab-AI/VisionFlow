# Graph Settling Issue Analysis & Fix

## Problem Description
The graph initially settles correctly, then jumps to randomized positions and stops without re-settling.

## Root Causes Identified

### 1. Position Broadcasting Issue
The graph is broadcasting positions but the settled state detection might be broken, causing:
- Initial settled state to be lost
- Physics simulation to restart with different parameters
- Positions to jump to new locations

### 2. Auto-Balance Triggering Incorrectly
The auto-balance system might be:
- Detecting false instability in settled graphs
- Applying corrections that destabilize the graph
- Not properly tracking the settled state

### 3. Z-Axis Position Issue
The z=-99.99 position indicates possible:
- Constraint violations pushing nodes to boundaries
- Numerical instability in physics calculations
- Incorrect initial position generation

## Immediate Fixes to Apply

### Fix 1: Disable Auto-Balance on Settled Graphs
In `/src/actors/graph_actor.rs`, around line 1240-1260, add a check:

```rust
// Skip auto-balance if graph is already settled
if self.stable_count > 30 {
    debug!("Graph is settled, skipping auto-balance checks");
    return;
}
```

### Fix 2: Add Position Validation
Before broadcasting positions, validate they're reasonable:

```rust
// Validate positions before broadcast
for node in self.node_map.values() {
    if node.data.position.z.abs() > 90.0 {
        warn!("Invalid z position detected: {}", node.data.position.z);
        // Reset to reasonable position
        node.data.position.z = node.data.position.z.clamp(-50.0, 50.0);
    }
}
```

### Fix 3: Prevent Position Jumps
Add hysteresis to prevent sudden position changes:

```rust
// Store previous positions
let prev_positions: HashMap<u32, Vec3Data> = self.node_map.iter()
    .map(|(id, node)| (*id, node.data.position.clone()))
    .collect();

// After physics update, check for jumps
for (id, node) in self.node_map.iter_mut() {
    if let Some(prev_pos) = prev_positions.get(&id) {
        let distance = ((node.data.position.x - prev_pos.x).powi(2) +
                       (node.data.position.y - prev_pos.y).powi(2) +
                       (node.data.position.z - prev_pos.z).powi(2)).sqrt();

        if distance > 100.0 {  // Threshold for jump detection
            warn!("Node {} jumped {} units, reverting", id, distance);
            node.data.position = prev_pos.clone();
        }
    }
}
```

## Configuration Adjustments

### In settings.yaml or via API:
```yaml
physics:
  auto_balance:
    enabled: false  # Temporarily disable until fixed
  stability:
    threshold: 0.01  # Make more sensitive
    min_stable_frames: 60  # Require more stability
```

## Testing Steps

1. Start server with logging enabled:
   ```bash
   RUST_LOG=debug cargo run
   ```

2. Connect a client and observe:
   - Initial graph load
   - Settling behavior
   - Any position jumps

3. Monitor for:
   - Z-axis values approaching -100
   - Sudden position changes > 50 units
   - Auto-balance triggering on settled graphs

## Long-term Solution

The fundamental issue is that the graph shouldn't be modifying positions after settling unless explicitly triggered by user interaction or data changes. We need to:

1. **Implement a "locked" state** for settled graphs
2. **Separate initial positioning from physics simulation**
3. **Add position history tracking** to detect and prevent jumps
4. **Improve the settled detection algorithm** to be more robust

## Immediate Workaround

If the issue persists, you can force the graph to stay in its initial state by commenting out the physics update in `update_node_positions`:

```rust
// Temporarily disable physics updates to debug
// self.send_update_to_gpu_actors(ctx, position_data, edge_data);
```

This will freeze the graph but help identify if the issue is in the physics simulation or elsewhere.