# Knowledge Graph Deadlock Recovery System

## Problem Overview

The knowledge graph visualization system can enter a deadlock state where all nodes become stuck at the boundary positions (typically around 980 units from center). This creates a perfect symmetry that prevents any movement, despite the physics simulation running.

### Symptoms of Deadlock
- All 177 nodes positioned exactly at boundary distance
- Average kinetic energy near zero (< 0.001)
- No visual movement despite physics enabled
- Auto-balance system unable to recover

## Solution Architecture

### 1. Aggressive Recovery Parameters

When deadlock is detected, the system applies much stronger forces:

| Parameter | Normal | Deadlock Recovery | Reason |
|-----------|---------|-------------------|---------|
| `repel_k` | 1.0-13.0 | 5.0-10.0 (8.0) | Strong repulsion to overcome boundary lock |
| `damping` | 0.85-0.95 | 0.5-0.6 (0.55) | Low damping allows movement |
| `max_velocity` | 1.0-5.0 | 8.0 | Higher velocity limits |
| `viewport_bounds` | 1000.0 | 1500.0 | Expanded boundary temporarily |
| `transition_rate` | 0.3 | 0.5 | Faster parameter application |

### 2. Symmetry Breaking Mechanism

The core innovation is **random perturbation** applied when deadlock is detected:

```rust
fn apply_deadlock_perturbation(&mut self) {
    let perturbation_strength = 2.5;
    
    for (node_id, node) in &mut self.node_map {
        // Random velocity injection
        let random_x = (rng.gen::<f32>() - 0.5) * perturbation_strength;
        let random_y = (rng.gen::<f32>() - 0.5) * perturbation_strength;
        node.vx += random_x;
        node.vy += random_y;
        
        // Small position offset to break boundary alignment
        let pos_offset = 0.5;
        node.x += (rng.gen::<f32>() - 0.5) * pos_offset;
        node.y += (rng.gen::<f32>() - 0.5) * pos_offset;
    }
}
```

### 3. Enhanced Detection

Improved deadlock detection with more sensitive thresholds:

- **Kinetic Energy Threshold**: Reduced from 0.0001 to 0.001 for faster detection
- **Comprehensive Logging**: Added detailed logging for boundary node counts and energy levels
- **State Monitoring**: Tracks recovery progress in real-time

## Implementation Details

### Detection Logic

```rust
let is_deadlocked = boundary_nodes == self.node_map.len() && avg_kinetic_energy < 0.001;

info!("[DEADLOCK-CHECK] Boundary nodes: {}/{}, Kinetic energy: {:.6}, Deadlocked: {}", 
      boundary_nodes, self.node_map.len(), avg_kinetic_energy, is_deadlocked);
```

### Recovery Sequence

1. **Detection**: All nodes at boundary with minimal kinetic energy
2. **Parameter Application**: Apply aggressive force parameters
3. **Symmetry Breaking**: Apply random perturbation to all nodes
4. **Monitoring**: Track recovery progress with enhanced logging
5. **Transition**: Gradually return to normal parameters once movement resumes

### Configuration Updates

Added to `settings.yaml`:

```yaml
auto_balance_config:
  # Deadlock recovery parameters
  deadlock_kinetic_threshold: 0.001
  recovery_repel_k_min: 5.0
  recovery_repel_k_max: 10.0
  recovery_damping_min: 0.5
  recovery_damping_max: 0.6
  recovery_max_velocity: 8.0
  recovery_transition_rate: 0.5
  perturbation_strength: 2.5
```

## Testing Strategy

### Test Scenarios

1. **Complete Deadlock**: All 177 nodes at boundary with zero velocity
2. **Parameter Validation**: Verify recovery parameters are within safe ranges
3. **Symmetry Breaking**: Confirm perturbation breaks perfect symmetry
4. **Recovery Effectiveness**: Validate nodes can escape boundary constraints

### Key Test Cases

- `test_complete_deadlock_detection()`: Verifies deadlock detection logic
- `test_aggressive_recovery_parameters()`: Validates parameter ranges
- `test_symmetry_breaking_perturbation()`: Confirms symmetry breaking works
- `test_recovery_strength_breaks_boundary_lock()`: Tests escape velocity calculations

## Performance Considerations

### Safety Mechanisms

1. **Parameter Clamping**: All parameters bounded to prevent explosion
2. **Gradual Transition**: Smooth parameter transitions to avoid jarring movements
3. **Logging**: Comprehensive logging for debugging and monitoring
4. **Fallback**: Normal bouncing logic remains for non-deadlock scenarios

### Monitoring

The system provides detailed logging:

```
[DEADLOCK-CHECK] Boundary nodes: 177/177, Kinetic energy: 0.000001, Deadlocked: true
[DEADLOCK-RECOVERY] Applying random perturbation to 177 nodes to break symmetry  
[DEADLOCK-RECOVERY] Applied AGGRESSIVE recovery: repel_k=8.0, damping=0.55, max_vel=8.0, bounds=1500
```

## Future Improvements

1. **Adaptive Perturbation**: Scale perturbation strength based on deadlock severity
2. **Predictive Detection**: Detect approaching deadlock before it occurs
3. **Machine Learning**: Learn optimal recovery parameters from successful recoveries
4. **Visualization**: Add visual indicators for deadlock states and recovery progress

## Summary

This enhanced deadlock recovery system addresses the core issue of symmetrical boundary locks by combining:

- **Aggressive Forces**: Strong enough to overcome boundary constraints
- **Symmetry Breaking**: Random perturbation to break perfect symmetry
- **Fast Response**: Rapid detection and recovery transition
- **Safe Operation**: Bounded parameters with comprehensive monitoring

The system transforms a complete deadlock scenario into a recoverable state, ensuring the knowledge graph remains interactive and useful even under extreme conditions.