# Auto-Pause Functionality

The auto-pause functionality automatically pauses the physics simulation when the graph reaches equilibrium but resumes on user interaction. This provides an interactive experience where the simulation settles into a stable state but remains responsive to user actions.

## Features

- **Equilibrium Detection**: Monitors average node velocity and total kinetic energy
- **Stability Validation**: Requires stability over multiple frames before pausing
- **Interactive Resume**: Automatically resumes physics on node interaction
- **Configurable Thresholds**: All parameters are configurable via settings
- **Real-time Notifications**: Broadcasts pause/resume events to connected clients

## Configuration

Auto-pause settings are configured in `data/settings.yaml` under the physics section:

```yaml
physics:
  autoPause:
    enabled: true                          # Enable/disable auto-pause
    equilibriumVelocityThreshold: 0.1      # Average velocity threshold
    equilibriumCheckFrames: 30             # Stability check duration (frames)
    equilibriumEnergyThreshold: 0.01       # Total kinetic energy threshold
    pauseOnEquilibrium: true               # Actually pause when detected
    resumeOnInteraction: true              # Resume on user interaction
```

## How It Works

### 1. Equilibrium Detection

The system continuously monitors the physics simulation and calculates:

- **Average Velocity**: Derived from total kinetic energy: `v = sqrt(2 * KE / mass)`
- **Average Kinetic Energy**: Total kinetic energy divided by number of nodes
- **Stability Counter**: Tracks consecutive frames where equilibrium conditions are met

### 2. Pause Conditions

Physics is paused when ALL conditions are met:

1. `auto_pause_config.enabled == true`
2. Average velocity < `equilibrium_velocity_threshold`
3. Average kinetic energy < `equilibrium_energy_threshold`  
4. Stability maintained for >= `equilibrium_check_frames`
5. `pause_on_equilibrium == true`

### 3. Resume Triggers

Physics automatically resumes when:

- User drags a node (`NodeInteractionType::Dragged`)
- User selects a node (`NodeInteractionType::Selected`) - if configured
- Manual resume command (`ForceResumePhysics` message)

### 4. State Management

- **Pause State**: `simulation_params.is_physics_paused` (separate from `enabled`)
- **Stability Counter**: `equilibrium_stability_counter` tracks frame count
- **Reset Logic**: Counter resets to 0 when system leaves equilibrium or resumes

## Implementation Details

### Core Components

1. **AutoPauseConfig**: Configuration structure in `config/mod.rs`
2. **Equilibrium Detection**: `check_and_handle_equilibrium()` in `graph_actor.rs`
3. **Message Handlers**: Handle pause/resume and interaction messages
4. **Physics Loop**: Respects pause state in `run_advanced_gpu_step()`

### Message Types

- `PhysicsPauseMessage`: Manual pause/resume control
- `NodeInteractionMessage`: User interaction events 
- `ForceResumePhysics`: Force resume regardless of state
- `GetEquilibriumStatus`: Query current pause state

### Integration Points

1. **Physics Simulation**: Checks pause state before GPU computation
2. **WebSocket Events**: Broadcasts pause/resume notifications to clients
3. **Node Interactions**: Frontend sends interaction messages
4. **Settings Loading**: Auto-pause config loaded from YAML

## Usage Examples

### Manual Control

```rust
// Pause physics
let pause_msg = PhysicsPauseMessage {
    pause: true,
    reason: "Manual pause".to_string(),
};
graph_actor.do_send(pause_msg);

// Resume physics
let resume_msg = ForceResumePhysics {
    reason: "Manual resume".to_string(),
};
graph_actor.do_send(resume_msg);
```

### Node Interaction

```rust
// Node dragged - will resume physics if paused
let interaction = NodeInteractionMessage {
    node_id: 123,
    interaction_type: NodeInteractionType::Dragged,
    position: Some(Vec3::new(10.0, 20.0, 30.0)),
};
graph_actor.do_send(interaction);
```

### Status Check

```rust
// Check if physics is paused
let status_msg = GetEquilibriumStatus;
let is_paused = graph_actor.send(status_msg).await?;
```

## Frontend Integration

The frontend should:

1. **Listen for Notifications**: Handle `physics_paused` and `physics_resumed` WebSocket messages
2. **Send Interactions**: Send `NodeInteractionMessage` when user interacts with nodes
3. **Visual Indicators**: Show pause state in UI (e.g., pause icon, status indicator)
4. **Manual Controls**: Provide pause/resume buttons if desired

### WebSocket Message Format

```json
{
  "type": "physics_paused",
  "reason": "Equilibrium reached (vel: 0.0832, energy: 0.0087)"
}

{
  "type": "physics_resumed", 
  "reason": "Node 42 dragged"
}
```

## Performance Considerations

- **Minimal Overhead**: Equilibrium check adds ~0.1ms per frame
- **Early Exit**: Disabled when `auto_pause_config.enabled = false`
- **Efficient Calculation**: Reuses existing kinetic energy computation
- **Trace Logging**: Detailed logging only when debug enabled

## Configuration Guidelines

### Conservative Settings (Slower to Pause)
```yaml
equilibriumVelocityThreshold: 0.05    # Very low velocity required
equilibriumCheckFrames: 60            # Must be stable for 60 frames
equilibriumEnergyThreshold: 0.005     # Very low energy required
```

### Aggressive Settings (Faster to Pause)
```yaml
equilibriumVelocityThreshold: 0.2     # Higher velocity tolerance
equilibriumCheckFrames: 15            # Shorter stability requirement
equilibriumEnergyThreshold: 0.02      # Higher energy tolerance
```

### Interactive Settings (Quick Resume)
```yaml
resumeOnInteraction: true             # Resume on any interaction
pauseOnEquilibrium: true              # Auto-pause enabled
```

## Troubleshooting

### Physics Never Pauses
- Check `enabled: true` in config
- Verify thresholds aren't too strict
- Check logs for equilibrium detection values
- Ensure `pauseOnEquilibrium: true`

### Physics Pauses Too Quickly
- Increase `equilibriumCheckFrames`
- Lower thresholds (more strict)
- Check for premature equilibrium detection

### Resume Not Working
- Verify `resumeOnInteraction: true`
- Check interaction message handling
- Ensure messages reach graph_actor

### Configuration Not Loading
- Verify YAML syntax is correct
- Check settings deserialization
- Confirm config struct updates

## Testing

Run the auto-pause functionality tests:

```bash
cargo test auto_pause_functionality_test
```

The test suite covers:
- Configuration validation
- Equilibrium detection logic
- Message handling
- State management
- Edge cases and error conditions