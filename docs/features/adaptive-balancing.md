# Adaptive Balancing

## Overview

Adaptive Balancing is an intelligent physics auto-tuning system that automatically adjusts force-directed graph parameters to maintain stable, aesthetically pleasing visualisations. The system continuously monitors graph behaviour and makes real-time adjustments to prevent common issues like node clustering, boundary bouncing, and explosive spreading.

## How It Works

```mermaid
flowchart TD
    A[Monitor Graph State] --> B{Detect Instability}
    B -->|Bouncing| C[Reduce Forces & Increase Damping]
    B -->|Spreading| D[Scale Forces & Enable Boundaries]
    B -->|Clustering| E[Increase Repulsion]
    B -->|Stable| F[Maintain Current Settings]
    
    C --> G[Smooth Parameter Transition]
    D --> G
    E --> G
    F --> H[Check Stability Metrics]
    
    G --> I[Apply to GPU Physics]
    I --> J[Update Positions]
    J --> A
    
    H -->|Stable for 3s| K[Notify Settings System]
    K --> L[Update UI Sliders]
```

## Core Components

### 1. Stability Detection

The system uses multiple metrics to assess graph stability:

- **Position Variance**: Tracks changes in maximum node distances over time
- **Kinetic Energy**: Monitors total system energy to detect true equilibrium
- **Boundary Analysis**: Counts nodes at boundary positions to detect bouncing
- **Oscillation Detection**: Identifies repetitive movement patterns

### 2. Parameter Adjustment Strategies

#### Bouncing Nodes
When nodes bounce at boundaries:
```rust
// Aggressive stabilisation
repel_k *= 0.5      // Reduce repulsion
damping = 0.99      // Maximum damping
max_velocity = 0.1  // Very low velocity
enable_bounds = true
viewport_bounds = 100.0
```

#### Spreading Nodes
When nodes spread too far:
```rust
// Scale based on spread distance
scale_factor = (100.0 / max_distance).clamp(0.3, 0.9)
repel_k *= scale_factor
damping += (1.0 - scale_factor) * 0.05
max_velocity *= (0.5 + scale_factor * 0.5)
```

#### Clustered Nodes
When nodes are too tightly packed:
```rust
// Gentle expansion
repel_k *= 1.2  // Increase repulsion slightly
```

### 3. Smooth Transitions

Parameters transition gradually to prevent destabilisation:

```rust
// Exponential smoothing with 10-20% per frame
current_param = current_param * (1.0 - rate) + target_param * rate
```

## Configuration

All thresholds are configurable in `settings.yaml`:

```yaml
physics:
  auto_balance: true
  auto_balance_interval_ms: 500
  auto_balance_config:
    # Stability detection
    stability_variance_threshold: 100.0
    stability_frame_count: 180
    kinetic_energy_threshold: 0.01
    kinetic_energy_variance_threshold: 0.001
    
    # Boundary detection
    bouncing_node_percentage: 0.33
    boundary_min_distance: 90.0
    boundary_max_distance: 110.0
    
    # Distance thresholds
    clustering_distance_threshold: 20.0
    spreading_distance_threshold: 500.0
    extreme_distance_threshold: 1000.0
    
    # Oscillation detection
    oscillation_detection_frames: 10
    oscillation_change_threshold: 5.0
    min_oscillation_changes: 5
```

## User Interface

### Control Centre Integration

The Adaptive Balancing toggle is located in the Physics Settings section:

```
Physics Settings
├── Physics Enabled        [Toggle]
├── ⚖️ Adaptive Balancing  [Toggle] ← Main control
├── Damping                [Slider]
├── Spring Strength        [Slider]
└── ...
```

### Visual Indicators

1. **Active Tuning Animation**: When actively adjusting parameters, an animated ⚖️ icon appears in the Control Centre header with pulsing dots

2. **Toast Notifications**: Real-time notifications appear for significant adjustments:
   - 🔵 Info: "Adaptive Balancing: Adjusting forces to contain spreading"
   - 🟡 Warning: "Adaptive Balancing: Stabilising bouncing nodes"
   - 🟢 Success: "Adaptive Balancing: Stable configuration found!"

## API Endpoints

### Get Auto-Balance Notifications

```http
GET /api/graph/auto-balance-notifications?since=1234567890
```

Response:
```json
{
  "success": true,
  "notifications": [
    {
      "message": "Adaptive Balancing: Stable configuration found!",
      "timestamp": 1234567890123,
      "severity": "success"
    }
  ]
}
```

## Performance Impact

- **CPU Overhead**: Minimal (~1-2% for metric calculations)
- **Memory Usage**: ~10KB for history buffers
- **Network Traffic**: 200 bytes/second during active tuning (REST polling)
- **GPU Impact**: None (calculations done on CPU)

## Best Practices

### When to Enable

✅ **Recommended for:**
- Initial graph exploration
- Unknown or varying node counts
- Dynamic graphs with frequent updates
- Demonstration environments

❌ **Not recommended for:**
- Fixed, pre-tuned layouts
- Performance-critical scenarios
- Graphs with custom constraints

### Manual Override

While Adaptive Balancing is active, manual slider adjustments are still possible but may be overridden. To make permanent manual changes:

1. Disable Adaptive Balancing
2. Adjust parameters manually
3. Save settings if authenticated

## Technical Implementation

### Backend Architecture

```mermaid
classDiagram
    class GraphServiceActor {
        -auto_balance_history: Vec~f32~
        -kinetic_energy_history: Vec~f32~
        -stable_count: u32
        -target_params: SimulationParams
        -param_transition_rate: f32
        +update_node_positions()
        +smooth_transition_params()
        +send_auto_balance_notification()
    }
    
    class AutoBalanceConfig {
        +stability_variance_threshold: f32
        +kinetic_energy_threshold: f32
        +bouncing_node_percentage: f32
        +boundary_min_distance: f32
        +boundary_max_distance: f32
    }
    
    class SimulationParams {
        +repel_k: f32
        +damping: f32
        +max_velocity: f32
        +spring_k: f32
        +enable_bounds: bool
        +auto_balance: bool
        +auto_balance_config: AutoBalanceConfig
    }
    
    GraphServiceActor --> SimulationParams
    SimulationParams --> AutoBalanceConfig
```

### Frontend Integration

```typescript
// Polling for notifications
const useAutoBalanceNotifications = () => {
  useEffect(() => {
    const interval = setInterval(async () => {
      const response = await fetch('/api/graph/auto-balance-notifications');
      const data = await response.json();
      
      data.notifications.forEach(notification => {
        toast[notification.severity](notification.message);
      });
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
};
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Graph still bouncing | Thresholds too high | Reduce `bouncing_node_percentage` in config |
| Parameters changing too rapidly | Transition rate too high | Increase `param_transition_rate` for smoother changes |
| Not detecting instability | Variance threshold too high | Lower `stability_variance_threshold` |
| False positive detections | Thresholds too sensitive | Increase threshold values |

### Debug Logging

Enable debug logging to monitor auto-balance decisions:

```yaml
system:
  debug:
    enabled: true
    enablePhysicsDebug: true
```

Look for log entries with `[AUTO-BALANCE]` prefix:
```
[AUTO-BALANCE] Stats - max: 156.3, avg: 45.2, KE: 0.023, boundary: 12/45
[AUTO-BALANCE] Bouncing detected! Boundary nodes: 15/45
[AUTO-BALANCE] Stabilisation applied - repel_k: 0.250, damping: 0.990
```

## Future Enhancements

- **Machine Learning**: Train models on successful stabilisation patterns
- **Presets**: Save and load successful configurations for similar graphs
- **Predictive Adjustment**: Anticipate instability before it occurs
- **Multi-Graph Support**: Independent auto-balance for multiple graphs
- **Custom Strategies**: User-defined adjustment rules and thresholds

---

[← Back to Features](index.md) | [API Reference →](../api/rest/graph.md)