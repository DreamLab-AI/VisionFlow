# CUDA Kernel Parameters Documentation

## Overview
The VisionFlow system now exposes CUDA kernel parameters through the REST API, allowing fine-grained control over GPU physics simulation behaviour.

## New Parameters

### Core Physics Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `restLength` | float | 50.0 | Natural spring rest length for force calculations |
| `repulsionCutoff` | float | 50.0 | Maximum distance for repulsion force calculations |
| `repulsionSofteningEpsilon` | float | 0.0001 | Prevents division by zero in force calculations |
| `centerGravityK` | float | 0.0 | Gravity strength towards centre (0 = disabled) |
| `gridCellSize` | float | 50.0 | Spatial grid resolution for neighbour searches |

### Warmup Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmupIterations` | integer | 100 | Number of warmup simulation steps |
| `coolingRate` | float | 0.001 | Rate of cooling during warmup phase |

### Boundary Behaviour Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `boundaryExtremeMultiplier` | float | 2.0 | Multiplier for extreme boundary detection |
| `boundaryExtremeForceMultiplier` | float | 10.0 | Force multiplier for extreme positions |
| `boundaryVelocityDamping` | float | 0.5 | Velocity reduction on boundary collision |

## API Endpoints

### GET /api/settings
Returns current settings including CUDA parameters in camelCase format.

### POST /api/settings
Updates settings with new CUDA parameter values. Accepts camelCase JSON.

Example request:
```json
{
  "visualisation": {
    "graphs": {
      "logseq": {
        "physics": {
          "restLength": 60.0,
          "repulsionCutoff": 75.0,
          "gridCellSize": 40.0,
          "coolingRate": 0.002
        }
      }
    }
  }
}
```

## Case Conversion
The API automatically handles conversion between:
- **Server-side**: snake_case (Rust convention)
- **Client-side**: camelCase (JavaScript/TypeScript convention)

## UI Integration
The Control Centre includes sliders for adjusting these parameters in real-time:
- Navigate to Settings → Physics → Advanced GPU
- Adjust parameters using the provided sliders
- Changes are applied immediately to the simulation

## Performance Considerations
- **gridCellSize**: Smaller values increase precision but require more computation
- **repulsionCutoff**: Larger values include more nodes in force calculations
- **warmupIterations**: Higher values provide better initial layout at the cost of startup time
- **coolingRate**: Lower values provide smoother transitions but slower convergence

## Validation
All parameters are validated on both client and server:
- Numeric ranges are enforced
- Invalid values are rejected with descriptive error messages
- Default values are used for missing parameters

## Integration with dev_config.toml
Server administrators can override defaults in `/ext/data/dev_config.toml`:
```toml
[cuda]
rest_length = 50.0
repulsion_cutoff = 50.0
repulsion_softening_epsilon = 0.0001
center_gravity_k = 0.0
grid_cell_size = 50.0
warmup_iterations = 100
cooling_rate = 0.001
```

## Monitoring
Use `cargo check` to validate the integration and ensure all parameters are properly synchronised across the codebase.