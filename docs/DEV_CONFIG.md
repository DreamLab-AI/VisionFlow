# Developer Configuration System

## Overview

The developer configuration system provides internal server-side settings that are not exposed to clients. This separates user-facing configuration (settings.yaml) from internal tuning parameters that developers need to adjust without recompiling.

## Files

- `/src/config/dev_config.rs` - Rust module defining configuration structures
- `/data/dev_config.toml` - Configuration file with current values
- `/data/settings.yaml` - User-facing settings (existing)

## Usage

### In Rust Code

```rust
use crate::config::dev_config;

// Get the full config
let config = dev_config::DevConfig::get();

// Use convenience functions for specific sections
let physics = dev_config::physics();
let cuda = dev_config::cuda();
let network = dev_config::network();
let rendering = dev_config::rendering();
let performance = dev_config::performance();
let debug = dev_config::debug();

// Examples:
let golden_ratio = dev_config::physics().golden_ratio;
let agent_colors = &dev_config::rendering().agent_colors;
let max_nodes = dev_config::cuda().max_nodes;
```

### Configuration Sections

#### Physics
Controls force calculations, boundary behaviour, node distribution, and clustering:
- `force_epsilon` - Prevents division by zero in force calculations
- `spring_length_multiplier` - Natural spring length calculation
- `boundary_extreme_multiplier` - When nodes are considered "extreme"
- `golden_ratio` - For initial node positioning
- `cross_graph_repulsion_scale` - Repulsion between different graphs

#### CUDA
GPU-specific parameters and safety limits:
- `warmup_iterations_default` - Initial settling phase
- `max_kernel_time_ms` - GPU kernel timeout
- `debug_output_throttle` - How often to output debug info
- `max_nodes` / `max_edges` - Memory limits

#### Network
Connection pooling, circuit breakers, and retry logic:
- `pool_max_idle_per_host` - Connection pool size
- `circuit_failure_threshold` - When to open circuit
- `retry_max_delay_ms` - Maximum retry delay
- `ws_frame_size` - WebSocket frame size

#### Rendering
Visual appearance and animations:
- `agent_colors` - Colour scheme for different agent types
- `agent_base_size` - Base size for agents
- `animation speeds` - Pulse, rotate, glow speeds
- `lod_distance_*` - Level of detail thresholds

#### Performance
Batching, caching, and resource management:
- `batch_size_nodes` - Node processing batch size
- `cache_ttl_secs` - Cache expiration
- `worker_threads` - Thread pool size
- `memory_warning_threshold_mb` - Memory alerts

#### Debug
Development and debugging flags:
- `enable_*_debug` - Debug output toggles
- `log_slow_operations_ms` - Performance monitoring
- `profile_sample_rate` - Profiling frequency

## Modifying Values

1. Edit `/data/dev_config.toml`
2. Restart the server (config is loaded once at startup)
3. No recompilation needed!

## Adding New Settings

1. Add field to appropriate struct in `dev_config.rs`
2. Add default value in `Default` implementation
3. Add value to `dev_config.toml`
4. Use in code via `dev_config::section().field`

## Best Practices

1. **Group related settings** - Keep settings organised by purpose
2. **Document units** - Always specify ms, secs, MB, etc.
3. **Provide sensible defaults** - System should work without config file
4. **Use static access** - Config is loaded once for performance
5. **Don't expose to clients** - These are internal tuning parameters

## Migration from Hardcoded Values

When you find hardcoded values in the code:

1. Identify the purpose and context
2. Add to appropriate section in dev_config
3. Replace hardcoded value with `dev_config::section().field`
4. Test with different values via config file

## Examples of Migrated Values

### Before:
```rust
const MIN_DISTANCE: f32 = 0.15;
const MAX_NODES: u32 = 1_000_000;
let colour = "#00FFFF"; // coordinator colour
```

### After:
```rust
let min_distance = dev_config::physics().min_distance;
let max_nodes = dev_config::cuda().max_nodes;
let colour = &dev_config::rendering().agent_colors.coordinator;
```

## Performance Considerations

The dev config is loaded once at startup and stored in a static `OnceLock`. This means:
- Zero runtime overhead for config access
- Changes require server restart
- Config is immutable after initialization

## Future Enhancements

Potential improvements:
- Hot reload capability for development
- Config validation on load
- Environment variable overrides
- Per-graph type configurations
- A/B testing different configurations

## See Also

- [Configuration Guide](getting-started/configuration.md)
- [Getting Started with VisionFlow](getting-started/index.md)
- [Guides](guides/README.md)
- [Installation Guide](getting-started/installation.md)
- [Quick Start Guide](getting-started/quickstart.md)
- [VisionFlow Quick Start Guide](guides/quick-start.md)
- [VisionFlow Settings System Guide](guides/settings-guide.md)
