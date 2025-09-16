# Configuration Guide

Comprehensive configuration documentation for VisionFlow system settings, environment variables, and deployment options.

## Quick Reference

- [Environment Variables](environment.md) - Complete environment variable reference
- [System Settings](settings.md) - Application configuration options
- [GPU Configuration](gpu-config.md) - CUDA and GPU acceleration settings
- [Network Configuration](network.md) - WebSocket and API settings

## Configuration Files

### Main Configuration
- `config.toml` - Primary system configuration
- `.env` - Environment variables
- `docker-compose.yml` - Container orchestration

### Feature-Specific Configuration
- `gpu.toml` - GPU acceleration settings
- `agents.toml` - AI agent configuration
- `visualization.toml` - 3D rendering options

## Configuration Sections

### System Configuration
```toml
[system]
log_level = "info"
max_nodes = 100000
enable_gpu = true
```

### Database Configuration
```toml
[database]
url = "sqlite://data/visionflow.db"
max_connections = 100
```

### WebSocket Configuration
```toml
[websocket]
port = 3001
binary_protocol = true
compression = true
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VISIONFLOW_GPU_ENABLED` | Enable GPU acceleration | `true` |
| `VISIONFLOW_LOG_LEVEL` | Logging verbosity | `info` |
| `VISIONFLOW_WS_PORT` | WebSocket port | `3001` |
| `VISIONFLOW_MAX_NODES` | Maximum graph nodes | `100000` |

See [Environment Variables](environment.md) for complete reference.

## GPU Configuration

### CUDA Settings
```toml
[gpu]
device_id = 0
memory_pool_size = "2GB"
enable_unified_memory = true
```

### Performance Tuning
```toml
[physics]
iterations_per_frame = 10
force_multiplier = 1.0
damping_factor = 0.95
```

## Network Configuration

### WebSocket Settings
```toml
[websocket]
max_connections = 1000
heartbeat_interval = 30
binary_compression = true
```

### API Configuration
```toml
[api]
rate_limit = 1000
cors_origins = ["http://localhost:3000"]
request_timeout = 30
```

## Security Configuration

### Authentication
```toml
[auth]
provider = "nostr"
session_timeout = 3600
enable_2fa = false
```

### HTTPS/TLS
```toml
[tls]
cert_path = "/etc/ssl/certs/visionflow.crt"
key_path = "/etc/ssl/private/visionflow.key"
```

## Development vs Production

### Development Settings
```toml
[development]
hot_reload = true
debug_gpu = true
log_level = "debug"
```

### Production Settings
```toml
[production]
log_level = "warn"
enable_metrics = true
cache_static_files = true
```

## Validation and Testing

Use the built-in configuration validator:
```bash
visionflow config validate
```

Test configuration changes:
```bash
visionflow config test
```

## Troubleshooting

### Common Issues
- GPU not detected: Check CUDA installation
- WebSocket connection failed: Verify port and firewall
- High memory usage: Adjust node limits and GPU memory

### Configuration Debugging
```bash
visionflow config dump    # Show current configuration
visionflow config check   # Validate configuration
visionflow logs config    # Show configuration-related logs
```

## Related Documentation

- [Deployment Guide](../deployment/README.md)
- [GPU Compute Architecture](../architecture/gpu-compute.md)
- [Server Configuration](../server/config.md)
- [Getting Started](../getting-started/configuration.md)

---

[← Back to Documentation](../README.md)