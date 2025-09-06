# VisionFlow Configuration Quick Reference

*[Configuration](../index.md)*

## Essential Environment Variables

```bash
# Core Settings
AGENT_CONTROL_URL=tcp://multi-agent-container:9500
RUST_LOG=info
DATABASE_URL=postgres://user:pass@localhost/visionflow

# GPU Configuration
ENABLE_GPU_PHYSICS=true
NVIDIA_GPU_UUID=auto  # Auto-detect first GPU

# AI Services (Required for AI features)
RAGFLOW_API_KEY=your-key-here
PERPLEXITY_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here

# Feature Access
POWER_USER_KEY_1=secret-key-1
POWER_USER_KEY_2=secret-key-2
```

## Quick Configuration Tasks

### Enable VisionFlow multi-agent Visualisation
```yaml
# In data/settings.yaml
features:
  enabled_features:
    - visionflow_multi-agent
    - graph_visualization
```

### Configure GPU Physics
```bash
# Auto-detect GPU
export NVIDIA_GPU_UUID=auto
export ENABLE_GPU_PHYSICS=true

# Or disable GPU
export ENABLE_GPU_PHYSICS=false
```

### Set Up AI Services
```yaml
# In data/settings.yaml
ragflow:
  api_key: "${RAGFLOW_API_KEY}"
  base_url: "https://api.ragflow.com"

perplexity:
  api_key: "${PERPLEXITY_API_KEY}"
  model: "mixtral-8x7b-instruct"
```

## File Locations

| Configuration Type | File Path | Purpose |
|-------------------|-----------|---------|
| Main Settings | `/app/data/settings.yaml` | Primary configuration |
| User Settings | `/app/user_settings/<pubkey>.yaml` | Per-user preferences |
| Environment | `.env` (project root) | Secrets and overrides |
| Docker Config | `docker-compose.yml` | Container settings |

## Settings Priority (Highest to Lowest)

1. **Environment Variables** - Override everything
2. **Command-line args** - Runtime overrides
3. **User settings** - Personal preferences
4. **settings.yaml** - Main configuration
5. **Defaults** - Built-in values

## Common Commands

```bash
# Check current configuration
curl http://localhost:8080/api/settings

# Update a setting via API
curl -X POST http://localhost:8080/api/settings \
  -H "Content-Type: application/json" \
  -d '{"path": "gpu.enabled", "value": true}'

# Verify GPU detection
nvidia-smi --query-gpu=uuid --format=csv,noheader

# Test MCP connection
nc -zv multi-agent-container 9500
```

## Debugging Configuration

```bash
# Enable debug logging
export RUST_LOG=debug
export CONFIG_DEBUG=true

# Check loaded configuration
grep "Loaded config" logs/server.log

# Validate YAML syntax
yamllint data/settings.yaml
```

## Docker Network Setup

```bash
# Create network (if not exists)
docker network create mcp-visionflow-net

# Verify containers on network
docker network inspect mcp-visionflow-net
```



## See Also

- [Configuration Guide](../getting-started/configuration.md)
- [Getting Started with VisionFlow](../getting-started/index.md)
- [Guides](../guides/README.md)
- [Installation Guide](../getting-started/installation.md)
- [Quick Start Guide](../getting-started/quickstart.md)
- [VisionFlow Quick Start Guide](../guides/quick-start.md)
- [VisionFlow Settings System Guide](../guides/settings-guide.md)

## Related Topics

- [Configuration Architecture](../server/config.md)
- [Configuration Guide](../getting-started/configuration.md)
- [Developer Configuration System](../DEV_CONFIG.md)
- [Modern Settings API - Path-Based Architecture](../MODERN_SETTINGS_API.md)
- [Production Configuration Guide](../configuration/index.md)
- [Settings API Reference](../api/rest/settings.md)
- [Settings Panel](../client/settings-panel.md)
- [Settings Performance Optimisation Report](../SETTINGS_PERFORMANCE_OPTIMIZATION.md)
- [Settings Sync Integration Tests](../testing/SETTINGS_SYNC_INTEGRATION_TESTS.md)
- [VisionFlow Configuration Guide](../CONFIGURATION.md)
- [VisionFlow Settings System Guide](../guides/settings-guide.md)
