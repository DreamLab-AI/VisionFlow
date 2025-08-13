# VisionFlow Configuration Guide

## Configuration Hierarchy

VisionFlow uses multiple configuration files with a clear precedence order. This document explains the role of each configuration file and how they interact.

## Configuration Files Overview

```
ext/
├── .env                     # Environment variables & secrets
├── .env_template            # Template for .env file
├── data/
│   └── settings.yaml        # Application settings (runtime configurable)
├── config.yml               # Cloudflare tunnel configuration
├── docker-compose.yml       # Docker service configuration with profiles
└── supervisord.*.conf       # Process management configuration
```

## File Roles and Precedence

### 1. Environment Variables (`.env`)
**Purpose:** Store secrets, API keys, and environment-specific configuration
**Precedence:** Highest - overrides all other configurations
**Scope:** Docker containers and application runtime

Key variables:
- `GITHUB_TOKEN` - Required for Logseq graph synchronisation
- `OPENAI_API_KEY` - AI service integration
- `PERPLEXITY_API_KEY` - Search and query service
- `CLOUDFLARE_TUNNEL_TOKEN` - Production tunnel authentication
- `CUDA_ARCH` - GPU architecture for compilation
- `MCP_TCP_PORT` - Claude Flow MCP port (9500)
- `RUST_LOG` - Logging level configuration

**Best Practice:** Never commit `.env` to version control. Use `.env_template` as reference.

### 2. Application Settings (`data/settings.yaml`)
**Purpose:** Runtime application configuration and defaults
**Precedence:** Medium - can be overridden by environment variables
**Scope:** Application behaviour and feature configuration

Structure:
```yaml
system:
  network:
    bind_address: "0.0.0.0"
    port: 4000
  websocket:
    min_update_rate: 10
    max_update_rate: 60

visualisation:
  graphs:
    logseq:      # Primary knowledge graph
      physics: ...
      nodes: ...
    visionflow:  # AI agent graph
      physics: ...
      nodes: ...
```

**Note:** Legacy top-level keys (`nodes`, `edges`, `physics`) under `visualisation` are deprecated. Use the `graphs` namespace.

### 3. Docker Compose Configuration (`docker-compose.yml`)
**Purpose:** Container orchestration and service definitions
**Precedence:** Defines how environment variables are passed to containers
**Scope:** Container runtime, networking, volumes

Profiles:
- `dev` - Development environment with hot reloading
- `production` / `prod` - Optimised production deployment

Usage:
```bash
# Development
docker-compose --profile dev up

# Production
docker-compose --profile production up
```

### 4. Cloudflare Configuration (`config.yml`)
**Purpose:** Tunnel routing for production deployment
**Precedence:** Independent - only used by Cloudflare tunnel
**Scope:** External access and routing

Example:
```yaml
tunnel: your-tunnel-id
credentials-file: /etc/cloudflared/credentials.json
ingress:
  - hostname: your-domain.com
    service: http://webxr:4000
  - service: http_status:404
```

### 5. Supervisor Configuration (`supervisord.*.conf`)
**Purpose:** Process management within containers
**Precedence:** Independent - manages process lifecycle
**Scope:** Container internal processes

Files:
- `supervisord.dev.conf` - Development processes (Vite, Rust watch)
- `supervisord.conf` - Production processes (Nginx, Rust server)

## Configuration Loading Order

1. **Container Start:**
   - Docker Compose reads `.env` file
   - Environment variables set in container

2. **Application Start:**
   - Rust backend loads `settings.yaml`
   - Environment variables override settings values
   - Configuration merged and validated

3. **Runtime:**
   - Settings can be modified via API (`/api/settings`)
   - Changes persist to `settings.yaml`
   - Some settings require restart

## Environment Variable Overrides

Environment variables can override `settings.yaml` values using this pattern:

```bash
# Override system.network.port
SYSTEM_NETWORK_PORT=5000

# Override visualisation.graphs.logseq.physics.repulsion_strength
VISUALISATION_GRAPHS_LOGSEQ_PHYSICS_REPULSION_STRENGTH=2000
```

## Configuration Best Practices

### Development
1. Copy `.env_template` to `.env`
2. Set required API keys
3. Use `docker-compose --profile dev up`
4. Modify `settings.yaml` for testing

### Production
1. Use environment variables for all secrets
2. Mount `settings.yaml` as read-only
3. Use `docker-compose --profile production up`
4. Configure Cloudflare tunnel for external access

### Testing Different Configurations
```bash
# Test with custom settings
docker run -v ./custom-settings.yaml:/app/settings.yaml ...

# Override specific values
SYSTEM_NETWORK_PORT=8080 docker-compose --profile dev up

# Use different GPU
CUDA_ARCH=89 NVIDIA_VISIBLE_DEVICES=1 docker-compose up
```

## Debugging Configuration Issues

### Check Loaded Configuration
```bash
# View current settings via API
curl http://localhost:4000/api/settings

# Check environment variables in container
docker exec visionflow_container env | sort

# Validate settings.yaml
docker exec visionflow_container cat /app/settings.yaml
```

### Common Issues

1. **Port conflicts:**
   - Check `SYSTEM_NETWORK_PORT` in environment
   - Verify `settings.yaml` port configuration
   - Ensure Docker port mapping matches

2. **GPU not detected:**
   - Verify `CUDA_ARCH` matches your GPU
   - Check `NVIDIA_VISIBLE_DEVICES` is correct
   - Ensure Docker has GPU runtime

3. **API keys not working:**
   - Confirm `.env` file is loaded
   - Check environment variables in container
   - Verify key format and validity

## Migration Notes

### From Legacy Configuration
If upgrading from an older version:

1. Move top-level physics settings to `graphs.logseq`:
   ```yaml
   # Old (deprecated)
   visualisation:
     physics:
       repulsion_strength: 1500

   # New
   visualisation:
     graphs:
       logseq:
         physics:
           repulsion_strength: 1500
   ```

2. Update environment variable names:
   ```bash
   # Old
   WEBSOCKET_PORT=3002

   # New
   MCP_TCP_PORT=9500
   MCP_TRANSPORT=tcp
   ```

## Security Considerations

1. **Never commit secrets:**
   - Use `.gitignore` for `.env`
   - Store production secrets in secure vault
   - Rotate API keys regularly

2. **Restrict file permissions:**
   ```bash
   chmod 600 .env
   chmod 644 settings.yaml
   ```

3. **Use read-only mounts in production:**
   ```yaml
   volumes:
     - ./settings.yaml:/app/settings.yaml:ro
   ```

## Support

For configuration issues:
1. Check this guide first
2. Review logs: `docker logs visionflow_container`
3. Consult `/docs/deployment/` for specific scenarios
4. Open an issue with configuration details (excluding secrets)