# Docker Compose Profiles Configuration

*[Deployment](../index.md)*

## Overview

VisionFlow uses Docker Compose profiles to manage different deployment environments from a single `docker-compose.yml` file. This simplifies configuration management and reduces duplication.

## Available Profiles

### Development Profile (`dev`)
For local development with hot reloading and debugging capabilities.

**Features:**
- Hot module replacement (HMR) for React
- Rust source code mounting for live rebuilding
- Volume mounts for all development files
- Debug logging enabled
- Nginx proxy on port 3001
- Development tools and scripts mounted

**Usage:**
```bash
docker-compose --profile dev up
```

### Production Profile (`production` or `prod`)
For production deployments with optimised builds and security.

**Features:**
- Optimised production builds
- Cloudflare tunnel integration
- Health checks configured
- Minimal volume mounts (data only)
- API exposed on port 4000
- No source code mounting

**Usage:**
```bash
docker-compose --profile production up
# or
docker-compose --profile prod up
```

## Environment-Specific Commands

### Development
```bash
# Start development environment
docker-compose --profile dev up

# Rebuild development containers
docker-compose --profile dev build

# View logs
docker-compose --profile dev logs -f

# Stop development environment
docker-compose --profile dev down
```

### Production
```bash
# Start production environment
docker-compose --profile production up -d

# Deploy with Cloudflare tunnel
CLOUDFLARE_TUNNEL_TOKEN=your-token docker-compose --profile production up -d

# View production logs
docker-compose --profile production logs -f webxr-prod

# Stop production environment
docker-compose --profile production down
```

## Service Names by Profile

| Profile | Service Name | Container Name | Port |
|---------|-------------|----------------|------|
| dev | webxr-dev | visionflow_container | 3001 |
| production | webxr-prod | visionflow_container | 4000 |
| production | cloudflared | cloudflared-tunnel | - |
| all | powerdev | powerdev | 8080 |

## Environment Variables

Both profiles respect the `.env` file. Key variables:

### Development
- `NODE_ENV=development`
- `VITE_DEBUG=true`
- `SYSTEM_NETWORK_PORT=4000`
- `VITE_DEV_SERVER_PORT=5173`

### Production
- `NODE_ENV=production`
- `CLOUDFLARE_TUNNEL_TOKEN` (required for tunnel)
- `NVIDIA_GPU_UUID` (for specific GPU targeting)

## Migration from Separate Files

Previously, we maintained three separate files:
- `docker-compose.yml` (base)
- `docker-compose.dev.yml`
- `docker-compose.production.yml`

Now consolidated into a single `docker-compose.yml` using profiles.

### Migration Steps

1. **Update deployment scripts:**
   ```bash
   # Old
   docker-compose -f docker-compose.dev.yml up

   # New
   docker-compose --profile dev up
   ```

2. **Update CI/CD pipelines:**
   ```yaml
   # Old
   docker-compose -f docker-compose.production.yml up -d

   # New
   docker-compose --profile production up -d
   ```

3. **Remove old files** (after verification):
   ```bash
   rm docker-compose.dev.yml
   rm docker-compose.production.yml
   ```

## Troubleshooting

### Profile not starting
Ensure you specify a profile:
```bash
# Wrong - no services will start
docker-compose up

# Correct
docker-compose --profile dev up
```

### Wrong ports exposed
Check which profile is running:
- Dev: Port 3001 (Nginx proxy)
- Production: Port 4000 (API directly)

### GPU not available
Verify NVIDIA runtime is installed and the profile's GPU configuration matches your system.

## Advanced Usage

### Running multiple profiles
```bash
# Start both dev and production (not recommended)
docker-compose --profile dev --profile production up
```

### Override profiles with environment
```bash
# Use production profile but override port
COMPOSE_PROFILES=production PORT=8080 docker-compose up
```

### Profile-specific builds
```bash
# Build only development image
docker-compose --profile dev build webxr-dev

# Build only production image
docker-compose --profile production build webxr-prod
```

## Best Practices

1. **Never mix profiles** in the same environment
2. **Use production profile** for any public-facing deployment
3. **Keep `.env` file** updated with environment-specific values
4. **Test profile switching** in staging before production
5. **Document profile choice** in deployment runbooks

## Benefits of Profile Consolidation

1. **Single source of truth** - One file to maintain
2. **Reduced duplication** - Common settings shared
3. **Easier comparison** - Dev vs prod side-by-side
4. **Simplified CI/CD** - Same file, different profiles
5. **Better documentation** - Clear profile boundaries

## Related Topics

- [Deployment Guide](../deployment/index.md)
- [Docker Deployment Guide](../deployment/docker.md)
- [Docker MCP Integration - Production Deployment Guide](../deployment/docker-mcp-integration.md)
- [Multi-Agent Container Setup](../deployment/multi-agent-setup.md)
