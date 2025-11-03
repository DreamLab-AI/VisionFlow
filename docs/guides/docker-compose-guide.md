# Docker Compose Unified Configuration - Usage Guide

## Overview

The unified `docker-compose.unified.yml` replaces three separate compose files with a single, profile-based configuration that supports both development and production environments.

## Quick Start

### Development Mode
```bash
# Start development environment
docker compose -f docker-compose.unified.yml --profile development up

# Or use short form
docker compose -f docker-compose.unified.yml --profile dev up

# Build and start
docker compose -f docker-compose.unified.yml --profile dev up --build

# Detached mode
docker compose -f docker-compose.unified.yml --profile dev up -d
```

### Production Mode
```bash
# Start production environment
docker compose -f docker-compose.unified.yml --profile production up

# Or use short form
docker compose -f docker-compose.unified.yml --profile prod up

# Build and start
docker compose -f docker-compose.unified.yml --profile prod up --build -d
```

## Profiles Explained

### Development Profile (`development` or `dev`)
- **Services**: `visionflow` + `cloudflared`
- **Dockerfile**: `Dockerfile.dev`
- **Ports**: 3001 (Nginx), 4000 (API)
- **Volumes**:
  - Data volumes (visionflow-data, visionflow-logs)
  - Build cache volumes (npm, cargo)
  - Docker socket (read-only)
- **Environment**:
  - `NODE_ENV=development`
  - `RUST_LOG=debug`
  - HMR enabled on port 24678
  - Vite dev server on port 5173
- **Use Case**: Local development with hot reload

### Production Profile (`production` or `prod`)
- **Services**: `visionflow-production` + `cloudflared`
- **Dockerfile**: `Dockerfile.production`
- **Ports**: 4000 (API only)
- **Volumes**:
  - Data volumes only (NO source mounts)
  - Build cache volumes
- **Environment**:
  - `NODE_ENV=production`
  - `RUST_LOG=warn`
  - Debug disabled
- **Use Case**: Production deployment

## Configuration via Environment Variables

Create a `.env` file in the project root:

```bash
# GPU Configuration
CUDA_ARCH=86                    # RTX A6000 = 86, RTX 4090 = 89
NVIDIA_VISIBLE_DEVICES=0        # GPU device ID

# Container Names
CONTAINER_NAME=visionflow_container
HOSTNAME=webxr

# Network
EXTERNAL_NETWORK=docker_ragflow
NETWORK_ALIAS=webxr

# Ports
DEV_NGINX_PORT=3001
API_PORT=4000
PROD_API_PORT=4000
VITE_DEV_SERVER_PORT=5173
VITE_HMR_PORT=24678

# Build Configuration
DOCKERFILE=Dockerfile.dev       # or Dockerfile.production
BUILD_TARGET=development        # or production
REBUILD_PTX=false               # Set to true to rebuild PTX kernels

# Logging
RUST_LOG=debug                  # development: debug, production: warn
DEBUG_ENABLED=true

# MCP & Claude Flow
CLAUDE_FLOW_HOST=agentic-workstation
MCP_HOST=agentic-workstation
MCP_TCP_PORT=9500
ORCHESTRATOR_WS_URL=ws://mcp-orchestrator:9001/ws

# Cloudflare Tunnel
CLOUDFLARE_TUNNEL_TOKEN=your_token_here

# Volume Names (optional customization)
DATA_VOLUME_NAME=visionflow-data
LOGS_VOLUME_NAME=visionflow-logs
NPM_CACHE_VOLUME=visionflow-npm-cache
CARGO_CACHE_VOLUME=visionflow-cargo-cache
```

## Common Operations

### View Running Services
```bash
docker compose -f docker-compose.unified.yml --profile dev ps
```

### View Logs
```bash
# All services
docker compose -f docker-compose.unified.yml --profile dev logs -f

# Specific service
docker compose -f docker-compose.unified.yml --profile dev logs -f visionflow
```

### Stop Services
```bash
docker compose -f docker-compose.unified.yml --profile dev down
```

### Stop and Remove Volumes
```bash
docker compose -f docker-compose.unified.yml --profile dev down -v
```

### Rebuild Containers
```bash
docker compose -f docker-compose.unified.yml --profile dev up --build
```

### Execute Commands in Container
```bash
# Development
docker compose -f docker-compose.unified.yml --profile dev exec visionflow bash

# Production
docker compose -f docker-compose.unified.yml --profile prod exec visionflow-production bash
```

## Volume Strategy

### Development
- **visionflow-data**: Persistent databases, markdown, metadata, user settings
- **visionflow-logs**: Application logs
- **npm-cache**: NPM package cache (speeds up rebuilds)
- **cargo-cache**: Cargo registry cache
- **cargo-git-cache**: Cargo git dependencies
- **cargo-target-cache**: Compiled Rust artifacts
- **Docker socket**: Read-only access for container management

### Production
- **visionflow-data**: Persistent databases, markdown, metadata, user settings
- **visionflow-logs**: Application logs
- **cargo-target-cache**: Compiled Rust artifacts
- **NO source mounts**: All code is baked into the image
- **NO Docker socket**: Enhanced security

## Architecture Features

### DRY Configuration with Extension Fields
- `x-common-environment`: Shared environment variables
- `x-common-healthcheck`: Standardized health checks
- `x-common-logging`: Consistent log rotation
- `x-gpu-resources`: GPU allocation settings

### Service Composition
- **visionflow**: Base service for development
- **visionflow-production**: Extends base service with production overrides
- **cloudflared**: Cloudflare tunnel (works with both profiles)

### Network Configuration
- External network: `docker_ragflow`
- Service aliases for DNS resolution
- Inter-service communication

### Health Checks
- HTTP endpoint: `http://localhost:4000/`
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds

### GPU Support
- NVIDIA runtime
- CUDA 12.4.1 base image
- Configurable GPU device selection
- Compute and utility capabilities

## Migration from Old Compose Files

### From `docker-compose.yml` (old development)
```bash
# Old command
docker compose --profile dev up

# New command
docker compose -f docker-compose.unified.yml --profile dev up
```

### From `docker-compose.production.yml`
```bash
# Old command
docker compose -f docker-compose.production.yml up

# New command
docker compose -f docker-compose.unified.yml --profile prod up
```

## Troubleshooting

### Container Won't Start
1. Check GPU availability: `nvidia-smi`
2. Verify network exists: `docker network ls | grep docker_ragflow`
3. Check logs: `docker compose -f docker-compose.unified.yml --profile dev logs`

### Port Already in Use
Change port in `.env`:
```bash
DEV_NGINX_PORT=3002  # Instead of 3001
API_PORT=4001         # Instead of 4000
```

### Volume Permissions
```bash
# Reset volume permissions
docker compose -f docker-compose.unified.yml --profile dev down -v
docker volume rm visionflow-data visionflow-logs
docker compose -f docker-compose.unified.yml --profile dev up
```

### GPU Not Detected
```bash
# Verify NVIDIA Docker runtime
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Check CUDA architecture
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Advanced Usage

### Multiple Profiles (Not Recommended)
```bash
# Run both dev and prod simultaneously (different ports required)
docker compose -f docker-compose.unified.yml --profile dev --profile prod up
```

### Custom Dockerfile
```bash
# Override Dockerfile in .env
DOCKERFILE=Dockerfile.custom
docker compose -f docker-compose.unified.yml --profile dev up --build
```

### Debug Mode
```bash
# Override entrypoint for debugging
docker compose -f docker-compose.unified.yml --profile dev run --entrypoint bash visionflow
```

## Best Practices

1. **Always use `.env` file**: Never hardcode sensitive values
2. **Volume management**: Regular backups of `visionflow-data`
3. **Log rotation**: Configured via `x-common-logging`
4. **Health checks**: Monitor service health with `docker compose ps`
5. **Resource limits**: Add memory/CPU limits in production
6. **Security**: Production uses minimal volumes and no Docker socket

## Benefits Over Multiple Compose Files

✅ **Single source of truth**: One file to maintain
✅ **DRY principle**: Shared configuration via extension fields
✅ **Environment-based**: Easy switching between dev/prod
✅ **Consistent naming**: Standardized volume and network names
✅ **Better defaults**: Sensible fallbacks for all variables
✅ **Simplified CI/CD**: Single file for all environments
✅ **Type safety**: Docker Compose validates the entire configuration
