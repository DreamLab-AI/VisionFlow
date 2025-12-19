---
title: Docker Compose Unified Configuration - Usage Guide
description: The unified `docker-compose.unified.yml` replaces three separate compose files with a single, profile-based configuration that supports both development and production environments.
category: guide
tags:
  - tutorial
  - docker
  - backend
updated-date: 2025-12-18
difficulty-level: intermediate
---


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
  - `NODE-ENV=development`
  - `RUST-LOG=debug`
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
  - `NODE-ENV=production`
  - `RUST-LOG=warn`
  - Debug disabled
- **Use Case**: Production deployment

## Configuration via Environment Variables

Create a `.env` file in the project root:

```bash
# GPU Configuration
CUDA-ARCH=86                    # RTX A6000 = 86, RTX 4090 = 89
NVIDIA-VISIBLE-DEVICES=0        # GPU device ID

# Container Names
CONTAINER-NAME=visionflow-container
HOSTNAME=webxr

# Network
EXTERNAL-NETWORK=docker-ragflow
NETWORK-ALIAS=webxr

# Ports
DEV-NGINX-PORT=3001
API-PORT=4000
PROD-API-PORT=4000
VITE-DEV-SERVER-PORT=5173
VITE-HMR-PORT=24678

# Build Configuration
DOCKERFILE=Dockerfile.dev       # or Dockerfile.production
BUILD-TARGET=development        # or production
REBUILD-PTX=false               # Set to true to rebuild PTX kernels

# Logging
RUST-LOG=debug                  # development: debug, production: warn
DEBUG-ENABLED=true

# MCP & Claude Flow
CLAUDE-FLOW-HOST=agentic-workstation
MCP-HOST=agentic-workstation
MCP-TCP-PORT=9500
ORCHESTRATOR-WS-URL=ws://mcp-orchestrator:9001/ws

# Cloudflare Tunnel
CLOUDFLARE-TUNNEL-TOKEN=your-token-here

# Volume Names (optional customization)
DATA-VOLUME-NAME=visionflow-data
LOGS-VOLUME-NAME=visionflow-logs
NPM-CACHE-VOLUME=visionflow-npm-cache
CARGO-CACHE-VOLUME=visionflow-cargo-cache
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
- External network: `docker-ragflow`
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
2. Verify network exists: `docker network ls | grep docker-ragflow`
3. Check logs: `docker compose -f docker-compose.unified.yml --profile dev logs`

### Port Already in Use
Change port in `.env`:
```bash
DEV-NGINX-PORT=3002  # Instead of 3001
API-PORT=4001         # Instead of 4000
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
nvidia-smi --query-gpu=compute-cap --format=csv
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

---

---

## Related Documentation

- [Vircadia Multi-User XR Integration - User Guide](vircadia-multi-user-guide.md)
- [Ontology Storage Guide](ontology-storage-guide.md)
- [Multi-Agent Docker Environment - Complete Documentation](infrastructure/docker-environment.md)
- [VisionFlow Guides](index.md)
- [Documentation Contributing Guidelines](contributing.md)

## Benefits Over Multiple Compose Files

✅ **Single source of truth**: One file to maintain
✅ **DRY principle**: Shared configuration via extension fields
✅ **Environment-based**: Easy switching between dev/prod
✅ **Consistent naming**: Standardized volume and network names
✅ **Better defaults**: Sensible fallbacks for all variables
✅ **Simplified CI/CD**: Single file for all environments
✅ **Type safety**: Docker Compose validates the entire configuration
