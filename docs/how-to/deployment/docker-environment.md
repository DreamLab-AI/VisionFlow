---
title: Docker Environment Setup
description: Complete guide for setting up the VisionFlow multi-agent Docker environment with GPU acceleration, Claude Code skills, and unified compose configuration.
category: how-to
tags:
  - docker
  - deployment
  - multi-agent
  - gpu
  - mcp
updated-date: 2025-01-29
difficulty-level: intermediate
---

# Docker Environment Setup

Complete guide for deploying VisionFlow in Docker with multi-agent support, GPU acceleration, and unified compose configuration.

---

## Overview

The VisionFlow Docker environment provides:

- **VisionFlow container** - Graph engine with GPU acceleration
- **Agentic workstation** - Claude Code with 13+ skills
- **GUI tools container** - Blender, QGIS, and graphics applications
- **Unified compose** - Single profile-based configuration

---

## Prerequisites

### System Requirements

**Minimum:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- 24GB RAM (8GB per container)
- 6 CPU cores (2 per container)
- 20GB free disk space
- Linux, macOS, or Windows with WSL2

**Recommended:**
- 48GB RAM (16GB per container)
- 12+ CPU cores
- NVIDIA GPU with CUDA 12.0+ drivers
- 100GB NVMe SSD
- Ubuntu 22.04 LTS or later

### Verification Commands

```bash
# Check Docker version
docker --version
# Requires: Docker version 20.10+

# Check Docker Compose version
docker compose version
# Requires: Docker Compose version 2.0+

# Check available memory
free -h
# Recommended: 48GB+

# Check disk space
df -h
# Required: 20GB minimum, 100GB recommended
```

### NVIDIA GPU Setup (Optional)

For GPU-accelerated physics and rendering:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

---

## Quick Start (5 Minutes)

### Step 1: Clone Repository

```bash
cd /path/to/your/projects
git clone <repository-url> VisionFlow
cd VisionFlow
```

### Step 2: Build Multi-Agent Environment

```bash
cd multi-agent-docker
./build-unified.sh
```

**What this does:**
- Builds agentic-workstation container with Claude Code
- Builds GUI tools container with Blender, QGIS, etc.
- Installs 13+ Claude skills
- Mounts Docker socket for inter-container management
- Sets up MCP servers (TCP and WebSocket)

**Build time:** ~10-15 minutes (first time)

### Step 3: Start VisionFlow

```bash
# From VisionFlow root directory
cd ..
./scripts/launch.sh build
./scripts/launch.sh up
```

### Step 4: Verify Setup

```bash
# Check all containers are running
docker ps

# Expected output:
# visionflow-container        Up
# agentic-workstation         Up
# gui-tools-container         Up
```

**Access points:**
- VisionFlow Client: http://localhost:3001
- VisionFlow API: http://localhost:9090
- Code Server: http://localhost:8080
- VNC (GUI tools): vnc://localhost:5901

---

## Unified Compose Configuration

The unified `docker-compose.unified.yml` replaces three separate compose files with a single, profile-based configuration.

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

### Profile Comparison

| Feature | Development | Production |
|---------|-------------|------------|
| **Dockerfile** | Dockerfile.dev | Dockerfile.production |
| **Ports** | 3001 (Nginx), 4000 (API) | 4000 (API only) |
| **NODE_ENV** | development | production |
| **RUST_LOG** | debug | warn |
| **HMR** | Enabled (port 24678) | Disabled |
| **Source mounts** | Yes | No |
| **Docker socket** | Yes | No |

---

## Environment Configuration

Create `.env` file in project root:

```bash
# GPU Configuration
CUDA_ARCH=86                    # RTX A6000 = 86, RTX 4090 = 89
NVIDIA_VISIBLE_DEVICES=0        # GPU device ID

# Container Names
CONTAINER_NAME=visionflow-container
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
CLOUDFLARE_TUNNEL_TOKEN=your-token-here

# Resource limits
DOCKER_MEMORY=16g
DOCKER_CPUS=4

# MCP Authentication (CHANGE THESE!)
WS_AUTH_TOKEN=your-secure-websocket-token
TCP_AUTH_TOKEN=your-secure-tcp-token
```

---

## Container Access

### SSH Access (Agentic Workstation)

```bash
# Default credentials
ssh -p 2222 devuser@localhost
# Password: turboflow

# Or use docker exec
docker exec -it agentic-workstation /bin/zsh
```

### VNC Access (GUI Tools)

```bash
# Using any VNC client:
# Host: localhost:5901
# Password: turboflow

# Recommended VNC clients:
# - RealVNC Viewer (Windows, macOS, Linux)
# - TigerVNC (Linux)
# - Screen Sharing (macOS built-in)
```

### Web Interfaces

```bash
# Code Server (VS Code in browser)
open http://localhost:8080

# VisionFlow Client
open http://localhost:3001

# VisionFlow API
curl http://localhost:9090/api/health
```

---

## Network Configuration

### Docker Network: docker_ragflow

All containers communicate via the `docker_ragflow` bridge network:

```bash
# Inspect network
docker network inspect docker_ragflow

# View connected containers
docker network inspect docker_ragflow | grep Name
```

**Network topology:**
```
docker_ragflow (172.18.0.0/16)
+-- visionflow-container (172.18.0.2)
+-- agentic-workstation (172.18.0.3)
+-- gui-tools-container (172.18.0.4)
```

**Container-to-container communication:**
```bash
# From agentic-workstation, connect to VisionFlow
curl http://visionflow-container:9090/api/health

# From VisionFlow, connect to MCP server
curl http://agentic-workstation:9500/health
```

### Firewall Configuration

For production deployments:

```bash
# Allow VisionFlow ports
sudo ufw allow 3001/tcp  # VisionFlow client
sudo ufw allow 9090/tcp  # VisionFlow API

# Allow agentic-workstation ports (if remote access needed)
sudo ufw allow 2222/tcp  # SSH
sudo ufw allow 8080/tcp  # Code Server
sudo ufw allow 9500/tcp  # MCP TCP
sudo ufw allow 3002/tcp  # MCP WebSocket

# Allow VNC (if remote GUI access needed)
sudo ufw allow 5901/tcp  # VNC
```

---

## Volume Strategy

### Development Volumes

| Volume | Purpose |
|--------|---------|
| `visionflow-data` | Persistent databases, markdown, metadata, user settings |
| `visionflow-logs` | Application logs |
| `npm-cache` | NPM package cache (speeds up rebuilds) |
| `cargo-cache` | Cargo registry cache |
| `cargo-git-cache` | Cargo git dependencies |
| `cargo-target-cache` | Compiled Rust artifacts |
| Docker socket | Read-only access for container management |

### Production Volumes

| Volume | Purpose |
|--------|---------|
| `visionflow-data` | Persistent databases, markdown, metadata, user settings |
| `visionflow-logs` | Application logs |
| `cargo-target-cache` | Compiled Rust artifacts |

**Note:** Production uses NO source mounts and NO Docker socket for enhanced security.

---

## Skill Installation

Skills are automatically installed during build. To verify or reinstall:

```bash
# Access agentic-workstation
docker exec -it agentic-workstation /bin/zsh

# List installed skills
ls -la /home/devuser/.claude/skills/

# Expected output (13 skills):
# docker-manager/
# wardley-maps/
# chrome-devtools/
# blender/
# imagemagick/
# pbr-rendering/
# playwright/
# web-summary/
# import-to-ontology/
# qgis/
# kicad/
# ngspice/
# logseq-formatted/

# Test all skills
/app/mcp-helper.sh test-all
```

### Manual Skill Installation

```bash
# Inside agentic-workstation container
cd /home/devuser/.claude/skills
mkdir my-custom-skill
cd my-custom-skill

# Create SKILL.md
cat > SKILL.md << 'EOF'
---
name: my-custom-skill
description: Description of what this skill does
---

# My Custom Skill

## Capabilities
- Feature 1
- Feature 2
EOF

# Test skill
./test-skill.sh
```

---

## MCP Server Configuration

### TCP Server (High Performance)

```bash
# Start MCP TCP server
docker exec agentic-workstation mcp-tcp-start

# Check status
docker exec agentic-workstation mcp-tcp-status

# View logs
docker logs -f agentic-workstation | grep "MCP TCP"

# Test connection
docker exec agentic-workstation mcp-tcp-test
```

### WebSocket Server

```bash
# WebSocket server starts automatically
# Check status:
curl http://localhost:3002/health

# Test connection from host:
wscat -c ws://localhost:3002
```

---

## GPU Acceleration

### VisionFlow GPU Setup

Enable CUDA for physics simulation:

```yaml
# In docker-compose.yml (VisionFlow)
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Verify GPU access:**
```bash
docker exec visionflow-container nvidia-smi
```

### GUI Tools GPU Setup

Enable GPU rendering in Blender:

```yaml
# In docker-compose.unified.yml (GUI tools)
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, graphics]
```

---

## Architecture Features

### DRY Configuration with Extension Fields

- `x-common-environment`: Shared environment variables
- `x-common-healthcheck`: Standardised health checks
- `x-common-logging`: Consistent log rotation
- `x-gpu-resources`: GPU allocation settings

### Service Composition

- **visionflow**: Base service for development
- **visionflow-production**: Extends base service with production overrides
- **cloudflared**: Cloudflare tunnel (works with both profiles)

### Health Checks

- HTTP endpoint: `http://localhost:4000/`
- Interval: 30 seconds
- Timeout: 10 seconds
- Retries: 3
- Start period: 40 seconds

---

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

---

## Troubleshooting

### Build Issues

#### "Docker socket not found"

```bash
# Verify Docker is running
sudo systemctl status docker

# Check socket exists on host
ls -la /var/run/docker.sock
```

#### "Port already in use"

```bash
# Find process using port
sudo lsof -i :9090

# Change port in .env:
# VISIONFLOW_PORT=9091
```

#### "Out of disk space"

```bash
# Clean up Docker
docker system prune -a

# Remove unused volumes
docker volume prune

# Check space
df -h
```

### Container Issues

#### Container won't start

```bash
# Check logs
docker logs agentic-workstation

# Try starting with fresh volumes
./multi-agent.sh cleanup  # WARNING: Deletes all data
./multi-agent.sh build
./multi-agent.sh start
```

#### MCP server not responding

```bash
# Check MCP server status
docker exec agentic-workstation mcp-tcp-status

# Restart MCP server
docker exec agentic-workstation mcp-tcp-restart

# View MCP logs
docker logs agentic-workstation | grep MCP
```

#### VNC connection fails

```bash
# Check GUI container is running
docker ps | grep gui-tools-container

# Verify VNC server is running
docker exec gui-tools-container ps aux | grep vnc

# Restart GUI container
docker restart gui-tools-container
```

### GPU Issues

#### GPU not detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Check CUDA architecture
nvidia-smi --query-gpu=compute_cap --format=csv
```

#### Volume Permissions

```bash
# Reset volume permissions
docker compose -f docker-compose.unified.yml --profile dev down -v
docker volume rm visionflow-data visionflow-logs
docker compose -f docker-compose.unified.yml --profile dev up
```

---

## Performance Optimisation

### Resource Tuning

Edit `.env` to adjust resource limits:

```bash
# High-performance configuration
DOCKER_MEMORY=32g
DOCKER_CPUS=8

# Minimal configuration (development laptop)
DOCKER_MEMORY=8g
DOCKER_CPUS=2
```

### Build Cache

```bash
# Fast rebuild (uses cache)
./build-unified.sh

# Clean rebuild (no cache, slower)
./build-unified.sh --no-cache
```

### Network Performance

Use MCP TCP server for better performance:

```python
# Prefer TCP over WebSocket
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('agentic-workstation', 9500))

# TCP is 60% faster than WebSocket
```

---

## Production Deployment

### Security Hardening

1. **Change default passwords:**
   ```bash
   # Update .env
   WS_AUTH_TOKEN=$(openssl rand -hex 32)
   TCP_AUTH_TOKEN=$(openssl rand -hex 32)
   JWT_SECRET=$(openssl rand -hex 32)
   ```

2. **Enable SSL/TLS:**
   ```bash
   SSL_ENABLED=true
   SSL_CERT_PATH=/app/certs/server.crt
   SSL_KEY_PATH=/app/certs/server.key
   ```

3. **Enable rate limiting:**
   ```bash
   RATE_LIMIT_ENABLED=true
   RATE_LIMIT_MAX_REQUESTS=100
   RATE_LIMIT_WINDOW_MS=60000
   ```

4. **Restrict network access:**
   ```yaml
   # Only expose necessary ports
   ports:
     - "127.0.0.1:9090:9090"  # VisionFlow API (localhost only)
     - "127.0.0.1:3001:3001"  # VisionFlow client (localhost only)
   ```

### Backup Strategy

```bash
# Backup critical volumes
docker cp agentic-workstation:/workspace ./backup/workspace
docker cp agentic-workstation:/home/devuser/.claude ./backup/claude
docker exec visionflow-container neo4j-admin dump --to=/tmp/neo4j-backup.dump
docker cp visionflow-container:/tmp/neo4j-backup.dump ./backup/

# Automate with cron
0 2 * * * /path/to/backup-script.sh
```

### Monitoring

```bash
# Container health checks
docker ps --filter health=healthy

# Resource usage
docker stats

# Aggregated logs
docker logs -f --tail 100 agentic-workstation
docker logs -f --tail 100 visionflow-container
docker logs -f --tail 100 gui-tools-container
```

---

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

---

## Benefits of Unified Configuration

- **Single source of truth**: One file to maintain
- **DRY principle**: Shared configuration via extension fields
- **Environment-based**: Easy switching between dev/prod
- **Consistent naming**: Standardised volume and network names
- **Better defaults**: Sensible fallbacks for all variables
- **Simplified CI/CD**: Single file for all environments
- **Type safety**: Docker Compose validates the entire configuration

---

## Related Documentation

- [Skills Catalog](../../reference/agents/skills-catalog.md)
- [WebSocket API Endpoints](../../reference/api/websocket-endpoints.md)
- [Database Schema Catalog](../../reference/database/schema-catalog.md)

---

**Last Updated**: January 29, 2025
**Maintainer**: VisionFlow Documentation Team
