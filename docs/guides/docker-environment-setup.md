---
layout: default
title: Docker Environment Setup - Multi-Agent System
parent: Guides
nav_order: 48
description: Docker environment setup for the multi-agent system
---


# Docker Environment Setup - Multi-Agent System

**Last Updated:** November 5, 2025
**Difficulty:** Intermediate
**Time Required:** 15-30 minutes

---

## Overview

This guide walks through setting up the VisionFlow multi-agent Docker environment, which provides:
- **VisionFlow container** - Graph engine with GPU acceleration
- **Agentic workstation** - Claude Code with 13+ skills
- **GUI tools container** - Blender, QGIS, and other graphics applications

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

### Software Prerequisites

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

### Optional: NVIDIA GPU Setup

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

**What this does:**
- Builds VisionFlow Rust backend
- Starts Neo4j database
- Launches React client
- Joins `docker_ragflow` network

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

## Detailed Setup

### Configuration

#### Environment Variables

Create `.env` file in `multi-agent-docker/`:

```bash
# Copy example configuration
cp .env.example .env
```

**Key settings:**
```bash
# Resource limits
DOCKER_MEMORY=16g
DOCKER_CPUS=4

# MCP Authentication (CHANGE THESE!)
WS_AUTH_TOKEN=your-secure-websocket-token
TCP_AUTH_TOKEN=your-secure-tcp-token

# Port configuration
MCP_TCP_PORT=9500
MCP_WS_PORT=3002
VNC_PORT=5901
SSH_PORT=2222

# Enable/disable features
RATE_LIMIT_ENABLED=true
SSL_ENABLED=false
```

#### Docker Compose Profiles

The system uses Docker Compose profiles for different deployment scenarios:

**Available profiles:**
- `dev` - Development mode with hot-reload (default)
- `production` - Optimized for production deployment
- `gpu` - GPU acceleration enabled
- `minimal` - Core services only (no GUI container)

**Usage:**
```bash
# Development (default)
./multi-agent.sh start

# Production deployment
./multi-agent.sh start --profile production

# GPU-accelerated
./multi-agent.sh start --profile gpu

# Minimal (no GUI)
./multi-agent.sh start --profile minimal
```

---

### Accessing Containers

#### SSH Access (Agentic Workstation)

```bash
# Default credentials
ssh -p 2222 devuser@localhost
# Password: turboflow

# Or use docker exec
docker exec -it agentic-workstation /bin/zsh
```

#### VNC Access (GUI Tools)

```bash
# Using any VNC client:
# Host: localhost:5901
# Password: turboflow

# Recommended VNC clients:
# - RealVNC Viewer (Windows, macOS, Linux)
# - TigerVNC (Linux)
# - Screen Sharing (macOS built-in)
```

#### Web Interfaces

```bash
# Code Server (VS Code in browser)
open http://localhost:8080

# VisionFlow Client
open http://localhost:3001

# VisionFlow API
curl http://localhost:9090/api/health
```

---

### Network Configuration

#### Docker Network: docker_ragflow

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
├── visionflow-container (172.18.0.2)
├── agentic-workstation (172.18.0.3)
└── gui-tools-container (172.18.0.4)
```

**Container-to-container communication:**
```bash
# From agentic-workstation, connect to VisionFlow
curl http://visionflow-container:9090/api/health

# From VisionFlow, connect to MCP server
curl http://agentic-workstation:9500/health
```

#### Firewall Configuration

For production deployments, configure firewall rules:

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

### Skill Installation

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

#### Manual Skill Installation

If a skill is missing:

```bash
# Inside agentic-workstation container
cd /home/devuser/.claude/skills

# Clone skill repository or copy skill files
cp -r /app/skills/docker-manager ./

# Verify SKILL.md exists
cat docker-manager/SKILL.md

# Test skill
./docker-manager/test-skill.sh
```

---

### Docker Socket Configuration

The Docker socket allows agentic-workstation to manage VisionFlow:

**Automatic setup:**
```yaml
# In docker-compose.unified.yml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock:rw
```

**Verify socket access:**
```bash
# From agentic-workstation
docker exec agentic-workstation ls -la /var/run/docker.sock

# Expected: srw-rw---- 1 root docker
```

**Troubleshooting socket permissions:**
```bash
# Add devuser to docker group (inside container)
docker exec agentic-workstation sudo usermod -aG docker devuser

# Restart container for group change
docker restart agentic-workstation
```

**⚠️ Security Warning:**
Docker socket access grants root-equivalent privileges. Only use in trusted environments.

---

## Advanced Configuration

### GPU Acceleration

#### VisionFlow GPU Setup

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

#### GUI Tools GPU Setup

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

### Custom Skill Development

Create new skills in `/home/devuser/.claude/skills/`:

```bash
# Inside agentic-workstation
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

## Natural Language Examples
```
Use my-custom-skill to do something
```
EOF

# Create implementation
# (Python script, Node.js module, binary, etc.)

# Create test script
cat > test-skill.sh << 'EOF'
#!/bin/bash
echo "Testing my-custom-skill..."
# Add tests here
EOF
chmod +x test-skill.sh

# Test skill
./test-skill.sh
```

### MCP Server Configuration

#### TCP Server (High Performance)

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

#### WebSocket Server

```bash
# WebSocket server starts automatically
# Check status:
curl http://localhost:3002/health

# Test connection from host:
wscat -c ws://localhost:3002
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

# Stop conflicting process or change port
# Edit .env:
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

### Skill Issues

#### Skill not found

```bash
# List installed skills
docker exec agentic-workstation ls -la /home/devuser/.claude/skills/

# Re-run setup
docker exec agentic-workstation /app/setup-workspace.sh --force
```

#### Docker Manager can't control VisionFlow

```bash
# Verify socket is mounted
docker exec agentic-workstation ls -la /var/run/docker.sock

# Check devuser is in docker group
docker exec agentic-workstation groups devuser

# Verify VisionFlow container exists
docker ps -a | grep visionflow
```

#### GUI skill timeout

GUI skills (Blender, QGIS, KiCad) require the GUI container:

```bash
# Check GUI container status
docker ps | grep gui-tools-container

# View GUI container logs
docker logs gui-tools-container

# Restart GUI container
docker restart gui-tools-container

# Connect via VNC to verify desktop is running
# vnc://localhost:5901
```

---

## Performance Optimization

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

Use Docker's build cache for faster rebuilds:

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

# Integration with monitoring systems
# Export logs to ELK, Prometheus, Grafana, etc.
```

---

## Next Steps

✅ **Setup complete!** Now you can:

1. **Learn the skills:** See [Multi-Agent Skills Guide](./multi-agent-skills.md)
2. **Understand architecture:** Read [Multi-Agent System Architecture](../../explanations/architecture/multi-agent-system.md)
3. **Start developing:** Use Claude Code with natural language commands
4. **Build custom skills:** Create your own specialized capabilities

**Example workflow:**
```
# Natural language command in Claude Code:
Use Docker Manager to restart VisionFlow
Create a Wardley map for our GPU architecture
Use Chrome DevTools to debug localhost:3001
Use Blender to create a 3D network visualization
```

---

## References

- [Multi-Agent Skills Reference](./multi-agent-skills.md)
- [System Architecture](../../explanations/architecture/multi-agent-system.md)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [Docker Documentation](https://docs.docker.com/)
- 

---

**Setup Guide Version:** 1.0
**Last Updated:** November 5, 2025
**Support:** Check VisionFlow GitHub issues for help
