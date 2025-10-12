# Docker Reference Guide

Complete reference for Docker operations in the multi-agent-docker project.

## Table of Contents

1. [Docker Compose Commands](#docker-compose-commands)
2. [Container Management](#container-management)
3. [Volume Management](#volume-management)
4. [Network Management](#network-management)
5. [Image Building and Updating](#image-building-and-updating)
6. [Resource Management](#resource-management)
7. [Common Commands Cheat Sheet](#common-commands-cheat-sheet)
8. [Dockerfile Reference](#dockerfile-reference)

---

## Docker Compose Commands

### Basic Operations

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d agentic-flow-cachyos

# Stop all services
docker-compose down

# Stop without removing containers
docker-compose stop

# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart agentic-flow-cachyos
```

### Service Management

```bash
# View running services
docker-compose ps

# View all services (including stopped)
docker-compose ps -a

# View service logs
docker-compose logs -f agentic-flow-cachyos

# View logs from all services
docker-compose logs -f

# View last 100 lines
docker-compose logs --tail=100 agentic-flow-cachyos

# Scale a service (if supported)
docker-compose up -d --scale claude-zai=3
```

### Configuration and Validation

```bash
# Validate docker-compose.yml syntax
docker-compose config

# View effective configuration
docker-compose config --services

# List images used by services
docker-compose images

# Pull latest images without starting
docker-compose pull
```

---

## Container Management

### Starting and Stopping

```bash
# Start container
docker start agentic-flow-cachyos

# Stop container (graceful)
docker stop agentic-flow-cachyos

# Force stop container
docker stop -t 0 agentic-flow-cachyos

# Restart container
docker restart agentic-flow-cachyos

# Pause container (freeze processes)
docker pause agentic-flow-cachyos

# Unpause container
docker unpause agentic-flow-cachyos
```

### Accessing Containers

```bash
# Open interactive shell (zsh)
docker exec -it agentic-flow-cachyos zsh

# Open bash shell
docker exec -it agentic-flow-cachyos bash

# Run single command
docker exec agentic-flow-cachyos ls -la /home/devuser

# Run command as root
docker exec -u root -it agentic-flow-cachyos bash

# Run command in specific directory
docker exec -w /home/devuser/workspace -it agentic-flow-cachyos zsh
```

### Container Inspection

```bash
# View container details
docker inspect agentic-flow-cachyos

# View specific field (JSON path)
docker inspect agentic-flow-cachyos --format='{{.State.Status}}'

# View environment variables
docker inspect agentic-flow-cachyos --format='{{range .Config.Env}}{{println .}}{{end}}'

# View mounted volumes
docker inspect agentic-flow-cachyos --format='{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'

# View network settings
docker inspect agentic-flow-cachyos --format='{{range .NetworkSettings.Networks}}{{println .IPAddress}}{{end}}'
```

### Logs and Monitoring

```bash
# View logs (follow)
docker logs -f agentic-flow-cachyos

# View logs with timestamps
docker logs -f --timestamps agentic-flow-cachyos

# View last N lines
docker logs --tail=100 agentic-flow-cachyos

# View logs since timestamp
docker logs --since 2025-10-12T10:00:00 agentic-flow-cachyos

# View logs between time ranges
docker logs --since 1h --until 30m agentic-flow-cachyos

# Live resource usage
docker stats agentic-flow-cachyos

# One-time stats
docker stats --no-stream agentic-flow-cachyos

# Process list inside container
docker top agentic-flow-cachyos

# Container port mappings
docker port agentic-flow-cachyos
```

### Container Lifecycle

```bash
# Remove stopped container
docker rm agentic-flow-cachyos

# Force remove running container
docker rm -f agentic-flow-cachyos

# Remove container and volumes
docker rm -v agentic-flow-cachyos

# Rename container
docker rename agentic-flow-cachyos agentic-workstation

# Export container filesystem
docker export agentic-flow-cachyos > container-backup.tar

# Create image from container
docker commit agentic-flow-cachyos my-custom-image:latest
```

---

## Volume Management

### Volume Operations

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect workspace

# Create volume
docker volume create my-volume

# Remove volume
docker volume rm workspace

# Remove unused volumes
docker volume prune

# Remove all volumes (dangerous)
docker volume prune -a
```

### Volume Inspection

```bash
# View volume details
docker volume inspect workspace

# View volume mountpoint
docker volume inspect workspace --format='{{.Mountpoint}}'

# List containers using volume
docker ps -a --filter volume=workspace

# View volume size
docker system df -v
```

### Backup and Restore

```bash
# Backup volume to tar
docker run --rm \
  -v workspace:/data \
  -v $(pwd):/backup \
  archlinux:latest \
  tar czf /backup/workspace-backup.tar.gz -C /data .

# Restore volume from tar
docker run --rm \
  -v workspace:/data \
  -v $(pwd):/backup \
  archlinux:latest \
  tar xzf /backup/workspace-backup.tar.gz -C /data

# Copy files from volume to host
docker cp agentic-flow-cachyos:/home/devuser/workspace/myfile.txt ./

# Copy files from host to volume
docker cp ./myfile.txt agentic-flow-cachyos:/home/devuser/workspace/

# Clone volume
docker volume create workspace-clone
docker run --rm \
  -v workspace:/source:ro \
  -v workspace-clone:/dest \
  archlinux:latest \
  sh -c "cd /source && cp -av . /dest"
```

### Volume Information

```bash
# Project volumes (from docker-compose.yml)
workspace         # Persistent workspace for development
model-cache       # Model cache storage
agent-memory      # Session data and agent memory
config-persist    # Configuration persistence
management-logs   # Management API logs
```

---

## Network Management

### Network Operations

```bash
# List networks
docker network ls

# Inspect network
docker network inspect agentic-network

# Create network
docker network create my-network

# Remove network
docker network rm agentic-network

# Remove unused networks
docker network prune

# Connect container to network
docker network connect agentic-network my-container

# Disconnect container from network
docker network disconnect agentic-network my-container
```

### Network Inspection

```bash
# View network details
docker network inspect agentic-network

# List containers on network
docker network inspect agentic-network --format='{{range .Containers}}{{.Name}} {{end}}'

# View container IP on network
docker inspect agentic-flow-cachyos --format='{{.NetworkSettings.Networks.agentic_network.IPAddress}}'

# Test connectivity between containers
docker exec agentic-flow-cachyos ping claude-zai
```

### Network Configuration

```bash
# Project network (from docker-compose.yml)
agentic-network   # Bridge network for service communication

# Container hostnames
agentic-workstation  # Main workstation container
claude-zai           # Claude Z.ai service
```

---

## Image Building and Updating

### Building Images

```bash
# Build from docker-compose.yml
docker-compose build

# Build specific service
docker-compose build agentic-flow-cachyos

# Build without cache
docker-compose build --no-cache

# Build with progress output
docker-compose build --progress=plain

# Build with custom Dockerfile
docker build -f Dockerfile.workstation -t agentic-workstation:latest .

# Build with build arguments
docker build --build-arg VERSION=1.0 -t agentic-workstation:1.0 .
```

### Updating Images

```bash
# Pull latest base images
docker-compose pull

# Rebuild and restart services
docker-compose up -d --build

# Force recreate containers
docker-compose up -d --force-recreate

# Update single service
docker-compose up -d --build agentic-flow-cachyos
```

### Image Management

```bash
# List images
docker images

# Remove image
docker rmi agentic-workstation:latest

# Remove unused images
docker image prune

# Remove all unused images
docker image prune -a

# View image history
docker history agentic-workstation:latest

# View image size details
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Tag image
docker tag agentic-workstation:latest agentic-workstation:v1.0

# Save image to file
docker save agentic-workstation:latest > workstation.tar

# Load image from file
docker load < workstation.tar
```

---

## Resource Management

### Resource Limits (docker-compose.yml)

```yaml
deploy:
  resources:
    limits:
      memory: 64G        # Maximum memory
      cpus: '32'         # Maximum CPU cores
    reservations:
      memory: 16G        # Guaranteed memory
      cpus: '8'          # Guaranteed CPU cores
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, compute, utility]
```

### Runtime Configuration

```bash
# Shared memory size
shm_size: 32gb

# GPU runtime
runtime: nvidia

# Device access
devices:
  - /dev/dri:/dev/dri
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
```

### Monitoring Resources

```bash
# Live stats for all containers
docker stats

# Container-specific stats
docker stats agentic-flow-cachyos --no-stream

# System-wide resource usage
docker system df

# Detailed system usage
docker system df -v

# Container processes
docker top agentic-flow-cachyos

# GPU utilization (if nvidia-smi available)
docker exec agentic-flow-cachyos nvidia-smi
```

### Resource Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Remove everything unused
docker system prune

# Aggressive cleanup (includes volumes)
docker system prune -a --volumes
```

---

## Common Commands Cheat Sheet

### Quick Reference

```bash
# Start project
docker-compose up -d

# View logs
docker-compose logs -f

# Access shell
docker exec -it agentic-flow-cachyos zsh

# Restart service
docker-compose restart agentic-flow-cachyos

# Stop project
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# View running containers
docker ps

# View all containers
docker ps -a

# Check container health
docker inspect agentic-flow-cachyos --format='{{.State.Health.Status}}'

# View resource usage
docker stats --no-stream

# Clean up unused resources
docker system prune

# Backup volume
docker run --rm -v workspace:/data -v $(pwd):/backup archlinux tar czf /backup/workspace.tar.gz -C /data .

# View container IP
docker inspect agentic-flow-cachyos --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'
```

### Troubleshooting Commands

```bash
# Check container status
docker ps -a --filter name=agentic-flow-cachyos

# View recent logs
docker logs --tail=50 agentic-flow-cachyos

# Check health status
docker inspect agentic-flow-cachyos | jq '.[0].State.Health'

# Verify port mappings
docker port agentic-flow-cachyos

# Test service endpoint
curl http://localhost:9090/health

# Enter as root for debugging
docker exec -u root -it agentic-flow-cachyos bash

# View environment variables
docker exec agentic-flow-cachyos env

# Check disk usage
docker exec agentic-flow-cachyos df -h

# Check running processes
docker exec agentic-flow-cachyos ps aux
```

---

## Dockerfile Reference

### Dockerfile.workstation Structure

```dockerfile
# Base image
FROM archlinux:latest

# System packages installation
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm base-devel git nodejs npm python

# User creation
RUN useradd -m -G wheel,video,audio -s /usr/bin/zsh devuser

# Application installation
RUN npm install -g agentic-flow@latest pm2

# File copying
COPY --chown=devuser:devuser config/ /home/devuser/.config/

# Environment variables
ENV SHELL=/usr/bin/zsh \
    DISPLAY=:1 \
    WORKSPACE=/home/devuser/workspace

# Port exposure
EXPOSE 8080 6901 9090 3000

# Volume mount points
VOLUME ["/home/devuser/workspace", "/home/devuser/models"]

# Startup command
CMD ["/opt/venv/bin/supervisord", "-c", "/etc/supervisord.conf"]
```

### Build Arguments

```bash
# Use build arguments for customization
docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg NODE_VERSION=20 \
  -t agentic-workstation:custom .
```

### Multi-stage Builds

```dockerfile
# Build stage
FROM archlinux:latest AS builder
RUN pacman -S --noconfirm base-devel
COPY . /build
RUN cd /build && make

# Runtime stage
FROM archlinux:latest
COPY --from=builder /build/output /app
CMD ["/app/server"]
```

### Best Practices

```dockerfile
# Combine RUN commands to reduce layers
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm package1 package2 && \
    rm -rf /var/cache/pacman/pkg/*

# Use specific versions for reproducibility
FROM archlinux:latest
RUN npm install agentic-flow@1.2.3

# Leverage build cache
COPY package*.json ./
RUN npm install
COPY . .

# Use .dockerignore
# Create .dockerignore with:
.git
node_modules
*.log
.env

# Set proper ownership
COPY --chown=devuser:devuser app/ /home/devuser/app/

# Use HEALTHCHECK
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:9090/health || exit 1
```

---

## Service-Specific Information

### Agentic Flow Cachyos

```bash
Container: agentic-flow-cachyos
Hostname: agentic-workstation
Ports:
  - 9090: Management API
  - 5901: VNC
  - 6901: noVNC
  - 8080: code-server

Volumes:
  - workspace: /home/devuser/workspace
  - model-cache: /home/devuser/models
  - agent-memory: /home/devuser/.agentic-flow
  - config-persist: /home/devuser/.config
  - management-logs: /home/devuser/logs

Healthcheck: http://localhost:9090/health
```

### Claude Z.ai Service

```bash
Container: claude-zai-service
Hostname: claude-zai
Port: 9600
Healthcheck: http://localhost:9600/health
Restart policy: unless-stopped
```

---

## Environment Variables

### API Keys

```bash
ANTHROPIC_API_KEY          # Claude API access
OPENAI_API_KEY             # OpenAI API access
GOOGLE_GEMINI_API_KEY      # Gemini API access
OPENROUTER_API_KEY         # OpenRouter access
CONTEXT7_API_KEY           # Context7 access
GITHUB_TOKEN               # GitHub integration
BRAVE_API_KEY              # Brave search
```

### Router Configuration

```bash
ROUTER_MODE                # performance|balanced|cost-optimized
PRIMARY_PROVIDER           # gemini|openai|claude|openrouter
FALLBACK_CHAIN             # Comma-separated provider list
```

### System Configuration

```bash
GPU_ACCELERATION           # true|false
CUDA_VISIBLE_DEVICES       # GPU device selection
ENABLE_DESKTOP             # true|false
ENABLE_CODE_SERVER         # true|false
LOG_LEVEL                  # debug|info|warn|error
NODE_ENV                   # production|development
```

### Management API

```bash
MANAGEMENT_API_KEY         # API authentication key
MANAGEMENT_API_PORT        # Default: 9090
MANAGEMENT_API_HOST        # Default: 0.0.0.0
```

---

## Advanced Operations

### Container Debugging

```bash
# Attach to container process
docker attach agentic-flow-cachyos

# View container changes
docker diff agentic-flow-cachyos

# Export container filesystem
docker export agentic-flow-cachyos > container.tar

# Check container events
docker events --filter container=agentic-flow-cachyos

# Copy files while preserving permissions
docker cp -a ./source/. agentic-flow-cachyos:/dest/
```

### Performance Tuning

```bash
# Limit container restart attempts
docker update --restart=on-failure:3 agentic-flow-cachyos

# Adjust memory limits
docker update --memory=32g agentic-flow-cachyos

# Adjust CPU limits
docker update --cpus=16 agentic-flow-cachyos

# Set memory swap limit
docker update --memory-swap=64g agentic-flow-cachyos
```

### Security Operations

```bash
# Run security scan
docker scan agentic-workstation:latest

# Check for vulnerabilities
docker scout cves agentic-workstation:latest

# Run as non-root user
docker exec -u devuser -it agentic-flow-cachyos zsh

# Limit capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE myimage

# Use read-only root filesystem
docker run --read-only -v /tmp:/tmp:rw myimage
```

---

## References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker CLI Reference](https://docs.docker.com/engine/reference/commandline/cli/)
