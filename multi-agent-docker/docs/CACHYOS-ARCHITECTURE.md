# CachyOS Ecosystem Architecture

## Overview

All containers in this stack use CachyOS v3 as the base image to ensure binary compatibility with the host system. This architecture addresses glibc and libstdc++ version mismatches that occur when running containers built on older distributions (Ubuntu 22.04, Debian 12) on hosts running bleeding-edge kernels and libraries.

Key benefits:
- Binary compatibility with CachyOS host systems
- x86-64-v3 instruction set optimizations (AVX, AVX2, BMI1/2, FMA)
- Access to latest toolchains and libraries
- Consistent ABI across host and container boundaries

## Container Base Images

| Container | Base Image | Purpose |
|-----------|------------|---------|
| agentic-workstation | cachyos/cachyos-v3 | Primary development environment with Claude Code, supervisord services |
| comfyui-cachyos | cachyos/cachyos-v3 | AI image generation with SAM3D, Stable Diffusion workflows |
| claude-zai-cachyos | cachyos/cachyos-v3 | Cost-effective Claude API proxy with worker pool |

## CUDA Configuration

CUDA is installed to `/opt/cuda` to avoid conflicts with system packages.

| Component | Path |
|-----------|------|
| Base Path | /opt/cuda |
| Version | 13.1 |
| Compiler (nvcc) | /opt/cuda/bin/nvcc |
| PTX Assembler | /opt/cuda/bin/ptxas |
| Libraries | /opt/cuda/lib64 |
| CUPTI | /opt/cuda/extras/CUPTI/lib64 |
| Include Headers | /opt/cuda/include |

Environment variables set in container:
```bash
CUDA_HOME=/opt/cuda
PATH=/opt/cuda/bin:$PATH
LD_LIBRARY_PATH=/opt/cuda/lib64:/opt/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

## Build Commands

```bash
# Build all CachyOS-aligned containers
./build-unified.sh

# Force rebuild without cache
./build-unified.sh --no-cache

# Build specific service only
docker compose -f docker-compose.unified.yml build agentic-workstation
```

## Service Deployment Modes

### Monolithic (Default)

All services run inside the agentic-workstation container, managed by supervisord.

| Service | Internal Port | Access |
|---------|---------------|--------|
| SSH | 22 | Mapped to host 2222 |
| VNC | 5901 | Direct access |
| code-server | 8080 | Direct access |
| Management API | 9090 | Direct access |
| Z.AI | 9600 | Internal only (localhost) |

Configuration:
```yaml
environment:
  ZAI_INTERNAL: "true"
  ZAI_URL: "http://localhost:9600"
```

### Microservices (Optional)

Use the docker-compose overlay for distributed deployment with separate containers.

```bash
docker compose -f docker-compose.unified.yml -f docker-compose.visionflow-cachyos.yml up -d
```

Configuration for microservices mode:
```yaml
environment:
  ZAI_INTERNAL: "false"
  ZAI_URL: "http://claude-zai:9600"
```

| Service | Container | Port |
|---------|-----------|------|
| Z.AI | claude-zai-cachyos | 9600 |
| ComfyUI | comfyui-cachyos | 8188 |
| Management API | agentic-workstation | 9090 |

## SSH Key Handling

SSH keys are mounted read-only and copied with correct permissions at container startup.

| Source | Destination | Permissions |
|--------|-------------|-------------|
| ~/.ssh (host) | ~/.ssh-host (container, read-only mount) | - |
| ~/.ssh-host/* | ~/.ssh/* (copied by entrypoint) | 600 (files), 700 (directory) |

Supported key types:
- ed25519 (recommended)
- rsa
- ecdsa
- dsa (legacy)

The entrypoint script handles key copying:
```bash
if [ -d "$HOME/.ssh-host" ]; then
    mkdir -p "$HOME/.ssh"
    cp -r "$HOME/.ssh-host/"* "$HOME/.ssh/" 2>/dev/null || true
    chmod 700 "$HOME/.ssh"
    chmod 600 "$HOME/.ssh/"* 2>/dev/null || true
fi
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| ZAI_INTERNAL | true | Enable internal Z.AI service via supervisord |
| ZAI_URL | http://localhost:9600 | Z.AI endpoint URL for API calls |
| MANAGEMENT_API_KEY | (required) | Authentication key for management API |
| RUVECTOR_PG_HOST | ruvector-postgres | PostgreSQL host for RuVector memory |
| RUVECTOR_PG_PORT | 5432 | PostgreSQL port |
| RUVECTOR_PG_USER | ruvector | PostgreSQL username |
| RUVECTOR_PG_DB | ruvector | PostgreSQL database name |
| ANTHROPIC_API_KEY | (required) | Anthropic API key for Claude |
| CUDA_HOME | /opt/cuda | CUDA installation path |
| DISPLAY | :1 | X11 display for VNC |

## Network Architecture

All containers connect to the shared `docker_ragflow` network for inter-service communication.

| Network | Subnet | Purpose |
|---------|--------|---------|
| docker_ragflow | 172.19.0.0/16 | Shared container network |

Service discovery uses Docker DNS with container names as hostnames.
