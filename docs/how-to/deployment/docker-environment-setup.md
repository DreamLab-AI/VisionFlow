---
title: Docker Environment Setup (Local Development)
description: Setting up a local Docker development environment for VisionFlow with hot reload, port mapping, and volume mounts.
category: how-to
tags:
  - development
  - docker
  - local
  - setup
updated-date: 2026-02-12
difficulty-level: beginner
---

# Docker Environment Setup (Local Development)

This guide walks through setting up a local development environment for VisionFlow using Docker Compose, including hot reload for both the Rust backend and the React frontend.

## Prerequisites

- Docker Engine 24+ with Compose V2 (`docker compose` subcommand)
- NVIDIA Container Toolkit (`nvidia-ctk`) for GPU passthrough
- NVIDIA GPU with compute capability 8.6+ and CUDA 12.4 drivers
- Git and a text editor

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-org/VisionFlow.git
cd VisionFlow

# 2. Create the shared Docker network (one time)
docker network create docker_ragflow

# 3. Copy and configure environment
cp .env.example .env
# Edit .env -- at minimum set NEO4J_PASSWORD

# 4. Start the development stack
docker compose -f docker-compose.unified.yml --profile dev up -d

# 5. Verify all services are healthy
docker compose -f docker-compose.unified.yml --profile dev ps
```

The application will be available at `http://localhost:3001` once the health check passes (allow ~40 seconds for initial startup).

## Port Mapping Reference

| Port | Service | Protocol | Description |
|------|---------|----------|-------------|
| 3001 | Nginx (dev entry) | HTTP | Frontend + API reverse proxy |
| 4000 | Actix-web API | HTTP | Direct Rust backend access |
| 5173 | Vite dev server | HTTP | HMR dev server (internal, proxied through Nginx) |
| 7474 | Neo4j Browser | HTTP | Graph database web UI |
| 7687 | Neo4j Bolt | TCP | Application database connections |
| 7880 | LiveKit | HTTP/WS | WebRTC signaling (voice overlay only) |
| 7881 | LiveKit RTC | TCP | WebRTC over TCP fallback |
| 7882 | LiveKit RTC | UDP | WebRTC primary media transport |
| 8100 | Turbo Whisper | HTTP/WS | Speech-to-text API (voice overlay only) |
| 8880 | Kokoro TTS | HTTP | Text-to-speech API (voice overlay only) |
| 24678 | Vite HMR | WS | Hot Module Replacement WebSocket (internal) |

## Volume Mounts for Hot Reload

The development configuration in `docker-compose.unified.yml` mounts source code read-only from the host, enabling live editing without rebuilding the container.

### Rust Backend Mounts

```yaml
- ./src:/app/src:ro
- ./Cargo.toml:/app/Cargo.toml:ro
- ./Cargo.lock:/app/Cargo.lock:ro
- ./build.rs:/app/build.rs:ro
- ./whelk-rs:/app/whelk-rs:ro
```

When you edit Rust source files on the host, the container sees the changes immediately. The entrypoint script watches for file changes and triggers `cargo build` automatically. Build artifacts are cached in the `cargo-target-cache` volume to speed up incremental compilation.

### React Frontend Mounts

```yaml
- ./client/src:/app/client/src:ro
- ./client/public:/app/client/public:ro
- ./client/index.html:/app/client/index.html:ro
- ./client/vite.config.ts:/app/client/vite.config.ts:ro
- ./client/tsconfig.json:/app/client/tsconfig.json:ro
```

Vite's HMR picks up TypeScript/React changes instantly. The dev server runs on port 5173 internally and is proxied through Nginx on port 3001.

### Data Volumes

```yaml
- visionflow-data:/app/data       # Persistent application data
- visionflow-logs:/app/logs       # Log files
- ./data/markdown:/workspace/ext/data/markdown:ro   # Graph content from host
- ./data/metadata:/workspace/ext/data/metadata:rw   # Metadata (read-write)
```

## Adding Voice Services

To enable the voice pipeline (LiveKit, Whisper, Kokoro TTS), layer the voice compose file:

```bash
docker compose \
  -f docker-compose.unified.yml \
  -f docker-compose.voice.yml \
  --profile dev up -d
```

This adds three additional containers. LiveKit requires UDP ports 50000-50200 for WebRTC media. Verify with:

```bash
docker logs visionflow-livekit --tail 20
```

## Accessing Services

| URL | What You Get |
|-----|-------------|
| `http://localhost:3001` | VisionFlow UI (through Nginx) |
| `http://localhost:3001/api/health` | Backend health check JSON |
| `http://localhost:7474` | Neo4j Browser (login with NEO4J_USER/NEO4J_PASSWORD) |
| `ws://localhost:3001/wss` | Graph data WebSocket |
| `ws://localhost:3001/ws/speech` | Voice WebSocket |

## Common Development Tasks

### Rebuild the Container

```bash
docker compose -f docker-compose.unified.yml --profile dev build --no-cache visionflow
docker compose -f docker-compose.unified.yml --profile dev up -d visionflow
```

### View Logs

```bash
# All services
docker compose -f docker-compose.unified.yml --profile dev logs -f

# Single service
docker logs -f visionflow_container
```

### Shell into the Container

```bash
docker exec -it visionflow_container bash
```

### Reset Neo4j Data

```bash
docker compose -f docker-compose.unified.yml --profile dev down
docker volume rm visionflow-neo4j-data
docker compose -f docker-compose.unified.yml --profile dev up -d
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container exits immediately | Missing `.env` or bad `NEO4J_PASSWORD` | Check `docker logs visionflow_container` |
| Port 3001 unreachable | Nginx not started yet | Wait for health check (40s start period) |
| Neo4j connection refused | Neo4j still starting | VisionFlow waits via `depends_on: condition: service_healthy` |
| GPU not detected | Missing NVIDIA Container Toolkit | Install `nvidia-ctk` and restart Docker |
| Slow Rust builds | Cold cargo cache | First build is slow; subsequent builds use `cargo-target-cache` volume |

## See Also

- [Docker Compose Guide](./docker-compose-guide.md) -- Full compose file reference
- [Docker Deployment](./docker-deployment.md) -- Production deployment with TLS
- [Infrastructure: Docker Environment](../infrastructure/docker-environment.md) -- Container and network reference
- [Testing Guide](../development/testing-guide.md) -- Running tests inside the container
