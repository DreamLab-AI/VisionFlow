---
title: Docker Compose Guide
description: Guide to deploying VisionFlow with Docker Compose, covering all compose files, service definitions, and environment configuration.
category: how-to
tags:
  - deployment
  - docker
  - docker-compose
  - infrastructure
updated-date: 2026-02-12
difficulty-level: intermediate
---

# Docker Compose Guide

This guide covers the Docker Compose configuration for VisionFlow, including the primary compose file, the voice-routing overlay, Neo4j graph database setup, and all required environment variables.

## Compose File Overview

VisionFlow uses several compose files for different deployment scenarios:

| File | Purpose | Profile |
|------|---------|---------|
| `docker-compose.yml` | Base development services (webxr + Cloudflare tunnel) | `dev` |
| `docker-compose.unified.yml` | Unified stack with Neo4j, JSS, and profile-based config | `dev`, `prod` |
| `docker-compose.voice.yml` | Voice pipeline overlay (LiveKit, Whisper, Kokoro TTS) | `dev`, `prod` |
| `docker-compose.production.yml` | Legacy production-only compose | default |
| `docker-compose.vircadia.yml` | Vircadia XR integration | varies |

## Primary Stack: docker-compose.unified.yml

The unified compose file is the recommended entry point. It defines shared YAML anchors for DRY configuration and includes both development and production service variants.

### Starting the Development Stack

```bash
# Create the external network (first time only)
docker network create docker_ragflow

# Start development profile with Neo4j
docker compose -f docker-compose.unified.yml --profile dev up -d

# Start with voice services layered in
docker compose -f docker-compose.unified.yml -f docker-compose.voice.yml --profile dev up -d
```

### Starting the Production Stack

```bash
docker compose -f docker-compose.unified.yml --profile prod up -d
```

## Neo4j Container Configuration

Neo4j 5.13.0 serves as the sole graph database. It is defined in `docker-compose.unified.yml` and starts before the VisionFlow application container via `depends_on: condition: service_healthy`.

**Ports:**
- `7474` -- Neo4j Browser HTTP interface
- `7687` -- Bolt protocol (application connections)

**Volumes:**
- `neo4j-data` -- Graph data files
- `neo4j-logs` -- Database logs
- `neo4j-conf` -- Custom configuration
- `neo4j-plugins` -- APOC and other plugins

**Health check:** `wget --spider --quiet http://localhost:7474` every 10 seconds with 5 retries and a 30-second start period.

**Memory tuning (environment):**
- `NEO4J_server_memory_pagecache_size=512M`
- `NEO4J_server_memory_heap_max__size=1G`

## Voice Routing: docker-compose.voice.yml

Layer this file on top of the unified compose to enable voice-to-voice audio. It adds three GPU-aware services.

| Service | Image | Port | Role |
|---------|-------|------|------|
| `livekit` | `livekit/livekit-server:v1.7` | 7880 (HTTP/WS), 7881 (TCP), 7882/udp | WebRTC SFU for spatial audio |
| `turbo-whisper` | `fedirz/faster-whisper-server:latest-cuda` | 8100 | Streaming speech-to-text (OpenAI-compatible) |
| `kokoro-tts` | `ghcr.io/remsky/kokoro-fastapi-cpu:latest` | 8880 | Text-to-speech with per-agent voice presets |

LiveKit reads its configuration from `config/livekit.yaml`, mounted read-only into the container at `/etc/livekit.yaml`. The config sets Opus codec defaults, 50-participant room limits, and WebRTC media ports 50000-50200/udp.

## Environment Variables

Create a `.env` file in the project root. Key variables consumed by the compose files:

### Neo4j
| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URI` | `bolt://neo4j:7687` | Bolt connection URI |
| `NEO4J_USER` | `neo4j` | Database username |
| `NEO4J_PASSWORD` | (required) | Database password |
| `NEO4J_DATABASE` | `neo4j` | Database name |

### LiveKit
| Variable | Default | Description |
|----------|---------|-------------|
| `LIVEKIT_API_KEY` | `visionflow` | API key for LiveKit server |
| `LIVEKIT_API_SECRET` | `visionflow-voice-secret-change-in-prod` | API secret (change in production) |
| `LIVEKIT_URL` | `ws://livekit:7880` | Internal WebSocket URL |

### Application
| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ENV` | `development` | Node environment |
| `RUST_LOG` | `debug` | Rust log level filter |
| `SYSTEM_NETWORK_PORT` | `4000` | Internal API port |
| `CUDA_ARCH` | `86` | CUDA compute capability |
| `CLOUDFLARE_TUNNEL_TOKEN` | (required for tunnel) | Cloudflare Argo tunnel token |

### MCP / Agent Coordination
| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_HOST` | `agentic-workstation` | MCP server hostname |
| `MCP_TCP_PORT` | `9500` | MCP TCP port |
| `ORCHESTRATOR_WS_URL` | `ws://mcp-orchestrator:9001/ws` | Orchestrator WebSocket |

## Network Configuration

All services join the external `docker_ragflow` network. Create it once before first run:

```bash
docker network create docker_ragflow
```

Service hostnames on this network: `webxr`, `neo4j`, `livekit`, `turbo-whisper`, `kokoro-tts`, `jss`, `cloudflared-tunnel`.

## Volumes Summary

| Volume | Purpose |
|--------|---------|
| `visionflow-data` | Application data (databases, markdown, metadata) |
| `visionflow-logs` | Application and Nginx logs |
| `npm-cache` | npm package cache |
| `cargo-cache` | Cargo registry cache |
| `cargo-git-cache` | Cargo git dependency cache |
| `cargo-target-cache` | Rust build artifact cache |
| `neo4j-data` | Neo4j graph store |
| `jss-data` | JavaScript Solid Server pod storage |

## See Also

- [Docker Environment Setup](./docker-environment-setup.md) -- Local development environment walkthrough
- [Docker Deployment](./docker-deployment.md) -- Production deployment with TLS and reverse proxy
- [Infrastructure: Docker Environment](../infrastructure/docker-environment.md) -- Complete container and network reference
- `config/livekit.yaml` -- LiveKit SFU configuration
- `docker-compose.unified.yml` -- Primary compose file (source of truth)
