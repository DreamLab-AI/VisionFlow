---
title: Docker Environment Reference
description: Complete reference for all Docker containers, networks, volumes, and environment variables in the VisionFlow infrastructure.
category: how-to
tags:
  - infrastructure
  - docker
  - containers
  - reference
updated-date: 2026-02-12
difficulty-level: intermediate
---

# Docker Environment Reference

This document provides a comprehensive reference for every container, network, volume, and environment variable in the VisionFlow Docker infrastructure, including the multi-agent orchestration container.

## Container Inventory

### Core Application Containers

| Container | Image / Dockerfile | Role | Profiles |
|-----------|-------------------|------|----------|
| `visionflow_container` | `Dockerfile.unified` (dev target) | Rust API server + Nginx + Vite dev server | `dev` |
| `visionflow_prod_container` | `Dockerfile.production` | Optimized release build with Nginx | `prod` |
| `visionflow-neo4j` | `neo4j:5.13.0` | Graph database (sole data store) | all |
| `visionflow-jss` | `Dockerfile.jss` (JavaScriptSolidServer) | Solid pods for user data and ontology fragments | `dev`, `prod` |
| `cloudflared-tunnel` | `cloudflare/cloudflared:latest` | Cloudflare Argo tunnel for TLS and routing | `dev`, `prod` |

### Voice Pipeline Containers

Activated by layering `docker-compose.voice.yml`:

| Container | Image | Role | Profiles |
|-----------|-------|------|----------|
| `visionflow-livekit` | `livekit/livekit-server:v1.7` | WebRTC Selective Forwarding Unit for spatial voice | `dev`, `prod` |
| `visionflow-turbo-whisper` | `fedirz/faster-whisper-server:latest-cuda` | Streaming speech-to-text (faster-whisper, GPU) | `dev`, `prod` |
| `visionflow-kokoro-tts` | `ghcr.io/remsky/kokoro-fastapi-cpu:latest` | Text-to-speech with per-agent voice presets | `dev`, `prod` |

### Multi-Agent Docker Container

The `multi-agent-docker/` directory contains a separate unified container (`Dockerfile.unified`) that runs the AI orchestration stack. This container provides:

- Claude Flow agent coordination via MCP (Model Context Protocol)
- MCP infrastructure services (relay, orchestrator)
- Management API on port 9090
- SSH access for remote agent control

It communicates with the VisionFlow application container over the shared `docker_ragflow` network. Key environment variables for agent coordination:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_FLOW_HOST` | `agentic-workstation` | Hostname of the agent orchestration container |
| `MCP_HOST` | `agentic-workstation` | MCP server hostname |
| `MCP_TCP_PORT` | `9500` | MCP relay TCP port |
| `MCP_TRANSPORT` | `tcp` | MCP transport protocol |
| `ORCHESTRATOR_WS_URL` | `ws://mcp-orchestrator:9001/ws` | Orchestrator WebSocket endpoint |
| `BOTS_ORCHESTRATOR_URL` | `ws://agentic-workstation:3002` | Bot orchestration endpoint |
| `MANAGEMENT_API_HOST` | `agentic-workstation` | Management API hostname |
| `MANAGEMENT_API_PORT` | `9090` | Management API port |

## Networks

| Network | Type | Purpose |
|---------|------|---------|
| `docker_ragflow` | External bridge | Shared network connecting all VisionFlow containers, multi-agent containers, and any external RAGFlow services |

All containers join `docker_ragflow`. Create it before first run:

```bash
docker network create docker_ragflow
```

Service discovery uses Docker DNS. Each container registers hostnames via its `hostname` field or network aliases.

## Volumes

### Application Volumes

| Volume | Named As | Mount Point | Purpose |
|--------|----------|-------------|---------|
| `visionflow-data` | `visionflow-data` | `/app/data` | Markdown, metadata, user settings |
| `visionflow-logs` | `visionflow-logs` | `/app/logs` | Application and Nginx logs |

### Build Cache Volumes

| Volume | Named As | Mount Point | Purpose |
|--------|----------|-------------|---------|
| `npm-cache` | `visionflow-npm-cache` | `/root/.npm` | npm package cache |
| `cargo-cache` | `visionflow-cargo-cache` | `/root/.cargo/registry` | Cargo crate registry |
| `cargo-git-cache` | `visionflow-cargo-git-cache` | `/root/.cargo/git` | Cargo git dependencies |
| `cargo-target-cache` | `visionflow-cargo-target-cache` | `/app/target` | Rust build artifacts |

### Database Volumes

| Volume | Named As | Mount Point | Purpose |
|--------|----------|-------------|---------|
| `neo4j-data` | `visionflow-neo4j-data` | `/data` | Neo4j graph store |
| `neo4j-logs` | `visionflow-neo4j-logs` | `/logs` | Neo4j server logs |
| `neo4j-conf` | `visionflow-neo4j-conf` | `/conf` | Neo4j custom configuration |
| `neo4j-plugins` | `visionflow-neo4j-plugins` | `/plugins` | APOC and extensions |
| `jss-data` | `visionflow-jss-data` | `/data` | Solid pod storage |

### Voice Pipeline Volumes

| Volume | Named As | Purpose |
|--------|----------|---------|
| `whisper-models` | `visionflow-whisper-models` | Cached Whisper model weights |

## Environment Variable Reference

### Core Application

| Variable | Default | Used By |
|----------|---------|---------|
| `NODE_ENV` | `development` | Vite, Nginx behavior |
| `RUST_LOG` | `debug` (dev) / `warn` (prod) | Rust tracing subscriber |
| `RUST_LOG_REDIRECT` | `true` | Route Rust logs through tracing |
| `SYSTEM_NETWORK_PORT` | `4000` | Actix-web listen port |
| `DOCKER_ENV` | `true` | Detect containerized runtime |
| `CUDA_ARCH` | `86` | CUDA compute capability target |
| `NVIDIA_VISIBLE_DEVICES` | `0` | GPU device index |
| `FORCE_FULL_SYNC` | `1` | Trigger full graph sync on startup |

### Neo4j

| Variable | Default | Used By |
|----------|---------|---------|
| `NEO4J_URI` | `bolt://neo4j:7687` | Rust neo4rs driver |
| `NEO4J_USER` | `neo4j` | Database authentication |
| `NEO4J_PASSWORD` | (required) | Database authentication |
| `NEO4J_DATABASE` | `neo4j` | Target database name |

### Solid Server (JSS)

| Variable | Default | Used By |
|----------|---------|---------|
| `JSS_URL` | `http://jss:3030` | Rust HTTP client |
| `JSS_WS_URL` | `ws://jss:3030/.notifications` | WebSocket subscription client |

### Vite / Frontend

| Variable | Default | Used By |
|----------|---------|---------|
| `VITE_DEBUG` | `true` (dev) | Enable frontend debug panels |
| `VITE_DEV_SERVER_PORT` | `5173` | Vite listen port |
| `VITE_API_PORT` | `4000` | API port for client fetch calls |
| `VITE_HMR_PORT` | `24678` | Hot Module Replacement port |

### MCP / Agent Coordination

| Variable | Default | Used By |
|----------|---------|---------|
| `MCP_RECONNECT_ATTEMPTS` | `3` | MCP client retry logic |
| `MCP_RECONNECT_DELAY` | `1000` | Delay between retries (ms) |
| `MCP_CONNECTION_TIMEOUT` | `30000` | Connection timeout (ms) |
| `MCP_RELAY_FALLBACK_TO_MOCK` | `true` | Use mock MCP if relay unavailable |

## Dockerfiles

| File | Purpose | Build Target |
|------|---------|-------------|
| `Dockerfile.dev` | Legacy development image | single stage |
| `Dockerfile.unified` | Multi-stage dev/prod image | `development` or `production` |
| `Dockerfile.production` | Optimized production image | single stage, release build |

## See Also

- [Docker Compose Guide](../deployment/docker-compose-guide.md) -- Compose file walkthrough
- [Docker Environment Setup](../deployment/docker-environment-setup.md) -- Local development setup
- [Docker Deployment](../deployment/docker-deployment.md) -- Production deployment guide
- [Port Configuration](./port-configuration.md) -- Detailed port allocation reference
- [Architecture](./architecture.md) -- Infrastructure architecture overview
