---
layout: default
title: "Multi-Agent System Architecture"
parent: Architecture
grand_parent: Explanations
nav_order: 26
---

# Multi-Agent System Architecture

**Version:** 1.0
**Last Updated:** November 5, 2025
**Status:** Production

---

## Overview

VisionFlow integrates with a sophisticated dual-container multi-agent system that separates AI orchestration and CLI tools from resource-intensive GUI applications. This architecture enables advanced AI-driven workflows while maintaining performance isolation and resource management.

---

## System Architecture

### High-Level Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Host Machine                              │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            Docker Network: docker_ragflow              │  │
│  │                                                         │  │
│  │  ┌──────────────────┐      ┌──────────────────┐       │  │
│  │  │   VisionFlow     │◄────►│   Agentic        │       │  │
│  │  │   Container      │      │   Workstation    │       │  │
│  │  │                  │      │                  │       │  │
│  │  │ • Graph Engine   │      │ • Claude Code    │       │  │
│  │  │ • Neo4j DB       │      │ • 13+ Skills     │       │  │
│  │  │ • Physics Sim    │      │ • MCP Protocol   │       │  │
│  │  │ • WebXR Client   │      │ • Tool Bridges   │       │  │
│  │  │                  │      │                  │       │  │
│  │  │ :9090 API        │      │ :9500 MCP TCP    │       │  │
│  │  │ :3001 Client     │      │ :3002 MCP WS     │       │  │
│  │  └──────────────────┘      └──────────────────┘       │  │
│  │           │                          │                 │  │
│  │           │                          ▼                 │  │
│  │           │                ┌──────────────────┐       │  │
│  │           │                │   GUI Tools      │       │  │
│  │           │                │   Container      │       │  │
│  │           │                │                  │       │  │
│  │           │                │ • Blender        │       │  │
│  │           │                │ • QGIS           │       │  │
│  │           │                │ • KiCad          │       │  │
│  │           │                │ • PBR Render     │       │  │
│  │           │                │ • XFCE Desktop   │       │  │
│  │           │                │                  │       │  │
│  │           │                │ :5901 VNC        │       │  │
│  │           │                │ :9876-9878 TCP   │       │  │
│  │           │                └──────────────────┘       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   Port Bindings                        │  │
│  │                                                         │  │
│  │  3001 → VisionFlow Client                             │  │
│  │  9090 → VisionFlow API                                │  │
│  │  5901 → GUI Tools VNC                                 │  │
│  │  8080 → Code Server (agentic-workstation)            │  │
│  │  2222 → SSH (agentic-workstation)                    │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## Container Breakdown

### 1. VisionFlow Container

**Purpose:** Core graph database, physics simulation, and WebXR visualization

**Key Components:**
- **Rust Backend** - Actix-web server with GPU acceleration
- **Neo4j Database** - Knowledge graph storage
- **Physics Engine** - CUDA-accelerated 3D simulation
- **WebXR Client** - React + Babylon.js visualization
- **Ontology Reasoning** - Whelk-rs inference engine

**Exposed Ports:**
- `9090` - REST API and WebSocket endpoints
- `3001` - Client development server

**Resource Limits:**
- Memory: 16GB default (configurable)
- CPUs: 4 cores default (configurable)
- GPU: NVIDIA GPU if available (CUDA 12.0)

**Network:**
- Connected to `docker_ragflow` bridge network
- Can communicate with agentic-workstation and GUI tools

---

### 2. Agentic Workstation Container

**Purpose:** AI orchestration, Claude Code, and CLI tooling

**Key Components:**
- **Claude Code** - AI assistant with MCP protocol support
- **13+ Skills** - Specialized capabilities (see Skills Reference)
- **Development Environment** - Python 3.12, Node.js 22, Rust, Deno
- **MCP Servers** - TCP (9500) and WebSocket (3002) endpoints
- **Docker Socket** - Mounted for inter-container management

**Key Features:**
- **Polyglot Runtime** - Multi-language development environment
- **Tool Bridges** - TCP bridges to GUI container applications
- **Natural Language Interface** - Skills invocable via text commands
- **SSH Access** - Port 2222 for direct shell access
- **Code Server** - Web-based VS Code on port 8080

**Exposed Ports:**
- `9500` - MCP TCP server (high-performance)
- `3002` - MCP WebSocket server
- `8080` - Code Server web interface
- `2222` - SSH access

**Mounted Volumes:**
- `/var/run/docker.sock` - Docker socket for container management
- `/workspace` - Shared project workspace
- `/home/devuser/.claude` - Claude Code configuration and skills

**Resource Limits:**
- Memory: 16GB default
- CPUs: 4 cores default

---

### 3. GUI Tools Container

**Purpose:** Resource-intensive GUI applications with VNC access

**Key Components:**
- **Blender 4.x** - 3D modeling and rendering
- **QGIS 3.x** - Geospatial analysis
- **KiCad** - Electronic design automation
- **PBR Generator** - Physically-based rendering textures
- **XFCE Desktop** - Lightweight desktop environment

**Architecture Pattern:**
```
┌─────────────────────────────────────┐
│      GUI Tools Container            │
│                                     │
│  ┌────────────────────────────┐    │
│  │   XFCE Desktop (VNC)       │    │
│  │                            │    │
│  │  ┌──────────┐ ┌─────────┐ │    │
│  │  │ Blender  │ │  QGIS   │ │    │
│  │  └────┬─────┘ └────┬────┘ │    │
│  │       │            │      │    │
│  │       ▼            ▼      │    │
│  │  ┌────────────────────┐  │    │
│  │  │  MCP Servers       │  │    │
│  │  │  (TCP Ports)       │  │    │
│  │  └────────────────────┘  │    │
│  └────────────────────────────┘    │
│                                     │
│  :5901 VNC                          │
│  :9876 Blender MCP                  │
│  :9877 QGIS MCP                     │
│  :9878 PBR MCP                      │
└─────────────────────────────────────┘
```

**Exposed Ports:**
- `5901` - VNC server (password: turboflow)
- `9876` - Blender MCP bridge
- `9877` - QGIS MCP bridge
- `9878` - PBR Generator MCP bridge

**Resource Requirements:**
- Memory: 8GB minimum (16GB recommended for Blender rendering)
- CPUs: 2 cores minimum (4+ for rendering)
- Display: Requires X11/VNC for GUI applications

---

## Communication Protocols

### Model Context Protocol (MCP)

VisionFlow uses MCP for AI agent communication with tools and services.

**MCP Endpoints:**

| Endpoint | Type | Port | Use Case |
|----------|------|------|----------|
| TCP Server | High-performance | 9500 | Inter-container, production |
| WebSocket | Web-compatible | 3002 | Browser clients, debugging |
| Stdio | Direct | N/A | Local skill invocation |

**MCP Message Flow:**
```
┌──────────────┐
│ Claude Code  │
└──────┬───────┘
       │
       │ JSON-RPC 2.0
       ▼
┌──────────────┐
│  MCP Server  │
└──────┬───────┘
       │
       ├──────► Skill: Docker Manager
       ├──────► Skill: Wardley Mapper
       ├──────► Skill: Chrome DevTools
       └──────► ... (10+ more skills)
```

**MCP Request Example:**
```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "tools/call",
  "params": {
    "name": "visionflow_build",
    "arguments": {
      "no_cache": true,
      "profile": "dev"
    }
  }
}
```

---

### Docker Socket Communication

The agentic-workstation container manages VisionFlow through the Docker socket:

```
┌──────────────────────────────────┐
│  Agentic Workstation             │
│                                  │
│  ┌────────────────────────────┐  │
│  │  Docker Manager Skill      │  │
│  └────────┬───────────────────┘  │
│           │                      │
│           ▼                      │
│  ┌────────────────────────────┐  │
│  │  /var/run/docker.sock      │  │
│  │  (mounted from host)       │  │
│  └────────┬───────────────────┘  │
└───────────┼──────────────────────┘
            │
            │ Docker API
            ▼
    ┌────────────────────────────┐
    │  Docker Daemon (host)      │
    └────────┬───────────────────┘
             │
             │ Controls
             ▼
    ┌────────────────────────────┐
    │  VisionFlow Container      │
    │  • Build                   │
    │  • Start/Stop/Restart      │
    │  • Exec commands           │
    │  • Stream logs             │
    └────────────────────────────┘
```

**Security Considerations:**
- Docker socket access grants root-equivalent privileges
- Only trusted code should have socket access
- Consider using Docker socket proxy for production

---

### GUI Application Bridges

GUI tools communicate via TCP bridges:

```
Agentic Workstation              GUI Tools Container
┌────────────────────┐          ┌──────────────────┐
│                    │          │                  │
│  blender-client.js ├─TCP─────►│  Blender MCP     │
│                    │  :9876   │  (Python addon)  │
│                    │          │                  │
│  qgis-client.py    ├─TCP─────►│  QGIS MCP        │
│                    │  :9877   │  (Plugin)        │
│                    │          │                  │
│  pbr-client.py     ├─TCP─────►│  PBR Generator   │
│                    │  :9878   │  (Server)        │
└────────────────────┘          └──────────────────┘
```

**Bridge Pattern:**
1. Client sends JSON-RPC request via TCP
2. Server executes in GUI application context
3. Results returned via same TCP connection
4. Async execution supported

---

## Data Flow Patterns

### Pattern 1: Development Workflow

```
Developer
    │
    ├─► Edit VisionFlow code
    │
    ▼
Claude Code (Natural Language)
    │
    ├─► "Use Docker Manager to rebuild VisionFlow"
    │
    ▼
Docker Manager Skill
    │
    ├─► Execute: docker build, docker restart
    │
    ▼
VisionFlow Container
    │
    ├─► Rebuild Rust backend
    ├─► Restart server
    └─► Ready at localhost:3001
```

### Pattern 2: Strategic Analysis

```
User Request
    │
    ├─► "Create a Wardley map for our architecture"
    │
    ▼
Wardley Mapper Skill
    │
    ├─► Extract components from codebase
    ├─► Analyze dependencies
    ├─► Assess evolution stages
    └─► Generate SVG map
    │
    ▼
Output: architecture-wardley-map.svg
```

### Pattern 3: 3D Visualization

```
Natural Language
    │
    ├─► "Create a 3D network graph in Blender"
    │
    ▼
Blender Skill (TCP Client)
    │
    ├─► Send Python script via TCP
    │
    ▼
Blender MCP Server (GUI Container)
    │
    ├─► Execute bpy commands
    ├─► Create geometry
    └─► Render image
    │
    ▼
Output: network-graph.png (VNC visible)
```

---

## Scalability & Performance

### Resource Allocation

**Default Configuration:**
```yaml
visionflow:
  mem_limit: 16g
  cpus: 4
  gpus: all  # If NVIDIA GPU available

agentic-workstation:
  mem_limit: 16g
  cpus: 4

gui-tools:
  mem_limit: 8g
  cpus: 2
```

**Minimum Requirements:**
- Total RAM: 24GB (8GB VisionFlow + 8GB agentic + 8GB GUI)
- CPUs: 6 cores (2 per container)
- Disk: 20GB for images + workspace

**Recommended for Production:**
- Total RAM: 48GB (16+16+16)
- CPUs: 12 cores (4+4+4)
- GPU: NVIDIA RTX 3080+ for physics and rendering
- Disk: NVMe SSD with 100GB+

### Performance Optimizations

1. **MCP TCP over WebSocket**
   - 60% lower latency
   - 2x higher throughput
   - Use for inter-container communication

2. **GPU Acceleration**
   - VisionFlow: CUDA physics (55x speedup)
   - GUI Container: Blender Cycles GPU rendering

3. **Docker Socket**
   - Direct API access (no HTTP overhead)
   - Native Docker commands
   - Stream logs efficiently

4. **Resource Isolation**
   - Separate containers prevent resource contention
   - GUI tools don't slow down VisionFlow
   - Independent scaling per service

---

## Security Architecture

### Container Isolation

```
┌─────────────────────────────────┐
│      Host Security Boundary     │
│                                 │
│  ┌──────────┐  ┌──────────┐    │
│  │ Vision   │  │ Agentic  │    │
│  │ Flow     │  │ Work.    │    │
│  │          │  │          │    │
│  │ User:    │  │ User:    │    │
│  │ vfuser   │  │ devuser  │    │
│  │ UID:1000 │  │ UID:1000 │    │
│  └──────────┘  └────┬─────┘    │
│                     │           │
│                     ▼           │
│          ┌─────────────────┐   │
│          │ docker.sock     │   │
│          │ (privileged)    │   │
│          └─────────────────┘   │
└─────────────────────────────────┘
```

**Security Measures:**
- Non-root users in all containers (UID 1000)
- Read-only root filesystems where possible
- Network segmentation via Docker networks
- Secret management via environment variables
- MCP authentication tokens (production)

**⚠️ Important:**
- Docker socket grants root-equivalent access
- Only mount in trusted environments
- Consider socket proxy for production
- Use secrets management for tokens

---

## Monitoring & Observability

### Log Aggregation

All containers log to stdout/stderr for unified monitoring:

```bash
# View all VisionFlow logs
docker logs visionflow-container

# View agentic-workstation logs
docker logs agentic-workstation

# View GUI container logs
docker logs gui-tools-container

# Follow logs in real-time
docker logs -f agentic-workstation

# Last 100 lines
docker logs --tail 100 visionflow-container
```

### Health Checks

```bash
# Container health status
docker ps --filter health=healthy

# MCP server health
curl http://localhost:9500/health

# VisionFlow API health
curl http://localhost:9090/api/health

# Check GUI tools
xdpyinfo -display :1 (from GUI container)
```

### Performance Metrics

```bash
# Container resource usage
docker stats

# Network throughput
docker exec agentic-workstation iftop

# GPU usage (if available)
nvidia-smi
```

---

## Disaster Recovery

### Backup Strategy

**Persistent Volumes:**
- `/workspace` - Project files and code
- `~/.claude` - Skills and configuration
- Neo4j data - VisionFlow database

**Backup Commands:**
```bash
# Backup workspace
docker cp agentic-workstation:/workspace ./backup/workspace

# Backup Claude skills
docker cp agentic-workstation:/home/devuser/.claude ./backup/claude

# Backup Neo4j (from VisionFlow)
docker exec visionflow-container neo4j-admin dump
```

### Recovery Procedures

1. **Complete System Failure:**
   ```bash
   ./multi-agent.sh stop
   ./multi-agent.sh cleanup  # WARNING: Deletes all data
   ./multi-agent.sh build
   # Restore backups
   docker cp ./backup/workspace agentic-workstation:/workspace
   ```

2. **Single Container Failure:**
   ```bash
   docker restart <container-name>
   # If restart fails:
   docker stop <container-name>
   docker rm <container-name>
   ./multi-agent.sh start
   ```

3. **Network Issues:**
   ```bash
   docker network inspect docker_ragflow
   # If broken:
   docker network rm docker_ragflow
   docker network create docker_ragflow
   ./multi-agent.sh restart
   ```

---

## References

- [Multi-Agent Skills Reference](../../guides/multi-agent-skills.md)
- [Docker Environment Setup](../../guides/docker-environment-setup.md)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Security Best Practices](../../guides/security.md)

---

**Architecture Version:** 1.0
**Last Review:** November 5, 2025
**Maintainer:** VisionFlow Architecture Team
