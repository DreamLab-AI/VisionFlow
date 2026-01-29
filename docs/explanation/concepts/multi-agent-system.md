---
title: Multi-Agent System
description: Understanding VisionFlow's dual-container multi-agent architecture for AI orchestration and development workflows
category: explanation
tags:
  - multi-agent
  - docker
  - mcp
  - architecture
related-docs:
  - concepts/actor-model.md
  - architecture/agents/multi-agent.md
  - guides/multi-agent-skills.md
updated-date: 2025-12-18
difficulty-level: intermediate
---

# Multi-Agent System

VisionFlow integrates with a sophisticated multi-agent system that separates AI orchestration from resource-intensive GUI applications, enabling advanced AI-driven development workflows.

---

## Core Concept

The multi-agent architecture addresses several challenges:

1. **Resource isolation**: GUI applications (Blender, QGIS) shouldn't slow down VisionFlow
2. **AI orchestration**: Claude Code needs access to tools without coupling to the main application
3. **Skill composition**: Modular skills combine for complex workflows
4. **Development workflow**: Natural language commands drive development tasks

The solution: three cooperating containers on a shared Docker network.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Host Machine                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Docker Network: docker_ragflow               │ │
│  │                                                         │ │
│  │  ┌──────────────────┐       ┌──────────────────┐       │ │
│  │  │   VisionFlow     │ ←───→ │    Agentic       │       │ │
│  │  │   Container      │       │   Workstation    │       │ │
│  │  │                  │       │                  │       │ │
│  │  │ • Graph Engine   │       │ • Claude Code    │       │ │
│  │  │ • Neo4j DB       │       │ • 13+ Skills     │       │ │
│  │  │ • Physics Sim    │       │ • MCP Protocol   │       │ │
│  │  │ • WebXR Client   │       │ • Tool Bridges   │       │ │
│  │  │                  │       │                  │       │ │
│  │  │ :9090 API        │       │ :9500 MCP TCP    │       │ │
│  │  │ :3001 Client     │       │ :3002 MCP WS     │       │ │
│  │  └──────────────────┘       └────────┬─────────┘       │ │
│  │                                      │                  │ │
│  │                                      ↓                  │ │
│  │                             ┌──────────────────┐       │ │
│  │                             │   GUI Tools      │       │ │
│  │                             │   Container      │       │ │
│  │                             │                  │       │ │
│  │                             │ • Blender        │       │ │
│  │                             │ • QGIS           │       │ │
│  │                             │ • XFCE Desktop   │       │ │
│  │                             │                  │       │ │
│  │                             │ :5901 VNC        │       │ │
│  │                             │ :9876-9878 MCP   │       │ │
│  │                             └──────────────────┘       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Container Roles

### VisionFlow Container

**Purpose**: Core graph database, physics simulation, and WebXR visualisation

**Components**:
- Rust backend (Actix-web server)
- Neo4j database
- CUDA physics engine
- React + Babylon.js client
- whelk-rs ontology reasoner

**Resource profile**:
- Memory: 16 GB
- CPUs: 4 cores
- GPU: NVIDIA CUDA 12.0

### Agentic Workstation Container

**Purpose**: AI orchestration, Claude Code, and development tooling

**Components**:
- Claude Code with MCP protocol
- 13+ specialised skills
- Polyglot runtime (Python, Node.js, Rust, Deno)
- Docker socket access

**Key capabilities**:
- Natural language interface
- Tool bridges to GUI container
- Container management via Docker socket
- SSH and Code Server access

### GUI Tools Container

**Purpose**: Resource-intensive GUI applications

**Components**:
- Blender 4.x (3D modelling)
- QGIS 3.x (geospatial)
- XFCE desktop (VNC-accessible)
- MCP bridges for remote control

---

## Model Context Protocol (MCP)

MCP enables Claude Code to interact with tools and services.

### Endpoints

| Endpoint | Type | Port | Use Case |
|----------|------|------|----------|
| TCP Server | High-performance | 9500 | Inter-container, production |
| WebSocket | Web-compatible | 3002 | Browser clients, debugging |
| Stdio | Direct | N/A | Local skill invocation |

### Message Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Message Flow                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Claude Code                                                 │
│       ↓                                                      │
│  "Build VisionFlow with the dev profile"                    │
│       ↓                                                      │
│  MCP Router                                                  │
│       ↓                                                      │
│  Tool: docker_manager.visionflow_build                      │
│       ↓                                                      │
│  Docker Manager Skill                                        │
│       ↓                                                      │
│  Docker Socket API                                          │
│       ↓                                                      │
│  VisionFlow Container: docker build ...                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### MCP Request Example

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

## Skill System

Skills encapsulate reusable capabilities invokable via natural language.

### Skill Categories

| Category | Skills | Purpose |
|----------|--------|---------|
| Development | Docker Manager, Git | Container and version control |
| Visualisation | Blender, QGIS | 3D and geospatial |
| Analysis | Wardley Mapper | Strategic analysis |
| Debugging | Chrome DevTools | Browser inspection |
| Communication | Slack, Discord | Team coordination |

### Skill Invocation

```
User: "Create a Wardley map of our architecture"

Claude Code:
1. Parses intent
2. Selects Wardley Mapper skill
3. Extracts components from codebase
4. Analyses dependencies
5. Assesses evolution stages
6. Generates SVG map

Output: architecture-wardley-map.svg
```

### Skill Definition Structure

```yaml
# .claude/skills/wardley-mapper.yaml
name: wardley-mapper
description: Generate Wardley maps from requirements
triggers:
  - "create wardley map"
  - "analyse architecture evolution"
tools:
  - extract_components
  - analyse_dependencies
  - generate_svg
parameters:
  format:
    type: string
    enum: [svg, png, json]
    default: svg
```

---

## GUI Application Bridges

GUI tools communicate via TCP bridges for remote AI control.

### Architecture

```
Agentic Workstation              GUI Tools Container
┌────────────────────┐          ┌──────────────────┐
│                    │          │                  │
│  blender-client.js ├──TCP────►│  Blender MCP     │
│                    │  :9876   │  (Python addon)  │
│                    │          │                  │
│  qgis-client.py    ├──TCP────►│  QGIS MCP        │
│                    │  :9877   │  (Plugin)        │
│                    │          │                  │
└────────────────────┘          └──────────────────┘
```

### Bridge Protocol

1. Client sends JSON-RPC request via TCP
2. Server executes in GUI application context
3. Results returned via same TCP connection
4. Async execution supported

### Example: Blender Scene Creation

```python
# From agentic workstation
response = blender_client.call("create_mesh", {
    "type": "CUBE",
    "location": [0, 0, 0],
    "scale": [1, 1, 1]
})

# Executes in Blender
# Returns: {"object_id": "Cube.001", "success": true}
```

---

## Docker Socket Communication

The agentic workstation manages VisionFlow through the Docker socket.

### Capabilities

- Build containers
- Start/stop/restart services
- Execute commands inside containers
- Stream logs
- Inspect container state

### Security Considerations

- Docker socket grants root-equivalent privileges
- Only mount in trusted environments
- Consider socket proxy for production
- Use secrets management for tokens

---

## Data Flow Patterns

### Development Workflow

```
Developer edit → Claude Code → Docker Manager → VisionFlow rebuild
```

### Strategic Analysis

```
User request → Wardley Mapper → Codebase analysis → SVG generation
```

### 3D Visualisation

```
Natural language → Blender Skill → TCP bridge → Blender execution → VNC visible
```

---

## Resource Management

### Default Allocation

| Container | Memory | CPUs | GPU |
|-----------|--------|------|-----|
| VisionFlow | 16 GB | 4 | NVIDIA |
| Agentic Workstation | 16 GB | 4 | - |
| GUI Tools | 8 GB | 2 | Optional |

### Minimum Requirements

- Total RAM: 24 GB
- CPUs: 6 cores
- Disk: 20 GB for images + workspace

### Performance Optimisations

1. **MCP TCP over WebSocket**: 60% lower latency
2. **GPU acceleration**: VisionFlow CUDA physics (55x speedup)
3. **Resource isolation**: Separate containers prevent contention

---

## Agent Graph Integration

VisionFlow visualises AI agents as nodes in the knowledge graph:

### Agent Representation

```
┌─────────────────────────────────────────────────────────────┐
│                   Agent Graph Nodes                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Claude Agent (Agent Type)                                  │
│  ├── Status: active | idle | processing                     │
│  ├── Current Task: "Building VisionFlow"                    │
│  ├── Token Usage: 15,234                                    │
│  └── Connections:                                           │
│      ├── → Docker Manager (tool)                            │
│      ├── → VisionFlow API (service)                         │
│      └── → Knowledge Graph (data source)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Dual-Graph Broadcasting

Both knowledge graph and agent graph share the WebSocket stream:

```
Node ID flags:
  Bit 31 (0x80000000): Agent node
  Bit 30 (0x40000000): Knowledge node
  Bits 0-29: Actual node ID
```

---

## Monitoring and Debugging

### Log Access

```bash
# VisionFlow logs
docker logs visionflow-container

# Agentic workstation logs
docker logs agentic-workstation

# Follow in real-time
docker logs -f agentic-workstation
```

### Health Checks

```bash
# MCP server health
curl http://localhost:9500/health

# VisionFlow API health
curl http://localhost:9090/api/health

# Container resource usage
docker stats
```

---

## Related Concepts

- **[Actor Model](actor-model.md)**: How VisionFlow actors integrate with MCP
- **[Real-Time Sync](real-time-sync.md)**: Agent graph broadcasting
- **[Hexagonal Architecture](hexagonal-architecture.md)**: Multi-agent as external adapter
