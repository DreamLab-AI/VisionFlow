# Multi-Agent Docker - Unified Container

A high-performance, GPU-accelerated development environment combining AI tools, GUI applications, and MCP servers in a single optimized container.

## Quick Start

```bash
# Build and start
docker-compose up -d --build

# Access VNC desktop
# VNC: localhost:5901 (password: password)
# Web: http://localhost:6901

# Enter container shell
docker exec -it multi-agent-container bash
```

## Features

### 🖥️ GUI Desktop (VNC)
- **XFCE Desktop** - Lightweight environment with D-Bus support
- **Resolution** - 1600x1200 (configurable)
- **Access** - VNC port 5901, noVNC port 6901
- **VNC Password** - `password`
- **User Password** - `password` (for desktop prompts)

### 🛠️ Development Tools
- **Python 3.12** - ML stack (PyTorch, CUDA 12.9.1)
- **Node.js 22+** - claude-flow, playwright, goalie
- **Rust** - Full toolchain
- **Blender 4.5.1** - CLI accessible: `blender`
- **QGIS** - Geospatial tools
- **LaTeX** - Full environment

### 🔌 MCP Servers (Local)
- **Port 9876** - Blender MCP
- **Port 9877** - QGIS MCP
- **Port 9878** - PBR Generator MCP
- **Port 9879** - Playwright MCP
- **Port 9880** - Web Summary MCP (Google AI Studio)
- **Port 9500** - Main MCP TCP Server
- **Port 3002** - WebSocket Bridge

### 📦 MCP Tools (Direct)
- `imagemagick-mcp` - Image creation/manipulation
- `kicad-mcp` - Electronic design automation
- `ngspice-mcp` - Circuit simulation

## Configuration

### Environment Variables (.env)
```bash
DOCKER_MEMORY=32g                      # RAM allocation
DOCKER_CPUS=16                         # CPU cores
HOST_UID=1000                          # User ID
HOST_GID=1000                          # Group ID
EXTERNAL_DIR=/path/to/projects         # External project mount
```

### External Projects
Projects from `EXTERNAL_DIR` are mounted at `/workspace/ext` inside the container, accessible from both VNC desktop and CLI.

## Architecture

### Single Unified Container
```
┌──────────────────────────────────────────┐
│  multi-agent-container                   │
│                                          │
│  VNC/XFCE Desktop ─┬─ Development Tools │
│  Blender, QGIS    │  Python/Node/Rust  │
│  Chromium         │  Claude CLI        │
│                   │                     │
│  Local MCP Servers (9876-9879)         │
│  Direct GPU Access (NVIDIA)            │
└──────────────────────────────────────────┘
```

**Benefits:**
- No network overhead (previous dual-container setup eliminated)
- Direct GPU access for all tools
- Unified logging and management
- Faster startup and MCP communication

## Resource Configuration

### Default Allocation
- **Memory** - 32GB (configurable)
- **CPUs** - 16 cores (configurable)
- **Shared Memory** - 4GB (for GUI/browsers)
- **GPU** - All NVIDIA GPUs with full capabilities

### High-Performance Tuning
For systems with >64GB RAM:
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 64g
      cpus: "32"
shm_size: '8gb'
```

## Directory Structure

```
multi-agent-docker/
├── README.md                  # This file
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Service configuration
├── entrypoint.sh             # Container startup
├── supervisord.conf          # Service management
├── setup-workspace.sh        # Workspace initialization
├── docs/                     # Documentation
│   ├── UNIFIED-CONTAINER-MIGRATION.md
│   ├── SECURITY.md
│   └── ...
├── core-assets/              # Scripts and tools
│   ├── scripts/              # MCP servers and utilities
│   ├── mcp-tools/            # Direct MCP tools
│   └── mcp.json             # MCP configuration
├── gui-tools-assets/         # GUI MCP server scripts
│   ├── addon.py              # Blender MCP addon
│   ├── playwright-mcp-server.js
│   ├── qgis-mcp-server.js
│   └── pbr-mcp-simple.py
└── workspace/                # Persistent workspace
    └── ext/                  # External projects
```

## Usage

### VNC Desktop
```bash
# VNC Client
Host: localhost:5901
Password: password

# noVNC Browser
URL: http://localhost:6901
Password: password
```

### Container Shell
```bash
# As dev user
docker exec -it multi-agent-container bash

# As root
docker exec -u root -it multi-agent-container bash
```

### MCP Services
```bash
# Check all services
docker exec multi-agent-container supervisorctl status

# Test MCP tool
docker exec -u dev multi-agent-container \
  python3 /app/core-assets/mcp-tools/imagemagick_mcp.py
```

## Troubleshooting

### VNC Issues
```bash
# Check VNC and desktop status
docker exec multi-agent-container supervisorctl status vnc xfce dbus

# Restart desktop
docker exec multi-agent-container supervisorctl restart vnc xfce dbus
```

### MCP Server Issues
```bash
# Check supervisor status
docker exec -it multi-agent-container supervisorctl status

# Restart specific server
docker exec -it multi-agent-container supervisorctl restart playwright-mcp-server
```

### GPU Issues
```bash
# Verify NVIDIA runtime
docker exec multi-agent-container nvidia-smi

# Check GPU environment
docker exec multi-agent-container env | grep NVIDIA
```

## Documentation

- **[Migration Guide](docs/UNIFIED-CONTAINER-MIGRATION.md)** - Architecture changes and verification
- **[Security Guide](docs/SECURITY.md)** - Authentication and security features
- **[Agent Briefing](docs/AGENT-BRIEFING.md)** - Agent-specific documentation
- **[Cleanup Candidates](docs/CLEANUP-CANDIDATES.md)** - Legacy files removed

## Development

### Helper Scripts
- `./multi-agent.sh shell` - Enter container as dev user
- `./setup-workspace.sh` - Initialize workspace (runs automatically)
- `./mcp-helper.sh` - MCP utility commands

### Logs
```bash
# Container logs
docker-compose logs -f multi-agent

# Service logs
docker exec multi-agent-container supervisorctl tail -f vnc
```

## Performance

Optimized for:
- **AI Development** - GPU acceleration, ML frameworks
- **3D Modeling** - Blender with GPU rendering
- **Browser Automation** - Playwright with Chromium/Firefox/WebKit
- **Geospatial Analysis** - QGIS with full capabilities
- **Multi-Agent Systems** - Claude-Flow orchestration

## License

See project root for licensing information.

## Support

For issues or questions:
1. Check the [Migration Guide](docs/UNIFIED-CONTAINER-MIGRATION.md)
2. Review [troubleshooting](#troubleshooting) section
3. File an issue in the project repository
