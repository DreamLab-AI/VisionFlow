# Multi-Agent Docker - Merged GUI Container

## Overview

This container merges the headless agent environment with VNC-accessible GUI applications into a single high-performance container. All GUI tools (Blender, QGIS, Playwright, PBR Generator) run locally with MCP servers accessible both internally and externally.

## Architecture Changes

### Previous Setup
- **gui-tools-service**: Separate container running Blender, QGIS, Playwright, PBR Generator with VNC
- **multi-agent**: Headless container with proxy scripts connecting to GUI container

### New Unified Setup
- **Single Container**: All services run in one optimized container
- **Local MCP Servers**: Direct communication, no proxying overhead
- **Integrated GUI**: VNC server provides desktop access to all tools
- **High Performance**: 32GB RAM default, 16 CPU cores, full NVIDIA GPU support

## Features

### GUI Applications (via VNC on port 5901)
- **Blender 4.5.1**: 3D modeling with MCP addon pre-installed
- **QGIS**: GIS application with MCP plugin
- **XFCE Desktop**: Lightweight desktop environment
- **Chrome**: Full browser with DevTools MCP support
- **Playwright**: Browser automation (Chromium, Firefox, WebKit)

### MCP Servers (Local)
- Port 9876: Blender MCP
- Port 9877: QGIS MCP
- Port 9878: PBR Generator MCP
- Port 9879: Playwright MCP
- Port 9500: Main MCP TCP Server
- Port 3002: WebSocket Bridge

### Development Tools
- Python 3.12 with ML stack (PyTorch, CUDA 12.9.1)
- Node.js 22+ with claude-flow, playwright, goalie
- Rust toolchain
- Docker-in-Docker support
- Full LaTeX environment

## Performance Configuration

### Default Resources
```yaml
Memory: 32GB (configurable via DOCKER_MEMORY)
CPUs: 16 cores (configurable via DOCKER_CPUS)
Shared Memory: 4GB (for GUI apps and browsers)
GPU: All NVIDIA GPUs with full capabilities
```

### Environment Variables
Set in `.env` file:
```bash
DOCKER_MEMORY=32g      # Adjust based on available RAM
DOCKER_CPUS=16         # Adjust based on available cores
HOST_UID=1000          # Match your user ID
HOST_GID=1000          # Match your group ID
```

## Usage

### Build and Start
```bash
cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker
docker-compose up -d --build
```

### Access VNC Desktop
Connect to `localhost:5901` with any VNC client (password not required)

### Access MCP Services
All MCP servers are accessible at `localhost:987x` where x is 6-9

### Enter Container Shell
```bash
docker exec -it multi-agent-container bash
```

### View Logs
```bash
docker-compose logs -f multi-agent
```

## Directory Structure

```
multi-agent-docker/
├── Dockerfile                 # Unified container definition
├── docker-compose.yml         # Single service configuration
├── entrypoint.sh             # Startup script with GUI initialization
├── supervisord.conf          # Service management (MCP servers)
├── workspace/                # Persistent workspace
├── core-assets/              # Scripts and tools
└── gui-based-tools-docker/   # GUI tool assets (copied into container)
    ├── blender-4.5.1-linux-x64.tar.xz
    ├── tessellating-pbr-generator/
    ├── addon.py              # Blender MCP addon
    ├── autostart.py          # Blender startup script
    ├── playwright-mcp-server.js
    ├── qgis-mcp-server.js
    └── pbr-mcp-simple.py
```

## Migration Notes

### Removed Components
- `gui-tools-service` container definition removed from docker-compose.yml
- Proxy scripts no longer needed (blender-mcp-proxy.js, qgis-mcp-proxy.js, etc.)
- Network communication between containers eliminated

### Benefits
1. **Reduced Latency**: Direct local MCP communication
2. **Simplified Architecture**: Single container to manage
3. **Better Resource Utilization**: Shared GPU/RAM pool
4. **Easier Debugging**: All logs in one place
5. **Faster Startup**: No inter-container dependencies

## Troubleshooting

### VNC Connection Issues
```bash
# Check VNC server status
docker exec multi-agent-container ps aux | grep x11vnc

# Restart VNC manually
docker exec multi-agent-container pkill x11vnc
docker exec multi-agent-container x11vnc -display :1 -nopw -forever -xkb -listen 0.0.0.0 -rfbport 5901 &
```

### MCP Server Issues
```bash
# Check supervisor status
docker exec -it multi-agent-container supervisorctl status

# Restart specific MCP server
docker exec -it multi-agent-container supervisorctl restart playwright-mcp-server
```

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker exec multi-agent-container nvidia-smi

# Check GPU environment
docker exec multi-agent-container env | grep NVIDIA
```

## Performance Tuning

For maximum performance on powerful systems:

```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 64g        # If you have >64GB RAM
      cpus: "32"         # If you have 32+ cores
```

```yaml
# Increase shared memory for heavy browser workloads
shm_size: '8gb'
```
