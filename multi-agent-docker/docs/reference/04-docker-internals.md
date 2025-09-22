# Reference: Docker Internals

This document provides a breakdown of the `Dockerfile` and `docker-compose.yml` files, explaining the container build process and service orchestration.

### `docker-compose.yml`

This file defines and orchestrates the two main services: `multi-agent` and `gui-tools-service`.

#### Service: `gui-tools-service`
-   **Image**: Built from `gui-based-tools-docker/Dockerfile`.
-   **Purpose**: To run heavy GUI applications in an isolated environment.
-   **Key Features**:
    -   **GPU Support**: Configured with `deploy.resources` to grant access to NVIDIA GPUs on the host.
    -   **Ports**: Exposes VNC (`5901`) and TCP ports for each GUI application's MCP server (`9876`, `9877`, `9878`).
    -   **Healthcheck**: Considers the service healthy only when all three MCP servers (Blender, QGIS, PBR) are accepting connections.
    -   **Network**: Attached to the `docker_ragflow` network to communicate with the `multi-agent` service.

#### Service: `multi-agent`
-   **Image**: Built from the root `Dockerfile`.
-   **Purpose**: The primary environment for AI agents, CLI tools, and development runtimes.
-   **Key Features**:
    -   **Build Arguments**: Uses `HOST_UID` and `HOST_GID` from the `.env` file to create a `dev` user with matching permissions to the host user, preventing file ownership issues.
    -   **Environment Variables**: Injects a large number of variables from the `.env` file to configure everything from application hosts to security settings.
    -   **Volumes**:
        -   `./workspace:/workspace`: Mounts the local workspace for persistent project files.
        -   `~/.claude:/home/dev/.claude`: Mounts Claude configuration directory for authentication.
        -   `~/.claude.json:/home/dev/.claude.json:ro`: Mounts main Claude config file.
        -   `${EXTERNAL_DIR}:/workspace/ext`: Mounts external directory for user projects.
    -   **Entrypoint**: Overridden to use `entrypoint-wrapper.sh` for pre-start initialization.
    -   **Healthcheck**: Uses the comprehensive `health-check.sh` script to verify the status of all core services.

### `Dockerfile` (multi-agent)

This file defines the build process for the main `multi-agent-container`.

#### Build Stages & Key Layers

1.  **Base Image**: `nvidia/cuda:12.9.1-devel-ubuntu24.04`
    -   Provides a modern Ubuntu base with the full CUDA development toolkit for GPU acceleration.

2.  **System Dependencies & Repositories**:
    -   Installs essential build tools, network utilities, and libraries (`build-essential`, `git`, `curl`, `supervisor`).
    -   Adds PPAs and repositories for up-to-date versions of Python (`deadsnakes`), Node.js (`nodesource`), and KiCad.

3.  **Non-Root User Creation**:
    -   A critical security step. It creates a `dev` user with a UID/GID matching the host (passed via build args). This ensures that files created in the mounted `/workspace` have the correct ownership on the host.

4.  **Runtimes & Toolchains**:
    -   **Python**: Installs Python 3.12 and creates a virtual environment (`/opt/venv312`).
    -   **Node.js**: Installs Node.js 22+.
    -   **Rust**: Installs the latest Rust toolchain in the `dev` user's home directory.
    -   **Deno**: Installs the Deno runtime.

5.  **Specialized Tooling Installation**:
    -   **Graphics & 3D**: `imagemagick`, `inkscape`, `ffmpeg`, `colmap`.
    -   **EDA**: `kicad`, `ngspice`.
    -   **AI/ML**: Python libraries like `torch` and `tensorflow` are installed via `requirements.txt` into the venv. The PyTorch installation specifically targets the CUDA-enabled version.

6.  **Application & Script Setup**:
    -   Copies in all the `core-assets` and helper scripts (`setup-workspace.sh`, `entrypoint.sh`, etc.).
    -   Installs global and local Node.js packages defined in `package.json`, including `claude-flow`.
    -   Sets up the `dev` user's shell environment (`.bashrc`) to include all necessary paths and aliases.

7.  **Final Configuration**:
    -   Copies the `supervisord.conf` file to manage background services.
    -   Sets the `ENTRYPOINT` to the wrapper script.
    -   Sets the default `CMD` to start `supervisord`.

### Key Docker Compose Configuration

```yaml
version: '3.8'

services:
  gui-tools-service:
    build:
      context: ./gui-based-tools-docker
    container_name: gui-tools-container
    ports:
      - "5901:5901"  # VNC
      - "9876:9876"  # Blender MCP
      - "9877:9877"  # QGIS MCP
      - "9878:9878"  # PBR Generator MCP
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  multi-agent:
    build:
      context: .
      args:
        - HOST_UID=${HOST_UID:-1000}
        - HOST_GID=${HOST_GID:-1000}
    container_name: multi-agent-container
    ports:
      - "3000:3000"  # Claude Flow UI
      - "3002:3002"  # MCP WebSocket Bridge
      - "9500:9500"  # MCP TCP Server
      - "9501:9501"  # MCP Health Check
    volumes:
      - ./workspace:/workspace
      - ~/.claude:/home/dev/.claude
      - ~/.claude.json:/home/dev/.claude.json:ro
    environment:
      - CLAUDE_CONFIG_DIR=/home/dev/.claude
      - BLENDER_HOST=gui-tools-service
      - QGIS_HOST=gui-tools-service
```

### Supervisord Configuration

The `supervisord.conf` file manages background services:

```ini
[supervisord]
nodaemon=true
logfile=/app/logs/supervisord.log

[program:mcp-tcp-server]
command=node /app/core-assets/scripts/mcp-tcp-server.js
autostart=true
autorestart=true
stdout_logfile=/app/mcp-logs/mcp-tcp-server.log
stderr_logfile=/app/mcp-logs/mcp-tcp-server.error.log
environment=NODE_ENV="%(ENV_NODE_ENV)s",TCP_AUTH_TOKEN="%(ENV_TCP_AUTH_TOKEN)s"

[program:mcp-ws-relay]
command=node /app/core-assets/scripts/mcp-ws-relay.js
autostart=true
autorestart=true
stdout_logfile=/app/mcp-logs/mcp-ws-relay.log
stderr_logfile=/app/mcp-logs/mcp-ws-relay.error.log
environment=NODE_ENV="%(ENV_NODE_ENV)s",WS_AUTH_TOKEN="%(ENV_WS_AUTH_TOKEN)s"
```

### Build Process Flow

1. **Host Machine**: User runs `./multi-agent.sh build`
2. **Docker Compose**: Reads `docker-compose.yml` and `.env` file
3. **Docker Build**: 
   - Builds `gui-tools-service` from its Dockerfile
   - Builds `multi-agent` from root Dockerfile with build args
4. **Layer Caching**: Docker caches layers for faster rebuilds
5. **Image Creation**: Final images are tagged and ready for use

### Container Startup Sequence

1. **Docker Compose Up**: `./multi-agent.sh start` triggers `docker-compose up -d`
2. **Container Creation**: Docker creates containers from images
3. **Volume Mounting**: Host directories are mounted into containers
4. **Network Creation**: `docker_ragflow` network is created if not exists
5. **Entrypoint Execution**: 
   - `gui-tools-service`: Starts X server, VNC, and GUI apps
   - `multi-agent`: Runs `entrypoint-wrapper.sh`
6. **Service Initialization**:
   - Permission fixes
   - Claude authentication setup
   - Workspace initialization
   - Automated setup scripts
7. **Supervisord Start**: Background services begin
8. **Health Checks**: Containers report healthy when all services are up

### Performance Optimizations

- **Multi-stage builds**: Reduce final image size
- **Layer caching**: Speed up rebuilds
- **Selective COPY**: Only copy needed files
- **Dependency caching**: Python and Node packages cached
- **GPU passthrough**: Direct GPU access for ML workloads

### Security Hardening

- **Non-root user**: All processes run as `dev` user
- **Read-only mounts**: Sensitive files mounted read-only
- **Token validation**: All services require authentication
- **Network isolation**: Custom Docker network
- **Resource limits**: CPU and memory constraints available