# Turbo Flow Unified - System Architecture

This document provides a comprehensive overview of the unified devpod architecture, combining devpod and multi-agent-docker features into a single monolithic CachyOS container.

## Table of Contents

1. [Overview](#overview)
2. [Container Architecture](#container-architecture)
3. [Multi-User System](#multi-user-system)
4. [Service Layer](#service-layer)
5. [Development Environments](#development-environments)
6. [Claude Code Skills](#claude-code-skills)
7. [Network Architecture](#network-architecture)
8. 
9. [Security Model](#security-model)
10. [Process Management](#process-management)

## Overview

### Design Philosophy

The Turbo Flow Unified container follows these core principles:

1. **Single Monolithic Design**: All features in one container for simplicity
2. **Multi-User Isolation**: Separate Linux users for different AI services
3. **Complete Skill Migration**: MCP servers replaced with Claude Code skills
4. **Full-Stack Development**: Comprehensive language and tool support
5. **GPU-First**: Native NVIDIA CUDA integration
6. **Network Integration**: Seamless docker_ragflow network participation

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                  Turbo Flow Unified Container               │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │  devuser   │  │ gemini-user│  │openai-user │           │
│  │ (Claude)   │  │  (Gemini)  │  │  (OpenAI)  │           │
│  │ 1000:1000  │  │ 1001:1001  │  │ 1002:1002  │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│                      ↓                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │          Supervisord (Process Manager)              │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │  SSH │ VNC │ XFCE4 │ Management API │ tmux          │   │
│  └─────────────────────────────────────────────────────┘   │
│                      ↓                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   Development Stack (LaTeX, Python, Rust, CUDA)     │   │
│  └─────────────────────────────────────────────────────┘   │
│                      ↓                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │      CachyOS Base (ArchLinux + Latest Packages)     │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Container Architecture

### Base System

- **Distribution**: CachyOS (ArchLinux-based)
- **Rationale**: Rolling release with latest packages, excellent GPU support
- **Kernel**: Compatible with host Linux 6.17+
- **Init System**: supervisord (not systemd, for Docker compatibility)

### Build Phases

The Dockerfile is organized into 17 phases:

1. **Base System Packages**: Core utilities, build tools
2. **CUDA Development**: NVIDIA toolkit, cuDNN
3. **Rust Toolchain**: Stable Rust with components
4. **Multi-User Setup**: Create 4 user accounts
5. **Node.js Packages**: Global npm packages
6. **Python Environment**: Virtual environment with packages
7. **devuser Setup**: Primary user configuration
8. **gemini-user Setup**: Google Gemini tools
9. **openai-user Setup**: OpenAI tools
10. **zai-user Setup**: Z.AI service user
11. **VNC Configuration**: TigerVNC with XFCE4
12. **SSH Server**: Secure shell access
13. **Application Files**: Copy skills, scripts, config
14. **Supervisord**: Service management setup
15. **Environment**: Runtime configuration
16. **Ports & Volumes**: Network and storage
17. **Entrypoint**: Initialization script

### Resource Allocation

Default configuration (adjustable):

- **Memory**: 64GB limit, 16GB reservation
- **CPUs**: 32 cores limit, 8 cores reservation
- **GPU**: All NVIDIA GPUs with compute capability
- **Shared Memory**: 32GB for GPU operations

## Multi-User System

### User Accounts

| User | UID:GID | Home | Shell | Purpose |
|------|---------|------|-------|---------|
| root | 0:0 | /root | bash | System administration |
| devuser | 1000:1000 | /home/devuser | zsh | Primary Claude Code development |
| gemini-user | 1001:1001 | /home/gemini-user | zsh | Google Gemini tools |
| openai-user | 1002:1002 | /home/openai-user | zsh | OpenAI Codex tools |
| zai-user | 1003:1003 | /home/zai-user | zsh | Z.AI service operations |

### User Isolation

```
devuser (primary)
  ├─ Full sudo access (NOPASSWD)
  ├─ Can switch to other users: sudo -u <user> -i
  ├─ Groups: wheel, video, audio, docker
  └─ Workspace: /home/devuser/workspace

gemini-user (isolated)
  ├─ No sudo access
  ├─ Google Gemini API credentials
  ├─ Workspace: /home/gemini-user/workspace
  └─ Can be accessed from devuser via: as-gemini

openai-user (isolated)
  ├─ No sudo access
  ├─ OpenAI API credentials
  ├─ Workspace: /home/openai-user/workspace
  └─ Can be accessed from devuser via: as-openai

zai-user (isolated)
  ├─ No sudo access
  ├─ Z.AI configuration (Anthropic base URL)
  ├─ Workspace: /home/zai-user/workspace
  └─ Can be accessed from devuser via: as-zai
```

### Credential Distribution

On container startup, the entrypoint script distributes API keys from `.env`:

```
.env file
  ├─ ANTHROPIC_API_KEY → /home/devuser/.config/claude/config.json
  ├─ GOOGLE_GEMINI_API_KEY → /home/gemini-user/.config/gemini/config.json
  ├─ OPENAI_API_KEY → /home/openai-user/.config/openai/config.json
  └─ ANTHROPIC_BASE_URL → /home/zai-user/.config/zai/config.json
```

## Service Layer

### Supervisord Services

All services are managed by supervisord with priority-based startup:

```
Priority 10:  dbus           (Desktop messaging bus)
Priority 50:  sshd           (SSH server)
Priority 100: xvnc           (VNC server)
Priority 200: xfce4          (Desktop environment)
Priority 300: management-api (Health check API)
Priority 900: tmux-autostart (Workspace manager)
```

### Service Details

#### DBus (Priority 10)
- **Command**: `/usr/bin/dbus-daemon --system --nofork`
- **User**: root
- **Purpose**: Inter-process communication for desktop
- **Dependencies**: None (starts first)

#### SSH Server (Priority 50)
- **Command**: `/usr/sbin/sshd -D -e`
- **User**: root
- **Port**: 22 (mapped to 2222 on host)
- **Authentication**: Password (devuser:turboflow)
- **Purpose**: Remote shell access

#### VNC Server (Priority 100)
- **Command**: `vncserver :1 -geometry 1920x1080 -depth 24`
- **User**: devuser
- **Port**: 5901
- **Display**: :1
- **Password**: turboflow
- **Features**: No screen lock, no screensaver

#### XFCE4 Desktop (Priority 200)
- **Command**: `/usr/bin/startxfce4`
- **User**: devuser
- **Display**: :1
- **Purpose**: Full desktop environment for GUI tools

#### Management API (Priority 300)
- **Command**: `/usr/bin/node /opt/management-api/server.js`
- **User**: devuser
- **Port**: 9090
- **Purpose**: Health check endpoint
- **Endpoints**:
  - `GET /health` - Container health status

#### tmux Auto-Start (Priority 900)
- **Command**: `/home/devuser/.config/tmux-autostart.sh`
- **User**: devuser
- **Delay**: 5 seconds after startup
- **Purpose**: Create 8-window development workspace

### tmux Workspace Layout

```
┌─────────────────────────────────────────────────────────┐
│ 0: Claude-Main     │ Primary Claude Code workspace      │
│ 1: Claude-Agent    │ Agent execution and testing        │
│ 2: Services        │ Supervisord monitoring             │
│ 3: Development     │ Python/Rust/CUDA development       │
│ 4: Logs            │ Service logs (split panes)         │
│ 5: System          │ htop resource monitoring           │
│ 6: VNC-Status      │ VNC connection information         │
│ 7: SSH-Shell       │ General purpose shell              │
└─────────────────────────────────────────────────────────┘
```

## Development Environments

### Python Environment

**Version**: 3.12+

**Package Management**:
- System pip
- virtualenv for projects
- poetry for dependency management
- uv for fast package installation

**Installed Packages**:
- **Data Science**: numpy, pandas, scipy, matplotlib, seaborn
- **Machine Learning**: scikit-learn, torch, torchvision, torchaudio
- **Development**: black, flake8, mypy, pylint, pytest
- **Utilities**: requests, beautifulsoup4, pyyaml, toml

**Virtual Environment**: `/opt/venv` (system-wide)

### Rust Environment

**Version**: Latest stable

**Components**:
- rustc (compiler)
- cargo (package manager)
- rustfmt (code formatter)
- clippy (linter)
- rust-analyzer (LSP server)

**Targets**:
- `x86_64-unknown-linux-gnu` (native)
- `wasm32-unknown-unknown` (WebAssembly)

**Installed Tools**:
- bat (cat replacement)
- exa (ls replacement)
- ripgrep (grep replacement)
- fd-find (find replacement)
- tokei (code statistics)

### Node.js Environment

**Version**: Latest LTS

**Global Packages**:
- `@anthropic-ai/claude-code` - Claude Code CLI
- `claude-usage-cli` - API usage monitoring
- `agentic-flow` - Agent orchestration
- `pm2` - Process manager
- `typescript` - Type system
- `ts-node` - TypeScript execution
- `playwright` - Browser automation

### CUDA Environment

**Components**:
- CUDA Toolkit (latest)
- cuDNN (deep learning)
- NVIDIA drivers

**Environment Variables**:
```bash
CUDA_HOME=/opt/cuda
PATH=/opt/cuda/bin:$PATH
LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
```

**Usage**:
```bash
nvcc --version         # CUDA compiler
nvidia-smi            # GPU status
cuda-gdb              # CUDA debugger
```

### LaTeX Environment

**Distribution**: TeX Live

**Packages**:
- texlive-basic
- texlive-bin
- texlive-binextra
- texlive-fontsrecommended
- texlive-latexrecommended
- biber (bibliography)

**Compilers**:
- pdflatex
- xelatex
- lualatex
- latexmk (build automation)

### GUI Tools

All GUI tools run on VNC desktop (XFCE4):

- **Blender**: 3D modeling and rendering
- **QGIS**: Geographic information system
- **KiCAD**: PCB design and schematics
- **ImageMagick**: Image processing (CLI + display)
- **Chromium**: Web browser
- **Firefox**: Alternative browser

## Claude Code Skills

### Skills Architecture

```
/home/devuser/.claude/skills/
├── web-summary/
│   ├── SKILL.md           (Manifest with YAML frontmatter)
│   └── tools/
│       └── web_summary_tool.py
├── blender/
│   ├── SKILL.md
│   └── tools/
│       └── blender_tool.py
├── qgis/
│   ├── SKILL.md
│   └── tools/
│       └── qgis_tool.py
├── kicad/
│   ├── SKILL.md
│   └── tools/
│       └── kicad_tool.py
├── imagemagick/
│   ├── SKILL.md
│   └── tools/
│       └── imagemagick_tool.py
└── pbr-rendering/
    ├── SKILL.md
    └── tools/
        └── pbr_tool.py
```

### Skill Communication

Skills use **stdio-based** communication (not HTTP):

```python
# Tool reads JSON from stdin
for line in sys.stdin:
    request = json.loads(line.strip())
    result = process_request(request)
    response = {"result": result}
    print(json.dumps(response))
    sys.stdout.flush()
```

### Available Skills

1. **web-summary**
   - YouTube transcript extraction
   - Web page summarization
   - Semantic topic generation for Logseq

2. **blender**
   - 3D modeling operations
   - Material creation
   - Rendering via socket communication

3. **qgis**
   - Geographic data processing
   - Map generation
   - Spatial analysis

4. **kicad**
   - PCB design
   - Schematic creation
   - Component placement

5. **imagemagick**
   - Image format conversion
   - Batch processing
   - Image manipulation

6. **pbr-rendering**
   - Physically-based rendering materials
   - Texture generation
   - Material property calculation

## Network Architecture

### Docker Network

```
docker_ragflow (bridge network)
  ├── turbo-flow-unified (this container)
  │   ├── Hostname: turbo-devpod
  │   ├── Aliases: turbo-devpod.ragflow, turbo-unified.local
  │   └── IP: Dynamic (DHCP from Docker)
  └── Other containers (e.g., ragflow services)
```

### Port Mappings

| Host | Container | Service | Protocol | Access |
|------|-----------|---------|----------|--------|
| 2222 | 22 | SSH | TCP | LAN |
| 5901 | 5901 | VNC | TCP | LAN |
| 9090 | 9090 | Management API | HTTP | LAN |

### Service Discovery

Containers in docker_ragflow can access this container via:
- `turbo-devpod:PORT`
- `turbo-devpod.ragflow:PORT`
- `turbo-unified.local:PORT`

### Network Capabilities

The container has enhanced network capabilities:
- `NET_ADMIN` - Network management
- `SYS_ADMIN` - Required for Chromium sandbox
- `SYS_PTRACE` - Debugging capabilities

## Storage & Persistence

### Docker Volumes

All persistent data uses Docker named volumes:

```
turbo-flow-unified_workspace      → /home/devuser/workspace
turbo-flow-unified_agents         → /home/devuser/agents
turbo-flow-unified_claude-config  → /home/devuser/.claude
turbo-flow-unified_gemini-workspace → /home/gemini-user/workspace
turbo-flow-unified_openai-workspace → /home/openai-user/workspace
turbo-flow-unified_model-cache    → /home/devuser/models
turbo-flow-unified_logs           → /var/log
```

### Directory Structure

```
/home/devuser/
├── workspace/          (Primary development workspace)
├── agents/            (610+ agent template markdown files)
├── models/            (AI model cache)
├── logs/              (User-level logs)
├── .claude/
│   ├── skills/        (Custom Claude Code skills)
│   └── config.json    (Claude configuration)
├── .config/
│   └── tmux-autostart.sh
└── .zshrc             (Shell configuration)

/home/gemini-user/
├── workspace/
├── .config/
│   └── gemini/
│       └── config.json

/home/openai-user/
├── workspace/
├── .config/
│   └── openai/
│       └── config.json

/home/zai-user/
├── workspace/
├── .config/
│   └── zai/
│       └── config.json

/var/log/              (System-wide logs)
├── supervisord.log
├── management-api.log
├── sshd.log
├── xvnc.log
└── xfce4.log
```

### Backup Strategy

```bash
# Backup workspace
docker run --rm \
  -v turbo-flow-unified_workspace:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/workspace-$(date +%Y%m%d).tar.gz -C /data .

# Restore workspace
docker run --rm \
  -v turbo-flow-unified_workspace:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/workspace-20241019.tar.gz -C /data
```

## Security Model

### Authentication

**Default Credentials** (⚠️ DEVELOPMENT ONLY):
- SSH/VNC Password: `turboflow`
- Management API: No authentication (port only exposed to trusted network)

**Production Recommendations**:
```bash
# Change SSH password
docker exec -it turbo-flow-unified passwd devuser

# Change VNC password
docker exec -u devuser turbo-flow-unified vncpasswd
docker exec turbo-flow-unified supervisorctl restart xvnc
```

### User Permissions

```
root
├─ Full system access
└─ Used only for: supervisord, sshd, system maintenance

devuser
├─ Sudo access (NOPASSWD)
├─ Can switch to other users
├─ Groups: wheel, video, audio, docker
└─ Primary development user

gemini-user, openai-user, zai-user
├─ No sudo access
├─ Isolated home directories (mode 700)
├─ Process isolation
└─ Credential isolation
```

### Container Capabilities

Required capabilities:
- `SYS_ADMIN` - For Chromium sandbox and mount operations
- `NET_ADMIN` - For network configuration
- `SYS_PTRACE` - For debugging tools (gdb, strace)

### GPU Device Access

```
/dev/dri/*              (Intel/AMD GPU)
/dev/nvidia0            (NVIDIA GPU 0)
/dev/nvidiactl          (NVIDIA control)
/dev/nvidia-uvm         (Unified memory)
/dev/nvidia-modeset     (Modesetting)
```

## Process Management

### Supervisord Hierarchy

```
supervisord (PID 1)
├── dbus-daemon         (System messaging)
├── sshd -D             (SSH server)
│   └── sshd sessions   (Per-connection processes)
├── vncserver :1        (VNC server)
│   └── Xvnc :1         (X11 display)
│       └── startxfce4  (Desktop)
│           ├── xfwm4   (Window manager)
│           ├── xfce4-panel
│           └── xfdesktop
├── node server.js      (Management API)
└── tmux-autostart.sh   (Workspace initialization)
    └── tmux server     (Terminal multiplexer)
        ├── Window 0: zsh (Claude-Main)
        ├── Window 1: zsh (Claude-Agent)
        ├── Window 2: zsh (Services)
        ├── Window 3: zsh (Development)
        ├── Window 4: tail (Logs, split)
        ├── Window 5: htop (System)
        ├── Window 6: zsh (VNC-Status)
        └── Window 7: zsh (SSH-Shell)
```

### Process Monitoring

```bash
# View all services
sudo supervisorctl status

# Restart a service
sudo supervisorctl restart xvnc

# View logs
sudo supervisorctl tail -f management-api

# Stop all services
sudo supervisorctl stop all

# Reload configuration
sudo supervisorctl reread
sudo supervisorctl update
```

### Health Checks

Docker health check runs every 30 seconds:
```bash
curl -f http://localhost:9090/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-19T12:34:56.789Z",
  "uptime": 1234.56
}
```

## Initialization Flow

### Container Startup Sequence

```
1. Docker starts container
   ↓
2. /entrypoint.sh executes
   ├─ Create directories
   ├─ Distribute API keys from .env
   ├─ Set file permissions
   ├─ Generate SSH host keys (if needed)
   └─ Copy 610+ agent templates
   ↓
3. supervisord starts (PID 1)
   ↓
4. Services start in priority order
   ├─ 10:  dbus (required for desktop)
   ├─ 50:  sshd (SSH access)
   ├─ 100: xvnc (VNC server)
   ├─ 200: xfce4 (Desktop environment)
   ├─ 300: management-api (Health endpoint)
   └─ 900: tmux-autostart (after 5s delay)
   ↓
5. Container ready
   └─ Health check passes
```

### Entrypoint Script Flow

```bash
#!/bin/bash
set -e

# Phase 1: Directory Setup
mkdir -p /home/devuser/{workspace,agents,.claude/skills}
mkdir -p /home/{gemini-user,openai-user,zai-user}/workspace

# Phase 2: Credential Distribution
if [ -f /.env ]; then
  export $(cat .env | grep -v '^#' | xargs)

  # devuser (Claude)
  echo "{\"api_key\": \"$ANTHROPIC_API_KEY\"}" > \
    /home/devuser/.config/claude/config.json

  # gemini-user
  echo "{\"api_key\": \"$GOOGLE_GEMINI_API_KEY\"}" > \
    /home/gemini-user/.config/gemini/config.json

  # openai-user
  echo "{\"api_key\": \"$OPENAI_API_KEY\"}" > \
    /home/openai-user/.config/openai/config.json

  # zai-user
  echo "{\"api_key\": \"$ANTHROPIC_API_KEY\", \"base_url\": \"$ANTHROPIC_BASE_URL\"}" > \
    /home/zai-user/.config/zai/config.json
fi

# Phase 3: Permissions
chown -R devuser:devuser /home/devuser
chown -R gemini-user:gemini-user /home/gemini-user
chown -R openai-user:openai-user /home/openai-user
chown -R zai-user:zai-user /home/zai-user

chmod 700 /home/{gemini-user,openai-user,zai-user}

# Phase 4: SSH Setup
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
  ssh-keygen -A
fi

# Phase 5: Agent Templates
if [ -d /home/devuser/agents ] && [ -z "$(ls -A /home/devuser/agents)" ]; then
  echo "Agents directory empty, templates already in image"
fi

# Phase 6: Start supervisord
exec /opt/venv/bin/supervisord -c /etc/supervisord.conf
```

## Troubleshooting Reference

### Common Issues

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| Container won't start | Check logs | `docker-compose logs` |
| VNC not accessible | Check VNC service | `docker exec turbo-flow-unified supervisorctl status xvnc` |
| SSH connection refused | Verify port mapping | `docker port turbo-flow-unified` |
| Skills not working | Check permissions | `docker exec turbo-flow-unified ls -la /home/devuser/.claude/skills` |
| High memory usage | Check processes | `docker exec turbo-flow-unified htop` |

### Diagnostic Commands

```bash
# Service status
docker exec turbo-flow-unified supervisorctl status

# Resource usage
docker stats turbo-flow-unified

# Process tree
docker exec turbo-flow-unified ps auxf

# Network connectivity
docker exec turbo-flow-unified curl -v http://localhost:9090/health

# User switching test
docker exec -u devuser turbo-flow-unified sudo -u gemini-user -i whoami

# Volume inspection
docker volume ls | grep turbo-flow
docker volume inspect turbo-flow-unified_workspace
```

## Performance Considerations

### Resource Usage Patterns

**Idle State**:
- Memory: ~2-4 GB
- CPU: <5%
- Disk: 20-25 GB (image + volumes)

**Active Development**:
- Memory: 8-16 GB (with Claude Code, browser)
- CPU: 20-60% (depending on tasks)
- Disk: Grows with workspace content

**GPU Workloads**:
- Memory: +GPU VRAM usage
- CPU: Depends on data preprocessing
- GPU: Varies by task (rendering, training)

### Optimization Tips

1. **Use Volume Drivers**: Consider local-persist or NFS for shared storage
2. **GPU Selection**: Use `CUDA_VISIBLE_DEVICES` to limit GPU access
3. **Memory Limits**: Adjust docker-compose.yml based on host capacity
4. **Build Cache**: Use BuildKit for faster rebuilds
5. **Layer Optimization**: Pre-built base images for quicker iterations

## Future Enhancements

Potential architectural improvements:

1. **Multi-Stage Build**: Separate build and runtime stages
2. **Dynamic Service Toggle**: Enable/disable services via environment variables
3. **Health Monitoring**: Enhanced monitoring with Prometheus metrics
4. **Secrets Management**: Integration with Docker secrets or vault
5. **Backup Automation**: Scheduled volume backups via cron or sidecar
6. **Resource Quotas**: Per-user CPU/memory limits with cgroups
7. **Service Mesh**: Integration with Istio or Linkerd for advanced networking

---

**Document Version**: 1.0
**Last Updated**: 2025-10-19
**Maintained By**: Turbo Flow Project
