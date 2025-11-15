# Multi-Agent Docker Workstation

Full-featured CachyOS development container with VNC desktop, multi-user AI isolation, and comprehensive tooling.

## Quick Start

```bash
# Build and start
docker build -f Dockerfile.unified -t turbo-flow-unified:latest .
docker compose -f docker-compose.unified.yml up -d

# Connect via VNC
vncviewer localhost:5901  # Password: none

# Or via SSH
ssh devuser@localhost -p 2222  # Password: turboflow
```

**See [QUICKSTART.md](./QUICKSTART.md) for detailed connection info and usage**

## Features

### Desktop Environment
- **2K Resolution**: 2048x2048 via VNC
- **Window Manager**: Openbox + tint2 panel
- **9 Color-Coded Terminals**: 3x3 grid, each with custom banner and helpful commands
- **Applications**: Chromium with DevTools, terminal grid auto-launches

### Multi-User Isolation
Four isolated Linux users with separate credentials:
- **devuser** (1000): Primary development, Claude Code, sudo access
- **gemini-user** (1001): Google Gemini API isolation
- **openai-user** (1002): OpenAI API isolation
- **zai-user** (1003): Z.AI service (cost-effective Claude API wrapper)

### Services (19 total)
Managed by supervisord:
- VNC (x11vnc + Xvfb), SSH, dbus
- Management API (port 9090), code-server (port 8080)
- Claude Z.AI service (port 9600, internal)
- Gemini Flow orchestration
- 13 MCP servers for Claude Code skills
- Automated terminal grid and tmux workspace

### Development Tools
- **Languages**: Python, Rust, Node.js v22, Go
- **GPU**: CUDA toolkit, Mesa software rendering
- **Version Control**: git, GitHub CLI
- **Containers**: Docker, docker-compose
- **AI Frameworks**: claude-flow, gemini-flow, 610+ agent templates

### Claude Code Skills (18+)

**Development & Research:**
- jupyter-notebooks - Interactive notebook execution with MCP
- latex-documents - Academic paper compilation with TeX Live
- rust-development - Complete Rust toolchain with cargo/clippy
- pytorch-ml - Deep learning with CUDA acceleration
- ffmpeg-processing - Professional video/audio transcoding

**Web & Content:**
- perplexity - Real-time AI research (UK-focused)
- web-summary - YouTube + web summaries (Z.AI powered)
- playwright - Browser automation
- chrome-devtools - Chrome debugging (port 9222)

**Graphics & Design:**
- blender - 3D modeling (socket 2800)
- qgis - GIS operations (socket 2801)
- imagemagick - Image processing
- pbr-rendering - PBR materials

**Engineering:**
- kicad - PCB design
- ngspice - Circuit simulation
- docker-manager - Container management

**Plus specialized skills**: wardley-maps, logseq-formatted, import-to-ontology

## Documentation

- **[QUICKSTART.md](./QUICKSTART.md)** - Connection info, VNC clients, quick commands
- **[docs/TERMINAL_GRID.md](./docs/TERMINAL_GRID.md)** - Terminal window configuration and customization
- **[docs/development-notes/](./docs/development-notes/)** - Session notes and migration history

## Key Configuration Files

```
multi-agent-docker/
├── Dockerfile.unified              # Main container definition
├── docker-compose.unified.yml      # Service orchestration
├── unified-config/
│   ├── supervisord.unified.conf    # Service manager (19 services)
│   ├── entrypoint-unified.sh       # Container startup script
│   ├── tmux-autostart.sh           # SSH tmux workspace (11 windows)
│   ├── autostart-terminals.sh      # VNC terminal grid launcher
│   ├── disable-screensaver.sh      # VNC screensaver disable
│   ├── 10-headless.conf            # Xorg headless configuration
│   └── terminal-init/              # 9 colorful terminal banners
│       ├── init-claude-main.sh
│       ├── init-claude-agent.sh
│       ├── init-services.sh
│       ├── init-development.sh
│       ├── init-docker.sh
│       ├── init-git.sh
│       ├── init-gemini.sh
│       ├── init-openai.sh
│       └── init-zai.sh
└── .env                            # API keys (not in git)
```

## Environment Variables

Required in `.env` file:
```bash
# Essential
ANTHROPIC_API_KEY=sk-ant-xxxxx
PROJECT_DIR=/path/to/your/project  # Mounts to /home/devuser/workspace/project

# Optional
PERPLEXITY_API_KEY=pplx-xxxxx
GOOGLE_GEMINI_API_KEY=xxxxx
OPENAI_API_KEY=sk-xxxxx
GITHUB_TOKEN=ghp_xxxxx
ZAI_ANTHROPIC_API_KEY=sk-ant-xxxxx  # Separate key for Z.AI
```

## Architecture

### Network
- **Docker network**: `docker_ragflow` (bridge)
- **Hostname**: `agentic-workstation`
- **Exposed ports**: 2222 (SSH), 5901 (VNC), 8080 (code-server), 9090 (Management API)
- **Internal ports**: 9600 (Z.AI - not exposed)

### Volumes
```yaml
workspace:         /home/devuser/workspace
agents:            /home/devuser/agents (610+ templates)
gemini-workspace:  /home/gemini-user/workspace
openai-workspace:  /home/openai-user/workspace
model-cache:       /home/devuser/models
logs:              /var/log
${PROJECT_DIR}:    /home/devuser/workspace/project (RW mount)
```

### Resource Limits
```yaml
Memory: 64GB limit, 16GB reservation
CPUs: 32 limit, 8 reservation
GPU: NVIDIA runtime, all GPUs, full capabilities
```

## Terminal Grid Details

VNC desktop shows 9 terminals in 3x3 grid:
1. 🤖 **Claude-Main** (cyan) - `/home/devuser/workspace`
2. 🤖 **Claude-Agent** (magenta) - `/home/devuser/agents`
3. ⚙️ **Services** (yellow) - Service monitoring with sudo
4. 💻 **Development** (green) - External project mount
5. 🐳 **Docker** (magenta) - Container management
6. 🔀 **Git** (blue) - Version control operations
7. 🔮 **Gemini-Shell** (light magenta) - gemini-user
8. 🧠 **OpenAI-Shell** (light blue) - openai-user
9. ⚡ **Z.AI-Shell** (bright yellow) - zai-user

Each terminal displays a color-coded banner with:
- Current working directory
- User and UID
- Purpose description
- Relevant quick commands

**Important**: VNC terminals do NOT use tmux (independent sessions). tmux is available for SSH only.

## Management

### Service Control
```bash
# Inside container
sudo supervisorctl status              # List all services
sudo supervisorctl restart <service>   # Restart service
sudo supervisorctl tail -f <service>   # View logs
```

### Management API
```bash
# Health check (no auth)
curl http://localhost:9090/health

# System status (requires API key)
curl -H "X-API-Key: change-this-secret-key" http://localhost:9090/api/status

# Swagger docs
http://localhost:9090/documentation
```

### User Switching
```bash
as-gemini   # Switch to gemini-user
as-openai   # Switch to openai-user
as-zai      # Switch to zai-user

# Or directly
sudo -u gemini-user -i
```

## Troubleshooting

### VNC not connecting
```bash
docker exec agentic-workstation supervisorctl status xvfb x11vnc
docker exec agentic-workstation supervisorctl restart xvfb x11vnc
```

### Terminals not appearing
```bash
docker exec agentic-workstation supervisorctl restart terminal-grid
# Or manually:
docker exec -u devuser agentic-workstation bash -c 'export DISPLAY=:1 && /home/devuser/.config/autostart-terminals.sh' &
```

### Services failing
```bash
docker exec agentic-workstation tail -50 /var/log/<service>.error.log
docker exec agentic-workstation supervisorctl restart <service>
```

## Security Notes

**⚠️ DEFAULT CREDENTIALS (CHANGE IN PRODUCTION):**
- SSH: `devuser:turboflow`
- VNC: No password
- Management API: `X-API-Key: change-this-secret-key`
- code-server: No authentication

## Development

Built on CachyOS (Arch Linux) with:
- Optimized kernel and packages
- GPU support (NVIDIA + Mesa)
- Modern development stack
- Multi-stage Dockerfile (17 phases)

## Support

- GitHub: [AR-AI-Knowledge-Graph/multi-agent-docker](https://github.com/yourusername/AR-AI-Knowledge-Graph)
- Issues: Report via GitHub Issues
- Docs: See `docs/` directory

## License

[Add your license here]
