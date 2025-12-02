# 🚀 Turbo Flow Claude

Origin repo, need attribution block
https://github.com/marcuspat/turbo-flow-claude/tree/main

**Advanced Agentic Development Environment with Multi-User AI Isolation**

[![DevPod](https://img.shields.io/badge/DevPod-Ready-blue?style=flat-square)](https://devpod.sh) [![Claude Flow](https://img.shields.io/badge/Claude%20Flow-SPARC-purple?style=flat-square)](https://github.com/ruvnet/claude-flow) [![Agents](https://img.shields.io/badge/Agents-610+-green?style=flat-square)](https://github.com/ChrisRoyse/610ClaudeSubagents)

---

## Overview

Turbo Flow Claude provides a complete AI-powered development environment with:

- **610+ AI Agents** from [610ClaudeSubagents](https://github.com/ChrisRoyse/610ClaudeSubagents)
- **Multi-User Isolation** for Claude, Gemini, OpenAI, and Z.AI services
- **Claude Flow Integration** with SPARC methodology and verification-first development
- **Full Development Stack** including Python, Rust, CUDA, LaTeX, and GUI tools
- **Two Deployment Modes** for flexibility and power

## Quick Start

### Option 1: DevPod (Cloud Development)

```bash
devpod up https://github.com/marcuspat/turbo-flow-claude --ide vscode
```

Perfect for GitHub Codespaces, Google Cloud Shell, or any DevPod provider.

### Option 2: Docker Unified Container (Full Workstation)

```bash
git clone https://github.com/marcuspat/turbo-flow-claude.git
cd turbo-flow-claude

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Optional: Mount external project directory
# Edit PROJECT_DIR in .env to point to your codebase:
# PROJECT_DIR=/path/to/your/project

# Build and run
docker build -f Dockerfile.unified -t turbo-flow-unified .
docker-compose -f docker-compose.unified.yml up -d

# Access container
docker exec -u devuser -it turbo-flow-unified zsh
```

Access via:
- **SSH**: `ssh -p 2222 devuser@localhost` (password: turboflow)
- **VNC**: Connect to `localhost:5901` (password: turboflow)
- **Web IDE**: http://localhost:8080
- **API**: http://localhost:9090/documentation

## Features

### Multi-User Architecture

Four isolated Linux users with credential separation:

| User | UID | Purpose | Access |
|------|-----|---------|--------|
| **devuser** | 1000 | Primary Claude Code development | `ssh devuser@localhost -p 2222` |
| **gemini-user** | 1001 | Google Gemini tools | `as-gemini` |
| **openai-user** | 1002 | OpenAI tools | `as-openai` |
| **zai-user** | 1003 | Z.AI service (cost-effective API) | `as-zai` |

### Services (Unified Container)

All managed by supervisord:

- **SSH** (port 22 → 2222) - Remote shell access
- **VNC** (port 5901) - XFCE4 desktop environment
- **code-server** (port 8080) - Web-based VS Code
- **Management API** (port 9090) - Full REST API with Swagger
- **Z.AI Service** (port 9600) - Cost-effective Claude API wrapper
- **Gemini Flow** - Google Gemini orchestration daemon

### Development Tools

**Languages & Runtimes:**
- Python 3.12+ with virtualenv, poetry, and extensive ML libraries
- Rust stable with complete toolchain
- Node.js LTS with global packages
- CUDA Toolkit for GPU development

**GUI Applications:**
- Blender (3D modeling)
- QGIS (GIS operations)
- KiCAD (PCB design)
- ImageMagick (image processing)
- Chromium & Firefox browsers

**Claude Code Skills:**
- `web-summary` - YouTube transcripts and web summarization
- `blender` - 3D modeling automation
- `qgis` - Geographic data processing
- `kicad` - Electronic circuit design
- `imagemagick` - Image manipulation
- `pbr-rendering` - Material generation

## Claude Flow Integration

### Automatic Context Loading

Commands automatically load context files:

```bash
cf-swarm "build a tic-tac-toe game"    # Swarm with auto-loaded context
cf-hive "create a REST API"            # Hive-mind with context
cf "analyze this codebase"             # Any command with context
```

Auto-loaded context includes:
- `CLAUDE.md` - Development rules and patterns
- `agents/doc-planner.md` - SPARC methodology agent
- `agents/microtask-breakdown.md` - Task decomposition agent
- Agent library information (610+ agents)

### SPARC Workflow

Systematic development methodology:

```bash
npx claude-flow@alpha sparc run <mode> "task"      # Execute specific mode
npx claude-flow@alpha sparc tdd "feature"          # Complete TDD workflow
npx claude-flow@alpha sparc pipeline "task"        # Full pipeline
```

Phases: **S**pecification → **P**seudocode → **A**rchitecture → **R**efinement → **C**ompletion

### Verification-First Development

Truth verification at 95% threshold:

```bash
npx claude-flow@alpha verify init strict  # Initialize verification
npx claude-flow@alpha truth                # View truth scores
npx claude-flow@alpha pair --start         # Pair programming mode
```

### GitHub Integration

13 specialized GitHub agents for:
- AI-powered PR reviews
- Automated releases with changelogs
- Intelligent issue management
- Multi-reviewer code analysis
- CI/CD optimization
- Security scanning

```bash
npx claude-flow@alpha github init --verify --pair
npx claude-flow@alpha github pr-manager setup --multi-reviewer
```

## Usage Examples

### Game Development
```bash
cf-swarm "build a multiplayer tic-tac-toe with real-time updates"
```

### Web Development
```bash
cf-hive "create a full-stack blog with authentication and admin panel"
```

### Agent Discovery
```bash
# Search for relevant agents
find agents/ -name "*test*"
find agents/ -name "*github*"

# Sample available agents
ls agents/*.md | shuf | head -10
```

### Full Hive Deployment
```bash
npx claude-flow@alpha hive-mind spawn \
  "Deploy complete enterprise development environment" \
  --agents 25 \
  --github-agents all-13 \
  --verify --pair --claude
```

## Architecture

### Deployment Modes

**DevPod Mode:**
- Lightweight cloud development setup
- Focus on Claude Flow integration with agent library
- Ideal for Codespaces, Google Cloud Shell, DevPod providers

**Unified Container Mode:**
- Full-featured CachyOS workstation
- VNC desktop with GUI tools
- Multi-user AI service isolation
- Complete development stack

### Multi-User Isolation

Credentials are distributed at startup:
- `.env` → `devuser/.config/claude/config.json`
- `.env` → `gemini-user/.config/gemini/config.json`
- `.env` → `openai-user/.config/openai/config.json`
- `.env` → `zai-user/.config/zai/config.json` (with Z.AI base URL)

### Service Architecture

```
supervisord (PID 1)
├── dbus (priority 10)
├── sshd (priority 50)
├── xvnc (priority 100)
├── xfce4 (priority 200)
├── management-api (priority 300)
├── code-server (priority 400)
├── claude-zai (priority 500, runs as zai-user)
├── gemini-flow (priority 600, runs as gemini-user)
└── tmux-autostart (priority 900)
```

## Performance Metrics

Claude Flow achieves:
- **84.8%** SWE-Bench solve rate
- **32.3%** token reduction
- **2.8-4.4x** speed improvement
- **>95%** truth verification accuracy
- **>90%** integration success rate

## Documentation

### Quick Start

- **[SETUP.md](SETUP.md)** - Installation and configuration
- **** - Project configuration for Claude Code
- **[SECURITY.md](SECURITY.md)** - Security best practices

### User Guides

- **** - First steps and basic usage
- **** - VNC, SSH, docker exec
- **** - Skills and agents
- **** - Complete reference
- **** - HTTP automation

### Developer Guides

- **** - System design
- **** - Custom skill development
- **** - Cloud environments
- **** - Production deployment
- **** - CLI commands

## Directory Structure

```
/
├── docs/
│   ├── user/              # User documentation
│   │   ├── getting-started.md
│   │   ├── container-access.md
│   │   ├── using-claude-cli.md
│   │   ├── skills-and-agents.md
│   │   └── management-api.md
│   └── developer/         # Developer documentation
│       ├── architecture.md
│       ├── building-skills.md
│       ├── devpod-setup.md
│       ├── cloud-deployment.md
│       └── command-reference.md
├── devpods/              # DevPod configuration
├── unified-config/       # Container configuration
├── skills/               # 6 Claude Code skills
├── agents/               # 610+ AI agent templates
├── Dockerfile.unified    # Main container build
├── docker-compose.unified.yml  # Service orchestration
├── .env.example          # Environment template
├── SETUP.md             # Installation guide
├── CLAUDE.md            # Project configuration
├── SECURITY.md          # Security guide
└── README.md            # This file
```

## Development Principles

1. **Verification-First** - Truth is enforced, not assumed (95% threshold)
2. **Doc-First** - Always start with doc-planner and microtask-breakdown agents
3. **GitHub-Centric** - All operations integrate with GitHub workflows
4. **Batch Everything** - Multiple operations in single messages
5. **Iterate Until Success** - Deep research when stuck, never give up
6. **Concurrent Execution** - Parallel operations for maximum efficiency

## Volume Mounts and External Projects

### Docker Named Volumes (Persistent)

The unified container uses named volumes for persistent data:
- `workspace` → `/home/devuser/workspace`
- `agents` → `/home/devuser/agents` (610+ agent templates)
- `claude-config` → `/home/devuser/.claude` (skills and configuration)
- `model-cache` → `/home/devuser/models`
- `logs` → `/var/log`

### External Project Directory

Mount an existing codebase from your host machine:

```bash
# Edit .env file
PROJECT_DIR=/mnt/mldata/githubs/my-project

# Restart container
docker-compose -f docker-compose.unified.yml up -d

# Access your project inside container
docker exec -u devuser -it turbo-flow-unified zsh
cd ~/workspace/project  # Your external project is here
```

The `PROJECT_DIR` is mounted at `/home/devuser/workspace/project` with read-write access.

### Host Claude Config

Copy your host machine's Claude configuration:

```bash
# Default: mounts ~/.claude from host (read-only)
HOST_CLAUDE_DIR=${HOME}/.claude

# Container copies to ~/.claude/ on startup
# This preserves your existing Claude Code settings
```

## Common Tasks

### Access Services (Docker Container)

```bash
# SSH
ssh -p 2222 devuser@localhost  # Password: turboflow

# VNC
vnc://localhost:5901           # Password: turboflow

# code-server
http://localhost:8080

# Management API
http://localhost:9090/documentation  # Swagger UI
```

### Service Management

```bash
# Check status
docker exec turbo-flow-unified supervisorctl status

# Restart service
docker exec turbo-flow-unified supervisorctl restart management-api

# View logs
docker exec turbo-flow-unified supervisorctl tail -f claude-zai
```

### tmux Workspace

```bash
# Attach to workspace
tmux attach -t workspace

# 8 windows: Claude-Main, Claude-Agent, Services, Development,
#            Logs, System, VNC-Status, SSH-Shell
```

## Resources

- [Claude Flow](https://github.com/ruvnet/claude-flow) - SPARC orchestration by Reuven Cohen
- [610ClaudeSubagents](https://github.com/ChrisRoyse/610ClaudeSubagents) - Agent library by Christopher Royse
- [Claude Monitor](https://github.com/Maciek-roboblog/Claude-Code-Usage-Monitor) - Usage monitoring
- [DevPod Documentation](https://devpod.sh/docs) - Cloud development setup

## Contributing

Contributions welcome! Please read the documentation and submit pull requests.

## License

See LICENSE file for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=marcuspat/turbo-flow-claude&type=Date)](https://www.star-history.com/#marcuspat/turbo-flow-claude&Date)

---

**Ready to supercharge your development with 610+ AI agents and verification-first workflows?**

```bash
devpod up https://github.com/marcuspat/turbo-flow-claude --ide vscode
```
