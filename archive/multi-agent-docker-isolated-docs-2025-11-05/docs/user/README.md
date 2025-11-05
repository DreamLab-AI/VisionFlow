# User Guide

**Complete user documentation for Turbo Flow Claude.**

---

## Getting Started

**New to Turbo Flow?** Start here:

1. **[Getting Started](getting-started.md)** - Installation and first steps
2. **[Container Access](container-access.md)** - VNC, SSH, docker exec methods
3. **[Using Claude CLI](using-claude-cli.md)** - Skills and agents overview
4. **[Skills and Agents](skills-and-agents.md)** - Complete reference
5. **[Management API](management-api.md)** - HTTP automation

---

## Quick Links

### Installation

```bash
git clone https://github.com/marcuspat/turbo-flow-claude.git
cd turbo-flow-claude
cp .env.example .env
# Edit .env with your API keys
docker build -f Dockerfile.unified -t turbo-flow-unified:latest .
# See Getting Started guide for full docker run command
```

### Access Methods

| Method | URL/Command | Best For |
|--------|-------------|----------|
| **VNC** | `localhost:5901` (password: turboflow) | GUI apps, desktop work |
| **SSH** | `ssh -p 2222 devuser@localhost` | Remote shell access |
| **Docker Exec** | `docker exec -u devuser -it turbo-flow-unified zsh` | Quick commands |
| **Management API** | `http://localhost:9090` | Automation, CI/CD |

### Essential Commands

```bash
# Start Claude Code
claude

# Load essential agents
> cat ~/agents/doc-planner.md ~/agents/microtask-breakdown.md

# Use a skill
> Summarize this article: https://example.com

# Switch AI users
as-gemini  # Gemini tools
as-openai  # OpenAI tools
```

---

## User Guide Contents

### [Getting Started](getting-started.md)

- Prerequisites
- Quick start guide
- Configuration
- First steps
- Troubleshooting

### [Container Access](container-access.md)

- VNC desktop
- SSH access
- Docker exec methods
- Multi-user switching
- Service management
- File operations

### [Using Claude CLI](using-claude-cli.md)

- Starting Claude Code
- Skills overview (6 available)
- Agent templates (610 available)
- Essential workflows
- Advanced usage
- Tips and best practices

### [Skills and Agents](skills-and-agents.md)

- Complete skill reference:
  - Web Summary
  - Blender
  - QGIS
  - KiCAD
  - ImageMagick
  - PBR Rendering
- Agent categories:
  - Essential agents
  - GitHub integration (13 agents)
  - Development agents
  - Language-specific agents
  - Domain-specific agents

### [Management API](management-api.md)

- HTTP endpoints
- Authentication
- Task creation and tracking
- Examples
- API reference
- Automation

---

## Features

### What's Included

- **610 AI Agents** - Specialized workflows and methodologies
- **6 Claude Code Skills** - Automated tool integrations
- **Multi-User Isolation** - Claude, Gemini, OpenAI, Z.AI
- **Full Development Stack** - Python, Rust, Node.js, CUDA
- **GUI Applications** - Blender, QGIS, KiCAD, browsers
- **Management API** - HTTP task submission and tracking

### Services

All services run automatically on container startup:

| Service | Port | Purpose |
|---------|------|---------|
| **VNC Desktop** | 5901 | XFCE4 desktop environment |
| **SSH Server** | 22 (→2222) | Remote shell access |
| **code-server** | 8080 | Web-based VS Code |
| **Management API** | 9090 | HTTP task automation |
| **Z.AI Service** | 9600 | Cost-effective Claude API |

---

## Common Tasks

### Development Session

```bash
# Access container
docker exec -u devuser -it turbo-flow-unified zsh

# Navigate to workspace
cd ~/workspace

# Start Claude Code
claude

# Load agents
> cat ~/agents/doc-planner.md ~/agents/microtask-breakdown.md

# Begin development
> Help me build a REST API with authentication
```

### Submit Task via API

```bash
curl -X POST http://localhost:9090/v1/tasks \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  -d '{
    "agent": "python-developer",
    "task": "Create a FastAPI REST API for a todo app",
    "provider": "claude-flow"
  }'
```

### Use GUI Applications

1. Connect to VNC (`localhost:5901`)
2. Open application:
   - Blender - 3D modeling
   - QGIS - GIS operations
   - Chromium - Web browsing
   - KiCAD - PCB design

---

## Default Credentials

**⚠️ Change these in production!**

```bash
# VNC
Password: turboflow

# SSH
User: devuser
Password: turboflow

# Management API
Authorization: Bearer change-this-secret-key-to-something-secure
```

---

## Support and Resources

- **Main README**: [../README.md](../README.md)
- **Setup Guide**: [../SETUP.md](../SETUP.md)
- **Developer Docs**: [../developer/](../developer/)
- **Security**: [../SECURITY.md](../SECURITY.md)
- **Issues**: [GitHub Issues](https://github.com/marcuspat/turbo-flow-claude/issues)

---

## Quick Reference

### Key Directories

```bash
~/workspace          # Working directory
~/agents             # 610 agent templates
~/.claude/skills     # 6 Claude Code skills
~/workspace/project  # External project mount (if configured)
~/logs/tasks         # Task execution logs
```

### User Accounts

| User | UID | Purpose | Switch Command |
|------|-----|---------|----------------|
| devuser | 1000 | Primary development (Claude) | Default |
| gemini-user | 1001 | Gemini tools | `as-gemini` |
| openai-user | 1002 | OpenAI tools | `as-openai` |
| zai-user | 1003 | Z.AI service | `as-zai` |

### Service Management

```bash
# Check status
docker exec turbo-flow-unified supervisorctl status

# Restart service
docker exec turbo-flow-unified supervisorctl restart {service}

# View logs
docker exec turbo-flow-unified supervisorctl tail -f {service}
```

---

**Ready to build? Start with [Getting Started](getting-started.md)!** 🚀
