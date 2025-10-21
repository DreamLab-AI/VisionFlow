# Getting Started with Turbo Flow Claude

**Complete guide to launching and using the Turbo Flow development environment.**

---

## Prerequisites

- Docker with NVIDIA runtime (for GPU support)
- Docker Compose v2.0+
- At least one AI API key (Anthropic recommended)
- 32GB+ RAM (64GB recommended)
- 100GB+ disk space

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/marcuspat/turbo-flow-claude.git
cd turbo-flow-claude
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

**Minimum required**:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

**Optional but recommended**:
```bash
GOOGLE_GEMINI_API_KEY=...
OPENAI_API_KEY=...
GITHUB_TOKEN=...
ZAI_API_KEY=...
```

### 3. Launch Container

```bash
# Build image
docker build -f Dockerfile.unified -t turbo-flow-unified:latest .

# Start services
docker run --rm --network docker_ragflow \
  --name turbo-flow-unified -d \
  -p 2222:22 -p 5901:5901 -p 8080:8080 -p 9090:9090 \
  --env-file .env \
  --runtime=nvidia \
  --shm-size=32g \
  -v ${HOME}/.claude:/mnt/host-claude:ro \
  -v workspace:/home/devuser/workspace \
  -v agents:/home/devuser/agents \
  -v claude-config:/home/devuser/.claude \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  turbo-flow-unified:latest
```

### 4. Access the Container

Choose your preferred method:

**VNC Desktop** (recommended for GUI work):
```bash
# Connect to localhost:5901
# Password: turboflow
# VNC clients: TigerVNC, Remmina, RealVNC
```

**SSH**:
```bash
ssh -p 2222 devuser@localhost
# Password: turboflow
```

**Docker Exec**:
```bash
docker exec -u devuser -it turbo-flow-unified zsh
```

---

## First Steps

### Inside the Container

Once logged in as `devuser`:

```bash
# 1. Verify environment
whoami                    # devuser
pwd                      # /home/devuser
echo $WORKSPACE          # /home/devuser/workspace

# 2. Check resources
ls ~/agents/*.md | wc -l  # 610 agent templates
ls ~/.claude/skills/      # 6 Claude Code skills

# 3. Start Claude Code
claude
```

### Loading Agents

Inside Claude CLI:

```bash
# Load essential agents
> cat ~/agents/doc-planner.md
> cat ~/agents/microtask-breakdown.md

# Now describe your task
> Using doc-planner methodology, help me build a REST API with authentication
```

### Using Skills

Skills activate automatically when needed:

```bash
# Web summary skill
> Summarize this article: https://example.com/article

# Blender skill
> Create a 3D cube in Blender and render it

# ImageMagick skill
> Resize image.jpg to 800x600 and save as thumbnail.jpg
```

---

## Using the Management API

Submit tasks via HTTP without VNC/SSH:

```bash
# Create task
curl -X POST http://localhost:9090/v1/tasks \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure' \
  -d '{
    "agent": "python-developer",
    "task": "Create a FastAPI REST API for a todo app",
    "provider": "claude-flow"
  }'

# Response
{
  "taskId": "uuid-here",
  "status": "accepted",
  "message": "Task started successfully"
}

# Check progress
curl http://localhost:9090/v1/tasks/{taskId} \
  -H 'Authorization: Bearer change-this-secret-key-to-something-secure'
```

See [Management API Guide](management-api.md) for complete reference.

---

## Common Workflows

### Development Session

```bash
# 1. Access container
docker exec -u devuser -it turbo-flow-unified zsh

# 2. Go to workspace
cd ~/workspace

# 3. Start coding with Claude
claude

# 4. Load agents for structured development
> cat ~/agents/doc-planner.md ~/agents/microtask-breakdown.md
> Help me build [your project]
```

### Using External Projects

Mount your existing codebase:

```bash
# Edit .env
PROJECT_DIR=/path/to/your/project

# Restart container
docker restart turbo-flow-unified

# Access in container
cd ~/workspace/project
```

### Multi-User AI Isolation

Switch between AI providers:

```bash
# As devuser (Claude)
whoami  # devuser

# Switch to Gemini
as-gemini
whoami  # gemini-user

# Switch to OpenAI
as-openai
whoami  # openai-user
```

Each user has isolated credentials and workspaces.

---

## Available Services

| Service | Access | Purpose |
|---------|--------|---------|
| **VNC Desktop** | localhost:5901 | Full XFCE4 desktop with GUI apps |
| **SSH** | localhost:2222 | Remote shell access |
| **code-server** | http://localhost:8080 | Web-based VS Code |
| **Management API** | http://localhost:9090 | HTTP task submission |
| **Swagger Docs** | http://localhost:9090/docs | API documentation |

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs turbo-flow-unified

# Verify network exists
docker network ls | grep ragflow

# Create network if missing
docker network create docker_ragflow
```

### VNC Connection Fails

```bash
# Check VNC service
docker exec turbo-flow-unified supervisorctl status xvnc

# Restart VNC
docker exec turbo-flow-unified supervisorctl restart xvnc xfce4
```

### Claude CLI OAuth Issues

```bash
# Mount host credentials
-v ${HOME}/.claude:/mnt/host-claude:ro

# Or authenticate once in VNC
docker exec -u devuser -it turbo-flow-unified claude
# Complete OAuth in browser
```

### API Returns 401 Unauthorized

```bash
# Check you're using Bearer token
Authorization: Bearer change-this-secret-key-to-something-secure

# Not X-API-Key header
```

---

## Next Steps

- **[Container Access Guide](container-access.md)** - All access methods
- **[Using Claude CLI](using-claude-cli.md)** - Skills and agents
- **[Management API](management-api.md)** - HTTP automation
- **[Skills and Agents](skills-and-agents.md)** - Available tools

---

## Quick Reference

**Default Credentials**:
- VNC password: `turboflow`
- SSH: `devuser:turboflow`
- API key: `change-this-secret-key-to-something-secure`

**Key Directories**:
- `~/workspace` - Your working directory
- `~/agents` - 610 agent templates
- `~/.claude/skills` - 6 Claude Code skills
- `~/workspace/project` - External project mount (if configured)

**User Switching**:
- `as-gemini` - Switch to gemini-user
- `as-openai` - Switch to openai-user
- `as-zai` - Switch to zai-user

**You're ready to build!** 🚀
