# Setup Guide - Turbo Flow Claude

Complete setup instructions for all deployment platforms.

## Quick Start

### DevPod (Recommended for Cloud)
```bash
# Install DevPod
# macOS: brew install loft-sh/devpod/devpod
# Linux: curl -L -o devpod "https://github.com/loft-sh/devpod/releases/latest/download/devpod-linux-amd64" && sudo install devpod /usr/local/bin

# Launch workspace
devpod up https://github.com/marcuspat/turbo-flow-claude --ide vscode
```

### Docker Unified Container (Full Workstation)
```bash
# Clone repository
git clone https://github.com/marcuspat/turbo-flow-claude.git
cd turbo-flow-claude

# Create .env file with API keys
cp .env.example .env
# Edit .env with your API keys

# Build and run
docker build -f Dockerfile.unified -t turbo-flow-unified .
docker-compose -f docker-compose.unified.yml up -d

# Access
# SSH:  ssh -p 2222 devuser@localhost  (password: turboflow)
# VNC:  vnc://localhost:5901  (password: turboflow)
# Code: http://localhost:8080
# API:  http://localhost:9090/health
```

## Platform-Specific Setup

### GitHub Codespaces

1. **Create Codespace**
   - Navigate to repository on GitHub
   - Click "Code" → "Codespaces" → "Create codespace on main"

2. **Run Setup**
   ```bash
   cd devpods
   bash codespace_setup.sh
   ```

3. **Access**
   - VS Code opens automatically
   - Terminal available in bottom panel

### Google Cloud Shell

1. **Launch Cloud Shell**
   - Go to https://console.cloud.google.com
   - Click Cloud Shell icon (top right)

2. **Clone and Setup**
   ```bash
   git clone https://github.com/marcuspat/turbo-flow-claude.git
   cd turbo-flow-claude/devpods
   bash boot_google_cloud_shell.sh
   ```

3. **Configure**
   - Add API keys to `.env`
   - Run `bash setup.sh`

### macOS / Linux Local

1. **Install Docker**
   - macOS: Docker Desktop
   - Linux: Docker Engine

2. **Clone Repository**
   ```bash
   git clone https://github.com/marcuspat/turbo-flow-claude.git
   cd turbo-flow-claude
   ```

3. **Setup**
   ```bash
   # macOS
   cd devpods && bash boot_macosx.sh

   # Linux
   cd devpods && bash boot_linux.sh
   ```

### Rackspace Spot

Detailed instructions: See `spot_rackspace_setup_guide.md`

### DevPod Providers

#### DigitalOcean
```bash
devpod provider add digitalocean
devpod provider use digitalocean
devpod provider update digitalocean --option DIGITALOCEAN_ACCESS_TOKEN=your_token
devpod provider update digitalocean --option DROPLET_SIZE=s-4vcpu-8gb
```

#### AWS
```bash
devpod provider add aws
devpod provider use aws
devpod provider update aws --option AWS_INSTANCE_TYPE=t3.medium
devpod provider update aws --option AWS_REGION=us-east-1
```

#### Azure
```bash
devpod provider add azure
devpod provider use azure
devpod provider update azure --option AZURE_VM_SIZE=Standard_B2s
```

#### Google Cloud
```bash
devpod provider add gcp
devpod provider use gcp
devpod provider update gcp --option GOOGLE_PROJECT_ID=your-project
```

Full provider details: See `devpod_provider_setup_guide.md`

## Configuration

### Environment Variables

Create `.env` file in repository root:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_GEMINI_API_KEY=...
OPENAI_API_KEY=sk-...

# Optional
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
GITHUB_TOKEN=ghp_...
CONTEXT7_API_KEY=...
BRAVE_API_KEY=...

# Docker Configuration
ENABLE_DESKTOP=true
GPU_ACCELERATION=true
MANAGEMENT_API_KEY=change-this-secret-key
```

### Claude Flow Aliases

After setup, these commands are available:

```bash
# Quick aliases
cf-init         # Initialize with verification and GitHub
cf-swarm "task" # Swarm with auto-loaded context
cf-hive "task"  # Hive-mind deployment
cf-verify       # Verification system
cf-truth        # View truth scores
cf-pair         # Start pair programming
```

See `claude-flow-aliases-guide.md` for complete reference.

## Post-Setup

### Verify Installation

```bash
# Check services (Docker container)
docker exec turbo-flow-unified supervisorctl status

# Check API keys
echo $ANTHROPIC_API_KEY

# Test Claude Code
claude --version

# List available agents
ls agents/*.md | wc -l  # Should show 610+
```

### Access Methods

**DevPod Mode:**
- VS Code opens automatically
- Terminal integrated in IDE

**Docker Container:**
- SSH: `ssh -p 2222 devuser@localhost`
- VNC: `vnc://localhost:5901`
- code-server: `http://localhost:8080`
- Management API: `http://localhost:9090/documentation`

### tmux Workspace

Access with `tmux attach -t workspace`:

| Window | Name | Purpose |
|--------|------|---------|
| 0 | Claude-Main | Primary workspace |
| 1 | Claude-Agent | Agent execution |
| 2 | Services | Service monitoring |
| 3 | Development | Python/Rust/CUDA |
| 4 | Logs | Service logs |
| 5 | System | Resource monitor |
| 6 | VNC-Status | Connection info |
| 7 | SSH-Shell | General shell |

## Troubleshooting

### Permission Issues (macOS)
```bash
sudo chown -R $(whoami):staff ~/.devpod
find ~/.devpod -type d -exec chmod 755 {} \;
```

### Container Won't Start
```bash
# Check logs
docker-compose -f docker-compose.unified.yml logs

# Rebuild
docker-compose -f docker-compose.unified.yml down
docker build -f Dockerfile.unified -t turbo-flow-unified --no-cache .
docker-compose -f docker-compose.unified.yml up -d
```

### Services Not Running
```bash
# Inside container
sudo supervisorctl status
sudo supervisorctl restart management-api
sudo supervisorctl tail -f management-api
```

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check docker-compose.yml has:
# runtime: nvidia
```

## Next Steps

1. **Read Documentation**
   - `CLAUDE.md` - Claude Code configuration
   - `ARCHITECTURE.md` - System architecture
   - `SKILLS.md` - Skill development

2. **Start Development**
   - Load mandatory agents: `doc-planner.md` + `microtask-breakdown.md`
   - Initialize verification: `npx claude-flow@alpha verify init strict`
   - Start pair programming: `npx claude-flow@alpha pair --start`

3. **Explore Features**
   - 610+ agent templates in `agents/`
   - 6 custom skills in `.claude/skills/`
   - Multi-user isolation (devuser, gemini-user, openai-user, zai-user)
   - Z.AI cost-effective service on port 9600

## Support

- GitHub Issues: https://github.com/marcuspat/turbo-flow-claude/issues
- Claude Flow: https://github.com/ruvnet/claude-flow
- DevPod Docs: https://devpod.sh/docs
