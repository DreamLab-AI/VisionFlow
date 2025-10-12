# Getting Started with Agentic Flow Docker System

A complete guide to setting up and running the Agentic Flow multi-agent orchestration platform with GPU acceleration, multiple AI providers, and Claude-ZAI integration.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [API Key Configuration](#api-key-configuration)
4. [First Time Build and Start](#first-time-build-and-start)
5. [Verification Steps](#verification-steps)
6. [Creating Your First Task](#creating-your-first-task)
7. [Using Claude-ZAI](#using-claude-zai)
8. [Accessing the Desktop Environment](#accessing-the-desktop-environment)
9. [Understanding the Directory Structure](#understanding-the-directory-structure)
10. [Next Steps and Resources](#next-steps-and-resources)

---

## Prerequisites

### System Requirements

**Minimum:**
- 16GB RAM
- 8 CPU cores
- 50GB free disk space
- Linux/macOS/Windows with WSL2

**Recommended:**
- 32GB+ RAM
- 16+ CPU cores
- 100GB+ free disk space
- NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Required Software

**Docker and Docker Compose:**

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

**NVIDIA GPU Support (Optional):**

If you have an NVIDIA GPU and want GPU acceleration:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### API Keys

You'll need at least ONE of these API keys:

- **Anthropic Claude** (recommended): Get from [console.anthropic.com](https://console.anthropic.com/)
- **Google Gemini**: Get from [aistudio.google.com](https://aistudio.google.com/)
- **OpenAI**: Get from [platform.openai.com](https://platform.openai.com/)
- **OpenRouter**: Get from [openrouter.ai](https://openrouter.ai/)

Optional:
- **GitHub Token**: For GitHub integration features
- **Context7**: For advanced search
- **Brave**: For web search capabilities

---

## Initial Setup

### 1. Clone the Repository

```bash
# Navigate to your projects directory
cd ~/projects

# Clone the repository
git clone https://github.com/your-org/AR-AI-Knowledge-Graph.git
cd AR-AI-Knowledge-Graph/multi-agent-docker
```

### 2. Review the Docker Compose Configuration

```bash
# View the docker-compose configuration
cat docker-compose.yml

# Check available Dockerfiles
ls -la Dockerfile*
```

The system consists of two main services:
- **agentic-flow-cachyos**: Main orchestration container (CachyOS-based)
- **claude-zai**: Z.AI semantic processing service

---

## API Key Configuration

### 1. Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# Open in your preferred editor
nano .env
# or
vim .env
# or
code .env
```

### 2. Configure Required API Keys

Edit `.env` and add your API keys:

```bash
# ============================================================================
# API Keys (At least one provider required)
# ============================================================================

# Anthropic Claude (recommended)
ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Z.AI (required for claude-zai service)
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic

# OpenAI
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google Gemini
GOOGLE_GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# OpenRouter (multi-model access)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# GitHub Token (optional)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. Configure System Settings

```bash
# ============================================================================
# Management API Configuration
# ============================================================================

# IMPORTANT: Change this secret key for production
MANAGEMENT_API_KEY=your-secure-random-key-here
MANAGEMENT_API_PORT=9090
MANAGEMENT_API_HOST=0.0.0.0

# ============================================================================
# Model Router Configuration
# ============================================================================

# Router mode: performance, cost, quality, balanced, offline
ROUTER_MODE=performance

# Primary provider: gemini, openai, claude, openrouter
PRIMARY_PROVIDER=gemini

# Fallback chain (comma-separated)
FALLBACK_CHAIN=gemini,openai,claude,openrouter

# ============================================================================
# GPU Configuration
# ============================================================================

GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=all

# ============================================================================
# Optional Services
# ============================================================================

# Desktop environment (VNC/noVNC)
ENABLE_DESKTOP=false

# VS Code Server
ENABLE_CODE_SERVER=false
```

### 4. Secure Your Environment File

```bash
# Set restrictive permissions
chmod 600 .env

# Verify it's not tracked by git
echo ".env" >> .gitignore
```

---

## First Time Build and Start

### 1. Build the Docker Images

```bash
# Build both containers (first time will take 10-20 minutes)
./start-agentic-flow.sh --build
```

This command:
- Builds the agentic-flow-cachyos container from CachyOS base
- Builds the claude-zai service container
- Installs all dependencies and tools
- Sets up MCP servers and agents
- Configures GPU access if available

### 2. Start the Services

The build command automatically starts the services. For subsequent starts:

```bash
# Regular startup (without rebuild)
./start-agentic-flow.sh

# Or use docker-compose directly
docker-compose up -d
```

### 3. Monitor Startup Progress

```bash
# Watch container logs
./start-agentic-flow.sh --logs

# Or follow specific service logs
docker-compose logs -f agentic-flow-cachyos
docker-compose logs -f claude-zai
```

Wait for these messages:
```
✅ Management API server listening on http://0.0.0.0:9090
✅ claude-zai service is healthy
✅ agentic-flow-cachyos container is running
```

---

## Verification Steps

### 1. Check Service Status

```bash
# Check overall status
./start-agentic-flow.sh --status

# Verify containers are running
docker ps
```

You should see:
```
CONTAINER ID   IMAGE                          STATUS         PORTS
abc123...      cachyos-agentic-flow-cachyos  Up 2 minutes   0.0.0.0:9090->9090/tcp
def456...      cachyos-claude-zai            Up 2 minutes   0.0.0.0:9600->9600/tcp
```

### 2. Test Health Endpoints

```bash
# Test Management API
curl http://localhost:9090/health
# Expected: {"status":"healthy","timestamp":"..."}

# Test Claude-ZAI Service
curl http://localhost:9600/health
# Expected: {"status":"ok","uptime":...}

# Get API information
curl http://localhost:9090/
```

### 3. Verify GPU Access (If Available)

```bash
# Enter the container
docker exec -it agentic-flow-cachyos zsh

# Check GPU
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.x   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# ...
```

### 4. Test AI Provider Connectivity

```bash
# Inside container
docker exec -it agentic-flow-cachyos zsh

# Run provider tests
/opt/scripts/test-all-providers.sh

# Or test specific provider
agentic-flow --agent coder --task "Print hello world" --provider gemini --max-tokens 50
```

### 5. Check RAGFlow Network Integration (Optional)

```bash
# Check if connected to RAGFlow network
docker network inspect docker_ragflow | grep agentic-flow-cachyos

# Test Xinference connectivity (if RAGFlow is running)
docker exec -it agentic-flow-cachyos curl -s http://172.18.0.11:9997/v1/models
```

---

## Creating Your First Task

### Using the Management API

The Management API is the primary interface for task management.

**1. Create a Task via HTTP:**

```bash
# Create a simple task
curl -X POST http://localhost:9090/v1/tasks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secure-random-key-here" \
  -d '{
    "agent": "coder",
    "task": "Create a Python function that calculates the fibonacci sequence",
    "provider": "gemini",
    "maxTokens": 1000
  }'
```

Response:
```json
{
  "taskId": "task-1234567890",
  "status": "running",
  "createdAt": "2025-10-12T17:00:00.000Z"
}
```

**2. Check Task Status:**

```bash
# Get task status
curl http://localhost:9090/v1/tasks/task-1234567890 \
  -H "X-API-Key: your-secure-random-key-here"
```

**3. List All Tasks:**

```bash
curl http://localhost:9090/v1/tasks \
  -H "X-API-Key: your-secure-random-key-here"
```

### Using the Command-Line Interface

**1. Enter the Container:**

```bash
# Open interactive shell
docker exec -it agentic-flow-cachyos zsh
```

**2. Run Your First Agent:**

```bash
# Simple coding task
agentic-flow --agent coder --task "Create a Python REST API with FastAPI"

# Use specific provider
agentic-flow --agent coder --task "Build authentication system" --provider claude

# With intelligent routing
agentic-flow --agent coder --task "Complex task" --optimize --priority quality
```

**3. Common Agent Commands:**

```bash
# Code review
agentic-flow --agent reviewer --task "Review this code for security issues" --file ./app.py

# Research
agentic-flow --agent researcher --task "Latest developments in transformer architectures"

# Testing
agentic-flow --agent tester --task "Generate comprehensive unit tests" --file ./api.py

# Backend development
agentic-flow --agent backend-dev --task "Create GraphQL API with PostgreSQL"

# List all available agents
agentic-flow --list-agents
```

### Example: Multi-Step Workflow

```bash
# Inside container

# Step 1: Research and plan
agentic-flow --agent researcher --task "Best practices for REST API authentication" \
  --save-context auth-research

# Step 2: Implement
agentic-flow --agent coder --task "Implement JWT authentication based on research" \
  --load-context auth-research --save-context auth-implementation

# Step 3: Test
agentic-flow --agent tester --task "Create tests for authentication" \
  --load-context auth-implementation --save-context auth-tests

# Step 4: Review
agentic-flow --agent reviewer --task "Security review of authentication" \
  --load-context auth-implementation
```

---

## Using Claude-ZAI

Claude-ZAI provides semantic processing and enhanced Claude integration through Z.AI's optimized infrastructure.

### Basic Usage

**1. Direct API Access:**

```bash
# Send a request to Claude-ZAI
curl -X POST http://localhost:9600/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${ANTHROPIC_API_KEY}" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": "Explain quantum computing in simple terms"
      }
    ]
  }'
```

**2. Check Service Health:**

```bash
curl http://localhost:9600/health
```

**3. View Service Metrics:**

```bash
curl http://localhost:9600/metrics
```

### Integration with Agentic Flow

Claude-ZAI is automatically used when you specify the Claude provider:

```bash
# Uses claude-zai service internally
docker exec -it agentic-flow-cachyos \
  agentic-flow --agent coder --task "Your task" --provider claude
```

### Configuration

Claude-ZAI settings in `.env`:

```bash
# Z.AI Configuration
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
ANTHROPIC_AUTH_TOKEN=${ANTHROPIC_API_KEY}

# Worker Pool Settings
CLAUDE_WORKER_POOL_SIZE=4    # Number of concurrent workers
CLAUDE_MAX_QUEUE_SIZE=50     # Maximum queued requests
```

---

## Accessing the Desktop Environment

The system includes an optional XFCE desktop environment accessible via VNC/noVNC for visual debugging and GUI tools.

### Enable Desktop Environment

**1. Edit `.env`:**

```bash
ENABLE_DESKTOP=true
```

**2. Restart Services:**

```bash
./start-agentic-flow.sh --restart
```

### Access Methods

**Option 1: Web Browser (noVNC - Recommended)**

1. Open browser: `http://localhost:6901`
2. Click "Connect"
3. Password: `vncpassword` (or your custom VNC_PASSWORD)

**Option 2: VNC Client**

1. Install VNC client (TigerVNC, RealVNC, etc.)
2. Connect to: `localhost:5901`
3. Password: `vncpassword`

### Desktop Features

- **XFCE Desktop**: Lightweight, no screen locking
- **Chromium Browser**: For web-based tools
- **Terminal**: Integrated zsh terminal
- **File Manager**: Thunar file manager
- **Text Editor**: gedit and vim
- **Development Tools**: All CLI tools accessible from desktop terminal

### GUI Applications

```bash
# Inside container (from desktop terminal)

# Open file manager
thunar &

# Open browser
chromium &

# Open text editor
gedit &

# Visual monitoring
htop
```

---

## Understanding the Directory Structure

### Container Directory Layout

```mermaid
graph TD
    A[/home/devuser/]

    A --> B[workspace/]
    B --> B1[projects/]
    B --> B2[temp/]
    B --> B3[agents/]

    A --> C[models/]
    C --> C1[phi-4.onnx]
    C --> C2[embeddings/]

    A --> D[.agentic-flow/]
    D --> D1[memory/]
    D --> D2[metrics/]
    D --> D3[logs/]
    D --> D4[sessions/]

    A --> E[.config/]
    E --> E1[agentic-flow/]
    E1 --> E11[router.config.json]
    E1 --> E12[agents/]
    E --> E2[claude/]
    E2 --> E21[mcp-servers.json]

    A --> F[logs/]
    F --> F1[api.log]
    F --> F2[tasks/]

    A --> G[.claude/]
    G --> G1[agents/]
    G --> G2[CLAUDE.md]

    style A fill:#4a90e2,stroke:#2e5c8a,color:#fff
    style B fill:#7ed321,stroke:#5a9a18,color:#000
    style C fill:#f5a623,stroke:#b87a1a,color:#000
    style D fill:#bd10e0,stroke:#8a0ba6,color:#fff
    style E fill:#50e3c2,stroke:#3ab09e,color:#000
    style F fill:#f8e71c,stroke:#b8a915,color:#000
    style G fill:#ff6b6b,stroke:#cc5555,color:#fff
```

### Persistent Volumes

Data is stored in Docker volumes that persist across container restarts:

```bash
# View volumes
docker volume ls | grep agentic

# Outputs:
# workspace         - Development files
# model-cache       - AI models
# agent-memory      - Agent state
# config-persist    - Configuration
# management-logs   - API logs
```

### Important Paths

**Configuration:**
- `.env` - Environment variables (host)
- `/home/devuser/.config/agentic-flow/router.config.json` - Model router config
- `/home/devuser/.claude/agents/` - Custom agent definitions
- `/opt/management-api/` - Management API source

**Scripts:**
- `/opt/scripts/` - System scripts
- `/opt/scripts/test-all-providers.sh` - Provider testing
- `/opt/scripts/healthcheck.sh` - Health monitoring
- `/opt/scripts/init-workstation.sh` - Initialization

**Executables:**
- `/usr/local/bin/agentic-flow` - Main CLI
- `/usr/local/bin/claude` - Claude CLI
- `/opt/management-api/server.js` - Management API server

---

## Next Steps and Resources

### Recommended Learning Path

1. **Explore Available Agents** (15 minutes)
   ```bash
   docker exec -it agentic-flow-cachyos zsh
   agentic-flow --list-agents
   agentic-flow --help
   ```

2. **Try Different Providers** (30 minutes)
   ```bash
   # Test each provider
   /opt/scripts/test-all-providers.sh

   # Compare results
   agentic-flow --agent coder --task "Same task" --provider gemini
   agentic-flow --agent coder --task "Same task" --provider claude
   ```

3. **Build a Sample Project** (1 hour)
   ```bash
   cd ~/workspace
   mkdir my-first-project
   cd my-first-project

   # Use agents to build a complete application
   agentic-flow --agent backend-dev --task "Create Express.js API with JWT auth"
   agentic-flow --agent tester --task "Generate tests for the API"
   agentic-flow --agent reviewer --task "Review code security"
   ```

4. **Explore Advanced Features** (ongoing)
   - Model routing and optimization
   - Custom agent creation
   - MCP tool integration
   - Multi-agent orchestration
   - RAGFlow integration

### Documentation

**Core Documentation:**
- [Architecture Overview](ARCHITECTURE-SIMPLIFIED.md) - System design
- [Deployment Guide](DEPLOYMENT.md) - Production deployment
- [Workstation Guide](README.workstation.md) - Full workstation features
- [API Reference](http://localhost:9090/docs) - Management API docs (when running)

**Integration Guides:**
- [Gemini Flow](GEMINI-FLOW.md) - Google Gemini integration
- [ONNX Proxy](ONNX-PROXY-IMPLEMENTATION.md) - Offline inference
- [Test Framework](TEST-FRAMEWORK.md) - Testing strategies

### API Endpoints Reference

**Management API (Port 9090):**

```bash
# Base URL: http://localhost:9090

# Health and status
GET  /health                # Health check
GET  /ready                 # Readiness check
GET  /metrics               # Prometheus metrics
GET  /v1/status             # System status

# Task management
POST /v1/tasks              # Create task
GET  /v1/tasks              # List tasks
GET  /v1/tasks/:taskId      # Get task details
DELETE /v1/tasks/:taskId    # Cancel task

# API documentation
GET  /docs                  # Swagger UI
GET  /                      # API information
```

**Claude-ZAI API (Port 9600):**

```bash
# Base URL: http://localhost:9600

# Service endpoints
GET  /health                      # Health check
GET  /metrics                     # Service metrics
POST /v1/messages                 # Claude messages API
POST /v1/messages/stream          # Streaming messages
```

### Shell Aliases and Shortcuts

Inside the container, these aliases are available:

```bash
# Navigation
workspace       # cd ~/workspace
projects        # cd ~/workspace/projects

# Development
afh             # agentic-flow --help
afl             # agentic-flow --list-agents
test-gpu        # Check GPU status
test-providers  # Test all providers
check-keys      # Verify API keys

# Provider shortcuts
af-gemini       # Use Gemini provider
af-openai       # Use OpenAI provider
af-claude       # Use Claude provider
af-local        # Use Xinference (local)
af-offline      # Use ONNX (offline)

# Router shortcuts
af-optimize     # Intelligent routing
af-perf         # Performance mode
af-cost         # Cost optimization
af-quality      # Quality mode
```

### Troubleshooting Commands

```bash
# View logs
./start-agentic-flow.sh --logs

# Check status
./start-agentic-flow.sh --status

# Restart services
./start-agentic-flow.sh --restart

# Complete rebuild
./start-agentic-flow.sh --clean
./start-agentic-flow.sh --build

# Test specific components
docker exec -it agentic-flow-cachyos /opt/scripts/test-all-providers.sh
docker exec -it agentic-flow-cachyos /opt/scripts/healthcheck.sh

# Check GPU
docker exec -it agentic-flow-cachyos nvidia-smi

# Verify network
docker network inspect docker_ragflow
docker network inspect agentic-network
```

### Community and Support

- **GitHub Repository**: [https://github.com/your-org/AR-AI-Knowledge-Graph](https://github.com/your-org/AR-AI-Knowledge-Graph)
- **Issues**: Report bugs and request features
- **Discussions**: Share ideas and get help
- **Pull Requests**: Contribute improvements

### Performance Tips

1. **Use the Intelligent Router**: Let the system choose the optimal provider
   ```bash
   agentic-flow --agent coder --task "..." --optimize --priority performance
   ```

2. **Cache Common Operations**: Enable caching for repeated tasks
   ```bash
   export ENABLE_CACHE=true
   ```

3. **Leverage Local Models**: Use Xinference/ONNX for cost savings
   ```bash
   export PRIMARY_PROVIDER=xinference
   ```

4. **Monitor Resource Usage**:
   ```bash
   docker stats agentic-flow-cachyos
   docker exec -it agentic-flow-cachyos htop
   ```

5. **Optimize GPU Usage**: Ensure GPU is utilized for ONNX
   ```bash
   export ONNX_EXECUTION_PROVIDER=cuda
   ```

### Security Best Practices

1. **Secure API Keys**: Never commit `.env` file
2. **Change Default Secrets**: Update `MANAGEMENT_API_KEY` in production
3. **Network Isolation**: Use Docker networks appropriately
4. **Regular Updates**: Keep containers and dependencies updated
5. **Monitor Logs**: Regularly review logs for anomalies

---

## Quick Reference Card

```bash
# Start/Stop
./start-agentic-flow.sh              # Start services
./start-agentic-flow.sh --stop       # Stop services
./start-agentic-flow.sh --restart    # Restart services

# Build
./start-agentic-flow.sh --build      # Build containers

# Monitoring
./start-agentic-flow.sh --status     # Check status
./start-agentic-flow.sh --logs       # View logs

# Shell Access
docker exec -it agentic-flow-cachyos zsh

# Basic Task
agentic-flow --agent coder --task "your task"

# API Request
curl -X POST http://localhost:9090/v1/tasks \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"agent":"coder","task":"your task"}'

# Health Check
curl http://localhost:9090/health

# Test Everything
/opt/scripts/test-all-providers.sh
```

---

## Success Checklist

- [ ] Docker and Docker Compose installed
- [ ] NVIDIA drivers installed (if using GPU)
- [ ] Repository cloned
- [ ] `.env` file configured with at least one API key
- [ ] Containers built successfully
- [ ] Services started and healthy
- [ ] Health endpoints responding
- [ ] GPU detected (if applicable)
- [ ] At least one provider working
- [ ] First task completed successfully
- [ ] Management API accessible
- [ ] Documentation reviewed

**Congratulations!** You're ready to build with Agentic Flow.

For more advanced usage, explore the [Architecture Guide](ARCHITECTURE-SIMPLIFIED.md) and [API Documentation](http://localhost:9090/docs).

---

**Need Help?** Open an issue on GitHub or review the troubleshooting section above.
