# Agentic Flow Docker Environment

**GPU-accelerated Docker workstation for AI agent development with multi-model orchestration, MCP tool integration, and Claude Z.AI wrapper.**

## Overview

This Docker environment provides a complete, production-ready platform for developing and deploying AI agent workflows. It includes:

- **GPU-Accelerated Workstation**: CachyOS-based container with NVIDIA GPU support
- **Multi-Model Router**: Intelligent routing across Gemini, OpenAI, Claude, and OpenRouter
- **Management API**: RESTful HTTP API for task management and system monitoring
- **Claude-ZAI Service**: High-performance Claude AI wrapper with worker pool
- **MCP Integration**: Model Context Protocol tools for Claude integration
- **Desktop Environment**: Optional VNC/noVNC for GUI applications
- **Development Tools**: Full development toolchain with VS Code server option

## Quick Start

```bash
# 1. Clone and navigate
cd multi-agent-docker

# 2. Configure API keys
cp .env.example .env
nano .env  # Add your API keys

# 3. Start services
./start-agentic-flow.sh --build

# 4. Verify services
curl http://localhost:9090/health  # Management API
curl http://localhost:9600/health  # Claude-ZAI
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Management API** | 9090 | Primary interface for task management and monitoring |
| **Claude-ZAI** | 9600 | Claude AI wrapper with Z.AI integration |
| **VNC** | 5901 | Remote desktop (if `ENABLE_DESKTOP=true`) |
| **noVNC** | 6901 | Browser-based VNC (if `ENABLE_DESKTOP=true`) |
| **Code Server** | 8080 | VS Code in browser (if `ENABLE_CODE_SERVER=true`) |

## Key Features

### Multi-Model AI Orchestration
- **Intelligent Router**: Automatic provider selection based on task requirements
- **Fallback Chain**: Graceful degradation across multiple providers
- **Performance Optimization**: Route to fastest provider for latency-sensitive tasks
- **Cost Optimization**: Balance quality and cost based on configurable priorities

### GPU Acceleration
- **NVIDIA GPU Support**: Full CUDA and GPU compute capabilities
- **Resource Management**: Configurable GPU memory and compute limits
- **Multi-GPU**: Support for multiple GPU allocation
- **Monitoring**: Real-time GPU utilization and temperature tracking

### Task Isolation
- **Process Isolation**: Each task runs in dedicated workspace
- **Database Isolation**: Separate SQLite databases per task
- **Log Separation**: Individual log files for each task
- **Resource Limits**: Per-task CPU and memory constraints

### MCP Tool Ecosystem
- **Claude Flow**: Agentic workflow orchestration
- **Context7**: Up-to-date code documentation
- **Playwright**: Browser automation
- **Filesystem**: Workspace file operations
- **Git/GitHub**: Version control and repository management
- **Web Tools**: Fetch, search, and summarize web content

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Management API (Port 9090)               │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ Task Manager│  │ System Monitor│  │ Process Manager  │   │
│  └─────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────▼────────┐     │     ┌────────▼──────────┐
    │  Model Router    │     │     │ Claude-ZAI (9600) │
    │  ┌────────────┐  │     │     │  Worker Pool      │
    │  │  Gemini    │  │     │     └───────────────────┘
    │  │  OpenAI    │  │     │
    │  │  Claude    │  │     │
    │  │ OpenRouter │  │     │
    │  └────────────┘  │     │
    └──────────────────┘     │
                              │
                    ┌─────────▼──────────┐
                    │   MCP Tools        │
                    │  ┌──────────────┐  │
                    │  │ Claude Flow  │  │
                    │  │ Filesystem   │  │
                    │  │ Playwright   │  │
                    │  │ Git/GitHub   │  │
                    │  │ Web Tools    │  │
                    │  └──────────────┘  │
                    └────────────────────┘
```

## Documentation Structure

### Getting Started
- [**Getting Started Guide**](GETTING_STARTED.md) - Detailed setup and first steps
- [**Quick Reference**](reference/QUICK_REFERENCE.md) - Command cheat sheet

### Core Documentation
- [**Architecture**](ARCHITECTURE.md) - System design and components
- [**API Reference**](API_REFERENCE.md) - Complete API documentation
- [**Configuration**](CONFIGURATION.md) - All configuration options
- [**Deployment**](DEPLOYMENT.md) - Production deployment guide
- [**Troubleshooting**](TROUBLESHOOTING.md) - Common issues and solutions

### User Guides
- [**Multi-Model Router Guide**](guides/MULTI_MODEL_ROUTER.md) - Configure model routing
- [**MCP Tools Guide**](guides/MCP_TOOLS.md) - Using MCP integrations
- [**GPU Configuration**](guides/GPU_CONFIGURATION.md) - GPU setup and optimization
- [**Task Management**](guides/TASK_MANAGEMENT.md) - Creating and monitoring tasks
- [**Desktop Environment**](guides/DESKTOP_ENVIRONMENT.md) - Using VNC and GUI tools

### Reference
- [**Environment Variables**](reference/ENVIRONMENT_VARIABLES.md) - Complete env var reference
- [**Scripts Reference**](reference/SCRIPTS.md) - Helper script documentation
- [**Docker Reference**](reference/DOCKER.md) - Docker commands and volumes

## Common Tasks

### Create and Monitor a Task

```bash
# Set API key
export API_KEY="your-management-api-key"

# Create task
curl -X POST http://localhost:9090/v1/tasks \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Build a REST API with Express",
    "provider": "gemini"
  }'

# Monitor task (returns task ID)
TASK_ID="<task-id-from-response>"
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks/$TASK_ID
```

### Use Claude-ZAI Wrapper

```bash
# Simple prompt
curl -X POST http://localhost:9600/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain Docker in 3 sentences",
    "timeout": 15000
  }'
```

### Access Desktop Environment

```bash
# Enable desktop in .env
ENABLE_DESKTOP=true

# Rebuild and start
./start-agentic-flow.sh --build

# Access via browser
open http://localhost:6901
```

### Check System Status

```bash
# Overall system status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq

# GPU status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status | jq '.gpu'

# Active tasks
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks | jq '.count'
```

### Shell Access

```bash
# Access workstation container
docker exec -it agentic-flow-cachyos zsh

# Or use the helper script
./start-agentic-flow.sh --shell
```

## Environment Configuration

Create `.env` from template and configure:

```bash
# Required: At least one AI provider
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_GEMINI_API_KEY=...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=...

# Management API
MANAGEMENT_API_KEY=change-this-secret-key

# Router Configuration
ROUTER_MODE=performance
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,openai,claude,openrouter

# GPU
GPU_ACCELERATION=true

# Optional Services
ENABLE_DESKTOP=false
ENABLE_CODE_SERVER=false
```

## Deployment Modes

### Standalone (Production)
- Installs `agentic-flow` from npm
- Only requires this directory
- Always uses latest published version

```bash
docker compose up -d --build
```

### Development
- Uses local source code
- For development and testing
- Requires full repository

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

## Persistent Data

Data is stored in Docker volumes:

- `workspace` - User workspace files
- `model-cache` - Downloaded AI models
- `agent-memory` - Agent session data
- `config-persist` - Configuration files
- `management-logs` - API and task logs

```bash
# List volumes
docker volume ls | grep cachyos

# Backup workspace
docker run --rm -v workspace:/data -v $(pwd):/backup \
  alpine tar czf /backup/workspace-backup.tar.gz /data
```

## Requirements

- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA GPU (optional, for GPU acceleration)
- NVIDIA Docker runtime (for GPU support)
- At least 16GB RAM (32GB+ recommended)
- 50GB free disk space

## Security

1. **Change Default API Key**: Set strong `MANAGEMENT_API_KEY`
2. **Network Isolation**: Use Docker networks
3. **Restrict Ports**: Firewall access to 9090, 9600
4. **Use Secrets**: Mount secrets instead of environment variables
5. **Regular Updates**: Rebuild with `--build` for latest packages
6. **TLS/HTTPS**: Use reverse proxy (nginx/traefik) in production

## Monitoring

### Health Checks

```bash
# Management API health
curl http://localhost:9090/health

# Claude-ZAI health
curl http://localhost:9600/health

# Detailed system status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/status
```

### Logs

```bash
# Service logs
docker compose logs -f

# Management API logs
docker logs agentic-flow-cachyos | grep management-api

# Task logs
docker exec agentic-flow-cachyos cat ~/logs/tasks/<task-id>.log
```

## Support

- **Documentation**: This directory
- **Issues**: [GitHub Issues](https://github.com/ruvnet/agentic-flow/issues)
- **Package**: [npm: agentic-flow](https://www.npmjs.com/package/agentic-flow)

## License

See main repository for license information.

## Next Steps

1. [Set up your environment](GETTING_STARTED.md)
2. [Understand the architecture](ARCHITECTURE.md)
3. [Configure the model router](guides/MULTI_MODEL_ROUTER.md)
4. [Explore the API](API_REFERENCE.md)
5. [Deploy to production](DEPLOYMENT.md)
