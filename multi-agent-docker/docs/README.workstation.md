# Agentic Flow CachyOS Workstation

**Interactive Development Environment** for Agentic Flow with all model providers, GPU acceleration, and RAGFlow network integration.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage Patterns](#usage-patterns)
- [Provider Configuration](#provider-configuration)
- [Network Architecture](#network-architecture)
- [GPU Support](#gpu-support)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

---

## Overview

This CachyOS-based Docker workstation provides a full development environment where you **shell into the container** and run all commands internally, rather than calling from the host.

### Architecture Philosophy

```
┌─────────────────────────────────────────────────────────────┐
│  Host System                                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  CachyOS Docker Container (agentic-flow-cachyos)     │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │  Interactive Shell (zsh)                        │ │  │
│  │  │  ├─ Claude Code / Claude Flow                   │ │  │
│  │  │  ├─ Agentic Flow CLI                            │ │  │
│  │  │  ├─ Intelligent Router                          │ │  │
│  │  │  ├─ 6 Model Providers                           │ │  │
│  │  │  └─ 213 MCP Tools                               │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                                                         │  │
│  │  Connected to: docker_ragflow network                  │  │
│  │  GPU: NVIDIA/AMD passthrough                           │  │
│  │  RAM: 64GB allocation                                  │  │
│  │  Storage: 50GB volumes                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Features

### ✅ Complete Development Environment
- **CachyOS Base**: Optimized Arch Linux with performance kernel
- **Minimal XFCE Desktop**: For visual debugging (no screensaver/lock)
- **Zsh + Oh-My-Zsh**: Enhanced shell with powerful aliases
- **Development Tools**: git, vim, neovim, tmux, htop, and more

### ✅ Six Model Providers
1. **Google Gemini** - Fast, cost-effective (98% cost savings, 1M context)
2. **OpenAI** - GPT-4o, excellent quality
3. **Anthropic Claude** - Highest quality for complex tasks
4. **OpenRouter** - 99% cost savings, 100+ models
5. **Xinference** - FREE local inference via RAGFlow network
6. **ONNX** - FREE offline GPU-accelerated inference

### ✅ Intelligent Model Router
- **Performance mode**: Optimize for speed and quality
- **Cost mode**: Minimize API costs
- **Quality mode**: Best results for complex tasks
- **Balanced mode**: Even trade-offs
- **Offline mode**: Local inference only

### ✅ RAGFlow Network Integration
- Connected to existing `docker_ragflow` network
- Access to Xinference at `172.18.0.11:9997`
- Free local model inference
- No internet required for local models

### ✅ GPU Support
- NVIDIA GPU passthrough (64GB RAM allocated)
- AMD GPU support (ROCm)
- GPU-accelerated ONNX inference
- CUDA/ROCm execution providers

### ✅ MCP Tools (213 Total)
- **Claude Flow**: 101 tools (GitHub, neural networks, workflows)
- **Flow Nexus**: 96 cloud tools (sandboxes, swarms, storage)
- **Agentic Payments**: Payment authorization tools
- **Claude Flow SDK**: 6 in-process tools

---

## Quick Start

### 1. Setup Environment

```bash
cd docker/cachyos

# Copy environment template
cp config/providers.env.template .env

# Edit .env and add your API keys
nano .env
```

**Required** (at least one):
- `ANTHROPIC_API_KEY=sk-ant-...`
- `OPENAI_API_KEY=sk-proj-...`
- `GOOGLE_GEMINI_API_KEY=AIza...`
- `OPENROUTER_API_KEY=sk-or-v1-...`

### 2. Build and Start Container

```bash
# Build image
docker-compose -f docker-compose.workstation.yml build

# Start container
docker-compose -f docker-compose.workstation.yml up -d

# Check status
docker ps | grep agentic-flow-cachyos
```

### 3. Shell Into Container

```bash
# Enter interactive shell
docker exec -it agentic-flow-cachyos zsh

# You're now inside the container!
# All commands run from here
```

### 4. Test Setup

```bash
# Inside container

# Check API keys
check-keys

# List available agents
afl

# Test all providers
test-providers

# Check GPU
test-gpu
```

---

## Usage Patterns

### Basic Agent Execution

```bash
# Standard format
agentic-flow --agent <agent> --task "<task description>"

# Example
agentic-flow --agent coder --task "Build a REST API with JWT authentication"
```

### Provider-Specific Execution

```bash
# Google Gemini (fast, cost-effective)
af-gemini --agent coder --task "Build REST API"

# OpenAI GPT-4o
af-openai --agent coder --task "Build REST API"

# Anthropic Claude (highest quality)
af-claude --agent reviewer --task "Review this code"

# OpenRouter (99% cost savings)
af-router --agent coder --task "Build REST API"

# Xinference (free local)
af-local --agent coder --task "Build REST API"

# ONNX (offline)
af-offline --agent coder --task "Build REST API"
```

### Intelligent Router

```bash
# Auto-select best model (performance mode)
af-optimize --agent coder --task "Build REST API"

# Optimize for performance
af-perf --agent coder --task "Build complex system"

# Optimize for cost
af-cost --agent researcher --task "Analyze trends"

# Optimize for quality
af-quality --agent reviewer --task "Security audit"

# Balanced optimization
af-balanced --agent backend --task "Create GraphQL API"
```

### Quick Aliases

```bash
# Agent shortcuts
coder "Build REST API with JWT auth"
reviewer "Review this code for security"
researcher "Latest LLM benchmarks 2025"
tester "Generate unit tests for this function"

# Provider shortcuts
gemini-coder "Build API"
gpt-coder "Build API"
claude-coder "Build API"
local-coder "Build API"    # Xinference
offline-coder "Build API"  # ONNX
```

### MCP Server Management

```bash
# Start MCP servers
mcp-start

# List all 213 tools
mcp-list

# Check server status
mcp-status

# Add custom MCP server
mcp-add weather '{"command":"npx","args":["-y","weather-mcp"]}'

# Remove MCP server
mcp-remove weather
```

---

## Provider Configuration

### Model Provider Priority (Default)

1. **Gemini** (Primary) - Fast, cost-effective, 1M context
2. **OpenAI** - Excellent quality, GPT-4o
3. **Claude** - Highest quality, complex tasks only
4. **OpenRouter** - Budget option, many models
5. **Xinference** - Free local (RAGFlow network)
6. **ONNX** - Free offline (GPU-accelerated)

### Customizing Provider Order

Edit `~/.config/agentic-flow/router.config.json`:

```json
{
  "routing": {
    "modes": {
      "custom": {
        "defaultChain": ["xinference", "gemini", "openai"],
        "fallbackChain": ["xinference", "onnx", "gemini", "openai"]
      }
    }
  }
}
```

Then use:

```bash
export ROUTER_MODE=custom
af-optimize --agent coder --task "..."
```

---

## Network Architecture

### RAGFlow Integration

The container connects to the existing `docker_ragflow` network, providing access to:

```
docker_ragflow Network (172.18.0.0/16)
├── Xinference:          172.18.0.11:9997  (FREE local models)
├── RAGFlow MinIO:       172.18.0.2
├── RAGFlow MySQL:       172.18.0.8
├── RAGFlow Elasticsearch: 172.18.0.7
└── agentic-flow-cachyos: 172.18.0.x (this container)
```

### Xinference Access

```bash
# Inside container

# List available models
curl -s http://172.18.0.11:9997/v1/models | jq

# Use Xinference
af-local --agent coder --task "Hello world"

# Test connectivity
test-xinference
```

### Provider Endpoints

- **Xinference**: `http://172.18.0.11:9997/v1` (internal network)
- **Anthropic**: `https://api.anthropic.com/v1` (internet)
- **OpenAI**: `https://api.openai.com/v1` (internet)
- **Gemini**: `https://generativelanguage.googleapis.com/v1beta` (internet)
- **OpenRouter**: `https://openrouter.ai/api/v1` (internet)
- **ONNX**: Local filesystem (`/home/devuser/models/phi-4.onnx`)

---

## GPU Support

### NVIDIA GPU

The container has full NVIDIA GPU access:

```bash
# Check GPU status
nvidia-smi

# Show GPU details
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### AMD GPU

For AMD GPUs using ROCm:

```bash
# Check GPU
rocm-smi

# Show device info
rocm-smi --showproductname
```

### GPU-Accelerated ONNX

ONNX uses GPU automatically if available:

```bash
# Run with GPU acceleration
af-offline --agent coder --task "Build API"

# Performance:
# - CPU: ~6 tokens/sec
# - GPU (CUDA): 60-300 tokens/sec
```

To configure ONNX execution provider:

```bash
# NVIDIA
export ONNX_EXECUTION_PROVIDER=cuda

# AMD
export ONNX_EXECUTION_PROVIDER=rocm

# CPU only
export ONNX_EXECUTION_PROVIDER=cpu
```

---

## Troubleshooting

### API Key Issues

```bash
# Check which keys are set
check-keys

# Test specific provider
test-provider gemini
test-provider openai
test-provider anthropic
```

### Xinference Not Reachable

```bash
# Check network connectivity
ping 172.18.0.11

# Test Xinference API
curl http://172.18.0.11:9997/v1/models

# Verify network
docker network inspect docker_ragflow | grep -A5 agentic-flow-cachyos
```

### GPU Not Detected

```bash
# Check GPU visibility
nvidia-smi || rocm-smi

# Check environment
echo $CUDA_VISIBLE_DEVICES

# Verify runtime
docker inspect agentic-flow-cachyos | grep -i runtime
```

### MCP Servers Not Starting

```bash
# Manual start
mcp-start

# Check logs
cat /tmp/mcp-startup.log

# List processes
ps aux | grep mcp
```

### Agent Command Not Found

```bash
# Check installation
which agentic-flow

# Reinstall
cd /tmp/agentic-flow && npm link

# Verify PATH
echo $PATH
```

---

## Advanced Configuration

### Custom Model Selection

Create `~/.config/agentic-flow/custom-models.json`:

```json
{
  "providers": {
    "gemini": {
      "models": {
        "default": "gemini-2.0-flash-exp"
      }
    },
    "openrouter": {
      "models": {
        "default": "deepseek/deepseek-r1"
      }
    }
  }
}
```

### Cost Budgets

Edit `router.config.json`:

```json
{
  "costTracking": {
    "budgetLimits": {
      "daily": 5.0,
      "weekly": 25.0,
      "monthly": 100.0
    }
  }
}
```

### Performance Tuning

```bash
# Environment variables in .env

# Concurrent requests
MAX_CONCURRENT_REQUESTS=10

# Timeouts
REQUEST_TIMEOUT=120000
MAX_RETRIES=5

# Rate limiting
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=2000
```

### Adding Custom Agents

```bash
# Create custom agent
mkdir -p ~/.claude/agents/custom
cat > ~/.claude/agents/custom/my-agent.md <<EOF
---
name: my-agent
description: Custom agent for specific tasks
---

# System Prompt

You are a specialized agent for [your use case].

## Capabilities
- [List capabilities]

## Guidelines
- [Execution guidelines]
EOF

# Use custom agent
agentic-flow --agent my-agent --task "Your task"
```

---

## Workspace Management

### Directory Structure

```
/home/devuser/
├── workspace/          # Your projects
│   ├── projects/       # Active development
│   ├── temp/           # Temporary files
│   └── agents/         # Custom agents
├── models/             # ONNX models, embeddings
├── .claude-flow/       # Agent memory and metrics
│   ├── memory/         # Persistent agent memory
│   ├── metrics/        # Performance metrics
│   └── logs/           # Execution logs
└── .config/
    ├── agentic-flow/   # Router configuration
    └── claude/         # MCP server configuration
```

### Persistent Volumes

Data persists across container restarts:

- **Workspace**: `/home/devuser/workspace` → Docker volume
- **Models**: `/home/devuser/models` → Docker volume
- **Memory**: `/home/devuser/.claude-flow` → Docker volume
- **Config**: `/home/devuser/.config` → Docker volume

### Backup and Restore

```bash
# Backup volumes
docker run --rm -v agentic-cachyos-workspace:/data -v $(pwd):/backup \
  alpine tar czf /backup/workspace-backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v agentic-cachyos-workspace:/data -v $(pwd):/backup \
  alpine sh -c "cd /data && tar xzf /backup/workspace-backup.tar.gz"
```

---

## Development Workflow

### Typical Session

```bash
# 1. Start container (if not running)
docker-compose -f docker-compose.workstation.yml up -d

# 2. Shell in
docker exec -it agentic-flow-cachyos zsh

# 3. Navigate to workspace
cd ~/workspace

# 4. Clone/create project
git clone https://github.com/your/repo
cd repo

# 5. Use agentic-flow
af-gemini --agent coder --task "Add authentication"

# 6. Test changes
af-claude --agent reviewer --task "Review security"

# 7. Generate tests
af-cost --agent tester --task "Create unit tests"

# 8. Commit
git add . && git commit -m "Add auth" && git push
```

### Integrating with Claude Code

```bash
# Inside container
cd ~/workspace/your-project

# Open in VSCode (if installed)
code .

# Run agentic-flow from integrated terminal
af --optimize --agent backend-dev --task "Add GraphQL endpoint"
```

---

## Stopping and Cleanup

### Stop Container

```bash
# Stop (keeps volumes)
docker-compose -f docker-compose.workstation.yml stop

# Stop and remove container
docker-compose -f docker-compose.workstation.yml down

# Remove everything (including volumes)
docker-compose -f docker-compose.workstation.yml down -v
```

### Restart Container

```bash
# Restart
docker-compose -f docker-compose.workstation.yml restart

# Rebuild and restart
docker-compose -f docker-compose.workstation.yml up -d --build
```

---

## Cost Comparison

### Monthly Cost Estimates (1000 requests/month)

| Provider | Model | Cost | Use Case |
|----------|-------|------|----------|
| **Xinference** | Local | $0 | FREE local inference |
| **ONNX** | Phi-4 | $0 | FREE offline inference |
| **OpenRouter** | Llama 3.1 8B | ~$0.30 | Simple tasks (99% savings) |
| **Gemini** | 2.5 Flash | ~$3 | Fast, balanced (98% savings) |
| **OpenAI** | GPT-4o | ~$30 | High quality (75% savings) |
| **Claude** | Sonnet 3.5 | ~$80 | Highest quality (baseline) |

**Recommendation**: Use intelligent router to automatically select optimal provider:

```bash
af-optimize --agent coder --task "..."
```

Expected savings: **70-90%** compared to using Claude exclusively.

---

## Support and Resources

### Documentation
- [Main README](../../README.md) - Project overview
- [Docker README](../README.md) - Docker deployment guide
- [ONNX Integration](../../docs/ONNX_INTEGRATION.md) - ONNX usage
- [Model Capabilities](../../docs/agentic-flow/benchmarks/MODEL_CAPABILITIES.md) - Model comparison

### Community
- **GitHub**: https://github.com/ruvnet/agentic-flow
- **Issues**: https://github.com/ruvnet/agentic-flow/issues
- **npm**: https://www.npmjs.com/package/agentic-flow

### Inside Container Help

```bash
# Agentic Flow help
afh

# List agents
afl

# MCP status
mcp-status

# Test providers
test-providers

# Check keys
check-keys
```

---

## License

MIT License - see [LICENSE](../../LICENSE) for details.

---

**Deploy your AI workstation in minutes. Scale to thousands of agents. Pay only for what you use.** 🚀
