# Configuration Guide

Complete configuration reference for the Agentic Flow Docker environment.

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Router Configuration](#router-configuration)
3. [MCP Configuration](#mcp-configuration)
4. [Docker Compose Configuration](#docker-compose-configuration)
5. [Resource Limits](#resource-limits)
6. [GPU Configuration](#gpu-configuration)
7. [Provider Configuration](#provider-configuration)
8. [Optional Services](#optional-services)
9. [Configuration Examples](#configuration-examples)

---

## Environment Variables

Create a `.env` file from `.env.example` in the `multi-agent-docker` directory.

### API Keys

At least one AI provider API key is required.

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Recommended | Anthropic Claude API key | - |
| `ANTHROPIC_BASE_URL` | For Z.AI | Z.AI API endpoint for Claude | `https://api.z.ai/api/anthropic` |
| `OPENAI_API_KEY` | Optional | OpenAI API key | - |
| `GOOGLE_GEMINI_API_KEY` | Optional | Google Gemini API key | - |
| `OPENROUTER_API_KEY` | Optional | OpenRouter API key (multi-model access) | - |
| `GITHUB_TOKEN` | Optional | GitHub personal access token | - |
| `CONTEXT7_API_KEY` | Optional | Context7 API key for documentation | - |
| `BRAVE_API_KEY` | Optional | Brave Search API key | - |

### Management API

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `MANAGEMENT_API_KEY` | Yes | Secret key for API authentication | `change-this-secret-key` |
| `MANAGEMENT_API_PORT` | No | Management API port | `9090` |
| `MANAGEMENT_API_HOST` | No | Management API bind address | `0.0.0.0` |

### Model Router

| Variable | Required | Description | Default | Options |
|----------|----------|-------------|---------|---------|
| `ROUTER_MODE` | No | Routing optimization mode | `performance` | `performance`, `cost`, `quality`, `balanced`, `offline` |
| `PRIMARY_PROVIDER` | No | Primary AI provider | `gemini` | `gemini`, `openai`, `claude`, `openrouter`, `xinference`, `onnx` |
| `FALLBACK_CHAIN` | No | Comma-separated provider fallback order | `gemini,openai,claude,openrouter` | Any combination of providers |

### GPU Configuration

| Variable | Required | Description | Default | Options |
|----------|----------|-------------|---------|---------|
| `GPU_ACCELERATION` | No | Enable GPU acceleration | `true` | `true`, `false` |
| `CUDA_VISIBLE_DEVICES` | No | GPU device selection | `all` | `all`, `0`, `0,1`, etc. |

### Optional Services

| Variable | Required | Description | Default | Options |
|----------|----------|-------------|---------|---------|
| `ENABLE_DESKTOP` | No | Enable VNC/noVNC desktop | `false` | `true`, `false` |
| `ENABLE_CODE_SERVER` | No | Enable VS Code Server | `false` | `true`, `false` |

### System Configuration

| Variable | Required | Description | Default | Options |
|----------|----------|-------------|---------|---------|
| `LOG_LEVEL` | No | Logging verbosity | `info` | `error`, `warn`, `info`, `debug`, `trace` |
| `NODE_ENV` | No | Node.js environment | `production` | `production`, `development` |
| `DISPLAY` | No | X11 display for GUI apps | `:0` | Any valid display |

### Claude-ZAI Worker Configuration

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `CLAUDE_WORKER_POOL_SIZE` | No | Number of Claude worker processes | `4` |
| `CLAUDE_MAX_QUEUE_SIZE` | No | Maximum queue size for Claude tasks | `50` |

---

## Router Configuration

The router configuration file at `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/config/router.config.json` controls intelligent model routing.

### Configuration Structure

```json
{
  "version": "1.0.0",
  "mode": "performance",
  "providers": { ... },
  "routing": { ... },
  "costTracking": { ... },
  "performanceMonitoring": { ... },
  "fallbackBehavior": { ... }
}
```

### Routing Modes

Configure via `ROUTER_MODE` environment variable or `mode` field in config file.

#### performance (Default)
- **Weights**: Speed 50%, Quality 40%, Cost 10%
- **Default Chain**: `gemini` → `openai` → `claude`
- **Fallback Chain**: `gemini` → `openai` → `claude` → `openrouter` → `xinference` → `onnx`
- **Use Case**: Production workloads requiring fast, high-quality responses

#### cost
- **Weights**: Speed 20%, Quality 30%, Cost 50%
- **Default Chain**: `xinference` → `openrouter` → `gemini`
- **Fallback Chain**: `xinference` → `onnx` → `openrouter` → `gemini` → `openai`
- **Use Case**: Development, testing, high-volume low-stakes tasks

#### quality
- **Weights**: Speed 10%, Quality 70%, Cost 20%
- **Default Chain**: `claude` → `openai` → `gemini`
- **Fallback Chain**: `claude` → `openai` → `gemini` → `openrouter`
- **Use Case**: Code reviews, architecture decisions, complex reasoning

#### balanced
- **Weights**: Speed 33%, Quality 34%, Cost 33%
- **Default Chain**: `gemini` → `openai` → `openrouter`
- **Fallback Chain**: `gemini` → `openai` → `openrouter` → `claude` → `xinference`
- **Use Case**: General-purpose workflows

#### offline
- **Weights**: Speed 50%, Quality 50%, Cost 0%
- **Default Chain**: `xinference` → `onnx`
- **Fallback Chain**: `xinference` → `onnx`
- **Use Case**: Privacy-sensitive, air-gapped, or network-constrained environments

### Routing Rules

Conditional routing based on task characteristics:

| Condition | Provider | Reason |
|-----------|----------|--------|
| `privacy: high`, `localOnly: true` | `onnx` | Offline local inference |
| `task: code-review`, `quality: required` | `claude` | Highest quality reasoning |
| `task: code-generation`, `cost: free` | `xinference` (deepseek-coder) | Free code generation |
| `task: simple`, `cost: minimal` | `openrouter` (llama-3.1-8b) | Cheapest option |
| `speed: critical`, `latency: low` | `gemini` (flash) | Fastest model |

### Agent Preferences

Different agent types have different provider preferences:

| Agent | Preferred Providers | Min Quality | Use Case |
|-------|-------------------|-------------|----------|
| `coder` | claude, openai, gemini | 85 | Code generation |
| `reviewer` | claude, openai | 90 | Code review |
| `researcher` | gemini, openai, openrouter | 70 | Research |
| `tester` | openrouter, gemini, xinference | 70 | Test generation |
| `planner` | claude, openai, gemini | 85 | Planning |

### Cost Tracking

```json
{
  "enabled": true,
  "budgetLimits": {
    "daily": 10.0,
    "weekly": 50.0,
    "monthly": 200.0
  },
  "alerts": {
    "thresholds": [0.5, 0.8, 0.95],
    "notificationMethod": "log"
  }
}
```

Alerts trigger at 50%, 80%, and 95% of budget limits.

### Performance Monitoring

```json
{
  "enabled": true,
  "metrics": [
    "latency",
    "tokens_per_second",
    "cost_per_request",
    "success_rate",
    "error_rate"
  ],
  "loggingLevel": "info"
}
```

### Fallback Behavior

```json
{
  "maxRetries": 3,
  "retryDelay": 1000,
  "circuitBreaker": {
    "enabled": true,
    "failureThreshold": 5,
    "timeout": 60000
  }
}
```

- **maxRetries**: Retry failed requests up to 3 times
- **retryDelay**: Wait 1 second between retries
- **Circuit Breaker**: Open circuit after 5 failures, reset after 60 seconds

---

## Provider Configuration

Each provider has specific configuration in `router.config.json`.

### Gemini (Google)

```json
{
  "enabled": true,
  "baseUrl": "https://generativelanguage.googleapis.com/v1beta",
  "apiKey": "${GOOGLE_GEMINI_API_KEY}",
  "models": {
    "default": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro",
    "exp": "gemini-2.0-flash-exp"
  },
  "metrics": {
    "speed": 95,
    "quality": 85,
    "cost": 98,
    "reliability": 92
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 1000000,
    "maxTokens": 8192
  },
  "priority": 1
}
```

**Strengths**:
- 1M token context window
- Fast response times
- Cost-effective
- Good for high-volume tasks

**Best For**: Research, data analysis, long-context tasks

### OpenAI

```json
{
  "enabled": true,
  "baseUrl": "https://api.openai.com/v1",
  "apiKey": "${OPENAI_API_KEY}",
  "models": {
    "default": "gpt-4o",
    "mini": "gpt-4o-mini",
    "legacy": "gpt-4-turbo"
  },
  "metrics": {
    "speed": 85,
    "quality": 90,
    "cost": 70,
    "reliability": 95
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 128000,
    "maxTokens": 4096
  },
  "priority": 2
}
```

**Strengths**:
- Excellent quality
- Strong reasoning
- Reliable
- Good tool calling

**Best For**: General-purpose tasks, complex reasoning

### Anthropic Claude

```json
{
  "enabled": true,
  "baseUrl": "https://api.anthropic.com/v1",
  "apiKey": "${ANTHROPIC_API_KEY}",
  "models": {
    "default": "claude-3-5-sonnet-20241022",
    "haiku": "claude-3-5-haiku-20241022",
    "opus": "claude-3-opus-20240229"
  },
  "metrics": {
    "speed": 80,
    "quality": 95,
    "cost": 40,
    "reliability": 98
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 200000,
    "maxTokens": 8192
  },
  "priority": 3,
  "useFor": [
    "code-review",
    "architecture",
    "complex-reasoning",
    "refactoring",
    "security-analysis"
  ]
}
```

**Strengths**:
- Highest quality reasoning
- Excellent for code review
- Strong security analysis
- Reliable

**Best For**: Code review, architecture, security analysis, complex refactoring

### OpenRouter

```json
{
  "enabled": true,
  "baseUrl": "https://openrouter.ai/api/v1",
  "apiKey": "${OPENROUTER_API_KEY}",
  "models": {
    "default": "meta-llama/llama-3.1-8b-instruct",
    "code": "deepseek/deepseek-coder-v2",
    "chat": "deepseek/deepseek-chat",
    "reasoning": "deepseek/deepseek-r1",
    "claude": "anthropic/claude-3.5-sonnet"
  },
  "metrics": {
    "speed": 75,
    "quality": 75,
    "cost": 99,
    "reliability": 88
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 128000,
    "maxTokens": 4096
  },
  "priority": 4
}
```

**Strengths**:
- 99% cost savings
- 100+ models available
- Multiple specialized models
- Flexible routing

**Best For**: Cost-conscious development, simple tasks, experimentation

### Xinference (Local Network)

```json
{
  "enabled": true,
  "baseUrl": "http://172.18.0.11:9997/v1",
  "apiKey": "none",
  "models": {
    "default": "auto",
    "code": "deepseek-coder",
    "chat": "qwen2.5"
  },
  "metrics": {
    "speed": 70,
    "quality": 75,
    "cost": 100,
    "reliability": 85
  },
  "features": {
    "streaming": true,
    "toolCalling": true,
    "contextWindow": 32768,
    "maxTokens": 2048,
    "localInference": true,
    "networkRequired": true
  },
  "priority": 5
}
```

**Strengths**:
- FREE inference via RAGFlow network
- Local network latency
- Good for testing
- No API costs

**Best For**: Development, testing, cost-free experimentation

### ONNX (Offline Local)

```json
{
  "enabled": true,
  "modelPath": "/home/devuser/models/phi-4.onnx",
  "executionProviders": ["cuda", "cpu"],
  "models": {
    "default": "phi-4-mini-instruct"
  },
  "metrics": {
    "speed": 60,
    "quality": 70,
    "cost": 100,
    "reliability": 90
  },
  "features": {
    "streaming": false,
    "toolCalling": false,
    "contextWindow": 4096,
    "maxTokens": 2048,
    "localInference": true,
    "offline": true,
    "gpuAccelerated": true
  },
  "priority": 6
}
```

**Strengths**:
- Completely offline
- GPU accelerated
- Zero API costs
- Privacy-preserving

**Best For**: Air-gapped environments, privacy-sensitive tasks, offline operation

---

## MCP Configuration

MCP (Model Context Protocol) servers provide tools for AI agents. Configuration file: `config/mcp.json`

### Available MCP Servers

| Server | Command | Description | Required Env |
|--------|---------|-------------|--------------|
| `claude-flow` | `npx -y claude-flow mcp start` | Agentic workflow integration | - |
| `context7` | `npx -y @upstash/context7-mcp` | Up-to-date API documentation | `CONTEXT7_API_KEY` |
| `playwright` | `npx -y @modelcontextprotocol/server-playwright` | Browser automation | `DISPLAY` |
| `filesystem` | `npx -y @modelcontextprotocol/server-filesystem` | File operations in workspace | - |
| `git` | `npx -y @modelcontextprotocol/server-git` | Git operations | - |
| `github` | `npx -y @modelcontextprotocol/server-github` | GitHub API operations | `GITHUB_TOKEN` |
| `fetch` | `npx -y @modelcontextprotocol/server-fetch` | HTTP requests | - |
| `brave-search` | `npx -y @modelcontextprotocol/server-brave-search` | Web search | `BRAVE_API_KEY` |
| `web-summary` | Python script | Web/YouTube summarization | `GOOGLE_API_KEY` |

### Tool Categories

```json
{
  "documentation": ["context7"],
  "automation": ["playwright"],
  "filesystem": ["filesystem", "git"],
  "web": ["fetch", "brave-search", "web-summary"],
  "github": ["github"],
  "workflows": ["claude-flow"]
}
```

### MCP Configuration Options

```json
{
  "defaultTimeout": 30000,
  "retryAttempts": 3,
  "logLevel": "info"
}
```

### Architecture Notes

- All MCP servers run via stdio (not HTTP)
- Each worker session spawns its own tool instances
- Tools run on-demand (not persistent services)
- Complete session isolation

---

## Docker Compose Configuration

### Services

#### agentic-flow-cachyos (Main Container)

**Ports**:
- `9090`: Management API (primary interface)
- `5901`: VNC (if `ENABLE_DESKTOP=true`)
- `6901`: noVNC web interface (if `ENABLE_DESKTOP=true`)
- `8080`: VS Code Server (if `ENABLE_CODE_SERVER=true`)

**Volumes**:
- `workspace`: Persistent workspace at `/home/devuser/workspace`
- `model-cache`: Model files at `/home/devuser/models`
- `agent-memory`: Session data at `/home/devuser/.agentic-flow`
- `config-persist`: Configuration at `/home/devuser/.config`
- `management-logs`: Logs at `/home/devuser/logs`
- `/tmp/.X11-unix`: X11 socket for GUI apps

**Device Access**:
- `/dev/dri`: DRI for GPU rendering
- `/dev/nvidia*`: NVIDIA GPU devices

**Healthcheck**:
```yaml
test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
interval: 30s
timeout: 10s
retries: 3
start_period: 60s
```

#### claude-zai (Claude Service)

**Ports**:
- `9600`: Claude-ZAI service API

**Environment**:
- `ANTHROPIC_BASE_URL`: Z.AI endpoint
- `ANTHROPIC_AUTH_TOKEN`: Claude API key
- `CLAUDE_WORKER_POOL_SIZE`: Worker pool size
- `CLAUDE_MAX_QUEUE_SIZE`: Max queue size

**Healthcheck**:
```yaml
test: ["CMD", "curl", "-f", "http://localhost:9600/health"]
interval: 30s
timeout: 10s
retries: 3
start_period: 30s
```

---

## Resource Limits

### Memory Configuration

```yaml
deploy:
  resources:
    limits:
      memory: 64G
    reservations:
      memory: 16G
```

- **Limit**: Maximum 64GB RAM
- **Reservation**: Guaranteed 16GB RAM
- **Shared Memory**: 32GB (`shm_size: 32gb`)

### CPU Configuration

```yaml
deploy:
  resources:
    limits:
      cpus: '32'
    reservations:
      cpus: '8'
```

- **Limit**: Maximum 32 CPU cores
- **Reservation**: Guaranteed 8 CPU cores

### Adjusting Resources

Edit `docker-compose.yml` to match your system:

```yaml
# For 16GB RAM systems
limits:
  memory: 16G
  cpus: '8'
reservations:
  memory: 8G
  cpus: '4'

# For 32GB RAM systems
limits:
  memory: 32G
  cpus: '16'
reservations:
  memory: 12G
  cpus: '6'
```

---

## GPU Configuration

### NVIDIA GPU Setup

**Requirements**:
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed
- Docker with GPU runtime

**Docker Compose Configuration**:

```yaml
runtime: nvidia

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu, compute, utility]
```

### GPU Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `GPU_ACCELERATION` | `true` | Enable GPU acceleration |
| `CUDA_VISIBLE_DEVICES` | `all` | Use all GPUs |
| `CUDA_VISIBLE_DEVICES` | `0` | Use only GPU 0 |
| `CUDA_VISIBLE_DEVICES` | `0,1` | Use GPUs 0 and 1 |

### GPU Device Mapping

```yaml
devices:
  - /dev/dri:/dev/dri
  - /dev/nvidia0:/dev/nvidia0
  - /dev/nvidiactl:/dev/nvidiactl
  - /dev/nvidia-uvm:/dev/nvidia-uvm
  - /dev/nvidia-modeset:/dev/nvidia-modeset
```

### Disabling GPU

Set in `.env`:
```bash
GPU_ACCELERATION=false
```

Or remove GPU configuration from `docker-compose.yml`:
```yaml
# Comment out:
# runtime: nvidia
# devices: [nvidia devices]
```

---

## Optional Services

### Desktop Environment (VNC/noVNC)

Enable GUI applications with VNC access.

**Enable**:
```bash
ENABLE_DESKTOP=true
```

**Access**:
- VNC: `localhost:5901` (VNC client required)
- noVNC: `http://localhost:6901` (web browser)

**Use Cases**:
- Browser automation with Playwright
- GUI development tools
- Visual debugging
- QGIS, Blender, etc.

### VS Code Server

Web-based VS Code IDE inside the container.

**Enable**:
```bash
ENABLE_CODE_SERVER=true
```

**Access**:
- URL: `http://localhost:8080`

**Use Cases**:
- Remote development
- Code editing inside container
- Collaborative development

---

## Configuration Examples

### Example 1: High-Quality Code Review Setup

**.env**:
```bash
# API Keys
ANTHROPIC_API_KEY=your-claude-key
OPENAI_API_KEY=your-openai-key

# Router - Quality mode with Claude primary
ROUTER_MODE=quality
PRIMARY_PROVIDER=claude
FALLBACK_CHAIN=claude,openai,gemini

# Management API
MANAGEMENT_API_KEY=secure-secret-key
MANAGEMENT_API_PORT=9090

# GPU enabled
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=all

# Desktop for browser automation
ENABLE_DESKTOP=true
ENABLE_CODE_SERVER=false

LOG_LEVEL=info
NODE_ENV=production
```

**Use Case**: Code reviews, architecture decisions, security analysis

---

### Example 2: Cost-Optimized Development

**.env**:
```bash
# API Keys - minimal
GOOGLE_GEMINI_API_KEY=your-gemini-key
OPENROUTER_API_KEY=your-openrouter-key

# Router - Cost mode
ROUTER_MODE=cost
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,openrouter,xinference

# Management API
MANAGEMENT_API_KEY=dev-secret
MANAGEMENT_API_PORT=9090

# GPU disabled for cost savings
GPU_ACCELERATION=false

# Optional services disabled
ENABLE_DESKTOP=false
ENABLE_CODE_SERVER=false

LOG_LEVEL=info
NODE_ENV=development
```

**Use Case**: Development, testing, experimentation with minimal costs

---

### Example 3: Performance-Optimized Production

**.env**:
```bash
# API Keys - all providers
ANTHROPIC_API_KEY=your-claude-key
OPENAI_API_KEY=your-openai-key
GOOGLE_GEMINI_API_KEY=your-gemini-key
OPENROUTER_API_KEY=your-openrouter-key

# Router - Performance mode
ROUTER_MODE=performance
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,openai,claude,openrouter

# Management API
MANAGEMENT_API_KEY=production-secret-key
MANAGEMENT_API_PORT=9090

# GPU enabled
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=all

# Optional services
ENABLE_DESKTOP=false
ENABLE_CODE_SERVER=false

# Claude worker pool - large
CLAUDE_WORKER_POOL_SIZE=8
CLAUDE_MAX_QUEUE_SIZE=100

LOG_LEVEL=warn
NODE_ENV=production
```

**Use Case**: Production workloads requiring fast, reliable responses

---

### Example 4: Offline Privacy-Focused Setup

**.env**:
```bash
# No API keys needed

# Router - Offline mode
ROUTER_MODE=offline
PRIMARY_PROVIDER=onnx
FALLBACK_CHAIN=onnx,xinference

# Management API
MANAGEMENT_API_KEY=offline-secret
MANAGEMENT_API_PORT=9090

# GPU enabled for local inference
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=all

# Optional services
ENABLE_DESKTOP=true
ENABLE_CODE_SERVER=true

LOG_LEVEL=info
NODE_ENV=production
```

**Use Case**: Air-gapped environments, privacy-sensitive tasks, no internet

---

### Example 5: Balanced General-Purpose

**.env**:
```bash
# API Keys
ANTHROPIC_API_KEY=your-claude-key
OPENAI_API_KEY=your-openai-key
GOOGLE_GEMINI_API_KEY=your-gemini-key

# Router - Balanced mode
ROUTER_MODE=balanced
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,openai,openrouter,claude

# Management API
MANAGEMENT_API_KEY=balanced-secret
MANAGEMENT_API_PORT=9090

# GPU enabled
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=all

# Optional services
ENABLE_DESKTOP=true
ENABLE_CODE_SERVER=true

LOG_LEVEL=info
NODE_ENV=production
```

**Use Case**: General-purpose development with flexibility

---

### Example 6: Desktop Automation with Browser

**.env**:
```bash
# API Keys
ANTHROPIC_API_KEY=your-claude-key
GOOGLE_GEMINI_API_KEY=your-gemini-key

# GitHub integration
GITHUB_TOKEN=your-github-token

# Web search
BRAVE_API_KEY=your-brave-key

# Router
ROUTER_MODE=performance
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,claude,openrouter

# Management API
MANAGEMENT_API_KEY=automation-secret
MANAGEMENT_API_PORT=9090

# GPU enabled
GPU_ACCELERATION=true
CUDA_VISIBLE_DEVICES=all

# Desktop REQUIRED for Playwright
ENABLE_DESKTOP=true
ENABLE_CODE_SERVER=true

LOG_LEVEL=info
NODE_ENV=production
```

**Use Case**: Browser automation, web scraping, GUI testing

---

## Quick Start Configuration

### Minimal Setup (Gemini Only)

1. Create `.env`:
```bash
GOOGLE_GEMINI_API_KEY=your-api-key
MANAGEMENT_API_KEY=change-this-secret
ROUTER_MODE=performance
PRIMARY_PROVIDER=gemini
GPU_ACCELERATION=true
```

2. Start:
```bash
docker-compose up -d
```

3. Access:
```bash
curl http://localhost:9090/health
```

### Recommended Setup (Multiple Providers)

1. Create `.env`:
```bash
ANTHROPIC_API_KEY=your-claude-key
OPENAI_API_KEY=your-openai-key
GOOGLE_GEMINI_API_KEY=your-gemini-key
MANAGEMENT_API_KEY=your-secure-secret
ROUTER_MODE=performance
PRIMARY_PROVIDER=gemini
FALLBACK_CHAIN=gemini,openai,claude
GPU_ACCELERATION=true
```

2. Verify configuration:
```bash
docker-compose config
```

3. Start:
```bash
docker-compose up -d
```

4. Check logs:
```bash
docker-compose logs -f
```

---

## Troubleshooting Configuration

### Configuration Not Loading

Check environment variable substitution:
```bash
docker-compose config
```

### GPU Not Working

Verify NVIDIA runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### API Keys Not Working

Check environment inside container:
```bash
docker exec agentic-flow-cachyos env | grep API_KEY
```

### Provider Connection Issues

Test provider connectivity:
```bash
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
  https://api.anthropic.com/v1/messages
```

### Resource Limits

Check resource usage:
```bash
docker stats agentic-flow-cachyos
```

---

## Security Best Practices

1. **API Keys**:
   - Never commit `.env` to version control
   - Use strong `MANAGEMENT_API_KEY`
   - Rotate keys regularly

2. **Network**:
   - Expose only required ports
   - Use firewall rules for production
   - Consider VPN for remote access

3. **Resources**:
   - Set appropriate memory limits
   - Monitor resource usage
   - Implement budget alerts

4. **Updates**:
   - Keep Docker images updated
   - Update provider configurations
   - Review security advisories

---

## Additional Resources

- [Architecture Documentation](./ARCHITECTURE-SIMPLIFIED.md)
- [Getting Started Guide](./GETTING_STARTED.md)
- [Deployment Guide](./DEPLOYMENT.md)
- [Router Documentation](./router/)
- [Provider Integrations](./integrations/)

---

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Verify configuration: `docker-compose config`
- Review health endpoints: `curl http://localhost:9090/health`
- Consult documentation in `/docs`
