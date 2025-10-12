# CachyOS Docker Configuration Guide

## Overview

This guide explains how to configure the CachyOS Docker environment for different use cases, from development workstations to production deployments.

## Configuration Files

### Primary Configuration Files

#### `docker-compose.yml`
Main orchestration file defining services, volumes, ports, and environment variables.

```yaml
# Minimal production configuration
services:
  cachyos:
    build:
      context: ../..
      dockerfile: docker/cachyos/Dockerfile.workstation
    environment:
      - ENABLE_DESKTOP=false
      - ENABLE_CODE_SERVER=false
    ports:
      - "9090:9090"
```

#### `config/mcp.json`
Defines available MCP tools using stdio communication.

```json
{
  "mcpServers": {
    "tool-name": {
      "command": "npx",
      "args": ["-y", "package-name"],
      "type": "stdio",
      "description": "Tool description",
      "env": {
        "ENV_VAR": "${VALUE}"
      }
    }
  }
}
```

#### `config/supervisord.conf`
Process supervisor configuration managing core services.

#### `config/router.config.json`
Agentic flow router configuration for AI provider selection.

```json
{
  "mode": "performance",
  "providers": {
    "claude": { "enabled": true, "priority": 1 },
    "gemini": { "enabled": true, "priority": 2 }
  }
}
```

## Environment Variables

### Core Settings

#### `ENABLE_DESKTOP` (default: true)
Controls desktop environment (VNC/noVNC/XFCE4).
- `true`: Full desktop with VNC access
- `false`: Headless mode (saves ~500MB RAM)

```yaml
environment:
  - ENABLE_DESKTOP=false  # Headless for production
```

#### `ENABLE_CODE_SERVER` (default: true)
Controls web-based VS Code IDE.
- `true`: code-server on port 8080
- `false`: No web IDE

```yaml
environment:
  - ENABLE_CODE_SERVER=true  # Enable for remote development
```

#### `WORKSPACE` (default: /home/devuser/workspace)
Primary working directory for projects.

```yaml
environment:
  - WORKSPACE=/home/devuser/workspace
volumes:
  - ./projects:/home/devuser/workspace
```

#### `DISPLAY` (default: :1)
X11 display identifier for GUI applications.

```yaml
environment:
  - DISPLAY=:1
```

### AI Provider Configuration

#### `ENABLE_GEMINI` (default: true)
Enable Google Gemini AI provider.

```yaml
environment:
  - ENABLE_GEMINI=true
  - GEMINI_API_KEY=${GEMINI_API_KEY}
```

#### `ENABLE_OPENAI` (default: true)
Enable OpenAI GPT provider.

```yaml
environment:
  - ENABLE_OPENAI=true
  - OPENAI_API_KEY=${OPENAI_API_KEY}
```

#### `ENABLE_CLAUDE` (default: true)
Enable Anthropic Claude provider.

```yaml
environment:
  - ENABLE_CLAUDE=true
  - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

#### `ENABLE_OPENROUTER` (default: true)
Enable OpenRouter proxy provider.

```yaml
environment:
  - ENABLE_OPENROUTER=true
  - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
```

#### `ROUTER_MODE` (default: performance)
AI provider routing strategy.
- `performance`: Prefer fastest models
- `cost`: Prefer cheapest models
- `quality`: Prefer highest quality models
- `balanced`: Balance all factors

```yaml
environment:
  - ROUTER_MODE=cost
```

### Gemini Flow Settings

#### `GEMINI_FLOW_ENABLED` (default: true)
Enable Gemini Flow orchestration framework.

```yaml
environment:
  - GEMINI_FLOW_ENABLED=true
```

#### `GEMINI_FLOW_PROTOCOLS` (default: a2a,mcp)
Comma-separated list of enabled protocols.
- `a2a`: Agent-to-Agent communication
- `mcp`: Model Context Protocol
- `http`: HTTP API endpoints

```yaml
environment:
  - GEMINI_FLOW_PROTOCOLS=a2a,mcp
```

#### `GEMINI_FLOW_TOPOLOGY` (default: hierarchical)
Swarm coordination pattern.
- `hierarchical`: Queen-led coordination
- `mesh`: Peer-to-peer network
- `adaptive`: Dynamic topology switching

```yaml
environment:
  - GEMINI_FLOW_TOPOLOGY=mesh
```

#### `GEMINI_FLOW_MAX_AGENTS` (default: 66)
Maximum concurrent agents in swarm.

```yaml
environment:
  - GEMINI_FLOW_MAX_AGENTS=100
```

### MCP Configuration

#### `MCP_AUTO_START` (default: true)
Automatically configure MCP tools on startup.

```yaml
environment:
  - MCP_AUTO_START=true
```

#### MCP Tool Environment Variables
Individual MCP tools may require API keys or configuration.

```yaml
environment:
  - GITHUB_TOKEN=${GITHUB_TOKEN}
  - BRAVE_API_KEY=${BRAVE_API_KEY}
  - CONTEXT7_API_KEY=${CONTEXT7_API_KEY}
```

### System Settings

#### `GPU_ACCELERATION` (default: true)
Enable GPU acceleration for compatible workloads.

```yaml
environment:
  - GPU_ACCELERATION=true
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Use Case Configurations

### Development Workstation

Full-featured setup with desktop, IDE, and all tools.

```yaml
services:
  cachyos:
    image: agentic-flow-cachyos:latest
    environment:
      - ENABLE_DESKTOP=true
      - ENABLE_CODE_SERVER=true
      - ROUTER_MODE=performance
      - GEMINI_FLOW_ENABLED=true
    ports:
      - "8080:8080"   # code-server
      - "6901:6901"   # noVNC
      - "9090:9090"   # Management API
      - "3000:3000"   # Dev server
    volumes:
      - ./workspace:/home/devuser/workspace
      - ./models:/home/devuser/models
      - ./config:/home/devuser/.config
```

### Headless Production

Minimal resource footprint for API-only deployments.

```yaml
services:
  cachyos:
    image: agentic-flow-cachyos:latest
    environment:
      - ENABLE_DESKTOP=false
      - ENABLE_CODE_SERVER=false
      - ROUTER_MODE=cost
      - GEMINI_FLOW_MAX_AGENTS=50
    ports:
      - "9090:9090"
    volumes:
      - ./workspace:/home/devuser/workspace
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
```

### CI/CD Pipeline

Optimized for automated testing and builds.

```yaml
services:
  cachyos:
    image: agentic-flow-cachyos:latest
    environment:
      - ENABLE_DESKTOP=false
      - ENABLE_CODE_SERVER=false
      - MCP_AUTO_START=false
      - GEMINI_FLOW_ENABLED=false
    volumes:
      - ./code:/home/devuser/workspace
    command: |
      bash -c "cd /home/devuser/workspace && npm test"
```

### GPU-Accelerated Workstation

For ML/AI workloads requiring GPU access.

```yaml
services:
  cachyos:
    image: agentic-flow-cachyos:latest
    environment:
      - GPU_ACCELERATION=true
      - ENABLE_DESKTOP=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu, compute, utility]
    volumes:
      - ./models:/home/devuser/models
```

## MCP Tool Configuration

### Adding New Tools

Edit `config/mcp.json` to add stdio-based tools:

```json
{
  "mcpServers": {
    "my-tool": {
      "command": "npx",
      "args": ["-y", "@namespace/my-tool-mcp"],
      "type": "stdio",
      "description": "My custom MCP tool",
      "env": {
        "MY_TOOL_API_KEY": "${MY_TOOL_API_KEY}"
      }
    }
  },
  "toolCategories": {
    "custom": ["my-tool"]
  }
}
```

### Tool Categories

Organize tools into logical categories:

```json
{
  "toolCategories": {
    "automation": ["playwright"],
    "documentation": ["context7"],
    "filesystem": ["filesystem", "git"],
    "web": ["fetch", "brave-search"],
    "github": ["github"]
  }
}
```

### Tool Environment Variables

Pass environment variables to specific tools:

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "type": "stdio",
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}",
        "GITHUB_ORG": "my-org"
      }
    }
  }
}
```

## Router Configuration

### Provider Priority

Configure provider selection order in `config/router.config.json`:

```json
{
  "mode": "performance",
  "providers": {
    "claude": {
      "enabled": true,
      "priority": 1,
      "models": ["claude-3-opus", "claude-3-sonnet"]
    },
    "gemini": {
      "enabled": true,
      "priority": 2,
      "models": ["gemini-pro", "gemini-ultra"]
    },
    "openai": {
      "enabled": true,
      "priority": 3,
      "models": ["gpt-4", "gpt-3.5-turbo"]
    }
  }
}
```

### Routing Strategies

#### Performance Mode
```json
{
  "mode": "performance",
  "criteria": {
    "latency_weight": 0.6,
    "throughput_weight": 0.3,
    "cost_weight": 0.1
  }
}
```

#### Cost Mode
```json
{
  "mode": "cost",
  "criteria": {
    "cost_weight": 0.7,
    "quality_weight": 0.2,
    "latency_weight": 0.1
  }
}
```

#### Quality Mode
```json
{
  "mode": "quality",
  "criteria": {
    "quality_weight": 0.7,
    "capabilities_weight": 0.2,
    "reliability_weight": 0.1
  }
}
```

## Supervisord Configuration

### Custom Service Management

Add custom services to `config/supervisord.conf`:

```ini
[program:my-service]
command=/usr/bin/node /home/devuser/my-service/server.js
user=devuser
autorestart=true
stdout_logfile=/home/devuser/logs/my-service.log
stderr_logfile=/home/devuser/logs/my-service.err.log
environment=HOME="/home/devuser",NODE_ENV="production"
priority=60
```

### Conditional Services

Use environment variables to control service startup:

```ini
[program:optional-service]
command=/usr/bin/my-app
autostart=%(ENV_ENABLE_MY_APP)s
```

## Volume Configuration

### Persistent Volumes

Recommended volume mounts:

```yaml
volumes:
  # Project files - READ/WRITE
  - ./workspace:/home/devuser/workspace

  # Model cache - READ/WRITE
  - ./models:/home/devuser/models

  # Application logs - READ/WRITE
  - ./logs:/home/devuser/logs

  # Configuration - READ
  - ./config:/home/devuser/.config:ro

  # Claude Flow state - READ/WRITE
  - claude-flow-data:/home/devuser/.claude-flow

volumes:
  claude-flow-data:
```

### Read-Only Mounts

Protect configuration from modification:

```yaml
volumes:
  - ./config:/home/devuser/.config:ro
```

## Network Configuration

### Bridge Network (Default)

Suitable for standalone deployments:

```yaml
services:
  cachyos:
    networks:
      - default
```

### Custom Network

For multi-container deployments:

```yaml
networks:
  agentic-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  cachyos:
    networks:
      agentic-net:
        ipv4_address: 172.20.0.10
```

## Security Configuration

### User Permissions

Container runs as `devuser` (non-root):

```dockerfile
USER devuser
WORKDIR /home/devuser
```

### Sudo Access

`devuser` has passwordless sudo for package installation:

```dockerfile
RUN echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
```

### API Key Management

Never commit API keys. Use environment variables:

```yaml
environment:
  - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
  - OPENAI_API_KEY=${OPENAI_API_KEY}
```

Create `.env` file (add to `.gitignore`):

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
```

## Resource Limits

### Memory Limits

```yaml
deploy:
  resources:
    limits:
      memory: 8G
    reservations:
      memory: 4G
```

### CPU Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
    reservations:
      cpus: '2'
```

### Combined Example

```yaml
services:
  cachyos:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
```

## Logging Configuration

### Log Rotation

Configure in `config/supervisord.conf`:

```ini
[supervisord]
logfile=/home/devuser/logs/supervisord.log
logfile_maxbytes=50MB
logfile_backups=10
loglevel=info
```

### JSON Logging

Configure structured logging in application config:

```json
{
  "logging": {
    "format": "json",
    "level": "info",
    "output": "/home/devuser/logs/app.log"
  }
}
```

## Health Checks

### Docker Health Check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:9090/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Custom Health Script

```yaml
healthcheck:
  test: ["/home/devuser/scripts/healthcheck.sh"]
```

## Environment File Structure

### Development `.env`

```bash
# Desktop and Development
ENABLE_DESKTOP=true
ENABLE_CODE_SERVER=true

# AI Providers
ENABLE_GEMINI=true
ENABLE_OPENAI=true
ENABLE_CLAUDE=true
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
GEMINI_API_KEY=xxx

# Router
ROUTER_MODE=performance

# Gemini Flow
GEMINI_FLOW_ENABLED=true
GEMINI_FLOW_TOPOLOGY=hierarchical
GEMINI_FLOW_MAX_AGENTS=66

# MCP Tools
GITHUB_TOKEN=ghp_xxx
BRAVE_API_KEY=xxx
CONTEXT7_API_KEY=xxx
```

### Production `.env`

```bash
# Minimal services
ENABLE_DESKTOP=false
ENABLE_CODE_SERVER=false

# Essential providers only
ENABLE_CLAUDE=true
ANTHROPIC_API_KEY=sk-ant-xxx

# Cost-optimized routing
ROUTER_MODE=cost

# Limited resources
GEMINI_FLOW_MAX_AGENTS=25
```

## Configuration Validation

### Verify MCP Configuration

```bash
docker exec agentic-flow-cachyos jq '.' /home/devuser/.config/claude/mcp.json
```

### Test Tool Availability

```bash
docker exec agentic-flow-cachyos npx -y @modelcontextprotocol/server-playwright --help
```

### Check Service Status

```bash
docker exec agentic-flow-cachyos supervisorctl status
```

## Dependency Management

### Local vs Global Dependencies

The container uses a **local dependencies pattern** for better reproducibility:

#### devDependencies (Local)
Installed in project's `node_modules`:
- `pm2` - Process manager
- `@clduab11/gemini-flow` - AI swarm framework
- `@anthropic-ai/claude-code` - Claude CLI

Access via:
```bash
# Direct from node_modules/.bin (automatically in PATH)
pm2 --version
gemini-flow --help
claude-code --version

# Or via npm scripts
npm run pm2 -- --version
npm run gemini-flow -- --help
npm run claude-code -- --version
```

#### Global Installs
Only `agentic-flow` itself is installed globally:
```bash
npm install -g agentic-flow
```

This provides the `agentic-flow` command while keeping auxiliary tools version-locked.

### PATH Configuration

The `.zshrc` sets PATH priority:
```bash
# 1. Project local binaries (highest priority)
export PATH="/tmp/agentic-flow/node_modules/.bin:$PATH"
# 2. Global npm binaries
export PATH="$HOME/.npm-global/bin:$PATH"
```

This ensures local versions take precedence over any global installs.

### Version Pinning

All tool versions are locked in `package-lock.json`:
```json
{
  "devDependencies": {
    "pm2": "^5.4.3",
    "@clduab11/gemini-flow": "^1.0.0",
    "@anthropic-ai/claude-code": "^0.1.0"
  }
}
```

To update:
```bash
docker exec agentic-flow-cachyos bash -c "cd /tmp/agentic-flow && npm update"
```

## Troubleshooting Configuration

### Tool Not Found

If `pm2`, `gemini-flow`, or `claude-code` not found:

```bash
# Check PATH
docker exec agentic-flow-cachyos echo $PATH

# Check binaries exist
docker exec agentic-flow-cachyos ls -la /tmp/agentic-flow/node_modules/.bin/

# Check symlinks
docker exec agentic-flow-cachyos ls -la /usr/local/bin/ | grep -E 'pm2|gemini-flow|claude-code'

# Reinstall dependencies
docker exec agentic-flow-cachyos bash -c "cd /tmp/agentic-flow && npm install"
```

### Environment Variables Not Set

Check with:
```bash
docker exec agentic-flow-cachyos env | grep ENABLE
```

### Volume Mount Issues

Verify mounts:
```bash
docker inspect agentic-flow-cachyos | jq '.[0].Mounts'
```

### Port Conflicts

Check port availability before starting:
```bash
netstat -tulpn | grep -E '8080|9090|6901'
```

### Configuration Syntax Errors

Validate JSON configs:
```bash
jq '.' config/mcp.json
jq '.' config/router.config.json
```

Validate YAML:
```bash
docker-compose config
```
