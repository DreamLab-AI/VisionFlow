# CachyOS Docker Quick Start

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum (16GB recommended)
- 20GB free disk space

## Basic Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/agentic-flow.git
cd agentic-flow/docker/cachyos
```

### 2. Configure Environment

Create `.env` file with API keys:

```bash
cat > .env << EOF
# Required for Claude
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Optional providers
OPENAI_API_KEY=sk-your-key-here
GEMINI_API_KEY=your-key-here

# Optional MCP tools
GITHUB_TOKEN=ghp_your-token-here
BRAVE_API_KEY=your-key-here
CONTEXT7_API_KEY=your-key-here

# Desktop and development
ENABLE_DESKTOP=true
ENABLE_CODE_SERVER=true
EOF
```

### 3. Build Image

```bash
docker-compose build
```

### 4. Start Container

```bash
docker-compose up -d
```

### 5. Verify Running

```bash
docker-compose ps
```

Expected output:
```
NAME                    STATUS              PORTS
agentic-flow-cachyos    Up 30 seconds       0.0.0.0:8080->8080/tcp
                                            0.0.0.0:6901->6901/tcp
                                            0.0.0.0:9090->9090/tcp
```

## Access Points

### Management API
```bash
curl http://localhost:9090/health
```

### Code Server (Web IDE)
Open browser: http://localhost:8080

### noVNC (Desktop)
Open browser: http://localhost:6901

### Interactive Shell
```bash
docker exec -it agentic-flow-cachyos zsh
```

## Quick Commands

### View Logs
```bash
docker-compose logs -f
```

### Restart Services
```bash
docker exec agentic-flow-cachyos supervisorctl restart all
```

### Check Service Status
```bash
docker exec agentic-flow-cachyos supervisorctl status
```

### Stop Container
```bash
docker-compose down
```

### Remove Everything
```bash
docker-compose down -v
docker rmi agentic-flow-cachyos:latest
```

## Common Tasks

### Create Worker Session

```bash
curl -X POST http://localhost:9090/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "tools": ["playwright", "filesystem", "git"],
    "timeout": 3600
  }'
```

### List Active Sessions

```bash
curl http://localhost:9090/api/v1/sessions
```

### Execute Task

```bash
curl -X POST http://localhost:9090/api/v1/sessions/{session_id}/execute \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "playwright",
    "action": "navigate",
    "params": {"url": "https://example.com"}
  }'
```

### Destroy Session

```bash
curl -X DELETE http://localhost:9090/api/v1/sessions/{session_id}
```

## Development Workflow

### 1. Access Code Server

Open http://localhost:8080 in browser.

### 2. Open Workspace

Navigate to `/home/devuser/workspace` in code-server.

### 3. Create Project

```bash
docker exec -it agentic-flow-cachyos bash -c "
  cd ~/workspace
  mkdir my-project
  cd my-project
  npm init -y
"
```

### 4. Edit Code

Use code-server web IDE or mount workspace locally:

```yaml
# In docker-compose.yml
volumes:
  - ./my-workspace:/home/devuser/workspace
```

### 5. Test Changes

```bash
docker exec agentic-flow-cachyos bash -c "
  cd ~/workspace/my-project
  npm test
"
```

## Troubleshooting

### Container Won't Start

Check logs:
```bash
docker-compose logs
```

Verify ports available:
```bash
netstat -tulpn | grep -E '8080|9090|6901'
```

### API Not Responding

Check Management API status:
```bash
docker exec agentic-flow-cachyos supervisorctl status management-api
```

View API logs:
```bash
docker exec agentic-flow-cachyos cat /home/devuser/logs/management-api.log
```

### Desktop Not Loading

Verify desktop services:
```bash
docker exec agentic-flow-cachyos supervisorctl status xvnc dbus xfce4 novnc
```

Check VNC logs:
```bash
docker exec agentic-flow-cachyos cat /home/devuser/logs/xvnc.log
```

### MCP Tools Not Working

Verify tool configuration:
```bash
docker exec agentic-flow-cachyos cat /home/devuser/.config/claude/mcp.json
```

Test tool manually:
```bash
docker exec agentic-flow-cachyos npx -y @modelcontextprotocol/server-playwright --help
```

### Out of Memory

Check container memory:
```bash
docker stats agentic-flow-cachyos
```

Add memory limit to docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

## Next Steps

- Read [ARCHITECTURE.md](./ARCHITECTURE.md) for system design details
- Review [CONFIGURATION.md](./CONFIGURATION.md) for advanced configuration
- Check [MCP_TOOLS.md](./MCP_TOOLS.md) for available tools and usage
- See [DEPLOYMENT.md](./DEPLOYMENT.md) for production deployment guide

## Getting Help

- GitHub Issues: https://github.com/yourusername/agentic-flow/issues
- Documentation: https://docs.agentic-flow.com
- Discord: https://discord.gg/agentic-flow
