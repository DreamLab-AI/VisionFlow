# Agentic Flow Docker Deployment

**Standalone, production-ready Docker deployment for Agentic Flow AI orchestration platform.**

This directory contains everything needed to deploy Agentic Flow using Docker. No other files from the repository are required - the `agentic-flow` package is installed directly from npm.

## Quick Start

```bash
# 1. Configure API keys
cp .env.example .env
nano .env  # Add your API keys

# 2. Start services
./start-agentic-flow.sh --build

# 3. Verify
curl http://localhost:9090/health
curl http://localhost:9600/health
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| **Management API** | 9090 | Primary interface for task management |
| **Claude-ZAI** | 9600 | Claude AI wrapper with Z.AI integration |
| **VNC** | 5901 | Remote desktop (if enabled) |
| **noVNC** | 6901 | Browser-based VNC (if enabled) |
| **Code Server** | 8080 | VS Code in browser (if enabled) |

## Directory Structure

```
docker/cachyos/                      ← ONLY THIS DIRECTORY NEEDED
├── start-agentic-flow.sh           # Launch script
├── docker-compose.yml              # Standalone compose (installs from npm)
├── docker-compose.dev.yml          # Development compose (local source)
├── Dockerfile.workstation          # Standalone Dockerfile
├── Dockerfile.workstation.dev      # Development Dockerfile
├── .env.example                    # Environment template
├── management-api/                 # Management API service
│   ├── server.js
│   ├── routes/
│   ├── middleware/
│   └── utils/
├── claude-zai/                     # Claude Z.AI wrapper
│   ├── Dockerfile
│   └── wrapper/
├── config/                         # Configuration files
│   ├── .zshrc
│   ├── router.config.json
│   ├── mcp.json
│   ├── gemini-flow.config.ts
│   └── supervisord.conf
├── scripts/                        # Helper scripts
│   ├── init-workstation.sh
│   └── mcp-cli.sh
└── core-assets/                    # Core assets
    └── scripts/
```

## Usage

### Starting Services

```bash
# First time (builds images)
./start-agentic-flow.sh --build

# Regular start
./start-agentic-flow.sh

# With RAGFlow network
./start-agentic-flow.sh

# Without RAGFlow
./start-agentic-flow.sh --no-ragflow
```

### Managing Services

```bash
# Check status
./start-agentic-flow.sh --status

# View logs
./start-agentic-flow.sh --logs

# Restart
./start-agentic-flow.sh --restart

# Stop
./start-agentic-flow.sh --stop

# Shell access
./start-agentic-flow.sh --shell
# or
docker exec -it agentic-flow-cachyos zsh
```

### Cleanup

```bash
# Stop and remove containers
./start-agentic-flow.sh --stop

# Complete cleanup (removes volumes)
./start-agentic-flow.sh --clean
```

## API Usage

### Management API

```bash
# Authentication required (except /health, /ready, /metrics)
export API_KEY="change-this-secret-key"

# Get API info
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/

# Create task
curl -X POST http://localhost:9090/v1/tasks \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Write a Python function to reverse a string",
    "provider": "gemini"
  }'

# Check task status
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks/{taskId}

# List active tasks
curl -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks

# Stop task
curl -X DELETE \
  -H "Authorization: Bearer $API_KEY" \
  http://localhost:9090/v1/tasks/{taskId}
```

### Claude-ZAI Wrapper

```bash
# Simple prompt
curl -X POST http://localhost:9600/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain Docker in 3 sentences",
    "timeout": 15000
  }'

# Complex prompt
curl -X POST http://localhost:9600/prompt \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a Python function that calculates Fibonacci numbers",
    "timeout": 30000
  }'
```

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...        # For Claude
GOOGLE_GEMINI_API_KEY=...           # For Gemini
OPENAI_API_KEY=sk-...               # For GPT-4

# Management API
MANAGEMENT_API_KEY=your-secret-key

# Optional
ENABLE_DESKTOP=false                # Enable VNC desktop
ENABLE_CODE_SERVER=false            # Enable VS Code
GPU_ACCELERATION=true               # Use GPU if available
```

### API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:9090/docs
- **OpenAPI spec**: http://localhost:9090/docs/json

## Deployment Options

### 1. Standalone (Default)
- Installs `agentic-flow` from npm
- Only requires `docker/cachyos/` directory
- Production-ready
- Always uses latest published version

```bash
docker compose up -d --build
```

### 2. Development
- Uses local `agentic-flow` source
- Requires full repository
- For development/testing

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

## Persistent Storage

Data persisted in Docker volumes:

- `workspace` - User workspace files
- `model-cache` - Downloaded AI models
- `agent-memory` - Agent session data
- `config-persist` - Configuration files
- `management-logs` - API logs

View volumes:
```bash
docker volume ls | grep cachyos
```

Backup volumes:
```bash
docker run --rm -v cachyos_workspace:/data -v $(pwd):/backup \
  alpine tar czf /backup/workspace-backup.tar.gz /data
```

## GPU Support

Requires:
- NVIDIA GPU
- NVIDIA Docker runtime
- `nvidia-container-toolkit`

Test GPU access:
```bash
docker exec agentic-flow-cachyos nvidia-smi
```

## Troubleshooting

### Services not starting

```bash
# Check logs
docker compose logs -f

# Check specific service
docker logs agentic-flow-cachyos
docker logs claude-zai-service
```

### Management API not responding

```bash
# Check if running
curl http://localhost:9090/health

# Check inside container
docker exec agentic-flow-cachyos ps aux | grep node

# View logs
docker exec agentic-flow-cachyos cat ~/logs/management-api.log
```

### Permission errors

```bash
# Fix volume permissions
docker compose down
docker volume rm cachyos_workspace cachyos_config-persist
docker compose up -d --build
```

## Architecture

### Standalone vs Development

| Feature | Standalone | Development |
|---------|------------|-------------|
| **Build context** | `.` (cachyos dir) | `../..` (repo root) |
| **agentic-flow** | `npm install -g` | `COPY agentic-flow/` |
| **Version** | npm latest (1.5.10) | Local source |
| **Deployment** | Just cachyos dir | Full repo |
| **Use case** | Production | Development |

### Service Communication

```
External Request
    ↓
Management API (9090) ← Authentication
    ↓
Process Manager
    ↓
Agent Workers (isolated)
    ↓
Claude-ZAI (9600) ← AI Processing
    ↓
Z.AI API / Anthropic API
```

## Advanced Usage

### Custom Network Integration

Connect to existing Docker networks:

```yaml
# docker-compose.yml
networks:
  agentic-network:
    external: true
    name: your-network-name
```

### Resource Limits

Adjust in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 32G
      cpus: '16'
```

### Desktop Environment

Enable GUI access:

```bash
# .env
ENABLE_DESKTOP=true

# Access via browser
http://localhost:6901

# Or VNC client
vnc://localhost:5901
```

## Security

### Best Practices

1. **Change default API key**: Update `MANAGEMENT_API_KEY` in `.env`
2. **Restrict ports**: Use firewall to limit access to 9090, 9600
3. **Use secrets**: Mount secrets instead of environment variables
4. **Network isolation**: Use Docker networks
5. **Regular updates**: Rebuild with `--build` to get latest npm packages

### Production Deployment

```bash
# Use strong API key
openssl rand -hex 32 > .api_key
export MANAGEMENT_API_KEY=$(cat .api_key)

# Run behind reverse proxy
# nginx/traefik/caddy in front of port 9090

# Enable TLS
# Add SSL certificates to reverse proxy

# Restrict network access
# iptables or cloud security groups
```

## Support

- **Documentation**: See `docs/` directory in main repo
- **Issues**: https://github.com/ruvnet/agentic-flow/issues
- **Package**: https://www.npmjs.com/package/agentic-flow

## License

See main repository for license information.
