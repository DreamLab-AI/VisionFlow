# Agentic Flow Quick Reference Guide

> One-page cheat sheet for daily operations, API calls, and troubleshooting

---

## Essential Commands

### Container Lifecycle

```bash
# Start services (first time)
./start-agentic-flow.sh --build

# Regular start
./start-agentic-flow.sh

# Stop all services
./start-agentic-flow.sh --stop

# Restart services
./start-agentic-flow.sh --restart

# Check status
./start-agentic-flow.sh --status

# View logs
./start-agentic-flow.sh --logs

# Open shell
./start-agentic-flow.sh --shell
docker exec -it agentic-flow-cachyos zsh

# Clean up everything
./start-agentic-flow.sh --clean
```

### Docker Compose Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f --tail=100

# Check status
docker-compose ps

# Stop services
docker-compose down

# Rebuild containers
docker-compose build --no-cache

# Remove volumes
docker-compose down -v
```

---

## Port Mappings

| Service | Port | Purpose |
|---------|------|---------|
| Management API | 9090 | Primary API interface |
| Claude-ZAI | 9600 | Z.AI semantic processing |
| VNC | 5901 | Remote desktop (if enabled) |
| noVNC | 6901 | Browser-based VNC (if enabled) |
| code-server | 8080 | VS Code Web (if enabled) |

---

## API Quick Reference

### Base Configuration

```bash
# Set API key header
export API_KEY="your-management-api-key"
export BASE_URL="http://localhost:9090"
```

### Health & Status

```bash
# Health check
curl http://localhost:9090/health

# Readiness check
curl http://localhost:9090/ready

# System status
curl -H "X-API-Key: $API_KEY" \
  http://localhost:9090/v1/status

# Metrics
curl http://localhost:9090/metrics
```

### Task Management

```bash
# Create task
curl -X POST http://localhost:9090/v1/tasks \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Create a REST API endpoint",
    "provider": "gemini"
  }'

# Get task status
curl -H "X-API-Key: $API_KEY" \
  http://localhost:9090/v1/tasks/TASK_ID

# List active tasks
curl -H "X-API-Key: $API_KEY" \
  http://localhost:9090/v1/status | jq '.tasks'
```

### Claude-ZAI API

```bash
# Health check
curl http://localhost:9600/health

# Get worker status
curl http://localhost:9600/status

# Send completion request (internal use)
curl -X POST http://localhost:9600/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Important File Locations

### Configuration Files

```
/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/
├── .env                          # Environment variables (create from .env.example)
├── docker-compose.yml            # Main compose file
├── docker-compose.dev.yml        # Development overrides
└── start-agentic-flow.sh         # Primary control script
```

### Inside Container

```
/home/devuser/
├── workspace/                    # Persistent development workspace
├── models/                       # Model cache directory
├── .agentic-flow/               # Agent session data
├── .config/                      # Application configs
└── logs/                         # Management API logs
```

### Scripts Directory

```
/mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker/scripts/
├── healthcheck.sh               # Container health checks
├── init-workstation.sh          # Workstation initialization
├── mcp-cli.sh                   # MCP command-line interface
├── test-all-providers.sh        # Test all AI providers
└── test-gemini-flow.sh          # Gemini integration test
```

---

## Environment Variables

### Required API Keys (At least one)

```bash
ANTHROPIC_API_KEY=sk-ant-xxxxx      # Claude API key
GOOGLE_GEMINI_API_KEY=AIzaxxxxx     # Gemini API key
OPENAI_API_KEY=sk-xxxxx             # OpenAI API key
OPENROUTER_API_KEY=sk-or-xxxxx      # OpenRouter key
```

### Management API

```bash
MANAGEMENT_API_KEY=change-this-secret-key    # API authentication
MANAGEMENT_API_PORT=9090                      # API port
MANAGEMENT_API_HOST=0.0.0.0                   # Bind address
```

### Router Configuration

```bash
ROUTER_MODE=performance                       # or "balanced", "quality"
PRIMARY_PROVIDER=gemini                       # Primary AI provider
FALLBACK_CHAIN=gemini,openai,claude,openrouter
```

### GPU Configuration

```bash
GPU_ACCELERATION=true                         # Enable GPU support
CUDA_VISIBLE_DEVICES=all                      # GPU selection
```

### Optional Services

```bash
ENABLE_DESKTOP=false                          # VNC desktop environment
ENABLE_CODE_SERVER=false                      # VS Code web server
```

### System Settings

```bash
LOG_LEVEL=info                                # debug, info, warn, error
NODE_ENV=production                           # or "development"
DISPLAY=:0                                    # X11 display for GUI apps
```

### Claude-ZAI Configuration

```bash
ANTHROPIC_BASE_URL=https://api.z.ai/api/anthropic
CLAUDE_WORKER_POOL_SIZE=4                     # Worker threads
CLAUDE_MAX_QUEUE_SIZE=50                      # Request queue size
```

---

## Common curl Examples

### Create a Coding Task

```bash
curl -X POST http://localhost:9090/v1/tasks \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Implement user authentication with JWT",
    "provider": "gemini"
  }' | jq
```

### Create a Testing Task

```bash
curl -X POST http://localhost:9090/v1/tasks \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "tester",
    "task": "Write comprehensive tests for the user service",
    "provider": "claude"
  }' | jq
```

### Check Task Progress

```bash
# Get task ID from creation response
TASK_ID="task_20250312_123456_coder"

# Poll for updates
curl -H "X-API-Key: $API_KEY" \
  http://localhost:9090/v1/tasks/$TASK_ID | jq
```

### Monitor System Resources

```bash
# CPU, memory, GPU status
curl -H "X-API-Key: $API_KEY" \
  http://localhost:9090/v1/status | jq '.system, .gpu'

# Active tasks count
curl -H "X-API-Key: $API_KEY" \
  http://localhost:9090/v1/status | jq '.tasks'
```

### Get Prometheus Metrics

```bash
# All metrics
curl http://localhost:9090/metrics

# Filter specific metrics
curl http://localhost:9090/metrics | grep http_request
```

---

## Troubleshooting Quick Checks

### Container Issues

```bash
# Check if containers are running
docker ps | grep agentic

# View recent logs
docker logs agentic-flow-cachyos --tail=50
docker logs claude-zai-service --tail=50

# Check container resource usage
docker stats agentic-flow-cachyos claude-zai-service

# Restart specific container
docker restart agentic-flow-cachyos

# Check container health
docker inspect agentic-flow-cachyos | jq '.[0].State.Health'
```

### API Connection Problems

```bash
# Test Management API
curl -v http://localhost:9090/health

# Test Claude-ZAI
curl -v http://localhost:9600/health

# Check if ports are listening
netstat -tlnp | grep -E '9090|9600'
lsof -i :9090
lsof -i :9600

# Test API authentication
curl -v -H "X-API-Key: wrong-key" http://localhost:9090/v1/status
```

### Environment Configuration

```bash
# Verify .env file exists
ls -la .env

# Check key variables are set
grep -E "API_KEY|MANAGEMENT" .env

# Validate docker-compose config
docker-compose config

# Check environment inside container
docker exec agentic-flow-cachyos env | grep -E "API_KEY|MANAGEMENT"
```

### GPU Problems

```bash
# Check GPU visibility on host
nvidia-smi

# Check GPU inside container
docker exec agentic-flow-cachyos nvidia-smi

# Verify CUDA environment
docker exec agentic-flow-cachyos bash -c 'echo $CUDA_VISIBLE_DEVICES'

# Test GPU access
docker exec agentic-flow-cachyos python3 -c "import torch; print(torch.cuda.is_available())"
```

### Network Connectivity

```bash
# Check network exists
docker network ls | grep agentic

# Inspect network
docker network inspect multi-agent-docker_agentic-network

# Test inter-container communication
docker exec agentic-flow-cachyos ping -c 3 claude-zai-service

# Check RAGFlow network connection (if using)
docker network inspect docker_ragflow | grep agentic
```

### Volume and Storage

```bash
# List volumes
docker volume ls | grep multi-agent

# Inspect volume
docker volume inspect multi-agent-docker_workspace

# Check volume usage
docker system df -v

# Clean unused volumes (careful!)
docker volume prune
```

### Performance Issues

```bash
# Check system load
docker exec agentic-flow-cachyos top -bn1 | head -20

# Memory usage
docker exec agentic-flow-cachyos free -h

# Disk space
docker exec agentic-flow-cachyos df -h

# View API metrics
curl http://localhost:9090/metrics | grep -E 'cpu|memory|response_time'
```

---

## Useful Shell Aliases

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# Agentic Flow aliases
alias af-start='cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker && ./start-agentic-flow.sh'
alias af-stop='cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker && ./start-agentic-flow.sh --stop'
alias af-status='cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker && ./start-agentic-flow.sh --status'
alias af-logs='cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker && ./start-agentic-flow.sh --logs'
alias af-shell='docker exec -it agentic-flow-cachyos zsh'
alias af-restart='cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker && ./start-agentic-flow.sh --restart'

# API shortcuts
alias af-health='curl http://localhost:9090/health | jq'
alias af-api-status='curl -H "X-API-Key: $MANAGEMENT_API_KEY" http://localhost:9090/v1/status | jq'
alias af-metrics='curl http://localhost:9090/metrics'

# Task management
af-task() {
  curl -X POST http://localhost:9090/v1/tasks \
    -H "X-API-Key: $MANAGEMENT_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"agent\":\"$1\",\"task\":\"$2\",\"provider\":\"${3:-gemini}\"}" | jq
}

af-task-status() {
  curl -H "X-API-Key: $MANAGEMENT_API_KEY" \
    http://localhost:9090/v1/tasks/$1 | jq
}

# Logs shortcuts
alias af-logs-api='docker logs agentic-flow-cachyos -f --tail=100'
alias af-logs-zai='docker logs claude-zai-service -f --tail=100'

# Quick access
alias af-cd='cd /mnt/mldata/githubs/AR-AI-Knowledge-Graph/multi-agent-docker'
```

### Usage Examples

```bash
# After adding aliases, reload shell
source ~/.bashrc  # or source ~/.zshrc

# Start system
af-start

# Check health
af-health

# Create task
af-task coder "Build a REST API"

# Check task status
af-task-status task_20250312_123456_coder

# View logs
af-logs-api

# Open shell
af-shell

# Get system status
af-api-status
```

---

## Quick Reference Table

| Need to... | Command |
|------------|---------|
| Start everything | `./start-agentic-flow.sh` |
| Stop everything | `./start-agentic-flow.sh --stop` |
| Check health | `curl http://localhost:9090/health` |
| Create task | `curl -X POST http://localhost:9090/v1/tasks -H "X-API-Key: $API_KEY" -d '...'` |
| View logs | `docker-compose logs -f` |
| Open shell | `docker exec -it agentic-flow-cachyos zsh` |
| Restart service | `docker-compose restart` |
| Check GPU | `docker exec agentic-flow-cachyos nvidia-smi` |
| View metrics | `curl http://localhost:9090/metrics` |
| Clean up | `./start-agentic-flow.sh --clean` |

---

## Emergency Commands

```bash
# Kill all containers immediately
docker kill agentic-flow-cachyos claude-zai-service

# Force remove containers
docker rm -f agentic-flow-cachyos claude-zai-service

# Emergency cleanup
docker system prune -af --volumes

# Reset to clean state
./start-agentic-flow.sh --clean
./start-agentic-flow.sh --build
```

---

## Support Resources

- **Documentation**: `/docs/` directory
- **Getting Started**: `/docs/GETTING_STARTED.md`
- **Architecture**: `/docs/FINAL-ARCHITECTURE.md`
- **API Reference**: Management API Swagger at `http://localhost:9090/documentation`
- **Logs**: `/home/devuser/logs/` (inside container)
- **Issues**: Check GitHub repository

---

**Last Updated**: 2025-10-12
**Version**: 2.1.0
**Quick Print Tip**: Print this page at 70% scale for best results
