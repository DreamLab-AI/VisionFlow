# Standalone Docker Deployment

This directory contains everything needed for standalone deployment using the npm-published `agentic-flow` package.

## What's Included

```
docker/cachyos/
├── docker-compose.standalone.yml    # Standalone compose file
├── Dockerfile.workstation.standalone # Installs from npm
├── management-api/                   # Management API service
├── claude-zai/                       # Claude Z.AI wrapper
├── config/                           # Configuration files
├── scripts/                          # Helper scripts
└── core-assets/                      # Core assets
```

## Quick Start

1. **Configure environment:**
```bash
cd docker/cachyos
cp .env.example .env
# Edit .env with your API keys
```

2. **Build and start:**
```bash
docker compose -f docker-compose.standalone.yml up -d --build
```

3. **Access services:**
- Management API: http://localhost:9090
- Claude-ZAI: http://localhost:9600
- Documentation: http://localhost:9090/docs

## Key Differences from Development Version

| Feature | Development | Standalone |
|---------|-------------|------------|
| Build context | `../..` (repo root) | `.` (cachyos dir only) |
| agentic-flow source | Local copy | npm install -g |
| Version | Local dev version | npm latest (1.5.10) |
| Deployment | Requires full repo | Only cachyos dir |

## Testing the Standalone Build

```bash
# Test Management API
curl -H "Authorization: Bearer change-this-secret-key" \
  http://localhost:9090/

# Test Claude-ZAI
curl -X POST http://localhost:9600/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello!","timeout":10000}'

# Create a task
curl -X POST http://localhost:9090/v1/tasks \
  -H "Authorization: Bearer change-this-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Write a hello world function",
    "provider": "gemini"
  }'
```

## Environment Variables

```bash
# Required API Keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Management API
MANAGEMENT_API_KEY=your-secure-token

# Optional
ENABLE_DESKTOP=false
ENABLE_CODE_SERVER=false
GPU_ACCELERATION=true
```

## Volumes

Persistent data stored in Docker volumes:
- `workspace` - User workspace files
- `model-cache` - Downloaded AI models
- `agent-memory` - Agent session data
- `config-persist` - Configuration
- `management-logs` - API logs

## Advantages of Standalone Deployment

✓ **Minimal footprint** - Only cachyos directory needed
✓ **Always up-to-date** - Installs latest from npm
✓ **Easy distribution** - Single directory to share
✓ **Clean separation** - No dev dependencies
✓ **Production ready** - Optimized build

## Upgrading agentic-flow

The standalone version installs `agentic-flow@latest` from npm. To upgrade:

```bash
docker compose -f docker-compose.standalone.yml down
docker compose -f docker-compose.standalone.yml build --no-cache
docker compose -f docker-compose.standalone.yml up -d
```

Or pin to specific version in `Dockerfile.workstation.standalone`:
```dockerfile
RUN npm install -g agentic-flow@1.5.10 pm2 gemini-flow claude
```
