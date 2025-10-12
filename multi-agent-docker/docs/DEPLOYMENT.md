# Standalone Deployment Guide

## What Changed

The Docker deployment has been converted to **standalone mode** - it now installs `agentic-flow` from npm instead of requiring the full repository.

### Files Changed

1. **docker-compose.yml** (now standalone)
   - Build context: `.` instead of `../..`
   - Installs from npm

2. **Dockerfile.workstation** (now standalone)
   - `RUN npm install -g agentic-flow@latest`
   - No longer copies from `../../agentic-flow/`

3. **start-agentic-flow.sh** (updated paths)
   - Works from cachyos directory
   - Auto-creates .env from template

4. **New files**
   - `.env.example` - Environment template
   - `README.md` - Complete documentation
   - `docker-compose.dev.yml` - Development version (uses local source)
   - `Dockerfile.workstation.dev` - Development Dockerfile

### Original Files (Preserved)

- `docker-compose.dev.yml` - For development with local source
- `Dockerfile.workstation.dev` - For development builds

## Deployment

### Option 1: Standalone (Production)

```bash
cd docker/cachyos
cp .env.example .env
# Edit .env with your API keys
./start-agentic-flow.sh --build
```

### Option 2: Development (Local Source)

```bash
cd docker/cachyos
cp .env.example .env
docker compose -f docker-compose.dev.yml up -d --build
```

## What You Need

### Standalone Deployment
```
docker/cachyos/    ← Just this directory!
```

### Development Deployment
```
agentic-flow/
├── agentic-flow/
└── docker/cachyos/
```

## Testing

After deployment:

```bash
# Test Management API
curl http://localhost:9090/health

# Test Claude-ZAI
curl -X POST http://localhost:9600/prompt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello","timeout":10000}'

# Test task creation
curl -X POST http://localhost:9090/v1/tasks \
  -H "Authorization: Bearer change-this-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"agent":"coder","task":"test","provider":"gemini"}'
```

## Migration from Old Setup

If you were using the old setup:

```bash
# Stop old containers
docker compose down

# Start new standalone version
./start-agentic-flow.sh --build
```

Your data in Docker volumes is preserved.
