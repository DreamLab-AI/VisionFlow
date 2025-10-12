# Final Architecture: Simplified Worker Control System

## Executive Summary

**Clean, secure architecture focused on worker spawning, control, and monitoring.**

- **External Interface**: Single Management API (port 9090)
- **Worker Execution**: Isolated sessions via agentic-session-manager
- **MCP Tools**: Run via stdio when needed (no HTTP, no exposed ports)
- **Optional**: Desktop (VNC) and IDE (code-server) for development

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     External Access                              │
│                    (Authenticated)                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │ Management API   │ Port 9090
                    │  (Node.js/       │ - Spawn workers
                    │   Fastify)       │ - Monitor status
                    └────────┬─────────┘ - View logs
                             │           - System health
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
┌─────────▼────────┐ ┌───────▼───────┐ ┌───────▼───────┐
│ Worker Session 1 │ │ Worker Sess 2 │ │ Worker Sess 3 │
│ (Isolated Dir)   │ │ (Isolated Dir)│ │ (Isolated Dir)│
│                  │ │               │ │               │
│  agentic-flow    │ │ agentic-flow  │ │ agentic-flow  │
│    ├─ Task       │ │   ├─ Task     │ │   ├─ Task     │
│    └─ MCP Tools  │ │   └─ MCP Tools│ │   └─ MCP Tools│
│       (stdio)    │ │      (stdio)  │ │      (stdio)  │
│       ├─Context7 │ │      ├─Playwright│ │    ├─GitHub │
│       └─Git      │ │      └─Fetch  │ │      └─Context7│
└──────────────────┘ └───────────────┘ └───────────────┘

Optional Services (disabled by default):
┌────────────────┐  ┌─────────────────┐
│ VNC Desktop    │  │ code-server     │
│ Port 5901/6901 │  │ Port 8080       │
│ (GUI access)   │  │ (Web IDE)       │
└────────────────┘  └─────────────────┘
```

## Core Components

### 1. Management API (Port 9090)

**The ONLY required external interface.**

#### Endpoints

**Worker Management:**
- `POST /v1/sessions` - Spawn isolated worker
- `GET /v1/sessions/:id` - Get worker status
- `GET /v1/sessions/:id/log` - Stream worker logs
- `GET /v1/sessions` - List all workers
- `DELETE /v1/sessions/:id` - Stop worker

**System Monitoring:**
- `GET /v1/status` - GPU, system resources, active workers
- `GET /health` - Simple health check
- `GET /ready` - Readiness probe

**Authentication:**
- Bearer token via `Authorization` header
- Configured via `MANAGEMENT_API_KEY` environment variable

### 2. Session Manager

**Script:** `/app/assets/core-assets/scripts/agentic-session-manager.sh`

Provides worker isolation:

```bash
# Create isolated session
SESSION_ID=$(agentic-session-manager.sh create-and-start \
  "coder" \
  "Build REST API with Express" \
  "gemini")

# Each session gets:
/home/devuser/workspace/sessions/${SESSION_ID}/
  ├── .session-meta.json    # Status, timestamps
  ├── workspace files       # Task outputs
  └── .db files            # SQLite (no conflicts)

/home/devuser/logs/sessions/${SESSION_ID}.log
```

### 3. MCP Tools (stdio only)

**Configuration:** `/home/devuser/.config/agentic-flow/mcp.json`

Tools spawn on-demand when workers need them:

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "type": "stdio"
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-playwright"],
      "type": "stdio"
    }
  }
}
```

**Available Tools:**
- **context7**: Up-to-date documentation
- **playwright**: Browser automation
- **filesystem**: File operations
- **git**: Git operations
- **github**: GitHub API
- **fetch**: Web content fetching
- **brave-search**: Web search
- **claude-flow**: Workflow orchestration

All tools run via stdin/stdout - no HTTP, no exposed ports.

## Services (supervisord)

Minimal service footprint:

### Required:
- **management-api** - Worker control interface
- **session-cleanup** - Periodic cleanup of old sessions

### Optional (disabled by default):
- **xvnc** - VNC server (if ENABLE_DESKTOP=true)
- **xfce4** - Desktop environment
- **novnc** - Browser VNC
- **code-server** - Web IDE (if ENABLE_CODE_SERVER=true)

## Port Exposure

| Port | Service | Required | Purpose |
|------|---------|----------|---------|
| 9090 | Management API | ✅ Yes | Worker control & monitoring |
| 5901 | VNC | ❌ Optional | GUI access (ENABLE_DESKTOP=true) |
| 6901 | noVNC | ❌ Optional | Browser GUI (ENABLE_DESKTOP=true) |
| 8080 | code-server | ❌ Optional | Web IDE (ENABLE_CODE_SERVER=true) |

**Default: Only port 9090 exposed.**

## Usage Examples

### 1. Spawn Worker

```bash
curl -X POST http://localhost:9090/v1/sessions \
  -H "Authorization: Bearer $MANAGEMENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Create 3D cube in Blender. Use context7 for Blender Python API docs.",
    "provider": "gemini"
  }'

# Response:
{
  "sessionId": "abc-123-def",
  "status": "accepted",
  "message": "Worker session created",
  "sessionDir": "/home/devuser/workspace/sessions/abc-123-def",
  "logFile": "/home/devuser/logs/sessions/abc-123-def.log"
}
```

### 2. Monitor Worker

```bash
# Check status
curl -H "Authorization: Bearer $MANAGEMENT_API_KEY" \
  http://localhost:9090/v1/sessions/abc-123-def

# Response:
{
  "sessionId": "abc-123-def",
  "agent": "coder",
  "task": "Create 3D cube...",
  "provider": "gemini",
  "status": "running",
  "startTime": 1704110400000,
  "duration": 45000,
  "logTail": "... last 50 lines ..."
}
```

### 3. Stream Logs

```bash
curl -H "Authorization: Bearer $MANAGEMENT_API_KEY" \
  http://localhost:9090/v1/sessions/abc-123-def/log?follow=true
```

### 4. System Status

```bash
curl -H "Authorization: Bearer $MANAGEMENT_API_KEY" \
  http://localhost:9090/v1/status

# Response:
{
  "timestamp": "2025-01-01T12:00:00.000Z",
  "api": {
    "uptime": 3600,
    "version": "2.0.0"
  },
  "workers": {
    "active": 3,
    "total": 15
  },
  "gpu": {
    "available": true,
    "gpus": [...]
  },
  "system": {
    "cpu": {...},
    "memory": {...},
    "disk": {...}
  }
}
```

## Configuration

### Environment Variables

**Required:**
```bash
MANAGEMENT_API_KEY=your-secure-token-here
```

**API Keys (at least one provider):**
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GOOGLE_GEMINI_API_KEY=AIza...
```

**Optional Services:**
```bash
ENABLE_DESKTOP=false        # Enable VNC/GUI
ENABLE_CODE_SERVER=false    # Enable web IDE
```

**MCP Tool APIs (optional):**
```bash
CONTEXT7_API_KEY=...        # Higher rate limits
GITHUB_TOKEN=ghp_...        # GitHub API access
BRAVE_API_KEY=...           # Web search
```

## How Workers Use MCP Tools

When a worker spawns:

1. **Inherits MCP configuration** from `/home/devuser/.config/agentic-flow/mcp.json`
2. **Task mentions tool**: "Use context7 for Express docs"
3. **Worker spawns tool** via stdio: `npx -y @upstash/context7-mcp`
4. **Tool communicates** via stdin/stdout (JSON-RPC)
5. **Tool provides result** to worker
6. **Tool exits** when done (clean lifecycle)

**No HTTP servers, no port management, no service orchestration.**

## Benefits

### Security
- ✅ Single authenticated endpoint
- ✅ No exposed tool ports
- ✅ Minimal attack surface

### Simplicity
- ✅ 4 services instead of 15+
- ✅ No HTTP server overhead for tools
- ✅ No port conflicts
- ✅ Simpler supervisord config

### Isolation
- ✅ Workers in isolated directories
- ✅ Tools spawn per-worker
- ✅ No shared state
- ✅ Clean lifecycle

### Efficiency
- ✅ On-demand tool execution
- ✅ No idle servers
- ✅ Lower memory footprint
- ✅ Faster startup

### Scalability
- ✅ Workers spawn independently
- ✅ No port exhaustion
- ✅ Linear scaling
- ✅ No service coordination

## File Structure

```
docker/cachyos/
├── FINAL-ARCHITECTURE.md           (This file)
├── ARCHITECTURE-SIMPLIFIED.md      (Design rationale)
├── docker-compose-simple.yml       (Minimal port exposure)
├── config/
│   ├── supervisord-simple.conf     (4 services, not 15)
│   └── mcp-stdio.json              (stdio tools, not HTTP)
├── assets/
│   └── core-assets/
│       └── scripts/
│           └── agentic-session-manager.sh
└── management-api/
    ├── server.js                   (Worker control API)
    ├── utils/
    │   ├── process-manager.js      (Session spawning)
    │   └── system-monitor.js       (Health checks)
    └── routes/
        └── sessions.js             (Worker endpoints)
```

## Deployment

### 1. Configure Environment

```bash
cp docker/cachyos/.env.example docker/cachyos/.env
# Edit .env with your API keys
```

### 2. Build and Start

```bash
cd docker/cachyos
docker-compose -f docker-compose-simple.yml up -d
```

### 3. Verify

```bash
# Check API health
curl http://localhost:9090/health

# Test authentication
curl -H "Authorization: Bearer $MANAGEMENT_API_KEY" \
  http://localhost:9090/v1/status
```

### 4. Spawn First Worker

```bash
curl -X POST http://localhost:9090/v1/sessions \
  -H "Authorization: Bearer $MANAGEMENT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "coder",
    "task": "Hello world in Python",
    "provider": "gemini"
  }'
```

## Optional: Enable Desktop

For GUI tool access (Blender, QGIS, etc.):

```bash
# In .env
ENABLE_DESKTOP=true

# Restart container
docker-compose -f docker-compose-simple.yml restart

# Access via VNC
vnc://localhost:5901

# Or browser
http://localhost:6901
```

## Migration from Old Architecture

If you have the old HTTP-based MCP servers:

### What to Remove:
- ❌ HTTP server implementations (blender-mcp-server.js, etc.)
- ❌ Port mappings for MCP tools (9876-9882)
- ❌ Supervisord entries for MCP servers
- ❌ Health checks for MCP services
- ❌ HTTP client bridge scripts

### What to Keep:
- ✅ Management API
- ✅ Session manager
- ✅ mcp-stdio.json configuration
- ✅ Desktop environment (optional)

### Migration Steps:
1. Use `docker-compose-simple.yml` instead of old compose file
2. Use `supervisord-simple.conf` instead of old config
3. Use `mcp-stdio.json` for MCP configuration
4. Update Management API to use session manager
5. Remove HTTP MCP server code

## Monitoring

### Check All Services

```bash
docker exec -it agentic-flow-cachyos supervisorctl status
```

### View Logs

```bash
# Management API
docker exec -it agentic-flow-cachyos tail -f /home/devuser/logs/management-api.log

# Worker session
docker exec -it agentic-flow-cachyos tail -f /home/devuser/logs/sessions/<session-id>.log
```

### Resource Usage

```bash
docker stats agentic-flow-cachyos
```

## Troubleshooting

### Worker Won't Spawn

Check Management API logs:
```bash
docker exec -it agentic-flow-cachyos cat /home/devuser/logs/management-api.err.log
```

### MCP Tool Not Found

Verify tool in mcp.json:
```bash
docker exec -it agentic-flow-cachyos cat /home/devuser/.config/agentic-flow/mcp.json
```

### Authentication Failed

Check API key:
```bash
echo $MANAGEMENT_API_KEY
```

## Conclusion

This architecture provides:
- **Clean separation**: External control via API, internal tools via stdio
- **Security**: Single authenticated endpoint, no exposed tool ports
- **Simplicity**: Minimal services, on-demand tools
- **Isolation**: Workers in dedicated directories
- **Efficiency**: Lower resource usage, faster startup

**Focus:** Worker spawning, control, and monitoring - exactly what's needed, nothing more.
