# Simplified Architecture: Worker Control & Monitoring Only

## Core Principle

**MCP tools run internally via stdio when agents need them - NOT as exposed HTTP servers.**

Only expose:
- Management API (9090) - Worker spawning, control, monitoring
- VNC (5901/6901) - GUI access if needed
- code-server (8080) - Web IDE if needed

## Architecture Overview

```
External Access:
  ↓
Management API (Port 9090)
  ├── POST /v1/sessions - Spawn isolated worker
  ├── GET /v1/sessions/{id} - Check worker status
  ├── GET /v1/sessions - List active workers
  └── GET /v1/status - System health

Internal (No External Exposure):
  └── Agentic Flow Workers (isolated sessions)
      └── MCP Tools (stdio, invoked when needed)
          ├── Context7
          ├── Blender
          ├── QGIS
          ├── Playwright
          ├── Web Summary
          ├── KiCAD
          └── ImageMagick
```

## Why This Is Better

### 1. Security
- **Only one attack surface**: Management API with authentication
- **No exposed tool servers**: MCP tools run internally only
- **Reduced port exposure**: 3 ports instead of 10+

### 2. Simplicity
- **No HTTP server overhead**: Tools spawn on-demand via stdio
- **No port conflicts**: Tools don't need unique ports
- **Simpler supervisord config**: Just desktop + management API

### 3. Isolation
- **Tools run in worker context**: Each session gets fresh tool instances
- **No shared state**: No conflicts between concurrent workers
- **Clean lifecycle**: Tools start/stop with worker sessions

### 4. Resource Efficiency
- **On-demand execution**: Tools only run when actually needed
- **No idle servers**: Nothing running when not in use
- **Lower memory footprint**: No persistent HTTP servers

## Correct Implementation

### supervisord.conf (Simplified)

```ini
[supervisord]
nodaemon=true
logfile=/home/devuser/logs/supervisord.log

# Desktop Environment (optional - only if GUI access needed)
[program:xvnc]
command=/usr/bin/Xvnc :1 -geometry 1920x1080 -depth 24 -rfbport 5901
...

[program:xfce4]
command=/usr/bin/startxfce4
...

# Management API - THE ONLY EXTERNAL INTERFACE
[program:management-api]
command=/usr/bin/node /home/devuser/management-api/server.js
...

# That's it! No MCP HTTP servers needed.
```

### mcp.json (For Worker Sessions)

MCP tools configured for stdio use by workers:

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
      "args": ["@modelcontextprotocol/server-playwright"],
      "type": "stdio"
    }
  }
}
```

When a worker needs a tool, agentic-flow spawns it via stdio automatically.

### docker-compose.yml (Minimal Ports)

```yaml
ports:
  - "9090:9090"   # Management API - ONLY EXTERNAL INTERFACE
  - "5901:5901"   # VNC (optional, for GUI access)
  - "6901:6901"   # noVNC (optional, browser VNC)
  - "8080:8080"   # code-server (optional, web IDE)

# NO MCP tool ports exposed!
```

## Usage Pattern

### 1. Spawn Worker via API

```bash
curl -X POST http://localhost:9090/v1/sessions \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "agent": "coder",
    "task": "Create 3D model with Blender. Use context7 for Blender API docs.",
    "provider": "gemini"
  }'

# Response: { "sessionId": "abc-123" }
```

### 2. Worker Automatically Uses MCP Tools

The worker session:
1. Reads `mcp.json` configuration
2. When task mentions "Blender", spawns Blender MCP via stdio
3. When task mentions "context7", spawns Context7 MCP via stdio
4. Tools communicate via stdin/stdout (no HTTP, no ports)
5. Tools clean up when session ends

### 3. Monitor Worker

```bash
# Check status
curl http://localhost:9090/v1/sessions/abc-123 \
  -H "Authorization: Bearer $API_KEY"

# View logs
curl http://localhost:9090/v1/sessions/abc-123/log \
  -H "Authorization: Bearer $API_KEY"
```

## What We Don't Need

### ❌ HTTP MCP Servers
```javascript
// DON'T create HTTP servers for MCP tools
const fastify = require('fastify');
const app = fastify();
app.post('/tools/render', ...); // ❌ Not needed
app.listen(9876); // ❌ Don't expose ports
```

### ❌ Port Mappings for Each Tool
```yaml
# DON'T expose individual MCP tool ports
ports:
  - "9876:9876"  # ❌ Blender MCP
  - "9877:9877"  # ❌ QGIS MCP
  - "9878:9878"  # ❌ Playwright MCP
```

### ❌ Supervisord Entries for MCP Servers
```ini
# DON'T run MCP tools as persistent services
[program:blender-mcp]  # ❌ Not needed
command=node /app/blender-mcp-server.js
```

## What We DO Need

### ✅ Management API Only

```javascript
// Management API - spawn/control/monitor workers
app.post('/v1/sessions', async (req, res) => {
  const sessionId = await sessionManager.create(req.body);
  // Worker will use MCP tools internally via stdio
  res.send({ sessionId });
});
```

### ✅ Session Manager

```bash
# agentic-session-manager.sh
# Spawns isolated agentic-flow workers
# Each worker loads mcp.json and spawns tools as needed
```

### ✅ MCP Configuration (stdio only)

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "type": "stdio"  // ✅ stdio, not HTTP
    }
  }
}
```

## Simplified Port Map

| Port | Service | Purpose |
|------|---------|---------|
| 9090 | Management API | Spawn/control/monitor workers |
| 5901 | VNC | GUI access (optional) |
| 6901 | noVNC | Browser GUI (optional) |
| 8080 | code-server | Web IDE (optional) |

**Total: 4 ports (or just 1 if GUI/IDE not needed)**

## Benefits Summary

1. **Security**: Single authenticated endpoint
2. **Simplicity**: No server orchestration for tools
3. **Isolation**: Tools run in worker context, clean lifecycle
4. **Efficiency**: On-demand execution, no idle servers
5. **Scalability**: Workers spawn tools independently
6. **Maintenance**: Fewer moving parts to manage

## Implementation Impact

### Remove/Simplify:
- ❌ HTTP server implementations for MCP tools
- ❌ Port mappings for MCP tools
- ❌ Supervisord entries for MCP servers
- ❌ Health monitoring for MCP servers
- ❌ Service management endpoints for MCP tools

### Keep:
- ✅ Management API (worker control)
- ✅ Session manager (isolation)
- ✅ mcp.json configuration (stdio)
- ✅ Desktop environment (optional GUI access)
- ✅ code-server (optional IDE)

## Migration Notes

The current implementation created HTTP servers for MCP tools because it was adapted from the old hive-mind architecture. But agentic-flow **natively supports stdio MCP** - we don't need the HTTP layer at all.

The correct pattern:
1. User calls Management API to spawn worker
2. Worker inherits mcp.json configuration
3. Worker spawns MCP tools via stdio when needed
4. Tools communicate via stdin/stdout
5. Tools clean up when worker completes

This is simpler, more secure, and more efficient than running persistent HTTP servers.

## Conclusion

**The only thing that needs external exposure is the Management API for worker control and monitoring.**

MCP tools should run internally via stdio, spawned on-demand by workers, not as persistent HTTP servers with exposed ports.

This dramatically simplifies the architecture while maintaining all functionality.
