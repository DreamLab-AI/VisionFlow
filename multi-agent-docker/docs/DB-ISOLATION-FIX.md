# Database Isolation Fix for Container Exit Issue

## Problem Summary

The container was exiting when connecting to claude-flow MCP due to SQLite database locking conflicts. Multiple processes were trying to write to the same database file:

- **Supervisord's mcp-tcp-server** → `/workspace/.swarm/memory.db`
- **Claude's local MCP instance** → `/workspace/.swarm/memory.db` (default)
- **Hook invocations** → `/workspace/.swarm/memory.db` (default)

SQLite doesn't support concurrent writers, causing lock → crash → graceful container exit.

## Solution: Database Path Isolation

Each claude-flow instance now uses its own isolated database:

```
/workspace/.swarm/
├── tcp-server.db      # Supervisord's mcp-tcp-server
├── claude-local.db    # Claude's MCP instance (via .claude.json)
├── claude-hooks.db    # Pre/Post tool hooks
└── sessions/          # Isolated TCP proxy sessions
    └── {sessionId}/
        └── memory.db
```

## Changes Made

### 1. Supervisord Configuration (`supervisord.conf`)
**Changed:**
```diff
-CLAUDE_FLOW_DB_PATH="/workspace/.swarm/memory.db"
+CLAUDE_FLOW_DB_PATH="/workspace/.swarm/tcp-server.db"
```

### 2. MCP TCP Server (`core-assets/scripts/mcp-tcp-server.js`)
**Changed:**
```diff
-CLAUDE_FLOW_DB_PATH: '/workspace/.swarm/memory.db',
+CLAUDE_FLOW_DB_PATH: '/workspace/.swarm/tcp-server.db',
```

### 3. MCP Template Configuration (`core-assets/mcp.json`)
**Added:**
```json
{
  "mcpServers": {
    "claude-flow": {
      "env": {
        "CLAUDE_FLOW_DB_PATH": "/workspace/.swarm/claude-local.db"
      }
    }
  }
}
```

### 4. New Configuration Script (`scripts/configure-claude-mcp.sh`)
**Created:** Automatic configuration script that runs on container startup to:
- Update `/home/dev/.claude/.claude.json` with isolated DB path
- Update `/workspace/.mcp.json` with isolated DB path
- Update `/workspace/.claude/settings.json` hooks with isolated DB path
- Set proper ownership on all config files

### 5. Entrypoint Script (`entrypoint.sh`)
**Added:** Configuration step during startup:
```bash
# Configure Claude MCP with isolated databases
if [ -f /app/scripts/configure-claude-mcp.sh ]; then
    /app/scripts/configure-claude-mcp.sh || echo "⚠️ MCP config failed, continuing..."
fi
```

### 6. Dockerfile
**Changed:**
```diff
+COPY scripts/ /app/scripts/
+RUN chmod +x /app/scripts/*.sh 2>/dev/null || true
```

### 7. Docker Compose (`docker-compose.yml`)
**Added:**
```yaml
init: true  # Prevents container exit on child process failures
```

## How It Works

### Container Startup Flow

1. **Entrypoint starts** → Launches supervisord in background
2. **Supervisord starts services:**
   - `mcp-tcp-server` → Uses `/workspace/.swarm/tcp-server.db`
   - `claude-flow-tcp` → Creates isolated sessions in `/workspace/.swarm/sessions/{id}/memory.db`
3. **Background setup job:**
   - Runs `claude --dangerously-skip-permissions /exit` to initialize Claude
   - **Runs `configure-claude-mcp.sh`** to set up DB isolation:
     - Creates/updates `/home/dev/.claude/.claude.json`
     - Sets `CLAUDE_FLOW_DB_PATH=/workspace/.swarm/claude-local.db`
     - Updates hook commands with `CLAUDE_FLOW_DB_PATH=/workspace/.swarm/claude-hooks.db`
4. **User connects Claude MCP** → Uses isolated `claude-local.db`, no conflicts

### Hook Isolation

Before:
```bash
npx claude-flow@alpha hooks pre-command --command '{}'
# ❌ Uses default /workspace/.swarm/memory.db → LOCK CONFLICT
```

After:
```bash
CLAUDE_FLOW_DB_PATH=/workspace/.swarm/claude-hooks.db npx claude-flow@alpha hooks pre-command --command '{}'
# ✅ Uses isolated database, no conflicts
```

## Testing the Fix

### 1. Rebuild the Docker image:
```bash
docker-compose build multi-agent
```

### 2. Start the container:
```bash
docker-compose up -d multi-agent
```

### 3. Check logs for configuration:
```bash
docker exec multi-agent-container cat /workspace/.setup.log | grep "database isolation"
```

Should see:
```
[2025-10-05 XX:XX:XX] Configuring Claude MCP database isolation...
✅ Claude MCP configuration complete

Database isolation:
  - TCP Server:    /workspace/.swarm/tcp-server.db
  - Claude MCP:    /workspace/.swarm/claude-local.db
  - Hook calls:    /workspace/.swarm/claude-hooks.db
```

### 4. Verify Claude config:
```bash
docker exec multi-agent-container jq '.mcpServers["claude-flow"]' /home/dev/.claude/.claude.json
```

Should show:
```json
{
  "command": "/workspace/claude-flow",
  "args": ["mcp", "start"],
  "type": "stdio",
  "env": {
    "CLAUDE_FLOW_DB_PATH": "/workspace/.swarm/claude-local.db"
  }
}
```

### 5. Connect Claude MCP and verify no crash:
- In Claude Code inside container: `/mcp`
- Connect to `claude-flow` MCP
- Container should remain running
- Check `ps aux | grep claude-flow` to see isolated processes

### 6. Verify database files:
```bash
docker exec multi-agent-container ls -la /workspace/.swarm/*.db
```

Should see:
```
-rw-r--r-- 1 dev dev  XXXX tcp-server.db
-rw-r--r-- 1 dev dev  XXXX claude-local.db
-rw-r--r-- 1 dev dev  XXXX claude-hooks.db
```

## Rollback Plan

If issues occur, you can:

1. **Check backups:**
```bash
docker exec multi-agent-container ls -la /home/dev/.claude/*.bak.*
docker exec multi-agent-container ls -la /workspace/.claude/settings.json.bak.*
```

2. **Restore from backup:**
```bash
docker exec multi-agent-container bash -c 'cp $(ls -t /home/dev/.claude/.claude.json.bak.* | head -1) /home/dev/.claude/.claude.json'
```

3. **Disable auto-configuration:**
```bash
# Remove the configure script
docker exec multi-agent-container rm /app/scripts/configure-claude-mcp.sh
# Restart container
docker-compose restart multi-agent
```

## Benefits

✅ **No more container exits** - Each instance has its own database
✅ **No SQLite lock conflicts** - Isolated write operations
✅ **Faster MCP startup** - No waiting for locks
✅ **Better debugging** - Each component's data is separate
✅ **Automatic configuration** - Setup script runs on every container start

## Additional Notes

- The `init: true` in docker-compose ensures the container doesn't exit if child processes fail
- Database files are created on-demand when each service starts
- All configurations are idempotent - safe to run multiple times
- The configure script creates backups before making changes
