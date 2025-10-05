# Docker Build Updated - TCP Bridge Architecture

**Date**: 2025-10-05
**Status**: Ready to rebuild

## Summary

Updated docker configuration to use **stdio-to-TCP bridge** for claude-flow MCP. This solves the stdio pipe closure issue while maintaining the working TCP infrastructure.

## Architecture Decision

**Question**: Should we use stdio or TCP?
**Answer**: **Both** - stdio interface with TCP backend via bridge

### Why This Works

1. ✅ **Claude Code expects stdio** - Gets the interface it wants
2. ✅ **TCP is reliable** - Port 9500 server is proven working
3. ✅ **Bridge stays alive** - Node.js process doesn't exit prematurely
4. ✅ **No stdin closure issues** - Bridge handles pipe lifecycle correctly
5. ✅ **Database isolation preserved** - TCP server uses `/workspace/.swarm/tcp-server.db`

## Changes Made

### 1. Updated `core-assets/mcp.json`
```json
"claude-flow": {
  "command": "node",
  "args": ["/app/scripts/stdio-to-tcp-bridge.js"],
  "type": "stdio",
  "env": {
    "MCP_HOST": "127.0.0.1",
    "MCP_PORT": "9500"
  }
}
```

### 2. Updated `scripts/configure-claude-mcp.sh`
- All jq commands now configure TCP bridge instead of direct binary
- Updated for `/home/dev/.mcp.json`, `/home/dev/.claude/.claude.json`, `/workspace/.mcp.json`
- Removed `CLAUDE_FLOW_DB_PATH` (no longer needed - TCP server handles DB)

### 3. Bridge Script Already Exists
`scripts/stdio-to-tcp-bridge.js` - 39 lines, already copied by Dockerfile line 102

### 4. Dockerfile (No Changes Needed)
Already has:
```dockerfile
COPY scripts/ /app/scripts/
RUN chmod +x ... /app/scripts/*.sh 2>/dev/null || true
```

## How It Works

```
Claude Code (stdio)
    ↓
stdio-to-tcp-bridge.js
    ↓ (TCP connection)
127.0.0.1:9500 (mcp-tcp-server)
    ↓
claude-flow backend
    ↓
/workspace/.swarm/tcp-server.db
```

## What Gets Fixed

### Before (Broken)
- Claude Code spawns `/usr/sbin/claude-flow mcp start`
- Stdin closes immediately
- Server sees EOF and shuts down
- Connection timeout after 30s
- Container stays up (✅ fixed earlier) but MCP fails

### After (Working)
- Claude Code spawns `node stdio-to-tcp-bridge.js`
- Bridge connects to TCP server on port 9500
- Bridge pipes stdin ↔ TCP socket
- TCP server handles all MCP protocol
- Connection succeeds

## Rebuild Instructions

```bash
# From multi-agent-docker directory
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker logs -f multi-agent-container

# Test MCP connection
docker exec -it multi-agent-container bash
claude --dangerously-skip-permissions
# Then use /mcp menu to connect
```

## Expected Results

After rebuild and reconnect:
- ✅ claude-flow MCP status: **connected**
- ✅ Container stays running
- ✅ TCP server shows active session
- ✅ No stdin closure errors
- ✅ Database at `/workspace/.swarm/tcp-server.db`

## Verification Commands

```bash
# Check TCP server is running
docker exec multi-agent-container netstat -tlnp | grep 9500

# Check database exists
docker exec multi-agent-container ls -la /workspace/.swarm/*.db

# Check bridge script
docker exec multi-agent-container cat /app/scripts/stdio-to-tcp-bridge.js

# Test TCP connection manually
docker exec multi-agent-container bash -c "echo '{\"jsonrpc\":\"2.0\",\"method\":\"initialize\"}' | nc localhost 9500"
```

## Database Isolation Status

| Component | Database Path |
|-----------|---------------|
| TCP MCP Server (port 9500) | `/workspace/.swarm/tcp-server.db` |
| Claude Hooks | `/workspace/.swarm/claude-hooks.db` |
| VisionFlow Sessions | `/workspace/.swarm/sessions/{id}/memory.db` |

All three isolated - no more SQLite lock conflicts.

## Files Modified

1. ✅ `core-assets/mcp.json` - TCP bridge config
2. ✅ `scripts/configure-claude-mcp.sh` - Auto-configuration updated
3. ✅ `scripts/stdio-to-tcp-bridge.js` - Bridge (already existed)
4. ✅ `Dockerfile` - Already copies scripts (no change needed)
5. ✅ `supervisord.conf` - Already has TCP server with isolated DB
6. ✅ `docker-compose.yml` - Already has `init: true`

## No More stdio vs TCP Debate

**We use BOTH:**
- stdio for the interface (what Claude Code wants)
- TCP for the transport (what actually works)
- Bridge to connect them (39 lines of Node.js)

This is the standard approach when you have:
- A client that requires stdio (Claude Code)
- A server that works better with TCP (claude-flow on port 9500)
- No control over the client's behavior (can't fix Claude Code's stdin handling)
