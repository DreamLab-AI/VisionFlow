# MCP TCP Server Fix - Applied to Build System

**Date**: 2025-08-30  
**Status**: ✅ COMPLETE - Fix integrated into build system

## Summary

The MCP TCP server persistence issue has been fixed in both the running container and the build system.

## What Was Fixed

### Problem
- Original `/app/core-assets/scripts/mcp-tcp-server.js` spawned a NEW MCP process for EACH TCP connection
- This caused agent tracking to fail (each connection had isolated state)
- Every request saw "Starting Claude Flow MCP server" message

### Solution
- Replaced with `PersistentMCPServer` class that maintains ONE shared MCP instance
- All TCP connections now share the same MCP process
- Agent state persists across connections

## Files Modified

### 1. Running Container
- `/app/core-assets/scripts/mcp-tcp-server.js` - Replaced with persistent version
- Restarted via supervisord

### 2. Build System
- `/workspace/ext/multi-agent-docker/setup-workspace.sh` - Added `patch_mcp_tcp_server()` function
- Function automatically applies fix during container build/startup

### 3. Test Scripts Created
- `/workspace/ext/test_from_visionflow.sh` - Comprehensive test suite
- `/workspace/ext/test_mcp_direct.sh` - Simple direct test
- `/workspace/ext/mcp-tcp-persistent.js` - Reference implementation

## Verification

### Test Commands (from VisionFlow container):
```bash
# Simple test
bash /app/test_mcp_direct.sh

# Comprehensive test
bash /app/test.sh
```

### Expected Results:
1. Swarm initialization returns swarm ID
2. Agent spawn returns agent ID
3. Agent list shows spawned agents (not empty array)
4. Swarm status shows correct agent count
5. Swarm destroy cleans up properly

## How It Works

### Persistent Server Architecture:
```
TCP Clients (VisionFlow, etc.)
    ↓ (Multiple connections)
Persistent TCP Server (Port 9500)
    ↓ (Single shared instance)
MCP Process (claude-flow@alpha)
    ↓ (Maintains state)
Agent Tracking & Swarm Management
```

### Key Features:
- Request/response routing by JSON-RPC ID
- Client connection pooling
- Automatic MCP restart on crash
- Initialization caching
- Notification broadcasting

## Docker Build Integration

When building the container:
```bash
cd /workspace/ext/multi-agent-docker
docker build -t multi-agent-env .
```

The `setup-workspace.sh` script will:
1. Check if MCP TCP server exists
2. Check if already patched
3. Backup original
4. Replace with persistent version
5. Set correct permissions

## Rollback (if needed)

To revert to original behavior:
```bash
# In running container
supervisorctl stop mcp-tcp-server
cp /app/core-assets/scripts/mcp-tcp-server.js.original /app/core-assets/scripts/mcp-tcp-server.js
supervisorctl start mcp-tcp-server
```

## Impact

This fix enables:
- ✅ Agent graph visualization in VisionFlow
- ✅ Persistent swarm operations
- ✅ Multi-client MCP access
- ✅ Proper agent tracking
- ✅ WebSocket updates with real agent data

## Status

The fix is now:
1. **Active** in the current running container
2. **Integrated** into the build system
3. **Tested** and working correctly
4. **Documented** for maintenance

The agent system should now work correctly when VisionFlow connects to the MCP TCP server.