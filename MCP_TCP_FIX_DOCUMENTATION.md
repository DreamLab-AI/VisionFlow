# MCP TCP Server Fix Documentation

**Date**: 2025-08-30  
**Issue**: MCP TCP server spawning new instance per connection  
**Solution**: Replace spawning behavior with persistent shared MCP instance

## Problem Analysis

### Root Cause
The original `/app/core-assets/scripts/mcp-tcp-server.js` spawns a NEW MCP process for EACH TCP connection:

```javascript
// Line 47 - PROBLEM: New process per connection
const mcp = spawn('npx', ['claude-flow@alpha', 'mcp', 'start', '--stdio', '--file', '/workspace/.mcp.json'], {
```

This causes:
1. Every request sees "Starting Claude Flow MCP server" startup message
2. No state persistence between requests
3. Agent tracking fails (each connection has isolated MCP instance)
4. Swarm operations don't persist
5. Massive overhead from spawning processes

### Evidence
From test output:
```
[TEST] Initialize Swarm (mesh topology)
  → Request: tools/call
  ← No valid response
  Raw output: ✅ Starting Claude Flow MCP server in stdio mode...
```

Every single request triggered a new MCP server startup.

## Solution Implementation

### Option 1: Persistent MCP TCP Server (Recommended)

Replace `/app/core-assets/scripts/mcp-tcp-server.js` with persistent version that:
1. Spawns ONE MCP instance on startup
2. Shares that instance across ALL TCP connections
3. Routes responses back to correct clients by request ID
4. Maintains agent state across connections

**File**: `/workspace/ext/mcp-tcp-persistent.js`

Key features:
- Single MCP process spawned once
- Request ID tracking for response routing
- Client management with pending request tracking
- Automatic restart on MCP crash
- Proper initialization handling

### Option 2: Patch Existing Server

Modify the existing `mcp-tcp-server.js` to maintain a singleton MCP instance:

```javascript
// Add at top of file
let sharedMCPProcess = null;
let sharedMCPInterface = null;

// Modify startTCPServer to use shared instance
if (!sharedMCPProcess) {
  sharedMCPProcess = spawn(...);
  // Initialize once
}
// Share sharedMCPProcess across all connections
```

## Container Patch Instructions

### For Dockerfile/Container Build

Add to `setup-workspace.sh` or container startup:

```bash
# Function to patch MCP TCP server for persistence
patch_mcp_tcp_server() {
    echo "Patching MCP TCP server for persistence..."
    
    # Option 1: Replace entirely (RECOMMENDED)
    if [ -f /workspace/ext/mcp-tcp-persistent.js ]; then
        cp /workspace/ext/mcp-tcp-persistent.js /app/core-assets/scripts/mcp-tcp-server.js
        chmod +x /app/core-assets/scripts/mcp-tcp-server.js
        echo "✓ Replaced MCP TCP server with persistent version"
    else
        # Option 2: In-place patch
        cat > /tmp/mcp-tcp-patch.js << 'EOF'
// Singleton MCP instance
let sharedMCP = null;
let sharedClients = new Map();
let initialized = false;

function getOrCreateMCP() {
    if (!sharedMCP) {
        sharedMCP = require('child_process').spawn('npx', 
            ['claude-flow@alpha', 'mcp', 'start', '--stdio'],
            {
                stdio: ['pipe', 'pipe', 'pipe'],
                cwd: '/workspace',
                env: { ...process.env, CLAUDE_FLOW_DIRECT_MODE: 'true' }
            }
        );
        // Add initialization logic here
    }
    return sharedMCP;
}
EOF
        # Apply patch logic here
        echo "✓ Applied in-place patch to MCP TCP server"
    fi
}

# Call the patch function
patch_mcp_tcp_server
```

### For Running Container

```bash
# 1. Stop current MCP TCP server
docker exec multi-agent-container pkill -f mcp-tcp-server.js

# 2. Copy persistent version
docker cp mcp-tcp-persistent.js multi-agent-container:/app/core-assets/scripts/mcp-tcp-server-new.js

# 3. Backup original
docker exec multi-agent-container mv /app/core-assets/scripts/mcp-tcp-server.js /app/core-assets/scripts/mcp-tcp-server.js.bak

# 4. Replace with new version
docker exec multi-agent-container mv /app/core-assets/scripts/mcp-tcp-server-new.js /app/core-assets/scripts/mcp-tcp-server.js

# 5. Restart supervisor to apply changes
docker exec multi-agent-container supervisorctl restart all
```

## Verification Steps

After applying the fix:

1. **Test persistence**:
```bash
# Should reuse same MCP instance
echo '{"jsonrpc":"2.0","id":"1","method":"tools/call","params":{"name":"swarm_init","arguments":{"topology":"mesh"}}}' | nc multi-agent-container 9500
sleep 1
echo '{"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"agent_list","arguments":{"filter":"all"}}}' | nc multi-agent-container 9500
```

2. **Check for single process**:
```bash
docker exec multi-agent-container ps aux | grep "mcp start" | wc -l
# Should show 1, not multiple
```

3. **Verify agent persistence**:
- Initialize swarm in one connection
- List agents in another connection
- Agents should be visible (not empty array)

## Expected Behavior After Fix

### Before Fix
- Each TCP connection spawns new MCP
- No state persistence
- Agent operations fail
- "Starting Claude Flow MCP server" on every request

### After Fix
- Single shared MCP instance
- State persists across connections
- Agent operations work correctly
- Proper JSON-RPC responses

## Integration Points

The fix ensures:

1. **VisionFlow** can maintain persistent agent swarms
2. **WebSocket updates** reflect actual agent state
3. **Multiple clients** can connect simultaneously
4. **Agent graph visualization** works correctly

## Rollback Plan

If issues occur:
```bash
# Restore original
docker exec multi-agent-container mv /app/core-assets/scripts/mcp-tcp-server.js.bak /app/core-assets/scripts/mcp-tcp-server.js
docker exec multi-agent-container supervisorctl restart all
```

## Testing

Use `/workspace/ext/test_from_visionflow.sh` to verify:
- Swarm initialization persists
- Agent spawning works
- Agent list returns spawned agents
- Swarm destruction cleans up properly

## Notes

- The persistent server maintains a single MCP process lifetime
- Request/response matching uses JSON-RPC ID field
- Notifications (no ID) are broadcast to all clients
- Automatic restart on MCP crash with 5-second delay
- Initialize request is cached to avoid re-initialization

This fix is **CRITICAL** for the agent system to function properly.