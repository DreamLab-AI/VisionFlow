# HOWTO: Fix Agent Tracking in Claude-Flow MCP Server

## Problem Statement
The MCP TCP server's `agent_list` command was returning mock data instead of real spawned agents. This occurred because:
1. The AgentTracker wasn't properly initialized
2. Spawned agents weren't being tracked with the correct swarmId
3. The agent_list handler had a fallback to mock data that masked the real issue

## Solution Overview
The fix involves patching the globally installed claude-flow package at `/usr/lib/node_modules/claude-flow` during container initialization via `setup-workspace.sh`.

## Architecture Understanding

### Directory Structure
- `/workspace/ext/claude-flow/` - Reference code only (not the running version)
- `/usr/lib/node_modules/claude-flow/` - Actual installed and running claude-flow
- `/workspace/ext/multi-agent-docker/` - Docker setup and patches
- `/app/core-assets/scripts/mcp-tcp-server.js` - TCP wrapper that spawns MCP instances

### Communication Flow
```
VisionFlow (Rust) → TCP:9500 → mcp-tcp-server.js → npx claude-flow@alpha mcp start
```

## Patches Applied

### Patch 1: Fix Agent Tracker Initialization
**File**: `/workspace/ext/multi-agent-docker/patches/01-fix-agent-tracker-init.patch`

**Changes**:
- Enhanced error handling for agent tracker module loading
- Added verification that global.agentTracker is available
- Added fallback manual initialization if automatic loading fails
- Added detailed logging for debugging

### Patch 2: Fix Agent Spawn Tracking
**File**: `/workspace/ext/multi-agent-docker/patches/02-fix-agent-spawn-tracking.patch`

**Changes**:
- Ensure swarmId consistency throughout agent_spawn
- Store activeSwarmId once and reuse it
- Pass swarmId to agentTracker.trackAgent()
- Add logging to confirm agents are tracked

### Patch 3: Remove Mock Data Fallback
**Applied directly in setup-workspace.sh**

**Changes**:
```javascript
// BEFORE: Returns mock data
return {
  success: true,
  swarmId: args.swarmId || 'mock-swarm',
  agents: [
    { id: 'agent-1', name: 'coordinator-1', type: 'coordinator', status: 'active', capabilities: [] },
    { id: 'agent-2', name: 'researcher-1', type: 'researcher', status: 'active', capabilities: [] },
    { id: 'agent-3', name: 'coder-1', type: 'coder', status: 'busy', capabilities: [] },
  ],
  count: 3,
  timestamp: new Date().toISOString(),
};

// AFTER: Returns empty array if no agents
return {
  success: true,
  swarmId: swarmId || 'default-swarm',
  agents: [],
  count: 0,
  timestamp: new Date().toISOString(),
};
```

### Patch 4: Enhanced Agent Spawn
**Applied directly in setup-workspace.sh**

**Key fixes**:
- Calculate swarmId once at the beginning
- Use consistent swarmId for storage and tracking
- Add proper error handling
- Log agent creation and tracking

### Patch 5: Agent Tracker Verification
**Applied directly in setup-workspace.sh**

**Additions**:
- Verify agent tracker after loading
- Create manual instance if needed
- Add startup logging

## VisionFlow Integration Changes

### Auto-Spawn Agents After Swarm Creation
**Files Modified**:
- `/workspace/ext/src/handlers/bots_handler.rs`
- `/workspace/ext/src/utils/mcp_connection.rs`

**Reason**: `swarm_init` only creates the swarm structure, it doesn't spawn agents. We need to call `agent_spawn` for each agent.

**Implementation**:
```rust
// In bots_handler.rs - After swarm_init succeeds
let agent_types = match request.topology.as_str() {
    "hierarchical" => vec!["coordinator", "analyst", "optimizer"],
    "mesh" => vec!["coordinator", "researcher", "coder", "analyst"],
    "star" => vec!["coordinator", "optimizer", "documenter"],
    _ => vec!["coordinator", "analyst", "optimizer"],
};

for agent_type in agent_types {
    match call_agent_spawn(&claude_flow_host, &claude_flow_port, agent_type).await {
        Ok(spawn_result) => info!("✅ Spawned {} agent", agent_type),
        Err(e) => warn!("Failed to spawn {} agent: {}", agent_type, e),
    }
}
```

## Setup Workspace Script Updates

### Location
`/workspace/ext/multi-agent-docker/setup-workspace.sh`

### Key Changes
1. **Installation Check**: Ensures claude-flow is installed before patching
2. **Path Discovery**: Finds installation at `/usr/lib/node_modules/claude-flow`
3. **Apply Patches**: Applies all 5 patches to fix agent tracking
4. **Service Restart**: Triggers supervisord restart of mcp-tcp-server

### Usage
```bash
# Applied automatically on container start
# Or manually:
bash /workspace/ext/multi-agent-docker/setup-workspace.sh

# To update claude-flow before patching:
UPDATE_CLAUDE_FLOW=true bash /workspace/ext/multi-agent-docker/setup-workspace.sh
```

## Testing the Fix

### 1. Verify Patches Applied
```bash
# Check if patches were applied
grep "Agent tracker verified" /usr/lib/node_modules/claude-flow/src/mcp/mcp-server.js

# Check logs for patch application
docker logs multi-agent-container 2>&1 | grep "Applying patch"
```

### 2. Test Agent Spawning
```bash
# Use the test script
bash /workspace/ext/test_mcp_interface_final.sh multi-agent-container 9500

# Or manually test
echo '{"jsonrpc":"2.0","id":"1","method":"tools/call","params":{"name":"swarm_init","arguments":{"topology":"mesh"}}}' | nc multi-agent-container 9500

echo '{"jsonrpc":"2.0","id":"2","method":"tools/call","params":{"name":"agent_spawn","arguments":{"type":"researcher"}}}' | nc multi-agent-container 9500

echo '{"jsonrpc":"2.0","id":"3","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | nc multi-agent-container 9500
```

### 3. Verify in VisionFlow
1. Open WebXR visualization
2. Click "Spawn Hive Mind"
3. Should see 3-4 real agents (not mock data)
4. Check browser console for agent IDs like `agent_1756564254868_pcgsya`

## Troubleshooting

### If agent_list still returns empty array:
1. Check if patches were applied: `docker logs multi-agent-container | grep patch`
2. Verify agent tracker initialization: Look for "Agent tracker verified" in logs
3. Check if agents are being spawned: Look for agent IDs in logs
4. Restart container to reapply patches: `docker restart multi-agent-container`

### If patches fail to apply:
1. Check claude-flow installation: `docker exec multi-agent-container which claude-flow`
2. Verify installation path: `docker exec multi-agent-container ls -la /usr/lib/node_modules/claude-flow/`
3. Manually apply patches: `docker exec -it multi-agent-container bash /workspace/ext/multi-agent-docker/setup-workspace.sh`

## Summary of Files Changed

### Docker Setup Files
- `/workspace/ext/multi-agent-docker/setup-workspace.sh` - Enhanced patch_mcp_server() function
- `/workspace/ext/multi-agent-docker/patches/01-fix-agent-tracker-init.patch` - Agent tracker initialization
- `/workspace/ext/multi-agent-docker/patches/02-fix-agent-spawn-tracking.patch` - Agent spawn tracking

### VisionFlow Files
- `/workspace/ext/src/handlers/bots_handler.rs` - Added auto-spawning after swarm_init
- `/workspace/ext/src/utils/mcp_connection.rs` - Added call_agent_spawn function

### Result
After these changes, the agent_list command returns real spawned agents instead of mock data, enabling proper agent tracking and visualization in VisionFlow.

## Version Compatibility
These patches are designed to work with:
- claude-flow@alpha (v2.0.0-alpha.101 or later)
- They check for installation before patching
- Patches are idempotent (safe to apply multiple times)
- Compatible with future upgrades via UPDATE_CLAUDE_FLOW environment variable

---
*Last Updated: 2025-08-30*
*Issue: Agent tracking not working due to mock data fallback*
*Solution: Comprehensive patching of installed claude-flow package*