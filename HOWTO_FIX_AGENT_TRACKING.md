# How to Fix Agent Tracking in MCP/VisionFlow Integration

## Problem Summary
The MCP server's `agent_list` command was returning empty arrays even after agents were successfully spawned via `agent_spawn`. This was caused by two issues:
1. The agent tracker wasn't properly tracking spawned agents
2. The MCP TCP server was using the global npm package instead of the local modified code

## Solution Overview
1. Fix the MCP server to properly track agents
2. Configure the Docker container to use local claude-flow code
3. Modify VisionFlow to automatically spawn agents after swarm creation

## Detailed Changes

### 1. MCP TCP Server Configuration
**File**: `/workspace/ext/multi-agent-docker/core-assets/scripts/mcp-tcp-server.js`

**Change**: Modified the spawn command to check for local claude-flow first

```diff
- const mcp = spawn('npx', ['claude-flow@alpha', 'mcp', 'start', '--stdio', '--file', '/workspace/.mcp.json'], {
-   stdio: ['pipe', 'pipe', 'pipe'],
-   cwd: '/workspace',
-   env: { ...process.env, CLAUDE_FLOW_DIRECT_MODE: 'true' }
- });

+ // Use local claude-flow from ext if available, otherwise use global
+ let mcpCommand, mcpArgs;
+ const localMcpPath = '/workspace/ext/claude-flow/src/mcp/mcp-server.js';
+ 
+ if (fs.existsSync(localMcpPath)) {
+   this.log('info', `Using LOCAL claude-flow from ${localMcpPath}`);
+   mcpCommand = 'node';
+   mcpArgs = [localMcpPath, '--stdio', '--file', '/workspace/.mcp.json'];
+ } else {
+   this.log('info', 'Using GLOBAL claude-flow@alpha package');
+   mcpCommand = 'npx';
+   mcpArgs = ['claude-flow@alpha', 'mcp', 'start', '--stdio', '--file', '/workspace/.mcp.json'];
+ }
+ 
+ const mcp = spawn(mcpCommand, mcpArgs, {
+   stdio: ['pipe', 'pipe', 'pipe'],
+   cwd: '/workspace',
+   env: { ...process.env, CLAUDE_FLOW_DIRECT_MODE: 'true' }
+ });
```

Apply this change to both TCP handler (line ~47) and Unix socket handler (line ~171).

### 2. MCP Server Agent Tracking Fixes
**File**: `/workspace/ext/claude-flow/src/mcp/mcp-server.js`

#### Fix 1: Enhanced agent tracker initialization (line ~15)
```diff
- // Initialize agent tracker
- await import('./implementations/agent-tracker.js').catch(() => {
-   try {
-     require('./implementations/agent-tracker');
-   } catch (e) {
-     console.log('Agent tracker not loaded');
-   }
- });

+ // Initialize agent tracker - CRITICAL for agent management
+ try {
+   await import('./implementations/agent-tracker.js');
+   console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Agent tracker loaded successfully`);
+ } catch (importError) {
+   console.error(`[${new Date().toISOString()}] WARN [claude-flow-mcp] ES module import failed, trying require:`, importError.message);
+   try {
+     require('./implementations/agent-tracker');
+     console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Agent tracker loaded via require`);
+   } catch (requireError) {
+     console.error(`[${new Date().toISOString()}] ERROR [claude-flow-mcp] Agent tracker failed to load:`, requireError.message);
+   }
+ }
+ 
+ // Verify agent tracker is available
+ if (global.agentTracker) {
+   console.error(`[${new Date().toISOString()}] INFO [claude-flow-mcp] Agent tracker verified and ready`);
+ } else {
+   console.error(`[${new Date().toISOString()}] ERROR [claude-flow-mcp] Agent tracker NOT available - agent tracking will not work!`);
+ }
```

#### Fix 2: Ensure swarmId consistency in agent_spawn (line ~1140)
```diff
  case 'agent_spawn':
    const agentId = `agent_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    const resolvedType = resolveLegacyAgentType(args.type);
+   const activeSwarmId = args.swarmId || (await this.getActiveSwarmId());
    const agentData = {
      id: agentId,
-     swarmId: args.swarmId || (await this.getActiveSwarmId()),
+     swarmId: activeSwarmId,
      name: args.name || `${resolvedType}-${Date.now()}`,
      type: resolvedType,
      status: 'active',
      // ... rest of agentData
    };
    
    // Track spawned agent
    if (global.agentTracker) {
      global.agentTracker.trackAgent(agentId, {
        ...agentData,
+       swarmId: activeSwarmId,
        capabilities: args.capabilities || [],
      });
+     console.error(
+       `[${new Date().toISOString()}] INFO [claude-flow-mcp] Agent tracked: ${agentId} in swarm: ${activeSwarmId}`,
+     );
+   } else {
+     console.error(
+       `[${new Date().toISOString()}] WARN [claude-flow-mcp] Agent tracker not available, agent ${agentId} not tracked in memory`,
+     );
    }
```

#### Fix 3: Remove mock data fallback from agent_list (line ~1507)
```diff
- // Fallback mock response
- return {
-   success: true,
-   swarmId: args.swarmId || 'mock-swarm',
-   agents: [
-     { id: 'agent-1', name: 'coordinator-1', type: 'coordinator', status: 'active', capabilities: [] },
-     { id: 'agent-2', name: 'researcher-1', type: 'researcher', status: 'active', capabilities: [] },
-     { id: 'agent-3', name: 'coder-1', type: 'coder', status: 'busy', capabilities: [] },
-   ],
-   count: 3,
-   timestamp: new Date().toISOString(),
- };

+ // No mock fallback - return empty list if no real agents
+ console.error(
+   `[${new Date().toISOString()}] WARN [claude-flow-mcp] No agent tracker available, returning empty list`,
+ );
+ return {
+   success: true,
+   swarmId: listSwarmId || 'default',
+   agents: [],
+   count: 0,
+   timestamp: new Date().toISOString(),
+ };
```

### 3. VisionFlow Agent Spawning
**File**: `/workspace/ext/src/utils/mcp_connection.rs`

Add new function to spawn agents:
```rust
/// Simplified function to spawn an agent
pub async fn call_agent_spawn(
    host: &str,
    port: &str,
    agent_type: &str,
) -> Result<Value, Box<dyn std::error::Error>> {
    let pool = MCPConnectionPool::new(host.to_string(), port.to_string());
    
    info!("Spawning agent of type: {}", agent_type);
    
    let params = json!({
        "name": "agent_spawn",
        "arguments": {
            "type": agent_type
        }
    });
    
    pool.execute_command("agent_spawn", "tools/call", params).await
}
```

**File**: `/workspace/ext/src/handlers/bots_handler.rs`

Modify swarm initialization to spawn agents (after line ~1066):
```rust
// Now spawn some agents based on the topology
use crate::utils::mcp_connection::call_agent_spawn;

let agent_types = match request.topology.as_str() {
    "hierarchical" => vec!["coordinator", "analyst", "optimizer"],
    "mesh" => vec!["coordinator", "researcher", "coder", "analyst"],
    "star" => vec!["coordinator", "optimizer", "documenter"],
    _ => vec!["coordinator", "analyst", "optimizer"],
};

info!("Spawning {} agents for {} topology", agent_types.len(), request.topology);

for agent_type in agent_types {
    match call_agent_spawn(&claude_flow_host, &claude_flow_port, agent_type).await {
        Ok(spawn_result) => {
            info!("✅ Spawned {} agent: {:?}", agent_type, spawn_result);
        }
        Err(e) => {
            warn!("Failed to spawn {} agent: {}", agent_type, e);
        }
    }
}
```

## Testing

Use the test script `/workspace/ext/simple_agent_test.sh`:
```bash
#!/bin/bash
# Test MCP connection
echo '{"jsonrpc":"2.0","id":"test","method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"test","version":"1.0"}}}' | nc -w 2 localhost 9500

# Create swarm
echo '{"jsonrpc":"2.0","id":"s1","method":"tools/call","params":{"name":"swarm_init","arguments":{"topology":"mesh"}}}' | nc -w 2 localhost 9500

# Spawn agent
echo '{"jsonrpc":"2.0","id":"a1","method":"tools/call","params":{"name":"agent_spawn","arguments":{"type":"coordinator"}}}' | nc -w 2 localhost 9500

# List agents (should show the spawned agent)
echo '{"jsonrpc":"2.0","id":"l1","method":"tools/call","params":{"name":"agent_list","arguments":{}}}' | nc -w 2 localhost 9500
```

## Deployment

1. **Commit changes to multi-agent-docker**:
   - The modified `mcp-tcp-server.js` is in the Docker configuration
   - This will be used when the container is rebuilt

2. **Restart the container**:
   - The supervisord will start the modified mcp-tcp-server
   - It will detect and use the local claude-flow code

3. **Restart VisionFlow**:
   - The Rust backend will use the updated handlers
   - Agents will be spawned automatically when swarms are created

## Verification

After restart, check logs:
```bash
# Check if local claude-flow is being used
docker logs multi-agent-container 2>&1 | grep "Using LOCAL claude-flow"

# Check if agent tracker is initialized
docker logs multi-agent-container 2>&1 | grep "Agent tracker verified and ready"

# Test agent tracking
bash /workspace/ext/simple_agent_test.sh
```

## Summary

The fix involves three layers:
1. **Docker layer**: Configure TCP server to use local code
2. **MCP layer**: Fix agent tracking in claude-flow
3. **Application layer**: Auto-spawn agents in VisionFlow

This ensures agents are properly created, tracked, and returned by `agent_list`.