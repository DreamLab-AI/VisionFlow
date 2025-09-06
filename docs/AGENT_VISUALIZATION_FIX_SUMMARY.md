# Agent Visualization Fix Summary

## Problem Statement
The agent visualization was showing 0 nodes despite agents being successfully created and parsed from the MCP server.

## Root Cause
The system had **two competing agent fetching mechanisms** that conflicted:

1. **BotsClient** (`/src/services/bots_client.rs`)
   - ✅ Creates fresh TCP connections per request
   - ✅ Successfully parses agent data from MCP responses
   - ❌ Did NOT send UpdateBotsGraph messages to the graph

2. **ClaudeFlowActor** (`/src/actors/claude_flow_actor_tcp.rs`)
   - ❌ Uses persistent TCP connection (incompatible with MCP server)
   - ❌ Gets "Connection lost" errors
   - ✅ Sends UpdateBotsGraph messages (but always with 0 agents)

## Architecture Conflict

```
Before Fix:
BotsClient → Parses agents → ❌ No graph update → Agents lost
ClaudeFlowActor → Connection error → Sends empty graph → 0 nodes displayed
```

## Solution: Consolidation

Merged functionality into a single, clean data flow:

```
After Fix:
MCP Server (port 9500)
     ↓ [Fresh TCP connection]
BotsClient (parses agents)
     ↓ [UpdateBotsGraph message]
GraphServiceActor (manages state)
     ↓ [WebSocket broadcast]
Frontend (renders nodes)
```

## Technical Implementation

### 1. Disable ClaudeFlowActor Polling
```rust
// src/actors/claude_flow_actor_tcp.rs
fn poll_agent_statuses(&mut self, _ctx: &mut Context<Self>) {
    // DISABLED: MCP server incompatible with persistent connections
    return;
}
```

### 2. Add Graph Updates to BotsClient
```rust
// src/services/bots_client.rs
// After successfully parsing agents:
if let Some(graph_addr) = graph_service_addr {
    graph_addr.do_send(UpdateBotsGraph {
        agents: update.agents.clone()
    });
}
```

## Files Changed

1. **Patch File**: `/workspace/ext/patches/consolidated_agent_graph_fix.patch`
2. **Setup Script**: `/workspace/ext/setup-workspace.sh` (includes patch creation)
3. **Documentation**: 
   - `/workspace/ext/docs/agent-visualization-architecture.md` (updated flow)
   - `/workspace/ext/task.md` (complete resolution details)

## MCP Server Configuration

The MCP server also required fixes to return real agents instead of mock data:

1. Disabled agent tracker (returns mock data)
2. Disabled databaseManager (looks for non-existent table)
3. Fixed memoryStore queries to not use namespace parameter
4. Agents stored in SQLite at `/workspace/.swarm/memory.db`

## Benefits of Consolidated Approach

- ✅ **Single source of truth** - Only BotsClient fetches agents
- ✅ **No conflicting updates** - One system, one data flow
- ✅ **Compatible with MCP** - Fresh connections match server behavior
- ✅ **Simpler architecture** - Easier to maintain and debug
- ✅ **Working visualization** - Agents now display correctly

## Testing

After applying the patch and rebuilding:

1. Spawn agents via UI
2. Check logs for: `BotsClient sending X agents to graph`
3. Verify WebSocket broadcasts: `Sending bots graph: X nodes, Y edges`
4. Confirm visualization shows nodes

## Future Improvements

1. Consider removing ClaudeFlowActor entirely if not needed
2. Add retry logic to BotsClient for resilience
3. Implement agent position persistence
4. Add real-time agent status updates