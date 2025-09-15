# MCP Connection Architecture Summary

## Current Status
- **Date**: 2025-09-15
- **MCP Server**: ✅ Running on port 9500 in multi-agent-container
- **WebXR Connection**: ❌ Failing - "MCP session not initialized"

## Container Architecture

### We Are Here: multi-agent-container (172.18.0.3)
- MCP TCP Server running on port 9500
- Process: `/app/core-assets/scripts/mcp-tcp-server.js`
- Persistent MCP process (fixed from per-connection bug)
- Storage: SQLite at `/workspace/.swarm/memory.db`

### WebXR Runs In: logseq container (172.18.0.10)
- WebXR application with Rust backend
- Must connect to `multi-agent-container:9500`
- Currently failing to initialize MCP session

## Connection Issue

### Root Cause
The WebXR ClaudeFlowActor is not properly connecting because:
1. Host resolution issue (needs to use `multi-agent-container` not `localhost`)
2. MCP initialization sequence not completing

### Code Fix Applied
```rust
// In /workspace/ext/src/actors/claude_flow_actor.rs
let host = std::env::var("MCP_HOST")
    .unwrap_or_else(|_| {
        warn!("MCP_HOST not set, using multi-agent-container as default");
        "multi-agent-container".to_string()
    });
```

## MCP Server Details

### Working Test
```bash
# From multi-agent-container (here):
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' | nc localhost 9500
# ✅ Returns: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","serverInfo":{"name":"claude-flow","version":"2.0.0-alpha.101"}}}
```

### Required Fix for WebXR
```bash
# From logseq container (WebXR):
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05"}}' | nc multi-agent-container 9500
```

## Swarm Addressing Protocol

### Current Implementation
- Swarms get unique IDs: `swarm_[timestamp]_[random]`
- Agents get unique IDs: `agent_[timestamp]_[random]`
- Storage persists across connections (SQLite/memory)

### Needed Extensions
1. **Add swarmId to all tool calls** - Allow targeting specific swarms
2. **Support swarm context switching** - Maintain multiple swarm states
3. **Enable cross-swarm messaging** - Agent collaboration across swarms
4. **Implement swarm lifecycle hooks** - Creation, deletion, migration

### Example Addressing
```json
{
  "method": "tools/call",
  "params": {
    "name": "agent_task",
    "arguments": {
      "swarmId": "swarm_1757880683494_yl81sece5",
      "agentId": "agent_1757967065850_dv2zg7",
      "task": "analyze_code"
    }
  }
}
```

## Next Steps

1. **Rebuild WebXR** with the connection fix
2. **Set environment variable**: `MCP_HOST=multi-agent-container`
3. **Test connection** from logseq container
4. **Implement UpdateBotsGraph** message flow
5. **Add position assignment** for agent visualization

## Architecture Diagrams
Complete detailed diagrams available in `/workspace/ext/docs/diagrams.md` including:
- Docker container architecture
- MCP connection lifecycle
- Swarm addressing protocol
- Multi-agent system integration