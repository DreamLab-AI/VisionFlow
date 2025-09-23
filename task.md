# MCP TCP Client Method Call Format Fix

## Issue Summary

The VisionFlow container's MCP TCP client is failing to communicate with the multi-agent-container's MCP server due to incorrect method call format. The client is receiving "Method not found" errors when attempting to query agent status.

## Current State

### Infrastructure
- **MCP Server**: Running in `multi-agent-container` (172.18.0.9:9500)
- **MCP Client**: Running in `visionflow_container` (172.18.0.10)
- **Network**: Both containers on `docker_ragflow` network (172.18.0.0/16)
- **Connection Status**: TCP connection established successfully

### Active Hive Mind
- **Swarm ID**: `swarm_1758656599853_033vowoj7`
- **Topology**: Mesh
- **Active Agents**: 3
  - coordinator-1 (active)
  - researcher-1 (active)
  - coder-1 (busy)

### Error Pattern
```
[2025-09-23T19:44:59Z ERROR webxr::actors::claude_flow_actor] MCP TCP query failed, falling back to JSON-RPC: Request failed after 6 attempts: MCP Error -32601: Method not found (data: None)
```

## Root Cause

The MCP client in `/mnt/mldata/githubs/AR-AI-Knowledge-Graph/src/utils/mcp_tcp_client.rs` is sending direct method calls like:

```json
{
  "jsonrpc": "2.0",
  "method": "agent_list",
  "params": {...},
  "id": 1
}
```

However, the MCP server only recognizes these top-level methods:
- `initialize`
- `tools/list`
- `tools/call`

All actual functionality must be accessed through `tools/call`.

## Required Fix

### 1. Update Method Call Format

The client needs to wrap all tool invocations in `tools/call`. For example:

**Current (incorrect):**
```rust
let result = self.send_request("agent_list", params).await?;
```

**Required (correct):**
```rust
let wrapped_params = json!({
    "name": "agent_list",
    "arguments": params
});
let result = self.send_request("tools/call", wrapped_params).await?;
```

### 2. Files to Modify

1. **`/mnt/mldata/githubs/AR-AI-Knowledge-Graph/src/utils/mcp_tcp_client.rs`**
   - Update `query_agent_list()` method (around line 210)
   - Update `query_swarm_status()` method
   - Update `query_server_info()` method
   - Any other methods that directly call MCP tools

2. **Response Parsing**
   - The response format will also change from direct results to wrapped content
   - Update response parsing to handle the `tools/call` response format

### 3. Testing Instructions

After implementing the fix, verify with these commands:

```bash
# Test from visionflow_container
docker exec visionflow_container sh -c 'echo "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"agent_list\",\"arguments\":{}}}" | nc -w 2 multi-agent-container 9500'

# Check logs for successful queries
tail -f /mnt/mldata/githubs/AR-AI-Knowledge-Graph/logs/rust-error.log
```

### 4. Expected Response Format

When correctly called through `tools/call`, responses will be:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"success\": true, \"agents\": [...]}"
      }
    ]
  }
}
```

## Implementation Priority

1. **High Priority**: Fix the `send_request` method to wrap calls in `tools/call`
2. **Medium Priority**: Update response parsing to handle nested content structure
3. **Low Priority**: Add proper error handling for MCP protocol violations

## Success Criteria

- No more "Method not found" errors in logs
- Agent status successfully displayed in VisionFlow UI
- TCP connection remains stable without fallback to JSON-RPC
- All MCP tool calls working correctly

## Additional Context

The MCP server provides 90+ tools for hive mind operations including:
- `swarm_init`, `agent_spawn`, `task_orchestrate`
- `neural_train`, `neural_patterns`, `neural_predict`
- `memory_usage`, `performance_report`, `bottleneck_analyze`
- And many more...

All of these must be accessed through the `tools/call` wrapper method.