# MCP TCP Server Documentation - THIS Container

## Overview
The MCP (Model Context Protocol) server is running **IN THIS CONTAINER** on port 9500 and provides a complete agent management system via JSON-RPC.

## Server Details
- **Port**: 9500 (TCP)
- **Protocol**: JSON-RPC 2.0 over TCP
- **MCP Version**: 2024-11-05
- **Server**: claude-flow v2.0.0-alpha.101
- **Process**: `/app/core-assets/scripts/mcp-tcp-server.js`

## Available MCP Tools (85 total)

### Core Agent Management
1. **swarm_init** - Initialize swarm with topology (hierarchical, mesh, ring, star)
2. **agent_spawn** - Create specialized AI agents
3. **agent_list** - List all active agents in swarm
4. **task_orchestrate** - Orchestrate complex task workflows
5. **swarm_status** - Monitor swarm health and performance

### Agent Types Supported
- coordinator / task-orchestrator
- researcher
- coder
- tester
- reviewer
- analyst / code-analyzer
- optimizer / perf-analyzer
- documenter / api-docs
- monitor / performance-benchmarker
- specialist / system-architect
- architect

### Memory & State Management
- **memory_usage** - Store/retrieve persistent memory with TTL
- **memory_search** - Search memory with patterns
- **memory_persist** - Cross-session persistence
- **memory_backup** - Backup memory stores
- **memory_restore** - Restore from backups

### Performance & Analytics
- **performance_report** - Generate performance reports
- **bottleneck_analyze** - Identify performance bottlenecks
- **token_usage** - Analyze token consumption
- **benchmark_run** - Performance benchmarks
- **metrics_collect** - Collect system metrics

### Neural & AI Features
- **neural_status** - Check neural network status
- **neural_train** - Train neural patterns with WASM SIMD
- **neural_patterns** - Analyze cognitive patterns
- **neural_predict** - Make AI predictions
- **neural_compress** - Compress neural models

## Response Format

### Agent Spawn Response
```json
{
  "success": true,
  "agentId": "agent_1757967065850_dv2zg7",
  "type": "researcher",
  "name": "test-researcher-1757967065850",
  "status": "active",
  "capabilities": [],
  "persisted": false,
  "timestamp": "2025-09-15T20:11:05.851Z"
}
```

### Agent List Response
```json
{
  "success": true,
  "swarmId": "swarm_1757880683494_yl81sece5",
  "agents": [
    {
      "id": "agent_1757967065850_dv2zg7",
      "swarmId": "swarm_1757880683494_yl81sece5",
      "name": "test-researcher-1757967065850",
      "type": "researcher",
      "status": "active",
      "capabilities": [],
      "metadata": "{...}",
      "createdAt": "2025-09-15T20:11:05.851Z",
      "lastActive": "2025-09-15T20:11:05.851Z"
    }
  ],
  "count": 1,
  "timestamp": "2025-09-15T20:11:05.852Z"
}
```

## Connection Protocol

### 1. Initialize Connection
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "tools": { "listChanged": true }
    },
    "clientInfo": {
      "name": "webxr-client",
      "version": "1.0.0"
    }
  }
}
```

### 2. Call Tool
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "agent_spawn",
    "arguments": {
      "type": "researcher",
      "name": "my-agent"
    }
  }
}
```

## Key Findings

### ✅ What's Working
1. **Agent Creation**: Successfully creates agents with unique IDs
2. **Agent Persistence**: Agents stored in memory/SQLite
3. **Swarm Management**: Automatic swarm ID assignment
4. **Real-time Updates**: Immediate response after operations
5. **Multiple Agent Types**: All 11+ agent types functional

### ⚠️ Current Issues
1. **Mock Data**: Initial agent_list returns mock agents until real ones spawned
2. **Response Repetition**: Server sends multiple duplicate responses (needs debouncing)
3. **No Position Data**: Agents created without x,y,z coordinates
4. **No Graph Integration**: Missing UpdateBotsGraph message to WebXR

## Integration with WebXR

### Current Data Flow
```
MCP Server (port 9500)
    ↓ [agent_spawn]
Agent Created in Memory
    ↓ [agent_list]
Returns Agent Data
    ❌ [Missing: UpdateBotsGraph]
WebXR Graph (not updated)
```

### Required Data Flow
```
MCP Server (port 9500)
    ↓ [agent_spawn]
Agent Created
    ↓ [BotsClient fetches]
UpdateBotsGraph message
    ↓ [GraphServiceActor]
WebXR Visualization
```

## Test Scripts Location
- `/tmp/test_mcp.js` - Test agent listing
- `/tmp/test_spawn.js` - Test agent spawning

## Memory Storage
- **SQLite**: `/workspace/.swarm/memory.db` (when available)
- **Fallback**: In-memory store (when SQLite unavailable)
- **Session ID**: `session-cf-[timestamp]-[random]`

## Network Configuration
- This container: Part of docker_ragflow network
- Can connect to multi-agent-container (172.18.0.3) if needed
- WebXR container at 172.18.0.10

## Next Steps for WebXR Integration

1. **Fix BotsClient Integration**
   - Make BotsClient call MCP server on port 9500
   - Parse real agent data from responses
   - Send UpdateBotsGraph messages

2. **Add Position Assignment**
   - Generate x,y,z coordinates for spawned agents
   - Store positions in agent metadata
   - Include in graph updates

3. **Remove Mock Data**
   - Disable mock agent generation
   - Always use real MCP data
   - Handle empty agent lists gracefully

4. **Add Real-time Updates**
   - Poll MCP server periodically
   - Update agent status changes
   - Reflect in visualization