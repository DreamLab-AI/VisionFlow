# MCP Infrastructure

## Overview

The Multi-Agent Docker System provides multiple MCP (Model Context Protocol) servers for AI tool integration and telemetry streaming.

## MCP Server Types

### 1. TCP MCP Server (Primary)
**Port**: 9500
**Protocol**: TCP + JSON-RPC
**Purpose**: Primary MCP communication endpoint

**Features**:
- Persistent claude-flow MCP process
- Isolated database: `/workspace/.swarm/tcp-server-instance/.swarm/memory.db`
- Multiple concurrent client connections (max: 50)
- Session-aware telemetry routing
- Optional authentication via `TCP_AUTH_TOKEN`

**Use Cases**:
- External system telemetry queries
- Agent status monitoring
- Real-time metrics streaming
- Session-specific data retrieval

**Connection**:
```bash
# From host
nc localhost 9500

# From another container
nc multi-agent-container 9500
```

**Example Query**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "swarm_monitor",
    "arguments": {}
  }
}
```

### 2. WebSocket Bridge
**Port**: 3002
**Protocol**: WebSocket
**Purpose**: Real-time telemetry streaming to external visualization

**Features**:
- Bidirectional streaming
- Client subscription filtering
- Bandwidth limiting
- Compression support
- Session-based routing

**Use Cases**:
- GPU spring visualization updates
- Live dashboard feeds
- Real-time agent topology
- Performance metrics streaming

**Connection**:
```javascript
const ws = new WebSocket('ws://localhost:3002');

ws.on('message', (data) => {
  const telemetry = JSON.parse(data);
  // Process telemetry for visualization
});
```

### 3. Claude-Flow TCP Proxy
**Port**: 9502
**Protocol**: TCP
**Purpose**: Isolated claude-flow sessions for external projects

**Features**:
- Separate from main MCP server
- Session isolation per connection
- Health endpoint on port 9503

**Use Cases**:
- External project integrations
- Isolated AI workflows
- Development/testing environments

### 4. GUI MCP Servers

#### Playwright MCP
**Port**: 9879
**Purpose**: Browser automation and testing

**Tools**:
- `playwright_navigate`
- `playwright_screenshot`
- `playwright_click`
- `playwright_fill`
- `playwright_eval`

#### QGIS MCP
**Port**: 9877
**Purpose**: Geographic Information System operations

**Tools**:
- `qgis_load_layer`
- `qgis_process`
- `qgis_export`

#### Blender MCP
**Port**: 9876
**Purpose**: 3D modeling and rendering

**Tools**:
- `blender_create_mesh`
- `blender_render`
- `blender_export`

#### PBR Generator MCP
**Port**: 9878
**Purpose**: Physically-Based Rendering texture generation

**Tools**:
- `pbr_generate`
- `pbr_tessellate`

#### Web Summary MCP
**Port**: 9880
**Purpose**: Web page summarization

**Tools**:
- `web_summarize`
- `web_extract`

## MCP Server Registry

**Location**: `/app/core-assets/mcp.json`

Defines all available MCP servers for Claude Code to discover.

**Example Entry**:
```json
{
  "mcpServers": {
    "claude-flow": {
      "command": "node",
      "args": ["/app/scripts/stdio-to-tcp-bridge.js"],
      "type": "stdio",
      "env": {
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": "9500"
      }
    },
    "playwright": {
      "command": "node",
      "args": ["/app/core-assets/scripts/playwright-mcp-local.js"],
      "type": "stdio"
    }
  }
}
```

## stdio-to-TCP Bridge

**Purpose**: Allows Claude Code (which expects stdio) to connect to TCP MCP server

**Location**: `/app/scripts/stdio-to-tcp-bridge.js`

**How It Works**:
```
Claude Code (stdio)
    ↓ stdin/stdout pipes
stdio-to-tcp-bridge.js
    ↓ TCP connection
localhost:9500 (MCP TCP Server)
```

**Why Needed**:
- Claude Code spawns MCP servers via stdio
- Our TCP server is persistent (not spawned per-connection)
- Bridge adapts stdio interface to TCP protocol

## Session-Aware Telemetry

The MCP TCP server can route telemetry requests to specific sessions.

### Query by Session UUID

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "session_metrics",
    "arguments": {
      "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

### Available Session Tools

| Tool | Purpose |
|------|---------|
| `session_status` | Get current session state |
| `session_metrics` | Performance metrics for session |
| `session_agents` | List agents in session's swarm |
| `session_memory` | Query session's memory store |
| `session_logs` | Stream session log tail |

## Telemetry Data Structures

### Swarm Status
```json
{
  "swarm_id": "swarm-1759686180441-xetod3wsp",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "agents": [
    {
      "agent_id": "queen-1",
      "type": "queen",
      "role": "coordinator",
      "status": "active",
      "metrics": {
        "tasks_assigned": 12,
        "decisions_made": 45
      }
    },
    {
      "agent_id": "worker-1",
      "type": "worker",
      "role": "researcher",
      "status": "busy",
      "current_task": "data-gathering",
      "metrics": {
        "tasks_completed": 3,
        "tasks_pending": 2
      }
    }
  ],
  "metrics": {
    "total_tasks": 15,
    "completed_tasks": 8,
    "active_agents": 5,
    "memory_usage_mb": 45.2
  }
}
```

### Agent Metrics
```json
{
  "agent_id": "worker-2",
  "performance": {
    "tasks_per_hour": 12.5,
    "success_rate": 0.95,
    "average_task_duration_ms": 2400
  },
  "resources": {
    "cpu_usage": 0.23,
    "memory_mb": 128.5
  },
  "neural_state": {
    "activation_pattern": [0.8, 0.3, 0.1, 0.9],
    "decision_confidence": 0.87
  }
}
```

### Topology Updates
```json
{
  "type": "topology_change",
  "timestamp": "2025-10-05T20:00:00Z",
  "session_id": "550e8400-...",
  "changes": [
    {
      "action": "agent_spawned",
      "agent_id": "worker-5",
      "parent_id": "queen-1"
    },
    {
      "action": "connection_established",
      "from": "worker-3",
      "to": "worker-5",
      "connection_type": "data_sharing"
    }
  ]
}
```

## Authentication

### TCP MCP Server

Set environment variable:
```bash
TCP_AUTH_TOKEN=your-secret-token
```

Include in connection:
```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "auth",
  "params": {
    "token": "your-secret-token"
  }
}
```

### WebSocket Bridge

Set environment variable:
```bash
WS_AUTH_TOKEN=your-secret-token
```

Include in WebSocket connection:
```
ws://localhost:3002?token=your-secret-token
```

## Health Monitoring

### TCP Server Health
```bash
# Check if listening
ss -tlnp | grep 9500

# Test connection
echo '{"jsonrpc":"2.0","method":"ping","id":1}' | nc localhost 9500

# Check supervisor status
docker exec multi-agent-container supervisorctl status mcp-tcp-server
```

### WebSocket Bridge Health
```bash
# Check if listening
ss -tlnp | grep 3002

# Test connection
wscat -c ws://localhost:3002

# Check supervisor status
docker exec multi-agent-container supervisorctl status mcp-ws-bridge
```

## Bandwidth and Performance

### Expected Bandwidth

| Component | Bandwidth | Update Frequency |
|-----------|-----------|------------------|
| Session status | ~1 KB/s | 1 Hz |
| Agent metrics | ~5 KB/s per agent | 10 Hz |
| Topology updates | ~2 KB/s | On change |
| Neural patterns | ~10 KB/s | 20 Hz (if enabled) |
| Full swarm telemetry | ~50 KB/s | Variable |

### Optimization Strategies

1. **Subscription Filtering**: Clients subscribe to specific session UUIDs
2. **Compression**: gzip compression for large payloads
3. **Batching**: Aggregate multiple updates into single message
4. **Rate Limiting**: Client-specific bandwidth quotas
5. **Delta Updates**: Send only changed fields

## Logs

### TCP Server Logs
```bash
tail -f ./logs/mcp/tcp-server.log
tail -f ./logs/mcp/tcp-server-error.log
```

### WebSocket Bridge Logs
```bash
tail -f ./logs/mcp/ws-bridge.log
tail -f ./logs/mcp/ws-bridge-error.log
```

### MCP Connection Logs
```bash
# Security events
tail -f ./logs/mcp/security/*.log

# Individual MCP server logs
docker logs multi-agent-container 2>&1 | grep -i mcp
```

## Troubleshooting

### MCP Server Not Responding

```bash
# Check if process is running
docker exec multi-agent-container \
  supervisorctl status mcp-tcp-server

# Check port is listening
docker exec multi-agent-container \
  ss -tlnp | grep 9500

# Restart MCP server
docker exec multi-agent-container \
  supervisorctl restart mcp-tcp-server

# Check logs for errors
tail -50 ./logs/mcp/tcp-server-error.log
```

### WebSocket Connection Drops

```bash
# Check supervisor status
docker exec multi-agent-container \
  supervisorctl status mcp-ws-bridge

# Check for network issues
docker exec multi-agent-container netstat -s | grep -i error

# Restart WebSocket bridge
docker exec multi-agent-container \
  supervisorctl restart mcp-ws-bridge
```

### Telemetry Data Mismatch

```bash
# Verify session exists
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh get {UUID}

# Check if hive-mind process is running
docker exec multi-agent-container \
  ps aux | grep $UUID

# Query MCP directly
echo '{
  "jsonrpc":"2.0",
  "id":1,
  "method":"tools/call",
  "params": {
    "name":"session_status",
    "arguments":{"session_id":"'$UUID'"}
  }
}' | nc localhost 9500
```

## Integration Patterns

### Polling Pattern
```rust
let mut interval = tokio::time::interval(Duration::from_secs(5));

loop {
    interval.tick().await;

    let status = tcp_client.query_session_status(uuid).await?;

    if status == "completed" || status == "failed" {
        break;
    }

    // Update local state
    session_tracker.update(uuid, status).await;
}
```

### Streaming Pattern
```rust
let mut ws_stream = connect_websocket("ws://localhost:3002").await?;

while let Some(message) = ws_stream.next().await {
    let telemetry: TelemetryUpdate = serde_json::from_str(&message)?;

    // Route to visualization system
    spring_system.update_agents(telemetry.agents).await;
}
```

### Hybrid Pattern
```rust
// Control: Docker exec for task spawn
let uuid = create_session(task).await?;
start_session(uuid).await?;

// Data: TCP MCP for monitoring
let metrics = tcp_client.query_metrics(uuid).await?;

// Visualization: WebSocket for real-time updates
let ws = subscribe_to_session(uuid).await?;
```
