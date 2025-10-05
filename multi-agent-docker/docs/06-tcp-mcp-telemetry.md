# TCP/MCP Telemetry and Monitoring

## Overview

The TCP/MCP infrastructure provides real-time telemetry for hive-mind sessions, enabling external visualization systems (like GPU spring systems) to monitor agent activity.

## Architecture

```
Hive-Mind Sessions (Multiple)
    ↓ Spawn processes, create agents
MCP TCP Server (Port 9500)
    ↓ Queries running processes
Telemetry Data (JSON-RPC responses)
    ↓ Via WebSocket Bridge (Port 3002)
External Visualization System
    (GPU Spring System, Dashboards, etc.)
```

## Connecting to MCP TCP Server

### Direct TCP Connection

```bash
# Test connection
nc localhost 9500

# Send JSON-RPC request
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | nc localhost 9500
```

### From External Rust System

```rust
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

async fn connect_mcp() -> Result<TcpStream, std::io::Error> {
    let stream = TcpStream::connect("localhost:9500").await?;
    Ok(stream)
}

async fn send_request(
    stream: &mut TcpStream,
    request: serde_json::Value
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    // Send request
    let request_str = serde_json::to_string(&request)?;
    stream.write_all(request_str.as_bytes()).await?;
    stream.write_all(b"\n").await?;

    // Read response
    let mut buffer = vec![0u8; 8192];
    let n = stream.read(&mut buffer).await?;
    let response_str = String::from_utf8_lossy(&buffer[..n]);

    Ok(serde_json::from_str(&response_str)?)
}
```

## Available MCP Tools

### Session-Aware Tools

| Tool | Parameters | Returns | Use Case |
|------|------------|---------|----------|
| `session_status` | `{session_id}` | Session state | Check if session is running |
| `session_metrics` | `{session_id}` | Performance data | Monitor session resources |
| `session_agents` | `{session_id}` | Agent list | Get agent topology |
| `swarm_monitor` | `{swarm_id}` | Swarm details | Full swarm state |
| `agent_metrics` | `{agent_id}` | Agent performance | Individual agent data |

### Global Tools

| Tool | Returns | Use Case |
|------|---------|----------|
| `tools/list` | Available tools | Discover capabilities |
| `swarm_list` | All active swarms | System overview |
| `performance_summary` | System-wide metrics | Resource monitoring |

## Telemetry Query Examples

### Get Session Status

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "session_status",
    "arguments": {
      "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "running",
    "swarm_id": "swarm-1759686180441-xetod3wsp",
    "uptime_seconds": 125,
    "last_activity": "2025-10-05T20:15:30Z"
  }
}
```

### Get Agent Topology

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "session_agents",
    "arguments": {
      "session_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "agents": [
      {
        "agent_id": "queen-1",
        "type": "queen",
        "role": "coordinator",
        "status": "active",
        "position": {"x": 0, "y": 0, "z": 0},
        "connections": ["worker-1", "worker-2", "worker-3", "worker-4"]
      },
      {
        "agent_id": "worker-1",
        "type": "worker",
        "role": "researcher",
        "status": "busy",
        "position": {"x": 10, "y": 5, "z": 0},
        "connections": ["queen-1", "worker-2"],
        "current_task": "data-gathering"
      }
    ],
    "topology": {
      "total_agents": 5,
      "agent_types": {
        "queen": 1,
        "worker": 4
      },
      "connections": 8
    }
  }
}
```

### Stream All Swarms

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "swarm_list",
    "arguments": {}
  }
}
```

**Response**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "swarms": [
      {
        "swarm_id": "swarm-1759686180441-xetod3wsp",
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "status": "running",
        "agents": 5,
        "tasks": {"total": 15, "completed": 8, "active": 7}
      },
      {
        "swarm_id": "swarm-1759686342957-hmlp73urn",
        "session_id": "660f9511-f3ac-52e5-b827-557766551111",
        "status": "running",
        "agents": 4,
        "tasks": {"total": 3, "completed": 2, "active": 1}
      }
    ],
    "total_swarms": 2,
    "total_agents": 9
  }
}
```

## WebSocket Bridge for Real-Time Streaming

### Connecting

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:3002');

ws.on('open', () => {
  console.log('Connected to telemetry stream');

  // Subscribe to specific session
  ws.send(JSON.stringify({
    action: 'subscribe',
    session_id: '550e8400-e29b-41d4-a716-446655440000',
    filters: {
      include_performance: true,
      include_topology: true,
      update_frequency_hz: 10
    }
  }));
});

ws.on('message', (data) => {
  const telemetry = JSON.parse(data);
  console.log('Telemetry update:', telemetry);

  // Route to visualization system
  updateSpringSystem(telemetry);
});
```

### Subscription Filters

```json
{
  "action": "subscribe",
  "session_id": "550e8400-...",  // Optional: filter by session
  "swarm_id": "swarm-1759...",    // Optional: filter by swarm
  "filters": {
    "include_performance": true,   // Include CPU/memory metrics
    "include_topology": true,      // Include agent connections
    "include_neural": false,       // Exclude neural patterns (high bandwidth)
    "update_frequency_hz": 10,     // Max 10 updates per second
    "agent_types": ["queen", "worker"]  // Filter by agent type
  }
}
```

### Telemetry Message Format

```json
{
  "type": "telemetry_update",
  "timestamp": "2025-10-05T20:30:00Z",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "swarm_id": "swarm-1759686180441-xetod3wsp",
  "agents": [
    {
      "agent_id": "worker-1",
      "position": {"x": 10.5, "y": 5.2, "z": 0},
      "velocity": {"x": 0.1, "y": -0.05, "z": 0},
      "status": "busy",
      "performance": {
        "cpu_usage": 0.23,
        "memory_mb": 128.5,
        "tasks_completed": 5
      }
    }
  ],
  "topology_changes": [
    {
      "type": "connection_added",
      "from": "worker-1",
      "to": "worker-3",
      "strength": 0.8
    }
  ]
}
```

## Integration with GPU Spring System

### Rust Integration Example

```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::StreamExt;

pub struct SpringSystemUpdater {
    ws_url: String,
    session_id: String,
}

impl SpringSystemUpdater {
    pub async fn start_streaming(&self) -> Result<(), Box<dyn std::error::Error>> {
        let (ws_stream, _) = connect_async(&self.ws_url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Subscribe
        let subscribe = serde_json::json!({
            "action": "subscribe",
            "session_id": self.session_id,
            "filters": {
                "include_performance": true,
                "include_topology": true,
                "update_frequency_hz": 20  // 20 FPS for smooth visualization
            }
        });

        write.send(Message::Text(serde_json::to_string(&subscribe)?))
            .await?;

        // Stream updates
        while let Some(message) = read.next().await {
            let message = message?;

            if let Message::Text(text) = message {
                let telemetry: TelemetryUpdate = serde_json::from_str(&text)?;

                // Update GPU spring system
                self.update_spring_positions(telemetry.agents).await?;
                self.update_spring_connections(telemetry.topology_changes).await?;
            }
        }

        Ok(())
    }

    async fn update_spring_positions(
        &self,
        agents: Vec<AgentTelemetry>
    ) -> Result<(), Box<dyn std::error::Error>> {
        for agent in agents {
            // Update GPU buffer with agent position
            let spring_node = SpringNode {
                id: agent.agent_id,
                position: agent.position,
                velocity: agent.velocity,
                mass: match agent.type_name.as_str() {
                    "queen" => 2.0,
                    "worker" => 1.0,
                    _ => 1.0,
                },
                color: agent_type_color(&agent.type_name),
            };

            gpu_spring_system.update_node(spring_node).await?;
        }

        Ok(())
    }
}
```

## Telemetry Data Bandwidth

### Expected Bandwidth Per Session

| Data Type | Size | Frequency | Bandwidth |
|-----------|------|-----------|-----------|
| Agent positions | ~100 bytes/agent | 10-20 Hz | ~5 KB/s per agent |
| Topology updates | ~50 bytes/change | On change | ~1 KB/s |
| Performance metrics | ~200 bytes | 1 Hz | ~200 bytes/s |
| Neural patterns | ~500 bytes | 20 Hz | ~10 KB/s (optional) |

**Total per session**: ~20-50 KB/s depending on agent count and features enabled

### Bandwidth Optimization

```json
// Minimal subscription (positions only)
{
  "filters": {
    "include_performance": false,
    "include_topology": false,
    "include_neural": false,
    "update_frequency_hz": 10
  }
}
// Bandwidth: ~5 KB/s

// Full telemetry
{
  "filters": {
    "include_performance": true,
    "include_topology": true,
    "include_neural": true,
    "update_frequency_hz": 20
  }
}
// Bandwidth: ~50 KB/s
```

## Monitoring Multiple Sessions

```rust
pub struct MultiSessionMonitor {
    sessions: HashMap<String, SessionHandle>,
    mcp_client: Arc<McpTelemetryClient>,
}

impl MultiSessionMonitor {
    pub async fn monitor_all(&mut self) -> Result<(), Error> {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            for (uuid, handle) in &self.sessions {
                // Query each session
                let metrics = self.mcp_client
                    .query_session_metrics(uuid)
                    .await?;

                // Update dashboard
                self.update_dashboard(uuid, metrics).await;

                // Check for completion
                let status = handle.get_status().await?;
                if matches!(status.as_str(), "completed" | "failed") {
                    self.sessions.remove(uuid);
                }
            }
        }
    }
}
```

## Troubleshooting Telemetry

### No Telemetry Data

```bash
# Check MCP server is running
docker exec multi-agent-container supervisorctl status mcp-tcp-server

# Test connection
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | nc localhost 9500

# Check if session is actually running
docker exec multi-agent-container \
  /app/scripts/hive-session-manager.sh status $UUID

# Check MCP logs
tail -50 logs/mcp/tcp-server.log
```

### Stale Data

```bash
# MCP server might be cached, restart it
docker exec multi-agent-container \
  supervisorctl restart mcp-tcp-server

# Or the session might be paused/dead
docker exec multi-agent-container ps aux | grep $UUID
```

### High Bandwidth Usage

```bash
# Reduce update frequency
# Change subscription filters to:
{
  "update_frequency_hz": 5,  // Lower from 20
  "include_neural": false    // Disable high-bandwidth data
}

# Monitor bandwidth
docker exec multi-agent-container \
  iftop -i eth0 -f "port 3002"
```

## Advanced: Custom Telemetry Queries

You can extend the MCP server with custom tools. See `/core-assets/scripts/mcp-tcp-server.js` for implementation details.

Example custom tool:
```javascript
// In mcp-tcp-server.js
async function customSessionQuery(sessionId) {
  // Query session database directly
  const db = await openSessionDatabase(sessionId);
  const customData = await db.query('SELECT * FROM custom_metrics');
  return customData;
}
```

Register as MCP tool and query via:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "custom_session_query",
    "arguments": {"session_id": "..."}
  }
}
```
