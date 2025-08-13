# MCP Integration Architecture

VisionFlow integrates with Claude Flow's Model Context Protocol (MCP) through the EnhancedClaudeFlowActor to orchestrate and visualise AI Multi Agents in real-time.

## Overview

The MCP integration enables VisionFlow to:
- Connect directly to Claude Flow via TCP (backend-only)
- Visualize agent interactions through REST API to frontend
- Monitor agent performance and resource usage
- Coordinate multi-agent collaboration in parallel graphs

## Architecture

```mermaid
graph LR
    subgraph "VisionFlow Backend"
        ECFA[ClaudeFlowActorTcp] -->|Direct TCP| MCP
        GSA[GraphServiceActor] --> ECFA
        API[REST API] --> ECFA
        PGC[ParallelGraphCoordinator] --> GSA
    end

    subgraph "Claude Flow Container"
        MCP[MCP Server :9500] --> CF[Claude Flow]
        CF --> AGENTS[Agent Pool]
        AGENTS --> LLM[LLM APIs]
    end

    subgraph "Frontend"
        UI[React UI] -->|REST /api/bots/* ✅| API
        UI -->|WebSocket Binary Protocol ✅| GSA
        UI -.->|❌ NO DIRECT MCP ACCESS ✅| MCP
    end
```

## Key Design Principles (VERIFIED IMPLEMENTATION ✅)

### 1. Backend-Only MCP Connection

The frontend **never** connects directly to MCP. All MCP communication flows through the Rust backend:

**✅ IMPLEMENTATION STATUS**: This architecture is now correctly implemented.
- Frontend MCP WebSocket code removed
- BotsClient re-enabled in backend
- REST-only communication from frontend verified

```rust
// ClaudeFlowActorTcp handles all MCP communication
pub struct ClaudeFlowActorTcp {
    mcp_client: Option<MCPClient>,
    tcp_connection: Option<TcpStream>,
    graph_service_addr: Addr<GraphServiceActor>,
}
```

### 2. TCP Integration

The ClaudeFlowActorTcp maintains a direct TCP connection:

```rust
pub struct ClaudeFlowActorTcp {
    _client: ClaudeFlowClient,
    graph_service_addr: Addr<GraphServiceActor>,
    is_connected: bool,
    tcp_connection: Option<TcpStream>,
    agent_cache: HashMap<String, AgentStatus>,
    message_flow_history: Vec<MessageFlowEvent>,
    pending_additions: Vec<AgentStatus>,
    pending_removals: Vec<String>,
    pending_updates: Vec<AgentUpdate>,
}

// Direct TCP connection to Claude Flow
async fn establish_mcp_connection() -> Result<TcpStream> {
    let addr = "multi-agent-container:9500";
    let stream = TcpStream::connect(addr).await?;
    Ok(stream)
}
```

### 3. Real-time Telemetry

Agent telemetry streams at 10Hz for smooth visualisation:

```rust
// Telemetry subscription
let subscribe_req = json!({
    "jsonrpc": "2.0",
    "method": "telemetry.subscribe",
    "params": {
        "events": ["agent.*", "message.*", "metrics.*"],
        "interval_ms": 100  // 10Hz updates
    }
});
```

## MCP Message Protocol

### Request Format

```json
{
    "jsonrpc": "2.0",
    "id": "uuid-v4",
    "method": "agent.spawn",
    "params": {
        "type": "coordinator",
        "task": "Build a REST API",
        "config": {
            "max_tokens": 4000,
            "temperature": 0.7
        }
    }
}
```

### Response Format

```json
{
    "jsonrpc": "2.0",
    "id": "uuid-v4",
    "result": {
        "agent_id": "agent-001",
        "status": "active",
        "session_id": "session-xyz"
    }
}
```

### Telemetry Events

```json
{
    "type": "agent.status",
    "data": {
        "agent_id": "agent-001",
        "status": "active",
        "cpu_usage": 45.2,
        "memory_usage": 128.5,
        "tasks_active": 3,
        "tokens_used": 1523
    },
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## Core MCP Methods

### Agent Management

| Method | Description | Parameters |
|--------|-------------|------------|
| `agent.spawn` | Create new agent | `type`, `task`, `config` |
| `agent.list` | List all agents | `filter`, `limit` |
| `agent.terminate` | Stop an agent | `agent_id` |
| `agent.status` | Get agent status | `agent_id` |

### multi-agent Orchestration

| Method | Description | Parameters |
|--------|-------------|------------|
| `multi-agent.initialise` | Create Multi Agent | `topology`, `agents`, `task` |
| `multi-agent.status` | Get multi-agent status | `multi-agent_id` |
| `multi-agent.dissolve` | Terminate multi-agent | `multi-agent_id` |

### Task Management

| Method | Description | Parameters |
|--------|-------------|------------|
| `task.assign` | Assign task to agent | `agent_id`, `task` |
| `task.status` | Get task progress | `task_id` |
| `task.cancel` | Cancel running task | `task_id` |

## Data Flow

### 1. multi-agent Initialization

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant API as REST API
    participant ECFA as EnhancedClaudeFlowActor
    participant MCP as MCP Server
    participant CF as Claude Flow

    UI->>API: POST /api/bots/multi-agent/init
    API->>ECFA: initializeMultiAgent message
    ECFA->>MCP: Direct WebSocket: multi-agent.initialize
    MCP->>CF: Create agents
    CF-->>MCP: Agent IDs
    MCP-->>ECFA: multi-agent created via WebSocket
    ECFA->>ECFA: Update agent_cache
    ECFA->>API: Push to GraphServiceActor
    API-->>UI: 200 OK + agent data
```

### 2. Agent Data Flow

```mermaid
sequenceDiagram
    participant MCP as MCP Server
    participant ECFA as EnhancedClaudeFlowActor
    participant GSA as GraphServiceActor
    participant GPU as GPUComputeActor
    participant UI as Frontend
    participant WS as WebSocket Binary

    loop Every 100ms
        MCP->>ECFA: Agent telemetry via WebSocket
        ECFA->>ECFA: Update agent_cache
        ECFA->>GSA: UpdateBotsGraph
        GSA->>GPU: Unified kernel with DualGraph mode
        GPU-->>GSA: Updated positions
        GSA->>WS: Binary protocol position updates
    end

    loop Every 10s
        UI->>API: GET /api/bots/agents
        API->>ECFA: Request cached agents
        ECFA-->>API: Return agent_cache
        API-->>UI: JSON agent data
    end
```

## Actor Integration

### EnhancedClaudeFlowActor

Manages direct MCP WebSocket connection and differential updates:

```rust
impl EnhancedClaudeFlowActor {
    // Handle incoming WebSocket messages
    fn handle_websocket_message(&mut self, message: WsMessage) {
        if let Ok(event) = serde_json::from_slice::<MCPEvent>(&message.into_data()) {
            match event.event_type.as_str() {
                "agent.spawned" => {
                    self.pending_additions.push(event.data.into());
                }
                "agent.terminated" => {
                    self.pending_removals.push(event.data.id);
                }
                "agent.status" => {
                    self.pending_updates.push(event.data.into());
                }
                "message.flow" => {
                    self.message_flow_history.push(event.data.into());
                }
                _ => {}
            }
        }

        // Apply differential updates
        self.apply_pending_changes();
    }

    // Push changes to parallel graph system
    fn apply_pending_changes(&mut self) {
        if self.has_changes() {
            let graph_data = self.build_graph_data();
            self.graph_service_addr.do_send(UpdateBotsGraph {
                agents: graph_data.agents,
                edges: graph_data.edges,
                communications: self.message_flow_history.clone(),
            });
        }
    }
}
```

### GraphServiceActor

Processes agent graph updates:

```rust
impl Handler<UpdateBotsGraph> for GraphServiceActor {
    fn handle(&mut self, msg: UpdateBotsGraph) {
        // Update agent graph buffer
        self.agent_nodes = msg.nodes;
        self.agent_edges = msg.edges;

        // Mark as agent nodes (set bit 31)
        for node in &mut self.agent_nodes {
            node.id |= 0x80000000;
        }

        // Send to GPU for physics
        if let Some(gpu) = &self.gpu_compute_addr {
            gpu.do_send(UpdateAgentGraph {
                nodes: self.agent_nodes.clone(),
                edges: self.agent_edges.clone(),
            });
        }
    }
}
```

## Configuration

MCP connection settings in environment variables:

```bash
# Claude Flow host (Docker service name)
CLAUDE_FLOW_HOST=multi-agent-container

# MCP TCP port
MCP_TCP_PORT=9500

# Enable MCP integration
ENABLE_MCP=true

# Connection settings
MCP_RECONNECT_ATTEMPTS=3
MCP_RECONNECT_DELAY=1000
MCP_CONNECTION_TIMEOUT=30000
MCP_MAX_AGENTS=50
```

## Error Handling

### Connection Failures

```rust
impl ClaudeFlowActor {
    async fn ensure_connection(&mut self) -> Result<()> {
        if !self.is_connected {
            match Self::connect_to_claude_flow().await {
                Ok(stream) => {
                    self.tcp_connection = Some(stream);
                    self.is_connected = true;
                    Ok(())
                }
                Err(e) => {
                    warn!("MCP connection failed: {}", e);
                    // Return empty state, no mock data
                    Err(e)
                }
            }
        } else {
            Ok(())
        }
    }
}
```

### Graceful Degradation (IMPLEMENTED ✅)

When MCP is unavailable:
1. ✅ Frontend shows "MCP Disconnected" status
2. ✅ Agent graph remains empty (no mock data)
3. ✅ Knowledge graph continues functioning
4. ✅ Reconnection attempts every 30 seconds
5. ✅ Real system metrics provided by agent_visualization_processor.rs

## Security Considerations

### Network Isolation

- MCP server only accessible within Docker network
- No external MCP exposure
- Frontend isolated from direct MCP access

### Authentication

- Optional API key authentication for MCP
- Session-based frontend authentication
- Rate limiting on API endpoints

### Data Validation

```rust
fn validate_mcp_response(response: &Value) -> Result<()> {
    // Validate JSON-RPC format
    if !response.get("jsonrpc").is_some() {
        return Err("Invalid JSON-RPC response");
    }

    // Check for errors
    if let Some(error) = response.get("error") {
        return Err(format!("MCP error: {:?}", error));
    }

    Ok(())
}
```

## Performance

### Metrics

- **Latency**: < 50ms MCP round-trip
- **Throughput**: 10,000+ telemetry events/sec
- **Agent Capacity**: 50+ concurrent agents
- **Update Rate**: 10Hz telemetry, 60 FPS rendering

### Optimizations

1. **Differential Updates**: Only send changed data
2. **Binary Protocol**: 28-byte position updates
3. **Connection Pooling**: Reuse WebSocket connections
4. **Batch Processing**: Aggregate telemetry events

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No agents visible | MCP not connected | Check Docker logs, verify ports |
| Laggy updates | Network latency | Reduce telemetry frequency |
| Connection drops | Container restart | Enable auto-reconnect |
| High CPU usage | Too many agents | Limit max agents in config |

### Debug Logging

Enable MCP debug logs:

```bash
RUST_LOG=webxr::services::claude_flow=debug cargo run
```

## See Also

- [System Overview](system-overview.md) - Overall architecture
- [Dual Graph](dual-graph.md) - Agent graph visualisation
- [API Reference](../api/rest.md) - REST endpoints for MCP control