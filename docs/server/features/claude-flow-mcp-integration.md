# Claude Flow MCP Integration

## Overview

The Claude Flow integration enables the Rust backend to communicate with claude-flow hive-mind AI agents through the Model Context Protocol (MCP). This integration supports spawning, managing, and visualising AI Multi Agents in real-time using TCP communication.

## Architecture

### TCP Communication (Current)

```
┌─────────────────────────┐
│   Rust Backend          │
│                         │
│  ClaudeFlowActorTcp     │
│  ├─ TcpTransport        │
│  │  └─ TCP Connection   │
│  └─ MCP Protocol        │
│                         │
└─────────────────────────┘
            │
            ▼ TCP:9500
  multi-agent-container
  (Claude Flow MCP Server)
```

### Key Components

#### ClaudeFlowActorTcp (`/src/actors/claude_flow_actor_tcp.rs`)
- Actor-based implementation for reliable MCP communication
- Connects to claude-flow MCP server via TCP
- Polls for agent updates every 5 seconds
- Gracefully falls back to mock data when MCP unavailable

#### TcpTransport (`/src/services/claude_flow/transport/tcp.rs`)
- Direct TCP connection management
- JSON-RPC 2.0 message format over TCP socket
- Configurable host and port settings
- Connection pooling for reliability

## MCP Protocol

### Initialization Flow

1. **Establish TCP Connection**:
   ```rust
   let stream = TcpStream::connect("multi-agent-container:9500").await?;
   ```

2. **Protocol Handshake**:
   ```json
   // Client initializes connection
   ← {"jsonrpc":"2.0","id":"init-1","method":"initialize","params":{...}}
   → {"jsonrpc":"2.0","id":"init-1","result":{...}}
   
   // Server announces capabilities
   → {"jsonrpc":"2.0","method":"server.initialized","params":{...}}
   ```

3. **Tool Invocation**:
   ```json
   // Call agent_list tool over TCP
   ← {"jsonrpc":"2.0","id":"1","method":"tools/call","params":{"name":"agent_list","arguments":{}}}
   → {"jsonrpc":"2.0","id":"1","result":{"content":[{"text":"..."}]}}
   ```

## Available MCP Tools

### Core Agent Management
- `multi-agent_init` - Initialize multi-agent with topology (mesh, hierarchical, ring, star)
- `agent_spawn` - Create specialized agents with capabilities
- `agent_list` - List active agents with status and metrics
- `agent_metrics` - Get detailed performance metrics
- `agent_command` - Send commands to specific agents

### Task Orchestration
- `task_orchestrate` - Coordinate complex multi-agent workflows
- `task_status` - Check task execution progress
- `task_results` - Retrieve completion results
- `pipeline_create` - Setup sequential processing pipelines

### Memory & Persistence
- `memory_usage` - Store and retrieve persistent data
- `memory_search` - Pattern-based memory search
- `memory_persist` - Cross-session data persistence
- `knowledge_graph` - Build and query knowledge graphs

### Neural & AI Capabilities
- `neural_train` - Train patterns with WASM SIMD acceleration
- `neural_predict` - Make predictions based on patterns
- `neural_patterns` - Analyze cognitive patterns
- `reasoning_chain` - Execute complex reasoning chains

### Performance & Monitoring
- `multi-agent_status` - Monitor overall multi-agent health
- `bottleneck_analyze` - Identify performance bottlenecks
- `performance_report` - Generate detailed reports
- `metrics_export` - Export metrics for external monitoring

## Agent Types

| Type | Role | Capabilities |
|------|------|-------------|
| Queen | Hive mind leader | Strategic coordination, multi-agent control |
| Coordinator | Task orchestration | Resource allocation, workflow management |
| Researcher | Information gathering | Web search, document analysis |
| Coder | Implementation | Code generation, refactoring |
| Analyst | Data analysis | Pattern recognition, insights |
| Architect | System design | Architecture planning, optimisation |
| Tester | Quality assurance | Test generation, validation |
| Reviewer | Code review | Quality checks, approval workflows |
| Optimizer | Performance tuning | Bottleneck analysis, optimisation |
| Documenter | Documentation | Technical writing, API docs |
| Monitor | System monitoring | Health checks, alerting |
| Specialist | Domain expert | Custom capabilities |

## Agent Data Model

```rust
pub struct BotsAgent {
    // Core properties
    pub id: String,
    pub agent_type: String,
    pub status: String,
    pub name: String,

    // Performance metrics
    pub cpu_usage: f32,
    pub health: f32,
    pub workload: f32,

    // Positioning (for visualisation)
    pub position: Vec3,
    pub velocity: Vec3,
    pub force: Vec3,
    pub connections: Vec<String>,

    // Hive-mind properties
    pub capabilities: Option<Vec<String>>,
    pub current_task: Option<String>,
    pub tasks_active: Option<u32>,
    pub tasks_completed: Option<u32>,
    pub success_rate: Option<f32>,
    pub tokens: Option<u64>,
    pub token_rate: Option<f32>,
    pub activity: Option<f32>,
    pub multi-agent_id: Option<String>,
    pub agent_mode: Option<String>,
    pub parent_queen_id: Option<String>,
}
```

## API Endpoints

### Initialize multi-agent
```http
POST /api/bots/initialize-multi-agent
Content-Type: application/json

{
  "topology": "hierarchical",
  "maxAgents": 8,
  "strategy": "balanced",
  "enableNeural": true,
  "agentTypes": ["coordinator", "researcher", "coder", "tester"],
  "customPrompt": "Build a REST API with authentication"
}
```

### Get Agent Status
```http
GET /api/bots/status
```

Returns lightweight agent status for real-time updates.

### Get Full Graph Data
```http
GET /api/bots/data
```

Returns complete graph with nodes, edges, and positioning.

## multi-agent Topologies

### Mesh
- Fully connected network
- All agents can communicate directly
- Best for collaborative tasks
- High communication overhead

### Hierarchical
- Tree structure with Queen at root
- Clear command chain
- Efficient for structured tasks
- Scalable to large multi-agents

### Ring
- Sequential processing pipeline
- Each agent communicates with neighbors
- Good for staged workflows
- Minimal communication overhead

### Star
- Central coordinator hub
- Peripheral worker agents
- Simple coordination model
- Single point of failure

## Error Handling

### Graceful Degradation
When MCP is unavailable:
1. Falls back to mock agent data
2. Continues visualisation with static data
3. Logs errors without crashing
4. Retries connection periodically

### Common Issues

1. **TCP Connection Failure**
   - Ensure `multi-agent-container` is running: `docker ps | grep multi-agent-container`
   - Check network connectivity: `telnet multi-agent-container 9500`
   - Verify port 9500 is accessible

2. **Communication Errors**
   - Monitor TCP stream for protocol errors
   - Check JSON-RPC message format
   - Verify tool names and parameters

3. **Performance Issues**
   - Adjust polling interval for agent updates
   - Limit maximum agents per multi-agent
   - Monitor process resource usage

## Configuration

### Environment Variables
```bash
# TCP configuration required
CLAUDE_FLOW_HOST=multi-agent-container  # Default: "multi-agent-container"
MCP_TCP_PORT=9500                       # Default: 9500
MCP_RECONNECT_ATTEMPTS=3                # Default: 3
MCP_RECONNECT_DELAY=1000                # Default: 1000ms
MCP_CONNECTION_TIMEOUT=30000            # Default: 30000ms
```

### Polling Configuration
```rust
// In ClaudeFlowActor
const POLL_INTERVAL: Duration = Duration::from_secs(5);
const RECONNECT_DELAY: Duration = Duration::from_secs(10);
```

## Performance Optimization

### Agent Update Batching
- Updates are batched in 5-second intervals
- Binary protocol reduces bandwidth by 85%
- Position updates use GPU-accelerated physics

### Connection Management
- TCP connection pooling
- Automatic reconnection on failure
- Configurable timeouts and retry logic

## Future Enhancements

1. **Connection Pool**: Reuse TCP connections for better performance
2. **Binary Protocol**: MessagePack for faster serialisation
3. **Streaming Updates**: Event-driven updates for real-time data
4. **Multi-multi-agent**: Support multiple independent multi-agents
5. **Persistence**: Save and restore multi-agent states

## Testing

### Manual Testing
```bash
# Initialize a test multi-agent
curl -X POST http://localhost:3001/api/bots/initialize-multi-agent \
  -H "Content-Type: application/json" \
  -d '{
    "topology": "mesh",
    "maxAgents": 5,
    "agentTypes": ["coordinator", "coder", "tester"]
  }'

# Check agent status
curl http://localhost:3001/api/bots/status | jq

# Get full graph data
curl http://localhost:3001/api/bots/data | jq
```

### Integration Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[actix_rt::test]
    async fn test_claude_flow_connection() {
        // Test implementation
    }
}
```

## References

- [MCP Specification](https://github.com/anthropics/mcp)
- [Claude Flow Documentation](https://github.com/Agentic-Insights/claude-flow)
- [Agent Control System](../../../agent-control-system/README.md)
- [Binary Protocol](../../api/binary-protocol.md)