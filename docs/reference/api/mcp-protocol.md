# MCP (Model Context Protocol) Integration

## Overview

VisionFlow integrates with Claude's Model Context Protocol (MCP) to enable multi-agent system coordination. The MCP server runs in a separate container and provides a TCP-based JSON-RPC interface for agent management, task orchestration, and swarm coordination.

**MCP Server Endpoint**: `multi-agent-container:9500` (TCP)

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   VisionFlow/Rust   │────▶│  MCP TCP Server     │
│   (logseq container)│ TCP │ (multi-agent)       │
└─────────────────────┘     └─────────────────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │   Agent Swarms      │
                            │ ┌─────┬─────┬─────┐│
                            │ │Agent│Agent│Agent││
                            │ └─────┴─────┴─────┘│
                            └─────────────────────┘
```

## Connection Setup

### Environment Configuration

```bash
# Container networking
MCP_HOST=multi-agent-container    # Hostname within Docker network
MCP_TCP_PORT=9500                # TCP port for MCP server
MCP_TRANSPORT=tcp                # Transport protocol

# Agent configuration
MAX_AGENTS=20                    # Maximum concurrent agents
AGENT_TIMEOUT=300                # Agent operation timeout (seconds)
```

### Establishing Connection

```rust
// Rust client connection
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LinesCodec};

async fn connect_to_mcp() -> Result<Framed<TcpStream, LinesCodec>> {
    let host = std::env::var("MCP_HOST")
        .unwrap_or_else(|_| "multi-agent-container".to_string());
    let port = std::env::var("MCP_TCP_PORT")
        .unwrap_or_else(|_| "9500".to_string());
    
    let stream = TcpStream::connect(format!("{}:{}", host, port)).await?;
    Ok(Framed::new(stream, LinesCodec::new()))
}
```

### Initialization Handshake

```json
// Client → Server
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "clientInfo": {
            "name": "visionflow",
            "version": "0.1.0"
        }
    }
}

// Server → Client
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "protocolVersion": "2024-11-05",
        "serverInfo": {
            "name": "claude-flow",
            "version": "2.0.0-alpha.101"
        },
        "capabilities": {
            "tools": ["agent_spawn", "agent_task", "swarm_init"],
            "resources": ["memory_bank", "task_queue"]
        }
    }
}
```

## Core Methods

### Agent Management

#### List Available Tools
Returns all available MCP tools for agent operations.

```json
{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "tools": [
            {
                "name": "agent_spawn",
                "description": "Spawn a new agent in the swarm",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "agentType": {"type": "string"},
                        "swarmId": {"type": "string"},
                        "config": {"type": "object"}
                    }
                }
            }
        ]
    }
}
```

#### Spawn Agent
Creates a new agent within a swarm.

```json
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "agent_spawn",
        "arguments": {
            "agentType": "coder",
            "swarmId": "swarm_123",
            "config": {
                "model": "claude-3-opus",
                "temperature": 0.7,
                "capabilities": ["python", "rust", "typescript"]
            }
        }
    }
}
```

#### Task Orchestration
Submits a task to the agent swarm.

```json
{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
        "name": "task_orchestrate",
        "arguments": {
            "swarmId": "swarm_123",
            "task": {
                "description": "Analyze security vulnerabilities in auth module",
                "priority": "high",
                "strategy": "adaptive",
                "timeout": 300
            }
        }
    }
}
```

### Swarm Operations

#### Initialize Swarm
Creates a new agent swarm with specified topology.

```json
{
    "jsonrpc": "2.0",
    "id": 5,
    "method": "tools/call",
    "params": {
        "name": "swarm_init",
        "arguments": {
            "topology": "mesh",
            "initialAgents": [
                {"type": "coordinator", "name": "SwarmLead"},
                {"type": "researcher", "name": "InfoGatherer"},
                {"type": "coder", "name": "CodeWriter"}
            ],
            "config": {
                "maxAgents": 10,
                "consensusThreshold": 0.7
            }
        }
    }
}
```

#### Get Swarm Status
Retrieves current swarm state and agent information.

```json
{
    "jsonrpc": "2.0",
    "id": 6,
    "method": "tools/call",
    "params": {
        "name": "swarm_status",
        "arguments": {
            "swarmId": "swarm_123"
        }
    }
}
```

**Response:**
```json
{
    "jsonrpc": "2.0",
    "id": 6,
    "result": {
        "swarmId": "swarm_123",
        "topology": "mesh",
        "agents": [
            {
                "id": "agent_001",
                "type": "coordinator",
                "status": "active",
                "health": 100,
                "currentTask": null
            },
            {
                "id": "agent_002",
                "type": "coder",
                "status": "busy",
                "health": 85,
                "currentTask": "task_456"
            }
        ],
        "metrics": {
            "tasksCompleted": 45,
            "avgCompletionTime": 120,
            "successRate": 0.92
        }
    }
}
```

### Memory and State

#### Store Memory
Persists data to the swarm's memory bank.

```json
{
    "jsonrpc": "2.0",
    "id": 7,
    "method": "tools/call",
    "params": {
        "name": "memory_store",
        "arguments": {
            "swarmId": "swarm_123",
            "key": "project_context",
            "value": {
                "language": "rust",
                "framework": "actix-web",
                "dependencies": ["tokio", "serde", "uuid"]
            },
            "ttl": 3600
        }
    }
}
```

#### Retrieve Memory
Fetches data from the swarm's memory bank.

```json
{
    "jsonrpc": "2.0",
    "id": 8,
    "method": "tools/call",
    "params": {
        "name": "memory_retrieve",
        "arguments": {
            "swarmId": "swarm_123",
            "key": "project_context"
        }
    }
}
```

## Swarm Addressing Protocol

### Unique Identifiers

All swarms and agents receive globally unique identifiers:

```
swarm_[timestamp]_[random]   // e.g., swarm_1757880683494_yl81sece5
agent_[timestamp]_[random]   // e.g., agent_1757967065850_dv2zg7
task_[timestamp]_[random]    // e.g., task_1757967065850_abc123
```

### Hierarchical Addressing

Agents can be addressed through their swarm:

```json
{
    "method": "tools/call",
    "params": {
        "name": "agent_message",
        "arguments": {
            "swarmId": "swarm_123",      // Target swarm
            "agentId": "agent_456",       // Target agent
            "message": {
                "type": "command",
                "action": "analyze_code",
                "params": {"file": "main.rs"}
            }
        }
    }
}
```

## Error Handling

### MCP Error Codes

| Code | Message | Description |
|------|---------|-------------|
| -32700 | Parse error | Invalid JSON |
| -32600 | Invalid request | Invalid JSON-RPC |
| -32601 | Method not found | Unknown method |
| -32602 | Invalid params | Invalid parameters |
| -32603 | Internal error | Server error |
| -32000 | Server error | MCP-specific error |
| -32001 | Agent error | Agent operation failed |
| -32002 | Swarm error | Swarm operation failed |
| -32003 | Resource error | Resource unavailable |

### Error Response Format

```json
{
    "jsonrpc": "2.0",
    "id": 10,
    "error": {
        "code": -32001,
        "message": "Agent operation failed",
        "data": {
            "agentId": "agent_123",
            "reason": "timeout",
            "duration": 305
        }
    }
}
```

## Connection Management

### Retry Logic

```rust
async fn call_with_retry<T>(method: &str, params: T) -> Result<Value> {
    let mut attempts = 0;
    let max_attempts = 3;
    let mut delay = Duration::from_millis(100);
    
    loop {
        match make_mcp_call(method, &params).await {
            Ok(result) => return Ok(result),
            Err(e) if attempts < max_attempts => {
                attempts += 1;
                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
            }
            Err(e) => return Err(e),
        }
    }
}
```

### Connection Pool

```rust
pub struct MCPConnectionPool {
    connections: Vec<MCPConnection>,
    max_connections: usize,
}

impl MCPConnectionPool {
    pub async fn get_connection(&self) -> Result<MCPConnection> {
        // Round-robin or least-loaded selection
        // Return available connection
    }
}
```

## Performance Optimization

### Batch Operations

```json
{
    "jsonrpc": "2.0",
    "id": 11,
    "method": "batch",
    "params": [
        {
            "method": "tools/call",
            "params": {
                "name": "agent_status",
                "arguments": {"agentId": "agent_001"}
            }
        },
        {
            "method": "tools/call",
            "params": {
                "name": "agent_status",
                "arguments": {"agentId": "agent_002"}
            }
        }
    ]
}
```

### Keep-Alive

```json
{
    "jsonrpc": "2.0",
    "id": 12,
    "method": "ping",
    "params": {}
}
```

## Security Considerations

### Authentication

MCP connections can be secured with:
1. **Network isolation**: Docker network segmentation
2. **TLS encryption**: For production deployments
3. **Token-based auth**: JWT or API key authentication

### Rate Limiting

Default MCP rate limits:
- 100 requests per second per connection
- 1000 concurrent operations per swarm
- 10MB maximum message size

## Integration Example

### Complete Rust Integration

```rust
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_util::codec::{Framed, LinesCodec};
use futures::{SinkExt, StreamExt};

pub struct MCPClient {
    connection: Framed<TcpStream, LinesCodec>,
    request_id: u64,
}

impl MCPClient {
    pub async fn connect() -> Result<Self> {
        let stream = TcpStream::connect("multi-agent-container:9500").await?;
        let mut connection = Framed::new(stream, LinesCodec::new());
        
        // Initialize connection
        let init_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05"
            }
        });
        
        connection.send(init_request.to_string()).await?;
        
        Ok(Self {
            connection,
            request_id: 2,
        })
    }
    
    pub async fn spawn_agent(&mut self, agent_type: &str, swarm_id: &str) -> Result<String> {
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": "tools/call",
            "params": {
                "name": "agent_spawn",
                "arguments": {
                    "agentType": agent_type,
                    "swarmId": swarm_id
                }
            }
        });
        
        self.request_id += 1;
        self.connection.send(request.to_string()).await?;
        
        // Wait for response
        if let Some(Ok(response)) = self.connection.next().await {
            let parsed: Value = serde_json::from_str(&response)?;
            if let Some(result) = parsed.get("result") {
                return Ok(result["agentId"].as_str().unwrap().to_string());
            }
        }
        
        Err("Failed to spawn agent".into())
    }
}
```

## Monitoring and Debugging

### MCP Health Check

```bash
# Test MCP connection from host
echo '{"jsonrpc":"2.0","id":1,"method":"ping","params":{}}' | nc multi-agent-container 9500

# Expected response:
{"jsonrpc":"2.0","id":1,"result":{"status":"ok","timestamp":1706006400000}}
```

### Debug Logging

Enable MCP debug logging:
```bash
MCP_LOG_LEVEL=debug
MCP_LOG_FILE=/app/logs/mcp.log
```

## Related Documentation

- [REST API](rest-api.md) - Agent management via REST
- [WebSocket API](websocket-api.md) - Real-time updates
- [Agent Architecture](../../agents/README.md) - Agent system design
- [Multi-Agent Docker](../../../multi-agent-docker/README.md) - Container setup

---

**[← Binary Protocol](binary-protocol.md)** | **[Back to API Index →](index.md)**