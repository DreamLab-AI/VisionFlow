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

# Connection pooling
MCP_POOL_SIZE=10                 # Maximum concurrent connections
MCP_POOL_MIN=2                   # Minimum pool connections
MCP_RETRY_ATTEMPTS=3             # Retry attempts on failure
MCP_RETRY_DELAY=100              # Initial retry delay (ms)
MCP_KEEPALIVE=30                 # Keep-alive interval (seconds)
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
Creates a new agent within a swarm with real TCP connection and spawning.

```json
{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
        "name": "agent_spawn",
        "arguments": {
            "agentType": "coder",
            "swarmId": "swarm_1757880683494_yl81sece5",
            "config": {
                "model": "claude-3-opus",
                "temperature": 0.7,
                "capabilities": ["python", "rust", "typescript"],
                "maxTokens": 4096,
                "timeout": 300,
                "retryAttempts": 3
            },
            "resources": {
                "cpuLimit": "2000m",
                "memoryLimit": "4Gi",
                "gpuAccess": true
            }
        }
    }
}
```

**Real Agent Spawning Response:**
```json
{
    "jsonrpc": "2.0",
    "id": 3,
    "result": {
        "agentId": "agent_1757967065850_dv2zg7",
        "swarmId": "swarm_1757880683494_yl81sece5",
        "status": "spawning",
        "estimatedReadyTime": "2025-01-22T10:00:30Z",
        "tcpEndpoint": "multi-agent-container:9500",
        "capabilities": ["python", "rust", "typescript"],
        "resources": {
            "allocated": true,
            "cpu": "2000m",
            "memory": "4Gi",
            "gpuDevice": 0
        },
        "connectionPool": {
            "poolId": "pool_123",
            "connections": 3,
            "healthCheck": "passing"
        },
        "initialisationMetrics": {
            "spawnTime": 1247,
            "modelLoadTime": 892,
            "memoryAllocated": "3.2 GB"
        }
    }
}
```

#### Task Orchestration
Submits a task to the agent swarm with real execution and coordination.

```json
{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
        "name": "task_orchestrate",
        "arguments": {
            "swarmId": "swarm_1757880683494_yl81sece5",
            "task": {
                "description": "Analyze security vulnerabilities in auth module",
                "priority": "high",
                "strategy": "adaptive",
                "timeout": 300,
                "requiredCapabilities": ["security", "code", "review"],
                "parallelism": 3,
                "consensusRequired": true
            },
            "execution": {
                "mode": "distributed",
                "retryPolicy": "exponential_backoff",
                "resultAggregation": "majority_vote",
                "failureThreshold": 0.2
            }
        }
    }
}
```

**Real Task Orchestration Response:**
```json
{
    "jsonrpc": "2.0",
    "id": 4,
    "result": {
        "taskId": "task_1757967065850_abc123",
        "swarmId": "swarm_1757880683494_yl81sece5",
        "status": "orchestrating",
        "assignedAgents": [
            {
                "agentId": "agent_1757967065850_dv2zg7",
                "role": "coordinator",
                "capabilities": ["security", "code"]
            },
            {
                "agentId": "agent_1757967065851_def456",
                "role": "worker",
                "capabilities": ["code", "review"]
            },
            {
                "agentId": "agent_1757967065852_ghi789",
                "role": "validator",
                "capabilities": ["security", "review"]
            }
        ],
        "orchestrationPlan": {
            "phases": [
                {
                    "name": "analysis",
                    "agents": ["agent_1757967065851_def456"],
                    "estimatedDuration": 120
                },
                {
                    "name": "security_scan",
                    "agents": ["agent_1757967065850_dv2zg7"],
                    "estimatedDuration": 180,
                    "dependencies": ["analysis"]
                },
                {
                    "name": "validation",
                    "agents": ["agent_1757967065852_ghi789"],
                    "estimatedDuration": 60,
                    "dependencies": ["security_scan"]
                }
            ]
        },
        "coordination": {
            "consensusThreshold": 0.7,
            "votingMechanism": "weighted",
            "conflictResolution": "expert_priority"
        },
        "estimatedCompletion": "2025-01-22T10:05:00Z",
        "mcpConnections": {
            "active": 3,
            "poolUtilization": 0.6
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
                "action": "analyse_code",
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
use tokio::sync::{Semaphore, RwLock};
use std::collections::VecDeque;

pub struct MCPConnectionPool {
    connections: RwLock<VecDeque<MCPConnection>>,
    semaphore: Semaphore,
    max_connections: usize,
    min_connections: usize,
    retry_attempts: u32,
    retry_delay: Duration,
}

impl MCPConnectionPool {
    pub fn new(max_connections: usize, min_connections: usize) -> Self {
        Self {
            connections: RwLock::new(VecDeque::new()),
            semaphore: Semaphore::new(max_connections),
            max_connections,
            min_connections,
            retry_attempts: 3,
            retry_delay: Duration::from_millis(100),
        }
    }

    pub async fn get_connection(&self) -> Result<MCPConnection> {
        // Acquire permit from semaphore (connection limit)
        let _permit = self.semaphore.acquire().await?;

        // Try to get existing connection from pool
        {
            let mut pool = self.connections.write().await;
            if let Some(conn) = pool.pop_front() {
                if conn.is_healthy().await {
                    return Ok(conn);
                }
            }
        }

        // Create new connection with retry logic
        self.create_connection_with_retry().await
    }

    async fn create_connection_with_retry(&self) -> Result<MCPConnection> {
        let mut attempts = 0;
        let mut delay = self.retry_delay;

        loop {
            match MCPConnection::connect("multi-agent-container:9500").await {
                Ok(conn) => return Ok(conn),
                Err(e) if attempts < self.retry_attempts => {
                    attempts += 1;
                    log::warn!("MCP connection attempt {} failed: {}", attempts, e);
                    tokio::time::sleep(delay).await;
                    delay *= 2; // Exponential backoff
                }
                Err(e) => return Err(e),
            }
        }
    }

    pub async fn return_connection(&self, conn: MCPConnection) {
        if conn.is_healthy().await {
            let mut pool = self.connections.write().await;
            if pool.len() < self.max_connections {
                pool.push_back(conn);
            }
        }
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

- [REST API](../reference/api/rest-api.md) - Agent management via REST
- [WebSocket API](../reference/api/websocket-api.md) - Real-time updates
- [Agent Architecture](../../agents/README.md) - Agent system design
- [Multi-Agent Docker](../../../multi-agent-docker/README.md) - Container setup

---

**[← Binary Protocol](../reference/api/binary-protocol.md)** | **[Back to API Index →](README.md)**