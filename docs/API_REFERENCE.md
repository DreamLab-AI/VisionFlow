# API Reference - MCP TCP Client

## Core Components

### ClaudeFlowClient

The main client for interacting with the MCP server via TCP.

```rust
use crate::services::claude_flow::client::ClaudeFlowClient;
```

#### Methods

##### `new(transport: Box<dyn Transport>) -> Self`
Creates a new client with the specified transport.

##### `connect(&mut self) -> Result<()>`
Establishes connection to the MCP server.

##### `disconnect(&mut self) -> Result<()>`
Closes the connection gracefully.

##### `initialize(&mut self) -> Result<InitializeResult>`
Initializes the MCP session with protocol negotiation.

##### `call_tool(&mut self, name: &str, arguments: Value) -> Result<Value>`
Calls an MCP tool with the specified arguments.

##### `list_tools(&mut self) -> Result<Vec<Tool>>`
Lists all available MCP tools.

### ClaudeFlowClientBuilder

Builder pattern for creating configured clients.

```rust
use crate::services::claude_flow::client_builder::ClaudeFlowClientBuilder;
```

#### Methods

##### `new() -> Self`
Creates a new builder with default settings.

##### `with_tcp() -> Self`
Configures the client to use TCP transport (port 9500).

##### `host(host: &str) -> Self`
Sets the target host (default: "multi-agent-container").

##### `port(port: u16) -> Self`
Sets the target port (default: 9500 for TCP).

##### `with_retry(attempts: u32, delay: Duration) -> Self`
Configures retry behavior for connection failures.

##### `with_timeout(timeout: Duration) -> Self`
Sets the connection timeout.

##### `build() -> Result<ClaudeFlowClient>`
Builds and connects the client (async).

#### Example Usage

```rust
// Simple TCP client
let client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .build()
    .await?;

// Custom configuration
let client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .host("custom-host")
    .port(9600)
    .with_retry(5, Duration::from_secs(2))
    .with_timeout(Duration::from_secs(60))
    .build()
    .await?;
```

### TcpTransport

Low-level TCP transport implementation.

```rust
use crate::services::claude_flow::transport::tcp::TcpTransport;
```

#### Methods

##### `new(host: &str, port: u16) -> Self`
Creates a new TCP transport.

##### `new_with_defaults() -> Self`
Creates transport with environment variable defaults.

#### Configuration

Environment variables:
- `CLAUDE_FLOW_HOST`: Target host (default: "multi-agent-container")
- `MCP_TCP_PORT`: Target port (default: 9500)
- `MCP_RECONNECT_ATTEMPTS`: Retry attempts (default: 3)
- `MCP_RECONNECT_DELAY`: Delay between retries in ms (default: 1000)
- `MCP_CONNECTION_TIMEOUT`: Connection timeout in ms (default: 30000)

### ClaudeFlowActorTcp

Actix actor for MCP operations via TCP.

```rust
use crate::actors::claude_flow_actor_tcp::ClaudeFlowActorTcp;
```

#### Messages

##### `CallTool`
```rust
pub struct CallTool {
    pub name: String,
    pub arguments: Value,
}
```

##### `ListTools`
```rust
pub struct ListTools;
```

##### `GetConnectionInfo`
```rust
pub struct GetConnectionInfo;
```

#### Example Usage

```rust
use actix::prelude::*;

let actor = ClaudeFlowActorTcp::new().start();

// Call a tool
let result = actor.send(CallTool {
    name: "agent_spawn".to_string(),
    arguments: json!({
        "type": "researcher",
        "name": "test-agent"
    })
}).await??;

// List tools
let tools = actor.send(ListTools).await??;

// Get connection info
let info = actor.send(GetConnectionInfo).await?;
```

## MCP Tools

### Agent Management

#### `agent_spawn`
Spawns a new agent in the multi-agent.

**Parameters:**
- `type: String` - Agent type (researcher, coder, analyst, etc.)
- `name: String` - Agent name
- `capabilities: Vec<String>` - Optional capabilities

**Example:**
```rust
client.call_tool("agent_spawn", json!({
    "type": "researcher",
    "name": "data-analyzer",
    "capabilities": ["data-analysis", "statistics"]
})).await?;
```

#### `agent_list`
Lists all active agents.

**Parameters:** None

**Returns:** Array of agent information

#### `agent_metrics`
Gets performance metrics for agents.

**Parameters:**
- `agentId: String` - Optional specific agent ID

### Task Orchestration

#### `task_orchestrate`
Orchestrates a complex task across the multi-agent.

**Parameters:**
- `task: String` - Task description
- `strategy: String` - Execution strategy (parallel, sequential, adaptive)
- `priority: String` - Priority level (low, medium, high, critical)

**Example:**
```rust
client.call_tool("task_orchestrate", json!({
    "task": "Analyze market data and generate report",
    "strategy": "parallel",
    "priority": "high"
})).await?;
```

#### `task_status`
Checks the status of running tasks.

**Parameters:**
- `taskId: String` - Optional specific task ID

#### `task_results`
Retrieves results from completed tasks.

**Parameters:**
- `taskId: String` - Task ID to get results for

### multi-agent Management

#### `multi-agent_init`
Initializes a new multi-agent with specified topology.

**Parameters:**
- `topology: String` - Topology type (mesh, hierarchical, ring, star)
- `maxAgents: u32` - Maximum number of agents
- `strategy: String` - Distribution strategy

#### `multi-agent_status`
Gets current multi-agent status and statistics.

**Parameters:**
- `verbose: bool` - Include detailed information

#### `multi-agent_monitor`
Monitors multi-agent activity in real-time.

**Parameters:**
- `duration: u32` - Monitoring duration in seconds
- `interval: u32` - Update interval in seconds

### Memory Operations

#### `memory_usage`
Store or retrieve persistent memory.

**Parameters:**
- `action: String` - Action (store, retrieve, list, delete, search)
- `key: String` - Memory key
- `value: String` - Value (for store action)
- `namespace: String` - Namespace (default: "default")
- `ttl: u32` - Time to live in seconds

**Example:**
```rust
// Store data
client.call_tool("memory_usage", json!({
    "action": "store",
    "key": "analysis_results",
    "value": "...",
    "namespace": "research",
    "ttl": 3600
})).await?;

// Retrieve data
client.call_tool("memory_usage", json!({
    "action": "retrieve",
    "key": "analysis_results",
    "namespace": "research"
})).await?;
```

### Neural Operations

#### `neural_train`
Trains neural patterns with hardware acceleration.

**Parameters:**
- `pattern_type: String` - Pattern type (coordination, optimization, prediction)
- `training_data: String` - Training data
- `epochs: u32` - Number of epochs (default: 50)

#### `neural_predict`
Makes predictions using trained models.

**Parameters:**
- `modelId: String` - Model identifier
- `input: String` - Input data for prediction

#### `neural_patterns`
Analyzes cognitive patterns.

**Parameters:**
- `action: String` - Action (analyze, learn, predict)
- `pattern: String` - Pattern type

### Performance Monitoring

#### `performance_report`
Generates performance reports.

**Parameters:**
- `format: String` - Report format (summary, detailed, json)
- `timeframe: String` - Time frame (24h, 7d, 30d)

#### `bottleneck_analyze`
Identifies performance bottlenecks.

**Parameters:**
- `component: String` - Component to analyze
- `metrics: Vec<String>` - Metrics to evaluate

## Error Types

### ConnectorError

Main error type for MCP operations.

```rust
pub enum ConnectorError {
    ConnectionError(String),
    NotConnected,
    Timeout(String),
    SerializationError(String),
    ProtocolError(String),
    ToolError(String),
}
```

### Result Type

```rust
pub type Result<T> = std::result::Result<T, ConnectorError>;
```

## Types

### InitializeResult

```rust
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: Option<ServerInfo>,
}
```

### Tool

```rust
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Value,
}
```

### ConnectionInfo

```rust
pub struct ConnectionInfo {
    pub connected: bool,
    pub transport: String,
    pub host: String,
    pub port: u16,
    pub stats: ConnectionStats,
}
```

## Testing

### Unit Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tcp_connection() {
        let client = ClaudeFlowClientBuilder::new()
            .with_tcp()
            .build()
            .await;

        assert!(client.is_ok());

        let mut client = client.unwrap();
        let tools = client.list_tools().await;
        assert!(tools.is_ok());
        assert!(!tools.unwrap().is_empty());
    }
}
```

### Integration Test Example

```rust
#[tokio::test]
async fn test_agent_lifecycle() {
    let mut client = create_test_client().await;

    // Spawn agent
    let result = client.call_tool("agent_spawn", json!({
        "type": "researcher",
        "name": "test-agent"
    })).await.unwrap();

    let agent_id = result["agentId"].as_str().unwrap();

    // Check agent exists
    let agents = client.call_tool("agent_list", json!({})).await.unwrap();
    assert!(agents.as_array().unwrap().iter()
        .any(|a| a["id"] == agent_id));

    // Get metrics
    let metrics = client.call_tool("agent_metrics", json!({
        "agentId": agent_id
    })).await.unwrap();

    assert_eq!(metrics["status"], "active");
}
```

## Performance Considerations

### Connection Pooling

For high-throughput scenarios:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

struct ConnectionPool {
    connections: Arc<RwLock<Vec<ClaudeFlowClient>>>,
    max_size: usize,
}

impl ConnectionPool {
    async fn get(&self) -> ClaudeFlowClient {
        // Round-robin or least-recently-used selection
    }
}
```

### Batching Requests

```rust
// Batch multiple tool calls
let futures: Vec<_> = tools.iter()
    .map(|tool| client.call_tool(&tool.name, json!({})))
    .collect();

let results = futures::future::join_all(futures).await;
```

---

*API Reference Version: 1.0*
*MCP Protocol: 2024-11-05*
*Transport: TCP*