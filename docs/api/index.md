# Claude Flow MCP API Reference - Definitive Guide

*Version: 2.0 | MCP Protocol: 2024-11-05 | Transport: TCP Only*

## Overview

The Claude Flow MCP (Model Context Protocol) API provides comprehensive integration capabilities for multi-agent systems, task orchestration, neural processing, and performance optimisation. This reference covers all components, methods, error handling, and integration patterns.

---

## Table of Contents

1. [Core Components](#core-components)
2. [Client Integration](#client-integration)
3. [Transport Layer](#transport-layer)
4. [Actor System](#actor-system)
5. [MCP Tools](#mcp-tools)
6. [Error Handling](#error-handling)
7. [Type Definitions](#type-definitions)
8. [Testing Patterns](#testing-patterns)
9. [Performance Optimization](#performance-optimisation)
10. [Environment Configuration](#environment-configuration)

---

## Core Components

### ClaudeFlowClient

The primary interface for all MCP operations via TCP transport.

```rust
use crate::services::claude_flow::client::ClaudeFlowClient;

impl ClaudeFlowClient {
    /// Creates a new client with specified transport
    pub fn new(transport: Box<dyn Transport>) -> Self;
    
    /// Establishes connection to MCP server
    pub async fn connect(&mut self) -> Result<()>;
    
    /// Closes connection gracefully with proper cleanup
    pub async fn disconnect(&mut self) -> Result<()>;
    
    /// Initializes MCP session with protocol negotiation
    pub async fn initialize(&mut self) -> Result<InitializeResult>;
    
    /// Calls MCP tool with arguments and error handling
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value>;
    
    /// Lists all available MCP tools with descriptions
    pub async fn list_tools(&mut self) -> Result<Vec<Tool>>;
    
    /// Gets current connection status and statistics
    pub async fn get_connection_info(&self) -> Result<ConnectionInfo>;
    
    /// Performs health check with timeout
    pub async fn health_check(&mut self, timeout: Duration) -> Result<bool>;
}
```

**Key Features:**
- Automatic reconnection with exponential backoff
- Connection pooling for high throughput
- Built-in health monitoring
- Protocol version negotiation
- Graceful error recovery

---

## Client Integration

### ClaudeFlowClientBuilder

Fluent builder pattern for creating configured clients with advanced options.

```rust
use crate::services::claude_flow::client_builder::ClaudeFlowClientBuilder;

impl ClaudeFlowClientBuilder {
    /// Creates builder with default configuration
    pub fn new() -> Self;
    
    /// Configures TCP transport (default port 9500)
    pub fn with_tcp(self) -> Self;
    
    /// Sets target host (supports DNS and IP)
    pub fn host(self, host: &str) -> Self;
    
    /// Sets target port with validation
    pub fn port(self, port: u16) -> Self;
    
    /// Configures retry policy with exponential backoff
    pub fn with_retry(self, attempts: u32, delay: Duration) -> Self;
    
    /// Sets connection timeout with fallback
    pub fn with_timeout(self, timeout: Duration) -> Self;
    
    /// Enables connection pooling
    pub fn with_pool(self, pool_size: usize) -> Self;
    
    /// Sets authentication credentials
    pub fn with_auth(self, token: String) -> Self;
    
    /// Builds and connects client asynchronously
    pub async fn build(self) -> Result<ClaudeFlowClient>;
}
```

### Usage Examples

#### Basic TCP Client
```rust
let client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .build()
    .await?;
```

#### Production Configuration
```rust
let client = ClaudeFlowClientBuilder::new()
    .with_tcp()
    .host("claude-flow.example.com")
    .port(9500)
    .with_retry(5, Duration::from_secs(2))
    .with_timeout(Duration::from_secs(60))
    .with_pool(10)
    .with_auth("your-api-token".to_string())
    .build()
    .await?;
```

---

## Transport Layer

### TcpTransport

The only supported transport layer - provides low-level TCP transport with connection management.

```rust
use crate::services::claude_flow::transport::tcp::TcpTransport;

impl TcpTransport {
    /// Creates TCP transport with host and port
    pub fn new(host: &str, port: u16) -> Self;
    
    /// Creates transport with environment defaults
    pub fn new_with_defaults() -> Self;
    
    /// Establishes TCP connection with timeout
    pub async fn connect(&mut self) -> Result<()>;
    
    /// Sends data with framing protocol
    pub async fn send(&mut self, data: &[u8]) -> Result<()>;
    
    /// Receives data with proper parsing
    pub async fn receive(&mut self) -> Result<Vec<u8>>;
    
    /// Closes connection gracefully
    pub async fn close(&mut self) -> Result<()>;
}
```

### Environment Configuration

```bash
# TCP Transport (ONLY supported transport)
export CLAUDE_FLOW_HOST="multi-agent-container"
export MCP_TCP_PORT=9500
export MCP_RECONNECT_ATTEMPTS=3
export MCP_RECONNECT_DELAY=1000        # milliseconds
export MCP_CONNECTION_TIMEOUT=30000    # milliseconds
export MCP_POOL_SIZE=5
```

---

## Actor System

### ClaudeFlowActorTcp

Actix-based actor for managing TCP-based MCP operations with supervision.

```rust
use crate::actors::claude_flow_actor_tcp::ClaudeFlowActorTcp;
use actix::prelude::*;

#[derive(Message)]
#[rtype(result = "Result<Value>")]
pub struct CallTool {
    pub name: String,
    pub arguments: Value,
    pub timeout: Option<Duration>,
}

#[derive(Message)]  
#[rtype(result = "Result<Vec<Tool>>")]
pub struct ListTools {
    pub category: Option<String>,
}

#[derive(Message)]
#[rtype(result = "Result<ConnectionInfo>")]
pub struct GetConnectionInfo;

impl Actor for ClaudeFlowActorTcp {
    type Context = Context<Self>;
    
    fn started(&mut self, ctx: &mut Self::Context) {
        // Setup heartbeat and health monitoring
        ctx.run_interval(Duration::from_secs(30), |act, ctx| {
            act.perform_health_check(ctx);
        });
    }
    
    fn stopping(&mut self, _: &mut Self::Context) -> Running {
        // Graceful shutdown
        Running::Stop
    }
}
```

### Actor Usage Patterns

```rust
use actix::prelude::*;

let actor = ClaudeFlowActorTcp::new().start();

// Call tool with timeout
let result = actor.send(CallTool {
    name: "agent_spawn".to_string(),
    arguments: json!({
        "type": "researcher",
        "name": "analysis-agent",
        "capabilities": ["data-analysis", "report-generation"]
    }),
    timeout: Some(Duration::from_secs(30)),
}).await??;

// List available tools by category
let tools = actor.send(ListTools {
    category: Some("agent".to_string()),
}).await??;

// Get connection info
let info = actor.send(GetConnectionInfo).await?;
```

---

## MCP Tools

Comprehensive tool catalog with parameters, examples, and error handling.

### Agent Management

#### `agent_spawn`
Creates and initializes new agents in the swarm.

**Parameters:**
- `type: String` - Agent type (researcher, coder, analyst, optimizer, coordinator)
- `name: String` - Unique agent identifier  
- `capabilities: Vec<String>` - Agent capabilities (optional)
- `resources: Object` - Resource allocation (optional)
- `config: Object` - Agent-specific configuration (optional)

**Response:**
```json
{
  "agentId": "agent-uuid-12345",
  "status": "active",
  "type": "researcher", 
  "capabilities": ["data-analysis", "statistics"],
  "resources": {
    "cpu": "0.5",
    "memory": "512MB"
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

**Example:**
```rust
let result = client.call_tool("agent_spawn", json!({
    "type": "researcher",
    "name": "market-analyst",
    "capabilities": ["data-analysis", "statistics", "visualisation"],
    "resources": {
        "cpu": "1.0",
        "memory": "1GB",
        "gpu": false
    },
    "config": {
        "learning_rate": 0.01,
        "batch_size": 32
    }
})).await?;

let agent_id = result["agentId"].as_str().unwrap();
println!("Spawned agent: {}", agent_id);
```

#### `agent_list`
Lists all active agents with filtering and pagination.

**Parameters:**
- `filter: String` - Filter by status (all, active, idle, busy) (optional)
- `type_filter: String` - Filter by agent type (optional)
- `limit: u32` - Maximum results (default: 50) (optional)
- `offset: u32` - Pagination offset (default: 0) (optional)

**Response:**
```json
{
  "agents": [
    {
      "id": "agent-uuid-12345",
      "type": "researcher",
      "status": "active",
      "current_task": "market-analysis-001",
      "uptime": 3600,
      "performance": {
        "tasks_completed": 15,
        "success_rate": 0.95,
        "avg_response_time": 1.2
      }
    }
  ],
  "total": 1,
  "page": 1
}
```

#### `agent_metrics`
Retrieves detailed performance metrics for agents.

**Parameters:**
- `agentId: String` - Specific agent ID (optional)
- `metric: String` - Metric type (all, cpu, memory, tasks, performance) (optional)
- `timeframe: String` - Time range (1h, 24h, 7d, 30d) (optional)

### Task Orchestration

#### `task_orchestrate`
Orchestrates complex multi-step tasks across the swarm.

**Parameters:**
- `task: String` - Task description or JSON specification
- `strategy: String` - Execution strategy (parallel, sequential, adaptive, balanced)
- `priority: String` - Priority level (low, medium, high, critical)
- `maxAgents: u32` - Maximum agents to use (optional)
- `timeout: u32` - Task timeout in seconds (optional)
- `dependencies: Array` - Task dependencies (optional)

**Example:**
```rust
let result = client.call_tool("task_orchestrate", json!({
    "task": {
        "description": "Comprehensive market analysis",
        "steps": [
            {"name": "data_collection", "agent_type": "researcher"},
            {"name": "data_analysis", "agent_type": "analyst"},  
            {"name": "visualisation", "agent_type": "coder"},
            {"name": "report_generation", "agent_type": "documenter"},
            {"name": "quality_review", "agent_type": "reviewer"}
        ]
    },
    "strategy": "adaptive",
    "priority": "high",
    "maxAgents": 5,
    "timeout": 3600
})).await?;

let task_id = result["taskId"].as_str().unwrap();
```

#### `task_status`
Monitors task execution progress with detailed status information.

**Parameters:**
- `taskId: String` - Task ID to check (optional)
- `detailed: bool` - Include detailed progress info (default: false)
- `include_logs: bool` - Include execution logs (default: false)

#### `task_results`
Retrieves comprehensive results from completed tasks.

**Parameters:**
- `taskId: String` - Task ID to get results for
- `format: String` - Result format (summary, detailed, raw) (optional)
- `include_artifacts: bool` - Include generated files/artifacts (optional)

### Swarm Management

#### `swarm_init`
Initializes a new swarm with specified topology and configuration.

**Parameters:**
- `topology: String` - Topology type (mesh, hierarchical, ring, star)
- `maxAgents: u32` - Maximum number of agents (default: 5)
- `strategy: String` - Distribution strategy (balanced, specialized, adaptive)
- `config: Object` - Swarm-specific configuration (optional)

#### `swarm_status`
Gets current swarm status and comprehensive statistics.

**Parameters:**
- `swarmId: String` - Specific swarm ID (optional)
- `verbose: bool` - Include detailed information (default: false)
- `include_metrics: bool` - Include performance metrics (optional)

#### `swarm_monitor`
Enables real-time monitoring of swarm activity.

**Parameters:**
- `swarmId: String` - Swarm to monitor (optional)
- `duration: u32` - Monitoring duration in seconds (default: 10)
- `interval: u32` - Update interval in seconds (default: 1)
- `metrics: Array` - Specific metrics to track (optional)

### Memory Operations

#### `memory_usage`
Manages persistent memory with namespacing and TTL support.

**Parameters:**
- `action: String` - Action (store, retrieve, list, delete, search)
- `key: String` - Memory key
- `value: String` - Value for store operations
- `namespace: String` - Namespace (default: "default")
- `ttl: u32` - Time to live in seconds (optional)
- `tags: Array` - Searchable tags (optional)

**Examples:**
```rust
// Store with TTL and tags
client.call_tool("memory_usage", json!({
    "action": "store",
    "key": "market_data_2024_q1",
    "value": "{\"growth_rate\": 0.15, \"market_cap\": 2500000}",
    "namespace": "market_analysis",
    "ttl": 86400, // 24 hours
    "tags": ["2024", "q1", "financial", "analysis"]
})).await?;

// Retrieve with namespace
let data = client.call_tool("memory_usage", json!({
    "action": "retrieve", 
    "key": "market_data_2024_q1",
    "namespace": "market_analysis"
})).await?;
```

### Neural Operations

#### `neural_train`
Trains neural patterns with WASM SIMD acceleration.

**Parameters:**
- `pattern_type: String` - Pattern type (coordination, optimisation, prediction)
- `training_data: String` - Training dataset or reference
- `epochs: u32` - Number of training epochs (default: 50)
- `learning_rate: f32` - Learning rate (default: 0.01)

#### `neural_predict`
Makes predictions using trained neural models.

**Parameters:**
- `modelId: String` - Model identifier  
- `input: String` - Input data for prediction
- `confidence_threshold: f32` - Minimum confidence (optional)

#### `neural_patterns`
Analyzes and manages cognitive patterns.

**Parameters:**
- `action: String` - Action (analyze, learn, predict, optimize)
- `pattern: String` - Pattern type (all, convergent, divergent, lateral, systems)
- `data: Object` - Pattern-specific data
- `agent_id: String` - Target agent (optional)

### Performance Monitoring

#### `performance_report`
Generates comprehensive performance reports with metrics.

**Parameters:**
- `format: String` - Report format (summary, detailed, json, pdf)
- `timeframe: String` - Time range (1h, 24h, 7d, 30d)
- `components: Array` - Specific components to include (optional)

#### `bottleneck_analyze`  
Identifies and analyzes performance bottlenecks.

**Parameters:**
- `component: String` - Component to analyze (optional)
- `metrics: Array` - Metrics to evaluate (optional)  
- `threshold: f32` - Performance threshold (optional)

---

## Error Handling

### ConnectorError Types

Comprehensive error handling with detailed context and recovery suggestions.

```rust
#[derive(Debug, Clone)]
pub enum ConnectorError {
    // Connection Issues
    ConnectionError(String),
    NotConnected,
    ConnectionTimeout(Duration),
    ConnectionRefused(String),
    ConnectionLost(String),
    
    // Protocol Issues
    ProtocolError(String),
    UnsupportedProtocolVersion(String),
    HandshakeFailure(String),
    
    // Data Issues  
    SerializationError(String),
    DeserializationError(String),
    InvalidMessageFormat(String),
    
    // Tool Issues
    ToolError(String),
    ToolNotFound(String),
    ToolExecutionFailed(String),
    ToolTimeout(Duration),
    
    // Authentication Issues
    AuthenticationFailed(String),
    AuthorizationFailed(String),
    TokenExpired,
    
    // Resource Issues
    ResourceExhausted(String),
    QuotaExceeded(String),
    RateLimitExceeded(Duration),
    
    // System Issues
    InternalError(String),
    ServiceUnavailable(String),
    MaintenanceMode,
}

impl ConnectorError {
    /// Returns whether this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            ConnectorError::ConnectionTimeout(_) => true,
            ConnectorError::ConnectionLost(_) => true,
            ConnectorError::RateLimitExceeded(_) => true,
            ConnectorError::ServiceUnavailable(_) => true,
            _ => false,
        }
    }
    
    /// Returns suggested retry delay
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            ConnectorError::RateLimitExceeded(delay) => Some(*delay),
            ConnectorError::ConnectionTimeout(_) => Some(Duration::from_secs(5)),
            ConnectorError::ServiceUnavailable(_) => Some(Duration::from_secs(30)),
            _ => None,
        }
    }
    
    /// Returns user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            ConnectorError::NotConnected => 
                "Not connected to Claude Flow server. Please check connection.".to_string(),
            ConnectorError::ToolNotFound(tool) => 
                format!("Tool '{}' is not available. Check tool name or server capabilities.", tool),
            ConnectorError::AuthenticationFailed(_) => 
                "Authentication failed. Please check your API token.".to_string(),
            _ => format!("An error occurred: {}", self),
        }
    }
}
```

### Error Recovery Patterns

#### Automatic Retry with Exponential Backoff
```rust
use tokio::time::{sleep, Duration};

pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub backoff_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            backoff_factor: 2.0,
        }
    }
}

pub async fn retry_with_backoff<F, T, E>(
    operation: F,
    config: RetryConfig,
) -> Result<T>
where
    F: Fn() -> Result<T, E>,
    E: Into<ConnectorError>,
{
    let mut delay = config.initial_delay;
    
    for attempt in 1..=config.max_attempts {
        match operation() {
            Ok(result) => return Ok(result),
            Err(error) => {
                let error = error.into();
                
                if !error.is_recoverable() || attempt == config.max_attempts {
                    return Err(error);
                }
                
                // Use error-specific delay if available
                let retry_delay = error.retry_delay().unwrap_or(delay);
                sleep(retry_delay).await;
                
                // Exponential backoff
                delay = std::cmp::min(
                    Duration::from_millis(
                        (delay.as_millis() as f64 * config.backoff_factor) as u64
                    ),
                    config.max_delay
                );
            }
        }
    }
    
    Err(ConnectorError::InternalError("Max retries exceeded".to_string()))
}
```

---

## Type Definitions

### Core Types

```rust
use serde::{Deserialize, Serialize};

/// Result type for all MCP operations
pub type Result<T> = std::result::Result<T, ConnectorError>;

/// MCP protocol initialization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    pub server_info: Option<ServerInfo>,
    pub session_id: String,
}

/// Tool definition with schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    pub output_schema: Option<serde_json::Value>,
    pub examples: Option<Vec<ToolExample>>,
    pub tags: Option<Vec<String>>,
    pub deprecated: Option<bool>,
}

/// Connection status and statistics  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub connected: bool,
    pub transport: String,
    pub host: String,
    pub port: u16,
    pub protocol_version: Option<String>,
    pub session_id: Option<String>,
    pub stats: ConnectionStats,
    pub capabilities: Option<ServerCapabilities>,
}

/// Connection performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub connected_at: Option<chrono::DateTime<chrono::Utc>>,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time: f64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub reconnection_count: u32,
}
```

---

## Testing Patterns

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    async fn create_test_client() -> ClaudeFlowClient {
        ClaudeFlowClientBuilder::new()
            .with_tcp()
            .build()
            .await
            .expect("Failed to create test client")
    }

    #[tokio::test]
    async fn test_client_connection() {
        let mut client = create_test_client().await;
        let result = client.connect().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tool_call_success() {
        let mut client = create_test_client().await;
        client.connect().await.unwrap();
        
        let result = client.call_tool("agent_list", json!({})).await;
        assert!(result.is_ok());
    }

    #[tokio::test] 
    async fn test_error_handling() {
        let mut client = create_test_client().await;
        
        // Test calling tool without connection
        let result = client.call_tool("agent_list", json!({})).await;
        assert!(matches!(result, Err(ConnectorError::NotConnected)));
    }
}
```

### Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    async fn setup_test_environment() -> ClaudeFlowClient {
        ClaudeFlowClientBuilder::new()
            .with_tcp()
            .host("localhost")
            .port(9500)
            .with_timeout(Duration::from_secs(30))
            .build()
            .await
            .expect("Failed to create test client")
    }

    #[tokio::test]
    async fn test_full_agent_lifecycle() {
        let mut client = setup_test_environment().await;
        
        // 1. Spawn agent
        let spawn_result = client.call_tool("agent_spawn", json!({
            "type": "researcher",
            "name": "integration-test-agent"
        })).await.expect("Failed to spawn agent");
        
        let agent_id = spawn_result["agentId"].as_str().unwrap();
        
        // 2. Verify agent exists
        let agents = client.call_tool("agent_list", json!({}))
            .await.expect("Failed to list agents");
            
        assert!(agents.as_array().unwrap().iter()
            .any(|a| a["id"] == agent_id));
        
        // 3. Get agent metrics
        let metrics = client.call_tool("agent_metrics", json!({
            "agentId": agent_id
        })).await.expect("Failed to get metrics");
        
        assert_eq!(metrics["agentId"], agent_id);
        assert!(metrics["metrics"].is_object());
    }
}
```

---

## Performance Optimization

### Connection Pooling

```rust
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use std::collections::VecDeque;

pub struct ConnectionPool {
    connections: Arc<RwLock<VecDeque<ClaudeFlowClient>>>,
    semaphore: Arc<Semaphore>,
    config: PoolConfig,
}

pub struct PoolConfig {
    pub max_size: usize,
    pub min_size: usize,
    pub max_idle_time: Duration,
    pub health_check_interval: Duration,
}

impl ConnectionPool {
    pub fn new(config: PoolConfig) -> Self {
        Self {
            connections: Arc::new(RwLock::new(VecDeque::new())),
            semaphore: Arc::new(Semaphore::new(config.max_size)),
            config,
        }
    }
    
    pub async fn acquire(&self) -> Result<PooledConnection> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| ConnectorError::ResourceExhausted("Pool exhausted".to_string()))?;
        
        // Try to get existing connection
        if let Some(client) = self.try_get_connection().await {
            return Ok(PooledConnection::new(client, self.clone()));
        }
        
        // Create new connection
        let client = ClaudeFlowClientBuilder::new()
            .with_tcp()
            .with_timeout(Duration::from_secs(30))
            .build()
            .await?;
            
        Ok(PooledConnection::new(client, self.clone()))
    }
}
```

---

## Environment Configuration

### Complete Environment Variables Reference

```bash
# Core Connection Settings
export CLAUDE_FLOW_HOST="multi-agent-container"     # Target host
export MCP_TCP_PORT=9500                            # TCP port
export MCP_WS_PORT=9501                             # WebSocket port

# Connection Management
export MCP_CONNECTION_TIMEOUT=30000                 # Connection timeout (ms)
export MCP_RECONNECT_ATTEMPTS=3                     # Max reconnection attempts
export MCP_RECONNECT_DELAY=1000                     # Delay between attempts (ms)

# Connection Pooling
export MCP_POOL_SIZE=5                              # Connection pool size
export MCP_POOL_MIN_SIZE=2                          # Minimum connections
export MCP_POOL_MAX_IDLE_TIME=300000               # Max idle time (ms)

# Performance Tuning
export MCP_BUFFER_SIZE=8192                         # Buffer size (bytes)
export MCP_MAX_CONCURRENT_REQUESTS=100             # Max concurrent requests
export MCP_REQUEST_TIMEOUT=300000                  # Request timeout (ms)

# Authentication & Security
export CLAUDE_FLOW_API_TOKEN="your-api-token"      # API authentication token
export MCP_AUTH_TIMEOUT=10000                      # Auth timeout (ms)

# Monitoring & Logging
export MCP_LOG_LEVEL="info"                        # Log level
export MCP_METRICS_ENABLED=true                   # Enable metrics collection
export MCP_HEALTH_CHECK_ENABLED=true              # Enable health checks

# Agent & Task Management
export MCP_MAX_AGENTS=50                          # Maximum agents per swarm
export MCP_AGENT_TIMEOUT=3600                     # Agent operation timeout (seconds)
export MCP_TASK_QUEUE_SIZE=1000                   # Task queue size
```

---

*This API reference provides comprehensive coverage of the Claude Flow MCP integration, including all core components, transport layers, error handling, testing patterns, and performance optimisation techniques. It serves as the definitive guide for integrating with and extending the Claude Flow multi-agent system.*

**Version**: 2.0  
**Last Updated**: January 15, 2025  
**MCP Protocol**: 2024-11-05  
**Supported Transports**: TCP Only