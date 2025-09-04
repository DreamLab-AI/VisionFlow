# Multi-MCP WebSocket Handler Instructions

## File: `src/handlers/multi_mcp_websocket_handler.rs`

### Purpose
Provides real-time WebSocket streaming of agent visualization data from multiple MCP servers to the VisionFlow graph renderer. Implements resilience patterns including circuit breakers, health checks, and timeout management.

### Key Components

#### 1. WebSocket Actor Structure
```rust
pub struct MultiMcpVisualizationWs {
    app_state: web::Data<AppState>,
    client_id: String,
    last_heartbeat: Instant,
    last_discovery_request: Instant,
    subscription_filters: SubscriptionFilters,
    performance_mode: PerformanceMode,
    // Resilience components
    timeout_config: TimeoutConfig,
    circuit_breaker: Option<std::sync::Arc<CircuitBreaker>>,
    health_manager: Option<std::sync::Arc<HealthCheckManager>>,
}
```

#### 2. Subscription Filters
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionFilters {
    pub server_types: Vec<McpServerType>,      // ClaudeFlow, RuvSwarm, Daa
    pub agent_types: Vec<String>,              // coordinator, coder, etc.
    pub swarm_ids: Vec<String>,                // Specific swarm filtering
    pub include_performance: bool,             // Performance metrics
    pub include_neural: bool,                  // Neural agent data
    pub include_topology: bool,                // Topology updates
}
```

#### 3. Performance Modes
```rust
pub enum PerformanceMode {
    HighFrequency,    // 60Hz - active monitoring
    Normal,           // 10Hz - default mode  
    LowFrequency,     // 1Hz - dashboard overview
    OnDemand,         // minimal CPU usage
}
```

### Implementation Instructions

#### WebSocket Connection Setup
1. **Client Registration**: Generate unique client ID using UUID
2. **Resilience Initialization**: 
   - Create circuit breaker for MCP operations
   - Initialize health check manager
   - Configure timeout settings
3. **Filter Setup**: Apply default subscription filters
4. **Performance Mode**: Set based on client capabilities

#### Message Handling Patterns

##### Incoming Messages
1. **Subscription Updates**: 
   ```rust
   // Handle filter changes from client
   match msg_type {
       "update_filters" => apply_subscription_filters(filters),
       "performance_mode" => change_performance_mode(mode),
       "heartbeat" => update_last_heartbeat(),
   }
   ```

2. **Discovery Requests**: 
   - Query available MCP servers
   - Return server capabilities and status
   - Apply rate limiting for discovery calls

##### Outgoing Messages
1. **Agent Updates**: Stream real-time agent status changes
2. **Performance Metrics**: Send performance data based on filters
3. **Topology Changes**: Broadcast swarm topology updates
4. **Error Notifications**: Send client-friendly error messages

#### MCP Server Integration

##### Multi-Server Coordination
1. **Server Discovery**: Dynamically discover available MCP servers
2. **Load Balancing**: Distribute requests across healthy servers
3. **Fallback Handling**: Graceful degradation when servers fail
4. **Data Aggregation**: Combine data from multiple MCP sources

##### Circuit Breaker Implementation
```rust
// Protect against cascading failures
match circuit_breaker.call(mcp_operation).await {
    Ok(result) => process_successful_response(result),
    Err(CircuitBreakerError::Open) => use_cached_data(),
    Err(e) => handle_operation_failure(e),
}
```

#### Performance Optimization

##### Update Rate Management
1. **Adaptive Rates**: Adjust update frequency based on client performance
2. **Batch Updates**: Group multiple changes into single messages
3. **Selective Updates**: Send only changed data
4. **Compression**: Use message compression for large payloads

##### Memory Management
1. **Connection Cleanup**: Remove inactive client connections
2. **Data Caching**: Cache frequently accessed MCP data
3. **Memory Monitoring**: Track WebSocket memory usage
4. **Garbage Collection**: Periodic cleanup of stale data

#### Error Handling & Resilience

##### Connection Resilience
```rust
// Handle connection interruptions
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MultiMcpVisualizationWs {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => handle_text_message(text, ctx),
            Ok(ws::Message::Close(_)) => graceful_shutdown(ctx),
            Err(e) => handle_protocol_error(e, ctx),
        }
    }
}
```

##### Health Check Integration
1. **MCP Server Health**: Monitor MCP server availability
2. **WebSocket Health**: Track connection quality metrics  
3. **Circuit Breaker Status**: Expose breaker state to clients
4. **Recovery Procedures**: Automatic recovery from failures

#### Security & Validation

##### Client Authentication
1. **Connection Validation**: Verify client permissions
2. **Rate Limiting**: Prevent WebSocket abuse
3. **Message Validation**: Sanitize incoming messages
4. **Access Control**: Filter data based on client permissions

##### Data Sanitization
1. **Input Filtering**: Validate subscription filter parameters
2. **Output Sanitization**: Remove sensitive data before sending
3. **Injection Prevention**: Protect against message injection
4. **Schema Validation**: Enforce message structure requirements

### Message Protocol Specification

#### Client to Server Messages
```json
{
    "type": "update_filters",
    "payload": {
        "server_types": ["claude-flow", "ruv-swarm"],
        "agent_types": ["coordinator", "coder"],
        "include_performance": true
    }
}
```

#### Server to Client Messages
```json
{
    "type": "agent_update",
    "timestamp": "2024-01-01T12:00:00Z",
    "payload": {
        "agents": [...],
        "performance_metrics": {...},
        "topology_changes": [...]
    }
}
```

### Testing Requirements

1. **WebSocket Integration Tests**: Full connection lifecycle
2. **MCP Server Mocking**: Test with simulated MCP responses
3. **Circuit Breaker Tests**: Failure scenarios and recovery
4. **Performance Tests**: High-frequency update handling
5. **Security Tests**: Authentication and rate limiting
6. **Resilience Tests**: Network interruption handling

### Monitoring & Observability

1. **Connection Metrics**: Track active connections and throughput
2. **MCP Server Metrics**: Monitor server response times and failures
3. **Error Tracking**: Log and analyze WebSocket errors
4. **Performance Monitoring**: Track message processing latency
5. **Health Dashboards**: Visual monitoring of system health