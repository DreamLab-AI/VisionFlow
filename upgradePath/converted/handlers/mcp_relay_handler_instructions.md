# MCP Relay Handler Instructions

## File: `src/handlers/mcp_relay_handler.rs`

### Purpose
WebSocket relay handler that bridges client connections to the MCP (Model Context Protocol) orchestrator. Provides bidirectional message forwarding between web clients and the multi-agent orchestration system.

### Key Components

#### 1. MCP Relay Actor Structure
```rust
pub struct MCPRelayActor {
    client_id: String,                    // Unique client identifier
    orchestrator_tx: Option<Arc<Mutex<SplitSink<...>>>>, // Orchestrator connection
    self_addr: Option<Addr<Self>>,        // Self-reference for async operations
}
```

#### 2. Message Types
```rust
// Messages from client to orchestrator
struct ClientText(String);
struct ClientBinary(Vec<u8>);

// Messages from orchestrator to client  
struct OrchestratorText(String);
struct OrchestratorBinary(Vec<u8>);
```

### Implementation Instructions

#### Actor Initialization
1. **Client ID Generation**: Create unique UUID for each client connection
2. **Self-Reference Storage**: Store actor address for async message handling
3. **Connection State**: Initialize with no orchestrator connection
4. **Logging**: Log client connection establishment

```rust
fn new() -> Self {
    Self {
        client_id: uuid::Uuid::new_v4().to_string(),
        orchestrator_tx: None,
        self_addr: None,
    }
}
```

#### Orchestrator Connection Management

##### Connection Establishment
1. **Environment Configuration**: Read `ORCHESTRATOR_WS_URL` or use default
2. **Async Connection**: Use `tokio_tungstenite::connect_async` for WebSocket connection
3. **Stream Splitting**: Split connection into sink (tx) and stream (rx) components
4. **Thread-Safe Access**: Wrap sink in `Arc<Mutex<>>` for concurrent access

```rust
fn connect_to_orchestrator(&mut self, ctx: &mut <Self as Actor>::Context) {
    let orchestrator_url = std::env::var("ORCHESTRATOR_WS_URL")
        .unwrap_or_else(|_| "ws://multi-agent-container:3002/ws".to_string());
    
    // Spawn async connection task
    actix::spawn(async move {
        // Connection logic with error handling
    });
}
```

##### Message Forwarding Loop
1. **Bidirectional Relay**: Forward messages in both directions
2. **Message Type Handling**: Support Text, Binary, Close, Ping, Pong messages
3. **Connection Monitoring**: Detect and handle connection failures
4. **Heartbeat Management**: Respond to ping/pong for connection health

#### WebSocket Message Handling

##### Client to Orchestrator Flow
```rust
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for MCPRelayActor {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        match msg {
            Ok(ws::Message::Text(text)) => {
                // Forward to orchestrator
                self.forward_to_orchestrator(text, ctx);
            }
            Ok(ws::Message::Binary(bin)) => {
                // Forward binary data
                self.forward_binary_to_orchestrator(bin, ctx);
            }
            Ok(ws::Message::Close(_)) => {
                // Handle client disconnect
                self.handle_client_disconnect(ctx);
            }
            Err(e) => {
                // Handle protocol errors
                error!("WebSocket protocol error: {}", e);
            }
        }
    }
}
```

##### Orchestrator to Client Flow
```rust
impl Handler<OrchestratorText> for MCPRelayActor {
    type Result = ();
    
    fn handle(&mut self, msg: OrchestratorText, ctx: &mut Self::Context) {
        // Forward orchestrator messages to connected client
        ctx.text(msg.0);
    }
}

impl Handler<OrchestratorBinary> for MCPRelayActor {
    type Result = ();
    
    fn handle(&mut self, msg: OrchestratorBinary, ctx: &mut Self::Context) {
        // Forward binary data to client
        ctx.binary(msg.0);
    }
}
```

#### Error Handling & Resilience

##### Connection Failure Recovery
1. **Retry Logic**: Implement exponential backoff for reconnection attempts
2. **Circuit Breaker**: Prevent cascade failures during orchestrator outages
3. **Graceful Degradation**: Handle orchestrator unavailability gracefully
4. **Client Notification**: Inform clients of connection status changes

```rust
// Connection retry with exponential backoff
async fn retry_orchestrator_connection(addr: Addr<MCPRelayActor>, attempt: u32) {
    let delay = std::cmp::min(300, 5 * 2_u64.pow(attempt)); // Max 5 minutes
    actix::clock::sleep(Duration::from_secs(delay)).await;
    addr.do_send(OrchestratorText("retry".to_string()));
}
```

##### Message Handling Errors
1. **Serialization Errors**: Handle JSON parsing failures gracefully
2. **Network Errors**: Manage network connectivity issues
3. **Protocol Errors**: Handle WebSocket protocol violations
4. **Resource Limits**: Manage memory and connection limits

#### Performance Optimizations

##### Connection Pooling
1. **Shared Connections**: Reuse orchestrator connections across clients
2. **Connection Limits**: Implement maximum connection limits
3. **Load Balancing**: Distribute clients across multiple orchestrator instances
4. **Health Monitoring**: Monitor connection health and performance

##### Message Buffering
1. **Message Queuing**: Buffer messages during temporary disconnections
2. **Backpressure Handling**: Manage high message volume scenarios
3. **Memory Management**: Prevent unbounded message accumulation
4. **Priority Queuing**: Prioritize critical messages

#### Security Considerations

##### Client Authentication
1. **Connection Validation**: Verify client authorization before relay
2. **Rate Limiting**: Prevent client message flooding
3. **Message Filtering**: Filter sensitive data from forwarded messages
4. **Access Control**: Implement role-based message filtering

##### Data Protection
1. **Message Encryption**: Encrypt sensitive message content
2. **Audit Logging**: Log all message relay activities
3. **Input Validation**: Validate message format and content
4. **Injection Prevention**: Prevent message injection attacks

### Configuration Management

#### Environment Variables
```rust
// Configuration options
const DEFAULT_ORCHESTRATOR_URL: &str = "ws://multi-agent-container:3002/ws";
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
const MAX_RETRY_ATTEMPTS: u32 = 5;
```

#### Runtime Configuration
1. **Dynamic URL Updates**: Support orchestrator URL changes
2. **Timeout Configuration**: Configurable connection timeouts
3. **Retry Policy**: Configurable retry behavior
4. **Feature Flags**: Enable/disable specific relay features

### Monitoring & Observability

#### Connection Metrics
1. **Active Connections**: Track number of active client connections
2. **Message Throughput**: Monitor message forwarding rates
3. **Connection Duration**: Track connection lifetimes
4. **Error Rates**: Monitor connection and message errors

#### Health Checks
1. **Orchestrator Connectivity**: Monitor orchestrator connection health
2. **Client Connection Health**: Track client connection quality
3. **Message Delivery**: Verify message delivery success rates
4. **Resource Usage**: Monitor memory and CPU usage

### Testing Requirements

1. **WebSocket Integration Tests**: Test full client-orchestrator relay
2. **Connection Failure Tests**: Test reconnection logic and error handling
3. **Message Forwarding Tests**: Verify bidirectional message relay
4. **Performance Tests**: Test under high message volume
5. **Security Tests**: Validate authentication and rate limiting
6. **Load Tests**: Test multiple concurrent client connections

### Deployment Considerations

1. **Container Networking**: Ensure proper container-to-container communication
2. **Service Discovery**: Support dynamic orchestrator discovery
3. **Load Balancing**: Deploy multiple relay instances for high availability
4. **Monitoring Integration**: Integrate with observability stack
5. **Graceful Shutdown**: Handle clean shutdown of relay connections