# MCP Backend Connection Configuration

## Current Architecture

The bot control and telemetry system connects to the multi-agent-docker container via TCP.

## Connection Details

### Network Configuration
- **Docker Network:** ragflow
- **Container:** multi-agent-docker  
- **Port:** 9500 (TCP)
- **Protocol:** MCP (Multi-agent Communication Protocol)

### Connection String
```rust
// For ClaudeFlowActor connection
const MCP_HOST: &str = "multi-agent-docker";
const MCP_PORT: u16 = 9500;
const MCP_NETWORK: &str = "ragflow";
```

## Implementation for ClaudeFlowActor

### Update claude_flow_actor_enhanced.rs

```rust
use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

impl ClaudeFlowActor {
    async fn connect_to_mcp(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Connect to multi-agent-docker on ragflow network
        let addr = "multi-agent-docker:9500";
        
        info!("Connecting to MCP backend at {}", addr);
        
        let stream = TcpStream::connect(addr).await?;
        info!("Successfully connected to MCP backend on port 9500");
        
        self.mcp_connection = Some(stream);
        self.is_connected = true;
        
        // Start telemetry receiver
        self.start_telemetry_receiver().await?;
        
        Ok(())
    }
    
    async fn start_telemetry_receiver(&mut self) {
        if let Some(ref mut stream) = self.mcp_connection {
            let mut buffer = [0; 1024];
            
            loop {
                match stream.read(&mut buffer).await {
                    Ok(n) if n > 0 => {
                        // Parse telemetry data
                        self.process_telemetry(&buffer[..n]);
                    }
                    Ok(_) => {
                        warn!("MCP connection closed");
                        self.is_connected = false;
                        break;
                    }
                    Err(e) => {
                        error!("Error reading from MCP: {}", e);
                        self.is_connected = false;
                        break;
                    }
                }
            }
        }
    }
    
    fn process_telemetry(&mut self, data: &[u8]) {
        // Parse the telemetry data format from multi-agent-docker
        // This likely includes:
        // - Agent status updates
        // - Performance metrics
        // - Task assignments
        // - Message flow data
        
        match serde_json::from_slice::<TelemetryMessage>(data) {
            Ok(telemetry) => {
                match telemetry.msg_type {
                    TelemetryType::AgentStatus => {
                        self.update_agent_status(telemetry.data);
                    }
                    TelemetryType::Performance => {
                        self.update_performance_metrics(telemetry.data);
                    }
                    TelemetryType::TaskUpdate => {
                        self.update_task_status(telemetry.data);
                    }
                    TelemetryType::MessageFlow => {
                        self.update_message_flow(telemetry.data);
                    }
                }
            }
            Err(e) => {
                warn!("Failed to parse telemetry data: {}", e);
            }
        }
    }
}
```

## Docker Compose Configuration

Ensure the containers are on the same network:

```yaml
# docker-compose.yml
services:
  ext-app:
    container_name: ext-app
    networks:
      - ragflow
    depends_on:
      - multi-agent-docker

  multi-agent-docker:
    container_name: multi-agent-docker
    ports:
      - "9500:9500"
    networks:
      - ragflow

networks:
  ragflow:
    external: true
```

## Environment Variables

```bash
# .env
MCP_HOST=multi-agent-docker
MCP_PORT=9500
MCP_NETWORK=ragflow
MCP_RECONNECT_INTERVAL=5000
MCP_TIMEOUT=30000
```

## Testing the Connection

### 1. Verify Network Connectivity
```bash
# From within the ext container
ping multi-agent-docker

# Test TCP connection
nc -zv multi-agent-docker 9500
```

### 2. Test Data Flow
```bash
# Monitor MCP telemetry
tcpdump -i any -A port 9500

# Check actor logs
tail -f /workspace/ext/logs/claude_flow_actor.log
```

## Data Format Expected from MCP

Based on the telemetry system, the multi-agent-docker likely sends:

```json
{
  "type": "agent_status",
  "timestamp": "2024-01-09T12:00:00Z",
  "data": {
    "agent_id": "agent-001",
    "status": "active",
    "cpu_usage": 45.2,
    "memory_usage": 512,
    "current_task": "Processing request",
    "position": [0.0, 0.0, 0.0],
    "velocity": [0.1, 0.0, 0.0],
    "connections": ["agent-002", "agent-003"]
  }
}
```

## Migration Path from Mock Data

### Phase 1: Parallel Operation
- Keep mock data generation as fallback
- Attempt MCP connection on startup
- Log all received telemetry for analysis

### Phase 2: Hybrid Mode
- Use real data when available
- Fill gaps with mock data
- Validate data format consistency

### Phase 3: Full Integration
- Remove all mock data generation
- Rely entirely on MCP telemetry
- Implement error handling and reconnection

## Error Handling

```rust
impl ClaudeFlowActor {
    async fn handle_mcp_connection(&mut self, ctx: &mut Context<Self>) {
        loop {
            if !self.is_connected {
                // Attempt reconnection
                match self.connect_to_mcp().await {
                    Ok(()) => {
                        info!("Reconnected to MCP backend");
                    }
                    Err(e) => {
                        error!("Failed to reconnect to MCP: {}", e);
                        // Fall back to mock data
                        self.use_mock_data = true;
                        
                        // Retry after interval
                        ctx.run_later(
                            Duration::from_millis(5000),
                            |act, ctx| {
                                ctx.spawn(
                                    async move {
                                        act.handle_mcp_connection(ctx).await;
                                    }.into_actor(act)
                                );
                            }
                        );
                        break;
                    }
                }
            }
            
            // Process incoming telemetry
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}
```

## Monitoring and Debugging

### Health Check Endpoint
```rust
// In health_handler.rs
pub async fn check_mcp_connection(state: web::Data<AppState>) -> impl Responder {
    if let Some(claude_flow_addr) = &state.claude_flow_addr {
        match claude_flow_addr.send(GetConnectionStatus).await {
            Ok(Ok(status)) => {
                HttpResponse::Ok().json(json!({
                    "mcp_connected": status.is_connected,
                    "host": "multi-agent-docker",
                    "port": 9500,
                    "agents_count": status.agent_count,
                    "last_telemetry": status.last_telemetry_timestamp
                }))
            }
            _ => {
                HttpResponse::ServiceUnavailable().json(json!({
                    "error": "MCP connection unavailable"
                }))
            }
        }
    }
}
```

## Notes

- The multi-agent-docker container must be running before starting the ext application
- Ensure both containers are on the ragflow network
- Port 9500 must be exposed within the Docker network
- Consider implementing a message queue for buffering telemetry during disconnections