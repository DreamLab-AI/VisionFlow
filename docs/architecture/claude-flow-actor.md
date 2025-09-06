# ClaudeFlowActor Architecture

This document describes the ClaudeFlowActor implementation that integrates the Claude Flow MCP (Model Context Protocol) with the LogseqXR backend.

## Overview

The `ClaudeFlowActor` is a Rust Actix actor that manages the connection to the Claude Flow MCP server. It connects to a dedicated, containerized MCP service (`powerdev`) over the network using **WebSockets**, and provides agent data to the visualisation system. The previous `stdio` transport mechanism has been disabled.

## Architecture

### Actor Structure

```rust
pub struct ClaudeFlowActor {
    client: ClaudeFlowClient,           // MCP client
    graph_service_addr: Addr<GraphServiceActor>,  // Graph visualisation service
    is_connected: bool,                 // Connection status
}
```

### Transport Mechanism

The actor exclusively uses **TCP transport** to connect to the Claude Flow MCP server on port 9500. WebSocket and stdio transports have been removed.

```rust
// src/actors/claude_flow_actor.rs
let host = std::env::var("CLAUDE_FLOW_HOST").unwrap_or_else(|_| {
    if std::env::var("DOCKER_ENV").is_ok() {
        "claude-flow-mcp".to_string()  // Docker service name
    } else {
        "localhost".to_string()  // Local development
    }
});
let port = std::env::var("MCP_TCP_PORT").unwrap_or_else(|_| "9500".to_string());

// Direct TCP connection to Claude Flow MCP
let stream = TcpStream::connect(&format!("{}:{}", host, port)).await?;
stream.set_nodelay(true)?;  // TCP optimisation
```

### Disabled Stdio Transport

The `StdioTransport` implementation in `src/services/claude_flow/transport/stdio.rs` is no longer used. Its `connect` function now returns an error to prevent its use.

```rust
// TCP-only implementation - WebSocket and stdio transports removed
// ClaudeFlowActorTcp handles direct TCP connection with JSON-RPC over TCP
async fn connect_to_claude_flow_tcp() -> Result<(BufWriter<OwnedWriteHalf>, BufReader<OwnedReadHalf>), Box<dyn std::error::Error + Send + Sync>> {
    let stream = TcpStream::connect(&addr).await?;
    stream.set_nodelay(true)?;
    let (read_half, write_half) = stream.into_split();
    Ok((BufWriter::new(write_half), BufReader::new(read_half)))
}
```

## Data Flow

### 1. Initialisation Sequence

```mermaid
graph TD
    subgraph "Rust Backend"
        A[Client Application] -->|HTTP/WebSocket| B(Actix Web Server)
        B -->|WebSocket| C{MCP Relay Actor}
    end

    subgraph "Docker Network"
        C -->|TCP: claude-flow-mcp:9500| D[Claude Flow MCP Container]
    end

    subgraph "MCP Service"
        D --> E(Claude Flow MCP)
    end

    style B fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#ccf,stroke:#333,stroke-width:2px
    style D fill:#cfc,stroke:#333,stroke-width:2px
```

### 2. Agent Polling

The actor polls for agent updates periodically when connected.

```rust
// Polls every 5 seconds when connected
ctx.run_interval(Duration::from_secs(5), |act, _ctx| {
    let client = act.client.clone();
    let graph_addr = act.graph_service_addr.clone();

    actix::spawn(async move {
        match client.list_agents(false).await {
            Ok(agents) => {
                graph_addr.do_send(UpdateBotsGraph { agents });
            }
            Err(e) => error!("Failed to poll agents: {}", e),
        }
    });
});
```

### 3. Mock Mode

When the connection to the `powerdev` container is unavailable, the actor can provide mock data for development and testing purposes.

```rust
fn create_mock_agents() -> Vec<AgentStatus> {
    vec![
        AgentStatus {
            agent_id: "coordinator-001",
            status: "active",
            profile: AgentProfile {
                name: "System Coordinator",
                agent_type: AgentType::Coordinator,
                capabilities: vec!["orchestration", "task-management"],
            },
            // ... other fields
        },
        // ... more mock agents
    ]
}
```

## Message Handlers

The actor responds to several messages to manage the Multi Agent.

- **GetActiveAgents**: Returns the current list of agents (real or mock).
- **SpawnClaudeAgent**: Creates a new agent in the multi-agent.
- **initializeMultiAgent**: Initialises a complete multi-agent configuration.

## Environment Configuration

The actor's connection settings are configured via environment variables.

```bash
# Hostname of the container running Claude Flow MCP
CLAUDE_FLOW_HOST=claude-flow-mcp    # Default: "claude-flow-mcp" (Docker), "localhost" (dev)

# Port for the TCP MCP connection
MCP_TCP_PORT=9500                   # Default: 9500
```

## Health Monitoring

Health checks are performed every 30 seconds to ensure the MCP service is responsive.

```rust
ctx.run_interval(Duration::from_secs(30), |act, _ctx| {
    let client = act.client.clone();

    actix::spawn(async move {
        match client.get_system_health().await {
            Ok(health) if health.status != "healthy" => {
                warn!("System health check failed: {:?}", health);
            }
            Err(e) => error!("Health check failed: {}", e),
            _ => {}
        }
    });
});
```

## Error Handling and Degraded Mode

The actor is designed for resilience:
1.  **Connection Failure**: Logs the error and switches to mock mode, ensuring the UI remains functional.
2.  **Polling/Health Check Errors**: Errors are logged without crashing the actor, maintaining system stability.

## Container Integration

The system relies on Docker networking for service discovery. The backend service connects to the `powerdev` container using its service name (`powerdev`) as the hostname. No local Node.js/npm installation is required for the backend container to communicate with the MCP.

## API Integration

The actor integrates with the main application's REST API to expose bot management endpoints:

-   `GET /api/bots/agents`
-   `POST /api/bots/spawn`
-   `POST /api/bots/multi-agent/init`
-   `DELETE /api/bots/agent/:id`

## Performance Considerations

1.  **Network Communication**: All communication is over TCP using JSON-RPC protocol with line-delimited messages.
2.  **Polling Intervals**: Agent polling (5s) and health checks (30s) are asynchronous and non-blocking.
3.  **Resource Usage**: As a client, the actor's resource footprint is minimal. The `powerdev` container manages the resource-intensive MCP process.

## Future Enhancements

1.  **Event Streaming**: Replace polling with a full event-driven model for real-time updates.
2.  **Dynamic Reconnection**: Implement more robust logic to automatically re-establish lost WebSocket connections.
3.  **Metrics Collection**: Integrate with Prometheus or a similar tool for detailed performance monitoring.

## Troubleshooting

### Common Issues

1.  **Connection Refused Errors**
    -   Verify the `powerdev` container is running and healthy: `docker ps | grep powerdev`.
    -   Check Docker network settings to ensure the backend container can resolve the `powerdev` hostname.
    -   Confirm `CLAUDE_FLOW_HOST` and `CLAUDE_FLOW_PORT` environment variables are set correctly.

2.  **WebSocket Handshake Failures**
    -   Inspect logs from both the backend and the `powerdev` container for error messages.
    -   Enable debug logging for more detailed output: `RUST_LOG=debug`.

### Debug Commands

```bash
# Check container logs for errors
docker logs <backend_container_id>
docker logs <powerdev_container_id>

# Check actor-specific logs
RUST_LOG=logseq_spring_thing::actors::claude_flow_actor=debug ./target/release/visionflow
```