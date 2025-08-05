# Claude Flow WebSocket Architecture

## Overview
The Claude Flow integration has been updated to use WebSocket transport instead of stdio for improved reliability and real-time communication.

## Architecture Components

### 1. Multi-Agent Container
- **Role**: Hosts the agent control system with MCP integration
- **Port**: 9500 (TCP server for agent control)
- **Service**: Agent Control System (JavaScript)
- **Location**: `/workspace/agent-control-system/`

### 2. Communication Architecture
The system uses TCP for inter-container communication:
```
Rust Backend <--TCP--> Agent Control System <--stdio--> claude-flow@alpha MCP
(VisionFlow)           (multi-agent-container)           (MCP Server)
```

### 3. Rust Backend (VisionFlow)
- **AgentControlActor**: Manages TCP connection to multi-agent-container
- **Connection**: `tcp://multi-agent-container:9500`
- **Features**: 
  - JSON-RPC 2.0 protocol over TCP
  - Automatic reconnection with exponential backoff
  - Graceful degradation to mock data when disconnected
  - Health monitoring

## Key Configuration

### Environment Variables
```bash
# In Rust container (.env or docker-compose.yml)
AGENT_CONTROL_URL=multi-agent-container:9500  # TCP connection to agent container
```

### Docker Networking
- Both containers must be on the same Docker network
- Multi-agent-container exposes port 9500 internally
- No host port mapping required for container-to-container communication

## Startup Sequence

1. **Multi-agent-container starts** with Agent Control System
2. **Rust container starts** and attempts TCP connection
3. **If connection fails**, Rust enters degraded mode with mock data
4. **Automatic reconnection** attempts with exponential backoff

## Error Handling

### Connection Failures
- No panic on connection failure
- Provides mock agents for visualization
- Continues attempting reconnection
- Clear error messages in API responses

### API Error Responses
```json
{
  "success": false,
  "error": "Failed to connect to Claude Flow",
  "code": "CONNECTION_REFUSED",
  "details": {
    "suggestion": "Ensure the MCP WebSocket relay is running on powerdev:3000"
  }
}
```

## Monitoring & Diagnostics

### Health Check Endpoint
```bash
GET /api/v1/bots/health
```

### Check WebSocket Relay Status
```bash
# From PowerDev container
netstat -tulpn | grep 3000
ps aux | grep mcp-ws-relay

# View logs
tail -f /var/log/mcp-ws-relay.log
```

### Check Rust Logs
```bash
# From host
./scripts/check-rust-logs.sh -f
```

## Troubleshooting

### MCP Not Starting Automatically
1. Check docker-compose.yml has the correct command
2. Verify `/workspace/ext/src/mcp-ws-relay.js` exists
3. Check Node.js is installed in PowerDev

### Connection Refused Errors
1. Verify MCP relay is running: `docker exec powerdev netstat -tulpn | grep 3000`
2. Check Docker network connectivity
3. Restart both containers in correct order

### Manual Recovery
```bash
# From host system
docker exec powerdev pkill -f mcp-ws-relay
docker exec -d powerdev bash -c "cd /workspace/ext/src && node mcp-ws-relay.js > /var/log/mcp-ws-relay.log 2>&1"
docker restart logseq_spring_thing_webxr
```

## Future Enhancements

### Real-time Updates (Phase 3)
- Implement WebSocket event streaming from MCP
- Push agent status updates to Rust backend
- Eliminate polling in favor of event-driven updates

### Production Hardening
- Implement authentication/authorization
- Add SSL/TLS for WebSocket connections
- Rate limiting and connection management
- Metrics and monitoring integration