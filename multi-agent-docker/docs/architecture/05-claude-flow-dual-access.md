# Claude-Flow Dual Access Architecture

## Overview

Claude-Flow v110 is configured with a dual-access architecture that enables both local CLI usage within the container and external TCP access for remote projects. This design provides maximum flexibility while maintaining session isolation and security.

## Architecture Components

### 1. Local MCP Access (For Claude Code)

**Configuration**: `.mcp.json` in workspace  
**Usage**: Direct access via Claude Code CLI  
**Database**: Shared at `/workspace/.swarm/memory.db`  

This is the primary method for using Claude-Flow within the container:
- Claude Code can access all claude-flow tools via the `mcp__claude-flow__` prefix
- Agents share state and memory across all local operations
- Perfect for interactive development and testing

### 2. TCP Shared Access (Port 9500)

**Port**: 9500  
**Purpose**: External access to the same shared instance  
**Database**: Shared at `/workspace/.swarm/memory.db`  

The main TCP server provides:
- Access to the same claude-flow instance used locally
- Shared agent state with local operations
- Suitable for external tools that need to interact with the main workspace

### 3. TCP Isolated Sessions (Port 9502)

**Port**: 9502  
**Purpose**: Isolated sessions for external projects  
**Database**: Per-session at `/workspace/.swarm/sessions/{sessionId}/memory.db`  

The isolated TCP proxy provides:
- Fresh claude-flow instance per connection
- Complete session isolation
- Perfect for CI/CD, testing, or multi-tenant scenarios
- Automatic cleanup on disconnect

## Usage Patterns

### Local Development (Inside Container)

```bash
# Initialize agents
npx claude-flow@alpha goal init
npx claude-flow@alpha neural init

# Use via Claude Code
claude
> Use mcp__claude-flow__goal_create tool

# Direct CLI usage
npx claude-flow@alpha goal create --name "Build feature X"
```

### External Shared Access

```python
# Python example connecting to shared instance
import socket
import json

sock = socket.create_connection(('localhost', 9500))

# Initialize connection
init_msg = {
    "jsonrpc": "2.0",
    "id": "init",
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "clientInfo": {"name": "external-client"}
    }
}
sock.send(json.dumps(init_msg).encode() + b'\n')

# Use shared agents
goal_msg = {
    "jsonrpc": "2.0", 
    "id": "goal1",
    "method": "tools/call",
    "params": {
        "name": "goal_create",
        "arguments": {"name": "External goal"}
    }
}
sock.send(json.dumps(goal_msg).encode() + b'\n')
```

### External Isolated Access

```javascript
// Node.js example with isolated session
const net = require('net');

const client = net.createConnection({ port: 9502, host: 'localhost' });

client.on('connect', () => {
    // Each connection gets its own claude-flow instance
    const init = {
        jsonrpc: "2.0",
        id: "init",
        method: "initialize",
        params: {
            protocolVersion: "2024-11-05"
        }
    };
    client.write(JSON.stringify(init) + '\n');
});

// Session is completely isolated from other connections
```

## Configuration

### Environment Variables

```bash
# TCP Shared (Port 9500)
MCP_TCP_PORT=9500
MCP_TCP_COMMAND=npx  # Can override to use different MCP
MCP_TCP_ARGS="claude-flow@alpha mcp start"

# TCP Isolated (Port 9502)
CLAUDE_FLOW_TCP_PORT=9502
CLAUDE_FLOW_MAX_SESSIONS=10
```

### Monitoring

```bash
# Check service status
mcp-tcp-status        # Shared instance
cf-tcp-status         # Isolated proxy

# View logs
mcp-tcp-logs          # Shared instance logs
cf-tcp-logs           # Isolated sessions logs

# Test connections
cf-test-tcp           # Test isolated TCP
mcp-test-tcp          # Test shared TCP

# Health checks
curl http://localhost:9501/health  # Shared health
curl http://localhost:9503/health  # Isolated health
```

## Security Considerations

### Shared Instance (9500)
- Uses authentication tokens if configured
- Shares state with local operations
- Should be restricted to trusted clients

### Isolated Sessions (9502)
- Each session is sandboxed
- Automatic cleanup prevents resource leaks
- Suitable for untrusted or multi-tenant scenarios
- Session data persists only during connection

## Best Practices

1. **Use Local MCP** for development and testing within the container
2. **Use Shared TCP** when external tools need to coordinate with local work
3. **Use Isolated TCP** for:
   - CI/CD pipelines
   - External automation
   - Multi-user scenarios
   - Testing in isolation

## Troubleshooting

### Claude-Flow Not Starting

```bash
# Check if claude-flow is installed
npx claude-flow@alpha --version

# Check MCP server logs
mcp-tcp-logs | grep -i error

# Manually test
npx claude-flow@alpha mcp start
```

### TCP Connection Refused

```bash
# Verify services are running
supervisorctl status

# Check ports are listening
ss -tlnp | grep -E '9500|9502'

# Test direct connection
nc -zv localhost 9500
nc -zv localhost 9502
```

### Session Limits Reached

```bash
# Check active sessions
curl http://localhost:9503/health

# Increase limit in environment
CLAUDE_FLOW_MAX_SESSIONS=20
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Container                        │
│                                                          │
│  ┌─────────────────┐     ┌──────────────────────────┐   │
│  │  Claude Code    │────▶│   claude-flow (local)    │   │
│  │  (CLI in        │     │   via .mcp.json          │   │
│  │   container)    │     └──────────────────────────┘   │
│  └─────────────────┘                │                   │
│                                     │                   │
│                            ┌────────▼────────┐          │
│                            │ Shared Database │          │
│                            │ .swarm/memory.db│          │
│                            └────────▲────────┘          │
│                                     │                   │
│  ┌─────────────────┐     ┌──────────────────────────┐   │
│  │ TCP Server      │────▶│   claude-flow (shared)   │   │
│  │ Port 9500       │     │   Same instance as local │   │
│  └─────────────────┘     └──────────────────────────┘   │
│                                                          │
│  ┌─────────────────┐     ┌──────────────────────────┐   │
│  │ TCP Proxy       │────▶│   claude-flow (isolated) │   │
│  │ Port 9502       │     │   Per-session instances  │   │
│  └─────────────────┘     └──────────────────────────┘   │
│                                     │                   │
│                            ┌────────▼────────────────┐  │
│                            │ Session Databases       │  │
│                            │ .swarm/sessions/*/     │  │
│                            └──────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

This dual-access architecture ensures claude-flow can be used effectively both as an integrated development tool and as a service for external automation.