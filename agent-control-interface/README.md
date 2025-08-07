# Agent Control Interface

A TCP-based JSON-RPC 2.0 server that provides agent telemetry to the VisionFlow visualization system. This module aggregates data from multiple MCP (Model Context Protocol) sources and serves it to external clients for 3D visualization.

## Overview

The Agent Control Interface acts as a bridge between our container's MCP-based agent orchestration tools and the VisionFlow backend running in an external container. It provides real-time telemetry data while the remote Rust service handles all spatial topology calculations using GPU acceleration.

## Architecture

```
VisionFlow Client (172.18.0.12)
         ↓ TCP:9500
┌──────────────────────────────┐
│  Agent Control Interface      │
│  ┌────────────────────────┐  │
│  │ TCP Server (Port 9500) │  │ ← JSON-RPC 2.0
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │ Telemetry Aggregator   │  │ ← No position calculations
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │ MCP Bridge              │  │
│  │ • mcp-observability    │  │
│  │ • claude-flow          │  │
│  │ • ruv-swarm            │  │
│  └────────────────────────┘  │
└──────────────────────────────┘
```

### Key Design Principles

1. **Telemetry Only**: We provide raw agent data; the Rust service calculates positions
2. **Multi-Source Aggregation**: Combines data from multiple MCP tools
3. **Real-time Updates**: Telemetry refreshes every second
4. **Fault Tolerant**: Continues operating even if some MCP sources are unavailable

## Installation

```bash
cd /workspace/ext/agent-control-interface
npm install
```

## Quick Start

### Start the server:
```bash
./start.sh
```

### Run in debug mode:
```bash
./start.sh debug
```

### Run in background:
```bash
./start.sh background
```

### Test the connection:
```bash
./start.sh test
```

## API Reference

The server implements the JSON-RPC 2.0 protocol over TCP with newline-delimited JSON messages.

### Methods

#### `initialize`
Establishes a session with the server.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "method": "initialize",
  "params": {
    "protocolVersion": "0.1.0",
    "clientInfo": {
      "name": "rust-backend",
      "version": "1.0.0"
    }
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "serverInfo": {
      "name": "Agent Control System",
      "version": "1.0.0"
    },
    "protocolVersion": "0.1.0"
  }
}
```

#### `agents/list`
Returns a list of all active agents.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": "2",
  "method": "agents/list",
  "params": {}
}
```

#### `tools/call`
Executes various tools. Supported tools:
- `swarm.initialize` - Initialize a new swarm
- `visualization.snapshot` - Get telemetry snapshot
- `metrics.get` - Get system metrics

**Example - Initialize Swarm:**
```json
{
  "jsonrpc": "2.0",
  "id": "3",
  "method": "tools/call",
  "params": {
    "name": "swarm.initialize",
    "arguments": {
      "topology": "hierarchical",
      "agentTypes": ["coordinator", "coder", "tester"]
    }
  }
}
```

## Data Models

### Agent
```typescript
interface Agent {
  id: string;
  type: "coordinator" | "coder" | "analyst" | "tester" | ...;
  name: string;
  status: "active" | "idle" | "busy";
  health: number; // 0-100
  capabilities: string[];
  metrics: {
    tasksCompleted: number;
    tasksActive: number;
    successRate: number; // 0.0-1.0
    cpuUsage: number; // 0.0-1.0
    memoryUsage: number; // 0.0-1.0
  };
}
```

### Connection
```typescript
interface Connection {
  id: string;
  from: string; // Agent ID
  to: string; // Agent ID
  messageCount: number;
  lastActivity: string; // ISO 8601 timestamp
}
```

## Testing

### Run automated tests:
```bash
node tests/test-client.js
```

### Interactive testing:
```bash
node tests/test-client.js localhost 9500 interactive
```

### Test from external container:
```bash
# From the VisionFlow container
node test-client.js 172.18.0.10 9500
```

## Configuration

Environment variables:
- `AGENT_CONTROL_PORT` - Server port (default: 9500)
- `LOG_LEVEL` - Logging level: debug, info, warn, error (default: info)
- `NODE_ENV` - Environment: development, production (default: development)

## MCP Integration

The interface automatically detects and uses available MCP tools:

1. **mcp-observability** - Primary source for agent telemetry (included in `mcp-observability/` subdirectory)
2. **claude-flow** - Advanced swarm orchestration (from `/app/claude-flow` if available)
3. **ruv-swarm** - WASM-optimized agent management (via MCP protocol if available)

The module includes mcp-observability as a self-contained component, making it fully portable. If other MCP tools are unavailable, the system falls back to mock data for testing.

## Troubleshooting

### Port already in use
```bash
# Find process using port 9500
lsof -i :9500

# Kill the process
kill -9 <PID>
```

### Connection refused from external container
Ensure the server is binding to all interfaces:
```bash
# Check binding address (should be 0.0.0.0:9500)
netstat -tlnp | grep 9500
```

### MCP tools not found
The module includes mcp-observability. If it's not working:
```bash
# Check mcp-observability is present
ls -la mcp-observability/

# Install its dependencies
cd mcp-observability && npm install

# For other tools, ensure they're available
which claude-flow
```

## Development

### Project Structure
```
agent-control-interface/
├── src/
│   ├── index.js              # Main server entry point
│   ├── json-rpc-handler.js   # JSON-RPC protocol handler
│   ├── telemetry-aggregator.js # Data aggregation logic
│   ├── mcp-bridge.js         # MCP tool integration
│   └── logger.js             # Logging utility
├── mcp-observability/        # Bundled MCP observability tool
│   ├── src/                  # MCP observability source
│   └── package.json          # MCP dependencies
├── tests/
│   └── test-client.js        # Test client implementation
├── logs/                     # Server logs
├── .env                      # Environment configuration
├── .env.example              # Configuration template
├── start.sh                  # Startup script
├── stop.sh                   # Shutdown script
├── package.json              # Project dependencies
├── README.md                 # This file
└── IMPLEMENTATION.md         # Technical documentation
```

### Adding New MCP Sources

To add a new MCP tool as a data source:

1. Update `mcp-bridge.js`:
```javascript
async initializeMCPTools() {
  // Add your tool
  this.mcpClients.set('your-tool', {
    available: await this.checkToolAvailability('mcp__your_tool__init'),
    prefix: 'mcp__your_tool__'
  });
}
```

2. Add fetcher in `telemetry-aggregator.js`:
```javascript
async fetchFromYourTool() {
  try {
    const result = await this.mcpBridge.callMCPTool('your-tool', 'agent.list');
    return this.normalizeAgentData(result.agents || [], 'your-tool');
  } catch (error) {
    return null;
  }
}
```

## Integration with Container Build

This module is designed to be side-loadable for testing but can be integrated into the main container build:

1. Add to Dockerfile:
```dockerfile
COPY ext/agent-control-interface /app/agent-control-interface
RUN cd /app/agent-control-interface && npm install --production
```

2. Add to supervisor config:
```ini
[program:agent-control]
command=/app/agent-control-interface/start.sh
directory=/app/agent-control-interface
autostart=true
autorestart=true
stderr_logfile=/var/log/agent-control.err.log
stdout_logfile=/var/log/agent-control.out.log
```

## License

MIT