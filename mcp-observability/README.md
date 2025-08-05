# MCP Bot Observability Server

A comprehensive Model Context Protocol (MCP) server for bot swarm observability with spring-physics directed graph visualization support.

## Features

- **🤖 Agent Management**: Create, update, and monitor agents with spring-physics positioning
- **🌐 Swarm Coordination**: Initialize and manage swarms with various topologies (hierarchical, mesh, ring, star)
- **📨 Message Flow Tracking**: Real-time message tracking with spring force calculations
- **📊 Performance Monitoring**: Comprehensive metrics collection and bottleneck detection
- **🎨 3D Visualization Support**: Spring-physics engine for force-directed graph layouts
- **🧠 Neural Pattern Learning**: Train and predict optimal coordination patterns
- **💾 Persistent Memory**: Store and retrieve swarm state with TTL support

## Quick Start

### Installation

```bash
cd /workspace/mcp-observability
npm install
```

### Running the Server

```bash
npm start
```

The server will start on stdio, ready to accept MCP protocol connections.

## Architecture

```
┌─────────────────────────────────────┐
│        MCP Client (Claude)          │
└──────────────┬──────────────────────┘
               │ stdio (JSON-RPC 2.0)
┌──────────────▼──────────────────────┐
│      MCP Observability Server       │
├─────────────────────────────────────┤
│ • Agent Manager                     │
│ • Physics Engine (Spring-based)     │
│ • Message Flow Tracker              │
│ • Performance Monitor               │
│ • Neural Pattern Learning           │
│ • Memory Store                      │
└─────────────────────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    VisionFlow 3D Visualization      │
│    (Spring-Physics Directed Graph)  │
└─────────────────────────────────────┘
```

## Available Tools

### Agent Management (6 tools)
- `agent.create` - Create new agent with physics positioning
- `agent.update` - Update agent state and performance
- `agent.metrics` - Get detailed agent metrics
- `agent.list` - List agents with filtering
- `agent.remove` - Remove agent from swarm
- `agent.spawn` - Spawn multiple agents optimally

### Swarm Coordination (4 tools)
- `swarm.initialize` - Initialize swarm with topology
- `swarm.status` - Get comprehensive swarm status
- `swarm.monitor` - Real-time swarm monitoring
- `swarm.reconfigure` - Change topology and physics

### Message Flow (6 tools)
- `message.send` - Send messages with spring forces
- `message.flow` - Get message flow visualization data
- `message.acknowledge` - Acknowledge message receipt
- `message.stats` - Communication statistics
- `message.broadcast` - Broadcast to multiple agents
- `message.patterns` - Analyze communication patterns

### Performance Monitoring (5 tools)
- `performance.analyze` - Analyze with bottleneck detection
- `performance.optimize` - Suggest physics optimizations
- `performance.report` - Comprehensive performance report
- `performance.metrics` - Current system metrics
- `performance.benchmark` - Run performance tests

### Visualization (5 tools)
- `visualization.snapshot` - Get current 3D state
- `visualization.animate` - Generate animation sequences
- `visualization.layout` - Apply layout patterns
- `visualization.highlight` - Highlight agents/connections
- `visualization.camera` - Camera position recommendations

### Neural Learning (5 tools)
- `neural.train` - Train coordination patterns
- `neural.predict` - Predict optimal patterns
- `neural.status` - Neural network status
- `neural.patterns` - Get recognized patterns
- `neural.optimize` - Optimize swarm configuration

### Memory Management (7 tools)
- `memory.store` - Store data with TTL
- `memory.retrieve` - Retrieve stored data
- `memory.list` - List stored keys
- `memory.delete` - Delete stored data
- `memory.persist` - Save to disk
- `memory.search` - Search by content
- `memory.stats` - Memory usage statistics

## Usage Examples

### Initialize a Hierarchical Swarm

```json
{
  "method": "tools/call",
  "params": {
    "name": "swarm.initialize",
    "arguments": {
      "topology": "hierarchical",
      "physicsConfig": {
        "springStrength": 0.1,
        "damping": 0.95,
        "linkDistance": 8.0
      },
      "agentConfig": {
        "coordinatorCount": 1,
        "workerTypes": [
          { "type": "coder", "count": 3 },
          { "type": "tester", "count": 2 },
          { "type": "analyst", "count": 1 }
        ]
      }
    }
  }
}
```

### Send a Message with Physics

```json
{
  "method": "tools/call",
  "params": {
    "name": "message.send",
    "arguments": {
      "from": "coordinator-001",
      "to": ["coder-001", "coder-002"],
      "type": "task",
      "priority": 3,
      "content": {
        "task": "implement-feature",
        "deadline": "2024-01-20"
      }
    }
  }
}
```

### Get Visualization Snapshot

```json
{
  "method": "tools/call",
  "params": {
    "name": "visualization.snapshot",
    "arguments": {
      "includePositions": true,
      "includeVelocities": true,
      "includeForces": false,
      "includeConnections": true
    }
  }
}
```

## Spring Physics Configuration

The physics engine uses the following parameters:

```javascript
{
  springStrength: 0.1,      // Force between connected nodes
  linkDistance: 8.0,        // Ideal distance between connected nodes
  damping: 0.95,            // Velocity damping factor
  nodeRepulsion: 500.0,     // Force preventing node overlap
  gravityStrength: 0.02,    // Central gravity to prevent drift
  maxVelocity: 2.0,         // Maximum node velocity
  
  // Hive-mind specific
  queenGravity: 0.05,       // Attraction to coordinator nodes
  swarmCohesion: 0.08,      // Force keeping swarm together
  hierarchicalForce: 0.03,  // Force for hierarchy maintenance
  
  // Message flow
  messageAttraction: 0.15,  // Temporary attraction on message
  communicationDecay: 0.98  // Decay rate for message forces
}
```

## Agent Types

- **👑 queen**: Top-level orchestrator (largest node)
- **🎯 coordinator**: Team coordination and management
- **🏗️ architect**: System design and planning
- **⚡ specialist**: Domain-specific expertise
- **💻 coder**: Implementation and development
- **🔍 researcher**: Information gathering and analysis
- **🧪 tester**: Quality assurance and validation
- **📊 analyst**: Data analysis and metrics
- **⚙️ optimizer**: Performance optimization
- **👁️ monitor**: System monitoring and health

## Performance Considerations

- Supports 1000+ agents at 60 FPS
- Binary position updates (28 bytes/agent)
- Efficient spring physics calculations
- Level-of-detail for large swarms
- Automatic performance optimization

## Docker Integration

To integrate with the Docker agent project:

1. Mount the MCP server in your Docker container:
```dockerfile
COPY --from=mcp-observability /workspace/mcp-observability /app/mcp-observability
```

2. Set environment variables:
```bash
MCP_OBSERVABILITY_PORT=3100
MCP_PHYSICS_UPDATE_RATE=60
MCP_MAX_AGENTS=1000
```

3. Start the server in your container:
```bash
cd /app/mcp-observability && npm start
```

## Memory Sections

The memory system is organized into sections:

- **global**: General purpose storage
- **swarm**: Swarm-specific data
- **agents**: Agent state and history
- **patterns**: Learned coordination patterns
- **performance**: Performance metrics history
- **coordination**: Coordination events and decisions

## Development

### Running Tests
```bash
npm test
```

### Development Mode
```bash
npm run dev
```

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Support

For issues and questions, please open an issue in the repository.