# Multi-MCP Agent Visualization System

## Overview

The Multi-MCP Agent Visualization System provides comprehensive discovery, monitoring, and visualization of AI agents across multiple MCP (Model Context Protocol) servers including Claude Flow, RuvSwarm, and DAA (Decentralized Autonomous Agents). It creates beautiful force-directed 3D graphs showing agent relationships, topologies, and real-time metrics.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Multi-Agent Container                    │
│                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────┐ │
│  │  Claude Flow    │  │   RuvSwarm      │  │   DAA    │ │
│  │  MCP Server     │  │   MCP Server    │  │  Server  │ │
│  │  (Port 9500)    │  │  (Port 9501)    │  │  (9502)  │ │
│  └────────┬────────┘  └────────┬────────┘  └────┬─────┘ │
│           │                     │                 │       │
│           └──────────┬──────────┘                 │       │
│                      ▼                            ▼       │
│  ┌──────────────────────────────────────────────────────┐│
│  │         Real MCP Integration Bridge                  ││
│  │  - Connects to actual MCP servers via TCP            ││
│  │  - Discovers agents using MCP tools                  ││
│  │  - Monitors neural status and swarm topology         ││
│  └─────────────────────┬────────────────────────────────┘│
│                        ▼                                  │
│  ┌──────────────────────────────────────────────────────┐│
│  │      Multi-MCP Discovery Service                     ││
│  │  - Unified agent representation                      ││
│  │  - Connection mapping                                ││
│  │  - Performance metrics aggregation                   ││
│  └─────────────────────┬────────────────────────────────┘│
│                        ▼                                  │
│  ┌──────────────────────────────────────────────────────┐│
│  │    Topology Visualization Engine                     ││
│  │  - Force-directed graph layouts                      ││
│  │  - 6 topology types (Hierarchical, Mesh, Ring, etc.) ││
│  │  - Physics simulation with convergence               ││
│  └─────────────────────┬────────────────────────────────┘│
└────────────────────────┼──────────────────────────────────┘
                         │ TCP Port 9500
                         ▼
┌──────────────────────────────────────────────────────────┐
│                 VisionFlow Container                      │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐│
│  │        Multi-MCP Visualization Actor                 ││
│  │  - Coordinates discovery across all MCP servers      ││
│  │  - Manages layouts and visual hints                  ││
│  │  - Streams updates at 10Hz                           ││
│  └─────────────────────┬────────────────────────────────┘│
│                        ▼                                  │
│  ┌──────────────────────────────────────────────────────┐│
│  │         ClaudeFlowActorTcp (Enhanced)                ││
│  │  - TCP communication with MCP servers                ││
│  │  - Line-delimited JSON-RPC protocol                  ││
│  │  - Request/response correlation                      ││
│  └─────────────────────┬────────────────────────────────┘│
│                        ▼                                  │
│  ┌──────────────────────────────────────────────────────┐│
│  │          GraphServiceActor                           ││
│  │  - Manages 3D graph state                            ││
│  │  - GPU-accelerated physics                           ││
│  │  - Streams to clients via WebSocket                  ││
│  └──────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
                         │ WebSocket
                         ▼
                  [Browser Client]
```

## Key Components

### 1. Multi-MCP Discovery Service (`multi_mcp_discovery.rs`)

Discovers and monitors agents across all MCP servers:

- **Unified Agent Model**: Single representation for agents from different servers
- **Connection Detection**: Automatically detects agent relationships based on capabilities
- **Health Monitoring**: Tracks connection status and retry logic
- **Configurable Discovery**: Per-server intervals and timeouts

#### Agent Data Structure:
```rust
UnifiedAgent {
    id: String,
    name: String,
    server: McpServerType,
    agent_type: String,
    status: AgentStatus,
    capabilities: Vec<String>,
    position: Position3D,
    metrics: AgentMetrics,
    connections: Vec<AgentConnection>,
}
```

### 2. Topology Visualization Engine (`topology_visualization_engine.rs`)

Creates beautiful 3D layouts with force-directed physics:

#### Topology Types:

1. **Hierarchical**: Tree-like structure with layers
   - Coordinators at top levels
   - Workers in lower layers
   - Clear command hierarchy

2. **Mesh**: Full connectivity network
   - Every agent can communicate with every other
   - Optimal for collaborative tasks
   - High fault tolerance

3. **Ring**: Circular pipeline
   - Sequential processing
   - Efficient for streaming tasks
   - Bidirectional communication

4. **Star**: Central hub coordination
   - One coordinator, many workers
   - Simple and efficient
   - Low coordination overhead

5. **Hybrid**: Multiple groups with different strategies
   - Groups by server type
   - Inter-group connections
   - Flexible organization

6. **Adaptive**: Dynamic based on agent importance
   - Central placement for critical agents
   - Concentric rings by performance
   - Self-organizing

#### Physics Simulation:

- **Spring Forces**: Attraction between connected agents
- **Repulsion Forces**: Prevent overlapping with distance decay
- **Center Gravity**: Keep graph centered
- **Damping**: Stabilize movement
- **Convergence Detection**: Stop when stable

### 3. Real MCP Integration Bridge (`real_mcp_integration_bridge.rs`)

Connects to actual MCP servers running in the container:

- **TCP Connections**: Direct TCP to MCP servers
- **MCP Protocol**: JSON-RPC with proper initialization
- **Tool Calls**: Uses actual MCP tools (agent_list, neural_status, etc.)
- **Fallback**: Generates simulated agents if servers unavailable

#### MCP Tools Used:
- `agent_list`: Get list of active agents
- `neural_status`: Neural network training status
- `swarm_status`: Swarm topology and performance
- `agent_metrics`: CPU, memory, task statistics

### 4. Multi-MCP Visualization Actor (`multi_mcp_visualization_actor.rs`)

Actix actor that coordinates the entire system:

- **Discovery Coordination**: Manages discovery service
- **Layout Updates**: Triggers topology recalculation
- **Performance**: 10Hz update rate, 5-second discovery cycle
- **State Management**: Maintains agent cache and layouts

## Visual Features

### Node Visualization:

- **Color**: Based on server type
  - Blue (#4287f5): Claude Flow agents
  - Green (#42f554): RuvSwarm agents
  - Red (#f54242): DAA agents
  - Orange (#f5a442): Custom agents

- **Size**: Based on importance
  - Health score contribution
  - Tasks completed
  - Minimum and maximum bounds

- **Shape**: Based on agent type
  - Diamond: Coordinators
  - Sphere: Researchers
  - Cube: Coders
  - Pyramid: Testers
  - Octahedron: Optimizers

- **Effects**:
  - Glow intensity: Activity level
  - Pulse frequency: CPU usage
  - Connection width: Relationship strength

### Edge Visualization:

- **Connection Types**:
  - Coordination: Thick blue lines
  - DataFlow: Medium green lines
  - TaskDelegation: Thin yellow lines
  - Knowledge: Dotted purple lines
  - Consensus: Dashed red lines

## TCP Communication Protocol

The system uses TCP instead of WebSocket for inter-container communication:

### VisionFlow → Multi-Agent Container:
- Port: 9500
- Protocol: Line-delimited JSON-RPC
- Authentication: MCP initialization handshake

### Message Format:
```json
{
  "jsonrpc": "2.0",
  "id": "uuid",
  "method": "tools/call",
  "params": {
    "name": "agent_list",
    "arguments": {
      "filter": "active"
    }
  }
}
```

### Response Format:
```json
{
  "jsonrpc": "2.0",
  "id": "uuid",
  "result": {
    "agents": [...]
  }
}
```

## Performance Optimization

1. **Differential Updates**: Only send changed data
2. **Batch Discovery**: Discover all servers in parallel
3. **Async Processing**: Non-blocking discovery and updates
4. **Connection Pooling**: Reuse TCP connections
5. **Layout Caching**: Only recalculate on topology change
6. **GPU Acceleration**: Physics calculations on GPU when available

## Configuration

### Discovery Settings:
```yaml
discovery:
  claude_flow:
    interval_ms: 5000
    timeout_ms: 2000
    retry_attempts: 3
  ruv_swarm:
    interval_ms: 5000
    timeout_ms: 2000
    retry_attempts: 3
  daa:
    interval_ms: 10000
    timeout_ms: 3000
    retry_attempts: 5
```

### Layout Configuration:
```yaml
layout:
  width: 2000
  height: 2000
  depth: 1000
  layer_spacing: 300
  node_spacing: 150
  force_iterations: 100
```

### Physics Parameters:
```yaml
physics:
  spring_strength: 0.1
  repulsion_strength: 500
  damping: 0.9
  max_velocity: 10
  convergence_threshold: 0.01
```

## API Endpoints

### REST API:

- `GET /api/mcp/agents` - Get all discovered agents
- `GET /api/mcp/topology` - Get current topology
- `POST /api/mcp/topology` - Change topology type
- `GET /api/mcp/stats` - Get discovery statistics
- `POST /api/mcp/discover` - Trigger manual discovery

### WebSocket Events (Client-facing):

- `agent_discovered` - New agent found
- `agent_updated` - Agent status changed
- `topology_changed` - Topology reconfigured
- `performance_update` - Metrics update

## Usage Example

```rust
// Start the visualization system
let graph_service = GraphServiceActor::new(...);
let mcp_viz = MultiMcpVisualizationActor::new(graph_service.clone());

// The actor automatically:
// 1. Connects to MCP servers
// 2. Discovers agents every 5 seconds
// 3. Calculates force-directed layouts
// 4. Streams updates to graph service
// 5. Graph service streams to browser clients

// Change topology
mcp_viz.send(SetTopology {
    topology: TopologyType::Mesh
}).await;

// Get current state
let state = mcp_viz.send(GetVisualizationState).await?;
println!("Agents: {}, Topology: {:?}", state.agent_count, state.topology);
```

## Benefits

1. **Real-time Monitoring**: See all AI agents across containers
2. **Beautiful Visualization**: Force-directed 3D graphs
3. **Performance Insights**: CPU, memory, task metrics
4. **Topology Flexibility**: 6 different layout algorithms
5. **Scalable**: Handles hundreds of agents efficiently
6. **Container-native**: Direct TCP for low latency
7. **Fault Tolerant**: Automatic reconnection and fallbacks

## Future Enhancements

1. **Agent Interaction Recording**: Store message flows
2. **Predictive Analytics**: Forecast agent performance
3. **Auto-scaling**: Spawn/terminate agents based on load
4. **Custom Topologies**: User-defined layout algorithms
5. **3D VR Support**: Immersive agent management
6. **Distributed Consensus**: Byzantine fault tolerance
7. **Neural Training Visualization**: Show learning progress

## Troubleshooting

### No Agents Discovered:
- Check MCP servers are running: `ps aux | grep mcp`
- Verify TCP ports are open: `netstat -tlnp | grep 9500`
- Check logs: `/app/mcp-logs/`

### Layout Issues:
- Increase force_iterations for better convergence
- Adjust spring_strength and repulsion_strength
- Try different topology types

### Performance:
- Reduce discovery_interval for less frequent updates
- Lower force_iterations for faster layout
- Enable GPU acceleration if available

## Summary

The Multi-MCP Agent Visualization System provides a comprehensive solution for visualizing and monitoring AI agents across multiple MCP servers. It combines real-time discovery, beautiful force-directed layouts, and efficient TCP communication to create an engaging and informative visualization of your AI swarm.