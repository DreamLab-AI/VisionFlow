# VisionFlow: Enhanced Multi-Agent Visualization Update

## 🎉 New Multi-MCP Agent Visualization System

We've added comprehensive support for visualizing AI agents across multiple MCP (Model Context Protocol) servers, creating beautiful force-directed 3D graphs of your entire AI swarm ecosystem.

### What's New

#### 🌟 Multi-Server Agent Discovery
- **Claude Flow Integration**: Direct TCP connection to Claude Flow MCP server (port 9500)
- **RuvSwarm Support**: Monitors WASM-accelerated swarm agents  
- **DAA Agents**: Tracks Decentralized Autonomous Agents
- **Unified View**: All agents visualized in a single cohesive 3D graph

#### 🎨 Advanced Topology Layouts
Six different force-directed layout algorithms:
1. **Hierarchical**: Tree-like command structure with layers
2. **Mesh**: Full connectivity for collaborative networks
3. **Ring**: Circular pipeline for sequential processing
4. **Star**: Central hub with radiating workers
5. **Hybrid**: Multiple groups with inter-group connections
6. **Adaptive**: Self-organizing by agent importance

#### ⚡ Real-Time Performance Metrics
- **Live Agent Status**: Active, Processing, Training, Coordinating states
- **Resource Monitoring**: CPU, memory, task completion rates
- **Health Scores**: Visual indication of agent performance
- **Connection Strength**: Relationship visualization between agents

#### 🎭 Visual Enhancements
- **Color Coding**: Different colors for each MCP server type
- **Dynamic Sizing**: Node size based on agent importance
- **Shape Variety**: Different shapes for agent roles (coordinators, workers, etc.)
- **Visual Effects**: Glow intensity for activity, pulse frequency for CPU usage
- **Connection Types**: Different edge styles for coordination, data flow, task delegation

### Technical Implementation

#### New Components

1. **Multi-MCP Discovery Service** (`services/multi_mcp_discovery.rs`)
   - Discovers agents from all MCP servers
   - Unified agent representation
   - Connection detection based on capabilities
   - Health monitoring and retry logic

2. **Topology Visualization Engine** (`services/topology_visualization_engine.rs`)
   - Force-directed graph physics
   - Multiple layout algorithms
   - Convergence detection
   - Visual hint generation

3. **Real MCP Integration Bridge** (`services/real_mcp_integration_bridge.rs`)
   - Direct TCP connections to MCP servers
   - JSON-RPC protocol implementation
   - MCP tool calls (agent_list, neural_status, etc.)
   - Fallback to simulated agents

4. **Multi-MCP Visualization Actor** (`actors/multi_mcp_visualization_actor.rs`)
   - Coordinates discovery across servers
   - Manages layouts and updates
   - 10Hz update rate for smooth visualization
   - Integration with existing GraphServiceActor

#### Enhanced TCP Communication

- **Direct Docker-to-Docker**: TCP instead of WebSocket for container communication
- **Line-Delimited JSON-RPC**: Efficient protocol for MCP communication
- **Request Correlation**: Proper request/response tracking
- **Connection Pooling**: Reuse connections for efficiency

### Configuration

Add to your `settings.yaml`:

```yaml
mcp_discovery:
  claude_flow:
    host: localhost
    port: 9500
    interval_ms: 5000
    timeout_ms: 2000
  ruv_swarm:
    host: localhost
    port: 9501
    interval_ms: 5000
  daa:
    host: localhost
    port: 9502
    interval_ms: 10000

visualization:
  topology:
    type: hierarchical  # or mesh, ring, star, hybrid, adaptive
    force_iterations: 100
    spring_strength: 0.1
    repulsion_strength: 500
    damping: 0.9
```

### API Endpoints

New REST endpoints:
- `GET /api/mcp/agents` - List all discovered agents
- `GET /api/mcp/topology` - Get current topology configuration
- `POST /api/mcp/topology` - Change topology type
- `GET /api/mcp/stats` - Get discovery statistics
- `POST /api/mcp/refresh` - Trigger manual discovery

### Performance Improvements

- **Parallel Discovery**: All MCP servers queried simultaneously
- **Differential Updates**: Only changed data transmitted
- **Layout Caching**: Recalculate only on topology changes
- **Async Processing**: Non-blocking discovery and updates
- **GPU Acceleration**: Physics calculations utilize CUDA when available

### Visual Examples

#### Hierarchical Layout
Perfect for command structures:
```
      [Coordinator]
     /      |      \
[Worker1] [Worker2] [Worker3]
    |        |         |
[Task1]   [Task2]   [Task3]
```

#### Mesh Layout
Full connectivity for collaboration:
```
[Agent1]---[Agent2]
   |\     /|
   | \   / |
   |  \ /  |
   |   X   |
   |  / \  |
   | /   \ |
   |/     \|
[Agent3]---[Agent4]
```

#### Star Layout
Central hub coordination:
```
     [Worker1]
         |
[Worker2]-[Hub]-[Worker3]
         |
     [Worker4]
```

### Usage Example

The system starts automatically when you launch VisionFlow:

```bash
# Agents are discovered automatically
# View in browser at http://localhost:4000

# Change topology via API
curl -X POST http://localhost:4000/api/mcp/topology \
  -H "Content-Type: application/json" \
  -d '{"type": "mesh"}'

# Get current agents
curl http://localhost:4000/api/mcp/agents
```

### Future Enhancements

Planned features:
- Agent interaction recording and replay
- Predictive performance analytics
- Auto-scaling based on load
- Custom topology definitions
- VR mode for agent management
- Distributed consensus visualization

### Migration Guide

No breaking changes! The new system integrates seamlessly:

1. Update your Docker images
2. Ensure MCP servers are running in multi-agent-container
3. Access the visualization at the same URL
4. Optionally configure topology preferences

### Troubleshooting

#### No agents appearing?
- Check MCP servers are running: `docker exec multi-agent-container ps aux | grep mcp`
- Verify TCP port 9500 is accessible
- Check logs: `docker logs visionflow-container`

#### Layout looks cluttered?
- Try different topology types
- Adjust physics parameters in settings
- Increase force_iterations for better convergence

#### Performance issues?
- Reduce discovery_interval_ms
- Lower force_iterations
- Enable GPU acceleration if available

### Credits

This update integrates with:
- Claude Flow MCP protocol
- RuvSwarm distributed agents
- DAA autonomous systems

Built with the power of force-directed graphs and real-time discovery!