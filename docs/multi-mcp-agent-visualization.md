# Multi-MCP Agent Visualization System

A comprehensive agent discovery, monitoring, and visualization system for multiple MCP (Model Context Protocol) servers including Claude Flow, RuvSwarm, and DAA (Decentralized Autonomous Agents).

## 🚀 Overview

This system provides:

- **Multi-Server Discovery**: Automatically discovers agents across Claude Flow, RuvSwarm, DAA, and custom MCP servers
- **Real-time Monitoring**: Continuous monitoring of agent status, performance, and interactions
- **Topology Visualization**: Supports hierarchical, mesh, ring, and star topologies with 3D positioning
- **Performance Analytics**: Comprehensive metrics, bottleneck detection, and trend analysis  
- **WebSocket Streaming**: Real-time updates to VisionFlow graph renderer
- **Neural Tracking**: Monitors neural agent learning, adaptation, and cognitive patterns

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    VisionFlow Client                         │
│                 (Graph Renderer)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │ WebSocket
┌─────────────────────▼───────────────────────────────────────┐
│            MultiMcpVisualizationActor                       │
│  ┌─────────────────┬─────────────────┬─────────────────┐    │
│  │   Discovery     │   Topology      │   Performance   │    │
│  │   Service       │   Engine        │   Analytics     │    │
│  └─────────────────┴─────────────────┴─────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │ TCP/JSON-RPC
┌─────────────────────▼───────────────────────────────────────┐
│                 MCP Servers                                 │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │ Claude Flow │  RuvSwarm   │     DAA     │   Custom    │  │
│  │   :9500     │   :9501     │   :9502     │   :9503     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Services

#### 1. Multi-MCP Agent Discovery (`multi_mcp_agent_discovery.rs`)
- Discovers agents across multiple MCP servers
- Maintains connection health and retry logic
- Provides unified agent status aggregation
- Supports server-specific configurations

#### 2. Topology Visualization Engine (`topology_visualization_engine.rs`)
- Generates 3D layouts for different topology types
- Calculates agent positioning and connections
- Provides performance metrics and optimization suggestions
- Supports dynamic layout updates

#### 3. Enhanced Visualization Protocol (`agent_visualization_protocol.rs`)
- Defines comprehensive data structures for multi-MCP environments
- Supports differential updates for performance
- Handles neural agent data and learning events
- Provides filtering and subscription management

#### 4. Multi-MCP Visualization Actor (`multi_mcp_visualization_actor.rs`)
- Coordinates all visualization components
- Manages WebSocket client connections
- Provides real-time streaming of agent data
- Handles legacy compatibility with existing systems

## 📊 Data Model

### Agent Status
```rust
pub struct MultiMcpAgentStatus {
    pub agent_id: String,
    pub swarm_id: String,
    pub server_source: McpServerType,
    pub name: String,
    pub agent_type: String,
    pub status: String,
    pub capabilities: Vec<String>,
    pub metadata: AgentExtendedMetadata,
    pub performance: AgentPerformanceData,
    pub neural_info: Option<NeuralAgentData>,
    pub created_at: i64,
    pub last_active: i64,
}
```

### Topology Types

#### Hierarchical
- Tree-like structure with coordinators at each level
- Optimal for command-and-control scenarios
- High coordination efficiency, moderate fault tolerance

#### Mesh
- Full or partial connectivity between agents
- Best for collaborative workflows
- High fault tolerance, moderate coordination overhead

#### Ring
- Circular arrangement with neighbor connections
- Good for pipeline processing
- Low coordination overhead, limited fault tolerance

#### Star
- Central hub with spokes to all agents
- Simple coordination model
- High efficiency, single point of failure

### Performance Metrics

```rust
pub struct AgentPerformanceData {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub health_score: f32,
    pub activity_level: f32,
    pub tasks_active: u32,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub success_rate: f32,
    pub token_usage: u64,
    pub token_rate: f32,
    pub response_time_ms: f32,
    pub throughput: f32,
}
```

## 🌐 WebSocket API

### Connection
```
ws://localhost:8080/api/multi-mcp/ws
```

### Message Types

#### Client to Server
```json
{
  "action": "configure",
  "data": {
    "subscription_filters": {
      "server_types": ["claude_flow", "ruv_swarm"],
      "agent_types": ["coordinator", "coder"],
      "include_performance": true,
      "include_neural": true
    },
    "performance_mode": "normal"
  }
}
```

```json
{
  "action": "request_discovery"
}
```

```json
{
  "action": "request_agents"
}
```

#### Server to Client
```json
{
  "type": "discovery",
  "timestamp": 1692454800000,
  "servers": [...],
  "total_agents": 12,
  "swarms": [...]
}
```

```json
{
  "type": "multi_agent_update", 
  "timestamp": 1692454800000,
  "agents": [...],
  "differential_updates": [...]
}
```

```json
{
  "type": "performance_analysis",
  "timestamp": 1692454800000,
  "global_metrics": {...},
  "bottlenecks": [...],
  "optimization_suggestions": [...]
}
```

## 🛠️ Usage

### Basic Setup

```rust
use visionflow_ext::services::multi_mcp_agent_discovery::MultiMcpAgentDiscovery;
use visionflow_ext::actors::MultiMcpVisualizationActor;

// Create discovery service
let discovery = MultiMcpAgentDiscovery::new();
discovery.initialize_default_servers().await;
discovery.start_discovery().await;

// Create visualization actor
let graph_service = GraphServiceActor::new().start();
let viz_actor = MultiMcpVisualizationActor::new(graph_service).start();
```

### Adding Custom MCP Server

```rust
use visionflow_ext::services::multi_mcp_agent_discovery::McpServerConfig;
use visionflow_ext::services::agent_visualization_protocol::McpServerType;

let config = McpServerConfig {
    server_id: "my-custom-server".to_string(),
    server_type: McpServerType::Custom("my-type".to_string()),
    host: "custom-host".to_string(),
    port: 9504,
    enabled: true,
    discovery_interval_ms: 3000,
    timeout_ms: 10000,
    retry_attempts: 3,
};

discovery.add_server(config).await;
```

### Topology Visualization

```rust
use visionflow_ext::services::topology_visualization_engine::{
    TopologyVisualizationEngine, TopologyConfig, TopologyType
};

let config = TopologyConfig {
    topology_type: TopologyType::Hierarchical,
    layout_params: Default::default(),
    visual_params: Default::default(),
};

let mut engine = TopologyVisualizationEngine::new(config);
let layout = engine.generate_layout(
    "my-swarm".to_string(),
    agents,
    TopologyType::Mesh,
);
```

## 🚀 Running the Demo

1. **Start the demo server:**
```bash
cargo run --example multi_mcp_integration_demo
```

2. **Open browser:**
```
http://127.0.0.1:8080
```

3. **Connect WebSocket and explore:**
   - Click "Connect WebSocket" to establish connection
   - Use "Request Discovery" to get server/agent info
   - Monitor real-time updates in the log panel

## 🔧 Configuration

### Environment Variables

```bash
# Claude Flow MCP server
CLAUDE_FLOW_HOST=localhost
MCP_TCP_PORT=9500

# RuvSwarm MCP server  
RUV_SWARM_HOST=localhost
RUV_SWARM_PORT=9501

# DAA MCP server
DAA_HOST=localhost
DAA_PORT=9502

# Custom MCP server
CUSTOM_MCP_HOST=localhost
CUSTOM_MCP_PORT=9503
```

### Performance Modes

- **HighFrequency**: 60Hz updates - for active monitoring dashboards
- **Normal**: 10Hz updates - default balanced mode
- **LowFrequency**: 1Hz updates - for overview displays
- **OnDemand**: Manual updates only - minimal resource usage

### Subscription Filters

```rust
pub struct SubscriptionFilters {
    pub server_types: Vec<McpServerType>,
    pub agent_types: Vec<String>,
    pub swarm_ids: Vec<String>,
    pub include_performance: bool,
    pub include_neural: bool,
    pub include_topology: bool,
}
```

## 📈 Performance Optimization

### Discovery Service
- Configurable polling intervals per server
- Retry logic with exponential backoff
- Connection pooling and health monitoring
- Differential updates to minimize bandwidth

### Visualization Engine
- Cached layout calculations
- GPU-optimized positioning algorithms
- Level-of-detail rendering for large swarms
- Efficient collision detection for force-directed layouts

### WebSocket Streaming
- Message filtering and compression
- Rate limiting and backpressure handling
- Client-specific update frequencies
- Efficient JSON serialization

## 🔮 Integration with VisionFlow

The system integrates seamlessly with the existing VisionFlow graph renderer:

1. **Data Format**: Uses VisionFlow-compatible JSON structures
2. **Real-time Updates**: Streams via WebSocket to graph component
3. **3D Positioning**: Provides optimized coordinates for GPU physics
4. **Visual Hints**: Includes colors, sizes, and animation states
5. **Interaction Events**: Handles selection, hover, and click events

### Example Client Integration

```typescript
// Connect to multi-MCP WebSocket
const ws = new WebSocket('ws://localhost:8080/api/multi-mcp/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'discovery':
      updateSwarmTopology(data);
      break;
    case 'multi_agent_update':
      updateAgentPositions(data.agents);
      break;
    case 'performance_analysis':
      updatePerformanceMetrics(data.global_metrics);
      break;
  }
};

// Configure subscription
ws.send(JSON.stringify({
  action: 'configure',
  data: {
    subscription_filters: {
      server_types: ['claude_flow', 'ruv_swarm'],
      include_neural: true
    },
    performance_mode: 'normal'
  }
}));
```

## 🧪 Testing

### Unit Tests
```bash
cargo test services::multi_mcp_agent_discovery
cargo test services::topology_visualization_engine
```

### Integration Tests
```bash
cargo test --test multi_mcp_integration
```

### Performance Tests
```bash
cargo test --release --test performance_benchmarks
```

## 🐛 Troubleshooting

### Common Issues

1. **MCP Server Connection Failed**
   - Check server is running on expected port
   - Verify firewall settings
   - Check server logs for errors

2. **WebSocket Connection Drops**
   - Increase heartbeat timeout
   - Check network stability
   - Verify server capacity

3. **High CPU Usage**
   - Reduce update frequency
   - Enable subscription filters
   - Use OnDemand performance mode

4. **Memory Leaks**
   - Clear layout cache periodically
   - Limit performance history size
   - Monitor agent cache growth

### Debug Logging

```bash
RUST_LOG=debug cargo run --example multi_mcp_integration_demo
```

### Health Endpoints

```
GET /api/multi-mcp/status
POST /api/multi-mcp/refresh
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push branch: `git push origin feature/amazing-feature`  
5. Open Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/visionflow-ext.git
cd visionflow-ext

# Install dependencies
cargo build

# Run tests
cargo test

# Start demo
cargo run --example multi_mcp_integration_demo
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Claude Flow Team** for MCP protocol specifications
- **RuvSwarm Project** for distributed agent coordination
- **VisionFlow** for graph visualization foundation
- **Actix Web** for WebSocket and actor system capabilities