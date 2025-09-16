# Agent Visualization System

**Status: 90% Complete** | **Integration: MCP + UpdateBotsGraph + Physics** | **Architecture: Hierarchical Positioning + Binary Protocol**

## Implementation Status

### ✅ **FULLY IMPLEMENTED (90%)**

**Core Agent Positioning System:**
- ✅ **Hierarchical Algorithm**: Queen-coordinator-architect-worker hierarchy with physics integration
- ✅ **UpdateBotsGraph Flow**: BotsClient → GraphServiceActor → WebSocket broadcasting
- ✅ **MCP Integration**: Fresh TCP connections to multi-agent-container:9500 with compatibility fixes
- ✅ **Agent Type Positioning**: Specialized positioning for coordinators, architects, implementation agents
- ✅ **Binary Protocol**: Efficient 34-byte agent data streaming with metadata preservation

**Message System Architecture:**
- ✅ **UpdateBotsGraph Messages**: Complete implementation in `src/actors/messages.rs:602`
- ✅ **Data Flow Pipeline**: MCP Server → BotsClient → UpdateBotsGraph → WebSocket
- ✅ **Multi-source Support**: Both claude-flow hive-mind and legacy BotsClient agents
- ✅ **Error Handling**: Fallback chain with graceful degradation to mock data
- ✅ **Connection Health**: Circuit breakers and retry logic for MCP compatibility

### ⚠️ **REMAINING WORK (10%)**

**Final Integration Tasks:**
- 🔄 **Physics Force Integration**: Connect agent mass/hierarchy to GPU force calculations
- 🔄 **Real-time Position Updates**: Smooth animation between agent state changes
- 🔄 **Advanced Filtering**: Performance-based agent grouping and clustering

## Overview

The agent visualization system provides real-time 3D visualization of multi-agent swarms with sophisticated positioning algorithms, efficient binary data streaming, and comprehensive MCP protocol integration.

## 🏗️ Architecture

### Architecture Implementation

**Current Working Architecture:**
```
MCP Server (port 9500)
    ↓ [Fresh TCP connection per request]
 BotsClient::fetch_hive_mind_agents()
    ↓ [Parses agent data from MCP responses]
 UpdateBotsGraph message
    ↓ [Sent to GraphServiceActor]
 GraphServiceActor (manages bots_graph_data)
    ↓ [WebSocket broadcast via ClientManagerActor]
 Frontend React Components (BotsVisualization*)
```

**Key Implementation Details:**
- **Connection Strategy**: Fresh TCP connections prevent MCP server incompatibility
- **Error Handling**: Fallback chain (MCP → BotsClient → GraphService → Mock data)
- **Multi-source Data**: Supports both claude-flow hive-mind and legacy BotsClient agents

### Key Implementation Components

#### 1. Agent Positioning Algorithm (`src/handlers/bots_handler.rs`)

**Hierarchical Positioning System:**
```rust
fn position_agents_hierarchically(agents: &mut Vec<BotsAgent>) {
    // Find coordinators (acting as Queens)
    let coordinator_ids: Vec<String> = agents.iter()
        .filter(|a| a.agent_type == "coordinator")
        .map(|a| a.id.clone())
        .collect();

    // Position coordinators at center level (200px radius)
    // Position child agents around parents (300px+ radius)
    // Uses parent_queen_id relationships for hierarchy
}
```

**Agent Type-Based Positioning:**
- **Queen**: Center (0,0)
- **Coordinator**: Inner ring (8px radius)
- **Architect**: Architecture level (12px radius, +2px vertical)
- **Implementation agents**: 18-20px radius with hierarchical Z-axis offsets

#### 2. UpdateBotsGraph Message System (`src/actors/messages.rs`)

```rust
#[derive(Message)]
#[rtype(result = "()")]
pub struct UpdateBotsGraph {
    pub agents: Vec<AgentStatus>,
}
```

#### 3. Binary Agent Protocol (`src/utils/socket_flow_messages.rs`)

**Enhanced Agent-to-Node Conversion:**
```rust
fn convert_agents_to_nodes(agents: Vec<BotsAgent>) -> Vec<Node> {
    agents.into_iter().enumerate().map(|(idx, agent)| {
        // Mass calculation based on agent type and activity
        let base_mass = match agent.agent_type.as_str() {
            "queen" => 15.0,      // Heaviest (central gravity)
            "coordinator" => 10.0,
            "architect" => 8.0,
            _ => 5.0,
        };
        // Enhanced with workload and activity factors
    })
}
```

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
- Circular arrangement with neighbour connections
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

// Create visualisation actor
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

### Topology Visualisation

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

## 📈 Performance Optimisation

### Discovery Service
- Configurable polling intervals per server
- Retry logic with exponential backoff
- Connection pooling and health monitoring
- Differential updates to minimise bandwidth

### Visualisation Engine
- Cached layout calculations
- GPU-optimised positioning algorithms
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
3. **3D Positioning**: Provides optimised coordinates for GPU physics
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

## 📄 Licence

This project is licensed under the MIT Licence - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Claude Flow Team** for MCP protocol specifications
- **RuvSwarm Project** for distributed agent coordination
- **VisionFlow** for graph visualisation foundation
- **Actix Web** for WebSocket and actor system capabilities

## Related Topics

- [Claude Flow MCP Integration](server/features/claude-flow-mcp-integration.md)
- [Docker MCP Integration - Production Deployment Guide](deployment/docker-mcp-integration.md)
- [MCP Connection Architecture](architecture/mcp-connection.md)
- [MCP Integration Architecture](architecture/mcp-integration.md)
- [MCP Integration](server/mcp-integration.md)
- [MCP Tool Integration Analysis](technical/mcp-tool-usage.md)
- [MCP WebSocket Relay Architecture](architecture/mcp-websocket-relay.md)
- [Multi-MCP Agent Visualisation API Reference](api/multi-mcp-visualization-api.md)
- [Multi-MCP Agent Visualisation System](MCP_AGENT_VISUALIZATION.md)
- [VisionFlow MCP Integration Documentation](api/mcp/index.md)
