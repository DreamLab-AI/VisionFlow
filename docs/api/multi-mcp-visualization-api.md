# Multi-MCP Agent Visualization API Reference

Complete API documentation for the Multi-MCP Agent Visualization System.

## 🌐 WebSocket API

### Connection

**Endpoint:** `ws://localhost:8080/api/multi-mcp/ws`

**Protocol:** WebSocket with JSON message exchange

### Client to Server Messages

#### Configure Subscription
```json
{
  "action": "configure",
  "data": {
    "subscription_filters": {
      "server_types": ["claude_flow", "ruv_swarm", "daa"],
      "agent_types": ["coordinator", "coder", "researcher"],
      "swarm_ids": ["swarm-001", "swarm-002"],
      "include_performance": true,
      "include_neural": true,
      "include_topology": true
    },
    "performance_mode": "normal"
  }
}
```

**Performance Modes:**
- `"high_frequency"` - 60Hz updates (for active monitoring)
- `"normal"` - 10Hz updates (default)
- `"low_frequency"` - 1Hz updates (for dashboards)
- `"on_demand"` - Manual updates only

#### Request Discovery Data
```json
{
  "action": "request_discovery"
}
```

#### Request Agent Updates
```json
{
  "action": "request_agents"
}
```

#### Request Performance Analysis
```json
{
  "action": "request_performance"
}
```

#### Request Topology Updates
```json
{
  "action": "request_topology",
  "data": {
    "swarm_id": "swarm-001"
  }
}
```

### Server to Client Messages

#### Discovery Message
```json
{
  "type": "discovery",
  "timestamp": 1692454800000,
  "servers": [
    {
      "server_id": "claude-flow",
      "server_type": "claude_flow",
      "host": "localhost",
      "port": 9500,
      "is_connected": true,
      "last_heartbeat": 1692454800,
      "supported_tools": ["swarm_init", "agent_list", "neural_status"],
      "agent_count": 4
    }
  ],
  "total_agents": 12,
  "swarms": [
    {
      "swarm_id": "swarm-001",
      "server_source": "claude_flow",
      "topology": "hierarchical",
      "agent_count": 4,
      "health_score": 0.95,
      "coordination_efficiency": 0.87
    }
  ],
  "global_topology": {
    "inter_swarm_connections": [],
    "coordination_hierarchy": [
      {
        "level": 0,
        "coordinator_agents": ["coordinator-001"],
        "managed_agents": ["worker-001", "worker-002"],
        "coordination_load": 0.65
      }
    ],
    "data_flow_patterns": []
  }
}
```

#### Multi-Agent Update
```json
{
  "type": "multi_agent_update",
  "timestamp": 1692454800000,
  "agents": [
    {
      "agent_id": "coordinator-001",
      "swarm_id": "swarm-001",
      "server_source": "claude_flow",
      "name": "Main Coordinator",
      "agent_type": "coordinator",
      "status": "active",
      "capabilities": ["coordination", "task_management"],
      "metadata": {
        "session_id": "session-001",
        "parent_id": null,
        "topology_position": {
          "layer": 0,
          "index_in_layer": 0,
          "connections": ["worker-001", "worker-002"],
          "is_coordinator": true,
          "coordination_level": 1
        },
        "coordination_role": "primary",
        "task_queue_size": 5,
        "error_count": 0,
        "warning_count": 1,
        "tags": ["coordinator", "primary"]
      },
      "performance": {
        "cpu_usage": 0.45,
        "memory_usage": 0.62,
        "health_score": 0.95,
        "activity_level": 0.8,
        "tasks_active": 5,
        "tasks_completed": 123,
        "tasks_failed": 2,
        "success_rate": 0.98,
        "token_usage": 45000,
        "token_rate": 150.0,
        "response_time_ms": 45.0,
        "throughput": 25.5
      },
      "neural_info": {
        "model_type": "claude-3-sonnet",
        "model_size": "medium",
        "training_status": "active",
        "cognitive_pattern": "coordination",
        "learning_rate": 0.01,
        "adaptation_score": 0.85,
        "memory_capacity": 1000000,
        "knowledge_domains": ["project_management", "system_architecture"]
      },
      "created_at": 1692451200,
      "last_active": 1692454800
    }
  ],
  "differential_updates": [
    {
      "agent_id": "coordinator-001",
      "field_updates": {
        "status": "active",
        "last_active": 1692454800
      },
      "performance_delta": {
        "cpu_change": 0.05,
        "memory_change": -0.02,
        "task_completion_rate": 0.98,
        "error_rate_change": 0.0
      }
    }
  ],
  "removed_agents": []
}
```

#### Topology Update
```json
{
  "type": "topology_update",
  "timestamp": 1692454800000,
  "swarm_id": "swarm-001",
  "topology_changes": [
    {
      "change_type": "agent_added",
      "affected_agents": ["new-worker-003"],
      "new_structure": {...},
      "reason": "scaling_up"
    }
  ],
  "new_connections": [
    {
      "connection_id": "conn-001-003",
      "source_agent": "coordinator-001",
      "target_agent": "new-worker-003",
      "connection_type": "coordination",
      "strength": 0.8,
      "bidirectional": false
    }
  ],
  "removed_connections": [],
  "coordination_updates": [
    {
      "coordinator_id": "coordinator-001",
      "managed_agents": ["worker-001", "worker-002", "new-worker-003"],
      "coordination_load": 0.72,
      "efficiency_score": 0.85
    }
  ]
}
```

#### Neural Update
```json
{
  "type": "neural_update",
  "timestamp": 1692454800000,
  "neural_agents": [
    {
      "agent_id": "coordinator-001",
      "neural_data": {
        "model_type": "claude-3-sonnet",
        "training_status": "adapting",
        "cognitive_pattern": "coordination",
        "learning_rate": 0.012,
        "adaptation_score": 0.87
      },
      "learning_progress": 0.75,
      "recent_adaptations": ["task_prioritization", "resource_allocation"]
    }
  ],
  "learning_events": [
    {
      "event_id": "learn-001",
      "agent_id": "coordinator-001",
      "event_type": "pattern_recognition",
      "learning_data": {...},
      "performance_impact": 0.15
    }
  ],
  "adaptation_metrics": [
    {
      "metric_name": "coordination_efficiency",
      "current_value": 0.87,
      "target_value": 0.90,
      "progress": 0.73
    }
  ]
}
```

#### Performance Analysis
```json
{
  "type": "performance_analysis",
  "timestamp": 1692454800000,
  "global_metrics": {
    "total_throughput": 125.5,
    "average_latency": 67.2,
    "system_efficiency": 0.87,
    "resource_utilization": 0.62,
    "error_rate": 0.02,
    "coordination_overhead": 0.15
  },
  "bottlenecks": [
    {
      "agent_id": "worker-002",
      "bottleneck_type": "cpu",
      "severity": 0.85,
      "impact_agents": ["coordinator-001"],
      "suggested_action": "Scale resources or redistribute workload"
    }
  ],
  "optimization_suggestions": [
    {
      "suggestion_id": "opt-001",
      "target_component": "worker-002",
      "optimization_type": "resource_scaling",
      "expected_improvement": 0.25,
      "implementation_complexity": "low",
      "risk_level": "minimal"
    }
  ],
  "trend_analysis": [
    {
      "metric_name": "throughput",
      "trend_direction": "increasing",
      "rate_of_change": 0.12,
      "confidence": 0.89,
      "prediction_horizon_minutes": 30
    }
  ]
}
```

#### Coordination Event
```json
{
  "type": "coordination_event",
  "timestamp": 1692454800000,
  "event_type": "task_delegation",
  "source_agent": "coordinator-001",
  "target_agents": ["worker-001", "worker-002"],
  "event_data": {
    "task_id": "task-789",
    "priority": "high",
    "estimated_duration": 300
  },
  "coordination_impact": 0.75
}
```

## 🔌 REST API

### Get MCP Server Status
**GET** `/api/multi-mcp/status`

**Response:**
```json
{
  "servers": [
    {
      "server_id": "claude-flow",
      "server_type": "claude_flow",
      "host": "localhost",
      "port": 9500,
      "is_connected": true,
      "agent_count": 4
    },
    {
      "server_id": "ruv-swarm",
      "server_type": "ruv_swarm",
      "host": "localhost",
      "port": 9501,
      "is_connected": false,
      "agent_count": 0
    }
  ],
  "total_agents": 4,
  "timestamp": 1692454800000
}
```

### Refresh MCP Discovery
**POST** `/api/multi-mcp/refresh`

**Response:**
```json
{
  "success": true,
  "message": "Discovery refresh initiated",
  "timestamp": 1692454800000
}
```

## 📚 Data Types

### McpServerType
```rust
pub enum McpServerType {
    ClaudeFlow,
    RuvSwarm,
    Daa,
    Custom(String),
}
```

### MultiMcpAgentStatus
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

### AgentPerformanceData
```rust
pub struct AgentPerformanceData {
    pub cpu_usage: f32,           // 0.0 - 1.0
    pub memory_usage: f32,        // 0.0 - 1.0
    pub health_score: f32,        // 0.0 - 1.0
    pub activity_level: f32,      // 0.0 - 1.0
    pub tasks_active: u32,
    pub tasks_completed: u32,
    pub tasks_failed: u32,
    pub success_rate: f32,        // 0.0 - 1.0
    pub token_usage: u64,
    pub token_rate: f32,          // tokens per second
    pub response_time_ms: f32,
    pub throughput: f32,
}
```

### NeuralAgentData
```rust
pub struct NeuralAgentData {
    pub model_type: String,
    pub model_size: String,
    pub training_status: String,
    pub cognitive_pattern: String,
    pub learning_rate: f32,
    pub adaptation_score: f32,    // 0.0 - 1.0
    pub memory_capacity: u64,
    pub knowledge_domains: Vec<String>,
}
```

### TopologyPosition
```rust
pub struct TopologyPosition {
    pub layer: u32,
    pub index_in_layer: u32,
    pub connections: Vec<String>,  // Connected agent IDs
    pub is_coordinator: bool,
    pub coordination_level: u32,
}
```

## 🎯 Client Integration Examples

### JavaScript WebSocket Client
```javascript
class MultiMcpClient {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.callbacks = {};
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (this.callbacks[data.type]) {
        this.callbacks[data.type](data);
      }
    };
  }
  
  subscribe(messageType, callback) {
    this.callbacks[messageType] = callback;
  }
  
  configure(filters, performanceMode = 'normal') {
    this.send({
      action: 'configure',
      data: {
        subscription_filters: filters,
        performance_mode: performanceMode
      }
    });
  }
  
  requestDiscovery() {
    this.send({ action: 'request_discovery' });
  }
  
  send(message) {
    this.ws.send(JSON.stringify(message));
  }
}

// Usage
const client = new MultiMcpClient('ws://localhost:8080/api/multi-mcp/ws');

client.subscribe('discovery', (data) => {
  console.log(`Found ${data.total_agents} agents across ${data.servers.length} servers`);
  updateSwarmVisualization(data);
});

client.subscribe('multi_agent_update', (data) => {
  console.log(`Agent update: ${data.agents.length} agents`);
  updateAgentPositions(data.agents);
});

client.configure({
  server_types: ['claude_flow', 'ruv_swarm'],
  agent_types: ['coordinator', 'coder'],
  include_neural: true
}, 'normal');
```

### Python Client
```python
import asyncio
import websockets
import json

class MultiMcpClient:
    def __init__(self, url):
        self.url = url
        self.ws = None
        self.callbacks = {}
    
    async def connect(self):
        self.ws = await websockets.connect(self.url)
        
        async for message in self.ws:
            data = json.loads(message)
            if data['type'] in self.callbacks:
                await self.callbacks[data['type']](data)
    
    def subscribe(self, message_type, callback):
        self.callbacks[message_type] = callback
    
    async def configure(self, filters, performance_mode='normal'):
        await self.send({
            'action': 'configure',
            'data': {
                'subscription_filters': filters,
                'performance_mode': performance_mode
            }
        })
    
    async def send(self, message):
        await self.ws.send(json.dumps(message))

# Usage
async def handle_discovery(data):
    print(f"Found {data['total_agents']} agents across {len(data['servers'])} servers")

async def main():
    client = MultiMcpClient('ws://localhost:8080/api/multi-mcp/ws')
    client.subscribe('discovery', handle_discovery)
    
    await client.configure({
        'server_types': ['claude_flow'],
        'include_performance': True
    })
    
    await client.connect()

asyncio.run(main())
```

### Rust Client
```rust
use tokio_tungstenite::{connect_async, tungstenite::Message};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let url = "ws://localhost:8080/api/multi-mcp/ws";
    let (ws_stream, _) = connect_async(url).await?;
    let (mut write, mut read) = ws_stream.split();
    
    // Configure subscription
    let config = json!({
        "action": "configure",
        "data": {
            "subscription_filters": {
                "server_types": ["claude_flow", "ruv_swarm"],
                "include_neural": true
            },
            "performance_mode": "normal"
        }
    });
    
    write.send(Message::Text(config.to_string())).await?;
    
    // Listen for messages
    while let Some(msg) = read.next().await {
        let msg = msg?;
        if let Message::Text(text) = msg {
            let data: Value = serde_json::from_str(&text)?;
            match data["type"].as_str() {
                Some("discovery") => {
                    println!("Discovery: {} agents", data["total_agents"]);
                }
                Some("multi_agent_update") => {
                    println!("Agent update: {} agents", data["agents"].as_array().unwrap().len());
                }
                _ => {}
            }
        }
    }
    
    Ok(())
}
```

## 🔧 Error Handling

### WebSocket Errors
```json
{
  "type": "error",
  "error_code": "SUBSCRIPTION_FAILED",
  "message": "Invalid server type specified",
  "timestamp": 1692454800000
}
```

### Common Error Codes
- `INVALID_ACTION` - Unknown action in client message
- `SUBSCRIPTION_FAILED` - Invalid subscription configuration  
- `SERVER_UNAVAILABLE` - MCP server not reachable
- `RATE_LIMITED` - Too many requests from client
- `INVALID_MESSAGE` - Malformed JSON message
- `AUTHENTICATION_REQUIRED` - Client authentication needed

## 🎛️ Configuration Options

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

# WebSocket server
WEBSOCKET_PORT=8080
WEBSOCKET_MAX_CONNECTIONS=100

# Performance tuning
DISCOVERY_INTERVAL_MS=3000
UPDATE_BATCH_SIZE=50
MAX_MESSAGE_SIZE=1048576
```

### Server Configuration
```rust
pub struct ServerConfig {
    pub discovery_interval_ms: u64,
    pub max_connections: usize,
    pub message_buffer_size: usize,
    pub heartbeat_interval_ms: u64,
    pub client_timeout_ms: u64,
}
```

## 📊 Monitoring and Metrics

### Health Check Endpoint
**GET** `/api/multi-mcp/health`

```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "connected_clients": 5,
  "active_servers": 2,
  "total_agents": 12,
  "message_rate": 125.5,
  "memory_usage_mb": 256,
  "cpu_usage": 0.35
}
```

### Metrics Endpoint
**GET** `/api/multi-mcp/metrics`

```json
{
  "discovery_stats": {
    "successful_discoveries": 150,
    "failed_discoveries": 3,
    "average_discovery_time_ms": 45.2
  },
  "websocket_stats": {
    "total_connections": 27,
    "active_connections": 5,
    "messages_sent": 5432,
    "messages_received": 876
  },
  "agent_stats": {
    "total_agents_discovered": 12,
    "agents_by_server": {
      "claude_flow": 8,
      "ruv_swarm": 4,
      "daa": 0
    }
  }
}
```

This comprehensive API reference provides everything needed to integrate with the Multi-MCP Agent Visualization System.