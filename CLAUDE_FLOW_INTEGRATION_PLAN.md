# Claude Flow-Centric Integration Plan

## Executive Summary

Replace the current redundant physics-heavy `agent-control-interface` with a streamlined Claude Flow-centric MCP interface that directly exposes the orchestrator's capabilities without unnecessary overhead.

## Current Architecture Problems

### 1. Redundant Systems
- **agent-control-interface** (Node.js): Has its own physics engine (wasteful)
- **claude_flow_client.rs**: Full MCP client already exists but underutilized
- **Multiple mock data generators**: Scattered throughout codebase
- **Dual telemetry paths**: AgentControlActor vs ClaudeFlowActor confusion

### 2. Port Confusion
- Port 3001: MCP Relay (external)
- Port 3002: Claude Flow WebSocket (internal)
- Port 9500: TCP JSON-RPC (current agent-control)
- Multiple overlapping services

## Proposed Architecture

```
┌─────────────────────────────────────────────────────┐
│           VisionFlow Rust Backend (ext-app)          │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │       Enhanced ClaudeFlowActor              │   │
│  │  (Single source of truth for agents)        │   │
│  └────────────────┬─────────────────────────────┘   │
│                   │                                  │
│         Direct MCP Protocol                          │
│         (No intermediate layer)                      │
└───────────────────┼──────────────────────────────────┘
                    │
              Port 3002 (WebSocket)
                    │
┌───────────────────▼──────────────────────────────────┐
│        Claude Flow Orchestrator Container            │
│           (multi-agent-docker)                       │
│                                                      │
│  ┌─────────────────────────────────────────────┐   │
│  │      Native Claude Flow MCP Server          │   │
│  │   Exposes full orchestrator capabilities    │   │
│  │   • Agent lifecycle (spawn/terminate)       │   │
│  │   • Task management                         │   │
│  │   • Memory operations                       │   │
│  │   • Neural network functions                │   │
│  │   • System metrics                          │   │
│  │   • Message flow telemetry                  │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Direct MCP Connection (Week 1)

#### 1.1 Enhance ClaudeFlowActor to be Primary Agent Source

```rust
// src/actors/claude_flow_actor_enhanced.rs

impl ClaudeFlowActor {
    async fn connect_direct_mcp(&mut self) -> Result<(), Box<dyn Error>> {
        // Connect directly to Claude Flow MCP on port 3002
        let mcp_client = MCPClient::new(MCPConfig {
            url: "ws://multi-agent-docker:3002",
            transport: TransportType::WebSocket,
            capabilities: vec![
                "agent-management",
                "task-orchestration",
                "memory-management",
                "neural-operations",
                "telemetry-streaming"
            ],
        });
        
        await mcp_client.connect()?;
        
        // Subscribe to real-time telemetry
        mcp_client.subscribe("telemetry/*", |event| {
            self.handle_telemetry_event(event);
        })?;
        
        self.mcp_client = Some(mcp_client);
        self.is_connected = true;
        Ok(())
    }
    
    async fn handle_telemetry_event(&mut self, event: MCPEvent) {
        match event.event_type {
            "agent.spawned" => self.add_agent(event.data),
            "agent.terminated" => self.remove_agent(event.data.id),
            "agent.status" => self.update_agent_status(event.data),
            "task.assigned" => self.update_agent_task(event.data),
            "message.sent" => self.add_message_flow(event.data),
            "metrics.update" => self.update_system_metrics(event.data),
            _ => {}
        }
        
        // Immediately update GraphServiceActor
        self.push_to_graph();
    }
    
    fn push_to_graph(&self) {
        // Convert internal state to GraphData
        let graph_data = self.to_graph_data();
        
        // Send to GraphServiceActor
        self.graph_service_addr.do_send(UpdateBotsGraph {
            nodes: graph_data.nodes,
            edges: graph_data.edges,
            metadata: graph_data.metadata,
        });
    }
}
```

#### 1.2 Remove AgentControlActor Completely

```bash
# Files to remove:
rm src/actors/agent_control_actor.rs
rm src/handlers/agent_control_handler.rs
rm src/services/agent_control_client.rs

# Remove from main.rs initialization
# Remove from app_state.rs
```

#### 1.3 Consolidate API Endpoints

```rust
// src/handlers/api_handler/bots/mod.rs

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/bots")
            // All agent operations go through ClaudeFlowActor
            .route("/agents", web::get().to(get_agents))
            .route("/spawn", web::post().to(spawn_agent))
            .route("/terminate", web::delete().to(terminate_agent))
            .route("/assign-task", web::post().to(assign_task))
            .route("/telemetry", web::get().to(get_telemetry))
            .route("/swarm/initialize", web::post().to(initialize_swarm))
    );
}
```

### Phase 2: Claude Flow MCP Server Enhancement (Week 1-2)

#### 2.1 Create Lightweight MCP Server in multi-agent-docker

```javascript
// claude-flow-mcp-server.js

const WebSocket = require('ws');
const { ClaudeFlowOrchestrator } = require('./orchestrator');

class ClaudeFlowMCPServer {
    constructor() {
        this.orchestrator = new ClaudeFlowOrchestrator();
        this.wss = new WebSocket.Server({ port: 3002 });
        this.clients = new Set();
    }
    
    start() {
        this.wss.on('connection', (ws) => {
            this.handleConnection(ws);
        });
        
        // Start telemetry streaming
        this.startTelemetryStream();
    }
    
    handleConnection(ws) {
        this.clients.add(ws);
        
        ws.on('message', async (data) => {
            const request = JSON.parse(data);
            const response = await this.handleMCPRequest(request);
            ws.send(JSON.stringify(response));
        });
        
        ws.on('close', () => {
            this.clients.delete(ws);
        });
    }
    
    async handleMCPRequest(request) {
        switch(request.method) {
            case 'agent.spawn':
                return await this.orchestrator.spawnAgent(request.params);
            case 'agent.list':
                return await this.orchestrator.listAgents();
            case 'agent.terminate':
                return await this.orchestrator.terminateAgent(request.params.id);
            case 'task.assign':
                return await this.orchestrator.assignTask(request.params);
            case 'swarm.initialize':
                return await this.orchestrator.initializeSwarm(request.params);
            case 'metrics.get':
                return await this.orchestrator.getMetrics();
            case 'telemetry.subscribe':
                // Add to telemetry subscribers
                return { subscribed: true };
            default:
                return { error: 'Unknown method' };
        }
    }
    
    startTelemetryStream() {
        // Push real-time updates every 100ms
        setInterval(() => {
            const telemetry = this.orchestrator.getCurrentTelemetry();
            this.broadcast({
                type: 'telemetry.update',
                data: telemetry
            });
        }, 100);
    }
    
    broadcast(message) {
        const data = JSON.stringify(message);
        this.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(data);
            }
        });
    }
}
```

#### 2.2 Remove Redundant agent-control-interface

```bash
# After Claude Flow MCP server is running:
rm -rf /workspace/ext/agent-control-interface

# Update Docker configuration to run claude-flow-mcp-server instead
```

### Phase 3: Telemetry Pipeline Completion (Week 2)

#### 3.1 Complete the Data Flow

```rust
// src/actors/claude_flow_actor_enhanced.rs

impl Handler<GetAgentTelemetry> for ClaudeFlowActor {
    type Result = Result<AgentTelemetry, String>;
    
    fn handle(&mut self, _msg: GetAgentTelemetry, _ctx: &mut Context<Self>) -> Self::Result {
        // Return current telemetry state
        Ok(AgentTelemetry {
            agents: self.agents.values().cloned().collect(),
            connections: self.message_flows.clone(),
            metrics: self.system_metrics.clone(),
            timestamp: Instant::now(),
        })
    }
}

impl ClaudeFlowActor {
    fn to_graph_data(&self) -> GraphData {
        // Convert agents to nodes with proper positioning
        let nodes = self.agents.values().map(|agent| {
            Node {
                id: agent.id.parse().unwrap_or(0),
                label: agent.name.clone(),
                data: BinaryNodeData {
                    // Position will be calculated by GPU
                    position: Vec3Data::zero(),
                    velocity: Vec3Data::zero(),
                    mass: agent.workload as u8,
                    flags: if agent.status == "active" { 1 } else { 0 },
                    padding: [0, 0],
                },
                metadata: agent.to_metadata(),
            }
        }).collect();
        
        // Convert message flows to edges
        let edges = self.message_flows.iter().map(|flow| {
            Edge {
                id: flow.id.clone(),
                source: flow.from_id,
                target: flow.to_id,
                weight: flow.intensity,
            }
        }).collect();
        
        GraphData {
            nodes,
            edges,
            metadata: self.build_metadata_store(),
        }
    }
}
```

#### 3.2 GPU Integration

```rust
// src/actors/gpu_compute_actor.rs

impl Handler<UpdateAgentGraph> for GPUComputeActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateAgentGraph, _ctx: &mut Context<Self>) -> Self::Result {
        // Update agent nodes in GPU buffer
        self.update_agent_buffer(msg.nodes)?;
        
        // Update edges
        self.update_edge_buffer(msg.edges)?;
        
        // Trigger physics computation
        self.compute_forces_dual_graph()?;
        
        Ok(())
    }
}
```

### Phase 4: Remove Mock Data (Week 2-3)

#### 4.1 Delete All Mock Generators

```rust
// Remove these functions entirely:
// - generate_mock_bots_update() from bots_client.rs
// - create_hive_mind_mock_data() from bots_handler.rs
// - generate_enhanced_mock_agents() from claude_flow_actor_enhanced.rs
// - All mock data in telemetry-aggregator.js
```

#### 4.2 Implement Fallback Strategy

```rust
impl ClaudeFlowActor {
    async fn ensure_connection(&mut self) -> Result<(), String> {
        if !self.is_connected {
            // Attempt reconnection
            if let Err(e) = self.connect_direct_mcp().await {
                // Return empty state instead of mock data
                return Err(format!("MCP connection failed: {}", e));
            }
        }
        Ok(())
    }
}
```

### Phase 5: Performance Optimization (Week 3)

#### 5.1 Implement Differential Updates

```rust
struct TelemetryDelta {
    added_agents: Vec<Agent>,
    removed_agents: Vec<String>,
    updated_agents: Vec<AgentUpdate>,
    new_messages: Vec<MessageFlow>,
}

impl ClaudeFlowActor {
    fn compute_delta(&mut self) -> TelemetryDelta {
        // Only send changes since last update
        let delta = TelemetryDelta {
            added_agents: self.pending_additions.drain(..).collect(),
            removed_agents: self.pending_removals.drain(..).collect(),
            updated_agents: self.pending_updates.drain(..).collect(),
            new_messages: self.pending_messages.drain(..).collect(),
        };
        delta
    }
}
```

#### 5.2 Binary Protocol for Telemetry

```rust
// Extend binary protocol for agent telemetry
struct AgentTelemetryPacket {
    header: u8,     // Packet type
    count: u16,     // Number of agents
    timestamp: u64, // Unix timestamp
    data: Vec<CompactAgent>,
}

struct CompactAgent {
    id: u32,
    status: u8,
    workload: u8,
    task_count: u16,
    cpu_usage: u8,  // 0-255 mapped to 0-100%
    memory_mb: u16,
}
```

## Benefits of This Approach

### 1. Efficiency Gains
- **Remove redundant physics**: No wasted CPU on unused calculations
- **Direct connection**: Eliminate intermediate JSON-RPC layer
- **Binary protocol**: Reduce network overhead by 80%
- **Differential updates**: Only send changes, not full state

### 2. Simplification
- **Single source of truth**: ClaudeFlowActor owns all agent data
- **Clear data flow**: MCP → ClaudeFlowActor → GraphServiceActor → GPU
- **Fewer moving parts**: Remove entire agent-control-interface directory
- **Unified API**: All agent operations through one endpoint

### 3. Real-Time Performance
- **100ms telemetry updates**: Direct WebSocket streaming
- **Sub-second control**: No intermediate hops
- **Predictable latency**: P99 < 50ms for all operations
- **Scalable**: Support 1000+ agents

## Migration Strategy

### Week 1: Parallel Operation
1. Implement direct MCP connection in ClaudeFlowActor
2. Keep agent-control-interface running as fallback
3. A/B test both paths

### Week 2: Switchover
1. Make ClaudeFlowActor primary
2. Disable agent-control-interface
3. Remove mock data generators

### Week 3: Optimization
1. Implement differential updates
2. Add binary protocol
3. Performance tuning

## Testing Plan

### Integration Tests
```rust
#[tokio::test]
async fn test_claude_flow_mcp_connection() {
    let actor = ClaudeFlowActor::new();
    actor.connect_direct_mcp().await.unwrap();
    
    // Spawn test agent
    let agent = actor.spawn_agent("test", "coder").await.unwrap();
    assert_eq!(agent.status, "active");
    
    // Verify telemetry
    let telemetry = actor.get_telemetry().await.unwrap();
    assert!(telemetry.agents.len() > 0);
}
```

### Load Tests
- 1000 concurrent agents
- 10,000 messages/second
- Measure latency percentiles
- Monitor resource usage

## Success Metrics

1. **Latency**: P99 < 50ms for all operations
2. **Throughput**: 10,000+ telemetry updates/second
3. **Resource Usage**: 50% reduction in CPU/memory
4. **Code Reduction**: Remove 5,000+ lines of redundant code
5. **Reliability**: 99.9% uptime for MCP connection

## Risk Mitigation

### Risk: MCP Service Unavailable
**Mitigation**: Implement exponential backoff reconnection with circuit breaker

### Risk: Breaking Changes
**Mitigation**: Version the MCP protocol, support multiple versions

### Risk: Performance Regression
**Mitigation**: Comprehensive benchmarks before/after migration

## Conclusion

This plan eliminates the redundant physics engine, simplifies the architecture, and creates a direct, efficient pipeline from Claude Flow to GPU visualization. The result will be a cleaner, faster, and more maintainable system that fully leverages the existing MCP infrastructure.