# Bots Visualization Architecture

This document describes the current architecture for visualising AI agents (bots) in the VisionFlow 3D environment using the unified GPU kernel and parallel graph coordinator.

## Overview

The bots visualisation system displays Claude Flow agents as interactive 3D nodes with real-time position updates via binary protocol. It runs parallel to the Logseq graph visualisation using the ParallelGraphCoordinator and unified backend processing.

## Core Components

### 1. Data Types

```typescript
interface BotsAgent {
  id: string;
  name: string;
  type: 'coordinator' | 'researcher' | 'coder' | 'analyst' | 'tester' | ...;
  status: 'active' | 'idle' | 'busy' | 'error';
  capabilities: string[];
  metrics: {
    tasksActive: number;
    tasksCompleted: number;
    successRate: number;
    avgResponseTime: number;
  };
  position?: Vector3;  // 3D position
}

interface BotsEdge {
  id: string;
  source: string;      // Agent ID
  target: string;      // Agent ID
  dataVolume: number;  // Bytes transferred
  messageCount: number;
  lastMessageTime: number;
}

interface BotsCommunication {
  id: string;
  timestamp: string;
  sender: string;
  receivers: string[];
  messageType: string;
  metadata: {
    size: number;
    tokens?: number;
    model?: string;
  };
}
```

### 2. Unified Backend Physics

Physics processing happens on the backend using the unified CUDA kernel:

```rust
// Backend unified GPU processing
impl GPUComputeActor {
    fn process_agent_graph(&mut self, agents: Vec<AgentStatus>) -> Result<()> {
        // Convert agents to Structure of Arrays format
        let (pos_x, pos_y, pos_z) = self.convert_agents_to_soa(&agents)?;

        // Use DualGraph compute mode
        let mut sim_params = SimParams::default();
        sim_params.compute_mode = ComputeMode::DualGraph as i32;
        sim_params.spring_k = 0.3;
        sim_params.damping = 0.95;
        sim_params.repel_k = 50.0;

        // Execute unified kernel
        self.unified_kernel.launch(
            &pos_x, &pos_y, &pos_z,
            &vel_x, &vel_y, &vel_z,
            &edge_sources, &edge_targets, &edge_weights,
            &sim_params
        )?;

        // Stream positions via binary protocol
        self.stream_positions_to_clients()?;
        Ok(())
    }
}
```

**Physics Configuration (settings.yaml):**
```yaml
visionflow:
  physics:
    spring_strength: 0.3     # Edge attraction
    damping: 0.95           # Velocity damping
    repulsion_strength: 0.8  # Node repulsion
    center_force: 0.002     # Center attraction
    max_velocity: 0.5       # Speed limit
    link_distance: 3.0      # Ideal edge length
```

### 3. Visual Representation

Agents are rendered as 3D objects with visual indicators:

```typescript
function renderAgent(agent: BotsAgent) {
  const geometry = getAgentGeometry(agent.type);  // Shape by type
  const material = getAgentMaterial(agent.status); // Color by status

  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(getAgentPosition(agent.id));

  // Add labels, icons, metrics display
  addAgentLabels(mesh, agent);
  addMetricsBadges(mesh, agent.metrics);

  return mesh;
}
```

**Visual Encoding:**
- **Shape**: Agent type (cube=coordinator, sphere=researcher, etc.)
- **Color**: Status (green=active, yellow=idle, red=error)
- **Size**: Task load or importance
- **Glow**: Recent activity
- **Connections**: Communication volume

### 4. Data Flow Pipeline

```
Claude Flow MCP (port 3002)
    ↓ Direct WebSocket connection
EnhancedClaudeFlowActor
    ↓ Agent telemetry processing
GraphServiceActor
    ↓ UpdateBotsGraph message
GPUComputeActor (Unified Kernel)
    ↓ DualGraph mode physics
Binary Protocol WebSocket
    ↓ Position/velocity updates
ParallelGraphCoordinator
    ↓ visionFlowPositions map
3D Renderer
    ↓ Visual output
User Interface

// Separate REST API for metadata
REST API (/api/bots/agents)
    ↓ Agent metadata (JSON)
Frontend UI
    ↓ Agent status, metrics
Agent Inspector Panels
```

### 5. Mock Data System

For development and demos, a comprehensive mock data system:

```typescript
class MockBotsDataProvider {
  private mockAgents: BotsAgent[] = [
    {
      id: "coordinator-001",
      name: "System Coordinator",
      type: "coordinator",
      status: "active",
      capabilities: ["orchestration", "task-management"],
      metrics: {
        tasksActive: 3,
        tasksCompleted: 15,
        successRate: 100,
        avgResponseTime: 250
      }
    },
    // ... more mock agents
  ];

  generateCommunications(count: number): BotsCommunication[] {
    // Generate realistic communication patterns
  }
}
```

## Integration Points

### 1. Backend Integration

The `EnhancedClaudeFlowActor` manages MCP connection and pushes to `GraphServiceActor`:

```rust
impl EnhancedClaudeFlowActor {
    fn apply_pending_changes(&mut self) {
        if self.has_changes() {
            // Apply differential updates to agent_cache
            self.process_pending_additions();
            self.process_pending_updates();
            self.process_pending_removals();

            // Convert to graph format and push to GPU
            let graph_data = self.build_graph_data();
            self.graph_service_addr.do_send(UpdateBotsGraph {
                agents: graph_data.agents,
                edges: graph_data.edges,
                communications: self.message_flow_history.clone(),
            });
        }
    }
}

// GraphServiceActor processes both knowledge and agent graphs
impl Handler<UpdateBotsGraph> for GraphServiceActor {
    fn handle(&mut self, msg: UpdateBotsGraph, _ctx: &mut Context<Self>) {
        // Update agent data
        self.bots_agents = msg.agents;

        // Send to unified GPU kernel with DualGraph mode
        if let Some(gpu) = &self.gpu_compute_addr {
            gpu.do_send(ComputeGraphLayout {
                compute_mode: ComputeMode::DualGraph,
                agents: Some(msg.agents),
                knowledge_nodes: self.knowledge_nodes.clone(),
            });
        }
    }
}
```

### 2. REST API Endpoints

Current implementation provides agent metadata via REST:

```
GET /api/bots/agents          # Get cached agents from EnhancedClaudeFlowActor
GET /api/bots/multi-agent/status    # Get multi-agent status and topology
POST /api/bots/multi-agent/init     # Initialize new multi-agent
POST /api/bots/spawn          # Spawn individual agent
DELETE /api/bots/agent/:id    # Terminate agent
GET /api/bots/health          # MCP connection health check
```

**Position updates are handled separately via binary protocol WebSocket**

### 3. Frontend Services

**ParallelGraphCoordinator**: Central management of both graphs
```typescript
class ParallelGraphCoordinator {
  async enableVisionFlow(enabled: boolean) {
    this.state.visionflow.enabled = enabled;

    if (enabled) {
      // Start REST API polling for agent metadata
      this.pollInterval = setInterval(() => {
        this.fetchAgentsFromAPI();
      }, 10000);

      // Position updates come automatically via binary protocol WebSocket
    } else {
      clearInterval(this.pollInterval);
    }

    this.notifyListeners();
  }

  private async fetchAgentsFromAPI() {
    const response = await fetch('/api/bots/agents');
    const data = await response.json();

    this.state.visionflow.agents = data.agents;
    this.state.visionflow.lastUpdate = Date.now();
    this.notifyListeners();
  }
}
```

**Binary Protocol Integration**: Position updates handled automatically
```typescript
// WebSocket binary protocol automatically updates both graphs
// ParallelGraphCoordinator receives position updates for both:
// - Logseq nodes (knowledge graph)
// - Agent nodes (visionflow graph)
// Frontend components use useParallelGraphs hook to access positions
```

## Performance Optimizations

### 1. Differential Updates
Only changed agents are sent from backend to frontend

### 2. Physics LOD (Level of Detail)
Distant agents use simplified physics calculations

### 3. Render Culling
Only visible agents are rendered each frame

### 4. Data Aggregation
Communications are aggregated into edges to reduce data volume

## User Interactions

### 1. Agent Selection
- Click to select and view detailed metrics
- Double-click to focus camera on agent

### 2. Filtering
- Filter by agent type
- Filter by status
- Show/hide inactive agents

### 3. Time Controls
- Replay communication history
- Adjust simulation speed
- Pause/resume physics

## Configuration

### Visual Themes

```typescript
const AGENT_THEMES = {
  default: {
    coordinator: { color: 0x4A90E2, shape: 'cube' },
    researcher: { color: 0x7ED321, shape: 'sphere' },
    coder: { color: 0xF5A623, shape: 'octahedron' },
    analyst: { color: 0xBD10E0, shape: 'tetrahedron' },
    tester: { color: 0x50E3C2, shape: 'icosahedron' }
  },
  dark: { /* ... */ },
  highContrast: { /* ... */ }
};
```

### Physics Presets

```typescript
const PHYSICS_PRESETS = {
  tight: { springStrength: 0.5, linkDistance: 2.0 },
  loose: { springStrength: 0.1, linkDistance: 5.0 },
  organic: { springStrength: 0.3, damping: 0.8 },
  grid: { springStrength: 0.8, centerForce: 0.0 }
};
```

## Future Enhancements

1. **Real-time WebSocket**: Replace polling with push updates
2. **Agent Trails**: Visualize movement history
3. **Communication Replay**: Replay message sequences
4. **3D Charts**: Embed performance metrics in 3D space
5. **VR Interaction**: Manipulate agents in VR
6. **multi-agent Patterns**: Visualize emergent multi-agent behaviors
7. **Cross-Graph Links**: Connect agents to Logseq nodes