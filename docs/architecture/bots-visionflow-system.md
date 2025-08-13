# Bots/VisionFlow System Architecture

## Overview

The Bots/VisionFlow system is a core feature that enables real-time visualisation and control of AI Multi Agents through the Model Context Protocol (MCP). It provides a spring-physics based 3D visualisation of agent interactions, communication patterns, and system health.

## System Components

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React + Three.js)              │
├─────────────────────────────────────────────────────────────┤
│  - BotsVisualization (3D agent nodes with spring physics)    │
│  - multiAgentInitializationPrompt (Hive mind spawning UI)        │
│  - SystemHealthPanel, ActivityLogPanel, AgentDetailPanel    │
│  - WebSocket Client (Binary position + JSON state updates)   │
└─────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
        ┌───────────▼────────────┐ ┌─────────▼──────────┐
        │   WebSocket Server     │ │   REST API Server   │
        ├────────────────────────┤ ├────────────────────┤
        │ - Binary position data │ │ - /api/bots/status  │
        │ - bots-full-update msg │ │ - /api/bots/init... │
        │ - Real-time events     │ │ - Agent management  │
        └────────────────────────┘ └────────────────────┘
                    │                         │
        ┌───────────▼─────────────────────────▼──────────┐
        │      Agent Control Actor (Rust Backend)         │
        ├─────────────────────────────────────────────────┤
        │ - AgentControlActor (TCP client to agent dock)  │
        │ - Converts agent data to graph format          │
        │ - Integrates with GPU physics simulation       │
        └─────────────────────────────────────────────────┘
                              │
                         TCP :9500
                              │
        ┌─────────────────────▼──────────────────────────┐
        │    Agent Control System (Agent Container)       │
        ├─────────────────────────────────────────────────┤
        │ - MCP Server (stdio interface)                  │
        │ - Agent lifecycle management                    │
        │ - Spring physics engine                         │
        │ - Real agent execution                          │
        └─────────────────────────────────────────────────┘
```

## Data Flow

### 1. Agent State Updates

The system maintains real-time agent state through multiple channels:

#### Binary Position Stream (60 FPS)
- **Protocol**: 28-byte binary format per agent
- **Structure**: `[4 bytes ID | 12 bytes position (3 floats) | 12 bytes velocity (3 floats)]`
- **Flag**: Nodes with ID & 0x80 are identified as bots
- **Transport**: WebSocket binary frames

#### JSON State Updates (Event-based)
- **Message Type**: `bots-full-update`
- **Contains**: Complete agent data including metrics, capabilities, and relationships
- **Frequency**: On state changes or periodic sync

### 2. Agent Types and Capabilities

The system supports 12 specialised agent types:

- **Queen**: Hive mind leader with strategic coordination
- **Coordinator**: Task orchestration and resource allocation
- **Researcher**: Information gathering and analysis
- **Coder**: Implementation and code generation
- **Analyst**: Data analysis and insights
- **Architect**: System design and planning
- **Tester**: Quality assurance and validation
- **Reviewer**: Code review and approval workflows
- **Optimizer**: Performance tuning and optimisation
- **Documenter**: Documentation and knowledge management
- **Monitor**: System monitoring and diagnostics
- **Specialist**: Domain-specific expertise

### 3. multi-agent Topologies

The system supports multiple multi-agent organizational patterns:

- **Mesh**: Fully connected, best for collaborative tasks
- **Hierarchical**: Tree structure with queen/coordinator at root
- **Ring**: Sequential processing pipeline
- **Star**: Central hub with peripheral workers

## Frontend Components

### BotsVisualization
- 3D force-directed graph using React Three Fiber
- Spring physics simulation for natural movement
- Visual encoding of agent state through size, colour, and animation
- Real-time message flow visualisation

### UI Panels
- **SystemHealthPanel**: Overall multi-agent health and metrics
- **ActivityLogPanel**: Real-time activity feed with colour coding
- **AgentDetailPanel**: Detailed view of individual agents

### multiAgentInitializationPrompt
Interactive modal for spawning new multi-agents with configuration options:
- Topology selection
- Agent type selection
- Maximum agent count (3-20)
- Task description
- Neural enhancement toggle

## Backend Integration

### AgentControlActor
The Rust backend uses an actor-based architecture to manage agent control:

```rust
pub struct AgentControlActor {
    client: AgentControlClient, // TCP client to agent container
}
```

Key responsibilities:
- Maintains TCP connection to agent control system
- Translates between agent data and graph visualisation format
- Handles multi-agent initialisation requests
- Provides real-time metrics

### API Endpoints

- `GET /api/bots/status` - Current agent states and metrics
- `POST /api/bots/initialise-multi-agent` - Spawn new Multi Agent
- `GET /api/bots/data` - Full graph data for visualisation
- WebSocket `/ws` - Real-time position and state updates

## Configuration

### Environment Variables

```bash
# Rust Backend
AGENT_CONTROL_URL=agent-container:9500

# Agent Container
TCP_SERVER_ENABLED=true
TCP_SERVER_PORT=9500
MAX_AGENTS=100
PHYSICS_UPDATE_RATE=60
```

### Visual Theme

The VisionFlow graph uses a distinct gold/green colour palette:
- Coordinator agents: Gold (#F1C40F)
- Worker agents: Various greens
- Active connections: Blue (#3498DB)
- Health indicators: Green/Yellow/Red gradient

## Performance Considerations

- Physics simulation runs at 60 FPS in dedicated worker
- Binary protocol minimizes bandwidth for position updates
- Agent states are batched in full-update messages
- GPU acceleration available for large multi-agents

## See Also

- [Parallel Graphs Architecture](./parallel-graphs.md) - How VisionFlow coexists with Logseq graph
- [MCP Integration](../server/features/mcp-integration.md) - Details on Model Context Protocol
- [Agent Control System](../../agent-control-system/README.md) - Standalone agent control package