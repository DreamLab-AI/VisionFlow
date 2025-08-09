# Actor System Architecture

## Overview

VisionFlow uses the Actix actor system to orchestrate AI agent swarms and dual graph visualization. The actor architecture enables real-time streaming of Claude Flow MCP telemetry and GPU-accelerated physics simulation for 100,000+ nodes at 60 FPS.

## Architecture

The actor system implements a Claude Flow-centric design with direct MCP integration:

```mermaid
graph TB
    subgraph "MCP Integration"
        MCP[Claude Flow MCP :3002] --> CFA[ClaudeFlowActor]
    end
    
    subgraph "Core Actors"
        CFA --> |Agent Updates| GS[GraphServiceActor]
        MA[MetadataActor] --> |Knowledge Graph| GS
        GS --> |Dual Graph| GPU[GPUComputeActor]
        GPU --> |Binary Stream| CM[ClientManagerActor]
        CM --> |WebSocket| WS[Frontend Clients]
    end
    
    subgraph "Configuration"
        SA[SettingsActor] --> GS
        PSA[ProtectedSettingsActor] --> SA
    end
    
    style CM fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style GS fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style GPU fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style MA fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style SA fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style PSA fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
    style CFA fill:#3A3F47,stroke:#61DAFB,color:#FFFFFF
```

## Actor Descriptions

### GraphServiceActor

**Location**: `src/actors/graph_actor.rs`

Central hub for dual graph management (knowledge + agents).

**Responsibilities**:
- Maintains separate buffers for knowledge and agent graphs
- Applies type flags to node IDs (bit 31 for agents, bit 30 for knowledge)
- Coordinates GPU physics for both graph types
- Merges updates from ClaudeFlowActor (agents) and MetadataActor (knowledge)
- Streams differential updates at 60 FPS

**Key Messages**:
- `GetGraphData` - Retrieve the complete graph
- `UpdateGraphData` - Update the entire graph structure
- `UpdateNodePosition` - Update a single node's position
- `SimulationStep` - Trigger a single physics simulation step
- `AddNode`/`RemoveNode` - Modify nodes
- `AddEdge`/`RemoveEdge` - Modify edges
- `BuildGraphFromMetadata` - Initialize graph from metadata

### ClientManagerActor

**Location**: `src/actors/client_manager_actor.rs`

Manages WebSocket client connections and broadcasts.

**Responsibilities**:
- Tracks connected WebSocket clients
- Broadcasts graph updates to all clients
- Manages client registration/unregistration
- Handles binary protocol encoding for efficient updates

**Key Messages**:
- `RegisterClient` - Register a new WebSocket client
- `UnregisterClient` - Remove a disconnected client
- `BroadcastNodePositions` - Send binary node updates to all clients
- `BroadcastMessage` - Send text messages to all clients
- `GetClientCount` - Get number of connected clients

### GPUComputeActor

**Location**: `src/actors/gpu_compute_actor.rs`

NVIDIA CUDA-accelerated physics for 100,000+ nodes.

**Responsibilities**:
- Dual graph physics with separate parameters
- Force-directed layout with stress majorization
- Visual analytics mode for pattern detection
- Isolation layers for temporal-spatial analysis
- Real-time performance: 60 FPS for 100K nodes

**Key Messages**:
- `InitializeGPU` - Initialize with graph data
- `UpdateGPUGraphData` - Update graph structure
- `UpdateSimulationParams` - Modify physics parameters
- `ComputeForces` - Run one physics iteration
- `GetNodeData` - Retrieve computed positions
- `GetGPUStatus` - Check GPU/CPU status

### SettingsActor

**Location**: `src/actors/settings_actor.rs`

Manages application settings and configuration.

**Responsibilities**:
- Stores and retrieves application settings
- Handles settings updates from clients
- Provides path-based setting access
- Integrates with file-based settings persistence

**Key Messages**:
- `GetSettings` - Retrieve all settings
- `UpdateSettings` - Update complete settings
- `GetSettingByPath` - Get specific setting by path
- `SetSettingByPath` - Update specific setting by path

### MetadataActor

**Location**: `src/actors/metadata_actor.rs`

Manages graph metadata and layout information.

**Responsibilities**:
- Loads metadata from JSON files
- Provides metadata for graph construction
- Handles metadata refresh operations
- Stores and manages node and edge metadata (via `MetadataStore` which is a `HashMap`)

**Key Messages**:
- `GetMetadata` - Retrieve complete metadata
- `UpdateMetadata` - Update metadata store
- `RefreshMetadata` - Reload metadata from disk

### ProtectedSettingsActor

**Location**: `src/actors/protected_settings_actor.rs`

Manages protected settings like API keys.

**Responsibilities**:
- Stores sensitive configuration separately
- Provides controlled access to API keys
- Handles user-specific protected settings
- Integrates with authentication system

**Key Messages**:
- `GetApiKeys` - Retrieve API keys for a specific user
- `ValidateClientToken` - Validate a client's session token
- `StoreClientToken` - Store a new client session token
- `UpdateUserApiKeys` - Update a user's API keys
- `CleanupExpiredTokens` - Trigger cleanup of expired session tokens
- `MergeSettings` - Merge new settings into protected settings
- `SaveSettings` - Persist protected settings to file
- `GetUser` - Retrieve a specific user's protected data

### ClaudeFlowActor (Primary)

**Location**: `src/actors/claude_flow_actor_enhanced.rs`

Direct integration with Claude Flow MCP for real-time agent orchestration.

**Responsibilities**:
- Maintains persistent WebSocket connection to MCP on port 3002
- Streams agent telemetry at 10Hz
- Handles swarm initialization via JSON-RPC
- NO mock data generation - only real Claude Flow data
- Automatic reconnection with exponential backoff

**Key Messages**:
- `InitializeSwarm` - Create agent swarm with topology and task
- `GetAgentTelemetry` - Retrieve real-time agent metrics
- `TerminateSwarm` - Dissolve entire swarm
- `StreamTelemetry` - Start/stop telemetry subscription

**MCP Methods**:
- `agent.spawn` - Create new agent
- `swarm.initialize` - Initialize swarm topology
- `telemetry.subscribe` - Stream metrics at 100ms intervals
- `task.assign` - Assign task to agent

**Agent Types**:
coordinator, researcher, coder, analyst, architect, tester, optimizer, reviewer, documenter

## Message Flow Examples

### Client Connection and Graph Update

```mermaid
sequenceDiagram
    participant Client
    participant WebSocket
    participant ClientManagerActor
    participant GraphServiceActor
    participant GPUComputeActor
    
    Client->>WebSocket: Connect
    WebSocket->>ClientManagerActor: RegisterClient
    ClientManagerActor-->>WebSocket: client_id
    
    loop Simulation Loop
        GraphServiceActor->>GPUComputeActor: ComputeForces
        GPUComputeActor-->>GraphServiceActor: positions
        GraphServiceActor->>ClientManagerActor: BroadcastNodePositions
        ClientManagerActor->>WebSocket: Binary update
        WebSocket->>Client: Node positions
    end
```

### Settings Update Flow

```mermaid
sequenceDiagram
    participant Client
    participant Handler
    participant SettingsActor
    participant ProtectedSettingsActor
    
    Client->>Handler: Update setting
    Handler->>SettingsActor: SetSettingByPath
    
    alt Protected setting
        SettingsActor->>ProtectedSettingsActor: Update protected
        ProtectedSettingsActor-->>SettingsActor: OK
    end
    
    SettingsActor-->>Handler: Success
    Handler-->>Client: Updated
```

## Implementation Patterns

### Actor Initialization

Actors are typically initialized in the application state setup:

```rust
// In app_state.rs
let client_manager = ClientManagerActor::new().start();
let gpu_compute = GPUComputeActor::new(/* params */).start();
let graph_service = GraphServiceActor::new(
    client_manager.clone(),
    Some(gpu_compute.clone())
).start();
```

### Message Handling

Each actor implements handlers for its messages:

```rust
impl Handler<GetGraphData> for GraphServiceActor {
    type Result = Result<GraphData, String>;

    fn handle(&mut self, _: GetGraphData, _: &mut Self::Context) -> Self::Result {
        Ok(self.get_graph_data().clone())
    }
}
```

### Async Communication

Actors communicate asynchronously using addresses:

```rust
// Send message and await response
let graph_data = graph_actor.send(GetGraphData).await??;

// Send without waiting
graph_actor.do_send(StartSimulation);
```

## Best Practices

1. **Message Design**: Keep messages small and focused on single operations
2. **Error Handling**: All actor messages return `Result<T, String>` for consistent error handling
3. **State Encapsulation**: Actors own their state; no shared mutable state
4. **Async Operations**: Use `AsyncContext` for long-running operations
5. **Supervision**: Actors automatically restart on failure

## Performance Considerations

- The actor system eliminates lock contention from `Arc<RwLock<T>>`
- Messages are processed sequentially within each actor
- Use `do_send` for fire-and-forget operations
- Binary protocol reduces serialization overhead
- GPU computation offloads expensive physics calculations

## Testing

Actors can be tested in isolation:

```rust
#[actix_rt::test]
async fn test_graph_actor() {
    let client_manager = ClientManagerActor::new().start();
    let actor = GraphServiceActor::new(client_manager, None).start();
    
    let result = actor.send(GetGraphData).await.unwrap();
    assert!(result.is_ok());
}
```

## Performance Metrics

| Component | Metric | Target | Actual |
|-----------|--------|--------|--------|
| MCP Connection | Latency | < 100ms | ~50ms |
| Telemetry Stream | Update Rate | 10 Hz | 10 Hz |
| GPU Physics | Frame Rate | 60 FPS | 60-120 FPS |
| Binary Protocol | Bandwidth | < 10 MB/s | ~1.2 MB/s |
| Agent Capacity | Max Agents | 50 | 50+ |

## Related Documentation

- [MCP Integration](../architecture/mcp-integration.md)
- [Binary Protocol](../api/binary-protocol.md)
- [GPU Compute](../architecture/gpu-compute.md)
- [Dual Graph Architecture](../architecture/dual-graph.md)