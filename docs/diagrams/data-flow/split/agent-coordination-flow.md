---
title: Agent Coordination Data Flow
description: Detailed data flow for multi-agent coordination and state synchronization
category: explanation
tags:
  - architecture
  - data-flow
  - agents
  - actors
  - coordination
updated-date: 2026-01-29
difficulty-level: advanced
---

# Agent Coordination Data Flow

This document details the data flow for agent coordination in VisionFlow, including the actor message system, state synchronization, and fault tolerance patterns.

## Overview

VisionFlow uses Actix actors for concurrent processing with 24 specialized actors organized in a supervision hierarchy. This document covers how agents coordinate, share state, and recover from failures.

## Actor Hierarchy

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph TB
    subgraph "GraphServiceSupervisor - Root"
        GSS[GraphServiceSupervisor<br/>Strategy: OneForOne<br/>Max Restarts: 3]
    end

    subgraph "Core State Actors"
        GSS --> GSA[GraphStateActor<br/>Graph data management]
        GSS --> PO[PhysicsOrchestratorActor<br/>GPU coordination]
        GSS --> SP[SemanticProcessorActor<br/>AI analysis]
        GSS --> CC[ClientCoordinatorActor<br/>WebSocket broadcast]
    end

    subgraph "GPU Supervisor Hierarchy"
        PO --> GPUMgr[GPUManagerActor]
        GPUMgr --> ResSup[ResourceSupervisor]
        GPUMgr --> PhysSup[PhysicsSupervisor]
        GPUMgr --> AnalSup[AnalyticsSupervisor]
        GPUMgr --> GraphSup[GraphAnalyticsSupervisor]
    end

    subgraph "Support Actors"
        GSS --> MCP[MultiMcpVisualizationActor]
        GSS --> Task[TaskOrchestratorActor]
        GSS --> Mon[AgentMonitorActor]
    end

    style GSS fill:#ff6b6b,color:#fff
    style GSA fill:#4ecdc4
    style PO fill:#ffe66d
    style GPUMgr fill:#ffccbc
```

## Message Flow Pattern

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
sequenceDiagram
    participant API as HTTP Handler
    participant GSS as GraphServiceSupervisor
    participant GSA as GraphStateActor
    participant PO as PhysicsOrchestrator
    participant CC as ClientCoordinator
    participant WS as WebSocket Clients

    API->>GSS: AddNode(node_data)
    GSS->>GSA: AddNode message

    activate GSA
    GSA->>GSA: Validate node
    GSA->>GSA: Update internal state
    GSA-->>GSS: Ok(node_id)
    deactivate GSA

    GSS->>PO: GraphUpdated(node_id)
    activate PO
    PO->>PO: Recalculate physics
    PO-->>GSS: Ack
    deactivate PO

    GSS->>CC: BroadcastGraphUpdate
    activate CC
    CC->>WS: Binary update (all clients)
    CC-->>GSS: Broadcast complete
    deactivate CC

    GSS-->>API: Response(node_id)
```

## State Synchronization

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph TB
    subgraph "Source of Truth"
        NEO4J[(Neo4j Database<br/>Persistent storage)]
    end

    subgraph "In-Memory State"
        GSA_STATE[GraphStateActor<br/>Graph data cache]
        PO_STATE[PhysicsOrchestrator<br/>Position/velocity state]
        CC_STATE[ClientCoordinator<br/>Client registry]
    end

    subgraph "Derived State"
        GPU_STATE[GPU Buffers<br/>Device memory]
        CLIENT_STATE[Client State<br/>Browser memory]
    end

    NEO4J -->|Initial load| GSA_STATE
    GSA_STATE -->|Graph updates| PO_STATE
    PO_STATE -->|Positions| GPU_STATE
    PO_STATE -->|Broadcast| CC_STATE
    CC_STATE -->|WebSocket| CLIENT_STATE

    NEO4J <-->|Persist changes| GSA_STATE

    style NEO4J fill:#f0e1ff
    style GSA_STATE fill:#e1f5ff
    style GPU_STATE fill:#e1ffe1
```

## Fault Tolerance

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph TB
    subgraph "Failure Detection"
        CRASH[Actor Crash] --> DETECT[Supervisor detects]
        DETECT --> EVAL{Evaluate strategy}
    end

    subgraph "Recovery Strategies"
        EVAL -->|OneForOne| RESTART_ONE[Restart failed actor only]
        EVAL -->|AllForOne| RESTART_ALL[Restart all siblings]
        EVAL -->|Escalate| PARENT[Notify parent supervisor]
    end

    subgraph "State Recovery"
        RESTART_ONE --> CHECKPOINT[Load last checkpoint]
        CHECKPOINT --> DELTA[Apply incremental changes]
        DELTA --> READY[Actor ready]
    end

    subgraph "Backoff"
        FAIL_COUNT{Fail count < 3?}
        FAIL_COUNT -->|Yes| BACKOFF[Exponential backoff]
        BACKOFF --> RESTART_ONE
        FAIL_COUNT -->|No| ESCALATE[Escalate to parent]
    end

    style CRASH fill:#ffe1e1
    style READY fill:#e1ffe1
    style ESCALATE fill:#ff6b6b,color:#fff
```

## Inter-Actor Message Types

| Category | Messages | Purpose |
|----------|----------|---------|
| Graph State | GetGraphData, AddNode, UpdateNode, DeleteNode | CRUD operations |
| Physics | SimulationStep, ComputeForces, UpdatePositions | Physics simulation |
| Client | RegisterClient, BroadcastPositions, Disconnect | WebSocket management |
| GPU | AllocateMemory, ExecuteKernel, FreeResources | GPU resource management |
| Semantic | ProcessMetadata, GenerateConstraints | AI/semantic analysis |

## Event-Driven Updates

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
sequenceDiagram
    participant GH as GitHub Webhook
    participant Sync as GitHubSyncService
    participant Events as EventBus
    participant GSA as GraphStateActor
    participant SP as SemanticProcessor
    participant CC as ClientCoordinator

    GH->>Sync: Push event
    Sync->>Sync: Process markdown files
    Sync->>Events: Emit(OntologyModified)

    par Parallel event handlers
        Events->>GSA: Handle(OntologyModified)
        GSA->>GSA: Invalidate cache
        GSA->>GSA: Reload from Neo4j

        Events->>SP: Handle(OntologyModified)
        SP->>SP: Regenerate constraints

        Events->>CC: Handle(OntologyModified)
        CC->>CC: Force full broadcast
    end
```

## Performance Metrics

| Actor | Messages/sec | Latency P50 | Latency P99 |
|-------|--------------|-------------|-------------|
| GraphStateActor | 5,000 (write), 20,000 (read) | 50us | 200us |
| PhysicsOrchestrator | 60 | 2ms | 10ms |
| ClientCoordinator | 20 | 10ms | 100ms |
| SemanticProcessor | 1,000 | 20ms | 100ms |
| GPUResourceActor | 100 | 100us | 500us |

## Coordination Patterns

### Request-Response (Synchronous)
```rust
let result: Arc<GraphData> = graph_state_actor.send(GetGraphData).await?;
```

### Fire-and-Forget (Asynchronous)
```rust
client_coordinator.do_send(UpdateNodePositions { positions });
```

### Pub/Sub (Event-Driven)
```rust
event_bus.publish(GraphUpdateEvent { nodes_added, nodes_removed });
```

## Related Documentation

- [Actor System Complete](../../server/actors/actor-system-complete.md)
- [GPU Supervisor Hierarchy](../../infrastructure/gpu/gpu-supervisor-hierarchy.md)
- [CQRS Handlers](../../../architecture/patterns/hexagonal-cqrs.md)
