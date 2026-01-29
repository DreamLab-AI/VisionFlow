---
title: C4 Container Diagram - VisionFlow Architecture
description: C4 Level 2 diagram showing VisionFlow's internal container structure and interactions
category: explanation
tags:
  - architecture
  - c4
  - diagrams
  - container
updated-date: 2026-01-29
difficulty-level: advanced
---

# C4 Container Diagram - VisionFlow Architecture

This diagram shows VisionFlow at the container level (C4 Level 2), illustrating the major deployable units and their interactions.

## Container Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
graph TB
    subgraph "Client Layer"
        WebClient[Web Client<br/>React 18 + TypeScript<br/>Three.js + WebXR<br/>Zustand State]
        Mobile[Mobile PWA<br/>Responsive UI<br/>Touch controls]
    end

    subgraph "API Layer"
        REST[REST API<br/>Actix Web 4.11<br/>~114 CQRS Handlers<br/>JSON responses]
        WS[WebSocket Server<br/>Binary Protocol V4<br/>50K concurrent<br/>28 bytes/node]
        Voice[Voice WebSocket<br/>Whisper STT<br/>Kokoro TTS<br/>Audio streaming]
    end

    subgraph "Application Layer"
        Actors[Actor System<br/>24 Actix Actors<br/>Supervisor hierarchy<br/>Fault tolerant]
        CQRS[CQRS Handlers<br/>Command/Query split<br/>Event sourcing ready<br/>Neo4j adapters]
        Events[Event Bus<br/>Async dispatch<br/>Cache invalidation<br/>Real-time updates]
    end

    subgraph "GPU Layer"
        GPUManager[GPU Manager Actor<br/>4 supervisors<br/>Resource coordination]
        Physics[Physics Supervisor<br/>Force computation<br/>Stress majorization<br/>Constraint solving]
        Analytics[Analytics Supervisor<br/>K-means clustering<br/>Anomaly detection<br/>PageRank]
        Graph[Graph Analytics<br/>SSSP + APSP<br/>Connected components<br/>Community detection]
    end

    subgraph "Infrastructure Layer"
        Neo4j[(Neo4j 5.13<br/>Graph Database<br/>Cypher queries<br/>Source of truth)]
        CUDA[CUDA Runtime<br/>87 kernels<br/>12.4 compute<br/>100K nodes @ 60fps]
        OWL[Whelk Reasoner<br/>OWL 2 DL<br/>Semantic validation<br/>Inference engine]
    end

    WebClient -->|HTTPS| REST
    WebClient -->|WSS Binary| WS
    WebClient -->|WSS Audio| Voice
    Mobile -->|HTTPS/WSS| REST

    REST --> CQRS
    WS --> Actors
    Voice --> Actors

    CQRS --> Actors
    Actors --> Events
    Events --> WS

    Actors --> GPUManager
    GPUManager --> Physics
    GPUManager --> Analytics
    GPUManager --> Graph

    Actors --> Neo4j
    Actors --> OWL
    Physics --> CUDA
    Analytics --> CUDA
    Graph --> CUDA

    style WebClient fill:#e3f2fd,stroke:#333
    style REST fill:#c8e6c9,stroke:#333
    style WS fill:#c8e6c9,stroke:#333
    style Voice fill:#c8e6c9,stroke:#333
    style Actors fill:#ffe66d,stroke:#333
    style CQRS fill:#ffe66d,stroke:#333
    style GPUManager fill:#ffccbc,stroke:#333
    style Neo4j fill:#f0e1ff,stroke:#333
    style CUDA fill:#e1ffe1,stroke:#333
    style OWL fill:#fff9c4,stroke:#333
```

## Container Details

### Client Layer

| Container | Technology | Responsibility |
|-----------|------------|----------------|
| Web Client | React 18, TypeScript, Three.js | Interactive 3D visualization, WebXR |
| Mobile PWA | Progressive Web App | Responsive mobile experience |

### API Layer

| Container | Technology | Throughput | Protocol |
|-----------|------------|------------|----------|
| REST API | Actix Web 4.11 | 10K req/s | HTTPS JSON |
| WebSocket Server | Actix WS | 50K concurrent | WSS Binary V4 |
| Voice WebSocket | Whisper + Kokoro | Audio streams | WSS PCM |

### Application Layer

| Container | Actors | Messages | Purpose |
|-----------|--------|----------|---------|
| Actor System | 24 actors | 100+ types | Concurrent processing |
| CQRS Handlers | ~114 handlers | Commands/Queries | Business logic |
| Event Bus | N/A | Events | Async communication |

### GPU Layer

| Supervisor | Child Actors | Kernels | Purpose |
|------------|--------------|---------|---------|
| GPU Manager | 4 supervisors | Coordination | Resource allocation |
| Physics | 5 actors | 37 kernels | Force-directed layout |
| Analytics | 3 actors | 12 kernels | ML algorithms |
| Graph Analytics | 2 actors | 12 kernels | Graph algorithms |

### Infrastructure Layer

| Component | Technology | Performance |
|-----------|------------|-------------|
| Neo4j | 5.13 Enterprise | ~12ms full graph |
| CUDA Runtime | 12.4 | 100K nodes @ 60fps |
| Whelk Reasoner | Rust OWL2 | ~100ms reasoning |

## Communication Patterns

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'primaryColor': '#4A90D9',
  'secondaryColor': '#67B26F',
  'tertiaryColor': '#FFA500',
  'primaryTextColor': '#333',
  'lineColor': '#666'
}}}%%
sequenceDiagram
    participant Client
    participant REST
    participant Actors
    participant GPU
    participant Neo4j
    participant WS

    Client->>REST: HTTP GET /api/graph
    REST->>Actors: GetGraphDataQuery
    Actors->>Neo4j: Cypher: MATCH (n)-[r]->(m)
    Neo4j-->>Actors: GraphData (316 nodes)
    Actors-->>REST: JSON response
    REST-->>Client: 200 OK (graph data)

    Client->>WS: WSS Connect
    WS-->>Client: Binary: Full graph (28n bytes)

    loop Physics Simulation (60Hz)
        Actors->>GPU: ComputeForces
        GPU-->>Actors: Positions (2.4MB)
        Actors->>WS: BroadcastPositions
        WS-->>Client: Binary update (28n bytes)
    end
```

## Scalability Characteristics

| Container | Horizontal Scale | Vertical Scale | Bottleneck |
|-----------|------------------|----------------|------------|
| Web Client | N/A (client-side) | Browser limits | WebGL memory |
| REST API | Load balancer | CPU cores | Handler throughput |
| WebSocket | Session affinity | RAM | Connection count |
| Actor System | Distributed actors | CPU + RAM | Message queue |
| GPU Compute | Multi-GPU | GPU memory | VRAM (24GB) |
| Neo4j | Read replicas | RAM + SSD | Query complexity |

## Related Documentation

- [C4 Context Diagram](c4-context.md)
- [Actor System Documentation](../../diagrams/server/actors/actor-system-complete.md)
- [GPU Architecture Documentation](../../diagrams/infrastructure/gpu/cuda-architecture-complete.md)
