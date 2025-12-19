---
title: System Architecture Overview - Complete Mermaid Diagrams
description: subgraph "Presentation Layer"         WebUI[Web UI<br/>React 18 + TypeScript]         ThreeJS[Three. js Renderer<br/>WebGL 2.
category: explanation
tags:
  - architecture
  - structure
  - api
  - api
  - api
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Neo4j database
---

# System Architecture Overview - Complete Mermaid Diagrams

## 1. Full System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        Browser[Web Browser<br/>React + Three.js]
        XR[XR/VR Devices<br/>Meta Quest 3]
        Mobile[Mobile Clients<br/>iOS/Android]
    end

    subgraph "Presentation Layer"
        WebUI[Web UI<br/>React 18 + TypeScript]
        ThreeJS[Three.js Renderer<br/>WebGL 2.0]
        XRRuntime[WebXR Runtime<br/>Hand Tracking]
    end

    subgraph "Communication Layer"
        WS[WebSocket Server<br/>Binary Protocol V4]
        REST[REST API<br/>Actix Web 4.11]
        Voice[Voice WebSocket<br/>Whisper STT + Kokoro TTS]
    end

    subgraph "Application Layer - Hexagonal Core"
        direction TB

        subgraph "Ports (9 Interfaces)"
            P1[GraphRepository]
            P2[KnowledgeGraphRepository]
            P3[OntologyRepository]
            P4[SettingsRepository]
            P5[PhysicsSimulator]
            P6[SemanticAnalyzer]
            P7[GpuPhysicsAdapter]
            P8[GpuSemanticAnalyzer]
            P9[InferenceEngine]
        end

        subgraph "Business Logic"
            CQRS[CQRS Handlers<br/>~114 Commands/Queries]
            Domain[Domain Services<br/>Graph Operations]
            Workflow[Workflow Engine<br/>Multi-step Processes]
        end

        subgraph "Adapters (12 Implementations)"
            A1[Neo4jAdapter]
            A2[Neo4jGraphRepository]
            A3[Neo4jSettingsRepository]
            A4[Neo4jOntologyRepository]
            A5[GpuSemanticAnalyzerAdapter]
            A6[ActixPhysicsAdapter]
        end
    end

    subgraph "Actor System (21 Actors)"
        Supervisor[GraphServiceSupervisor<br/>Fault Tolerance]

        subgraph "Core Actors"
            GSA[GraphStateActor<br/>7-State Machine]
            PO[PhysicsOrchestratorActor<br/>GPU Coordinator]
            SP[SemanticProcessorActor<br/>AI Analysis]
            CC[ClientCoordinatorActor<br/>Multi-client Sync]
        end

        subgraph "GPU Actors (11)"
            FC[ForceComputeActor]
            SM[StressMajorizationActor]
            SF[SemanticForcesActor]
            CA[ConstraintActor]
            OC[OntologyConstraintActor]
            SPA[ShortestPathActor]
            PR[PageRankActor]
            CLA[ClusteringActor]
            AD[AnomalyDetectionActor]
            CCO[ConnectedComponentsActor]
            GR[GPUResourceActor]
        end

        subgraph "Support Actors"
            WA[WorkspaceActor]
            SA[SettingsActor]
            OSA[OptimizedSettingsActor]
        end
    end

    subgraph "Infrastructure Layer"
        subgraph "Database"
            Neo4j[(Neo4j 5.13<br/>Graph Database)]
            OWL[OWL Ontologies<br/>Whelk Reasoner]
        end

        subgraph "GPU Compute"
            CUDA[CUDA 12.4<br/>87 Kernels]
            Physics[Physics Engine<br/>Barnes-Hut + Verlet]
            ML[ML Algorithms<br/>Clustering + Pathfinding]
        end

        subgraph "External Services"
            GitHub[GitHub API<br/>Markdown Sync]
            AI[AI Services<br/>Claude + Perplexity]
            Auth[Nostr Auth<br/>Decentralized Identity]
        end
    end

    subgraph "Data Storage"
        GraphData[(Knowledge Graph<br/>Nodes + Edges)]
        Ontologies[(Ontology Store<br/>OWL Classes)]
        UserSettings[(User Settings<br/>Per-client Config)]
        Workspaces[(Workspaces<br/>Multi-tenant Data)]
    end

    Browser --> WebUI
    XR --> XRRuntime
    Mobile --> WebUI

    WebUI --> ThreeJS
    ThreeJS --> WS
    WebUI --> REST
    XRRuntime --> Voice

    WS --> CC
    REST --> CQRS
    Voice --> SP

    CQRS --> P1
    CQRS --> P2
    CQRS --> P3
    CQRS --> P4
    Domain --> P5
    Domain --> P6
    Workflow --> P7
    Workflow --> P8
    SP --> P9

    P1 --> A2
    P2 --> A1
    P3 --> A4
    P4 --> A3
    P5 --> A6
    P6 --> A5
    P7 --> A6
    P8 --> A5

    A1 --> Neo4j
    A2 --> Neo4j
    A3 --> Neo4j
    A4 --> Neo4j
    A4 --> OWL

    A5 --> CUDA
    A6 --> CUDA

    Supervisor --> GSA
    Supervisor --> PO
    Supervisor --> SP
    Supervisor --> CC
    Supervisor --> WA
    Supervisor --> SA
    Supervisor --> OSA

    PO --> FC
    PO --> SM
    PO --> SF
    PO --> CA
    PO --> OC
    PO --> SPA
    PO --> PR
    PO --> CLA
    PO --> AD
    PO --> CCO
    PO --> GR

    GSA --> Neo4j
    CC --> WS

    FC --> Physics
    SPA --> ML
    CLA --> ML
    PR --> ML

    Neo4j --> GraphData
    Neo4j --> Ontologies
    Neo4j --> UserSettings
    Neo4j --> Workspaces

    GitHub --> A1
    AI --> SP
    Auth --> A3

    style Browser fill:#e1f5ff
    style Supervisor fill:#ff6b6b,color:#fff
    style Neo4j fill:#f0e1ff
    style CUDA fill:#e1ffe1
    style CQRS fill:#fff9e1
```

## 2. Hexagonal Architecture Pattern

```mermaid
graph LR
    subgraph "Outside World - Infrastructure"
        Web[Web Clients<br/>Browser/Mobile]
        DB[(Neo4j Database)]
        GPU[GPU Compute<br/>CUDA Kernels]
        External[External APIs<br/>GitHub/AI]
    end

    subgraph "Ports (Inbound - Left)"
        HTTP[HTTP Port<br/>REST API]
        WS[WebSocket Port<br/>Binary Protocol]
        CLI[CLI Port<br/>Admin Commands]
    end

    subgraph "Application Core - Domain Logic"
        direction TB
        UseCase1[Graph Operations<br/>Add/Remove Nodes]
        UseCase2[Physics Simulation<br/>Force-Directed Layout]
        UseCase3[Semantic Analysis<br/>Ontology Reasoning]
        UseCase4[Multi-client Sync<br/>WebSocket Broadcast]

        Domain[Domain Model<br/>Node, Edge, Graph]
        Rules[Business Rules<br/>Validation + Constraints]
    end

    subgraph "Ports (Outbound - Right)"
        GraphPort[GraphRepository<br/>Interface]
        PhysicsPort[PhysicsSimulator<br/>Interface]
        SemanticPort[SemanticAnalyzer<br/>Interface]
        SettingsPort[SettingsRepository<br/>Interface]
    end

    subgraph "Adapters (Outbound - Right)"
        Neo4jAdapter[Neo4j Adapter<br/>Cypher Queries]
        CudaAdapter[CUDA Adapter<br/>Kernel Execution]
        WhelkAdapter[Whelk Adapter<br/>OWL Reasoning]
        ConfigAdapter[Settings Adapter<br/>Persistence]
    end

    Web --> HTTP
    Web --> WS
    External --> CLI

    HTTP --> UseCase1
    HTTP --> UseCase4
    WS --> UseCase2
    CLI --> UseCase3

    UseCase1 --> Domain
    UseCase2 --> Domain
    UseCase3 --> Domain
    UseCase4 --> Domain

    Domain --> Rules

    UseCase1 --> GraphPort
    UseCase2 --> PhysicsPort
    UseCase3 --> SemanticPort
    UseCase4 --> SettingsPort

    GraphPort --> Neo4jAdapter
    PhysicsPort --> CudaAdapter
    SemanticPort --> WhelkAdapter
    SettingsPort --> ConfigAdapter

    Neo4jAdapter --> DB
    CudaAdapter --> GPU
    WhelkAdapter --> DB
    ConfigAdapter --> DB

    style Domain fill:#fff9e1
    style Rules fill:#ffe1e1
    style HTTP fill:#e1f5ff
    style WS fill:#e1f5ff
    style GraphPort fill:#e8f5e9
    style PhysicsPort fill:#e8f5e9
```

## 3. CQRS Pattern Implementation

```mermaid
graph TB
    subgraph "Client Layer"
        UI[User Interface<br/>React Components]
    end

    subgraph "API Gateway"
        Router[Actix Web Router<br/>Route Dispatcher]
    end

    subgraph "CQRS Handlers - Command Side (Write)"
        direction LR
        CMD1[AddNodeCommand<br/>Handler]
        CMD2[UpdatePositionsCommand<br/>Handler]
        CMD3[DeleteEdgeCommand<br/>Handler]
        CMD4[SyncGitHubCommand<br/>Handler]
        CMD5[UpdateSettingsCommand<br/>Handler]
    end

    subgraph "CQRS Handlers - Query Side (Read)"
        direction LR
        QRY1[GetGraphQuery<br/>Handler]
        QRY2[GetNodeDetailsQuery<br/>Handler]
        QRY3[SearchNodesQuery<br/>Handler]
        QRY4[GetAnalyticsQuery<br/>Handler]
        QRY5[GetSettingsQuery<br/>Handler]
    end

    subgraph "Command Processing"
        Validator[Command Validator<br/>Business Rules]
        EventBus[Event Bus<br/>Async Notifications]
        CommandRepo[Write Repository<br/>Neo4j Write Operations]
    end

    subgraph "Query Processing"
        Cache[Query Cache<br/>Redis/In-Memory]
        QueryRepo[Read Repository<br/>Neo4j Read Operations]
        Projections[Read Projections<br/>Optimized Views]
    end

    subgraph "Domain Events"
        E1[NodeAdded Event]
        E2[PositionsUpdated Event]
        E3[EdgeDeleted Event]
        E4[SettingsChanged Event]
    end

    subgraph "Event Handlers"
        H1[WebSocket Broadcaster<br/>Push to Clients]
        H2[Cache Invalidator<br/>Clear Stale Data]
        H3[Analytics Updater<br/>Recompute Metrics]
    end

    subgraph "Data Store"
        WriteSide[(Neo4j Write<br/>Master)]
        ReadSide[(Neo4j Read<br/>Replica/Cache)]
    end

    UI -->|Commands| Router
    UI -->|Queries| Router

    Router -->|Route Command| CMD1
    Router -->|Route Command| CMD2
    Router -->|Route Command| CMD3
    Router -->|Route Command| CMD4
    Router -->|Route Command| CMD5

    Router -->|Route Query| QRY1
    Router -->|Route Query| QRY2
    Router -->|Route Query| QRY3
    Router -->|Route Query| QRY4
    Router -->|Route Query| QRY5

    CMD1 --> Validator
    CMD2 --> Validator
    CMD3 --> Validator
    CMD4 --> Validator
    CMD5 --> Validator

    Validator --> CommandRepo
    CommandRepo --> WriteSide

    CommandRepo --> EventBus
    EventBus --> E1
    EventBus --> E2
    EventBus --> E3
    EventBus --> E4

    E1 --> H1
    E2 --> H1
    E3 --> H2
    E4 --> H3

    H1 --> UI
    H2 --> Cache
    H3 --> ReadSide

    QRY1 --> Cache
    QRY2 --> Cache
    QRY3 --> QueryRepo
    QRY4 --> Projections
    QRY5 --> Cache

    Cache --> QueryRepo
    QueryRepo --> ReadSide
    Projections --> ReadSide

    WriteSide -.->|Replication| ReadSide

    style CMD1 fill:#ffe1e1
    style CMD2 fill:#ffe1e1
    style CMD3 fill:#ffe1e1
    style QRY1 fill:#e1ffe1
    style QRY2 fill:#e1ffe1
    style QRY3 fill:#e1ffe1
    style EventBus fill:#fff9e1
```

## 4. Actor System Supervision Tree

```mermaid
graph TB
    Root[ActorSystem Root<br/>Actix System]

    Root --> Supervisor[GraphServiceSupervisor<br/>OneForOne Strategy<br/>Max Restarts: 3]

    Supervisor --> |Spawn & Monitor| GSA[GraphStateActor<br/>State: 7-State Machine<br/>Restart: Always]
    Supervisor --> |Spawn & Monitor| PO[PhysicsOrchestratorActor<br/>Strategy: AllForOne<br/>Restart: Always]
    Supervisor --> |Spawn & Monitor| SP[SemanticProcessorActor<br/>State: Ready/Processing<br/>Restart: Always]
    Supervisor --> |Spawn & Monitor| CC[ClientCoordinatorActor<br/>State: Active Connections<br/>Restart: OnFailure]
    Supervisor --> |Spawn & Monitor| WA[WorkspaceActor<br/>State: CRUD Operations<br/>Restart: OnFailure]
    Supervisor --> |Spawn & Monitor| SA[SettingsActor<br/>State: Config State<br/>Restart: Always]

    PO --> |AllForOne| FC[ForceComputeActor<br/>CUDA Physics]
    PO --> |AllForOne| SM[StressMajorizationActor<br/>Layout Optimization]
    PO --> |AllForOne| SF[SemanticForcesActor<br/>AI Forces]
    PO --> |AllForOne| CA[ConstraintActor<br/>Collision Detection]
    PO --> |AllForOne| OC[OntologyConstraintActor<br/>OWL Constraints]
    PO --> |AllForOne| SPA[ShortestPathActor<br/>SSSP GPU]
    PO --> |AllForOne| PR[PageRankActor<br/>Centrality]
    PO --> |AllForOne| CLA[ClusteringActor<br/>Community Detection]
    PO --> |AllForOne| AD[AnomalyDetectionActor<br/>Outlier Detection]
    PO --> |AllForOne| CCO[ConnectedComponentsActor<br/>Graph Components]
    PO --> |AllForOne| GR[GPUResourceActor<br/>Memory Management]

    style Root fill:#333,color:#fff
    style Supervisor fill:#ff6b6b,color:#fff
    style GSA fill:#4ecdc4
    style PO fill:#ffe66d
    style SP fill:#a8e6cf
    style CC fill:#ff8b94
    style FC fill:#95e1d3
    style SM fill:#95e1d3
    style SF fill:#95e1d3
```

---

---

## Related Documentation

- [Deployment & Infrastructure Diagrams](03-deployment-infrastructure.md)
- [Complete System Data Flow Documentation](../data-flow/complete-data-flows.md)
- [ASCII Diagram Deprecation - Complete Report](../../ASCII_DEPRECATION_COMPLETE.md)
- [Server Architecture](../../concepts/architecture/core/server.md)
- [Hexagonal Architecture Migration Status Report](../../architecture/HEXAGONAL_ARCHITECTURE_STATUS.md)

## 5. Component Interaction Matrix

```mermaid
graph LR
    subgraph "Client Components"
        GC[GraphCanvas<br/>Three.js]
        SS[SettingsStore<br/>Zustand]
        BD[BotsData<br/>Agent Context]
    end

    subgraph "Communication Services"
        WSS[WebSocketService<br/>Binary Protocol]
        API[UnifiedApiClient<br/>REST HTTP]
        BP[BinaryProtocol<br/>34-byte Parser]
    end

    subgraph "Server Actors"
        CCA[ClientCoordinator<br/>Actor]
        GSA2[GraphState<br/>Actor]
        POA[PhysicsOrchestrator<br/>Actor]
    end

    subgraph "Data Sources"
        N4J[(Neo4j<br/>Database)]
        GPU2[GPU Compute<br/>CUDA]
    end

    GC --> WSS
    GC --> API
    SS --> API
    BD --> WSS
    BD --> API

    WSS --> BP
    BP --> CCA
    API --> GSA2
    API --> SS

    CCA --> GSA2
    CCA --> WSS
    GSA2 --> N4J
    GSA2 --> POA
    POA --> GPU2

    style GC fill:#e3f2fd
    style WSS fill:#c8e6c9
    style CCA fill:#ff8b94
    style N4J fill:#f0e1ff
```
