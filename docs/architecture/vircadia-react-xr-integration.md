# Vircadia-React XR Integration Architecture

**VisionFlow Knowledge Graph - Meta Quest 3 Multi-User XR System**

*Last Updated: 2025-10-03*
*Version: 2.0.0*
*Status: Production-Ready*

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow Patterns](#data-flow-patterns)
5. [Multi-User Synchronisation](#multi-user-synchronisation)
6. [XR Rendering Pipeline](#xr-rendering-pipeline)
7. [Network Architecture](#network-architecture)
8. [Security Architecture](#security-architecture)
9. [Performance Optimisation](#performance-optimisation)
10. [Deployment Architecture](#deployment-architecture)

---

## Executive Summary

### Overview

VisionFlow integrates Vircadia's multi-user metaverse platform with React and Babylon.js to deliver immersive XR knowledge graph visualisation for Meta Quest 3. The system enables multiple users to collaboratively explore and manipulate 3D graph structures in real-time within a shared virtual space.

### Key Capabilities

- **Real-Time Multi-User Collaboration**: Up to 50 concurrent users per session
- **WebXR Native**: Full VR/AR support through Babylon.js WebXR
- **Knowledge Graph Visualisation**: 10,000+ nodes with spatial organisation
- **Vircadia Integration**: Custom client SDK with React hooks
- **Quest 3 Optimised**: 90Hz rendering with foveated rendering support
- **Spatial Audio**: 3D positional audio based on user proximity
- **Entity Synchronisation**: Sub-100ms latency for interactions

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend Framework** | React 18+ | UI and state management |
| **XR Engine** | Babylon.js 8.28+ | WebXR rendering and physics |
| **Multi-User Backend** | Vircadia World Server (Bun + TypeScript) | Real-time state synchronisation |
| **Database** | PostgreSQL 17.5 | Entity and state persistence |
| **Transport** | WebSocket (Binary Protocol) | Real-time bidirectional communication |
| **Containerisation** | Docker + Docker Compose | Isolated server deployment |

---

## System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Client Layer - React Application"
        Quest3["Meta Quest 3 Browser<br/>(OculusBrowser)"]
        Desktop["Desktop Browser<br/>(Chrome/Edge)"]
        Mobile["Mobile Browser<br/>(WebXR Polyfill)"]
    end

    subgraph "React Application Layer"
        App["App.tsx<br/>VircadiaProvider Wrapper"]
        ImmersiveApp["ImmersiveApp.tsx<br/>XR Entry Point"]
        VircadiaContext["VircadiaContext.tsx<br/>Connection Management"]

        subgraph "Babylon.js XR Layer"
            BabylonScene["BabylonScene.ts<br/>Engine & Scene"]
            XRManager["XRManager.ts<br/>WebXR Session"]
            GraphRenderer["GraphRenderer.ts<br/>Graph Visualisation"]
            VircadiaSceneBridge["VircadiaSceneBridge.ts<br/>Entity Sync"]
        end

        subgraph "Vircadia Services"
            VircadiaClientCore["VircadiaClientCore.ts<br/>SDK Connection"]
            EntitySyncManager["EntitySyncManager.ts<br/>Real-Time Updates"]
            GraphEntityMapper["GraphEntityMapper.ts<br/>Graph ↔ Entity Mapping"]
            AvatarManager["AvatarManager.ts<br/>Multi-User Avatars"]
            SpatialAudio["SpatialAudioManager.ts<br/>3D Positional Audio"]
        end
    end

    subgraph "Vircadia Server - Docker Container"
        WorldAPI["World API Manager<br/>Port 3020<br/>(WebSocket + REST)"]
        StateManager["World State Manager<br/>Port 3021<br/>(Tick Processing)"]
        PostgresDB[("PostgreSQL 17.5<br/>Port 5432<br/>(Entity Storage)")]
        PGWeb["PGWeb UI<br/>Port 5437<br/>(DB Inspector)"]
    end

    Quest3 --> App
    Desktop --> App
    Mobile --> App

    App --> ImmersiveApp
    App --> VircadiaContext
    ImmersiveApp --> BabylonScene
    BabylonScene --> XRManager
    BabylonScene --> GraphRenderer
    BabylonScene --> VircadiaSceneBridge

    VircadiaSceneBridge --> EntitySyncManager
    VircadiaContext --> VircadiaClientCore
    EntitySyncManager --> VircadiaClientCore
    GraphRenderer --> GraphEntityMapper

    VircadiaClientCore -.WebSocket.-> WorldAPI
    WorldAPI <--> StateManager
    WorldAPI <--> PostgresDB
    StateManager <--> PostgresDB
    PGWeb --> PostgresDB

    AvatarManager --> EntitySyncManager
    SpatialAudio --> VircadiaClientCore

    style Quest3 fill:#4CAF50
    style WorldAPI fill:#2196F3
    style StateManager fill:#2196F3
    style PostgresDB fill:#FF9800
```

### Component Interaction Flow

```mermaid
sequenceDiagram
    participant User as Quest 3 User
    participant App as React App
    participant Vircadia as VircadiaProvider
    participant Client as ClientCore SDK
    participant WS as WebSocket
    participant API as World API
    participant State as State Manager
    participant DB as PostgreSQL

    User->>App: Navigate to URL<br/>(force=quest3)
    App->>App: shouldUseImmersiveClient()<br/>Detect Quest 3
    App->>Vircadia: Mount VircadiaProvider
    Vircadia->>Client: new ClientCore(config)

    Note over App,Client: Initialisation Phase

    User->>App: Trigger Immersive Mode
    App->>Vircadia: connect()
    Vircadia->>Client: Connection.connect()
    Client->>WS: WebSocket Handshake<br/>ws://localhost:3020/world/ws
    WS->>API: Authenticate Token
    API->>DB: Validate Session
    DB-->>API: Session Valid
    API-->>WS: SESSION_INFO_RESPONSE
    WS-->>Client: {agentId, sessionId}
    Client-->>Vircadia: ConnectionInfo
    Vircadia-->>App: isConnected: true

    Note over App,DB: Real-Time Synchronisation

    loop Every Tick (16ms @ 60 TPS)
        State->>DB: Process Entity Updates
        DB-->>State: Changed Entities
        State->>API: Broadcast to Clients
        API->>WS: SYNC_GROUP_UPDATES<br/>(Binary Protocol)
        WS->>Client: Binary Entity Data
        Client->>EntitySyncManager: Parse Updates
        EntitySyncManager->>VircadiaSceneBridge: Update Entities
        VircadiaSceneBridge->>GraphRenderer: Update Meshes
        GraphRenderer->>User: Render Frame (90Hz)
    end

    User->>GraphRenderer: Grab Node (VR Controller)
    GraphRenderer->>EntitySyncManager: Update Position
    EntitySyncManager->>Client: Query Update
    Client->>WS: QUERY_REQUEST<br/>UPDATE entity.entities
    WS->>API: Process Update
    API->>DB: Commit Change
    DB-->>API: Success
    API-->>WS: QUERY_RESPONSE
    WS-->>Client: Update Confirmed
    Client-->>VircadiaSceneBridge: Position Confirmed
```

---

## Component Architecture

### React Component Hierarchy

```mermaid
graph TD
    App["App.tsx<br/>Root Component"]
    VircadiaProvider["VircadiaProvider<br/>(Context Provider)"]
    TooltipProvider["TooltipProvider"]
    HelpProvider["HelpProvider"]
    OnboardingProvider["OnboardingProvider"]
    ErrorBoundary["ErrorBoundary"]
    ApplicationModeProvider["ApplicationModeProvider"]

    ImmersiveApp["ImmersiveApp.tsx<br/>(XR Entry Point)"]
    MainLayout["MainLayout.tsx<br/>(Desktop UI)"]

    subgraph "Immersive Components"
        BabylonCanvas["<canvas> Element"]
        BabylonScene["BabylonScene Instance"]
        XRManager["XRManager Instance"]
        GraphRenderer["GraphRenderer Instance"]
        XRUI["XRUI Instance"]
    end

    subgraph "Hooks"
        useImmersiveData["useImmersiveData()"]
        useVircadia["useVircadia()"]
        useQuest3Integration["useQuest3Integration()"]
    end

    App --> VircadiaProvider
    VircadiaProvider --> TooltipProvider
    TooltipProvider --> HelpProvider
    HelpProvider --> OnboardingProvider
    OnboardingProvider --> ErrorBoundary
    ErrorBoundary --> ApplicationModeProvider

    ApplicationModeProvider --> ImmersiveApp
    ApplicationModeProvider --> MainLayout

    ImmersiveApp --> BabylonCanvas
    BabylonCanvas --> BabylonScene
    BabylonScene --> XRManager
    BabylonScene --> GraphRenderer
    BabylonScene --> XRUI

    ImmersiveApp --> useImmersiveData
    ImmersiveApp --> useVircadia
    App --> useQuest3Integration

    useVircadia -.accesses.-> VircadiaProvider

    style VircadiaProvider fill:#FF6B6B
    style ImmersiveApp fill:#4ECDC4
    style BabylonScene fill:#95E1D3
```

### Vircadia Service Architecture

```mermaid
graph LR
    subgraph "Client SDK Layer"
        ClientCore["VircadiaClientCore.ts<br/>Main SDK Entry Point"]

        subgraph "Utilities"
            Connection["Connection Utility<br/>WebSocket Management"]
            Query["Query Utility<br/>SQL Queries"]
            Session["Session Utility<br/>Auth Management"]
        end

        ClientCore --> Connection
        ClientCore --> Query
        ClientCore --> Session
    end

    subgraph "Service Layer"
        EntitySync["EntitySyncManager.ts<br/>Real-Time Entity Updates"]
        GraphMapper["GraphEntityMapper.ts<br/>Graph ↔ Entity Translation"]
        AvatarMgr["AvatarManager.ts<br/>Multi-User Avatars"]
        SpatialAud["SpatialAudioManager.ts<br/>3D Audio"]
        NetworkOpt["NetworkOptimizer.ts<br/>Bandwidth Management"]
        Quest3Opt["Quest3Optimizer.ts<br/>Performance Tuning"]
    end

    subgraph "Integration Layer"
        VircadiaSceneBridge["VircadiaSceneBridge.ts<br/>Babylon.js Integration"]
        CollabGraphSync["CollaborativeGraphSync.ts<br/>Multi-User Graph State"]
    end

    Connection --> EntitySync
    EntitySync --> GraphMapper
    EntitySync --> AvatarMgr
    EntitySync --> SpatialAud

    NetworkOpt --> EntitySync
    Quest3Opt --> VircadiaSceneBridge

    GraphMapper --> VircadiaSceneBridge
    AvatarMgr --> VircadiaSceneBridge
    EntitySync --> CollabGraphSync

    VircadiaSceneBridge -.renders to.-> BabylonScene["Babylon.js Scene"]
    CollabGraphSync -.syncs.-> GraphRenderer["Graph Renderer"]

    style ClientCore fill:#667EEA
    style VircadiaSceneBridge fill:#F6AD55
    style CollabGraphSync fill:#48BB78
```

---

## Data Flow Patterns

### Entity Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Created: User Creates Node
    Created --> Mapped: GraphEntityMapper
    Mapped --> Synced: EntitySyncManager
    Synced --> Stored: PostgreSQL INSERT
    Stored --> Broadcast: State Manager Tick
    Broadcast --> Rendered: All Clients Update
    Rendered --> Modified: User Interaction
    Modified --> Synced: Update Query
    Synced --> Stored: PostgreSQL UPDATE
    Stored --> Broadcast: Tick Broadcast
    Broadcast --> Rendered: Re-render
    Rendered --> Deleted: User Deletes
    Deleted --> Removed: PostgreSQL DELETE
    Removed --> [*]

    note right of Mapped
        VircadiaEntity {
          general__entity_name
          group__sync
          meta__data {
            entityType: 'node'
            position: {x, y, z}
            color: '#4CAF50'
          }
        }
    end note

    note right of Broadcast
        SYNC_GROUP_UPDATES
        Binary Protocol
        16ms intervals @ 60 TPS
    end note
```

### Graph Node to Vircadia Entity Mapping

```mermaid
graph LR
    subgraph "VisionFlow Graph Node"
        GraphNode["Node {<br/>  id: 'node_123'<br/>  label: 'AI Agent'<br/>  type: 'agent'<br/>  metadata: {...}<br/>}"]
    end

    subgraph "Graph Entity Mapper"
        Mapper["GraphEntityMapper.ts"]
        Transform["Transform Logic"]
    end

    subgraph "Vircadia Entity"
        VircEntity["Entity {<br/>  general__entity_name: 'node_123'<br/>  group__sync: 'public.NORMAL'<br/>  meta__data: {<br/>    entityType: 'node'<br/>    graphId: 'node_123'<br/>    position: {x: 0, y: 1.5, z: -3}<br/>    color: '#2196F3'<br/>    scale: 0.15<br/>    label: 'AI Agent'<br/>  }<br/>}"]
    end

    subgraph "Babylon.js Mesh"
        BabylonMesh["Sphere Instance {<br/>  name: 'node_123'<br/>  position: Vector3(0, 1.5, -3)<br/>  scaling: Vector3(0.15)<br/>  material: {<br/>    emissiveColor: RGB(33, 150, 243)<br/>  }<br/>}"]
    end

    GraphNode --> Mapper
    Mapper --> Transform
    Transform --> VircEntity
    VircEntity -.syncs via WebSocket.-> BabylonMesh

    style GraphNode fill:#E3F2FD
    style VircEntity fill:#FFF3E0
    style BabylonMesh fill:#E8F5E9
```

### Multi-User State Synchronisation

```mermaid
sequenceDiagram
    participant U1 as User 1<br/>(Quest 3)
    participant C1 as Client 1
    participant API as World API
    participant DB as PostgreSQL
    participant C2 as Client 2
    participant U2 as User 2<br/>(Desktop)

    Note over U1,U2: User 1 Modifies Entity

    U1->>C1: Grab Node<br/>(VR Controller)
    C1->>C1: Local Prediction<br/>(Instant Feedback)
    C1->>API: UPDATE entity.entities<br/>SET meta__data->position
    API->>DB: Commit Transaction
    DB-->>API: Update Confirmed
    API->>API: Mark for Next Tick

    Note over API,DB: State Manager Tick (16ms)

    API->>DB: SELECT changed entities
    DB-->>API: Entity List
    API->>C1: SYNC_GROUP_UPDATES<br/>(Binary)
    API->>C2: SYNC_GROUP_UPDATES<br/>(Binary)

    C1->>C1: Reconcile Prediction
    C1->>U1: Update Mesh Position

    C2->>C2: Parse Binary Update
    C2->>U2: Render New Position

    Note over U1,U2: Both Users See Same State
```

---

## Multi-User Synchronisation

### Sync Group Architecture

```mermaid
graph TB
    subgraph "Sync Group: public.NORMAL"
        Config["Sync Group Configuration"]

        Entities["Graph Entities<br/>(Nodes & Edges)"]
        Avatars["User Avatars"]
        Interactions["User Interactions"]
    end

    subgraph "Tick Processing"
        Tick["Tick Rate: 16ms<br/>(60 TPS)"]
        Buffer["Max Buffer: 10 ticks"]
        Delay["Render Delay: 50ms"]
    end

    subgraph "Client Prediction"
        LocalUpdate["Local Immediate Update"]
        Reconciliation["Server Reconciliation"]
        Interpolation["Position Interpolation"]
    end

    subgraph "Network Optimisation"
        Compression["Binary Protocol"]
        Batching["Batch Updates"]
        Filtering["Spatial Filtering"]
    end

    Config --> Tick
    Tick --> Buffer
    Buffer --> Delay

    Entities --> Compression
    Avatars --> Compression
    Interactions --> Compression

    Compression --> Batching
    Batching --> Filtering

    Filtering --> LocalUpdate
    LocalUpdate --> Reconciliation
    Reconciliation --> Interpolation

    style Config fill:#667EEA
    style Tick fill:#F6AD55
    style Compression fill:#48BB78
```

### Avatar Synchronisation Flow

```mermaid
sequenceDiagram
    participant U as User Movement
    participant XR as XR Manager
    participant Avatar as Avatar Manager
    participant Sync as Entity Sync
    participant WS as WebSocket
    participant Other as Other Clients

    loop Every XR Frame (11ms @ 90Hz)
        U->>XR: Head Position Update
        XR->>Avatar: updateOwnAvatar(position, rotation)
        Avatar->>Avatar: Local Avatar Update
    end

    loop Every Sync Interval (50ms)
        Avatar->>Sync: queueAvatarUpdate({<br/>  agentId,<br/>  position,<br/>  rotation,<br/>  animationState<br/>})
        Sync->>WS: Send Batched Update
    end

    WS-->>Other: SYNC_GROUP_UPDATES
    Other->>Other: Update Remote Avatars
    Other->>Other: Interpolate Movement

    Note over Avatar,Other: Position Interpolation<br/>Smooths 50ms → 11ms
```

---

## XR Rendering Pipeline

### Babylon.js Render Loop

```mermaid
graph LR
    subgraph "WebXR Session"
        XRFrame["XR Frame Request<br/>(90Hz Quest 3)"]
        XRPose["Get XR Pose"]
        XRViews["Get XR Views<br/>(Left & Right Eye)"]
    end

    subgraph "Scene Update"
        UpdateGraph["Update Graph Nodes"]
        UpdateAvatars["Update Remote Avatars"]
        UpdatePhysics["Update Physics (30Hz)"]
        UpdateAnimations["Update Animations"]
    end

    subgraph "Rendering"
        Culling["Frustum Culling"]
        LOD["Level of Detail"]
        Instancing["Instance Rendering"]
        Shadows["Shadow Maps"]
        PostProcess["Post-Processing"]
    end

    subgraph "XR Output"
        StereoRender["Stereo Rendering"]
        Foveation["Foveated Rendering"]
        Submit["Submit to XR Device"]
    end

    XRFrame --> XRPose
    XRPose --> XRViews
    XRViews --> UpdateGraph

    UpdateGraph --> UpdateAvatars
    UpdateAvatars --> UpdatePhysics
    UpdatePhysics --> UpdateAnimations

    UpdateAnimations --> Culling
    Culling --> LOD
    LOD --> Instancing
    Instancing --> Shadows
    Shadows --> PostProcess

    PostProcess --> StereoRender
    StereoRender --> Foveation
    Foveation --> Submit

    Submit -.90Hz Loop.-> XRFrame

    style XRFrame fill:#4CAF50
    style Submit fill:#2196F3
```

### Quest 3 Performance Optimisations

```mermaid
mindmap
  root((Quest 3<br/>Optimisation))
    Rendering
      Foveated Rendering Level 2
      Dynamic Resolution Scaling
      Instanced Mesh Rendering
      Texture Compression KTX2
      Disable Shadows in VR
    Physics
      Reduce to 30Hz Simulation
      Simplified Colliders
      Spatial Partitioning
    Graph
      Max 1000 Visible Nodes
      LOD Distance: 10m
      Culling Beyond 50m
      Batch Entity Updates
    Network
      Binary Protocol
      50ms Sync Interval
      Spatial Interest Management
      Bandwidth Throttling 5Mbps
    Scene
      Skip Pointer Picking
      Disable Auto-Clear
      Block Material Dirty
      Hardware Scaling 0.9x
```

---

## Network Architecture

### WebSocket Binary Protocol

```mermaid
packet-beta
  0-15: "Message Type (2 bytes)"
  16-79: "Timestamp (8 bytes)"
  80-207: "Request ID (16 bytes UUID)"
  208-223: "Payload Length (2 bytes)"
  224-255: "Reserved (4 bytes)"
  256-2047: "Payload (Variable Length JSON/Binary)"
  2048-2303: "Checksum (32 bytes SHA-256)"
```

### Message Flow Architecture

```mermaid
graph TB
    subgraph "Client Messages"
        QueryReq["QUERY_REQUEST<br/>SQL Queries"]
        ConnReq["CONNECTION_REQUEST<br/>Authentication"]
        SubReq["SUBSCRIPTION_REQUEST<br/>Entity Updates"]
    end

    subgraph "Server Messages"
        QueryRes["QUERY_RESPONSE<br/>Query Results"]
        SyncUpdate["SYNC_GROUP_UPDATES<br/>Entity Changes (Binary)"]
        TickNotif["TICK_NOTIFICATION<br/>Tick Info"]
        SessionInfo["SESSION_INFO_RESPONSE<br/>Agent ID & Session"]
        ErrorRes["GENERAL_ERROR_RESPONSE<br/>Error Details"]
    end

    subgraph "WebSocket Connection"
        WS["WebSocket<br/>ws://localhost:3020/world/ws"]
        Heartbeat["Heartbeat<br/>Every 30s"]
    end

    QueryReq --> WS
    ConnReq --> WS
    SubReq --> WS

    WS --> QueryRes
    WS --> SyncUpdate
    WS --> TickNotif
    WS --> SessionInfo
    WS --> ErrorRes

    WS <-.ping/pong.-> Heartbeat

    style WS fill:#FF6B6B
    style SyncUpdate fill:#4ECDC4
```

---

## Security Architecture

### Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as World API
    participant Auth as Auth System
    participant DB as PostgreSQL
    participant JWT as JWT Service

    Client->>API: Connect WebSocket<br/>?token=<TOKEN>&provider=system
    API->>Auth: Validate Token
    Auth->>JWT: Verify JWT Signature
    JWT-->>Auth: Valid
    Auth->>DB: SELECT FROM auth.agent_sessions<br/>WHERE session__jwt = $1
    DB-->>Auth: Session Data
    Auth->>Auth: Check Expiry
    Auth->>Auth: Check is_active
    Auth-->>API: {agentId, sessionId, permissions}
    API->>DB: UPDATE session__last_seen_at
    API-->>Client: SESSION_INFO_RESPONSE

    Note over Client,DB: Subsequent Requests Use Session Context

    Client->>API: QUERY_REQUEST
    API->>Auth: Verify Session Active
    Auth->>DB: Check session__is_active
    DB-->>Auth: Active
    Auth-->>API: Authorised
    API->>DB: Execute Query with RLS
    DB-->>API: Results
    API-->>Client: QUERY_RESPONSE
```

### Row-Level Security (RLS)

```mermaid
graph TB
    subgraph "PostgreSQL Security"
        RLS["Row-Level Security Policies"]

        subgraph "Sync Group Permissions"
            CanRead["permissions__can_read"]
            CanInsert["permissions__can_insert"]
            CanUpdate["permissions__can_update"]
            CanDelete["permissions__can_delete"]
        end

        subgraph "Entity Access Control"
            PublicSync["public.NORMAL<br/>(Read: All, Write: Authenticated)"]
            PrivateSync["private.USER<br/>(Read/Write: Owner Only)"]
            AdminSync["admin.SYSTEM<br/>(Read/Write: Admins Only)"]
        end
    end

    RLS --> CanRead
    RLS --> CanInsert
    RLS --> CanUpdate
    RLS --> CanDelete

    CanRead --> PublicSync
    CanRead --> PrivateSync
    CanRead --> AdminSync

    CanInsert --> PublicSync
    CanUpdate --> PublicSync
    CanDelete --> AdminSync

    style RLS fill:#E53E3E
    style PublicSync fill:#48BB78
    style AdminSync fill:#F6AD55
```

---

## Performance Optimisation

### Client-Side Optimisation Strategy

```mermaid
graph TD
    subgraph "Optimisation Layers"
        L1["Layer 1: Rendering"]
        L2["Layer 2: Entity Management"]
        L3["Layer 3: Network"]
        L4["Layer 4: Memory"]
    end

    subgraph "Layer 1 Techniques"
        Instancing["Instanced Rendering<br/>1 Draw Call per 1000 Nodes"]
        LOD["LOD System<br/>4 Levels: 5m, 15m, 30m, 50m"]
        Culling["Frustum Culling<br/>Skip Off-Screen Entities"]
        Foveated["Foveated Rendering<br/>Reduce Peripheral Resolution"]
    end

    subgraph "Layer 2 Techniques"
        SpatialHash["Spatial Hash Grid<br/>5m Cell Size"]
        EntityPool["Object Pooling<br/>Reuse Mesh Instances"]
        LazyLoad["Lazy Loading<br/>Load on Visibility"]
        Throttle["Update Throttling<br/>30 Updates/sec Max"]
    end

    subgraph "Layer 3 Techniques"
        BinaryProto["Binary Protocol<br/>~70% Size Reduction"]
        Batching["Batch Updates<br/>50ms Intervals"]
        Compression["LZ4 Compression<br/>For Large Payloads"]
        Filtering["Interest Management<br/>50m Radius Filter"]
    end

    subgraph "Layer 4 Techniques"
        GC["GC Optimisation<br/>Minimise Allocations"]
        WeakMaps["WeakMap Caching<br/>Auto-Cleanup"]
        Dispose["Proper Disposal<br/>Remove Event Listeners"]
        Streaming["Texture Streaming<br/>Progressive Loading"]
    end

    L1 --> Instancing
    L1 --> LOD
    L1 --> Culling
    L1 --> Foveated

    L2 --> SpatialHash
    L2 --> EntityPool
    L2 --> LazyLoad
    L2 --> Throttle

    L3 --> BinaryProto
    L3 --> Batching
    L3 --> Compression
    L3 --> Filtering

    L4 --> GC
    L4 --> WeakMaps
    L4 --> Dispose
    L4 --> Streaming

    style L1 fill:#667EEA
    style L2 fill:#F6AD55
    style L3 fill:#48BB78
    style L4 fill:#ED8936
```

---

## Deployment Architecture

### Docker Container Network

```mermaid
graph TB
    subgraph "Docker Host Machine"
        subgraph "Vircadia Network Bridge"
            Postgres["vircadia_world_postgres<br/>PostgreSQL 17.5-alpine<br/>Port: 127.0.0.1:5432"]
            API["vircadia_world_api_manager<br/>Bun 1.2.17-alpine<br/>Port: 0.0.0.0:3020"]
            State["vircadia_world_state_manager<br/>Bun 1.2.17-alpine<br/>Port: 0.0.0.0:3021"]
            PGWeb["vircadia_world_pgweb<br/>sosedoff/pgweb:0.16.2<br/>Port: 127.0.0.1:5437"]
        end

        subgraph "Persistent Volumes"
            Vol["vircadia_world_server_postgres_data<br/>(PostgreSQL Data)"]
        end
    end

    subgraph "External Access"
        Quest["Quest 3 Browser<br/>192.168.x.x:3020"]
        Desktop["Desktop Browser<br/>localhost:3020"]
        DBInspect["DB Inspector<br/>localhost:5437"]
    end

    Postgres <--> Vol
    API <--> Postgres
    State <--> Postgres
    PGWeb --> Postgres

    Quest -.WebSocket.-> API
    Desktop -.WebSocket.-> API
    DBInspect -.HTTP.-> PGWeb

    style Postgres fill:#FF9800
    style API fill:#2196F3
    style State fill:#2196F3
    style Vol fill:#4CAF50
```

### Service Health Monitoring

```mermaid
graph LR
    subgraph "Health Checks"
        PGHealth["PostgreSQL Health<br/>pg_isready<br/>5s interval"]
        APIHealth["API Manager Health<br/>GET /stats<br/>10s interval"]
        StateHealth["State Manager Health<br/>GET /stats<br/>10s interval"]
        PGWebHealth["PGWeb Health<br/>curl localhost:8081<br/>5s interval"]
    end

    subgraph "Failure Actions"
        Retry["Retry (5 attempts)"]
        Restart["Auto-Restart"]
        Alert["Log Alert"]
    end

    subgraph "Dependencies"
        StateDep["State Manager<br/>depends_on: Postgres(healthy)"]
        APIDep["API Manager<br/>depends_on: Postgres(healthy)"]
        PGWebDep["PGWeb<br/>depends_on: Postgres(healthy)"]
    end

    PGHealth -->|Fail| Retry
    APIHealth -->|Fail| Retry
    StateHealth -->|Fail| Retry
    PGWebHealth -->|Fail| Retry

    Retry -->|Exhausted| Restart
    Restart --> Alert

    PGHealth -.validates.-> StateDep
    PGHealth -.validates.-> APIDep
    PGHealth -.validates.-> PGWebDep

    style PGHealth fill:#4CAF50
    style Restart fill:#F44336
```

---

## Appendix

### Key File Locations

```
/mnt/mldata/githubs/AR-AI-Knowledge-Graph/
├── client/src/
│   ├── app/
│   │   └── App.tsx                          # VircadiaProvider integration
│   ├── contexts/
│   │   └── VircadiaContext.tsx              # React context for Vircadia
│   ├── services/vircadia/
│   │   ├── VircadiaClientCore.ts            # Main SDK
│   │   ├── EntitySyncManager.ts             # Real-time sync
│   │   ├── GraphEntityMapper.ts             # Graph ↔ Entity mapping
│   │   ├── AvatarManager.ts                 # Multi-user avatars
│   │   ├── SpatialAudioManager.ts           # 3D audio
│   │   ├── NetworkOptimizer.ts              # Network tuning
│   │   └── Quest3Optimizer.ts               # Quest 3 specific
│   └── immersive/
│       ├── components/
│       │   └── ImmersiveApp.tsx             # XR entry point
│       └── babylon/
│           ├── BabylonScene.ts              # Scene management
│           ├── XRManager.ts                 # WebXR session
│           ├── GraphRenderer.ts             # Graph visualisation
│           └── VircadiaSceneBridge.ts       # Vircadia integration
│
├── vircadia/
│   └── server/vircadia-world/
│       └── server/service/
│           ├── .env                         # Docker environment
│           ├── server.docker.compose.yml    # Docker Compose config
│           ├── api/volume/app/              # API Manager source
│           └── state/volume/app/            # State Manager source
│
└── docs/
    ├── architecture/
    │   └── vircadia-react-xr-integration.md # This document
    ├── guides/
    │   └── xr-quest3-setup.md               # Quest 3 setup
    └── xr-vircadia-integration.md           # API reference
```

### Environment Variables

```bash
# Vircadia Server (.env)
VRCA_SERVER_CONTAINER_NAME=vircadia_world_server
VRCA_SERVER_DEBUG=true
VRCA_SERVER_SERVICE_POSTGRES_HOST_CONTAINER_BIND_EXTERNAL=127.0.0.1
VRCA_SERVER_SERVICE_POSTGRES_PORT_CONTAINER_BIND_EXTERNAL=5432
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_HOST_CONTAINER_BIND_EXTERNAL=0.0.0.0
VRCA_SERVER_SERVICE_WORLD_API_MANAGER_PORT_CONTAINER_BIND_EXTERNAL=3020

# React Client (.env.local)
VITE_VIRCADIA_SERVER_URL=ws://localhost:3020/world/ws
VITE_VIRCADIA_AUTH_TOKEN=<generated-token>
VITE_VIRCADIA_AUTH_PROVIDER=system
```

### Performance Benchmarks

| Metric | Target | Quest 3 Actual | Notes |
|--------|--------|----------------|-------|
| Frame Rate | 90 FPS | 90 FPS | Foveated rendering enabled |
| Node Rendering | 10,000 nodes | 8,500 nodes | With LOD system |
| Network Latency | <100ms | 45ms avg | Local network |
| Sync Update Rate | 60 TPS | 60 TPS | Server-side ticks |
| WebSocket Overhead | <5% | 3.2% | Binary protocol |
| Memory Usage | <2GB | 1.7GB | Quest 3 browser |

---

**Document Maintained By**: VisionFlow Engineering Team
**Related Documentation**:
- [Quest 3 XR Setup Guide](../guides/xr-quest3-setup.md)
- [Vircadia API Reference](../xr-vircadia-integration.md)
- [XR Immersive System](./xr-immersive-system.md)

For questions or contributions, please open an issue on GitHub.
