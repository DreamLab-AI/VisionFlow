# Architecture Overview

This document describes the architecture of the VisionFlow Vircadia immersive experience -- a multi-user, real-time 3D knowledge graph visualization system with WebXR support.

## System Overview

VisionFlow connects a React/Three.js client to a Vircadia World Server over WebSocket. The server exposes a SQL-over-WebSocket interface backed by PostgreSQL, enabling clients to query and mutate shared entity state in real time. Peer-to-peer audio is handled via WebRTC.

```mermaid
flowchart TB
    subgraph Client["Browser Client"]
        direction TB
        UI["React 19 + Three.js"]
        VCC["VircadiaClientCore"]
        TAR["ThreeJSAvatarRenderer"]
        SAM["SpatialAudioManager"]
        CGS["CollaborativeGraphSync"]
        ESM["EntitySyncManager"]
        NO["NetworkOptimizer"]
        Q3O["Quest3Optimizer"]
        FF["FeatureFlags"]
        BWP["BinaryWebSocketProtocol"]
    end

    subgraph Server["Vircadia World Server"]
        WS["WebSocket Endpoint"]
        AUTH["Auth Provider"]
        DB[(PostgreSQL)]
        SYNC["Entity Sync Engine"]
    end

    subgraph Peers["Remote Peers"]
        P1["Peer Client A"]
        P2["Peer Client B"]
    end

    VCC <-->|"JSON over WebSocket"| WS
    CGS <-->|"Binary Protocol V3"| WS
    SAM <-->|"WebRTC (ICE/STUN/TURN)"| P1
    SAM <-->|"WebRTC (ICE/STUN/TURN)"| P2
    WS <--> AUTH
    WS <--> SYNC
    SYNC <--> DB

    UI --> VCC
    UI --> TAR
    UI --> SAM
    UI --> CGS
    TAR --> VCC
    SAM --> VCC
    CGS --> VCC
    CGS --> BWP
    ESM --> VCC
    NO --> VCC
    Q3O --> VCC

    style Client fill:#e1f5ff,stroke:#0288d1
    style Server fill:#fff3e0,stroke:#ff9800
    style Peers fill:#f3e5f5,stroke:#9c27b0
```

## Client-Server Communication

### SQL-over-WebSocket Pattern

VisionFlow uses the Vircadia World Server's SQL-over-WebSocket interface. The client sends parameterized SQL queries as JSON messages over a persistent WebSocket connection. The server executes queries against PostgreSQL and returns results.

```mermaid
sequenceDiagram
    participant C as Client
    participant WS as WebSocket
    participant S as Vircadia Server
    participant DB as PostgreSQL

    C->>WS: Connect (token + provider in URL params)
    WS->>S: Authenticate
    S-->>C: SESSION_INFO_RESPONSE (agentId, sessionId)

    C->>WS: QUERY_REQUEST {query, parameters, requestId}
    WS->>S: Parse and validate
    S->>DB: Execute parameterized SQL
    DB-->>S: Result rows
    S-->>C: QUERY_RESPONSE {result, requestId}

    S-->>C: SYNC_GROUP_UPDATES_RESPONSE (push)
    S-->>C: TICK_NOTIFICATION_RESPONSE (push)
```

### Message Types

| Type | Direction | Description |
|:-----|:----------|:------------|
| `QUERY_REQUEST` | Client to Server | Parameterized SQL query |
| `QUERY_RESPONSE` | Server to Client | Query result |
| `SESSION_INFO_RESPONSE` | Server to Client | Agent and session identifiers |
| `SYNC_GROUP_UPDATES_RESPONSE` | Server to Client | Entity change notifications |
| `TICK_NOTIFICATION_RESPONSE` | Server to Client | Server tick heartbeat |
| `GENERAL_ERROR_RESPONSE` | Server to Client | Error details |

### Binary Protocol V3

For high-frequency data (positions, agent state, graph operations), VisionFlow uses a compact binary protocol that reduces bandwidth by up to 80% compared to JSON.

| Field | Size | Description |
|:------|:-----|:------------|
| Message Type | 1 byte | Identifies the payload type |
| Protocol Version | 1 byte | Currently `3` |
| Payload Length | 2 bytes | Little-endian uint16 |
| Payload | Variable | Type-specific binary data |

Position updates use 21 bytes per agent (4-byte ID, 12-byte position, 4-byte timestamp, 1-byte flags).

## Service Layer Architecture

```mermaid
flowchart LR
    subgraph Core["Core Layer"]
        VCC["VircadiaClientCore"]
        BWP["BinaryWebSocketProtocol"]
        FF["FeatureFlags"]
    end

    subgraph Rendering["Rendering Layer"]
        TAR["ThreeJSAvatarRenderer"]
        Q3O["Quest3Optimizer"]
    end

    subgraph Collaboration["Collaboration Layer"]
        CGS["CollaborativeGraphSync"]
        ESM["EntitySyncManager"]
        SAM["SpatialAudioManager"]
    end

    subgraph Network["Network Layer"]
        NO["NetworkOptimizer"]
        GEM["GraphEntityMapper"]
    end

    TAR --> VCC
    Q3O --> VCC
    CGS --> VCC
    CGS --> BWP
    ESM --> VCC
    ESM --> GEM
    SAM --> VCC
    NO --> VCC
    FF -.->|"Guards all services"| TAR
    FF -.->|"Guards all services"| CGS
    FF -.->|"Guards all services"| SAM
    FF -.->|"Guards all services"| Q3O

    style Core fill:#e8f5e9,stroke:#4caf50
    style Rendering fill:#e1f5ff,stroke:#0288d1
    style Collaboration fill:#fff3e0,stroke:#ff9800
    style Network fill:#f3e5f5,stroke:#9c27b0
```

### Service Dependency Summary

| Service | Depends On | Responsibility |
|:--------|:-----------|:---------------|
| **VircadiaClientCore** | None (foundation) | WebSocket lifecycle, SQL queries, event bus |
| **BinaryWebSocketProtocol** | None (singleton) | Binary message encoding/decoding |
| **FeatureFlags** | None (singleton) | Runtime feature gating and rollout |
| **ThreeJSAvatarRenderer** | VircadiaClientCore, Three.js Scene | Avatar GLTF loading, position sync, nameplates |
| **SpatialAudioManager** | VircadiaClientCore, Three.js Scene | WebRTC peer connections, HRTF spatialization |
| **CollaborativeGraphSync** | VircadiaClientCore, BinaryWebSocketProtocol | Selections, annotations, presence, OT conflict resolution |
| **EntitySyncManager** | VircadiaClientCore, GraphEntityMapper | Bidirectional graph-entity mapping and sync |
| **NetworkOptimizer** | VircadiaClientCore | Delta compression, binary batching, adaptive quality |
| **Quest3Optimizer** | VircadiaClientCore, Three.js Renderer | Foveated rendering, dynamic resolution, hand tracking |
| **GraphEntityMapper** | None (utility) | Graph node/edge to Vircadia entity conversion |

## Data Flow Diagrams

### Entity Sync Lifecycle

```mermaid
sequenceDiagram
    participant App as Application
    participant ESM as EntitySyncManager
    participant GEM as GraphEntityMapper
    participant VCC as VircadiaClientCore
    participant DB as PostgreSQL

    Note over App,DB: Push: Application to Vircadia
    App->>ESM: pushGraphToVircadia(graphData)
    ESM->>GEM: mapGraphToEntities(graphData)
    GEM-->>ESM: VircadiaEntity[]
    loop Batch Insert (100 per batch)
        ESM->>VCC: query(INSERT INTO entity.entities ...)
        VCC->>DB: Execute parameterized SQL
    end

    Note over App,DB: Pull: Vircadia to Application
    App->>ESM: pullGraphFromVircadia()
    ESM->>VCC: query(SELECT * FROM entity.entities WHERE ...)
    VCC->>DB: Execute parameterized SQL
    DB-->>VCC: Entity rows
    VCC-->>ESM: QueryResult
    ESM->>GEM: entitiesToGraph(entities)
    GEM-->>ESM: GraphData {nodes, edges}
    ESM-->>App: GraphData

    Note over App,DB: Real-time Position Sync
    App->>ESM: updateNodePosition(nodeId, position)
    ESM->>ESM: Queue in pendingPositionUpdates
    ESM->>ESM: flushPositionUpdates() [every 100ms]
    ESM->>GEM: generatePositionUpdateSQL(entityName, pos)
    ESM->>VCC: query(UPDATE ... SET position)
```

### Avatar Position Broadcast

```mermaid
sequenceDiagram
    participant Cam as Camera
    participant TAR as ThreeJSAvatarRenderer
    participant VCC as VircadiaClientCore
    participant DB as PostgreSQL
    participant Remote as Remote Clients

    Note over Cam,Remote: Local Avatar Broadcast (every 100ms)
    Cam->>TAR: Camera position/rotation
    TAR->>TAR: Copy camera transform to local avatar
    TAR->>VCC: query(UPDATE entity.entities SET position=$1, rotation=$2)
    VCC->>DB: Execute parameterized SQL

    Note over Cam,Remote: Remote Avatar Fetch (on syncUpdate event)
    DB-->>VCC: SYNC_GROUP_UPDATES_RESPONSE
    VCC-->>TAR: syncUpdate event
    TAR->>VCC: query(SELECT * FROM entity.entities WHERE name LIKE 'avatar_%')
    VCC->>DB: Execute parameterized SQL
    DB-->>VCC: Avatar entity rows
    VCC-->>TAR: QueryResult
    TAR->>TAR: loadRemoteAvatar() or updateAvatarPosition()
```

### WebRTC Signaling Flow

```mermaid
sequenceDiagram
    participant A as Client A (SpatialAudioManager)
    participant DB as Vircadia DB (Signaling Store)
    participant B as Client B (SpatialAudioManager)

    Note over A,B: Offer/Answer Exchange via Entity Storage
    A->>A: createOffer()
    A->>DB: INSERT webrtc_offer_A_B (offer SDP)

    B->>DB: SELECT webrtc_* WHERE to = B
    DB-->>B: Offer from A
    B->>B: setRemoteDescription(offer)
    B->>B: createAnswer()
    B->>DB: INSERT webrtc_answer_B_A (answer SDP)

    A->>DB: SELECT webrtc_* WHERE to = A
    DB-->>A: Answer from B
    A->>A: setRemoteDescription(answer)

    Note over A,B: ICE Candidate Exchange
    A->>DB: INSERT webrtc_ice_A_B (candidate)
    B->>DB: SELECT webrtc_ice WHERE to = B
    DB-->>B: ICE candidate
    B->>B: addIceCandidate()

    Note over A,B: Peer Connection Established
    A<-->B: Direct WebRTC Audio (HRTF spatialized)
```

### Collaborative Graph Editing

```mermaid
sequenceDiagram
    participant U1 as User 1
    participant CGS1 as CollaborativeGraphSync (User 1)
    participant WS as WebSocket Server
    participant CGS2 as CollaborativeGraphSync (User 2)
    participant U2 as User 2

    U1->>CGS1: selectNodes([nodeA, nodeB])
    CGS1->>WS: Binary SELECTION_UPDATE (0x52)
    WS-->>CGS2: Binary SELECTION_UPDATE
    CGS2->>CGS2: updateSelectionHighlight()
    CGS2-->>U2: Torus highlight rings on nodeA, nodeB

    U1->>CGS1: createAnnotation(nodeA, "Important finding")
    CGS1->>WS: Binary ANNOTATION_UPDATE (0x51)
    WS-->>CGS2: Binary ANNOTATION_UPDATE
    CGS2->>CGS2: createAnnotationMesh()
    CGS2-->>U2: Annotation plane rendered at node position

    Note over U1,U2: Conflict Resolution (Operational Transform)
    U1->>CGS1: moveNode(nodeA, pos1)
    U2->>CGS2: moveNode(nodeA, pos2)
    CGS1->>CGS1: resolveConflict() -- last-write-wins with userId tiebreak
    CGS2->>CGS2: resolveConflict() -- deterministic ordering
```

## Technology Stack

| Layer | Technology | Version |
|:------|:-----------|:--------|
| **UI Framework** | React | 19.x |
| **3D Rendering** | Three.js (React Three Fiber) | 0.182.x |
| **XR Runtime** | WebXR (@react-three/xr) | 6.x |
| **Language** | TypeScript | 5.9.x |
| **Build Tool** | Vite | 6.x |
| **Testing** | Vitest + Playwright | 4.x / 1.57.x |
| **Styling** | Tailwind CSS | 4.x |
| **Backend** | Rust (Actix-web) | 1.75+ |
| **Database** | PostgreSQL (via Vircadia World Server) | 15 |
| **GPU Compute** | CUDA 12.4 | 100+ kernels |
| **Ontology** | OWL 2 EL (Whelk-rs) | - |
| **Networking** | WebSocket (JSON + Binary V3), WebRTC | - |
| **Spatial Audio** | Web Audio API (HRTF PannerNode) | - |
| **Avatar Format** | glTF/GLB (via GLTFLoader) | 2.0 |
| **License** | Mozilla Public License 2.0 | - |

## Deployment Topology

```mermaid
flowchart TB
    subgraph Docker["Docker Compose Stack"]
        VWS["Vircadia World Server\n:3020 (WS) / :3021 (HTTP)"]
        PG[(PostgreSQL 15\n:5432)]
        VWS <--> PG
    end

    subgraph Clients["Browser Clients"]
        C1["Client 1\n(Desktop)"]
        C2["Client 2\n(Quest 3)"]
        C3["Client N\n(Mobile)"]
    end

    C1 <-->|"wss://host:3020/world/ws"| VWS
    C2 <-->|"wss://host:3020/world/ws"| VWS
    C3 <-->|"wss://host:3020/world/ws"| VWS
    C1 <-->|"WebRTC P2P"| C2
    C2 <-->|"WebRTC P2P"| C3

    style Docker fill:#fff3e0,stroke:#ff9800
    style Clients fill:#e1f5ff,stroke:#0288d1
```

The Vircadia World Server is deployed as a Docker container alongside PostgreSQL. Authentication supports system tokens and Nostr identity. Clients connect via WebSocket with token-based auth, and establish direct WebRTC connections for spatial audio.
