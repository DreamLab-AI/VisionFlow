# VisionFlow System Architecture Diagrams

**Version:** 1.0.0
**Last Updated:** 2025-10-27
**Status:** Production-Ready Architecture

---

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        WEB[React + Three.js Client<br/>Port 3001 Dev / 4000 Prod]
        MOBILE[Logseq Mobile<br/>Markdown Sync]
    end

    subgraph "VisionFlow Container"
        NGINX[Nginx Reverse Proxy<br/>Port 3001]

        subgraph "Actix-Web Server<br/>Port 4000"
            HANDLERS[HTTP/WebSocket Handlers]
            MIDDLEWARE[Middleware Layer<br/>Timeout, CORS, Error Recovery]
        end

        subgraph "Application Layer - CQRS"
            DIRECTIVES[Directives<br/>Write Operations]
            QUERIES[Queries<br/>Read Operations]
            EVENTS[Event Emitters]
        end

        subgraph "Domain Layer - Ports"
            PORT_SETTINGS[Settings Repository Port]
            PORT_KG[Knowledge Graph Repository Port]
            PORT_ONT[Ontology Repository Port]
            PORT_GPU[GPU Physics Port]
            PORT_SEM[Semantic Analyzer Port]
            PORT_INF[Inference Engine Port]
        end

        subgraph "Infrastructure - Adapters"
            ADAPT_SETTINGS[SQLite Settings Adapter]
            ADAPT_KG[SQLite KG Adapter]
            ADAPT_ONT[SQLite Ontology Adapter]
            ADAPT_GPU[Physics Orchestrator Adapter]
            ADAPT_SEM[Semantic Processor Adapter]
            ADAPT_WHELK[Whelk Inference Adapter]
        end

        subgraph "Actor System<br/>Legacy Integration"
            GRAPH_ACTOR[Graph Actor]
            ONTOLOGY_ACTOR[Ontology Actor]
            PHYSICS_ACTOR[Physics Orchestrator]
            GPU_ACTORS[GPU Manager + Compute Actors]
            WORKSPACE_ACTOR[Workspace Actor]
            CLIENT_COORD[Client Coordinator]
        end

        subgraph "Data Layer"
            DB_SETTINGS[(settings.db<br/>Config & Preferences)]
            DB_KG[(knowledge_graph.db<br/>Nodes & Edges)]
            DB_ONT[(ontology.db<br/>OWL Classes & Axioms)]
        end

        subgraph "External Services"
            GITHUB[GitHub API<br/>Ontology Source]
            NOSTR[Nostr Network<br/>Decentralized Messaging]
            RAGFLOW[RAGFlow Service<br/>AI Embeddings]
            MCP[MCP Orchestrator<br/>ws://agentic-workstation:3002]
        end

        subgraph "GPU Layer<br/>CUDA Kernels"
            CUDA_PHYSICS[39 Physics Kernels<br/>Force, Collision, Integration]
            CUDA_CLUSTER[Leiden Clustering<br/>Community Detection]
            CUDA_PATH[Shortest Path SSSP<br/>Multi-hop Reasoning]
            CUDA_CONSTRAINT[Ontology Constraints<br/>Validation Forces]
        end
    end

    WEB --> NGINX
    NGINX --> HANDLERS
    HANDLERS --> MIDDLEWARE
    MIDDLEWARE --> DIRECTIVES
    MIDDLEWARE --> QUERIES

    DIRECTIVES --> PORT_SETTINGS
    DIRECTIVES --> PORT_KG
    DIRECTIVES --> PORT_ONT
    QUERIES --> PORT_SETTINGS
    QUERIES --> PORT_KG
    QUERIES --> PORT_ONT

    PORT_SETTINGS --> ADAPT_SETTINGS
    PORT_KG --> ADAPT_KG
    PORT_ONT --> ADAPT_ONT
    PORT_GPU --> ADAPT_GPU
    PORT_SEM --> ADAPT_SEM
    PORT_INF --> ADAPT_WHELK

    ADAPT_SETTINGS --> DB_SETTINGS
    ADAPT_KG --> DB_KG
    ADAPT_ONT --> DB_ONT

    ADAPT_GPU --> PHYSICS_ACTOR
    ADAPT_SEM --> GPU_ACTORS
    ADAPT_WHELK --> ONTOLOGY_ACTOR

    PHYSICS_ACTOR --> CUDA_PHYSICS
    GPU_ACTORS --> CUDA_CLUSTER
    GPU_ACTORS --> CUDA_PATH
    ONTOLOGY_ACTOR --> CUDA_CONSTRAINT

    WORKSPACE_ACTOR --> GITHUB
    GRAPH_ACTOR --> NOSTR
    HANDLERS --> RAGFLOW
    HANDLERS --> MCP

    MOBILE -.Markdown Files.-> DB_KG

    style WEB fill:#e1f5ff
    style HANDLERS fill:#fff4e1
    style DIRECTIVES fill:#e8f5e9
    style QUERIES fill:#e8f5e9
    style DB_SETTINGS fill:#f3e5f5
    style DB_KG fill:#f3e5f5
    style DB_ONT fill:#f3e5f5
    style CUDA_PHYSICS fill:#ffebee
    style CUDA_CLUSTER fill:#ffebee
    style CUDA_PATH fill:#ffebee
    style CUDA_CONSTRAINT fill:#ffebee
```

---

## 2. Hexagonal Architecture - Ports and Adapters

```mermaid
graph LR
    subgraph "External World"
        HTTP[HTTP Clients]
        WS[WebSocket Clients]
        FILES[File System<br/>Markdown]
        GPU_HW[NVIDIA GPU<br/>Hardware]
    end

    subgraph "Adapters - Inbound"
        REST_ADAPTER[REST API Handler<br/>settings_handler.rs]
        WS_ADAPTER[WebSocket Handler<br/>socket_flow_handler.rs]
        FILE_ADAPTER[File Watcher<br/>workspace_handler.rs]
    end

    subgraph "Application Layer"
        direction TB
        SETTINGS_DIR[Settings Directives<br/>hexser::Directive]
        SETTINGS_QRY[Settings Queries<br/>hexser::Query]
        GRAPH_DIR[Graph Directives]
        GRAPH_QRY[Graph Queries]
        ONT_DIR[Ontology Directives]
        ONT_QRY[Ontology Queries]
    end

    subgraph "Domain - Ports"
        direction TB
        PORT_SETTINGS_W[SettingsRepository<br/>write methods]
        PORT_SETTINGS_R[SettingsRepository<br/>read methods]
        PORT_KG_W[KnowledgeGraphRepository<br/>write methods]
        PORT_KG_R[KnowledgeGraphRepository<br/>read methods]
        PORT_ONT_W[OntologyRepository<br/>write methods]
        PORT_ONT_R[OntologyRepository<br/>read methods]
        PORT_GPU[GpuPhysicsAdapter]
    end

    subgraph "Adapters - Outbound"
        direction TB
        SQLITE_SETTINGS[SqliteSettingsRepository<br/>src/adapters/]
        SQLITE_KG[SqliteKnowledgeGraphRepository]
        SQLITE_ONT[SqliteOntologyRepository]
        ACTOR_GPU[PhysicsOrchestratorAdapter<br/>Wraps Actor System]
    end

    subgraph "Infrastructure"
        DB_S[(settings.db)]
        DB_K[(knowledge_graph.db)]
        DB_O[(ontology.db)]
        ACTOR_SYS[Actix Actor System<br/>Legacy]
        CUDA_K[CUDA Kernels<br/>39 Production]
    end

    HTTP --> REST_ADAPTER
    WS --> WS_ADAPTER
    FILES --> FILE_ADAPTER

    REST_ADAPTER --> SETTINGS_DIR
    REST_ADAPTER --> SETTINGS_QRY
    WS_ADAPTER --> GRAPH_DIR
    WS_ADAPTER --> GRAPH_QRY
    FILE_ADAPTER --> ONT_DIR

    SETTINGS_DIR --> PORT_SETTINGS_W
    SETTINGS_QRY --> PORT_SETTINGS_R
    GRAPH_DIR --> PORT_KG_W
    GRAPH_QRY --> PORT_KG_R
    ONT_DIR --> PORT_ONT_W
    ONT_QRY --> PORT_ONT_R

    PORT_SETTINGS_W --> SQLITE_SETTINGS
    PORT_SETTINGS_R --> SQLITE_SETTINGS
    PORT_KG_W --> SQLITE_KG
    PORT_KG_R --> SQLITE_KG
    PORT_ONT_W --> SQLITE_ONT
    PORT_ONT_R --> SQLITE_ONT
    PORT_GPU --> ACTOR_GPU

    SQLITE_SETTINGS --> DB_S
    SQLITE_KG --> DB_K
    SQLITE_ONT --> DB_O
    ACTOR_GPU --> ACTOR_SYS
    ACTOR_SYS --> CUDA_K

    GPU_HW -.GPU Access.-> CUDA_K

    style REST_ADAPTER fill:#e1f5ff
    style WS_ADAPTER fill:#e1f5ff
    style SETTINGS_DIR fill:#e8f5e9
    style SETTINGS_QRY fill:#fff9c4
    style PORT_SETTINGS_W fill:#f3e5f5
    style PORT_SETTINGS_R fill:#f3e5f5
    style SQLITE_SETTINGS fill:#ffebee
    style DB_S fill:#fce4ec
```

---

## 3. Component Interaction - Data Flow

```mermaid
sequenceDiagram
    participant Client as React Client<br/>(Browser)
    participant Handler as Actix Handler<br/>settings_handler.rs
    participant Directive as Settings Directive<br/>CQRS Write
    participant Port as SettingsRepository<br/>Port Interface
    participant Adapter as SQLite Adapter<br/>Implementation
    participant DB as settings.db<br/>Database
    participant WS as WebSocket<br/>Broadcast

    Note over Client,DB: Example: User updates physics settings

    Client->>Handler: POST /api/settings/physics<br/>{profile: "default", settings: {...}}
    activate Handler

    Handler->>Directive: execute_directive(<br/>  UpdatePhysicsSettings<br/>)
    activate Directive

    Directive->>Port: update_physics_settings(<br/>  profile, settings<br/>)
    activate Port

    Port->>Adapter: update_physics_settings_impl(<br/>  profile, settings<br/>)
    activate Adapter

    Adapter->>DB: UPDATE physics_settings<br/>SET settings_json = ?<br/>WHERE profile_name = ?
    activate DB
    DB-->>Adapter: OK
    deactivate DB

    Adapter-->>Port: Result::Ok(())
    deactivate Adapter

    Port-->>Directive: Result::Ok(())
    deactivate Port

    Directive->>Directive: emit_event(<br/>  PhysicsSettingsUpdated<br/>)

    Directive-->>Handler: Result::Ok(response)
    deactivate Directive

    Handler->>WS: broadcast_to_clients(<br/>  PhysicsSettingsUpdated<br/>)
    activate WS
    WS-->>Client: WebSocket Message<br/>{type: "settings_updated"}
    deactivate WS

    Handler-->>Client: HTTP 200 OK<br/>{success: true}
    deactivate Handler

    Note over Client,DB: Server is authoritative - no client caching
```

---

## 4. Binary WebSocket Protocol Flow

```mermaid
sequenceDiagram
    participant Client as React Client
    participant WS as WebSocket Handler<br/>socket_flow_handler.rs
    participant PreRead as PreReadSocketSettings
    participant Graph as Graph Query/Directive
    participant GPU as GPU Compute Actor
    participant DB as knowledge_graph.db

    Note over Client,DB: Binary Protocol V2 (36 bytes per update)

    Client->>WS: WebSocket Connect<br/>ws://localhost:4000/ws/flow
    activate WS

    WS->>PreRead: Read settings.db<br/>get_user_preferences()
    activate PreRead
    PreRead->>DB: SELECT * FROM settings<br/>WHERE key LIKE 'user.%'
    DB-->>PreRead: Settings JSON
    PreRead-->>WS: UserSettings struct
    deactivate PreRead

    WS-->>Client: Initial State Broadcast<br/>(Binary Protocol V2)

    loop Every 16ms (60 FPS)
        GPU->>GPU: Compute Physics<br/>39 CUDA Kernels
        GPU->>Graph: query_node_positions()
        activate Graph
        Graph->>DB: SELECT id, position_x, position_y, position_z<br/>FROM nodes WHERE updated > ?
        DB-->>Graph: Updated nodes
        Graph-->>GPU: NodePositions
        deactivate Graph

        GPU->>WS: Binary Update Packet<br/>[header(4) + node_id(4) + xyz(12) + ...]
        WS->>Client: Binary WebSocket Frame<br/>~10ms latency
    end

    Client->>WS: User Interaction<br/>(Node drag)
    WS->>Graph: execute_directive(<br/>  UpdateNodePosition<br/>)
    activate Graph
    Graph->>DB: UPDATE nodes<br/>SET position_x=?, position_y=?, position_z=?<br/>WHERE id=?
    DB-->>Graph: OK
    Graph-->>WS: Success
    deactivate Graph

    WS-->>Client: Broadcast to all clients<br/>Binary update

    deactivate WS

    Note over Client,DB: 36-byte updates vs 200-byte JSON<br/>80% bandwidth reduction
```

---

## 5. Three-Database Architecture

```mermaid
graph TB
    subgraph "Application Services"
        SETTINGS_SVC[Settings Service<br/>User Preferences]
        GRAPH_SVC[Graph Service<br/>Knowledge Nodes]
        ONTOLOGY_SVC[Ontology Service<br/>Semantic Validation]
        PHYSICS_SVC[Physics Service<br/>GPU Simulation]
    end

    subgraph "Repository Layer"
        REPO_SETTINGS[SqliteSettingsRepository]
        REPO_KG[SqliteKnowledgeGraphRepository]
        REPO_ONT[SqliteOntologyRepository]
    end

    subgraph "Database Files<br/>/data/"
        DB1[(settings.db<br/>━━━━━━━━━<br/>Tables:<br/>• settings<br/>• physics_settings<br/>• user_preferences<br/>• feature_flags<br/>• namespace_mappings)]

        DB2[(knowledge_graph.db<br/>━━━━━━━━━<br/>Tables:<br/>• nodes<br/>• edges<br/>• file_metadata<br/>• node_positions<br/>• clustering_results<br/>• communities)]

        DB3[(ontology.db<br/>━━━━━━━━━<br/>Tables:<br/>• owl_classes<br/>• owl_properties<br/>• owl_axioms<br/>• inference_results<br/>• validation_reports<br/>• constraint_violations)]
    end

    subgraph "External Sources"
        LOGSEQ[Logseq Markdown<br/>Local Files]
        GITHUB_MD[GitHub Markdown<br/>Ontology Definitions]
        USER_INPUT[User Configuration<br/>UI Settings]
    end

    SETTINGS_SVC --> REPO_SETTINGS
    GRAPH_SVC --> REPO_KG
    ONTOLOGY_SVC --> REPO_ONT
    PHYSICS_SVC --> REPO_KG

    REPO_SETTINGS --> DB1
    REPO_KG --> DB2
    REPO_ONT --> DB3

    USER_INPUT -.Writes.-> DB1
    LOGSEQ -.Ingestion Pipeline.-> DB2
    GITHUB_MD -.Sync Service.-> DB3

    style DB1 fill:#e8f5e9
    style DB2 fill:#fff9c4
    style DB3 fill:#f3e5f5
    style LOGSEQ fill:#e1f5ff
    style GITHUB_MD fill:#e1f5ff
    style USER_INPUT fill:#e1f5ff
```

---

## 6. Actor System Integration (Legacy)

```mermaid
graph TB
    subgraph "Hexagonal Application"
        APP[Application Layer<br/>CQRS Directives/Queries]
        ADAPT[Adapter Layer<br/>Actor Wrappers]
    end

    subgraph "Actix Actor System"
        SUPERVISOR[Graph Service Supervisor<br/>Actor Lifecycle Management]

        GRAPH[Graph Actor<br/>Node/Edge Operations]
        ONTOLOGY[Ontology Actor<br/>OWL Validation]
        PHYSICS[Physics Orchestrator<br/>Simulation Loop]
        WORKSPACE[Workspace Actor<br/>File Watching]
        CLIENT_COORD[Client Coordinator<br/>WebSocket Management]

        subgraph "GPU Actors"
            GPU_MGR[GPU Manager Actor]
            FORCE_COMPUTE[Force Compute Actor]
            CLUSTERING[Leiden Clustering Actor]
            CONSTRAINT[Constraint Actor]
            ANOMALY[Anomaly Detection Actor]
        end
    end

    subgraph "CUDA Layer"
        STREAM1[CUDA Stream 1<br/>Physics]
        STREAM2[CUDA Stream 2<br/>Clustering]
        STREAM3[CUDA Stream 3<br/>Pathfinding]

        KERNELS[39 Production Kernels<br/>Force, Collision, Integration]
    end

    APP --> ADAPT
    ADAPT -.Wraps Actor Calls.-> SUPERVISOR

    SUPERVISOR --> GRAPH
    SUPERVISOR --> ONTOLOGY
    SUPERVISOR --> PHYSICS
    SUPERVISOR --> WORKSPACE
    SUPERVISOR --> CLIENT_COORD

    PHYSICS --> GPU_MGR
    GPU_MGR --> FORCE_COMPUTE
    GPU_MGR --> CLUSTERING
    GPU_MGR --> CONSTRAINT
    GPU_MGR --> ANOMALY

    FORCE_COMPUTE --> STREAM1
    CLUSTERING --> STREAM2
    CONSTRAINT --> STREAM3

    STREAM1 --> KERNELS
    STREAM2 --> KERNELS
    STREAM3 --> KERNELS

    style APP fill:#e8f5e9
    style ADAPT fill:#fff9c4
    style SUPERVISOR fill:#e1f5ff
    style GPU_MGR fill:#ffebee
    style KERNELS fill:#fce4ec
```

---

## 7. Deployment Architecture

```mermaid
graph TB
    subgraph "External Network"
        INTERNET[Internet]
        CLOUDFLARE[Cloudflare Tunnel<br/>cloudflared container]
    end

    subgraph "Docker Host"
        subgraph "docker_ragflow Network<br/>Bridge Network"

            subgraph "VisionFlow Container<br/>visionflow_container"
                NGINX[Nginx<br/>Port 3001]

                subgraph "Supervisord<br/>Process Manager"
                    ACTIX[Actix-Web Server<br/>Port 4000]
                    VITE[Vite Dev Server<br/>Port 5173]
                    RUST_BUILD[Cargo Watch<br/>Hot Reload]
                end

                VOL_SRC["/app/src<br/>Source Code"]
                VOL_DATA["/app/data<br/>Databases"]
                VOL_CLIENT["/app/client<br/>React App"]
            end

            subgraph "External Services<br/>agentic-workstation container"
                MCP_ORCH[MCP Orchestrator<br/>Port 3002]
                MGMT_API[Management API<br/>Port 9090]
            end

            RAGFLOW_SVC[RAGFlow Container<br/>AI Embeddings]
        end

        subgraph "Host Volumes"
            HOST_SRC[./src]
            HOST_DATA[visionflow-data<br/>Named Volume]
            HOST_CLIENT[./client]
        end

        subgraph "GPU Access"
            NVIDIA_GPU[NVIDIA GPU<br/>/dev/nvidia0]
        end
    end

    INTERNET --> CLOUDFLARE
    CLOUDFLARE --> NGINX

    NGINX --> ACTIX
    NGINX --> VITE

    ACTIX --> MCP_ORCH
    ACTIX --> RAGFLOW_SVC
    ACTIX --> MGMT_API

    HOST_SRC -.Live Mount.-> VOL_SRC
    HOST_DATA --> VOL_DATA
    HOST_CLIENT -.Live Mount.-> VOL_CLIENT

    ACTIX -.CUDA API.-> NVIDIA_GPU

    style CLOUDFLARE fill:#fff9c4
    style NGINX fill:#e1f5ff
    style ACTIX fill:#e8f5e9
    style NVIDIA_GPU fill:#ffebee
```

---

## 8. API Endpoint Architecture

```mermaid
graph LR
    subgraph "HTTP REST API"
        API_SETTINGS[/api/settings/*<br/>Settings Management]
        API_GRAPH[/api/graph/*<br/>Graph Operations]
        API_ONTOLOGY[/api/ontology/*<br/>OWL Validation]
        API_WORKSPACE[/api/workspace/*<br/>File Management]
        API_HEALTH[/api/health<br/>Health Check]
    end

    subgraph "WebSocket API"
        WS_FLOW[/ws/flow<br/>Binary Protocol V2<br/>Real-time Graph Updates]
        WS_SPEECH[/ws/speech<br/>Voice Commands<br/>Spatial Audio]
        WS_MCP[/ws/mcp<br/>MCP Relay<br/>Agent Communication]
        WS_REALTIME[/ws/realtime<br/>Multi-user Sync<br/>Collaborative Editing]
    end

    subgraph "Static Routes"
        PAGES[/<br/>React SPA<br/>index.html]
        ASSETS[/assets/*<br/>Static Resources<br/>JS, CSS, WASM]
    end

    subgraph "Handler Modules<br/>src/handlers/"
        H_SETTINGS[settings_handler.rs<br/>1,100 lines]
        H_GRAPH[graph_state_handler.rs<br/>400 lines]
        H_ONTOLOGY[ontology_handler.rs<br/>600 lines]
        H_SOCKET[socket_flow_handler.rs<br/>1,800 lines]
        H_SPEECH[speech_socket_handler.rs<br/>700 lines]
    end

    API_SETTINGS --> H_SETTINGS
    API_GRAPH --> H_GRAPH
    API_ONTOLOGY --> H_ONTOLOGY
    WS_FLOW --> H_SOCKET
    WS_SPEECH --> H_SPEECH

    style API_SETTINGS fill:#e8f5e9
    style API_GRAPH fill:#e8f5e9
    style API_ONTOLOGY fill:#e8f5e9
    style WS_FLOW fill:#fff9c4
    style WS_SPEECH fill:#fff9c4
    style H_SETTINGS fill:#e1f5ff
    style H_SOCKET fill:#e1f5ff
```

---

## Architecture Principles

### 1. Database-First Design
- All state persists in three separate databases
- No in-memory caching at application layer
- Database is single source of truth

### 2. Server-Authoritative
- Client never caches state
- All updates flow through server
- Binary protocol for efficiency (36 bytes/update)

### 3. CQRS Pattern
- Directives for write operations (mutations)
- Queries for read operations (no side effects)
- Clear separation of concerns

### 4. Hexagonal Architecture
- Ports define domain interfaces
- Adapters implement infrastructure
- Business logic independent of frameworks

### 5. GPU-First Performance
- 39 production CUDA kernels
- 100x speedup over CPU
- 60 FPS at 100k+ nodes
- Sub-10ms WebSocket latency

---

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Frame Rate** | 60 FPS | At 100k+ nodes |
| **WebSocket Latency** | <10ms | Binary protocol V2 |
| **GPU Speedup** | 100x | vs CPU baseline |
| **Bandwidth Reduction** | 80% | Binary vs JSON |
| **Database Size** | ~300 MB | 100k nodes + edges |
| **Memory Usage** | 8-16 GB | Active development |
| **Concurrent Users** | 50+ | Real-time collaboration |

---

## Migration Status

**Current Phase:** ✅ Completed
**Architecture Version:** 3.1.0
**Last Verified:** 2025-10-25

All legacy file-based configuration has been migrated to database-backed storage. The hexagonal architecture is fully implemented with CQRS pattern and three-database separation.

---

**For detailed migration history, see:** [MIGRATION_PLAN.md](../MIGRATION_PLAN.md)
**For database schemas, see:** [DATABASE.md](../DATABASE.md)
**For API documentation, see:** [REST API](../reference/api/rest-api.md) and [WebSocket API](../reference/api/websocket-api.md)
