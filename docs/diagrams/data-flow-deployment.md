# VisionFlow Data Flow & Deployment Diagrams

**Version:** 1.0.0
**Last Updated:** 2025-10-27
**Focus:** Data pipelines, file processing, and deployment infrastructure

---

## 1. File Processing Pipeline

```mermaid
flowchart TB
    subgraph "External Sources"
        LOGSEQ_MD[Logseq Markdown Files<br/>./data/markdown/]
        GITHUB_OWL[GitHub Repository<br/>Ontology Definitions]
        USER_UPLOAD[User File Upload<br/>HTTP POST /api/workspace/upload]
    end

    subgraph "Ingestion Layer"
        FILE_WATCHER[File System Watcher<br/>notify crate<br/>Hot Reload]
        GITHUB_SYNC[GitHub Sync Service<br/>Direct DB Sync]
        UPLOAD_HANDLER[Upload Handler<br/>workspace_handler.rs]
    end

    subgraph "Processing Pipeline"
        MD_PARSER[Markdown Parser<br/>Extract Nodes & Edges]
        OWL_PARSER[OWL/RDF Parser<br/>horned-owl crate]
        METADATA_EXTRACT[Metadata Extractor<br/>File Hash, Topics, Links]
        VECTOR_EMBED[Vector Embeddings<br/>RAGFlow Service]
    end

    subgraph "Actor Processing"
        WORKSPACE_ACTOR[Workspace Actor<br/>File Management]
        GRAPH_ACTOR[Graph Actor<br/>Node Creation]
        ONTOLOGY_ACTOR[Ontology Actor<br/>OWL Validation]
    end

    subgraph "Database Storage"
        DB_KG[(knowledge_graph.db<br/>━━━━━━━━━<br/>• nodes<br/>• edges<br/>• file_metadata)]
        DB_ONT[(ontology.db<br/>━━━━━━━━━<br/>• owl_classes<br/>• owl_properties<br/>• owl_axioms)]
    end

    subgraph "Post-Processing"
        WHELK[Whelk Reasoner<br/>OWL Inference]
        CLUSTERING[Leiden Clustering<br/>GPU-Accelerated]
        PHYSICS_INIT[Initial Physics<br/>Node Positioning]
    end

    LOGSEQ_MD --> FILE_WATCHER
    GITHUB_OWL --> GITHUB_SYNC
    USER_UPLOAD --> UPLOAD_HANDLER

    FILE_WATCHER --> MD_PARSER
    GITHUB_SYNC --> OWL_PARSER
    UPLOAD_HANDLER --> MD_PARSER

    MD_PARSER --> METADATA_EXTRACT
    OWL_PARSER --> METADATA_EXTRACT

    METADATA_EXTRACT --> VECTOR_EMBED
    METADATA_EXTRACT --> WORKSPACE_ACTOR

    VECTOR_EMBED --> GRAPH_ACTOR
    WORKSPACE_ACTOR --> GRAPH_ACTOR
    WORKSPACE_ACTOR --> ONTOLOGY_ACTOR

    GRAPH_ACTOR --> DB_KG
    ONTOLOGY_ACTOR --> DB_ONT

    DB_KG --> CLUSTERING
    DB_ONT --> WHELK

    WHELK --> PHYSICS_INIT
    CLUSTERING --> PHYSICS_INIT

    style LOGSEQ_MD fill:#e1f5ff
    style GITHUB_OWL fill:#e1f5ff
    style DB_KG fill:#fff9c4
    style DB_ONT fill:#f3e5f5
    style WHELK fill:#ffebee
    style CLUSTERING fill:#ffebee
```

---

## 2. Real-Time Data Synchronization Flow

```mermaid
sequenceDiagram
    participant User1 as User 1 Browser
    participant User2 as User 2 Browser
    participant WS as WebSocket Handler
    participant Physics as Physics Orchestrator
    participant GPU as GPU Compute
    participant DB as knowledge_graph.db
    participant Broadcast as Client Coordinator

    Note over User1,Broadcast: Multi-User Real-Time Collaboration

    User1->>WS: Connect WebSocket<br/>/ws/flow
    User2->>WS: Connect WebSocket<br/>/ws/flow

    WS->>DB: SELECT * FROM nodes<br/>LIMIT 10000
    DB-->>WS: Initial graph state
    WS-->>User1: Binary Packet (Initial State)
    WS-->>User2: Binary Packet (Initial State)

    loop Physics Loop (60 FPS)
        Physics->>GPU: Compute Forces<br/>CUDA Kernels
        GPU->>GPU: Force Calculation<br/>Collision Detection<br/>Integration
        GPU-->>Physics: Updated Positions
        Physics->>DB: UPDATE nodes<br/>SET position_x=?, position_y=?, position_z=?
        Physics->>Broadcast: notify_clients(<br/>  NodePositionsUpdated<br/>)
        Broadcast->>User1: Binary Update (36 bytes/node)
        Broadcast->>User2: Binary Update (36 bytes/node)
    end

    User1->>WS: User Interaction<br/>{action: "drag_node", id: 42, x: 10, y: 20}
    WS->>Physics: apply_user_force(<br/>  node_id: 42,<br/>  target: (10, 20)<br/>)
    Physics->>GPU: Add External Force
    GPU->>DB: UPDATE nodes<br/>SET position_x=10, position_y=20<br/>WHERE id=42
    Physics->>Broadcast: notify_clients(<br/>  UserInteraction<br/>)
    Broadcast->>User2: Binary Update<br/>(User 1's drag)
    Broadcast-->>User1: ACK

    Note over User1,Broadcast: Server-Authoritative:<br/>User 1's action instantly visible to User 2
```

---

## 3. GPU Processing Pipeline

```mermaid
flowchart LR
    subgraph "CPU Layer"
        ACTIX[Actix-Web Server<br/>Main Thread]
        PHYSICS_ACTOR[Physics Orchestrator Actor<br/>Async Task]
        GPU_MGR[GPU Manager Actor<br/>Resource Allocation]
    end

    subgraph "CUDA Memory Management"
        HOST_MEM[Host Memory<br/>Pinned RAM<br/>8-16 GB]
        DEVICE_MEM[Device Memory<br/>GPU VRAM<br/>12-24 GB]
    end

    subgraph "GPU Compute Streams"
        STREAM1[Stream 1: Physics<br/>Force Compute<br/>Collision Detection<br/>Integration]

        STREAM2[Stream 2: Clustering<br/>Leiden Algorithm<br/>Community Detection<br/>Modularity]

        STREAM3[Stream 3: Pathfinding<br/>SSSP Shortest Path<br/>Multi-hop Reasoning<br/>Graph Traversal]

        STREAM4[Stream 4: Constraints<br/>Ontology Validation<br/>Physics Forces<br/>Consistency Checks]
    end

    subgraph "CUDA Kernels<br/>39 Production Kernels"
        K_FORCE[force_directed.cu<br/>Compute attractive/repulsive]
        K_COLLISION[collision_detection.cu<br/>Spatial hashing]
        K_INTEGRATE[velocity_integration.cu<br/>Position updates]
        K_LEIDEN[leiden_clustering.cu<br/>Community detection]
        K_SSSP[sssp_pathfinding.cu<br/>Shortest paths]
        K_CONSTRAINT[constraint_forces.cu<br/>OWL validation]
    end

    subgraph "Database Write-Back"
        DB_KG[(knowledge_graph.db)]
        DB_ONT[(ontology.db)]
    end

    ACTIX --> PHYSICS_ACTOR
    PHYSICS_ACTOR --> GPU_MGR

    GPU_MGR --> HOST_MEM
    HOST_MEM -.DMA Transfer.-> DEVICE_MEM

    DEVICE_MEM --> STREAM1
    DEVICE_MEM --> STREAM2
    DEVICE_MEM --> STREAM3
    DEVICE_MEM --> STREAM4

    STREAM1 --> K_FORCE
    STREAM1 --> K_COLLISION
    STREAM1 --> K_INTEGRATE

    STREAM2 --> K_LEIDEN

    STREAM3 --> K_SSSP

    STREAM4 --> K_CONSTRAINT

    K_FORCE -.Results.-> DEVICE_MEM
    K_COLLISION -.Results.-> DEVICE_MEM
    K_INTEGRATE -.Results.-> DEVICE_MEM
    K_LEIDEN -.Results.-> DEVICE_MEM
    K_SSSP -.Results.-> DEVICE_MEM
    K_CONSTRAINT -.Results.-> DEVICE_MEM

    DEVICE_MEM -.DMA Transfer.-> HOST_MEM
    HOST_MEM --> PHYSICS_ACTOR

    PHYSICS_ACTOR --> DB_KG
    PHYSICS_ACTOR --> DB_ONT

    style DEVICE_MEM fill:#ffebee
    style STREAM1 fill:#fff9c4
    style STREAM2 fill:#fff9c4
    style STREAM3 fill:#fff9c4
    style STREAM4 fill:#fff9c4
    style K_FORCE fill:#fce4ec
    style K_LEIDEN fill:#fce4ec
```

---

## 4. External Service Integration

```mermaid
graph TB
    subgraph "VisionFlow Core"
        ACTIX[Actix-Web Server<br/>Port 4000]
        HANDLERS[Handler Layer]
        SERVICES[Service Layer]
    end

    subgraph "External AI Services"
        RAGFLOW[RAGFlow Service<br/>Vector Embeddings<br/>External Container]
        MCP_ORCH[MCP Orchestrator<br/>Agent Coordination<br/>ws://agentic-workstation:3002]
        CLAUDE[Claude API<br/>Text Generation<br/>HTTPS]
    end

    subgraph "Data Sources"
        GITHUB[GitHub API<br/>REST + GraphQL<br/>Ontology Sync]
        NOSTR[Nostr Network<br/>Decentralized Relay<br/>Event Broadcasting]
    end

    subgraph "External Storage"
        QDRANT[Qdrant Vector DB<br/>Semantic Search<br/>External Container<br/>./qdrant_data volume]
    end

    subgraph "Management Services"
        MGMT_API[Management API<br/>Health Checks<br/>http://agentic-workstation:9090]
    end

    HANDLERS --> SERVICES

    SERVICES --> RAGFLOW
    SERVICES --> MCP_ORCH
    SERVICES --> CLAUDE
    SERVICES --> GITHUB
    SERVICES --> NOSTR
    SERVICES --> QDRANT

    ACTIX --> MGMT_API

    RAGFLOW -.Vector Embeddings.-> SERVICES
    MCP_ORCH -.Agent Events.-> HANDLERS
    GITHUB -.OWL Files.-> SERVICES

    style RAGFLOW fill:#e1f5ff
    style MCP_ORCH fill:#e1f5ff
    style GITHUB fill:#e1f5ff
    style QDRANT fill:#fff9c4
```

---

## 5. Docker Compose Service Topology

```mermaid
graph TB
    subgraph "docker_ragflow Network<br/>External Bridge Network"

        subgraph "VisionFlow Container<br/>visionflow_container"
            NGINX_DEV[Nginx<br/>Port 3001 → 3001<br/>Dev Mode]
            ACTIX_WEB[Actix-Web<br/>Port 4000 → 4000<br/>API Server]
            SUPERVISORD[Supervisord<br/>Process Manager]
            VITE_DEV[Vite Dev Server<br/>Port 5173<br/>Internal Only]
        end

        subgraph "VisionFlow Production<br/>visionflow_prod_container<br/>Profile: production"
            ACTIX_PROD[Actix-Web<br/>Port 4001 → 4000<br/>Production Build]
            SUPERVISORD_PROD[Supervisord<br/>Production Config]
        end

        subgraph "Cloudflare Tunnel<br/>cloudflared-tunnel"
            CLOUDFLARED[cloudflared<br/>TUNNEL_TOKEN from .env]
        end

        subgraph "External Services<br/>agentic-workstation container"
            MCP_SERVICE[MCP Orchestrator<br/>Port 3002<br/>BOTS_ORCHESTRATOR_URL]
            MGMT_SERVICE[Management API<br/>Port 9090<br/>Health Endpoints]
        end

        subgraph "RAGFlow Container"
            RAGFLOW_API[RAGFlow API<br/>AI Embeddings]
        end

        subgraph "Qdrant Container"
            QDRANT_API[Qdrant Vector DB<br/>Semantic Search]
        end
    end

    subgraph "Host Filesystem"
        VOL_SRC[./src<br/>Live Mount<br/>Hot Reload]
        VOL_CLIENT[./client<br/>React Source]
        VOL_DATA[visionflow-data<br/>Named Volume<br/>Databases]
        VOL_LOGS[./logs<br/>Bind Mount<br/>Nginx Logs]
        VOL_CARGO[cargo-target-cache<br/>Named Volume<br/>Build Cache]
    end

    subgraph "External Network"
        INTERNET[Internet]
    end

    INTERNET --> CLOUDFLARED
    CLOUDFLARED -.Tunnel.-> NGINX_DEV
    CLOUDFLARED -.Tunnel.-> ACTIX_PROD

    SUPERVISORD --> NGINX_DEV
    SUPERVISORD --> ACTIX_WEB
    SUPERVISORD --> VITE_DEV

    ACTIX_WEB --> MCP_SERVICE
    ACTIX_WEB --> MGMT_SERVICE
    ACTIX_WEB --> RAGFLOW_API
    ACTIX_WEB --> QDRANT_API

    VOL_SRC -.ro Mount.-> ACTIX_WEB
    VOL_CLIENT -.rw Mount.-> VITE_DEV
    VOL_DATA --> ACTIX_WEB
    VOL_LOGS --> NGINX_DEV
    VOL_CARGO --> ACTIX_WEB

    style ACTIX_WEB fill:#e8f5e9
    style CLOUDFLARED fill:#fff9c4
    style VOL_DATA fill:#f3e5f5
    style MCP_SERVICE fill:#e1f5ff
```

---

## 6. Network Port Mapping

```mermaid
graph LR
    subgraph "Host Ports<br/>External Access"
        P3001[Host:3001<br/>Development Entry]
        P4000[Host:4000<br/>API Direct Access]
    end

    subgraph "VisionFlow Container<br/>Internal Ports"
        C3001[Container:3001<br/>Nginx Proxy]
        C4000[Container:4000<br/>Actix-Web API]
        C5173[Container:5173<br/>Vite Dev Server<br/>Internal Only]
        C24678[Container:24678<br/>Vite HMR<br/>Hot Module Reload]
    end

    subgraph "External Service Ports<br/>docker_ragflow Network"
        MCP3002[agentic-workstation:3002<br/>MCP Orchestrator WebSocket]
        MGMT9090[agentic-workstation:9090<br/>Management API HTTP]
        MCP9500[agentic-workstation:9500<br/>MCP TCP Transport]
    end

    P3001 -.Port Mapping.-> C3001
    P4000 -.Port Mapping.-> C4000

    C3001 --> C4000
    C3001 --> C5173

    C4000 -.WebSocket.-> MCP3002
    C4000 -.HTTP.-> MGMT9090
    C4000 -.TCP.-> MCP9500

    style P3001 fill:#e1f5ff
    style P4000 fill:#e1f5ff
    style C4000 fill:#e8f5e9
    style MCP3002 fill:#fff9c4
```

---

## 7. Volume Mount Strategy

```mermaid
graph TB
    subgraph "Host Filesystem"
        H_SRC[./src<br/>Source Code<br/>Read-Only]
        H_CLIENT[./client<br/>React App<br/>Read-Write]
        H_CARGO_TOML[./Cargo.toml<br/>Read-Only]
        H_SCHEMA[./schema<br/>SQL Schemas<br/>Read-Only]
        H_LOGS[./logs<br/>Application Logs<br/>Read-Write]
    end

    subgraph "Docker Named Volumes"
        V_DATA[visionflow-data<br/>Databases<br/>Persistent]
        V_NPM[npm-cache<br/>NPM Packages<br/>Cache]
        V_CARGO_REG[cargo-cache<br/>Cargo Registry<br/>Cache]
        V_CARGO_GIT[cargo-git-cache<br/>Git Dependencies<br/>Cache]
        V_CARGO_TARGET[cargo-target-cache<br/>Build Artifacts<br/>Cache]
    end

    subgraph "Container Paths"
        C_SRC[/app/src<br/>Live Code<br/>Hot Reload]
        C_CLIENT[/app/client<br/>React Source<br/>Vite Build]
        C_DATA[/app/data<br/>3 Databases<br/>WAL Mode]
        C_TARGET[/app/target<br/>Rust Build<br/>Incremental]
        C_NPM[/root/.npm<br/>Package Cache]
        C_LOGS[/app/logs<br/>Application Logs]
    end

    H_SRC -.Bind Mount (ro).-> C_SRC
    H_CLIENT -.Bind Mount (rw).-> C_CLIENT
    H_LOGS -.Bind Mount (rw).-> C_LOGS

    V_DATA --> C_DATA
    V_NPM --> C_NPM
    V_CARGO_TARGET --> C_TARGET

    C_SRC -.Watched by.-> CARGO_WATCH[Cargo Watch<br/>Auto Rebuild]
    C_CLIENT -.Watched by.-> VITE[Vite Dev Server<br/>HMR]

    C_DATA -.Contains.-> DB_FILES[settings.db<br/>knowledge_graph.db<br/>ontology.db]

    style V_DATA fill:#f3e5f5
    style V_CARGO_TARGET fill:#fff9c4
    style C_DATA fill:#fce4ec
    style CARGO_WATCH fill:#e1f5ff
    style VITE fill:#e1f5ff
```

---

## 8. Development vs Production Deployment

```mermaid
flowchart TB
    subgraph "Development Profile<br/>docker-compose --profile dev up"
        DEV_BUILD[Build: Dockerfile.dev<br/>Multi-stage build]

        DEV_SVC[VisionFlow Dev Container<br/>visionflow_container]

        DEV_SUPERVISORD[Supervisord Services:<br/>• Nginx (proxy)<br/>• Actix-Web (API)<br/>• Vite Dev Server (HMR)<br/>• Cargo Watch (auto-rebuild)]

        DEV_VOLUMES[Volumes:<br/>• ./src → /app/src (ro)<br/>• ./client → /app/client (rw)<br/>• visionflow-data (databases)]

        DEV_ENV[Environment:<br/>NODE_ENV=development<br/>RUST_LOG=debug<br/>VITE_DEBUG=true<br/>SYSTEM_NETWORK_PORT=4000]

        DEV_PORTS[Ports:<br/>3001:3001 (Nginx)<br/>4000:4000 (API)]
    end

    subgraph "Production Profile<br/>docker-compose --profile production up"
        PROD_BUILD[Build: Dockerfile.dev<br/>Production Entrypoint]

        PROD_SVC[VisionFlow Prod Container<br/>visionflow_prod_container]

        PROD_SUPERVISORD[Supervisord Services:<br/>• Nginx (prod config)<br/>• Actix-Web (optimized)<br/>• NO Vite (pre-built)<br/>• NO Cargo Watch]

        PROD_VOLUMES[Volumes:<br/>• visionflow-data (databases)<br/>• cargo-target-cache (build)]

        PROD_ENV[Environment:<br/>NODE_ENV=production<br/>RUST_LOG=warn<br/>SYSTEM_NETWORK_PORT=4001]

        PROD_PORTS[Ports:<br/>4000:4000 (API only)]
    end

    DEV_BUILD --> DEV_SVC
    DEV_SVC --> DEV_SUPERVISORD
    DEV_SUPERVISORD --> DEV_VOLUMES
    DEV_VOLUMES --> DEV_ENV
    DEV_ENV --> DEV_PORTS

    PROD_BUILD --> PROD_SVC
    PROD_SVC --> PROD_SUPERVISORD
    PROD_SUPERVISORD --> PROD_VOLUMES
    PROD_VOLUMES --> PROD_ENV
    PROD_ENV --> PROD_PORTS

    style DEV_SVC fill:#e8f5e9
    style PROD_SVC fill:#ffebee
```

---

## 9. CI/CD & Build Pipeline

```mermaid
flowchart LR
    subgraph "Source Control"
        GIT[Git Repository<br/>Local/Remote]
        BRANCH[Feature Branch]
    end

    subgraph "Development Build"
        CARGO[Cargo Build<br/>Incremental Compilation]
        VITE_BUILD[Vite Build<br/>TypeScript + React]
        CUDA_COMPILE[CUDA Compilation<br/>nvcc + PTX]
    end

    subgraph "Docker Build"
        DOCKERFILE[Dockerfile.dev<br/>Multi-stage Build]
        BASE[Stage 1: Base System<br/>CachyOS + CUDA]
        RUST_STAGE[Stage 2: Rust Build<br/>cargo build --release]
        NODE_STAGE[Stage 3: Node Build<br/>npm install + vite build]
        FINAL[Stage 4: Final Image<br/>Copy artifacts]
    end

    subgraph "Deployment"
        COMPOSE[Docker Compose<br/>Up -d]
        HEALTH[Health Check<br/>curl /api/health]
        READY[Service Ready<br/>Port 4000]
    end

    GIT --> BRANCH
    BRANCH --> CARGO
    BRANCH --> VITE_BUILD
    BRANCH --> CUDA_COMPILE

    CARGO --> DOCKERFILE
    VITE_BUILD --> DOCKERFILE
    CUDA_COMPILE --> DOCKERFILE

    DOCKERFILE --> BASE
    BASE --> RUST_STAGE
    RUST_STAGE --> NODE_STAGE
    NODE_STAGE --> FINAL

    FINAL --> COMPOSE
    COMPOSE --> HEALTH
    HEALTH --> READY

    style DOCKERFILE fill:#e1f5ff
    style FINAL fill:#e8f5e9
    style READY fill:#c8e6c9
```

---

## Key Data Flow Principles

### 1. Server-Authoritative Model
- **Single Source of Truth**: All state in databases
- **No Client Caching**: Clients always fetch from server
- **Real-time Sync**: WebSocket broadcasts to all connected clients

### 2. Binary Protocol Optimization
- **Protocol V2**: 36 bytes per node update
- **80% Bandwidth Reduction**: vs 200-byte JSON updates
- **Sub-10ms Latency**: GPU → Database → WebSocket → Client

### 3. Hot Reload Architecture
- **Source Code**: Live mounted, watched by cargo
- **React App**: Vite HMR on file change
- **Databases**: Persistent volumes, WAL mode for concurrency

### 4. GPU Memory Management
- **Pinned Host Memory**: 8-16 GB for DMA transfers
- **Device Memory**: 12-24 GB VRAM for compute
- **4 Concurrent Streams**: Physics, Clustering, Pathfinding, Constraints

### 5. Multi-Container Coordination
- **docker_ragflow Network**: All services on same bridge
- **Service Discovery**: DNS-based (agentic-workstation:PORT)
- **Health Checks**: HTTP endpoints for orchestration

---

## Performance Characteristics

| Component | Throughput | Latency | Bottleneck |
|-----------|------------|---------|------------|
| **File Ingestion** | 1000 files/sec | 1ms/file | Disk I/O |
| **WebSocket** | 10k updates/sec | <10ms | Network |
| **GPU Physics** | 100k nodes @ 60 FPS | 16ms/frame | GPU Compute |
| **Database Write** | 5k writes/sec | 0.2ms | SQLite WAL |
| **Clustering** | 100k nodes | 50ms | GPU Memory |

---

**For implementation details, see:**
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Hexagonal architecture
- [DATABASE.md](../DATABASE.md) - Database schemas
- [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md) - Development workflows
