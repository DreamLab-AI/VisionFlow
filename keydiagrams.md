# VisionFlow Key Architecture Diagrams
- **Audience**: Technical ML engineers and researchers new to the project
- **Purpose**: Understand how VisionFlow transforms OWL ontologies into GPU-accelerated 3D knowledge graph visualizations
- **Core Innovation**: Ontological relationships (SubClassOf, DisjointWith, EquivalentClasses) are translated into physical forces that drive self-organizing graph layouts at 60 FPS on 100K+ nodes

---

## 1. System Context (C4 Level 1)
- Shows VisionFlow's external boundaries: who uses it, what it connects to
- Three user personas: Developer (web), Data Scientist (analytics), XR User (immersive VR)
- External dependencies: GitHub for data ingestion, AI services (Claude/Perplexity) for semantic analysis, Nostr for decentralized identity
- Infrastructure: Neo4j graph database + CUDA 12.4 GPU compute

```mermaid
graph TB
    subgraph "External Users"
        Developer[Developer<br/>Knowledge exploration<br/>via web interface]
        DataScientist[Data Scientist<br/>Graph pattern analysis<br/>and clustering]
        XRUser[XR/VR User<br/>Immersive 3D<br/>visualization]
    end

    subgraph "External Systems"
        GitHubAPI[GitHub API<br/>Source of markdown<br/>documentation]
        AIServices[AI Services<br/>Claude + Perplexity<br/>Semantic analysis]
        NostrNetwork[Nostr Network<br/>Decentralized identity<br/>and authentication]
    end

    subgraph "VisionFlow Platform"
        VF[VisionFlow<br/>Knowledge Graph Visualization<br/>with GPU-Accelerated Physics<br/>and AI-Powered Analysis]
    end

    subgraph "Infrastructure"
        Neo4j[(Neo4j 5.13<br/>Graph Database)]
        GPU[GPU Compute<br/>CUDA 12.4<br/>39 physics kernels]
    end

    Developer -->|HTTPS/WSS| VF
    DataScientist -->|HTTPS/WSS| VF
    XRUser -->|WebXR| VF

    GitHubAPI -->|REST API| VF
    AIServices -->|API calls| VF
    NostrNetwork -->|NIP-07| VF

    VF -->|Bolt protocol| Neo4j
    VF -->|CUDA FFI| GPU

    style VF fill:#4A90D9,color:#fff,stroke:#333,stroke-width:3px
    style Neo4j fill:#f0e1ff,stroke:#333
    style GPU fill:#e1ffe1,stroke:#333
```

---

## 2. Container Architecture (C4 Level 2)
- Deployable units and their interactions
- **Client Layer**: React 18 + Three.js + WebXR for 3D rendering
- **API Layer**: REST (CQRS handlers) + Binary WebSocket (28 bytes/node) + Voice (Whisper STT / Kokoro TTS)
- **Application Layer**: 24 Actix actors with supervisor hierarchy, event sourcing, CQRS command/query split
- **GPU Layer**: 4 supervisors coordinating 39+ CUDA kernels for physics, clustering (Leiden), graph analytics (SSSP, PageRank)
- **Infrastructure**: Neo4j as source of truth, Whelk OWL 2 EL reasoner for semantic inference

```mermaid
graph TB
    subgraph "Client Layer"
        WebClient[Web Client<br/>React 18 + TypeScript<br/>Three.js + WebXR]
    end

    subgraph "API Layer"
        REST[REST API<br/>Actix Web 4.11<br/>114 CQRS Handlers]
        WS[WebSocket Server<br/>Binary Protocol V2<br/>36 bytes/node]
        Voice[Voice WebSocket<br/>Whisper STT<br/>Kokoro TTS]
    end

    subgraph "Application Layer"
        Actors[Actor System<br/>24 Actix Actors<br/>Supervisor hierarchy]
        CQRS[CQRS Handlers<br/>Command/Query split<br/>Event sourcing ready]
        Events[Event Bus<br/>Async dispatch<br/>Domain events]
    end

    subgraph "GPU Layer"
        GPUManager[GPU Manager Actor<br/>4 supervisors]
        Physics[Physics Supervisor<br/>Force-directed layout<br/>Constraint solving]
        Analytics[Analytics Supervisor<br/>Leiden clustering<br/>Anomaly detection]
        GraphAlgo[Graph Analytics<br/>SSSP + APSP<br/>Community detection]
    end

    subgraph "Infrastructure Layer"
        Neo4j[(Neo4j 5.13<br/>Graph Database<br/>Source of truth)]
        CUDA[CUDA Runtime<br/>39 kernels<br/>100K nodes at 60fps]
        OWL[Whelk Reasoner<br/>OWL 2 EL<br/>Inference engine]
    end

    WebClient -->|HTTPS| REST
    WebClient -->|WSS Binary| WS
    WebClient -->|WSS Audio| Voice

    REST --> CQRS
    WS --> Actors
    Voice --> Actors

    CQRS --> Actors
    Actors --> Events
    Events --> WS

    Actors --> GPUManager
    GPUManager --> Physics
    GPUManager --> Analytics
    GPUManager --> GraphAlgo

    Actors --> Neo4j
    Actors --> OWL
    Physics --> CUDA
    Analytics --> CUDA
    GraphAlgo --> CUDA

    style WebClient fill:#e3f2fd,stroke:#333
    style Actors fill:#ffe66d,stroke:#333
    style CQRS fill:#ffe66d,stroke:#333
    style GPUManager fill:#ffccbc,stroke:#333
    style Neo4j fill:#f0e1ff,stroke:#333
    style CUDA fill:#e1ffe1,stroke:#333
    style OWL fill:#fff9c4,stroke:#333
```

---

## 3. End-to-End Data Pipeline
- The complete flow from GitHub markdown files to 3D rendered graph
- **Data ingestion**: Differential sync (SHA1 comparison) fetches only changed files from GitHub
- **Dual parsing**: Knowledge graph nodes from `public:: true` pages, OWL classes from `### OntologyBlock` sections
- **Reasoning**: Whelk-rs (Rust OWL 2 EL reasoner, 10-100x faster than Java alternatives) computes inferred axioms
- **Constraint generation**: 8 semantic constraint types translate ontological relationships into physics forces
- **GPU simulation**: 39 CUDA kernels compute forces, integrate velocities, update positions at 60 Hz
- **Binary streaming**: 36-byte binary WebSocket protocol (80% bandwidth reduction vs JSON) delivers positions to clients

```mermaid
graph TB
    subgraph GitHub["GitHub Repository"]
        MD1["Knowledge Graph<br/>.md files with public:: true"]
        MD2["Ontology<br/>.md files with OntologyBlock"]
    end

    subgraph Sync["GitHub Sync Service"]
        DIFF["Differential Sync<br/>SHA1 comparison"]
        KGP["KnowledgeGraphParser"]
        ONTOP["OntologyParser"]
    end

    subgraph Database["Data Layer"]
        GRAPH_TABLES["Neo4j Nodes<br/>Neo4j Relationships"]
        OWL_TABLES["OntologyRepository<br/>In-Memory Store"]
    end

    subgraph Reasoning["Ontology Reasoning"]
        WHELK["Whelk-rs Reasoner<br/>OWL 2 EL"]
        INFER["Inferred Axioms<br/>is-inferred=true"]
        CACHE["LRU Cache<br/>90x speedup"]
    end

    subgraph PhysicsLayer["GPU Semantic Physics"]
        CONSTRAINTS["Semantic Constraints<br/>8 types"]
        CUDAEngine["CUDA Physics Engine<br/>39 kernels"]
        FORCES["Force Calculations<br/>Ontology-driven"]
    end

    subgraph Client["Client Visualization"]
        WSProto["Binary WebSocket<br/>36 bytes/node"]
        RENDER["3D Rendering<br/>Three.js"]
        GRAPH["Self-Organizing Graph"]
    end

    MD1 --> DIFF
    MD2 --> DIFF
    DIFF --> KGP
    DIFF --> ONTOP
    KGP --> GRAPH_TABLES
    ONTOP --> OWL_TABLES

    OWL_TABLES --> WHELK
    WHELK --> INFER
    INFER --> OWL_TABLES
    WHELK --> CACHE

    OWL_TABLES --> CONSTRAINTS
    CONSTRAINTS --> CUDAEngine
    GRAPH_TABLES --> CUDAEngine
    CUDAEngine --> FORCES

    FORCES --> WSProto
    WSProto --> RENDER
    RENDER --> GRAPH

    style GitHub fill:#e1f5ff
    style Sync fill:#fff3e0
    style Database fill:#f0e1ff
    style Reasoning fill:#e8f5e9
    style PhysicsLayer fill:#ffe1e1
    style Client fill:#fff9c4
```

---

## 4. Ontology Reasoning Pipeline
- The core ML-relevant pipeline: how OWL axioms become physics constraints
- **Step 1**: Load classes, asserted axioms, and properties from OntologyRepository
- **Step 2**: Whelk-rs computes inferences (e.g., `Cat SubClassOf Animal` + `Animal SubClassOf LivingThing` => `Cat SubClassOf LivingThing`)
- **Step 3**: Store inferred axioms back with `is_inferred=true` flag
- **Step 4**: Generate physics constraints per axiom type:
	- `SubClassOf` => Spring attraction (k=0.5) — child classes cluster near parents
	- `DisjointWith` => Coulomb repulsion (k=-0.8) — disjoint classes pushed apart
	- `EquivalentClasses` => Strong spring (k=1.0) — synonyms rendered together
	- `ObjectProperty` => Directional alignment — domains/ranges aligned
- **Step 5**: Inferred axioms get 0.3x force multiplier (subtle influence vs asserted)
- **Step 6**: Upload constraint buffers to GPU for real-time simulation

```mermaid
graph TB
    START["Sync Complete"]

    subgraph Load["1. Load Ontology"]
        LOAD_CLASSES["Load classes from<br/>OntologyRepository"]
        LOAD_AXIOMS["Load asserted axioms<br/>is_inferred=false"]
        LOAD_PROPS["Load properties from<br/>OntologyRepository"]
    end

    subgraph Reason["2. Whelk-rs Reasoning"]
        BUILD["Build OWL graph"]
        COMPUTE["Compute inferences<br/>10-100x faster than Java"]
        CHECK["Consistency check"]
    end

    subgraph Store["3. Store Results"]
        INFER_AX["Store inferred axioms<br/>is_inferred=true"]
        UPDATE_META["Update reasoning metadata"]
        CACHE_WARM["Warm LRU cache"]
    end

    subgraph Generate["4. Generate Constraints"]
        SUBCLASS["SubClassOf => Attraction"]
        DISJOINT["DisjointWith => Repulsion"]
        EQUIV["EquivalentClasses => Strong Attraction"]
        PROP["ObjectProperty => Alignment"]
        WEAKEN["Inferred axioms => 0.3x force"]
    end

    START --> LOAD_CLASSES
    LOAD_CLASSES --> LOAD_AXIOMS
    LOAD_AXIOMS --> LOAD_PROPS

    LOAD_PROPS --> BUILD
    BUILD --> COMPUTE
    COMPUTE --> CHECK

    CHECK --> INFER_AX
    INFER_AX --> UPDATE_META
    UPDATE_META --> CACHE_WARM

    CACHE_WARM --> SUBCLASS
    SUBCLASS --> DISJOINT
    DISJOINT --> EQUIV
    EQUIV --> PROP
    PROP --> WEAKEN

    WEAKEN --> GPU_UPLOAD["Upload to GPU"]

    style START fill:#c8e6c9
    style GPU_UPLOAD fill:#ffe1e1
```

---

## 5. CUDA Physics Kernel Pipeline
- Shows the CPU-to-GPU data flow for each physics simulation frame
- **CPU side**: Generates semantic constraints from ontology, uploads to GPU memory
- **GPU kernels execute sequentially**: Spring forces (attraction), repulsion forces (separation), alignment forces (directional), inferred axiom weighting (0.3x reduction), velocity integration, position update
- **Output**: New node positions downloaded back to CPU for WebSocket broadcast
- **Performance**: 16ms per frame on RTX 3080 with 10K+ nodes and 50K+ constraints

```mermaid
graph LR
    subgraph CPU["CPU - Rust"]
        CONS["Generate<br/>Constraints"]
        UPLOAD["Upload to GPU"]
    end

    subgraph GPUKernels["GPU - CUDA"]
        K1["Kernel 1:<br/>Spring Forces"]
        K2["Kernel 2:<br/>Repulsion Forces"]
        K3["Kernel 3:<br/>Alignment Forces"]
        K_INFER["Apply 0.3x<br/>to inferred"]
        INTEGRATE["Integrate<br/>Velocities"]
        UPDATE["Update<br/>Positions"]
    end

    subgraph Output["Output"]
        POSITIONS["New Node<br/>Positions"]
        DOWNLOAD["Download to CPU"]
    end

    CONS --> UPLOAD
    UPLOAD --> K1
    K1 --> K2
    K2 --> K3
    K3 --> K_INFER
    K_INFER --> INTEGRATE
    INTEGRATE --> UPDATE
    UPDATE --> POSITIONS
    POSITIONS --> DOWNLOAD

    style CPU fill:#fff3e0
    style GPUKernels fill:#ffe1e1
    style Output fill:#e8f5e9
```

---

## 6. GitHub Sync Sequence
- Detailed sequence showing how markdown files become graph nodes and ontology classes
- **Differential sync**: Only processes files whose SHA1 hash has changed (90%+ skip rate on subsequent syncs)
- **File routing**: `public:: true` header => knowledge graph node, `### OntologyBlock` => OWL class/property
- **Post-sync**: Reloads graph into memory, initializes GPU physics, starts 60 Hz simulation loop
- **Binary streaming**: Each frame broadcasts 36 bytes per node (id, xyz position, xyz velocity, group, flags)

```mermaid
sequenceDiagram
    participant User
    participant REST as REST API
    participant Sync as GitHubSyncService
    participant GH as GitHub
    participant Neo4j as Neo4j
    participant GSA as GraphStateActor
    participant POA as PhysicsOrchestrator
    participant GPU as GPU
    participant CCA as ClientCoordinator
    participant WS as WebSocket
    participant Browser

    User->>REST: POST /api/admin/sync/streaming
    REST->>Sync: trigger_sync()

    Sync->>GH: GET /repos/:owner/:repo/git/trees/:sha
    GH-->>Sync: File tree

    loop For each markdown file
        Sync->>GH: GET /repos/:owner/:repo/contents/:path
        GH-->>Sync: File content (base64)

        alt Contains OntologyBlock
            Sync->>Neo4j: save_ontology_class()
        end

        alt Contains public:: true
            Sync->>Neo4j: add_node(), add_edge()
        end
    end

    Sync->>GSA: ReloadGraphFromDatabase
    GSA->>Neo4j: MATCH (n:Node)-[e:EDGE]-(m:Node)
    Neo4j-->>GSA: Graph data

    GSA->>POA: InitializePhysics
    POA->>GPU: Transfer node data to GPU memory

    loop Simulation Loop 60 Hz
        POA->>GPU: Execute force calculation kernels
        GPU-->>POA: Updated positions
        POA->>CCA: BroadcastPositions
        CCA->>WS: Binary protocol V2 36 bytes/node
        WS->>Browser: WebSocket frame
    end

    REST-->>User: 200 OK
```

---

## 7. Hexagonal Architecture (Ports and Adapters)
- VisionFlow's core architectural pattern: business logic depends on abstractions, not implementations
- **Inside**: Domain logic + Port traits (interfaces) define what the system needs
- **Boundary**: Adapters implement ports using specific technologies
- **Outside**: Infrastructure (Neo4j, CUDA, HTTP clients, WebSocket)
- **Key ports**: `OntologyRepository`, `KnowledgeGraphRepository`, `InferenceEngine`, `GpuPhysicsAdapter`
- **Benefit for ML engineers**: You can swap the inference engine (Whelk) for any OWL reasoner by implementing the `InferenceEngine` port trait

```mermaid
graph LR
    subgraph "Outside - Infrastructure"
        DB[(Neo4j)]
        GPU[CUDA GPU]
        HTTP[HTTP Clients]
        WS[WebSocket Clients]
    end

    subgraph "Boundary - Adapters"
        DA["Neo4j Adapters"]
        GA["GPU Adapters"]
        HA["HTTP Handlers"]
        WA["WS Handlers"]
    end

    subgraph "Inside - Domain + Ports"
        P["Port Traits"]
        D["Domain Logic"]
    end

    DB <--> DA
    GPU <--> GA
    HTTP <--> HA
    WS <--> WA

    DA --> P
    GA --> P
    HA --> P
    WA --> P

    P --- D
```

---

## 8. Event-Driven Architecture
- Domain events decouple system components: graph mutations, ontology changes, physics events, settings updates
- **Event Bus**: Central pub/sub with middleware pipeline (logging, metrics, validation, retry)
- **Event Store**: Persistent event log for audit trails and event sourcing
- **Key domain events**: `NodeAdded`, `EdgeRemoved`, `ClassAdded`, `InferenceCompleted`, `SimulationStarted`
- **Automatic inference triggers**: When ontology changes (new class/property/axiom), the event bus auto-triggers Whelk reasoning with configurable debouncing

```mermaid
sequenceDiagram
    participant Service as Service Layer
    participant Bus as Event Bus
    participant MW as Middleware
    participant Handler as Event Handlers
    participant Store as Event Store

    Service->>Bus: publish(event)
    Bus->>MW: before_publish(event)
    MW->>Bus: enriched event
    Bus->>Store: save(event)
    Bus->>Handler: handle(event)
    Handler->>Bus: result
    Bus->>MW: after_publish(event)
    Bus->>Service: success
```

---

## 9. GPU Actor Supervision Tree
- Actix actor system coordinates GPU operations through a supervisor hierarchy
- **GPUManagerActor**: Top-level supervisor, routes messages to specialized child actors
- **GPUResourceActor**: CUDA device management (context creation, memory allocation, data transfer)
- **ForceComputeActor**: Executes physics simulation kernels (preserves iteration state across settings updates)
- **Error recovery**: GPU initialization failures trigger CPU fallback; runtime kernel failures trigger GPU state reset and retry
- **Physics loop**: GraphSupervisor requests physics step => GPUManager delegates to ForceCompute => kernels execute on GPU => positions downloaded via GPUResource => broadcast to clients

```mermaid
sequenceDiagram
    participant AppState
    participant Supervisor as GraphSupervisor
    participant GPUManager as GPUManagerActor
    participant GPUResource as GPUResourceActor
    participant ForceCompute as ForceComputeActor
    participant GPU as GPU Hardware

    Note over AppState: System Startup
    AppState->>GPUManager: Create GPUManagerActor
    AppState->>Supervisor: Create GraphSupervisor

    Note over Supervisor, GPU: GPU Initialization
    Supervisor->>GPUManager: InitializeGPU
    GPUManager->>GPUResource: Initialize CUDA device
    GPUResource->>GPU: CUDA context creation
    GPU-->>GPUResource: Context created
    GPUResource->>GPU: Memory allocation
    GPU-->>GPUResource: Memory allocated
    GPUResource-->>GPUManager: GPU ready

    Note over Supervisor, GPU: Graph Data Upload
    Supervisor->>GPUManager: UpdateGPUGraphData(nodes, edges)
    GPUManager->>GPUResource: Upload graph data
    GPUResource->>GPU: Data transfer
    GPU-->>GPUResource: Data uploaded
    GPUResource-->>GPUManager: Graph data ready

    Note over Supervisor, GPU: Physics Loop
    loop Simulation Step 60 Hz
        Supervisor->>GPUManager: RequestPhysicsStep
        GPUManager->>ForceCompute: Execute simulation
        ForceCompute->>GPU: Force calculation kernel
        GPU-->>ForceCompute: Forces computed
        ForceCompute->>GPU: Position integration kernel
        GPU-->>ForceCompute: Positions updated
        ForceCompute-->>GPUManager: Step complete
        GPUManager->>GPUResource: Download positions
        GPUResource->>GPU: Memory transfer
        GPU-->>GPUResource: Position data
        GPUResource-->>GPUManager: Data available
        GPUManager-->>Supervisor: Updated positions
    end
```

---

## 10. Semantic Forces System
- The core innovation: ontological relationships become physical forces that drive graph self-organization
- **5 force types**: DAG layout (hierarchy), type clustering (grouping), collision detection (overlap prevention), attribute-weighted (custom), edge-type-weighted (per-relationship spring strength)
- **3-tier architecture**: Frontend controls => Rust SemanticPhysicsEngine => CUDA kernels
- **ML relevance**: This is effectively a physics-based embedding where spatial position encodes ontological structure

```mermaid
graph TD
    A[Semantic Forces] --> B[DAG Layout]
    A --> C[Type Clustering]
    A --> D[Collision Detection]
    A --> E[Attribute Weighted]
    A --> F[Edge Type Weighted]

    B --> B1[Hierarchical Positioning<br/>SubClassOf chains]
    C --> C1[Group by Node Type<br/>Person, Organization, Concept]
    D --> D1[Prevent Overlap<br/>Radius-based repulsion]
    E --> E1[Custom Attribute Forces<br/>Property-driven layout]
    F --> F1[Per-Edge Spring Strengths<br/>Dependency vs Hierarchy vs Association]
```

---

## 11. Data Lineage (Source to Display)
- Complete traceability: every pixel on screen traces back to a source file in GitHub
- **GitHub file** => **Sync metadata** (SHA1 hash) => **Neo4j node** (graph storage) => **OWL class** (ontology) => **Asserted axiom** => **Inferred axiom** (by Whelk) => **Semantic constraint** (physics force) => **GPU force computation** => **Node position** (x,y,z) => **Client display** (Three.js render)
- **Key detail**: Inferred axioms produce constraints with 0.3x strength multiplier, creating subtle spatial influence vs the full-strength asserted axioms

```mermaid
graph TB
    GH["GitHub File:<br/>artificial-intelligence.md"]

    META["Sync Metadata:<br/>SHA1: abc123...<br/>last-modified: 2025-11-03"]

    NODE["Neo4j GraphNode:<br/>id: 1<br/>metadataId: artificial-intelligence<br/>label: Artificial Intelligence"]

    CLASS["OWL Class:<br/>iri: AI<br/>label: AI System"]

    AXIOM_A["Asserted Axiom:<br/>AI subClassOf ComputationalSystem<br/>is_inferred: false"]

    AXIOM_I["Inferred Axiom:<br/>AI subClassOf InformationProcessor<br/>is_inferred: true<br/>by Whelk-rs"]

    CONS1["Semantic Constraint:<br/>type: Spring<br/>strength: 0.5<br/>is_inferred: false"]

    CONS2["Semantic Constraint:<br/>type: Spring<br/>strength: 0.15<br/>is_inferred: true -- 0.3x"]

    FORCE["GPU Force:<br/>node 1 attracted to 2 -- strong<br/>node 1 attracted to 3 -- weak"]

    POS["Node Position:<br/>x: 42.3, y: 15.7, z: -8.2"]

    CLIENT_DISPLAY["Client Display:<br/>3D rendered at 42.3, 15.7, -8.2"]

    GH --> META
    GH --> NODE
    GH --> CLASS
    CLASS --> AXIOM_A
    AXIOM_A --> AXIOM_I
    AXIOM_A --> CONS1
    AXIOM_I --> CONS2
    CONS1 --> FORCE
    CONS2 --> FORCE
    FORCE --> POS
    POS --> CLIENT_DISPLAY

    style GH fill:#e1f5ff
    style NODE fill:#f0e1ff
    style CLASS fill:#e8f5e9
    style AXIOM_A fill:#fff9c4
    style AXIOM_I fill:#ffecb3
    style CONS1 fill:#ffe1e1
    style CONS2 fill:#ffcdd2
    style FORCE fill:#ff8a80
    style POS fill:#c8e6c9
    style CLIENT_DISPLAY fill:#a5d6a7
```

---

## 12. Pipeline Timing (End-to-End)
- Total cold-start latency: ~5.8 seconds from GitHub fetch to first rendered frame
- After initial sync, differential sync achieves 90%+ skip rate
- **GPU physics**: 16ms per frame sustained (60 FPS) even with 100K+ nodes
- **Binary WebSocket**: 50ms transmission latency for full graph update
- **Key optimization**: LRU cache provides 90x speedup for repeated reasoning queries

```mermaid
gantt
    title Complete Data Flow Timing - GitHub to Client
    dateFormat X
    axisFormat %L ms

    section GitHub Sync
    Fetch files          :0, 2000
    Parse content        :2000, 1000
    Store to Neo4j       :3000, 500

    section Reasoning
    Load ontology        :3500, 200
    Whelk-rs inference   :3700, 1500
    Store inferred       :5200, 300

    section GPU Physics
    Generate constraints :5500, 100
    Upload to GPU        :5600, 50
    Compute forces       :5650, 16
    Download positions   :5666, 34

    section Client
    WebSocket transmit   :5700, 50
    Render frame         :5750, 16
```

---

## 13. Ontology-to-Physics Mapping Reference
- How each OWL axiom type translates to a GPU physics force
- This is the bridge between symbolic AI (ontology reasoning) and numerical simulation (GPU physics)

| OWL Axiom Type | Physics Force | Spring Constant | Visual Effect |
|---|---|---|---|
| `SubClassOf(A, B)` | Spring attraction | k=0.5 | Child classes cluster near parents |
| `DisjointWith(A, B)` | Coulomb repulsion | k=-0.8 | Disjoint classes pushed apart |
| `EquivalentClasses(A, B)` | Strong spring | k=1.0 | Synonyms rendered together |
| `ObjectProperty(A, B)` | Directional alignment | k=0.3 | Property domains/ranges aligned |
| **Inferred axiom** (any type) | Same as above | **0.3x multiplier** | Subtle influence vs asserted |

---

## 14. Performance Characteristics
- Benchmarks demonstrating GPU acceleration impact for graph computations

| Operation | CPU (100K nodes) | GPU (100K nodes) | Speedup |
|---|---|---|---|
| Physics Simulation | 1,600ms | 16ms | **100x** |
| Leiden Clustering | 800ms | 12ms | **67x** |
| SSSP Pathfinding | 500ms | 8ms | **62x** |
| Force-Directed Layout | 2,000ms | 20ms | **100x** |

| System Metric | Target | Measured |
|---|---|---|
| Rendering FPS | 60 | 60 sustained at 150K nodes |
| WebSocket Latency | <20ms | <10ms average |
| Graph Query (Cypher) | <100ms | 50ms average |
| Ontology Reasoning | <500ms | 200ms average |
| Whelk-rs vs Java Reasoners | - | **10-100x faster** |
| LRU Cache Hit | - | **90x speedup** |
| Binary Protocol vs JSON | - | **80% bandwidth reduction** |

---

## 15. Solid Sidecar Architecture
- Decentralized data ownership via Linked Data Platform (LDP) using JSON Solid Server as a sidecar container
- **Neo4j => Solid sync**: Rust backend serializes graph data to RDF Turtle, batch-uploads to user-owned pods
- **NIP-98 auth**: Nostr-based decentralized identity for pod access control
- **Real-time notifications**: WebSocket-based resource change events from JSS to clients

```mermaid
flowchart TB
    subgraph Neo4j["Neo4j Graph Database"]
        N1[Nodes]
        N2[Edges]
        N3[Metadata]
    end

    subgraph Backend["Rust Backend"]
        B1[Sync Service]
        B2[RDF Serializer]
        B3[Batch Processor]
    end

    subgraph JSS["JSS Sidecar - Node.js"]
        J1[LDP Server]
        J2[Pod Manager]
        J3[Notification Hub]
    end

    subgraph Storage["Pod Storage"]
        S1["/pods/user1/graph/"]
        S2["node-1.ttl"]
        S3["node-2.ttl"]
    end

    N1 --> B1
    N2 --> B1
    N3 --> B1
    B1 --> B2
    B2 --> B3
    B3 -->|"PUT /pods/user/graph/id.ttl"| J1
    J1 --> J2
    J2 --> S1
    S1 --> S2
    S1 --> S3
    J2 --> J3
    J3 -->|"WebSocket notification"| Client
```

---

## Quick Reference: Technology Stack
- **Language**: Rust 1.75+ (backend), TypeScript (client)
- **Web Framework**: Actix Web 4.11
- **3D Rendering**: Three.js 0.175 + React Three Fiber
- **Graph Database**: Neo4j 5.13 (Cypher queries, Bolt protocol)
- **GPU Compute**: CUDA 12.4 (39 custom kernels via cudarc Rust bindings)
- **OWL Reasoning**: Whelk-rs (Rust OWL 2 EL++ reasoner)
- **Authentication**: Nostr NIP-98 (decentralized, passkey-based)
- **Real-time Protocol**: Custom 36-byte binary WebSocket (80% smaller than JSON)
- **XR Support**: WebXR (Meta Quest 3 with hand tracking)
- **Decentralized Storage**: Solid pods via JSON Solid Server sidecar
