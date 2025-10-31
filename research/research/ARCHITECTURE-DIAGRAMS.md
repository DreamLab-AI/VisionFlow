# Legacy Knowledge Graph - Architecture Diagrams

## System Overview

```mermaid
graph TB
    subgraph "Client Layer"
        WS[WebSocket Client]
    end

    subgraph "Actor System"
        GS[GraphServiceActor<br/>State Manager]
        PO[PhysicsOrchestratorActor<br/>60 Hz Tick]
        FC[ForceComputeActor<br/>GPU Physics]
        GPU[GPUResourceActor<br/>Context Sharing]
        CA[ConstraintActor]
        CL[ClusteringActor]
        OCA[OntologyConstraintActor]
    end

    subgraph "GPU Layer"
        CUDA[CUDA Kernels]
        SM[Shared Memory]
        GM[GPU Memory]
    end

    subgraph "Storage Layer"
        DB[(SQLite DB)]
    end

    WS -->|Binary Protocol| GS
    GS -->|Physics Tick| PO
    PO -->|Compute Forces| FC
    FC -->|GPU Lock| GPU
    GPU -->|Context| FC
    FC -->|Forces/Positions| CUDA
    CUDA -->|Results| FC
    FC -->|Positions| GS
    GS -->|Broadcast| WS

    CA --> PO
    CL --> PO
    OCA --> CA

    FC -.->|Read/Write| GM
    CUDA -.->|Shared Mem| SM
    GS -.->|Persist| DB

    style FC fill:#ff6b6b
    style CUDA fill:#4ecdc4
    style GPU fill:#ffe66d
    style GS fill:#95e1d3
```

## GPU Kernel Pipeline

```mermaid
graph LR
    subgraph "Input Data"
        POS[Positions<br/>x,y,z]
        VEL[Velocities<br/>vx,vy,vz]
        EDGES[Edges<br/>CSR Format]
        CONSTRAINTS[Constraints]
    end

    subgraph "Kernel 1: Build Grid"
        HASH[Spatial Hash]
        CELL[Cell Assignment]
    end

    subgraph "Kernel 2: Force Pass"
        REP[Repulsion<br/>via Grid]
        SPRING[Spring<br/>via CSR]
        CONS[Constraint<br/>Forces]
        SUM[Force Sum]
    end

    subgraph "Kernel 3: Integrate"
        ACCEL[Acceleration<br/>F/m]
        VELUP[Velocity<br/>Update]
        POSUP[Position<br/>Update]
        BOUNDARY[Boundary<br/>Check]
    end

    subgraph "Output"
        NEWPOS[New Positions]
        NEWVEL[New Velocities]
        STATS[Kinetic Energy<br/>Active Nodes]
    end

    POS --> HASH
    HASH --> CELL

    CELL --> REP
    EDGES --> SPRING
    CONSTRAINTS --> CONS

    REP --> SUM
    SPRING --> SUM
    CONS --> SUM

    SUM --> ACCEL
    VEL --> ACCEL
    ACCEL --> VELUP
    VELUP --> POSUP
    POSUP --> BOUNDARY

    BOUNDARY --> NEWPOS
    BOUNDARY --> NEWVEL
    BOUNDARY --> STATS

    style HASH fill:#4ecdc4
    style REP fill:#ff6b6b
    style SPRING fill:#95e1d3
    style CONS fill:#ffe66d
    style ACCEL fill:#a8e6cf
```

## Clustering Pipeline

```mermaid
graph TB
    subgraph "Input"
        NODES[Node Positions<br/>10K nodes]
    end

    subgraph "K-means Clustering"
        INIT[K-means++<br/>Init Centroids]
        ASSIGN[Assign to<br/>Nearest Centroid]
        UPDATE[Update<br/>Centroids]
        CHECK{Converged?}
    end

    subgraph "LOF Anomaly Detection"
        KNN[k-NN Search<br/>via Spatial Grid]
        RD[Reachability<br/>Distance]
        LRD[Local Reachability<br/>Density]
        LOF[LOF Score]
    end

    subgraph "Label Propagation"
        VOTE[Neighbor<br/>Voting]
        LABEL[Update<br/>Labels]
        MOD[Modularity<br/>Computation]
    end

    NODES --> INIT
    INIT --> ASSIGN
    ASSIGN --> UPDATE
    UPDATE --> CHECK
    CHECK -->|No| ASSIGN
    CHECK -->|Yes| RESULT1[Cluster IDs]

    NODES --> KNN
    KNN --> RD
    RD --> LRD
    LRD --> LOF
    LOF --> RESULT2[Anomaly Scores]

    NODES --> VOTE
    VOTE --> LABEL
    LABEL --> MOD
    MOD --> RESULT3[Communities]

    style INIT fill:#4ecdc4
    style KNN fill:#ff6b6b
    style VOTE fill:#95e1d3
    style RESULT1 fill:#ffe66d
    style RESULT2 fill:#ffe66d
    style RESULT3 fill:#ffe66d
```

## SSSP Hybrid Architecture

```mermaid
graph TB
    subgraph "CPU Coordinator"
        INPUT[Source Node]
        DECISION{Graph Size<br/>Frontier Size}
        OUTPUT[Distance Array]
    end

    subgraph "GPU Path (Large Graphs)"
        KSTEP[K-Step Relaxation]
        PIVOT[Detect Pivots]
        PARTITION[Partition<br/>Frontier]
        COMPACT[Frontier<br/>Compaction]
    end

    subgraph "CPU/WASM Path (Small Graphs)"
        DIJKSTRA[Bounded<br/>Dijkstra]
    end

    INPUT --> DECISION
    DECISION -->|Large<br/>> 1K nodes| KSTEP
    DECISION -->|Small<br/>< 1K nodes| DIJKSTRA

    KSTEP --> PIVOT
    PIVOT --> PARTITION
    PARTITION --> COMPACT
    COMPACT -->|Next Iter| KSTEP
    COMPACT -->|Done| OUTPUT

    DIJKSTRA --> OUTPUT

    style DECISION fill:#ffe66d
    style KSTEP fill:#4ecdc4
    style COMPACT fill:#ff6b6b
    style DIJKSTRA fill:#95e1d3
```

## Constraint Application Flow

```mermaid
sequenceDiagram
    participant Client
    participant ConstraintActor
    participant ForceComputeActor
    participant GPU

    Client->>ConstraintActor: UpdateConstraints
    ConstraintActor->>ConstraintActor: Validate Constraints
    ConstraintActor->>ForceComputeActor: UploadConstraintsToGPU
    ForceComputeActor->>GPU: Copy Constraint Data

    loop Every Physics Frame
        ForceComputeActor->>GPU: force_pass_kernel
        GPU->>GPU: Compute Progressive Weight
        GPU->>GPU: Apply Constraint Forces
        GPU->>ForceComputeActor: Updated Positions
    end

    ForceComputeActor->>Client: Position Updates (Binary)
    ConstraintActor->>Client: Constraint Stats (JSON)
```

## Adaptive Throttling Logic

```mermaid
graph TD
    START[Physics Iteration] --> CHECK_STABLE{Stable for<br/>> 600 iters?}

    CHECK_STABLE -->|Yes| INTERVAL_30[Download Interval = 30<br/>~2 Hz]
    CHECK_STABLE -->|No| CHECK_SIZE{Node Count}

    CHECK_SIZE -->|> 10K| INTERVAL_10[Download Interval = 10<br/>~6 Hz]
    CHECK_SIZE -->|> 1K| INTERVAL_5[Download Interval = 5<br/>~12 Hz]
    CHECK_SIZE -->|< 1K| INTERVAL_2[Download Interval = 2<br/>~30 Hz]

    INTERVAL_30 --> DOWNLOAD{iter %<br/>interval == 0?}
    INTERVAL_10 --> DOWNLOAD
    INTERVAL_5 --> DOWNLOAD
    INTERVAL_2 --> DOWNLOAD

    DOWNLOAD -->|Yes| GPU_DOWNLOAD[Download from GPU]
    DOWNLOAD -->|No| SKIP[Skip Download]

    GPU_DOWNLOAD --> BROADCAST[Broadcast to Clients]
    SKIP --> NEXT[Next Iteration]
    BROADCAST --> NEXT

    style CHECK_STABLE fill:#ffe66d
    style GPU_DOWNLOAD fill:#4ecdc4
    style BROADCAST fill:#95e1d3
    style SKIP fill:#ff6b6b
```

## Stability Gate Decision

```mermaid
graph TD
    START[Calculate Kinetic Energy] --> AVG[Average KE]
    START --> COUNT[Active Node Count]

    AVG --> CHECK_ENERGY{KE <<br/>threshold?}
    COUNT --> CHECK_MOTION{Active <<br/>1% total?}

    CHECK_ENERGY -->|Yes| STABLE[STABLE STATE]
    CHECK_MOTION -->|Yes| STABLE

    CHECK_ENERGY -->|No| CHECK_BOTH{Both No?}
    CHECK_MOTION -->|No| CHECK_BOTH

    CHECK_BOTH -->|Yes| ACTIVE[ACTIVE STATE]

    STABLE --> PAUSE[Pause Physics<br/>0% GPU Usage]
    ACTIVE --> RUN[Run Physics<br/>60-90% GPU]

    PAUSE --> SAVE[Save 80%<br/>GPU Cycles]
    RUN --> COMPUTE[Full Computation]

    style STABLE fill:#95e1d3
    style ACTIVE fill:#ff6b6b
    style SAVE fill:#a8e6cf
```

## Database Schema Relationships

```mermaid
erDiagram
    NODES ||--o{ EDGES : has
    NODES {
        int id PK
        text metadata_id UK
        text label
        real x
        real y
        real z
        real vx
        real vy
        real vz
        real ax
        real ay
        real az
        real mass
        real charge
        text color
        real size
        real opacity
        text node_type
        int is_pinned
        real pin_x
        real pin_y
        real pin_z
        text metadata
    }

    EDGES {
        int id PK
        int source_id FK
        int target_id FK
        text relation_type
        real weight
        text metadata
    }

    CONSTRAINTS ||--o{ CONSTRAINT_NODES : affects
    CONSTRAINTS {
        int id PK
        text kind
        text params
        real weight
        int active
    }

    CONSTRAINT_NODES {
        int constraint_id FK
        int node_id FK
    }
```

## Memory Layout

```mermaid
graph TB
    subgraph "GPU Memory (~1.2 MB for 10K nodes)"
        subgraph "Node Arrays"
            POS_X[pos_x: 40 KB]
            POS_Y[pos_y: 40 KB]
            POS_Z[pos_z: 40 KB]
            VEL_X[vel_x: 40 KB]
            VEL_Y[vel_y: 40 KB]
            VEL_Z[vel_z: 40 KB]
            MASS[mass: 40 KB]
            CHARGE[charge: 40 KB]
        end

        subgraph "Edge Arrays (CSR)"
            ROW[row_offsets: 40 KB]
            COL[col_indices: 200 KB]
            WEIGHT[weights: 200 KB]
        end

        subgraph "Spatial Grid"
            GRID_KEY[grid_keys: 40 KB]
            CELL_START[cell_start: 100 KB]
            CELL_END[cell_end: 100 KB]
        end

        subgraph "Constraints"
            CONS_DATA[constraint_data: ~10 KB]
        end
    end

    TOTAL[Total: ~1.2 MB] --> POS_X
    TOTAL --> ROW
    TOTAL --> GRID_KEY
    TOTAL --> CONS_DATA

    style TOTAL fill:#ffe66d
    style POS_X fill:#4ecdc4
    style ROW fill:#95e1d3
    style GRID_KEY fill:#ff6b6b
```

## Performance Scaling

```mermaid
graph LR
    subgraph "Small Graphs (< 1K nodes)"
        SMALL_FPS[200-500 FPS]
        SMALL_GPU[60-70% GPU]
        SMALL_DOWNLOAD[~30 Hz]
    end

    subgraph "Medium Graphs (1K-10K nodes)"
        MED_FPS[50-100 FPS]
        MED_GPU[70-85% GPU]
        MED_DOWNLOAD[~12 Hz]
    end

    subgraph "Large Graphs (> 10K nodes)"
        LARGE_FPS[20-60 FPS]
        LARGE_GPU[80-90% GPU]
        LARGE_DOWNLOAD[~6 Hz]
    end

    SMALL_FPS -->|Scale Up| MED_FPS
    MED_FPS -->|Scale Up| LARGE_FPS

    SMALL_GPU -->|Higher Load| MED_GPU
    MED_GPU -->|Higher Load| LARGE_GPU

    SMALL_DOWNLOAD -->|Throttle| MED_DOWNLOAD
    MED_DOWNLOAD -->|Throttle| LARGE_DOWNLOAD

    style SMALL_FPS fill:#a8e6cf
    style MED_FPS fill:#ffe66d
    style LARGE_FPS fill:#ff6b6b
```

## Concurrency Model

```mermaid
graph TB
    subgraph "Main Thread"
        ORCHESTRATOR[PhysicsOrchestratorActor<br/>60 Hz Tick]
    end

    subgraph "GPU Thread"
        FORCE[ForceComputeActor<br/>Exclusive GPU Access]
    end

    subgraph "Query Threads (Multiple)"
        Q1[Clustering Query]
        Q2[Analytics Query]
        Q3[SSSP Query]
    end

    subgraph "GPU Resource Manager"
        RWLOCK[RwLock<br/>Shared GPU Context]
    end

    ORCHESTRATOR -->|Message| FORCE
    FORCE -->|Write Lock| RWLOCK

    Q1 -->|Read Lock| RWLOCK
    Q2 -->|Read Lock| RWLOCK
    Q3 -->|Read Lock| RWLOCK

    RWLOCK -->|Context| FORCE
    RWLOCK -->|Context| Q1
    RWLOCK -->|Context| Q2
    RWLOCK -->|Context| Q3

    style FORCE fill:#ff6b6b
    style RWLOCK fill:#4ecdc4
    style Q1 fill:#95e1d3
    style Q2 fill:#95e1d3
    style Q3 fill:#95e1d3
```

---

## Notes

These diagrams visualize the key architectural patterns found in the legacy knowledge graph system. They serve as reference for understanding data flow, concurrency, and performance characteristics during migration.

**See Also**:
- Full Analysis: `Legacy-Knowledge-Graph-System-Analysis.md`
- Executive Summary: `EXECUTIVE-SUMMARY.md`
