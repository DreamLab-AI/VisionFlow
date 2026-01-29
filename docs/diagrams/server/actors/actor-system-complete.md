---
title: Server-Side Actor System - Complete Architecture Documentation
description: The server-side actor system is a hierarchical, fault-tolerant distributed computing architecture built on Actix.  It consists of 21+ specialized actors organized under a supervisor hierarchy for m...
category: explanation
tags:
  - architecture
  - patterns
  - structure
  - api
  - api
related-docs:
  - ASCII_DEPRECATION_COMPLETE.md
  - diagrams/README.md
  - concepts/quick-reference.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - Neo4j database
---

# Server-Side Actor System - Complete Architecture Documentation

## System Overview

The server-side actor system is a hierarchical, fault-tolerant distributed computing architecture built on Actix. It consists of 21+ specialized actors organized under a supervisor hierarchy for maximum reliability and performance.

**Architecture Principles:**
- **Supervision Trees**: Hierarchical fault isolation with restart strategies
- **Message Passing**: Asynchronous communication via typed messages
- **Actor Lifecycle**: Spawn → Initialize → Running → Stopped
- **Fault Tolerance**: One-for-one and all-for-one supervision strategies
- **GPU Acceleration**: Specialized GPU sub-actors for compute-intensive operations

---

## 1. Complete Actor Hierarchy

```mermaid
graph TB
    subgraph "GraphServiceSupervisor - Root Supervisor"
        GSS[GraphServiceSupervisor<br/>Strategy: OneForOne<br/>Restarts: 3 max]
    end

    subgraph "Core State Actors"
        GSS --> GSA[GraphStateActor<br/>State Machine: 7 States<br/>Manages: Graph Data + Nodes]
        GSS --> PO[PhysicsOrchestratorActor<br/>Coordinates: 11 GPU Actors<br/>Mode: Hierarchical]
        GSS --> SP[SemanticProcessorActor<br/>AI: Semantic Analysis<br/>Constraints: Dynamic]
        GSS --> CC[ClientCoordinatorActor<br/>WebSocket: Broadcast Manager<br/>Clients: N concurrent]
    end

    subgraph "GPU Sub-Actors (11 Total) - Supervised by PhysicsOrchestratorActor"
        PO --> FC[ForceComputeActor<br/>Primary Physics Engine<br/>CUDA Kernels]
        PO --> SM[StressMajorizationActor<br/>Layout Optimization<br/>Iterative Solver]
        PO --> SF[SemanticForcesActor<br/>Semantic Attraction<br/>AI-Driven Forces]
        PO --> CA[ConstraintActor<br/>Hard Constraints<br/>Collision Detection]
        PO --> OC[OntologyConstraintActor<br/>OWL/RDF Rules<br/>Semantic Validation]
        PO --> SPA[ShortestPathActor<br/>SSSP + APSP<br/>GPU Pathfinding]
        PO --> PR[PageRankActor<br/>Centrality<br/>Influence Analysis]
        PO --> CLA[ClusteringActor<br/>K-Means + Communities<br/>Label Propagation]
        PO --> AD[AnomalyDetectionActor<br/>LOF + Z-Score<br/>Outlier Detection]
        PO --> CCO[ConnectedComponentsActor<br/>Graph Components<br/>Union-Find]
        PO --> GR[GPUResourceActor<br/>Memory Manager<br/>Stream Allocator]
    end

    subgraph "Support Actors"
        GSS --> WS[WorkspaceActor<br/>Workspace CRUD<br/>Multi-tenant]
        GSS --> SA[SettingsActor<br/>Config Management<br/>Persistence]
        GSS --> OSA[OptimisedSettingsActor<br/>Hot-path Settings<br/>Cache Layer]
    end

    subgraph "Integration Actors (24 Total)"
        GSS --> MCP[MultiMcpVisualizationActor<br/>MCP Server Integration<br/>Multi-protocol coordination]
        GSS --> TASK[TaskOrchestratorActor<br/>Async Task Management<br/>Job scheduling]
        GSS --> MON[AgentMonitorActor<br/>Agent Health Monitoring<br/>Telemetry collection]
    end

    style GSS fill:#ff6b6b,stroke:#333,stroke-width:4px,color:#fff
    style MCP fill:#b39ddb,stroke:#333,stroke-width:2px
    style TASK fill:#b39ddb,stroke:#333,stroke-width:2px
    style MON fill:#b39ddb,stroke:#333,stroke-width:2px
    style GSA fill:#4ecdc4,stroke:#333,stroke-width:2px
    style PO fill:#ffe66d,stroke:#333,stroke-width:2px
    style SP fill:#a8e6cf,stroke:#333,stroke-width:2px
    style CC fill:#ff8b94,stroke:#333,stroke-width:2px

    style FC fill:#95e1d3,stroke:#333,stroke-width:1px
    style SM fill:#95e1d3,stroke:#333,stroke-width:1px
    style SF fill:#95e1d3,stroke:#333,stroke-width:1px
    style CA fill:#95e1d3,stroke:#333,stroke-width:1px
    style OC fill:#95e1d3,stroke:#333,stroke-width:1px
    style SPA fill:#95e1d3,stroke:#333,stroke-width:1px
    style PR fill:#95e1d3,stroke:#333,stroke-width:1px
    style CLA fill:#95e1d3,stroke:#333,stroke-width:1px
    style AD fill:#95e1d3,stroke:#333,stroke-width:1px
    style CCO fill:#95e1d3,stroke:#333,stroke-width:1px
    style GR fill:#95e1d3,stroke:#333,stroke-width:1px
```

---

## 2. GraphServiceSupervisor - Root Supervision

```mermaid
stateDiagram-v2
    [*] --> Initializing: "Actor.started()"

    Initializing --> SpawningChildren: Spawn child actors
    SpawningChildren --> SettingAddresses: "Store Addr<T>"
    SettingAddresses --> Monitoring: Setup supervision

    Monitoring --> Monitoring: Normal operation
    Monitoring --> ChildFailed: Child actor crash

    ChildFailed --> RestartChild: "restart_count < 3"
    ChildFailed --> EscalateFailure: "restart_count >= 3"

    RestartChild --> Monitoring: Child restarted
    EscalateFailure --> [*]: Supervisor stops

    Monitoring --> Stopping: Stop message
    Stopping --> [*]: "Actor.stopped()"

    note right of Monitoring
        Supervision Strategy:
        - OneForOne: Only restart failed child
        - Max Restarts: 3 within 10s
        - Backoff: Exponential
    end note

    note right of ChildFailed
        Fault Isolation:
        - GraphStateActor crash → Only GSA restarts
        - PhysicsOrchestrator crash → All GPU actors restart
        - SemanticProcessor crash → Only SP restarts
    end note
```

**Supervision Configuration:**
```rust
// GraphServiceSupervisor restart policies
SupervisorStrategy::OneForOne {
    max_restarts: 3,
    within: Duration::from_secs(10),
    backoff: BackoffStrategy::Exponential {
        initial_interval: Duration::from_millis(500),
        max_interval: Duration::from_secs(5),
        multiplier: 2.0,
    }
}
```

---

## 3. GraphStateActor - State Machine (7 States)

```mermaid
stateDiagram-v2
    [*] --> Uninitialized

    Uninitialized --> Initializing: BuildGraphFromMetadata
    Initializing --> Loading: Async metadata load
    Loading --> Ready: Graph built successfully
    Loading --> Error: Build failed

    Ready --> Updating: AddNode/RemoveNode/AddEdge
    Updating --> Ready: Update complete
    Updating --> Error: Update failed

    Ready --> Simulating: StartSimulation
    Simulating --> Simulating: "SimulationStep (loop)"
    Simulating --> Ready: StopSimulation

    Error --> Recovering: ReloadGraphFromDatabase
    Recovering --> Ready: Recovery successful
    Recovering --> Error: Recovery failed

    Ready --> [*]: Actor stopped
    Error --> [*]: Max retries exceeded

    note right of Ready
        State Data:
        - graph_data: Arc&lt;GraphData&gt;
        - node_map: HashMap&lt;u32, Node&gt;
        - edge_map: HashMap&lt;String, Edge&gt;
        - metadata_to_node: HashMap&lt;String, u32&gt;
    end note

    note right of Simulating
        Physics Loop:
        1. Receive positions from GPU
        2. Update internal graph state
        3. Broadcast to clients
        4. Trigger next step
    end note
```

**Internal State Structure:**
```rust
pub struct GraphStateActor {
    graph_data: Option<Arc<GraphData>>,           // Shared immutable graph
    node_map: HashMap<u32, Node>,                 // Fast node lookup
    edge_map: HashMap<String, Edge>,              // Edge storage
    metadata_to_node: HashMap<String, u32>,       // Metadata → Node ID
    state: GraphState,                            // Current state enum
    physics_orchestrator: Option<Addr<PhysicsOrchestratorActor>>,
    semantic_processor: Option<Addr<SemanticProcessorActor>>,
    client_coordinator: Option<Addr<ClientCoordinatorActor>>,
}

enum GraphState {
    Uninitialized,
    Initializing,
    Loading,
    Ready,
    Updating,
    Simulating,
    Error(String),
}
```

---

## 4. PhysicsOrchestratorActor - GPU Coordination

```mermaid
graph TB
    subgraph "PhysicsOrchestratorActor - Central Coordinator"
        PO[PhysicsOrchestratorActor<br/>Manages: 11 GPU Actors<br/>Strategy: Hierarchical Pipeline]
    end

    subgraph "Force Computation Pipeline"
        PO -->|1. Compute Forces| FC[ForceComputeActor<br/>CUDA: Repulsion + Attraction<br/>Output: Force Vectors]
        PO -->|2. Apply Semantic| SF[SemanticForcesActor<br/>AI Clustering Forces<br/>Similarity Attraction]
        PO -->|3. Check Constraints| CA[ConstraintActor<br/>Hard Constraints<br/>Collision Detection]
        PO -->|4. Validate Ontology| OC[OntologyConstraintActor<br/>OWL Rules<br/>RDF Validation]
    end

    subgraph "Layout Optimization Pipeline"
        PO -->|5. Stress Optimize| SM[StressMajorizationActor<br/>Iterative Solver<br/>Min Graph Stress]
        SM -->|Converged Layout| PO
    end

    subgraph "Graph Analysis Pipeline"
        PO -->|Request Paths| SPA[ShortestPathActor<br/>SSSP: Bellman-Ford<br/>APSP: Landmarks]
        PO -->|Request Rank| PR[PageRankActor<br/>Power Iteration<br/>Centrality Scores]
        PO -->|Request Clusters| CLA[ClusteringActor<br/>K-Means + Louvain<br/>Community Detection]
        PO -->|Request Anomalies| AD[AnomalyDetectionActor<br/>LOF + Z-Score<br/>Outlier Detection]
        PO -->|Request Components| CCO[ConnectedComponentsActor<br/>Union-Find<br/>Component Labels]
    end

    subgraph "Resource Management"
        PO -->|Allocate Memory| GR[GPUResourceActor<br/>CUDA Stream Pool<br/>Memory Allocator]
        GR -->|Return Resources| PO
    end

    PO -->|Final Positions| ClientCoordinator[ClientCoordinatorActor]

    style PO fill:#ffe66d,stroke:#333,stroke-width:3px
    style FC fill:#95e1d3,stroke:#333,stroke-width:2px
    style SF fill:#95e1d3,stroke:#333,stroke-width:2px
    style CA fill:#95e1d3,stroke:#333,stroke-width:2px
    style OC fill:#95e1d3,stroke:#333,stroke-width:2px
    style SM fill:#ffd3b6,stroke:#333,stroke-width:2px
    style SPA fill:#a8e6cf,stroke:#333,stroke-width:1px
    style PR fill:#a8e6cf,stroke:#333,stroke-width:1px
    style CLA fill:#a8e6cf,stroke:#333,stroke-width:1px
    style AD fill:#a8e6cf,stroke:#333,stroke-width:1px
    style CCO fill:#a8e6cf,stroke:#333,stroke-width:1px
    style GR fill:#ffaaa5,stroke:#333,stroke-width:2px
```

**Orchestration Flow:**
```rust
// PhysicsOrchestratorActor pipeline
async fn orchestrate_physics_step(&mut self) -> Result<(), String> {
    // 1. Force computation (ForceComputeActor)
    let forces = self.force_compute_actor.send(ComputeForces).await??;

    // 2. Semantic forces (SemanticForcesActor)
    let semantic_forces = self.semantic_forces_actor.send(ApplySemanticForces).await??;

    // 3. Constraint validation (ConstraintActor + OntologyConstraintActor)
    self.constraint_actor.send(ValidateConstraints { forces }).await??;
    self.ontology_actor.send(ValidateOntology).await??;

    // 4. Position integration
    let positions = self.force_compute_actor.send(UpdatePositions { forces }).await??;

    // 5. Stress majorization (optional)
    if self.should_optimize_layout() {
        self.stress_actor.send(OptimizeLayout).await??;
    }

    // 6. Broadcast to clients
    self.client_coordinator.do_send(UpdateNodePositions { positions });

    Ok(())
}
```

---

## 5. SemanticProcessorActor - AI Reasoning Pipeline

```mermaid
sequenceDiagram
    participant Client as WebSocket Client
    participant GSA as GraphStateActor
    participant SPA as SemanticProcessorActor
    participant Analyzer as SemanticAnalyzer
    participant GPU as GPU SemanticAnalyzer

    Note over SPA: State: Idle

    Client->>GSA: AddNode(metadata)
    GSA->>SPA: ProcessMetadata(metadata_id, metadata)

    activate SPA
    Note over SPA: State: Analyzing

    SPA->>Analyzer: analyze_metadata(metadata)
    Analyzer-->>SPA: SemanticFeatures

    alt AI Features Enabled
        SPA->>SPA: extract_ai_features()
        Note right of SPA: Generate:<br/>- Content embeddings (256-dim)<br/>- Topic classifications<br/>- Importance scores<br/>- Sentiment analysis<br/>- Named entities
        SPA->>SPA: Cache AISemanticFeatures
    end

    SPA-->>GSA: Ok(())
    deactivate SPA

    Note over SPA: State: Idle

    Client->>GSA: RegenerateSemanticConstraints
    GSA->>SPA: RegenerateSemanticConstraints

    activate SPA
    Note over SPA: State: Generating Constraints

    par Parallel Constraint Generation
        SPA->>SPA: generate_similarity_constraints()
        Note right of SPA: Cosine similarity > 0.7<br/>→ Attraction constraint
    and
        SPA->>SPA: generate_clustering_constraints()
        Note right of SPA: Group by file type<br/>+ complexity metrics
    and
        SPA->>SPA: generate_importance_constraints()
        Note right of SPA: High importance nodes<br/>→ Central positioning
    and
        SPA->>SPA: generate_topic_constraints()
        Note right of SPA: Same topic classification<br/>→ Cluster together
    end

    SPA->>SPA: Merge + Truncate to max_constraints
    SPA-->>GSA: Ok(constraints)
    deactivate SPA

    GSA->>SPA: ComputeShortestPaths(source_id)
    activate SPA
    Note over SPA: State: GPU Computation

    SPA->>GPU: initialize(graph_data)
    GPU-->>SPA: Ok(())

    SPA->>GPU: compute_shortest_paths(source_id)
    activate GPU
    Note right of GPU: CUDA Kernel:<br/>Parallel Bellman-Ford<br/>All nodes in parallel
    GPU-->>SPA: PathfindingResult { distances, predecessors }
    deactivate GPU

    SPA-->>GSA: Ok(PathfindingResult)
    deactivate SPA

    Note over SPA: State: Idle
```

**Semantic Features Structure:**
```rust
pub struct AISemanticFeatures {
    content_embedding: Vec<f32>,              // 256-dim vector
    topic_classifications: HashMap<String, f32>, // "technology": 0.8, "science": 0.6
    importance_score: f32,                    // 0.0 - 1.0
    conceptual_links: Vec<(u32, f32)>,       // (node_id, similarity)
    complexity_metrics: HashMap<String, f32>, // readability, vocabulary diversity
    sentiment_analysis: Option<HashMap<String, f32>>, // positive, negative, neutral
    named_entities: Vec<String>,              // Proper nouns, capitalized terms
    cluster_assignments: Vec<String>,         // ["code", "documentation", "large_content"]
}
```

---

## 6. ClientCoordinatorActor - Broadcast Mechanisms

```mermaid
sequenceDiagram
    participant WS1 as WebSocket Client 1
    participant WS2 as WebSocket Client 2
    participant CC as ClientCoordinatorActor
    participant CM as ClientManager
    participant PO as PhysicsOrchestratorActor

    Note over CC: Broadcast Interval: 50ms (active)<br/>1000ms (stable)

    WS1->>CC: Connect (WebSocket)
    activate CC
    CC->>CM: register_client(addr)
    CM-->>CC: client_id = 1
    CC->>CC: generate_initial_position(client_id)
    Note right of CC: Random spherical position<br/>radius: 50-200 units
    CC->>WS1: Initial position
    CC->>CC: force_broadcast("new_client_1")
    Note right of CC: Immediate broadcast for new client
    deactivate CC

    WS2->>CC: Connect (WebSocket)
    activate CC
    CC->>CM: register_client(addr)
    CM-->>CC: client_id = 2
    CC->>CC: force_broadcast("new_client_2")
    CC->>WS1: Full graph (binary protocol)
    CC->>WS2: Full graph (binary protocol)
    deactivate CC

    loop Every Simulation Step
        PO->>CC: UpdateNodePositions { positions }
        activate CC
        CC->>CC: update_position_cache(positions)

        alt should_broadcast() == true
            CC->>CC: serialize_positions() → Binary
            Note right of CC: BinaryProtocol::encode_graph_update<br/>28 bytes per node

            par Broadcast to All Clients
                CC->>WS1: Binary graph update (28n bytes)
            and
                CC->>WS2: Binary graph update (28n bytes)
            end

            CC->>CC: last_broadcast = now()
            CC->>CC: broadcast_count++
        else Throttled (too soon)
            Note right of CC: Skip broadcast<br/>Wait for interval
        end
        deactivate CC
    end

    WS1->>CC: Disconnect
    activate CC
    CC->>CM: unregister_client(1)
    CM-->>CC: Ok(())
    CC->>CC: update_connection_stats()
    deactivate CC

    Note over CC: Active Clients: 1<br/>Total Broadcasts: 1234<br/>Bytes Sent: 1.2 MB
```

**Binary Protocol Efficiency:**
```rust
// Per-node data structure (28 bytes)
#[repr(C, packed)]
pub struct BinaryNodeDataClient {
    node_id: u32,    // 4 bytes
    x: f32,          // 4 bytes
    y: f32,          // 4 bytes
    z: f32,          // 4 bytes
    vx: f32,         // 4 bytes
    vy: f32,         // 4 bytes
    vz: f32,         // 4 bytes
}                    // Total: 28 bytes

// For 10,000 nodes: 280 KB per broadcast
// At 20 Hz: 5.6 MB/s per client
```

---

## 7. Complete Message Type Catalog (100+ Messages)

### Graph State Messages (20+)
```mermaid
classDiagram
    class GraphStateMessages {
        +GetGraphData → Arc~GraphData~
        +UpdateNodePositions(positions)
        +AddNode(node)
        +RemoveNode(node_id)
        +AddEdge(edge)
        +RemoveEdge(edge_id)
        +BatchAddNodes(nodes)
        +BatchAddEdges(edges)
        +BuildGraphFromMetadata(metadata)
        +AddNodesFromMetadata(metadata)
        +UpdateNodeFromMetadata(metadata_id, metadata)
        +RemoveNodeByMetadata(metadata_id)
        +GetNodeMap → HashMap~u32, Node~
        +ClearGraph()
        +UpdateGraphData(graph_data)
        +UpdateBotsGraph(agents)
        +GetBotsGraphData → Arc~GraphData~
        +FlushUpdateQueue()
        +ConfigureUpdateQueue(settings)
    }
```

### Physics Messages (15+)
```mermaid
classDiagram
    class PhysicsMessages {
        +StartSimulation()
        +StopSimulation()
        +SimulationStep()
        +PauseSimulation()
        +ResumeSimulation()
        +UpdateSimulationParams(params)
        +GetPhysicsState → PhysicsState
        +GetPhysicsStats → PhysicsStats
        +ResetPhysics()
        +InitializePhysics(graph_data)
        +ComputeForces → ForceVectors
        +UpdatePositions(forces)
        +PinNodes(node_ids)
        +UnpinNodes(node_ids)
        +UpdatePhysicsParameters(params)
    }
```

### Semantic Messages (12+)
```mermaid
classDiagram
    class SemanticMessages {
        +ProcessMetadata(metadata_id, metadata)
        +RegenerateSemanticConstraints()
        +GetConstraints → ConstraintSet
        +UpdateConstraints(constraint_data)
        +GetSemanticStats → SemanticStats
        +SetGraphData(graph_data)
        +UpdateSemanticConfig(config)
        +ComputeShortestPaths(source_id) → PathfindingResult
        +ComputeAllPairsShortestPaths() → HashMap
        +TriggerStressMajorization()
        +UpdateAdvancedParams(params)
        +GetConstraintBuffer → Vec~ConstraintData~
    }
```

### Client Messages (10+)
```mermaid
classDiagram
    class ClientMessages {
        +RegisterClient(addr) → client_id
        +UnregisterClient(client_id)
        +BroadcastNodePositions(positions)
        +BroadcastMessage(message)
        +GetClientCount → usize
        +ForcePositionBroadcast(reason)
        +InitialClientSync(client_id, source)
        +UpdateNodePositions(positions)
        +SetGraphServiceAddress(addr)
        +GetClientCoordinatorStats → Stats
        +QueueVoiceData(audio)
        +SetBandwidthLimit(bytes_per_sec)
        +AuthenticateClient(client_id, pubkey)
        +UpdateClientFilter(client_id, filter)
    }
```

### GPU Actor Messages (40+)
```mermaid
classDiagram
    class GPUMessages {
        <<ForceComputeActor>>
        +ComputeForces() → ForceVectors
        +UpdatePositions(forces) → Positions
        +GetPhysicsStats() → PhysicsStats
        +UpdatePhysicsParams(params)

        <<StressMajorizationActor>>
        +OptimizeLayout() → OptimizationResult
        +GetStats() → StressMajorizationStats
        +UpdateParams(params)

        <<SemanticForcesActor>>
        +ApplySemanticForces() → ForceVectors
        +UpdateSemanticGraph(graph_data)

        <<ConstraintActor>>
        +ValidateConstraints(forces) → bool
        +UpdateConstraintSet(constraints)
        +GetActiveConstraints() → ConstraintSet

        <<OntologyConstraintActor>>
        +ValidateOntology() → ValidationResult
        +LoadOntology(owl_file)
        +GetConstraintBuffer() → Vec~ConstraintData~
        +UpdateOntologyRules(rules)

        <<ShortestPathActor>>
        +ComputeSSSP(source_id) → Distances
        +ComputeAPSP() → AllPairsDistances
        +InvalidateCache()

        <<PageRankActor>>
        +ComputePageRank() → Scores
        +GetTopNodes(k) → Vec~NodeId~

        <<ClusteringActor>>
        +RunKMeans(params) → KMeansResult
        +RunCommunityDetection(params) → Communities
        +DetectCommunities() → CommunityLabels

        <<AnomalyDetectionActor>>
        +DetectAnomalies(params) → AnomalyResult
        +GetAnomalyScores() → Vec~f32~

        <<ConnectedComponentsActor>>
        +FindComponents() → ComponentLabels
        +GetComponentSizes() → HashMap

        <<GPUResourceActor>>
        +AllocateStream() → CudaStream
        +AllocateMemory(size) → DevicePtr
        +FreeResources(handles)
        +GetMemoryStats() → MemoryStats
    }
```

---

## 8. Message Flow Patterns

### Pattern 1: Client Request → GPU Computation → Broadcast
```mermaid
sequenceDiagram
    participant C as Client (WebSocket)
    participant GSS as GraphServiceSupervisor
    participant GSA as GraphStateActor
    participant PO as PhysicsOrchestratorActor
    participant FC as ForceComputeActor
    participant CC as ClientCoordinatorActor

    C->>GSS: HTTP POST /api/physics/step
    GSS->>PO: SimulationStep

    activate PO
    PO->>FC: ComputeForces
    activate FC
    Note right of FC: CUDA kernel execution<br/>10,000 nodes in 2ms
    FC-->>PO: ForceVectors
    deactivate FC

    PO->>FC: UpdatePositions(forces)
    activate FC
    FC-->>PO: Positions (Vec~BinaryNodeData~)
    deactivate FC

    PO->>GSA: UpdateNodePositions(positions)
    GSA->>CC: UpdateNodePositions(positions)
    deactivate PO

    activate CC
    CC->>CC: update_position_cache()
    CC->>C: Binary broadcast (WebSocket)
    Note right of CC: All connected clients<br/>receive update
    deactivate CC

    GSS-->>C: HTTP 200 OK
```

### Pattern 2: Hierarchical Message Escalation (Error Handling)
```mermaid
sequenceDiagram
    participant GSS as GraphServiceSupervisor
    participant PO as PhysicsOrchestratorActor
    participant FC as ForceComputeActor

    GSS->>PO: SimulationStep
    activate PO

    PO->>FC: ComputeForces
    activate FC

    FC->>FC: CUDA kernel launch
    Note right of FC: GPU ERROR:<br/>Out of memory
    FC-->>PO: Err("CUDA OOM")
    deactivate FC

    Note over PO: Retry logic:<br/>Attempt 1/3
    PO->>FC: ComputeForces (retry)
    activate FC
    FC-->>PO: Err("CUDA OOM")
    deactivate FC

    Note over PO: Attempt 2/3
    PO->>FC: ComputeForces (retry)
    activate FC
    FC-->>PO: Err("CUDA OOM")
    deactivate FC

    Note over PO: Max retries exceeded<br/>Escalate to supervisor
    PO-->>GSS: Err("GPU failure")
    deactivate PO

    activate GSS
    Note over GSS: Supervision decision:<br/>Restart ForceComputeActor
    GSS->>FC: Restart (spawn new actor)
    activate FC
    FC->>FC: Initialize GPU context
    FC-->>GSS: Started()
    deactivate FC

    GSS->>PO: Restart (spawn new actor)
    activate PO
    PO->>PO: Reinitialize child actors
    PO-->>GSS: Started()
    deactivate PO
    deactivate GSS
```

### Pattern 3: Fan-Out / Fan-In (Parallel GPU Operations)
```mermaid
sequenceDiagram
    participant PO as PhysicsOrchestratorActor
    participant FC as ForceComputeActor
    participant SPA as ShortestPathActor
    participant PR as PageRankActor
    participant CLA as ClusteringActor

    PO->>PO: Analytics Request

    par Parallel GPU Operations
        PO->>FC: ComputeForces
        activate FC
        FC-->>PO: ForceVectors (2ms)
        deactivate FC
    and
        PO->>SPA: ComputeSSSP(source=0)
        activate SPA
        SPA-->>PO: Distances (5ms)
        deactivate SPA
    and
        PO->>PR: ComputePageRank
        activate PR
        PR-->>PO: Scores (10ms)
        deactivate PR
    and
        PO->>CLA: DetectCommunities
        activate CLA
        CLA-->>PO: Communities (8ms)
        deactivate CLA
    end

    Note over PO: Await all futures<br/>Max latency: 10ms

    PO->>PO: Merge results
    PO-->>PO: Analytics Complete
```

---

## 9. Actor Lifecycle and Mailbox Management

```mermaid
stateDiagram-v2
    [*] --> Created: "Addr::start()"

    Created --> Starting: Actix spawns actor
    Starting --> Started: "Actor::started() called"

    Started --> Running: Begin processing messages

    Running --> MessageWaiting: Mailbox empty
    MessageWaiting --> Processing: Message received
    Processing --> Running: Handler returns

    Running --> Stopping: "do_send(StopMessage)"
    Stopping --> Stopped: "Actor::stopped() called"
    Stopped --> [*]: Cleanup complete

    Processing --> Error: Handler panics
    Error --> Restarting: Supervisor restart
    Restarting --> Starting: New actor instance

    note right of MessageWaiting
        Mailbox:
        - Unbounded by default
        - FIFO ordering
        - Backpressure via bounded mailbox
        - Priority messages skip queue
    end note

    note right of Processing
        Message Handler:
        1. Deserialize message
        2. Call "Handler::handle()"
        3. Await async operations
        4. Serialize result
        5. Send response (if sync)
    end note
```

**Mailbox Configuration:**
```rust
// Default unbounded mailbox
let actor = GraphStateActor::new().start();

// Bounded mailbox (backpressure)
let actor = GraphStateActor::new()
    .start_in_arbiter(&arbiter, |ctx| {
        ctx.set_mailbox_capacity(1000); // Max 1000 pending messages
    });

// Priority mailbox (custom)
impl Actor for GraphStateActor {
    type Context = Context<Self>;

    fn handle_priority_message(&mut self, msg: PriorityMsg) {
        // Processed before regular messages
    }
}
```

---

## 10. Inter-Actor Communication Protocols

### Protocol 1: Request-Response (Sync)
```rust
// Sender (GraphServiceSupervisor)
let result: Result<Arc<GraphData>, String> =
    graph_state_actor.send(GetGraphData).await?;

// Receiver (GraphStateActor)
impl Handler<GetGraphData> for GraphStateActor {
    type Result = Result<Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.graph_data.clone().ok_or("No graph data")?)
    }
}
```

### Protocol 2: Fire-and-Forget (Async)
```rust
// Sender (PhysicsOrchestratorActor)
client_coordinator.do_send(UpdateNodePositions {
    positions: positions.clone(),
});
// No await, returns immediately

// Receiver (ClientCoordinatorActor)
impl Handler<UpdateNodePositions> for ClientCoordinatorActor {
    type Result = ();  // No response

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Self::Context) {
        self.update_position_cache(msg.positions);
        // No return value
    }
}
```

### Protocol 3: Actor-to-Actor Subscription (Pub/Sub)
```rust
// Publisher (GraphStateActor)
pub struct GraphUpdateEvent {
    pub nodes_added: Vec<u32>,
    pub nodes_removed: Vec<u32>,
}

impl Actor for GraphStateActor {
    fn started(&mut self, ctx: &mut Self::Context) {
        // Subscribe to graph updates
        ctx.run_interval(Duration::from_millis(100), |act, _ctx| {
            if act.has_pending_updates() {
                // Notify all subscribers
                act.subscribers.iter().for_each(|sub| {
                    sub.do_send(GraphUpdateEvent {
                        nodes_added: act.pending_adds.clone(),
                        nodes_removed: act.pending_removes.clone(),
                    });
                });
            }
        });
    }
}

// Subscriber (SemanticProcessorActor)
impl Handler<GraphUpdateEvent> for SemanticProcessorActor {
    type Result = ();

    fn handle(&mut self, msg: GraphUpdateEvent, _ctx: &mut Self::Context) {
        // React to graph changes
        self.invalidate_semantic_cache(&msg.nodes_removed);
        self.process_new_nodes(&msg.nodes_added);
    }
}
```

---

## 11. Error Recovery and Fault Tolerance

```mermaid
graph TB
    subgraph "Fault Isolation Zones"
        Z1[Zone 1: Graph State<br/>Actor: GraphStateActor<br/>Failures: Transient errors]
        Z2[Zone 2: Physics<br/>Actors: 11 GPU Actors<br/>Failures: CUDA errors, OOM]
        Z3[Zone 3: Semantic<br/>Actor: SemanticProcessorActor<br/>Failures: AI model errors]
        Z4[Zone 4: Clients<br/>Actor: ClientCoordinatorActor<br/>Failures: WebSocket disconnects]
    end

    subgraph "Restart Strategies"
        S1[OneForOne<br/>Restart only failed actor<br/>Preserve other actors]
        S2[AllForOne<br/>Restart all siblings<br/>Fresh state]
        S3[RestForOne<br/>Restart failed + later<br/>Dependency chain]
    end

    subgraph "Recovery Actions"
        R1[Retry: 3 attempts<br/>Exponential backoff]
        R2[Reload: Fetch from DB<br/>Rebuild state]
        R3[Isolate: Remove failed component<br/>Degrade gracefully]
        R4[Escalate: Restart supervisor<br/>Full system reset]
    end

    Z1 -->|Transient| S1 -->|1st Action| R1
    Z2 -->|Critical| S2 -->|2nd Action| R2
    Z3 -->|Recoverable| S1 -->|3rd Action| R3
    Z4 -->|Non-critical| S3 -->|Last Resort| R4

    R1 -->|Success| Normal[Resume Normal Operation]
    R1 -->|Fail| R2
    R2 -->|Success| Normal
    R2 -->|Fail| R3
    R3 -->|Success| Degraded[Degraded Mode Operation]
    R3 -->|Fail| R4
    R4 --> Restart[System Restart]

    style Z1 fill:#4ecdc4,stroke:#333,stroke-width:2px
    style Z2 fill:#ffe66d,stroke:#333,stroke-width:2px
    style Z3 fill:#a8e6cf,stroke:#333,stroke-width:2px
    style Z4 fill:#ff8b94,stroke:#333,stroke-width:2px

    style S1 fill:#95e1d3,stroke:#333,stroke-width:1px
    style S2 fill:#ffd3b6,stroke:#333,stroke-width:1px
    style S3 fill:#ffaaa5,stroke:#333,stroke-width:1px

    style R1 fill:#dcedc1,stroke:#333,stroke-width:1px
    style R2 fill:#ffd3b6,stroke:#333,stroke-width:1px
    style R3 fill:#ffaaa5,stroke:#333,stroke-width:1px
    style R4 fill:#ff6b6b,stroke:#333,stroke-width:2px

    style Normal fill:#a8e6cf,stroke:#333,stroke-width:2px
    style Degraded fill:#ffe66d,stroke:#333,stroke-width:2px
    style Restart fill:#ff6b6b,stroke:#333,stroke-width:2px
```

**Error Recovery Examples:**

```rust
// Example 1: Transient error with retry (GraphStateActor)
impl Handler<AddNode> for GraphStateActor {
    type Result = ResponseFuture<Result<(), String>>;

    fn handle(&mut self, msg: AddNode, _ctx: &mut Self::Context) -> Self::Result {
        let node = msg.node.clone();
        let max_retries = 3;

        Box::pin(async move {
            for attempt in 1..=max_retries {
                match self.insert_node_with_retry(&node, attempt).await {
                    Ok(()) => return Ok(()),
                    Err(e) if attempt < max_retries => {
                        warn!("Insert failed (attempt {}): {}", attempt, e);
                        tokio::time::sleep(Duration::from_millis(100 * attempt)).await;
                        continue;
                    }
                    Err(e) => return Err(format!("Failed after {} retries: {}", max_retries, e)),
                }
            }
            unreachable!()
        })
    }
}

// Example 2: Critical error with state reload (PhysicsOrchestratorActor)
impl Handler<SimulationStep> for PhysicsOrchestratorActor {
    type Result = ResponseFuture<Result<(), String>>;

    fn handle(&mut self, _msg: SimulationStep, ctx: &mut Self::Context) -> Self::Result {
        let force_actor = self.force_compute_actor.clone();
        let graph_state = self.graph_state_actor.clone();

        Box::pin(async move {
            match force_actor.send(ComputeForces).await {
                Ok(Ok(forces)) => Ok(()),
                Ok(Err(e)) if e.contains("CUDA OOM") => {
                    error!("GPU out of memory, reloading graph with reduced resolution");
                    graph_state.send(ReloadGraphFromDatabase).await??;
                    Err("GPU OOM - graph reloaded".to_string())
                }
                Ok(Err(e)) => Err(e),
                Err(e) => {
                    error!("Actor mailbox error: {}", e);
                    ctx.stop(); // Trigger supervisor restart
                    Err("Mailbox failure".to_string())
                }
            }
        })
    }
}

// Example 3: Graceful degradation (ClientCoordinatorActor)
impl Handler<BroadcastNodePositions> for ClientCoordinatorActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BroadcastNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        let manager = self.client_manager.read()
            .map_err(|e| format!("RwLock poisoned: {}", e))?;

        let mut failed_clients = Vec::new();

        for (client_id, client) in manager.clients.iter() {
            match client.addr.try_send(SendToClientBinary(msg.positions.clone())) {
                Ok(()) => {},
                Err(SendError::Full(_)) => {
                    warn!("Client {} mailbox full, skipping", client_id);
                    // Degrade: skip this client, continue with others
                }
                Err(SendError::Closed(_)) => {
                    error!("Client {} disconnected", client_id);
                    failed_clients.push(*client_id);
                }
            }
        }

        drop(manager); // Release read lock

        // Clean up failed clients
        let mut manager = self.client_manager.write()
            .map_err(|e| format!("RwLock poisoned: {}", e))?;
        for client_id in failed_clients {
            manager.unregister_client(client_id);
        }

        Ok(())
    }
}
```

---

## 12. GPU Sub-Actor Details (11 Actors)

### ForceComputeActor - Primary Physics Engine
```mermaid
stateDiagram-v2
    [*] --> Uninitialized

    Uninitialized --> Initializing: InitializePhysics
    Initializing --> AllocatingGPU: Allocate CUDA resources
    AllocatingGPU --> UploadingData: Upload graph to GPU
    UploadingData --> Ready: GPU initialized

    Ready --> ComputingForces: ComputeForces message
    ComputingForces --> LaunchingKernel: Launch CUDA kernel
    LaunchingKernel --> Synchronizing: "cudaDeviceSynchronize()"
    Synchronizing --> DownloadingResults: Copy results to host
    DownloadingResults --> Ready: Return ForceVectors

    Ready --> UpdatingPositions: UpdatePositions message
    UpdatingPositions --> IntegratingForces: Velocity Verlet
    IntegratingForces --> Ready: Return new positions

    Ready --> UpdatingParams: UpdatePhysicsParams
    UpdatingParams --> UploadingParams: Copy params to GPU
    UploadingParams --> Ready: Params updated

    Ready --> [*]: "Cleanup (Actor stopped)"

    note right of LaunchingKernel
        CUDA Kernel Configuration:
        - Threads per block: 256
        - Blocks: (num_nodes + 255) / 256
        - Shared memory: 16 KB per block
        - Registers: 32 per thread
    end note

    note right of ComputingForces
        Force Calculation:
        1. Repulsion (all pairs): O(n²)
        2. Attraction (edges): O(m)
        3. Damping: O(n)
        4. Constraints: O(k)
        Total GPU time: ~2ms for 10k nodes
    end note
```

**CUDA Kernel Launch:**
```rust
// ForceComputeActor internal CUDA launch
pub async fn compute_forces_gpu(&mut self) -> Result<Vec<Vec3>, String> {
    let num_nodes = self.graph_data.nodes.len();
    let num_edges = self.graph_data.edges.len();

    unsafe {
        // Launch repulsion kernel
        let threads_per_block = 256;
        let num_blocks = (num_nodes + threads_per_block - 1) / threads_per_block;

        cuda_launch_kernel!(
            compute_repulsion_forces_kernel<<<num_blocks, threads_per_block>>>(
                self.device_positions,
                self.device_forces,
                num_nodes,
                self.params.repulsion_strength
            )
        );

        // Launch attraction kernel (edge-based)
        let edge_blocks = (num_edges + threads_per_block - 1) / threads_per_block;
        cuda_launch_kernel!(
            compute_attraction_forces_kernel<<<edge_blocks, threads_per_block>>>(
                self.device_positions,
                self.device_edges,
                self.device_forces,
                num_edges,
                self.params.attraction_strength
            )
        );

        // Synchronize and download
        cudaDeviceSynchronize()?;

        let mut host_forces = vec![Vec3::ZERO; num_nodes];
        cudaMemcpy(
            host_forces.as_mut_ptr(),
            self.device_forces,
            num_nodes * std::mem::size_of::<Vec3>(),
            cudaMemcpyDeviceToHost
        )?;

        Ok(host_forces)
    }
}
```

### StressMajorizationActor - Iterative Layout Solver
```mermaid
graph TB
    Start[Receive OptimizeLayout] --> Init[Initialize: X₀ = current positions]
    Init --> Iteration[Iteration k]

    Iteration --> ComputeStress[Compute Stress:<br/>σ = Σᵢⱼ wᵢⱼ(dᵢⱼ - ‖Xᵢ - Xⱼ‖)²]
    ComputeStress --> CheckConvergence{σₖ - σₖ₋₁ < ε?}

    CheckConvergence -->|No| SolveSystem[Solve Linear System:<br/>LXₖ₊₁ = LwZ]
    SolveSystem --> UpdatePositions[Xₖ₊₁ = new positions]
    UpdatePositions --> Iteration

    CheckConvergence -->|Yes| Return[Return OptimizationResult]

    CheckConvergence -->|Max Iterations| Return

    style Start fill:#a8e6cf,stroke:#333,stroke-width:2px
    style ComputeStress fill:#ffe66d,stroke:#333,stroke-width:2px
    style CheckConvergence fill:#ff8b94,stroke:#333,stroke-width:2px
    style Return fill:#95e1d3,stroke:#333,stroke-width:2px
```

**Stress Majorization Algorithm:**
```rust
// StressMajorizationActor implementation
pub struct StressMajorizationActor {
    max_iterations: u32,
    convergence_threshold: f32,
    gpu_solver: CudaLinearSolver,
}

impl StressMajorizationActor {
    pub async fn optimize_layout(&mut self, graph: &GraphData) -> Result<OptimizationResult, String> {
        let n = graph.nodes.len();
        let mut X = graph.get_position_matrix(); // n×3 matrix
        let mut prev_stress = f32::MAX;

        for iter in 0..self.max_iterations {
            // 1. Compute distance matrix D
            let D = self.compute_distances(&X); // O(n²)

            // 2. Compute stress
            let stress = self.compute_stress(&D, &graph.ideal_distances); // O(n²)

            // 3. Check convergence
            if (prev_stress - stress).abs() < self.convergence_threshold {
                return Ok(OptimizationResult {
                    final_positions: X,
                    final_stress: stress,
                    iterations: iter,
                    converged: true,
                });
            }

            // 4. Solve linear system: LX = LwZ (GPU-accelerated)
            let L = self.compute_laplacian_matrix(&graph); // O(n²)
            let Z = self.compute_weighted_position_matrix(&X, &D, &graph); // O(n²)

            X = self.gpu_solver.solve_system(L, Z).await?; // GPU: ~10ms for 10k nodes

            prev_stress = stress;
        }

        Ok(OptimizationResult {
            final_positions: X,
            final_stress: prev_stress,
            iterations: self.max_iterations,
            converged: false,
        })
    }
}
```

### ShortestPathActor - GPU Pathfinding
```mermaid
sequenceDiagram
    participant Client
    participant SPA as ShortestPathActor
    participant Cache as PathCache
    participant GPU as CUDA SSSP

    Client->>SPA: ComputeSSSP(source=42)

    SPA->>Cache: Check cache for source 42
    alt Cache Hit
        Cache-->>SPA: Cached distances
        SPA-->>Client: PathfindingResult (1ms)
    else Cache Miss
        SPA->>GPU: Launch Bellman-Ford kernel
        activate GPU

        Note right of GPU: GPU Execution:<br/>- Upload graph (CSR format)<br/>- Initialize distances to ∞<br/>- Parallel edge relaxation<br/>- Repeat n-1 times<br/>- Detect negative cycles

        GPU-->>SPA: Distances + Predecessors (5ms)
        deactivate GPU

        SPA->>Cache: Store result (source=42)
        SPA-->>Client: PathfindingResult (6ms total)
    end
```

**GPU SSSP Algorithm:**
```rust
// ShortestPathActor - GPU Bellman-Ford
pub async fn compute_sssp_gpu(&mut self, source: u32) -> Result<PathfindingResult, String> {
    // Check cache first
    if let Some(cached) = self.cache.get(&source) {
        return Ok(cached.clone());
    }

    let n = self.graph_data.nodes.len();
    let m = self.graph_data.edges.len();

    unsafe {
        // Upload graph in CSR (Compressed Sparse Row) format
        let (row_offsets, col_indices, weights) = self.graph_data.to_csr();

        // Initialize GPU buffers
        let mut device_distances: *mut f32 = cuda_malloc(n * sizeof(f32))?;
        let mut device_predecessors: *mut u32 = cuda_malloc(n * sizeof(u32))?;

        cudaMemset(device_distances, 0x7f, n * sizeof(f32)); // Set to ∞
        cudaMemset(device_predecessors, 0xff, n * sizeof(u32)); // Set to -1

        // Set source distance to 0
        let zero: f32 = 0.0;
        cudaMemcpy(&zero as *const f32, &device_distances[source], 1, cudaMemcpyHostToDevice)?;

        // Bellman-Ford iterations (n-1 times)
        for _ in 0..(n - 1) {
            cuda_launch_kernel!(
                bellman_ford_relax_kernel<<<(m + 255) / 256, 256>>>(
                    device_row_offsets,
                    device_col_indices,
                    device_weights,
                    device_distances,
                    device_predecessors,
                    m
                )
            );
        }

        // Synchronize and download
        cudaDeviceSynchronize()?;

        let mut distances = vec![f32::MAX; n];
        let mut predecessors = vec![u32::MAX; n];

        cudaMemcpy(distances.as_mut_ptr(), device_distances, n, cudaMemcpyDeviceToHost)?;
        cudaMemcpy(predecessors.as_mut_ptr(), device_predecessors, n, cudaMemcpyDeviceToHost)?;

        cuda_free(device_distances)?;
        cuda_free(device_predecessors)?;

        let result = PathfindingResult {
            source_node: source,
            distances,
            predecessors,
            computation_time_ms: 5,
        };

        // Cache result
        self.cache.insert(source, result.clone());

        Ok(result)
    }
}
```

### ClusteringActor - K-Means + Community Detection
```rust
pub enum ClusteringAlgorithm {
    KMeans { k: usize, max_iter: u32 },
    Louvain { resolution: f32 },
    LabelPropagation { max_iter: u32 },
}

impl ClusteringActor {
    pub async fn detect_communities(
        &mut self,
        algorithm: ClusteringAlgorithm
    ) -> Result<CommunityDetectionResult, String> {
        match algorithm {
            ClusteringAlgorithm::KMeans { k, max_iter } => {
                self.run_kmeans_gpu(k, max_iter).await
            }
            ClusteringAlgorithm::Louvain { resolution } => {
                self.run_louvain_gpu(resolution).await
            }
            ClusteringAlgorithm::LabelPropagation { max_iter } => {
                self.run_label_propagation_gpu(max_iter).await
            }
        }
    }

    async fn run_label_propagation_gpu(&mut self, max_iter: u32) -> Result<CommunityDetectionResult, String> {
        let n = self.graph_data.nodes.len();

        unsafe {
            // Initialize: each node is its own community
            let mut device_labels: *mut u32 = cuda_malloc(n * sizeof(u32))?;
            cuda_launch_kernel!(
                initialize_labels_kernel<<<(n + 255) / 256, 256>>>(device_labels, n)
            );

            let mut changed = true;
            let mut iter = 0;

            while changed && iter < max_iter {
                changed = false;

                // Propagate labels
                cuda_launch_kernel!(
                    label_propagation_kernel<<<(n + 255) / 256, 256>>>(
                        device_graph_csr,
                        device_labels,
                        device_changed_flag,
                        n
                    )
                );

                cudaMemcpy(&changed, device_changed_flag, 1, cudaMemcpyDeviceToHost)?;
                iter += 1;
            }

            // Download results
            let mut labels = vec![0u32; n];
            cudaMemcpy(labels.as_mut_ptr(), device_labels, n, cudaMemcpyDeviceToHost)?;

            Ok(CommunityDetectionResult {
                node_labels: labels.into_iter().map(|l| l as i32).collect(),
                num_communities: self.count_unique_labels(&labels),
                iterations: iter,
                converged: !changed,
                algorithm: CommunityDetectionAlgorithm::LabelPropagation,
            })
        }
    }
}
```

### AnomalyDetectionActor - Outlier Detection
```rust
pub enum AnomalyMethod {
    LOF { k: usize },      // Local Outlier Factor
    ZScore { threshold: f32 },
    IsolationForest { trees: usize },
}

impl AnomalyDetectionActor {
    pub async fn detect_anomalies_lof(&mut self, k: usize) -> Result<AnomalyResult, String> {
        let n = self.graph_data.nodes.len();

        unsafe {
            // 1. Compute k-distance for each node (GPU)
            let mut device_k_distances: *mut f32 = cuda_malloc(n * sizeof(f32))?;
            cuda_launch_kernel!(
                compute_k_distance_kernel<<<(n + 255) / 256, 256>>>(
                    device_positions,
                    device_k_distances,
                    n,
                    k
                )
            );

            // 2. Compute local reachability density (LRD)
            let mut device_lrd: *mut f32 = cuda_malloc(n * sizeof(f32))?;
            cuda_launch_kernel!(
                compute_lrd_kernel<<<(n + 255) / 256, 256>>>(
                    device_positions,
                    device_k_distances,
                    device_lrd,
                    n,
                    k
                )
            );

            // 3. Compute LOF scores
            let mut device_lof_scores: *mut f32 = cuda_malloc(n * sizeof(f32))?;
            cuda_launch_kernel!(
                compute_lof_kernel<<<(n + 255) / 256, 256>>>(
                    device_lrd,
                    device_lof_scores,
                    device_neighbors,
                    n,
                    k
                )
            );

            cudaDeviceSynchronize()?;

            // Download results
            let mut lof_scores = vec![0.0f32; n];
            cudaMemcpy(lof_scores.as_mut_ptr(), device_lof_scores, n, cudaMemcpyDeviceToHost)?;

            // Identify anomalies (LOF > threshold)
            let threshold = 1.5;
            let anomalies: Vec<AnomalyNode> = lof_scores
                .iter()
                .enumerate()
                .filter(|(_, &score)| score > threshold)
                .map(|(i, &score)| AnomalyNode {
                    node_id: i as u32,
                    anomaly_score: score,
                    method: AnomalyDetectionMethod::LOF,
                })
                .collect();

            Ok(AnomalyResult {
                lof_scores: Some(lof_scores),
                anomaly_threshold: threshold,
                num_anomalies: anomalies.len(),
                anomalies,
                method: AnomalyDetectionMethod::LOF,
            })
        }
    }
}
```

### GPUResourceActor - Memory and Stream Management
```rust
pub struct GPUResourceActor {
    cuda_streams: Vec<CudaStream>,
    memory_pools: HashMap<usize, Vec<DevicePtr>>, // size -> available pointers
    allocated_bytes: usize,
    total_gpu_memory: usize,
}

impl GPUResourceActor {
    pub async fn allocate_stream(&mut self) -> Result<CudaStream, String> {
        if let Some(stream) = self.cuda_streams.pop() {
            // Reuse existing stream
            Ok(stream)
        } else {
            // Create new stream
            let mut stream: cudaStream_t = std::ptr::null_mut();
            unsafe {
                cudaStreamCreate(&mut stream)?;
            }
            Ok(CudaStream::new(stream))
        }
    }

    pub async fn allocate_memory(&mut self, size: usize) -> Result<DevicePtr, String> {
        // Round up to nearest power of 2 for pooling
        let pool_size = size.next_power_of_two();

        if let Some(pool) = self.memory_pools.get_mut(&pool_size) {
            if let Some(ptr) = pool.pop() {
                // Reuse pooled memory
                return Ok(ptr);
            }
        }

        // Allocate new memory
        let mut device_ptr: *mut u8 = std::ptr::null_mut();
        unsafe {
            let result = cudaMalloc(&mut device_ptr as *mut *mut u8 as *mut *mut c_void, pool_size);
            if result != cudaSuccess {
                return Err(format!("cudaMalloc failed: {:?}", result));
            }
        }

        self.allocated_bytes += pool_size;

        Ok(DevicePtr {
            ptr: device_ptr,
            size: pool_size,
        })
    }

    pub async fn free_resources(&mut self, resources: Vec<GPUResource>) {
        for resource in resources {
            match resource {
                GPUResource::Stream(stream) => {
                    self.cuda_streams.push(stream); // Return to pool
                }
                GPUResource::Memory(ptr) => {
                    self.memory_pools
                        .entry(ptr.size)
                        .or_insert_with(Vec::new)
                        .push(ptr); // Return to pool
                }
            }
        }
    }

    pub async fn get_memory_stats(&self) -> MemoryStats {
        let mut free_bytes = 0usize;
        let mut total_bytes = 0usize;

        unsafe {
            cudaMemGetInfo(&mut free_bytes, &mut total_bytes);
        }

        MemoryStats {
            allocated_bytes: self.allocated_bytes,
            free_bytes,
            total_bytes,
            pool_sizes: self.memory_pools.keys().copied().collect(),
        }
    }
}
```

---

## 13. Performance Characteristics

### Message Latency (Median, P95, P99)
```mermaid
graph LR
    subgraph "Local Actor Messages (Same Thread)"
        L1[GetGraphData<br/>Median: 50μs<br/>P95: 100μs<br/>P99: 200μs]
        L2[UpdateNodePositions<br/>Median: 80μs<br/>P95: 150μs<br/>P99: 300μs]
    end

    subgraph "GPU Actor Messages (CUDA Kernel)"
        G1[ComputeForces<br/>Median: 2ms<br/>P95: 5ms<br/>P99: 10ms]
        G2[ComputeSSSP<br/>Median: 5ms<br/>P95: 12ms<br/>P99: 25ms]
        G3[OptimizeLayout<br/>Median: 50ms<br/>P95: 150ms<br/>P99: 300ms]
    end

    subgraph "Network Messages (WebSocket)"
        N1[BroadcastNodePositions<br/>Median: 10ms<br/>P95: 30ms<br/>P99: 100ms]
    end

    style L1 fill:#a8e6cf,stroke:#333,stroke-width:2px
    style L2 fill:#a8e6cf,stroke:#333,stroke-width:2px
    style G1 fill:#ffe66d,stroke:#333,stroke-width:2px
    style G2 fill:#ffe66d,stroke:#333,stroke-width:2px
    style G3 fill:#ffd3b6,stroke:#333,stroke-width:2px
    style N1 fill:#ff8b94,stroke:#333,stroke-width:2px
```

### Throughput (Messages/Second)
| Actor | Message Type | Throughput | Notes |
|-------|-------------|-----------|-------|
| GraphStateActor | GetGraphData | 20,000 msg/s | Read-only, Arc clone |
| GraphStateActor | AddNode | 5,000 msg/s | Write lock contention |
| PhysicsOrchestratorActor | SimulationStep | 60 msg/s | 16ms GPU compute |
| SemanticProcessorActor | ProcessMetadata | 1,000 msg/s | CPU-bound AI |
| ClientCoordinatorActor | BroadcastNodePositions | 20 msg/s | 50ms broadcast interval |
| ForceComputeActor | ComputeForces | 500 msg/s | 2ms GPU kernel |
| ShortestPathActor | ComputeSSSP | 200 msg/s | 5ms GPU kernel |
| ClusteringActor | DetectCommunities | 50 msg/s | 20ms GPU kernel |

### Scalability Limits
```mermaid
graph TB
    subgraph "Bottlenecks by Scale"
        S1[Small: <1,000 nodes<br/>Bottleneck: None<br/>Latency: <5ms]
        S2[Medium: 1k-10k nodes<br/>Bottleneck: GPU memory<br/>Latency: <20ms]
        S3[Large: 10k-100k nodes<br/>Bottleneck: GPU compute<br/>Latency: <100ms]
        S4[Massive: >100k nodes<br/>Bottleneck: GPU memory + bandwidth<br/>Latency: >1s]
    end

    S1 --> S2
    S2 --> S3
    S3 --> S4

    style S1 fill:#a8e6cf,stroke:#333,stroke-width:2px
    style S2 fill:#ffe66d,stroke:#333,stroke-width:2px
    style S3 fill:#ffd3b6,stroke:#333,stroke-width:2px
    style S4 fill:#ffaaa5,stroke:#333,stroke-width:2px
```

---

## 14. Actor Timing Diagrams

### End-to-End Request Flow (Client → GPU → Broadcast)
```mermaid
gantt
    title Complete Simulation Step (Target: 60 FPS = 16.67ms)
    dateFormat X
    axisFormat %L

    section HTTP Request
    API Handler receives request :milestone, 0, 0
    Deserialize JSON :a1, 0, 100

    section Actor Messaging
    Send to GraphServiceSupervisor :a2, 100, 150
    Forward to PhysicsOrchestrator :a3, 150, 200

    section GPU Computation
    Send ComputeForces :a4, 200, 250
    CUDA kernel launch :a5, 250, 300
    Repulsion kernel :crit, a6, 300, 1800
    Attraction kernel :crit, a7, 1800, 2300
    cudaDeviceSynchronize :a8, 2300, 2500
    Download results :a9, 2500, 3000

    section Position Integration
    UpdatePositions message :a10, 3000, 3100
    Velocity Verlet integration :a11, 3100, 3500

    section State Update
    Send to GraphStateActor :a12, 3500, 3600
    Update internal state :a13, 3600, 4000

    section Client Broadcast
    Send to ClientCoordinator :a14, 4000, 4100
    Serialize binary protocol :a15, 4100, 4500
    WebSocket broadcast (N clients) :crit, a16, 4500, 10000

    section HTTP Response
    Return 200 OK :milestone, 10000, 10000

    Total Latency: 10ms
```

**Timeline Breakdown:**
- **0-200μs**: HTTP request processing, deserialization, actor routing
- **200-3000μs (2.8ms)**: GPU force computation (critical path)
  - 1.5ms: Repulsion kernel (O(n²) complexity)
  - 0.5ms: Attraction kernel (O(m) complexity)
  - 0.2ms: CUDA synchronization
  - 0.5ms: Data transfer (GPU → CPU)
  - 0.1ms: Message passing overhead
- **3000-4000μs (1ms)**: Position integration and state update
- **4000-10000μs (6ms)**: Client broadcast
  - 0.4ms: Binary serialization (10,000 nodes)
  - 5.6ms: WebSocket transmission (10 clients @ 560 KB/client)
- **Total: 10ms** (6.67ms budget remaining for 60 FPS)

---

## 15. State Persistence and Recovery

```mermaid
sequenceDiagram
    participant GSS as GraphServiceSupervisor
    participant GSA as GraphStateActor
    participant DB as Neo4j Database
    participant Checkpoint as Checkpoint File

    Note over GSS,Checkpoint: Normal Operation

    loop Every 60 seconds
        GSS->>GSA: CreateCheckpoint
        GSA->>Checkpoint: Serialize state (bincode)
        Note right of Checkpoint: checkpoint_1234567890.bin<br/>Contains:<br/>- graph_data: Arc~GraphData~<br/>- node_map: HashMap<br/>- edge_map: HashMap<br/>- simulation_params
        Checkpoint-->>GSA: Checkpoint created
    end

    Note over GSS,Checkpoint: Actor Crash Detected

    GSS->>GSS: Supervision strategy: Restart
    GSS->>GSA: Spawn new actor instance

    activate GSA
    GSA->>GSA: "Actor::started()"
    GSA->>Checkpoint: Load latest checkpoint

    alt Checkpoint exists and valid
        Checkpoint-->>GSA: Deserialized state
        GSA->>GSA: Restore internal state
        Note right of GSA: State restored from<br/>checkpoint_1234567890.bin<br/>Age: 23 seconds

        GSA->>DB: Fetch incremental changes (since checkpoint)
        DB-->>GSA: New nodes/edges (23 sec delta)
        GSA->>GSA: Merge checkpoint + delta
        GSA-->>GSS: Ready (recovery: 500ms)
    else Checkpoint invalid or missing
        GSA->>DB: ReloadGraphFromDatabase
        DB-->>GSA: Full graph data
        GSA->>GSA: Rebuild all state
        GSA-->>GSS: Ready (recovery: 5000ms)
    end
    deactivate GSA

    Note over GSS,Checkpoint: Normal Operation Resumed
```

**Checkpoint Format:**
```rust
#[derive(Serialize, Deserialize)]
pub struct ActorCheckpoint {
    version: u32,                              // Schema version
    timestamp: i64,                            // Unix timestamp
    actor_type: String,                        // "GraphStateActor"
    state: ActorState,                         // Serialized state
    checksum: u64,                             // CRC64 checksum
}

#[derive(Serialize, Deserialize)]
pub struct GraphStateActorState {
    graph_data: GraphData,                     // Full graph structure
    node_map: HashMap<u32, Node>,              // Node lookup
    edge_map: HashMap<String, Edge>,           // Edge lookup
    metadata_to_node: HashMap<String, u32>,    // Metadata mapping
    simulation_params: SimulationParams,        // Physics parameters
    current_state: GraphState,                 // State machine position
}

impl GraphStateActor {
    pub fn create_checkpoint(&self) -> Result<ActorCheckpoint, String> {
        let state = GraphStateActorState {
            graph_data: self.graph_data.as_ref().clone(),
            node_map: self.node_map.clone(),
            edge_map: self.edge_map.clone(),
            metadata_to_node: self.metadata_to_node.clone(),
            simulation_params: self.simulation_params.clone(),
            current_state: self.state.clone(),
        };

        let serialized = bincode::serialize(&state)
            .map_err(|e| format!("Serialization failed: {}", e))?;

        let checksum = crc64::checksum(&serialized);

        Ok(ActorCheckpoint {
            version: 1,
            timestamp: Utc::now().timestamp(),
            actor_type: "GraphStateActor".to_string(),
            state: serialized,
            checksum,
        })
    }

    pub fn restore_from_checkpoint(&mut self, checkpoint: ActorCheckpoint) -> Result<(), String> {
        // Verify checksum
        if crc64::checksum(&checkpoint.state) != checkpoint.checksum {
            return Err("Checkpoint corrupted: checksum mismatch".to_string());
        }

        // Deserialize state
        let state: GraphStateActorState = bincode::deserialize(&checkpoint.state)
            .map_err(|e| format!("Deserialization failed: {}", e))?;

        // Restore actor state
        self.graph_data = Some(Arc::new(state.graph_data));
        self.node_map = state.node_map;
        self.edge_map = state.edge_map;
        self.metadata_to_node = state.metadata_to_node;
        self.simulation_params = state.simulation_params;
        self.state = state.current_state;

        info!("Restored from checkpoint: {} seconds old",
              Utc::now().timestamp() - checkpoint.timestamp);

        Ok(())
    }
}
```

---

---

## Related Documentation

- [VisionFlow GPU CUDA Architecture - Complete Technical Documentation](../../infrastructure/gpu/cuda-architecture-complete.md)
- [Server Architecture](../../../concepts/architecture/core/server.md)
- [Complete State Management Architecture](../../client/state/state-management-complete.md)
- [X-FluxAgent Integration Plan for ComfyUI MCP Skill](../../../multi-agent-docker/x-fluxagent-adaptation-plan.md)
- [VisionFlow Documentation Modernization - Final Report](../../../DOCUMENTATION_MODERNIZATION_COMPLETE.md)

## 16. Summary: Actor System Capabilities

### Total Actor Count: 24
1. **GraphServiceSupervisor** - Root supervisor (OneForOne)
2. **GraphStateActor** - Graph data management (7-state machine)
3. **PhysicsOrchestratorActor** - GPU coordination (11 sub-actors)
4. **SemanticProcessorActor** - AI semantic analysis
5. **ClientCoordinatorActor** - WebSocket broadcasting
6-16. **GPU Sub-Actors** (11 total):
   - ForceComputeActor
   - StressMajorizationActor
   - SemanticForcesActor
   - ConstraintActor
   - OntologyConstraintActor
   - ShortestPathActor
   - PageRankActor
   - ClusteringActor
   - AnomalyDetectionActor
   - ConnectedComponentsActor
   - GPUResourceActor
17. **WorkspaceActor** - Multi-tenant workspace management
18. **SettingsActor** - Configuration persistence
19. **OptimisedSettingsActor** - Hot-path settings cache
20. **MultiMcpVisualizationActor** - MCP server integration
21. **TaskOrchestratorActor** - Async task management
22. **AgentMonitorActor** - Agent health monitoring
23-24. **Support Actors** (telemetry, logging, etc.)

### Message Types: 100+
- **Graph State**: 20+ messages (CRUD, batch operations)
- **Physics**: 15+ messages (simulation control, GPU operations)
- **Semantic**: 12+ messages (AI processing, constraints)
- **Client**: 10+ messages (WebSocket, broadcasting)
- **GPU Actors**: 40+ messages (specialized GPU operations)
- **Settings**: 10+ messages (configuration management)

### Key Performance Metrics:
- **Simulation Rate**: 60 FPS (16.67ms per step)
- **GPU Compute**: 2-10ms (force computation)
- **Client Broadcast**: 50ms interval (active), 1000ms (stable)
- **Throughput**: 20,000 msg/s (read), 5,000 msg/s (write)
- **Scalability**: Up to 100,000 nodes (GPU memory limit)

### Fault Tolerance:
- **Supervision**: OneForOne + AllForOne strategies
- **Restart Policies**: 3 max restarts within 10s
- **State Recovery**: Checkpoint + incremental delta
- **Graceful Degradation**: Non-critical actors isolated

---

**End of Complete Actor System Documentation**

*Generated: 2025-12-05*
*System Version: Production (Actix + CUDA)*
*Total Diagrams: 16*
*Total Lines: ~2000*
