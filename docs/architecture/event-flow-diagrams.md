# Event Flow Diagrams
**Detailed Event Flows for Hexagonal/CQRS Architecture**

---

## 1. GitHub Sync Event Flow (BUG FIX)

### Current Problem Flow

```mermaid
sequenceDiagram
    participant GitHub as GitHub API
    participant ContentAPI as EnhancedContentAPI
    participant SyncService as GitHubSyncService
    participant Repo as SqliteKnowledgeGraphRepository
    participant DB as knowledge_graph.db
    participant Actor as GraphServiceActor
    participant Client as API Client

    rect rgb(255, 200, 200)
        Note over GitHub,Client: ❌ CURRENT BROKEN FLOW - Cache Coherency Issue
    end

    GitHub->>ContentAPI: 1. Fetch markdown files
    ContentAPI-->>SyncService: 2. Return 316 files

    SyncService->>SyncService: 3. Parse files<br/>→ 316 nodes, 450 edges

    SyncService->>Repo: 4. INSERT INTO nodes (316 nodes)
    Repo->>DB: INSERT INTO edges (450 edges)

    DB-->>Repo: ✅ Database has 316 nodes

    Note over SyncService,Actor: ❌ NO EVENT EMITTED!<br/>Cache remains stale

    rect rgb(255, 200, 200)
        Note over Actor: GraphServiceActor state:<br/>In-memory cache: 63 nodes (STALE!)<br/>Never refreshed
    end

    Client->>Actor: GET /api/graph/data<br/>Sends GetGraphData message

    Actor->>Actor: Returns in-memory cache<br/>(63 nodes)

    Actor-->>Client: ❌ CLIENT SEES: 63 nodes (WRONG!)

    Note over Client: Expected: 316 nodes<br/>Actual: 63 nodes<br/>❌ Cache coherency bug
```

### Fixed Event-Driven Flow

```mermaid
sequenceDiagram
    participant GitHub as GitHub API
    participant ContentAPI as EnhancedContentAPI
    participant SyncService as GitHubSyncService
    participant Port as GraphRepository (Port)
    participant Adapter as SqliteGraphRepository (Adapter)
    participant DB as knowledge_graph.db
    participant EventBus as EventBus
    participant CacheInv as CacheInvalidation Subscriber
    participant WSub as WebSocket Subscriber
    participant LogSub as Logging Subscriber
    participant MetricsSub as Metrics Subscriber
    participant Client as API Client

    rect rgb(200, 255, 200)
        Note over GitHub,Client: ✅ NEW EVENT-DRIVEN FLOW - Cache Coherency Fixed
    end

    GitHub->>ContentAPI: 1. Fetch markdown files
    ContentAPI-->>SyncService: 2. Return 316 files

    SyncService->>SyncService: 3. Parse files<br/>→ 316 nodes, 450 edges

    SyncService->>Port: 4. save_graph(316 nodes, 450 edges)
    Port->>Adapter: Delegate to adapter
    Adapter->>DB: 5. INSERT INTO nodes/edges

    DB-->>Adapter: ✅ Database has 316 nodes
    Adapter-->>Port: Success
    Port-->>SyncService: Persisted

    rect rgb(200, 255, 200)
        SyncService->>EventBus: 6. ✅ Emit GitHubSyncCompletedEvent
        Note over SyncService,EventBus: Event-driven invalidation
    end

    par Event Distribution
        EventBus->>CacheInv: GitHubSyncCompletedEvent
        EventBus->>WSub: GitHubSyncCompletedEvent
        EventBus->>LogSub: GitHubSyncCompletedEvent
        EventBus->>MetricsSub: GitHubSyncCompletedEvent
    end

    par Subscriber Actions
        CacheInv->>CacheInv: Clear ALL caches<br/>(graph, nodes, edges, semantic)
        WSub->>Client: Broadcast "graphReloaded"
        LogSub->>LogSub: Log: "Sync completed: 316 nodes"
        MetricsSub->>MetricsSub: Track sync duration and stats
    end

    Note over Client: Client reloads graph

    Client->>Port: Next API call (GET /api/graph/data)
    Port->>Adapter: get_graph()
    Adapter->>DB: SELECT * FROM nodes
    DB-->>Adapter: 316 nodes
    Adapter-->>Port: Graph data
    Port-->>Client: ✅ Returns 316 nodes

    rect rgb(200, 255, 200)
        Note over Client: ✅ CLIENT SEES: 316 nodes (CORRECT!)
    end
```

---

## 2. Physics Simulation Event Flow

### Physics Step Execution

```mermaid
sequenceDiagram
    actor User
    participant Controller as PhysicsController
    participant Handler as TriggerPhysicsStepCommandHandler
    participant Repo as GraphRepository
    participant DB as knowledge_graph.db
    participant PhysicsService as PhysicsService (Domain)
    participant GPU as GPU Acceleration
    participant EventBus as EventBus
    participant WSub as WebSocket Subscriber
    participant CacheInv as Cache Invalidation
    participant Metrics as Metrics Subscriber
    participant EqCheck as Equilibrium Checker
    participant Client as WebSocket Client

    Note over User,Client: PHYSICS SIMULATION EVENT FLOW

    User->>Controller: POST /api/physics/start<br/>"Start Simulation"

    Controller->>Handler: Creates TriggerPhysicsStepCommand

    Handler->>Handler: 1. Validate simulation params

    Handler->>Repo: 2. Load graph from repository<br/>get_graph()
    Repo->>DB: SELECT * FROM nodes, edges
    DB-->>Repo: Returns 316 nodes, 450 edges
    Repo-->>Handler: Graph data

    Handler->>PhysicsService: Run simulation

    rect rgb(200, 220, 255)
        Note over PhysicsService,GPU: For each iteration (e.g., 100 iterations):

        loop Physics Iterations
            PhysicsService->>GPU: 3. Calculate forces (GPU)<br/>- Repulsion between nodes<br/>- Attraction along edges<br/>- Apply constraints
            GPU-->>PhysicsService: Force vectors

            PhysicsService->>PhysicsService: 4. Update velocities<br/>v = v + (F/m) * dt

            PhysicsService->>PhysicsService: 5. Update positions<br/>new_pos = old_pos + velocity * dt

            PhysicsService->>PhysicsService: 6. Check convergence<br/>if max_velocity < threshold:<br/>equilibrium = true
        end
    end

    PhysicsService->>Repo: Batch position updates<br/>Vec<(node_id, position)>
    Repo->>DB: UPDATE nodes SET x=?, y=?, z=? WHERE id=?
    DB-->>Repo: ✅ Positions saved

    PhysicsService->>EventBus: Emit PhysicsStepCompletedEvent

    par Event Distribution
        EventBus->>WSub: PhysicsStepCompletedEvent
        EventBus->>CacheInv: PhysicsStepCompletedEvent
        EventBus->>Metrics: PhysicsStepCompletedEvent
        EventBus->>EqCheck: PhysicsStepCompletedEvent
    end

    par Subscriber Actions
        WSub->>Client: Broadcast positions<br/>Every 16ms (60 FPS)
        CacheInv->>CacheInv: Clear position cache
        Metrics->>Metrics: Track FPS and perf
        EqCheck->>EqCheck: Check if complete
    end

    Note over Client: Receives WebSocket message:
    Note over Client: {<br/>  "type": "physicsUpdate",<br/>  "positions": [...],<br/>  "iteration": 45,<br/>  "equilibrium": false<br/>}

    rect rgb(200, 255, 200)
        Note over Client: ✅ Smooth real-time animation at 60 FPS
    end
```

---

## 3. Node Creation Event Flow (User Interaction)

### User Creates New Node

```mermaid
sequenceDiagram
    actor User
    participant Controller as NodeController
    participant Handler as CreateNodeCommandHandler
    participant Repo as GraphRepository
    participant DB as knowledge_graph.db
    participant EventBus as EventBus
    participant WSub as WebSocket Subscriber
    participant EventStore as Event Store Subscriber
    participant AuditLog as Audit Log Subscriber
    participant Analytics as Analytics Subscriber
    participant Clients as WebSocket Clients

    Note over User,Clients: NODE CREATION EVENT FLOW

    User->>Controller: POST /api/graph/nodes<br/>Body: {"label": "New Concept",<br/>"x": 10, "y": 20, "z": 0}

    Controller->>Handler: Parse request → CreateNodeCommand

    rect rgb(255, 240, 200)
        Note over Handler: 1. Validate command
        Handler->>Handler: Check label not empty
        Handler->>Handler: Check position valid
        Handler->>Handler: Check node_id unique

        alt Validation OK
            Handler->>Handler: ✅ Validation passed
        else Validation Failed
            Handler-->>Controller: ❌ ValidationError
            Controller-->>User: 400 Bad Request
        end
    end

    Handler->>Handler: 2. Create Node entity<br/>node = Node {<br/>  id: generate_id(),<br/>  label: "New Concept",<br/>  position: (10.0, 20.0, 0.0)<br/>}

    Handler->>Repo: add_node(node)
    Repo->>DB: INSERT INTO nodes<br/>(id, label, x, y, z)<br/>VALUES (?, ?, ?, ?, ?)
    DB-->>Repo: ✅ Node persisted
    Repo-->>Handler: Success

    rect rgb(200, 255, 200)
        Handler->>Handler: 3. Emit domain event<br/>NodeCreatedEvent {<br/>  node_id: node.id,<br/>  label: "New Concept",<br/>  timestamp: now(),<br/>  source: UserInteraction<br/>}
    end

    Handler->>EventBus: publish(NodeCreatedEvent)

    par Event Distribution
        EventBus->>WSub: NodeCreatedEvent
        EventBus->>EventStore: NodeCreatedEvent
        EventBus->>AuditLog: NodeCreatedEvent
        EventBus->>Analytics: NodeCreatedEvent
    end

    par Subscriber Actions
        WSub->>Clients: Broadcast to all clients
        EventStore->>EventStore: Save event for replay
        AuditLog->>AuditLog: Log user action
        Analytics->>Analytics: Track node creation rate
    end

    Note over Clients: Receive WebSocket message:
    Note over Clients: {<br/>  "type": "nodeCreated",<br/>  "node": {<br/>    "id": 317,<br/>    "label": "New Concept",<br/>    "position": [10, 20, 0],<br/>    "createdAt": "2025-10-26T20:42:00Z"<br/>  }<br/>}

    rect rgb(200, 255, 200)
        Note over Clients: ✅ All clients see new node instantly
    end
```

---

## 4. WebSocket Connection Event Flow

### Client Connects to WebSocket

```mermaid
sequenceDiagram
    participant Client as WebSocket Client
    participant ActixWS as Actix WebSocket Handler
    participant Gateway as WebSocketGateway
    participant EventBus as EventBus
    participant InitSync as Initial Sync Subscriber
    participant Metrics as Metrics Subscriber
    participant Logging as Logging Subscriber
    participant QueryHandler as GetGraphDataQueryHandler
    participant Repo as GraphRepository
    participant DB as knowledge_graph.db

    Note over Client,DB: WEBSOCKET CONNECTION EVENT FLOW

    Client->>ActixWS: ws://localhost:8080/ws/graph<br/>Open WebSocket connection

    ActixWS->>ActixWS: 1. Accept connection<br/>2. Generate client_id

    ActixWS->>Gateway: register_client(client_id)
    Gateway->>Gateway: Add to connected clients map

    rect rgb(200, 255, 200)
        Gateway->>EventBus: Emit WebSocketClientConnectedEvent<br/>{<br/>  client_id: "abc123",<br/>  timestamp: now()<br/>}
    end

    par Event Distribution
        EventBus->>InitSync: WebSocketClientConnectedEvent
        EventBus->>Metrics: WebSocketClientConnectedEvent
        EventBus->>Logging: WebSocketClientConnectedEvent
    end

    par Subscriber Actions
        InitSync->>InitSync: Send current graph state to new client
        Metrics->>Metrics: Track active connections
        Logging->>Logging: Log new connection
    end

    InitSync->>QueryHandler: Execute GetGraphDataQuery

    QueryHandler->>Repo: get_graph()
    Repo->>DB: SELECT * FROM nodes, edges
    DB-->>Repo: 316 nodes, 450 edges
    Repo-->>QueryHandler: Graph data
    QueryHandler-->>InitSync: Graph data

    InitSync->>Gateway: send_to_client(client_id, graph_data)
    Gateway->>Client: Send initial state

    Note over Client: Receives initial sync:
    Note over Client: {<br/>  "type": "initialSync",<br/>  "nodes": [...316 nodes...],<br/>  "edges": [...450 edges...],<br/>  "timestamp": "2025-10-26T20:42:00Z"<br/>}

    rect rgb(200, 255, 200)
        Note over Client: ✅ Client has full graph state
    end

    Note over Client,EventBus: After connection, client subscribes to all events:

    rect rgb(240, 240, 255)
        Note over Client: Subscribed to:<br/>• NodeCreated events<br/>• NodePositionChanged events<br/>• PhysicsUpdate events<br/>• GitHubSyncCompleted events<br/>• GraphReloaded events
    end

    rect rgb(200, 255, 200)
        Note over Client: ✅ Client stays in sync with server
    end
```

---

## 5. Cache Invalidation Event Flow

### When Cache Gets Invalidated

```mermaid
flowchart TD
    subgraph EventSources["Event Sources"]
        GitHubSync["GitHubSyncCompletedEvent"]
        NodeModified["NodeCreatedEvent /<br/>NodeUpdatedEvent /<br/>EdgeCreatedEvent"]
        PhysicsUpdate["PhysicsStepCompletedEvent"]
    end

    subgraph CacheInvalidation["Cache Invalidation Subscriber"]
        CacheSub["CacheInvalidationSubscriber"]
    end

    subgraph CacheService["CacheService Implementation"]
        direction TB
        CacheInfo["• In-memory LRU cache (optional)<br/>• Redis cache (distributed, optional)<br/>• TTL-based expiration<br/>• Event-driven invalidation"]

        subgraph CacheLayers["Cache Layers"]
            GraphCache["1. GraphData Cache<br/>Key: 'graph:full'<br/>TTL: 5 min OR event<br/>Invalidated by: GitHubSync,<br/>NodeCreated, EdgeCreated"]
            NodeCache["2. Node Cache<br/>Key: 'node:{id}'<br/>TTL: 10 min OR event<br/>Invalidated by: NodeUpdated,<br/>NodeDeleted"]
            PosCache["3. Position Cache<br/>Key: 'positions:snapshot'<br/>TTL: 1 min OR event<br/>Invalidated by: PhysicsUpdate"]
        end
    end

    GitHubSync -->|Event| CacheSub
    NodeModified -->|Event| CacheSub
    PhysicsUpdate -->|Event| CacheSub

    CacheSub -->|invalidate_all<br/>Clear all caches| GraphCache
    CacheSub -->|invalidate_all| NodeCache
    CacheSub -->|invalidate_all| PosCache

    CacheSub -.->|invalidate_graph_data<br/>Clear affected caches| GraphCache
    CacheSub -.->|invalidate_graph_data| NodeCache

    CacheSub -.->|invalidate_positions<br/>Clear position cache only| PosCache

    style GitHubSync fill:#ffcccc
    style NodeModified fill:#ffffcc
    style PhysicsUpdate fill:#ccccff
    style CacheSub fill:#ccffcc
    style GraphCache fill:#e6f3ff
    style NodeCache fill:#e6f3ff
    style PosCache fill:#e6f3ff
```

### Read Flow with Cache

```mermaid
flowchart TD
    Start([GetGraphDataQuery]) --> CheckCache{Check cache:<br/>cache_service.get<br/>'graph:full'}

    CheckCache -->|Cache HIT ⚡| ReturnCached["Return cached data<br/>(Fast! ~1ms)"]
    CheckCache -->|Cache MISS| QueryRepo["GraphRepository.get_graph()"]

    QueryRepo --> ReadDB["Read from SQLite<br/>(Slower ~50-100ms)"]
    ReadDB --> SetCache["cache_service.set<br/>'graph:full', data,<br/>TTL=5min"]
    SetCache --> ReturnData["Return data"]

    ReturnCached --> End([Response])
    ReturnData --> End

    style CheckCache fill:#fff4cc
    style ReturnCached fill:#ccffcc
    style QueryRepo fill:#ffcccc
    style ReadDB fill:#ffcccc
    style SetCache fill:#cce6ff
    style ReturnData fill:#ccffcc
```

---

## 6. Semantic Analysis Event Flow

### AI-Powered Semantic Analysis

```mermaid
sequenceDiagram
    actor User
    participant Controller as SemanticController
    participant Handler as SemanticAnalysisCommandHandler
    participant GraphRepo as GraphRepository
    participant SemanticService as SemanticService (Domain)
    participant GPU as GpuSemanticAnalyzer
    participant SemanticRepo as SemanticRepository
    participant SemanticDB as semantic_analysis.db
    participant EventBus as EventBus
    participant WSub as WebSocket Subscriber
    participant PhysicsSub as Physics Service Subscriber
    participant CacheInv as Cache Invalidation
    participant Metrics as Metrics Subscriber
    participant Client as WebSocket Client

    Note over User,Client: SEMANTIC ANALYSIS EVENT FLOW

    User->>Controller: POST /api/graph/analyze<br/>"Analyze Semantics"

    Controller->>Handler: Creates TriggerSemanticAnalysisCommand

    Handler->>GraphRepo: 1. Load graph data<br/>get_graph()
    GraphRepo-->>Handler: Returns nodes + edges

    Handler->>SemanticService: Run semantic analysis

    rect rgb(240, 230, 255)
        Note over SemanticService,GPU: 2. Analyze node relationships

        SemanticService->>SemanticService: Extract keywords from labels
        SemanticService->>SemanticService: Compute semantic similarity
        SemanticService->>SemanticService: Identify clusters

        SemanticService->>GPU: Use GPU for embedding generation<br/>compute_embeddings()
        GPU-->>SemanticService: Returns: Vec<(node_id, embedding[768])>

        SemanticService->>SemanticService: 3. Generate semantic constraints<br/>- Nodes with similar embeddings should be closer<br/>- Constraint: distance(A, B) < threshold

        SemanticService->>SemanticService: 4. Detect communities<br/>- Use Louvain algorithm<br/>- Identify node clusters

        SemanticService->>SemanticService: 5. Compute importance scores<br/>- PageRank on graph<br/>- Identify central nodes
    end

    SemanticService-->>Handler: SemanticAnalysisResults {<br/>  constraints: Vec<SemanticConstraint>,<br/>  communities: Vec<Community>,<br/>  importance_scores: HashMap<u32, f32><br/>}

    Handler->>SemanticRepo: 6. Save results<br/>save_analysis(results)
    SemanticRepo->>SemanticDB: INSERT results
    SemanticDB-->>SemanticRepo: ✅ Results persisted

    rect rgb(200, 255, 200)
        Handler->>EventBus: 7. Emit SemanticAnalysisCompletedEvent<br/>{<br/>  constraints_count: 150,<br/>  communities_count: 8,<br/>  timestamp: now()<br/>}
    end

    par Event Distribution
        EventBus->>WSub: SemanticAnalysisCompletedEvent
        EventBus->>PhysicsSub: SemanticAnalysisCompletedEvent
        EventBus->>CacheInv: SemanticAnalysisCompletedEvent
        EventBus->>Metrics: SemanticAnalysisCompletedEvent
    end

    par Subscriber Actions
        WSub->>Client: Notify clients of new constraints
        PhysicsSub->>PhysicsSub: Apply new semantic constraints<br/>to physics simulation
        CacheInv->>CacheInv: Clear semantic cache
        Metrics->>Metrics: Track analysis duration
    end

    Note over Client: Receives notification:
    Note over Client: {<br/>  "type": "semanticAnalysisComplete",<br/>  "constraintsGenerated": 150,<br/>  "communities": 8,<br/>  "message": "Semantic analysis complete.<br/>  Physics will apply new constraints."<br/>}

    rect rgb(200, 255, 200)
        Note over Client: ✅ Semantic analysis complete
    end
```

---

## 7. Error Handling Event Flow

### When Commands Fail

```mermaid
sequenceDiagram
    actor User
    participant Controller as NodeController
    participant Handler as CreateNodeCommandHandler
    participant Repo as GraphRepository
    participant DB as SQLite Database
    participant EventBus as EventBus
    participant ErrorLog as Error Logging Subscriber
    participant Notify as Notification Subscriber

    Note over User,Notify: ERROR HANDLING EVENT FLOW

    rect rgb(255, 230, 230)
        Note over User,Handler: Scenario 1: Validation Failure
    end

    User->>Controller: POST /api/graph/nodes<br/>Body: {"label": "", "x": "invalid", ...} ❌

    Controller->>Handler: Parse request → CreateNodeCommand

    Handler->>Handler: 1. Validate command

    rect rgb(255, 200, 200)
        Handler->>Handler: Validation FAILS<br/>- Label is empty<br/>- Position is invalid type

        Handler-->>Controller: Return Error(ValidationError)

        Note over Handler,Controller: ⚠️ No event emitted (validation failed)<br/>⚠️ No database write (transaction not started)
    end

    Controller-->>User: HTTP Response: 400 Bad Request<br/>{<br/>  "error": "ValidationError",<br/>  "message": "Node label cannot be empty",<br/>  "field": "label"<br/>}

    Note over User,Notify: ────────────────────────────────────

    rect rgb(255, 230, 230)
        Note over User,Notify: Scenario 2: Database Operation Fails
    end

    User->>Controller: POST /api/graph/nodes<br/>Body: {"label": "Node", "x": 10, "y": 20}

    Controller->>Handler: Parse request → CreateNodeCommand

    Handler->>Handler: Validation OK ✅

    Handler->>Repo: add_node(node)

    Repo->>DB: INSERT INTO nodes...

    rect rgb(255, 200, 200)
        DB-->>Repo: ❌ SQLite INSERT fails<br/>(e.g., duplicate ID)

        Repo-->>Handler: Return Error(DatabaseError)
    end

    rect rgb(255, 230, 200)
        Handler->>EventBus: Emit ErrorEvent (optional)

        par Event Distribution
            EventBus->>ErrorLog: ErrorEvent
            EventBus->>Notify: ErrorEvent
        end

        par Subscriber Actions
            ErrorLog->>ErrorLog: Log error to file<br/>Send to monitoring (e.g., Sentry)
            Notify->>Notify: Alert administrators
        end
    end

    Handler-->>Controller: Error response

    Controller-->>User: HTTP Response: 500 Internal Server Error<br/>{<br/>  "error": "DatabaseError",<br/>  "message": "Failed to insert node: UNIQUE constraint",<br/>  "requestId": "abc-123-def"<br/>}

    Note over User: ❌ Operation failed with proper error handling
```

---

## 8. Event Store Replay (Event Sourcing)

### Rebuilding State from Events

```mermaid
sequenceDiagram
    participant Admin as System Administrator
    participant EventStore as EventStore
    participant Replayer as EventReplayer
    participant InMemGraph as In-Memory Graph
    participant Repo as GraphRepository
    participant DB as knowledge_graph.db

    Note over Admin,DB: EVENT STORE REPLAY FLOW

    rect rgb(255, 230, 230)
        Note over Admin: Scenario: Database corruption<br/>or state rebuild needed
    end

    Admin->>EventStore: Start replay from version 0

    EventStore->>EventStore: get_events(from_version: 0)

    rect rgb(230, 240, 255)
        Note over EventStore: EventStore contains:<br/>┌────────────────────────────────┐<br/>│ Ver │ Event Type │ Data     │<br/>├────────────────────────────────┤<br/>│ 1   │ NodeCreated │ {id: 1} │<br/>│ 2   │ NodeCreated │ {id: 2} │<br/>│ 3   │ EdgeCreated │ {1-2}   │<br/>│ 4   │ NodePositionChanged │ ... │<br/>│ 5   │ GitHubSyncCompleted │ ... │<br/>│ ... │ ... │ ...     │<br/>│ 1000│ NodePositionChanged │ ... │<br/>└────────────────────────────────┘
    end

    EventStore-->>Replayer: Returns all events (1 → 1000)

    Replayer->>Replayer: Initialize empty in-memory graph

    rect rgb(240, 255, 240)
        Note over Replayer,InMemGraph: For each event:

        loop Event Replay
            alt NodeCreated
                Replayer->>InMemGraph: Apply: Add node to graph
            else EdgeCreated
                Replayer->>InMemGraph: Apply: Add edge to graph
            else NodePositionChanged
                Replayer->>InMemGraph: Apply: Update node position
            else GitHubSyncCompleted
                Replayer->>InMemGraph: Apply: Note sync timestamp
            else PhysicsStepCompleted
                Replayer->>InMemGraph: Apply: Note physics iteration
            end
        end
    end

    Replayer->>Replayer: Verify final state

    rect rgb(200, 255, 200)
        Note over Replayer: Final State:<br/>✅ Graph with 316 nodes<br/>✅ Edges correctly linked<br/>✅ Latest positions applied<br/>✅ State consistent with event history
    end

    Replayer->>Repo: save_graph(replayed_state)
    Repo->>DB: Write replayed state to database
    DB-->>Repo: ✅ State persisted
    Repo-->>Replayer: Success

    Replayer-->>Admin: ✅ State rebuilt successfully from events

    rect rgb(200, 255, 200)
        Note over Admin: Database restored to consistent state
    end
```

---

## Event Flow Summary Table

| Event Type | Triggers | Subscribers | Latency Target |
|------------|----------|-------------|----------------|
| `GitHubSyncCompleted` | GitHub sync service | Cache invalidation, WebSocket, Metrics, Logging | <100ms |
| `NodeCreated` | User creates node | WebSocket, Event store, Audit log | <50ms |
| `NodePositionChanged` | Physics or user drag | WebSocket (batched), Cache invalidation | <16ms (60 FPS) |
| `PhysicsStepCompleted` | Physics simulation | WebSocket, Metrics, Equilibrium checker | <16ms (60 FPS) |
| `WebSocketClientConnected` | Client connects | Initial sync, Metrics | <200ms |
| `SemanticAnalysisCompleted` | Semantic analysis | Physics service, WebSocket, Cache | <100ms |

---

**Event flows designed by**: Hive Mind Architecture Planner
**Date**: 2025-10-26
