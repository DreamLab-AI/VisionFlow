# Event Flow Diagrams
**Detailed Event Flows for Hexagonal/CQRS Architecture**

---

## 1. GitHub Sync Event Flow (BUG FIX)

### Current Problem Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT BROKEN FLOW                          │
└─────────────────────────────────────────────────────────────────┘

GitHub API
    │
    │ 1. Fetch markdown files
    ▼
EnhancedContentAPI
    │
    │ 2. Return 316 files
    ▼
GitHubSyncService
    │
    │ 3. Parse files → 316 nodes, 450 edges
    ▼
SqliteKnowledgeGraphRepository
    │
    │ 4. INSERT INTO nodes (316 nodes)
    │    INSERT INTO edges (450 edges)
    ▼
knowledge_graph.db
    │
    │ ✅ Database has 316 nodes
    │
    ❌ NO EVENT EMITTED!
    │
    │ GraphServiceActor still has:
    │ - In-memory cache: 63 nodes (STALE!)
    │ - Never refreshed
    │
    ▼
GET /api/graph/data
    │
    │ Sends GetGraphData message
    ▼
GraphServiceActor
    │
    │ Returns in-memory cache
    ▼
❌ CLIENT SEES: 63 nodes (WRONG!)
```

### Fixed Event-Driven Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                    NEW EVENT-DRIVEN FLOW                        │
└─────────────────────────────────────────────────────────────────┘

GitHub API
    │
    │ 1. Fetch markdown files
    ▼
EnhancedContentAPI
    │
    │ 2. Return 316 files
    ▼
GitHubSyncService
    │
    │ 3. Parse files → 316 nodes, 450 edges
    ▼
GraphRepository (Port)
    │
    │ 4. save_graph(316 nodes, 450 edges)
    ▼
SqliteGraphRepository (Adapter)
    │
    │ 5. INSERT INTO nodes/edges
    ▼
knowledge_graph.db
    │
    │ ✅ Database has 316 nodes
    ▼
GitHubSyncService
    │
    │ 6. ✅ Emit GitHubSyncCompletedEvent
    ▼
EventBus
    │
    ├─────────────────────┬─────────────────────┬──────────────────────┐
    │                     │                     │                      │
    ▼                     ▼                     ▼                      ▼
CacheInvalidation    WebSocket           Logging              Metrics
Subscriber          Subscriber          Subscriber          Subscriber
    │                     │                     │                      │
    │                     │                     │                      │
    ▼                     ▼                     ▼                      ▼
Clear ALL            Broadcast           Log: "Sync          Track sync
caches               "graphReloaded"     completed:          duration
                     to clients          316 nodes"          and stats
    │                     │
    │                     │
    ▼                     ▼
Next API call        Clients reload
reads fresh          and fetch new
from database        graph data
    │                     │
    ▼                     ▼
✅ Returns 316     ✅ Display 316
   nodes               nodes
```

---

## 2. Physics Simulation Event Flow

### Physics Step Execution
```
┌─────────────────────────────────────────────────────────────────┐
│               PHYSICS SIMULATION EVENT FLOW                     │
└─────────────────────────────────────────────────────────────────┘

User clicks "Start Simulation"
    │
    │ POST /api/physics/start
    ▼
PhysicsController (HTTP Handler)
    │
    │ Creates TriggerPhysicsStepCommand
    ▼
TriggerPhysicsStepCommandHandler
    │
    │ 1. Validate simulation params
    │ 2. Load graph from repository
    ▼
GraphRepository.get_graph()
    │
    │ Returns 316 nodes, 450 edges
    ▼
PhysicsService (Domain Service)
    │
    │ For each iteration (e.g., 100 iterations):
    │
    ├──> 3. Calculate forces (GPU)
    │    - Repulsion between nodes
    │    - Attraction along edges
    │    - Apply constraints
    │
    ├──> 4. Update velocities
    │
    ├──> 5. Update positions
    │        new_pos = old_pos + velocity * dt
    │
    ├──> 6. Check convergence
    │        if max_velocity < threshold:
    │            equilibrium = true
    │
    └──> Batch position updates: Vec<(node_id, position)>
         │
         ▼
GraphRepository.batch_update_positions()
    │
    │ UPDATE nodes SET x=?, y=?, z=? WHERE id=?
    ▼
knowledge_graph.db
    │
    │ ✅ Positions saved
    ▼
PhysicsService
    │
    │ Emit PhysicsStepCompletedEvent
    ▼
EventBus
    │
    ├──────────────────┬──────────────────┬─────────────────┐
    │                  │                  │                 │
    ▼                  ▼                  ▼                 ▼
WebSocket         Cache            Metrics          Equilibrium
Subscriber     Invalidation      Subscriber          Checker
    │              Subscriber         │                 │
    │                  │               │                 │
    ▼                  ▼               ▼                 ▼
Broadcast          Clear          Track FPS       Check if
positions          cache          and perf        complete
to clients
    │
    │ Every 16ms (60 FPS)
    ▼
WebSocket clients receive:
{
  "type": "physicsUpdate",
  "positions": [
    {"id": 1, "x": 10.5, "y": 20.3, "z": 0.0},
    {"id": 2, "x": -5.2, "y": 15.1, "z": 0.0},
    ...
  ],
  "iteration": 45,
  "equilibrium": false
}
    │
    ▼
✅ Smooth real-time animation
```

---

## 3. Node Creation Event Flow (User Interaction)

### User Creates New Node
```
┌─────────────────────────────────────────────────────────────────┐
│                  NODE CREATION EVENT FLOW                       │
└─────────────────────────────────────────────────────────────────┘

User clicks "Add Node" in UI
    │
    │ POST /api/graph/nodes
    │ Body: {"label": "New Concept", "x": 10, "y": 20, "z": 0}
    ▼
NodeController (HTTP Handler)
    │
    │ Parse request → CreateNodeCommand
    ▼
CreateNodeCommandHandler
    │
    │ 1. Validate command
    │    - Check label not empty
    │    - Check position valid
    │    - Check node_id unique
    │
    ├──> Validation OK?
    │    │
    │    ▼ Yes
    │
    │ 2. Create Node entity
    │    node = Node {
    │      id: generate_id(),
    │      label: "New Concept",
    │      position: (10.0, 20.0, 0.0),
    │      ...
    │    }
    │
    ▼
GraphRepository.add_node(node)
    │
    │ INSERT INTO nodes (id, label, x, y, z) VALUES (?, ?, ?, ?, ?)
    ▼
knowledge_graph.db
    │
    │ ✅ Node persisted
    ▼
CreateNodeCommandHandler
    │
    │ 3. Emit domain event
    │    event = NodeCreatedEvent {
    │      node_id: node.id,
    │      label: "New Concept",
    │      timestamp: now(),
    │      source: UserInteraction
    │    }
    │
    ▼
EventBus.publish(event)
    │
    ├─────────────────┬─────────────────┬────────────────┐
    │                 │                 │                │
    ▼                 ▼                 ▼                ▼
WebSocket        Event Store      Audit Log      Analytics
Subscriber       Subscriber      Subscriber     Subscriber
    │                 │                 │                │
    │                 │                 │                │
    ▼                 ▼                 ▼                ▼
Broadcast         Save event      Log user       Track node
to all clients    for replay      action         creation rate
    │
    │ Send to all connected clients
    ▼
WebSocket clients receive:
{
  "type": "nodeCreated",
  "node": {
    "id": 317,
    "label": "New Concept",
    "position": [10, 20, 0],
    "createdAt": "2025-10-26T20:42:00Z"
  }
}
    │
    ▼
✅ All clients see new node instantly
```

---

## 4. WebSocket Connection Event Flow

### Client Connects to WebSocket
```
┌─────────────────────────────────────────────────────────────────┐
│             WEBSOCKET CONNECTION EVENT FLOW                     │
└─────────────────────────────────────────────────────────────────┘

Client opens WebSocket connection
    │
    │ ws://localhost:8080/ws/graph
    ▼
Actix WebSocket Handler
    │
    │ 1. Accept connection
    │ 2. Generate client_id
    │
    ▼
WebSocketGateway.register_client(client_id)
    │
    │ Add to connected clients map
    │
    ▼
Emit WebSocketClientConnectedEvent
    │
    │ event = WebSocketClientConnectedEvent {
    │   client_id: "abc123",
    │   timestamp: now(),
    │ }
    │
    ▼
EventBus.publish(event)
    │
    ├────────────────────┬────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
Initial Sync       Metrics            Logging
Subscriber       Subscriber         Subscriber
    │                    │                    │
    │                    │                    │
    ▼                    ▼                    ▼
Send current     Track active       Log new
graph state      connections        connection
to new client
    │
    │ Execute GetGraphDataQuery
    ▼
GetGraphDataQueryHandler
    │
    ▼
GraphRepository.get_graph()
    │
    ▼
knowledge_graph.db (316 nodes)
    │
    ▼
Return graph data
    │
    ▼
WebSocketGateway.send_to_client(client_id, graph_data)
    │
    │ Send initial state to new client
    ▼
Client receives:
{
  "type": "initialSync",
  "nodes": [...316 nodes...],
  "edges": [...450 edges...],
  "timestamp": "2025-10-26T20:42:00Z"
}
    │
    ▼
✅ Client has full graph state

────────────────────────────────────────────────────────────────

After connection, client subscribes to all events:
    │
    ├──> NodeCreated events
    ├──> NodePositionChanged events
    ├──> PhysicsUpdate events
    ├──> GitHubSyncCompleted events
    └──> GraphReloaded events
         │
         ▼
Client stays in sync with server
```

---

## 5. Cache Invalidation Event Flow

### When Cache Gets Invalidated
```
┌─────────────────────────────────────────────────────────────────┐
│              CACHE INVALIDATION EVENT FLOW                      │
└─────────────────────────────────────────────────────────────────┘

Multiple Event Sources Can Trigger Cache Invalidation:

GitHub Sync Event:
    │
    ▼
GitHubSyncCompletedEvent
    │
    ├──> CacheInvalidationSubscriber
    │
    └──> cache_service.invalidate_all()
         - Clear graph data cache
         - Clear node cache
         - Clear edge cache
         - Clear semantic cache

Node/Edge Modified Events:
    │
    ▼
NodeCreatedEvent / NodeUpdatedEvent / EdgeCreatedEvent
    │
    ├──> CacheInvalidationSubscriber
    │
    └──> cache_service.invalidate_graph_data()
         - Clear affected caches
         - Keep node-specific caches

Physics Update Events:
    │
    ▼
PhysicsStepCompletedEvent
    │
    ├──> CacheInvalidationSubscriber
    │
    └──> cache_service.invalidate_positions()
         - Clear position cache
         - Keep node metadata cache

────────────────────────────────────────────────────────────────

Cache Service Implementation:
┌─────────────────────────────────────────────────────────────────┐
│                    CacheService                                 │
│                                                                  │
│  - In-memory LRU cache (optional optimization)                  │
│  - Redis cache (distributed cache, optional)                    │
│  - TTL-based expiration (e.g., 5 minutes)                       │
│  - Event-driven invalidation (immediate)                        │
└─────────────────────────────────────────────────────────────────┘

Cache Layers:
1. GraphData cache (full graph)
   - Key: "graph:full"
   - TTL: 5 minutes OR event invalidation
   - Invalidated by: GitHubSync, NodeCreated, EdgeCreated

2. Node cache (individual nodes)
   - Key: "node:{id}"
   - TTL: 10 minutes OR event invalidation
   - Invalidated by: NodeUpdated, NodeDeleted

3. Position cache (physics state)
   - Key: "positions:snapshot"
   - TTL: 1 minute OR event invalidation
   - Invalidated by: PhysicsUpdate

────────────────────────────────────────────────────────────────

Read Flow with Cache:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  GetGraphDataQuery                                              │
│         │                                                        │
│         ▼                                                        │
│  Check cache: cache_service.get("graph:full")                   │
│         │                                                        │
│         ├──> Cache HIT                                           │
│         │    └──> Return cached data ⚡ (fast!)                  │
│         │                                                        │
│         └──> Cache MISS                                          │
│              │                                                   │
│              ▼                                                   │
│         GraphRepository.get_graph()                             │
│              │                                                   │
│              ▼                                                   │
│         Read from SQLite 🐌 (slower)                            │
│              │                                                   │
│              ▼                                                   │
│         cache_service.set("graph:full", data, TTL=5min)         │
│              │                                                   │
│              ▼                                                   │
│         Return data                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Semantic Analysis Event Flow

### AI-Powered Semantic Analysis
```
┌─────────────────────────────────────────────────────────────────┐
│            SEMANTIC ANALYSIS EVENT FLOW                         │
└─────────────────────────────────────────────────────────────────┘

User clicks "Analyze Semantics"
    │
    │ POST /api/graph/analyze
    ▼
SemanticController
    │
    │ Creates TriggerSemanticAnalysisCommand
    ▼
SemanticAnalysisCommandHandler
    │
    │ 1. Load graph data
    ▼
GraphRepository.get_graph()
    │
    │ Returns nodes + edges
    ▼
SemanticService (Domain Service)
    │
    │ 2. Analyze node relationships
    │    - Extract keywords from labels
    │    - Compute semantic similarity
    │    - Identify clusters
    │
    ├──> Use GPU for embedding generation
    │    │
    │    ▼
    │    GpuSemanticAnalyzer.compute_embeddings()
    │    │
    │    ▼
    │    Returns: Vec<(node_id, embedding[768])>
    │
    ├──> 3. Generate semantic constraints
    │    - Nodes with similar embeddings should be closer
    │    - Constraint: distance(A, B) < threshold
    │
    ├──> 4. Detect communities
    │    - Use Louvain algorithm
    │    - Identify node clusters
    │
    └──> 5. Compute importance scores
         - PageRank on graph
         - Identify central nodes
         │
         ▼
SemanticAnalysisResults {
    constraints: Vec<SemanticConstraint>,
    communities: Vec<Community>,
    importance_scores: HashMap<u32, f32>,
}
    │
    │ 6. Save results to repository
    ▼
SemanticRepository.save_analysis(results)
    │
    ▼
semantic_analysis.db
    │
    │ ✅ Results persisted
    ▼
SemanticAnalysisCommandHandler
    │
    │ 7. Emit event
    │    event = SemanticAnalysisCompletedEvent {
    │      constraints_count: 150,
    │      communities_count: 8,
    │      timestamp: now(),
    │    }
    │
    ▼
EventBus.publish(event)
    │
    ├─────────────────┬─────────────────┬─────────────────┐
    │                 │                 │                 │
    ▼                 ▼                 ▼                 ▼
WebSocket       Physics Service   Cache           Metrics
Subscriber      Subscriber     Invalidation    Subscriber
    │                 │            Subscriber        │
    │                 │                 │            │
    ▼                 ▼                 ▼            ▼
Notify clients   Apply new       Clear           Track
of new          semantic        semantic        analysis
constraints     constraints     cache           duration
                to physics
    │
    ▼
Client receives:
{
  "type": "semanticAnalysisComplete",
  "constraintsGenerated": 150,
  "communities": 8,
  "message": "Semantic analysis complete. Physics will apply new constraints."
}
```

---

## 7. Error Handling Event Flow

### When Commands Fail
```
┌─────────────────────────────────────────────────────────────────┐
│                ERROR HANDLING EVENT FLOW                        │
└─────────────────────────────────────────────────────────────────┘

User sends invalid command
    │
    │ POST /api/graph/nodes
    │ Body: {"label": "", "x": "invalid", ...} ❌
    ▼
NodeController
    │
    │ Parse request → CreateNodeCommand
    ▼
CreateNodeCommandHandler
    │
    │ 1. Validate command
    │
    ├──> Validation FAILS
    │    - Label is empty
    │    - Position is invalid type
    │
    ▼
Return Error(ValidationError)
    │
    │ No event emitted (validation failed)
    │ No database write (transaction not started)
    │
    ▼
HTTP Response: 400 Bad Request
{
  "error": "ValidationError",
  "message": "Node label cannot be empty",
  "field": "label"
}

────────────────────────────────────────────────────────────────

Database Operation Fails:
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  CreateNodeCommandHandler                                       │
│         │                                                        │
│         ▼                                                        │
│  Validation OK ✅                                               │
│         │                                                        │
│         ▼                                                        │
│  GraphRepository.add_node(node)                                 │
│         │                                                        │
│         ▼                                                        │
│  SQLite INSERT fails (e.g., duplicate ID) ❌                   │
│         │                                                        │
│         ▼                                                        │
│  Return Error(DatabaseError)                                    │
│         │                                                        │
│         ▼                                                        │
│  Emit ErrorEvent (optional)                                     │
│         │                                                        │
│         ├──> ErrorLoggingSubscriber                             │
│         │    - Log error to file                                │
│         │    - Send to monitoring (e.g., Sentry)                │
│         │                                                        │
│         └──> NotificationSubscriber                             │
│              - Alert administrators                             │
│                                                                  │
│  HTTP Response: 500 Internal Server Error                       │
│  {                                                               │
│    "error": "DatabaseError",                                    │
│    "message": "Failed to insert node: UNIQUE constraint",       │
│    "requestId": "abc-123-def"                                   │
│  }                                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Event Store Replay (Event Sourcing)

### Rebuilding State from Events
```
┌─────────────────────────────────────────────────────────────────┐
│                EVENT STORE REPLAY FLOW                          │
└─────────────────────────────────────────────────────────────────┘

Scenario: Database corruption or state rebuild needed

EventStore contains:
┌──────────────────────────────────────────────────────────────┐
│ Version │ Event Type             │ Timestamp │ Data          │
├──────────────────────────────────────────────────────────────┤
│ 1       │ NodeCreated            │ 10:00:00  │ {id: 1, ...}  │
│ 2       │ NodeCreated            │ 10:00:01  │ {id: 2, ...}  │
│ 3       │ EdgeCreated            │ 10:00:02  │ {id: "1-2"...}│
│ 4       │ NodePositionChanged    │ 10:00:03  │ {id: 1, ...}  │
│ 5       │ GitHubSyncCompleted    │ 10:05:00  │ {nodes: 316...}│
│ 6       │ PhysicsStepCompleted   │ 10:05:01  │ {iter: 1...}  │
│ ...     │ ...                    │ ...       │ ...           │
│ 1000    │ NodePositionChanged    │ 11:00:00  │ {id: 50, ...} │
└──────────────────────────────────────────────────────────────┘

Replay Process:
    │
    │ Start from version 0
    ▼
EventStore.get_events(from_version: 0)
    │
    │ Returns all events (1 → 1000)
    ▼
EventReplayer
    │
    │ For each event:
    │
    ├──> NodeCreated
    │    │
    │    └──> Apply: Add node to in-memory graph
    │
    ├──> EdgeCreated
    │    │
    │    └──> Apply: Add edge to in-memory graph
    │
    ├──> NodePositionChanged
    │    │
    │    └──> Apply: Update node position
    │
    ├──> GitHubSyncCompleted
    │    │
    │    └──> Apply: Note sync timestamp
    │
    └──> PhysicsStepCompleted
         │
         └──> Apply: Note physics iteration
    │
    │ After all events replayed:
    ▼
Final State:
- Graph with 316 nodes ✅
- Edges correctly linked ✅
- Latest positions applied ✅
- State consistent with event history ✅
    │
    │ Save to database
    ▼
GraphRepository.save_graph(replayed_state)
    │
    ▼
✅ State rebuilt successfully from events
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
