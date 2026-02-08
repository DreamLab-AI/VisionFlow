# VisionFlow Architecture

System architecture reference for VisionFlow -- a real-time 3D knowledge graph visualization
platform with Rust backend, TypeScript/Three.js client, Neo4j graph database, and CUDA GPU compute.

Last updated: 2026-02-08

---

## System Overview

```
+-------------------+     WebSocket (JSON + Binary V3)     +-------------------+
|                   | <----------------------------------> |                   |
|  React/Three.js   |                                      |  Actix-web Rust   |
|  Client (Browser) |     HTTP REST (JSON)                 |  Backend Server   |
|                   | <----------------------------------> |                   |
+-------------------+                                      +--------+----------+
        |                                                           |
        | WebRTC P2P (voice)                              +---------+---------+
        |                                                 |         |         |
        v                                            Neo4j DB   CUDA GPU   Whelk-rs
   Remote Peers                                      (Cypher)   (100+ kernels) (OWL 2 EL)
```

### Component Summary

| Component | Technology | Role |
|:----------|:-----------|:-----|
| Client | React 19, Three.js, R3F | 3D graph rendering, WebXR, UI |
| Backend | Rust, Actix-web 4, Actix actors | HTTP/WS server, business logic, CQRS |
| Database | Neo4j 5 | Graph storage, Cypher queries |
| GPU | CUDA 12.4 | Physics simulation, analytics (100+ kernels) |
| Ontology | Whelk-rs (OWL 2 EL) | Reasoning, classification, inference |

---

## Actor System Architecture

The backend uses Actix actors for concurrent, message-driven processing. Each actor owns its
state and communicates exclusively through typed messages (`Handler<Msg>` pattern).

### Actor Hierarchy

```
                    main()
                      |
            +---------+---------+
            |                   |
    GraphServiceSupervisor  PhysicsSupervisor
            |                   |
    +-------+-------+    +-----+-----+
    |       |       |    |     |     |
 GraphState Ontology Settings Force  SSSP  Clustering
  Actor     Actor    Actor  Actor  Actor  Actor
                              |
                        AgentMonitorActor
```

### Key Actors

| Actor | Messages Handled | Responsibility |
|:------|:-----------------|:---------------|
| `GraphServiceSupervisor` | `UpdateBotsGraph`, `GetGraphData` | Coordinates graph state, broadcasts binary updates |
| `GraphStateActor` | `AddNode`, `RemoveNode`, `AddEdge` | Authoritative graph state, node/edge CRUD |
| `OntologyActor` | `LoadOntology`, `Classify`, `AddAxiom` | OWL reasoning via Whelk-rs |
| `OptimizedSettingsActor` | `GetSettings`, `UpdateSettings` | Settings persistence to Neo4j |
| `AgentMonitorActor` | `UpdateBotsGraph` | Receives agent telemetry from Management API |
| `PhysicsSupervisor` | `StartSimulation`, `StopSimulation` | Manages GPU compute actors |
| `ForceComputeActor` | `ComputeForces` | CUDA force-directed layout |
| `ClientCoordinatorActor` | `RegisterClient`, `BroadcastBinary` | WebSocket session management |

### Message Flow (Agent Telemetry Example)

```
Management API POST /api/bots/update
    -> AgentMonitorActor::handle(UpdateBotsGraph)
    -> GraphServiceSupervisor::handle(UpdateBotsGraph)
    -> Binary encode (V3, 48 bytes/node with agent flag bit 31)
    -> ClientCoordinatorActor::broadcast(binary_data)
    -> WebSocket -> Client AgentNodesLayer (Three.js instanced mesh)
```

---

## Binary WebSocket Protocol

### Protocol Versions

| Version | Bytes/Node | Status | Use Case |
|:--------|:-----------|:-------|:---------|
| V1 | 34 | Deprecated | Legacy (node ID <= 16383) |
| V2 | 36 | Stable | Production default |
| V3 | 48 | Stable | Analytics + node type classification |
| V4 | 16/change | Experimental | Delta encoding |

### V3 Wire Format (48 bytes per node)

```
Offset  Size   Type    Field
[0]     1      u8      Protocol version = 3
--- per node (48 bytes) ---
[0]     4      u32     Node ID + type flags (see below)
[4]     12     3xf32   Position (x, y, z)
[16]    12     3xf32   Velocity (vx, vy, vz)
[28]    4      f32     SSSP distance
[32]    4      i32     SSSP parent
[36]    4      u32     Cluster ID (K-means)
[40]    4      f32     Anomaly score (LOF, 0.0-1.0)
[44]    4      u32     Community ID (Louvain)
```

Total message size: `1 + (48 * node_count)` bytes.

### Node Type Bit Flags (u32 ID Field)

Bits 26-31 of the node ID u32 encode node type. Bits 0-25 are the actual node ID
(max 67,108,863 nodes).

```
Bit 31: Agent node       (0x80000000)
Bit 30: Knowledge node   (0x40000000)
Bit 28: Ontology property (0x10000000)
Bit 27: Ontology individual (0x08000000)
Bit 26: Ontology class   (0x04000000)
Bits 0-25: Actual node ID (0x03FFFFFF mask)
```

Client decoding (TypeScript):
```typescript
const raw = view.getUint32(offset, true);
const id = raw & 0x03FFFFFF;
const isAgent = (raw & 0x80000000) !== 0;
const isKnowledge = (raw & 0x40000000) !== 0;
const isOntologyClass = (raw & 0x04000000) !== 0;
const isOntologyIndividual = (raw & 0x08000000) !== 0;
const isOntologyProperty = (raw & 0x10000000) !== 0;
```

### Client Protocol Message Types

| Code | Name | Direction | Description |
|:-----|:-----|:----------|:------------|
| 0x01 | GRAPH_UPDATE | S->C | Graph topology (5-byte header with graph type flag) |
| 0x02 | VOICE_DATA | Bidirectional | Voice audio chunks |
| 0x10 | POSITION_UPDATE | C->S | Client node position change |
| 0x11 | AGENT_POSITIONS | S->C | Batch agent positions |
| 0x20 | AGENT_STATE_FULL | S->C | Full agent state (49 bytes/agent) |
| 0x21 | AGENT_STATE_DELTA | S->C | Delta agent state |
| 0x23 | AGENT_ACTION | S->C | Agent-to-data interaction event |
| 0x30 | CONTROL_BITS | C->S | Client control flags |
| 0x32 | HANDSHAKE | Bidirectional | Protocol version negotiation |
| 0x33 | HEARTBEAT | Bidirectional | Keep-alive |
| 0x34 | BROADCAST_ACK | C->S | Backpressure flow control |
| 0x50 | SYNC_UPDATE | Bidirectional | Graph operation sync (OT) |
| 0x51 | ANNOTATION_UPDATE | Bidirectional | Annotation CRUD |
| 0x52 | SELECTION_UPDATE | Bidirectional | Node selection broadcast |
| 0x53 | USER_POSITION | Bidirectional | Cursor/avatar position |
| 0x54 | VR_PRESENCE | Bidirectional | VR head + hand tracking |
| 0xFF | ERROR | S->C | Error response |

---

## Authentication Flow

### AuthenticatedUser Extractor

The `AuthenticatedUser` struct is an Actix `FromRequest` extractor. Adding it as a handler
parameter automatically enforces authentication on that endpoint.

```rust
pub async fn update_settings(
    _auth: AuthenticatedUser,  // <-- presence enforces auth
    body: web::Json<SettingsUpdate>,
) -> ActixResult<HttpResponse> { ... }
```

### Authentication Methods

| Method | Header | Validation |
|:-------|:-------|:-----------|
| Bearer token + Nostr pubkey | `Authorization: Bearer <token>` + `X-Nostr-Pubkey: <hex>` | `NostrService::validate_session()` |
| Dev bypass | env `SETTINGS_AUTH_BYPASS=true` | Skips validation, returns `dev-user` with power_user=true |

### Auth Decision Flow

```
Request arrives
    |
    +-- SETTINGS_AUTH_BYPASS=true? --> Return dev-user (power_user=true)
    |
    +-- Extract Authorization header
    |       Missing? --> 401 Unauthorized
    |
    +-- Strip "Bearer " prefix
    |       Missing? --> 401 Unauthorized
    |
    +-- Extract X-Nostr-Pubkey header
    |       Missing? --> 401 Unauthorized
    |
    +-- NostrService.validate_session(pubkey, token)
    |       Invalid? --> 401 Unauthorized
    |
    +-- NostrService.get_user(pubkey)
            Not found? --> 401 Unauthorized
            Found? --> AuthenticatedUser { pubkey, is_power_user }
```

### Secured Endpoint Groups (44 handlers)

| Group | Handler Count | Module |
|:------|:-------------|:-------|
| Analytics | 16 | `api_handler/analytics/mod.rs` |
| Semantic forces | 7 | `api_handler/semantic_forces.rs` |
| Workspace | 5 | `workspace_handler.rs` |
| Bots | 4 | `bots_handler.rs` |
| Constraints | 4 | `constraints_handler.rs` |
| Ontology physics | 3 | `api_handler/ontology_physics/mod.rs` |
| RAGFlow | 3 | `ragflow_handler.rs` |
| Quest3 | 1 | `api_handler/quest3/mod.rs` |
| Settings | 1 | `settings/api/settings_routes.rs` |

Read-only GET endpoints remain public.

---

## Multi-Graph Rendering Pipeline

The client renders three graph layers simultaneously, each sourced from the same binary
WebSocket stream but filtered by node type bit flags.

```
Binary WebSocket Frame (V3)
    |
    +-- Decode node ID + type flags
    |
    +-- isAgent (bit 31)?
    |       --> AgentNodesLayer (green instanced spheres)
    |
    +-- isKnowledge (bit 30)?
    |       --> KnowledgeGraphLayer (blue instanced spheres, cluster hulls)
    |
    +-- isOntologyClass/Individual/Property (bits 26-28)?
    |       --> OntologyGraphLayer (purple/orange/cyan, hierarchical rings)
    |
    +-- Default (no flags)?
            --> DefaultGraphLayer (standard rendering)
```

Each layer maintains its own:
- Instanced mesh buffer (GPU-side)
- Edge geometry (per-mode coloring)
- Physics simulation parameters
- LOD (level of detail) thresholds

---

## EventBus Pattern

### Architecture

```
                 publish(event)
DomainEvent -----> EventBus -----> wildcard subscribers (*)
                      |
                      +-----> type-specific subscribers
                      |
                      +-----> middleware chain (logging, metrics)
```

### EventHandler Trait

```rust
pub trait EventHandler: Send + Sync {
    fn handler_id(&self) -> &str;
    async fn handle(&self, event: &StoredEvent) -> Result<(), EventError>;
}
```

### Registered Handlers

| Handler | Subscribes To | Action |
|:--------|:-------------|:-------|
| `GraphEventHandler` | `node.*`, `edge.*`, `graph.cleared` | Updates actor state, triggers re-render |
| `OntologyEventHandler` | `ontology.*` | Triggers re-classification on class/property/axiom changes |
| `AuditEventHandler` | `*` (wildcard) | Logs all events to audit store with timestamp and actor ID |
| `NotificationEventHandler` | `*` (wildcard) | Pushes real-time notifications to connected WebSocket clients |

### Domain Events

| Event Type | Aggregate | Trigger |
|:-----------|:----------|:--------|
| `node.created` | Graph | AddNode directive |
| `node.updated` | Graph | UpdateNode directive |
| `node.deleted` | Graph | RemoveNode directive |
| `edge.created` | Graph | AddEdge directive |
| `edge.deleted` | Graph | RemoveEdge directive |
| `graph.cleared` | Graph | ClearGraph directive |
| `ontology.loaded` | Ontology | LoadOntology directive |
| `ontology.class_added` | Ontology | AddOwlClass directive |
| `ontology.property_added` | Ontology | AddProperty directive |
| `ontology.axiom_added` | Ontology | AddAxiom directive |
| `ontology.inference_completed` | Ontology | RunInference directive |

---

## CQRS Architecture

### Layer Structure

```
HTTP Request
    |
    v
Handler (Actix endpoint)
    |
    v
Application Service (orchestration)
    |
    +-- Directive (write) --> Port (trait) --> Adapter (Neo4j) --> Database
    |                                |
    |                                +--> EventBus.publish(DomainEvent)
    |
    +-- Query (read) -----> Port (trait) --> Adapter (Neo4j) --> Database
```

### Ports (Traits)

| Port | Methods | Adapter |
|:-----|:--------|:--------|
| `KnowledgeGraphRepository` | `get_graph`, `add_node`, `remove_node`, `add_edge` | `Neo4jGraphRepository` |
| `OntologyRepository` | `load_ontology`, `add_class`, `add_axiom`, `classify` | `Neo4jOntologyRepository` |
| `SettingsRepository` | `get_settings`, `update_settings`, `get_physics_config` | `Neo4jSettingsRepository` |
| `PhysicsSimulator` | `start`, `stop`, `compute_forces` | `CudaPhysicsAdapter` |
| `SemanticAnalyzer` | `analyze`, `get_embeddings`, `compute_similarity` | `GpuSemanticAnalyzer` |

### Directives (Commands)

Directives are write operations that mutate state and publish domain events:

```rust
// Example: AddNode directive
pub struct AddNodeDirective {
    pub graph_id: String,
    pub node: NodeData,
}

impl AddNodeDirective {
    pub async fn execute(
        &self,
        repo: &dyn KnowledgeGraphRepository,
        event_bus: &EventBus,
    ) -> Result<NodeId, DomainError> {
        let node_id = repo.add_node(&self.graph_id, &self.node).await?;
        event_bus.publish(NodeCreatedEvent { graph_id, node_id }).await?;
        Ok(node_id)
    }
}
```

---

## Technology Stack

| Layer | Technology | Version |
|:------|:-----------|:--------|
| UI Framework | React | 19.x |
| 3D Rendering | Three.js (React Three Fiber) | 0.182.x |
| XR Runtime | WebXR (@react-three/xr) | 6.x |
| Client Language | TypeScript | 5.9.x |
| Build Tool | Vite | 6.x |
| Backend Framework | Actix-web | 4.x |
| Backend Language | Rust | 1.75+ |
| Database | Neo4j | 5.x |
| GPU Compute | CUDA | 12.4 |
| Ontology Engine | Whelk-rs (OWL 2 EL) | -- |
| WebSocket | JSON + Binary V3 | -- |
| Spatial Audio | Web Audio API (HRTF) | -- |

---

## Docker Build (5-Stage)

```dockerfile
# Stage 1: Dependency cache (rebuilds only on Cargo.lock change)
FROM rust AS deps
COPY Cargo.toml Cargo.lock ./
RUN cargo build --release --locked

# Stage 2: Full Rust build
FROM deps AS build
COPY src/ ./src/
RUN cargo build --release

# Stage 3: CUDA kernel compilation (parallel with Stage 4)
FROM nvidia/cuda AS cuda
COPY src/gpu/kernels/ ./
RUN nvcc --compile ...

# Stage 4: Frontend build (parallel with Stage 3)
FROM node AS frontend
COPY client/ ./
RUN npm ci && npm run build

# Stage 5: Runtime
FROM debian:bookworm-slim
COPY --from=build /app/target/release/visionflow .
COPY --from=cuda /app/kernels/ ./kernels/
COPY --from=frontend /app/dist/ ./static/
```

Key optimizations:
- Stages 3 and 4 run in parallel (no dependency)
- Stage 1 is cached unless `Cargo.lock` changes
- Final image contains only release binary + assets

---

## File Organization

```
src/
  actors/           # Actix actors (message handlers)
    gpu/            # GPU compute actors (CUDA wrappers)
  adapters/         # Port implementations (Neo4j, CUDA, etc.)
  application/      # CQRS directives, queries, services
  config/           # Configuration loading
  constraints/      # Semantic physics constraint system
  cqrs/             # CQRS types and handler registration
  events/           # EventBus, domain events, handlers
    handlers/       # GraphEventHandler, AuditEventHandler, etc.
  gpu/              # CUDA kernel management, memory pools
  handlers/         # HTTP/WS endpoint handlers
    api_handler/    # REST API grouped by domain
  inference/        # OWL reasoning optimization
  middleware/       # Auth, rate limiting, CORS
  services/         # Shared services (Nostr, bots, etc.)
  settings/         # Settings management + auth extractor
  utils/            # Binary protocol, validation, helpers

client/
  src/
    components/     # React components
    services/       # BinaryWebSocketProtocol, FeatureFlags, etc.
    hooks/          # React hooks (useActionConnections, useTelemetry)
    layers/         # Three.js rendering layers (AgentNodes, KnowledgeGraph)
```
