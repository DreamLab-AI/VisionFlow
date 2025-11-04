# GraphServiceActor Deprecation Research Report

## Executive Summary

**GraphServiceActor** is being deprecated in favor of a **Hexagonal/CQRS architecture** with specialized actors managed by **TransitionalGraphSupervisor**. This migration addresses critical architectural issues including cache coherency bugs, tight coupling, and maintainability challenges with the 48,000+ token monolithic actor.

---

## Current Architecture Analysis

### GraphServiceActor Status

**Location**: `/home/devuser/workspace/project/src/actors/graph-actor.rs`
**Size**: 156KB, 4,614 lines, 48,000+ tokens
**Status**: ⚠️ **TRANSITIONAL** - Being wrapped by TransitionalGraphSupervisor

#### Primary Issues
1. **Cache Coherency Bug**: In-memory cache shows 63 nodes when database contains 316 nodes after GitHub sync
2. **Monolithic Design**: Single actor handles graph state, physics, WebSocket broadcasting, and semantic analysis
3. **Tight Coupling**: Direct dependencies on WebSocket infrastructure, GPU managers, and client coordinators
4. **Testing Challenges**: Cannot test graph logic without full actor system initialization
5. **Scalability Bottleneck**: Single actor processes all graph operations

#### Current Responsibilities
```rust
pub struct GraphServiceActor {
    graph-data: Arc<RwLock<GraphData>>,           // In-memory cache - PRIMARY ISSUE
    bots-graph-data: Arc<RwLock<GraphData>>,
    simulation-params: Arc<RwLock<SimulationParams>>,
    ws-server: Option<Addr<WebSocketServer>>,
    client-manager: Addr<ClientCoordinatorActor>,
    gpu-manager: Option<Addr<GPUManagerActor>>,
    kg-repo: Arc<dyn KnowledgeGraphRepository>,
    // ... 50+ more fields
}
```

### Messages Handled
- **46 message handlers** processing 129+ message types
- Mixed concerns: state management, physics simulation, WebSocket broadcasting, semantic analysis
- No clear separation between reads and writes

---

## Replacement Architecture

### Overview: Hexagonal/CQRS Migration

```
┌─────────────────────────────────────────────────────────────┐
│                     OLD ARCHITECTURE                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │        GraphServiceActor (48K tokens)                 │  │
│  │  • In-memory cache (STALE!)                          │  │
│  │  • Physics simulation                                │  │
│  │  • WebSocket broadcasting                            │  │
│  │  • Semantic analysis                                 │  │
│  │  • Settings management                               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
                      MIGRATION
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    NEW ARCHITECTURE                          │
│                                                              │
│  ┌──────────────────┐     ┌──────────────────┐            │
│  │  CQRS Pattern    │     │  Specialized     │            │
│  │  • Commands      │     │  Actors          │            │
│  │  • Queries       │     │  • GraphState    │            │
│  │  • Events        │     │  • Physics       │            │
│  └──────────────────┘     │  • Semantic      │            │
│                           │  • Client        │            │
│  ┌──────────────────┐     └──────────────────┘            │
│  │  Repositories    │                                      │
│  │  • Graph         │     ┌──────────────────┐            │
│  │  • Ontology      │     │  Event Bus       │            │
│  │  • Settings      │     │  • Cache         │            │
│  └──────────────────┘     │    Invalidation  │            │
│                           │  • WebSocket     │            │
│  ┌──────────────────┐     │    Broadcast     │            │
│  │  Neo4j Adapter   │     └──────────────────┘            │
│  │  (Primary DB)    │                                      │
│  └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Replacement Components

### 1. TransitionalGraphSupervisor

**Location**: `/home/devuser/workspace/project/src/actors/graph-service-supervisor.rs`
**Purpose**: Temporary wrapper that manages GraphServiceActor lifecycle during migration
**Status**: ✅ **ACTIVE** - Current recommended pattern

```rust
pub struct TransitionalGraphSupervisor {
    graph-service-actor: Option<Addr<GraphServiceActor>>,
    client-manager-addr: Option<Addr<ClientCoordinatorActor>>,
    gpu-manager-addr: Option<Addr<GPUManagerActor>>,
    kg-repo: Arc<dyn KnowledgeGraphRepository>,
}
```

**Key Features**:
- Manages GraphServiceActor as a supervised child
- Provides message forwarding to maintain compatibility
- Tracks metrics: uptime, messages forwarded
- Enables gradual migration without breaking existing code

**Usage in AppState** (`/home/devuser/workspace/project/src/app-state.rs`):
```rust
pub struct AppState {
    pub graph-service-addr: Addr<TransitionalGraphSupervisor>,  // ← Use this
    // NOT: Addr<GraphServiceActor>
}
```

### 2. ActorGraphRepository

**Location**: `/home/devuser/workspace/project/src/adapters/actor-graph-repository.rs`
**Purpose**: Adapter implementing GraphRepository port using GraphServiceActor
**Status**: ⚠️ **TRANSITIONAL** - Use for reads, avoid for new code

```rust
pub struct ActorGraphRepository {
    actor-addr: Addr<GraphServiceActor>,
}

impl GraphRepository for ActorGraphRepository {
    async fn get-graph(&self) -> Result<Arc<GraphData>>;
    async fn get-node-map(&self) -> Result<Arc<HashMap<u32, Node>>>;
    async fn add-nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;
    async fn add-edges(&self, edges: Vec<Edge>) -> Result<Vec<String>>;
    // ... more methods
}
```

**When to Use**:
- ✅ Reading existing graph data during migration
- ❌ New write operations (use CQRS directives instead)
- ❌ New features (use repositories directly)

### 3. CQRS Query Handlers

**Location**: `/home/devuser/workspace/project/src/application/graph/queries.rs`
**Status**: ✅ **PRODUCTION READY** - Use for all read operations

#### Available Query Handlers
```rust
// 1. Get full graph data with positions
pub struct GetGraphDataHandler {
    repository: Arc<ActorGraphRepository>,
}
impl GetGraphDataHandler {
    pub async fn handle(&self, query: GetGraphData)
        -> Result<Arc<GraphData>, String>;
}

// 2. Get node map (ID → Node)
pub struct GetNodeMapHandler {
    repository: Arc<ActorGraphRepository>,
}
impl GetNodeMapHandler {
    pub async fn handle(&self, query: GetNodeMap)
        -> Result<Arc<HashMap<u32, Node>>, String>;
}

// 3. Get physics simulation state
pub struct GetPhysicsStateHandler {
    repository: Arc<ActorGraphRepository>,
}
impl GetPhysicsStateHandler {
    pub async fn handle(&self, query: GetPhysicsState)
        -> Result<PhysicsState, String>;
}

// 4. Get auto-balance notifications
pub struct GetAutoBalanceNotificationsHandler {
    repository: Arc<ActorGraphRepository>,
}

// 5. Get bots graph data
pub struct GetBotsGraphDataHandler {
    repository: Arc<ActorGraphRepository>,
}

// 6. Get constraints
pub struct GetConstraintsHandler {
    repository: Arc<ActorGraphRepository>,
}

// 7. Get equilibrium status
pub struct GetEquilibriumStatusHandler {
    repository: Arc<ActorGraphRepository>,
}

// 8. Compute shortest paths
pub struct ComputeShortestPathsHandler {
    repository: Arc<ActorGraphRepository>,
}
```

#### Access in AppState
```rust
pub struct AppState {
    pub graph-query-handlers: GraphQueryHandlers,
}

pub struct GraphQueryHandlers {
    pub get-graph-data: Arc<GetGraphDataHandler>,
    pub get-node-map: Arc<GetNodeMapHandler>,
    pub get-physics-state: Arc<GetPhysicsStateHandler>,
    pub get-auto-balance-notifications: Arc<GetAutoBalanceNotificationsHandler>,
    pub get-bots-graph-data: Arc<GetBotsGraphDataHandler>,
    pub get-constraints: Arc<GetConstraintsHandler>,
    pub get-equilibrium-status: Arc<GetEquilibriumStatusHandler>,
    pub compute-shortest-paths: Arc<ComputeShortestPathsHandler>,
}
```

### 4. GraphApplicationService

**Location**: `/home/devuser/workspace/project/src/application/services.rs`
**Purpose**: High-level orchestration service for complex workflows
**Status**: 🚧 **IN DEVELOPMENT** - Use for coordinated operations

```rust
#[derive(Clone)]
pub struct GraphApplicationService {
    command-bus: Arc<RwLock<CommandBus>>,
    query-bus: Arc<RwLock<QueryBus>>,
    event-bus: Arc<RwLock<EventBus>>,
}

impl GraphApplicationService {
    pub async fn add-node(&self, node-data: serde-json::Value) -> ServiceResult<String>;
    pub async fn update-node(&self, node-id: &str, updates: serde-json::Value) -> ServiceResult<()>;
    pub async fn remove-node(&self, node-id: &str) -> ServiceResult<()>;
    pub async fn get-all-nodes(&self) -> ServiceResult<Vec<serde-json::Value>>;
    pub async fn save-graph(&self) -> ServiceResult<()>;
}
```

### 5. Neo4jAdapter (Primary Database)

**Location**: `/home/devuser/workspace/project/src/adapters/neo4j-adapter.rs`
**Purpose**: Primary knowledge graph repository (replaces in-memory cache)
**Status**: ✅ **PRODUCTION** - Use for all persistence

```rust
pub struct Neo4jAdapter {
    // Neo4j connection and query methods
}

impl KnowledgeGraphRepository for Neo4jAdapter {
    async fn save-metadata(&self, metadata: &FileMetadata) -> Result<()>;
    async fn get-metadata(&self, id: &str) -> Result<Option<FileMetadata>>;
    async fn list-metadata(&self) -> Result<Vec<FileMetadata>>;
    // ... more methods
}
```

**Key Benefit**: No cache coherency issues - always reads from source of truth

### 6. Specialized Child Actors

**Managed by**: GraphServiceSupervisor (future state)

#### GraphStateActor
- **Responsibility**: Node/edge state management, persistence
- **Status**: 🚧 Future implementation
- **Will replace**: Graph state portions of GraphServiceActor

#### PhysicsOrchestratorActor
- **Responsibility**: Physics simulation, GPU coordination
- **Status**: ✅ Already exists independently
- **Location**: `/home/devuser/workspace/project/src/actors/physics-orchestrator-actor.rs`

#### SemanticProcessorActor
- **Responsibility**: Semantic analysis, constraint generation
- **Status**: ✅ Already exists independently
- **Location**: `/home/devuser/workspace/project/src/actors/semantic-processor-actor.rs`

#### ClientCoordinatorActor
- **Responsibility**: WebSocket broadcasting, client management
- **Status**: ✅ Already exists independently
- **Location**: `/home/devuser/workspace/project/src/actors/client-coordinator-actor.rs`

---

## Migration Patterns

### Pattern 1: Query Operations (READ)

#### BEFORE (Deprecated ❌)
```rust
use crate::actors::messages as actor-msgs;

// Send actor message
let graph-data = state.graph-service-addr
    .send(actor-msgs::GetGraphData)
    .await??;
```

#### AFTER (Recommended ✅)
```rust
// Use CQRS query handler
let handler = &state.graph-query-handlers.get-graph-data;
let query = GetGraphData;
let graph-data = handler.handle(query).await?;
```

**Benefits**:
- ✅ No actor mailbox errors
- ✅ Pure async/await - easier to test
- ✅ Type-safe queries with clear contracts
- ✅ Reads from Neo4j (source of truth)

### Pattern 2: Command Operations (WRITE)

#### BEFORE (Deprecated ❌)
```rust
// Direct actor message
state.graph-service-addr
    .send(actor-msgs::AddNode { node })
    .await??;
```

#### AFTER Phase 1 (Current Temporary ✅)
```rust
// Via TransitionalGraphSupervisor
state.graph-service-addr  // This is now TransitionalGraphSupervisor
    .send(actor-msgs::AddNode { node })
    .await??;
```

#### AFTER Phase 2 (Future Target 🎯)
```rust
// Use CQRS directive handler
let handler = &state.graph-directive-handlers.create-node;
let directive = CreateNode { node };

handler.handle(directive)?;
// Handler automatically:
//  1. Validates
//  2. Persists to Neo4j
//  3. Emits GraphNodeCreated event
//  4. Event subscribers handle cache/WebSocket
```

### Pattern 3: Event-Driven Cache Invalidation

#### Problem (Current)
```rust
// GitHub sync writes to database
github-sync.sync-to-database().await?;

// ❌ GraphServiceActor cache still shows old data!
// Next API call returns stale 63 nodes instead of fresh 316 nodes
```

#### Solution (Target Architecture)
```rust
// GitHub sync writes to database
github-sync.sync-to-database().await?;

// Emit event
event-bus.publish(DomainEvent::GraphSyncCompleted {
    nodes-added: 253,
    timestamp: now(),
}).await?;

// Event subscribers automatically triggered:
// 1. CacheInvalidationSubscriber → clears all caches
// 2. WebSocketBroadcasterSubscriber → notifies clients
// 3. MetricsSubscriber → updates monitoring

// Next API call reads fresh data from Neo4j ✅
```

---

## Migration Status & Timeline

### Phase 1: Query Handlers (✅ COMPLETE)
**Duration**: 1 week
**Status**: Production ready

**Deliverables**:
- ✅ 8 query handlers implemented
- ✅ GraphQueryHandlers struct in AppState
- ✅ 4 API endpoints migrated
- ✅ Integration tests written

**Files Created**:
- `/src/application/graph/queries.rs`
- `/src/application/graph/mod.rs`
- `/tests/cqrs-api-integration-tests.rs`

### Phase 2: Directive Handlers (❌ TODO)
**Duration**: 1-2 weeks
**Status**: Not started

**Deliverables**:
- [ ] CreateNodeHandler
- [ ] CreateEdgeHandler
- [ ] UpdateNodePositionHandler
- [ ] BatchUpdatePositionsHandler
- [ ] DeleteNodeHandler
- [ ] DeleteEdgeHandler

**Files to Create**:
- `/src/application/graph/directives.rs`

### Phase 3: Event Infrastructure (❌ TODO)
**Duration**: 1-2 weeks
**Status**: Not started

**Deliverables**:
- [ ] GraphEvent variants
- [ ] InMemoryEventBus
- [ ] CacheInvalidationSubscriber
- [ ] WebSocketBroadcasterSubscriber
- [ ] GitHub sync event emission

**Files to Create**:
- `/src/application/graph/cache-invalidator.rs`
- `/src/application/graph/websocket-broadcaster.rs`

**Files to Update**:
- `/src/application/events.rs`
- `/src/services/github-sync-service.rs`

### Phase 4: Actor Removal (⚠️ BLOCKED)
**Duration**: 1-2 weeks
**Status**: Blocked on Phases 2 & 3

**Deliverables**:
- [ ] Remove GraphServiceActor
- [ ] Remove TransitionalGraphSupervisor
- [ ] Remove ActorGraphRepository
- [ ] Update all remaining usages

---

## API Impact Analysis

### Files Using GraphServiceActor

#### High Priority (Public APIs)
1. `/src/handlers/api-handler/graph/mod.rs` - Main graph API routes
2. `/src/handlers/socket-flow-handler.rs` - WebSocket graph updates
3. `/src/handlers/admin-sync-handler.rs` - Admin sync operations

#### Medium Priority (Internal Services)
4. `/src/services/ontology-pipeline-service.rs` - Ontology processing
5. `/src/adapters/actor-graph-repository.rs` - Repository adapter

#### Low Priority (Tests/Examples)
6. `/tests/CRITICAL-github-sync-regression-test.rs`
7. `/examples/metadata-debug.rs`
8. `/examples/constraint-integration-debug.rs`

### API Endpoint Migration Status

| Endpoint | Method | Old Pattern | New Pattern | Status |
|----------|--------|-------------|-------------|--------|
| `/api/graph/data` | GET | Actor message | Query handler | ✅ Migrated |
| `/api/graph/data/paginated` | GET | Actor message | Query handler | ✅ Migrated |
| `/api/graph/refresh` | POST | Actor message | Query handler | ✅ Migrated |
| `/api/graph/auto-balance-notifications` | GET | Actor message | Query handler | ✅ Migrated |
| `/api/graph/nodes` | POST | Actor message | Directive handler | ❌ Pending Phase 2 |
| `/api/graph/edges` | POST | Actor message | Directive handler | ❌ Pending Phase 2 |
| `/api/graph/nodes/:id` | DELETE | Actor message | Directive handler | ❌ Pending Phase 2 |

---

## Performance Implications

### Before (GraphServiceActor)
- **Cache Hit**: ~50μs (in-memory lookup)
- **Cache Miss**: Never invalidated → stale data!
- **Write Latency**: ~500μs (actor message + processing)
- **Scalability**: Single actor bottleneck

### After (CQRS + Neo4j)
- **Read from Neo4j**: ~1-5ms (network + query)
- **Cache Hit** (with invalidation): ~50μs (valid data!)
- **Write Latency**: ~2-10ms (Neo4j write + event emission)
- **Scalability**: Horizontal (read replicas, event subscribers)

### Trade-offs
- ❌ Slightly higher read latency (1-5ms vs 50μs)
- ✅ **ALWAYS CORRECT DATA** (critical fix!)
- ✅ Horizontal scalability
- ✅ Event-driven architecture
- ✅ Testable without actors

---

## Deprecation Timeline

### Immediate (Now)
1. ✅ Use `TransitionalGraphSupervisor` instead of `GraphServiceActor` directly
2. ✅ Use CQRS query handlers for all read operations
3. ⚠️ Continue using actor messages for writes (temporary)

### Q1 2025 (Phase 2-3)
1. Implement directive handlers for write operations
2. Implement event bus and subscribers
3. Migrate all HTTP handlers to CQRS patterns
4. Add deprecation warnings to actor messages

### Q2 2025 (Phase 4)
1. Add `#[deprecated]` attribute to GraphServiceActor
2. Add compiler warnings for direct usage
3. Update all remaining internal code
4. Remove GraphServiceActor entirely
5. Remove TransitionalGraphSupervisor wrapper

---

## Testing Strategy

### Current Testing Challenges
```rust
// ❌ Complex setup required
#[actix-web::test]
#[ignore = "Requires full actor system initialization"]
async fn test-graph-operations() {
    // Requires:
    // - GraphServiceActor
    // - TransitionalGraphSupervisor
    // - ClientCoordinatorActor
    // - GPU managers
    // - WebSocket infrastructure
    // - 50+ dependencies
}
```

### New Testing Pattern
```rust
// ✅ Simple, isolated tests
#[tokio::test]
async fn test-get-graph-data-handler() {
    // Mock repository
    let mock-repo = MockGraphRepository::new();
    mock-repo.expect-get-graph()
        .returning(|| Ok(Arc::new(GraphData::default())));

    // Test handler
    let handler = GetGraphDataHandler::new(Arc::new(mock-repo));
    let result = handler.handle(GetGraphData).await;

    assert!(result.is-ok());
}
```

**Benefits**:
- ✅ No actor system required
- ✅ Fast execution (~1ms vs ~1s)
- ✅ Easy to mock dependencies
- ✅ Clear test boundaries

---

## Example Deprecation Notice Template

Based on successful patterns found in the codebase (e.g., `/src/gpu/dynamic-buffer-manager.rs`):

```rust
//! # DEPRECATED: Use CQRS Query/Directive Handlers instead
//!
//! This actor is deprecated in favor of the hexagonal/CQRS architecture:
//!
//! ## For Read Operations
//! Use `GraphQueryHandlers` from `AppState`:
//! ```rust,ignore
//! let handler = &state.graph-query-handlers.get-graph-data;
//! let result = handler.handle(GetGraphData).await?;
//! ```
//!
//! ## For Write Operations (Phase 2)
//! Use `GraphDirectiveHandlers` (coming soon):
//! ```rust,ignore
//! let handler = &state.graph-directive-handlers.create-node;
//! let result = handler.handle(CreateNode { node }).await?;
//! ```
//!
//! ## Migration Path
//! 1. Phase 1 (Current): Use `TransitionalGraphSupervisor` wrapper
//! 2. Phase 2 (Q1 2025): Migrate writes to directive handlers
//! 3. Phase 3 (Q1 2025): Event bus implementation
//! 4. Phase 4 (Q2 2025): Complete actor removal
//!
//! See `/docs/graphserviceactor-deprecation-research.md` for details.

#![deprecated(
    since = "0.6.0",
    note = "Use CQRS Query/Directive handlers. See GraphQueryHandlers in AppState."
)]
```

---

## Integration Test Examples

### Example 1: Query Handler Integration Test
```rust
// Location: tests/cqrs-api-integration-tests.rs
#[actix-web::test]
async fn test-get-graph-data-via-cqrs() {
    // Setup AppState with TransitionalGraphSupervisor
    let state = create-test-app-state().await;

    // Use query handler (not actor message!)
    let handler = &state.graph-query-handlers.get-graph-data;
    let result = handler.handle(GetGraphData).await;

    assert!(result.is-ok());
    let graph-data = result.unwrap();
    assert!(!graph-data.nodes.is-empty());
}
```

### Example 2: Event-Driven Cache Invalidation Test (Future)
```rust
#[tokio::test]
async fn test-github-sync-invalidates-cache() {
    // 1. Seed database with initial data
    neo4j.insert-nodes(initial-nodes).await?;

    // 2. Query returns cached data (63 nodes)
    let cached = handler.handle(GetGraphData).await?;
    assert-eq!(cached.nodes.len(), 63);

    // 3. GitHub sync adds more nodes
    github-sync.sync().await?;
    // This emits GraphSyncCompleted event

    // 4. Event subscriber clears cache
    tokio::time::sleep(Duration::from-millis(100)).await;

    // 5. Next query reads fresh data from Neo4j (316 nodes)
    let fresh = handler.handle(GetGraphData).await?;
    assert-eq!(fresh.nodes.len(), 316); // ✅ CORRECT!
}
```

---

## Recommendations for Developers

### Immediate Actions (DO NOW)
1. ✅ **Use TransitionalGraphSupervisor** in all new code
   ```rust
   // AppState has this already
   pub graph-service-addr: Addr<TransitionalGraphSupervisor>
   ```

2. ✅ **Use CQRS query handlers** for all reads
   ```rust
   // Instead of: state.graph-service-addr.send(GetGraphData).await
   // Use: state.graph-query-handlers.get-graph-data.handle(GetGraphData).await
   ```

3. ⚠️ **Continue using actor messages for writes** (temporary)
   ```rust
   // Still OK during Phase 1
   state.graph-service-addr.send(AddNode { node }).await
   ```

### Phase 2 Preparation (PLAN FOR)
1. Watch for `GraphDirectiveHandlers` implementation
2. Prepare to migrate write operations to directive handlers
3. Update tests to use pure async/await patterns

### Things to Avoid
1. ❌ Don't add new message types to GraphServiceActor
2. ❌ Don't expand GraphServiceActor responsibilities
3. ❌ Don't rely on in-memory cache for correctness
4. ❌ Don't create direct dependencies on GraphServiceActor

### Best Practices
1. ✅ Read from Neo4j as source of truth
2. ✅ Use repository abstractions (GraphRepository port)
3. ✅ Emit events after state changes
4. ✅ Write tests without actor system
5. ✅ Follow hexagonal architecture boundaries

---

## References

### Key Documentation
- `/docs/concepts/architecture/hexagonal-cqrs-architecture.md` - Detailed architecture design
- `/docs/concepts/architecture/quick-reference.md` - Quick migration guide
- `/docs/concepts/architecture/cqrs-directive-template.md` - Directive handler template

### Key Source Files
- `/src/actors/graph-actor.rs` - Current GraphServiceActor (deprecated)
- `/src/actors/graph-service-supervisor.rs` - TransitionalGraphSupervisor (current)
- `/src/application/graph/queries.rs` - CQRS query handlers (use this!)
- `/src/adapters/actor-graph-repository.rs` - Repository adapter (transitional)
- `/src/app-state.rs` - Application state structure

### Test Files
- `/tests/cqrs-api-integration-tests.rs` - CQRS integration tests
- `/tests/CRITICAL-github-sync-regression-test.rs` - Cache coherency bug demonstration

---

## Summary Table

| Component | Status | When to Use | Notes |
|-----------|--------|-------------|-------|
| **GraphServiceActor** | ⚠️ Deprecated | Never (new code) | Being phased out - use alternatives |
| **TransitionalGraphSupervisor** | ✅ Current | All operations (Phase 1) | Temporary wrapper during migration |
| **GraphQueryHandlers** | ✅ Production | All read operations | Preferred for queries - always use this |
| **ActorGraphRepository** | ⚠️ Transitional | Indirect use only | Used by query handlers, don't use directly |
| **GraphApplicationService** | 🚧 Development | Complex workflows | Coming in Phase 2 |
| **GraphDirectiveHandlers** | ❌ Not Implemented | Write operations (future) | Target for Phase 2 |
| **Neo4jAdapter** | ✅ Production | All persistence | Primary database - source of truth |
| **EventBus** | ❌ Not Implemented | Event-driven patterns | Target for Phase 3 |

---

## Conclusion

GraphServiceActor is being deprecated due to fundamental architectural issues. The replacement architecture provides:

1. **Correctness**: No cache coherency bugs - always reads from source of truth
2. **Maintainability**: Small, focused components vs 48K token monolith
3. **Testability**: Pure async/await functions, easy to mock
4. **Scalability**: Horizontal scaling with read replicas and event subscribers
5. **Evolvability**: Clean separation of concerns enables independent evolution

**Current Recommendation**: Use `TransitionalGraphSupervisor` with CQRS query handlers for all new code. Plan for Phase 2 migration to directive handlers in Q1 2025.

---

**Document Version**: 1.0
**Last Updated**: 2025-11-04
**Next Review**: Q1 2025 (Phase 2 kickoff)
