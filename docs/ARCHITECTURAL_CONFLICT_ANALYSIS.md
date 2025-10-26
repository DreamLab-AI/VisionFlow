# Architectural Conflict Analysis
## Three Competing Architectures in WebXR Knowledge Graph

**Date:** 2025-10-26
**Severity:** HIGH - Root cause of code duplication and complexity
**Status:** DOCUMENTED - Requires architectural decision

---

## Executive Summary

The codebase contains **three parallel and conflicting architectural patterns** for managing core application logic. This architectural duplication is the **root cause** of:

- Duplicated business logic across multiple layers
- Isolated and deprecated code paths
- Complexity in data flow and state management
- Difficulty in understanding which components are authoritative

---

## Architecture 1: Monolithic GraphServiceActor (Legacy)

### Location
`src/actors/graph_actor.rs` (2,501 lines)

### Characteristics
- **Pattern:** Large, centralized actor model
- **Responsibilities:**
  - Graph state management
  - Physics simulation
  - Client WebSocket communication
  - Semantic analysis
  - Real-time updates
  - Position broadcasting

### Code Evidence
```rust
pub struct GraphServiceActor {
    graph_data: Arc<RwLock<GraphData>>,
    clients: Arc<RwLock<HashMap<String, ClientConnection>>>,
    settings: Arc<RwLock<Settings>>,
    simulation_params: SimulationParams,
    physics_orchestrator_addr: Option<Addr<PhysicsOrchestratorActor>>,
    semantic_processor_addr: Option<Addr<SemanticProcessorActor>>,
    // ... many more fields
}
```

### Strengths
- **Mature:** Well-tested, production-ready
- **Integrated:** Handles full request/response lifecycle
- **Real-time:** Direct WebSocket communication with clients

### Weaknesses
- **Monolithic:** 2,501 lines in single file
- **Tight coupling:** Hard to test, modify, or extend
- **Mixed concerns:** Business logic intertwined with infrastructure
- **Scalability:** Single point of contention for all operations

### Current Usage
- **Active:** Used by main WebXR application
- **Entry point:** `src/main.rs` → GraphServiceActor
- **API routes:** Delegates to this actor via Actix messages

---

## Architecture 2: Modular GPU Supervisor Model (Transitional)

### Location
`src/actors/gpu/gpu_manager_actor.rs` + child actors

### Characteristics
- **Pattern:** Hierarchical actor supervision tree
- **Responsibilities:**
  - Delegates to specialized child actors
  - Physics → ForceComputeActor
  - Clustering → ClusteringActor
  - Anomaly Detection → AnomalyDetectionActor
  - GPU resource management → GPUResourceActor

### Code Evidence
```rust
pub struct GPUManagerActor {
    force_compute_actor: Option<Addr<ForceComputeActor>>,
    clustering_actor: Option<Addr<ClusteringActor>>,
    anomaly_detection_actor: Option<Addr<AnomalyDetectionActor>>,
    gpu_resource_actor: Option<Addr<GPUResourceActor>>,
    // ...
}
```

### Strengths
- **Modular:** Clear separation of concerns
- **Parallel:** Child actors run independently
- **Testable:** Each actor can be tested in isolation
- **Scalable:** Can distribute across threads/processes

### Weaknesses
- **Incomplete:** Not fully integrated with main application flow
- **Coordination overhead:** Message passing between actors
- **Complexity:** Requires understanding actor lifecycle
- **Transitional:** Unclear which paths use this vs. legacy actor

### Current Usage
- **Partially integrated:** Some code paths use GPU actors
- **Coexists:** Runs alongside GraphServiceActor
- **Selective:** Used for specific GPU-accelerated operations

---

## Architecture 3: CQRS/Hexagonal Architecture (Modern)

### Location
- **Application Layer:** `src/application/`
- **Domain Ports:** `src/ports/`
- **Infrastructure Adapters:** `src/adapters/`

### Characteristics
- **Pattern:** Clean Architecture / Hexagonal Architecture
- **Responsibilities:**
  - Command/Query separation (CQRS)
  - Repository ports for data access
  - Adapter pattern for infrastructure
  - Domain-driven design principles

### Code Evidence

**Ports (Interfaces):**
```rust
// src/ports/knowledge_graph_repository.rs
#[async_trait]
pub trait KnowledgeGraphRepository: Send + Sync {
    async fn save_graph(&self, nodes: Vec<Node>, edges: Vec<Edge>) -> Result<(), String>;
    async fn get_graph(&self) -> Result<(Vec<Node>, Vec<Edge>), String>;
}
```

**Adapters (Implementations):**
```rust
// src/adapters/sqlite_knowledge_graph_repository.rs
pub struct SqliteKnowledgeGraphRepository {
    pool: Arc<r2d2::Pool<r2d2_sqlite::SqliteConnectionManager>>,
}

#[async_trait]
impl KnowledgeGraphRepository for SqliteKnowledgeGraphRepository {
    async fn save_graph(&self, nodes: Vec<Node>, edges: Vec<Edge>) -> Result<(), String> {
        // SQLite implementation
    }
}
```

**Application Layer (Use Cases):**
```rust
// src/application/ontology/directives.rs
pub struct CreateOntologyGraphHandler {
    repository: Arc<dyn OntologyRepository>,
}

impl CreateOntologyGraphHandler {
    pub async fn execute(&self, directive: CreateOntologyGraph) -> Result<String> {
        // Business logic isolated from infrastructure
    }
}
```

### Strengths
- **Clean:** Clear separation between domain, application, and infrastructure
- **Testable:** Can mock repositories via trait objects
- **Flexible:** Easy to swap implementations (SQLite → PostgreSQL)
- **Maintainable:** Changes to infrastructure don't affect business logic

### Weaknesses
- **Incomplete:** Not integrated with main application flow
- **Parallel implementation:** Duplicate logic exists in GraphServiceActor
- **Unused:** Many handlers and queries not called by API routes
- **Disconnected:** Doesn't participate in real-time updates

### Current Usage
- **Limited:** Primarily used for ontology validation operations
- **Standalone:** GitHub sync service uses this pattern (our recent fix!)
- **Future-ready:** Intended architecture but not fully adopted

---

## Architectural Conflict Analysis

### Data Flow Confusion

**Example: Knowledge Graph Updates**

**Path 1 (GraphServiceActor):**
```
HTTP Request → API Handler → GraphServiceActor Message →
GraphServiceActor internal state → SQLite save → WebSocket broadcast
```

**Path 2 (CQRS/Hexagonal):**
```
HTTP Request → Application Handler → Repository Port →
SQLite Adapter → Response (NO real-time updates!)
```

**Path 3 (GitHub Sync - Our Fix):**
```
GitHub API → github_sync_service.rs → KnowledgeGraphParser →
SqliteKnowledgeGraphRepository (Hexagonal) → Database
(SEPARATE from GraphServiceActor state!)
```

### State Inconsistency Problem

The **same data** is managed by **multiple systems**:

1. **GraphServiceActor** maintains in-memory `graph_data: Arc<RwLock<GraphData>>`
2. **CQRS handlers** write to database via repository ports
3. **GitHub sync** writes to database independently

**Result:** GraphServiceActor's in-memory state can become stale or inconsistent with database!

### Code Duplication Examples

**Graph Data Structures:**
- `models/node.rs` - Domain models
- GraphServiceActor internal structures
- Separate graph representations in GPU actors

**Database Access:**
- GraphServiceActor: Direct SQLite connection
- CQRS adapters: Repository pattern with r2d2 pool
- Services: Mixed approaches

**Business Logic:**
- Semantic analysis in GraphServiceActor
- Semantic analysis in SemanticProcessorActor (GPU model)
- Ontology validation in application handlers

---

## Impact on Recent Privacy Bug Fix

### Why Three Architectures Complicated the Fix

1. **GitHub sync uses CQRS pattern** (`src/services/github_sync_service.rs`)
   - Uses `SqliteKnowledgeGraphRepository` (Hexagonal adapter)
   - Writes directly to database
   - **BYPASSES GraphServiceActor!**

2. **GraphServiceActor loads data separately**
   - Has its own initialization logic
   - May cache stale data
   - Doesn't automatically reload after GitHub sync

3. **API endpoints use GraphServiceActor state**
   - `GET /api/graph/data` returns in-memory data
   - This is why we saw **old cached data (188 nodes)** after our fix!

### The Real Problem

When GitHub sync runs with our fixed code:
1. ✅ Correctly parses and filters nodes (316 expected)
2. ✅ Saves to database via `SqliteKnowledgeGraphRepository`
3. ❌ **GraphServiceActor still has old in-memory state (188 nodes)**
4. ❌ API returns old data from GraphServiceActor, not database

**This is why validation showed 46.3% despite the fix being correct!**

---

## Architectural Decision Required

### Option 1: Commit to Monolithic GraphServiceActor (Short-term)

**Approach:**
- Accept GraphServiceActor as authoritative source
- Remove CQRS/Hexagonal layer (except as data access)
- Migrate all business logic to actor model

**Pros:**
- Minimal changes
- Works with existing real-time infrastructure
- Clear authority

**Cons:**
- Maintains tight coupling
- Difficult to test
- Scalability limitations

---

### Option 2: Complete CQRS/Hexagonal Migration (Long-term)

**Approach:**
- Make CQRS layer the authoritative source
- Refactor GraphServiceActor into thin orchestrator
- Use event sourcing for real-time updates
- Separate read (query) and write (command) models

**Pros:**
- Clean architecture
- Highly testable
- Flexible and maintainable
- Industry best practice

**Cons:**
- Significant refactoring effort
- May break existing integrations
- Requires architectural expertise

---

### Option 3: Hybrid Approach with Clear Boundaries (Pragmatic)

**Approach:**
- **GraphServiceActor:** Real-time UI state and WebSocket communication
- **CQRS Layer:** Data persistence and batch operations (like GitHub sync)
- **Event bus:** Synchronize between the two via domain events

**Implementation:**
```rust
// After GitHub sync completes
pub async fn sync_graphs(&self) -> Result<SyncStatistics, String> {
    // 1. Sync to database (CQRS)
    self.repository.save_graph(nodes, edges).await?;

    // 2. Notify GraphServiceActor to reload (Event)
    self.event_bus.publish(GraphDataUpdated {
        source: "github_sync",
        node_count: nodes.len(),
    }).await;

    Ok(stats)
}

// In GraphServiceActor
impl Handler<GraphDataUpdated> for GraphServiceActor {
    fn handle(&mut self, msg: GraphDataUpdated, _ctx: &mut Context<Self>) -> Self::Result {
        // Reload from database
        self.reload_graph_from_database().await;

        // Broadcast to clients
        self.broadcast_graph_update().await;
    }
}
```

**Pros:**
- Leverages existing systems
- Clear separation of concerns
- Incremental migration path
- Maintains real-time functionality

**Cons:**
- More complex than pure approaches
- Requires careful event design
- Potential for event ordering issues

---

## Recommended Path Forward

### Immediate (This Sprint)

1. **Document current state** ✅ (this document)
2. **Add event bus for sync notification**
   - GitHub sync → event → GraphServiceActor reload
   - Fixes the "old cached data" problem immediately
3. **Add integration tests**
   - Verify GraphServiceActor state matches database after sync

### Short-term (Next Sprint)

4. **Establish clear boundaries**
   - Document which operations use which architecture
   - API routes → explicit routing to correct layer
5. **Deprecate duplicate code**
   - Mark deprecated paths with `#[deprecated]`
   - Add logging to identify unused code paths

### Long-term (Next Quarter)

6. **Incremental CQRS migration**
   - Move read-only operations to query handlers
   - Keep writes in GraphServiceActor temporarily
   - Gradually extract business logic
7. **Event sourcing for real-time**
   - Replace WebSocket broadcasts with event streams
   - GraphServiceActor becomes event subscriber
8. **Phase out legacy actor**
   - Final migration of remaining logic
   - GraphServiceActor becomes thin WebSocket gateway

---

## Files Affected by Architectural Conflict

### Legacy Monolithic Pattern
- `src/actors/graph_actor.rs` (2,501 lines)
- `src/actors/physics_orchestrator_actor.rs`
- `src/actors/supervisor.rs`

### Modular GPU Pattern
- `src/actors/gpu/gpu_manager_actor.rs`
- `src/actors/gpu/force_compute_actor.rs`
- `src/actors/gpu/clustering_actor.rs`
- `src/actors/gpu/anomaly_detection_actor.rs`

### CQRS/Hexagonal Pattern
- `src/application/ontology/directives.rs`
- `src/application/ontology/queries.rs`
- `src/ports/knowledge_graph_repository.rs`
- `src/ports/ontology_repository.rs`
- `src/adapters/sqlite_knowledge_graph_repository.rs`
- `src/adapters/sqlite_ontology_repository.rs`

### Services (Mixed Pattern)
- `src/services/github_sync_service.rs` (uses Hexagonal)
- `src/services/local_markdown_sync.rs` (uses Hexagonal)
- `src/services/parsers/knowledge_graph_parser.rs` (domain logic)

---

## Metrics of Architectural Debt

### Code Duplication
- **3 different graph data structures** (models, actor state, GPU state)
- **3 database access patterns** (direct, pooled, repository)
- **2 semantic analysis implementations** (actor, GPU actor)

### Complexity
- **GraphServiceActor:** 2,501 lines (should be <500)
- **Total actor files:** 15 files, ~8,000 lines
- **Hexagonal layer:** 12 files, mostly unused

### Testing Gap
- **GraphServiceActor:** Difficult to test (integration tests only)
- **CQRS handlers:** Easily testable but not used
- **Integration between architectures:** No tests

---

## Conclusion

The presence of three competing architectures is the **root cause** of:
- ✅ Our difficulty validating the privacy bug fix (old cached data in actor)
- ✅ Code duplication and maintenance burden
- ✅ Confusion about which components are authoritative
- ✅ Testing challenges

**Recommended immediate action:**
Implement **hybrid approach with event bus** to synchronize GraphServiceActor state with database after operations like GitHub sync. This provides immediate value while establishing a migration path toward cleaner architecture.

**Long-term vision:**
Complete migration to CQRS/Hexagonal architecture with event sourcing for real-time updates.

---

**Document Status:** DRAFT
**Requires:** Architectural decision from tech lead
**Next Steps:** Team discussion and decision on path forward
