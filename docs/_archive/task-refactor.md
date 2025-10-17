# Actor System Refactoring Plan
**Created:** 2025-10-16
**Status:** Planning
**Target:** GraphService decomposition + Hexagonal Architecture with Hexser

---

## Executive Summary

This plan decomposes the monolithic `GraphServiceActor`/`GraphServiceSupervisor` into specialized actors following Hexagonal Architecture principles, with optional Hexser integration for formalization. The refactor addresses critical architectural issues identified in the codebase audit while positioning the system for AI-ready introspection.

**Key Objectives:**
1. Decompose GraphServiceSupervisor into 4 specialized actors
2. Apply Hexagonal Architecture (Ports & Adapters)
3. Optionally integrate Hexser for compile-time validation and AI blueprint generation
4. Improve testability, maintainability, and architectural clarity

**Expected Benefits:**
- **Separation of Concerns:** Each actor has single responsibility
- **Testability:** Ports (traits) enable easy mocking
- **AI Integration:** Machine-readable architecture via HexGraph
- **Compile-Time Safety:** Architectural rules enforced by compiler
- **Reduced Boilerplate:** Declarative component wiring

---

## Current State Analysis

### Problem: Monolithic GraphServiceSupervisor

**File:** `src/actors/graph_service_supervisor.rs` (currently ~2000 lines)
**Status:** All message handlers return "Supervisor not yet fully implemented" (lines 735-789)

**Current Responsibilities** (all in one actor):
1. **State Management:** Graph data storage, node/edge tracking
2. **Physics Coordination:** GPU kernel orchestration, force calculations
3. **Semantic Processing:** SSSP, clustering, community detection
4. **Client Communication:** WebSocket broadcasting, state synchronization

**Issues:**
- ❌ Violates Single Responsibility Principle
- ❌ Difficult to test (requires full actor system setup)
- ❌ No clear separation between domain logic and infrastructure
- ❌ Implicit dependencies make reasoning difficult
- ❌ Supervisor pattern incomplete/non-functional

### Current Architecture (Implicit Ports & Adapters)

**Domain Layer:**
- `src/models/graph.rs`, `node.rs`, `edge.rs` - Core data structures
- `src/physics/` - Physics simulation algorithms

**Application Layer:**
- `src/actors/` - Actor system (currently monolithic)

**Infrastructure Layer:**
- `src/handlers/` - Actix-Web HTTP/WebSocket handlers
- `src/gpu/` - CUDA kernel adapters
- `src/services/` - External API clients

**Problem:** Layers exist but boundaries are implicit and not enforced.

---

## Target Architecture

### Phase 1: Actor Decomposition (No Hexser)

Break GraphServiceSupervisor into 4 specialized actors:

#### 1. GraphStateActor
**Responsibility:** Pure state management
**File:** `src/actors/graph_state_actor.rs` (new)

**State:**
```rust
struct GraphStateActor {
    graph_data: Arc<RwLock<GraphData>>,      // Current graph state
    node_index: HashMap<u32, NodeHandle>,     // Fast node lookup
    edge_index: HashMap<u32, EdgeHandle>,     // Fast edge lookup
    dirty_nodes: HashSet<u32>,                // Changed nodes since last sync
    version: u64,                             // Optimistic locking version
}
```

**Messages:**
- `GetGraphData` → `Result<Arc<GraphData>>`
- `AddNodes` → `Result<Vec<u32>>` (returns new node IDs)
- `AddEdges` → `Result<Vec<u32>>` (returns new edge IDs)
- `UpdateNodePositions` → `Result<()>`
- `GetDirtyNodes` → `Result<HashSet<u32>>`
- `ClearDirtyNodes` → `Result<()>`
- `GetGraphVersion` → `u64`

**No Dependencies:** Pure state management, no external actors

---

#### 2. PhysicsOrchestratorActor
**Responsibility:** Physics simulation coordination
**File:** `src/actors/physics_orchestrator_actor.rs` (enhance existing)

**State:**
```rust
struct PhysicsOrchestratorActor {
    graph_state: Addr<GraphStateActor>,       // State source
    gpu_manager: Option<Addr<GPUManagerActor>>, // GPU compute
    spring_params: SpringParams,
    sssp_source: Option<u32>,
    simulation_running: bool,
    tick_count: u64,
}
```

**Messages:**
- `StartSimulation` → `Result<()>`
- `StopSimulation` → `Result<()>`
- `SimulationTick` → `Result<PhysicsUpdate>` (internal)
- `UpdatePhysicsParams` → `Result<()>`
- `SetSSSPSource` → `Result<()>`
- `ApplyConstraints` → `Result<()>` (from OntologyActor)

**Dependencies:**
- `GraphStateActor` - reads graph, writes position updates
- `GPUManagerActor` - delegates force calculations

---

#### 3. SemanticProcessorActor
**Responsibility:** Graph algorithms & semantic analysis
**File:** `src/actors/semantic_processor_actor.rs` (new)

**State:**
```rust
struct SemanticProcessorActor {
    graph_state: Addr<GraphStateActor>,
    gpu_manager: Option<Addr<GPUManagerActor>>,
    clustering_cache: HashMap<String, ClusteringResult>,
    community_cache: HashMap<String, CommunityResult>,
}
```

**Messages:**
- `RunSSSP { source: u32 }` → `Result<SSSPResult>`
- `RunClustering { algorithm: ClusterAlgo }` → `Result<ClusteringResult>`
- `DetectCommunities` → `Result<CommunityResult>`
- `GetShortestPath { source: u32, target: u32 }` → `Result<Vec<u32>>`
- `InvalidateCache` → `Result<()>`

**Dependencies:**
- `GraphStateActor` - reads graph structure
- `GPUManagerActor` - delegates algorithm execution

---

#### 4. ClientCoordinatorActor
**Responsibility:** Client communication & state synchronization
**File:** `src/actors/client_coordinator_actor.rs` (enhance existing)

**State:**
```rust
struct ClientCoordinatorActor {
    graph_state: Addr<GraphStateActor>,
    physics_orchestrator: Addr<PhysicsOrchestratorActor>,
    semantic_processor: Addr<SemanticProcessorActor>,
    connected_clients: HashMap<Uuid, ClientSession>,
    broadcast_interval: Duration,
}
```

**Messages:**
- `RegisterClient { session_id: Uuid, addr: Addr<...> }` → `Result<()>`
- `UnregisterClient { session_id: Uuid }` → `Result<()>`
- `BroadcastGraphUpdate` → `Result<()>` (periodic)
- `SendToClient { session_id: Uuid, data: Vec<u8> }` → `Result<()>`
- `GetConnectedClients` → `Vec<ClientInfo>`

**Dependencies:**
- `GraphStateActor` - reads dirty nodes for efficient updates
- `PhysicsOrchestratorActor` - forwards physics control messages
- `SemanticProcessorActor` - forwards algorithm requests

---

### Phase 2: Define Ports (Traits)

Extract interfaces for testability and swappable implementations:

#### GraphRepository Port
```rust
// src/ports/graph_repository.rs
#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_graph(&self) -> Result<Arc<GraphData>>;
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>>;
    async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<u32>>;
    async fn update_positions(&self, updates: Vec<(u32, BinaryNodeData)>) -> Result<()>;
    async fn get_dirty_nodes(&self) -> Result<HashSet<u32>>;
}
```

**Implementations:**
- `ActorGraphRepository` - wraps `Addr<GraphStateActor>`
- `MockGraphRepository` - for testing

#### PhysicsSimulator Port
```rust
// src/ports/physics_simulator.rs
#[async_trait]
pub trait PhysicsSimulator: Send + Sync {
    async fn run_simulation_step(&self, graph: &GraphData) -> Result<Vec<(u32, BinaryNodeData)>>;
    async fn update_params(&self, params: SimulationParams) -> Result<()>;
    async fn apply_constraints(&self, constraints: Vec<Constraint>) -> Result<()>;
}
```

**Implementations:**
- `GpuPhysicsSimulator` - wraps GPU actors
- `CpuPhysicsSimulator` - CPU fallback (to be implemented)
- `MockPhysicsSimulator` - for testing

#### SemanticAnalyzer Port
```rust
// src/ports/semantic_analyzer.rs
#[async_trait]
pub trait SemanticAnalyzer: Send + Sync {
    async fn run_sssp(&self, graph: &GraphData, source: u32) -> Result<SSSPResult>;
    async fn run_clustering(&self, graph: &GraphData, algo: ClusterAlgo) -> Result<ClusteringResult>;
    async fn detect_communities(&self, graph: &GraphData) -> Result<CommunityResult>;
}
```

**Implementations:**
- `GpuSemanticAnalyzer` - GPU-accelerated
- `CpuSemanticAnalyzer` - CPU fallback
- `MockSemanticAnalyzer` - for testing

---

### Phase 3: Hexser Integration (Optional but Recommended)

Add Hexser attributes to formalize architecture and enable AI introspection:

#### Domain Layer Tagging
```rust
// src/models/node.rs
#[derive(HexDomain, Entity, Clone, Debug)]
pub struct Node {
    pub id: u32,
    pub label: String,
    pub node_type: NodeType,
    pub properties: HashMap<String, String>,
}

// src/models/edge.rs
#[derive(HexDomain, Entity, Clone, Debug)]
pub struct Edge {
    pub id: u32,
    pub source: u32,
    pub target: u32,
    pub edge_type: EdgeType,
}
```

#### Port Tagging
```rust
// src/ports/graph_repository.rs
#[derive(HexPort)]
#[async_trait]
pub trait GraphRepository: Send + Sync {
    /* ... */
}

// src/ports/physics_simulator.rs
#[derive(HexPort)]
#[async_trait]
pub trait PhysicsSimulator: Send + Sync {
    /* ... */
}
```

#### Adapter Tagging
```rust
// src/adapters/actor_graph_repository.rs
#[derive(HexAdapter)]
pub struct ActorGraphRepository {
    graph_state: Addr<GraphStateActor>,
}

impl GraphRepository for ActorGraphRepository {
    /* ... */
}

// src/adapters/gpu_physics_adapter.rs
#[derive(HexAdapter)]
pub struct GpuPhysicsAdapter {
    physics_orchestrator: Addr<PhysicsOrchestratorActor>,
    gpu_manager: Option<Addr<GPUManagerActor>>,
}

impl PhysicsSimulator for GpuPhysicsAdapter {
    /* ... */
}
```

#### Application Service Tagging
```rust
// src/services/graph_service.rs
#[derive(HexService)]
pub struct GraphService {
    graph_repo: Arc<dyn GraphRepository>,
    physics_sim: Arc<dyn PhysicsSimulator>,
    semantic_analyzer: Arc<dyn SemanticAnalyzer>,
}

impl GraphService {
    pub async fn update_and_simulate(&self) -> Result<()> {
        let graph = self.graph_repo.get_graph().await?;
        let updates = self.physics_sim.run_simulation_step(&graph).await?;
        self.graph_repo.update_positions(updates).await?;
        Ok(())
    }
}
```

#### Benefits of Hexser Integration

1. **Compile-Time Architecture Validation:**
```rust
// Enforced at compile time via HexGraph
#[cfg(test)]
mod architecture_tests {
    #[test]
    fn domain_cannot_depend_on_infrastructure() {
        // Automatically validated by Hexser
        // src/models/* cannot import from src/handlers/* or src/gpu/*
    }
}
```

2. **Exported HexGraph for AI Agents:**
```json
{
  "nodes": [
    {"id": "GraphStateActor", "layer": "application"},
    {"id": "ActorGraphRepository", "layer": "adapter", "implements": "GraphRepository"},
    {"id": "GraphRepository", "layer": "port"},
    {"id": "Node", "layer": "domain"}
  ],
  "edges": [
    {"from": "ActorGraphRepository", "to": "GraphStateActor", "type": "uses"},
    {"from": "ActorGraphRepository", "to": "GraphRepository", "type": "implements"}
  ]
}
```

AI agents can consume this to:
- Understand data flow from WebSocket → Handler → Actor → GPU
- Identify bottlenecks or architectural violations
- Suggest refactorings or optimizations

3. **Actionable Errors with Context:**
```rust
// Instead of generic error
Err("Failed to update graph")

// Hexserror provides architectural context
Err(
    hexser::hex_application_error!(
        hexser::error::codes::application::STATE_UPDATE_FAILED,
        "GraphStateActor failed to apply position updates"
    )
    .with_layer_context("application", "GraphStateActor")
    .with_next_steps(&[
        "Check if GraphStateActor is running",
        "Verify graph_data lock is not deadlocked",
        "Check for version conflicts (optimistic locking)"
    ])
    .with_suggestions(&[
        "Run: cargo test -- graph_state_actor",
        "Check logs for deadlock warnings"
    ])
)
```

---

## Implementation Phases

### Phase 0: Preparation (Week 1)
**Goal:** Set up foundation without breaking existing system

**Tasks:**
1. ✅ Complete legacy code removal (Binary Protocol V1)
2. Create `/task-refactor.md` (this document)
3. Add Hexser to Cargo.toml (optional feature flag)
4. Create directory structure:
```
src/
├── ports/          # New: trait definitions
├── adapters/       # New: trait implementations
├── actors/         # Enhanced: decomposed actors
└── services/       # Enhanced: application services
```
5. Document current GraphServiceSupervisor behavior for reference
6. Set up integration tests to lock current behavior

**Deliverable:** Architecture ready for refactoring, no functionality changes

---

### Phase 1: GraphStateActor Creation (Week 2)
**Goal:** Extract pure state management

**Tasks:**
1. Create `GraphStateActor` with basic CRUD operations
2. Implement message handlers:
   - `GetGraphData`
   - `AddNodes`, `AddEdges`
   - `UpdateNodePositions`
3. Add optimistic locking with version tracking
4. Implement dirty node tracking for efficient updates
5. Write unit tests for state management logic
6. Create `GraphRepository` port trait
7. Implement `ActorGraphRepository` adapter
8. Optionally add `#[derive(HexDomain)]` to models
9. Optionally add `#[derive(HexPort)]` to GraphRepository

**Migration Path:**
- Start GraphStateActor alongside existing GraphServiceSupervisor
- Gradually redirect state operations to GraphStateActor
- No client-visible changes

**Deliverable:** Functional GraphStateActor with 100% test coverage

---

### Phase 2: PhysicsOrchestratorActor Enhancement (Week 3)
**Goal:** Consolidate physics coordination

**Tasks:**
1. Enhance existing `PhysicsOrchestratorActor`:
   - Add `graph_state: Addr<GraphStateActor>` dependency
   - Remove local state (delegate to GraphStateActor)
2. Implement simulation loop:
   - Read graph from GraphStateActor
   - Compute forces via GPUManagerActor
   - Write position updates back to GraphStateActor
3. Add constraint application from OntologyActor
4. Create `PhysicsSimulator` port trait
5. Implement `GpuPhysicsAdapter`
6. Write integration tests (actor → adapter → port)
7. Optionally add `#[derive(HexPort)]` to PhysicsSimulator
8. Optionally add `#[derive(HexAdapter)]` to GpuPhysicsAdapter

**Migration Path:**
- PhysicsOrchestratorActor now coordinates via GraphStateActor
- GPU operations unchanged
- No client-visible changes

**Deliverable:** Physics decoupled from state management

---

### Phase 3: SemanticProcessorActor Creation (Week 4)
**Goal:** Extract semantic algorithms

**Tasks:**
1. Create `SemanticProcessorActor`:
   - Add `graph_state: Addr<GraphStateActor>` dependency
   - Add `gpu_manager: Option<Addr<GPUManagerActor>>` for compute
2. Implement algorithm handlers:
   - `RunSSSP`
   - `RunClustering` (fix Issue #4 from audit - wire to existing impls)
   - `DetectCommunities`
3. Add caching layer for expensive algorithms
4. Create `SemanticAnalyzer` port trait
5. Implement `GpuSemanticAnalyzer` adapter
6. Write algorithm tests with mock graph data
7. Optionally add `#[derive(HexPort)]` to SemanticAnalyzer
8. Optionally add `#[derive(HexAdapter)]` to GpuSemanticAnalyzer

**Critical Fix:** Wire ClusteringActor handlers to existing implementations (addresses Audit Issue #4)

**Migration Path:**
- Semantic operations route through SemanticProcessorActor
- Existing GPU clustering code now accessible via proper handlers
- No client-visible changes

**Deliverable:** Semantic processing isolated and testable

---

### Phase 4: ClientCoordinatorActor Enhancement (Week 5)
**Goal:** Consolidate client communication

**Tasks:**
1. Enhance existing `ClientCoordinatorActor`:
   - Add `graph_state: Addr<GraphStateActor>`
   - Add `physics_orchestrator: Addr<PhysicsOrchestratorActor>`
   - Add `semantic_processor: Addr<SemanticProcessorActor>`
2. Implement efficient broadcast:
   - Use `GetDirtyNodes` from GraphStateActor
   - Only send changed nodes (not full graph)
   - Binary protocol encoding (V2 only now)
3. Add message forwarding:
   - Physics control → PhysicsOrchestratorActor
   - Algorithm requests → SemanticProcessorActor
4. Implement session management
5. Add backpressure handling (addresses Audit Issue #9)
6. Write WebSocket communication tests (addresses Test Gap #2)

**Migration Path:**
- Client communication now goes through ClientCoordinatorActor
- Handlers delegate to specialized actors
- Client-visible change: More efficient updates (dirty nodes only)

**Deliverable:** Client communication layer complete

---

### Phase 5: GraphServiceSupervisor Removal (Week 6)
**Goal:** Remove monolithic actor

**Tasks:**
1. Verify all GraphServiceSupervisor functionality migrated:
   - State → GraphStateActor ✓
   - Physics → PhysicsOrchestratorActor ✓
   - Semantics → SemanticProcessorActor ✓
   - Clients → ClientCoordinatorActor ✓
2. Update all references to GraphServiceSupervisor:
   - `app_state.rs` - change actor addresses
   - Handlers - route to specialized actors
   - Tests - update to new actor system
3. Delete `src/actors/graph_service_supervisor.rs`
4. Update architecture documentation
5. Run full integration test suite
6. Performance benchmark comparison (before/after)

**Verification:**
- All 107 existing tests pass
- No performance regression
- Cargo check passes

**Deliverable:** Monolithic actor removed, system decomposed

---

### Phase 6: Hexser Finalization (Week 7) - Optional
**Goal:** Complete Hexagonal Architecture formalization

**Tasks:**
1. Add remaining Hexser attributes:
   - Tag all domain models with `#[derive(HexDomain)]`
   - Tag all ports with `#[derive(HexPort)]`
   - Tag all adapters with `#[derive(HexAdapter)]`
   - Tag application services with `#[derive(HexService)]`
2. Export HexGraph:
```bash
cargo build --features hexser
# Generates target/hexgraph.json
```
3. Implement compile-time architectural rules:
```rust
// .hexser/rules.rs
HexGraph::enforce_rule(|graph| {
    graph.assert_no_dependencies("src/models", "src/handlers")
    graph.assert_no_dependencies("src/models", "src/gpu")
    graph.assert_all_adapters_implement_ports()
})
```
4. Create AI agent integration example:
   - Load hexgraph.json
   - Query architecture
   - Generate documentation
5. Add Hexserror to all error paths
6. Document Hexser patterns for team

**Deliverable:** Fully AI-introspectable architecture with compile-time safety

---

## Testing Strategy

### Unit Tests
Each actor testable in isolation using mock dependencies:

```rust
#[actix_rt::test]
async fn test_graph_state_actor_add_nodes() {
    let actor = GraphStateActor::new().start();

    let result = actor.send(AddNodes {
        nodes: vec![test_node(100), test_node(101)]
    }).await.unwrap();

    assert_eq!(result.unwrap(), vec![100, 101]);

    let graph = actor.send(GetGraphData).await.unwrap().unwrap();
    assert_eq!(graph.nodes.len(), 2);
}

#[actix_rt::test]
async fn test_physics_orchestrator_with_mock_graph() {
    let mock_graph_repo = Arc::new(MockGraphRepository::new());
    let mock_physics_sim = Arc::new(MockPhysicsSimulator::new());

    let actor = PhysicsOrchestratorActor::with_dependencies(
        mock_graph_repo,
        mock_physics_sim
    ).start();

    actor.send(StartSimulation).await.unwrap().unwrap();

    // Verify mock interactions
    assert_eq!(mock_physics_sim.step_count(), 1);
}
```

### Integration Tests
Test actor interactions:

```rust
#[actix_rt::test]
async fn test_full_simulation_pipeline() {
    // Start all actors
    let graph_state = GraphStateActor::new().start();
    let physics = PhysicsOrchestratorActor::new(graph_state.clone()).start();
    let semantic = SemanticProcessorActor::new(graph_state.clone()).start();
    let client = ClientCoordinatorActor::new(
        graph_state.clone(),
        physics.clone(),
        semantic.clone()
    ).start();

    // Add test data
    graph_state.send(AddNodes {
        nodes: create_test_graph(100)
    }).await.unwrap().unwrap();

    // Start simulation
    physics.send(StartSimulation).await.unwrap().unwrap();

    // Wait for updates
    tokio::time::sleep(Duration::from_secs(1)).await;

    // Verify positions changed
    let graph = graph_state.send(GetGraphData).await.unwrap().unwrap();
    assert!(has_non_zero_positions(&graph));

    // Verify dirty tracking
    let dirty = graph_state.send(GetDirtyNodes).await.unwrap().unwrap();
    assert_eq!(dirty.len(), 100);
}
```

### End-to-End Tests
Test complete request flow (addresses Audit Test Gap #1):

```rust
#[actix_rt::test]
async fn test_end_to_end_validation_pipeline() {
    // Setup: Start full actor system + HTTP server
    let app_state = create_test_app_state().await;
    let app = test::init_service(
        App::new()
            .app_data(app_state.clone())
            .configure(api_config)
    ).await;

    // Client → API Handler
    let req = test::TestRequest::post()
        .uri("/api/ontology/validate")
        .set_json(&test_ontology_graph())
        .to_request();

    let resp = test::call_service(&app, req).await;
    assert!(resp.status().is_success());

    // API → OntologyActor → Validator → Constraints
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Constraints → PhysicsOrchestrator → GPU
    let graph = app_state.graph_state_addr
        .send(GetGraphData).await.unwrap().unwrap();

    // Verify constraints applied
    assert!(graph.has_ontology_constraints());

    // Verify WebSocket notification sent
    // (WebSocket test infrastructure needed - Audit Test Gap #2)
}
```

---

## Migration Safety

### Incremental Rollout
1. **Week 1-2:** New actors start alongside old GraphServiceSupervisor
2. **Week 3-4:** Gradually redirect operations to new actors
3. **Week 5:** Run dual system (old + new) with comparison logging
4. **Week 6:** Remove old GraphServiceSupervisor after verification

### Rollback Plan
Each phase can be rolled back independently:
- Keep feature flags: `use_new_graph_state`, `use_physics_orchestrator_v2`
- Environment variable override: `LEGACY_GRAPH_SERVICE=true`
- Database versioning: No schema changes needed

### Performance Monitoring
Track metrics during migration:
```rust
struct MigrationMetrics {
    old_system_latency_ms: f64,
    new_system_latency_ms: f64,
    message_passing_overhead: f64,
    memory_usage_delta: i64,
}
```

**Acceptance Criteria:**
- Latency increase < 10%
- Memory increase < 20%
- All 107 tests pass
- No client-visible regressions

---

## Risk Analysis

### High Risk
1. **Actor Message Passing Overhead**
   - **Mitigation:** Benchmark before/after, optimize hot paths
   - **Fallback:** Keep monolithic path as option

2. **Deadlocks from Circular Dependencies**
   - **Mitigation:** Strict dependency hierarchy (GraphState → Physics → Semantic → Client)
   - **Detection:** Timeout monitoring on actor messages

### Medium Risk
3. **Hexser Learning Curve**
   - **Mitigation:** Phase 6 is optional, core refactor doesn't require Hexser
   - **Training:** Provide team Hexser workshop

4. **Test Coverage Gaps**
   - **Mitigation:** Address Audit Test Gaps #1, #2 before refactor
   - **Requirement:** 90%+ coverage on new actors before migration

### Low Risk
5. **Client Protocol Changes**
   - **Mitigation:** V2 binary protocol unchanged, only update frequency improved
   - **Testing:** Extensive WebSocket integration tests

---

## Success Metrics

### Code Quality
- **Lines of Code:** Expect 20-30% reduction via ports/adapters abstraction
- **Cyclomatic Complexity:** Target <15 per function (current ~25 in GraphServiceSupervisor)
- **Test Coverage:** 90%+ on new actors (currently 75% overall)

### Architecture Quality
- **Dependency Violations:** 0 (enforced by Hexser if used)
- **Single Responsibility:** Each actor <500 lines
- **Testability:** 100% of actors testable without full system

### System Performance
- **Latency:** <10% increase
- **Throughput:** No regression
- **Memory:** <20% increase
- **Startup Time:** No regression

### AI Integration (If Hexser Used)
- **HexGraph Accuracy:** 100% of dependencies captured
- **AI Query Success:** 95%+ of architectural queries answerable from HexGraph
- **Documentation Generation:** Auto-generate architecture docs from HexGraph

---

## Dependencies and Prerequisites

### Cargo Dependencies
```toml
[dependencies]
actix = "0.13"
actix-web = "4.4"
async-trait = "0.1"

[dependencies.hexser]
version = "0.3"  # Or latest
optional = true

[features]
default = []
hexser = ["dep:hexser"]
```

### External Systems
- No external system changes required
- GPU CUDA kernels unchanged
- Management API unchanged
- Database schema unchanged

### Team Skills
- Actix actor model (existing)
- Async Rust (existing)
- Hexagonal Architecture (training needed)
- Hexser framework (training needed if Phase 6)

---

## Timeline Summary

| Phase | Duration | Deliverable | Dependencies |
|-------|----------|-------------|--------------|
| Phase 0: Preparation | 1 week | Foundation ready | - |
| Phase 1: GraphStateActor | 1 week | State management isolated | Phase 0 |
| Phase 2: PhysicsOrchestrator | 1 week | Physics decoupled | Phase 1 |
| Phase 3: SemanticProcessor | 1 week | Semantics isolated | Phase 1 |
| Phase 4: ClientCoordinator | 1 week | Client communication complete | Phase 1-3 |
| Phase 5: Supervisor Removal | 1 week | Monolith removed | Phase 1-4 |
| Phase 6: Hexser Finalization | 1 week | AI-ready architecture | Phase 5 (optional) |

**Total Time:** 7 weeks (6 weeks if skipping Hexser)

---

## Next Steps

### Immediate Actions (This Week)
1. ✅ Complete Binary Protocol V1 removal
2. ✅ Create this refactoring plan
3. Review plan with team for feedback
4. Decide: Hexser integration yes/no?
5. Set up Phase 0 directory structure
6. Document current GraphServiceSupervisor behavior

### Week 1 Start
1. Create `src/actors/graph_state_actor.rs`
2. Implement basic CRUD operations
3. Write unit tests
4. Start GraphStateActor alongside existing system

---

## Appendices

### A. GraphServiceSupervisor Current Behavior
**File:** `src/actors/graph_service_supervisor.rs`
**Status:** All handlers stubbed (lines 735-789)
**Messages:** 15+ message types, all return "not yet fully implemented"
**Action:** Document expected behavior from test suite before removal

### B. Hexser Resources
- **Documentation:** https://docs.hexser.dev
- **GitHub:** https://github.com/hexser/hexser
- **Examples:** https://github.com/hexser/examples

### C. Alternative Approaches Considered

**Option 1: Keep Monolithic Actor**
- ❌ Rejected: Violates SRP, difficult to test, audit identified as P0 issue

**Option 2: Microservices Instead of Actors**
- ❌ Rejected: Unnecessary network overhead, actor model working well

**Option 3: Actor Decomposition Without Hexser**
- ✅ Valid: Phases 1-5 provide value independently
- ✅ Hexser (Phase 6) optional enhancement

---

**End of Refactoring Plan**
