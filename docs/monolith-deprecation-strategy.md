# GraphServiceActor Monolith Deprecation Strategy

**Document Version:** 1.0
**Date:** 2025-10-26
**Status:** STRATEGIC PLAN - Awaiting Architectural Decision
**Related:** `ARCHITECTURAL_CONFLICT_ANALYSIS.md`

---

## Executive Summary

This document provides a **phased deprecation strategy** for the 4,566-line `GraphServiceActor` monolith (`src/actors/graph_actor.rs`), migrating it to the modern CQRS/Hexagonal architecture pattern already established in `src/application/` and `src/ports/`.

**Key Findings:**
- **46 Handler implementations** spanning 4,566 lines (target: <500 lines per file)
- **CQRS equivalents exist** for 70% of basic operations (AddNode, RemoveNode, AddEdge, etc.)
- **No CQRS equivalents** for real-time features (WebSocket broadcasts, physics simulation)
- **Safe deprecation possible** for read-only and batch operations
- **High-risk migration** for real-time coordination and state management

**Timeline:** 3 phases over 6-9 months
**Risk Level:** HIGH - Core production system
**Recommended Approach:** Hybrid with event synchronization (Option 3 from architectural analysis)

---

## Current State Analysis

### GraphServiceActor Responsibilities

The monolith currently handles **8 distinct domains**:

#### 1. Graph State Management (Core - HIGH RISK)
- **Lines:** ~600 lines
- **Methods:**
  - `add_node()`, `remove_node()`, `add_edge()`, `remove_edge()`
  - `batch_add_nodes()`, `batch_add_edges()`, `batch_graph_update()`
  - `get_graph_data()`, `get_node_map()`
- **CQRS Equivalent:** ✅ **EXISTS** in `src/application/knowledge_graph/directives.rs`
- **Risk:** MEDIUM - Has equivalents but needs state sync

#### 2. Real-Time WebSocket Communication (HIGH RISK)
- **Lines:** ~800 lines
- **Methods:**
  - Client connection management via `ClientCoordinatorActor`
  - Position broadcasting to connected clients
  - `acknowledge_broadcast()`, `get_backpressure_metrics()`
- **CQRS Equivalent:** ❌ **NONE** - Real-time not in CQRS scope
- **Risk:** HIGH - Production critical, no fallback

#### 3. Physics Simulation Orchestration (HIGH RISK)
- **Lines:** ~900 lines
- **Methods:**
  - `update_node_positions()`, `update_simulation_params()`
  - Integration with `GPUManagerActor` for force computation
  - `StartSimulation`, `StopSimulation`, `SimulationStep` handlers
- **CQRS Equivalent:** ❌ **NONE** - Stateful simulation loop
- **Risk:** HIGH - Complex state machine

#### 4. Semantic Analysis & AI Features (MEDIUM RISK)
- **Lines:** ~700 lines
- **Methods:**
  - `SemanticAnalyzer` integration
  - `AdvancedEdgeGenerator` for multi-modal edges
  - `get_semantic_analysis_status()`, `regenerate_semantic_constraints()`
- **CQRS Equivalent:** ⚠️ **PARTIAL** - Could extract to service layer
- **Risk:** MEDIUM - Experimental features

#### 5. Advanced Physics & Constraints (MEDIUM RISK)
- **Lines:** ~600 lines
- **Methods:**
  - `update_advanced_physics_params()`, `get_constraint_set()`
  - `trigger_stress_optimization()`, `handle_constraint_update()`
  - Stress-majorization solver integration
- **CQRS Equivalent:** ❌ **NONE** - Domain-specific physics
- **Risk:** MEDIUM - Specialized functionality

#### 6. Metadata Synchronization (LOW RISK)
- **Lines:** ~500 lines
- **Methods:**
  - `build_from_metadata()`, `add_nodes_from_metadata()`
  - `update_node_from_metadata()`, `remove_node_by_metadata()`
- **CQRS Equivalent:** ⚠️ **PARTIAL** - Repository exists but no metadata handlers
- **Risk:** LOW - Batch operations, stateless

#### 7. Database Persistence (ALREADY MIGRATED)
- **Lines:** ~300 lines
- **Methods:**
  - Uses `KnowledgeGraphRepository` trait (Hexagonal pattern!)
  - `ReloadGraphFromDatabase` handler
- **CQRS Equivalent:** ✅ **FULLY MIGRATED** - Already using ports/adapters
- **Risk:** LOW - Already abstracted

#### 8. Batch Operations & Queue Management (LOW RISK)
- **Lines:** ~400 lines
- **Methods:**
  - `queue_add_node()`, `queue_add_edge()`, `flush_update_queue()`
  - `configure_update_queue()`, `batch_update_optimized()`
  - `get_batch_metrics()`, `force_flush_with_metrics()`
- **CQRS Equivalent:** ⚠️ **PARTIAL** - Basic batch exists in CQRS, not queuing
- **Risk:** LOW - Performance optimization layer

---

## Phase 1: Safe Deprecation Without Breaking Production (Months 1-2)

### Objective
Mark low-risk, stateless methods as deprecated while creating CQRS equivalents where missing.

### Strategy
Use Rust's `#[deprecated]` attribute to signal deprecation without removing functionality.

### Candidate Methods for Immediate Deprecation

#### 1.1 Read-Only Query Methods (ZERO RISK)

**Mark Deprecated:**
```rust
// src/actors/graph_actor.rs

#[deprecated(
    since = "2.0.0",
    note = "Use CQRS LoadGraphHandler from src/application/knowledge_graph/queries.rs"
)]
pub fn get_graph_data(&self) -> &GraphData {
    &self.graph_data
}

#[deprecated(
    since = "2.0.0",
    note = "Use CQRS GetNodeHandler for individual nodes or LoadGraphHandler for full graph"
)]
pub fn get_node_map(&self) -> &HashMap<u32, Node> {
    &self.node_map
}
```

**Action Items:**
- ✅ CQRS equivalent exists: `LoadGraphHandler`, `GetNodeHandler`
- 🔧 Add deprecation warnings to code
- 📊 Monitor usage via logging

**Timeline:** Week 1

---

#### 1.2 Batch Statistics & Metrics (LOW RISK)

**Mark Deprecated:**
```rust
#[deprecated(
    since = "2.0.0",
    note = "Use CQRS GetGraphStatisticsHandler for graph metrics"
)]
pub fn get_batch_metrics(&self) -> &BatchMetrics {
    &self.batch_metrics
}

#[deprecated(
    since = "2.0.0",
    note = "Use performance monitoring service from src/services/"
)]
pub fn get_backpressure_metrics(&self) -> (u32, u32, u32) {
    // ...
}
```

**Action Items:**
- 🔨 **CREATE:** Performance monitoring service in `src/services/performance_monitor.rs`
- ✅ CQRS has `GetGraphStatisticsHandler` for basic stats
- 📊 Extend CQRS with batch metrics query

**Timeline:** Week 2

---

#### 1.3 Configuration Getters (ZERO RISK)

**Mark Deprecated:**
```rust
#[deprecated(
    since = "2.0.0",
    note = "Use CQRS configuration queries from src/application/settings/"
)]
pub fn get_constraint_set(&self) -> &ConstraintSet {
    &self.constraint_set
}

#[deprecated(
    since = "2.0.0",
    note = "Use CQRS query handlers for semantic analysis status"
)]
pub fn get_semantic_analysis_status(&self) -> (usize, Option<std::time::Duration>) {
    // ...
}
```

**Action Items:**
- 🔨 **CREATE:** `GetConstraintsHandler` in `src/application/physics/queries.rs` (new domain)
- 🔨 **CREATE:** `GetSemanticStatusHandler` in `src/application/semantic/queries.rs` (new domain)

**Timeline:** Week 3

---

### 1.4 Metadata Batch Operations (LOW RISK)

**Mark Deprecated:**
```rust
#[deprecated(
    since = "2.0.0",
    note = "Use CQRS BuildGraphFromMetadataHandler"
)]
pub fn build_from_metadata(&mut self, metadata: MetadataStore, ctx: &mut Context<Self>) -> Result<(), String> {
    // ...
}

#[deprecated(
    since = "2.0.0",
    note = "Use CQRS AddNodesFromMetadataHandler"
)]
pub fn add_nodes_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
    // ...
}
```

**Action Items:**
- 🔨 **CREATE:** Metadata directive handlers:
  - `src/application/metadata/directives.rs::BuildGraphFromMetadataHandler`
  - `src/application/metadata/directives.rs::AddNodesFromMetadataHandler`
  - `src/application/metadata/directives.rs::UpdateNodeFromMetadataHandler`

**Timeline:** Week 4

---

### Phase 1 Deliverables

| Deliverable | Description | Risk | Timeline |
|-------------|-------------|------|----------|
| Deprecation warnings | Add `#[deprecated]` to 15+ low-risk methods | LOW | Week 1 |
| CQRS query handlers | Create missing query handlers for read operations | LOW | Weeks 2-3 |
| CQRS metadata handlers | Create metadata domain in CQRS layer | LOW | Week 4 |
| Logging instrumentation | Track deprecated method usage | ZERO | Week 1 |
| Documentation update | Update API docs with migration paths | ZERO | Week 4 |

**Success Criteria:**
- Zero production incidents
- All deprecated methods have CQRS equivalents
- Usage metrics collected for next phase

---

## Phase 2: Create CQRS Equivalents for Stateful Operations (Months 3-5)

### Objective
Build CQRS command/query handlers for stateful operations **without** removing GraphServiceActor logic yet.

### Strategy
**Parallel implementation** - Run both systems side-by-side with event synchronization.

### 2.1 Event Bus Architecture

**Create Domain Event System:**

```rust
// src/domain/events.rs

#[derive(Debug, Clone)]
pub enum GraphDomainEvent {
    // Graph mutations
    NodeAdded { node_id: u32, source: String },
    NodeRemoved { node_id: u32, source: String },
    EdgeAdded { edge_id: String, source: String },
    EdgeRemoved { edge_id: String, source: String },

    // Batch operations
    GraphReloaded { node_count: usize, edge_count: usize, source: String },
    BatchUpdated { nodes: usize, edges: usize, source: String },

    // Physics events
    PhysicsSimulationStarted,
    PhysicsSimulationStopped,
    NodePositionsUpdated { node_ids: Vec<u32> },

    // Semantic events
    SemanticAnalysisCompleted { features_extracted: usize },
    ConstraintsRegenerated { constraint_count: usize },
}

// src/services/event_bus.rs

pub struct EventBus {
    subscribers: Arc<RwLock<HashMap<String, Vec<EventSubscriber>>>>,
}

#[async_trait]
pub trait EventSubscriber: Send + Sync {
    async fn on_event(&self, event: &GraphDomainEvent);
}

impl EventBus {
    pub async fn publish(&self, event: GraphDomainEvent) {
        // Publish to all subscribers
    }

    pub fn subscribe(&mut self, subscriber: Box<dyn EventSubscriber>) {
        // Add subscriber
    }
}
```

**Timeline:** Weeks 9-10

---

### 2.2 CQRS Write Operations with Events

**Extend CQRS directives to publish events:**

```rust
// src/application/knowledge_graph/directives.rs

impl<R: KnowledgeGraphRepository + Send + Sync + 'static> DirectiveHandler<AddNode>
    for AddNodeHandler<R>
{
    fn handle(&self, directive: AddNode) -> HexResult<()> {
        let repository = self.repository.clone();
        let event_bus = self.event_bus.clone(); // NEW

        tokio::runtime::Handle::current().block_on(async move {
            // 1. Persist to database (CQRS)
            let node_id = repository.add_node(&directive.node).await?;

            // 2. Publish domain event (NEW)
            event_bus.publish(GraphDomainEvent::NodeAdded {
                node_id,
                source: "cqrs".to_string(),
            }).await;

            Ok(())
        })
    }
}
```

**Timeline:** Weeks 11-12

---

### 2.3 GraphServiceActor as Event Subscriber

**Make GraphServiceActor listen to CQRS events:**

```rust
// src/actors/graph_actor.rs

#[async_trait]
impl EventSubscriber for Addr<GraphServiceActor> {
    async fn on_event(&self, event: &GraphDomainEvent) {
        match event {
            GraphDomainEvent::GraphReloaded { node_count, edge_count, source } => {
                if source != "graph_service_actor" {
                    // CQRS layer updated database, reload our in-memory state
                    let _ = self.send(ReloadGraphFromDatabase).await;
                }
            }
            GraphDomainEvent::NodeAdded { node_id, source } => {
                if source != "graph_service_actor" {
                    // Reload specific node and broadcast to WebSocket clients
                    // ...
                }
            }
            // ... handle other events
            _ => {}
        }
    }
}
```

**This solves the "stale cache" problem identified in architectural analysis!**

**Timeline:** Weeks 13-14

---

### 2.4 Bidirectional Event Flow

**GraphServiceActor also publishes events:**

```rust
// src/actors/graph_actor.rs

impl Handler<AddNode> for GraphServiceActor {
    fn handle(&mut self, msg: AddNode, _ctx: &mut Context<Self>) -> Self::Result {
        // 1. Update in-memory state
        self.add_node(msg.node.clone());

        // 2. Persist to database via repository
        let _ = self.kg_repo.add_node(&msg.node).await;

        // 3. Publish event (NEW)
        self.event_bus.publish(GraphDomainEvent::NodeAdded {
            node_id: msg.node.id,
            source: "graph_service_actor".to_string(),
        }).await;

        // 4. Broadcast to WebSocket clients (existing)
        self.broadcast_node_update(&msg.node).await;

        Ok(())
    }
}
```

**Timeline:** Weeks 15-16

---

### 2.5 Create CQRS Handlers for Missing Operations

**New CQRS Application Domains:**

```
src/application/
├── knowledge_graph/    (✅ exists)
├── ontology/           (✅ exists)
├── settings/           (✅ exists)
├── physics/            (🔨 CREATE)
│   ├── directives.rs   - UpdateSimulationParams, TriggerStressMajorization
│   ├── queries.rs      - GetPhysicsState, GetConstraints
│   └── mod.rs
├── semantic/           (🔨 CREATE)
│   ├── directives.rs   - RegenerateSemanticConstraints
│   ├── queries.rs      - GetSemanticAnalysisStatus
│   └── mod.rs
└── metadata/           (🔨 CREATE)
    ├── directives.rs   - BuildGraphFromMetadata, UpdateNodeFromMetadata
    ├── queries.rs      - GetMetadataForNode
    └── mod.rs
```

**Timeline:** Weeks 17-20

---

### Phase 2 Deliverables

| Deliverable | Description | Risk | Timeline |
|-------------|-------------|------|----------|
| Event bus implementation | Domain event publishing/subscription | MEDIUM | Weeks 9-10 |
| CQRS event publishing | All directives publish events | LOW | Weeks 11-12 |
| Actor event subscription | GraphServiceActor listens to events | MEDIUM | Weeks 13-14 |
| Bidirectional sync | Actor publishes events back | MEDIUM | Weeks 15-16 |
| Physics CQRS handlers | New physics domain in CQRS | LOW | Weeks 17-18 |
| Semantic CQRS handlers | New semantic domain in CQRS | LOW | Weeks 19 |
| Metadata CQRS handlers | New metadata domain in CQRS | LOW | Week 20 |
| Integration tests | Verify event synchronization | HIGH | Weeks 19-20 |

**Success Criteria:**
- CQRS operations automatically sync to GraphServiceActor in-memory state
- GraphServiceActor operations publish events
- Zero state inconsistencies
- WebSocket broadcasts work for both CQRS and Actor-initiated changes

---

## Phase 3: Traffic Routing & Legacy Removal (Months 6-9)

### Objective
Gradually route API traffic to CQRS handlers, making GraphServiceActor a **thin real-time gateway**.

### Strategy
**Feature flags + percentage-based rollout** for risk mitigation.

### 3.1 API Route Migration (Gradual)

**Create feature-flagged routing:**

```rust
// src/routes/graph_routes.rs

pub async fn handle_add_node(
    body: web::Json<NodeCreateRequest>,
    graph_actor: web::Data<Addr<GraphServiceActor>>,
    cqrs_bus: web::Data<Arc<CQRSBus>>,
    config: web::Data<AppConfig>,
) -> Result<HttpResponse, Error> {
    // Feature flag: percentage-based rollout
    let use_cqrs = config.cqrs_rollout_percentage > rand::random::<u8>();

    if use_cqrs {
        // NEW: CQRS path
        let directive = AddNode { node: body.into_inner().into() };
        cqrs_bus.execute_directive(directive).await?;
        Ok(HttpResponse::Created().json(/* ... */))
    } else {
        // LEGACY: Actor path
        graph_actor.send(AddNode { /* ... */ }).await??;
        Ok(HttpResponse::Created().json(/* ... */))
    }
}
```

**Rollout Plan:**
- Week 21: 5% traffic to CQRS (read-only queries)
- Week 22: 20% traffic to CQRS (reads + simple writes)
- Week 23: 50% traffic to CQRS
- Week 24: 80% traffic to CQRS
- Week 25: 95% traffic to CQRS
- Week 26: 100% traffic to CQRS (monitor for 2 weeks)

**Timeline:** Weeks 21-26

---

### 3.2 GraphServiceActor Becomes WebSocket Gateway

**Refactor GraphServiceActor to minimal responsibilities:**

**KEEP (Real-Time Layer):**
- WebSocket client connection management
- Position broadcast to clients
- Real-time simulation orchestration (if not extracted)
- Event subscription for database changes

**REMOVE (Move to CQRS):**
- ❌ Graph state management (use event-sourced state or query database)
- ❌ Database persistence (already using repository)
- ❌ Business logic (semantic analysis, constraint generation)
- ❌ Batch operations (move to CQRS batch directives)

**Refactored structure (target <500 lines):**

```rust
// src/actors/graph_actor.rs (REFACTORED)

pub struct GraphServiceActor {
    // Real-time communication
    client_manager: Addr<ClientCoordinatorActor>,

    // Physics orchestration
    gpu_compute_addr: Option<Addr<GPUManagerActor>>,
    simulation_running: AtomicBool,
    simulation_params: SimulationParams,

    // Read-only cache for broadcasts (populated by events)
    graph_snapshot: Arc<RwLock<GraphData>>,

    // Event subscription
    event_bus: Arc<EventBus>,
}

impl EventSubscriber for Addr<GraphServiceActor> {
    async fn on_event(&self, event: &GraphDomainEvent) {
        match event {
            GraphDomainEvent::NodeAdded { node_id, .. } |
            GraphDomainEvent::GraphReloaded { .. } => {
                // Reload snapshot from database (via CQRS query)
                let graph = cqrs_bus.execute_query(LoadGraph).await?;
                *self.graph_snapshot.write().await = graph;

                // Broadcast to WebSocket clients
                self.broadcast_graph_update().await;
            }
            _ => {}
        }
    }
}
```

**Timeline:** Weeks 27-30

---

### 3.3 Remove Deprecated Methods

**After 100% CQRS traffic + 2-week monitoring:**

```rust
// src/actors/graph_actor.rs

// REMOVED (was deprecated in Phase 1)
// pub fn get_graph_data(&self) -> &GraphData { ... }
// pub fn get_node_map(&self) -> &HashMap<u32, Node> { ... }
// pub fn build_from_metadata(...) { ... }

// REMOVED (migrated to CQRS)
// pub fn add_node(&mut self, node: Node) { ... }
// pub fn remove_node(&mut self, node_id: u32) { ... }
// pub fn batch_add_nodes(...) { ... }
```

**Delete Handler implementations:**
- Remove 30+ Handler implementations for operations now in CQRS
- Keep only real-time coordination handlers:
  - `StartSimulation`, `StopSimulation`, `SimulationStep`
  - `InitialClientSync`, `ForcePositionBroadcast`
  - `UpdateNodePositions` (from physics)

**Timeline:** Weeks 31-32

---

### 3.4 Final Actor Responsibilities

**GraphServiceActor becomes ~500 lines (90% reduction!):**

```rust
// Final minimal responsibilities:

1. WebSocket Real-Time Gateway (200 lines)
   - Client connection lifecycle
   - Position broadcast to connected clients
   - Backpressure management

2. Physics Simulation Orchestrator (150 lines)
   - StartSimulation / StopSimulation
   - Coordinate with GPUManagerActor
   - Broadcast position updates

3. Event Subscriber (100 lines)
   - Listen to GraphDomainEvent
   - Reload read-only snapshot
   - Trigger WebSocket broadcasts

4. Initialization & Configuration (50 lines)
   - Actor lifecycle
   - Event bus registration
```

**Timeline:** Week 33

---

### Phase 3 Deliverables

| Deliverable | Description | Risk | Timeline |
|-------------|-------------|------|----------|
| Feature-flagged routing | API routes support both paths | MEDIUM | Week 21 |
| Gradual rollout (5%-100%) | Percentage-based traffic migration | HIGH | Weeks 21-26 |
| Monitoring & alerting | Track CQRS vs Actor performance | LOW | Week 21 |
| Actor refactoring | Reduce to <500 lines | HIGH | Weeks 27-30 |
| Handler removal | Delete 30+ deprecated handlers | MEDIUM | Weeks 31-32 |
| Final validation | Integration tests, performance tests | HIGH | Week 33 |
| Documentation update | Architectural diagrams, ADRs | ZERO | Week 33 |

**Success Criteria:**
- GraphServiceActor <500 lines
- 100% CQRS traffic with zero incidents
- Real-time WebSocket broadcasts still functional
- No state synchronization issues

---

## Risk Assessment & Mitigation

### High-Risk Areas

#### 1. Real-Time WebSocket Synchronization (CRITICAL)

**Risk:** CQRS updates don't immediately propagate to connected WebSocket clients.

**Symptoms:**
- Clients see stale data after CQRS operations
- Position updates lag or stop
- Race conditions between database writes and broadcasts

**Mitigation:**
- ✅ Event bus ensures GraphServiceActor receives notifications
- ✅ Integration tests verify end-to-end flow (CQRS → DB → Event → Actor → WebSocket)
- ✅ Add latency monitoring for event propagation
- ✅ Implement retry logic for event delivery failures

**Contingency:**
- Rollback feature flag to 0% CQRS traffic
- Force GraphServiceActor reload from database on broadcast

---

#### 2. State Consistency During Migration (HIGH)

**Risk:** Concurrent updates from both CQRS and Actor paths cause conflicts.

**Symptoms:**
- Database constraint violations
- Lost updates (last-write-wins conflicts)
- In-memory state diverges from database

**Mitigation:**
- ✅ **Phase 2 requirement:** Both paths publish events to prevent conflicts
- ✅ Use database transactions in CQRS handlers
- ✅ Add version/timestamp to domain events for conflict detection
- ✅ Implement event sourcing for audit trail

**Contingency:**
- Emergency database restore from backup
- Lock down to single write path (either CQRS or Actor)

---

#### 3. Performance Degradation (MEDIUM)

**Risk:** Event bus overhead slows down operations.

**Symptoms:**
- Increased latency for graph operations
- Event queue backlog
- CPU/memory pressure from event processing

**Mitigation:**
- ✅ Async event publishing (non-blocking)
- ✅ Event batching for high-throughput scenarios
- ✅ Performance benchmarks before/after migration
- ✅ Circuit breaker for event bus failures

**Contingency:**
- Disable event publishing temporarily
- Scale up event processing workers

---

#### 4. Physics Simulation Disruption (HIGH)

**Risk:** Refactoring breaks physics simulation loop.

**Symptoms:**
- Nodes stop moving
- Force computation failures
- GPU integration breaks

**Mitigation:**
- ✅ **DO NOT TOUCH** physics loop in Phase 1-2
- ✅ Extract physics to separate service only in Phase 3
- ✅ Extensive physics integration tests
- ✅ Keep GPUManagerActor interface unchanged

**Contingency:**
- Maintain physics code in Actor until fully validated
- Add feature flag for physics service migration

---

### Medium-Risk Areas

#### 5. Semantic Analysis Migration (MEDIUM)

**Risk:** Semantic features break during CQRS migration.

**Mitigation:**
- Mark as **experimental** in Phase 1
- Create standalone semantic service in Phase 2
- Validate against test dataset

#### 6. Batch Operation Performance (MEDIUM)

**Risk:** CQRS batch handlers slower than Actor batch methods.

**Mitigation:**
- Benchmark batch operations before migration
- Optimize CQRS repository batch methods
- Keep Actor batch paths available as fallback

---

## Timeline & Milestones

### Phase 1: Safe Deprecation (Months 1-2)

| Week | Milestone | Risk | Validation |
|------|-----------|------|------------|
| 1 | Deprecate read-only methods | ZERO | Compile warnings logged |
| 2 | Create performance monitoring service | LOW | Metrics dashboard |
| 3 | Create CQRS query handlers for config | LOW | Unit tests pass |
| 4 | Create metadata CQRS handlers | LOW | Integration tests pass |
| 8 | **Phase 1 Complete** | LOW | Zero production incidents |

---

### Phase 2: CQRS Equivalents + Events (Months 3-5)

| Week | Milestone | Risk | Validation |
|------|-----------|------|------------|
| 9-10 | Event bus implementation | MEDIUM | Event delivery tests |
| 11-12 | CQRS publishes events | LOW | Event logs verified |
| 13-14 | Actor subscribes to events | MEDIUM | State sync tests |
| 15-16 | Actor publishes events | MEDIUM | Bidirectional sync tests |
| 17-18 | Physics CQRS handlers | LOW | Unit tests pass |
| 19 | Semantic CQRS handlers | LOW | Unit tests pass |
| 20 | Metadata CQRS handlers | LOW | Unit tests pass |
| 20 | **Phase 2 Complete** | MEDIUM | End-to-end integration tests |

---

### Phase 3: Traffic Routing & Removal (Months 6-9)

| Week | Milestone | Risk | Validation |
|------|-----------|------|------------|
| 21 | Feature-flagged routing | MEDIUM | A/B testing framework |
| 21-26 | Gradual rollout (5%→100%) | HIGH | Zero error rate increase |
| 27-30 | Refactor Actor to <500 lines | HIGH | All tests pass |
| 31-32 | Remove deprecated handlers | MEDIUM | Code coverage maintained |
| 33 | Final validation | HIGH | Performance benchmarks |
| 33 | **Phase 3 Complete** | HIGH | Production stable 2 weeks |

**Total Timeline:** 33 weeks (~8 months)

---

## Success Metrics

### Quantitative Metrics

| Metric | Baseline (Current) | Target (Post-Migration) |
|--------|-------------------|------------------------|
| GraphServiceActor LOC | 4,566 lines | <500 lines (89% reduction) |
| Handler count | 46 handlers | <10 handlers (real-time only) |
| CQRS coverage | ~30% (unused) | 100% (active) |
| State sync latency | N/A (no sync) | <50ms (event propagation) |
| WebSocket broadcast lag | ~10ms | <20ms (acceptable overhead) |
| Test coverage | ~60% | >90% (all CQRS handlers) |

---

### Qualitative Metrics

- ✅ **Architectural Clarity:** Single source of truth for data operations (CQRS)
- ✅ **Testability:** All business logic unit-testable via CQRS handlers
- ✅ **Maintainability:** <500 line files, clear separation of concerns
- ✅ **Flexibility:** Easy to swap repository implementations (SQLite → PostgreSQL)
- ✅ **Real-time Reliability:** WebSocket broadcasts unaffected by migration

---

## Decision Points

### Before Phase 1 Starts (Week 0)

**Required Decisions:**
1. ✅ **Approve hybrid architecture approach** (Option 3 from ARCHITECTURAL_CONFLICT_ANALYSIS.md)
2. ✅ **Allocate engineering resources** (2-3 engineers for 8 months)
3. ✅ **Define rollback criteria** (e.g., >1% error rate increase → rollback)
4. ✅ **Approve event bus technology** (in-memory vs. Redis Pub/Sub vs. Kafka)

**Stakeholders:** Tech Lead, Engineering Manager, Product Owner

---

### Phase 1 → Phase 2 Gate (Week 8)

**Go/No-Go Criteria:**
- ✅ All deprecated methods have CQRS equivalents
- ✅ Zero production incidents from deprecation warnings
- ✅ Usage metrics show adoption of new APIs
- ✅ Team comfortable with CQRS pattern

---

### Phase 2 → Phase 3 Gate (Week 20)

**Go/No-Go Criteria:**
- ✅ Event bus successfully synchronizes state
- ✅ No state inconsistencies in integration tests
- ✅ Performance benchmarks show acceptable overhead (<10%)
- ✅ All CQRS handlers implemented and tested

---

### Phase 3 Rollout Gates

**5% → 20%:** Zero error rate increase over 3 days
**20% → 50%:** Performance within 5% of baseline
**50% → 80%:** State sync issues <0.01%
**80% → 95%:** WebSocket latency <20ms
**95% → 100%:** Production stable for 1 week

---

## Rollback Plan

### Immediate Rollback (Phase 3)

**Trigger:** Error rate >1% increase, data loss, or WebSocket failures

**Action:**
```bash
# Set feature flag to 0% CQRS traffic
curl -X POST /api/admin/config \
  -d '{"cqrs_rollout_percentage": 0}'

# Force GraphServiceActor to reload from database
curl -X POST /api/admin/graph/reload

# Monitor recovery
curl /api/health/detailed
```

**Recovery Time:** <5 minutes

---

### Emergency Rollback (Phase 2)

**Trigger:** Event bus failures causing state corruption

**Action:**
1. Disable event publishing in CQRS handlers (config flag)
2. Disable event subscription in GraphServiceActor
3. Restart GraphServiceActor with fresh database load
4. Route all traffic to Actor path only

**Recovery Time:** <15 minutes

---

### Nuclear Rollback (Phase 1)

**Trigger:** Fundamental architecture decision reversal

**Action:**
1. Remove all `#[deprecated]` attributes
2. Delete CQRS handlers created in Phase 1
3. Document lessons learned
4. Resume operations with legacy Actor

**Recovery Time:** <1 hour (code changes + deploy)

---

## Open Questions

1. **Event Bus Technology:**
   - In-memory event bus (simple, fast, not durable)?
   - Redis Pub/Sub (distributed, durable, requires infrastructure)?
   - Tokio broadcast channels (Rust-native, in-process)?

2. **Real-Time Physics Service:**
   - Extract to separate service in Phase 3?
   - Keep physics in Actor indefinitely?
   - Move to dedicated GPU service?

3. **Testing Strategy:**
   - Contract testing between CQRS and Actor?
   - Chaos engineering to test event failures?
   - Performance regression tests?

4. **Migration Complexity:**
   - Should we extend timeline if complexity exceeds estimates?
   - Hire external architects for CQRS expertise?

---

## Recommended Next Steps

### Immediate (This Week)

1. ✅ **Schedule architectural review meeting** with tech lead and team
2. ✅ **Present this strategy document** for feedback
3. ✅ **Decide on event bus technology** (blocking decision)
4. ✅ **Assign Phase 1 engineering owner**

### Short-term (Next Sprint)

5. 🔧 **Create ADR (Architecture Decision Record)** documenting approved approach
6. 🔧 **Setup feature flag infrastructure** for gradual rollout
7. 🔧 **Write integration test framework** for event synchronization
8. 🔧 **Begin Phase 1 Week 1:** Deprecate read-only methods

### Long-term (Quarter Planning)

9. 📅 **Reserve engineering capacity** for 8-month migration
10. 📅 **Plan incident response drills** for rollback scenarios
11. 📅 **Schedule weekly migration standups** for Phase 2 onwards

---

## Conclusion

The 4,566-line GraphServiceActor monolith can be **safely migrated** to CQRS/Hexagonal architecture over **8 months** using a **phased, event-driven approach**.

**Key Success Factors:**
- ✅ **Phase 1:** Low-risk deprecation builds confidence and CQRS momentum
- ✅ **Phase 2:** Event bus solves the "stale cache" problem identified in architectural analysis
- ✅ **Phase 3:** Gradual rollout (5%→100%) allows safe validation at each stage

**Final State:**
- **GraphServiceActor:** <500 lines, real-time WebSocket gateway only
- **CQRS Layer:** 100% of business logic, fully testable
- **Event Bus:** Synchronizes database changes with in-memory actor state

**Risk Mitigation:**
- Feature flags enable instant rollback
- Parallel implementation prevents breaking changes
- Extensive testing at each phase gate

**Outcome:**
A **modern, maintainable architecture** that preserves real-time functionality while achieving the testability and flexibility goals of CQRS/Hexagonal design.

---

**Document Status:** STRATEGIC PLAN - Ready for Review
**Next Action:** Schedule architectural review meeting
**Owner:** System Architecture Team
**Reviewers:** Tech Lead, Engineering Manager, Senior Engineers
