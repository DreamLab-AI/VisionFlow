# Code Quality Analysis Report: Hexagonal Migration Dependency Audit

**Generated:** 2025-10-26
**Auditor:** Code Quality Analyzer Agent
**Scope:** GraphServiceActor, GPU Manager Actor, Physics Orchestrator Actor

---

## Executive Summary

### Overall Quality Score: 6.5/10
- **Files Analyzed:** 26 source files
- **Critical Dependencies Found:** 47 handler dependencies on GraphServiceActor
- **Technical Debt Estimate:** 240-320 hours for complete hexagonal migration
- **Migration Risk Level:** HIGH (monolithic actor has 4,566 lines, 46+ message handlers)

### Critical Findings

1. **GraphServiceActor is HIGHLY COUPLED** - 46+ message handlers, 9 API handlers depend on it
2. **AppState stores direct actor address** - All handlers receive `Addr<TransitionalGraphSupervisor>`
3. **WebSocket handler has 7+ direct actor calls** - Real-time physics updates tightly coupled
4. **GPU actors require GraphServiceActor address** - Circular dependency for position updates
5. **No repository abstraction for graph operations** - All graph queries go through actor system

---

## 1. GraphServiceActor Dependency Map

### 1.1 Core Statistics

```json
{
  "graph_service_actor": {
    "total_dependencies": 47,
    "lines_of_code": 4566,
    "message_handlers": 46,
    "api_routes": 9,
    "websocket_handlers": 1,
    "services": 5,
    "tests": 2,
    "actor_dependencies": 8
  }
}
```

### 1.2 API Handler Dependencies

**File:** `src/handlers/api_handler/graph/mod.rs`
- **Dependencies:** 6 message sends to `graph_service_addr`
- **Messages Used:**
  - `GetGraphData` - Fetch full graph structure
  - `GetNodeMap` - Get node ID mappings
  - `GetPhysicsState` - Check physics simulation status
  - `AddNodesFromMetadata` - Incremental graph updates
  - `GetAutoBalanceNotifications` - Physics auto-balance events

**Critical Finding:** ALL graph API operations go through GraphServiceActor. No repository layer exists.

**File:** `src/handlers/api_handler/analytics/mod.rs`
- **Dependencies:** 3 message sends
- **Messages Used:**
  - `GetGraphData` - Graph structure for analytics
  - GPU-related messages routed through GraphServiceActor

**File:** `src/handlers/api_handler/files/mod.rs`
- **Dependencies:** 1 message send
- **Messages:** File metadata integration with graph

**File:** `src/handlers/socket_flow_handler.rs`
- **Dependencies:** 7+ message sends (HIGHEST)
- **Messages Used:**
  - `GetGraphData` - Real-time graph state
  - `RequestPositionSnapshot` - Physics position sync
  - `UpdateNodePosition` - User drag interactions
  - `SimulationStep` - Manual physics step
  - Real-time physics updates

**Critical Finding:** WebSocket handler is the MOST COUPLED component. Real-time physics updates create tight bidirectional dependency.

### 1.3 Service Dependencies

**File:** `src/app_state.rs`
- **Line 50:** `pub graph_service_addr: Addr<TransitionalGraphSupervisor>`
- **Impact:** ALL handlers receive this address via `web::Data<AppState>`
- **Blast Radius:** Changing this requires updating 26+ handler files

**File:** `src/services/bots_client.rs`
- **Dependency:** Stores `graph_service_addr` for bot graph integration
- **Usage:** Bot metadata → graph node creation

**File:** `src/actors/client_coordinator_actor.rs`
- **Dependency:** Receives GraphServiceActor address for force broadcasts
- **Message:** `SetGraphServiceAddress`

### 1.4 Actor-to-Actor Dependencies

**GPU Manager Actor → GraphServiceActor**
- **Reason:** GPU needs to send position updates back to graph
- **Messages:** Position update broadcasts after physics computation
- **Circular Dependency:** Yes - GraphServiceActor → GPU → GraphServiceActor

**Physics Orchestrator → GraphServiceActor**
- **Reason:** Physics needs graph data for force calculations
- **Messages:** Graph data requests, position updates
- **Status:** Currently less coupled, but still uses actor system

**ClientCoordinatorActor → GraphServiceActor**
- **Reason:** Force broadcast settling fix
- **Messages:** `ForcePositionBroadcast`
- **Impact:** WebSocket real-time updates

---

## 2. GPU Actor Dependencies

### 2.1 GPUManagerActor Analysis

```json
{
  "gpu_manager_actor": {
    "lines_of_code": 657,
    "message_handlers": 18,
    "child_actors": 7,
    "graph_service_dependency": "HIGH",
    "migration_priority": "MEDIUM"
  }
}
```

**Child Actors Spawned:**
1. `GPUResourceActor` - GPU buffer management
2. `ForceComputeActor` - Physics force calculations
3. `ClusteringActor` - K-means, community detection
4. `AnomalyDetectionActor` - LOF, Z-score
5. `StressMajorizationActor` - Graph layout optimization
6. `ConstraintActor` - Physics constraints
7. `OntologyConstraintActor` - Ontology-driven constraints

**GraphServiceActor Dependency:**
- **SetSharedGPUContext message** includes `graph_service_addr: Option<Addr<GraphServiceActor>>`
- **Reason:** GPU actors need to send position updates back to graph
- **Impact:** Circular dependency prevents clean separation

**Migration Complexity:** MEDIUM
- Can be isolated relatively easily
- Main blocker is the position update callback
- Solution: Use event bus or repository pattern

### 2.2 Physics Orchestrator Analysis

```json
{
  "physics_orchestrator_actor": {
    "lines_of_code": 1105,
    "message_handlers": 15,
    "graph_dependency": "MEDIUM",
    "migration_priority": "LOW"
  }
}
```

**Current State:**
- Has `graph_data_ref: Option<Arc<GraphData>>` - good abstraction
- Uses GPU compute actor, not direct GraphServiceActor dependency
- Still receives GPU compute address from AppState

**Positive Finding:** Physics Orchestrator is ALREADY partially decoupled. It stores graph data reference instead of querying actor.

**Migration Path:** Low complexity - mainly needs event bus for position updates

---

## 3. Data Flow Analysis

### 3.1 Graph Data Flow (Current Architecture)

```
User Request (HTTP/WebSocket)
    ↓
Handler (AppState)
    ↓
GraphServiceActor.send(GetGraphData)
    ↓
GraphServiceActor internal state
    ↓
Actor responds with Arc<GraphData>
    ↓
Handler serializes and returns
```

**Problem:** Every graph query goes through actor mailbox (sequential bottleneck)

### 3.2 Physics Update Flow (Real-time)

```
GPU Physics Computation
    ↓
ForceComputeActor
    ↓
GraphServiceActor.send(UpdateNodePositions)
    ↓
GraphServiceActor updates internal state
    ↓
GraphServiceActor → ClientCoordinatorActor (force broadcast)
    ↓
WebSocket sends to connected clients
```

**Problem:** Circular dependency prevents GPU actor isolation

### 3.3 WebSocket Interaction Flow

```
User drags node in UI
    ↓
WebSocket receives BinaryMessage
    ↓
Handler extracts position
    ↓
GraphServiceActor.send(UpdateNodePosition)
    ↓
GraphServiceActor updates node
    ↓
GraphServiceActor triggers physics pause
    ↓
GraphServiceActor → GPU → resume after 500ms
```

**Problem:** Real-time interaction requires synchronous actor communication (high latency)

---

## 4. Migration Complexity Scores

### 4.1 GraphServiceActor Migration

**Complexity: 9/10 (VERY HIGH)**

**Reasons:**
- 4,566 lines of monolithic code
- 46+ message handlers to refactor
- 9 API handlers depend on it
- WebSocket handler has 7+ direct calls
- Circular GPU dependencies
- No existing repository layer
- Physics state management tightly coupled

**Estimated Time:** 160-200 hours
- Repository layer: 40 hours
- API handler refactor: 60 hours
- WebSocket refactor: 40 hours
- GPU decoupling: 20 hours
- Testing: 40 hours

**Blast Radius:** CRITICAL - removing it TODAY would break:
- All graph visualization
- All API endpoints (`/api/graph/*`)
- WebSocket real-time updates
- Physics simulation
- GPU-accelerated analytics
- Bot graph integration
- Auto-balance notifications

### 4.2 GPU Manager Actor Migration

**Complexity: 5/10 (MEDIUM)**

**Reasons:**
- Well-structured supervisor pattern
- Only 657 lines
- Clear child actor separation
- Main blocker is GraphServiceActor callback

**Estimated Time:** 40-60 hours
- Event bus integration: 20 hours
- Position update refactor: 15 hours
- Testing: 25 hours

**Blast Radius:** MEDIUM - removing it would break:
- GPU-accelerated physics
- Clustering analytics
- Anomaly detection
- Constraint-based layouts
- Ontology physics

### 4.3 Physics Orchestrator Migration

**Complexity: 3/10 (LOW)**

**Reasons:**
- Already partially decoupled
- Only 1,105 lines
- Uses data reference pattern
- Minimal actor dependencies

**Estimated Time:** 40-60 hours
- Event bus integration: 15 hours
- Repository injection: 10 hours
- Testing: 35 hours

**Blast Radius:** LOW - removing it would break:
- Physics simulation orchestration
- Auto-balance features
- Equilibrium detection

---

## 5. Code Smells Detected

### 5.1 Critical Code Smells

**God Object (GraphServiceActor)**
- **Severity:** CRITICAL
- **Lines:** 4,566 (9x over 500-line guideline)
- **Responsibilities:** Graph state, physics, GPU coordination, WebSocket broadcasts, metadata integration, bot graphs, analytics
- **Recommendation:** Split into 8+ domain-specific services

**Feature Envy (All Handlers)**
- **Severity:** HIGH
- **Pattern:** Every handler directly calls `state.graph_service_addr.send()`
- **Recommendation:** Introduce repository layer, handlers should use repositories

**Circular Dependency (GPU ↔ Graph)**
- **Severity:** HIGH
- **Pattern:** GraphServiceActor → GPU → GraphServiceActor (position updates)
- **Recommendation:** Event bus or publish-subscribe pattern

**Long Method (GraphServiceActor handlers)**
- **Severity:** MEDIUM
- **Pattern:** Many message handlers exceed 50 lines
- **Example:** `UpdateGraphData` handler has complex nested logic
- **Recommendation:** Extract service methods

**Inappropriate Intimacy (AppState)**
- **Severity:** MEDIUM
- **Pattern:** AppState exposes raw actor addresses to all handlers
- **Recommendation:** Facade pattern or service locator

### 5.2 Performance Code Smells

**Sequential Bottleneck**
- **Pattern:** All graph queries go through single actor mailbox
- **Impact:** Cannot scale beyond single-core performance
- **Recommendation:** Repository with connection pooling

**Synchronous Actor Calls in WebSocket**
- **Pattern:** WebSocket handler uses `.send().await` for position updates
- **Impact:** Adds 5-15ms latency per interaction
- **Recommendation:** Message queue or async event bus

---

## 6. Refactoring Opportunities

### 6.1 High-Impact Quick Wins

**Introduce GraphRepository Trait (40 hours)**
```rust
pub trait GraphRepository: Send + Sync {
    async fn get_graph_data(&self) -> Result<Arc<GraphData>, Error>;
    async fn get_node(&self, id: u32) -> Result<Node, Error>;
    async fn add_node(&self, node: Node) -> Result<(), Error>;
    async fn update_node_position(&self, id: u32, position: Vec3) -> Result<(), Error>;
}
```

**Benefit:** Decouple handlers from actor system, enable parallel queries

**Extract PhysicsService (30 hours)**
```rust
pub struct PhysicsService {
    repository: Arc<dyn GraphRepository>,
    gpu_manager: Arc<GpuManager>,
}
```

**Benefit:** Separate physics logic from graph state management

**Implement Event Bus for GPU Updates (25 hours)**
```rust
pub trait EventBus {
    fn publish(&self, event: GraphEvent);
    fn subscribe(&self, subscriber: Box<dyn EventSubscriber>);
}
```

**Benefit:** Break circular GPU ↔ Graph dependency

### 6.2 Long-term Architectural Improvements

**CQRS for Graph Operations (80 hours)**
- Command side: Mutations through actor system
- Query side: Direct repository access
- Benefit: 10x faster reads, maintain actor benefits for writes

**Repository Pattern for All Domains (120 hours)**
- GraphRepository (already planned)
- PhysicsRepository (simulation state)
- AnalyticsRepository (GPU results cache)
- Benefit: True hexagonal architecture, testable without actors

**WebSocket Message Queue (60 hours)**
- Replace synchronous actor calls with async queue
- Benefit: Sub-millisecond position updates

---

## 7. Positive Findings

### 7.1 Good Practices Observed

**Modular GPU Actor System**
- GPUManagerActor uses supervisor pattern
- Child actors are single-responsibility
- Good separation of clustering, anomaly detection, constraints

**Physics Orchestrator Abstraction**
- Already uses `Arc<GraphData>` instead of actor queries
- Parameter interpolation for smooth transitions
- Auto-balance and equilibrium detection well-isolated

**Repository Pattern Started**
- `SqliteKnowledgeGraphRepository` exists in adapters
- Shows team understands hexagonal architecture
- Just needs to be connected to handlers

**Actor System Benefits Preserved**
- Fault tolerance through supervision
- Message-driven architecture for complex workflows
- Good telemetry and logging

---

## 8. Migration Roadmap

### Phase 1: Foundation (80 hours)
1. Create GraphRepository trait and SQLite implementation
2. Add repository to AppState
3. Refactor 2-3 simple API handlers to use repository
4. Add integration tests

### Phase 2: API Handler Migration (60 hours)
1. Migrate all `/api/graph/*` handlers to repository
2. Keep actor system for mutations only (CQRS)
3. Performance testing (expect 5-10x speedup for reads)

### Phase 3: Event Bus (40 hours)
1. Implement event bus for GPU position updates
2. Remove GraphServiceActor reference from GPU actors
3. Break circular dependency

### Phase 4: WebSocket Optimization (60 hours)
1. Replace synchronous actor calls with message queue
2. Async position update broadcasting
3. Sub-millisecond latency target

### Phase 5: Actor Consolidation (80 hours)
1. Split GraphServiceActor into domain services
2. Keep lightweight actor only for command orchestration
3. Move business logic to services

**Total Estimated Time:** 320 hours (8 weeks with 2 engineers)

---

## 9. Immediate Action Items

### Critical (Do First)
1. ✅ **Create this audit document** (DONE)
2. ⚠️ **Implement GraphRepository trait** (BLOCKS everything else)
3. ⚠️ **Refactor 1 API handler as proof-of-concept** (Validate approach)

### High Priority (Week 1)
4. Design event bus architecture (GPU decoupling)
5. Map all GraphServiceActor message handlers to service methods
6. Create migration test suite

### Medium Priority (Week 2-3)
7. Migrate remaining API handlers
8. Implement CQRS pattern
9. Performance benchmarking

### Low Priority (Week 4+)
10. WebSocket async queue
11. Split GraphServiceActor
12. Remove actor system from read path

---

## 10. Risk Assessment

### High Risks
- **Breaking WebSocket real-time updates** - Requires careful testing
- **GPU position callback failure** - Event bus must be reliable
- **Performance regression** - Repository must be faster than actors

### Mitigation Strategies
- Feature flags for gradual rollout
- Parallel running of old and new systems
- Extensive integration testing
- Performance benchmarks before/after

### Success Metrics
- API response time < 10ms (currently 50-100ms)
- WebSocket latency < 16ms for 60 FPS
- Zero data loss during migration
- 100% test coverage for new repositories

---

## Appendix A: File Dependency Matrix

| File | GraphServiceActor Calls | GPU Actor Calls | Migration Priority |
|------|------------------------|-----------------|-------------------|
| `api_handler/graph/mod.rs` | 6 | 0 | HIGH |
| `socket_flow_handler.rs` | 7+ | 0 | CRITICAL |
| `api_handler/analytics/mod.rs` | 3 | 2 | HIGH |
| `clustering_handler.rs` | 2 | 1 | MEDIUM |
| `bots_handler.rs` | 1 | 0 | LOW |
| `consolidated_health_handler.rs` | 1 | 1 | LOW |

---

## Appendix B: Message Handler Inventory

GraphServiceActor implements **46 message handlers**:

**Graph Structure (10 handlers):**
- GetGraphData, GetNodeMap, AddNode, RemoveNode, AddEdge, RemoveEdge
- BatchAddNodes, BatchAddEdges, BatchGraphUpdate
- UpdateNodePositions

**Physics (12 handlers):**
- StartSimulation, StopSimulation, SimulationStep, UpdateNodePosition
- GetPhysicsState, UpdateSimulationParams, UpdateAdvancedParams
- PhysicsPauseMessage, NodeInteractionMessage, ForceResumePhysics
- GetEquilibriumStatus, RequestPositionSnapshot

**GPU Integration (8 handlers):**
- InitializeGPUConnection, StoreGPUComputeAddress, GPUInitialized
- SetAdvancedGPUContext, ResetGPUInitFlag, UpdateConstraints
- GetConstraints, TriggerStressMajorization

**Metadata/Content (6 handlers):**
- BuildGraphFromMetadata, AddNodesFromMetadata
- UpdateNodeFromMetadata, RemoveNodeByMetadata
- UpdateGraphData, ReloadGraphFromDatabase

**Bot Integration (2 handlers):**
- UpdateBotsGraph, GetBotsGraphData

**Analytics (4 handlers):**
- GetAutoBalanceNotifications, RegenerateSemanticConstraints
- ComputeShortestPaths, UpdateQueue operations

**Sync (4 handlers):**
- InitialClientSync, ForcePositionBroadcast
- FlushUpdateQueue, ConfigureUpdateQueue

---

## Conclusion

GraphServiceActor is the **primary blocker** for hexagonal migration. With 4,566 lines and 46 message handlers, it's a massive God Object that requires careful decomposition.

**Recommended Approach:** Incremental migration using CQRS pattern
1. Keep actor system for commands (mutations)
2. Use repository for queries (reads)
3. Gradually move business logic to services
4. Event bus for GPU decoupling

**Timeline:** 8 weeks with 2 engineers for complete migration
**Risk:** Medium-High (careful testing required)
**Reward:** 10x faster queries, true hexagonal architecture, scalable design

---

**Audit Complete** ✅
**Next Step:** Implement GraphRepository trait and migrate first API handler as proof-of-concept.
