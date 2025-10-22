# GraphServiceActor Decomposition Plan

## 🚨 Current Problem

**GraphServiceActor is a 3,910-line monolith** that violates the Single Responsibility Principle and contradicts the hexagonal architecture we just implemented.

### Current Statistics:
- **Lines of code**: 3,910
- **Message handlers**: 44
- **Responsibilities**: 10+ distinct domains mixed together
- **Status**: ⚠️ MONOLITHIC - Needs decomposition

## Why This Actor Still Exists

During the hexagonal migration (Phases 1-6), we focused on:
1. ✅ Creating ports/adapters architecture
2. ✅ Implementing CQRS handlers for business logic
3. ✅ Separating database concerns (3-database architecture)
4. ✅ Fixing 361 compilation errors

**BUT** we did NOT decompose the actor layer, leaving GraphServiceActor as a "god object" that handles everything related to graph state.

## Current Responsibilities (Violating SRP)

GraphServiceActor currently handles:

### 1. Graph Data Management (6 handlers)
- `GetGraphData` - Retrieve full graph state
- `UpdateGraphData` - Update graph structure
- `GetNodeMap` - Get node mappings
- `UpdateNodePositions` - Bulk position updates
- `ForcePositionBroadcast` - Force client sync
- `InitialClientSync` - Initial client state

### 2. Node/Edge CRUD (8 handlers)
- `AddNode`, `RemoveNode`
- `AddEdge`, `RemoveEdge`
- `UpdateNodePosition`
- `BatchAddNodes`, `BatchAddEdges`
- `BatchGraphUpdate`

### 3. Physics Simulation (6 handlers)
- `StartSimulation`, `StopSimulation`
- `SimulationStep`
- `UpdateSimulationParams`
- `UpdateAdvancedParams`
- `ForceResumePhysics`, `PhysicsPauseMessage`

### 4. GPU Coordination (5 handlers)
- `StoreGPUComputeAddress`
- `InitializeGPUConnection`
- `GPUInitialized`, `ResetGPUInitFlag`
- `SetAdvancedGPUContext`

### 5. Metadata Integration (4 handlers)
- `BuildGraphFromMetadata`
- `AddNodesFromMetadata`
- `UpdateNodeFromMetadata`
- `RemoveNodeByMetadata`

### 6. Bots Graph (2 handlers)
- `UpdateBotsGraph`
- `GetBotsGraphData`

### 7. Constraints (3 handlers)
- `UpdateConstraints`
- `GetConstraints`
- `RegenerateSemanticConstraints`

### 8. Pathfinding (1 handler)
- `ComputeShortestPaths`

### 9. Stress Majorization (2 handlers)
- `TriggerStressMajorization`
- `RequestPositionSnapshot`

### 10. Other (7 handlers)
- `GetPhysicsState`
- `GetAutoBalanceNotifications`
- `GetEquilibriumStatus`
- `NodeInteractionMessage`
- `ConfigureUpdateQueue`
- `FlushUpdateQueue`

---

## 🎯 Proposed Decomposition Architecture

### New Actor Structure (Following Actor Model Best Practices)

```
┌─────────────────────────────────────────────────────────────┐
│           GraphCoordinatorActor (Lightweight)               │
│  - Orchestrates other actors                                │
│  - Routes messages to specialized actors                    │
│  - Maintains minimal coordination state                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ GraphStateActor  │  │ NodeManagementA. │  │ EdgeManagementA. │
│ - Graph data     │  │ - Node CRUD      │  │ - Edge CRUD      │
│ - State queries  │  │ - Node updates   │  │ - Edge updates   │
│ - Snapshots      │  │ - Batch ops      │  │ - Batch ops      │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ PhysicsActorCoord│  │ ClientSyncActor  │  │ MetadataAdapter  │
│ - Simulation     │  │ - WebSocket sync │  │ - Metadata→Graph │
│ - GPU delegation │  │ - Position broad │  │ - Build from MD  │
│ - Constraints    │  │ - Initial sync   │  │ - Update from MD │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                   │                   │
        └───────────────────┴───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ BotsGraphActor   │  │ PathfindingActor │  │ ConstraintsActor │
│ - Bots-specific  │  │ - Shortest paths │  │ - Constraint mgmt│
│ - Bots graph ops │  │ - GPU pathfinding│  │ - Semantic regen │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

### Benefits of This Architecture:

1. **Single Responsibility**: Each actor has ONE clear purpose
2. **Scalability**: Actors can be deployed on different threads/machines
3. **Testability**: Small, focused actors are easy to test
4. **Maintainability**: Changes to one domain don't affect others
5. **Performance**: Actors can process messages in parallel
6. **Fault Isolation**: Failure in one actor doesn't crash others
7. **Hexagonal Compliance**: Actors use CQRS handlers, not direct DB access

---

## 📋 Migration Plan (3 Phases)

### Phase 1: Extract Read-Only Actors (Low Risk) - 2 days
**Goal**: Extract actors that only query state

1. **Create PathfindingActor**
   - Extract `ComputeShortestPaths` handler
   - Use `LoadGraphQuery` from CQRS layer
   - Delegate to GPU semantic analyzer
   - ~200 lines

2. **Create BotsGraphActor**
   - Extract `GetBotsGraphData`, `UpdateBotsGraph`
   - Separate bots graph from main graph
   - ~300 lines

3. **Test Phase 1**
   - Unit tests for new actors
   - Integration tests with existing system
   - Verify performance unchanged

### Phase 2: Extract Core Domain Actors (Medium Risk) - 4 days
**Goal**: Extract primary CRUD operations

4. **Create NodeManagementActor**
   - Extract: `AddNode`, `RemoveNode`, `UpdateNodePosition`
   - Extract: `BatchAddNodes`
   - Use CQRS directives: `CreateNode`, `DeleteNode`
   - ~500 lines

5. **Create EdgeManagementActor**
   - Extract: `AddEdge`, `RemoveEdge`
   - Extract: `BatchAddEdges`, `BatchGraphUpdate`
   - Use CQRS directives: `CreateEdge`, `DeleteEdge`
   - ~400 lines

6. **Create MetadataIntegrationActor**
   - Extract: `BuildGraphFromMetadata`, `AddNodesFromMetadata`
   - Extract: `UpdateNodeFromMetadata`, `RemoveNodeByMetadata`
   - Bridge between metadata system and graph
   - ~600 lines

7. **Test Phase 2**
   - Full integration test suite
   - Load testing (10K nodes, 50K edges)
   - Verify CRUD operations work correctly

### Phase 3: Extract State & Coordination (High Risk) - 5 days
**Goal**: Final decomposition with coordination layer

8. **Create GraphStateActor**
   - Core state holder (nodes, edges, positions)
   - Handles: `GetGraphData`, `GetNodeMap`, `GetPhysicsState`
   - Immutable snapshots for queries
   - ~700 lines

9. **Create ClientSyncActor**
   - Extract: `InitialClientSync`, `ForcePositionBroadcast`
   - Extract: `UpdateNodePositions` (broadcast only)
   - WebSocket coordination
   - ~500 lines

10. **Create PhysicsCoordinatorActor**
    - Extract: `StartSimulation`, `StopSimulation`, `SimulationStep`
    - Extract: `UpdateSimulationParams`, `UpdateAdvancedParams`
    - Delegates to PhysicsOrchestratorActor (already exists)
    - ~400 lines

11. **Create ConstraintsActor**
    - Extract: `UpdateConstraints`, `GetConstraints`
    - Extract: `RegenerateSemanticConstraints`
    - ~300 lines

12. **Create GraphCoordinatorActor (Lightweight)**
    - Message router (replaces GraphServiceActor)
    - Orchestrates specialized actors
    - Minimal state (actor addresses only)
    - ~400 lines

13. **Test Phase 3**
    - Comprehensive end-to-end tests
    - Performance benchmarks (should improve 2-3x)
    - Stress testing (100K nodes, 500K edges)
    - Verify no regressions

---

## 🔄 Message Flow Example (After Decomposition)

**Before** (Current Monolith):
```
Client → GraphServiceActor (3910 lines)
           └─ Handles everything internally
```

**After** (Decomposed):
```
Client → GraphCoordinatorActor (router)
           ├─ AddNode → NodeManagementActor
           │              └─ Uses CQRS CreateNodeDirective
           │              └─ Notifies GraphStateActor
           │              └─ Triggers ClientSyncActor broadcast
           │
           ├─ GetGraphData → GraphStateActor
           │                   └─ Uses CQRS LoadGraphQuery
           │
           └─ ComputeShortestPaths → PathfindingActor
                                      └─ Queries GraphStateActor
                                      └─ Delegates to GPU
```

---

## 📊 Expected Metrics After Decomposition

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Largest Actor LOC** | 3,910 | ~700 | 5.6x smaller |
| **Actors** | 1 monolith | 9 specialized | Better separation |
| **Testability** | Low | High | +++++ |
| **Parallel Processing** | None | 9-way | 9x potential |
| **Message Latency** | 50ms | 15ms | 3.3x faster |
| **Fault Isolation** | None | Full | +++++ |
| **Maintainability** | Poor | Excellent | +++++ |

---

## 🚀 Quick Start (When Ready)

### Step 1: Create Feature Branch
```bash
git checkout -b feature/decompose-graph-actor
```

### Step 2: Start with Phase 1 (Low Risk)
```bash
# Create new actors
touch src/actors/pathfinding_actor.rs
touch src/actors/bots_graph_actor.rs

# Extract handlers
# (Follow Phase 1 plan above)
```

### Step 3: Incremental Testing
```bash
# After each actor extraction
cargo test --lib
cargo check --all-features
```

### Step 4: Update AppState
```rust
// src/app_state.rs
pub struct AppState {
    // OLD: pub graph_service: Addr<GraphServiceActor>,

    // NEW: Specialized actors
    pub graph_coordinator: Addr<GraphCoordinatorActor>,
    pub graph_state: Addr<GraphStateActor>,
    pub node_management: Addr<NodeManagementActor>,
    pub edge_management: Addr<EdgeManagementActor>,
    // ... other specialized actors
}
```

---

## ⚠️ Current Status

- **GraphServiceActor**: ⚠️ RETAINED (monolithic)
- **GraphServiceSupervisor**: ✅ RETAINED (needed for coordination)
- **Decomposition Status**: 📋 PLANNED (not yet implemented)
- **Recommended Priority**: HIGH (technical debt)

---

## 🎯 Why This Wasn't Done During Hexagonal Migration

The hexagonal migration (Phases 1-6) focused on:
1. ✅ **Application Layer** - CQRS handlers (45 handlers)
2. ✅ **Infrastructure Layer** - Adapters (8 adapters)
3. ✅ **Domain Layer** - Ports (10 traits)
4. ✅ **Database Layer** - 3-database architecture

**Actor layer decomposition is Phase 7** (not yet implemented):
- Requires careful message routing
- Risk of breaking real-time WebSocket updates
- Needs performance validation
- Estimated effort: 11 days (3 phases)

---

## 📚 References

- Actor Model Best Practices: https://doc.akka.io/docs/akka/current/typed/guide/actors-intro.html
- Actix Actor System: https://actix.rs/docs/actix/actor/
- Hexagonal Architecture: https://alistair.cockburn.us/hexagonal-architecture/
- Single Responsibility Principle: Clean Code by Robert C. Martin

---

**Status**: 📋 DOCUMENTED - Ready for implementation
**Priority**: HIGH - Technical debt that contradicts hexagonal architecture
**Estimated Effort**: 11 days (3 phases)
**Risk Level**: Medium-High (requires careful testing)

**Next Steps**:
1. Review this plan with stakeholders
2. Create Phase 1 feature branch
3. Extract PathfindingActor and BotsGraphActor (low risk)
4. Test thoroughly before proceeding to Phase 2
