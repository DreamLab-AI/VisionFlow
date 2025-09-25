# GraphServiceActor Refactoring Plan

## Executive Summary

The `GraphServiceActor` has grown to **3,104 lines** (estimated ~38,456 tokens) and handles multiple complex responsibilities. This document outlines a comprehensive refactoring plan to decompose it into smaller, focused actors while maintaining system stability and performance.

## Current State Analysis

### File Location
- **Primary File**: `/workspace/ext/src/actors/graph_actor.rs` (Lines: 3,104)
- **Related Files**:
  - `/workspace/ext/src/actors/messages/` (message definitions)
  - `/workspace/ext/src/models/` (data structures)
  - `/workspace/ext/src/services/` (semantic analysis, edge generation)

### Current Responsibilities

The `GraphServiceActor` currently handles:

1. **Graph State Management** (Lines 86-93)
   - Node and edge storage (`graph_data`, `node_map`)
   - Bots graph data management
   - Node ID generation

2. **Physics Simulation** (Lines 88, 94, 109-131)
   - GPU compute coordination
   - Simulation parameters management
   - Auto-balance tracking and notifications
   - Kinetic energy monitoring

3. **Advanced AI Features** (Lines 96-112)
   - Constraint management (`constraint_set`)
   - Semantic analysis (`semantic_analyzer`)
   - Edge generation (`edge_generator`)
   - Stress majorization (`stress_solver`)
   - Feature caching

4. **Client Communication** (Lines 89, 122-124)
   - WebSocket message handling
   - Position broadcasting
   - Initial synchronization

5. **Message Handling** (35+ Handler implementations, Lines 1990-2935)
   - Graph data operations
   - Simulation control
   - Advanced physics parameter updates
   - GPU initialization

### Performance Concerns

- **Token Count**: ~38,456 tokens (extremely large for a single actor)
- **Message Handlers**: 35+ different message types
- **Complexity**: Multiple state machines within a single actor
- **Maintainability**: Difficult to test and modify individual features

## Proposed Actor Decomposition

### 1. GraphServiceActor (Supervisor)
**New Role**: Lightweight coordinator and supervisor
**Location**: `/workspace/ext/src/actors/graph_service_supervisor.rs`
**Lines**: ~500-800

```rust
pub struct GraphServiceSupervisor {
    // Child actor addresses
    graph_state_addr: Addr<GraphStateActor>,
    physics_orchestrator_addr: Addr<PhysicsOrchestratorActor>,
    semantic_processor_addr: Addr<SemanticProcessorActor>,
    client_coordinator_addr: Addr<ClientCoordinatorActor>,

    // Minimal state
    shutdown_complete: Arc<AtomicBool>,
    settings_addr: Option<Addr<SettingsActor>>,
}
```

**Responsibilities**:
- Spawn and supervise child actors
- Route messages to appropriate actors
- Handle graceful shutdown
- Coordinate cross-actor operations

### 2. GraphStateActor
**Location**: `/workspace/ext/src/actors/graph_state_actor.rs`
**Lines**: ~800-1200

```rust
pub struct GraphStateActor {
    graph_data: Arc<GraphData>,
    node_map: Arc<HashMap<u32, Node>>,
    bots_graph_data: Arc<GraphData>,
    next_node_id: AtomicU32,
}
```

**Responsibilities**:
- Node and edge CRUD operations
- Graph data storage and retrieval
- Metadata integration
- Node ID management

**Message Handlers**:
- `GetGraphData` (Line 1990)
- `AddNode` (Line 2087)
- `RemoveNode` (Line 2096)
- `AddEdge` (Line 2105)
- `RemoveEdge` (Line 2114)
- `GetNodeMap` (Line 2123)
- `BuildGraphFromMetadata` (Line 2131)

### 3. PhysicsOrchestratorActor
**Location**: `/workspace/ext/src/actors/physics_orchestrator_actor.rs`
**Lines**: ~1000-1500

```rust
pub struct PhysicsOrchestratorActor {
    gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    simulation_running: AtomicBool,
    simulation_params: SimulationParams,
    target_params: SimulationParams,
    param_transition_rate: f32,

    // Auto-balance components
    auto_balance_history: Vec<f32>,
    stable_count: u32,
    kinetic_energy_history: Vec<f32>,
    current_state: AutoBalanceState,
    auto_balance_notifications: Arc<Mutex<Vec<AutoBalanceNotification>>>,
}
```

**Responsibilities**:
- Physics simulation control
- GPU compute coordination
- Auto-balance monitoring
- Parameter transitions
- Performance monitoring

**Message Handlers**:
- `StartSimulation` (Line 2188)
- `StopSimulation` (Line 2197)
- `SimulationStep` (Line 2252)
- `UpdateSimulationParams` (Line 2607)
- `GetAutoBalanceNotifications` (Line 2262)
- `PhysicsPauseMessage` (Line 2881)
- `ForceResumePhysics` (Line 2926)

### 4. SemanticProcessorActor
**Location**: `/workspace/ext/src/actors/semantic_processor_actor.rs`
**Lines**: ~800-1200

```rust
pub struct SemanticProcessorActor {
    constraint_set: ConstraintSet,
    semantic_analyzer: SemanticAnalyzer,
    edge_generator: AdvancedEdgeGenerator,
    stress_solver: StressMajorizationSolver,
    semantic_features_cache: HashMap<String, SemanticFeatures>,
    advanced_params: AdvancedParams,

    // Control flow
    stress_step_counter: u32,
    constraint_update_counter: u32,
    last_semantic_analysis: Option<std::time::Instant>,
}
```

**Responsibilities**:
- Semantic analysis and feature extraction
- Constraint generation and management
- Edge generation algorithms
- Stress majorization optimization
- Feature caching

**Message Handlers**:
- `UpdateAdvancedParams` (Line 2716)
- `UpdateConstraints` (Line 2739)
- `GetConstraints` (Line 2747)
- `TriggerStressMajorization` (Line 2755)
- `RegenerateSemanticConstraints` (Line 2765)

### 5. ClientCoordinatorActor
**Location**: `/workspace/ext/src/actors/client_coordinator_actor.rs`
**Lines**: ~600-1000

```rust
pub struct ClientCoordinatorActor {
    client_manager: Addr<ClientManagerActor>,
    last_broadcast_time: Option<std::time::Instant>,
    initial_positions_sent: bool,
    previous_positions: HashMap<u32, Vec3Data>,
}
```

**Responsibilities**:
- Client synchronization
- Position broadcasting
- WebSocket message routing
- Initial client setup

**Message Handlers**:
- `InitialClientSync` (Line 2048)
- `ForcePositionBroadcast` (Line 2009)
- `UpdateNodePositions` (Line 1999)
- `RequestPositionSnapshot` (Line 2668)

## Message Flow Diagrams

### Current Monolithic Flow
```
Client → GraphServiceActor ← GPU
              ↕
        [All Responsibilities]
              ↕
        WebSocket Response
```

### Proposed Distributed Flow

#### 1. Node Operations Flow
```
Client Request
      ↓
GraphServiceSupervisor
      ↓
GraphStateActor
      ↓
[Node/Edge Operations]
      ↓
Response via Supervisor
```

#### 2. Physics Simulation Flow
```
Simulation Timer
      ↓
PhysicsOrchestratorActor
      ↓
GPU Compute Actor
      ↓
Position Updates → ClientCoordinatorActor
      ↓
WebSocket Broadcast
```

#### 3. Semantic Analysis Flow
```
Graph Changes
      ↓
SemanticProcessorActor
      ↓
[Analysis & Constraint Generation]
      ↓
Constraint Updates → PhysicsOrchestratorActor
```

#### 4. Cross-Actor Coordination
```
GraphServiceSupervisor
    ├── GraphStateActor
    ├── PhysicsOrchestratorActor
    ├── SemanticProcessorActor
    └── ClientCoordinatorActor
         ↓
    Message Routing & State Sync
```

## Step-by-Step Implementation Plan

### Phase 1: Infrastructure Setup (Days 1-2)
**Goal**: Create new actor structures without breaking existing functionality

1. **Create Actor Skeletons**
   ```bash
   # New files to create
   /workspace/ext/src/actors/graph_service_supervisor.rs
   /workspace/ext/src/actors/graph_state_actor.rs
   /workspace/ext/src/actors/physics_orchestrator_actor.rs
   /workspace/ext/src/actors/semantic_processor_actor.rs
   /workspace/ext/src/actors/client_coordinator_actor.rs
   ```

2. **Update Module Structure**
   - Modify `/workspace/ext/src/actors/mod.rs` to include new actors
   - Update imports in dependent modules

3. **Create Message Routing**
   - Define inter-actor messages in `/workspace/ext/src/actors/messages/mod.rs`
   - Implement message forwarding in supervisor

4. **Cargo Check Strategy**
   ```bash
   cargo check --all-targets
   # Should compile with warnings but no errors
   ```

### Phase 2: GraphStateActor Extraction (Days 3-4)
**Goal**: Extract graph state management functionality

1. **Extract Core State (Lines 86-93, 2087-2180)**
   - Move `graph_data`, `node_map`, `bots_graph_data` to GraphStateActor
   - Implement handlers: `GetGraphData`, `AddNode`, `RemoveNode`, `AddEdge`, `RemoveEdge`
   - Move metadata-related handlers: `BuildGraphFromMetadata`, `AddNodesFromMetadata`

2. **Update GraphServiceActor**
   - Replace direct state access with message passing
   - Route graph operations to GraphStateActor

3. **Testing Strategy**
   ```bash
   cargo test actors::graph_state_actor::tests
   cargo test integration::graph_operations
   cargo check --all-targets
   ```

### Phase 3: PhysicsOrchestratorActor Extraction (Days 5-7)
**Goal**: Extract physics and simulation functionality

1. **Extract Physics Components (Lines 88, 94, 2188-2262, 2881-2935)**
   - Move `gpu_compute_addr`, `simulation_running`, `simulation_params`
   - Move auto-balance tracking components (Lines 115-121)
   - Implement handlers: `StartSimulation`, `StopSimulation`, `SimulationStep`

2. **Extract GPU Coordination (Lines 2532-2607)**
   - Move GPU initialization logic
   - Implement `StoreGPUComputeAddress`, `InitializeGPUConnection`

3. **Auto-Balance Integration**
   - Move auto-balance notification system
   - Implement `GetAutoBalanceNotifications`

4. **Testing Strategy**
   ```bash
   cargo test actors::physics_orchestrator_actor::tests
   cargo test simulation::physics_integration
   cargo check --all-targets
   ```

### Phase 4: SemanticProcessorActor Extraction (Days 8-10)
**Goal**: Extract AI and semantic analysis functionality

1. **Extract Semantic Components (Lines 99-112, 2716-2789)**
   - Move `constraint_set`, `semantic_analyzer`, `edge_generator`
   - Move `stress_solver`, `semantic_features_cache`
   - Implement handlers: `UpdateAdvancedParams`, `UpdateConstraints`

2. **Extract Advanced Features**
   - Move stress majorization logic
   - Implement `TriggerStressMajorization`, `RegenerateSemanticConstraints`

3. **Feature Caching**
   - Implement semantic feature cache management
   - Add cache invalidation strategies

4. **Testing Strategy**
   ```bash
   cargo test actors::semantic_processor_actor::tests
   cargo test semantic::analysis_integration
   cargo check --all-targets
   ```

### Phase 5: ClientCoordinatorActor Extraction (Days 11-12)
**Goal**: Extract client communication functionality

1. **Extract Client Communication (Lines 89, 122-124, 1999-2048)**
   - Move `client_manager` address
   - Move position broadcasting logic
   - Implement handlers: `InitialClientSync`, `ForcePositionBroadcast`

2. **Position Management**
   - Move position change tracking (Lines 129-131)
   - Implement efficient position diffing

3. **Testing Strategy**
   ```bash
   cargo test actors::client_coordinator_actor::tests
   cargo test websocket::client_integration
   cargo check --all-targets
   ```

### Phase 6: Supervisor Implementation (Days 13-14)
**Goal**: Implement coordination and message routing

1. **Message Routing**
   - Implement intelligent message forwarding
   - Handle cross-actor communication patterns

2. **State Synchronization**
   - Implement state sharing mechanisms
   - Handle actor lifecycle management

3. **Error Handling**
   - Implement graceful degradation
   - Add supervisor restart strategies

4. **Testing Strategy**
   ```bash
   cargo test actors::graph_service_supervisor::tests
   cargo test integration::full_system
   cargo check --all-targets
   ```

### Phase 7: Integration & Migration (Days 15-16)
**Goal**: Migrate existing code to use new actor system

1. **Update Actor Spawning**
   - Modify `/workspace/ext/src/main.rs` or equivalent
   - Replace GraphServiceActor spawn with GraphServiceSupervisor

2. **Update Message Senders**
   - Find all code that sends messages to GraphServiceActor
   - Update to send to GraphServiceSupervisor

3. **Comprehensive Testing**
   ```bash
   cargo test --all
   cargo check --all-targets
   cargo clippy --all-targets
   ```

## Risk Mitigation Strategies

### 1. Compilation Risks
**Risk**: Breaking existing compilation
**Mitigation**:
- Maintain backward compatibility during transition
- Use feature flags to toggle between old/new implementations
- Run `cargo check` after each extraction phase

### 2. Message Ordering Risks
**Risk**: Race conditions between actors
**Mitigation**:
- Implement message sequencing in supervisor
- Use tokio channels for ordered communication
- Add integration tests for concurrent scenarios

### 3. Performance Risks
**Risk**: Inter-actor communication overhead
**Mitigation**:
- Benchmark message passing performance
- Use shared memory for large data structures (Arc<>)
- Implement batching for frequent operations

### 4. State Synchronization Risks
**Risk**: Inconsistent state between actors
**Mitigation**:
- Implement atomic state updates
- Add state validation checks
- Use distributed locking for critical sections

### 5. Testing Coverage Risks
**Risk**: Reduced test coverage during refactoring
**Mitigation**:
- Write tests for each new actor before extraction
- Maintain integration tests throughout process
- Add regression tests for critical paths

## Testing Approach for Each Phase

### Unit Testing Strategy
```rust
// Example test structure for each actor
#[cfg(test)]
mod tests {
    use super::*;
    use actix::test;

    #[actix::test]
    async fn test_actor_initialization() {
        // Test actor creation and startup
    }

    #[actix::test]
    async fn test_message_handling() {
        // Test individual message handlers
    }

    #[actix::test]
    async fn test_error_conditions() {
        // Test error handling and recovery
    }
}
```

### Integration Testing Strategy
```rust
// Test inter-actor communication
#[actix::test]
async fn test_cross_actor_workflow() {
    let supervisor = GraphServiceSupervisor::new(/* params */);

    // Test complete workflow across multiple actors
    let result = supervisor.send(ComplexOperation).await;
    assert!(result.is_ok());
}
```

### Performance Testing
```rust
// Benchmark actor performance
#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_message_throughput(c: &mut Criterion) {
        c.bench_function("actor_message_throughput", |b| {
            b.iter(|| {
                // Benchmark message processing speed
            });
        });
    }
}
```

## Success Metrics

### Code Quality Metrics
- **Line Reduction**: Target <1000 lines per actor (currently 3,104 total)
- **Token Reduction**: Target <10,000 tokens per actor (currently ~38,456 total)
- **Handler Count**: Target <10 handlers per actor (currently 35+ total)
- **Cyclomatic Complexity**: Target <10 per function

### Performance Metrics
- **Message Latency**: <1ms inter-actor communication
- **Memory Usage**: No significant increase in memory footprint
- **Throughput**: Maintain current simulation performance (fps)
- **Startup Time**: <5% increase in system startup time

### Reliability Metrics
- **Test Coverage**: >90% line coverage for each new actor
- **Integration Tests**: 100% critical path coverage
- **Error Handling**: All error paths tested and documented
- **Compilation**: Zero compilation errors, minimal new warnings

### Maintainability Metrics
- **Documentation**: 100% public API documentation
- **Code Review**: All new code reviewed and approved
- **Technical Debt**: Reduce overall technical debt score
- **Onboarding**: New developers can understand individual actors in <1 hour

## File Locations Summary

### New Files to Create
```
/workspace/ext/src/actors/
├── graph_service_supervisor.rs     (~500-800 lines)
├── graph_state_actor.rs           (~800-1200 lines)
├── physics_orchestrator_actor.rs  (~1000-1500 lines)
├── semantic_processor_actor.rs    (~800-1200 lines)
└── client_coordinator_actor.rs    (~600-1000 lines)
```

### Files to Modify
```
/workspace/ext/src/actors/
├── mod.rs                         (add new actor modules)
└── messages/                      (add inter-actor messages)

/workspace/ext/src/
├── main.rs                        (update actor spawning)
└── [various files]                (update message senders)
```

### Files to Eventually Remove
```
/workspace/ext/src/actors/
└── graph_actor.rs                 (3,104 lines → deprecated)
```

## Conclusion

This refactoring plan will transform the monolithic GraphServiceActor into a well-structured, maintainable actor system. The phased approach ensures system stability while systematically reducing complexity. Each phase includes comprehensive testing and validation to minimize risks.

**Expected Outcomes**:
- **75% reduction** in individual actor complexity
- **Improved maintainability** through separation of concerns
- **Better testability** with focused unit tests
- **Enhanced scalability** through distributed processing
- **Reduced technical debt** and improved code quality

The total effort is estimated at **16 days** with a team of 2-3 developers, including comprehensive testing and documentation.