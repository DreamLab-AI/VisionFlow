# 🏗️ Agent 2: Architecture Planner - Mission Brief

**Agent ID:** architecture-planner
**Type:** System Architect
**Priority:** Critical
**Compute Units:** 20
**Memory Quota:** 512 MB

## Mission Statement

Design complete hexagonal migration strategy. For each GraphServiceActor responsibility (graph state, WebSocket, physics), design hexagonal replacement using application handlers, repository ports, and event sourcing. Create step-by-step migration plan.

## Current Hexagonal Layer

**Existing Files:**
```
src/application/
├── knowledge_graph/
│   ├── directives.rs  (Command handlers)
│   ├── queries.rs     (Query handlers)
├── ontology/
│   ├── directives.rs
│   ├── queries.rs
├── settings/
    ├── directives.rs
    ├── queries.rs

src/ports/
├── knowledge_graph_repository.rs
├── graph_repository.rs
├── physics_simulator.rs
├── gpu_physics_adapter.rs
├── semantic_analyzer.rs
├── gpu_semantic_analyzer.rs
├── inference_engine.rs
├── ontology_repository.rs
├── settings_repository.rs

src/adapters/
├── actor_graph_repository.rs
└── (more adapters needed)
```

## GraphServiceActor Responsibilities to Migrate

### 1. Graph State Management (Lines 90-200)
**Current:** Monolithic state in actor
**Target:** CQRS with event sourcing

**Design:**
- **Command:** `AddNode`, `UpdateNode`, `RemoveNode`
- **Query:** `GetNode`, `ListNodes`, `GetGraphStats`
- **Events:** `NodeAdded`, `NodeUpdated`, `NodeRemoved`
- **Repository:** `KnowledgeGraphRepository` port

### 2. WebSocket Real-time Updates (Lines 500-800)
**Current:** Direct actor messaging
**Target:** Event-driven architecture

**Design:**
- **Event Bus:** Publish domain events
- **Subscribers:** WebSocket handler subscribes to events
- **Binary Protocol:** Move to adapter layer
- **Client Updates:** Event → Adapter → WebSocket

### 3. Physics Simulation (Lines 1200-2000)
**Current:** Coupled to actor lifecycle
**Target:** Domain service with ports

**Design:**
- **Port:** `PhysicsSimulator` trait
- **Adapter:** `GPUPhysicsAdapter` implementing port
- **Command:** `UpdatePhysicsParams`, `StepSimulation`
- **Event:** `PhysicsStateUpdated`

### 4. GPU Computation (Lines 2000-3000)
**Current:** Direct GPU manager communication
**Target:** Adapter pattern

**Design:**
- **Port:** `GPUPhysicsAdapter` (already exists)
- **Implementation:** Uses GPU manager internally
- **Decoupling:** Application layer unaware of GPU

### 5. Semantic Analysis (Lines 3000-3500)
**Current:** Embedded in actor
**Target:** Service with port

**Design:**
- **Port:** `SemanticAnalyzer` trait
- **Service:** `SemanticAnalysisService`
- **Command:** `AnalyzeGraph`, `GenerateEdges`
- **Event:** `SemanticsUpdated`

### 6. GitHub Sync (Lines 3500-4000)
**Current:** Mixed with actor logic
**Target:** Application service

**Design:**
- **Service:** `GitHubSyncService`
- **Command:** `SyncRepository`, `UpdateMetadata`
- **Query:** `GetSyncStatus`, `ListSyncedNodes`
- **Event:** `RepositorySynced`

### 7. Constraint Management (Lines 4000-4566)
**Current:** Part of actor state
**Target:** Domain model with repository

**Design:**
- **Model:** `Constraint`, `ConstraintSet` (already exists)
- **Repository:** `ConstraintRepository` port
- **Command:** `AddConstraint`, `UpdateConstraints`
- **Event:** `ConstraintsChanged`

## Migration Strategy

### Phase 1: Extract Read Operations (Week 1)
**Goal:** All queries through hexagonal layer

**Tasks:**
1. Create query handlers in `src/application/knowledge_graph/queries.rs`
2. Implement repository pattern for read-only operations
3. Route API GET requests through query handlers
4. Keep GraphServiceActor as fallback
5. Validate: All reads work through new layer

### Phase 2: Extract Write Operations (Week 2)
**Goal:** All mutations through command handlers

**Tasks:**
1. Create command handlers in `src/application/knowledge_graph/directives.rs`
2. Implement command validation logic
3. Route API POST/PUT/DELETE through commands
4. Event sourcing for state changes
5. Validate: All writes work through new layer

### Phase 3: Event Sourcing for WebSocket (Week 3)
**Goal:** Real-time updates via events

**Tasks:**
1. Design domain events for all state changes
2. Create event bus/publisher
3. Migrate WebSocket handler to subscribe to events
4. Remove direct actor communication
5. Validate: Real-time updates work correctly

### Phase 4: Domain Services for Physics (Week 4)
**Goal:** Physics as pluggable service

**Tasks:**
1. Extract physics logic to domain service
2. Use `PhysicsSimulator` port
3. Implement GPU adapter
4. Remove physics from actor
5. Validate: Simulation runs correctly

### Phase 5: Legacy Removal (Week 5)
**Goal:** Delete GraphServiceActor entirely

**Tasks:**
1. Verify zero dependencies remain
2. Remove graph_actor.rs
3. Clean up GPU supervisor actors
4. Update documentation
5. Validate: System works without monolith

## Deliverables

Create: `/home/devuser/workspace/project/docs/migration/hexagonal-migration-plan.md`

**Required Sections:**
1. **Architecture Diagrams** - Before/after hexagonal layer
2. **Command/Query Catalog** - All operations mapped
3. **Event Catalog** - Domain events for state changes
4. **Port Definitions** - Interface contracts
5. **Adapter Specifications** - Implementation details
6. **Migration Phases** - Week-by-week breakdown
7. **Risk Analysis** - Potential issues and mitigation
8. **Rollback Strategy** - How to undo if needed

## Memory Storage

Store plan under: `hive-coordination/planning/hexagonal_migration_plan`

**JSON Structure:**
```json
{
  "phases": [
    {
      "id": 1,
      "name": "Extract Read Operations",
      "duration_days": 7,
      "tasks": [...],
      "validation_criteria": [...]
    },
    ...
  ],
  "commands": ["AddNode", "UpdateNode", ...],
  "queries": ["GetNode", "ListNodes", ...],
  "events": ["NodeAdded", "NodeUpdated", ...],
  "ports": ["KnowledgeGraphRepository", "PhysicsSimulator", ...],
  "risk_score": 7
}
```

## Success Criteria

✅ Complete architecture design for all 7 responsibilities
✅ CQRS command/query separation defined
✅ Event sourcing strategy documented
✅ Port interfaces specified
✅ 5-phase migration plan with validation
✅ Risk analysis with mitigation
✅ Findings stored in memory

---
*Assigned by Queen Coordinator*
