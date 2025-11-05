# GraphServiceActor → Modular Actors Migration Plan

**Status:** READY TO EXECUTE
**Date:** 2025-11-05
**Directive:** Complete migration without backward compatibility

## Current Architecture (DEPRECATED)

```
AppState
└── TransitionalGraphSupervisor (wrapper)
    └── GraphServiceActor (4615-line god object)
        ├── Graph state (691 lines of mixed concerns)
        ├── Physics (812 lines)
        ├── GPU management (420 lines)
        ├── Constraints (350 lines)
        ├── Semantic analysis (280 lines)
        ├── Auto-balancing (615 lines)
        ├── Client management (310 lines)
        └── Message queue (145 lines)
```

## Target Architecture (NEW)

```
AppState
└── GraphServiceSupervisor
    ├── GraphStateActor (712 lines - graph data only)
    ├── PhysicsOrchestratorActor (dedicated physics)
    ├── SemanticProcessorActor (dedicated semantic)
    └── ClientCoordinatorActor (existing)
```

## Files to DELETE

1. **src/actors/graph_actor.rs** (4615 lines)
   - The god object identified in audit
   - All functionality distributed to modular actors

2. **src/actors/backward_compat.rs** (240 lines)
   - Deprecated compatibility layer
   - Legacy message wrappers
   - Migration helpers no longer needed

3. **TransitionalGraphSupervisor from graph_service_supervisor.rs** (lines 882-1320)
   - Temporary wrapper around deprecated actor
   - All handlers forward to old GraphServiceActor

## Files to UPDATE

### 1. src/actors/graph_service_supervisor.rs

**Changes:**
- Line 44: Remove `GraphServiceActor` import, add `GraphStateActor`
- Line 167: Change `Option<Addr<GraphServiceActor>>` → `Option<Addr<GraphStateActor>>`
- Lines 365-389: Update actor spawn to use `GraphStateActor::new(repository)`
- Lines 375-386: Remove client_addr dependency (GraphStateActor only needs repository)
- Lines 882-1320: DELETE entire TransitionalGraphSupervisor section
- Lines 786-880: Implement proper message forwarding to modular actors

**Key actor spawn change:**
```rust
// OLD (line 375-380)
let actor = GraphServiceActor::new(
    client_addr,
    None,
    kg_repo.clone(),
    None,
).start();

// NEW
let actor = GraphStateActor::new(kg_repo.clone()).start();
```

### 2. src/actors/mod.rs

**Remove exports:**
```rust
pub mod graph_actor;  // DELETE
pub use graph_actor::GraphServiceActor;  // DELETE
pub mod backward_compat;  // DELETE
pub use backward_compat::{LegacyActorCompat, MigrationHelper};  // DELETE
```

**Add exports:**
```rust
pub mod graph_state_actor;  // ADD
pub use graph_state_actor::GraphStateActor;  // ADD
```

### 3. src/app_state.rs

**Line 27:** Remove `TransitionalGraphSupervisor` import
**Line 82:** Change type
```rust
// OLD
pub graph_service_addr: Addr<TransitionalGraphSupervisor>,

// NEW
pub graph_service_addr: Addr<GraphServiceSupervisor>,
```

**Line 270+:** Update initialization (remove `with_dependencies`, use `new()`)

### 4. src/main.rs

Update supervisor initialization to use new modular architecture:
```rust
// OLD
let graph_service_supervisor = GraphServiceSupervisor::with_dependencies(
    Some(client_manager_addr),
    gpu_manager_addr,
    kg_repo,
).start();

// NEW
let graph_service_supervisor = GraphServiceSupervisor::new().start();
```

### 5. src/handlers/* (6 files reference old actor)

Files to update:
- src/handlers/socket_flow_handler.rs
- src/handlers/api_handler/files/mod.rs
- src/handlers/api_handler/graph/mod.rs
- src/handlers/admin_sync_handler.rs

**Change pattern:**
```rust
// OLD
state.graph_service_addr.send(Message).await

// NEW - message routing through supervisor, OR
// Direct to specific actor if needed
```

## Implementation Order

1. ✅ Create MIGRATION_PLAN.md (this file)
2. Delete graph_actor.rs
3. Delete backward_compat.rs
4. Update graph_service_supervisor.rs (remove Transitional, fix supervisor)
5. Update actors/mod.rs (exports)
6. Update app_state.rs (type change)
7. Update main.rs (initialization)
8. Update handlers (message routing)
9. Test compilation
10. Commit migration

## Breaking Changes

- `TransitionalGraphSupervisor` no longer exists
- `GraphServiceActor` no longer exists
- `backward_compat` module deleted
- Direct actor messaging replaced with supervisor routing
- Some message types may need updating

## Benefits

- ✅ Removes 4615-line god object
- ✅ Clean separation of concerns
- ✅ Each actor <800 lines
- ✅ Easier testing and maintenance
- ✅ Fixes C6 critical issue from audit
- ✅ No backward compatibility burden

## Risks

- Handlers may need message routing updates
- Some tests may break temporarily
- Compilation errors expected until all references updated

## Rollback Plan

If issues arise:
- Git revert to current commit
- All old code preserved in git history

## Success Criteria

- ✅ Compiles without errors
- ✅ GraphServiceActor deleted
- ✅ TransitionalGraphSupervisor deleted
- ✅ backward_compat deleted
- ✅ All handlers use new supervisor or direct actors
- ✅ Tests pass
