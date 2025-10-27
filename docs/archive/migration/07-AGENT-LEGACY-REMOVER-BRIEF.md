# 🗑️ Agent 7: Legacy Code Remover - Mission Brief

**Agent ID:** legacy-remover
**Type:** Code Cleanup Engineer
**Priority:** Medium
**Compute Units:** 5
**Memory Quota:** 256 MB

## Mission Statement

Once migration is complete and validated, REMOVE deprecated files completely. Delete GraphServiceActor, GPU supervisor actors, physics orchestrator, and all deprecated adapters. Verify system compiles and all tests pass after removal. This is the final step to achieve pure hexagonal architecture.

## CRITICAL: Execute ONLY After Validation

**Prerequisites:**
- ✅ Integration Validator confirms ALL phases passed
- ✅ GitHub sync verified (316 nodes, 100% public metadata)
- ✅ All tests passing with hexagonal layer
- ✅ Zero dependencies on deprecated code confirmed
- ✅ Queen Coordinator approval granted

**If ANY prerequisite fails:** STOP and report to Queen immediately.

## Files to Remove

### Phase 1: Remove GraphServiceActor
**File:** `src/actors/graph_actor.rs` (4,566 lines)

**Before Deletion:**
1. **Verify No Dependencies**
```bash
# Search for remaining imports
grep -r "use.*graph_actor::" src/
grep -r "GraphServiceActor" src/

# Should return ZERO results (except in deprecated files)
```

2. **Backup for Rollback**
```bash
mkdir -p .migration_backup/$(date +%Y%m%d)
cp src/actors/graph_actor.rs .migration_backup/$(date +%Y%m%d)/
```

3. **Delete**
```bash
git rm src/actors/graph_actor.rs
git commit -m "[migration] Remove GraphServiceActor monolith - migrated to CQRS handlers"
```

4. **Verify Compilation**
```bash
cargo check
# Should compile without errors
```

### Phase 2: Remove Graph Service Supervisor
**File:** `src/actors/graph_service_supervisor.rs`

**Steps:**
1. Verify no usage: `grep -r "graph_service_supervisor" src/`
2. Backup: `cp src/actors/graph_service_supervisor.rs .migration_backup/$(date +%Y%m%d)/`
3. Delete: `git rm src/actors/graph_service_supervisor.rs`
4. Verify: `cargo check`

### Phase 3: Remove GPU Supervisor Actors

**Files to Remove:**
- `src/actors/gpu/gpu_manager_actor.rs`
- `src/actors/gpu/gpu_resource_actor.rs`
- `src/actors/gpu/force_compute_actor.rs`
- `src/actors/gpu/stress_majorization_actor.rs`
- `src/actors/gpu/clustering_actor.rs`
- `src/actors/gpu/anomaly_detection_actor.rs`
- `src/actors/gpu/constraint_actor.rs`
- `src/actors/gpu/ontology_constraint_actor.rs`

**Files to KEEP (utilities):**
- `src/actors/gpu/shared.rs`
- `src/actors/gpu/cuda_stream_wrapper.rs`
- `src/actors/gpu/mod.rs` (update to only export utilities)

**Steps:**
```bash
# Verify no dependencies on actor files
for file in gpu_manager_actor gpu_resource_actor force_compute_actor stress_majorization_actor clustering_actor anomaly_detection_actor constraint_actor ontology_constraint_actor; do
    echo "Checking $file..."
    grep -r "$file" src/ --exclude-dir=actors/gpu
done

# Backup
cp -r src/actors/gpu .migration_backup/$(date +%Y%m%d)/gpu_actors_backup

# Remove actor files
cd src/actors/gpu
git rm gpu_manager_actor.rs gpu_resource_actor.rs force_compute_actor.rs \
       stress_majorization_actor.rs clustering_actor.rs anomaly_detection_actor.rs \
       constraint_actor.rs ontology_constraint_actor.rs

# Update mod.rs to only expose utilities
cat > mod.rs << 'EOF'
//! GPU utilities for physics computation
//!
//! This module provides low-level CUDA/GPU utilities used by adapters.
//! GPU coordination is now handled by adapters implementing GPUPhysicsAdapter port.

pub mod shared;
pub mod cuda_stream_wrapper;

pub use shared::*;
pub use cuda_stream_wrapper::*;
EOF

git add mod.rs
git commit -m "[migration] Remove GPU supervisor actors - now using adapter pattern"
```

### Phase 4: Remove Physics Orchestrator
**File:** `src/actors/physics_orchestrator_actor.rs`

**Steps:**
1. Verify no usage: `grep -r "physics_orchestrator" src/`
2. Backup: `cp src/actors/physics_orchestrator_actor.rs .migration_backup/$(date +%Y%m%d)/`
3. Delete: `git rm src/actors/physics_orchestrator_actor.rs`
4. Verify: `cargo check`

### Phase 5: Clean Actor Module Exports

**File:** `src/actors/mod.rs`

**Update to remove deprecated exports:**
```rust
// src/actors/mod.rs - AFTER cleanup

// Remove these lines:
// pub mod graph_actor;                  // DELETED
// pub mod graph_service_supervisor;     // DELETED
// pub mod physics_orchestrator_actor;   // DELETED

// Keep only:
pub mod client_coordinator_actor;
pub mod metadata_actor;
pub mod task_orchestrator_actor;
pub mod multi_mcp_visualization_actor;
pub mod agent_monitor_actor;
pub mod graph_messages;
pub mod messages;
pub mod ontology_actor;
pub mod workspace_actor;
pub mod supervisor;
pub mod optimized_settings_actor;
pub mod protected_settings_actor;
pub mod graph_state_actor;
pub mod semantic_processor_actor;
pub mod voice_commands;

// GPU utilities only (actors removed)
pub mod gpu;

// Re-exports (remove deprecated)
pub use client_coordinator_actor::ClientCoordinatorActor;
// pub use graph_actor::GraphServiceActor;  // REMOVE THIS LINE
```

### Phase 6: Clean Deprecated Adapters

**File:** `src/adapters/actor_graph_repository.rs`

**If this adapter wraps GraphServiceActor:**
- Remove the file entirely
- Replace with direct hexagonal repository implementation

**If it's been updated to hexagonal pattern:**
- Keep it but verify it doesn't import graph_actor

### Phase 7: Update Main Application State

**File:** `src/app_state.rs`

**Remove GraphServiceActor references:**
```rust
// BEFORE
pub struct AppState {
    pub graph_actor: Addr<GraphServiceActor>,  // REMOVE
    pub gpu_manager: Addr<GPUManagerActor>,    // REMOVE
    // ...
}

// AFTER
pub struct AppState {
    pub query_handler: Arc<KnowledgeGraphQueryHandler>,
    pub command_handler: Arc<KnowledgeGraphCommandHandler>,
    pub event_publisher: Arc<dyn EventPublisher>,
    pub physics_service: Arc<dyn PhysicsSimulator>,
    // ...
}
```

### Phase 8: Update Main Initialization

**File:** `src/main.rs`

**Remove actor spawning:**
```rust
// BEFORE
let graph_actor = GraphServiceActor::new(...).start();
let gpu_manager = GPUManagerActor::new(...).start();

// AFTER
let repository = Arc::new(PostgresKnowledgeGraphRepository::new(pool));
let event_publisher = Arc::new(BroadcastEventPublisher::new(1000));
let query_handler = Arc::new(KnowledgeGraphQueryHandler::new(repository.clone()));
let command_handler = Arc::new(KnowledgeGraphCommandHandler::new(repository, event_publisher.clone()));
let physics_service = Arc::new(GPUPhysicsService::new(gpu_adapter, event_publisher.clone()));
```

## Validation After Each Removal

**After EVERY file deletion:**

1. **Compile Check**
```bash
cargo check
# Must succeed with zero errors
```

2. **Test Suite**
```bash
cargo test
# All tests must pass
```

3. **Integration Tests**
```bash
cargo test --test integration_tests
# All integration tests must pass
```

4. **Build Full Binary**
```bash
cargo build --release
# Must build successfully
```

## Final Validation

**After ALL removals complete:**

### 1. Full System Test
```bash
# Start application
cargo run --release

# In another terminal, run validation suite
./scripts/validate_migration.sh
```

### 2. GitHub Sync Validation
```bash
# Trigger sync
curl -X POST http://localhost:8080/api/github/sync \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"test_repository"}'

# Verify 316 nodes
curl http://localhost:8080/api/graph/stats | jq '.node_count'
# Expected: 316
```

### 3. WebSocket Real-time Test
```javascript
// Connect WebSocket
const ws = new WebSocket('ws://localhost:8080/ws/graph');

// Trigger update
curl -X POST http://localhost:8080/api/graph/nodes -d '{"name":"test"}'

// Verify WebSocket receives update within 50ms
```

### 4. Physics Simulation Test
```bash
# Verify simulation running
curl http://localhost:8080/api/physics/status
# Expected: {"status":"running","fps":60}
```

## Deliverables

### Primary Deliverable
Create: `/home/devuser/workspace/project/docs/migration/removal-log.md`

**Required Sections:**
1. **Files Removed**
   - File path
   - Line count
   - Deletion timestamp
   - Git commit hash
2. **Backup Locations**
   - Backup directory
   - Files backed up
   - Restore instructions
3. **Compilation Results**
   - Errors encountered (if any)
   - Fixes applied
   - Final build status
4. **Test Results**
   - Tests run
   - Tests passed
   - Any failures (should be zero)
5. **Code Metrics**
   - Lines removed
   - Files removed
   - Final codebase size reduction
6. **Rollback Procedure**
   - How to restore deleted code
   - Rollback validation steps

### Git Commit Summary
Example commit messages:
```
[migration] Remove GraphServiceActor monolith (4,566 lines)
[migration] Remove GPU supervisor actors (2,300 lines)
[migration] Remove physics orchestrator actor (850 lines)
[migration] Clean up actor module exports
[migration] Update AppState to use hexagonal services
[migration] Complete migration to hexagonal architecture
```

## Memory Storage

Store removal log under: `hive-coordination/removal/deleted_files`

**JSON Structure:**
```json
{
  "removal_summary": {
    "files_removed": 12,
    "lines_removed": 8566,
    "backup_location": ".migration_backup/20251026/",
    "git_commits": [
      "abc123 - Remove GraphServiceActor monolith",
      "def456 - Remove GPU supervisor actors",
      ...
    ]
  },
  "validation_results": {
    "compilation": "success",
    "tests_passed": 156,
    "tests_failed": 0,
    "github_sync_validated": true,
    "websocket_validated": true,
    "physics_validated": true
  },
  "codebase_metrics": {
    "before_lines": 85000,
    "after_lines": 76434,
    "reduction_percent": 10.1
  },
  "rollback_tested": true
}
```

## Rollback Procedure

**If ANYTHING goes wrong during removal:**

1. **Stop immediately**
2. **Restore from backup:**
```bash
cp -r .migration_backup/$(date +%Y%m%d)/* src/actors/
git checkout src/actors/mod.rs
git checkout src/app_state.rs
git checkout src/main.rs
```

3. **Verify restoration:**
```bash
cargo check
cargo test
```

4. **Report to Queen Coordinator** with details of what failed

## Coordination

### Before Starting Removal
```bash
npx claude-flow@alpha hooks pre-task --description "Legacy code removal - final cleanup"
```

### After Each File Removed
```bash
npx claude-flow@alpha hooks notify --message "Removed graph_actor.rs (4,566 lines)"
npx claude-flow@alpha hooks post-edit --file "src/actors/graph_actor.rs"
```

### After All Removals Complete
```bash
npx claude-flow@alpha hooks post-task --task-id "legacy-removal-complete"
npx claude-flow@alpha hooks notify --message "All legacy code removed. Hexagonal architecture complete!"
```

## Success Criteria

✅ All deprecated files removed
✅ System compiles with zero errors
✅ All tests pass (100% success rate)
✅ GitHub sync works (316 nodes)
✅ WebSocket updates functional
✅ Physics simulation operational
✅ 8,500+ lines of legacy code removed
✅ Backup created for rollback
✅ Removal log documented

## Report to Queen

Upon completion, notify Queen Coordinator:
- Files removed (count and line total)
- Final validation status
- Codebase size reduction
- Migration completion confirmation

**Expected Duration:** 30-45 minutes
**Blocker Escalation:** Immediate if compilation fails

---
*Assigned by Queen Coordinator - Final Cleanup*
