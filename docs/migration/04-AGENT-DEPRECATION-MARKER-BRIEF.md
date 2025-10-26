# ⚠️ Agent 4: Deprecation Marker - Mission Brief

**Agent ID:** deprecation-marker
**Type:** Code Maintenance
**Priority:** Medium
**Compute Units:** 5
**Memory Quota:** 128 MB

## Mission Statement

Mark ALL legacy code with `#[deprecated]` attributes. Add deprecation warnings to GraphServiceActor, GPU actors, and physics orchestrator. Update documentation showing hexagonal alternatives. Enable compiler warnings to guide migration.

## Deprecation Targets

### 1. Primary Target: GraphServiceActor
**File:** `src/actors/graph_actor.rs` (4,566 lines)

**Actions:**
```rust
#[deprecated(
    since = "0.3.0",
    note = "Use CQRS command/query handlers in src/application/knowledge_graph/ instead. \
            See docs/migration/hexagonal-migration-plan.md for migration guide."
)]
pub struct GraphServiceActor {
    // ... existing code
}

#[deprecated(
    since = "0.3.0",
    note = "Use AddNodeCommand in src/application/knowledge_graph/directives.rs"
)]
impl Handler<AddNode> for GraphServiceActor {
    // ... existing code
}
```

### 2. GPU Supervisor Actors (DEPRECATE)
**Files to Mark:**
- `src/actors/gpu/gpu_manager_actor.rs`
- `src/actors/gpu/gpu_resource_actor.rs`
- `src/actors/gpu/force_compute_actor.rs`
- `src/actors/gpu/stress_majorization_actor.rs`
- `src/actors/gpu/clustering_actor.rs`
- `src/actors/gpu/anomaly_detection_actor.rs`
- `src/actors/gpu/constraint_actor.rs`
- `src/actors/gpu/ontology_constraint_actor.rs`

**Preserve:**
- `src/actors/gpu/shared.rs` (utilities)
- `src/actors/gpu/cuda_stream_wrapper.rs` (low-level CUDA)

**Deprecation Message:**
```rust
#[deprecated(
    since = "0.3.0",
    note = "Use GPUPhysicsAdapter port (src/ports/gpu_physics_adapter.rs) instead. \
            GPU functionality moved to adapter layer for better separation of concerns."
)]
```

### 3. Physics Orchestrator
**File:** `src/actors/physics_orchestrator_actor.rs`

**Deprecation:**
```rust
#[deprecated(
    since = "0.3.0",
    note = "Use PhysicsSimulator port (src/ports/physics_simulator.rs) with domain service pattern. \
            Physics is now a pluggable service in hexagonal architecture."
)]
pub struct PhysicsOrchestratorActor {
    // ... existing code
}
```

### 4. Graph Service Supervisor
**File:** `src/actors/graph_service_supervisor.rs`

**Deprecation:**
```rust
#[deprecated(
    since = "0.3.0",
    note = "Supervisor pattern replaced by application layer service initialization. \
            See src/application/knowledge_graph/mod.rs for new initialization pattern."
)]
```

### 5. Legacy Adapters
**File:** `src/adapters/actor_graph_repository.rs`

**Actions:**
- Mark as deprecated if it wraps GraphServiceActor
- Update to use hexagonal repository directly
- Document migration path

## Documentation Updates

### 1. Create Migration Guide
**File:** `docs/migration/DEPRECATION_GUIDE.md`

**Content:**
```markdown
# Deprecation Guide - Hexagonal Migration

## Deprecated Code

All monolithic actor patterns are deprecated as of v0.3.0.

### GraphServiceActor → CQRS Handlers

**Before:**
```rust
let result = graph_actor.send(AddNode { ... }).await?;
```

**After:**
```rust
use crate::application::knowledge_graph::directives::AddNodeCommand;
let result = add_node_handler.handle(AddNodeCommand { ... }).await?;
```

### GPU Actors → Adapter Pattern

**Before:**
```rust
let gpu_actor = GPUManagerActor::new(...);
```

**After:**
```rust
use crate::ports::gpu_physics_adapter::GPUPhysicsAdapter;
let adapter = GPUPhysicsAdapterImpl::new(...);
```

[... more examples ...]
```

### 2. Update Main Documentation
**Files to Update:**
- `README.md` - Add deprecation notice
- `ARCHITECTURE.md` - Mark old patterns as legacy
- `CONTRIBUTING.md` - Guide new contributors to hexagonal layer

### 3. Add Compiler Warnings

Create: `src/actors/deprecated_notice.rs`

```rust
//! # DEPRECATION NOTICE
//!
//! This entire actors/ directory is being phased out in favor of hexagonal architecture.
//!
//! **New Code Guidelines:**
//! - Commands/Mutations: Use `src/application/*/directives.rs`
//! - Queries/Reads: Use `src/application/*/queries.rs`
//! - Ports: Define interfaces in `src/ports/`
//! - Adapters: Implement ports in `src/adapters/`
//!
//! **Migration Timeline:**
//! - v0.3.0: Deprecation warnings added
//! - v0.4.0: Dual support (old + new patterns)
//! - v0.5.0: Legacy code removed
//!
//! See: docs/migration/hexagonal-migration-plan.md

#![warn(deprecated)]
```

## Deliverables

### Primary Deliverable
Create: `/home/devuser/workspace/project/docs/migration/deprecated-files-catalog.md`

**Required Sections:**
1. **Deprecated Files List**
   - File path
   - Deprecation reason
   - Alternative to use
   - Removal timeline
2. **Migration Examples**
   - Before/after code samples
   - Common patterns
   - Gotchas and tips
3. **Compiler Warning Summary**
   - Expected warnings after marking
   - How to fix each warning
   - Suppression guidance (temporary)

### Code Changes
Mark these files with `#[deprecated]`:
1. `src/actors/graph_actor.rs`
2. `src/actors/graph_service_supervisor.rs`
3. `src/actors/physics_orchestrator_actor.rs`
4. `src/actors/gpu/gpu_manager_actor.rs`
5. `src/actors/gpu/gpu_resource_actor.rs`
6. `src/actors/gpu/force_compute_actor.rs`
7. `src/actors/gpu/*_actor.rs` (all GPU actors except shared utilities)

## Memory Storage

Store catalog under: `hive-coordination/deprecated/marked_files`

**JSON Structure:**
```json
{
  "deprecated_files": [
    {
      "path": "src/actors/graph_actor.rs",
      "since": "0.3.0",
      "alternative": "src/application/knowledge_graph/",
      "removal_version": "0.5.0",
      "marked": true
    },
    ...
  ],
  "total_files_marked": 12,
  "total_lines_deprecated": 8500,
  "migration_examples_created": 15,
  "documentation_updated": true
}
```

## Coordination

### Before Starting
```bash
npx claude-flow@alpha hooks pre-task --description "Mark legacy code as deprecated"
```

### During Work
```bash
# After marking each file
npx claude-flow@alpha hooks post-edit --file "src/actors/graph_actor.rs"
npx claude-flow@alpha hooks notify --message "Marked GraphServiceActor as deprecated"
```

### After Completion
```bash
npx claude-flow@alpha hooks post-task --task-id "deprecation-marker-complete"
```

## Success Criteria

✅ All legacy actors marked with #[deprecated]
✅ Deprecation messages include migration guidance
✅ Documentation updated with alternatives
✅ DEPRECATION_GUIDE.md created
✅ Compiler warnings enabled
✅ Migration examples provided
✅ Findings stored in memory

## Validation

After marking deprecated:
```bash
# Verify warnings appear
cargo check 2>&1 | grep "deprecated"

# Should see warnings for each marked item
```

## Report to Queen

Upon completion, notify Queen Coordinator:
- Number of files marked
- Total lines deprecated
- Documentation updates completed
- Compiler warnings confirmed

**Expected Duration:** 20-30 minutes
**Blocker Escalation:** None expected (mechanical task)

---
*Assigned by Queen Coordinator*
