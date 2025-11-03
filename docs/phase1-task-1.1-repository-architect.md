# Repository Architect Agent - Task 1.1: Generic Repository Trait

## MISSION BRIEF
You are the Repository Architect, responsible for eliminating 87% code duplication across 5 repositories by creating a unified GenericRepository<T, ID> trait.

## OBJECTIVE
Create `/home/devuser/workspace/project/src/repositories/generic_repository.rs` and refactor 3 SQLite repositories to use it.

## CONTEXT FROM AUDIT
- 496 duplicate CRUD operations across 137 files
- 400+ lines of duplicated async wrapper code
- 150 lines of duplicated transaction logic
- 41 identical mutex acquisition patterns
- Current duplication: 87% across repositories

## IMPLEMENTATION STEPS

### Step 1: Create Generic Repository Module (4 hours)
Create `/home/devuser/workspace/project/src/repositories/generic_repository.rs`

Implement:
- `GenericRepository<T, ID>` trait with base CRUD operations
- `SqliteRepository<T>` base class with:
  - Generic transaction wrapper
  - Async blocking wrapper (consolidates 400 lines)
  - Generic batch insert/update
  - Error conversion utilities
  - Connection management
- Default implementations for common operations

Reference: `/home/devuser/workspace/project/docs/REPOSITORY_DUPLICATION_ANALYSIS.md` lines 400-600

### Step 2: Update Repository Trait Definitions (3 hours)
Modify `/home/devuser/workspace/project/src/ports/knowledge_graph_repository.rs`
- Add default trait implementations for health_check, transaction management
- Ensure backward compatibility

### Step 3: Refactor UnifiedGraphRepository (6 hours)
Modify `/home/devuser/workspace/project/src/repositories/unified_graph_repository.rs` (1,939 lines)
- Inherit from SqliteRepository<T>
- Remove duplicate transaction management (lines 460-587, 668-744, 816-879, 930-963)
- Remove duplicate async wrappers (lines 350-1881)
- Keep domain-specific logic only

### Step 4: Refactor UnifiedOntologyRepository (3 hours)
Modify `/home/devuser/workspace/project/src/repositories/unified_ontology_repository.rs` (841 lines)
- Use generic base for transaction management
- Remove duplicated async wrappers (lines 259-660)

## ACCEPTANCE CRITERIA
- [ ] Generic repository trait compiles without errors
- [ ] All existing repository tests pass: `cargo test --lib repositories`
- [ ] UnifiedGraphRepository uses generic base (no duplicate transaction code)
- [ ] UnifiedOntologyRepository uses generic base
- [ ] Code reduction: Minimum 540 lines eliminated
- [ ] No performance regression

## TESTING COMMANDS
```bash
cargo test --lib repositories::generic_repository
cargo test --lib repositories::unified_graph_repository
cargo test --lib repositories::unified_ontology_repository
cargo test --workspace
```

## COORDINATION PROTOCOL
BEFORE starting:
```bash
npx claude-flow@alpha hooks pre-task --description "Generic Repository Trait Implementation"
npx claude-flow@alpha hooks session-restore --session-id "swarm-phase1"
```

DURING work - publish API after Step 1:
```bash
npx claude-flow@alpha hooks post-edit --file "src/repositories/generic_repository.rs" --memory-key "hive/phase1/repository-trait-api"
```

AFTER completion:
```bash
npx claude-flow@alpha hooks post-task --task-id "task-1.1-repository"
npx claude-flow@alpha hooks notify --message "Repository trait implementation complete: 540+ lines saved"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## ROLLBACK PLAN
- Keep originals in `/home/devuser/workspace/project/archive/repositories_pre_generic/`
- Git branch: `refactor/generic-repository-pattern`

## FILES TO MODIFY
- Created: 1 (`src/repositories/generic_repository.rs`)
- Modified: 4 (unified_graph_repository.rs, unified_ontology_repository.rs, settings_repository.rs, knowledge_graph_repository.rs)

Report completion status to memory key: `hive/phase1/completion-status`
Report any file conflicts to memory key: `hive/phase1/conflicts`
