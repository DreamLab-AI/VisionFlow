# 👑 QUEEN COORDINATOR - Hexagonal Migration Brief

**Mission Status:** CRITICAL - Immediate Action Required
**Date:** 2025-10-26
**Session ID:** hive-hexagonal-migration
**Sovereign:** Queen Coordinator

## Executive Summary

The monolithic GraphServiceActor pattern was supposed to be removed but still dominates the codebase:

- **Monolith Size:** 4,566 lines (82% larger than expected)
- **Dependencies:** 25 files directly depend on GraphServiceActor
- **Hexagonal Layer:** Partially implemented (20 files exist)
- **Status:** Incomplete migration, dual architecture causing complexity

## Three Competing Architectures

### 1. Monolithic GraphServiceActor (REMOVE)
- **File:** `src/actors/graph_actor.rs` (4,566 lines)
- **Pattern:** Single actor handling all concerns
- **Problems:** Tight coupling, hard to test, blocks evolution

### 2. GPU Supervisor Model (DEPRECATE)
- **Files:** `src/actors/gpu/*` (7 actors)
- **Pattern:** Actor-based GPU coordination
- **Status:** Redundant with hexagonal ports

### 3. CQRS/Hexagonal (KEEP & COMPLETE)
- **Layers:** Application, Ports, Adapters
- **Pattern:** Command/Query separation, dependency inversion
- **Status:** 20+ files exist but incomplete

## Royal Directives - 7 Worker Agents

### Agent 1: Code Auditor
**Mission:** Audit ALL dependencies on GraphServiceActor
**Deliverables:**
- Complete dependency map (25+ files)
- API route analysis
- Handler coupling analysis
- WebSocket dependency tree
- Storage: `audit/graph_actor_dependencies`

### Agent 2: Architecture Planner
**Mission:** Design complete hexagonal replacement
**Deliverables:**
- Command handlers for write operations
- Query handlers for read operations
- Event sourcing design for real-time updates
- Port definitions for physics/GPU/semantic
- Storage: `planning/hexagonal_migration_plan`

### Agent 3: Test Coverage Analyzer
**Mission:** Ensure safe deletion via comprehensive tests
**Deliverables:**
- Test coverage report for graph_actor.rs
- Missing test identification
- Critical path testing strategy
- Pre-migration test suite
- Storage: `testing/pre_migration_coverage`

### Agent 4: Deprecation Marker
**Mission:** Mark ALL legacy code as deprecated
**Deliverables:**
- #[deprecated] attributes on all legacy actors
- Documentation updates showing hexagonal alternatives
- Compiler warnings enabled
- Migration guide for dependents
- Storage: `deprecated/marked_files`

### Agent 5: Migration Executor
**Mission:** Execute migration in 4 phases
**Phases:**
1. Extract read operations → query handlers
2. Extract write operations → command handlers
3. Replace WebSocket → event sourcing
4. Migrate physics → domain services
**Storage:** `migration/phase_status`

### Agent 6: Integration Validator
**Mission:** Validate each migration phase
**Tests:**
- API endpoints return correct data
- WebSocket updates propagate
- Physics simulation runs correctly
- GitHub sync works (316 nodes)
**Storage:** `validation/integration_tests`

### Agent 7: Legacy Code Remover
**Mission:** Complete removal of deprecated code
**Removals:**
- src/actors/graph_actor.rs
- src/actors/gpu/* (except compute utilities)
- src/actors/physics_orchestrator_actor.rs
- All deprecated adapters
**Storage:** `removal/deleted_files`

## Resource Allocation

| Agent | Compute Units | Memory (MB) | Priority |
|-------|---------------|-------------|----------|
| Code Auditor | 15 | 256 | High |
| Architecture Planner | 20 | 512 | Critical |
| Test Analyzer | 10 | 256 | High |
| Deprecation Marker | 5 | 128 | Medium |
| Migration Executor | 30 | 1024 | Critical |
| Integration Validator | 15 | 512 | High |
| Legacy Remover | 5 | 256 | Medium |

## Success Criteria

✅ Zero dependencies on GraphServiceActor
✅ All functionality in hexagonal layer
✅ Real-time via event sourcing
✅ 100% test coverage maintained
✅ GitHub sync: 316 nodes, 100% public metadata
✅ Monolithic code deleted
✅ Clean `cargo build` with no warnings

## Coordination Protocol

All agents MUST:
1. Use hooks for session coordination
2. Store findings in memory under assigned keys
3. Report progress to Queen Coordinator
4. Validate work before marking complete
5. Enable cross-agent knowledge sharing

## Next Actions

Queen will spawn 7 workers concurrently. Each worker receives:
- Specific mission brief
- Memory namespace for storage
- Success criteria
- Coordination hooks

**The monolith ends NOW.**

---
*Issued by Queen Coordinator, Session: hive-hexagonal-migration*
