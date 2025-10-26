# CQRS Migration Documentation

## Overview

This directory contains comprehensive documentation for migrating the GraphServiceActor (2,501 lines) from an actor-based monolith to a CQRS/Hexagonal architecture. The migration is split into phases, starting with the safest operations first.

## Documentation Files

### 1. [cqrs-phase1-read-operations.md](./cqrs-phase1-read-operations.md) (20KB)
**Primary Blueprint for Phase 1 Implementation**

This is the main migration guide containing:
- ✅ Identified read operations (7 query handlers)
- ✅ Step-by-step implementation plan
- ✅ Rollback strategies at each stage
- ✅ Success criteria and metrics
- ✅ Risk assessment and mitigation
- ✅ Error code definitions

**Start Here** if you're ready to implement Phase 1.

### 2. [cqrs-research-findings.md](./cqrs-research-findings.md) (17KB)
**Detailed Research Analysis**

Complete analysis of the codebase including:
- ✅ Existing CQRS patterns (Settings, Ontology domains)
- ✅ GraphServiceActor handler analysis (all 40+ handlers)
- ✅ Repository port patterns and best practices
- ✅ Performance analysis and benchmarking strategy
- ✅ Testing strategy and examples
- ✅ Timeline estimates (5 days for Phase 1)

**Read This** to understand the rationale behind design decisions.

### 3. [cqrs-code-templates.md](./cqrs-code-templates.md) (30KB)
**Copy-Paste Ready Code Templates**

Complete code templates for immediate use:
- ✅ Query handler templates (3 variations)
- ✅ Repository port extensions
- ✅ Actor-based adapter (complete implementation)
- ✅ AppState updates
- ✅ API route migration patterns
- ✅ Feature flag implementation
- ✅ Unit and integration test templates
- ✅ Complete queries.rs file (all 7 handlers)

**Use This** for actual code implementation.

## Quick Start Guide

### For Implementers

1. **Read the blueprint**: [cqrs-phase1-read-operations.md](./cqrs-phase1-read-operations.md)
2. **Review existing patterns**: Study `src/application/settings/queries.rs`
3. **Copy templates**: Use [cqrs-code-templates.md](./cqrs-code-templates.md)
4. **Follow the implementation plan** in the blueprint

### For Reviewers

1. **Understand the research**: [cqrs-research-findings.md](./cqrs-research-findings.md)
2. **Review risk assessment**: See "Risk Assessment" in the blueprint
3. **Check success criteria**: See "Success Criteria" in the blueprint
4. **Verify testing strategy**: See "Testing Strategy" in research findings

## Migration Phases

### Phase 1: Read Operations (Current)
**Scope**: Migrate 7 read-only query operations
**Timeline**: ~5 days
**Risk**: Low (no side effects)

**Operations**:
1. `GetGraphData` → Returns `Arc<GraphData>`
2. `GetNodeMap` → Returns `Arc<HashMap<u32, Node>>`
3. `GetPhysicsState` → Returns `PhysicsState`
4. `GetNodePositions` → Returns `Vec<(u32, Vec3)>`
5. `GetBotsGraphData` → Returns `Arc<GraphData>`
6. `GetConstraints` → Returns `ConstraintSet`
7. `GetAutoBalanceNotifications` → Returns `Vec<AutoBalanceNotification>`

**Files to Create**:
- `src/application/graph/mod.rs`
- `src/application/graph/queries.rs`
- `src/adapters/actor_graph_repository.rs`
- `tests/integration/graph_queries_test.rs`

**Files to Modify**:
- `src/ports/graph_repository.rs` (extend interface)
- `src/app_state.rs` (add query handlers)
- `src/handlers/api_handler/graph/mod.rs` (update routes)
- `src/main.rs` (initialize handlers)

### Phase 2: Write Operations (Future)
**Scope**: Migrate write operations (directives)
**Timeline**: TBD
**Risk**: Medium (has side effects)

Operations: TBD after Phase 1 success

### Phase 3: Direct Repository (Future)
**Scope**: Replace actor with direct database access
**Timeline**: TBD
**Risk**: High (architectural change)

### Phase 4: Event Sourcing (Future)
**Scope**: Add event sourcing for audit trail
**Timeline**: TBD
**Risk**: Medium (new capability)

## Implementation Checklist

### Pre-Implementation
- [ ] Read all documentation files
- [ ] Study existing CQRS implementations:
  - `src/application/settings/queries.rs`
  - `src/application/ontology/queries.rs`
- [ ] Review GraphServiceActor handlers:
  - `src/actors/graph_actor.rs` (lines 3144-4343)
- [ ] Set up feature branch: `git checkout -b feature/cqrs-phase1-queries`

### Phase 1A: Repository Port
- [ ] Extend `src/ports/graph_repository.rs`
- [ ] Add 7 new read method signatures
- [ ] Define return types and error cases
- [ ] Add documentation to each method

### Phase 1B: Query Handlers
- [ ] Create `src/application/graph/mod.rs`
- [ ] Create `src/application/graph/queries.rs`
- [ ] Implement all 7 query handlers
- [ ] Add unit tests for each handler

### Phase 1C: Actor Adapter
- [ ] Create `src/adapters/actor_graph_repository.rs`
- [ ] Implement all 7 repository methods
- [ ] Verify Arc semantics (no cloning)
- [ ] Add error mapping tests

### Phase 1D: AppState Integration
- [ ] Create `GraphQueryHandlers` struct in `src/app_state.rs`
- [ ] Initialize handlers in `src/main.rs`
- [ ] Add repository to AppState
- [ ] Verify dependency injection

### Phase 1E: API Routes
- [ ] Update `src/handlers/api_handler/graph/mod.rs`
- [ ] Migrate `get_graph_data` route
- [ ] Add feature flag support
- [ ] Test both CQRS and legacy paths

### Phase 1F: Testing
- [ ] Create `tests/integration/graph_queries_test.rs`
- [ ] Add unit tests for all handlers
- [ ] Add integration tests for API routes
- [ ] Performance benchmarks

### Phase 1G: Deployment
- [ ] Code review
- [ ] Merge to main
- [ ] Deploy with feature flag OFF
- [ ] Enable feature flag gradually
- [ ] Monitor metrics

## Success Metrics

### Functional
- ✅ All 7 read operations migrated
- ✅ API routes use query handlers
- ✅ No behavioral changes for clients
- ✅ All existing tests pass

### Performance
- ✅ Response times within 5% of baseline
- ✅ No memory leaks detected
- ✅ Actor mailbox not congested

### Code Quality
- ✅ Follows established CQRS patterns
- ✅ 100% test coverage for handlers
- ✅ Comprehensive documentation
- ✅ Clean error handling

## Rollback Plan

### Development Rollback
```bash
git checkout main
git branch -D feature/cqrs-phase1-queries
# Start over with lessons learned
```

### Production Rollback
```bash
# Option 1: Disable feature flag
export USE_CQRS_QUERIES=false

# Option 2: Revert deployment
git revert <commit-hash>
git push origin main
```

### Emergency Rollback
See "Rollback Strategy" section in [cqrs-phase1-read-operations.md](./cqrs-phase1-read-operations.md).

## Key Design Decisions

### 1. Actor Adapter vs Direct Repository
**Decision**: Use actor adapter for Phase 1
**Rationale**:
- Minimizes risk
- Allows gradual migration
- Maintains existing actor behavior
- Easy rollback

### 2. Feature Flag Strategy
**Decision**: Environment variable `USE_CQRS_QUERIES`
**Rationale**:
- No code changes for rollback
- Can enable per-environment
- Gradual rollout capability

### 3. Error Code Scheme
**Decision**: `E_GRAPH_001` through `E_GRAPH_007`
**Rationale**:
- Follows hexser conventions
- Matches ontology pattern
- Easy debugging

### 4. No Caching in Phase 1
**Decision**: Defer caching to Phase 2
**Rationale**:
- Keep Phase 1 simple
- Actor already has in-memory state
- Caching more valuable with direct DB access

## Reference Implementations

### Existing CQRS Domains
1. **Settings Domain** (`src/application/settings/`)
   - Simple queries with caching
   - Batch operations
   - Profile management

2. **Ontology Domain** (`src/application/ontology/`)
   - Complex queries
   - Arc-based returns
   - Validation queries

### Study These Files
```
src/application/settings/queries.rs       # Basic query patterns
src/application/settings/directives.rs    # Directive patterns (Phase 2)
src/application/ontology/queries.rs       # Complex queries
src/ports/settings_repository.rs          # Port interface design
src/adapters/sqlite_settings_repository.rs # Adapter with caching
```

## Common Patterns

### Query Handler Pattern
```rust
// 1. Define query struct
#[derive(Debug, Clone)]
pub struct GetGraphData;

// 2. Create handler with repository
pub struct GetGraphDataHandler {
    repository: Arc<dyn GraphRepository>,
}

// 3. Implement QueryHandler trait
impl QueryHandler<GetGraphData, Arc<GraphData>> for GetGraphDataHandler {
    fn handle(&self, _query: GetGraphData) -> HexResult<Arc<GraphData>> {
        // Execute async repository call
        tokio::runtime::Handle::current().block_on(async move {
            self.repository.get_graph().await
                .map_err(|e| Hexserror::port("E_GRAPH_001", &format!("{}", e)))
        })
    }
}
```

### Repository Port Pattern
```rust
#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn get_graph(&self) -> Result<Arc<GraphData>>;
}
```

### Actor Adapter Pattern
```rust
#[async_trait]
impl GraphRepository for ActorGraphRepository {
    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        self.actor_addr.send(GetGraphData).await??
    }
}
```

## Troubleshooting

### Issue: Handler not found
**Solution**: Check `src/application/graph/mod.rs` exports

### Issue: Runtime error in block_on
**Solution**: Ensure tokio runtime is initialized

### Issue: Type mismatch errors
**Solution**: Verify repository method signatures match port interface

### Issue: Performance degradation
**Solution**:
1. Check Arc usage (should be Arc::clone, not data clone)
2. Profile with benchmarks
3. Review actor mailbox metrics

## Next Steps

After completing this research and documentation:

1. **Implementation**: Follow the blueprint in [cqrs-phase1-read-operations.md](./cqrs-phase1-read-operations.md)
2. **Testing**: Use templates from [cqrs-code-templates.md](./cqrs-code-templates.md)
3. **Review**: Reference [cqrs-research-findings.md](./cqrs-research-findings.md) for design decisions
4. **Deploy**: Use feature flag for gradual rollout

## Questions?

Refer to:
- **Implementation questions**: See [cqrs-code-templates.md](./cqrs-code-templates.md)
- **Design questions**: See [cqrs-research-findings.md](./cqrs-research-findings.md)
- **Process questions**: See [cqrs-phase1-read-operations.md](./cqrs-phase1-read-operations.md)

## Summary

This CQRS migration is designed to be:
- ✅ **Safe**: Read-only operations first
- ✅ **Gradual**: Actor adapter allows incremental migration
- ✅ **Reversible**: Multiple rollback options
- ✅ **Testable**: Comprehensive test templates
- ✅ **Well-documented**: 67KB of documentation

**Estimated Timeline**: 5 days for Phase 1
**Risk Level**: Low
**Success Probability**: High (based on existing patterns)

---

**Documentation Created**: 2025-10-26
**Last Updated**: 2025-10-26
**Status**: Ready for Implementation
