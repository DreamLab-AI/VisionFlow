# Legacy Code Removal Timeline

## Executive Summary

**Total Legacy Code**: 12,588 lines across 10 actors
**Total References**: 502 references across codebase
**Estimated Timeline**: 7-8 weeks (304 hours)
**Critical Path**: GraphServiceActor → PhysicsOrchestrator → GPU Actors

## Phase 0: Preparation (Week 1 - 40 hours)

### Week 1: Assessment & Foundation
**Goal**: Mark all legacy code for deprecation and establish baseline

#### Tasks:
- [x] Complete legacy code inventory
- [ ] Add `#[deprecated]` attributes to all legacy actors
- [ ] Create migration branch: `feature/hexagonal-migration`
- [ ] Establish cargo check baseline (current: 112 dead code warnings)
- [ ] Document current API contracts
- [ ] Create integration test suite for critical paths
- [ ] Set up feature flags for parallel migration

#### Deliverables:
- `docs/legacy-code-inventory.json` ✅
- `docs/legacy-removal-timeline.md` ✅
- `scripts/verify-no-legacy.sh` (pending)
- `docs/cargo-check-baseline.txt` (pending)
- Integration test suite covering 80%+ of legacy actor functionality

#### Success Criteria:
- ✅ All legacy code identified and documented
- [ ] Cargo check passes (0 errors, acceptable warnings documented)
- [ ] All tests pass
- [ ] Deprecation warnings visible to developers

---

## Phase 1: GraphServiceActor Migration (Weeks 2-4 - 120 hours)

### Week 2: Repository & Domain Layer (40 hours)

**Goal**: Complete hexagonal foundation for graph operations

#### Tasks:
1. **Graph Domain Models** (8 hours)
   - Create `src/domain/graph/mod.rs`
   - Define `GraphNode`, `GraphEdge`, `GraphData` as domain entities
   - Create domain events: `NodeAdded`, `EdgeAdded`, `NodePositionUpdated`

2. **Graph Repository** (12 hours)
   - Implement `src/domain/graph/repository.rs` trait
   - Extend `SqliteKnowledgeGraphRepository` for graph operations
   - Add batch operations support
   - Implement transaction support

3. **Graph Commands** (12 hours)
   - Create `src/application/graph/commands.rs`
   - Implement command handlers:
     - `AddNodeCommand`
     - `RemoveNodeCommand`
     - `AddEdgeCommand`
     - `UpdateNodePositionCommand`
     - `BatchUpdateCommand`
   - Add command validation

4. **Graph Queries** (8 hours)
   - Create `src/application/graph/queries.rs`
   - Implement query handlers:
     - `GetGraphDataQuery`
     - `GetNodeQuery`
     - `GetNeighborsQuery`
     - `SearchNodesQuery`

#### Deliverables:
- Complete CQRS layer for graph operations
- Repository tests with 90%+ coverage
- Command/Query handler tests
- Documentation for new architecture

### Week 3: Service Layer & Handler Migration (40 hours)

**Goal**: Replace actor calls with service calls in handlers

#### Tasks:
1. **Graph Service** (12 hours)
   - Create `src/services/graph_service.rs`
   - Coordinate commands, queries, and events
   - Handle WebSocket broadcasting (without actors)
   - Implement caching layer

2. **Handler Migration** (20 hours)
   - Migrate `src/handlers/api_handler/graph/mod.rs` (HIGH PRIORITY)
     - Replace `state.graph_service_addr.send()` calls
     - Use `state.graph_service.execute_command()` instead
   - Migrate graph-related endpoints in:
     - `src/handlers/settings_handler.rs`
     - `src/handlers/api_handler/analytics/mod.rs`
   - Update AppState with new services

3. **Integration Testing** (8 hours)
   - Create integration tests for migrated handlers
   - Test concurrent operations
   - Verify WebSocket broadcasting works
   - Performance benchmarks vs legacy

#### Deliverables:
- Handlers using services instead of actors
- Passing integration tests
- Performance parity with legacy system
- Migration guide documentation

### Week 4: GraphServiceActor Removal (40 hours)

**Goal**: Delete GraphServiceActor and verify system stability

#### Tasks:
1. **Final Migration** (16 hours)
   - Migrate remaining GraphServiceActor calls
   - Remove `graph_service_addr` from AppState
   - Update `src/main.rs` initialization
   - Remove TransitionalGraphSupervisor

2. **Cleanup** (12 hours)
   - Delete `src/actors/graph_actor.rs` (4566 lines)
   - Delete `src/actors/graph_messages.rs`
   - Delete `src/actors/graph_service_supervisor.rs`
   - Remove GraphServiceActor exports from `src/actors/mod.rs`

3. **Verification** (12 hours)
   - Run `scripts/verify-no-legacy.sh`
   - Cargo check (should pass with 0 errors)
   - Full test suite (should pass 100%)
   - Load testing
   - Manual QA

#### Success Criteria:
- ✅ GraphServiceActor completely removed
- ✅ Zero references to GraphServiceActor in codebase
- ✅ All tests pass
- ✅ Cargo check passes
- ✅ Performance equal or better than legacy

---

## Phase 2: PhysicsOrchestratorActor Migration (Week 5 - 32 hours)

### Week 5: Physics Service (32 hours)

**Goal**: Consolidate physics coordination into hexagonal service

#### Tasks:
1. **Physics Service** (16 hours)
   - Create `src/services/physics_service.rs`
   - Consolidate logic from:
     - PhysicsOrchestratorActor (1105 lines)
     - GraphServiceActor physics methods
     - ForceComputeActor coordination
   - Implement physics state management
   - Add simulation parameter handling

2. **Handler Migration** (8 hours)
   - Update physics-related handlers
   - Replace actor sends with service calls
   - Test physics simulation

3. **Cleanup** (8 hours)
   - Delete `src/actors/physics_orchestrator_actor.rs`
   - Remove references
   - Verify cargo check

#### Deliverables:
- Unified physics service
- No PhysicsOrchestratorActor references
- Passing tests

---

## Phase 3: GPU Actors Migration (Weeks 6-7 - 112 hours)

### Week 6: GPU Worker Actors (56 hours)

**Goal**: Migrate GPU computation actors to services

#### Tasks:

1. **ForceComputeActor → PhysicsService** (24 hours)
   - Integrate force computation into PhysicsService
   - Keep GPU kernels in `src/utils/unified_gpu_compute.rs`
   - Refactor for direct GPU calls (no actors)
   - Delete `src/actors/gpu/force_compute_actor.rs`

2. **ClusteringActor → ClusteringService** (18 hours)
   - Create `src/services/clustering_service.rs`
   - Implement K-means, community detection
   - Keep GPU implementations in utils
   - Migrate analytics handlers
   - Delete `src/actors/gpu/clustering_actor.rs`

3. **AnomalyDetectionActor → AnomalyService** (14 hours)
   - Create `src/services/anomaly_detection_service.rs`
   - Keep LOF, Z-score, Isolation Forest in utils
   - Migrate analytics handlers
   - Delete `src/actors/gpu/anomaly_detection_actor.rs`

### Week 7: GPU Resource Management (56 hours)

**Goal**: Migrate remaining GPU actors

#### Tasks:

1. **GPUResourceActor → GPUResourceManager** (16 hours)
   - Create `src/services/gpu_resource_manager.rs`
   - Manage GPU context, memory, devices
   - No actor overhead
   - Delete `src/actors/gpu/gpu_resource_actor.rs`

2. **StressMajorizationActor → LayoutService** (12 hours)
   - Create `src/services/layout_service.rs`
   - Keep GPU stress majorization in utils
   - Delete `src/actors/gpu/stress_majorization_actor.rs`

3. **OntologyConstraintActor + ConstraintActor → ConstraintService** (14 hours)
   - Create `src/application/ontology/constraint_service.rs`
   - Consolidate both actors
   - Keep GPU constraint solving in utils
   - Delete both actors

4. **GPUManagerActor Removal** (14 hours)
   - Remove coordinator (workers now services)
   - Delete `src/actors/gpu/gpu_manager_actor.rs`
   - Clean up `src/actors/gpu/mod.rs`
   - Remove `gpu_manager_addr` from AppState

---

## Phase 4: Final Cleanup (Week 8 - 40 hours)

### Week 8: Complete Legacy Removal

**Goal**: Remove ALL actor infrastructure

#### Tasks:

1. **Actor Directory Cleanup** (16 hours)
   - Delete remaining actor files:
     - `src/actors/gpu/` (entire directory)
     - `src/actors/supervisor.rs` (if unused)
     - Unused message types in `src/actors/messages.rs`
   - Keep only:
     - Non-legacy actors (if any remain)
     - Essential coordination code

2. **AppState Refactoring** (12 hours)
   - Remove all actor addresses:
     - `graph_service_addr`
     - `gpu_manager_addr`
     - `gpu_compute_addr`
   - Add service fields:
     - `graph_service: Arc<GraphService>`
     - `physics_service: Arc<PhysicsService>`
     - `gpu_compute_service: Arc<GPUComputeService>`
     - etc.

3. **Main.rs Cleanup** (4 hours)
   - Remove actor initialization
   - Initialize services instead
   - Simplify startup sequence

4. **Final Verification** (8 hours)
   - Run `scripts/verify-no-legacy.sh` (should return 0)
   - Cargo check (0 errors, minimal warnings)
   - Full test suite (100% pass)
   - Performance benchmarks
   - Load testing
   - Manual QA

#### Success Criteria:
- ✅ Zero actor references (except non-legacy if any)
- ✅ Cargo check passes
- ✅ All tests pass
- ✅ Performance improved (no actor overhead)
- ✅ Code quality improved (clean architecture)

---

## Verification Gates

Each phase must pass these gates before proceeding:

### Gate 1: After Phase 1 (GraphServiceActor)
```bash
# Should return 0
grep -rn "GraphServiceActor" src/ --include="*.rs" | wc -l

# Should pass
cargo check
cargo test

# Should show performance improvement
cargo bench --bench graph_operations
```

### Gate 2: After Phase 2 (PhysicsOrchestratorActor)
```bash
# Should return 0
grep -rn "PhysicsOrchestratorActor" src/ --include="*.rs" | wc -l

# Should pass
cargo check
cargo test --test physics_integration
```

### Gate 3: After Phase 3 (GPU Actors)
```bash
# Should return 0
grep -rn "GPUManagerActor|ForceComputeActor|ClusteringActor" src/ --include="*.rs" | wc -l

# Should pass
cargo check
cargo test --test gpu_integration
```

### Gate 4: Final (Complete Migration)
```bash
# Run comprehensive verification
./scripts/verify-no-legacy.sh

# Should show zero legacy references
# Should pass all checks
```

---

## Risk Mitigation

### High-Risk Items:
1. **GraphServiceActor (4566 lines)**: Break into smaller PRs, use feature flags
2. **WebSocket Broadcasting**: Test extensively, may need event bus
3. **GPU Coordination**: Complex state management, thorough testing required
4. **Physics Simulation**: Performance-critical, benchmark continuously

### Mitigation Strategies:
- **Feature Flags**: `legacy_actors` flag to toggle between old/new
- **Parallel Development**: Keep legacy code working while building new
- **Incremental Rollout**: Migrate one handler at a time
- **Automated Testing**: CI/CD catches regressions immediately
- **Performance Monitoring**: Benchmark after each phase
- **Rollback Plan**: Git branches allow quick revert if needed

---

## Success Metrics

### Code Quality:
- [ ] 12,588 lines removed
- [ ] Dead code warnings reduced from 112 to <20
- [ ] Cyclomatic complexity reduced
- [ ] Test coverage increased to >85%

### Architecture:
- [ ] Pure hexagonal architecture (no actors)
- [ ] Clear separation of concerns
- [ ] CQRS implemented for graph operations
- [ ] Repository pattern for all data access

### Performance:
- [ ] Response times equal or better
- [ ] Memory usage reduced (no actor overhead)
- [ ] Concurrent request handling improved
- [ ] GPU utilization maintained or improved

### Maintainability:
- [ ] Code is easier to understand
- [ ] Easier to test (no actor mocking)
- [ ] Clearer dependencies
- [ ] Better documentation

---

## Post-Migration

### Week 9+: Optimization
- Optimize service performance
- Add more comprehensive caching
- Improve GPU resource utilization
- Refactor utilities further
- Add more integration tests

### Continuous Improvement:
- Monitor production metrics
- Gather team feedback
- Iterate on architecture
- Document lessons learned
- Share knowledge with team

---

## Emergency Contacts & Resources

- **Architecture Lead**: [Review hexagonal architecture docs]
- **Migration Docs**: `docs/hexagonal-architecture.md`
- **Verification Script**: `scripts/verify-no-legacy.sh`
- **Test Coverage**: `cargo tarpaulin --out Html`
- **Benchmarks**: `cargo bench`

---

## Appendix A: File Deletion Checklist

### Phase 1 Deletions:
- [ ] `src/actors/graph_actor.rs` (4566 lines)
- [ ] `src/actors/graph_messages.rs`
- [ ] `src/actors/graph_service_supervisor.rs`

### Phase 2 Deletions:
- [ ] `src/actors/physics_orchestrator_actor.rs` (1105 lines)

### Phase 3 Deletions:
- [ ] `src/actors/gpu/gpu_manager_actor.rs` (657 lines)
- [ ] `src/actors/gpu/force_compute_actor.rs` (1047 lines)
- [ ] `src/actors/gpu/clustering_actor.rs` (715 lines)
- [ ] `src/actors/gpu/anomaly_detection_actor.rs` (918 lines)
- [ ] `src/actors/gpu/gpu_resource_actor.rs` (606 lines)
- [ ] `src/actors/gpu/stress_majorization_actor.rs` (452 lines)
- [ ] `src/actors/gpu/ontology_constraint_actor.rs` (549 lines)
- [ ] `src/actors/gpu/constraint_actor.rs` (327 lines)
- [ ] `src/actors/gpu/cuda_stream_wrapper.rs` (66 lines)
- [ ] `src/actors/gpu/shared.rs` (536 lines)
- [ ] `src/actors/gpu/mod.rs` (44 lines)

### Phase 4 Deletions:
- [ ] `src/actors/gpu/` (entire directory if empty)
- [ ] `src/actors/supervisor.rs` (if unused)
- [ ] Legacy message types in `src/actors/messages.rs`

**Total Lines to Delete**: 12,588 lines

---

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Status**: Phase 0 - Preparation In Progress
