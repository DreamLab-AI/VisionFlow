# Test Coverage Analysis Report
## Hexagonal Migration - GraphServiceActor

**Analysis Date:** 2025-10-26
**Agent:** Test Coverage Analyzer
**Mission:** Ensure safe migration through comprehensive testing

---

## Executive Summary

### Current Test Infrastructure: ⚠️ MODERATE COVERAGE

- **Total Test Files:** 140+ test files identified
- **Actor Tests:** Limited direct GraphServiceActor testing
- **Critical Gap:** ❌ NO GITHUB SYNC REGRESSION TEST (316 nodes)
- **WebSocket Tests:** ✅ Basic rate limiting tests exist
- **GPU/Physics Tests:** ✅ Comprehensive GPU safety and physics tests
- **API Tests:** ✅ Extensive validation and security tests

### Risk Assessment: 🔴 HIGH RISK FOR MIGRATION

**Why?**
1. **No GitHub sync validation** - The 316 nodes (185 files + 131 linked_pages) fix is UNTESTED
2. **No event sourcing tests** - Hexagonal architecture requires event flow validation
3. **Limited integration tests** - Actor system stability tested, but not graph CRUD flow
4. **Missing repository layer tests** - New hexagonal ports/adapters need test coverage

---

## 1. Current Test Inventory

### ✅ **Strong Coverage Areas**

#### GPU Safety & Physics (15+ test files)
- `gpu_safety_validation.rs` - Comprehensive GPU bounds checking
- `gpu_stability_test.rs` - GPU initialization and fallback
- `physics_parameter_flow_test.rs` - Physics simulation tests
- `stress_majorization_integration.rs` - Layout optimization tests
- `ptx_smoke_test.rs`, `ptx_validation_comprehensive.rs` - CUDA kernel validation

#### API Security & Validation (8+ test files)
- `api_validation_tests.rs` - **1,762 lines** of input validation, XSS, SQL injection tests
- `settings_validation_tests.rs` - Settings integrity tests
- `network_resilience_tests.rs` - Connection retry, timeout, circuit breaker
- `error_handling_tests.rs` - Error propagation and recovery

#### Actor System Stability (5+ test files)
- `core_runtime_test.rs` - Actor initialization tests (GraphServiceActor on line 81-106)
- `deadlock_recovery_test.rs` - Concurrency safety
- `production_validation_suite.rs` - **1,315 lines** of comprehensive production tests

### ⚠️ **Moderate Coverage Areas**

#### WebSocket & Real-time (3 test files)
- `test_websocket_rate_limit.rs` - Rate limiting tests
- `settings_sync_test.rs` - Settings broadcast
- ❌ **MISSING:** WebSocket reconnection tests
- ❌ **MISSING:** Binary protocol edge cases

#### Settings Management (6+ test files)
- `settings_deserialization_test.rs` - YAML/JSON parsing
- `test_settings_save_minimal.rs` - Persistence
- `e2e-settings-validation.rs` - End-to-end settings flow
- ✅ Good coverage for settings system

### ❌ **Critical Gaps**

#### 1. GitHub Sync Testing - **ZERO COVERAGE**
```rust
// ❌ THIS TEST DOES NOT EXIST!
#[tokio::test]
async fn test_github_sync_shows_316_nodes_with_100_percent_public() {
    // Should verify:
    // - 185 markdown files scanned
    // - 316 total nodes (185 page + 131 linked_page)
    // - 100% nodes have public=true metadata
    // - 330 private linked_pages filtered out (646 total - 316 public)
}
```

**Impact:** 🔴 **CRITICAL**
The core bug fix (316 nodes visible in API) has NO regression test!

#### 2. Event Sourcing - **NO COVERAGE**
```rust
// ❌ MISSING: Event flow tests
#[tokio::test]
async fn test_github_sync_invalidates_cache_via_event() {
    // Should verify:
    // - GitHubSyncCompletedEvent emitted
    // - Cache invalidated
    // - Next API call returns fresh 316 nodes
}
```

#### 3. Repository Layer - **NO COVERAGE**
```rust
// ❌ MISSING: Hexagonal ports testing
#[tokio::test]
async fn test_graph_repository_saves_and_retrieves() {
    // Should test:
    // - SqliteGraphRepository CRUD operations
    // - Transaction handling
    // - Connection pooling
}
```

#### 4. Integration Tests - **WEAK COVERAGE**
- ❌ No end-to-end test for: GitHub sync → Database save → API retrieval
- ❌ No test for: WebSocket broadcast after data change
- ❌ No test for: Physics simulation → Position updates → Client sync

---

## 2. Critical Path Analysis

### Graph Data Retrieval (GET /api/graph/data)

| Layer | Test Coverage | Risk |
|-------|--------------|------|
| API Handler | ⚠️ Partial (API validation tests) | Medium |
| GraphServiceActor | ⚠️ Minimal (initialization only) | **HIGH** |
| Database Query | ❌ None | **CRITICAL** |
| Response Serialization | ✅ Good (production tests) | Low |

**Test Gap:** No integration test verifying full request → response flow

### GitHub Sync → Database Save (316 nodes fix!)

| Layer | Test Coverage | Risk |
|-------|--------------|------|
| GitHub API Scan | ❌ None | **CRITICAL** |
| Markdown Parsing | ⚠️ Partial (ontology tests) | Medium |
| Metadata Extraction | ❌ None | **CRITICAL** |
| Database Save | ❌ None | **CRITICAL** |
| Event Emission | ❌ None | **CRITICAL** |

**Test Gap:** THE ENTIRE CRITICAL PATH IS UNTESTED!

### WebSocket Connections & Broadcasts

| Layer | Test Coverage | Risk |
|-------|--------------|------|
| Connection Setup | ✅ Good (rate limit tests) | Low |
| Binary Protocol | ⚠️ Minimal | Medium |
| Broadcast Logic | ❌ None | **HIGH** |
| Reconnection | ❌ None | **HIGH** |

**Test Gap:** No test for WebSocket message flow after data updates

### Physics Simulation

| Layer | Test Coverage | Risk |
|-------|--------------|------|
| Force Calculation | ✅ Excellent (GPU tests) | Low |
| Position Updates | ✅ Good (physics tests) | Low |
| Constraint Handling | ✅ Good (ontology tests) | Low |
| Performance | ✅ Good (stress tests) | Low |

**Test Gap:** Minimal - physics is well-tested!

### Semantic Analysis

| Layer | Test Coverage | Risk |
|-------|--------------|------|
| Feature Extraction | ❌ None | Medium |
| Edge Generation | ❌ None | Medium |
| Caching | ❌ None | Medium |

**Test Gap:** Semantic features untested, but lower priority for migration

---

## 3. Test Gap Matrix

| Feature | Unit Tests | Integration Tests | E2E Tests | Priority |
|---------|-----------|------------------|-----------|----------|
| **Graph CRUD** | ❌ None | ⚠️ Partial | ❌ None | 🔴 P0 |
| **WebSocket** | ✅ Rate limit | ❌ None | ❌ None | 🟡 P1 |
| **GitHub Sync** | ❌ None | ❌ None | ❌ None | 🔴 P0 |
| **Physics** | ✅ Excellent | ✅ Good | ⚠️ Partial | 🟢 P2 |
| **Hexagonal Ports** | ❌ None | ❌ None | ❌ None | 🔴 P0 |
| **Event Sourcing** | ❌ None | ❌ None | ❌ None | 🔴 P0 |
| **Cache Invalidation** | ❌ None | ❌ None | ❌ None | 🟡 P1 |
| **Error Handling** | ✅ Excellent | ✅ Good | ✅ Good | 🟢 P2 |
| **Security** | ✅ Excellent | ✅ Good | ✅ Good | 🟢 P2 |

**Legend:**
- 🔴 P0 - Critical for migration
- 🟡 P1 - Important for stability
- 🟢 P2 - Already well-covered

---

## 4. Migration Test Plan

### Phase 1: Pre-Migration Tests (Create BEFORE migration)

#### A. Baseline Functionality Tests
```rust
// File: tests/hexagonal_migration/baseline_graph_crud_test.rs

#[tokio::test]
async fn test_baseline_get_graph_data() {
    // Test with CURRENT actor-based implementation
    let app_state = create_test_app_state().await;
    let response = get_graph_data(app_state).await;

    assert_eq!(response.status(), 200);
    let graph: GraphData = response.json().await;

    // BASELINE: Record current behavior
    let baseline_node_count = graph.nodes.len();
    let baseline_edge_count = graph.edges.len();

    // Save to file for comparison after migration
    save_baseline("graph_data", &graph);
}
```

#### B. GitHub Sync Regression Test (THE CRITICAL ONE!)
```rust
// File: tests/hexagonal_migration/github_sync_regression_test.rs

#[tokio::test]
async fn test_github_sync_316_nodes_regression() {
    // Setup: Point to local markdown directory (185 files)
    let config = GitHubSyncConfig {
        local_path: Some("data/markdown"),
        ..Default::default()
    };

    // Run GitHub sync
    let sync_service = GitHubSyncService::new(config);
    let result = sync_service.scan_repository().await.unwrap();

    // CRITICAL VALIDATIONS:

    // 1. Verify 185 markdown files scanned
    assert_eq!(
        result.files_scanned, 185,
        "Should scan exactly 185 markdown files"
    );

    // 2. Verify database has 316 nodes
    let db_nodes = kg_repo.get_all_nodes().await.unwrap();
    assert_eq!(
        db_nodes.len(), 316,
        "Database should contain 316 nodes (185 page + 131 linked_page)"
    );

    // 3. Verify ALL nodes have public=true
    let public_nodes: Vec<_> = db_nodes.iter()
        .filter(|n| n.metadata.get("public") == Some(&"true".to_string()))
        .collect();
    assert_eq!(
        public_nodes.len(), 316,
        "100% of nodes should have public=true metadata"
    );

    // 4. Verify node type distribution
    let page_nodes = db_nodes.iter().filter(|n| n.node_type == "page").count();
    let linked_page_nodes = db_nodes.iter().filter(|n| n.node_type == "linked_page").count();

    assert_eq!(page_nodes, 185, "Should have 185 'page' nodes");
    assert_eq!(linked_page_nodes, 131, "Should have 131 'linked_page' nodes");

    // 5. Verify private linked_pages filtered out
    // (646 total linked_pages - 316 public = 330 private filtered)
    assert!(
        result.nodes_filtered >= 330,
        "Should filter at least 330 private linked_page nodes"
    );

    // 6. Verify API returns 316 nodes
    let api_response = get_graph_data(app_state).await;
    let graph: GraphData = api_response.json().await;

    assert_eq!(
        graph.nodes.len(), 316,
        "API should return exactly 316 nodes after GitHub sync"
    );
}
```

#### C. Performance Benchmarks
```rust
// File: tests/hexagonal_migration/performance_benchmarks.rs

#[tokio::test]
async fn benchmark_graph_retrieval_speed() {
    let app_state = create_test_app_state().await;

    let start = Instant::now();
    for _ in 0..100 {
        let _ = get_graph_data(app_state.clone()).await;
    }
    let duration = start.elapsed();

    let avg_ms = duration.as_millis() / 100;

    // BASELINE: Record current performance
    save_baseline("graph_retrieval_ms", avg_ms);

    assert!(
        avg_ms < 50,
        "Graph retrieval should average under 50ms, got {}ms",
        avg_ms
    );
}
```

### Phase 2: Hexagonal Architecture Tests

#### A. Repository Port Tests
```rust
// File: tests/hexagonal_migration/repository_tests.rs

#[tokio::test]
async fn test_sqlite_graph_repository_crud() {
    let repo = SqliteGraphRepository::new("test.db").await.unwrap();

    // Test save
    let nodes = vec![
        Node { id: 1, label: "Test".to_string(), ..Default::default() },
    ];
    let edges = vec![];

    repo.save_graph(nodes.clone(), edges.clone()).await.unwrap();

    // Test retrieve
    let (retrieved_nodes, retrieved_edges) = repo.get_graph().await.unwrap();

    assert_eq!(nodes.len(), retrieved_nodes.len());
    assert_eq!(nodes[0].label, retrieved_nodes[0].label);

    // Test update
    let updated_node = Node { id: 1, label: "Updated".to_string(), ..Default::default() };
    repo.update_node(updated_node).await.unwrap();

    let (updated_nodes, _) = repo.get_graph().await.unwrap();
    assert_eq!(updated_nodes[0].label, "Updated");
}
```

#### B. Command Handler Tests
```rust
// File: tests/hexagonal_migration/command_handler_tests.rs

#[tokio::test]
async fn test_update_node_command_emits_event() {
    let handler = UpdateNodeCommandHandler::new(repo);

    let cmd = UpdateNodeCommand {
        node_id: 1,
        new_label: "Updated".to_string(),
    };

    let result = handler.execute(cmd).await.unwrap();

    // Verify event emitted
    assert!(result.events.contains(&NodeUpdatedEvent {
        node_id: 1,
        old_label: "Test".to_string(),
        new_label: "Updated".to_string(),
    }));
}
```

#### C. Event Flow Tests
```rust
// File: tests/hexagonal_migration/event_flow_tests.rs

#[tokio::test]
async fn test_github_sync_event_invalidates_cache() {
    let cache = Arc::new(Mutex::new(Some(GraphData::default())));
    let event_bus = EventBus::new();

    // Subscribe cache invalidator
    event_bus.subscribe(move |event: GitHubSyncCompletedEvent| {
        *cache.lock().unwrap() = None; // Invalidate cache
    });

    // Emit event
    event_bus.publish(GitHubSyncCompletedEvent {
        nodes_updated: 316,
    });

    // Verify cache invalidated
    assert!(cache.lock().unwrap().is_none());
}
```

### Phase 3: Post-Migration Validation

#### A. Comparison Tests
```rust
// File: tests/hexagonal_migration/migration_comparison_test.rs

#[tokio::test]
async fn test_hexagonal_matches_baseline() {
    let baseline = load_baseline("graph_data");

    // Test with NEW hexagonal implementation
    let app_state = create_hexagonal_app_state().await;
    let response = get_graph_data(app_state).await;
    let graph: GraphData = response.json().await;

    // Compare
    assert_eq!(graph.nodes.len(), baseline.nodes.len());
    assert_eq!(graph.edges.len(), baseline.edges.len());

    // Node-by-node comparison
    for (new_node, old_node) in graph.nodes.iter().zip(baseline.nodes.iter()) {
        assert_eq!(new_node.id, old_node.id);
        assert_eq!(new_node.label, old_node.label);
        // ... compare all fields
    }
}
```

#### B. Regression Test Suite
```rust
// File: tests/hexagonal_migration/regression_suite.rs

#[tokio::test]
async fn test_all_regressions() {
    // Run ALL baseline tests against hexagonal implementation
    test_github_sync_316_nodes_regression().await;
    test_websocket_broadcast_after_update().await;
    test_physics_simulation_stability().await;
    test_cache_invalidation_flow().await;
    test_concurrent_updates().await;
}
```

---

## 5. Recommended Test Files to Create

### Priority Order:

1. **`tests/github_sync_regression_test.rs`** - 🔴 **CRITICAL P0**
   - Test 316 nodes (185 files + 131 linked_pages)
   - Test 100% public metadata
   - Test private node filtering

2. **`tests/hexagonal_repository_tests.rs`** - 🔴 **P0**
   - Test SqliteGraphRepository CRUD
   - Test transaction handling
   - Test error recovery

3. **`tests/hexagonal_event_flow_tests.rs`** - 🔴 **P0**
   - Test event emission
   - Test event handlers
   - Test cache invalidation

4. **`tests/graph_crud_integration_test.rs`** - 🟡 **P1**
   - Test end-to-end graph operations
   - Test API → Actor → Database flow

5. **`tests/websocket_integration_test.rs`** - 🟡 **P1**
   - Test WebSocket broadcast after update
   - Test reconnection handling

6. **`tests/migration_baseline_tests.rs`** - 🟡 **P1**
   - Capture current behavior BEFORE migration
   - Performance benchmarks

---

## 6. Existing Tests That Will Break

### Tests Using Actor System Directly:

```rust
// tests/core_runtime_test.rs (lines 80-106)
#[tokio::test]
async fn test_graph_service_initialization() {
    // ⚠️ This will BREAK when we remove GraphServiceActor!
    let graph_actor = GraphServiceActor::new(settings);
    let _addr = graph_actor.start();
}
```

**Impact:** Low - Only initialization test, easily updated

### Tests Expecting Actor Messages:

```rust
// Any test sending ActorMessage to GraphServiceActor will break
addr.send(GetGraphData).await  // ❌ Will fail after migration
```

**Solution:** Update to use application service layer instead

---

## 7. Success Criteria

### ✅ Pre-Migration Checklist:

- [ ] GitHub sync regression test passes (316 nodes)
- [ ] Repository layer tests pass
- [ ] Event flow tests pass
- [ ] Baseline performance captured
- [ ] All critical paths have integration tests

### ✅ Post-Migration Checklist:

- [ ] ALL baseline tests pass with hexagonal implementation
- [ ] Performance meets or exceeds baseline
- [ ] GitHub sync STILL shows 316 nodes
- [ ] WebSocket broadcasts STILL work
- [ ] Physics simulation STILL stable
- [ ] Zero regressions in production

---

## 8. Risk Mitigation Strategy

### HIGH RISK: GitHub Sync (316 nodes)

**Risk:** Migration breaks public metadata filtering
**Mitigation:** Create regression test FIRST, run in CI
**Rollback Plan:** Keep old actor code in `_legacy` module for quick revert

### MEDIUM RISK: Event Sourcing

**Risk:** Events not emitted, cache not invalidated
**Mitigation:** Test event flow before migration
**Rollback Plan:** Feature flag to disable event system

### LOW RISK: Physics/GPU

**Risk:** Performance degradation
**Mitigation:** Extensive existing tests + benchmarks
**Rollback Plan:** GPU system is independent, low risk

---

## 9. Estimated Test Coverage

### Current Coverage (Estimated):

- **Overall:** ~60-65%
- **Critical Paths:** ~40% ⚠️
- **GitHub Sync:** **0%** ❌
- **GPU/Physics:** ~90% ✅
- **Security:** ~95% ✅

### Target Coverage After Test Creation:

- **Overall:** 85%+
- **Critical Paths:** 90%+
- **GitHub Sync:** 90%+
- **Hexagonal Layers:** 85%+

---

## 10. Next Steps

### Immediate Actions (Before Migration):

1. **Create GitHub sync regression test** (316 nodes) - 1 day
2. **Create repository layer tests** - 1 day
3. **Create event flow tests** - 1 day
4. **Capture baseline metrics** - 0.5 day
5. **Run full test suite and document results** - 0.5 day

**Total Estimated Time:** 4 days

### During Migration:

1. Run baseline tests continuously
2. Update breaking tests incrementally
3. Add new hexagonal tests as components are built

### Post-Migration:

1. Run comparison tests
2. Validate performance
3. Run regression suite
4. Deploy with monitoring

---

## Conclusion

### Current State: ⚠️ NOT READY FOR MIGRATION

**Blockers:**
1. ❌ No GitHub sync regression test (THE CRITICAL BUG FIX!)
2. ❌ No hexagonal architecture test infrastructure
3. ❌ No baseline performance metrics
4. ❌ Weak integration test coverage

### Recommendation: 🛑 DO NOT MIGRATE YET

**Create these 3 critical tests first:**

1. **GitHub sync regression test** - Ensures 316 nodes still work
2. **Repository CRUD tests** - Validates new database layer
3. **Event flow tests** - Ensures cache invalidation works

**Then:** Migration can proceed safely with confidence.

---

**Report Generated By:** Test Coverage Analyzer Agent
**Hive Mind Mission:** Ensure hexagonal migration succeeds without breaking production
**Queen's Mandate:** "No regressions. Test everything twice."
