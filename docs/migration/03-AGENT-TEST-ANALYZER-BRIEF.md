# 🧪 Agent 3: Test Coverage Analyzer - Mission Brief

**Agent ID:** test-analyzer
**Type:** Test Engineer
**Priority:** High
**Compute Units:** 10
**Memory Quota:** 256 MB

## Mission Statement

Analyze test coverage for monolithic actors. Identify gaps in test coverage that would prevent safe deletion. Create tests for critical paths before migration. Ensure 100% coverage for migration-critical functionality.

## Test Coverage Targets

### 1. GraphServiceActor Tests
**Location:** `tests/` directory (if exists) or embedded in modules

**Required Coverage:**
- ✅ Node CRUD operations
- ✅ Edge generation and management
- ✅ WebSocket binary protocol
- ✅ Physics simulation step
- ✅ GPU computation integration
- ✅ Semantic analysis triggers
- ✅ GitHub sync workflows
- ✅ Constraint application

### 2. Integration Test Analysis
**Focus:** End-to-end workflows

**Critical Paths:**
1. **API → Actor → Database**
   - Test: POST /api/graph/nodes → GraphServiceActor → Persist
2. **WebSocket → Real-time Updates**
   - Test: State change → Binary frame → Client receives
3. **GitHub → Metadata → Graph**
   - Test: Sync repo → Extract metadata → Create nodes
4. **Physics → GPU → Visualization**
   - Test: Simulation step → GPU compute → Position updates

### 3. Missing Tests Identification

**Analyze:**
```bash
# Find test files
find tests/ -name "*.rs"

# Check for test modules
grep -r "#\[test\]" src/

# Coverage report (if tarpaulin available)
cargo tarpaulin --out Html
```

**Document:**
- Untested functions in graph_actor.rs
- Missing integration tests
- Edge cases not covered
- Error handling gaps

### 4. Pre-Migration Test Suite

**Create:** `/home/devuser/workspace/project/tests/migration/pre_migration_baseline.rs`

**Test Categories:**
1. **State Verification Tests**
   - Verify graph state before migration
   - Snapshot current behavior
   - Baseline for post-migration validation

2. **Regression Prevention Tests**
   - Test all known issues/bugs
   - Ensure fixes don't regress
   - Document expected behavior

3. **Performance Baseline Tests**
   - WebSocket message throughput
   - Physics simulation FPS
   - API response times
   - Memory usage baselines

4. **Data Integrity Tests**
   - GitHub sync: 316 nodes expected
   - Metadata completeness
   - Edge consistency
   - Constraint validation

## Deliverables

### Primary Deliverable
Create: `/home/devuser/workspace/project/docs/migration/test-coverage-analysis.md`

**Required Sections:**
1. **Current Coverage Report**
   - Line coverage percentage
   - Function coverage percentage
   - Branch coverage percentage
2. **Missing Tests Catalog**
   - Critical functions without tests
   - Integration gaps
   - Edge cases not covered
3. **Pre-Migration Test Plan**
   - Tests to write before migration
   - Baseline tests for validation
   - Performance benchmarks
4. **Test Infrastructure Needs**
   - Mocking framework requirements
   - Test data setup
   - CI/CD integration
5. **Risk Assessment**
   - High-risk untested areas
   - Migration blockers
   - Recommended test-first order

### Secondary Deliverable
Create: `/home/devuser/workspace/project/tests/migration/mod.rs`

**Test Modules:**
```rust
mod pre_migration_baseline;
mod graph_state_snapshot;
mod websocket_validation;
mod github_sync_validation;
mod physics_simulation_baseline;
```

## Memory Storage

Store analysis under: `hive-coordination/testing/pre_migration_coverage`

**JSON Structure:**
```json
{
  "coverage_summary": {
    "line_coverage_percent": 65,
    "function_coverage_percent": 72,
    "branch_coverage_percent": 58,
    "critical_paths_covered": 80
  },
  "missing_tests": [
    {"function": "handle_batch_update", "risk": "high", "reason": "No integration test"},
    ...
  ],
  "baseline_tests_created": 15,
  "migration_blockers": [
    {"issue": "WebSocket binary protocol untested", "severity": "critical"},
    ...
  ],
  "recommended_test_order": ["state_verification", "integration", "performance"]
}
```

## Test Writing Guidelines

### Example Baseline Test
```rust
#[tokio::test]
async fn test_github_sync_baseline_316_nodes() {
    // Setup
    let actor = setup_graph_actor().await;

    // Execute GitHub sync
    let result = actor.send(SyncGitHubRepository {
        repo_url: "test_repo"
    }).await.unwrap();

    // Verify expected state
    assert_eq!(result.nodes_created, 316);
    assert_eq!(result.public_metadata_percent, 100.0);

    // Snapshot state for post-migration comparison
    let snapshot = actor.send(GetGraphSnapshot).await.unwrap();
    save_migration_baseline("github_sync_316", snapshot);
}
```

## Tools Available

- `cargo test` - Run test suite
- `cargo tarpaulin` - Coverage reporting
- `cargo nextest` - Fast test runner
- `criterion` - Benchmarking
- Test fixtures in `tests/fixtures/`

## Coordination

### Before Starting
```bash
npx claude-flow@alpha hooks pre-task --description "Test coverage analysis for migration"
npx claude-flow@alpha hooks session-restore --session-id "hive-hexagonal-migration"
```

### During Work
```bash
npx claude-flow@alpha hooks notify --message "Coverage analysis: 65% line coverage found"
npx claude-flow@alpha hooks notify --message "Created 15 baseline tests"
```

### After Completion
```bash
npx claude-flow@alpha hooks post-task --task-id "test-analyzer-coverage"
```

## Success Criteria

✅ Coverage report generated
✅ All missing tests documented
✅ Critical paths have baseline tests
✅ GitHub sync test verifies 316 nodes
✅ Pre-migration test suite created
✅ Risk assessment completed
✅ Findings stored in memory
✅ Report delivered in markdown

## Report to Queen

Upon completion, notify Queen Coordinator:
- Current coverage percentage
- Number of baseline tests created
- Critical gaps identified
- Migration readiness score (1-10)

**Expected Duration:** 45-60 minutes
**Blocker Escalation:** Report to Queen if coverage < 60%

---
*Assigned by Queen Coordinator*
