# ✅ Agent 6: Integration Validator - Mission Brief

**Agent ID:** integration-validator
**Type:** QA Engineer
**Priority:** High
**Compute Units:** 15
**Memory Quota:** 512 MB

## Mission Statement

Validate integration after each migration phase. Test that API endpoints return correct data, WebSocket updates propagate properly, physics simulation runs correctly, and GitHub sync works as expected (316 nodes with 100% public metadata). Act as quality gatekeeper preventing broken migrations.

## Validation Responsibilities

### Per-Phase Validation

After **each migration phase**, run comprehensive validation before allowing next phase to proceed.

### Phase 1 Validation: Read Operations
**Trigger:** After Migration Executor completes Phase 1

**Tests to Run:**
1. **API Endpoint Validation**
```bash
# Test all GET endpoints
curl http://localhost:8080/api/graph/nodes | jq '.length'
# Expected: > 0 nodes

curl http://localhost:8080/api/graph/nodes/1 | jq '.id'
# Expected: node data returned

curl http://localhost:8080/api/graph/stats | jq '.node_count'
# Expected: accurate statistics
```

2. **Data Consistency Check**
```rust
// tests/integration/phase1_validation.rs

#[tokio::test]
async fn test_query_handler_returns_same_data_as_actor() {
    let query_result = query_handler.handle_list_nodes(ListNodesQuery).await.unwrap();
    let actor_result = graph_actor.send(GetNodes).await.unwrap();

    assert_eq!(query_result, actor_result, "Query handler must return identical data");
}
```

3. **Performance Baseline**
```bash
# Measure query performance
ab -n 1000 -c 10 http://localhost:8080/api/graph/nodes
# Expected: < 50ms p95 latency
```

**Phase 1 Pass Criteria:**
✅ All GET endpoints functional
✅ Data matches actor results
✅ Performance within 10% of baseline
✅ No errors in logs

---

### Phase 2 Validation: Write Operations
**Trigger:** After Migration Executor completes Phase 2

**Tests to Run:**
1. **Command Execution Validation**
```bash
# Test POST/PUT/DELETE operations
curl -X POST http://localhost:8080/api/graph/nodes \
  -H "Content-Type: application/json" \
  -d '{"name":"test_node","type":"file"}'
# Expected: 201 Created with node data

curl -X PUT http://localhost:8080/api/graph/nodes/123 \
  -H "Content-Type: application/json" \
  -d '{"name":"updated_name"}'
# Expected: 200 OK with updated node

curl -X DELETE http://localhost:8080/api/graph/nodes/123
# Expected: 204 No Content
```

2. **Event Publishing Verification**
```rust
#[tokio::test]
async fn test_command_publishes_event() {
    let mut event_receiver = event_publisher.subscribe();

    // Execute command
    command_handler.handle_add_node(AddNodeCommand { ... }).await.unwrap();

    // Verify event published
    let event = timeout(Duration::from_secs(1), event_receiver.recv())
        .await
        .expect("Event should be published")
        .unwrap();

    match event {
        DomainEvent::NodeAdded(e) => assert_eq!(e.node_id, expected_id),
        _ => panic!("Wrong event type"),
    }
}
```

3. **Transaction Integrity**
```rust
#[tokio::test]
async fn test_command_rollback_on_failure() {
    // Simulate failure after partial execution
    let result = command_handler.handle_add_node(failing_command).await;

    assert!(result.is_err(), "Command should fail");

    // Verify no partial state persisted
    let node = repository.get_node(failed_node_id).await;
    assert!(node.is_err(), "Node should not exist after rollback");
}
```

**Phase 2 Pass Criteria:**
✅ All POST/PUT/DELETE work correctly
✅ Events published for all mutations
✅ Transactions atomic (rollback on failure)
✅ No data corruption

---

### Phase 3 Validation: WebSocket Real-time Updates
**Trigger:** After Migration Executor completes Phase 3

**Tests to Run:**
1. **WebSocket Connection Test**
```javascript
// tests/integration/websocket_validation.js

const ws = new WebSocket('ws://localhost:8080/ws/graph');

ws.on('open', () => {
    console.log('WebSocket connected');
});

ws.on('message', (data) => {
    const frame = parseBinaryFrame(data);
    console.log('Received update:', frame);
    assert(frame.type === 'node_update', 'Frame type should be node_update');
});
```

2. **Real-time Update Latency**
```rust
#[tokio::test]
async fn test_websocket_update_latency() {
    let ws_client = connect_websocket().await;
    let start = Instant::now();

    // Trigger state change
    command_handler.handle_add_node(AddNodeCommand { ... }).await.unwrap();

    // Wait for WebSocket update
    let update = ws_client.recv_frame().await.unwrap();
    let latency = start.elapsed();

    assert!(latency < Duration::from_millis(50), "Update latency too high: {:?}", latency);
}
```

3. **Binary Protocol Compatibility**
```rust
#[tokio::test]
async fn test_binary_protocol_unchanged() {
    let ws_client = connect_websocket().await;

    // Trigger update
    command_handler.handle_update_node(UpdateNodeCommand { ... }).await.unwrap();

    // Receive binary frame
    let frame = ws_client.recv_binary_frame().await.unwrap();

    // Verify binary format matches specification
    assert_eq!(frame[0], 0x01, "Frame type byte incorrect");
    assert_eq!(frame.len(), expected_frame_size(), "Frame size changed");
}
```

**Phase 3 Pass Criteria:**
✅ WebSocket connections stable
✅ Updates arrive in < 50ms
✅ Binary protocol unchanged
✅ No missed updates

---

### Phase 4 Validation: Physics Simulation
**Trigger:** After Migration Executor completes Phase 4

**Tests to Run:**
1. **Simulation Step Validation**
```rust
#[tokio::test]
async fn test_physics_simulation_runs() {
    let physics_service = setup_physics_service().await;

    // Execute simulation step
    let result = physics_service.step(0.016).await.unwrap();

    assert!(result.positions.len() > 0, "Positions should be updated");
    assert!(result.positions.iter().all(|p| p.x.is_finite()), "Positions must be valid");
}
```

2. **GPU Computation Correctness**
```rust
#[tokio::test]
async fn test_gpu_physics_matches_baseline() {
    let baseline_positions = load_baseline_positions();

    let physics_service = setup_physics_service().await;
    let result = physics_service.step(0.016).await.unwrap();

    // Allow small floating point differences
    for (actual, expected) in result.positions.iter().zip(baseline_positions.iter()) {
        assert!((actual.x - expected.x).abs() < 0.001, "Position diverged from baseline");
    }
}
```

3. **Performance Benchmark**
```rust
#[tokio::test]
async fn test_physics_performance() {
    let physics_service = setup_physics_service().await;

    let start = Instant::now();
    for _ in 0..60 {
        physics_service.step(0.016).await.unwrap();
    }
    let duration = start.elapsed();

    let fps = 60.0 / duration.as_secs_f32();
    assert!(fps >= 60.0, "Physics simulation should maintain 60 FPS, got: {}", fps);
}
```

**Phase 4 Pass Criteria:**
✅ Simulation produces valid positions
✅ Results match baseline (within tolerance)
✅ Performance >= 60 FPS
✅ GPU adapter works correctly

---

### Continuous Validation: GitHub Sync

**Run after EVERY phase** to ensure GitHub integration still works.

**Tests to Run:**
1. **Sync Execution**
```bash
# Trigger GitHub sync
curl -X POST http://localhost:8080/api/github/sync \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"test_repository"}'
```

2. **Node Count Verification**
```rust
#[tokio::test]
async fn test_github_sync_creates_316_nodes() {
    // Execute sync
    github_sync_service.sync_repository("test_repo").await.unwrap();

    // Verify node count
    let stats = query_handler.handle_get_stats(GetGraphStatsQuery).await.unwrap();
    assert_eq!(stats.node_count, 316, "GitHub sync should create exactly 316 nodes");
}
```

3. **Metadata Completeness**
```rust
#[tokio::test]
async fn test_github_sync_100_percent_public_metadata() {
    github_sync_service.sync_repository("test_repo").await.unwrap();

    let nodes = query_handler.handle_list_nodes(ListNodesQuery::all()).await.unwrap();

    let public_metadata_count = nodes.iter()
        .filter(|n| n.metadata.is_public)
        .count();

    let percentage = (public_metadata_count as f32 / nodes.len() as f32) * 100.0;
    assert_eq!(percentage, 100.0, "All nodes should have public metadata");
}
```

**GitHub Sync Pass Criteria:**
✅ Sync completes successfully
✅ Exactly 316 nodes created
✅ 100% public metadata
✅ Edges generated correctly

---

## Deliverables

### Primary Deliverable
Create: `/home/devuser/workspace/project/docs/migration/validation-report.md`

**Required Sections:**
1. **Phase Validation Summary**
   - Pass/fail for each phase
   - Test results
   - Performance metrics
2. **Regression Test Results**
   - Baseline vs current behavior
   - Any differences found
   - Issues identified
3. **Performance Comparison**
   - Latency measurements
   - Throughput metrics
   - Memory usage
4. **GitHub Sync Validation**
   - Node count verification
   - Metadata completeness
   - Sync timing
5. **Issues Found**
   - Bugs discovered
   - Performance regressions
   - Recommendations for fixes

### Test Suite
Create: `/home/devuser/workspace/project/tests/integration/migration_validation.rs`

**Test modules:**
```rust
mod phase1_read_operations;
mod phase2_write_operations;
mod phase3_websocket_events;
mod phase4_physics_simulation;
mod continuous_github_sync;
```

## Memory Storage

Store validation results under: `hive-coordination/validation/integration_tests`

**JSON Structure:**
```json
{
  "validation_summary": {
    "phase1": { "status": "passed", "tests_run": 15, "failures": 0 },
    "phase2": { "status": "passed", "tests_run": 20, "failures": 0 },
    "phase3": { "status": "passed", "tests_run": 18, "failures": 0 },
    "phase4": { "status": "passed", "tests_run": 12, "failures": 0 }
  },
  "github_sync_validation": {
    "nodes_created": 316,
    "public_metadata_percent": 100.0,
    "status": "passed"
  },
  "performance_metrics": {
    "api_latency_p95_ms": 45,
    "websocket_update_latency_ms": 30,
    "physics_fps": 62.5
  },
  "issues_found": [],
  "overall_status": "PASSED"
}
```

## Coordination

### Before Each Phase Validation
```bash
npx claude-flow@alpha hooks pre-task --description "Validating migration phase X"
```

### After Validation Complete
```bash
npx claude-flow@alpha hooks post-task --task-id "integration-validation-phaseX"
npx claude-flow@alpha hooks notify --message "Phase X validation: PASSED"
```

## Success Criteria

✅ All 4 phases validated successfully
✅ GitHub sync verified after every phase
✅ Performance within acceptable range
✅ No regressions detected
✅ Integration tests pass 100%
✅ Validation report delivered

## Blocker Protocol

**If validation FAILS:**
1. STOP migration immediately
2. Document failure in memory
3. Notify Queen Coordinator with details
4. Wait for fix before proceeding

## Report to Queen

Upon completion of each phase, notify Queen Coordinator:
- Phase validation status (PASS/FAIL)
- Test coverage (tests run / tests passed)
- Performance metrics
- Issues found (if any)
- Recommendation to proceed or block

**Expected Duration:** 20-30 minutes per phase
**Blocker Escalation:** Immediate escalation if any phase fails

---
*Assigned by Queen Coordinator - Quality Gatekeeper*
