# Settings Migration QA Strategy

**Document Version**: 1.0.0
**Date**: 2025-10-22
**Status**: Draft - Awaiting Review

---

## Executive Summary

This document outlines the comprehensive Quality Assurance strategy for migrating the VisionFlow settings system from file-based YAML/TOML configuration to a database-backed, CQRS-driven architecture with hot-reload capabilities.

**Migration Scope:**
- **Source**: YAML (`data/settings.yaml`, `data/settings_ontology_extension.yaml`) + TOML (`data/dev_config.toml`)
- **Target**: SQLite database with hexagonal architecture (ports/adapters pattern)
- **Architecture**: CQRS (Command Query Responsibility Segregation) via `hexser` library
- **Components**: Repository port → SQLite adapter → CQRS handlers → REST API → Frontend UI

---

## 1. Migration Correctness Testing

### 1.1 Data Integrity Validation

**Objective**: Ensure 100% of settings are migrated without data loss or corruption.

#### Test Cases:

**TC-MIG-001: Complete Settings Migration**
```rust
#[tokio::test]
async fn test_yaml_to_db_complete_migration() {
    // Setup
    let temp_db = setup_temp_database();
    let yaml_settings = load_yaml_settings("data/settings.yaml");
    let toml_config = load_toml_config("data/dev_config.toml");

    // Execute migration
    let migrator = SettingsMigrator::new(temp_db.clone());
    migrator.migrate_yaml(&yaml_settings).await.unwrap();
    migrator.migrate_toml(&toml_config).await.unwrap();

    // Verify record counts
    let total_records = count_database_records(&temp_db);
    let expected_records = count_yaml_keys(&yaml_settings) + count_toml_keys(&toml_config);
    assert_eq!(total_records, expected_records, "All settings must be migrated");

    // Verify specific critical settings
    assert_setting_exists(&temp_db, "visualisation.graphs.logseq.physics.damping");
    assert_setting_exists(&temp_db, "visualisation.graphs.visionflow.physics.spring_k");
    assert_setting_exists(&temp_db, "system.network.port");
}
```

**TC-MIG-002: Value Type Preservation**
```rust
#[tokio::test]
async fn test_value_types_preserved() {
    let temp_db = setup_temp_database();
    let settings = load_test_settings();

    // Migrate
    migrate_to_database(&temp_db, &settings).await.unwrap();

    // Verify types
    let repo = SqliteSettingsRepository::new(Arc::new(temp_db));

    // Float values
    let damping = repo.get_setting("visualisation.graphs.logseq.physics.damping").await.unwrap();
    assert!(matches!(damping, Some(SettingValue::Float(_))));

    // Integer values
    let port = repo.get_setting("system.network.port").await.unwrap();
    assert!(matches!(port, Some(SettingValue::Integer(_))));

    // Boolean values
    let enabled = repo.get_setting("ontology.enabled").await.unwrap();
    assert!(matches!(enabled, Some(SettingValue::Boolean(_))));

    // String values
    let domain = repo.get_setting("system.network.domain").await.unwrap();
    assert!(matches!(domain, Some(SettingValue::String(_))));
}
```

**TC-MIG-003: Nested Structure Flattening**
```rust
#[tokio::test]
async fn test_nested_yaml_flattened_correctly() {
    let yaml = r#"
    visualisation:
      graphs:
        logseq:
          physics:
            damping: 0.6
            spring_k: 4.6
    "#;

    let temp_db = setup_temp_database();
    migrate_yaml_string(&temp_db, yaml).await.unwrap();

    let repo = SqliteSettingsRepository::new(Arc::new(temp_db));

    // Verify dot-notation keys
    assert!(repo.get_setting("visualisation.graphs.logseq.physics.damping").await.is_ok());
    assert!(repo.get_setting("visualisation.graphs.logseq.physics.spring_k").await.is_ok());
}
```

### 1.2 TOML Override Preservation

**TC-MIG-004: Dev Config Overrides**
```rust
#[tokio::test]
async fn test_toml_overrides_preserved() {
    let temp_db = setup_temp_database();

    // Load both YAML and TOML
    let yaml_settings = load_yaml_settings("data/settings.yaml");
    let toml_overrides = load_toml_config("data/dev_config.toml");

    // Migrate with priority handling (TOML overrides YAML)
    let migrator = SettingsMigrator::new(temp_db.clone());
    migrator.migrate_with_priority(&yaml_settings, &toml_overrides).await.unwrap();

    // Verify TOML values take precedence
    let repo = SqliteSettingsRepository::new(Arc::new(temp_db));
    let rest_length = repo.get_setting("physics.rest_length").await.unwrap().unwrap();

    // TOML specifies 100.0, YAML might specify different value
    assert_eq!(rest_length, SettingValue::Float(100.0));
}
```

### 1.3 Graph Separation Validation

**CRITICAL**: Logseq and VisionFlow graphs must maintain separate configurations.

**TC-MIG-005: Graph Configuration Isolation**
```rust
#[tokio::test]
async fn test_graph_settings_separated() {
    let temp_db = setup_temp_database();
    let settings = load_yaml_settings("data/settings.yaml");

    migrate_to_database(&temp_db, &settings).await.unwrap();

    let repo = SqliteSettingsRepository::new(Arc::new(temp_db));

    // Get physics settings for both graphs
    let logseq_physics = repo.get_physics_settings("logseq").await.unwrap();
    let visionflow_physics = repo.get_physics_settings("visionflow").await.unwrap();

    // Verify they are different
    assert_ne!(
        logseq_physics.damping,
        visionflow_physics.damping,
        "Logseq and Visionflow must have separate physics configs"
    );

    // Verify specific expected values from YAML
    assert_eq!(logseq_physics.damping, 0.6);
    assert_eq!(visionflow_physics.damping, 0.1);
}
```

---

## 2. Functional Testing

### 2.1 CRUD Operations

**TC-FUNC-001: Create Setting**
```rust
#[tokio::test]
async fn test_create_setting_via_cqrs() {
    let repo = create_test_repository();
    let handler = UpdateSettingHandler::new(repo.clone());

    // Execute directive
    handler.handle(UpdateSetting {
        key: "test.new_setting".to_string(),
        value: SettingValue::String("test_value".to_string()),
        description: Some("Test setting".to_string()),
    }).unwrap();

    // Query to verify
    let query_handler = GetSettingHandler::new(repo);
    let result = query_handler.handle(GetSetting {
        key: "test.new_setting".to_string(),
    }).unwrap();

    assert!(result.is_some());
    assert_eq!(result.unwrap(), SettingValue::String("test_value".to_string()));
}
```

**TC-FUNC-002: Read Setting (camelCase/snake_case)**
```rust
#[tokio::test]
async fn test_read_setting_case_insensitive() {
    let repo = create_test_repository();

    // Store with snake_case
    repo.set_setting("max_velocity", SettingValue::Float(100.0), None).await.unwrap();

    // Retrieve with camelCase
    let result = repo.get_setting("maxVelocity").await.unwrap();
    assert!(result.is_some());

    // Retrieve with snake_case
    let result2 = repo.get_setting("max_velocity").await.unwrap();
    assert!(result2.is_some());

    assert_eq!(result, result2, "Both case formats should return same value");
}
```

**TC-FUNC-003: Update Setting**
```rust
#[tokio::test]
async fn test_update_setting_via_api() {
    let app_state = setup_test_app_state();

    // Initial value
    let initial = json!({ "damping": 0.5 });
    update_setting_via_http(&app_state, "physics.damping", &initial).await;

    // Update value
    let updated = json!({ "damping": 0.8 });
    let response = update_setting_via_http(&app_state, "physics.damping", &updated).await;

    assert_eq!(response.status(), 200);

    // Verify update
    let current = get_setting_via_http(&app_state, "physics.damping").await;
    assert_eq!(current["value"], 0.8);
}
```

**TC-FUNC-004: Delete Setting**
```rust
#[tokio::test]
async fn test_delete_setting() {
    let repo = create_test_repository();

    // Create setting
    repo.set_setting("temp.test", SettingValue::Boolean(true), None).await.unwrap();

    // Delete
    repo.delete_setting("temp.test").await.unwrap();

    // Verify deletion
    let result = repo.get_setting("temp.test").await.unwrap();
    assert!(result.is_none());
}
```

### 2.2 Validation Rules Enforcement

**TC-FUNC-005: Range Validation**
```rust
#[tokio::test]
async fn test_physics_value_range_validation() {
    let repo = create_test_repository();
    let handler = UpdateSettingHandler::new(repo);

    // Test damping out of range (must be 0.0-1.0)
    let result = handler.handle(UpdateSetting {
        key: "visualisation.graphs.logseq.physics.damping".to_string(),
        value: SettingValue::Float(1.5), // Invalid: > 1.0
        description: None,
    });

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("above maximum"));
}
```

**TC-FUNC-006: Type Validation**
```rust
#[tokio::test]
async fn test_type_mismatch_validation() {
    let repo = create_test_repository();

    // Try to set numeric setting to string
    let result = repo.set_setting(
        "system.network.port",
        SettingValue::String("not_a_number".to_string()),
        None
    ).await;

    assert!(result.is_err());
}
```

### 2.3 Default Value Fallback

**TC-FUNC-007: Missing Setting Returns Default**
```rust
#[tokio::test]
async fn test_missing_setting_fallback() {
    let repo = create_test_repository();

    // Query non-existent setting
    let result = repo.get_setting("nonexistent.setting").await.unwrap();

    // Should return None, allowing application to use default
    assert!(result.is_none());

    // Application layer should provide default
    let default_value = result.unwrap_or(SettingValue::Float(1.0));
    assert_eq!(default_value, SettingValue::Float(1.0));
}
```

### 2.4 Category Filtering

**TC-FUNC-008: Filter Settings by Prefix**
```rust
#[tokio::test]
async fn test_get_settings_by_category() {
    let repo = create_test_repository();

    // Create multiple physics settings
    repo.set_setting("physics.damping", SettingValue::Float(0.6), None).await.unwrap();
    repo.set_setting("physics.gravity", SettingValue::Float(0.0001), None).await.unwrap();
    repo.set_setting("system.port", SettingValue::Integer(4000), None).await.unwrap();

    // Query all physics settings
    let physics_settings = repo.get_settings_by_prefix("physics").await.unwrap();

    assert_eq!(physics_settings.len(), 2);
    assert!(physics_settings.contains_key("physics.damping"));
    assert!(physics_settings.contains_key("physics.gravity"));
    assert!(!physics_settings.contains_key("system.port"));
}
```

### 2.5 Search Functionality

**TC-FUNC-009: Search Settings by Keyword**
```rust
#[tokio::test]
async fn test_search_settings() {
    let repo = create_test_repository();

    // Populate test data
    setup_test_settings(&repo).await;

    // Search for "physics"
    let results = repo.search_settings("physics").await.unwrap();

    assert!(!results.is_empty());
    assert!(results.iter().all(|(key, _)| key.contains("physics")));
}
```

---

## 3. Integration Testing

### 3.1 Backend Settings Propagation

**TC-INT-001: Settings Update Triggers Physics Recalculation**
```rust
#[tokio::test]
async fn test_physics_update_propagation() {
    let app_state = setup_full_app_state().await;

    // Update physics setting
    update_physics_setting(&app_state, "logseq", "damping", 0.8).await;

    // Verify physics engine received update
    tokio::time::sleep(Duration::from_millis(100)).await;

    let physics_state = app_state.graph_service
        .unwrap()
        .send(GetPhysicsState)
        .await
        .unwrap();

    assert_eq!(physics_state.damping, 0.8);
}
```

### 3.2 Frontend UI Updates

**TC-INT-002: WebSocket Notification on Setting Change**
```rust
#[actix_web::test]
async fn test_websocket_setting_notification() {
    let (app_state, mut ws_receiver) = setup_websocket_test_server().await;

    // Update setting via API
    update_setting_via_http(&app_state, "physics.damping", &json!(0.7)).await;

    // Wait for WebSocket notification
    let notification = tokio::time::timeout(
        Duration::from_secs(1),
        ws_receiver.recv()
    ).await.unwrap().unwrap();

    // Verify notification content
    assert_eq!(notification["type"], "setting_updated");
    assert_eq!(notification["key"], "physics.damping");
    assert_eq!(notification["value"], 0.7);
}
```

### 3.3 Hot-Reload Without Restart

**TC-INT-003: Hot-Reload Verification**
```rust
#[tokio::test]
async fn test_hot_reload_no_restart() {
    let app_state = setup_full_app_state().await;

    // Record server start time
    let start_time = app_state.start_time;

    // Update multiple settings
    update_setting(&app_state, "physics.damping", SettingValue::Float(0.9)).await;
    update_setting(&app_state, "system.websocket.heartbeatInterval", SettingValue::Integer(5000)).await;

    // Wait for propagation
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Verify server didn't restart
    assert_eq!(app_state.start_time, start_time);

    // Verify settings are active
    let current_damping = get_active_physics_setting(&app_state, "damping").await;
    assert_eq!(current_damping, 0.9);
}
```

**TC-INT-004: Hot-Reload Latency Measurement**
```rust
#[tokio::test]
async fn test_hot_reload_latency() {
    let app_state = setup_full_app_state().await;

    let start = Instant::now();

    // Update setting
    update_setting(&app_state, "physics.temperature", SettingValue::Float(2.0)).await;

    // Wait for setting to be active in physics engine
    while get_active_physics_setting(&app_state, "temperature").await != 2.0 {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    let latency = start.elapsed();

    // Requirement: < 100ms hot-reload latency
    assert!(latency < Duration::from_millis(100), "Hot-reload took {:?}", latency);
}
```

### 3.4 Multi-User Concurrency

**TC-INT-005: Concurrent Setting Updates**
```rust
#[tokio::test]
async fn test_concurrent_updates() {
    let app_state = Arc::new(setup_full_app_state().await);

    // Spawn 10 concurrent update tasks
    let mut handles = vec![];
    for i in 0..10 {
        let state = app_state.clone();
        let handle = tokio::spawn(async move {
            update_setting(&state, &format!("test.concurrent_{}", i), SettingValue::Integer(i)).await
        });
        handles.push(handle);
    }

    // Wait for all updates
    futures::future::join_all(handles).await;

    // Verify all settings were created
    for i in 0..10 {
        let value = get_setting(&app_state, &format!("test.concurrent_{}", i)).await.unwrap();
        assert_eq!(value, SettingValue::Integer(i));
    }
}
```

---

## 4. Performance Testing

### 4.1 Database Query Performance

**TC-PERF-001: Single Setting Read (< 1ms)**
```rust
#[tokio::test]
async fn bench_single_setting_read() {
    let repo = create_test_repository_with_data(1000).await;

    // Warm up cache
    repo.get_setting("physics.damping").await.unwrap();

    // Measure 100 reads
    let start = Instant::now();
    for _ in 0..100 {
        repo.get_setting("physics.damping").await.unwrap();
    }
    let elapsed = start.elapsed();

    let avg_time = elapsed / 100;
    println!("Average read time: {:?}", avg_time);

    // Requirement: < 1ms per setting read
    assert!(avg_time < Duration::from_millis(1));
}
```

**TC-PERF-002: Batch Update Efficiency**
```rust
#[tokio::test]
async fn bench_batch_update() {
    let repo = create_test_repository().await;

    // Prepare 50 updates
    let mut updates = HashMap::new();
    for i in 0..50 {
        updates.insert(
            format!("test.setting_{}", i),
            SettingValue::Float(i as f64)
        );
    }

    // Measure batch update
    let start = Instant::now();
    repo.set_settings_batch(updates).await.unwrap();
    let elapsed = start.elapsed();

    println!("Batch update (50 settings): {:?}", elapsed);

    // Requirement: < 50ms for 50 settings
    assert!(elapsed < Duration::from_millis(50));
}
```

### 4.2 Memory Usage

**TC-PERF-003: Cache Memory Overhead**
```rust
#[test]
fn test_cache_memory_usage() {
    let repo = create_test_repository_sync();

    // Get baseline memory
    let baseline = get_process_memory();

    // Load 1000 settings into cache
    for i in 0..1000 {
        repo.set_setting_sync(&format!("test.{}", i), SettingValue::Integer(i));
    }

    // Measure memory increase
    let after_cache = get_process_memory();
    let cache_overhead = after_cache - baseline;

    println!("Cache overhead for 1000 settings: {} KB", cache_overhead / 1024);

    // Requirement: < 5MB for 1000 settings
    assert!(cache_overhead < 5 * 1024 * 1024);
}
```

### 4.3 Hot-Reload Latency

**TC-PERF-004: Update-to-Active Latency**
```rust
#[tokio::test]
async fn bench_hot_reload_latency() {
    let app_state = setup_full_app_state().await;

    let mut latencies = vec![];

    // Measure 20 hot-reloads
    for i in 0..20 {
        let start = Instant::now();

        // Update setting
        update_setting(&app_state, "physics.temperature", SettingValue::Float(i as f64)).await;

        // Wait for active
        while get_active_setting(&app_state, "physics.temperature").await != i as f64 {
            tokio::time::sleep(Duration::from_micros(100)).await;
        }

        latencies.push(start.elapsed());
    }

    let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    let max_latency = latencies.iter().max().unwrap();

    println!("Average hot-reload latency: {:?}", avg_latency);
    println!("Max hot-reload latency: {:?}", max_latency);

    // Requirements:
    assert!(avg_latency < Duration::from_millis(50), "Average < 50ms");
    assert!(*max_latency < Duration::from_millis(100), "Max < 100ms");
}
```

---

## 5. User Acceptance Testing

### 5.1 Control Panel Usability

**UAT-001: Settings Discovery**
- [ ] User can browse settings by category (Physics, Network, XR, etc.)
- [ ] Search function finds settings by keyword
- [ ] Settings have human-readable descriptions
- [ ] Settings show current values and default values
- [ ] Settings indicate which are modified vs defaults

**UAT-002: Settings Modification**
- [ ] User can edit setting value inline
- [ ] Changes preview before applying
- [ ] Invalid values show clear error messages
- [ ] Changes apply immediately (hot-reload)
- [ ] Visual feedback confirms setting was saved

**UAT-003: Error Handling**
- [ ] Clear error messages for validation failures
- [ ] Explanation of why value was rejected
- [ ] Suggested valid range/values shown
- [ ] Network errors handled gracefully
- [ ] Ability to retry failed operations

### 5.2 Documentation Completeness

**DOC-001: Setting Descriptions**
```sql
-- Each setting should have description in database
SELECT key, description
FROM settings
WHERE description IS NULL OR description = '';

-- Expected result: 0 rows (all settings documented)
```

**DOC-002: API Documentation**
- [ ] OpenAPI/Swagger spec complete
- [ ] All endpoints documented
- [ ] Request/response examples provided
- [ ] Error codes documented
- [ ] Authentication requirements clear

---

## 6. Rollback Procedures

### 6.1 Pre-Migration Backup

**BACKUP-001: Create Backups**
```bash
#!/bin/bash
# Run before migration

BACKUP_DIR="data/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup config files
cp data/settings.yaml "$BACKUP_DIR/"
cp data/settings_ontology_extension.yaml "$BACKUP_DIR/"
cp data/dev_config.toml "$BACKUP_DIR/"

# Backup database (if exists)
if [ -f "data/visionflow.db" ]; then
    cp data/visionflow.db "$BACKUP_DIR/"
fi

echo "Backups created in: $BACKUP_DIR"
```

### 6.2 Database Export

**BACKUP-002: Export Database**
```bash
#!/bin/bash
# Export database to SQL script

sqlite3 data/visionflow.db .dump > data/backups/settings_dump_$(date +%Y%m%d).sql
```

### 6.3 Rollback to File-Based System

**ROLLBACK-001: Restore from Backup**
```bash
#!/bin/bash
# Restore file-based configuration

BACKUP_DIR="$1"  # Pass backup directory as argument

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

# Restore config files
cp "$BACKUP_DIR/settings.yaml" data/
cp "$BACKUP_DIR/settings_ontology_extension.yaml" data/
cp "$BACKUP_DIR/dev_config.toml" data/

# Disable database mode (set environment variable)
export USE_DATABASE_SETTINGS=false

# Restart application
./restart.sh
```

### 6.4 Version Flag System

**ROLLBACK-002: Feature Flag**
```rust
// In application startup
let use_database_settings = std::env::var("USE_DATABASE_SETTINGS")
    .unwrap_or_else(|_| "true".to_string())
    .parse::<bool>()
    .unwrap_or(true);

if use_database_settings {
    // Use SQLite repository
    let repository = Arc::new(SqliteSettingsRepository::new(db));
    app_state.settings_repository = repository;
} else {
    // Fallback to file-based settings
    let repository = Arc::new(FileSettingsRepository::new("data/settings.yaml"));
    app_state.settings_repository = repository;
}
```

---

## 7. Test Execution Scripts

### 7.1 Migration Validation Script

**File**: `tests/scripts/validate_migration.sh`
```bash
#!/bin/bash
set -e

echo "=== Settings Migration Validation ==="

# Run migration
cargo run --bin migrate_settings

# Run validation tests
echo "Running migration correctness tests..."
cargo test test_yaml_to_db_migration --release -- --nocapture

echo "Running type preservation tests..."
cargo test test_value_types_preserved --release -- --nocapture

echo "Running graph separation tests..."
cargo test test_graph_settings_separated --release -- --nocapture

# Generate report
echo "Generating validation report..."
cargo run --bin generate_migration_report

echo "✓ Migration validation complete"
```

### 7.2 Functional Test Suite

**File**: `tests/scripts/run_functional_tests.sh`
```bash
#!/bin/bash
set -e

echo "=== Functional Test Suite ==="

# CRUD operations
cargo test test_create_setting_via_cqrs --release
cargo test test_read_setting_case_insensitive --release
cargo test test_update_setting_via_api --release
cargo test test_delete_setting --release

# Validation
cargo test test_physics_value_range_validation --release
cargo test test_type_mismatch_validation --release

# Category/Search
cargo test test_get_settings_by_category --release
cargo test test_search_settings --release

echo "✓ All functional tests passed"
```

### 7.3 Performance Benchmark Suite

**File**: `tests/scripts/run_benchmarks.sh`
```bash
#!/bin/bash
set -e

echo "=== Performance Benchmark Suite ==="

# Database query performance
cargo bench bench_single_setting_read
cargo bench bench_batch_update

# Hot-reload latency
cargo bench bench_hot_reload_latency

# Memory usage
cargo test test_cache_memory_usage --release -- --nocapture --test-threads=1

echo "✓ Performance benchmarks complete"
echo "See target/criterion/report/index.html for detailed results"
```

### 7.4 Integration Test Suite

**File**: `tests/scripts/run_integration_tests.sh`
```bash
#!/bin/bash
set -e

echo "=== Integration Test Suite ==="

# Start test server
cargo build --release
./target/release/visionflow &
SERVER_PID=$!

# Wait for server startup
sleep 2

# Run integration tests
cargo test test_physics_update_propagation --release -- --nocapture
cargo test test_websocket_setting_notification --release -- --nocapture
cargo test test_hot_reload_no_restart --release -- --nocapture
cargo test test_concurrent_updates --release -- --nocapture

# Cleanup
kill $SERVER_PID

echo "✓ All integration tests passed"
```

---

## 8. Success Criteria

### 8.1 Migration Completeness
- ✅ **100% of settings migrated**: All YAML/TOML keys present in database
- ✅ **Zero data loss**: Every value matches source files
- ✅ **Type preservation**: Integer remains integer, float remains float, etc.
- ✅ **Graph separation maintained**: Logseq ≠ Visionflow settings

### 8.2 Performance Requirements
- ✅ **< 1ms per setting read** (with cache)
- ✅ **< 5ms per setting update** (database write)
- ✅ **< 50ms batch update** (50 settings)
- ✅ **< 100ms hot-reload latency** (update to active)
- ✅ **< 5MB cache overhead** (1000 settings)

### 8.3 Functional Requirements
- ✅ **All validation rules working**: Min/max, types, patterns
- ✅ **Hot-reload works**: No server restart needed
- ✅ **WebSocket notifications**: Clients receive updates
- ✅ **Multi-user safe**: Concurrent updates don't corrupt data
- ✅ **Rollback available**: Can revert to file-based system

### 8.4 User Acceptance
- ✅ **Positive user testing feedback**: Settings UI is intuitive
- ✅ **Error messages clear**: Users understand validation failures
- ✅ **Documentation complete**: All settings documented
- ✅ **Search works**: Users can find settings easily

---

## 9. Risk Analysis and Mitigation

### 9.1 Critical Risks

**RISK-001: Data Loss During Migration**
- **Impact**: Critical
- **Probability**: Low
- **Mitigation**:
  - Automated backups before migration
  - Dry-run mode to validate without writing
  - Checksum verification post-migration
  - Rollback script tested in advance

**RISK-002: Performance Degradation**
- **Impact**: High
- **Probability**: Medium
- **Mitigation**:
  - Benchmark before/after comparison
  - Cache warming on startup
  - Database indexing on key columns
  - Connection pooling

**RISK-003: Breaking Changes to API**
- **Impact**: High
- **Probability**: Low
- **Mitigation**:
  - Maintain backward compatibility
  - Version API endpoints
  - Comprehensive integration tests
  - Staged rollout

**RISK-004: Graph Configuration Conflation**
- **Impact**: Critical
- **Probability**: Medium (based on previous conflation issues)
- **Mitigation**:
  - Explicit separation tests (TC-MIG-005)
  - Profile-based physics settings storage
  - Validation that logseq ≠ visionflow
  - Code review checklist item

### 9.2 Additional Testing Required

**GAP-001: Missing Load Testing**
- [ ] Stress test with 10,000+ settings
- [ ] Concurrent user load simulation (100+ users)
- [ ] Long-running stability test (24+ hours)

**GAP-002: Missing Security Testing**
- [ ] SQL injection testing on setting keys
- [ ] Authorization checks (user can only modify their settings)
- [ ] Input sanitization validation

**GAP-003: Missing Browser Compatibility**
- [ ] Test UI in Chrome, Firefox, Safari, Edge
- [ ] Mobile browser testing
- [ ] WebSocket fallback mechanisms

---

## 10. Test Data Management

### 10.1 Test Fixtures

**File**: `tests/fixtures/test_settings.yaml`
```yaml
# Minimal test fixture with known values
visualisation:
  graphs:
    logseq:
      physics:
        damping: 0.6
        spring_k: 4.6
        repel_k: 13.28
    visionflow:
      physics:
        damping: 0.1
        spring_k: 10.0
        repel_k: 100.0
system:
  network:
    port: 4000
    domain: "test.local"
```

**File**: `tests/fixtures/test_dev_config.toml`
```toml
# Test developer overrides
[physics]
rest_length = 100.0
repulsion_cutoff = 150.0
max_force = 50.0
max_velocity = 150.0
```

### 10.2 Test Database Setup

```rust
// tests/common/setup.rs

pub fn setup_temp_database() -> Arc<DatabaseService> {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let db = DatabaseService::new(&db_path.to_str().unwrap()).unwrap();
    db.initialize_schema().unwrap();

    Arc::new(db)
}

pub async fn create_test_repository() -> Arc<dyn SettingsRepository> {
    let db = setup_temp_database();
    Arc::new(SqliteSettingsRepository::new(db))
}

pub async fn setup_test_app_state() -> AppState {
    let db = setup_temp_database();
    let repo = Arc::new(SqliteSettingsRepository::new(db.clone()));
    let settings_service = SettingsService::new(db.clone()).unwrap();

    AppState {
        settings_repository: repo,
        settings_service: Arc::new(settings_service),
        // ... other fields
    }
}
```

---

## 11. Continuous Integration

### 11.1 CI Pipeline Configuration

**File**: `.github/workflows/settings-migration-tests.yml`
```yaml
name: Settings Migration Tests

on:
  push:
    paths:
      - 'src/adapters/sqlite_settings_repository.rs'
      - 'src/services/settings_service.rs'
      - 'src/handlers/settings_handler.rs'
      - 'tests/settings_*.rs'
  pull_request:
    paths:
      - 'src/adapters/**'
      - 'src/services/settings_service.rs'
      - 'src/handlers/settings_handler.rs'

jobs:
  migration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable

      - name: Run migration validation
        run: ./tests/scripts/validate_migration.sh

      - name: Run functional tests
        run: ./tests/scripts/run_functional_tests.sh

      - name: Run integration tests
        run: ./tests/scripts/run_integration_tests.sh

      - name: Run benchmarks
        run: ./tests/scripts/run_benchmarks.sh

      - name: Upload test reports
        uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: |
            target/criterion/
            data/migration_report.json
```

---

## 12. Monitoring and Observability

### 12.1 Migration Metrics

Track these metrics during and after migration:

- **Migration duration**: Time to complete full migration
- **Error rate**: Percentage of failed migrations
- **Data integrity**: Checksum verification pass rate
- **Performance delta**: Before/after query latency comparison

### 12.2 Post-Migration Monitoring

```rust
// Add to telemetry
pub struct SettingsMetrics {
    pub cache_hit_rate: f64,
    pub avg_query_latency_ms: f64,
    pub hot_reload_latency_ms: f64,
    pub database_errors_count: u64,
    pub validation_failures_count: u64,
}

// Expose as Prometheus metrics
metrics::gauge!("settings_cache_hit_rate", cache_hit_rate);
metrics::histogram!("settings_query_latency_ms", avg_query_latency_ms);
metrics::histogram!("settings_hot_reload_latency_ms", hot_reload_latency_ms);
metrics::counter!("settings_database_errors", database_errors_count);
metrics::counter!("settings_validation_failures", validation_failures_count);
```

---

## Appendix A: Migration Checklist

**Pre-Migration:**
- [ ] Backup all YAML/TOML configuration files
- [ ] Export current database (if exists)
- [ ] Review migration script
- [ ] Test migration on staging environment
- [ ] Verify rollback procedure works

**During Migration:**
- [ ] Run migration script
- [ ] Monitor for errors
- [ ] Verify record counts
- [ ] Validate critical settings
- [ ] Test hot-reload functionality

**Post-Migration:**
- [ ] Run full test suite
- [ ] Verify application functionality
- [ ] Check performance metrics
- [ ] User acceptance testing
- [ ] Document any issues encountered
- [ ] Archive legacy config files (don't delete yet)

**Sign-off:**
- [ ] Technical lead approval
- [ ] QA approval
- [ ] User acceptance approval
- [ ] Production deployment approved

---

## Appendix B: Contact Information

**Migration Team:**
- Migration Lead: [Name]
- QA Lead: [Name]
- Database Administrator: [Name]
- DevOps Engineer: [Name]

**Escalation Path:**
- Level 1: Migration team
- Level 2: Engineering manager
- Level 3: CTO

---

**Document End**
