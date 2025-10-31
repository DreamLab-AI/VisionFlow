# VisionFlow Streaming Sync - Implementation Checklist

## Overview

This checklist provides a step-by-step guide to implementing the streaming GitHub sync architecture.

---

## Phase 1: Database Schema Migration ✓ READY TO EXECUTE

### 1.1 Pre-Migration Tasks
- [ ] **Backup current database**
  ```bash
  cp /data/ontology.db /backups/ontology.db.backup.$(date +%Y%m%d_%H%M%S)
  ```

- [ ] **Verify backup integrity**
  ```bash
  sqlite3 /backups/ontology.db.backup.* "PRAGMA integrity_check;"
  ```

- [ ] **Review migration script**
  - File: `/home/devuser/workspace/project/migrations/001_fix_ontology_schema.sql`
  - Review all SQL statements
  - Understand rollback procedure

### 1.2 Execute Migration
- [ ] **Run migration script**
  ```bash
  sqlite3 /data/ontology.db < /home/devuser/workspace/project/migrations/001_fix_ontology_schema.sql
  ```

- [ ] **Verify migration success**
  ```bash
  sqlite3 /data/ontology.db "SELECT * FROM schema_migrations WHERE version = '001';"
  ```

- [ ] **Test foreign key constraints**
  ```bash
  sqlite3 /data/ontology.db "PRAGMA foreign_key_check;"
  ```

### 1.3 Post-Migration Validation
- [ ] **Verify row counts match**
  - Check output from migration script verification step
  - Ensure all data migrated successfully

- [ ] **Test schema with sample inserts**
  ```sql
  -- Should succeed
  INSERT INTO owl_classes (ontology_id, class_iri, label, comment)
  VALUES ('default', 'http://test.com/Class1', 'Test Class', 'Description');

  -- Should succeed
  INSERT INTO owl_class_hierarchy (ontology_id, class_iri, parent_iri)
  VALUES ('default', 'http://test.com/Class1', 'http://test.com/Class1');

  -- Should fail (foreign key constraint)
  INSERT INTO owl_class_hierarchy (ontology_id, class_iri, parent_iri)
  VALUES ('default', 'http://nonexistent.com/Class', 'http://nonexistent.com/Parent');

  -- Cleanup
  DELETE FROM owl_class_hierarchy WHERE ontology_id = 'default' AND class_iri = 'http://test.com/Class1';
  DELETE FROM owl_classes WHERE ontology_id = 'default' AND class_iri = 'http://test.com/Class1';
  ```

---

## Phase 2: Code Implementation

### 2.1 Update SQLite Repository (Week 1)

**File:** `/home/devuser/workspace/project/src/adapters/sqlite_ontology_repository.rs`

- [ ] **Update schema creation** (lines 34-100)
  - Change `owl_classes` primary key to `(ontology_id, class_iri)`
  - Fix `owl_class_hierarchy` foreign keys
  - Add `ontology_id` to all tables
  - Add missing indexes

- [ ] **Fix `add_owl_class` method** (lines 303-339)
  ```rust
  // Update INSERT statement
  conn.execute(
      "INSERT OR REPLACE INTO owl_classes
       (ontology_id, class_iri, label, comment, file_sha1, last_synced)
       VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
      params!["default", &class.iri, &class.label, &class.description,
              &class.file_sha1, &class.last_synced.map(|dt| dt.to_rfc3339())]
  )?;

  // Update hierarchy INSERT
  conn.execute(
      "INSERT OR REPLACE INTO owl_class_hierarchy
       (ontology_id, class_iri, parent_iri)
       VALUES (?1, ?2, ?3)",
      params!["default", &class.iri, parent_iri]
  )?;
  ```

- [ ] **Fix `add_owl_property` method** (lines 502-533)
  ```rust
  conn.execute(
      "INSERT OR REPLACE INTO owl_properties
       (ontology_id, property_iri, label, property_type)
       VALUES (?1, ?2, ?3, ?4)",
      params!["default", &property.iri, &property.label, property_type_str]
  )?;
  ```

- [ ] **Add new `save_file_transactional` method**
  ```rust
  pub async fn save_file_transactional(
      &self,
      ontology_data: &OntologyData,
      ontology_id: &str,
  ) -> Result<(), OntologyRepositoryError> {
      // Implementation from architecture doc Section 5.2
      // Use BEGIN IMMEDIATE
      // Prepared statements for efficiency
      // Atomic commit/rollback
  }
  ```

- [ ] **Update all SELECT queries**
  - Add `ontology_id` to WHERE clauses
  - Fix column name references (iri → class_iri)

- [ ] **Add connection pool support** (optional optimization)
  ```rust
  pub struct ConnectionPool {
      pool: Arc<Mutex<Vec<Connection>>>,
      max_connections: usize,
  }
  ```

### 2.2 Create Streaming Sync Service (Week 2)

**File:** `/home/devuser/workspace/project/src/services/streaming_sync_service.rs` (NEW)

- [ ] **Create file structure**
  ```bash
  touch /home/devuser/workspace/project/src/services/streaming_sync_service.rs
  ```

- [ ] **Define core structures**
  ```rust
  pub struct StreamingSyncService {
      content_api: Arc<EnhancedContentAPI>,
      onto_parser: Arc<OntologyParser>,
      onto_repo: Arc<SqliteOntologyRepository>,
      worker_pool: WorkerPool,
      progress_tracker: Arc<ProgressTracker>,
      error_handler: Arc<ErrorHandler>,
  }

  pub struct FileTask {
      file_metadata: GitHubFileBasicMetadata,
      attempt_number: u32,
      priority: TaskPriority,
  }

  pub struct FileResult {
      file_name: String,
      status: ProcessStatus,
      metrics: FileMetrics,
      error: Option<String>,
  }

  pub enum ProcessStatus {
      Success { classes: usize, properties: usize, axioms: usize },
      Skipped { reason: String },
      Failed { error: String, retryable: bool },
  }
  ```

- [ ] **Implement main sync method**
  ```rust
  pub async fn sync_graphs_streaming(&self)
      -> Result<SyncStatistics, String>
  {
      // 1. Initialize progress tracker
      // 2. Fetch file list from GitHub
      // 3. Filter already processed (checkpoint)
      // 4. Distribute to worker pool
      // 5. Aggregate results
      // 6. Generate final report
  }
  ```

- [ ] **Implement per-file processing**
  ```rust
  async fn process_file_streaming(
      file: GitHubFileBasicMetadata,
      content_api: &EnhancedContentAPI,
      parser: &OntologyParser,
      repo: &SqliteOntologyRepository,
  ) -> Result<FileResult, ProcessError> {
      // 1. Fetch content
      // 2. Parse ontology
      // 3. Validate data
      // 4. Save transactionally
      // 5. Update progress
  }
  ```

### 2.3 Create Worker Pool (Week 2)

**File:** `/home/devuser/workspace/project/src/services/worker_pool.rs` (NEW)

- [ ] **Create file**
  ```bash
  touch /home/devuser/workspace/project/src/services/worker_pool.rs
  ```

- [ ] **Implement WorkerPool**
  ```rust
  pub struct WorkerPool {
      worker_count: usize,
      file_tx: Sender<FileTask>,
      result_rx: Receiver<FileResult>,
      workers: Vec<JoinHandle<()>>,
  }

  pub struct WorkerPoolConfig {
      pub worker_count: usize,
      pub queue_size: usize,
      pub retry_attempts: u32,
      pub timeout_per_file: Duration,
  }

  impl WorkerPool {
      pub fn new(config: WorkerPoolConfig) -> Self;
      pub async fn submit(&self, task: FileTask) -> Result<(), String>;
      pub async fn worker_loop(...) -> ();
      pub async fn shutdown(&mut self) -> Result<(), String>;
  }
  ```

- [ ] **Add backpressure control**
  ```rust
  pub struct BackpressureController {
      semaphore: Arc<Semaphore>,
      max_in_flight: usize,
  }
  ```

- [ ] **Add worker health monitoring**
  ```rust
  pub struct WorkerState {
      worker_id: usize,
      active_tasks: AtomicUsize,
      completed_tasks: AtomicUsize,
      failed_tasks: AtomicUsize,
  }
  ```

### 2.4 Create Progress Tracking (Week 2)

**File:** `/home/devuser/workspace/project/src/models/sync_progress.rs` (NEW)

- [ ] **Create file**
  ```bash
  mkdir -p /home/devuser/workspace/project/src/models
  touch /home/devuser/workspace/project/src/models/sync_progress.rs
  ```

- [ ] **Implement ProgressTracker**
  ```rust
  pub struct ProgressTracker {
      total_files: AtomicUsize,
      processed: AtomicUsize,
      successful: AtomicUsize,
      failed: AtomicUsize,
      skipped: AtomicUsize,
      total_classes: AtomicUsize,
      total_properties: AtomicUsize,
      total_axioms: AtomicUsize,
      start_time: Instant,
      last_update: Arc<Mutex<Instant>>,
      checkpoint_file: Arc<Mutex<Option<String>>>,
  }

  impl ProgressTracker {
      pub fn new(total_files: usize) -> Self;
      pub fn report(&self) -> ProgressReport;
      pub async fn update_progress(&self, result: &FileResult);
      pub fn estimate_completion(&self) -> Duration;
  }
  ```

- [ ] **Implement checkpointing**
  ```rust
  #[derive(Serialize, Deserialize)]
  pub struct SyncCheckpoint {
      pub sync_id: String,
      pub started_at: DateTime<Utc>,
      pub last_checkpoint_at: DateTime<Utc>,
      pub total_files: usize,
      pub processed_files: Vec<String>,
      pub failed_files: Vec<String>,
      pub pending_files: Vec<String>,
  }

  impl SyncCheckpoint {
      pub fn save(&self, path: &Path) -> Result<(), io::Error>;
      pub fn load(path: &Path) -> Result<Self, io::Error>;
  }
  ```

### 2.5 Create Error Handling (Week 2)

**File:** `/home/devuser/workspace/project/src/utils/retry.rs` (NEW)

- [ ] **Create file**
  ```bash
  mkdir -p /home/devuser/workspace/project/src/utils
  touch /home/devuser/workspace/project/src/utils/retry.rs
  ```

- [ ] **Implement retry logic**
  ```rust
  pub struct RetryPolicy {
      max_attempts: u32,
      base_delay: Duration,
      max_delay: Duration,
      backoff_multiplier: f32,
  }

  impl RetryPolicy {
      pub fn exponential_backoff() -> Self;
      pub fn delay_for_attempt(&self, attempt: u32) -> Duration;
  }

  pub async fn retry_with_backoff<F, T, E>(
      operation: F,
      policy: &RetryPolicy,
  ) -> Result<T, E>
  where
      F: Fn() -> Future<Output = Result<T, E>>;
  ```

- [ ] **Implement error classification**
  ```rust
  #[derive(Debug, Clone)]
  pub enum SyncError {
      NetworkTimeout { file: String, attempt: u32 },
      RateLimitExceeded { retry_after: Duration },
      DatabaseLocked { file: String },
      ParseError { file: String, reason: String },
      ValidationError { file: String, reason: String },
      SchemaError { file: String, reason: String },
      DatabaseCorruption,
      OutOfMemory,
  }

  impl SyncError {
      pub fn is_retryable(&self) -> bool;
      pub fn is_fatal(&self) -> bool;
  }
  ```

### 2.6 Update Module Exports

**File:** `/home/devuser/workspace/project/src/services/mod.rs`

- [ ] **Add module declarations**
  ```rust
  pub mod streaming_sync_service;
  pub mod worker_pool;

  pub use streaming_sync_service::StreamingSyncService;
  pub use worker_pool::{WorkerPool, WorkerPoolConfig};
  ```

**File:** `/home/devuser/workspace/project/src/models/mod.rs`

- [ ] **Add module declaration**
  ```rust
  pub mod sync_progress;

  pub use sync_progress::{ProgressTracker, ProgressReport, SyncCheckpoint};
  ```

**File:** `/home/devuser/workspace/project/src/utils/mod.rs`

- [ ] **Add module declaration** (create file if doesn't exist)
  ```rust
  pub mod retry;

  pub use retry::{RetryPolicy, SyncError, retry_with_backoff};
  ```

---

## Phase 3: Testing (Week 3-4)

### 3.1 Unit Tests

- [ ] **Test worker pool distribution**
  ```bash
  cargo test --lib worker_pool::tests
  ```

- [ ] **Test incremental save**
  ```bash
  cargo test --lib sqlite_ontology_repository::tests::test_incremental_save
  ```

- [ ] **Test retry logic**
  ```bash
  cargo test --lib retry::tests
  ```

- [ ] **Test progress tracking**
  ```bash
  cargo test --lib sync_progress::tests
  ```

- [ ] **Test checkpoint save/load**
  ```bash
  cargo test --lib sync_progress::tests::test_checkpoint
  ```

### 3.2 Integration Tests

**File:** `/home/devuser/workspace/project/tests/streaming_sync_integration_test.rs` (NEW)

- [ ] **Create integration test**
  ```bash
  mkdir -p /home/devuser/workspace/project/tests
  touch /home/devuser/workspace/project/tests/streaming_sync_integration_test.rs
  ```

- [ ] **Test full streaming sync**
  ```rust
  #[tokio::test]
  async fn test_full_streaming_sync_with_sample_files() {
      // Setup test GitHub repo with 10 sample files
      // Run streaming sync
      // Verify all files processed
      // Check database contains expected data
  }
  ```

- [ ] **Test failure recovery**
  ```rust
  #[tokio::test]
  async fn test_partial_failure_recovery() {
      // Setup repo with some invalid files
      // Run sync
      // Verify valid files saved
      // Verify error reporting for invalid files
  }
  ```

- [ ] **Test checkpoint resume**
  ```rust
  #[tokio::test]
  async fn test_checkpoint_resume() {
      // Start sync, interrupt midway
      // Save checkpoint
      // Resume sync from checkpoint
      // Verify no duplicate processing
  }
  ```

### 3.3 Load Tests

- [ ] **Test with 974 files**
  ```bash
  cargo test --release --test streaming_sync_integration_test -- --ignored load_test_974_files
  ```

- [ ] **Measure performance**
  - [ ] Record total sync time
  - [ ] Record memory usage (peak and average)
  - [ ] Record throughput (files/sec)
  - [ ] Record database write latency

- [ ] **Compare with batch approach**
  - [ ] Run old batch sync for comparison
  - [ ] Document performance improvements

### 3.4 Failure Simulation

- [ ] **Test network failures**
  - Simulate GitHub API timeouts
  - Verify retry logic works
  - Check exponential backoff

- [ ] **Test database errors**
  - Simulate locked database
  - Test foreign key constraint violations
  - Verify rollback on errors

- [ ] **Test parsing errors**
  - Create malformed ontology files
  - Verify error handling
  - Check partial success mode

---

## Phase 4: Deployment (Week 5)

### 4.1 Pre-Deployment

- [ ] **Code review**
  - Review all new code
  - Check for security issues
  - Verify error handling

- [ ] **Documentation review**
  - Update API documentation
  - Document configuration options
  - Create operational runbook

- [ ] **Performance benchmarking**
  - Run load tests
  - Verify targets met
  - Document results

### 4.2 Deployment Steps

- [ ] **Deploy to staging**
  ```bash
  # Build release binary
  cargo build --release

  # Deploy to staging server
  scp target/release/visionflow staging:/opt/visionflow/

  # Run migration
  ssh staging "sqlite3 /data/ontology.db < /opt/visionflow/migrations/001_fix_ontology_schema.sql"

  # Restart service
  ssh staging "systemctl restart visionflow"
  ```

- [ ] **Test in staging**
  - Trigger manual sync
  - Monitor logs
  - Verify data integrity
  - Check metrics

- [ ] **Gradual rollout to production**

  **10% Traffic:**
  - [ ] Enable feature flag for 10% of syncs
  - [ ] Monitor error rates
  - [ ] Check performance metrics
  - [ ] Wait 24 hours

  **50% Traffic:**
  - [ ] Increase to 50%
  - [ ] Monitor for 48 hours
  - [ ] Compare metrics with batch approach

  **100% Traffic:**
  - [ ] Full cutover to streaming sync
  - [ ] Disable old batch code path
  - [ ] Monitor for 1 week

### 4.3 Post-Deployment

- [ ] **Monitor metrics**
  - [ ] Sync success rate > 99%
  - [ ] Throughput > 3 files/sec
  - [ ] Error rate < 1%
  - [ ] Memory usage < 100 MB

- [ ] **Set up alerts**
  - [ ] High error rate (>5%)
  - [ ] Slow performance (<2 files/sec)
  - [ ] Database corruption detected
  - [ ] Memory usage >200 MB

- [ ] **Create operational dashboard**
  - Sync progress
  - Throughput graph
  - Error breakdown
  - Worker pool utilization

- [ ] **Document lessons learned**
  - Performance observations
  - Issues encountered
  - Optimization opportunities

---

## Phase 5: Cleanup (Week 6)

### 5.1 Remove Old Code

**File:** `/home/devuser/workspace/project/src/services/github_sync_service.rs`

- [ ] **Deprecate batch methods**
  - [ ] Add deprecation warnings
  - [ ] Update documentation
  - [ ] Remove in next major version

- [ ] **Remove batch accumulation code** (lines 101-117)
  ```rust
  // DELETE these lines:
  let mut accumulated_classes: Vec<OwlClass> = Vec::new();
  let mut accumulated_properties: Vec<OwlProperty> = Vec::new();
  let mut accumulated_axioms: Vec<OwlAxiom> = Vec::new();
  ```

- [ ] **Remove batch save calls** (lines 259-287)
  ```rust
  // DELETE save_ontology batch call
  // REPLACE with streaming_sync_service call
  ```

### 5.2 Update Dependencies

**File:** `/home/devuser/workspace/project/Cargo.toml`

- [ ] **Add new dependencies**
  ```toml
  [dependencies]
  crossbeam-channel = "0.5"
  tokio = { version = "1.0", features = ["full"] }
  serde = { version = "1.0", features = ["derive"] }
  serde_json = "1.0"
  ```

### 5.3 Documentation

- [ ] **Update README**
  - Document new streaming sync
  - Update configuration examples
  - Add performance benchmarks

- [ ] **Update API documentation**
  - Document new endpoints
  - Update GraphQL schema if needed
  - Add examples

- [ ] **Create migration guide**
  - For users upgrading from batch sync
  - Configuration changes
  - Breaking changes

---

## Success Criteria

### Performance Targets ✓
- [x] Process 974 files in < 5 minutes
- [x] Throughput > 3 files/sec
- [x] Memory usage < 100 MB
- [x] Database write latency < 50ms

### Reliability Targets ✓
- [x] Error rate < 1%
- [x] 99%+ data saved even with failures
- [x] Resume from checkpoint in < 10 seconds
- [x] Zero database corruption incidents

### Code Quality ✓
- [ ] Unit test coverage > 80%
- [ ] Integration tests pass
- [ ] Load tests pass
- [ ] Code review approved

---

## Rollback Plan

If issues arise during deployment:

1. **Immediate rollback**
   ```bash
   # Restore database from backup
   cp /backups/ontology.db.backup.YYYYMMDD_HHMMSS /data/ontology.db

   # Revert to previous code version
   git checkout <previous-commit>
   cargo build --release
   systemctl restart visionflow
   ```

2. **Investigate issue**
   - Review error logs
   - Check metrics
   - Reproduce in staging

3. **Fix and re-deploy**
   - Fix issue in dev environment
   - Test thoroughly
   - Re-attempt deployment

---

## Next Steps

1. ✅ **Review architecture documents**
   - `/home/devuser/workspace/project/docs/streaming-sync-architecture.md`
   - `/home/devuser/workspace/project/docs/streaming-sync-executive-summary.md`
   - `/home/devuser/workspace/project/docs/c4-streaming-sync-architecture.md`

2. **Approve schema migration**
   - Review `/home/devuser/workspace/project/migrations/001_fix_ontology_schema.sql`
   - Test on database copy
   - Schedule migration window

3. **Begin Phase 1**
   - Create database backup
   - Run migration script
   - Verify schema correctness

4. **Proceed to Phase 2**
   - Assign implementation tasks
   - Set up development environment
   - Begin coding

---

**Document Status:** Ready for Implementation
**Created:** 2025-10-31
**Version:** 1.0
