# Pipeline Integration Architecture

## Overview

This document describes the end-to-end event-driven data pipeline that processes ontology data from GitHub through reasoning, constraint generation, GPU physics, and client delivery.

## System Architecture

### Pipeline Stages

```
┌─────────────┐
│   GitHub    │
│  Repository │
└──────┬──────┘
       │ OWL Files
       ▼
┌─────────────────────────────┐
│ GitHubSyncService           │
│ - Parse OntologyBlock       │
│ - Save to unified.db        │
│ - SHA1 deduplication        │
└──────┬──────────────────────┘
       │ OntologyModified Event
       ▼
┌─────────────────────────────┐
│ OntologyPipelineService     │
│ - Event orchestration       │
│ - Backpressure management   │
│ - Error recovery            │
└──────┬──────────────────────┘
       │ TriggerReasoning
       ▼
┌─────────────────────────────┐
│ ReasoningActor              │
│ - CustomReasoner inference  │
│ - Cache management          │
│ - EL++ subsumption         │
└──────┬──────────────────────┘
       │ InferredAxioms
       ▼
┌─────────────────────────────┐
│ ConstraintBuilder           │
│ - Axiom → Physics forces    │
│ - SubClassOf → Attraction   │
│ - DisjointWith → Repulsion  │
└──────┬──────────────────────┘
       │ ConstraintSet
       ▼
┌─────────────────────────────┐
│ OntologyConstraintActor     │
│ - GPU upload                │
│ - CUDA kernel execution     │
│ - CPU fallback              │
└──────┬──────────────────────┘
       │ Forces Applied
       ▼
┌─────────────────────────────┐
│ ForceComputeActor           │
│ - Physics simulation        │
│ - Position updates          │
│ - Velocity integration      │
└──────┬──────────────────────┘
       │ Node Positions
       ▼
┌─────────────────────────────┐
│ WebSocket Broadcasting      │
│ - Binary protocol           │
│ - Rate limiting             │
│ - Backpressure              │
└──────┬──────────────────────┘
       │ Binary Node Data
       ▼
┌─────────────┐
│   Client    │
│   Browser   │
└─────────────┘
```

## Event-Driven Architecture

### Event Types

#### 1. OntologyModifiedEvent
**Trigger**: GitHubSyncService saves OWL data
**Payload**:
```rust
struct OntologyModifiedEvent {
    ontology_id: i64,
    ontology: Ontology,
    source: String,
    timestamp: DateTime<Utc>,
    correlation_id: String,
}
```

**Handlers**:
- OntologyPipelineService::on_ontology_modified()

#### 2. ReasoningCompleteEvent
**Trigger**: ReasoningActor finishes inference
**Payload**:
```rust
struct ReasoningCompleteEvent {
    ontology_id: i64,
    inferred_axioms: Vec<InferredAxiom>,
    inference_time_ms: u64,
    cache_hit: bool,
    correlation_id: String,
}
```

**Handlers**:
- OntologyPipelineService::on_reasoning_complete()

#### 3. ConstraintsGeneratedEvent
**Trigger**: ConstraintBuilder generates physics constraints
**Payload**:
```rust
struct ConstraintsGeneratedEvent {
    constraint_set: ConstraintSet,
    axiom_count: usize,
    constraint_count: usize,
    correlation_id: String,
}
```

**Handlers**:
- OntologyConstraintActor::on_constraints_generated()

#### 4. GPUUploadCompleteEvent
**Trigger**: Constraints uploaded to GPU
**Payload**:
```rust
struct GPUUploadCompleteEvent {
    constraint_count: usize,
    upload_time_ms: f32,
    gpu_memory_used: usize,
    correlation_id: String,
}
```

**Handlers**:
- ForceComputeActor::on_gpu_upload_complete()

#### 5. PositionsUpdatedEvent
**Trigger**: Physics simulation updates node positions
**Payload**:
```rust
struct PositionsUpdatedEvent {
    positions: Vec<(u32, BinaryNodeData)>,
    iteration: u32,
    correlation_id: String,
}
```

**Handlers**:
- ClientManagerActor::broadcast_positions()

## Data Flow

### 1. GitHub → Database

```rust
// GitHubSyncService::save_ontology_data()
async fn save_ontology_data(&self, onto_data: OntologyData) -> Result<(), String> {
    // 1. Save to unified.db
    self.onto_repo.save_ontology(
        &onto_data.classes,
        &onto_data.properties,
        &onto_data.axioms
    ).await?;

    // 2. Build Ontology struct
    let ontology = build_ontology_from_data(&onto_data);

    // 3. Trigger pipeline (async, non-blocking)
    if let Some(pipeline) = &self.pipeline_service {
        let correlation_id = Uuid::new_v4().to_string();
        tokio::spawn(async move {
            pipeline.on_ontology_modified(1, ontology, correlation_id).await
        });
    }

    Ok(())
}
```

### 2. Reasoning Pipeline

```rust
// OntologyPipelineService::on_ontology_modified()
pub async fn on_ontology_modified(
    &self,
    ontology_id: i64,
    ontology: Ontology,
    correlation_id: String,
) -> Result<OntologyPipelineStats, String> {
    info!("[{}] Ontology modification detected", correlation_id);

    // 1. Trigger reasoning
    let msg = TriggerReasoning { ontology_id, ontology };
    let axioms = self.reasoning_actor.send(msg).await??;

    info!("[{}] Reasoning complete: {} axioms", correlation_id, axioms.len());

    // 2. Generate constraints
    let constraint_set = self.generate_constraints(&axioms).await?;

    info!("[{}] Constraints generated: {}", correlation_id, constraint_set.constraints.len());

    // 3. Upload to GPU
    self.upload_to_gpu(constraint_set, correlation_id).await?;

    Ok(stats)
}
```

### 3. GPU Pipeline

```rust
// OntologyConstraintActor::apply_constraints()
async fn apply_constraints(
    &mut self,
    constraint_set: ConstraintSet,
    correlation_id: String,
) -> Result<(), String> {
    info!("[{}] Uploading {} constraints to GPU", correlation_id, constraint_set.constraints.len());

    // 1. Convert to GPU format
    let gpu_data = constraint_set.to_gpu_data();

    // 2. Upload to GPU with retry
    match self.upload_with_retry(&gpu_data, 3).await {
        Ok(_) => info!("[{}] GPU upload successful", correlation_id),
        Err(e) => {
            warn!("[{}] GPU upload failed, using CPU fallback: {}", correlation_id, e);
            self.stats.cpu_fallback_count += 1;
        }
    }

    // 3. Trigger force computation
    self.force_compute_actor.send(ComputeForces).await??;

    Ok(())
}
```

### 4. Client Broadcasting

```rust
// ForceComputeActor::compute_and_broadcast()
async fn compute_and_broadcast(&mut self, correlation_id: String) -> Result<(), String> {
    // 1. Compute forces on GPU
    let positions = self.compute_forces().await?;

    info!("[{}] Forces computed: {} positions", correlation_id, positions.len());

    // 2. Apply backpressure
    if self.client_queue_size > MAX_QUEUE_SIZE {
        warn!("[{}] Backpressure: client queue full, throttling", correlation_id);
        return Ok(());
    }

    // 3. Broadcast to clients
    self.client_manager.send(BroadcastNodePositions {
        positions: serialize_to_binary(&positions),
    }).await?;

    Ok(())
}
```

## Backpressure Management

### Queue-Based Backpressure

```rust
struct PipelineBackpressure {
    // Stage capacities
    reasoning_queue: VecDeque<OntologyModifiedEvent>,
    constraint_queue: VecDeque<ReasoningCompleteEvent>,
    gpu_queue: VecDeque<ConstraintsGeneratedEvent>,

    // Capacity limits
    max_reasoning_queue: usize,    // 10
    max_constraint_queue: usize,   // 5
    max_gpu_queue: usize,          // 3

    // Metrics
    dropped_events: HashMap<String, u64>,
    throttle_events: HashMap<String, u64>,
}

impl PipelineBackpressure {
    async fn enqueue_reasoning(
        &mut self,
        event: OntologyModifiedEvent,
    ) -> Result<(), BackpressureError> {
        if self.reasoning_queue.len() >= self.max_reasoning_queue {
            warn!("Reasoning queue full, dropping event {}", event.correlation_id);
            self.dropped_events.entry("reasoning".to_string())
                .and_modify(|e| *e += 1)
                .or_insert(1);
            return Err(BackpressureError::QueueFull);
        }

        self.reasoning_queue.push_back(event);
        Ok(())
    }
}
```

### Rate Limiting

```rust
struct RateLimiter {
    // Token bucket algorithm
    tokens: f64,
    max_tokens: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl RateLimiter {
    fn try_acquire(&mut self, tokens: f64) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }
}
```

## Error Handling

### Circuit Breaker Pattern

```rust
#[derive(Clone)]
enum CircuitState {
    Closed,            // Normal operation
    Open(Instant),     // Failing, reject requests
    HalfOpen,          // Testing recovery
}

struct CircuitBreaker {
    state: CircuitState,
    failure_count: u32,
    failure_threshold: u32,    // 5 failures
    timeout_duration: Duration, // 30 seconds
    success_threshold: u32,     // 2 successes to close
}

impl CircuitBreaker {
    async fn call<F, T>(&mut self, f: F) -> Result<T, CircuitBreakerError>
    where
        F: Future<Output = Result<T, String>>,
    {
        match &self.state {
            CircuitState::Open(opened_at) => {
                if opened_at.elapsed() > self.timeout_duration {
                    info!("Circuit breaker entering half-open state");
                    self.state = CircuitState::HalfOpen;
                } else {
                    return Err(CircuitBreakerError::CircuitOpen);
                }
            }
            _ => {}
        }

        match f.await {
            Ok(result) => {
                self.on_success();
                Ok(result)
            }
            Err(e) => {
                self.on_failure();
                Err(CircuitBreakerError::CallFailed(e))
            }
        }
    }

    fn on_failure(&mut self) {
        self.failure_count += 1;
        if self.failure_count >= self.failure_threshold {
            warn!("Circuit breaker opening after {} failures", self.failure_count);
            self.state = CircuitState::Open(Instant::now());
        }
    }

    fn on_success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                info!("Circuit breaker closing after successful recovery");
                self.state = CircuitState::Closed;
                self.failure_count = 0;
            }
            _ => {
                self.failure_count = 0;
            }
        }
    }
}
```

### Retry Logic

```rust
async fn with_retry<F, T>(
    f: F,
    max_retries: u32,
    backoff: Duration,
) -> Result<T, String>
where
    F: Fn() -> Pin<Box<dyn Future<Output = Result<T, String>> + Send>>,
{
    let mut attempt = 0;
    let mut delay = backoff;

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                attempt += 1;
                if attempt >= max_retries {
                    return Err(format!("Failed after {} retries: {}", max_retries, e));
                }

                warn!("Retry {}/{}: {}. Waiting {:?}", attempt, max_retries, e, delay);
                tokio::time::sleep(delay).await;

                // Exponential backoff with jitter
                delay = delay * 2 + Duration::from_millis(rand::random::<u64>() % 100);
            }
        }
    }
}
```

## Observability

### Structured Logging

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(self), fields(correlation_id = %correlation_id))]
async fn process_ontology(
    &self,
    ontology_id: i64,
    correlation_id: String,
) -> Result<(), String> {
    info!("Processing ontology {}", ontology_id);

    // Tracing automatically includes correlation_id in all logs
    let result = self.reasoning_actor.send(msg).await;

    match result {
        Ok(axioms) => {
            info!(axiom_count = axioms.len(), "Reasoning complete");
        }
        Err(e) => {
            error!(error = %e, "Reasoning failed");
        }
    }

    Ok(())
}
```

### Metrics Collection

```rust
struct PipelineMetrics {
    // Counters
    total_ontologies_processed: AtomicU64,
    total_reasoning_calls: AtomicU64,
    total_constraints_generated: AtomicU64,
    total_gpu_uploads: AtomicU64,

    // Gauges
    active_pipeline_stages: AtomicU64,
    queue_sizes: HashMap<String, AtomicUsize>,

    // Histograms
    reasoning_duration_ms: Histogram,
    constraint_gen_duration_ms: Histogram,
    gpu_upload_duration_ms: Histogram,
    end_to_end_duration_ms: Histogram,

    // Error rates
    reasoning_errors: AtomicU64,
    gpu_errors: AtomicU64,
    client_broadcast_errors: AtomicU64,
}

impl PipelineMetrics {
    fn record_reasoning_complete(&self, duration_ms: u64, correlation_id: &str) {
        self.total_reasoning_calls.fetch_add(1, Ordering::Relaxed);
        self.reasoning_duration_ms.record(duration_ms);
        info!(
            correlation_id = %correlation_id,
            duration_ms = duration_ms,
            "Reasoning metrics recorded"
        );
    }
}
```

### Correlation IDs

Every pipeline execution gets a unique correlation ID that flows through all stages:

```rust
// Generate at entry point
let correlation_id = Uuid::new_v4().to_string();

// Pass through events
struct OntologyModifiedEvent {
    correlation_id: String,
    // ... other fields
}

// Include in logs
info!("[{}] Starting reasoning", correlation_id);

// Include in metrics
metrics.record_with_correlation(correlation_id, duration);

// Trace across services
span.set_attribute("correlation_id", correlation_id);
```

## Performance Characteristics

### Throughput

- **GitHub Sync**: 50 files/batch, ~100 files/second
- **Reasoning**: 100-1000 axioms/second (cached: 10x faster)
- **Constraint Generation**: 500 constraints/second
- **GPU Upload**: 1000 constraints/upload, <10ms
- **Physics Simulation**: 60 FPS (16.67ms/frame)
- **WebSocket Broadcasting**: 10,000 nodes @ 30 FPS

### Latency

| Stage | P50 | P95 | P99 |
|-------|-----|-----|-----|
| GitHub → Database | 50ms | 150ms | 300ms |
| Reasoning (cold) | 200ms | 500ms | 1s |
| Reasoning (cached) | 10ms | 20ms | 50ms |
| Constraint Generation | 30ms | 80ms | 150ms |
| GPU Upload | 5ms | 15ms | 30ms |
| Physics Iteration | 8ms | 12ms | 16ms |
| End-to-End (cold) | 300ms | 800ms | 1.5s |
| End-to-End (cached) | 60ms | 120ms | 250ms |

### Memory Usage

- **Reasoning Cache**: 100MB - 1GB (configurable)
- **Constraint Buffers**: 50MB - 200MB
- **GPU Memory**: 500MB - 2GB
- **WebSocket Buffers**: 10MB - 50MB per client

## Failure Modes

### 1. GitHub API Rate Limiting
**Symptoms**: 403 responses, sync failures
**Mitigation**:
- SHA1 deduplication (only changed files)
- Exponential backoff
- Manual trigger endpoint for retries

### 2. Reasoning Actor Crash
**Symptoms**: Pipeline stuck, no constraints
**Mitigation**:
- Actix supervisor restarts actor
- Cached results available immediately
- Circuit breaker prevents cascading failures

### 3. GPU Out of Memory
**Symptoms**: CUDA errors, constraint upload fails
**Mitigation**:
- Automatic CPU fallback
- Constraint batching (1000 at a time)
- Memory pool with cleanup

### 4. WebSocket Client Overload
**Symptoms**: High latency, dropped frames
**Mitigation**:
- Rate limiting (30 FPS default)
- Binary protocol (10x smaller)
- Backpressure (skip frames if queue full)

### 5. Database Lock Contention
**Symptoms**: Slow queries, timeouts
**Mitigation**:
- WAL mode enabled
- Read replicas for queries
- Write batching (50 files/transaction)

## Configuration

```toml
[pipeline]
# Backpressure
max_reasoning_queue = 10
max_constraint_queue = 5
max_gpu_queue = 3

# Retry
max_retries = 3
initial_backoff_ms = 100
max_backoff_ms = 5000

# Circuit breaker
failure_threshold = 5
timeout_duration_secs = 30
success_threshold = 2

# Rate limiting
reasoning_rate_limit = 10  # per second
gpu_upload_rate_limit = 5   # per second
client_broadcast_fps = 30

# Caching
reasoning_cache_size_mb = 500
constraint_cache_size_mb = 200

# Timeouts
reasoning_timeout_secs = 60
gpu_upload_timeout_secs = 10
client_broadcast_timeout_secs = 5
```

## Admin Endpoints

### Pipeline Control

```http
POST /api/admin/pipeline/trigger
Content-Type: application/json

{
  "force": true,
  "correlation_id": "manual-trigger-2025"
}
```

Response:
```json
{
  "status": "triggered",
  "correlation_id": "manual-trigger-2025",
  "estimated_duration_ms": 500
}
```

### Pipeline Status

```http
GET /api/admin/pipeline/status
```

Response:
```json
{
  "status": "running",
  "current_stage": "reasoning",
  "queue_sizes": {
    "reasoning": 2,
    "constraints": 0,
    "gpu_upload": 0
  },
  "active_correlation_ids": [
    "abc-123",
    "def-456"
  ],
  "backpressure": {
    "throttled": false,
    "dropped_events": 0
  }
}
```

### Pipeline Pause/Resume

```http
POST /api/admin/pipeline/pause
Content-Type: application/json

{
  "reason": "Maintenance window"
}
```

```http
POST /api/admin/pipeline/resume
```

### Pipeline Metrics

```http
GET /api/admin/pipeline/metrics
```

Response:
```json
{
  "total_ontologies_processed": 1542,
  "total_reasoning_calls": 3084,
  "total_constraints_generated": 15420,
  "total_gpu_uploads": 3084,
  "latencies": {
    "reasoning_p50_ms": 15,
    "reasoning_p95_ms": 45,
    "reasoning_p99_ms": 120,
    "constraint_gen_p50_ms": 25,
    "gpu_upload_p50_ms": 8,
    "end_to_end_p50_ms": 65
  },
  "error_rates": {
    "reasoning_errors": 12,
    "gpu_errors": 3,
    "client_broadcast_errors": 0
  },
  "cache_stats": {
    "reasoning_cache_hit_rate": 0.87,
    "reasoning_cache_size_mb": 342,
    "constraint_cache_size_mb": 128
  }
}
```

## Integration Testing

### End-to-End Test

```rust
#[actix_rt::test]
async fn test_pipeline_end_to_end() {
    // 1. Upload OWL file via GitHub sync
    let owl_content = r#"
        ### OntologyBlock
        - owl:Class: ex:Person
        - owl:Class: ex:Student
        - owl:SubClassOf: ex:Student ex:Person
    "#;

    let sync_result = github_sync.process_file("test.md", owl_content).await;
    assert!(sync_result.is_ok());

    // 2. Wait for pipeline completion (with timeout)
    let correlation_id = sync_result.unwrap();
    let timeout = Duration::from_secs(5);
    let start = Instant::now();

    loop {
        let status = pipeline.get_status(&correlation_id).await?;
        if status.stage == "complete" {
            break;
        }

        if start.elapsed() > timeout {
            panic!("Pipeline timeout");
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // 3. Verify reasoning results
    let axioms = reasoning_actor.send(GetInferredAxioms { ontology_id: 1 }).await??;
    assert!(!axioms.is_empty());

    // 4. Verify GPU constraints
    let constraints = constraint_actor.send(GetConstraintStats).await??;
    assert!(constraints.active_ontology_constraints > 0);

    // 5. Verify client received positions
    let client_data = websocket_client.receive_binary().await?;
    assert!(!client_data.is_empty());
}
```

### Cache Test

```rust
#[actix_rt::test]
async fn test_reasoning_cache_hit() {
    let ontology = create_test_ontology();

    // First call (cache miss)
    let start = Instant::now();
    let result1 = reasoning_actor.send(TriggerReasoning {
        ontology_id: 1,
        ontology: ontology.clone(),
    }).await??;
    let duration1 = start.elapsed();

    // Second call (cache hit)
    let start = Instant::now();
    let result2 = reasoning_actor.send(TriggerReasoning {
        ontology_id: 1,
        ontology: ontology.clone(),
    }).await??;
    let duration2 = start.elapsed();

    // Cache hit should be 10x faster
    assert!(duration2 < duration1 / 10);
    assert_eq!(result1.len(), result2.len());
}
```

### Backpressure Test

```rust
#[actix_rt::test]
async fn test_backpressure_throttling() {
    // Flood pipeline with events
    let mut correlation_ids = Vec::new();

    for i in 0..20 {
        let correlation_id = format!("test-{}", i);
        pipeline.on_ontology_modified(1, ontology.clone(), correlation_id.clone()).await;
        correlation_ids.push(correlation_id);
    }

    // Verify some events were throttled
    let metrics = pipeline.get_metrics().await?;
    assert!(metrics.throttled_events > 0);

    // Verify queue didn't overflow
    assert!(metrics.dropped_events == 0);
}
```

## Deployment Checklist

- [ ] Database migrations applied (`sql/migrations/`)
- [ ] Reasoning cache initialized (`inference_cache.db`)
- [ ] GPU drivers installed (CUDA 12.x)
- [ ] Environment variables set (see `config/pipeline.toml`)
- [ ] Circuit breaker thresholds configured
- [ ] Rate limits configured for production load
- [ ] Monitoring dashboards deployed (Grafana)
- [ ] Alert rules configured (Prometheus)
- [ ] Integration tests passing
- [ ] Load tests passing (10,000 nodes)
- [ ] Operator runbook reviewed

## References

- [CustomReasoner Implementation](../../src/reasoning/custom_reasoner.rs)
- [OntologyPipelineService](../../src/services/ontology_pipeline_service.rs)
- [ReasoningActor](../../src/reasoning/reasoning_actor.rs)
- [OntologyConstraintActor](../../src/actors/gpu/ontology_constraint_actor.rs)
- [GitHubSyncService](../../src/services/github_sync_service.rs)
- [WebSocket Protocol](../protocols/WEBSOCKET_PROTOCOL.md)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-03 | Initial pipeline integration architecture |
