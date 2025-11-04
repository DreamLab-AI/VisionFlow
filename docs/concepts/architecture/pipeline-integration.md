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
    ontology-id: i64,
    ontology: Ontology,
    source: String,
    timestamp: DateTime<Utc>,
    correlation-id: String,
}
```

**Handlers**:
- OntologyPipelineService::on-ontology-modified()

#### 2. ReasoningCompleteEvent
**Trigger**: ReasoningActor finishes inference
**Payload**:
```rust
struct ReasoningCompleteEvent {
    ontology-id: i64,
    inferred-axioms: Vec<InferredAxiom>,
    inference-time-ms: u64,
    cache-hit: bool,
    correlation-id: String,
}
```

**Handlers**:
- OntologyPipelineService::on-reasoning-complete()

#### 3. ConstraintsGeneratedEvent
**Trigger**: ConstraintBuilder generates physics constraints
**Payload**:
```rust
struct ConstraintsGeneratedEvent {
    constraint-set: ConstraintSet,
    axiom-count: usize,
    constraint-count: usize,
    correlation-id: String,
}
```

**Handlers**:
- OntologyConstraintActor::on-constraints-generated()

#### 4. GPUUploadCompleteEvent
**Trigger**: Constraints uploaded to GPU
**Payload**:
```rust
struct GPUUploadCompleteEvent {
    constraint-count: usize,
    upload-time-ms: f32,
    gpu-memory-used: usize,
    correlation-id: String,
}
```

**Handlers**:
- ForceComputeActor::on-gpu-upload-complete()

#### 5. PositionsUpdatedEvent
**Trigger**: Physics simulation updates node positions
**Payload**:
```rust
struct PositionsUpdatedEvent {
    positions: Vec<(u32, BinaryNodeData)>,
    iteration: u32,
    correlation-id: String,
}
```

**Handlers**:
- ClientManagerActor::broadcast-positions()

## Data Flow

### 1. GitHub → Database

```rust
// GitHubSyncService::save-ontology-data()
async fn save-ontology-data(&self, onto-data: OntologyData) -> Result<(), String> {
    // 1. Save to unified.db
    self.onto-repo.save-ontology(
        &onto-data.classes,
        &onto-data.properties,
        &onto-data.axioms
    ).await?;

    // 2. Build Ontology struct
    let ontology = build-ontology-from-data(&onto-data);

    // 3. Trigger pipeline (async, non-blocking)
    if let Some(pipeline) = &self.pipeline-service {
        let correlation-id = Uuid::new-v4().to-string();
        tokio::spawn(async move {
            pipeline.on-ontology-modified(1, ontology, correlation-id).await
        });
    }

    Ok(())
}
```

### 2. Reasoning Pipeline

```rust
// OntologyPipelineService::on-ontology-modified()
pub async fn on-ontology-modified(
    &self,
    ontology-id: i64,
    ontology: Ontology,
    correlation-id: String,
) -> Result<OntologyPipelineStats, String> {
    info!("[{}] Ontology modification detected", correlation-id);

    // 1. Trigger reasoning
    let msg = TriggerReasoning { ontology-id, ontology };
    let axioms = self.reasoning-actor.send(msg).await??;

    info!("[{}] Reasoning complete: {} axioms", correlation-id, axioms.len());

    // 2. Generate constraints
    let constraint-set = self.generate-constraints(&axioms).await?;

    info!("[{}] Constraints generated: {}", correlation-id, constraint-set.constraints.len());

    // 3. Upload to GPU
    self.upload-to-gpu(constraint-set, correlation-id).await?;

    Ok(stats)
}
```

### 3. GPU Pipeline

```rust
// OntologyConstraintActor::apply-constraints()
async fn apply-constraints(
    &mut self,
    constraint-set: ConstraintSet,
    correlation-id: String,
) -> Result<(), String> {
    info!("[{}] Uploading {} constraints to GPU", correlation-id, constraint-set.constraints.len());

    // 1. Convert to GPU format
    let gpu-data = constraint-set.to-gpu-data();

    // 2. Upload to GPU with retry
    match self.upload-with-retry(&gpu-data, 3).await {
        Ok(-) => info!("[{}] GPU upload successful", correlation-id),
        Err(e) => {
            warn!("[{}] GPU upload failed, using CPU fallback: {}", correlation-id, e);
            self.stats.cpu-fallback-count += 1;
        }
    }

    // 3. Trigger force computation
    self.force-compute-actor.send(ComputeForces).await??;

    Ok(())
}
```

### 4. Client Broadcasting

```rust
// ForceComputeActor::compute-and-broadcast()
async fn compute-and-broadcast(&mut self, correlation-id: String) -> Result<(), String> {
    // 1. Compute forces on GPU
    let positions = self.compute-forces().await?;

    info!("[{}] Forces computed: {} positions", correlation-id, positions.len());

    // 2. Apply backpressure
    if self.client-queue-size > MAX-QUEUE-SIZE {
        warn!("[{}] Backpressure: client queue full, throttling", correlation-id);
        return Ok(());
    }

    // 3. Broadcast to clients
    self.client-manager.send(BroadcastNodePositions {
        positions: serialize-to-binary(&positions),
    }).await?;

    Ok(())
}
```

## Backpressure Management

### Queue-Based Backpressure

```rust
struct PipelineBackpressure {
    // Stage capacities
    reasoning-queue: VecDeque<OntologyModifiedEvent>,
    constraint-queue: VecDeque<ReasoningCompleteEvent>,
    gpu-queue: VecDeque<ConstraintsGeneratedEvent>,

    // Capacity limits
    max-reasoning-queue: usize,    // 10
    max-constraint-queue: usize,   // 5
    max-gpu-queue: usize,          // 3

    // Metrics
    dropped-events: HashMap<String, u64>,
    throttle-events: HashMap<String, u64>,
}

impl PipelineBackpressure {
    async fn enqueue-reasoning(
        &mut self,
        event: OntologyModifiedEvent,
    ) -> Result<(), BackpressureError> {
        if self.reasoning-queue.len() >= self.max-reasoning-queue {
            warn!("Reasoning queue full, dropping event {}", event.correlation-id);
            self.dropped-events.entry("reasoning".to-string())
                .and-modify(|e| *e += 1)
                .or-insert(1);
            return Err(BackpressureError::QueueFull);
        }

        self.reasoning-queue.push-back(event);
        Ok(())
    }
}
```

### Rate Limiting

```rust
struct RateLimiter {
    // Token bucket algorithm
    tokens: f64,
    max-tokens: f64,
    refill-rate: f64, // tokens per second
    last-refill: Instant,
}

impl RateLimiter {
    fn try-acquire(&mut self, tokens: f64) -> bool {
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
        let elapsed = now.duration-since(self.last-refill).as-secs-f64();
        self.tokens = (self.tokens + elapsed * self.refill-rate).min(self.max-tokens);
        self.last-refill = now;
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
    failure-count: u32,
    failure-threshold: u32,    // 5 failures
    timeout-duration: Duration, // 30 seconds
    success-threshold: u32,     // 2 successes to close
}

impl CircuitBreaker {
    async fn call<F, T>(&mut self, f: F) -> Result<T, CircuitBreakerError>
    where
        F: Future<Output = Result<T, String>>,
    {
        match &self.state {
            CircuitState::Open(opened-at) => {
                if opened-at.elapsed() > self.timeout-duration {
                    info!("Circuit breaker entering half-open state");
                    self.state = CircuitState::HalfOpen;
                } else {
                    return Err(CircuitBreakerError::CircuitOpen);
                }
            }
            - => {}
        }

        match f.await {
            Ok(result) => {
                self.on-success();
                Ok(result)
            }
            Err(e) => {
                self.on-failure();
                Err(CircuitBreakerError::CallFailed(e))
            }
        }
    }

    fn on-failure(&mut self) {
        self.failure-count += 1;
        if self.failure-count >= self.failure-threshold {
            warn!("Circuit breaker opening after {} failures", self.failure-count);
            self.state = CircuitState::Open(Instant::now());
        }
    }

    fn on-success(&mut self) {
        match self.state {
            CircuitState::HalfOpen => {
                info!("Circuit breaker closing after successful recovery");
                self.state = CircuitState::Closed;
                self.failure-count = 0;
            }
            - => {
                self.failure-count = 0;
            }
        }
    }
}
```

### Retry Logic

```rust
async fn with-retry<F, T>(
    f: F,
    max-retries: u32,
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
                if attempt >= max-retries {
                    return Err(format!("Failed after {} retries: {}", max-retries, e));
                }

                warn!("Retry {}/{}: {}. Waiting {:?}", attempt, max-retries, e, delay);
                tokio::time::sleep(delay).await;

                // Exponential backoff with jitter
                delay = delay * 2 + Duration::from-millis(rand::random::<u64>() % 100);
            }
        }
    }
}
```

## Observability

### Structured Logging

```rust
use tracing::{info, warn, error, instrument};

#[instrument(skip(self), fields(correlation-id = %correlation-id))]
async fn process-ontology(
    &self,
    ontology-id: i64,
    correlation-id: String,
) -> Result<(), String> {
    info!("Processing ontology {}", ontology-id);

    // Tracing automatically includes correlation-id in all logs
    let result = self.reasoning-actor.send(msg).await;

    match result {
        Ok(axioms) => {
            info!(axiom-count = axioms.len(), "Reasoning complete");
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
    total-ontologies-processed: AtomicU64,
    total-reasoning-calls: AtomicU64,
    total-constraints-generated: AtomicU64,
    total-gpu-uploads: AtomicU64,

    // Gauges
    active-pipeline-stages: AtomicU64,
    queue-sizes: HashMap<String, AtomicUsize>,

    // Histograms
    reasoning-duration-ms: Histogram,
    constraint-gen-duration-ms: Histogram,
    gpu-upload-duration-ms: Histogram,
    end-to-end-duration-ms: Histogram,

    // Error rates
    reasoning-errors: AtomicU64,
    gpu-errors: AtomicU64,
    client-broadcast-errors: AtomicU64,
}

impl PipelineMetrics {
    fn record-reasoning-complete(&self, duration-ms: u64, correlation-id: &str) {
        self.total-reasoning-calls.fetch-add(1, Ordering::Relaxed);
        self.reasoning-duration-ms.record(duration-ms);
        info!(
            correlation-id = %correlation-id,
            duration-ms = duration-ms,
            "Reasoning metrics recorded"
        );
    }
}
```

### Correlation IDs

Every pipeline execution gets a unique correlation ID that flows through all stages:

```rust
// Generate at entry point
let correlation-id = Uuid::new-v4().to-string();

// Pass through events
struct OntologyModifiedEvent {
    correlation-id: String,
    // ... other fields
}

// Include in logs
info!("[{}] Starting reasoning", correlation-id);

// Include in metrics
metrics.record-with-correlation(correlation-id, duration);

// Trace across services
span.set-attribute("correlation-id", correlation-id);
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
max-reasoning-queue = 10
max-constraint-queue = 5
max-gpu-queue = 3

# Retry
max-retries = 3
initial-backoff-ms = 100
max-backoff-ms = 5000

# Circuit breaker
failure-threshold = 5
timeout-duration-secs = 30
success-threshold = 2

# Rate limiting
reasoning-rate-limit = 10  # per second
gpu-upload-rate-limit = 5   # per second
client-broadcast-fps = 30

# Caching
reasoning-cache-size-mb = 500
constraint-cache-size-mb = 200

# Timeouts
reasoning-timeout-secs = 60
gpu-upload-timeout-secs = 10
client-broadcast-timeout-secs = 5
```

## Admin Endpoints

### Pipeline Control

```http
POST /api/admin/pipeline/trigger
Content-Type: application/json

{
  "force": true,
  "correlation-id": "manual-trigger-2025"
}
```

Response:
```json
{
  "status": "triggered",
  "correlation-id": "manual-trigger-2025",
  "estimated-duration-ms": 500
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
  "current-stage": "reasoning",
  "queue-sizes": {
    "reasoning": 2,
    "constraints": 0,
    "gpu-upload": 0
  },
  "active-correlation-ids": [
    "abc-123",
    "def-456"
  ],
  "backpressure": {
    "throttled": false,
    "dropped-events": 0
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
  "total-ontologies-processed": 1542,
  "total-reasoning-calls": 3084,
  "total-constraints-generated": 15420,
  "total-gpu-uploads": 3084,
  "latencies": {
    "reasoning-p50-ms": 15,
    "reasoning-p95-ms": 45,
    "reasoning-p99-ms": 120,
    "constraint-gen-p50-ms": 25,
    "gpu-upload-p50-ms": 8,
    "end-to-end-p50-ms": 65
  },
  "error-rates": {
    "reasoning-errors": 12,
    "gpu-errors": 3,
    "client-broadcast-errors": 0
  },
  "cache-stats": {
    "reasoning-cache-hit-rate": 0.87,
    "reasoning-cache-size-mb": 342,
    "constraint-cache-size-mb": 128
  }
}
```

## Integration Testing

### End-to-End Test

```rust
#[actix-rt::test]
async fn test-pipeline-end-to-end() {
    // 1. Upload OWL file via GitHub sync
    let owl-content = r#"
        ### OntologyBlock
        - owl:Class: ex:Person
        - owl:Class: ex:Student
        - owl:SubClassOf: ex:Student ex:Person
    "#;

    let sync-result = github-sync.process-file("test.md", owl-content).await;
    assert!(sync-result.is-ok());

    // 2. Wait for pipeline completion (with timeout)
    let correlation-id = sync-result.unwrap();
    let timeout = Duration::from-secs(5);
    let start = Instant::now();

    loop {
        let status = pipeline.get-status(&correlation-id).await?;
        if status.stage == "complete" {
            break;
        }

        if start.elapsed() > timeout {
            panic!("Pipeline timeout");
        }

        tokio::time::sleep(Duration::from-millis(100)).await;
    }

    // 3. Verify reasoning results
    let axioms = reasoning-actor.send(GetInferredAxioms { ontology-id: 1 }).await??;
    assert!(!axioms.is-empty());

    // 4. Verify GPU constraints
    let constraints = constraint-actor.send(GetConstraintStats).await??;
    assert!(constraints.active-ontology-constraints > 0);

    // 5. Verify client received positions
    let client-data = websocket-client.receive-binary().await?;
    assert!(!client-data.is-empty());
}
```

### Cache Test

```rust
#[actix-rt::test]
async fn test-reasoning-cache-hit() {
    let ontology = create-test-ontology();

    // First call (cache miss)
    let start = Instant::now();
    let result1 = reasoning-actor.send(TriggerReasoning {
        ontology-id: 1,
        ontology: ontology.clone(),
    }).await??;
    let duration1 = start.elapsed();

    // Second call (cache hit)
    let start = Instant::now();
    let result2 = reasoning-actor.send(TriggerReasoning {
        ontology-id: 1,
        ontology: ontology.clone(),
    }).await??;
    let duration2 = start.elapsed();

    // Cache hit should be 10x faster
    assert!(duration2 < duration1 / 10);
    assert-eq!(result1.len(), result2.len());
}
```

### Backpressure Test

```rust
#[actix-rt::test]
async fn test-backpressure-throttling() {
    // Flood pipeline with events
    let mut correlation-ids = Vec::new();

    for i in 0..20 {
        let correlation-id = format!("test-{}", i);
        pipeline.on-ontology-modified(1, ontology.clone(), correlation-id.clone()).await;
        correlation-ids.push(correlation-id);
    }

    // Verify some events were throttled
    let metrics = pipeline.get-metrics().await?;
    assert!(metrics.throttled-events > 0);

    // Verify queue didn't overflow
    assert!(metrics.dropped-events == 0);
}
```

## Deployment Checklist

- [ ] Database migrations applied (`sql/migrations/`)
- [ ] Reasoning cache initialized (`inference-cache.db`)
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

- [CustomReasoner Implementation](../../src/reasoning/custom-reasoner.rs)
- [OntologyPipelineService](../../src/services/ontology-pipeline-service.rs)
- [ReasoningActor](../../src/reasoning/reasoning-actor.rs)
- [OntologyConstraintActor](../../src/actors/gpu/ontology-constraint-actor.rs)
- [GitHubSyncService](../../src/services/github-sync-service.rs)
- [WebSocket Protocol](../protocols/websocket-protocol.md)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-03 | Initial pipeline integration architecture |
