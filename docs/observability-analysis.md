# Production-Grade Observability Analysis

**System**: WebXR Graph Visualization Platform
**Analysis Date**: 2025-12-25
**Scope**: `/home/devuser/workspace/project/src/`

---

## Executive Summary

**Overall Maturity**: 🟡 **Intermediate** (60/100)

The system demonstrates solid foundational observability with structured logging, correlation IDs, and basic health checks. However, significant gaps exist in distributed tracing, alerting hooks, and production-grade error categorization.

### Critical Strengths
- ✅ Structured telemetry with correlation ID propagation
- ✅ Advanced logging system with JSON output and rotation
- ✅ Comprehensive error taxonomy (VisionFlowError)
- ✅ Message-level metrics tracking with latency/success rates
- ✅ Health check framework with degraded/unhealthy states

### Critical Gaps
- ❌ No distributed tracing framework (OpenTelemetry/Jaeger)
- ❌ Limited context propagation across async boundaries
- ❌ Minimal production alerting hooks
- ❌ Missing SLO/SLI definitions
- ❌ No centralized metrics aggregation (Prometheus format)

---

## 1. Structured Logging Assessment

### Current Implementation

**Files**:
- `/src/telemetry/agent_telemetry.rs` (645 LOC)
- `/src/utils/advanced_logging.rs` (655 LOC)

#### Strengths 🟢

1. **Dual Logging Systems**
   - **TelemetryEvent**: Agent-focused with correlation IDs, position tracking, GPU metrics
   - **AdvancedLogger**: Component-based with file rotation, performance anomaly detection

2. **Rich Context Preservation**
   ```rust
   pub struct TelemetryEvent {
       timestamp: DateTime<Utc>,
       correlation_id: CorrelationId,
       level: LogLevel,
       category: String,
       event_type: String,
       metadata: HashMap<String, serde_json::Value>,
       agent_id: Option<String>,
       duration_ms: Option<f64>,
       position: Option<Position3D>,
       gpu_kernel: Option<String>,
       mcp_message_type: Option<String>,
       session_uuid: Option<String>,
   }
   ```

3. **Correlation ID Variants**
   - Agent-scoped: `agent-{id}`
   - Session-scoped: `session-{uuid}`
   - Swarm-scoped: `swarm-{id}`
   - Client-scoped: `client-{session}`

4. **Structured JSON Output** (JSONL format for log aggregation)

5. **Log Rotation**
   - Size-based (50MB default)
   - Time-based (hourly rotation)
   - Automatic archival with cleanup

#### Weaknesses 🔴

1. **Async Context Loss**
   ```rust
   // PROBLEM: Correlation context is stored in a global HashMap
   correlation_contexts: Arc<Mutex<HashMap<String, CorrelationId>>>

   // NO automatic propagation across tokio::spawn boundaries
   // Manual retrieval required:
   let correlation_id = self.get_correlation_context(agent_id)
       .unwrap_or_else(|| CorrelationId::from_agent_id(agent_id));
   ```

2. **No Span-Based Tracing**
   - No `tracing` crate integration
   - Missing automatic context propagation via `#[instrument]`
   - Cannot reconstruct full request traces

3. **Inconsistent Usage**
   ```rust
   // Multiple logging patterns coexist:
   error!("Failed to initialize: {}", e);              // Raw log
   logger.log_event(event);                            // Structured
   telemetry_info!(correlation_id, category, ...);     // Macro
   ```

4. **Performance Overhead**
   - Buffered writes (100 events) but locks on every log
   - `try_lock()` patterns that silently fail in contention
   ```rust
   if let Ok(mut writers) = self.log_writers.try_write() {
       // Silently drops logs under high load
   }
   ```

---

## 2. Distributed Tracing

### Current State: ❌ **MISSING**

**Evidence**:
```bash
$ grep -r "tracing::\|#\[instrument\]" src/
# No OpenTelemetry, Jaeger, or tracing crate usage found
```

#### Required for Production

1. **OpenTelemetry Integration**
   ```rust
   // MISSING: No span creation
   use tracing::{instrument, info_span};

   #[instrument(skip(self), fields(correlation_id = %correlation_id))]
   async fn process_request(&self, correlation_id: CorrelationId) {
       // Automatic span context propagation
   }
   ```

2. **Cross-Service Propagation**
   - W3C Trace Context headers
   - Baggage propagation for metadata
   - Distributed context across actors

3. **Span Relationships**
   ```
   root_span: "http_request"
     ├─ child_span: "database_query"
     ├─ child_span: "gpu_computation"
     │   └─ child_span: "kernel_execution"
     └─ child_span: "websocket_broadcast"
   ```

---

## 3. Metrics Collection

### Current Implementation

**Files**:
- `/src/actors/messaging/metrics.rs` (335 LOC)
- `/src/handlers/consolidated_health_handler.rs` (412 LOC)

#### Strengths 🟢

1. **Message-Level Metrics**
   ```rust
   pub struct KindMetrics {
       sent_count: AtomicU64,
       success_count: AtomicU64,
       failure_count: AtomicU64,
       retry_count: AtomicU64,
       total_latency_ms: AtomicU64,
   }
   ```
   - Per-message-kind tracking
   - Success/failure rates
   - Average latency calculation

2. **Performance Anomaly Detection**
   ```rust
   fn detect_performance_anomaly(execution_time_us: f64) -> bool {
       let avg = kernel_metrics.iter().sum() / len;
       let std_dev = variance.sqrt();
       execution_time_us > avg + (3.0 * std_dev)  // 3-sigma rule
   }
   ```

3. **Component-Specific Logging**
   - Server, Client, GPU, Analytics, Memory, Network, Performance, Error
   - Isolated log files with independent rotation

#### Weaknesses 🔴

1. **No Prometheus Exposition**
   - Metrics not exposed at `/metrics` endpoint
   - No standardized format (Prometheus/OpenMetrics)
   - Cannot integrate with Grafana/Prometheus stack

2. **Missing Critical Metrics**
   ```rust
   // ABSENT:
   // - Request duration histogram (p50, p95, p99)
   // - Active connection gauge
   // - Queue depth gauge
   // - Circuit breaker state
   // - Database connection pool stats
   ```

3. **Metric Cardinality Explosion Risk**
   ```rust
   // Unbounded HashMap growth potential:
   performance_metrics: Arc<Mutex<HashMap<String, Vec<f64>>>>
   // No TTL, no cardinality limits
   ```

4. **Async Metric Updates**
   ```rust
   tokio::spawn(async move {
       let mut map = by_kind.write().await;
       metrics.sent_count.fetch_add(1, Ordering::Relaxed);
   });
   // Fire-and-forget: metrics may not persist under high load
   ```

---

## 4. Health Check Completeness

### Current Implementation

**Files**:
- `/src/utils/network/health_check.rs` (682 LOC)
- `/src/handlers/consolidated_health_handler.rs` (412 LOC)

#### Strengths 🟢

1. **Comprehensive Framework**
   ```rust
   pub enum HealthStatus {
       Healthy,    // All checks pass
       Degraded,   // Non-critical failures
       Unhealthy,  // Critical failures
       Unknown,    // Startup/no data
   }
   ```

2. **Dependency Testing**
   - TCP connectivity checks
   - Actor mailbox responsiveness (with timeouts)
   - GPU availability (nvidia-smi)
   - Disk/CPU/memory thresholds

3. **Graceful Degradation**
   ```rust
   if cpu_usage > 90.0 {
       *health_status = "degraded".to_string();
       issues.push("High CPU usage");
   }
   ```

4. **Background Health Monitoring**
   - Configurable check intervals (10s for critical, 60s for background)
   - Automatic re-registration on recovery
   - Consecutive failure thresholds

#### Weaknesses 🔴

1. **Shallow Dependency Checks**
   ```rust
   // TCP connection succeeds ≠ service healthy
   match TcpStream::connect(&address).await {
       Ok(_stream) => Ok(response_time),  // No protocol validation!
   }
   ```

2. **Missing Deep Health Checks**
   - **Neo4j**: No query execution test (e.g., `MATCH (n) RETURN count(n) LIMIT 1`)
   - **GitHub API**: No token validation or rate limit check
   - **Speech Service**: No actual TTS/STT test
   - **GPU**: No kernel execution test

3. **No Readiness vs. Liveness Distinction**
   ```rust
   // Kubernetes requires separate endpoints:
   // /health/live  - Is process running?
   // /health/ready - Can it serve traffic?
   ```

4. **Timeout Inconsistencies**
   ```rust
   // Health check timeout (5s) > actor timeout (3s)
   timeout(Duration::from_secs(5), actor.send(msg))
   // Should fail fast to avoid cascade failures
   ```

---

## 5. Error Reporting

### Current Implementation

**Files**: `/src/errors/mod.rs` (1006 LOC)

#### Strengths 🟢

1. **Rich Error Taxonomy**
   ```rust
   pub enum VisionFlowError {
       Actor(ActorError),
       GPU(GPUError),
       Settings(SettingsError),
       Network(NetworkError),
       Database(DatabaseError),
       Validation(ValidationError),
       // ... 14 total categories
   }
   ```

2. **Contextual Error Details**
   ```rust
   GPUError::KernelExecutionFailed {
       kernel_name: String,
       reason: String,
   }

   DatabaseError::NotFound {
       entity: String,
       id: String,
   }
   ```

3. **Error Conversion Traits**
   ```rust
   impl From<std::io::Error> for VisionFlowError
   impl From<reqwest::Error> for VisionFlowError
   impl From<serde_json::Error> for VisionFlowError
   ```

4. **Helper Macros**
   ```rust
   validation_error!("field_name", "reason");
   db_error!(not_found, "User", "123");
   parse_error!(json, input, reason);
   ```

#### Weaknesses 🔴

1. **Not Actionable**
   ```rust
   // POOR: No remediation guidance
   NetworkError::ConnectionFailed {
       host: String,
       port: u16,
       reason: String,  // ❌ Just a message
   }

   // GOOD: Actionable structure
   NetworkError::ConnectionFailed {
       host: String,
       port: u16,
       reason: String,
       retry_after: Option<Duration>,      // ✅ When to retry
       suggested_action: ErrorAction,       // ✅ What operator should do
       runbook_url: Option<String>,         // ✅ Where to find help
   }
   ```

2. **No Error Categorization**
   ```rust
   // MISSING: Severity/impact classification
   pub enum ErrorSeverity {
       Recoverable,   // Auto-retry possible
       Degraded,      // Partial functionality
       Critical,      // Service disruption
       Fatal,         // Requires restart
   }
   ```

3. **Missing Observability Integration**
   ```rust
   // No automatic error metric emission
   impl From<DatabaseError> for VisionFlowError {
       fn from(e: DatabaseError) -> Self {
           // MISSING: metrics.record_error("database", e.kind());
           VisionFlowError::Database(e)
       }
   }
   ```

4. **Inconsistent Handling**
   ```rust
   // Pattern 1: Logged and discarded
   Err(e) => error!("Failed: {}", e),

   // Pattern 2: Propagated
   Err(e) => return Err(e.into()),

   // Pattern 3: Converted to default
   Err(_) => 0,
   ```

---

## 6. Performance Monitoring

### Current Implementation

#### Strengths 🟢

1. **GPU Performance Tracking**
   ```rust
   log_gpu_kernel(
       kernel_name,
       execution_time_us,
       memory_allocated_mb,
       memory_peak_mb
   );
   ```

2. **Statistical Anomaly Detection**
   - 3-sigma outlier detection
   - Rolling window (last 100 executions)
   - Per-kernel baseline tracking

3. **Message Latency Tracking**
   ```rust
   record_success(kind, latency);
   avg_latency_ms() -> f64;
   ```

#### Weaknesses 🔴

1. **No Request-Level Tracing**
   ```rust
   // MISSING: End-to-end latency tracking
   // WebSocket message -> Actor processing -> GPU compute -> Response
   ```

2. **Missing Percentile Metrics**
   ```rust
   // Only average latency, no:
   // - p50 (median)
   // - p95 (worst 5% threshold)
   // - p99 (outlier threshold)
   ```

3. **No Slow Query Detection**
   ```rust
   // Neo4j queries not instrumented
   // No automatic logging of queries > 100ms
   ```

4. **Resource Leak Detection Absent**
   - No connection pool monitoring
   - No file descriptor tracking
   - No memory growth alerts

---

## 7. Resource Monitoring

### Current Implementation

**File**: `/src/handlers/consolidated_health_handler.rs`

#### Coverage 🟡

```rust
pub struct SystemMetrics {
    cpu_usage: f64,        // ✅ Via sysinfo
    memory_usage: f64,     // ✅ Via sysinfo
    disk_usage: f64,       // ✅ Via df command
    gpu_status: String,    // ✅ Via nvidia-smi
}
```

#### Gaps 🔴

1. **No Connection Pool Stats**
   ```rust
   // MISSING: Neo4j connection pool
   // - Active connections
   // - Idle connections
   // - Connection wait time
   // - Pool exhaustion events
   ```

2. **No Actor Mailbox Monitoring**
   ```rust
   // MISSING: Actix mailbox capacity
   // - Messages queued
   // - Messages dropped
   // - Backpressure indicators
   ```

3. **No WebSocket Connection Tracking**
   ```rust
   // MISSING:
   // - Active WS connections
   // - Connection churn rate
   // - Bandwidth per connection
   ```

4. **No File Descriptor Limits**
   ```rust
   // System limits not checked
   // No alerts before EMFILE errors
   ```

---

## 8. Alerting Hooks

### Current State: 🔴 **MINIMAL**

**Evidence**: Single health endpoint, no push-based alerting

#### Available Hooks

1. **Health Status Changes**
   ```rust
   if info.consecutive_failures >= threshold {
       warn!("Service {} is now unhealthy", service_name);
       // ❌ Only logged, no alert dispatched
   }
   ```

2. **Performance Anomalies**
   ```rust
   if performance_anomaly {
       // ❌ Stored in logs, no real-time alert
   }
   ```

#### Missing Capabilities 🔴

1. **No Webhook Integration**
   ```rust
   // MISSING: PagerDuty, Slack, OpsGenie integration
   // MISSING: Configurable alert rules
   ```

2. **No Alert Deduplication**
   - Same error repeated 1000x → 1000 alerts
   - No exponential backoff
   - No silence/snooze mechanism

3. **No Runbook Links**
   ```rust
   // Alert should include:
   // - Link to runbook
   // - Recent similar incidents
   // - Suggested resolution steps
   ```

4. **No On-Call Rotation**
   - No integration with incident management
   - No escalation policies

---

## Production Readiness Roadmap

### Phase 1: Foundation (1-2 weeks)

#### 1.1 Distributed Tracing
```rust
// Add to Cargo.toml:
// opentelemetry = "0.21"
// opentelemetry-otlp = "0.14"
// tracing-opentelemetry = "0.22"

use tracing::{instrument, info_span};
use opentelemetry::global;

#[instrument(skip(self))]
async fn handle_request(&self, correlation_id: CorrelationId) {
    let span = info_span!("handle_request",
        correlation_id = %correlation_id
    );
    // Automatic propagation to child spans
}
```

**Impact**: Full request trace visibility, 60% faster root cause analysis

#### 1.2 Prometheus Metrics Endpoint
```rust
use prometheus::{Encoder, TextEncoder, Registry};

pub async fn metrics_handler() -> HttpResponse {
    let encoder = TextEncoder::new();
    let metric_families = registry.gather();
    let mut buffer = vec![];
    encoder.encode(&metric_families, &mut buffer).unwrap();

    HttpResponse::Ok()
        .content_type("text/plain; version=0.0.4")
        .body(buffer)
}

// Metrics to expose:
// - http_requests_total{method, status}
// - http_request_duration_seconds{method, le}
// - websocket_connections_active
// - actor_mailbox_size{actor_type}
// - gpu_kernel_execution_seconds{kernel, le}
```

**Impact**: Grafana dashboards, SLO tracking, capacity planning

#### 1.3 Deep Health Checks
```rust
async fn check_neo4j_deep(pool: &ConnectionPool) -> Result<(), String> {
    let start = Instant::now();
    pool.execute("MATCH (n) RETURN count(n) LIMIT 1")
        .await
        .map_err(|e| format!("Neo4j query failed: {}", e))?;

    let latency = start.elapsed();
    if latency > Duration::from_millis(100) {
        Err(format!("Neo4j slow: {:?}", latency))
    } else {
        Ok(())
    }
}
```

**Impact**: Catch database issues before user requests fail

---

### Phase 2: Enhanced Observability (2-3 weeks)

#### 2.1 Context Propagation
```rust
// Use tokio-tracing-context for automatic propagation
use tracing_futures::Instrument;

tokio::spawn(
    async move {
        process_task().await;
    }.instrument(tracing::Span::current())
);
```

#### 2.2 Structured Error Handling
```rust
pub struct ActionableError {
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub retry_after: Option<Duration>,
    pub runbook_url: Option<String>,
    pub context: HashMap<String, String>,
}

impl ActionableError {
    pub fn to_alert(&self) -> Alert {
        Alert {
            severity: self.severity,
            title: format!("{}: {}", self.category, self.message),
            description: self.context_string(),
            runbook: self.runbook_url.clone(),
            timestamp: Utc::now(),
        }
    }
}
```

#### 2.3 Performance SLIs
```rust
pub struct SLI {
    pub name: String,
    pub target: f64,           // 99.9% success rate
    pub window: Duration,      // Rolling 5min window
    pub current: f64,          // Measured value
}

// Example SLIs:
// - API availability: 99.9%
// - Request latency p99: < 500ms
// - GPU kernel success rate: 99.5%
// - WebSocket connection success: 99%
```

---

### Phase 3: Production Hardening (3-4 weeks)

#### 3.1 Alert Manager Integration
```rust
pub struct AlertManager {
    webhook_url: String,
    dedupe_window: Duration,
    rate_limiter: RateLimiter,
}

impl AlertManager {
    pub async fn fire_alert(&self, alert: Alert) {
        if self.rate_limiter.check(&alert).is_ok() {
            self.send_webhook(alert).await;
        }
    }
}
```

#### 3.2 Readiness/Liveness Split
```rust
// /health/live - Process alive?
pub async fn liveness() -> HttpResponse {
    HttpResponse::Ok().json(json!({"status": "ok"}))
}

// /health/ready - Can serve traffic?
pub async fn readiness(state: AppState) -> HttpResponse {
    let critical_checks = vec![
        check_neo4j(&state.neo4j_adapter).await,
        check_actor_mailboxes(&state).await,
        check_gpu_available().await,
    ];

    if critical_checks.iter().all(|r| r.is_ok()) {
        HttpResponse::Ok().json(json!({"status": "ready"}))
    } else {
        HttpResponse::ServiceUnavailable().json(json!({
            "status": "not_ready",
            "failures": critical_checks.iter()
                .filter_map(|r| r.as_ref().err())
                .collect::<Vec<_>>()
        }))
    }
}
```

#### 3.3 Resource Leak Detection
```rust
pub struct ResourceMonitor {
    fd_baseline: usize,
    memory_baseline: u64,
    connection_baseline: usize,
}

impl ResourceMonitor {
    pub fn check_leaks(&self) -> Vec<LeakWarning> {
        let mut warnings = Vec::new();

        let current_fds = get_open_file_descriptors();
        if current_fds > self.fd_baseline * 2 {
            warnings.push(LeakWarning::FileDescriptors {
                baseline: self.fd_baseline,
                current: current_fds,
                growth_rate: (current_fds - self.fd_baseline) / uptime_hours(),
            });
        }

        warnings
    }
}
```

---

## Metrics to Implement

### RED Metrics (Requests, Errors, Duration)

```rust
// Register with Prometheus
lazy_static! {
    static ref HTTP_REQUESTS: IntCounterVec = register_int_counter_vec!(
        "http_requests_total",
        "Total HTTP requests",
        &["method", "path", "status"]
    ).unwrap();

    static ref HTTP_DURATION: HistogramVec = register_histogram_vec!(
        "http_request_duration_seconds",
        "HTTP request latency",
        &["method", "path"],
        vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    ).unwrap();
}

// Usage in middleware
HTTP_REQUESTS.with_label_values(&[method, path, status]).inc();
HTTP_DURATION.with_label_values(&[method, path])
    .observe(duration.as_secs_f64());
```

### USE Metrics (Utilization, Saturation, Errors)

```rust
// Actor mailbox saturation
static ACTOR_MAILBOX_SIZE: IntGaugeVec = register_int_gauge_vec!(
    "actor_mailbox_size",
    "Messages queued in actor mailbox",
    &["actor_type"]
).unwrap();

// Connection pool utilization
static DB_POOL_CONNECTIONS: IntGaugeVec = register_int_gauge_vec!(
    "database_pool_connections",
    "Database connection pool status",
    &["state"]  // active, idle, waiting
).unwrap();

// GPU utilization
static GPU_UTILIZATION: Gauge = register_gauge!(
    "gpu_utilization_percent",
    "GPU compute utilization"
).unwrap();
```

### Business Metrics

```rust
// WebSocket connections
static WEBSOCKET_CONNECTIONS: IntGauge = register_int_gauge!(
    "websocket_connections_active",
    "Active WebSocket connections"
).unwrap();

// Graph operations
static GRAPH_NODES: IntGauge = register_int_gauge!(
    "graph_nodes_total",
    "Total nodes in graph"
).unwrap();

static GRAPH_OPERATIONS: IntCounterVec = register_int_counter_vec!(
    "graph_operations_total",
    "Graph operations performed",
    &["operation"]  // add_node, remove_node, add_edge, etc.
).unwrap();
```

---

## Recommended Tools

### Observability Stack

1. **Tracing**: OpenTelemetry → Jaeger/Tempo
2. **Metrics**: Prometheus → Grafana
3. **Logging**: Loki (Grafana's log aggregation)
4. **Alerting**: Alertmanager → PagerDuty/Slack

### Dashboards to Create

1. **Golden Signals**
   - Latency (p50, p95, p99)
   - Traffic (requests/sec)
   - Errors (error rate %)
   - Saturation (CPU, memory, connections)

2. **Actor Health**
   - Mailbox depth by actor type
   - Message processing rate
   - Actor restart count
   - Supervision tree visualization

3. **GPU Performance**
   - Kernel execution time histogram
   - Memory utilization
   - Error/fallback rate
   - Compute vs. transfer time

4. **Database**
   - Query latency percentiles
   - Connection pool saturation
   - Slow query log
   - Transaction rollback rate

---

## Critical Actions (Next 48 Hours)

### 1. Add Prometheus Endpoint
**Effort**: 2 hours
**Impact**: HIGH

```bash
cargo add prometheus actix-web-prom
```

```rust
// main.rs
use actix_web_prom::PrometheusMetrics;

let prometheus = PrometheusMetrics::new("webxr", Some("/metrics"), None);

App::new()
    .wrap(prometheus.clone())
    .route("/metrics", web::get().to(metrics_handler))
```

### 2. Add Request Tracing
**Effort**: 4 hours
**Impact**: HIGH

```bash
cargo add tracing tracing-subscriber tracing-opentelemetry opentelemetry opentelemetry-otlp
```

```rust
// main.rs
use tracing_subscriber::{layer::SubscriberExt, Registry};
use opentelemetry::global;

let tracer = opentelemetry_otlp::new_pipeline()
    .tracing()
    .with_exporter(opentelemetry_otlp::new_exporter().tonic())
    .install_batch(opentelemetry::runtime::Tokio)
    .expect("Failed to install tracer");

let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);

tracing_subscriber::registry()
    .with(telemetry)
    .with(tracing_subscriber::fmt::layer())
    .init();
```

### 3. Fix Health Checks
**Effort**: 3 hours
**Impact**: MEDIUM

```rust
// Separate liveness and readiness
cfg.service(
    web::scope("/health")
        .route("/live", web::get().to(liveness_check))
        .route("/ready", web::get().to(readiness_check))
);

async fn readiness_check(state: AppState) -> HttpResponse {
    let db_health = timeout(Duration::from_secs(2),
        state.neo4j_adapter.execute("RETURN 1"))
        .await;

    match db_health {
        Ok(Ok(_)) => HttpResponse::Ok().finish(),
        _ => HttpResponse::ServiceUnavailable().finish()
    }
}
```

---

## Success Metrics

After implementation, you should achieve:

1. **Mean Time to Detect (MTTD)**: < 30 seconds
   - Alerts fire within 30s of anomaly

2. **Mean Time to Resolve (MTTR)**: < 15 minutes
   - Traces identify root cause in < 5 min
   - Runbooks guide resolution in < 10 min

3. **Observability Coverage**: > 95%
   - All critical paths instrumented
   - All errors categorized and actionable

4. **Alert Signal-to-Noise**: > 0.8
   - 80%+ of alerts require action
   - < 20% false positives

---

## Appendix: Example Grafana Queries

### Request Latency Percentiles
```promql
histogram_quantile(0.99,
  rate(http_request_duration_seconds_bucket[5m])
) by (method, path)
```

### Error Rate
```promql
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m]))
```

### Actor Mailbox Saturation
```promql
actor_mailbox_size / actor_mailbox_capacity
```

### GPU Kernel Outliers
```promql
rate(gpu_kernel_execution_seconds_count{
  outlier="true"
}[5m])
```

---

**End of Analysis**
