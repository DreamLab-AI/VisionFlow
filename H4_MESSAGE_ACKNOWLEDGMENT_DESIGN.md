# H4: Message Acknowledgment Protocol Design

**Date:** 2025-11-05
**Status:** 🔧 DESIGN PHASE
**Priority:** High (Audit Finding)

---

## Problem Statement

### Current State Analysis

After analyzing the VisionFlow actor system, I identified critical message loss scenarios:

**High-Traffic Actors:**
- `ForceComputeActor`: 23 message handlers (GPU physics)
- `PhysicsOrchestratorActor`: 21 message handlers (coordination)
- `Settings handlers`: 10 message sends (configuration)

**Critical Issue:**
Most messages use `addr.do_send(message)` (fire-and-forget) with **zero delivery guarantees**.

### Example Failure Scenarios

```rust
// src/actors/physics_orchestrator_actor.rs:813
// ❌ PROBLEM: Position updates can be silently dropped
if let Some(ref graph_addr) = self.graph_service_addr {
    graph_addr.do_send(UpdateNodePositions {
        positions: node_updates
    });
}

// src/actors/physics_orchestrator_actor.rs:284
// ❌ PROBLEM: GPU initialization never confirmed
gpu_addr.do_send(InitializeGPU {
    graph: Arc::clone(graph_data),
    // ...
});
```

**What can go wrong:**
1. **Actor mailbox full** → Message silently dropped
2. **Actor crashed/stopped** → Message lost
3. **Processing failure** → No retry attempted
4. **Network partition** (distributed actors) → Message never arrives

---

## Design Options

### Option 1: Convert to Ask Pattern (Simple but Limited)

**Approach:** Change `do_send()` to `.send().await` for critical messages

```rust
// Before (fire-and-forget):
gpu_addr.do_send(UpdateGPUGraphData { graph });

// After (ask pattern):
match gpu_addr.send(UpdateGPUGraphData { graph }).await {
    Ok(Ok(())) => { /* success */ },
    Ok(Err(e)) => { /* processing error */ },
    Err(e) => { /* delivery error */ }
}
```

**Pros:**
- Uses existing Actix infrastructure
- Simple implementation
- Built-in timeout handling

**Cons:**
- ❌ Adds latency (blocks on response)
- ❌ Can't batch messages efficiently
- ❌ Doesn't track message history
- ❌ No exponential backoff retry logic

### Option 2: Explicit Acknowledgment Protocol (Recommended)

**Approach:** Layer acknowledgment system on top of existing messaging

```rust
// 1. Tag critical messages with correlation ID
let msg_id = MessageId::new();
tracker.track(msg_id, MessageKind::UpdateGPUGraphData);
gpu_addr.do_send(UpdateGPUGraphData {
    graph,
    correlation_id: Some(msg_id),  // NEW
});

// 2. Actor sends acknowledgment after processing
ctx.address().do_send(MessageAck {
    correlation_id: msg_id,
    status: AckStatus::Success,
});

// 3. Tracker receives ack and clears timeout
// 4. If timeout expires → retry with exponential backoff
```

**Pros:**
- ✅ Keeps `do_send()` performance (non-blocking)
- ✅ Explicit tracking of message status
- ✅ Retry logic with exponential backoff
- ✅ Works with existing code (backward compatible)
- ✅ Detailed monitoring/metrics

**Cons:**
- More complex implementation
- Requires modifying message types

**Decision:** **Option 2 recommended** - matches audit requirement for explicit acknowledgment protocol.

---

## Proposed Architecture

### 1. Core Components

```rust
/// Unique identifier for tracking messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MessageId(uuid::Uuid);

impl MessageId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

/// Status of message acknowledgment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AckStatus {
    Success,
    PartialSuccess { reason: String },
    Failed { error: String },
    Retrying { attempt: u32 },
}

/// Generic acknowledgment message
#[derive(Debug, Clone, Message)]
#[rtype(result = "()")]
pub struct MessageAck {
    pub correlation_id: MessageId,
    pub status: AckStatus,
    pub timestamp: Instant,
    pub metadata: HashMap<String, String>,
}

/// Tracks outstanding messages with timeouts
pub struct MessageTracker {
    pending: Arc<RwLock<HashMap<MessageId, PendingMessage>>>,
    retry_tx: mpsc::UnboundedSender<RetryRequest>,
    metrics: MessageMetrics,
}

struct PendingMessage {
    id: MessageId,
    kind: MessageKind,
    sent_at: Instant,
    timeout: Duration,
    retry_count: u32,
    max_retries: u32,
    payload: Box<dyn Any + Send>,
}

#[derive(Debug, Clone, Copy)]
pub enum MessageKind {
    UpdateGPUGraphData,
    UploadConstraintsToGPU,
    ComputeForces,
    InitializeGPU,
    UpdateNodePositions,
    // ... more critical messages
}
```

### 2. Retry Strategy

**Exponential Backoff:**
```rust
impl MessageTracker {
    fn calculate_retry_delay(attempt: u32) -> Duration {
        let base_delay = Duration::from_millis(100);
        let max_delay = Duration::from_secs(30);

        let delay = base_delay * 2u32.pow(attempt);
        delay.min(max_delay)  // Cap at 30 seconds
    }

    async fn retry_message(&self, msg: PendingMessage) {
        if msg.retry_count >= msg.max_retries {
            error!("Message {} exhausted retries", msg.id);
            self.metrics.record_failure(msg.kind);
            return;
        }

        let delay = Self::calculate_retry_delay(msg.retry_count);
        tokio::time::sleep(delay).await;

        // Re-send message
        info!("Retrying message {} (attempt {})", msg.id, msg.retry_count + 1);
        // ... resend logic
    }
}
```

**Retry Schedule:**
- Attempt 1: 100ms delay
- Attempt 2: 200ms delay
- Attempt 3: 400ms delay
- Attempt 4: 800ms delay
- Attempt 5: 1.6s delay
- Attempt 6+: 30s delay (capped)

### 3. Integration Pattern

**Before (no acknowledgment):**
```rust
// src/actors/physics_orchestrator_actor.rs:813
if let Some(ref gpu_addr) = self.gpu_compute_addr {
    gpu_addr.do_send(UpdateGPUGraphData {
        graph: Arc::clone(graph_data),
    });
}
```

**After (with acknowledgment):**
```rust
if let Some(ref gpu_addr) = self.gpu_compute_addr {
    let msg_id = MessageId::new();

    // Track message
    self.message_tracker.track(
        msg_id,
        MessageKind::UpdateGPUGraphData,
        Duration::from_secs(5),  // 5s timeout
        3,  // max 3 retries
    );

    // Send with correlation ID
    gpu_addr.do_send(UpdateGPUGraphData {
        graph: Arc::clone(graph_data),
        correlation_id: Some(msg_id),
    });
}

// In UpdateGPUGraphData handler:
impl Handler<UpdateGPUGraphData> for ForceComputeActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGPUGraphData, ctx: &mut Self::Context) -> Self::Result {
        info!("UpdateGPUGraphData received");

        // Process message...
        self.gpu_state.num_nodes = msg.graph.nodes.len() as u32;

        // Send acknowledgment
        if let Some(correlation_id) = msg.correlation_id {
            if let Some(ref orchestrator_addr) = self.physics_orchestrator_addr {
                orchestrator_addr.do_send(MessageAck {
                    correlation_id,
                    status: AckStatus::Success,
                    timestamp: Instant::now(),
                    metadata: HashMap::new(),
                });
            }
        }

        Ok(())
    }
}
```

---

## Critical Messages Identified

### ForceComputeActor (GPU Operations)

1. **`ComputeForces`**
   - **Risk:** GPU computation failure not detected
   - **Impact:** Physics stops updating silently
   - **Timeout:** 5 seconds
   - **Retries:** 3 attempts

2. **`UpdateGPUGraphData`**
   - **Risk:** Graph data out of sync
   - **Impact:** Rendering shows stale positions
   - **Timeout:** 2 seconds
   - **Retries:** 5 attempts (critical for consistency)

3. **`UploadConstraintsToGPU`**
   - **Risk:** Constraints not applied
   - **Impact:** Physics behaves incorrectly
   - **Timeout:** 3 seconds
   - **Retries:** 3 attempts

4. **`SetSharedGPUContext`**
   - **Risk:** GPU never initializes
   - **Impact:** System unusable
   - **Timeout:** 10 seconds
   - **Retries:** 5 attempts (critical)

### PhysicsOrchestratorActor (Coordination)

1. **`UpdateNodePositions`**
   - **Risk:** Position updates lost
   - **Impact:** Clients see frozen graph
   - **Timeout:** 1 second
   - **Retries:** 2 attempts (high frequency message)

2. **`InitializeGPU`**
   - **Risk:** Initialization never completes
   - **Impact:** Physics never starts
   - **Timeout:** 10 seconds
   - **Retries:** 5 attempts

---

## Existing Acknowledgment Pattern

**Good News:** The codebase already has one example!

```rust
// src/actors/physics_orchestrator_actor.rs:1114
#[cfg(feature = "gpu")]
impl Handler<GPUInitialized> for PhysicsOrchestratorActor {
    type Result = ();

    fn handle(&mut self, _msg: GPUInitialized, _ctx: &mut Self::Context) -> Self::Result {
        info!("✅ GPU initialization CONFIRMED - GPUInitialized message received");
        self.gpu_initialized = true;
        self.gpu_init_in_progress = false;
        // ...
    }
}
```

This proves the pattern works! We just need to generalize it.

---

## Implementation Plan

### Phase 1: Core Infrastructure (Priority 1)
- [ ] Create `MessageId`, `MessageAck`, `AckStatus` types
- [ ] Implement `MessageTracker` with timeout handling
- [ ] Add retry logic with exponential backoff
- [ ] Create metrics tracking (success rate, retry count, latency)

### Phase 2: Critical Message Integration (Priority 1)
- [ ] Add `correlation_id: Option<MessageId>` to critical messages:
  - `UpdateGPUGraphData`
  - `UploadConstraintsToGPU`
  - `ComputeForces`
  - `SetSharedGPUContext`
  - `UpdateNodePositions`
  - `InitializeGPU`

- [ ] Add acknowledgment sending to handlers
- [ ] Integrate `MessageTracker` into `PhysicsOrchestratorActor`
- [ ] Integrate `MessageTracker` into other critical actors

### Phase 3: Monitoring & Testing (Priority 2)
- [ ] Add Prometheus metrics for message tracking
- [ ] Create dashboard for message acknowledgment status
- [ ] Write integration tests for retry logic
- [ ] Test failure scenarios (actor crash, mailbox full, timeout)

### Phase 4: Documentation (Priority 2)
- [ ] Document acknowledgment protocol for developers
- [ ] Create examples for adding new tracked messages
- [ ] Update architecture documentation

---

## Backwards Compatibility

**Key Design Principle:** `correlation_id` is `Option<MessageId>`

```rust
pub struct UpdateGPUGraphData {
    pub graph: Arc<GraphData>,
    pub correlation_id: Option<MessageId>,  // NEW - optional!
}
```

**Benefits:**
- ✅ Existing code without correlation IDs continues to work
- ✅ Gradual migration possible (add tracking incrementally)
- ✅ No breaking changes to existing message handlers

---

## Metrics & Monitoring

```rust
pub struct MessageMetrics {
    total_sent: AtomicU64,
    total_acked: AtomicU64,
    total_failed: AtomicU64,
    total_retried: AtomicU64,

    // Per-message-kind metrics
    by_kind: HashMap<MessageKind, KindMetrics>,
}

pub struct KindMetrics {
    sent_count: AtomicU64,
    success_count: AtomicU64,
    failure_count: AtomicU64,
    retry_count: AtomicU64,
    avg_latency_ms: AtomicU64,
}
```

**Exposed Metrics:**
- `messages_total{kind="UpdateGPUGraphData", status="sent"}`
- `messages_total{kind="UpdateGPUGraphData", status="acked"}`
- `messages_total{kind="UpdateGPUGraphData", status="failed"}`
- `message_retry_count{kind="UpdateGPUGraphData"}`
- `message_latency_ms{kind="UpdateGPUGraphData", quantile="0.95"}`

---

## Alternative: Hybrid Approach

For the **highest-priority** messages, use both:

1. **Synchronous Ask Pattern** for critical initialization:
```rust
// GPU initialization - MUST succeed before continuing
let result = gpu_addr.send(InitializeGPU { ... }).await?;
```

2. **Asynchronous Acknowledgment** for high-frequency messages:
```rust
// Position updates - use acknowledgment protocol
tracker.track_and_send(UpdateNodePositions { ... });
```

**Decision Matrix:**
- **Low frequency + critical** → Ask pattern (`.send().await`)
- **High frequency + critical** → Acknowledgment protocol
- **Non-critical** → Keep `do_send()` as-is

---

## Security Considerations

1. **Message ID Generation:** Use cryptographically secure UUIDs
2. **Replay Protection:** Track processed message IDs (prevent duplicate processing)
3. **Rate Limiting:** Limit retry attempts to prevent DoS
4. **Timeout Bounds:** Cap maximum timeout to prevent resource exhaustion

---

## Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_message_tracker_timeout() {
    let tracker = MessageTracker::new();
    let msg_id = MessageId::new();

    tracker.track(msg_id, MessageKind::ComputeForces, Duration::from_millis(100), 3);

    // Wait for timeout
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Verify retry triggered
    assert_eq!(tracker.metrics.total_retried.load(Ordering::Relaxed), 1);
}
```

### Integration Tests
```rust
#[actix_rt::test]
async fn test_gpu_update_with_acknowledgment() {
    let orchestrator = PhysicsOrchestratorActor::new(...);
    let gpu_actor = ForceComputeActor::new();

    // Send message with tracking
    let msg_id = MessageId::new();
    orchestrator.message_tracker.track(msg_id, ...);

    // Simulate GPU processing and ack
    gpu_actor.handle(UpdateGPUGraphData { ..., correlation_id: Some(msg_id) });

    // Verify ack received
    assert!(orchestrator.message_tracker.is_acked(msg_id));
}
```

### Failure Scenarios
- Actor crash mid-processing
- Mailbox full (backpressure)
- Network partition (distributed actors)
- Slow processing (timeout triggered)

---

## Performance Impact

**Overhead Analysis:**
- **Memory:** ~100 bytes per tracked message (UUID + metadata)
- **CPU:** Minimal (async timeout handling in background)
- **Latency:** Zero added latency (acks sent async)

**Optimization:**
- Use lock-free data structures for metrics
- Batch acknowledgments for high-frequency messages
- Automatic cleanup of acked messages (prevent memory leak)

---

## Success Criteria

1. ✅ Zero message loss for critical operations
2. ✅ Automatic retry with exponential backoff
3. ✅ <1% overhead on message throughput
4. ✅ Comprehensive metrics for monitoring
5. ✅ Backwards compatible with existing code

---

## Next Steps

1. **Review this design** with stakeholders
2. **Approve implementation approach** (Option 2 recommended)
3. **Implement Phase 1** (core infrastructure)
4. **Integrate Phase 2** (critical messages)
5. **Test and monitor Phase 3**

---

## References

- [Actix Actor Documentation](https://actix.rs/docs/actix/actor/)
- [At-Least-Once Delivery Pattern](https://doc.akka.io/docs/akka/current/typed/reliable-delivery.html)
- [Exponential Backoff Algorithm](https://en.wikipedia.org/wiki/Exponential_backoff)
- VisionFlow Audit Findings: H4 - Message Acknowledgment Protocol

---

**Status:** 📋 DESIGN COMPLETE - Ready for review and implementation
