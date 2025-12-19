---
title: Actor System Guide
description: VisionFlow uses **Actix actors** for concurrent, fault-tolerant processing of graph operations, physics simulation, and real-time client coordination.  This guide covers actor patterns, best practi...
category: guide
tags:
  - architecture
  - design
  - patterns
  - api
  - api
related-docs:
  - concepts/architecture/core/server.md
  - explanations/architecture/hexagonal-cqrs.md
  - explanations/architecture/database-architecture.md
  - README.md
updated-date: 2025-12-18
difficulty-level: advanced
dependencies:
  - ✅ Document message ordering requirements
  - ✅ Add message acknowledgments for critical paths
  - Neo4j database
---

# Actor System Guide

**Status**: Production
**Last Updated**: December 2, 2025
**Audience**: Developers working with the actor system

---

## Overview

VisionFlow uses **Actix actors** for concurrent, fault-tolerant processing of graph operations, physics simulation, and real-time client coordination. This guide covers actor patterns, best practices, common pitfalls, and debugging techniques.

---

## Actor Hierarchy

### Supervision Tree

```mermaid
graph TB
    ROOT["Actix System Root<br/>━━━━━━━━━━━━━<br/>Managed by Actix runtime"]

    SUPER["GraphServiceSupervisor<br/>━━━━━━━━━━━━━<br/>Strategy: OneForOne<br/>Max Restarts: 3 per 60s<br/>Escalation: System shutdown"]

    STATE["GraphStateActor<br/>━━━━━━━━━━━━━<br/>Restart: Always<br/>Mailbox: Unbounded"]

    PHYS["PhysicsOrchestratorActor<br/>━━━━━━━━━━━━━<br/>Restart: Always<br/>Manages 11 GPU actors"]

    SEM["SemanticProcessorActor<br/>━━━━━━━━━━━━━<br/>Restart: Always<br/>Coordinates AI analysis"]

    CLIENT["ClientCoordinatorActor<br/>━━━━━━━━━━━━━<br/>Restart: On failure<br/>WebSocket management"]

    GPU1["ForceComputeActor"]
    GPU2["SemanticForcesActor"]
    GPU3["PageRankActor"]
    GPU_MORE["+ 8 more GPU actors..."]

    ROOT --> SUPER
    SUPER -->|supervises| STATE
    SUPER -->|supervises| PHYS
    SUPER -->|supervises| SEM
    SUPER -->|supervises| CLIENT

    PHYS -->|spawns| GPU1
    PHYS -->|spawns| GPU2
    PHYS -->|spawns| GPU3
    PHYS -->|spawns| GPU_MORE

    classDef root fill:#ffeb3b,stroke:#f57f17,stroke-width:3px
    classDef supervisor fill:#ff9800,stroke:#e65100,stroke-width:3px
    classDef core fill:#81c784,stroke:#2e7d32,stroke-width:2px
    classDef gpu fill:#64b5f6,stroke:#1565c0,stroke-width:2px

    class ROOT root
    class SUPER supervisor
    class STATE,PHYS,SEM,CLIENT core
    class GPU1,GPU2,GPU3,GPU_MORE gpu
```

### Actor Responsibilities Matrix

| Actor | Primary Role | Spawned By | Restart Policy | Typical Mailbox Size |
|-------|--------------|------------|----------------|---------------------|
| GraphServiceSupervisor | Coordination | System | Never | 10-50 |
| GraphStateActor | State management | Supervisor | Always | 100-1000 |
| PhysicsOrchestratorActor | GPU coordination | Supervisor | Always | 50-200 |
| SemanticProcessorActor | AI analysis | Supervisor | Always | 20-100 |
| ClientCoordinatorActor | WebSocket | Supervisor | On failure | 500-5000 |
| ForceComputeActor | CUDA forces | Physics | On failure | 10-50 |
| PageRankActor | Centrality | Physics | On failure | 5-20 |
| OntologyActor | Reasoning | Supervisor | Always | 10-50 |

---

## Message Patterns

### Request-Response Pattern

**When to use**: Client needs immediate response

```rust
// Message definition
#[derive(Message)]
#[rtype(result = "Result<GraphData, ActorError>")]
pub struct GetGraphData;

// Handler implementation
impl Handler<GetGraphData> for GraphStateActor {
    type Result = ResponseFuture<Result<GraphData, ActorError>>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        let repo = self.graph_repo.clone();

        Box::pin(async move {
            repo.get_graph()
                .await
                .map_err(|e| ActorError::RepositoryError(e))
        })
    }
}

// Client usage
let result = graph_state_actor
    .send(GetGraphData)
    .await??; // First ? for mailbox error, second ? for handler error
```

### Fire-and-Forget Pattern

**When to use**: No response needed, fire-and-forget

```rust
// Message definition
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastGraphUpdate {
    pub update: GraphUpdate,
}

// Handler implementation
impl Handler<BroadcastGraphUpdate> for ClientCoordinatorActor {
    type Result = ();

    fn handle(&mut self, msg: BroadcastGraphUpdate, _ctx: &mut Context<Self>) {
        // No return value, just broadcast
        for client in &self.clients {
            client.do_send(msg.update.clone());
        }
    }
}

// Client usage (no await needed)
client_coordinator.do_send(BroadcastGraphUpdate { update });
```

### Actor-to-Actor Coordination Pattern

**When to use**: Multiple actors need to coordinate

```rust
// Coordination message
#[derive(Message)]
#[rtype(result = "Result<(), ActorError>")]
pub struct PhysicsStepComplete {
    pub node_positions: Vec<(u32, (f32, f32, f32))>,
}

// PhysicsOrchestratorActor → GraphStateActor
impl Handler<PhysicsStepComplete> for GraphStateActor {
    type Result = ResponseFuture<Result<(), ActorError>>;

    fn handle(&mut self, msg: PhysicsStepComplete, _ctx: &mut Context<Self>) -> Self::Result {
        let repo = self.graph_repo.clone();
        let client_coord = self.client_coordinator.clone();

        Box::pin(async move {
            // 1. Persist positions to Neo4j
            repo.batch_update_positions(msg.node_positions.clone()).await?;

            // 2. Notify clients
            if let Some(client_actor) = client_coord {
                client_actor.do_send(BroadcastGraphUpdate {
                    update: GraphUpdate::Positions(msg.node_positions),
                });
            }

            Ok(())
        })
    }
}
```

---

## Message Ordering Guarantees

### Within Single Actor: GUARANTEED ✅

Messages to the **same actor** are processed in **FIFO order**:

```rust
// These messages will execute in order for the same actor
actor.do_send(Message1);
actor.do_send(Message2);
actor.do_send(Message3);
// Guaranteed: Message1 → Message2 → Message3
```

### Across Multiple Actors: NO GUARANTEE ⚠️

Messages to **different actors** have **no ordering guarantee**:

```rust
// NO ordering guarantee between these actors
actor_a.do_send(UpdateNodePosition { node_id: 1, pos: (1.0, 2.0, 3.0) });
actor_b.do_send(RecalculatePhysics { node_id: 1 });
// Race condition: actor_b might process before actor_a!
```

### Coordination Techniques

#### 1. Message Acknowledgment

```rust
// Use request-response for coordination
let ack = actor_a.send(UpdateNodePosition { ... }).await??;
// Now safe to send to actor_b
actor_b.do_send(RecalculatePhysics { ... });
```

#### 2. Coordinator Pattern

```rust
// GraphServiceSupervisor coordinates multi-actor operations
impl Handler<CoordinatedUpdate> for GraphServiceSupervisor {
    type Result = ResponseFuture<Result<(), ActorError>>;

    fn handle(&mut self, msg: CoordinatedUpdate, _ctx: &mut Context<Self>) -> Self::Result {
        let state = self.graph_state.clone();
        let physics = self.physics_orch.clone();

        Box::pin(async move {
            // Step 1: Update state
            state.send(UpdateState { ... }).await??;

            // Step 2: Trigger physics (guaranteed after step 1)
            physics.do_send(RecalculatePhysics { ... });

            Ok(())
        })
    }
}
```

#### 3. Event Bus (Future Enhancement)

```rust
// Planned: Event-driven coordination
event_bus.publish(GraphUpdated { node_id: 1, ... });
// Subscribers handle in parallel, no ordering issues
```

---

## Fault Tolerance

### Supervision Strategies

**OneForOne** (Current for GraphServiceSupervisor):
- Only restart the failed actor
- Other actors continue running
- Best for independent actors

```rust
let restart_policy = RestartPolicy {
    max_restarts: 3,
    within_time_period: Duration::from_secs(60),
    backoff_strategy: BackoffStrategy::Exponential {
        initial: Duration::from_millis(100),
        max: Duration::from_secs(10),
    },
    escalation_threshold: 3,
};
```

**OneForAll** (Alternative):
- Restart all children when one fails
- Use when actors have shared state
- More disruptive but guarantees consistency

### Restart Scenarios

#### Transient Failure (e.g., temporary GPU error)

```
1. ForceComputeActor fails (GPU timeout)
2. Supervisor detects failure
3. Supervisor restarts ForceComputeActor (attempt 1/3)
4. Actor re-initializes GPU resources
5. Processing resumes
```

#### Persistent Failure (e.g., GPU out of memory)

```
1. ForceComputeActor fails (GPU OOM)
2. Supervisor restarts (attempt 1/3) → fails
3. Supervisor restarts (attempt 2/3) → fails
4. Supervisor restarts (attempt 3/3) → fails
5. Supervisor escalates to system → logs error
6. PhysicsOrchestratorActor falls back to CPU physics
```

#### Cascading Failure Prevention

```rust
// Circuit breaker pattern
struct ActorCircuitBreaker {
    failure_threshold: u32,
    reset_timeout: Duration,
    state: CircuitState,
}

enum CircuitState {
    Closed,          // Normal operation
    Open,            // Too many failures, reject requests
    HalfOpen,        // Testing if actor recovered
}
```

---

## Performance Considerations

### Mailbox Sizes

| Actor Type | Typical Size | Max Observed | Recommendation |
|------------|--------------|--------------|----------------|
| GraphStateActor | 200-500 | 2,000 | Monitor >1000 |
| PhysicsOrchestratorActor | 50-100 | 500 | Monitor >300 |
| ClientCoordinatorActor | 1,000-5,000 | 20,000 | Monitor >10,000 |
| GPU Actors | 10-50 | 200 | Monitor >100 |

**Warning signs**:
- Mailbox size growing continuously → Actor is overloaded
- High latency for simple operations → Check mailbox size
- Memory usage increasing → Mailbox accumulation

### Message Throughput

Measured on development hardware (RTX 3060, Ryzen 5600X):

| Message Type | Throughput | Latency (p50) | Latency (p95) |
|--------------|-----------|---------------|---------------|
| GetGraphData | 500 msg/s | 2ms | 8ms |
| UpdateNodePosition | 1,000 msg/s | 1ms | 4ms |
| TriggerPhysicsStep | 100 msg/s | 10ms | 30ms |
| BroadcastToClients | 2,000 msg/s | 0.5ms | 2ms |

### Optimization Techniques

#### 1. Batch Messages

```rust
// ❌ BAD: 1000 individual messages
for node in nodes {
    actor.do_send(UpdateNodePosition { node_id: node.id, ... });
}

// ✅ GOOD: 1 batched message
actor.do_send(BatchUpdatePositions { positions: node_positions });
```

#### 2. Use `do_send` for Fire-and-Forget

```rust
// ❌ BAD: Unnecessary await
actor.send(Notification { ... }).await?;

// ✅ GOOD: Fire-and-forget
actor.do_send(Notification { ... });
```

#### 3. Reduce Message Size

```rust
// ❌ BAD: Large message (10 KB+)
actor.send(FullGraphUpdate {
    nodes: all_10000_nodes,
    edges: all_50000_edges
}).await?;

// ✅ GOOD: Reference-based message
actor.send(GraphUpdateNotification {
    changed_node_ids: vec![1, 5, 12]
}).await?;
// Actor fetches from repository as needed
```

---

## Common Pitfalls

### 1. Deadlock via Circular Dependencies

**Problem**: Actor A waits for Actor B, Actor B waits for Actor A

```rust
// ❌ DEADLOCK RISK
impl Handler<Message1> for ActorA {
    type Result = ResponseFuture<Result<(), ActorError>>;

    fn handle(&mut self, _msg: Message1, _ctx: &mut Context<Self>) -> Self::Result {
        let actor_b = self.actor_b.clone();
        Box::pin(async move {
            actor_b.send(Message2).await??; // Waits for ActorB
            Ok(())
        })
    }
}

impl Handler<Message2> for ActorB {
    type Result = ResponseFuture<Result<(), ActorError>>;

    fn handle(&mut self, _msg: Message2, _ctx: &mut Context<Self>) -> Self::Result {
        let actor_a = self.actor_a.clone();
        Box::pin(async move {
            actor_a.send(Message3).await??; // Waits for ActorA → DEADLOCK!
            Ok(())
        })
    }
}
```

**Solution**: Use coordinator pattern or one-way messages

```rust
// ✅ COORDINATOR PATTERN
impl Handler<CoordinatedOperation> for Supervisor {
    fn handle(&mut self, _msg: CoordinatedOperation, _ctx: &mut Context<Self>) -> Self::Result {
        let actor_a = self.actor_a.clone();
        let actor_b = self.actor_b.clone();

        Box::pin(async move {
            actor_a.send(Message1).await??;
            actor_b.send(Message2).await??; // Sequential, no deadlock
            Ok(())
        })
    }
}
```

### 2. Forgetting Error Handling

**Problem**: Actor crashes silently, supervisor doesn't know why

```rust
// ❌ BAD: Panics crash the actor
impl Handler<ProcessData> for Actor {
    type Result = ();

    fn handle(&mut self, msg: ProcessData, _ctx: &mut Context<Self>) {
        let result = self.process(msg.data);
        result.unwrap(); // 💥 PANIC if error → actor crash
    }
}
```

**Solution**: Return `Result` and log errors

```rust
// ✅ GOOD: Graceful error handling
impl Handler<ProcessData> for Actor {
    type Result = ResponseFuture<Result<(), ActorError>>;

    fn handle(&mut self, msg: ProcessData, _ctx: &mut Context<Self>) -> Self::Result {
        Box::pin(async move {
            match self.process(msg.data).await {
                Ok(_) => Ok(()),
                Err(e) => {
                    error!("ProcessData failed: {:?}", e);
                    Err(ActorError::ProcessingError(e))
                }
            }
        })
    }
}
```

### 3. Unbounded Mailbox Growth

**Problem**: Actor can't keep up with incoming messages

```rust
// ❌ BAD: No backpressure
for i in 0..1_000_000 {
    actor.do_send(ProcessTask { id: i }); // Mailbox grows to 1M messages!
}
```

**Solution**: Add backpressure or rate limiting

```rust
// ✅ GOOD: Bounded queue with backpressure
let semaphore = Arc::new(Semaphore::new(100)); // Max 100 in-flight

for i in 0..1_000_000 {
    let permit = semaphore.clone().acquire_owned().await?;
    let actor = actor.clone();

    tokio::spawn(async move {
        actor.send(ProcessTask { id: i }).await?;
        drop(permit); // Release permit when done
        Ok::<_, ActorError>(())
    });
}
```

### 4. Ignoring Actor Lifecycle

**Problem**: Actor holds resources that aren't cleaned up

```rust
// ❌ BAD: GPU memory leaks on actor crash
struct GpuActor {
    gpu_buffer: CudaBuffer, // Not cleaned up on crash!
}
```

**Solution**: Implement `Drop` or use `stopping()` hook

```rust
// ✅ GOOD: Explicit cleanup
impl Actor for GpuActor {
    type Context = Context<Self>;

    fn stopping(&mut self, _ctx: &mut Self::Context) -> Running {
        info!("GpuActor stopping, cleaning up GPU resources");
        self.gpu_buffer.free(); // Explicit cleanup
        Running::Stop
    }
}
```

---

## Debugging Actor Issues

### 1. Tracing Message Flow

Enable debug logging for actor messages:

```rust
// In actor handler
impl Handler<MyMessage> for MyActor {
    fn handle(&mut self, msg: MyMessage, ctx: &mut Context<Self>) -> Self::Result {
        debug!("MyActor received MyMessage {{ field: {} }}", msg.field);
        debug!("Mailbox size: {}", ctx.address().recipient().connected());

        // ... handler logic

        debug!("MyActor finished processing MyMessage");
    }
}
```

**Log output**:
```
[DEBUG] MyActor received MyMessage { field: 42 }
[DEBUG] Mailbox size: 127
[DEBUG] MyActor finished processing MyMessage
```

### 2. Detecting Stuck Actors

Use heartbeat mechanism:

```rust
struct ActorWithHeartbeat {
    last_heartbeat: Instant,
}

impl Actor for ActorWithHeartbeat {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        // Send heartbeat every 5 seconds
        ctx.run_interval(Duration::from_secs(5), |act, ctx| {
            debug!("Heartbeat from actor {}", ctx.address().recipient());
            act.last_heartbeat = Instant::now();
        });
    }
}

// Monitor actor (separate actor)
impl Handler<CheckHealth> for MonitorActor {
    fn handle(&mut self, _msg: CheckHealth, _ctx: &mut Context<Self>) -> Self::Result {
        for (name, actor_info) in &self.actors {
            if actor_info.last_heartbeat.elapsed() > Duration::from_secs(30) {
                error!("Actor {} is stuck! No heartbeat for 30s", name);
            }
        }
    }
}
```

### 3. Mailbox Size Monitoring

Add instrumentation to track mailbox growth:

```rust
// In supervisor or monitoring actor
impl Handler<GetActorStats> for GraphServiceSupervisor {
    fn handle(&mut self, _msg: GetActorStats, _ctx: &mut Context<Self>) -> Self::Result {
        let stats = vec![
            ActorStat {
                name: "GraphStateActor",
                mailbox_size: self.graph_state.recipient().connected(),
                health: if mailbox_size < 1000 { "healthy" } else { "degraded" },
            },
            // ... other actors
        ];

        ResponseActFuture::new(async { stats })
    }
}
```

### 4. Reproduce Race Conditions

Use deterministic testing with delays:

```rust
#[tokio::test]
async fn test_race_condition() {
    let actor_a = ActorA::start_default();
    let actor_b = ActorB::start_default();

    // Send messages in specific order with delays
    actor_a.do_send(Message1);
    tokio::time::sleep(Duration::from_millis(10)).await; // Simulate timing
    actor_b.do_send(Message2);

    tokio::time::sleep(Duration::from_millis(100)).await; // Wait for processing

    // Assert state
    let state = actor_a.send(GetState).await??;
    assert_eq!(state.value, expected_value);
}
```

---

## When to Create New Actors

### ✅ Good Reasons

1. **Isolated State**: Actor needs its own mutable state (e.g., GPU resources)
2. **Concurrency**: Operation is I/O-bound or CPU-intensive (e.g., physics simulation)
3. **Fault Isolation**: Failure should not affect other components (e.g., client connections)
4. **Lifecycle Management**: Resource needs explicit lifecycle (e.g., WebSocket connection)

### ❌ Bad Reasons

1. **Simple Functions**: Pure functions don't need actors
2. **Shared Read-Only Data**: Use `Arc<T>` instead
3. **Synchronous Operations**: Use regular structs if no concurrency needed
4. **Over-Decomposition**: Too many actors add message passing overhead

### Decision Tree

```
Is the operation:
  CPU-intensive or I/O-bound?
    YES → Consider actor for concurrency
    NO → Does it need isolated mutable state?
      YES → Consider actor for state isolation
      NO → Does it need fault isolation?
        YES → Consider actor for fault tolerance
        NO → Use regular struct/function
```

---

## Best Practices Summary

### Message Design
- ✅ Keep messages small (<1 KB)
- ✅ Use `do_send` for fire-and-forget
- ✅ Return `Result` for error handling
- ✅ Batch operations when possible

### Actor Lifecycle
- ✅ Implement `stopping()` for cleanup
- ✅ Use heartbeats for monitoring
- ✅ Handle restart gracefully
- ✅ Document restart policy

### Coordination
- ✅ Use supervisor for multi-actor operations
- ✅ Avoid circular dependencies
- ✅ Document message ordering requirements
- ✅ Add message acknowledgments for critical paths

### Performance
- ✅ Monitor mailbox sizes
- ✅ Add backpressure for high-throughput actors
- ✅ Profile message handling time
- ✅ Use `tokio::spawn` for CPU-bound work

### Debugging
- ✅ Enable debug logging in development
- ✅ Add tracing spans for message flow
- ✅ Monitor actor health metrics
- ✅ Write integration tests for message flows

---

### Architecture Docs
- [Server Architecture](../../concepts/architecture/core/server.md) - Overall system design with 21-actor hierarchy
- [Hexagonal Architecture](../../explanations/architecture/hexagonal-cqrs.md) - Ports/adapters/CQRS patterns
- [Database Architecture](../../explanations/architecture/database-architecture.md) - Neo4j integration

### Implementation Guides
- [Message Flow Debugging](../debugging/actor-message-tracing.md) - Tracing actor messages (if exists)
- [Performance Tuning](../performance/actor-optimization.md) - Advanced optimization (if exists)

### External Resources
- [Actix Documentation](https://actix.rs/) - Official Actix framework docs
- [Actix Best Practices](https://actix.rs/docs/whatis/) - Actor patterns and anti-patterns

---

---

---

## Related Documentation

- [Client State Management with Zustand](../client/state-management.md)
- [Adding Features](../developer/04-adding-features.md)
- [Testing Guide](../../archive/docs/guides/developer/05-testing-guide.md)
- [RAGFlow Knowledge Management Integration](../ai-models/ragflow-integration.md)
- [Working with Agents](../../archive/docs/guides/user/working-with-agents.md)

## Changelog

**December 2, 2025**
- Initial actor system guide created
- Added 21-actor hierarchy documentation
- Included common pitfalls and debugging techniques
- Added performance benchmarks and best practices
