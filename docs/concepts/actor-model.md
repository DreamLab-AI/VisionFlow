---
title: Actor Model
description: Understanding VisionFlow's Actix-based actor system for fault-tolerant, concurrent graph processing
category: explanation
tags:
  - actor
  - actix
  - concurrency
  - architecture
related-docs:
  - concepts/hexagonal-architecture.md
  - concepts/gpu-acceleration.md
  - reference/api/rest-api-reference.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# Actor Model

VisionFlow uses the actor model for concurrent, fault-tolerant graph processing. Built on Actix, the system coordinates 21+ specialised actors under hierarchical supervision.

---

## Core Concept

The actor model treats computation as message-passing between isolated actors:

- **Actors** are independent units with private state
- **Messages** are the only way to communicate
- **Mailboxes** queue incoming messages
- **Supervision** handles failures hierarchically

This model naturally fits VisionFlow's needs:
- WebSocket clients connect/disconnect asynchronously
- GPU computations run in parallel
- Ontology changes trigger cascading updates
- Failures isolate to individual actors

---

## Actor Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Actor Hierarchy                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  GraphServiceSupervisor (Root - OneForOne strategy)         │
│  ├── GraphStateActor (State management - 797 lines)         │
│  │                                                           │
│  ├── PhysicsOrchestratorActor (GPU coordination)            │
│  │   ├── ForceComputeActor (CUDA physics)                   │
│  │   ├── StressMajorizationActor (Layout optimisation)      │
│  │   ├── SemanticForcesActor (Semantic clustering)          │
│  │   ├── ConstraintActor (Hard constraints)                 │
│  │   ├── OntologyConstraintActor (OWL rules)                │
│  │   ├── ShortestPathActor (SSSP/APSP)                      │
│  │   ├── PageRankActor (Centrality)                         │
│  │   ├── ClusteringActor (K-means/Louvain)                  │
│  │   ├── AnomalyDetectionActor (LOF/Z-score)                │
│  │   ├── ConnectedComponentsActor (Union-Find)              │
│  │   └── GPUResourceActor (Memory/stream management)        │
│  │                                                           │
│  ├── SemanticProcessorActor (AI/ML features)                │
│  │                                                           │
│  ├── ClientCoordinatorActor (WebSocket - 1,593 lines)       │
│  │                                                           │
│  ├── WorkspaceActor (Multi-tenant workspaces)               │
│  │                                                           │
│  └── SettingsActor (Configuration)                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Supervision Strategies

Supervision determines how parent actors handle child failures.

### OneForOne (Default)

Only restart the failed actor; siblings continue unaffected.

```
Child A crashes → Only A restarts
                  B and C continue
```

**Used by**: GraphServiceSupervisor

### AllForOne

Restart all children when any fails.

```
Child A crashes → A, B, C all restart
                  Ensures consistent state
```

**Used by**: PhysicsOrchestratorActor (GPU actors share state)

### RestForOne

Restart failed actor and all actors started after it.

```
Start order: A, B, C
C crashes → Only C restarts
B crashes → B and C restart
A crashes → A, B, C restart
```

**Used for**: Dependency chains

### Restart Configuration

```rust
SupervisorStrategy::OneForOne {
    max_restarts: 3,          // Max restarts within window
    within: Duration::from_secs(10),
    backoff: BackoffStrategy::Exponential {
        initial_interval: Duration::from_millis(500),
        max_interval: Duration::from_secs(5),
        multiplier: 2.0,
    }
}
```

---

## Key Actors

### GraphStateActor

**Purpose**: Central graph state management

**State machine**:
```
Uninitialized → Initializing → Loading → Ready
                                           ↓
                                       Updating
                                           ↓
                                       Simulating
                                           ↓
                                       Error → Recovering
```

**Key responsibilities**:
- Node/edge CRUD operations
- Metadata-to-node mapping
- Graph serialisation for clients
- Position cache for physics

### PhysicsOrchestratorActor

**Purpose**: Coordinates 11 GPU sub-actors

**Orchestration flow**:
```
1. ComputeForces → ForceComputeActor
2. ApplySemanticForces → SemanticForcesActor
3. ValidateConstraints → ConstraintActor
4. ValidateOntology → OntologyConstraintActor
5. UpdatePositions → ForceComputeActor
6. (Optional) OptimizeLayout → StressMajorizationActor
7. Broadcast → ClientCoordinatorActor
```

### ClientCoordinatorActor

**Purpose**: WebSocket client management and broadcasting

**Key features**:
- Client registration/deregistration
- Binary protocol serialisation
- Adaptive broadcast intervals (60 FPS active, 5 Hz settled)
- Bandwidth throttling per client

### SemanticProcessorActor

**Purpose**: AI-driven semantic analysis

**Capabilities**:
- Content embedding generation (256-dim vectors)
- Topic classification
- Importance scoring
- Constraint generation based on similarity

---

## Message Patterns

### Request-Response (Synchronous)

Wait for actor response:

```rust
// Sender
let result: Arc<GraphData> = graph_actor.send(GetGraphData).await?;

// Receiver
impl Handler<GetGraphData> for GraphStateActor {
    type Result = Result<Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(self.graph_data.clone().ok_or("No data")?)
    }
}
```

### Fire-and-Forget (Asynchronous)

Send without waiting:

```rust
// Sender (no await, returns immediately)
client_coordinator.do_send(UpdateNodePositions { positions });

// Receiver
impl Handler<UpdateNodePositions> for ClientCoordinatorActor {
    type Result = ();  // No response

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Context<Self>) {
        self.broadcast_positions(msg.positions);
    }
}
```

### Pub/Sub (Event Broadcasting)

Notify multiple subscribers:

```rust
// Publisher
impl GraphStateActor {
    fn notify_graph_update(&self, event: GraphUpdateEvent) {
        for subscriber in &self.subscribers {
            subscriber.do_send(event.clone());
        }
    }
}

// Subscriber
impl Handler<GraphUpdateEvent> for SemanticProcessorActor {
    type Result = ();

    fn handle(&mut self, msg: GraphUpdateEvent, _ctx: &mut Context<Self>) {
        self.invalidate_cache(&msg.changed_nodes);
    }
}
```

---

## Mailbox Management

### Default Configuration

- **Capacity**: Unbounded (no backpressure)
- **Ordering**: FIFO
- **Priority**: All messages equal

### Bounded Mailbox (Backpressure)

```rust
let actor = GraphStateActor::new()
    .start_in_arbiter(&arbiter, |ctx| {
        ctx.set_mailbox_capacity(1000);  // Max pending
    });
```

When full, senders receive `MailboxError::Closed`.

### Priority Messages

Some messages skip the queue:

```rust
impl Actor for GraphStateActor {
    fn handle_priority_message(&mut self, msg: EmergencyShutdown) {
        // Processed before queued messages
    }
}
```

---

## Actor Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                     Actor Lifecycle                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Created → Starting → Started → Running                     │
│                                     ↓                        │
│                             MessageWaiting ←→ Processing     │
│                                     ↓                        │
│                                 Stopping                     │
│                                     ↓                        │
│                                  Stopped                     │
│                                                              │
│  On Error: Processing → Error → Restarting → Starting       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Lifecycle hooks**:
- `started()`: Initialise resources, start timers
- `stopping()`: Begin cleanup
- `stopped()`: Final cleanup, release resources

---

## Error Recovery

### Fault Isolation Zones

| Zone | Actors | Failure Impact |
|------|--------|---------------|
| Graph State | GraphStateActor | Transient errors, retry |
| Physics | 11 GPU actors | CUDA errors, OOM |
| Semantic | SemanticProcessorActor | AI model errors |
| Clients | ClientCoordinatorActor | WebSocket disconnects |

### Recovery Actions

1. **Retry**: Exponential backoff (3 attempts)
2. **Reload**: Fetch state from database
3. **Isolate**: Remove failed component, degrade gracefully
4. **Escalate**: Restart supervisor (full system reset)

### Example: GPU OOM Recovery

```rust
impl Handler<ComputeForces> for ForceComputeActor {
    type Result = ResponseFuture<Result<ForceVectors, String>>;

    fn handle(&mut self, _msg: ComputeForces, ctx: &mut Context<Self>) -> Self::Result {
        match self.compute_gpu() {
            Ok(forces) => Ok(forces),
            Err(e) if e.contains("CUDA OOM") => {
                // Escalate to supervisor for GPU reset
                ctx.stop();
                Err("GPU OOM - escalating".into())
            }
            Err(e) => Err(e),
        }
    }
}
```

---

## Performance Characteristics

### Message Latency

| Pattern | P50 | P95 | P99 |
|---------|-----|-----|-----|
| Local actor (same thread) | 50 us | 100 us | 200 us |
| GPU actor (CUDA kernel) | 2 ms | 5 ms | 10 ms |
| WebSocket broadcast | 10 ms | 30 ms | 100 ms |

### Throughput

| Actor | Message Type | Throughput |
|-------|-------------|------------|
| GraphStateActor | GetGraphData | 20,000/s |
| GraphStateActor | AddNode | 5,000/s |
| PhysicsOrchestratorActor | SimulationStep | 60/s |
| ClientCoordinatorActor | Broadcast | 20/s |

---

## Checkpointing

### Periodic State Snapshots

```rust
impl GraphStateActor {
    fn start_checkpointing(&mut self, ctx: &mut Context<Self>) {
        ctx.run_interval(Duration::from_secs(60), |act, _| {
            let checkpoint = act.create_checkpoint();
            act.save_checkpoint(checkpoint);
        });
    }
}
```

### Recovery from Checkpoint

```rust
impl Actor for GraphStateActor {
    fn started(&mut self, _ctx: &mut Context<Self>) {
        if let Some(checkpoint) = self.load_latest_checkpoint() {
            self.restore_from_checkpoint(checkpoint);
            info!("Restored from checkpoint: {} seconds old",
                  checkpoint.age_seconds());
        } else {
            self.load_from_database();
        }
    }
}
```

---

## Related Concepts

- **[Hexagonal Architecture](hexagonal-architecture.md)**: How actors integrate with ports/adapters
- **[GPU Acceleration](gpu-acceleration.md)**: GPU actor implementation details
- **[Real-Time Sync](real-time-sync.md)**: ClientCoordinatorActor WebSocket handling
