---
title: Pipeline Sequence Diagrams
description: ```mermaid sequenceDiagram participant GitHub
category: explanation
tags:
  - architecture
  - api
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: advanced
---


# Pipeline Sequence Diagrams

## Complete End-to-End Flow

```mermaid
sequenceDiagram
    participant GitHub
    participant GitHubSync as GitHubSyncService
    participant DB as Neo4j/OntologyRepo
    participant Pipeline as OntologyPipelineService
    participant Reasoning as ReasoningActor
    participant Constraint as ConstraintBuilder
    participant GPU as OntologyConstraintActor
    participant Physics as ForceComputeActor
    participant WS as WebSocket
    participant Client

    GitHub->>GitHubSync: Webhook (OWL file changed)
    GitHubSync->>GitHubSync: Parse OntologyBlock
    GitHubSync->>DB: Save classes, properties, axioms
    DB-->>GitHubSync: Saved

    GitHubSync->>Pipeline: on-ontology-modified(ontology, correlation-id)
    activate Pipeline

    Pipeline->>Reasoning: TriggerReasoning
    activate Reasoning
    Reasoning->>Reasoning: CustomReasoner.infer-axioms()
    Reasoning->>Reasoning: Check cache
    alt Cache Hit
        Reasoning-->>Pipeline: Cached axioms (10ms)
    else Cache Miss
        Reasoning->>Reasoning: EL++ inference (200ms)
        Reasoning->>DB: Store inferred axioms
        Reasoning-->>Pipeline: New axioms (200ms)
    end
    deactivate Reasoning

    Pipeline->>Constraint: generate-constraints(axioms)
    activate Constraint
    Constraint->>Constraint: SubClassOf → Attraction
    Constraint->>Constraint: DisjointWith → Repulsion
    Constraint-->>Pipeline: ConstraintSet
    deactivate Constraint

    Pipeline->>GPU: ApplyOntologyConstraints
    activate GPU
    GPU->>GPU: Convert to GPU format
    GPU->>GPU: Upload to CUDA
    alt GPU Success
        GPU-->>Pipeline: Upload success
    else GPU Error
        GPU->>GPU: CPU fallback
        GPU-->>Pipeline: CPU fallback
    end
    deactivate GPU

    Pipeline-->>GitHubSync: Pipeline complete
    deactivate Pipeline

    GPU->>Physics: Trigger force computation
    activate Physics
    Physics->>Physics: Compute forces (CUDA)
    Physics->>Physics: Update positions
    Physics->>WS: BroadcastNodePositions
    deactivate Physics

    WS->>Client: Binary position update
    Client->>Client: Render graph
```

## Happy Path (Cached)

```mermaid
sequenceDiagram
    participant GitHub
    participant Pipeline as OntologyPipelineService
    participant Reasoning as ReasoningActor
    participant Cache as InferenceCache
    participant GPU as OntologyConstraintActor
    participant Client

    GitHub->>Pipeline: OWL file (correlation-id: abc-123)
    Note over Pipeline: [abc-123] Ontology modified

    Pipeline->>Reasoning: TriggerReasoning
    Reasoning->>Cache: get-or-compute(ontology-id)
    Cache-->>Reasoning: Cached axioms (8ms)
    Note over Reasoning: [abc-123] Cache hit!

    Reasoning-->>Pipeline: 15 inferred axioms (10ms total)
    Pipeline->>Pipeline: generate-constraints()
    Note over Pipeline: [abc-123] Generated 30 constraints

    Pipeline->>GPU: ApplyOntologyConstraints
    GPU->>GPU: Upload to CUDA
    GPU-->>Pipeline: Success (5ms)
    Note over GPU: [abc-123] GPU upload complete

    Pipeline-->>GitHub: Stats (total: 65ms)
    Note over Pipeline: [abc-123] Pipeline complete

    GPU->>Client: Position updates
```

## Error Path (GPU Failure)

```mermaid
sequenceDiagram
    participant Pipeline as OntologyPipelineService
    participant Reasoning as ReasoningActor
    participant GPU as OntologyConstraintActor
    participant Circuit as CircuitBreaker
    participant Metrics as MetricsCollector

    Pipeline->>Reasoning: TriggerReasoning
    Reasoning-->>Pipeline: Axioms OK

    Pipeline->>GPU: ApplyOntologyConstraints
    GPU->>GPU: CUDA upload
    GPU->>Circuit: Check circuit state
    Circuit-->>GPU: CLOSED (OK to proceed)

    GPU->>GPU: cudaMemcpy() → ERROR
    Note over GPU: Out of memory!

    GPU->>Metrics: Record GPU error
    GPU->>Circuit: on-failure()
    Circuit->>Circuit: Increment failure count (1/5)

    GPU->>GPU: CPU fallback
    GPU-->>Pipeline: Success (CPU fallback)

    Note over Pipeline: Retry #2
    Pipeline->>GPU: ApplyOntologyConstraints
    GPU->>GPU: CUDA upload → ERROR again

    GPU->>Circuit: on-failure()
    Circuit->>Circuit: Increment failure count (2/5)
    GPU->>GPU: CPU fallback
    GPU-->>Pipeline: Success (CPU fallback)

    Note over Circuit: After 5 failures...
    Circuit->>Circuit: Open circuit
    Note over Circuit: Circuit OPEN (30s timeout)

    Note over Pipeline: Future requests
    Pipeline->>GPU: ApplyOntologyConstraints
    GPU->>Circuit: Check circuit state
    Circuit-->>GPU: OPEN (reject immediately)
    GPU-->>Pipeline: CircuitBreakerOpen error

    Note over Circuit: After 30s timeout...
    Circuit->>Circuit: Transition to HALF-OPEN

    Pipeline->>GPU: ApplyOntologyConstraints
    GPU->>Circuit: Check circuit state
    Circuit-->>GPU: HALF-OPEN (try again)
    GPU->>GPU: CUDA upload → SUCCESS
    GPU->>Circuit: on-success()
    Circuit->>Circuit: Close circuit
    Note over Circuit: Circuit CLOSED (recovered)
```

## Backpressure Flow

```mermaid
sequenceDiagram
    participant GitHub
    participant Pipeline as OntologyPipelineService
    participant ReasoningQ as Reasoning Queue
    participant ConstraintQ as Constraint Queue
    participant GPUQ as GPU Queue
    participant RateLimiter

    Note over GitHub: Rapid changes (20 files)

    loop 20 files
        GitHub->>Pipeline: OWL file change
        Pipeline->>ReasoningQ: enqueue-reasoning()

        alt Queue not full
            ReasoningQ-->>Pipeline: Enqueued
        else Queue full (10/10)
            ReasoningQ-->>Pipeline: QueueFull error
            Pipeline->>Pipeline: Drop event
            Note over Pipeline: Dropped event (backpressure)
        end
    end

    Note over ReasoningQ: Process queue (10 events)

    loop Process reasoning queue
        ReasoningQ->>Pipeline: Dequeue event
        Pipeline->>RateLimiter: try-acquire(1 token)

        alt Tokens available
            RateLimiter-->>Pipeline: Acquired
            Pipeline->>Pipeline: Trigger reasoning
        else No tokens
            RateLimiter-->>Pipeline: Rate limited
            Pipeline->>ReasoningQ: Re-enqueue
            Note over Pipeline: Throttled (rate limit)
        end
    end

    Pipeline->>ConstraintQ: enqueue-constraint()
    ConstraintQ->>GPUQ: enqueue-gpu()

    Note over GPUQ: Process with backpressure
    GPUQ->>GPUQ: Check client queue size
    alt Client queue OK
        GPUQ->>GPU: Upload constraints
    else Client queue full
        GPUQ->>GPUQ: Skip frame (backpressure)
        Note over GPUQ: Client overloaded, throttling
    end
```

## Cache Invalidation Flow

```mermaid
sequenceDiagram
    participant Pipeline
    participant Reasoning
    participant Cache as InferenceCache
    participant DB as Neo4j/OntologyRepo

    Note over Pipeline: Ontology modified
    Pipeline->>Reasoning: TriggerReasoning(ontology-id=1)

    Reasoning->>Cache: get-or-compute(1)
    Cache->>Cache: calculate-checksum(ontology)
    Cache->>DB: SELECT checksum WHERE ontology-id=1

    alt Checksum matches
        DB-->>Cache: Cached checksum matches
        Cache->>DB: SELECT inferred-axioms
        DB-->>Cache: Cached axioms
        Cache-->>Reasoning: Cached result (fast)
        Note over Cache: Cache HIT
    else Checksum different
        DB-->>Cache: Checksum mismatch
        Cache-->>Reasoning: Cache miss
        Note over Cache: Cache MISS

        Reasoning->>Reasoning: CustomReasoner.infer-axioms()
        Reasoning->>DB: INSERT inferred-axioms
        Reasoning->>DB: UPDATE checksum

        Reasoning-->>Cache: Store new result
    end

    Note over Pipeline: Manual cache invalidation
    Pipeline->>Reasoning: InvalidateCache(ontology-id=1)
    Reasoning->>DB: DELETE FROM cache WHERE ontology-id=1
    DB-->>Reasoning: Deleted
```

## Correlation ID Tracing

```mermaid
sequenceDiagram
    participant GitHub
    participant GitHubSync
    participant Pipeline
    participant Reasoning
    participant GPU
    participant Logger
    participant Metrics

    GitHub->>GitHubSync: OWL file change
    GitHubSync->>GitHubSync: Generate correlation-id = "xyz-789"

    GitHubSync->>Logger: info("[xyz-789] Starting sync")
    GitHubSync->>Pipeline: on-ontology-modified(correlation-id="xyz-789")

    Pipeline->>Logger: info("[xyz-789] Pipeline triggered")
    Pipeline->>Reasoning: TriggerReasoning(correlation-id="xyz-789")

    Reasoning->>Logger: info("[xyz-789] Reasoning started")
    Reasoning->>Reasoning: inference()
    Reasoning->>Logger: info("[xyz-789] Reasoning complete: 20 axioms")
    Reasoning-->>Pipeline: axioms

    Pipeline->>Logger: info("[xyz-789] Generating constraints")
    Pipeline->>GPU: ApplyOntologyConstraints(correlation-id="xyz-789")

    GPU->>Logger: info("[xyz-789] GPU upload started")
    GPU->>GPU: upload to CUDA
    GPU->>Logger: info("[xyz-789] GPU upload complete")
    GPU->>Metrics: record("xyz-789", duration=5ms)

    Pipeline->>Logger: info("[xyz-789] Pipeline complete")
    Pipeline->>Metrics: record("xyz-789", total-duration=150ms)

    Note over Logger: All logs tagged with [xyz-789]
    Note over Metrics: All metrics tagged with xyz-789
```

## Event-Driven Updates

```mermaid
sequenceDiagram
    participant GitHubSync
    participant EventBus
    participant Handler1 as SemanticProcessor
    participant Handler2 as MetricsCollector
    participant Handler3 as AuditLogger

    GitHubSync->>EventBus: publish(OntologyModifiedEvent)
    Note over EventBus: Event: OntologyModified

    par Parallel event handlers
        EventBus->>Handler1: handle(event)
        Handler1->>Handler1: Invalidate semantic cache
        Handler1-->>EventBus: OK

    and
        EventBus->>Handler2: handle(event)
        Handler2->>Handler2: Record metric
        Handler2-->>EventBus: OK

    and
        EventBus->>Handler3: handle(event)
        Handler3->>Handler3: Log audit trail
        Handler3-->>EventBus: OK
    end

    EventBus-->>GitHubSync: All handlers complete
```

## Retry Logic Flow

```mermaid
sequenceDiagram
    participant Pipeline
    participant Reasoning
    participant Retry as RetryLogic

    Pipeline->>Reasoning: TriggerReasoning
    Reasoning->>Reasoning: infer-axioms()
    Reasoning-->>Pipeline: ERROR (timeout)

    Pipeline->>Retry: with-retry(f, max=3, backoff=100ms)

    Note over Retry: Attempt 1/3
    Retry->>Reasoning: infer-axioms()
    Reasoning-->>Retry: ERROR

    Note over Retry: Wait 100ms (exponential backoff)
    Retry->>Retry: sleep(100ms)

    Note over Retry: Attempt 2/3
    Retry->>Reasoning: infer-axioms()
    Reasoning-->>Retry: ERROR

    Note over Retry: Wait 200ms + jitter (50ms)
    Retry->>Retry: sleep(250ms)

    Note over Retry: Attempt 3/3
    Retry->>Reasoning: infer-axioms()
    Reasoning-->>Retry: SUCCESS

    Retry-->>Pipeline: Result (succeeded on retry 3)
```

## Admin API Flow

```mermaid
sequenceDiagram
    participant Admin as Admin UI
    participant API as PipelineAdminHandler
    participant Pipeline as OntologyPipelineService
    participant EventBus

    Admin->>API: POST /api/admin/pipeline/trigger
    API->>API: Generate correlation-id
    API->>Pipeline: on-ontology-modified()
    Pipeline-->>API: Stats
    API-->>Admin: {"status": "triggered", "correlation-id": "abc-123"}

    Admin->>API: GET /api/admin/pipeline/status
    API->>Pipeline: get-status()
    API->>EventBus: get-stats()
    API-->>Admin: {"status": "running", "queue-sizes": {...}}

    Admin->>API: POST /api/admin/pipeline/pause
    API->>Pipeline: pause()
    Pipeline->>Pipeline: Set paused flag
    API-->>Admin: {"status": "paused"}

    Admin->>API: GET /api/admin/pipeline/events/abc-123
    API->>EventBus: get-events-by-correlation("abc-123")
    EventBus-->>API: [OntologyModified, ReasoningComplete, ...]
    API-->>Admin: {"events": [...]}

    Admin->>API: POST /api/admin/pipeline/resume
    API->>Pipeline: resume()
    Pipeline->>Pipeline: Clear paused flag
    API-->>Admin: {"status": "resumed"}
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01-03 | Initial sequence diagrams |
