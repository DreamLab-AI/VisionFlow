# Pipeline Admin API Guide

> ⚠️ **DEPRECATION NOTICE** ⚠️
> **GraphServiceActor** is deprecated. See `/docs/guides/graphserviceactor-migration.md` for current patterns.

**Status**: ⚙️ Code exists, wiring needed
**Last Updated**: November 3, 2025

---

## Overview

The Ontology Pipeline Admin API provides REST endpoints for managing the semantic intelligence pipeline: triggering reasoning, monitoring status, and controlling execution flow.

---

## Quick Start

### Trigger Pipeline
```bash
curl -X POST http://localhost:4000/api/admin/pipeline/trigger \
  -H "Content-Type: application/json" \
  -d '{"force": true}'
```

### Check Status
```bash
curl http://localhost:4000/api/admin/pipeline/status
```

### View Metrics
```bash
curl http://localhost:4000/api/admin/pipeline/metrics
```

---

## REST Endpoints

### POST /api/admin/pipeline/trigger
**Description**: Manually trigger ontology reasoning pipeline

**Request**:
```json
{
  "force": true
}
```

**Response**:
```json
{
  "status": "triggered",
  "correlation_id": "abc123",
  "timestamp": "2025-11-03T18:00:00Z"
}
```

---

### GET /api/admin/pipeline/status
**Description**: Get current pipeline execution status

**Response**:
```json
{
  "status": "running",
  "current_stage": "reasoning",
  "progress_percent": 45,
  "started_at": "2025-11-03T17:59:00Z",
  "queue_sizes": {
    "reasoning": 3,
    "constraints": 0,
    "gpu_upload": 0
  }
}
```

---

### POST /api/admin/pipeline/pause
**Description**: Pause pipeline execution

**Request**:
```json
{
  "reason": "Maintenance"
}
```

**Response**:
```json
{
  "status": "paused",
  "reason": "Maintenance",
  "timestamp": "2025-11-03T18:00:00Z"
}
```

---

### POST /api/admin/pipeline/resume
**Description**: Resume paused pipeline

**Response**:
```json
{
  "status": "running",
  "timestamp": "2025-11-03T18:05:00Z"
}
```

---

### GET /api/admin/pipeline/metrics
**Description**: Get pipeline performance metrics

**Response**:
```json
{
  "metrics": {
    "reasoning_latency_ms": 45.2,
    "constraint_gen_latency_ms": 12.3,
    "gpu_upload_latency_ms": 8.7,
    "total_pipeline_latency_ms": 66.2,
    "error_rate": 0.01,
    "cache_hit_rate": 0.85
  },
  "window": "last_1000_executions"
}
```

---

### GET /api/admin/pipeline/events/:correlation_id
**Description**: Query event log for specific execution

**Response**:
```json
{
  "correlation_id": "abc123",
  "events": [
    {
      "type": "OntologyModified",
      "timestamp": "2025-11-03T17:59:00Z",
      "data": {"file_count": 5}
    },
    {
      "type": "ReasoningComplete",
      "timestamp": "2025-11-03T17:59:45Z",
      "data": {"inference_count": 42}
    }
  ]
}
```

---

### POST /api/admin/pipeline/cache/clear
**Description**: Clear pipeline cache

**Response**:
```json
{
  "status": "cleared",
  "cache_entries_removed": 156
}
```

---

## Integration Checklist

### Step 1: Add Module to mod.rs

**File**: `src/handlers/mod.rs`

Add after line 42:
```rust
pub mod pipeline_admin_handler;
pub use pipeline_admin_handler::configure_routes as configure_pipeline_admin_routes;
```

### Step 2: Add Module to services/mod.rs

**File**: `src/services/mod.rs`

Add line:
```rust
pub mod pipeline_events;
```

### Step 3: Add AppState Fields

**File**: `src/app_state.rs`

Add imports:
```rust
use crate::services::pipeline_events::PipelineEventBus;
use crate::services::ontology_pipeline_service::{OntologyPipelineService, SemanticPhysicsConfig};
```

Add fields to AppState struct:
```rust
pub pipeline_event_bus: Arc<RwLock<PipelineEventBus>>,
pub pipeline_service: Arc<OntologyPipelineService>,
pub pipeline_paused: Arc<RwLock<bool>>,
pub pipeline_pause_reason: Arc<RwLock<Option<String>>>,
```

### Step 4: Initialize in AppState::new()

Add after actor initialization:
```rust
// Initialize pipeline infrastructure
info!("[AppState::new] Initializing pipeline event bus and orchestration service");
let pipeline_event_bus = Arc::new(RwLock::new(PipelineEventBus::new(10000)));

let pipeline_config = SemanticPhysicsConfig::default();
let mut pipeline_service = OntologyPipelineService::new(pipeline_config);

// Wire existing actors
if let Some(ref ontology_addr) = ontology_actor_addr {
    pipeline_service.set_ontology_actor(ontology_addr.clone());
}
pipeline_service.set_graph_actor(graph_service_addr.clone());
pipeline_service.set_graph_repository(knowledge_graph_repository.clone());

let pipeline_service = Arc::new(pipeline_service);
let pipeline_paused = Arc::new(RwLock::new(false));
let pipeline_pause_reason = Arc::new(RwLock::new(None));
```

### Step 5: Register Routes in main.rs

**File**: `src/main.rs`

Add import:
```rust
use webxr::handlers::configure_pipeline_admin_routes;
```

Create pipeline admin state:
```rust
let pipeline_admin_state = web::Data::new(
    webxr::handlers::pipeline_admin_handler::PipelineAdminState {
        pipeline_service: app_state_data.pipeline_service.clone(),
        event_bus: app_state_data.pipeline_event_bus.clone(),
        paused: app_state_data.pipeline_paused.clone(),
        pause_reason: app_state_data.pipeline_pause_reason.clone(),
    }
);
```

Add to app configuration:
```rust
.app_data(pipeline_admin_state.clone())
```

Register routes:
```rust
.configure(configure_pipeline_admin_routes)
```

---

## Pipeline Flow

### Automatic Execution

```
GitHub Sync → OWL Parse → unified.db → [TRIGGER]
                                            ↓
                                       Reasoning
                                            ↓
                                  Constraint Generation
                                            ↓
                                       GPU Upload
                                            ↓
                                    Client Visualization
```

### Event Bus

All pipeline stages emit events:
- `OntologyModifiedEvent`
- `ReasoningCompleteEvent`
- `ConstraintsGeneratedEvent`
- `GPUUploadCompleteEvent`
- `PositionsUpdatedEvent`
- `PipelineErrorEvent`

---

## Configuration

### SemanticPhysicsConfig

Default values:
```rust
SemanticPhysicsConfig {
    auto_trigger_reasoning: true,       // Auto-run on ontology change
    auto_generate_constraints: true,     // Auto-generate constraints
    constraint_strength: 1.0,            // Force multiplier
    use_gpu_constraints: true,           // Upload to GPU
    max_reasoning_depth: 10,             // Inference depth limit
    cache_inferences: true,              // Enable caching
}
```

---

## Actor Integration

### Required Actor Addresses

The pipeline service needs:
1. **OntologyActor** - OWL parsing and storage
2. **GraphServiceActor** - Graph data access ❌ DEPRECATED (Nov 2025) - Use unified_gpu_compute.rs
3. **ReasoningActor** - Inference execution (to be added)
4. **OntologyConstraintActor** - GPU constraint upload (to be added)

### Wiring Example

```rust
pipeline_service.set_ontology_actor(ontology_addr.clone());
pipeline_service.set_graph_actor(graph_addr.clone());
pipeline_service.set_graph_repository(repo.clone());
```

---

## Monitoring

### Log Output

```
[PipelineEventBus] Event published: OntologyModified (correlation: abc123)
[OntologyPipelineService] Triggering reasoning for 5 OWL files
[CustomReasoner] Generated 42 new inferences
[PipelineEventBus] Event published: ReasoningComplete (correlation: abc123)
```

### Metrics Collection

Metrics are tracked for:
- Reasoning latency (ms)
- Constraint generation latency (ms)
- GPU upload latency (ms)
- Error rates
- Cache hit rates

---

## Troubleshooting

### Issue: "Pipeline Service Not Initialized"
**Solution**: Ensure AppState fields are wired correctly.

### Issue: Queue Sizes Always Zero
**Solution**: Queues are placeholders until async task queues are implemented.

### Issue: Metrics Return Zeros
**Solution**: Metrics are placeholders until ReasoningActor integration is complete.

---

## Future Enhancements

### Planned Features
- [ ] Real-time metrics from ReasoningActor
- [ ] Queue depth tracking
- [ ] Performance profiling
- [ ] WebSocket event streaming
- [ ] Pipeline analytics dashboard

---

## References

- [Pipeline Events Service](../../src/services/pipeline_events.rs)
- [Pipeline Admin Handler](../../src/handlers/pipeline_admin_handler.rs)
- [Ontology Pipeline Service](../../src/services/ontology_pipeline_service.rs)
- [Integration Checklist (Historical)](../PIPELINE_INTEGRATION_CHECKLIST.md)
