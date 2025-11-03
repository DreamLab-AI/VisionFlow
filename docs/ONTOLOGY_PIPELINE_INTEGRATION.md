# Ontology Pipeline Integration

## Overview

This document describes the end-to-end data pipeline that connects GitHub synchronization, ontology reasoning, constraint generation, and GPU-accelerated semantic physics.

## Architecture Diagram

```
GitHub Sync → Parse Ontology → Save to unified.db → Trigger Reasoning →
Cache Inferences → Generate Constraints → Upload to GPU →
Apply Semantic Forces → Stream to Client → Render Hierarchy
```

## Component Overview

### 1. GitHubSyncService
**Location**: `src/services/github_sync_service.rs`

**Responsibilities**:
- Fetch markdown files from GitHub repository
- Parse knowledge graph data (nodes/edges) from public pages
- Extract and parse ontology blocks (`### OntologyBlock`)
- Save graph data to `unified.db`
- **NEW**: Trigger ontology reasoning pipeline after ontology save

**Key Methods**:
- `sync_graphs()` - Main synchronization entry point
- `process_single_file()` - Parse individual markdown file
- `save_ontology_data()` - **Triggers pipeline on ontology modifications**

### 2. OntologyPipelineService
**Location**: `src/services/ontology_pipeline_service.rs`

**Responsibilities**:
- Orchestrate the complete ontology → physics pipeline
- Configure automatic reasoning and constraint generation
- Manage integration between all components

**Configuration** (`SemanticPhysicsConfig`):
```rust
pub struct SemanticPhysicsConfig {
    pub auto_trigger_reasoning: bool,      // Enable auto-reasoning
    pub auto_generate_constraints: bool,    // Auto-generate constraints
    pub constraint_strength: f32,           // Strength multiplier (0-10)
    pub use_gpu_constraints: bool,          // Use GPU acceleration
    pub max_reasoning_depth: usize,         // Max inference depth
    pub cache_inferences: bool,             // Cache reasoning results
}
```

**Pipeline Flow**:
```rust
on_ontology_modified(ontology_id, ontology) {
    1. trigger_reasoning()
       ↓
    2. generate_constraints_from_axioms()
       ↓
    3. upload_constraints_to_gpu()
}
```

### 3. ReasoningActor
**Location**: `src/reasoning/reasoning_actor.rs`

**Responsibilities**:
- Execute ontology reasoning (RDFS, OWL inference)
- Cache inferred axioms in SQLite (`inference_cache.db`)
- Return inferred relationships (SubClassOf, EquivalentClass, etc.)

**Message Handling**:
- `TriggerReasoning` - Start reasoning process
- `GetInferredAxioms` - Retrieve cached results
- `InvalidateCache` - Clear cache for specific ontology

### 4. OntologyConstraintActor
**Location**: `src/actors/gpu/ontology_constraint_actor.rs`

**Responsibilities**:
- Convert OWL axioms to physics constraints
- Upload constraint buffers to GPU
- Manage constraint lifecycle (add/remove/update)

**Constraint Types** (from inferred axioms):

| Axiom Type | Constraint Type | Effect |
|------------|----------------|--------|
| `SubClassOf(A, B)` | Clustering | Nodes of class A cluster near B nodes |
| `EquivalentClass(A, B)` | Alignment | A and B nodes align (stronger force) |
| `DisjointWith(A, B)` | Separation | A and B nodes repel (2x strength) |

### 5. GPU Compute Pipeline
**Location**: `src/gpu/unified_gpu_compute.rs` (referenced, not in this repo)

**Responsibilities**:
- Execute force computations on GPU
- Apply semantic constraints to node positions
- Stream position updates to WebSocket clients

## Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│ 1. GitHub Sync                                                 │
├────────────────────────────────────────────────────────────────┤
│ GitHubSyncService.sync_graphs()                                │
│   ↓                                                             │
│ process_single_file("page.md")                                 │
│   ↓                                                             │
│ OntologyParser.parse(content) → OntologyData                   │
│   ↓                                                             │
│ save_ontology_data(onto_data)                                  │
│   → UnifiedOntologyRepository.save_ontology()                  │
│   → Saves classes, properties, axioms to unified.db            │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. Trigger Reasoning Pipeline                                  │
├────────────────────────────────────────────────────────────────┤
│ if pipeline_service configured:                                │
│   Convert OntologyData → Ontology struct                       │
│   pipeline.on_ontology_modified(ontology_id, ontology)         │
│     → Spawns async task to avoid blocking sync                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Reasoning Execution                                         │
├────────────────────────────────────────────────────────────────┤
│ OntologyPipelineService.trigger_reasoning()                    │
│   ↓                                                             │
│ ReasoningActor.send(TriggerReasoning {                         │
│   ontology_id, ontology                                        │
│ })                                                              │
│   ↓                                                             │
│ InferenceCache.get_or_compute()                                │
│   → Check cache for ontology_id                                │
│   → If miss: CustomReasoner.infer()                            │
│   → Store results in inference_cache.db                        │
│   ↓                                                             │
│ Returns: Vec<InferredAxiom>                                    │
│   Example: [                                                   │
│     InferredAxiom {                                            │
│       axiom_type: "SubClassOf",                                │
│       subject: "Engineer",                                     │
│       object: "Person"                                         │
│     }                                                           │
│   ]                                                             │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Constraint Generation                                       │
├────────────────────────────────────────────────────────────────┤
│ OntologyPipelineService.generate_constraints_from_axioms()     │
│   For each InferredAxiom:                                      │
│     match axiom_type:                                          │
│       "SubClassOf" → ConstraintType::Clustering                │
│       "EquivalentClass" → ConstraintType::Alignment            │
│       "DisjointWith" → ConstraintType::Separation              │
│   ↓                                                             │
│ Returns: ConstraintSet {                                       │
│   constraints: Vec<Constraint>,                                │
│   metadata: { source, axiom_count, timestamp }                 │
│ }                                                               │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. GPU Upload                                                  │
├────────────────────────────────────────────────────────────────┤
│ OntologyPipelineService.upload_constraints_to_gpu()            │
│   ↓                                                             │
│ OntologyConstraintActor.send(ApplyOntologyConstraints {        │
│   constraint_set,                                              │
│   merge_mode: ConstraintMergeMode::Merge,                      │
│   graph_id: 0                                                  │
│ })                                                              │
│   ↓                                                             │
│ OntologyConstraintTranslator.translate_axioms_to_constraints() │
│   → Convert ConstraintSet to GPU buffer format                 │
│   → Upload to CUDA constraint buffer                           │
│   ↓                                                             │
│ GPU now has semantic constraints applied                       │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 6. Physics Simulation                                          │
├────────────────────────────────────────────────────────────────┤
│ ForceComputeActor runs GPU kernels:                            │
│   1. compute_forces_kernel()                                   │
│      → Apply repulsion, attraction, damping                    │
│   2. apply_ontology_constraints_kernel()                       │
│      → Apply semantic clustering/alignment/separation          │
│   3. integrate_forces_kernel()                                 │
│      → Update node positions and velocities                    │
│   ↓                                                             │
│ Results: Updated node positions                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 7. Client Streaming                                            │
├────────────────────────────────────────────────────────────────┤
│ GraphServiceActor broadcasts to ClientManager                  │
│   ↓                                                             │
│ WebSocket clients receive position updates                     │
│   → Real-time graph visualization with semantic physics        │
└────────────────────────────────────────────────────────────────┘
```

## Configuration

### Enable Semantic Physics Pipeline

**In your application initialization**:

```rust
use crate::services::ontology_pipeline_service::{
    OntologyPipelineService, SemanticPhysicsConfig
};

// Create configuration
let config = SemanticPhysicsConfig {
    auto_trigger_reasoning: true,
    auto_generate_constraints: true,
    constraint_strength: 1.5,  // Adjust strength
    use_gpu_constraints: true,
    max_reasoning_depth: 10,
    cache_inferences: true,
};

// Create pipeline service
let pipeline = Arc::new(OntologyPipelineService::new(config));

// Register actor addresses
pipeline.set_reasoning_actor(reasoning_actor_addr);
pipeline.set_ontology_actor(ontology_actor_addr);
pipeline.set_graph_actor(graph_service_addr);
pipeline.set_constraint_actor(constraint_actor_addr);

// Attach to GitHub sync service
github_sync.set_pipeline_service(Arc::clone(&pipeline));
```

### Disable Automatic Reasoning

```rust
let config = SemanticPhysicsConfig {
    auto_trigger_reasoning: false,  // Disable
    ..Default::default()
};
```

### Manual Reasoning Trigger

```rust
// Manually trigger reasoning for specific ontology
let stats = pipeline.on_ontology_modified(ontology_id, ontology).await?;

println!("Inferred {} axioms", stats.inferred_axioms_count);
println!("Generated {} constraints", stats.constraints_generated);
println!("GPU upload: {}", stats.gpu_upload_success);
```

## Constraint Strength Tuning

The `constraint_strength` parameter acts as a multiplier for all semantic constraints:

```rust
config.constraint_strength = 0.5;  // Subtle semantic clustering
config.constraint_strength = 1.0;  // Default balanced forces
config.constraint_strength = 2.0;  // Strong semantic grouping
config.constraint_strength = 5.0;  // Very strong (may dominate layout)
```

**Per-axiom strength modifiers** (applied after multiplier):
- `SubClassOf`: 1.0x
- `EquivalentClass`: 1.5x (stronger alignment)
- `DisjointWith`: 2.0x (strong separation)

## Error Handling

The pipeline includes comprehensive error handling at each stage:

1. **Reasoning Failure**: Logs error, returns early (no constraints applied)
2. **Constraint Generation Failure**: Logs error, skips GPU upload
3. **GPU Upload Failure**: Logs error but doesn't block sync
4. **Async Task Spawning**: Pipeline runs in background, doesn't block sync

**All errors are logged** with emoji prefixes for visibility:
- ✅ Success events
- ❌ Error events
- 🔄 Processing events
- 📤 Upload events
- 🧠 Reasoning events

## Cache Management

### Inference Cache

**Location**: `.swarm/inference_cache.db` (SQLite)

**Schema**:
```sql
CREATE TABLE inference_cache (
    ontology_id INTEGER PRIMARY KEY,
    inferred_axioms BLOB,  -- Serialized Vec<InferredAxiom>
    computed_at INTEGER,   -- Unix timestamp
    cache_key TEXT
);
```

**Cache Invalidation**:
```rust
// Invalidate cache when ontology changes
reasoning_actor.send(InvalidateCache {
    ontology_id: 1
}).await?;
```

## Monitoring & Debugging

### Check Pipeline Statistics

```rust
let stats = pipeline.on_ontology_modified(ontology_id, ontology).await?;

info!("Pipeline Stats:");
info!("  Reasoning triggered: {}", stats.reasoning_triggered);
info!("  Axioms inferred: {}", stats.inferred_axioms_count);
info!("  Constraints generated: {}", stats.constraints_generated);
info!("  GPU upload success: {}", stats.gpu_upload_success);
info!("  Total time: {}ms", stats.total_time_ms);
```

### Enable Debug Logging

```rust
env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Debug)
    .init();
```

**Key log patterns**:
- `🔄 Triggering ontology reasoning pipeline` - Pipeline started
- `✅ Reasoning complete: X axioms` - Reasoning succeeded
- `🔧 Generating constraints from X axioms` - Constraint generation
- `📤 Uploading X constraints to GPU` - GPU upload started
- `✅ Constraints uploaded to GPU successfully` - Pipeline complete

## Performance Characteristics

- **GitHub Sync**: ~50 files/sec (batched)
- **Ontology Parsing**: ~10ms per file with ontology blocks
- **Reasoning**: ~100-500ms (cached: <1ms)
- **Constraint Generation**: ~1ms per 100 axioms
- **GPU Upload**: ~10-50ms for 1000 constraints
- **Total Pipeline**: ~200-600ms end-to-end

## Future Enhancements

1. **EventBus Integration**: Broadcast `OntologyUpdated` events
2. **Real-time WebSocket Notifications**: Notify clients of reasoning progress
3. **Incremental Reasoning**: Only re-infer changed portions of ontology
4. **Constraint Visualization**: Show active semantic constraints in UI
5. **Performance Metrics**: Track pipeline latency and throughput

## Troubleshooting

### Reasoning Not Triggered

**Check**:
1. Is `auto_trigger_reasoning` enabled in config?
2. Is pipeline service registered with GitHub sync?
   ```rust
   github_sync.set_pipeline_service(pipeline);
   ```
3. Are ontology blocks being detected? Look for log:
   ```
   🦉 Detected OntologyBlock in file.md
   ```

### Constraints Not Applied

**Check**:
1. Is `auto_generate_constraints` enabled?
2. Are axioms being inferred? Check log:
   ```
   ✅ Reasoning complete: X axioms
   ```
3. Is GPU upload successful? Look for:
   ```
   ✅ Constraints uploaded to GPU successfully
   ```

### GPU Upload Failures

**Check**:
1. Is constraint actor properly initialized?
2. Is GPU context available?
3. Check CUDA/GPU logs for memory issues

### Performance Issues

**Solutions**:
1. Reduce `max_reasoning_depth` (default: 10)
2. Enable `cache_inferences` to avoid re-computation
3. Reduce `constraint_strength` if too many constraints
4. Use incremental sync (SHA1 filtering) to skip unchanged files

## Related Documentation

- **Ontology Parsing**: See `src/services/parsers/ontology_parser.rs`
- **Reasoning Engine**: See `src/reasoning/custom_reasoner.rs`
- **Inference Cache**: See `src/reasoning/inference_cache.rs`
- **GPU Constraints**: See `src/actors/gpu/ontology_constraint_actor.rs`
- **GitHub Sync**: See `src/services/github_sync_service.rs`

## Contact & Support

For questions or issues with the ontology pipeline:
1. Check logs with `DEBUG` level enabled
2. Review this documentation
3. Examine the integration tests in `tests/`
