# OntologyReasoningService Integration Guide

## Complete Implementation Summary

This guide documents the complete implementation of the OntologyReasoningService in VisionFlow, including all integration points and usage patterns.

## Files Created

### 1. Core Service Implementation

**File**: `/home/devuser/workspace/project/src/services/ontology_reasoning_service.rs`

Complete OWL EL++ reasoning service with:
- Full whelk-rs integration
- Axiom inference with confidence scores
- Class hierarchy computation
- Disjoint class detection
- Blake3-based caching
- Database persistence

**Key Methods**:
```rust
pub async fn infer_axioms(&self, ontology_id: &str) -> Result<Vec<InferredAxiom>>
pub async fn get_class_hierarchy(&self, ontology_id: &str) -> Result<ClassHierarchy>
pub async fn get_disjoint_classes(&self, ontology_id: &str) -> Result<Vec<DisjointPair>>
pub async fn clear_cache(&self)
```

### 2. Database Migration

**File**: `/home/devuser/workspace/project/migration/003_add_inference_cache.sql`

Creates:
- `inference_cache` table for storing reasoning results
- `user_defined` column in `owl_axioms` to distinguish inferred vs explicit axioms
- Indexes for efficient querying
- View for monitoring expired cache entries

### 3. Documentation

**Files**:
- `/home/devuser/workspace/project/docs/ontology_reasoning_service.md`
- `/home/devuser/workspace/project/docs/ontology_reasoning_integration_guide.md`

## Integration Points

### 1. Service Registration

**File**: `src/services/mod.rs`

```rust
pub mod ontology_reasoning_service;
```

The service is now exported and available for use throughout the application.

### 2. OntologyActor Integration

**File**: `src/actors/ontology_actor.rs`

The `TriggerReasoning` message handler has been updated with TODO comments for complete integration:

```rust
impl Handler<TriggerReasoning> for OntologyActor {
    // TODO: Add OntologyReasoningService to OntologyActor state
    // TODO: Call reasoning_service.infer_axioms(&ontology_id).await
    // TODO: Broadcast OntologyUpdated event to EventBus
    // TODO: Store inferred axioms with user_defined=false
}
```

**Required Changes for Full Integration**:

1. Add service to OntologyActor state:
```rust
pub struct OntologyActor {
    // ... existing fields
    reasoning_service: Option<Arc<OntologyReasoningService>>,
}
```

2. Update constructor:
```rust
pub fn with_reasoning_service(
    config: OntologyActorConfig,
    reasoning_service: Arc<OntologyReasoningService>,
) -> Self {
    Self {
        // ... existing fields
        reasoning_service: Some(reasoning_service),
    }
}
```

3. Implement handler:
```rust
impl Handler<TriggerReasoning> for OntologyActor {
    fn handle(&mut self, msg: TriggerReasoning, _ctx: &mut Self::Context) -> Self::Result {
        let reasoning_service = self.reasoning_service.clone();
        let ontology_id = msg.ontology_id.to_string();

        Box::pin(async move {
            if let Some(service) = reasoning_service {
                // Run inference
                let inferred = service.infer_axioms(&ontology_id).await
                    .map_err(|e| format!("Inference failed: {}", e))?;

                info!("Inferred {} new axioms", inferred.len());

                // TODO: Broadcast OntologyUpdated event

                Ok(format!("Inferred {} axioms", inferred.len()))
            } else {
                Ok("Reasoning service not configured".to_string())
            }
        })
    }
}
```

### 3. GitHub Sync Service Integration

**File**: `src/services/github_sync_service.rs`

The `save_ontology_data()` method already triggers a reasoning pipeline (line 599-640):

```rust
// 🔥 TRIGGER REASONING PIPELINE if configured
if let Some(pipeline) = &self.pipeline_service {
    info!("🔄 Triggering ontology reasoning pipeline after ontology save");
    // ... existing pipeline trigger
}
```

**Note**: The existing `pipeline_service` appears to be a custom reasoner. The new `OntologyReasoningService` using whelk-rs can:
- Replace the existing pipeline (recommended for EL++ compliance)
- Work alongside it (for comparison/validation)
- Be used as a fallback if pipeline is not configured

**Recommended Integration**:

```rust
// Option 1: Replace existing pipeline
async fn save_ontology_data(&self, onto_data: OntologyData) -> Result<(), String> {
    self.onto_repo.save_ontology(...).await?;

    // Use new OntologyReasoningService
    if let Some(reasoning_service) = &self.reasoning_service {
        reasoning_service.infer_axioms("default").await
            .map_err(|e| format!("Reasoning failed: {}", e))?;
    }

    Ok(())
}

// Option 2: Use both (comparison mode)
async fn save_ontology_data(&self, onto_data: OntologyData) -> Result<(), String> {
    self.onto_repo.save_ontology(...).await?;

    // Existing pipeline
    if let Some(pipeline) = &self.pipeline_service {
        pipeline.trigger().await?;
    }

    // New whelk-rs reasoning
    if let Some(reasoning_service) = &self.reasoning_service {
        reasoning_service.infer_axioms("default").await?;
    }

    Ok(())
}
```

## Data Flow

```
GitHub Markdown Files
        ↓
GitHubSyncService::process_files()
        ↓
OntologyParser::parse()
        ↓
save_ontology_data()
        ↓
UnifiedOntologyRepository::save_ontology()
        ↓
OntologyReasoningService::infer_axioms()
        ↓
WhelkInferenceEngine::infer()
        ↓
Store inferred axioms (user_defined=false)
        ↓
Cache results in inference_cache table
        ↓
Broadcast OntologyUpdated event
```

## Usage Examples

### Basic Usage

```rust
use std::sync::Arc;
use crate::adapters::whelk_inference_engine::WhelkInferenceEngine;
use crate::repositories::unified_ontology_repository::UnifiedOntologyRepository;
use crate::services::ontology_reasoning_service::OntologyReasoningService;

// Initialize
let engine = Arc::new(WhelkInferenceEngine::new());
let repo = Arc::new(UnifiedOntologyRepository::new("data/unified.db")?);
let service = OntologyReasoningService::new(engine, repo);

// Infer axioms
let axioms = service.infer_axioms("default").await?;
println!("Inferred {} axioms", axioms.len());

// Get hierarchy
let hierarchy = service.get_class_hierarchy("default").await?;
println!("Root classes: {:?}", hierarchy.root_classes);

// Find disjoint classes
let disjoint = service.get_disjoint_classes("default").await?;
println!("Found {} disjoint pairs", disjoint.len());
```

### Integration with Actor System

```rust
// In app initialization
let reasoning_service = Arc::new(OntologyReasoningService::new(engine, repo));
let ontology_actor = OntologyActor::with_reasoning_service(
    OntologyActorConfig::default(),
    reasoning_service.clone(),
).start();

// Trigger reasoning via message
ontology_actor.do_send(TriggerReasoning {
    ontology_id: 1,
    source: "github_sync".to_string(),
});
```

### Querying Inferred vs Explicit Axioms

```rust
// Get all axioms
let all_axioms = repo.get_axioms().await?;

// Filter inferred axioms
let inferred: Vec<_> = all_axioms.iter()
    .filter(|a| a.annotations.get("inferred") == Some(&"true".to_string()))
    .collect();

// Filter explicit axioms
let explicit: Vec<_> = all_axioms.iter()
    .filter(|a| a.annotations.get("inferred") != Some(&"true".to_string()))
    .collect();

println!("Explicit: {}, Inferred: {}", explicit.len(), inferred.len());
```

## Testing

### Unit Tests

```bash
# Run all reasoning service tests
cargo test --package webxr --lib services::ontology_reasoning_service

# Run specific test
cargo test --package webxr --lib services::ontology_reasoning_service::tests::test_infer_axioms
```

### Integration Tests

```bash
# Test full pipeline
cargo test --package webxr --test ontology_integration_test
```

### Manual Testing

```rust
// Create test ontology
let classes = vec![
    OwlClass {
        iri: "http://example.org/Person".to_string(),
        label: Some("Person".to_string()),
        parent_classes: vec![],
        // ...
    },
    OwlClass {
        iri: "http://example.org/Employee".to_string(),
        label: Some("Employee".to_string()),
        parent_classes: vec!["http://example.org/Person".to_string()],
        // ...
    },
];

let axioms = vec![
    OwlAxiom {
        axiom_type: AxiomType::SubClassOf,
        subject: "http://example.org/Employee".to_string(),
        object: "http://example.org/Person".to_string(),
        // ...
    },
];

// Save and infer
repo.save_ontology(&classes, &[], &axioms).await?;
let inferred = service.infer_axioms("test").await?;

// Verify inferred axioms
assert!(inferred.len() > 0);
```

## Performance Tuning

### Cache Configuration

The service uses in-memory LRU cache with database persistence:

```rust
// Cache entry structure
struct InferenceCacheEntry {
    ontology_id: String,
    ontology_checksum: String,  // Blake3 hash
    inferred_axioms: Vec<InferredAxiom>,
    timestamp: DateTime<Utc>,
    inference_time_ms: u64,
}
```

**Tuning Parameters**:
- Cache size: Configurable in-memory cache (default: unlimited)
- TTL: 7 days for database cache (configurable in migration)
- Invalidation: Automatic on ontology changes via checksum

### Database Optimization

```sql
-- Add index for faster inference cache lookups
CREATE INDEX idx_inference_cache_ontology_checksum
    ON inference_cache(ontology_id, ontology_checksum);

-- Clean up old cache entries
DELETE FROM inference_cache
WHERE timestamp < (strftime('%s', 'now') - 604800);  -- 7 days

-- Vacuum after cleanup
VACUUM;
```

## Troubleshooting

### Issue: Inference Not Triggering

**Check**:
1. OntologyReasoningService is initialized
2. Service is passed to OntologyActor
3. TriggerReasoning message is being sent
4. No errors in logs

**Solution**:
```bash
# Enable debug logging
RUST_LOG=webxr::services::ontology_reasoning_service=debug cargo run

# Check for errors
grep "Reasoning" logs/visionflow.log
```

### Issue: Cache Not Invalidating

**Check**:
1. Ontology checksum is changing
2. Cache entries have correct timestamps
3. Database migration applied

**Solution**:
```rust
// Force cache clear
service.clear_cache().await;

// Verify checksum calculation
let checksum1 = service.calculate_ontology_checksum("default").await?;
// Modify ontology
let checksum2 = service.calculate_ontology_checksum("default").await?;
assert_ne!(checksum1, checksum2);
```

### Issue: Slow Inference

**Check**:
1. Ontology size (classes, axioms)
2. Cache hit rate
3. Database indexes

**Solution**:
```sql
-- Check cache performance
SELECT
    ontology_id,
    COUNT(*) as entries,
    AVG(inference_time_ms) as avg_time,
    MAX(inference_time_ms) as max_time
FROM inference_cache
GROUP BY ontology_id;

-- Verify indexes exist
.indexes owl_axioms
.indexes inference_cache
```

## Future Enhancements

### Planned Features

1. **Inference Explanation**
   - Track reasoning paths for each inferred axiom
   - Provide human-readable explanations
   - Export proof trees

2. **Incremental Reasoning**
   - Only recompute affected portions on changes
   - Delta-based inference updates
   - Faster response times

3. **Distributed Reasoning**
   - Partition ontology across multiple workers
   - Parallel inference computation
   - Aggregate results

4. **Advanced Caching**
   - Redis integration for distributed cache
   - Partial cache invalidation
   - Cache warming strategies

### API Extensions

```rust
// Planned methods
impl OntologyReasoningService {
    /// Explain why an axiom was inferred
    pub async fn explain_inference(
        &self,
        axiom_id: &str,
    ) -> Result<InferenceExplanation>;

    /// Incremental update (only changed portions)
    pub async fn infer_axioms_incremental(
        &self,
        ontology_id: &str,
        changed_iris: &[String],
    ) -> Result<Vec<InferredAxiom>>;

    /// Batch inference for multiple ontologies
    pub async fn infer_axioms_batch(
        &self,
        ontology_ids: &[String],
    ) -> Result<HashMap<String, Vec<InferredAxiom>>>;
}
```

## References

- [VisionFlow Architecture](./architecture.md)
- [OWL 2 EL Profile](https://www.w3.org/TR/owl2-profiles/#OWL_2_EL)
- [whelk-rs Documentation](https://github.com/balhoff/whelk-rs)
- [Unified Database Schema](../schema/README.md)

## Contact

For questions or issues:
- Check logs: `tail -f logs/visionflow.log`
- GitHub Issues: Create issue with `reasoning` label
- Documentation: See `docs/ontology_reasoning_service.md`
