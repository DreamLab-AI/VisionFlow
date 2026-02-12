---
title: OntologyReasoningService Integration Guide
description: This guide documents the complete implementation of the OntologyReasoningService in VisionFlow, including all integration points and usage patterns.
category: how-to
tags:
  - tutorial
  - backend
  - documentation
  - reference
  - visionflow
updated-date: 2025-12-18
difficulty-level: intermediate
---


# OntologyReasoningService Integration Guide

## Complete Implementation Summary

This guide documents the complete implementation of the OntologyReasoningService in VisionFlow, including all integration points and usage patterns.

## Files Created

### 1. Core Service Implementation

**File**: `/home/devuser/workspace/project/src/services/ontology-reasoning-service.rs`

Complete OWL EL++ reasoning service with:
- Full whelk-rs integration
- Axiom inference with confidence scores
- Class hierarchy computation
- Disjoint class detection
- Blake3-based caching
- Database persistence

**Key Methods**:
```rust
pub async fn infer-axioms(&self, ontology-id: &str) -> Result<Vec<InferredAxiom>>
pub async fn get-class-hierarchy(&self, ontology-id: &str) -> Result<ClassHierarchy>
pub async fn get-disjoint-classes(&self, ontology-id: &str) -> Result<Vec<DisjointPair>>
pub async fn clear-cache(&self)
```

### 2. Database Migration

**File**: `/home/devuser/workspace/project/migration/003-add-inference-cache.sql`

Creates:
- `inference-cache` table for storing reasoning results
- `user-defined` column in `owl-axioms` to distinguish inferred vs explicit axioms
- Indexes for efficient querying
- View for monitoring expired cache entries

### 3. Documentation

**Files**:
- `/home/devuser/workspace/project/docs/ontology-reasoning-service.md`
- `/home/devuser/workspace/project/docs/ontology-reasoning-integration-guide.md`

## Integration Points

### 1. Service Registration

**File**: `src/services/mod.rs`

```rust
pub mod ontology-reasoning-service;
```

The service is now exported and available for use throughout the application.

### 2. OntologyActor Integration

**File**: `src/actors/ontology-actor.rs`

The `TriggerReasoning` message handler has been updated with TODO comments for complete integration:

```rust
impl Handler<TriggerReasoning> for OntologyActor {
    // TODO: Add OntologyReasoningService to OntologyActor state
    // TODO: Call reasoning-service.infer-axioms(&ontology-id).await
    // TODO: Broadcast OntologyUpdated event to EventBus
    // TODO: Store inferred axioms with user-defined=false
}
```

**Required Changes for Full Integration**:

1. Add service to OntologyActor state:
```rust
pub struct OntologyActor {
    // ... existing fields
    reasoning-service: Option<Arc<OntologyReasoningService>>,
}
```

2. Update constructor:
```rust
pub fn with-reasoning-service(
    config: OntologyActorConfig,
    reasoning-service: Arc<OntologyReasoningService>,
) -> Self {
    Self {
        // ... existing fields
        reasoning-service: Some(reasoning-service),
    }
}
```

3. Implement handler:
```rust
impl Handler<TriggerReasoning> for OntologyActor {
    fn handle(&mut self, msg: TriggerReasoning, -ctx: &mut Self::Context) -> Self::Result {
        let reasoning-service = self.reasoning-service.clone();
        let ontology-id = msg.ontology-id.to-string();

        Box::pin(async move {
            if let Some(service) = reasoning-service {
                // Run inference
                let inferred = service.infer-axioms(&ontology-id).await
                    .map-err(|e| format!("Inference failed: {}", e))?;

                info!("Inferred {} new axioms", inferred.len());

                // TODO: Broadcast OntologyUpdated event

                Ok(format!("Inferred {} axioms", inferred.len()))
            } else {
                Ok("Reasoning service not configured".to-string())
            }
        })
    }
}
```

### 3. GitHub Sync Service Integration

**File**: `src/services/github-sync-service.rs`

The `save-ontology-data()` method already triggers a reasoning pipeline (line 599-640):

```rust
// 🔥 TRIGGER REASONING PIPELINE if configured
if let Some(pipeline) = &self.pipeline-service {
    info!("🔄 Triggering ontology reasoning pipeline after ontology save");
    // ... existing pipeline trigger
}
```

**Note**: The existing `pipeline-service` appears to be a custom reasoner. The new `OntologyReasoningService` using whelk-rs can:
- Replace the existing pipeline (recommended for EL++ compliance)
- Work alongside it (for comparison/validation)
- Be used as a fallback if pipeline is not configured

**Recommended Integration**:

```rust
// Option 1: Replace existing pipeline
async fn save-ontology-data(&self, onto-data: OntologyData) -> Result<(), String> {
    self.onto-repo.save-ontology(...).await?;

    // Use new OntologyReasoningService
    if let Some(reasoning-service) = &self.reasoning-service {
        reasoning-service.infer-axioms("default").await
            .map-err(|e| format!("Reasoning failed: {}", e))?;
    }

    Ok(())
}

// Option 2: Use both (comparison mode)
async fn save-ontology-data(&self, onto-data: OntologyData) -> Result<(), String> {
    self.onto-repo.save-ontology(...).await?;

    // Existing pipeline
    if let Some(pipeline) = &self.pipeline-service {
        pipeline.trigger().await?;
    }

    // New whelk-rs reasoning
    if let Some(reasoning-service) = &self.reasoning-service {
        reasoning-service.infer-axioms("default").await?;
    }

    Ok(())
}
```

## Data Flow

```
GitHub Markdown Files
        ↓
GitHubSyncService::process-files()
        ↓
OntologyParser::parse()
        ↓
save-ontology-data()
        ↓
OntologyRepository::save-ontology()
        ↓
OntologyReasoningService::infer-axioms()
        ↓
WhelkInferenceEngine::infer()
        ↓
Store inferred axioms (user-defined=false)
        ↓
Cache results in inference-cache table
        ↓
Broadcast OntologyUpdated event
```

## Usage Examples

### Basic Usage

```rust
use std::sync::Arc;
use crate::adapters::whelk-inference-engine::WhelkInferenceEngine;
use crate::ports::ontology-repository::OntologyRepository;
use crate::services::ontology-reasoning-service::OntologyReasoningService;

// Initialize
let engine = Arc::new(WhelkInferenceEngine::new());
let repo: Arc<dyn OntologyRepository> = /* obtain from DI */;
let service = OntologyReasoningService::new(engine, repo);

// Infer axioms
let axioms = service.infer-axioms("default").await?;
println!("Inferred {} axioms", axioms.len());

// Get hierarchy
let hierarchy = service.get-class-hierarchy("default").await?;
println!("Root classes: {:?}", hierarchy.root-classes);

// Find disjoint classes
let disjoint = service.get-disjoint-classes("default").await?;
println!("Found {} disjoint pairs", disjoint.len());
```

### Integration with Actor System

```rust
// In app initialization
let reasoning-service = Arc::new(OntologyReasoningService::new(engine, repo));
let ontology-actor = OntologyActor::with-reasoning-service(
    OntologyActorConfig::default(),
    reasoning-service.clone(),
).start();

// Trigger reasoning via message
ontology-actor.do-send(TriggerReasoning {
    ontology-id: 1,
    source: "github-sync".to-string(),
});
```

### Querying Inferred vs Explicit Axioms

```rust
// Get all axioms
let all-axioms = repo.get-axioms().await?;

// Filter inferred axioms
let inferred: Vec<-> = all-axioms.iter()
    .filter(|a| a.annotations.get("inferred") == Some(&"true".to-string()))
    .collect();

// Filter explicit axioms
let explicit: Vec<-> = all-axioms.iter()
    .filter(|a| a.annotations.get("inferred") != Some(&"true".to-string()))
    .collect();

println!("Explicit: {}, Inferred: {}", explicit.len(), inferred.len());
```

## Testing

### Unit Tests

```bash
# Run all reasoning service tests
cargo test --package webxr --lib services::ontology-reasoning-service

# Run specific test
cargo test --package webxr --lib services::ontology-reasoning-service::tests::test-infer-axioms
```

### Integration Tests

```bash
# Test full pipeline
cargo test --package webxr --test ontology-integration-test
```

### Manual Testing

```rust
// Create test ontology
let classes = vec![
    OwlClass {
        iri: "http://example.org/Person".to-string(),
        label: Some("Person".to-string()),
        parent-classes: vec![],
        // ...
    },
    OwlClass {
        iri: "http://example.org/Employee".to-string(),
        label: Some("Employee".to-string()),
        parent-classes: vec!["http://example.org/Person".to-string()],
        // ...
    },
];

let axioms = vec![
    OwlAxiom {
        axiom-type: AxiomType::SubClassOf,
        subject: "http://example.org/Employee".to-string(),
        object: "http://example.org/Person".to-string(),
        // ...
    },
];

// Save and infer
repo.save-ontology(&classes, &[], &axioms).await?;
let inferred = service.infer-axioms("test").await?;

// Verify inferred axioms
assert!(inferred.len() > 0);
```

## Performance Tuning

### Cache Configuration

The service uses in-memory LRU cache with database persistence:

```rust
// Cache entry structure
struct InferenceCacheEntry {
    ontology-id: String,
    ontology-checksum: String,  // Blake3 hash
    inferred-axioms: Vec<InferredAxiom>,
    timestamp: DateTime<Utc>,
    inference-time-ms: u64,
}
```

**Tuning Parameters**:
- Cache size: Configurable in-memory cache (default: unlimited)
- TTL: 7 days for database cache (configurable in migration)
- Invalidation: Automatic on ontology changes via checksum

### Database Optimization

```sql
-- Add index for faster inference cache lookups
CREATE INDEX idx-inference-cache-ontology-checksum
    ON inference-cache(ontology-id, ontology-checksum);

-- Clean up old cache entries
DELETE FROM inference-cache
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
RUST-LOG=webxr::services::ontology-reasoning-service=debug cargo run

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
service.clear-cache().await;

// Verify checksum calculation
let checksum1 = service.calculate-ontology-checksum("default").await?;
// Modify ontology
let checksum2 = service.calculate-ontology-checksum("default").await?;
assert-ne!(checksum1, checksum2);
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
    ontology-id,
    COUNT(*) as entries,
    AVG(inference-time-ms) as avg-time,
    MAX(inference-time-ms) as max-time
FROM inference-cache
GROUP BY ontology-id;

-- Verify indexes exist
.indexes owl-axioms
.indexes inference-cache
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
    pub async fn explain-inference(
        &self,
        axiom-id: &str,
    ) -> Result<InferenceExplanation>;

    /// Incremental update (only changed portions)
    pub async fn infer-axioms-incremental(
        &self,
        ontology-id: &str,
        changed-iris: &[String],
    ) -> Result<Vec<InferredAxiom>>;

    /// Batch inference for multiple ontologies
    pub async fn infer-axioms-batch(
        &self,
        ontology-ids: &[String],
    ) -> Result<HashMap<String, Vec<InferredAxiom>>>;
}
```

## References

- 
- [OWL 2 EL Profile](https://www.w3.org/TR/owl2-profiles/#OWL-2-EL)
- [whelk-rs Documentation](https://github.com/balhoff/whelk-rs)
- 

---

## Related Documentation

- [VisionFlow Guides](index.md)
- [Natural Language Queries Tutorial](features/natural-language-queries.md)
- [Intelligent Pathfinding Guide](features/intelligent-pathfinding.md)
- [Goalie Integration - Goal-Oriented AI Research](infrastructure/goalie-integration.md)
- [Troubleshooting Guide](infrastructure/troubleshooting.md)

## Contact

For questions or issues:
- Check logs: `tail -f logs/visionflow.log`
- GitHub Issues: Create issue with `reasoning` label
- Documentation: See `docs/ontology-reasoning-service.md`
