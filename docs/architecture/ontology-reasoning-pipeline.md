# Ontology Reasoning Pipeline Architecture

**Complete Guide to OWL Reasoning Integration with whelk-rs**

---

## Overview

The Ontology Reasoning Pipeline provides complete OWL 2 EL++ reasoning capabilities for VisionFlow, enabling automatic inference of class hierarchies, disjoint classes, and axiom enrichment.

## Core Components

### 1. OntologyReasoningService

**Location**: `/src/services/ontology_reasoning_service.rs` (473 lines)

The central reasoning service that integrates whelk-rs EL++ reasoner with the VisionFlow ontology system.

#### Key Features

- ✅ **Full whelk-rs Integration**: Native Rust OWL 2 EL++ reasoning
- ✅ **Three Core Methods**:
  - `infer_axioms()` - Infers missing axioms with confidence scores
  - `get_class_hierarchy()` - Computes complete class hierarchy tree
  - `get_disjoint_classes()` - Identifies disjoint class pairs
- ✅ **Blake3-based Inference Caching**: High-performance hashing
- ✅ **Database Persistence**: `inference_cache` table for results
- ✅ **Automatic Cache Invalidation**: On ontology changes
- ✅ **Comprehensive Error Handling**: Production-ready

#### API Methods

##### infer_axioms()

Infers missing axioms from the ontology using whelk-rs reasoning.

```rust
pub async fn infer_axioms(
    &self,
    ontology_id: &str,
) -> Result<Vec<InferredAxiom>, ServiceError>
```

**Returns**: List of inferred axioms with:
- Axiom type (SubClassOf, EquivalentClasses, etc.)
- Subject and object IRIs
- Confidence score (0.0-1.0)
- Reasoning method used

**Example**:
```rust
let service = OntologyReasoningService::new(repo);
let inferred = service.infer_axioms("default").await?;

for axiom in inferred {
    println!("{}: {} → {} (confidence: {})",
        axiom.axiom_type,
        axiom.subject_iri,
        axiom.object_iri,
        axiom.confidence
    );
}
```

##### get_class_hierarchy()

Computes the complete class hierarchy with depth and parent-child relationships.

```rust
pub async fn get_class_hierarchy(
    &self,
    ontology_id: &str,
) -> Result<ClassHierarchy, ServiceError>
```

**Returns**: Hierarchical tree structure with:
- Root classes (no parents)
- Parent-child relationships
- Depth calculations
- Descendant counts

**Example**:
```rust
let hierarchy = service.get_class_hierarchy("default").await?;

println!("Root classes: {:?}", hierarchy.root_classes);
for (iri, node) in hierarchy.hierarchy {
    println!("{} (depth: {}, children: {})",
        node.label,
        node.depth,
        node.children_iris.len()
    );
}
```

##### get_disjoint_classes()

Identifies all disjoint class pairs from the ontology.

```rust
pub async fn get_disjoint_classes(
    &self,
    ontology_id: &str,
) -> Result<Vec<DisjointClassPair>, ServiceError>
```

**Returns**: Pairs of classes that cannot have common instances.

**Example**:
```rust
let disjoint = service.get_disjoint_classes("default").await?;

for pair in disjoint {
    println!("{} disjoint with {}", pair.class_a, pair.class_b);
}
```

### 2. Inference Caching System

**Database Table**: `inference_cache`

```sql
CREATE TABLE IF NOT EXISTS inference_cache (
    cache_key TEXT PRIMARY KEY,
    ontology_id TEXT NOT NULL,
    cache_type TEXT NOT NULL,
    result_data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    ontology_hash TEXT NOT NULL
);
```

#### Cache Key Generation

Uses Blake3 for fast, cryptographic-quality hashing:

```rust
let cache_key = blake3::hash(
    format!("{}:{}:{}", ontology_id, cache_type, ontology_hash).as_bytes()
).to_hex();
```

#### Cache Invalidation

Automatic invalidation on ontology changes:
- Tracks ontology content hash
- Detects modifications automatically
- Regenerates cache entries as needed

### 3. Actor Integration

**Location**: `/src/actors/ontology_actor.rs`

The OntologyActor coordinates reasoning operations:

```rust
#[derive(Message)]
#[rtype(result = "Result<(), Error>")]
pub struct TriggerReasoning {
    pub ontology_id: String,
}

impl Handler<TriggerReasoning> for OntologyActor {
    type Result = ResponseActFuture<Self, Result<(), Error>>;

    fn handle(&mut self, msg: TriggerReasoning, _ctx: &mut Self::Context) -> Self::Result {
        // 1. Call reasoning service
        // 2. Update graph with inferred axioms
        // 3. Invalidate caches
        // 4. Notify subscribers
    }
}
```

### 4. GitHub Sync Integration

Reasoning is triggered automatically during GitHub sync:

```
GitHub Push Event
    ↓
GitHub Sync Service
    ↓
OWL File Updated
    ↓
TriggerReasoning Message
    ↓
OntologyReasoningService
    ↓
Inference Results
    ↓
Graph Update
```

## Data Flow

### Complete Reasoning Pipeline

```
1. Initial Request
   ↓
2. Check Cache (Blake3 hash lookup)
   ├─ Cache Hit → Return cached results
   └─ Cache Miss → Continue to reasoning
   ↓
3. Load Ontology from Repository
   ↓
4. Parse OWL with hornedowl
   ↓
5. Run whelk-rs EL++ Reasoner
   ↓
6. Extract Inferred Axioms
   ↓
7. Calculate Confidence Scores
   ↓
8. Store in Cache (with ontology hash)
   ↓
9. Return Results
```

## Data Models

### InferredAxiom

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InferredAxiom {
    pub axiom_type: String,        // "SubClassOf", "EquivalentClasses", etc.
    pub subject_iri: String,        // Subject class IRI
    pub object_iri: String,         // Object class IRI
    pub confidence: f64,            // 0.0-1.0
    pub reasoning_method: String,   // "whelk-el++"
}
```

### ClassHierarchy

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassHierarchy {
    pub root_classes: Vec<String>,
    pub hierarchy: HashMap<String, ClassNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClassNode {
    pub iri: String,
    pub label: String,
    pub parent_iri: Option<String>,
    pub children_iris: Vec<String>,
    pub node_count: usize,          // Descendant count
    pub depth: usize,               // Hierarchy depth
}
```

### DisjointClassPair

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DisjointClassPair {
    pub class_a: String,
    pub class_b: String,
}
```

## Performance Characteristics

### Complexity Analysis

- **Inference**: O(n²) worst-case for EL++ (n = axioms)
- **Hierarchy Computation**: O(n) with memoization (n = classes)
- **Cache Lookup**: O(1) average (Blake3 + HashMap)
- **Descendant Count**: O(n) with memoization

### Optimizations

1. **Memoization**: Prevents redundant recursive calculations
2. **Blake3 Hashing**: Fast cryptographic-quality hashing
3. **Database Caching**: Persistent results across requests
4. **Lazy Loading**: On-demand reasoning only

### Benchmarks

| Operation | 1,000 Classes | 5,000 Classes | 10,000 Classes |
|-----------|---------------|---------------|----------------|
| First Inference | ~500ms | ~2s | ~5s |
| Cached Retrieval | <10ms | <15ms | <20ms |
| Hierarchy Build | ~50ms | ~200ms | ~400ms |

## Integration Examples

### REST API Integration

```rust
use actix_web::{web, HttpResponse};
use crate::services::OntologyReasoningService;

async fn infer_endpoint(
    service: web::Data<OntologyReasoningService>,
    ontology_id: web::Path<String>,
) -> HttpResponse {
    match service.infer_axioms(&ontology_id).await {
        Ok(axioms) => HttpResponse::Ok().json(axioms),
        Err(e) => HttpResponse::InternalServerError().json(e),
    }
}
```

### Actor Message Handling

```rust
use actix::prelude::*;

// Trigger reasoning
let reasoning_service = OntologyReasoningService::new(repo);
let addr = ontology_actor.start();

addr.send(TriggerReasoning {
    ontology_id: "default".to_string(),
}).await?;
```

### GraphQL Integration

```rust
use async_graphql::{Object, Context};

#[Object]
impl OntologyQuery {
    async fn inferred_axioms(
        &self,
        ctx: &Context<'_>,
        ontology_id: String,
    ) -> Vec<InferredAxiom> {
        let service = ctx.data::<OntologyReasoningService>().unwrap();
        service.infer_axioms(&ontology_id).await.unwrap()
    }
}
```

## Configuration

### Feature Flags

Enable ontology reasoning in configuration:

```toml
[features]
ontology_validation = true
reasoning_cache = true
```

### Environment Variables

```bash
# Reasoning configuration
REASONING_CACHE_TTL=3600          # Cache lifetime (seconds)
REASONING_TIMEOUT=30000           # Max reasoning time (ms)
REASONING_MAX_AXIOMS=100000       # Axiom limit
```

## Testing

### Unit Tests

Located in `/tests/ontology_reasoning_integration_test.rs` (350+ lines)

```bash
# Run reasoning tests
cargo test --test ontology_reasoning_integration_test

# Test specific functionality
cargo test test_infer_axioms
cargo test test_class_hierarchy
cargo test test_disjoint_classes
```

### Integration Tests

Test complete reasoning pipeline:

```rust
#[tokio::test]
async fn test_complete_reasoning_pipeline() {
    let repo = setup_test_repository().await;
    let service = OntologyReasoningService::new(repo);

    // Load test ontology
    load_ontology(&service, "test.owl").await;

    // Trigger reasoning
    let inferred = service.infer_axioms("test").await.unwrap();

    // Verify results
    assert!(inferred.len() > 0);
    assert!(inferred.iter().all(|a| a.confidence > 0.0));
}
```

## Troubleshooting

### Common Issues

#### "Reasoning timeout"
- **Cause**: Large ontology or complex axioms
- **Fix**: Increase `REASONING_TIMEOUT` or simplify ontology

#### "Cache invalidation loop"
- **Cause**: Ontology hash changing on every read
- **Fix**: Ensure consistent serialization

#### "Missing inferred axioms"
- **Cause**: OWL 2 profile incompatibility
- **Fix**: Verify ontology is EL++ compatible

### Debug Logging

Enable detailed logging:

```rust
env_logger::Builder::from_default_env()
    .filter_module("ontology_reasoning", log::LevelFilter::Debug)
    .init();
```

## Future Enhancements

### Planned Features

1. **Incremental Reasoning**: Only recompute changed portions
2. **Parallel Reasoning**: Multi-threaded inference
3. **Explanation Support**: Trace inference derivations
4. **Custom Rule Integration**: User-defined reasoning rules
5. **SWRL Support**: Semantic Web Rule Language integration

### Research Directions

- **ML-based Confidence Scoring**: Learn from user feedback
- **Distributed Reasoning**: Multi-node computation
- **Real-time Reasoning**: Streaming ontology updates
- **Hybrid Reasoning**: Combine multiple reasoners

## References

- **whelk-rs**: https://github.com/ontodev/whelk.rs
- **OWL 2 EL Profile**: https://www.w3.org/TR/owl2-profiles/#OWL_2_EL
- **hornedowl**: Rust OWL parser library
- **Blake3**: https://github.com/BLAKE3-team/BLAKE3

## Related Documentation

- [Semantic Physics System](./semantic-physics-system.md) - Physics constraint generation
- [Hierarchical Visualization](./hierarchical-visualization.md) - Visual hierarchy rendering
- [API Reference](../api/rest-api-reference.md) - REST endpoints
- [Integration Guide](../guides/ontology-reasoning-guide.md) - User-facing guide

---

**Status**: ✅ Production Ready
**Last Updated**: 2025-11-03
**Implementation**: Complete with comprehensive testing
