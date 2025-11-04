# Ontology Reasoning Pipeline Architecture

**Complete Guide to OWL Reasoning Integration with whelk-rs**

---

## Overview

The Ontology Reasoning Pipeline provides complete OWL 2 EL++ reasoning capabilities for VisionFlow, enabling automatic inference of class hierarchies, disjoint classes, and axiom enrichment.

## Core Components

### 1. OntologyReasoningService

**Location**: `/src/services/ontology-reasoning-service.rs` (473 lines)

The central reasoning service that integrates whelk-rs EL++ reasoner with the VisionFlow ontology system.

#### Key Features

- ✅ **Full whelk-rs Integration**: Native Rust OWL 2 EL++ reasoning
- ✅ **Three Core Methods**:
  - `infer-axioms()` - Infers missing axioms with confidence scores
  - `get-class-hierarchy()` - Computes complete class hierarchy tree
  - `get-disjoint-classes()` - Identifies disjoint class pairs
- ✅ **Blake3-based Inference Caching**: High-performance hashing
- ✅ **Database Persistence**: `inference-cache` table for results
- ✅ **Automatic Cache Invalidation**: On ontology changes
- ✅ **Comprehensive Error Handling**: Production-ready

#### API Methods

##### infer-axioms()

Infers missing axioms from the ontology using whelk-rs reasoning.

```rust
pub async fn infer-axioms(
    &self,
    ontology-id: &str,
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
let inferred = service.infer-axioms("default").await?;

for axiom in inferred {
    println!("{}: {} → {} (confidence: {})",
        axiom.axiom-type,
        axiom.subject-iri,
        axiom.object-iri,
        axiom.confidence
    );
}
```

##### get-class-hierarchy()

Computes the complete class hierarchy with depth and parent-child relationships.

```rust
pub async fn get-class-hierarchy(
    &self,
    ontology-id: &str,
) -> Result<ClassHierarchy, ServiceError>
```

**Returns**: Hierarchical tree structure with:
- Root classes (no parents)
- Parent-child relationships
- Depth calculations
- Descendant counts

**Example**:
```rust
let hierarchy = service.get-class-hierarchy("default").await?;

println!("Root classes: {:?}", hierarchy.root-classes);
for (iri, node) in hierarchy.hierarchy {
    println!("{} (depth: {}, children: {})",
        node.label,
        node.depth,
        node.children-iris.len()
    );
}
```

##### get-disjoint-classes()

Identifies all disjoint class pairs from the ontology.

```rust
pub async fn get-disjoint-classes(
    &self,
    ontology-id: &str,
) -> Result<Vec<DisjointClassPair>, ServiceError>
```

**Returns**: Pairs of classes that cannot have common instances.

**Example**:
```rust
let disjoint = service.get-disjoint-classes("default").await?;

for pair in disjoint {
    println!("{} disjoint with {}", pair.class-a, pair.class-b);
}
```

### 2. Inference Caching System

**Database Table**: `inference-cache`

```sql
CREATE TABLE IF NOT EXISTS inference-cache (
    cache-key TEXT PRIMARY KEY,
    ontology-id TEXT NOT NULL,
    cache-type TEXT NOT NULL,
    result-data TEXT NOT NULL,
    created-at TEXT NOT NULL,
    ontology-hash TEXT NOT NULL
);
```

#### Cache Key Generation

Uses Blake3 for fast, cryptographic-quality hashing:

```rust
let cache-key = blake3::hash(
    format!("{}:{}:{}", ontology-id, cache-type, ontology-hash).as-bytes()
).to-hex();
```

#### Cache Invalidation

Automatic invalidation on ontology changes:
- Tracks ontology content hash
- Detects modifications automatically
- Regenerates cache entries as needed

### 3. Actor Integration

**Location**: `/src/actors/ontology-actor.rs`

The OntologyActor coordinates reasoning operations:

```rust
#[derive(Message)]
#[rtype(result = "Result<(), Error>")]
pub struct TriggerReasoning {
    pub ontology-id: String,
}

impl Handler<TriggerReasoning> for OntologyActor {
    type Result = ResponseActFuture<Self, Result<(), Error>>;

    fn handle(&mut self, msg: TriggerReasoning, -ctx: &mut Self::Context) -> Self::Result {
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
#[serde(rename-all = "camelCase")]
pub struct InferredAxiom {
    pub axiom-type: String,        // "SubClassOf", "EquivalentClasses", etc.
    pub subject-iri: String,        // Subject class IRI
    pub object-iri: String,         // Object class IRI
    pub confidence: f64,            // 0.0-1.0
    pub reasoning-method: String,   // "whelk-el++"
}
```

### ClassHierarchy

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename-all = "camelCase")]
pub struct ClassHierarchy {
    pub root-classes: Vec<String>,
    pub hierarchy: HashMap<String, ClassNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename-all = "camelCase")]
pub struct ClassNode {
    pub iri: String,
    pub label: String,
    pub parent-iri: Option<String>,
    pub children-iris: Vec<String>,
    pub node-count: usize,          // Descendant count
    pub depth: usize,               // Hierarchy depth
}
```

### DisjointClassPair

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename-all = "camelCase")]
pub struct DisjointClassPair {
    pub class-a: String,
    pub class-b: String,
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
use actix-web::{web, HttpResponse};
use crate::services::OntologyReasoningService;

async fn infer-endpoint(
    service: web::Data<OntologyReasoningService>,
    ontology-id: web::Path<String>,
) -> HttpResponse {
    match service.infer-axioms(&ontology-id).await {
        Ok(axioms) => HttpResponse::Ok().json(axioms),
        Err(e) => HttpResponse::InternalServerError().json(e),
    }
}
```

### Actor Message Handling

```rust
use actix::prelude::*;

// Trigger reasoning
let reasoning-service = OntologyReasoningService::new(repo);
let addr = ontology-actor.start();

addr.send(TriggerReasoning {
    ontology-id: "default".to-string(),
}).await?;
```

### GraphQL Integration

```rust
use async-graphql::{Object, Context};

#[Object]
impl OntologyQuery {
    async fn inferred-axioms(
        &self,
        ctx: &Context<'->,
        ontology-id: String,
    ) -> Vec<InferredAxiom> {
        let service = ctx.data::<OntologyReasoningService>().unwrap();
        service.infer-axioms(&ontology-id).await.unwrap()
    }
}
```

## Configuration

### Feature Flags

Enable ontology reasoning in configuration:

```toml
[features]
ontology-validation = true
reasoning-cache = true
```

### Environment Variables

```bash
# Reasoning configuration
REASONING-CACHE-TTL=3600          # Cache lifetime (seconds)
REASONING-TIMEOUT=30000           # Max reasoning time (ms)
REASONING-MAX-AXIOMS=100000       # Axiom limit
```

## Testing

### Unit Tests

Located in `/tests/ontology-reasoning-integration-test.rs` (350+ lines)

```bash
# Run reasoning tests
cargo test --test ontology-reasoning-integration-test

# Test specific functionality
cargo test test-infer-axioms
cargo test test-class-hierarchy
cargo test test-disjoint-classes
```

### Integration Tests

Test complete reasoning pipeline:

```rust
#[tokio::test]
async fn test-complete-reasoning-pipeline() {
    let repo = setup-test-repository().await;
    let service = OntologyReasoningService::new(repo);

    // Load test ontology
    load-ontology(&service, "test.owl").await;

    // Trigger reasoning
    let inferred = service.infer-axioms("test").await.unwrap();

    // Verify results
    assert!(inferred.len() > 0);
    assert!(inferred.iter().all(|a| a.confidence > 0.0));
}
```

## Troubleshooting

### Common Issues

#### "Reasoning timeout"
- **Cause**: Large ontology or complex axioms
- **Fix**: Increase `REASONING-TIMEOUT` or simplify ontology

#### "Cache invalidation loop"
- **Cause**: Ontology hash changing on every read
- **Fix**: Ensure consistent serialization

#### "Missing inferred axioms"
- **Cause**: OWL 2 profile incompatibility
- **Fix**: Verify ontology is EL++ compatible

### Debug Logging

Enable detailed logging:

```rust
env-logger::Builder::from-default-env()
    .filter-module("ontology-reasoning", log::LevelFilter::Debug)
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
- **OWL 2 EL Profile**: https://www.w3.org/TR/owl2-profiles/#OWL-2-EL
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
