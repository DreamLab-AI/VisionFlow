---
title: Ontology Reasoning
description: Understanding VisionFlow's OWL 2 EL++ reasoning pipeline for automatic inference and constraint generation
category: explanation
tags:
  - ontology
  - owl
  - reasoning
  - architecture
related-docs:
  - concepts/constraint-system.md
  - concepts/physics-engine.md
  - reference/database/ontology-schema-v2.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# Ontology Reasoning

VisionFlow integrates OWL 2 EL++ reasoning to automatically infer class hierarchies, disjoint relationships, and missing axioms from loaded ontologies.

---

## Core Concept

Ontology reasoning answers questions the ontology doesn't explicitly state:

- If `Dog SubClassOf Mammal` and `Mammal SubClassOf Animal`, then `Dog SubClassOf Animal` (inferred)
- If `Cat DisjointWith Dog`, instances of Cat cannot be instances of Dog
- If `hasPart` is transitive, part-of-part relationships propagate

These inferences enrich the constraint system, producing more meaningful layouts.

---

## The Reasoning Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  Ontology Reasoning Pipeline                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Check Cache (Blake3 hash lookup)                        │
│     ├── Cache Hit  → Return cached results (< 10ms)         │
│     └── Cache Miss → Continue to reasoning                  │
│                                                              │
│  2. Load Ontology from Repository                           │
│                    ↓                                         │
│  3. Parse OWL with hornedowl                                │
│                    ↓                                         │
│  4. Run whelk-rs EL++ Reasoner                              │
│     • Compute class hierarchy                               │
│     • Find inferred SubClassOf relations                    │
│     • Detect inconsistencies                                │
│                    ↓                                         │
│  5. Extract Inferred Axioms                                 │
│                    ↓                                         │
│  6. Calculate Confidence Scores                             │
│                    ↓                                         │
│  7. Store in Cache (with ontology hash)                     │
│                    ↓                                         │
│  8. Return Results                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## OWL 2 EL++ Profile

VisionFlow uses the EL++ profile for tractable reasoning:

### Supported Constructs

| Construct | Example | Supported |
|-----------|---------|-----------|
| Class assertions | `Individual(john)` | Yes |
| SubClassOf | `Dog SubClassOf Animal` | Yes |
| EquivalentClasses | `Human = Person` | Yes |
| DisjointClasses | `Cat DisjointWith Dog` | Yes |
| ObjectSomeValuesFrom | `hasPart some Heart` | Yes |
| ObjectHasValue | `livesIn value Earth` | Yes |
| ObjectIntersectionOf | `Male and Adult` | Yes |
| Transitive properties | `partOf` | Yes |
| Reflexive properties | `knows` | Yes |

### Unsupported (Full OWL 2)

| Construct | Reason |
|-----------|--------|
| Universal quantification | Intractable |
| Negation | Undecidable |
| Cardinality restrictions | Expensive |
| Disjunction | Exponential |

EL++ provides polynomial-time reasoning while covering most biomedical and scientific ontologies.

---

## Key Reasoning Operations

### 1. Class Hierarchy Inference

Computes the complete inheritance tree:

```
Input:                        Output:
Dog SubClassOf Mammal         Dog (depth: 2)
Mammal SubClassOf Animal        └── Mammal (depth: 1)
                                     └── Animal (depth: 0, root)

Inferred: Dog SubClassOf Animal
```

**Complexity**: O(n) with memoisation

### 2. Disjoint Class Detection

Identifies classes that cannot share instances:

```
Input:
Cat DisjointWith Dog
Siamese SubClassOf Cat

Inferred:
Siamese DisjointWith Dog  (inherited disjointness)
```

**Use in physics**: Disjoint classes receive strong separation forces.

### 3. Axiom Inference

Discovers implicit axioms from explicit ones:

```
Input:
hasPart is Transitive
Heart partOf Body
LeftVentricle partOf Heart

Inferred:
LeftVentricle partOf Body
```

**Confidence scores**: Inferred axioms receive lower priority (0.7-0.9) than asserted axioms (1.0).

---

## Caching System

### Blake3 Hash Keys

Reasoning results are cached by ontology content hash:

```
cache_key = blake3(ontology_id + cache_type + ontology_hash)
```

**Cache types**:
- `inferred_axioms`: All inferred axioms
- `class_hierarchy`: Hierarchy tree
- `disjoint_classes`: Disjoint pairs

### Automatic Invalidation

Cache invalidates when:
- Ontology file modified (hash changes)
- Explicit cache clear request
- TTL expired (configurable, default 1 hour)

### Performance Impact

| Operation | Uncached | Cached |
|-----------|----------|--------|
| 1,000 classes | 500 ms | < 10 ms |
| 5,000 classes | 2,000 ms | < 15 ms |
| 10,000 classes | 5,000 ms | < 20 ms |

---

## Data Models

### Inferred Axiom

```rust
pub struct InferredAxiom {
    axiom_type: String,        // "SubClassOf", "DisjointClasses", etc.
    subject_iri: String,       // Subject class IRI
    object_iri: String,        // Object class IRI
    confidence: f64,           // 0.0-1.0
    reasoning_method: String,  // "whelk-el++"
}
```

### Class Hierarchy Node

```rust
pub struct ClassNode {
    iri: String,
    label: String,
    parent_iri: Option<String>,
    children_iris: Vec<String>,
    node_count: usize,         // Descendant count
    depth: usize,              // Hierarchy depth
}
```

### Class Hierarchy

```rust
pub struct ClassHierarchy {
    root_classes: Vec<String>,
    hierarchy: HashMap<String, ClassNode>,
}
```

---

## Integration with Constraints

Reasoning results feed directly into the constraint system:

```
┌─────────────────────────────────────────────────────────────┐
│              Reasoning → Constraints Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Inferred SubClassOf                                        │
│       ↓                                                      │
│  HierarchicalAttraction constraint (priority: 7)            │
│       ↓                                                      │
│  GPU physics applies parent-child spring force              │
│                                                              │
│  ─────────────────────────────────────────────────────      │
│                                                              │
│  Inferred DisjointWith                                      │
│       ↓                                                      │
│  Separation constraint (priority: 7)                        │
│       ↓                                                      │
│  GPU physics applies strong repulsion force                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

Inferred constraints receive priority 7 (vs priority 5 for asserted), ensuring user-defined and explicit axioms take precedence.

---

## GitHub Sync Integration

Reasoning triggers automatically when ontologies sync from GitHub:

```
GitHub Push Event
      ↓
GitHub Sync Service
      ↓
OWL File Updated
      ↓
TriggerReasoning Message → OntologyActor
      ↓
OntologyReasoningService.infer_axioms()
      ↓
Inference Results
      ↓
Graph Update (new constraints)
      ↓
WebSocket Broadcast (layout changes)
```

---

## Configuration

### Environment Variables

```bash
REASONING_CACHE_TTL=3600          # Cache lifetime (seconds)
REASONING_TIMEOUT=30000           # Max reasoning time (ms)
REASONING_MAX_AXIOMS=100000       # Axiom limit
```

### Feature Flags

```toml
[features]
ontology_validation = true
reasoning_cache = true
```

---

## Troubleshooting

### Common Issues

**"Reasoning timeout"**
- Cause: Large ontology or complex axioms
- Fix: Increase `REASONING_TIMEOUT` or simplify ontology

**"Cache invalidation loop"**
- Cause: Ontology hash changing on every read
- Fix: Ensure consistent serialisation (normalise whitespace)

**"Missing inferred axioms"**
- Cause: Axiom uses OWL 2 construct outside EL++ profile
- Fix: Verify ontology is EL++ compatible (no universal quantification, negation, or cardinality)

### Debug Logging

```rust
RUST_LOG=ontology_reasoning=debug cargo run
```

---

## Future Enhancements

### Planned

1. **Incremental reasoning**: Only recompute changed portions
2. **Parallel reasoning**: Multi-threaded inference
3. **Explanation support**: Trace inference derivations
4. **Custom rules**: User-defined reasoning rules

### Research Directions

- **ML-based confidence**: Learn optimal scores from user feedback
- **Distributed reasoning**: Multi-node computation for very large ontologies
- **Hybrid reasoning**: Combine multiple reasoners (whelk + HermiT)

---

## Related Concepts

- **[Constraint System](constraint-system.md)**: How inferred axioms become physics constraints
- **[Physics Engine](physics-engine.md)**: How constraints affect node positioning
- **[Hexagonal Architecture](hexagonal-architecture.md)**: OntologyRepository port and adapter
