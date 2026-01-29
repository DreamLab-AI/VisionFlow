---
title: Constraint System
description: Understanding how semantic constraints translate OWL axioms into GPU-optimised physics rules for meaningful graph layouts
category: explanation
tags:
  - constraints
  - ontology
  - gpu
  - architecture
related-docs:
  - concepts/physics-engine.md
  - concepts/ontology-reasoning.md
  - reference/api/semantic-features-api.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# Constraint System

The constraint system bridges semantic knowledge and visual layout by translating OWL 2 ontology axioms into GPU-accelerated physics constraints.

---

## Core Concept

Traditional force-directed layouts treat all edges equally. VisionFlow's constraint system adds semantic meaning:

- **DisjointWith** axioms create separation forces
- **SubClassOf** axioms create hierarchical attraction
- **EquivalentClasses** axioms force colocation
- **PartOf** relationships enforce containment boundaries

This enables layouts where spatial relationships convey logical relationships.

---

## Constraint Types

### 1. Separation Constraint

**Source**: `owl:disjointWith` axioms

**Purpose**: Ensures logically disjoint classes are visually separated.

**Parameters**:
- `node_a`, `node_b`: Class IRIs
- `min_distance`: Minimum separation (default: 70 units)
- `strength`: Force multiplier (default: 0.8)
- `priority`: 1-10 (1 = highest)
- `axis`: Optional axis restriction (X, Y, Z, or None)

**Physics**: Applies `repel_k * 2.0` multiplier for strong separation when nodes are closer than `min_distance`.

### 2. Hierarchical Attraction Constraint

**Source**: `rdfs:subClassOf` axioms

**Purpose**: Pulls subclasses toward their parent classes, creating visual hierarchies.

**Parameters**:
- `child`: Subclass IRI
- `parent`: Superclass IRI
- `ideal_distance`: Target separation (default: 20 units)
- `strength`: Spring constant (default: 0.3)
- `z_offset`: Vertical offset for 3D hierarchy visualisation

**Physics**: Applies `spring_k * 0.5` for gentle attraction, maintaining parent-child proximity without overlap.

### 3. Alignment Constraint

**Source**: User-defined or inferred from ontology structure

**Purpose**: Forces nodes to align along specific axes.

**Parameters**:
- `node`: Node IRI
- `axis`: X, Y, or Z
- `target_value`: Position on axis
- `strength`: Alignment force

**Use case**: Aligning all classes at the same depth in a hierarchy to the same Y coordinate.

### 4. Bidirectional Edge Constraint

**Source**: `owl:inverseOf` properties

**Purpose**: Creates symmetric relationship forces.

**Parameters**:
- `node_a`, `node_b`: Related nodes
- `distance`: Ideal separation
- `strength`: Symmetrical force magnitude

**Example**: If `hasPart` is inverse of `isPartOf`, both directions receive equal spring force.

### 5. Colocation Constraint

**Source**: `owl:equivalentClass` or `owl:sameAs` axioms

**Purpose**: Forces equivalent entities to cluster tightly.

**Parameters**:
- `node_a`, `node_b`: Equivalent class IRIs
- `max_distance`: Maximum separation (default: 2.0 units)
- `strength`: Strong force (default: 0.9)

**Physics**: Very strong spring force pulls equivalent classes nearly on top of each other.

### 6. Containment Constraint

**Source**: Part-whole relationships (e.g., `hasPart`, `contains`)

**Purpose**: Keeps parts spatially within their whole's boundary.

**Parameters**:
- `part`: Part node IRI
- `whole`: Containing node IRI
- `radius`: Boundary radius (default: 30 units)
- `strength`: Boundary enforcement (default: 0.8)

**Physics**: If part strays beyond `radius` from whole, applies inward force proportional to overshoot.

---

## Priority Blending System

When multiple constraints affect the same node pair, priorities determine the outcome.

### Priority Levels

```
Priority 1:  User-defined constraints     100% weight
Priority 2:  Critical system constraints   78% weight
Priority 3:  Important user preferences    60% weight
Priority 4:  Important system rules        47% weight
Priority 5:  Asserted axioms (default)     36% weight
Priority 6:  Medium importance             28% weight
Priority 7:  Inferred axioms               22% weight
Priority 8:  Low importance                17% weight
Priority 9:  Very low importance           13% weight
Priority 10: Suggestions                   10% weight
```

### Weight Calculation

Exponential decay provides smooth priority falloff:

```
weight(p) = 10^(-(p-1)/9)
```

### Blending Strategies

| Strategy | Behaviour |
|----------|-----------|
| **Weighted** (default) | Blend by priority weights |
| **HighestPriority** | Lowest priority number wins |
| **Strongest** | Highest strength value wins |
| **Equal** | Simple average |

---

## GPU Buffer System

Constraints are packed into GPU-optimised buffers for CUDA kernel processing.

### GPU Constraint Layout

```c
#[repr(C, align(16))]  // 16-byte alignment for CUDA
struct GPUSemanticConstraint {
    constraint_type: u32,    // 0=Separation, 1=Hierarchical, etc.
    node_a_id: u32,
    node_b_id: u32,
    param1: f32,             // Type-specific (e.g., min_distance)
    param2: f32,             // Type-specific (e.g., strength)
    param3: f32,             // Type-specific (e.g., radius)
    strength: f32,
    priority: u8,
    axis: u8,                // 0=None, 1=X, 2=Y, 3=Z
    _padding: [u8; 14],      // Align to 80 bytes
}
```

**Total size**: 80 bytes per constraint (16-byte multiple for optimal CUDA access)

### Memory Benefits

- **Coalesced reads**: Sequential memory access patterns
- **Zero-copy upload**: Direct pointer mapping to GPU
- **Cache-friendly**: 80-byte structure fits L1 cache lines

### Memory Usage

| Constraint Count | Memory |
|-----------------|--------|
| 2,000 | 160 KB |
| 12,000 | 960 KB |
| 30,000 | 2.4 MB |
| 200,000 | 16 MB |

---

## Axiom Translation Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                  Axiom Translation Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load OWL Ontology (hornedowl parser)                    │
│                    ↓                                         │
│  2. Parse Axioms → OWLAxiom structs                         │
│                    ↓                                         │
│  3. Create SemanticAxiomTranslator                          │
│                    ↓                                         │
│  4. Translate Axioms → SemanticPhysicsConstraints           │
│     • DisjointWith → Separation                             │
│     • SubClassOf → HierarchicalAttraction                   │
│     • EquivalentClasses → Colocation + Bidirectional        │
│     • PartOf → Containment                                  │
│                    ↓                                         │
│  5. Create SemanticGPUConstraintBuffer                      │
│                    ↓                                         │
│  6. Pack Constraints (auto-registers IRIs → IDs)            │
│                    ↓                                         │
│  7. Upload to GPU via CUDA                                  │
│                    ↓                                         │
│  8. Run Physics Simulation                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Translator Configuration

```rust
pub struct SemanticPhysicsConfig {
    // Force multipliers
    disjoint_repel_multiplier: f32,      // Default: 2.0
    subclass_spring_multiplier: f32,     // Default: 0.5
    equivalent_colocation_dist: f32,     // Default: 2.0
    partof_containment_radius: f32,      // Default: 30.0

    // Feature flags
    enable_hierarchy_alignment: bool,    // Default: true
    enable_bidirectional_constraints: bool, // Default: true

    // Priority settings
    user_defined_priority: u8,           // Default: 1
    asserted_priority: u8,               // Default: 5
    inferred_priority: u8,               // Default: 10
}
```

### Runtime Adjustment

Constraint parameters can be adjusted at runtime via the REST API without reloading the ontology.

---

## Constraint Validation

The system validates constraints before GPU upload:

1. **IRI resolution**: All IRIs must map to known nodes
2. **Parameter bounds**: Strengths and distances within valid ranges
3. **Circular dependency**: Detects containment cycles
4. **Priority conflicts**: Warns when high-priority constraints contradict

---

## Performance Characteristics

### Translation Speed

- **DisjointClasses(n)**: O(n^2) constraints generated (all pairs)
- **SubClassOf**: O(1) per axiom
- **Batch processing**: ~100,000 axioms/second (single-threaded)

### GPU Constraint Application

| Node Count | Constraints | Application Time |
|------------|-------------|------------------|
| 1,000 | 2,000 | 0.3 ms |
| 10,000 | 30,000 | 1.2 ms |
| 100,000 | 200,000 | 8 ms |

---

## Related Concepts

- **[Physics Engine](physics-engine.md)**: How constraints integrate with the force simulation
- **[Ontology Reasoning](ontology-reasoning.md)**: How axioms are inferred before constraint generation
- **[GPU Acceleration](gpu-acceleration.md)**: CUDA kernel implementation for constraint application
