# Semantic Physics Architecture

## Overview

The Semantic Physics Architecture implements a constraint generation and physics integration system that translates OWL (Web Ontology Language) axioms into GPU-accelerated physics constraints for visualization and reasoning.

## Architecture Components

### 1. Semantic Physics Types (`semantic-physics-types.rs`)

**Enhanced Constraint Types:**

```rust
pub enum SemanticPhysicsConstraint {
    // DisjointWith → Separation (repel-k * 2.0)
    Separation {
        class-a: String,
        class-b: String,
        min-distance: f32,
        strength: f32,
        priority: u8,
    },

    // SubClassOf → HierarchicalAttraction (spring-k * 0.5)
    HierarchicalAttraction {
        child-class: String,
        parent-class: String,
        ideal-distance: f32,
        strength: f32,
        priority: u8,
    },

    // Axis alignment for organizing hierarchies
    Alignment {
        class-iri: String,
        axis: Axis,  // X, Y, or Z
        target-position: f32,
        strength: f32,
        priority: u8,
    },

    // InverseOf → Bidirectional edge constraints
    BidirectionalEdge {
        class-a: String,
        class-b: String,
        strength: f32,
        priority: u8,
    },

    // EquivalentTo → Colocation
    Colocation {
        class-a: String,
        class-b: String,
        target-distance: f32,
        strength: f32,
        priority: u8,
    },

    // PartOf → Containment
    Containment {
        child-class: String,
        parent-class: String,
        radius: f32,
        strength: f32,
        priority: u8,
    },
}
```

**Priority System:** 1-10 (1 = highest priority, 10 = lowest)
- Priority 1: User-defined constraints (weight = 1.0)
- Priority 5: Asserted axioms (weight ≈ 0.32)
- Priority 10: Inferred axioms (weight = 0.1)

### 2. Semantic Axiom Translator (`semantic-axiom-translator.rs`)

**Axiom → Constraint Translation Rules:**

| OWL Axiom | Physics Constraint | Parameters |
|-----------|-------------------|------------|
| `DisjointWith(A, B)` | `Separation` | `min-distance = 35.0 * 2.0`, `strength = 0.8` |
| `SubClassOf(Child, Parent)` | `HierarchicalAttraction` | `ideal-distance = 20.0`, `strength = 0.6 * 0.5` |
| `EquivalentClasses(A, B)` | `Colocation + BidirectionalEdge` | `target-distance = 2.0`, `strength = 0.9` |
| `SameAs(I1, I2)` | `Colocation` | `target-distance = 0.0`, `strength = 1.0` |
| `DifferentFrom(I1, I2)` | `Separation` | `min-distance = 35.0`, `strength = 0.8` |
| `PartOf(Part, Whole)` | `Containment` | `radius = 30.0`, `strength = 0.8` |
| `PropertyDomainRange` | `Alignment (X-axis)` | Domain at X=-50, Range at X=50 |

**Configuration:**

```rust
pub struct SemanticPhysicsConfig {
    pub disjoint-repel-multiplier: f32,      // Default: 2.0
    pub subclass-spring-multiplier: f32,     // Default: 0.5
    pub enable-hierarchy-alignment: bool,    // Default: true
    pub enable-bidirectional-constraints: bool, // Default: true
    pub priority-blending: PriorityBlendingStrategy,
}
```

**Priority Blending Strategies:**
- `Weighted`: Exponential weight based on priority
- `HighestPriority`: Take constraint with lowest priority number
- `Strongest`: Take constraint with highest strength
- `Equal`: Blend all constraints equally

### 3. GPU Constraint Buffer (`semantic-gpu-buffer.rs`)

**CUDA-Compatible Memory Layout:**

```rust
#[repr(C, align(16))]
pub struct SemanticGPUConstraint {
    pub constraint-type: i32,       // 1-6 for constraint types
    pub priority: i32,              // 1-10
    pub node-indices: [i32; 4],     // Up to 4 nodes
    pub params: [f32; 4],           // Primary parameters
    pub params2: [f32; 4],          // Secondary parameters
    pub weight: f32,                // Precomputed priority weight
    pub axis: i32,                  // 0=None, 1=X, 2=Y, 3=Z
    -padding: [f32; 2],            // 16-byte alignment
}
```

**GPU Buffer Features:**
- ✅ 16-byte memory alignment for CUDA
- ✅ IRI to index mapping for class references
- ✅ Automatic constraint validation
- ✅ Buffer overflow protection
- ✅ Statistics tracking

## Usage Examples

### Basic Translation

```rust
use constraints::{
    SemanticAxiomTranslator,
    SemanticPhysicsConfig,
    SemanticGPUConstraintBuffer,
    AxiomType,
    OWLAxiom,
};

// Create translator
let mut translator = SemanticAxiomTranslator::new();

// Define axioms
let axioms = vec![
    OWLAxiom::asserted(AxiomType::DisjointClasses {
        classes: vec![1, 2, 3],
    }),
    OWLAxiom::asserted(AxiomType::SubClassOf {
        subclass: 10,
        superclass: 20,
    }),
];

// Translate to semantic constraints
let constraints = translator.translate-axioms(&axioms);

// Create GPU buffer
let mut gpu-buffer = SemanticGPUConstraintBuffer::new(1000);
gpu-buffer.add-constraints(&constraints).unwrap();

// Upload to CUDA
unsafe {
    cuda-upload(
        gpu-buffer.as-ptr(),
        gpu-buffer.size-bytes(),
    );
}
```

### Custom Configuration

```rust
let config = SemanticPhysicsConfig {
    disjoint-repel-multiplier: 3.0,  // Stronger repulsion
    subclass-spring-multiplier: 0.3, // Tighter hierarchies
    enable-hierarchy-alignment: true,
    enable-bidirectional-constraints: true,
    priority-blending: PriorityBlendingStrategy::Weighted,
    ..Default::default()
};

let mut translator = SemanticAxiomTranslator::with-config(config);
```

### Priority Blending Example

```rust
// User-defined constraint (priority 1, weight 1.0)
let user-constraint = OWLAxiom::user-defined(AxiomType::SubClassOf {
    subclass: 1,
    superclass: 2,
});

// Inferred constraint (priority 7, weight ≈ 0.33)
let inferred-constraint = OWLAxiom::inferred(AxiomType::SubClassOf {
    subclass: 1,
    superclass: 3,
});

// User constraint will dominate in blending
let constraints = translator.translate-axioms(&[
    user-constraint,
    inferred-constraint,
]);
```

## Performance Characteristics

### Memory Layout
- Constraint size: 80 bytes (16-byte aligned)
- Buffer overhead: ~24 bytes + HashMap overhead
- 1000 constraints ≈ 80 KB GPU memory

### Translation Performance
- DisjointClasses(n): O(n²) constraints generated
- SubClassOf: O(1) + O(1) if alignment enabled
- Batch translation: ~100K axioms/sec (estimated)

### GPU Upload
- Direct memory mapping via `as-ptr()`
- Zero-copy transfer to CUDA
- Contiguous memory layout for coalesced access

## Integration Points

### With Existing Systems

1. **AxiomMapper Integration:**
   ```rust
   // Use standard AxiomMapper for legacy code
   let standard-constraints = axiom-mapper.translate-axioms(&axioms);

   // Use SemanticAxiomTranslator for enhanced features
   let semantic-constraints = semantic-translator.translate-axioms(&axioms);
   ```

2. **GPU Converter Compatibility:**
   ```rust
   // Convert semantic to standard physics constraints
   let physics-constraints = translator.to-physics-constraints(&semantic-constraints);

   // Use existing GPU converter
   let gpu-buffer = to-gpu-constraint-batch(&physics-constraints);
   ```

3. **Priority Resolver Integration:**
   ```rust
   // Semantic constraints can be resolved using standard resolver
   let mut resolver = PriorityResolver::new();
   resolver.add-constraints(physics-constraints);
   let resolved = resolver.resolve();
   ```

## Advanced Features

### Hierarchy Alignment

When `enable-hierarchy-alignment = true`, SubClassOf axioms generate:
1. HierarchicalAttraction constraint (primary)
2. Alignment constraint on Y-axis (secondary, priority +2)

This creates visually organized tree structures.

### Bidirectional Constraints

For symmetric relationships (EquivalentClasses, InverseOf):
1. Colocation constraint (forces proximity)
2. BidirectionalEdge constraint (ensures symmetric forces)

### Constraint Priority Weighting

```rust
// Exponential falloff: weight = 10^(-(priority-1)/9)
Priority 1:  weight = 1.000 (100%)
Priority 2:  weight = 0.774 (77%)
Priority 3:  weight = 0.599 (60%)
Priority 4:  weight = 0.464 (46%)
Priority 5:  weight = 0.359 (36%)
Priority 6:  weight = 0.278 (28%)
Priority 7:  weight = 0.215 (22%)
Priority 8:  weight = 0.167 (17%)
Priority 9:  weight = 0.129 (13%)
Priority 10: weight = 0.100 (10%)
```

## Testing

Comprehensive test coverage includes:
- ✅ Constraint type generation
- ✅ Priority calculation
- ✅ GPU memory alignment
- ✅ IRI to index mapping
- ✅ Buffer overflow protection
- ✅ Statistics tracking
- ✅ Blending strategies
- ✅ Hierarchy cache management

Run tests:
```bash
cargo test --package ontology-visualizer --lib constraints::semantic
```

## Future Enhancements

1. **Dynamic LOD for Semantic Constraints**
   - Distance-based constraint activation
   - Hierarchy-aware culling

2. **Temporal Constraints**
   - Time-based activation frames
   - Progressive constraint introduction

3. **Multi-GPU Support**
   - Buffer partitioning
   - Load balancing across GPUs

4. **Advanced Blending**
   - Machine learning-based priority prediction
   - Context-aware constraint strength adjustment

## References

- OWL 2 Web Ontology Language: https://www.w3.org/TR/owl2-syntax/
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- Force-Directed Graph Drawing: Fruchterman-Reingold algorithm
