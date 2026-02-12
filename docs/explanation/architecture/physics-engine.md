---
title: Physics Engine Architecture
description: Complete reference for VisionFlow's semantic physics engine that translates OWL ontology axioms into GPU-accelerated force-directed layouts with 5 CUDA kernels and sub-2ms performance.
category: explanation
tags:
  - physics
  - gpu
  - cuda
  - architecture
  - ontology
  - force-directed
updated-date: 2025-01-29
difficulty-level: advanced
---

# Physics Engine Architecture

VisionFlow's physics engine is a semantic-aware, GPU-accelerated force-directed graph layout system that translates OWL 2 ontology axioms into meaningful visual arrangements.

---

## Executive Summary

The physics engine bridges two domains:

1. **Semantic Domain**: OWL axioms define logical relationships (SubClassOf, DisjointWith, EquivalentClasses)
2. **Physical Domain**: Forces and constraints position nodes spatially to reflect those relationships

This translation enables users to intuitively understand complex ontologies through spatial proximity, hierarchy, and grouping.

### Implementation Status

| Component | Status | Performance |
|-----------|--------|-------------|
| **CUDA Kernels** | Complete | 5 kernels, ~2.3ms for 10K nodes |
| **Constraint Models** | Complete | 10 semantic constraint kinds |
| **Ontology Translator** | Complete | Maps OWL axioms to constraints |
| **OntologyConstraintActor** | Complete | GPU upload with CPU fallback |
| **CustomReasoner** | Complete | Infers SubClassOf, DisjointWith, EquivalentTo |

---

## Force-Directed Layout Principles

### The Fundamental Algorithm

Force-directed layouts treat graphs as physical systems where:

- **Nodes** behave as charged particles that repel each other
- **Edges** act as springs that attract connected nodes
- **Constraints** impose additional forces based on semantic meaning

The system iteratively applies forces until reaching equilibrium (settled state).

```
+-------------------------------------------------------------+
|                    Force Components                          |
+-------------------------------------------------------------+
|                                                              |
|   Repulsion (all pairs)    Attraction (edges only)          |
|   +---+        +---+       +---+--------+---+               |
|   | A | <----> | B |       | A |~~~~~~~~| B |               |
|   +---+        +---+       +---+--------+---+               |
|     ^            ^           Spring connection              |
|     +------------+                                          |
|     Push apart                                              |
|                                                              |
|   Semantic Forces (ontology-derived)                        |
|   +-----------+                                             |
|   | Parent    | <-- Hierarchical attraction                 |
|   +-----+-----+                                             |
|         |                                                   |
|   +-----+-----+                                             |
|   | Child     |                                             |
|   +-----------+                                             |
|                                                              |
+-------------------------------------------------------------+
```

### Integration Method

VisionFlow uses Verlet integration for stability:

```
Position(t+dt) = Position(t) + Velocity(t)*dt + 0.5*Acceleration(t)*dt^2
Velocity(t+dt) = Velocity(t) + 0.5*(Acceleration(t) + Acceleration(t+dt))*dt
```

Adaptive timesteps prevent instability when forces spike during initial layout or major changes.

---

## Semantic Force Types

The physics engine implements five specialised semantic forces derived from OWL semantics.

### 1. DAG Layout (Hierarchical Positioning)

Arranges nodes in directed acyclic graph (DAG) patterns based on class hierarchy.

**Use case**: SubClassOf relationships

**Modes**:
- **Top-Down**: Vertical layers (root at top)
- **Left-Right**: Horizontal layers (root at left)
- **Radial**: Concentric circles (root at centre)

**Algorithm**:
1. Calculate hierarchy levels via topological sort
2. Lock one axis to hierarchy level
3. Apply spring forces to maintain level position

### 2. Type Clustering

Groups nodes by semantic type (Class, Individual, Property).

**Use case**: Visual organisation by node category

**Algorithm**:
1. Calculate cluster centre for each node type
2. Apply attraction toward type's cluster centre
3. Reduce repulsion between same-type nodes

### 3. Collision Detection

Prevents node overlap through size-aware repulsion.

**Use case**: Readability, especially for labelled nodes

**Algorithm**:
1. Calculate effective radius for each node
2. Detect overlapping pairs
3. Apply proportional separation forces

### 4. Attribute-Weighted Springs

Modifies edge spring strength based on semantic weight.

**Use case**: Stronger connections for important relationships

**Example**: `partOf` relationships use stronger springs than `relatedTo`

### 5. Ontology Relationship Forces

Specialised forces for domain-specific ontology patterns:

- **Physicality clustering**: Groups VirtualEntity/PhysicalEntity/ConceptualEntity
- **Role clustering**: Groups Process/Agent/Resource nodes
- **Maturity layout**: Positions by lifecycle stage (Emerging/Mature/Declining)

---

## OWL-to-Physics Translation

The axiom translator converts OWL 2 constructs into physics constraints:

| OWL Axiom | Physics Constraint | Default Parameters |
|-----------|-------------------|-------------------|
| `DisjointWith(A, B)` | Strong separation | min-distance: 70, strength: 0.8 |
| `SubClassOf(C, P)` | Hierarchical attraction | ideal-distance: 20, strength: 0.3 |
| `EquivalentClasses(A, B)` | Colocation + bidirectional | distance: 2.0, strength: 0.9 |
| `SameAs(A, B)` | Strong colocation | distance: 0.0, strength: 1.0 |
| `PartOf(P, W)` | Containment boundary | radius: 30, strength: 0.8 |
| `InverseOf(P, Q)` | Bidirectional edge | strength: 0.7 |

### Priority Blending

When multiple constraints affect the same nodes, priorities resolve conflicts:

```
Priority 1:  User-defined (highest)    -> weight = 100%
Priority 5:  Asserted axioms           -> weight = 36%
Priority 10: Inferred axioms (lowest)  -> weight = 10%
```

Weight calculation uses exponential decay:
```
weight(priority) = 10^(-(priority-1)/9)
```

---

## CUDA Kernel Implementation

### Kernel Performance Characteristics

| Kernel | Purpose | Block Size | Complexity | Time (10K nodes) |
|--------|---------|------------|------------|------------------|
| `apply_disjoint_classes_kernel` | Repulsion | 256 threads | O(n^2) pairs | ~0.8ms |
| `apply_subclass_hierarchy_kernel` | Attraction | 256 threads | O(n*m) | ~0.6ms |
| `apply_sameas_colocate_kernel` | Colocation | 256 threads | O(n) | ~0.3ms |
| `apply_inverse_symmetry_kernel` | Symmetry | 256 threads | O(n) | ~0.2ms |
| `apply_functional_cardinality_kernel` | Cardinality | 256 threads | O(n*c) | ~0.4ms |
| **TOTAL** | | | | **~2.3ms** |

### Memory Alignment

All GPU structures use **64-byte alignment** for optimal cache line utilization:

```cuda
struct OntologyNode {          // 64 bytes total
    uint32_t graph_id;         // 4 bytes
    uint32_t node_id;          // 4 bytes
    uint32_t ontology_type;    // 4 bytes
    uint32_t constraint_flags; // 4 bytes
    float3 position;           // 12 bytes
    float3 velocity;           // 12 bytes
    float mass;                // 4 bytes
    float radius;              // 4 bytes
    uint32_t parent_class;     // 4 bytes
    uint32_t property_count;   // 4 bytes
    uint32_t padding[6];       // 24 bytes -> TOTAL: 64 bytes
};

struct OntologyConstraint {    // 64 bytes total
    uint32_t type;             // 4 bytes
    uint32_t source_id;        // 4 bytes
    uint32_t target_id;        // 4 bytes
    uint32_t graph_id;         // 4 bytes
    float strength;            // 4 bytes
    float distance;            // 4 bytes
    float padding[10];         // 40 bytes -> TOTAL: 64 bytes
};
```

**Why 64 bytes?**
- Modern GPU cache lines are 128 bytes
- Two constraints fit perfectly in one cache line
- Minimizes memory bandwidth (critical for physics simulation)

### Axiom-to-Force Mappings

#### DisjointWith -> Separation (Repulsion)

**Ontology Axiom**:
```owl
DisjointClasses(Neuron, Astrocyte)
```

**CUDA Implementation**:
```cuda
// Repulsion force: F = -k * penetration
float penetration = min_distance - dist;
float force_magnitude = separation_strength * constraint.strength * penetration;
float3 force = direction * (-force_magnitude);  // Negative = repel
```

**Effect**: Disjoint classes visually separate in 3D space

#### SubClassOf -> HierarchicalAttraction

**Ontology Axiom**:
```owl
SubClassOf(Neuron, Cell)  // Neuron is-a Cell
```

**CUDA Implementation**:
```cuda
// Spring force to ideal distance: F = k * displacement
float displacement = dist - ideal_distance;
float force_magnitude = alignment_strength * constraint.strength * displacement;
float3 force = direction * force_magnitude;
```

**Effect**: Subclass instances cluster near superclass instances

#### EquivalentTo -> Colocation (Strong Attraction)

**Ontology Axiom**:
```owl
EquivalentClasses(Person, Human)
```

**CUDA Implementation**:
```cuda
// Strong spring force to minimize distance
float force_magnitude = colocate_strength * constraint.strength * dist;
float3 force = direction * force_magnitude;

// Additional velocity damping for faster convergence
nodes[idx].velocity = nodes[idx].velocity * 0.95f;
```

**Effect**: Equivalent classes align very closely in space

---

## Stability and Convergence

### Energy Monitoring

The system tracks total kinetic energy:
```
E = 0.5 * sum(mass * velocity^2)
```

When energy drops below threshold, the simulation enters "settled" mode with reduced update frequency.

### Convergence Criteria

- **Energy threshold**: System settles when E < 0.001 per node
- **Movement threshold**: Settles when max displacement < 0.1 units
- **Time limit**: Forces settlement after 10 seconds if not converged

### Damping

Velocity damping (default 0.9) prevents oscillation:
```
velocity(t+dt) = velocity(t) * damping_factor
```

---

## Stress Majorization

For high-quality final layouts, stress majorization optimises node positions to minimise graph stress:

```
Stress = sum(weight_ij * (distance_ij - ideal_distance_ij)^2)
```

**Algorithm**:
1. Compute current pairwise distances
2. Calculate stress gradient
3. Solve linear system for optimal positions
4. Iterate until convergence

Stress majorization runs on GPU for graphs up to 100K nodes.

---

## Performance Characteristics

### Complexity

| Algorithm | Complexity | GPU Parallelism |
|-----------|-----------|-----------------|
| Repulsion (Barnes-Hut) | O(n log n) | Per-node threads |
| Attraction | O(m) | Per-edge threads |
| Collision detection | O(n log n) | Spatial grid |
| Stress majorization | O(n^2) | Matrix operations |

### Typical Frame Times (RTX 3080)

| Node Count | Frame Time | FPS |
|------------|-----------|-----|
| 1,000 | 1.2 ms | 800+ |
| 10,000 | 3.5 ms | 285 |
| 100,000 | 12 ms | 83 |

---

## Configuration Parameters

Key physics parameters exposed via API and UI:

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `repulsion_strength` | Node-node repulsion | 1000 | 0-10000 |
| `attraction_strength` | Edge spring strength | 0.01 | 0-1 |
| `damping` | Velocity damping | 0.9 | 0.5-0.99 |
| `timestep` | Integration step size | 0.016 | 0.001-0.1 |
| `gravity_strength` | Centre attraction | 0.1 | 0-1 |
| `theta` | Barnes-Hut approximation | 0.5 | 0-1 |

### Force Magnitude Recommendations

Based on empirical testing with 1K-10K node graphs:

| Constraint Type | Default Weight | Force Multiplier | Max Force Clamp | Ideal Distance |
|-----------------|----------------|------------------|-----------------|----------------|
| **DisjointWith** | 2.0 | 2.0x separation_strength | 1000.0 | 100-150 units |
| **SubClassOf** | 1.0 | 0.5x spring_k | 500.0 | 30-50 units |
| **EquivalentTo** | 1.5 | 1.5x colocate_strength | 800.0 | 5-10 units |
| **InverseOf** | 0.7 | 0.7x symmetry_strength | 400.0 | Equal from midpoint |
| **FunctionalProperty** | 0.7 | 1.0x cardinality_penalty | 600.0 | Boundary box |

---

## Data Flow Pipeline

**Pipeline Stages:**

1. **GitHub Sync**: Parse .md files -> OntologyBlock extraction
   - UnifiedOntologyRepository::save_ontology_class()
   - Stores classes with IRIs in OntologyRepository (in-memory)

2. **Reasoning**: CustomReasoner infers transitive axioms
   - Input: Ontology { subclass_of, disjoint_classes, ... }
   - Output: Vec<InferredAxiom> { SubClassOf, DisjointWith, ... }

3. **Constraint Generation**: OntologyPipelineService
   - Creates Constraint objects from axioms
   - Resolves IRIs to database node IDs

4. **GPU Upload**: OntologyConstraintActor
   - Converts Constraint -> ConstraintData (GPU format)
   - Uploads to CUDA kernels via SharedGPUContext

5. **Physics Simulation**: CUDA Kernels
   - apply_disjoint_classes_kernel() - Repulsion forces
   - apply_subclass_hierarchy_kernel() - Attraction forces
   - apply_sameas_colocate_kernel() - Strong attraction
   - Performance: ~2.3ms for 10K nodes with 64-byte alignment

---

## Key Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/utils/ontology-constraints.cu` | 488 | CUDA kernels (5 semantic force types) |
| `src/physics/ontology-constraints.rs` | 822 | OWL axiom -> Constraint translator |
| `src/services/ontology-pipeline-service.rs` | 370 | End-to-end semantic physics pipeline |
| `src/actors/gpu/ontology-constraint-actor.rs` | 550 | GPU upload and actor coordination |
| `src/reasoning/custom-reasoner.rs` | 466 | OWL reasoning engine |
| `src/models/constraints.rs` | 412 | Constraint data structures |

### CUDA Kernel Entry Points

```c
extern "C" {
    void launch_disjoint_classes_kernel(/*...*/);
    void launch_subclass_hierarchy_kernel(/*...*/);
    void launch_sameas_colocate_kernel(/*...*/);
    void launch_inverse_symmetry_kernel(/*...*/);
    void launch_functional_cardinality_kernel(/*...*/);
}
```

---

## Usage Examples

### Biomedical Ontology Visualization

**Ontology** (Cell Ontology subset):
```turtle
@prefix cell: <http://purl.obolibrary.org/obo/CL_> .

cell:0000540 rdf:type owl:Class ;  # Neuron
    rdfs:subClassOf cell:0000000 .  # Cell

cell:0000127 rdf:type owl:Class ;  # Astrocyte
    rdfs:subClassOf cell:0000000 .  # Cell

[ rdf:type owl:AllDisjointClasses ;
  owl:members ( cell:0000540 cell:0000127 ) ] .  # Neurons != Astrocytes
```

**Visual Effect**:
- All `Neuron` instances form a cluster
- All `Astrocyte` instances form a separate cluster
- Both clusters orbit around `Cell` superclass
- Neurons and Astrocytes maintain >100 unit separation (repulsion)

**Code**:
```rust
let config = SemanticPhysicsConfig {
    auto_trigger_reasoning: true,
    constraint_strength: 1.2,  // Slightly stronger forces
    ..Default::default()
};

let pipeline = OntologyPipelineService::new(config);
pipeline.on_ontology_modified(ontology_id, cell_ontology).await?;
```

---

## Related Documentation

- [Actor System Architecture](./actor-system.md)
- [Hexagonal Architecture with CQRS](./hexagonal-cqrs-unified.md)
- [Database Schema Catalog](../../reference/database/schema-catalog.md)

---

## Further Reading

- Fruchterman, T. & Reingold, E. (1991). "Graph Drawing by Force-directed Placement"
- Gansner, E. et al. (2004). "A Technique for Drawing Directed Graphs" (stress majorization)
- W3C OWL 2 Specification: https://www.w3.org/TR/owl2-syntax/

---

**Last Updated**: January 29, 2025
**Maintainer**: VisionFlow Architecture Team
