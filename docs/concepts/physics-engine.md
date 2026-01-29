---
title: Physics Engine
description: Understanding the semantic physics engine that translates OWL ontology relationships into GPU-accelerated force-directed layouts
category: explanation
tags:
  - physics
  - gpu
  - architecture
  - ontology
related-docs:
  - concepts/constraint-system.md
  - concepts/gpu-acceleration.md
  - concepts/ontology-reasoning.md
  - guides/physics-configuration.md
updated-date: 2025-12-18
difficulty-level: intermediate
---

# Physics Engine

VisionFlow's physics engine is a semantic-aware, GPU-accelerated force-directed graph layout system that translates OWL 2 ontology axioms into meaningful visual arrangements.

---

## Core Concept

The physics engine bridges two domains:

1. **Semantic Domain**: OWL axioms define logical relationships (SubClassOf, DisjointWith, EquivalentClasses)
2. **Physical Domain**: Forces and constraints position nodes spatially to reflect those relationships

This translation enables users to intuitively understand complex ontologies through spatial proximity, hierarchy, and grouping.

---

## Force-Directed Layout Principles

### The Fundamental Algorithm

Force-directed layouts treat graphs as physical systems where:

- **Nodes** behave as charged particles that repel each other
- **Edges** act as springs that attract connected nodes
- **Constraints** impose additional forces based on semantic meaning

The system iteratively applies forces until reaching equilibrium (settled state).

```
┌─────────────────────────────────────────────────────────┐
│                    Force Components                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Repulsion (all pairs)    Attraction (edges only)       │
│   ┌───┐        ┌───┐       ┌───┐────────┌───┐          │
│   │ A │ ←────→ │ B │       │ A │~~~~~~~~│ B │          │
│   └───┘        └───┘       └───┘────────└───┘          │
│     ↑            ↑           Spring connection           │
│     └────────────┘                                       │
│     Push apart                                           │
│                                                          │
│   Semantic Forces (ontology-derived)                     │
│   ┌───────────┐                                         │
│   │ Parent    │ ←── Hierarchical attraction              │
│   └─────┬─────┘                                         │
│         │                                                │
│   ┌─────┴─────┐                                         │
│   │ Child     │                                         │
│   └───────────┘                                         │
│                                                          │
└─────────────────────────────────────────────────────────┘
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

The physics engine implements five specialised semantic forces derived from OWL semantics:

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
Priority 1:  User-defined (highest)    → weight = 100%
Priority 5:  Asserted axioms           → weight = 36%
Priority 10: Inferred axioms (lowest)  → weight = 10%
```

Weight calculation uses exponential decay:
```
weight(priority) = 10^(-(priority-1)/9)
```

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

---

## Related Concepts

- **[Constraint System](constraint-system.md)**: How semantic constraints are defined and enforced
- **[GPU Acceleration](gpu-acceleration.md)**: CUDA kernel implementation details
- **[Ontology Reasoning](ontology-reasoning.md)**: How OWL axioms are inferred before translation

---

## Further Reading

- Fruchterman, T. & Reingold, E. (1991). "Graph Drawing by Force-directed Placement"
- Gansner, E. et al. (2004). "A Technique for Drawing Directed Graphs" (stress majorization)
- W3C OWL 2 Specification: https://www.w3.org/TR/owl2-syntax/
