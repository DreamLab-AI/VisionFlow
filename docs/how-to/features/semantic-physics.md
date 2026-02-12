---
title: Semantic Physics Engine
description: How VisionFlow uses OWL class relationships to drive physical forces in the graph layout, mapping subClassOf to attraction and disjointWith to repulsion.
category: how-to
tags:
  - ontology
  - physics
  - gpu
  - graph-layout
updated-date: 2026-02-12
difficulty-level: advanced
---

# Semantic Physics Engine

## Overview

The semantic physics engine translates OWL ontology relationships into physical
forces that govern node placement in the 3D graph. Instead of treating every
edge identically, the engine assigns force profiles based on the semantic type
of each relationship. The result is a layout where conceptual closeness in the
ontology maps to spatial closeness in the visualisation.

## Core Principle: Ontology-Driven Layout

Traditional force-directed layouts use generic spring and repulsion forces.
VisionFlow augments these with ontology-aware rules:

| OWL Relationship     | Physical Effect          | Rationale                                  |
|----------------------|--------------------------|--------------------------------------------|
| `subClassOf`         | Attraction (strong)      | Children cluster near parents              |
| `equivalentClass`    | Attraction (very strong) | Synonymous concepts overlap                |
| `disjointWith`       | Repulsion (strong)       | Incompatible concepts separate             |
| `has-part`           | Attraction (medium)      | Parts stay near their wholes               |
| `requires`           | Attraction (weak)        | Soft dependency pull                       |
| `bridges-to`         | Attraction (weak)        | Cross-domain links pull gently             |

Force magnitudes are configurable per relationship type in `SimParams`.

## Force Computation

For each pair of connected nodes the engine computes a net force vector:

```text
F_ij = spring(d_ij, rest_length) + semantic_modifier(rel_type)
```

Where:

- `spring(d, r)` is the standard Hooke's-law spring: `k * (d - r)`
- `semantic_modifier` scales the spring constant and rest length according to
  the relationship table above.

For `disjointWith`, the spring is replaced with an inverse-square repulsion:

```text
F_repel = -repulsion_strength / (d_ij^2 + epsilon)
```

This ensures disjoint classes are pushed apart even when the generic repulsion
would allow them to drift close.

## Configuration

Tune semantic force weights in the physics settings:

```json
{
  "semantic-subclass-attraction": 0.5,
  "semantic-equivalent-attraction": 1.0,
  "semantic-disjoint-repulsion": 0.8,
  "semantic-has-part-attraction": 0.3,
  "semantic-requires-attraction": 0.15,
  "semantic-bridges-attraction": 0.1,
  "semantic-rest-length-base": 50.0,
  "semantic-force-blend": 0.6
}
```

`semantic-force-blend` controls the mix between pure force-directed layout and
semantic overrides. At `0.0` the engine behaves like a vanilla force layout; at
`1.0` semantic forces dominate entirely.

## GPU Acceleration

Semantic forces are computed inside the same CUDA kernels used by the general
physics adapter. The kernel reads a relationship-type buffer alongside the
adjacency list:

```cuda
__global__ void compute_semantic_forces(
    const float3 *positions,
    const int    *adj_row_ptr,
    const int    *adj_col_idx,
    const int    *rel_types,
    const float  *force_weights,
    float3       *out_forces,
    int           n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 f = make_float3(0, 0, 0);
    for (int e = adj_row_ptr[i]; e < adj_row_ptr[i + 1]; e++) {
        int j = adj_col_idx[e];
        int rt = rel_types[e];
        float w = force_weights[rt];

        float3 delta = positions[j] - positions[i];
        float dist = length(delta) + 1e-6f;
        f += normalize(delta) * w * (dist - REST_LENGTH);
    }
    out_forces[i] = f;
}
```

The `force_weights` array is uploaded once when `SimParams` change and indexed
by a small integer relationship-type enum (0 = subClassOf, 1 = disjointWith,
etc.).

## Whelk Inference Integration

Before forces are computed, the Whelk EL++ reasoner infers implicit
relationships. For example, if `Cat subClassOf Animal` and
`Dog subClassOf Animal`, Whelk may infer `Cat disjointWith Dog` from class
axioms. These inferred edges are included in the adjacency list so that
implicit ontological structure is reflected in the layout without manual
annotation.

## Practical Tips

- Start with a low `semantic-force-blend` (0.2) and increase gradually to
  observe how semantic forces reshape the layout.
- Use `disjointWith` assertions to split unrelated domains into distinct
  spatial clusters.
- Combine with stress majorization for a two-phase layout: stress majorization
  sets global structure, then semantic forces refine local grouping.

## See Also

- [Semantic Forces User Guide](semantic-forces.md) -- DAG layout and force tuning
- [Ontology Semantic Forces Implementation](ontology-semantic-forces.md) -- code-level details
- [Stress Majorization](stress-majorization.md) -- complementary layout algorithm
- [GPU Physics Adapter Port](../../reference/architecture/ports/06-gpu-physics-adapter.md) -- CUDA integration
- [GPU Semantic Analyzer Port](../../reference/architecture/ports/07-gpu-semantic-analyzer.md) -- semantic similarity on GPU
