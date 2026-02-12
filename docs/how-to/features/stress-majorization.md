---
title: Stress Majorization for Graph Layout
description: GPU-accelerated stress majorization algorithm for optimising node positions in VisionFlow knowledge graphs, with CUDA kernels and client-side tweening.
category: how-to
tags:
  - graph-layout
  - gpu
  - cuda
  - stress-majorization
updated-date: 2026-02-12
difficulty-level: advanced
---

# Stress Majorization for Graph Layout

## Overview

Stress majorization minimises the difference between graph-theoretic distances
and Euclidean distances between nodes. VisionFlow runs the heavy computation on
the server GPU via CUDA kernels, then streams updated positions to clients that
apply tweening for smooth animation.

## Algorithm Summary

Given a graph with _n_ nodes, the stress function is:

```
stress = SUM_{i<j} w_ij * (||p_i - p_j|| - d_ij)^2
```

where `d_ij` is the shortest-path distance and `w_ij = 1/d_ij^2` is the
weight. Each iteration solves a weighted least-squares system to reduce stress
monotonically.

## Enabling Stress Majorization

Toggle the feature in the simulation parameters (`SimParams`):

```json
{
  "stress-optimization-enabled": 1,
  "stress-optimization-frequency": 100,
  "stress-learning-rate": 0.05,
  "stress-momentum": 0.5,
  "stress-max-displacement": 10.0,
  "stress-convergence-threshold": 0.01,
  "stress-max-iterations": 50,
  "stress-blend-factor": 0.2
}
```

| Parameter                        | Default | Description                              |
|----------------------------------|---------|------------------------------------------|
| `stress-optimization-enabled`    | 0       | 1 to enable, 0 to disable               |
| `stress-optimization-frequency`  | 100     | Run every _N_ physics ticks              |
| `stress-learning-rate`           | 0.05    | Step size per iteration                  |
| `stress-momentum`               | 0.5     | Momentum for gradient updates            |
| `stress-max-displacement`        | 10.0    | Clamp per-node displacement              |
| `stress-convergence-threshold`   | 0.01    | Stop when stress delta falls below this  |
| `stress-max-iterations`          | 50      | Hard cap on iterations per cycle         |
| `stress-blend-factor`            | 0.2     | Blend ratio with force-directed output   |

## GPU Acceleration via CUDA

The distance matrix and iterative solve are off-loaded to CUDA:

1. **APSP kernel** -- compute all-pairs shortest paths using a BFS/Bellman-Ford
   hybrid on CSR graph representation.
2. **Stress solve kernel** -- each thread updates one node's position toward the
   weighted centroid of its graph neighbours.
3. **Convergence check** -- a parallel reduction computes total stress delta;
   early-exit when below threshold.

```cuda
__global__ void stress_solve_step(
    float3 *positions,
    const float *dist_matrix,
    const float *weights,
    int n,
    float learning_rate
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float3 numerator = make_float3(0, 0, 0);
    float denominator = 0;

    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        float w = weights[i * n + j];
        float d = dist_matrix[i * n + j];
        float3 delta = positions[j] - positions[i];
        float cur = length(delta) + 1e-6f;
        numerator += w * (positions[j] - delta * (d / cur));
        denominator += w;
    }

    positions[i] = lerp(positions[i], numerator / denominator, learning_rate);
}
```

When no CUDA device is available the server falls back to a CPU solver that
uses the same algorithm with Rayon parallel iterators.

## Server-Authoritative Computation

Stress majorization runs exclusively on the server to guarantee every connected
client sees identical node positions. The flow is:

1. Server physics loop triggers stress optimisation every _N_ ticks.
2. CUDA kernel computes new positions and writes them to the position buffer.
3. Updated positions are broadcast over WebSocket as a binary frame
   (node_id: u32, x: f32, y: f32, z: f32 per node).
4. Clients receive the frame and begin tweening from old to new positions.

## Client-Side Tweening

To avoid jarring jumps, the client interpolates between the previous and new
server-authoritative positions over a configurable duration:

```typescript
function tweenPositions(
  current: Float32Array,
  target: Float32Array,
  alpha: number
): void {
  for (let i = 0; i < current.length; i++) {
    current[i] += (target[i] - current[i]) * alpha;
  }
}
```

The default tween duration matches the server tick interval (100 ms), producing
fluid motion even when stress updates arrive at a lower cadence.

## Performance Notes

| Graph Size   | CUDA Time per Cycle | CPU Fallback   |
|-------------|---------------------|----------------|
| 1,000 nodes | ~8 ms               | ~120 ms        |
| 5,000 nodes | ~35 ms              | ~1,400 ms      |
| 10,000 nodes| ~90 ms              | ~6,000 ms      |

For graphs over 5,000 nodes, GPU acceleration is strongly recommended.

## See Also

- [Stress Majorization Layout Optimization Guide](stress-majorization-guide.md) -- detailed parameter tuning
- [Semantic Physics Engine](semantic-physics.md) -- ontology-driven forces
- [GPU Physics Adapter Port](../../reference/architecture/ports/06-gpu-physics-adapter.md) -- CUDA kernel bindings
- [Semantic Forces User Guide](semantic-forces.md) -- force semantics overview
