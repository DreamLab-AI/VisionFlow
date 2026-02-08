---
title: GPU Acceleration
description: Understanding VisionFlow's CUDA-based GPU acceleration for physics simulation, graph algorithms, and analytics
category: explanation
tags:
  - gpu
  - cuda
  - performance
  - architecture
related-docs:
  - concepts/physics-engine.md
  - concepts/constraint-system.md
  - reference/performance-benchmarks.md
updated-date: 2025-12-18
difficulty-level: advanced
---

# GPU Acceleration

VisionFlow leverages NVIDIA CUDA to achieve real-time performance on graphs with 100,000+ nodes through massively parallel computation.

---

## Core Concept

Force-directed layout algorithms are inherently parallel: each node's forces can be computed independently. GPUs excel at this pattern, providing:

- **Massive parallelism**: Thousands of CUDA cores vs dozens of CPU cores
- **Memory bandwidth**: 500+ GB/s vs 50 GB/s on CPU
- **SIMD efficiency**: Same operation on many data points

VisionFlow achieves **55x speedup** over CPU for physics simulation on large graphs.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Compute Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Host Layer (Rust)                  │    │
│  │                                                      │    │
│  │  GpuMemoryManager ─→ Buffer Allocation              │    │
│  │                   ─→ Stream Management (3 streams)  │    │
│  │                   ─→ Leak Detection                 │    │
│  │                   ─→ RAII Safety                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   FFI Layer (extern C)               │    │
│  │                                                      │    │
│  │  CUDA FFI Bindings ─→ Kernel Launch                 │    │
│  │                    ─→ Dynamic Grid Sizing           │    │
│  │                    ─→ Error Handling                │    │
│  └─────────────────────────────────────────────────────┘    │
│                            ↓                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Device Layer (CUDA)                 │    │
│  │                                                      │    │
│  │  Physics Kernels    Clustering Kernels              │    │
│  │  Graph Kernels      Semantic Kernels                │    │
│  │  Ontology Kernels   Utility Kernels                 │    │
│  │                                                      │    │
│  │  GPU Memory: Global + Shared + Constant             │    │
│  │                            ↓                        │    │
│  │  Streaming Multiprocessors (Parallel Execution)     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Kernel Inventory

VisionFlow implements **87 production CUDA kernels** across 13 files:

### Physics Simulation (37 kernels)

- `build_grid_kernel`: Spatial hashing for O(n) neighbour detection
- `force_pass_kernel`: Multi-force integration with Barnes-Hut approximation
- `integrate_pass_kernel`: Verlet integration with adaptive timestep
- `calculate_kinetic_energy_kernel`: System energy monitoring
- `check_system_stability_kernel`: Convergence detection

### Clustering & Analytics (12 kernels)

- `init_centroids_kernel`: K-means++ parallel initialisation
- `assign_clusters_kernel`: GPU-parallel cluster assignment
- `update_centroids_kernel`: Cooperative groups reduction
- `compute_lof_kernel`: Local Outlier Factor anomaly detection
- `louvain_local_pass_kernel`: Parallel community detection

### Graph Algorithms (12 kernels)

- `compact_frontier_kernel`: SSSP with parallel prefix sum
- `label_propagation_kernel`: Connected component detection
- `pagerank_iteration_kernel`: Power iteration PageRank
- `approximate_apsp_kernel`: Landmark-based all-pairs shortest paths (now wired into ShortestPathActor)
- `delta_stepping_kernel`: Delta-stepping SSSP with configurable bucket width
- `batched_sssp_kernel`: Multi-source batched SSSP for efficient landmark computation
- SSSP distances now feed back into the GPU force kernel via `d_sssp_dist` buffer (SSSP-aware spring forces active)

### Semantic Forces (15 kernels)

- `apply_dag_force`: Hierarchical DAG layout
- `apply_type_cluster_force`: Type-based grouping
- `apply_collision_force`: Physical separation
- `apply_ontology_relationship_force`: Domain relationship forces

### Ontology Constraints (5 kernels)

- `apply_disjoint_classes_kernel`: Separation for disjoint classes
- `apply_subclass_hierarchy_kernel`: Spring alignment for SubClassOf
- `apply_sameas_colocate_kernel`: Colocation for equivalent classes

---

## Memory Management

### Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU Memory Hierarchy                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  On-Chip (Fast)                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Registers    (64K per SM, ~1 cycle latency)        │    │
│  │  Shared Mem   (48-96 KB per SM, ~5 cycle latency)   │    │
│  │  L1 Cache     (48 KB per SM)                        │    │
│  │  Constant     (64 KB total, cached)                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Off-Chip (Slower)                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  L2 Cache     (40-72 MB total)                      │    │
│  │  Global Mem   (16-80 GB HBM/GDDR)                   │    │
│  │  Host Mem     (via PCIe, ~12 GB/s)                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Structure of Arrays (SoA)

VisionFlow uses SoA layout for coalesced memory access:

```c
// Bad: Array of Structures (strided access)
struct Node { float x, y, z, vx, vy, vz; };
Node nodes[N];

// Good: Structure of Arrays (coalesced access)
float pos_x[N], pos_y[N], pos_z[N];
float vel_x[N], vel_y[N], vel_z[N];
```

Coalesced access achieves **8-10x** better memory bandwidth.

### Working Set

For 100K nodes:
- Positions: 2.4 MB (3 x 100K x 4 bytes)
- Velocities: 2.4 MB
- Forces: 2.4 MB
- Graph CSR: 3.2 MB
- **Total: ~10-22 MB** (fits entirely in L2 cache)

---

## Algorithmic Optimisations

### Barnes-Hut Approximation

Reduces O(n^2) repulsion to O(n log n) by treating distant node clusters as single masses:

```
┌─────────────────────────────────────────────────────────────┐
│                    Barnes-Hut Algorithm                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  For each node:                                              │
│                                                              │
│    Is cluster far enough?  (distance / size > theta)        │
│         │                                                    │
│         ├── Yes → Use centre of mass (1 interaction)        │
│         │                                                    │
│         └── No  → Recurse into children (8 interactions)    │
│                                                              │
│  Theta = 0.5 typical (trade accuracy for speed)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Spatial Grid

3D grid partitions space for O(1) neighbour queries:

1. Hash each node to grid cell
2. Check only 27 neighbouring cells
3. Update grid each frame (~0.3 ms for 100K nodes)

### Parallel Reduction

Warp-level primitives eliminate shared memory barriers:

```c
// Warp reduction: no __syncthreads needed
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

---

## Kernel Execution

### Block/Grid Configuration

| Kernel Type | Block Size | Grid Size | Shared Memory |
|-------------|-----------|-----------|---------------|
| Force calculation | 256 | (N+255)/256 | 16 KB |
| Reduction | 512 | (N+511)/512 | 32 KB |
| Graph traversal | 256 | (N+255)/256 | 8 KB |
| Clustering | 256 | K clusters | 48 KB |

### Occupancy Targets

- **Force kernels**: 75-100% occupancy (memory-bound)
- **Reduction kernels**: 100% occupancy (compute-bound)
- **Graph kernels**: 80-100% occupancy (latency-bound)

### Stream Parallelism

Three CUDA streams enable overlap:

```
Stream 0: Physics simulation (critical path)
Stream 1: Graph algorithms (independent)
Stream 2: Analytics/clustering (independent)
```

---

## Performance Characteristics

### Benchmark Results (RTX 4090)

| Operation | 10K nodes | 100K nodes |
|-----------|----------|------------|
| Force computation | 0.8 ms | 2.5 ms |
| Integration | 0.2 ms | 0.5 ms |
| Grid construction | 0.1 ms | 0.3 ms |
| PageRank (1 iter) | 1.0 ms | 3.5 ms |
| K-means assignment | 0.5 ms | 4.0 ms |
| **Total frame** | 2.6 ms | 10.8 ms |

### Frame Budget (60 FPS = 16.67 ms)

```
┌─────────────────────────────────────────────────────────────┐
│            Frame Time Budget (100K nodes)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Grid construction:      0.3 ms  [====]                     │
│  Force computation:      2.5 ms  [============]              │
│  Integration:            0.5 ms  [===]                       │
│  Graph algorithms:       3.5 ms  [================]          │
│  Clustering:             4.0 ms  [==================]        │
│  Semantic forces:        2.5 ms  [============]              │
│  Stability checks:       0.7 ms  [====]                      │
│  ─────────────────────────────────────────────────────      │
│  Total:                 14.0 ms  (85% budget used)          │
│  Remaining:              2.7 ms  (headroom)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## CPU Fallback

When CUDA is unavailable, VisionFlow falls back to CPU with Rayon parallelism:

| Algorithm | GPU Time | CPU (Serial) | CPU (Rayon) | CPU (Rayon + SIMD) |
|-----------|----------|--------------|-------------|---------------------|
| Force (10K) | 2.5 ms | 180 ms | 25 ms | 4-6 ms |
| K-means (10K) | 4.0 ms | 120 ms | 18 ms | 3-5 ms |
| PageRank (10K) | 3.5 ms | 95 ms | 15 ms | 3-4 ms |

CPU fallback with Rayon provides **5-7x** speedup over serial. Adding SIMD yields an estimated **4-8x** further improvement over scalar Rayon, bringing CPU performance within **2-4x** of GPU.

### CPU SIMD Acceleration

When running on the CPU fallback path, VisionFlow uses SIMD intrinsics to close the gap with GPU performance:

- **Runtime detection**: AVX2 and SSE4.1 support detected at startup via `is_x86_feature_detected!`
- **SIMD distance computation**: Processes 8 node-pairs per cycle with AVX2 (256-bit lanes), 4 pairs with SSE4.1
- **SIMD force accumulation**: Vectorized repulsive/attractive force summation across x/y/z components
- **SIMD position integration**: Velocity and position updates batched into 256-bit operations
- **Scalar fallback**: Non-x86 platforms (ARM, RISC-V) use an auto-vectorisation-friendly scalar path

| Algorithm | CPU (Serial) | CPU (Rayon) | CPU (Rayon + SIMD) | SIMD Speedup |
|-----------|-------------|-------------|---------------------|--------------|
| Force (10K) | 180 ms | 25 ms | 4-6 ms | 4-6x over Rayon |
| K-means (10K) | 120 ms | 18 ms | 3-5 ms | 4-6x over Rayon |
| PageRank (10K) | 95 ms | 15 ms | 3-4 ms | 4-5x over Rayon |

CPU fallback with SIMD provides an estimated **4-8x speedup over scalar Rayon** and remains **2-4x slower** than GPU.

---

## Error Handling

### GPU Error Recovery

```
┌─────────────────────────────────────────────────────────────┐
│                   GPU Error Recovery Flow                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Kernel Launch                                               │
│       ↓                                                      │
│  Check cudaGetLastError()                                   │
│       │                                                      │
│       ├── Success → Continue                                │
│       │                                                      │
│       └── Error → Log error                                 │
│                    ↓                                         │
│              Retry (3 attempts)                              │
│                    │                                         │
│                    ├── Success → Continue                   │
│                    │                                         │
│                    └── Failed → cudaDeviceReset()           │
│                                   ↓                          │
│                              Reinitialise GPU                │
│                                   │                          │
│                                   ├── Success → Continue    │
│                                   │                          │
│                                   └── Failed → CPU Fallback │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

### Minimum

- NVIDIA GPU with Compute Capability 3.5+
- 4 GB VRAM
- CUDA 11.0+ driver

### Recommended

- NVIDIA RTX 3080 or better
- 10+ GB VRAM
- CUDA 12.0+ driver
- NVLink for multi-GPU (future)

---

## Related Concepts

- **[Physics Engine](physics-engine.md)**: High-level physics simulation design
- **[Constraint System](constraint-system.md)**: How constraints are packed for GPU
- **[Actor Model](actor-model.md)**: GPU actor coordination via PhysicsOrchestratorActor
