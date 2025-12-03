---
title: GPU Semantic Force Kernels
description: The GPU semantic force system implements ontology-aware physics for knowledge graph visualization. It adds three types of forces based on semantic relationships between nodes:
type: explanation
status: stable
---

# GPU Semantic Force Kernels

## Overview

The GPU semantic force system implements ontology-aware physics for knowledge graph visualization. It adds three types of forces based on semantic relationships between nodes:

1. **Separation Forces**: Push nodes of disjoint classes apart
2. **Hierarchical Attraction**: Pull child class nodes toward parent centroids
3. **Alignment Forces**: Align nodes along axes based on ontology structure

## Architecture

### CUDA Kernels

#### `apply-semantic-forces`
**Location**: `src/utils/visionflow-unified.cu:1581-1737`

Computes semantic forces for each node based on constraint data.

**Grid/Block Configuration**:
- Grid: `(ceil(num-nodes/256), 1, 1)`
- Block: `(256, 1, 1)`
- Each thread processes one node

**Parameters**:
```cuda
--global-- void apply-semantic-forces(
    const float* pos-x,           // Node X positions
    const float* pos-y,           // Node Y positions
    const float* pos-z,           // Node Z positions
    float3* semantic-forces,      // Output: semantic forces per node
    const ConstraintData* constraints,  // Semantic constraints
    const int num-constraints,
    const int* node-class-indices,      // OWL class IDs per node
    const int num-nodes,
    const float dt                // Time step
);
```

**Force Calculations**:

1. **Separation Forces** (Disjoint Classes):
   ```cuda
   force-magnitude = separation-strength * (min-distance - dist) / dist
   force = normalize(pos-i - pos-j) * force-magnitude
   ```
   - Only applied when `class-i != class-j`
   - Uses `constraint.params[0]` for strength
   - Uses `constraint.params[3]` for minimum separation distance

2. **Hierarchical Attraction** (Parent-Child):
   ```cuda
   force-magnitude = attraction-strength * dist
   force = normalize(parent-pos - child-pos) * force-magnitude
   ```
   - Only applied to child nodes (node-role > 0)
   - First node in constraint is parent
   - Uses `constraint.params[1]` for strength

3. **Alignment Forces** (Axis-based):
   ```cuda
   centroid = average(group-positions)
   alignment-force = (centroid - my-pos) * alignment-strength
   ```
   - Aligns along X, Y, or Z axis based on `constraint.params[2]`
   - Uses `constraint.params[4]` for strength
   - Forces nodes onto alignment plane

#### `blend-semantic-physics-forces`
**Location**: `src/utils/visionflow-unified.cu:1743-1800`

Blends semantic forces with physics forces using priority-based weighting.

**Grid/Block Configuration**:
- Grid: `(ceil(num-nodes/256), 1, 1)`
- Block: `(256, 1, 1)`

**Blending Logic**:
```cuda
priority-weight = min(avg-priority / 10.0, 1.0)
final-force = base-force * (1 - priority-weight) + semantic-force * priority-weight
```

- Higher constraint weight → more semantic influence
- Priority range: 0-10 (normalized to 0-1)
- Automatic fallback to physics forces if NaN/Inf

## Integration with Physics Pipeline

### Execution Order

```
1. force-pass-kernel()          // Compute base physics forces
2. apply-semantic-forces()      // Compute semantic forces
3. blend-semantic-physics-forces()  // Blend forces
4. integrate-pass-kernel()      // Update positions/velocities
```

### Constraint Data Structure

```rust
#[repr(C)]
pub struct ConstraintData {
    pub kind: i32,              // ConstraintKind::SEMANTIC (3)
    pub count: i32,             // Number of nodes (max 4)
    pub node-idx: [i32; 4],     // Node indices
    pub params: [f32; 8],       // Force parameters
    pub weight: f32,            // Priority weight (0-10)
    pub activation-frame: i32,  // For progressive activation
}
```

**Parameter Layout for SEMANTIC constraints**:
- `params[0]`: Separation strength
- `params[1]`: Attraction strength
- `params[2]`: Alignment axis (0=X, 1=Y, 2=Z)
- `params[3]`: Minimum separation distance
- `params[4]`: Alignment strength

## GPU Buffer Management

### Required Buffers

1. **Constraint Buffer** (`ConstraintData*`)
   - Uploaded once per frame
   - Cached when ontology doesn't change
   - Size: `num-constraints * sizeof(ConstraintData)`

2. **Semantic Forces Buffer** (`float3*`)
   - Temporary storage for semantic forces
   - Size: `num-nodes * sizeof(float3)`

3. **Class Indices Buffer** (`int*`)
   - Maps nodes to OWL class IDs
   - Updated when ontology changes
   - Size: `num-nodes * sizeof(int)`

### Memory Management Strategy

```rust
// Upload constraints to GPU (once per ontology update)
gpu-compute.upload-constraints(&constraint-data)?;

// Allocate semantic forces buffer (once per initialization)
gpu-compute.allocate-semantic-forces-buffer(num-nodes)?;

// Upload class indices (when ontology changes)
gpu-compute.update-class-indices(&class-ids)?;
```

## Progressive Activation

Constraints use progressive activation to prevent sudden force application:

```cuda
if (c-params.constraint-ramp-frames > 0) {
    int frames = c-params.iteration - constraint.activation-frame;
    if (frames >= 0 && frames < c-params.constraint-ramp-frames) {
        multiplier = float(frames) / float(c-params.constraint-ramp-frames);
    }
}
```

- Ramps from 0 to 1 over `constraint-ramp-frames`
- Prevents physics instability
- Configurable per-constraint via `activation-frame`

## Performance Characteristics

### Computational Complexity
- **Per Node**: O(C) where C = number of constraints involving node
- **Total**: O(N * C-avg) where C-avg = average constraints per node
- **Typical**: ~3-5 constraints per node → O(N)

### Memory Bandwidth
- **Read**: Positions (12 bytes/node) + Constraints (48 bytes/constraint)
- **Write**: Semantic forces (12 bytes/node)
- **Total**: ~24 bytes/node + constraint overhead

### Optimization Opportunities
1. **Constraint Caching**: Cache constraints on GPU across frames
2. **Early Exit**: Skip nodes with no constraints
3. **Shared Memory**: Cache constraint data in shared memory
4. **Warp Divergence**: Group similar constraint types

## Usage Example

```rust
use visionflow-unified::UnifiedGPUCompute;

// Initialize GPU compute
let mut gpu-compute = UnifiedGPUCompute::new(num-nodes)?;

// Upload semantic constraints
let constraints = generate-semantic-constraints(&ontology);
let constraint-data: Vec<ConstraintData> = constraints
    .iter()
    .map(|c| c.to-gpu-format())
    .collect();

gpu-compute.upload-constraints(&constraint-data)?;

// Upload class indices
let class-indices = map-nodes-to-classes(&nodes, &ontology);
gpu-compute.update-class-indices(&class-indices)?;

// Physics loop
loop {
    gpu-compute.execute-physics-step(&simulation-params)?;

    // Semantic forces are automatically applied during physics step
    let positions = gpu-compute.get-node-positions()?;

    // Update visualization...
}
```

## Testing

### Unit Tests
- `tests/gpu-semantic-forces-test.rs`: Kernel correctness
- `tests/ontology-constraints-gpu-test.rs`: Integration tests

### Validation Metrics
1. **Force Magnitude**: Check forces are within `max-force` bounds
2. **Separation Distance**: Verify disjoint classes maintain minimum distance
3. **Alignment**: Measure deviation from alignment axes
4. **Stability**: Monitor kinetic energy convergence

## References

- [Force-Directed Graph Drawing](https://en.wikipedia.org/wiki/Force-directed-graph-drawing)
- [OWL 2 Web Ontology Language](https://www.w3.org/TR/owl2-overview/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
