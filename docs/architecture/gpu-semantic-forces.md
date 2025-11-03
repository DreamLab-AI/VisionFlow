# GPU Semantic Force Kernels

## Overview

The GPU semantic force system implements ontology-aware physics for knowledge graph visualization. It adds three types of forces based on semantic relationships between nodes:

1. **Separation Forces**: Push nodes of disjoint classes apart
2. **Hierarchical Attraction**: Pull child class nodes toward parent centroids
3. **Alignment Forces**: Align nodes along axes based on ontology structure

## Architecture

### CUDA Kernels

#### `apply_semantic_forces`
**Location**: `src/utils/visionflow_unified.cu:1581-1737`

Computes semantic forces for each node based on constraint data.

**Grid/Block Configuration**:
- Grid: `(ceil(num_nodes/256), 1, 1)`
- Block: `(256, 1, 1)`
- Each thread processes one node

**Parameters**:
```cuda
__global__ void apply_semantic_forces(
    const float* pos_x,           // Node X positions
    const float* pos_y,           // Node Y positions
    const float* pos_z,           // Node Z positions
    float3* semantic_forces,      // Output: semantic forces per node
    const ConstraintData* constraints,  // Semantic constraints
    const int num_constraints,
    const int* node_class_indices,      // OWL class IDs per node
    const int num_nodes,
    const float dt                // Time step
);
```

**Force Calculations**:

1. **Separation Forces** (Disjoint Classes):
   ```cuda
   force_magnitude = separation_strength * (min_distance - dist) / dist
   force = normalize(pos_i - pos_j) * force_magnitude
   ```
   - Only applied when `class_i != class_j`
   - Uses `constraint.params[0]` for strength
   - Uses `constraint.params[3]` for minimum separation distance

2. **Hierarchical Attraction** (Parent-Child):
   ```cuda
   force_magnitude = attraction_strength * dist
   force = normalize(parent_pos - child_pos) * force_magnitude
   ```
   - Only applied to child nodes (node_role > 0)
   - First node in constraint is parent
   - Uses `constraint.params[1]` for strength

3. **Alignment Forces** (Axis-based):
   ```cuda
   centroid = average(group_positions)
   alignment_force = (centroid - my_pos) * alignment_strength
   ```
   - Aligns along X, Y, or Z axis based on `constraint.params[2]`
   - Uses `constraint.params[4]` for strength
   - Forces nodes onto alignment plane

#### `blend_semantic_physics_forces`
**Location**: `src/utils/visionflow_unified.cu:1743-1800`

Blends semantic forces with physics forces using priority-based weighting.

**Grid/Block Configuration**:
- Grid: `(ceil(num_nodes/256), 1, 1)`
- Block: `(256, 1, 1)`

**Blending Logic**:
```cuda
priority_weight = min(avg_priority / 10.0, 1.0)
final_force = base_force * (1 - priority_weight) + semantic_force * priority_weight
```

- Higher constraint weight → more semantic influence
- Priority range: 0-10 (normalized to 0-1)
- Automatic fallback to physics forces if NaN/Inf

## Integration with Physics Pipeline

### Execution Order

```
1. force_pass_kernel()          // Compute base physics forces
2. apply_semantic_forces()      // Compute semantic forces
3. blend_semantic_physics_forces()  // Blend forces
4. integrate_pass_kernel()      // Update positions/velocities
```

### Constraint Data Structure

```rust
#[repr(C)]
pub struct ConstraintData {
    pub kind: i32,              // ConstraintKind::SEMANTIC (3)
    pub count: i32,             // Number of nodes (max 4)
    pub node_idx: [i32; 4],     // Node indices
    pub params: [f32; 8],       // Force parameters
    pub weight: f32,            // Priority weight (0-10)
    pub activation_frame: i32,  // For progressive activation
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
   - Size: `num_constraints * sizeof(ConstraintData)`

2. **Semantic Forces Buffer** (`float3*`)
   - Temporary storage for semantic forces
   - Size: `num_nodes * sizeof(float3)`

3. **Class Indices Buffer** (`int*`)
   - Maps nodes to OWL class IDs
   - Updated when ontology changes
   - Size: `num_nodes * sizeof(int)`

### Memory Management Strategy

```rust
// Upload constraints to GPU (once per ontology update)
gpu_compute.upload_constraints(&constraint_data)?;

// Allocate semantic forces buffer (once per initialization)
gpu_compute.allocate_semantic_forces_buffer(num_nodes)?;

// Upload class indices (when ontology changes)
gpu_compute.update_class_indices(&class_ids)?;
```

## Progressive Activation

Constraints use progressive activation to prevent sudden force application:

```cuda
if (c_params.constraint_ramp_frames > 0) {
    int frames = c_params.iteration - constraint.activation_frame;
    if (frames >= 0 && frames < c_params.constraint_ramp_frames) {
        multiplier = float(frames) / float(c_params.constraint_ramp_frames);
    }
}
```

- Ramps from 0 to 1 over `constraint_ramp_frames`
- Prevents physics instability
- Configurable per-constraint via `activation_frame`

## Performance Characteristics

### Computational Complexity
- **Per Node**: O(C) where C = number of constraints involving node
- **Total**: O(N * C_avg) where C_avg = average constraints per node
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
use visionflow_unified::UnifiedGPUCompute;

// Initialize GPU compute
let mut gpu_compute = UnifiedGPUCompute::new(num_nodes)?;

// Upload semantic constraints
let constraints = generate_semantic_constraints(&ontology);
let constraint_data: Vec<ConstraintData> = constraints
    .iter()
    .map(|c| c.to_gpu_format())
    .collect();

gpu_compute.upload_constraints(&constraint_data)?;

// Upload class indices
let class_indices = map_nodes_to_classes(&nodes, &ontology);
gpu_compute.update_class_indices(&class_indices)?;

// Physics loop
loop {
    gpu_compute.execute_physics_step(&simulation_params)?;

    // Semantic forces are automatically applied during physics step
    let positions = gpu_compute.get_node_positions()?;

    // Update visualization...
}
```

## Testing

### Unit Tests
- `tests/gpu_semantic_forces_test.rs`: Kernel correctness
- `tests/ontology_constraints_gpu_test.rs`: Integration tests

### Validation Metrics
1. **Force Magnitude**: Check forces are within `max_force` bounds
2. **Separation Distance**: Verify disjoint classes maintain minimum distance
3. **Alignment**: Measure deviation from alignment axes
4. **Stability**: Monitor kinetic energy convergence

## References

- [Force-Directed Graph Drawing](https://en.wikipedia.org/wiki/Force-directed_graph_drawing)
- [OWL 2 Web Ontology Language](https://www.w3.org/TR/owl2-overview/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
