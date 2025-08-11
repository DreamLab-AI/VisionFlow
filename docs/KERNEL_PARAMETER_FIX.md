# Kernel Parameter Mismatch Fix

## Problem
The unified CUDA kernel (`visionflow_unified.cu`) uses **Structure of Arrays (SoA)** layout for better GPU memory coalescing:
```cuda
__global__ void visionflow_compute_kernel(
    float* pos_x, float* pos_y, float* pos_z,  // Separate arrays for each component
    float* vel_x, float* vel_y, float* vel_z,
    ...
)
```

But the Rust code (`advanced_gpu_compute.rs`) uses **Array of Structures (AoS)**:
```rust
struct EnhancedBinaryNodeData {
    pub position: Vec3Data,  // Packed struct
    pub velocity: Vec3Data,
    ...
}
```

This mismatch causes `CUDA_ERROR_INVALID_VALUE` when launching the kernel.

## Temporary Solution
Disabled the advanced kernel path in `advanced_gpu_compute.rs`:
```rust
use_advanced_kernel: false, // Disabled until SoA conversion is implemented
```

This forces the system to use the legacy kernel which still works.

## Permanent Solution Options

### Option 1: Convert Rust to SoA (Recommended)
Update `unified_gpu_compute.rs` to use separate arrays:
```rust
pub struct UnifiedGPUCompute {
    // Node buffers (Structure of Arrays)
    pos_x: CudaSlice<f32>,
    pos_y: CudaSlice<f32>,
    pos_z: CudaSlice<f32>,
    vel_x: CudaSlice<f32>,
    vel_y: CudaSlice<f32>,
    vel_z: CudaSlice<f32>,
    ...
}
```

### Option 2: Add AoS to SoA Converter
Create a conversion layer that unpacks structs before kernel launch:
```rust
fn convert_aos_to_soa(nodes: &[EnhancedBinaryNodeData]) -> (Vec<f32>, Vec<f32>, Vec<f32>, ...) {
    let mut pos_x = Vec::with_capacity(nodes.len());
    let mut pos_y = Vec::with_capacity(nodes.len());
    let mut pos_z = Vec::with_capacity(nodes.len());
    
    for node in nodes {
        pos_x.push(node.position.x);
        pos_y.push(node.position.y);
        pos_z.push(node.position.z);
    }
    
    (pos_x, pos_y, pos_z, ...)
}
```

### Option 3: Update Kernel to Support AoS
Modify the CUDA kernel to accept packed structs (not recommended due to performance impact).

## Performance Impact
- **SoA**: Better memory coalescing, ~2-4x faster for large graphs
- **AoS**: Simpler code but worse GPU memory access patterns

## Implementation Priority
1. Keep advanced kernel disabled for stability
2. Use `unified_gpu_compute.rs` which already has SoA support
3. Gradually migrate other modules to use the unified compute module
4. Remove legacy AoS-based modules once migration is complete