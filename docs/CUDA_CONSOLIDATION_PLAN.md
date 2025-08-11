# CUDA/PTX Consolidation Plan for VisionFlow

## Executive Summary

The VisionFlow GPU compute system currently has 7 CUDA kernels with mixed compilation states, redundant implementations, and complex fallback mechanisms. This plan outlines a streamlined architecture consolidating to a single unified PTX with all required features.

## Current State Analysis

### 1. CUDA Kernel Inventory

| Kernel | Lines | Status | Purpose | Issues |
|--------|-------|--------|---------|---------|
| `compute_forces.cu` | 331 | ✅ Compiles | Basic force-directed layout | Legacy, limited features |
| `compute_dual_graphs.cu` | 383 | ✅ Compiles | Dual graph support | Redundant with unified |
| `dual_graph_unified.cu` | 1225 | ✅ Compiles | Unified dual graph physics | Good candidate for base |
| `unified_physics.cu` | 1092 | ✅ Compiles | Unified physics engine | Overlaps with dual_graph_unified |
| `visual_analytics_core.cu` | 1323 | ✅ Compiles | Visual analytics | Complex but working |
| `advanced_compute_forces.cu` | 2000+ | ❌ Fails | Advanced constraints | Missing dependencies |
| `advanced_gpu_algorithms.cu` | 2000+ | ❌ Fails | Complex analytics | Missing dependencies |

### 2. Compilation Issues

**Failed Kernels Root Causes:**
- Missing CUDA libraries: `cublas_v2.h`, `cusparse.h`, `cusolverDn.h`
- Unsupported features: `cooperative_groups.h`, `cub/cub.cuh`
- Complex template metaprogramming incompatible with PTX generation
- Excessive includes causing compilation timeout

### 3. Architecture Problems

**Current Issues:**
1. **Redundancy**: 3 kernels doing force calculations (`compute_forces`, `dual_graph_unified`, `unified_physics`)
2. **Fallback Complexity**: Multiple PTX paths with complex fallback logic
3. **Inconsistent Features**: Different kernels support different constraint types
4. **Memory Layout Mismatches**: SoA vs AoS inconsistencies
5. **Client Integration**: Control center can't specify which kernel features to use

### 4. Communication Path

```
Client (TypeScript/React)
    ↓ [WebSocket/REST]
Rust Server (actix-web)
    ↓ [Actor Messages]
GPUComputeActor
    ↓ [PTX Loading]
CUDA Kernels → PTX
    ↓ [Binary Protocol]
Client Visualization
```

## Consolidation Strategy

### Phase 1: Create Unified Kernel (Week 1)

**Objective**: Single `visionflow_unified.cu` kernel with all features

```cuda
// visionflow_unified.cu - Unified kernel structure
extern "C" {
    // Core data structures
    struct NodeData { /* SoA format */ };
    struct EdgeData { /* CSR format */ };
    struct SimParams { /* All parameters */ };
    
    // Single entry point with mode selection
    __global__ void visionflow_compute_kernel(
        NodeData* nodes,
        EdgeData* edges,
        SimParams* params,
        int mode  // 0=basic, 1=dual, 2=advanced, 3=analytics
    );
    
    // Feature-specific device functions
    __device__ void compute_basic_forces(...);
    __device__ void compute_dual_graph_forces(...);
    __device__ void compute_with_constraints(...);
    __device__ void compute_visual_analytics(...);
}
```

**Implementation Steps:**
1. Extract working code from `dual_graph_unified.cu` as base
2. Merge force calculation from `unified_physics.cu`
3. Add visual analytics from `visual_analytics_core.cu`
4. Simplify constraint system from failed kernels
5. Remove all external dependencies

### Phase 2: Simplify PTX Loading (Week 1)

**Current Complex Loading:**
```rust
// Multiple paths and fallbacks
let paths = [
    "/app/src/utils/ptx/advanced_compute_forces.ptx",
    "src/utils/ptx/compute_forces.ptx",
    "./fallback/legacy.ptx"
];
```

**New Simple Loading:**
```rust
// Single PTX with runtime mode selection
pub struct UnifiedGPU {
    kernel: CudaFunction,
    mode: ComputeMode,
}

impl UnifiedGPU {
    pub fn new(device: Arc<CudaDevice>) -> Result<Self> {
        let ptx = Ptx::from_file("/app/src/utils/ptx/visionflow_unified.ptx");
        let kernel = device.load_ptx(ptx, "visionflow_compute_kernel", 
                                     &["visionflow_compute_kernel"])?;
        Ok(Self { kernel, mode: ComputeMode::Basic })
    }
    
    pub fn set_mode(&mut self, mode: ComputeMode) {
        self.mode = mode;
    }
}
```

### Phase 3: Fix Control Center Integration (Week 2)

**Current Issue**: Client can't control GPU features

**Solution**: Add GPU control to settings API

```typescript
// Client-side GPU control
interface GPUSettings {
    mode: 'basic' | 'dual' | 'advanced' | 'analytics';
    constraints: ConstraintConfig[];
    visualization: {
        trajectories: boolean;
        isolationLayers: boolean;
        temporalAnalysis: boolean;
    };
}

// Settings update
await apiService.updateGPUSettings({
    mode: 'advanced',
    constraints: [
        { type: 'separation', enabled: true, strength: 0.5 },
        { type: 'alignment', enabled: false }
    ]
});
```

**Server-side handler:**
```rust
// New GPU settings endpoint
pub async fn update_gpu_settings(
    state: web::Data<AppState>,
    settings: web::Json<GpuSettings>
) -> impl Responder {
    state.gpu_actor.send(UpdateGpuMode {
        mode: settings.mode,
        constraints: settings.constraints,
    }).await
}
```

### Phase 4: Remove Legacy Code (Week 2)

**Files to Remove:**
- `src/utils/advanced_compute_forces.cu` (non-compiling)
- `src/utils/advanced_gpu_algorithms.cu` (non-compiling)
- `src/utils/compute_forces.cu` (legacy)
- `src/utils/compute_dual_graphs.cu` (redundant)
- `src/utils/advanced_gpu_compute.rs` (complex fallbacks)

**Code to Simplify:**
- Remove all PTX fallback logic
- Remove `ComputeMode::Legacy`
- Consolidate `GPUComputeActor` modes

### Phase 5: Optimize Compilation (Week 3)

**Build System Improvements:**

```bash
#!/bin/bash
# Optimized compilation script
nvcc -ptx \
    -arch=sm_86 \
    -O3 \
    --use_fast_math \
    --restrict \
    --ftz=true \
    --prec-div=false \
    --prec-sqrt=false \
    --maxrregcount=64 \
    visionflow_unified.cu \
    -o visionflow_unified.ptx
```

**Docker Integration:**
```dockerfile
# Multi-stage build
FROM nvidia/cuda:12.0-devel AS cuda-builder
COPY src/utils/visionflow_unified.cu /build/
RUN nvcc -ptx -arch=sm_86 -O3 /build/visionflow_unified.cu -o /build/visionflow_unified.ptx

FROM nvidia/cuda:12.0-runtime
COPY --from=cuda-builder /build/visionflow_unified.ptx /app/src/utils/ptx/
```

## Implementation Checklist

### Week 1: Core Consolidation
- [ ] Create `visionflow_unified.cu` from working kernels
- [ ] Test compilation with all features
- [ ] Update `GPUComputeActor` to use single kernel
- [ ] Remove kernel mode fallback logic

### Week 2: Integration
- [ ] Add GPU settings to REST API
- [ ] Update client PhysicsEngineControls component
- [ ] Connect settings to GPU mode selection
- [ ] Test end-to-end control flow

### Week 3: Cleanup & Optimization
- [ ] Remove all legacy CUDA files
- [ ] Optimize PTX compilation flags
- [ ] Profile performance improvements
- [ ] Update documentation

## Success Metrics

1. **Single PTX file** replacing 7 separate kernels
2. **100% compilation success** rate
3. **Client control** over all GPU features
4. **50% reduction** in code complexity
5. **No performance regression** vs current system

## Risk Mitigation

1. **Backup Strategy**: Keep working PTX files until new system proven
2. **Feature Flags**: Gradual rollout with ability to revert
3. **Performance Testing**: Benchmark before/after each change
4. **Incremental Migration**: Test each feature independently

## Technical Details

### Unified Kernel Structure

```cuda
// Simplified, dependency-free implementation
__global__ void visionflow_compute_kernel(
    float* pos_x, float* pos_y, float* pos_z,  // SoA node positions
    float* vel_x, float* vel_y, float* vel_z,  // SoA velocities
    int* edge_src, int* edge_dst, float* edge_weight,  // CSR edges
    SimParams params,
    int num_nodes,
    int num_edges,
    int compute_mode
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    
    switch(compute_mode) {
        case 0: // Basic forces
            force = compute_basic_forces(idx, /*...*/);
            break;
        case 1: // Dual graph
            force = compute_dual_forces(idx, /*...*/);
            break;
        case 2: // With constraints
            force = compute_constrained_forces(idx, /*...*/);
            break;
        case 3: // Visual analytics
            force = compute_analytics_forces(idx, /*...*/);
            break;
    }
    
    // Update velocities with damping
    vel_x[idx] = (vel_x[idx] + force.x * params.dt) * params.damping;
    vel_y[idx] = (vel_y[idx] + force.y * params.dt) * params.damping;
    vel_z[idx] = (vel_z[idx] + force.z * params.dt) * params.damping;
    
    // Update positions
    pos_x[idx] += vel_x[idx] * params.dt;
    pos_y[idx] += vel_y[idx] * params.dt;
    pos_z[idx] += vel_z[idx] * params.dt;
}
```

### Memory Layout Optimization

**Current Mixed Layout:**
- AoS in some kernels: `struct Node { float3 pos; float3 vel; }`
- SoA in others: `float* pos_x, *pos_y, *pos_z`

**Unified SoA Layout:**
- Better coalescing: All threads access contiguous memory
- Cache efficiency: Separate hot/cold data
- SIMD friendly: Vectorized operations

### Constraint System Simplification

**Remove Complex Dependencies:**
- No cuBLAS/cuSPARSE/cuSOLVER
- Pure CUDA C implementation
- Device-only functions
- No dynamic allocation

**Simplified Constraints:**
```cuda
__device__ float3 apply_constraints(
    int idx, float3 force, ConstraintParams* constraints
) {
    // Simple, efficient constraint application
    if (constraints->separation_enabled) {
        force = apply_separation(idx, force, constraints->separation_dist);
    }
    if (constraints->boundary_enabled) {
        force = apply_boundary(idx, force, constraints->bounds);
    }
    return force;
}
```

## Conclusion

This consolidation will transform a complex, fragile system with 7 kernels and multiple failure points into a robust, single-kernel solution with full client control. The unified approach eliminates compilation issues, reduces maintenance burden, and provides a clear upgrade path for future enhancements.